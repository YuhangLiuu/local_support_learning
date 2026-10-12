"""Explicit, independent checkpoint/probe workflow. Not called by training.

Supply already tokenized fixed batches, stable sample IDs and an exact preprocessing
description. No datasets/models are downloaded by this module itself.
"""

from contextlib import contextmanager
import copy
import random

import numpy as np

from .collector import cpu, get_observer
from .drift_metrics import DriftTracker, probe_fingerprint


@contextmanager
def isolated_state(model):
    """Restore modes, RNG and module-local EMA/prefix/diagnostic state on exceptions too.

    KV cache must be newly created by the caller for each run; run_probe below
    explicitly sets use_cache=False and does not accept a past_key_values input.
    Parameters/gradients are never written or zeroed by the probe workflow.
    """
    import torch
    from modeling_qwen2 import ConditionalLoRALinear
    modes = [(m, m.training) for m in model.modules()]
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    states = []
    lists = []
    for m in model.modules():
        if isinstance(m, ConditionalLoRALinear):
            # Keep the original list objects (siblings alias is_disabled) AND
            # their contents. Saving just a reference cannot undo append/setitem.
            for values in (m.is_disabled, m._pending_x):
                lists.append((values, list(values)))
            states.append((m, dict(
                _smoothing_state={k: v.clone() for k, v in m._smoothing_state.items()},
                prefix_mask=m.prefix_mask, smoothing_alpha=m.smoothing_alpha,
                is_cl_lora=m.is_cl_lora, is_disabled=m.is_disabled,
                is_cl_accum=m.is_cl_accum, _pending_x=m._pending_x,
                is_hit_rate_tracking=m.is_hit_rate_tracking,
                hit_count=m.hit_count, token_count=m.token_count,
                seq_hit_count=m.seq_hit_count, seq_count=m.seq_count)))
    obs = get_observer(model)
    obs_state = None if obs is None else (obs.context, obs.calls, obs.sampled, copy.deepcopy(obs.budgets))
    if obs is not None:
        obs.flush()
    try:
        yield
    finally:
        for original, contents in lists:
            original[:] = contents
        for m, state in states:
            for key, value in state.items():
                setattr(m, key, value)
        for m, training in modes:
            m.training = training
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        if obs is not None:
            try:
                obs.flush()
            finally:
                obs.context, obs.calls, obs.sampled, obs.budgets = obs_state


def logits_kl(reference, candidate, valid):
    """KL(P_reference || P_candidate), bounded valid token positions only."""
    import torch
    r = torch.log_softmax(reference[valid].float(), dim=-1)
    c = torch.log_softmax(candidate[valid].float(), dim=-1)
    return (r.exp() * (r-c)).sum(-1)


class ProbeRunner:
    def __init__(self, model, max_probes=16):
        self.model = model.module if hasattr(model, "module") else model
        self.tracker = DriftTracker(max_probes)

    def run(self, probe_id, sample_ids, batch, preprocessing, stage, epoch,
            modes=("full", "base", "all_gated_no_smoothing", "current_forced"), adapter_id=None):
        import torch
        from modeling_qwen2 import ConditionalLoRALinear
        obs = get_observer(self.model)
        if obs is None or obs.config.level != "probe":
            raise ValueError("attach diagnostics with level='probe' to an independently loaded checkpoint")
        if set(batch) - {"input_ids", "attention_mask", "labels"}:
            raise ValueError("only fixed token IDs, attention mask and optional labels allowed; no shared KV cache")
        if len(sample_ids) != batch["input_ids"].shape[0]:
            raise ValueError("one stable sample ID required per row")
        fingerprint = probe_fingerprint(sample_ids, batch["input_ids"].detach().cpu().numpy(),
                                        batch["attention_mask"].detach().cpu().numpy(),
                                        dict(rules=preprocessing, labels=batch["labels"].detach().cpu().tolist()
                                             if "labels" in batch else None))
        if any(mode not in {"full", "base", "all_gated_no_smoothing", "current_forced", "adapter_off"} for mode in modes):
            raise ValueError("unknown diagnostic mode")
        if "adapter_off" in modes and adapter_id is None:
            raise ValueError("adapter_off requires adapter_id")
        outputs, snapshots, layer_outputs = {}, {}, {}
        for mode in modes:
            with isolated_state(self.model), torch.no_grad():
                self.model.eval()
                for module in self.model.modules():
                    if not isinstance(module, ConditionalLoRALinear):
                        continue
                    module._smoothing_state = {}
                    module.is_cl_accum = False
                    module._pending_x = []
                    module.is_hit_rate_tracking = False
                    module.is_cl_lora = True
                    if mode == "base":
                        module.is_disabled = [True] * module.num_loras
                    elif mode == "adapter_off":
                        if not 0 <= adapter_id < module.num_loras:
                            raise ValueError("adapter index out of range")
                        module.is_disabled = [d or i == adapter_id for i, d in enumerate(module.is_disabled)]
                    elif mode in {"all_gated_no_smoothing", "current_forced"}:
                        module.smoothing_alpha = 1.0
                        # Only the Linear wrapper uses training to force the current
                        # adapter. All dropout modules remain eval for a controlled comparison.
                        module.training = mode == "current_forced"
                obs.set_context(dataset=probe_id, stage=stage, epoch=epoch, mode="probe_" + mode,
                                probe_fingerprint=fingerprint)
                obs.sampled = 0
                handles = []
                states = {}
                # Full layer outputs are sampled before transfer, never stored in full.
                def capture(index):
                    def hook(module, args, output):
                        mask = torch.as_tensor(obs.valid, device=output.device)
                        states[index] = cpu(output[mask])
                    return hook
                for i, layer in enumerate(self.model.model.layers):
                    handles.append(layer.register_forward_hook(capture(i)))
                try:
                    inputs = {k: v for k, v in batch.items() if k != "labels"}
                    output = self.model(**inputs, use_cache=False)
                    mask = torch.as_tensor(obs.valid, device=output.logits.device)
                    outputs[mode] = output.logits.detach()[mask].float().cpu()
                    snapshots[mode] = copy.deepcopy(obs.last_snapshot)
                    layer_outputs[mode] = states
                    if "labels" in batch:
                        labels = batch["labels"][:, 1:]
                        logits = output.logits[:, :-1]
                        keep = (labels != -100) & batch["attention_mask"][:, 1:].bool()
                        nll = torch.nn.functional.cross_entropy(logits[keep].float(), labels[keep], reduction="sum") if keep.any() else logits.new_zeros(())
                        obs.writer.write("probe_metrics", obs.context, metric_name="answer_nll",
                                         numerator=float(nll.cpu()), denominator=int(keep.sum().cpu()),
                                         sample_ids=list(sample_ids))
                finally:
                    for handle in handles:
                        handle.remove()
        context = dict(dataset=probe_id, stage=stage, epoch=epoch, mode="probe_comparison",
                       probe_fingerprint=fingerprint)
        if "base" in outputs and "full" in outputs:
            values = logits_kl(outputs["base"], outputs["full"], slice(None))
            obs.writer.write("probe_metrics", context, metric_name="kl_base_to_lsl",
                             numerator=float(values.sum()), denominator=len(values), sample_ids=list(sample_ids))
        if "full" in outputs:
            for mode, logits in outputs.items():
                if mode == "full":
                    continue
                kl = logits_kl(outputs["full"], logits, slice(None))
                obs.writer.write("probe_metrics", {**context, "comparison_mode": mode}, metric_name="kl_full_to_mode",
                                 numerator=float(kl.sum()), denominator=len(kl), sample_ids=list(sample_ids))
                for key, full in snapshots["full"].items():
                    if key not in snapshots[mode]:
                        continue
                    other = snapshots[mode][key]
                    flips = full["mask"] != other["mask"]
                    obs.writer.write("probe_metrics", {**context, "comparison_mode": mode}, key[0], key[0], key[1],
                                     metric_name="mode_gate_disagreement", numerator=int(flips.sum()), denominator=len(flips))
                    if full.get("update") is not None and other.get("update") is not None:
                        delta = np.linalg.norm(full["update"] - other["update"], axis=-1)
                        obs.writer.write("probe_metrics", {**context, "comparison_mode": mode}, key[0], key[0], key[1],
                                         metric_name="mode_update_difference_l2", numerator=float(delta.sum()), denominator=len(delta))
            for layer, values in layer_outputs["full"].items():
                snapshots["full"][(f"model.layers.{layer}", None)] = dict(
                    z=values.copy(), mask=np.zeros(len(values), bool), margin=None,
                    valid_positions=np.flatnonzero(obs.valid))
            for record in self.tracker.update(probe_id, fingerprint, stage, snapshots["full"], epoch=epoch):
                record = dict(record)
                owner, adapter = record.pop("owner"), record.pop("adapter_id")
                obs.writer.write("drift_metrics", context, owner, owner, adapter, **record,
                                 sample_ids=list(sample_ids))
        if "base" in layer_outputs and "full" in layer_outputs:
            for layer, full in layer_outputs["full"].items():
                delta = np.linalg.norm(full - layer_outputs["base"][layer], axis=-1)
                name = f"model.layers.{layer}"
                obs.writer.write("probe_metrics", context, name, name, metric_name="layer_output_change_l2",
                                 numerator=float(delta.sum()), denominator=len(delta))
        return dict(probe_fingerprint=fingerprint, sampled_logits=outputs,
                    snapshots=snapshots, sampled_layer_outputs=layer_outputs)


def main():
    """Load only local artifacts. Never invoked by the official training entry."""
    import argparse
    import json
    from pathlib import Path
    from .collector import attach_diagnostics
    from .config import DiagnosticsConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="local base-model directory")
    parser.add_argument("--cl-state", required=True)
    parser.add_argument("--probe-file", required=True, help="fixed token IDs, masks, IDs, preprocessing JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment-id", required=True, help="same ID as training experiment")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--history", help="bounded NPZ history, reused across stage checkpoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--adapter-off", type=int)
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer
    from modeling_qwen2 import (Qwen2ForCausalLM, apply_conditional_lora_wrapper,
                                load_cl_state, set_shared_prefix_ids)
    manifest = json.loads(Path(args.cl_state).with_suffix(".json").read_text(encoding="utf-8"))
    if (args.stage, args.epoch) != (manifest["stage"], manifest["epoch"]):
        raise ValueError("requested stage/epoch does not match checkpoint sidecar")
    configuration = manifest["configuration"]
    model = Qwen2ForCausalLM.from_pretrained(args.base_model, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation=args.attention_implementation).to(args.device)
    apply_conditional_lora_wrapper(model, is_cl_lora=configuration["activate_cl_method"],
        jl_k=configuration["jl_k"] if configuration["activate_jl"] else 0,
        jl_seed=configuration["seed"], cl_memory_type=configuration["cl_memory_type"],
        smoothing_alpha=configuration["smoothing_alpha"])
    load_cl_state(model, args.cl_state)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    set_shared_prefix_ids(model, tokenizer)
    observer = attach_diagnostics(model, DiagnosticsConfig(enabled=True, level="probe",
        max_tokens_per_batch=args.max_tokens), args.output, args.experiment_id,
        metadata=dict(manifest=manifest, attention_implementation=args.attention_implementation, device=args.device))
    runner = ProbeRunner(model)
    if args.history and Path(args.history).exists():
        runner.tracker.load(args.history)
    data = json.loads(Path(args.probe_file).read_text(encoding="utf-8"))
    modes = ("full", "base", "all_gated_no_smoothing", "current_forced")
    if args.adapter_off is not None:
        modes += ("adapter_off",)
    try:
        for probe in data["probes"]:
            batch = {name: torch.tensor(probe[name], dtype=torch.long, device=args.device)
                     for name in ("input_ids", "attention_mask", "labels") if name in probe}
            runner.run(probe["probe_id"], probe["sample_ids"], batch, data["preprocessing"],
                       args.stage, args.epoch, modes=modes, adapter_id=args.adapter_off)
        if args.history:
            runner.tracker.save(args.history)
    finally:
        observer.close()


if __name__ == "__main__":
    main()

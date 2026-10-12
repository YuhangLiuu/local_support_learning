"""Read-only observer of actual decisions, scores and updates in the normal forward.

No diagnostic model forwards, projection matmuls or LoRA matmuls are added.
GPU tensors survive only until the sampled root forward ends.
"""

from dataclasses import asdict
import os

import numpy as np

from .adapter_metrics import cosine_values, correlation_sums, update_statistics
from .gate_metrics import (adjacent_disagreement, coactivation_counts, diag_gmm_scores,
                           gate_counts, score_distribution, valid_positions)
from .report import RecordWriter


def cpu(tensor):
    return tensor.detach().float().cpu().numpy()


def owner_name(name):
    parent, _, tail = name.rpartition(".")
    return parent + "." + {"k_proj": "q_proj", "v_proj": "q_proj", "up_proj": "gate_proj"}.get(tail, tail)


class DiagnosticsCollector:
    def __init__(self, model, config, path, experiment_id, metadata=None):
        self.config = config
        self.model = model
        self.writer = RecordWriter(path, experiment_id, int(os.environ.get("RANK", 0)),
                                   int(os.environ.get("WORLD_SIZE", 1)))
        self.context = None
        self.active = False
        self.calls = self.sampled = 0
        self.budgets = {}
        self.pending = []
        self.totals = {}
        self.handles = []
        self.trace_rows = 0
        self.last_snapshot = {}
        self.sequence = 0
        self.current_ticket = None
        self.score_bindings = []
        self.writer.metadata(dict(configuration=metadata or {}, diagnostics=asdict(config),
                                  score_precision="original GPU GMM outputs; CPU summaries only",
                                  sampling="first bounded valid tokens, deterministic; no RNG",
                                  sequence_unit="sampled row per forward; decode is not full-sequence",
                                  rank_semantics="local models; no GMM synchronization added",
                                  versions=self._versions()))

    @staticmethod
    def _versions():
        from importlib.metadata import PackageNotFoundError, version
        result = {}
        for name in ("torch", "transformers", "pomegranate", "numpy", "datasets", "lm_eval"):
            try:
                result[name] = version(name)
            except PackageNotFoundError:
                result[name] = None
        return result

    def set_context(self, **context):
        if context != self.context:
            self.flush()
            if self.context is not None:
                self.budgets[tuple(sorted(self.context.items()))] = (self.calls, self.sampled)
            self.context = dict(context)
            self.calls, self.sampled = self.budgets.get(tuple(sorted(context.items())), (0, 0))

    def _pre(self, model, args, kwargs):
        self.active = False
        self.pending = []
        self.current_ticket = None
        self.last_snapshot = {}
        if self.context is None or self.config.level == "profile":
            return
        self.calls += 1
        force = self.context.get("mode", "").startswith("probe")
        if self.sampled >= self.config.max_batches_per_context:
            return
        if not force and (self.calls - 1) % self.config.every_n_forwards:
            return
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            return  # inputs_embeds-only callers must supply their own diagnostic mask
        import torch
        from cl_memory import drop_shared_prefix_mask
        with torch.no_grad():
            pad = kwargs.get("attention_mask")
            if pad is None:
                pad = torch.ones_like(ids)
            if pad.ndim != 2:
                return
            pad = pad[:, -ids.shape[1]:]
            prefix = getattr(model.model, "_shared_prefix_ids", None)
            keep = pad.bool()
            if prefix is not None and ids.shape[1] > 1:
                keep = drop_shared_prefix_mask(ids, pad, prefix.to(ids.device))
            self.valid = valid_positions(keep.detach().cpu().numpy(), budget=self.config.max_tokens_per_batch)
        if not self.valid.any():
            return
        self.sampled += 1
        self.sequence += 1
        self.active = True

    def begin(self, module, x, base):
        if not self.active:
            return None
        import torch
        if tuple(x.shape[:2]) != self.valid.shape:
            return None
        with torch.no_grad():
            idx = torch.as_tensor(np.flatnonzero(self.valid), device=x.device)
            xs = x.detach().reshape(-1, x.shape[-1]).index_select(0, idx) if self.config.extended else None
            ticket = dict(module=module, name=module._lsl_diagnostic_name, idx=idx,
                          x=xs if self.config.extended else None, z=None, scores={}, offsets={},
                          base=base.detach().reshape(-1, base.shape[-1]).index_select(0, idx)
                          if self.config.extended else None, adapters={})
        self.pending.append(ticket)
        self.current_ticket = ticket
        if module.is_cl_memory_owner:
            for clf in module.cl_gmm:
                clf._lsl_diagnostics = self
                self.score_bindings.append(clf)
        return ticket

    def projected(self, ticket, flat_jl):
        if self.config.extended:
            ticket["z"] = flat_jl.detach().index_select(0, ticket["idx"])

    def gmm_scores(self, clf, x, pos, neg):
        ticket = self.current_ticket
        if not self.active or ticket is None or not ticket["module"].is_cl_memory_owner:
            return
        module = ticket["module"]
        adapter = next((i for i, c in enumerate(module.cl_gmm) if c is clf), None)
        if adapter is None:
            return
        import torch
        with torch.no_grad():
            offset = ticket["offsets"].get(adapter, 0)
            idx = ticket["idx"]
            selected = idx[(idx >= offset) & (idx < offset + x.shape[0])] - offset
            ticket["offsets"][adapter] = offset + x.shape[0]
            ticket["scores"].setdefault(adapter, []).append(
                (pos.detach().index_select(0, selected), neg.detach().index_select(0, selected)))

    def observe(self, ticket, adapter, raw=None, applied=None, forced=False, disabled=False, update=None):
        if ticket is None:
            return
        import torch
        with torch.no_grad():
            def select(mask):
                return None if mask is None else mask.detach().reshape(-1).index_select(0, ticket["idx"])
            data = dict(raw=select(raw), applied=select(applied), forced=forced, disabled=disabled)
            if self.config.extended and update is not None:
                data["update"] = update.detach().reshape(-1, update.shape[-1]).index_select(0, ticket["idx"])
            ticket["adapters"][adapter] = data

    def _add(self, family, name, owner, adapter, metric, numerator, denominator, **dims):
        key = (family, name, owner, adapter, metric, tuple(sorted(dims.items())))
        values = self.totals.setdefault(key, [0., 0])
        values[0] += float(numerator)
        values[1] += int(denominator)

    def _distribution(self, family, name, owner, adapter, metric, values):
        d = score_distribution(values, self.config.histogram_edges, self.config.near_zero_margin)
        self._add(family, name, owner, adapter, metric, d["sum"], d["count"])
        self.writer.write(family, self.context, name, owner, adapter, metric_name=metric + "_distribution",
                          numerator=d["sum"], denominator=d["count"], forward_id=self.sequence,
                          **{k: v for k, v in d.items() if k not in {"sum", "count"}})

    def _post(self, model, args, kwargs, output):
        try:
            if self.active and output is not None:
                self._consume()
        finally:
            self.pending = []
            self.current_ticket = None
            for clf in self.score_bindings:
                if getattr(clf, "_lsl_diagnostics", None) is self:
                    del clf._lsl_diagnostics
            self.score_bindings.clear()
            self.active = False

    def _consume(self):
        layer_masks = {}
        self.last_snapshot = {}
        for ticket in self.pending:
            m, name = ticket["module"], ticket["name"]
            owner = owner_name(name)
            valid = self.valid
            masks, potentials = [], {}
            z = cpu(ticket["z"]) if ticket["z"] is not None else None
            is_owner = m.is_cl_memory_owner
            for i in range(m.num_loras):
                data = ticket["adapters"].get(i, dict(disabled=True, forced=False, raw=None, applied=None))
                raw = np.zeros(valid.shape, bool)
                applied = np.zeros(valid.shape, bool)
                scores = None
                clf = m.cl_gmm[i] if m.cl_memory_type == "gmm" and i < len(m.cl_gmm) else None
                captured = ticket["scores"].get(i)
                if is_owner:
                    self._add("gmm_metrics", name, owner, i, "score_observed_rate",
                              int(bool(captured)) * valid.sum(), valid.sum())
                if is_owner and captured:
                    pos = np.concatenate([cpu(p) for p, _ in captured])
                    neg = np.concatenate([cpu(n) for _, n in captured])
                    scores = pos - neg
                    for metric, values in (("positive_log_likelihood", pos), ("negative_log_likelihood", neg), ("margin", scores)):
                        self._distribution("gmm_metrics", name, owner, i, metric, values)
                    self._add("gmm_metrics", name, owner, i, "near_zero_margin_rate",
                              (np.abs(scores) <= self.config.near_zero_margin).sum(), len(scores))
                    if self.config.extended:
                        params = tuple(cpu(t) for t in clf.params_vectorized["pos"])
                        _, resp = diag_gmm_scores(z, *params, responsibilities=True)
                        occupancy = resp.sum(0)
                        self.writer.write("gmm_metrics", self.context, name, owner, i,
                                          metric_name="positive_component_occupancy", numerator=0, denominator=len(z),
                                          posterior_sum=occupancy.tolist(), hard_counts=np.bincount(resp.argmax(1), minlength=resp.shape[1]).tolist())
                        self._distribution("gmm_metrics", name, owner, i, "component_entropy",
                                           -(resp * np.log(np.maximum(resp, 1e-300))).sum(1))
                if data["raw"] is not None:
                    raw[valid] = cpu(data["raw"]).astype(bool)
                if data["applied"] is not None:
                    applied[valid] = cpu(data["applied"]).astype(bool)
                elif data["forced"]:
                    applied[valid] = True
                masks.append(applied)
                for metric, (num, den) in gate_counts(raw, applied, valid).items():
                    if metric in {"raw_activation_rate", "raw_applied_disagreement"} and data["raw"] is None:
                        continue  # forced-on/unfitted/disabled paths have no observed raw GMM decision
                    self._add("gate_metrics", name, owner, i, metric, num, den, statistic_scope="matrix")
                    if is_owner:
                        self._add("gate_metrics", name, owner, i, metric, num, den, statistic_scope="owner")
                self._add("gate_metrics", name, owner, i, "forced_on_rate", int(data["forced"]) * valid.sum(), valid.sum())
                self._add("gate_metrics", name, owner, i, "disabled_rate", int(data["disabled"]) * valid.sum(), valid.sum())
                if self.config.extended and "update" in data:
                    actual = cpu(data["update"])
                    potentials[i] = actual
                    for metric, values in update_statistics(actual, cpu(ticket["base"])).items():
                        self._distribution("adapter_metrics", name, owner, i, metric, values)
                    if scores is not None:
                        self.writer.write("adapter_metrics", self.context, name, owner, i,
                                          metric_name="margin_update_norm_correlation", numerator=0, denominator=len(scores),
                                          correlation_sums=correlation_sums(scores, np.linalg.norm(actual, axis=-1)))
                if is_owner:
                    if self.config.level == "probe":
                        self.last_snapshot[(name, i)] = dict(mask=applied[valid].copy(),
                                                        margin=None if scores is None else scores.copy(), z=z.copy(),
                                                        positive=None if scores is None else pos.copy(),
                                                        negative=None if scores is None else neg.copy(),
                                                        update=cpu(data["update"]) if "update" in data else None,
                                                        valid_positions=np.flatnonzero(valid).copy())
                    if self.config.extended:
                        suffix = name.split("layers.")[-1].split(".", 1)[-1]
                        layer = int(name.split("layers.")[-1].split(".", 1)[0])
                        layer_masks[(suffix, i, layer)] = (name, applied)
                    if self.config.save_token_traces and self.trace_rows < self.config.max_trace_rows:
                        self.writer.write("token_traces", self.context, name, owner, i, metric_name="gate_trace",
                                          numerator=int(applied[valid].sum()), denominator=int(valid.sum()),
                                          forward_id=self.sequence, positions=np.argwhere(valid).tolist(),
                                          decisions=applied[valid].astype(int).tolist())
                        self.trace_rows += 1
            for metric, (num, den) in coactivation_counts(np.asarray(masks, bool).reshape(m.num_loras, *valid.shape), valid).items():
                self._add("gate_metrics", name, owner, None, metric, num, den, statistic_scope="matrix")
                if is_owner:
                    self._add("gate_metrics", name, owner, None, metric, num, den, statistic_scope="owner")
            for i, left in potentials.items():
                for j, right in potentials.items():
                    if j > i:
                        vals = cosine_values(left, right)
                        self._add("adapter_metrics", name, owner, i, "applied_update_cosine", vals.sum(), len(vals), peer_adapter_id=j)
        for (suffix, i, layer), (name, mask) in layer_masks.items():
            previous = layer_masks.get((suffix, i, layer - 1))
            if previous:
                num, den = adjacent_disagreement(previous[1], mask, self.valid)
                self._add("gate_metrics", name, name, i, "adjacent_layer_disagreement", num, den, peer_module=previous[0])

    def flush(self):
        if self.context is None:
            return
        for (family, name, owner, adapter, metric, dims), (num, den) in self.totals.items():
            self.writer.write(family, self.context, name, owner, adapter, metric_name=metric,
                              numerator=num, denominator=den, **dict(dims))
        self.totals.clear()

    def record_fit(self, module, which):
        """Record original fit verdict; never retry, recalibrate or repair."""
        i = module.accum_lora_idx
        name = module._lsl_diagnostic_name
        self.writer.write("gmm_metrics", {**(self.context or {}), "mode": "fit_" + which},
                          name, owner_name(name), i, metric_name="fit_disabled",
                          numerator=int(module.is_disabled[i]), denominator=1,
                          negative_fit_failed=getattr(module, "_neg_fit_failed", False))

    def close(self):
        self.flush()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for module in self.model.modules():
            if getattr(module, "_lsl_diagnostics", None) is self:
                del module._lsl_diagnostics
            if hasattr(module, "_lsl_diagnostic_name"):
                del module._lsl_diagnostic_name
        if getattr(self.model, "_lsl_observer", None) is self:
            del self.model._lsl_observer
        self.pending.clear()
        self.last_snapshot.clear()


def attach_diagnostics(model, config, path, experiment_id, metadata=None):
    if not config.enabled:
        return None
    if hasattr(model, "_lsl_observer"):
        raise ValueError("diagnostics already attached")
    from modeling_qwen2 import ConditionalLoRALinear
    collector = DiagnosticsCollector(model, config, path, experiment_id, metadata)
    for name, module in model.named_modules():
        if isinstance(module, ConditionalLoRALinear):
            module._lsl_diagnostics = collector
            module._lsl_diagnostic_name = name
    collector.handles.append(model.register_forward_pre_hook(collector._pre, with_kwargs=True))
    collector.handles.append(model.register_forward_hook(collector._post, with_kwargs=True, always_call=True))
    model._lsl_observer = collector
    return collector


def get_observer(model):
    base = model.module if hasattr(model, "module") else model
    return getattr(base, "_lsl_observer", None)


def set_context(model, **context):
    observer = get_observer(model)
    if observer is not None:
        observer.set_context(**context)


def flush(model):
    observer = get_observer(model)
    if observer is not None:
        observer.flush()

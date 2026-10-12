"""Small experiment boundary adapters; no training/prediction policy changes."""

from contextlib import contextmanager
from dataclasses import asdict
from functools import wraps
import inspect
import json
from pathlib import Path
import shutil

from .collector import attach_diagnostics, get_observer
from .config import DiagnosticsConfig
from .report import json_safe


def initialize(model, tokenizer, config):
    if not config.diagnostics_enabled:
        return None
    if config.diagnostics_level not in {"basic", "extended"}:
        raise ValueError("formal training supports basic/extended only; use the independent probe/profile workflows")
    experiment_id = f"{config.start_time}_{config.wandb_run_name}"
    root = Path(config.diagnostics_output_dir or f"results/{experiment_id}")
    dc = DiagnosticsConfig(enabled=True, level=config.diagnostics_level,
                           every_n_forwards=config.diagnostics_every_n_forwards,
                           max_batches_per_context=config.diagnostics_max_batches,
                           max_tokens_per_batch=config.diagnostics_max_tokens)
    obs = attach_diagnostics(model, dc, root / "diagnostics", experiment_id,
                             metadata=dict(vars(config)))
    obs.experiment_root = root
    obs.writer.metadata(dict(configuration=dict(vars(config)), diagnostics=asdict(dc),
                             model_config=model.config.to_dict(),
                             tokenizer_revision=tokenizer.init_kwargs.get("_commit_hash"),
                             dataset_revisions="not pinned by upstream; record HF revisions before reproduction",
                             versions=obs._versions(),
                             sample_semantics="bounded valid tokens; sequence is forward row",
                             count_semantics="matrix and owner scopes must not be summed together",
                             score_semantics="actual fp32 GPU scores; forced-on paths do not calculate scores",
                             checkpoint_semantics="adapter/GMM snapshot, not optimizer resume state"))
    if obs.writer.identity["rank"] == 0:
        (root / "config.json").write_text(json.dumps(json_safe(dict(vars(config))), indent=2), encoding="utf-8")
        (root / "metadata.json").write_text(json.dumps(json_safe(obs.writer.identity), indent=2), encoding="utf-8")
    return obs


@contextmanager
def observation_context(model, **context):
    obs = get_observer(model)
    if obs is None:
        yield
        return
    previous = obs.context
    obs.set_context(**{**(previous or {}), **context})
    try:
        yield
    finally:
        obs.flush()
        if previous is not None:
            obs.set_context(**previous)
        else:
            obs.context = None


def observe_eval(mode):
    """Label original eval forwards; does not change model train/eval state."""
    def decorate(fn):
        signature = inspect.signature(fn)
        @wraps(fn)
        def wrapped(*args, **kwargs):
            model = kwargs.get("model", args[0] if args else None)
            if get_observer(model) is None:
                return fn(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            dataset = bound.arguments.get("dataset_name")
            dataset = dataset or ",".join(bound.arguments.get("tasks", [])) or mode
            with observation_context(model, mode=mode, dataset=dataset):
                return fn(*args, **kwargs)
        return wrapped
    return decorate


def stage_context(model, dataset, stage, epoch, mode="train"):
    obs = get_observer(model)
    if obs is not None:
        obs.set_context(dataset=dataset, stage=stage, epoch=epoch, mode=mode)


def dataset_context(model, dataset):
    obs = get_observer(model)
    if obs is not None:
        obs.set_context(**{**(obs.context or {}), "dataset": dataset})


def save_checkpoint(model, config, task, epoch):
    obs = get_observer(model)
    if obs is None or not config.diagnostics_save_checkpoints or obs.writer.identity["rank"] != 0:
        return
    from modeling_qwen2 import save_cl_state
    path = obs.experiment_root / "checkpoints" / f"cl_state_stage{config.task_idx}_{task}_epoch{epoch}.pt"
    save_cl_state(model, str(path))
    manifest = dict(configuration=dict(vars(config)), stage=config.task_idx, epoch=epoch,
                    git_commit=obs.writer.identity["git_commit"],
                    model_revision=getattr((model.module if hasattr(model, "module") else model).config, "_commit_hash", None),
                    contents="all LoRA A/B/scalings, JL, positive/negative GMMs, disabled flags",
                    not_saved=["base weights", "optimizer", "scheduler", "RNG", "data cursor", "KV cache", "EMA"],
                    purpose="independent fixed-probe diagnostics; not exact training resumption")
    path.with_suffix(".json").write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")


def reproduction_stage(model, stage, dataset, results):
    obs = get_observer(model)
    if obs is None or obs.writer.identity["rank"] != 0:
        return
    root = obs.experiment_root / "reproduction"
    root.mkdir(parents=True, exist_ok=True)
    record = dict(obs.writer.identity, stage=stage, dataset=dataset,
                  epoch_index="index 0 zero-shot; index e+1 trained epoch e", **results)
    with (root / "original_metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(record), allow_nan=False) + "\n")


def reproduction_summary(model, original_dir):
    obs = get_observer(model)
    if obs is not None:
        obs.flush()
        if obs.writer.identity["rank"] == 0:
            dest = obs.experiment_root / "reproduction"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(original_dir) / "cl_results.csv", dest / "task_results.csv")

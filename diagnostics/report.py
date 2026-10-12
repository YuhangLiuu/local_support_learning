"""Append-only rank-local records and explicit offline pooled-count aggregation."""

import json
import math
import subprocess
from pathlib import Path


def git_identity(root=None):
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return json_safe(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class RecordWriter:
    def __init__(self, path, experiment_id, rank=0, world_size=1, git=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.identity = dict(experiment_id=experiment_id, rank=rank, world_size=world_size,
                             aggregation_scope="rank_local", **(git or git_identity()))

    def metadata(self, data):
        target = self.path / f"metadata.rank{self.identity['rank']}.json"
        target.write_text(json.dumps(json_safe({**data, **self.identity}), indent=2,
                                     ensure_ascii=False, allow_nan=False), encoding="utf-8")

    def write(self, family, context, module=None, owner=None, adapter_id=None,
              metric_name="", numerator=0., denominator=0, **extra):
        parts = (module or "").split(".")
        layer = int(parts[parts.index("layers") + 1]) if "layers" in parts else None
        record = dict(self.identity, dataset=None, stage=None, epoch=None, mode=None)
        record.update(context)
        record.update(module=module, owner=owner, layer_id=layer, adapter_id=adapter_id,
                      metric_name=metric_name, metric_value=numerator / denominator if denominator else None,
                      numerator=numerator, denominator=denominator, sample_count=denominator)
        record.update(extra)
        target = self.path / f"{family}.rank{self.identity['rank']}.jsonl"
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
        return record


def pool_records(records):
    """Pool sufficient statistics for the SAME metric/context across disjoint rank shards.

    Never average rank means. This describes the empirical mixture of rank-local
    models/data, not a synchronized-GMM model or deduplicated DistributedSampler.
    Histograms/correlation records remain local; only sum/count records pool here.
    """
    keys = ("experiment_id", "git_commit", "dataset", "stage", "epoch", "mode", "window",
            "module", "owner", "layer_id", "adapter_id", "metric_name", "peer_module", "peer_adapter_id", "statistic_scope")
    grouped = {}
    for r in records:
        if r.get("aggregation_rule", "").startswith("not_poolable") or any(k in r for k in ("histogram", "correlation_sums", "sample_ids")):
            raise ValueError("specialized records require their own reducer")
        key = tuple(r.get(k) for k in keys)
        if key not in grouped:
            grouped[key] = {**r, "numerator": 0., "denominator": 0, "ranks": set()}
        g = grouped[key]
        g["numerator"] += r["numerator"]
        g["denominator"] += r["denominator"]
        g["ranks"].add(r["rank"])
    for g in grouped.values():
        g.update(metric_value=g["numerator"] / g["denominator"] if g["denominator"] else None,
                 sample_count=g["denominator"], rank=None, aggregation_scope="pooled_rank_local",
                 ranks=sorted(g["ranks"]))
    return list(grouped.values())


def summarize_gate(records, dimensions=("layer_id", "adapter_id"), scope="owner"):
    """Token-exposure-weighted summaries; not 'any gate in layer open' rates.

    Owners count q/k/v once, matrix scope counts the actual three updates. Never
    mix scopes. Keep stage/epoch/dataset/mode and rank provenance in every group.
    """
    groups = {}
    dims = ("experiment_id", "git_commit", "stage", "epoch", "dataset", "mode", "rank", *dimensions)
    for record in records:
        if record.get("statistic_scope") != scope or record["metric_name"] != "activation_rate":
            continue
        key = tuple(record.get(d) for d in dims)
        group = groups.setdefault(key, {**dict(zip(dims, key)), "numerator": 0., "denominator": 0})
        group["numerator"] += record["numerator"]
        group["denominator"] += record["denominator"]
    for group in groups.values():
        group["metric_value"] = group["numerator"] / group["denominator"] if group["denominator"] else None
        group["aggregation_scope"] = f"{scope}_token_exposure_weighted"
    return list(groups.values())

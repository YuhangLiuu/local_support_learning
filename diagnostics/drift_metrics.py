"""Fixed-probe identities, bounded history and offline window-change proxies."""

from collections import deque
import hashlib
import json

import numpy as np


def probe_fingerprint(sample_ids, input_ids, attention_mask, preprocessing):
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("probe sample IDs must be unique")
    payload = dict(sample_ids=list(sample_ids), input_ids=np.asarray(input_ids).tolist(),
                   attention_mask=np.asarray(attention_mask).tolist(), preprocessing=preprocessing)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class DriftTracker:
    """Bounded per-probe first/previous snapshots; same owner and dimensions only."""
    def __init__(self, max_probes=16):
        self.max_probes = max_probes
        self.first = {}
        self.previous = {}

    def update(self, probe_id, fingerprint, stage, snapshots, epoch=None):
        key = (probe_id, fingerprint)
        if key not in self.first and len(self.first) >= self.max_probes:
            raise ValueError("fixed probe budget exhausted")
        if any(p == probe_id and f != fingerprint for p, f in self.first):
            raise ValueError("probe content/order/preprocessing changed")
        records = []
        for reference, collection in (("first", self.first), ("previous", self.previous)):
            if key not in collection:
                continue
            old_stage, old_epoch, old = collection[key]
            for owner_adapter, cur in snapshots.items():
                if owner_adapter not in old:
                    continue  # an adapter did not exist in the reference stage
                prev = old[owner_adapter]
                if not np.array_equal(prev["valid_positions"], cur["valid_positions"]):
                    raise ValueError("probe token positions changed")
                if prev["z"].shape != cur["z"].shape:
                    raise ValueError("incompatible owner feature dimensions")
                delta = np.linalg.norm(cur["z"] - prev["z"], axis=-1)
                flip = cur["mask"] != prev["mask"]
                measures = {("layer_output_drift_l2" if owner_adapter[1] is None else "projected_representation_drift_l2"):
                            (float(delta.sum()), len(delta))}
                if owner_adapter[1] is not None:
                    measures["gate_decision_flip"] = (int(flip.sum()), len(flip))
                if prev["margin"] is not None and cur["margin"] is not None:
                    shift = cur["margin"] - prev["margin"]
                    measures["gmm_margin_drift"] = (float(shift.sum()), len(shift))
                for score in ("positive", "negative"):
                    if prev.get(score) is not None and cur.get(score) is not None:
                        shift = cur[score] - prev[score]
                        measures[f"{score}_log_likelihood_drift"] = (float(shift.sum()), len(shift))
                for metric, (num, den) in measures.items():
                    records.append(dict(owner=owner_adapter[0], adapter_id=owner_adapter[1],
                                        reference=reference, reference_stage=old_stage,
                                        reference_epoch=old_epoch,
                                        stage=stage, metric_name=metric, numerator=num, denominator=den))
        # All arrays are already bounded CPU arrays, never autograd tensors.
        snapshot = {k: {n: None if v is None else v.copy() for n, v in s.items()} for k, s in snapshots.items()}
        self.first.setdefault(key, (stage, epoch, snapshot))
        self.previous[key] = (stage, epoch, snapshot)
        return records

    def save(self, path):
        """Numeric NPZ plus JSON metadata; no pickled model/data objects."""
        arrays, entries = {}, []
        for label, collection in (("first", self.first), ("previous", self.previous)):
            for (probe, fingerprint), (stage, epoch, snapshots) in collection.items():
                if not snapshots:
                    entries.append(dict(label=label, probe=probe, fingerprint=fingerprint,
                                        stage=stage, epoch=epoch, empty=True))
                for (owner, adapter), snapshot in snapshots.items():
                    fields = {}
                    for name, value in snapshot.items():
                        if value is not None:
                            key = f"a{len(arrays)}"
                            arrays[key] = value
                            fields[name] = key
                        else:
                            fields[name] = None
                    entries.append(dict(label=label, probe=probe, fingerprint=fingerprint, stage=stage, epoch=epoch,
                                        owner=owner, adapter=adapter, fields=fields))
        arrays["metadata"] = np.asarray(json.dumps(entries))
        np.savez_compressed(path, **arrays)

    def load(self, path):
        # Parse and validate into fresh collections. Loading must not combine old
        # stage metadata with new arrays, or mutate history on a rejected file.
        first, previous = {}, {}
        with np.load(path, allow_pickle=False) as archive:
            for row in json.loads(str(archive["metadata"])):
                if row["label"] not in {"first", "previous"}:
                    raise ValueError("invalid probe history collection")
                target = first if row["label"] == "first" else previous
                key = (row["probe"], row["fingerprint"])
                snapshot = target.setdefault(key, (row["stage"], row.get("epoch"), {}))[2]
                if row.get("empty"):
                    continue
                snapshot[(row["owner"], row["adapter"])] = {
                    name: None if field is None else archive[field].copy() for name, field in row["fields"].items()}
        if len(first) > self.max_probes:
            raise ValueError("saved history exceeds probe budget")
        self.first, self.previous = first, previous


class WindowTracker:
    """Explicit offline observation interface, NOT a task-boundary detector."""
    def __init__(self, size=128):
        if size <= 0:
            raise ValueError("window size must be positive")
        self.size = size
        self.values = deque(maxlen=size)
        self.previous = None

    def add(self, margin, activation):
        self.values.append((float(margin), float(activation)))
        if len(self.values) < self.size:
            return None
        values = np.asarray(self.values)
        current = values.mean(0)
        out = dict(margin_mean=float(current[0]), activation_rate=float(current[1]), sample_count=len(values))
        if self.previous is not None:
            out.update(margin_mean_change=float(current[0] - self.previous[0]),
                       activation_change=float(current[1] - self.previous[1]))
        self.previous = current
        self.values.clear()
        return out

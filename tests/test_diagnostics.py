import json
from pathlib import Path

import numpy as np
import pytest

from diagnostics.config import DiagnosticsConfig
from diagnostics.drift_metrics import DriftTracker, WindowTracker, probe_fingerprint
from diagnostics.gate_metrics import (coactivation_counts, diag_gmm_scores, gate_counts,
                                      score_distribution, valid_positions)
from diagnostics.report import RecordWriter, pool_records


def test_config_and_disabled_import():
    assert not DiagnosticsConfig().enabled
    with pytest.raises(ValueError):
        DiagnosticsConfig(every_n_forwards=0)


def test_padding_prefix_and_exact_counts():
    pad = np.array([[0, 1, 1, 1], [0, 0, 1, 1]], bool)
    prefix = np.array([[1, 0, 1, 1], [1, 1, 0, 1]], bool)
    valid = valid_positions(pad, prefix)
    mask = np.array([[1, 1, 1, 0], [1, 1, 1, 1]], bool)
    counts = gate_counts(mask, mask, valid)
    assert counts["activation_rate"] == (2, 3)
    assert counts["sequence_activation_rate"] == (2, 2)
    assert counts["switch_rate"] == (1, 1)
    assert gate_counts(~mask, mask, valid)["raw_applied_disagreement"] == (3, 3)
    assert valid_positions(pad, prefix, budget=2).sum() == 2


def test_coactivation_counts_each_token_once():
    valid = np.ones((1, 4), bool)
    masks = np.array([[[1, 0, 1, 0]], [[1, 1, 0, 0]]], bool)
    counts = coactivation_counts(masks, valid)
    assert counts["coactivation_rate"] == (1, 4)
    assert counts["all_closed_rate"] == (1, 4)
    assert counts["sequence_all_closed_rate"] == (0, 1)
    assert coactivation_counts(np.empty((0, 1, 4), bool), valid)["all_closed_rate"] == (4, 4)


def test_likelihood_and_responsibility():
    x = np.array([[0.], [1.]])
    score, resp = diag_gmm_scores(x, np.log([1.]), np.zeros((1, 1)), np.ones((1, 1)), True)
    np.testing.assert_allclose(score, [-.5*np.log(2*np.pi), -.5*(np.log(2*np.pi)+1)])
    np.testing.assert_allclose(resp.sum(1), 1.)
    distribution = score_distribution([-.1, 0., .1, np.nan], [-1, 0, 1], .1)
    assert distribution["near_zero"] == 3 and distribution["nonfinite"] == 1
    assert sum(distribution["histogram"]) == distribution["count"] == 3


def test_schema_and_rank_pooling(tmp_path):
    records = []
    for rank, num, den in [(0, 1, 2), (1, 9, 10)]:
        writer = RecordWriter(tmp_path, "test", rank, 2, git={"git_commit": "abc", "git_dirty": False})
        records.append(writer.write("gate_metrics", dict(dataset="d", stage=0, epoch=1, mode="train"),
                                    "model.layers.0.self_attn.q_proj", adapter_id=0,
                                    metric_name="activation_rate", numerator=num, denominator=den))
    pooled = pool_records(records)[0]
    assert pooled["metric_value"] == pytest.approx(10/12)
    assert pooled["aggregation_scope"] == "pooled_rank_local"
    assert pooled["ranks"] == [0, 1]
    assert pooled["layer_id"] == 0
    required = {"experiment_id", "git_commit", "dataset", "stage", "epoch", "adapter_id", "layer_id",
                "module", "owner", "metric_name", "metric_value", "sample_count", "numerator", "denominator"}
    for path in tmp_path.glob("*.jsonl"):
        assert required <= json.loads(path.read_text()).keys()
    separate = [dict(records[0], statistic_scope=s) for s in ("owner", "matrix")]
    assert len(pool_records(separate)) == 2


def test_fixed_probe_identity_and_drift():
    fp = probe_fingerprint(["a"], [[1, 2]], [[1, 1]], "v1")
    assert fp != probe_fingerprint(["a"], [[2, 1]], [[1, 1]], "v1")
    tracker = DriftTracker()
    key = ("model.layers.0.self_attn.q_proj", 0)
    first = {key: dict(z=np.zeros((2, 3)), mask=np.array([0, 1], bool),
                       margin=np.array([0., 1.]), valid_positions=np.array([0, 1]))}
    assert tracker.update("p", fp, 0, first) == []
    second = {key: {**first[key], "z": np.ones((2, 3)), "mask": np.array([1, 1], bool)}}
    records = tracker.update("p", fp, 1, second)
    flip = next(r for r in records if r["metric_name"] == "gate_decision_flip")
    assert flip["numerator"] == 1 and flip["denominator"] == 2 and flip["reference_stage"] == 0
    with pytest.raises(ValueError):
        tracker.update("p", "changed", 2, second)


def test_windows_are_observers_not_detectors():
    tracker = WindowTracker(2)
    assert tracker.add(1, 0) is None
    assert tracker.add(3, 1)["margin_mean"] == 2
    tracker.add(5, 1)
    assert tracker.add(7, 1)["margin_mean_change"] == 4


def test_probe_history_roundtrip(tmp_path):
    tracker = DriftTracker()
    snapshots = {("model.layers.0.q_proj", 0): dict(z=np.zeros((2,3)), margin=None,
        mask=np.array([1,0], bool), valid_positions=np.array([0,1]))}
    tracker.update("p", "fp", 0, snapshots)
    path = tmp_path / "history.npz"
    tracker.save(path)
    restored = DriftTracker()
    restored.load(path)
    records = restored.update("p", "fp", 1, snapshots)
    assert all(r["numerator"] == 0 for r in records)

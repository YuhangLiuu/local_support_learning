import numpy as np
import pytest

from diagnostics.drift_metrics import DriftTracker


def test_empty_history_roundtrip_preserves_probe_identity(tmp_path):
    tracker=DriftTracker()
    tracker.update("empty","original",0,{})
    tracker.save(tmp_path/"history.npz")
    restored=DriftTracker();restored.load(tmp_path/"history.npz")
    assert ("empty","original") in restored.first
    with pytest.raises(ValueError,match="changed"):
        restored.update("empty","changed",1,{})


def snapshot():
    return {("model.layers.0.q_proj",0):dict(z=np.zeros((1,2)),mask=np.array([True]),
                margin=None,valid_positions=np.array([0]))}


def test_history_load_replaces_old_state_instead_of_mixing_stages(tmp_path):
    saved=DriftTracker();saved.update("p","fp",3,snapshot(),epoch=2);saved.save(tmp_path/"history.npz")
    restored=DriftTracker();restored.update("p","fp",0,snapshot(),epoch=1)
    restored.load(tmp_path/"history.npz")
    rows=restored.update("p","fp",4,snapshot())
    assert all(r["reference_stage"] == 3 and r["reference_epoch"] == 2 for r in rows)


def test_over_budget_load_leaves_previous_history_intact(tmp_path):
    saved=DriftTracker();saved.update("a","a",0,snapshot());saved.update("b","b",0,snapshot())
    saved.save(tmp_path/"history.npz")
    restored=DriftTracker(max_probes=1);restored.update("original","fp",0,snapshot())
    with pytest.raises(ValueError,match="budget"):
        restored.load(tmp_path/"history.npz")
    assert set(restored.first) == {("original","fp")}
    assert set(restored.previous) == {("original","fp")}

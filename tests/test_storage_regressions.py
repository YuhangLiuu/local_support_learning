"""Regressions found during the CPU observability audit."""
import json

import pytest

from diagnostics.report import RecordWriter, pool_records


def row(**extra):
    return dict(experiment_id="run", git_commit="commit", rank=0, metric_name="gate_flip",
                numerator=1, denominator=4, **extra)


def test_reference_and_counterfactual_contexts_never_pool_together():
    records = [row(reference="first", reference_stage=0, reference_epoch=0),
               row(reference="previous", reference_stage=1, reference_epoch=2)]
    assert len(pool_records(records)) == 2
    assert len(pool_records([row(comparison_mode="base"), row(comparison_mode="adapter_off")])) == 2
    assert len(pool_records([row(probe_fingerprint="a"), row(probe_fingerprint="b")])) == 2


@pytest.mark.parametrize("extra", [dict(posterior_sum=[1., 2.], hard_counts=[1, 2]),
                                    dict(error_type="ValueError", error="fit failure")])
def test_structured_metrics_require_specialized_reducer(extra):
    with pytest.raises(ValueError, match="reducer"):
        pool_records([row(**extra)])


def test_output_directory_rejects_different_experiment_and_preserves_metadata(tmp_path):
    one = RecordWriter(tmp_path, "one", git={"git_commit": "commit"})
    one.metadata({"configuration": {"seed": 7}})
    before = (tmp_path / "metadata.rank0.json").read_bytes()
    one.write("gate_metrics", {}, numerator=1, denominator=2)
    with pytest.raises(ValueError, match="experiment"):
        RecordWriter(tmp_path, "two", git={"git_commit": "commit"})
    assert (tmp_path / "metadata.rank0.json").read_bytes() == before
    assert json.loads((tmp_path / "gate_metrics.rank0.jsonl").read_text())["experiment_id"] == "one"


def test_different_experiment_cannot_append_even_without_metadata(tmp_path):
    RecordWriter(tmp_path, "one", git={}).write("gate_metrics", {}, numerator=1, denominator=2)
    with pytest.raises(ValueError, match="experiment"):
        RecordWriter(tmp_path, "two", git={})

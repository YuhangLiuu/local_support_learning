"""CPU-only full model/real EM acceptance tests, subprocess-isolated A/B/C/D."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("pomegranate",reason="full CPU integration requires real pomegranate; no substitute")
from importlib.metadata import version
if version("transformers") != "4.56.2":
    pytest.skip("full integration targets repository Transformers 4.56.2",allow_module_level=True)

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root=tmp_path_factory.mktemp("full-qwen-real-em")
    result={}
    for mode in ("baseline","off","basic","extended"):
        target=root/mode
        command=[sys.executable,"-X","utf8",str(ROOT/"tests/tiny_end_to_end_runner.py"),"--mode",mode,"--output",str(target)]
        env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","PYTHONHASHSEED":"0"}
        proc=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,encoding="utf-8",timeout=180)
        target.mkdir(exist_ok=True)
        (target/"process.log").write_text(proc.stdout+proc.stderr,encoding="utf-8")
        assert proc.returncode==0,(mode,proc.stdout,proc.stderr)
        result[mode]=dict(path=target,summary=json.loads((target/"summary.json").read_text()),
                          snapshots=torch.load(target/"snapshots.pt",weights_only=False,map_location="cpu"))
    return result


def compare(a,b,errors):
    if isinstance(a,torch.Tensor):
        assert a.dtype==b.dtype and a.shape==b.shape
        if a.numel() and a.is_floating_point():
            left,right=a.double(),b.double()
            diff=(left-right).abs()
            errors["max_abs"]=max(errors["max_abs"],float(diff.max()))
            errors["max_rel"]=max(errors["max_rel"],float((diff/torch.maximum(left.abs(),right.abs()).clamp_min(1e-12)).max()))
        assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k],errors)
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y,errors)
    else:
        assert a==b


@pytest.mark.cpu_integration
@pytest.mark.real_em
@pytest.mark.parametrize("mode",["off","basic","extended"])
def test_full_qwen_four_modes_real_em_bitwise(runs,mode):
    errors={"max_abs":0.,"max_rel":0.}
    compare(runs["baseline"]["snapshots"],runs[mode]["snapshots"],errors)
    assert errors==dict(max_abs=0.,max_rel=0.)
    (runs[mode]["path"]/"parity.json").write_text(json.dumps(dict(**errors,bitwise=True,reference="baseline-lsl-original")),encoding="utf-8")


@pytest.mark.cpu_integration
@pytest.mark.real_em
def test_actual_em_counts_freezing_roundtrip_no_extra_inline_work(runs):
    a=runs["baseline"]["summary"]
    assert "baseline_source" in a["source"]
    assert all(c["summary"]["counts"]==a["counts"] for c in runs.values())
    assert a["counts"]==dict(model_forward=13,backbone_forward=16,fit_pos=16,fit_neg=8,predict=300)
    for mode,r in runs.items():
        assert all(s["active_owners"]==8 for s in r["summary"]["stages"])
        assert all(np.isfinite(s["losses"]).all() for s in r["summary"]["stages"])
        assert len(r["summary"]["stages"])==2
        for stage in (0,1):
            before=r["snapshots"][f"stage{stage}/eval"]
            after=r["snapshots"][f"stage{stage}/roundtrip"]
            compare(before["logits"],after["logits"],dict(max_abs=0.,max_rel=0.))
            compare(before["gates"],after["gates"],dict(max_abs=0.,max_rel=0.))
        if mode in {"baseline","off"}:
            assert not (r["path"]/"diagnostics").exists()
        else:
            assert r["summary"]["independent_probe_counts"]==dict(model_forward=8,backbone_forward=8,
                                                                  fit_pos=0,fit_neg=0,predict=98)


def rows(path,family):
    return [json.loads(l) for l in (path/f"{family}.rank0.jsonl").read_text().splitlines()]


@pytest.mark.cpu_integration
@pytest.mark.parametrize("mode",["basic","extended"])
def test_real_model_diagnostics_semantics_against_same_forward(runs,mode):
    r=runs[mode];root=r["path"]/"diagnostics"
    gate=rows(root,"gate_metrics");gmm=rows(root,"gmm_metrics")
    for stage in (0,1):
        valid=[x for x in gate if x["stage"]==stage and x["mode"]=="eval" and x["metric_name"]=="activation_rate"]
        for scope,count in (("owner",8*(stage+1)),("matrix",14*(stage+1))):
            assert len([x for x in valid if x.get("statistic_scope")==scope])==count
        assert all(x["dataset"]==f"synthetic{stage}" and x["epoch"]==0 and x["layer_id"] in {0,1} for x in valid)
        train=[x for x in gate if x["stage"]==stage and x["mode"]=="train" and x["adapter_id"]==stage and x["metric_name"]=="forced_on_rate"]
        assert len(train)==14 and all(x["metric_value"]==1. for x in train)
        observed=[x for x in gmm if x["stage"]==stage and x["mode"]=="train" and x["adapter_id"]==stage and x["metric_name"]=="score_observed_rate"]
        assert len(observed)==8 and all(x["metric_value"]==0. for x in observed)
    for ref in r["summary"]["references"]:
        def get(source,metric,scope=None):
            selected=[x for x in source if x["stage"]==ref["stage"] and x["mode"]=="eval"
                and x["module"]=="model.layers.0.self_attn.q_proj" and x["adapter_id"]==ref["adapter"]
                and x["metric_name"]==metric and (scope is None or x.get("statistic_scope")==scope)]
            assert len(selected)==1
            return selected[0]
        rate=get(gate,"activation_rate","owner")
        assert (rate["numerator"],rate["denominator"])==(ref["activation"],ref["count"])
        raw=get(gate,"raw_activation_rate","owner")
        assert raw["numerator"]==ref["raw"]
        assert get(gmm,"margin")["metric_value"]==pytest.approx(ref["margin"],abs=2e-6)
        if mode=="extended":
            assert get(rows(root,"adapter_metrics"),"update_l2")["metric_value"]==pytest.approx(ref["update_norm"],abs=1e-8)
    from diagnostics.report import pool_records
    simple=[x for x in gate if x["metric_name"]=="activation_rate"]
    pooled=pool_records(simple)
    assert all(x["metric_value"]==x["numerator"]/x["denominator"] for x in pooled)
    assert all(x["sample_count"]==x["denominator"] for x in simple)


@pytest.mark.cpu_integration
def test_real_extended_crosslayer_coactivation_cosine_and_fixed_probe(runs):
    root=runs["extended"]["path"]
    gate=rows(root/"diagnostics","gate_metrics")
    adapter=rows(root/"diagnostics","adapter_metrics")
    assert any(x["metric_name"]=="coactivation_rate" and x["stage"]==1 and x["numerator"]>0 for x in gate)
    assert any(x["metric_name"]=="applied_update_cosine" and x["denominator"]>0 for x in adapter)
    cosine=[r for r in adapter if r["stage"]==1 and r["mode"]=="eval" and r["module"]=="model.layers.0.self_attn.q_proj"
            and r["metric_name"]=="applied_update_cosine"]
    ref=runs["extended"]["summary"]["cosine_reference"]
    assert len(cosine)==1 and cosine[0]["denominator"]==ref["denominator"]
    assert cosine[0]["numerator"]==pytest.approx(ref["numerator"],abs=1e-7)
    assert any(x["metric_name"]=="adjacent_layer_disagreement" for x in gate)
    # Reconstruct owner masks directly from executed smoother outputs, with the
    # same fixed padding/prefix token positions; compare counts, not file presence.
    prefix_mask=torch.tensor([[False,False,False,*([True]*10)], [False,False,*([True]*11)]])
    for stage in (0,1):
        events=runs["extended"]["snapshots"][f"stage{stage}/eval"]["gates"]
        masks={(name,i):value for name,kind,i,value in events if kind=="smooth"}
        for (name,i),mask in masks.items():
            if not any(name.endswith(t) for t in ("q_proj","o_proj","gate_proj","down_proj")):
                continue
            layer=int(name.split("layers.")[1].split(".")[0])
            if layer==1:
                previous=name.replace("layers.1.","layers.0.")
                actual=[r for r in gate if r["stage"]==stage and r["mode"]=="eval" and r["module"]==name
                        and r["adapter_id"]==i and r["metric_name"]=="adjacent_layer_disagreement"]
                assert len(actual)==1
                assert actual[0]["numerator"]==int(((masks[(previous,i)]!=mask)&prefix_mask).sum())
                assert actual[0]["denominator"]==int(prefix_mask.sum())
        for name in {name for name,i in masks}:
            active=sum(masks[(name,i)].int() for i in range(stage+1))
            for metric,condition in (("coactivation_rate",active>1),("all_closed_rate",active==0)):
                actual=[r for r in gate if r["stage"]==stage and r["mode"]=="eval" and r["module"]==name
                        and r.get("statistic_scope")=="matrix" and r["metric_name"]==metric]
                assert len(actual)==1 and actual[0]["numerator"]==int((condition&prefix_mask).sum())
    # Every scalar statistic is finite when its denominator is nonzero.
    for source in (gate,adapter,rows(root/"diagnostics","gmm_metrics")):
        for row in source:
            if row["denominator"]:
                assert np.isfinite(row["metric_value"])
    drift=rows(root/"probe","drift_metrics")
    assert drift and all(x["stage"]==1 and x["reference_stage"]==0 and x["reference_epoch"]==0 for x in drift)
    assert {x["adapter_id"] for x in drift}=={None,0}
    assert any(x["metric_name"]=="gate_decision_flip" for x in drift)
    from diagnostics.drift_metrics import DriftTracker
    history=DriftTracker();history.load(root/"probe_history.npz")
    assert len(history.first)==1 and len(history.previous)==1


@pytest.mark.cpu_integration
def test_real_fixed_probe_drift_numerical_oracles(runs):
    root=runs["extended"]["path"]
    snapshots=torch.load(root/"probe_snapshots.pt",weights_only=False,map_location="cpu")
    drift=rows(root/"probe","drift_metrics")
    for key,current in snapshots[1].items():
        if key not in snapshots[0]:
            continue
        previous=snapshots[0][key]
        name,adapter=key
        np.testing.assert_array_equal(current["valid_positions"],previous["valid_positions"])
        expected={}
        delta=np.sqrt(((current["z"].astype(np.float64)-previous["z"])**2).sum(-1))
        expected["layer_output_drift_l2" if adapter is None else "projected_representation_drift_l2"]=(delta.sum(),len(delta))
        if adapter is not None:
            flips=(current["mask"]!=previous["mask"])
            expected["gate_decision_flip"]=(flips.sum(),len(flips))
        for field,metric in (("margin","gmm_margin_drift"),("positive","positive_log_likelihood_drift"),
                             ("negative","negative_log_likelihood_drift")):
            if current.get(field) is not None and previous.get(field) is not None:
                change=current[field].astype(np.float64)-previous[field]
                expected[metric]=(change.sum(),len(change))
        for metric,(num,den) in expected.items():
            result=[r for r in drift if r["module"]==name and r["adapter_id"]==adapter and r["metric_name"]==metric]
            assert len(result)==2 and {r["reference"] for r in result}=={"first","previous"}
            assert all(r["denominator"]==den for r in result)
            assert all(r["numerator"]==pytest.approx(num,abs=2e-6,rel=2e-7) for r in result)

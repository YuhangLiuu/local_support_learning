"""Numerical oracles independent of diagnostics implementation, not JSON smoke tests."""
import json
import math
import types

import numpy as np
import pytest
import torch
from torch import nn

from cpu_lsl_fixture import bind_imports, load_core
from diagnostics.collector import attach_diagnostics
from diagnostics.config import DiagnosticsConfig
from diagnostics.drift_metrics import DriftTracker
from diagnostics.efficiency_metrics import storage_inventory
from diagnostics.gate_metrics import adjacent_disagreement, coactivation_counts, diag_gmm_scores, gate_counts, score_distribution
from diagnostics.report import RecordWriter, pool_records, summarize_gate


@pytest.fixture
def core(monkeypatch):
    c = load_core(); bind_imports(monkeypatch,c)
    return c


class KnownNetwork(nn.Module):
    def __init__(self, core, layers=1, smoothing=1., adapters=2):
        super().__init__()
        self.model = nn.Module(); self.model._shared_prefix_ids = None
        self.model.layers = nn.ModuleList()
        self.register_buffer("inputs", torch.tensor([[0.,1.],[.5,1.],[1.,1.],[2.,1.]]))
        for _ in range(layers):
            layer = nn.Module(); layer.self_attn = nn.Module()
            base = nn.Linear(2,2,bias=False)
            with torch.no_grad():
                base.weight.copy_(torch.eye(2))
            q = core.ConditionalLoRALinear(base, True, "gmm", smoothing_alpha=smoothing)
            layer.self_attn.q_proj = q
            self.model.layers.append(layer)
            for i in range(adapters):
                q.add_lora(2,2,gmm_kwargs=dict(pos_kwargs={"n_components":1}, neg_kwargs={"n_components":1}))
                with torch.no_grad():
                    q.lora_A[i].copy_(torch.eye(2))
                    q.lora_B[i].copy_(torch.diag(torch.tensor([3.,4.] if i == 0 else [4.,3.])))
                for which,mean in (("pos",[0.,0.]),("neg",[2.,0.])):
                    q.cl_gmm[i].load_gmm_state(which,dict(priors=torch.ones(1),means=[torch.tensor(mean)],covs=[torch.ones(2)]))

    def forward(self,input_ids,attention_mask=None,use_cache=False):
        x = self.inputs[input_ids]
        outputs = [l.self_attn.q_proj(x,attention_mask) for l in self.model.layers]
        return types.SimpleNamespace(logits=sum(outputs))


def records(path,family):
    return [json.loads(l) for l in (path/f"{family}.rank0.jsonl").read_text().splitlines()]


def find(rows,metric,adapter=0,scope=None,layer=0):
    chosen = [r for r in rows if r["metric_name"] == metric and r["adapter_id"] == adapter
              and r["layer_id"] == layer and (scope is None or r.get("statistic_scope") == scope)]
    assert len(chosen) == 1, (metric,chosen)
    return chosen[0]


def test_gate_all_known_counts_and_alignment():
    valid = np.array([[1,1,1,1],[0,1,0,1]],bool)
    raw = np.array([[1,0,1,0],[1,1,1,0]],bool)
    smooth = np.array([[0,1,1,0],[0,0,0,1]],bool)
    result = gate_counts(raw,smooth,valid)
    assert result == {"activation_rate":(3,6),"raw_activation_rate":(3,6),
                      "sequence_activation_rate":(2,2),"raw_applied_disagreement":(4,6),"switch_rate":(2,3)}
    co = coactivation_counts(np.stack([raw,smooth]),valid)
    assert co == {"coactivation_rate":(1,6),"all_closed_rate":(1,6),
                  "sequence_coactivation_rate":(1,2),"sequence_all_closed_rate":(0,2)}
    assert adjacent_disagreement(raw,smooth,valid) == (4,6)
    empty = gate_counts(raw,smooth,np.zeros_like(valid))
    assert all(pair == (0,0) for pair in empty.values())


@pytest.mark.parametrize("training",[True,False])
def test_actual_collector_training_inference_margin_smoothing_update_oracle(core,tmp_path,training):
    m = KnownNetwork(core,smoothing=.3).train(training)
    obs = attach_diagnostics(m,DiagnosticsConfig(enabled=True,level="extended",every_n_forwards=1,
        save_token_traces=True),tmp_path,"oracle")
    obs.set_context(stage=1,epoch=2,dataset="known",mode="train" if training else "eval")
    ids = torch.tensor([[0,1,2,3]])
    out = m(ids,attention_mask=torch.ones_like(ids)).logits
    x = m.inputs.numpy()
    raw = np.array([1,1,0,0],bool)
    applied = raw if training else np.array([0,1,0,0],bool)
    current = np.ones(4,bool) if training else applied
    update0 = x @ np.diag([3.,4.]) * applied[:,None]
    update1 = x @ np.diag([4.,3.]) * current[:,None]
    np.testing.assert_array_equal(out.detach().numpy()[0],x+update0+update1)
    q = m.model.layers[0].self_attn.q_proj
    if training:
        assert q._smoothing_state == {}
    else:
        # s=.3,.51,.357,.2499; strict >.5 opens only token 1.
        np.testing.assert_allclose(q._smoothing_state[0].numpy(),[.2499],atol=3e-8,rtol=0)
    obs.close()
    gate = records(tmp_path,"gate_metrics")
    gmm = records(tmp_path,"gmm_metrics")
    adapter = records(tmp_path,"adapter_metrics")
    assert find(gate,"activation_rate",scope="owner")["numerator"] == applied.sum()
    assert find(gate,"raw_activation_rate",scope="owner")["numerator"] == 2
    assert find(gate,"raw_applied_disagreement",scope="owner")["numerator"] == (0 if training else 1)
    assert find(gate,"forced_on_rate",adapter=1)["metric_value"] == int(training)
    assert find(gmm,"score_observed_rate",adapter=1)["metric_value"] == int(not training)
    if training:
        assert not any(r["adapter_id"] == 1 and r["metric_name"] == "margin" for r in gmm)
        assert not any(r["adapter_id"] == 1 and r["metric_name"] == "raw_activation_rate" for r in gate)
    # Independent isotropic 2D Gaussian formula, margin [2,1,0,-2].
    pos = -np.log(2*np.pi)-.5*(x*x).sum(-1)
    neg = -np.log(2*np.pi)-.5*((x-np.array([2.,0.]))**2).sum(-1)
    for name,values in (("positive_log_likelihood",pos),("negative_log_likelihood",neg),("margin",pos-neg)):
        np.testing.assert_allclose(find(gmm,name)["metric_value"],values.mean(),atol=3e-7,rtol=0)
    for i,update in enumerate((update0,update1)):
        norms = np.sqrt((update**2).sum(-1))
        ratio = norms/np.sqrt((x*x).sum(-1))
        np.testing.assert_allclose(find(adapter,"update_l2",adapter=i)["metric_value"],norms.mean(),atol=3e-7,rtol=0)
        np.testing.assert_allclose(find(adapter,"update_base_norm_ratio",adapter=i)["metric_value"],ratio.mean(),atol=3e-7,rtol=0)
    keep = (np.linalg.norm(update0,axis=-1)*np.linalg.norm(update1,axis=-1)) > 0
    expected_cosine = (update0*update1).sum(-1)[keep]/(np.linalg.norm(update0,axis=-1)*np.linalg.norm(update1,axis=-1))[keep]
    cosine = find(adapter,"applied_update_cosine")
    assert cosine["denominator"] == int(keep.sum())
    np.testing.assert_allclose(cosine["metric_value"],expected_cosine.mean(),atol=1e-7,rtol=0)
    occupancy = find(gmm,"positive_component_occupancy")
    assert occupancy["posterior_sum"] == [4.] and occupancy["hard_counts"] == [4]
    correlation = find(adapter,"margin_update_norm_correlation")["correlation_sums"]
    assert correlation["n"] == 4
    np.testing.assert_allclose(correlation["sxy"],((pos-neg)*np.linalg.norm(update0,axis=-1)).sum(),atol=3e-6,rtol=0)


def test_multicomponent_gaussian_responsibility_and_strict_threshold(core):
    x = np.array([[-1.],[0.],[1.],[2.]],np.float64)
    prior = np.array([.25,.75]); means=np.array([[-1.],[1.]]); covs=np.array([[.5],[2.]])
    # Independent density calculation in probability space (well-conditioned tiny data).
    densities = np.column_stack([prior[j]/math.sqrt(2*math.pi*covs[j,0]) *
                                np.exp(-((x[:,0]-means[j,0])**2)/(2*covs[j,0])) for j in range(2)])
    expected = np.log(densities.sum(1)); expected_resp = densities/densities.sum(1)[:,None]
    actual,resp = diag_gmm_scores(x,np.log(prior),means,covs,True)
    np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0)
    np.testing.assert_allclose(resp,expected_resp,atol=1e-14,rtol=0)
    actual_t = core._diag_gmm_logprob(torch.from_numpy(x),torch.from_numpy(np.log(prior)),
                                    torch.from_numpy(means),torch.from_numpy(covs))
    np.testing.assert_allclose(actual_t.numpy(),expected,atol=1e-14,rtol=0)
    clf = core.GMMClassifierGPU({"n_components":2},{"n_components":2},"cpu")
    params = tuple(torch.from_numpy(v) for v in (np.log(prior),means,covs))
    clf.params_vectorized = {"pos":params,"neg":params}
    assert torch.equal(clf.predict(torch.from_numpy(x)),torch.zeros(4,dtype=torch.float64))
    assert not core.is_in_gmm(clf,torch.from_numpy(x),chunk_m=1).any()  # zero margin is closed


def test_actual_cross_layer_alignment_oracle(core,tmp_path):
    m=KnownNetwork(core,layers=2,adapters=1).eval()
    clf=m.model.layers[1].self_attn.q_proj.cl_gmm[0]
    clf.params_vectorized["pos"],clf.params_vectorized["neg"] = clf.params_vectorized["neg"],clf.params_vectorized["pos"]
    obs=attach_diagnostics(m,DiagnosticsConfig(enabled=True,level="extended",every_n_forwards=1),tmp_path,"layers")
    obs.set_context(stage=0,epoch=0,dataset="d",mode="eval")
    # Drop token 0, leaving layer gates [1,0,0] vs [0,0,1]: two of three disagree.
    m(torch.tensor([[0,1,2,3]]),attention_mask=torch.tensor([[0,1,1,1]]))
    obs.close()
    result=find(records(tmp_path,"gate_metrics"),"adjacent_layer_disagreement",layer=1)
    assert (result["numerator"],result["denominator"]) == (2,3)
    assert result["peer_module"] == "model.layers.0.self_attn.q_proj"


def test_multistage_fixed_probe_exact_drift_oracle(tmp_path):
    tracker=DriftTracker()
    key=("model.layers.0.self_attn.q_proj",0)
    def snapshot(z,mask,margin,pos,neg):
        return {key:dict(z=np.asarray(z,float),mask=np.asarray(mask,bool),margin=np.asarray(margin,float),
                         positive=np.asarray(pos,float),negative=np.asarray(neg,float),valid_positions=np.array([1,3]))}
    first=snapshot([[0,0],[0,0]],[0,1],[0,2],[1,4],[1,2])
    second=snapshot([[3,4],[0,0]],[1,1],[2,4],[4,7],[2,3])
    third=snapshot([[6,8],[0,0]],[1,0],[4,6],[7,10],[3,4])
    assert tracker.update("p","fixed",0,first,epoch=1) == []
    tracker.update("p","fixed",1,second,epoch=2)
    tracker.save(tmp_path/"history.npz")
    restored=DriftTracker(); restored.load(tmp_path/"history.npz")
    rows=restored.update("p","fixed",2,third,epoch=3)
    expected={"first":{"projected_representation_drift_l2":(10.,2),"gate_decision_flip":(2,2),
                      "gmm_margin_drift":(8.,2),"positive_log_likelihood_drift":(12.,2),"negative_log_likelihood_drift":(4.,2)},
              "previous":{"projected_representation_drift_l2":(5.,2),"gate_decision_flip":(1,2),
                          "gmm_margin_drift":(4.,2),"positive_log_likelihood_drift":(6.,2),"negative_log_likelihood_drift":(2.,2)}}
    for r in rows:
        assert (r["numerator"],r["denominator"]) == expected[r["reference"]][r["metric_name"]]
        assert r["reference_stage"] == (0 if r["reference"] == "first" else 1)
        assert r["reference_epoch"] == (1 if r["reference"] == "first" else 2)
    changed={key:{**third[key],"valid_positions":np.array([0,3])}}
    with pytest.raises(ValueError,match="positions"):
        restored.update("p","fixed",3,changed)


def test_parameter_storage_inventory_shares_owners_and_negative(core):
    from cpu_lsl_fixture import KernelQwenCPU
    m=KernelQwenCPU(core)
    m.add_stage(0);m.add_stage(1);m.add_stage(2)
    inv=storage_inventory(m)
    # Per layer: Q/O 2*(64+64)*4, K/V 2*(64+32)*4,
    # Gate/Up 2*(64+128)*4, Down (128+64)*4 = 4096 parameters per phase.
    assert inv["adapter_parameters"] == 2*3*4096
    assert inv["adapter_bytes"] == 2*3*4096*4
    assert inv["adapter_count"] == 2*7*3
    # Four owners per layer, three 64x8 projections plus Down 128x8.
    assert inv["jl_bytes"] == 2*(3*64*8+128*8)*4
    # One NEG object and three POS objects per owner. Parameter vectors K=2,D=8:
    # (priors=2 + means=16 + covs=16)*4=136 bytes per GMM.
    assert inv["gmm_count"] == 2*4*(1+3)
    assert inv["gmm_model_parameter_bytes"] == 2*4*4*136
    # Vectors have log-priors+stacked means/covs separate from model parameters.
    # assign_fit explicitly reloads NEG each stage, so each adapter has separate
    # scoring vectors even though the original add_lora shares the NEG object.
    assert inv["gmm_scoring_bytes"] == 2*4*3*2*136
    assert inv["gmm_parameter_bytes"] == inv["gmm_model_parameter_bytes"]+inv["gmm_scoring_bytes"]


def test_multibatch_multirank_weighted_pool_roundtrip_empty_and_scope(tmp_path):
    rows=[]
    for rank,counts in ((0,[(1,2),(0,1)]),(1,[(8,10),(0,0)])):
        w=RecordWriter(tmp_path,"r",rank,2,git={"git_commit":"fixed","git_dirty":False})
        for num,den in counts:
            w.write("gate_metrics",dict(dataset="d",stage=1,epoch=2,mode="train"),
                    "model.layers.0.self_attn.q_proj",adapter_id=0,metric_name="activation_rate",
                    numerator=num,denominator=den,statistic_scope="owner")
        rows += [json.loads(l) for l in (tmp_path/f"gate_metrics.rank{rank}.jsonl").read_text().splitlines()]
    pooled=pool_records(rows)[0]
    assert (pooled["numerator"],pooled["denominator"],pooled["sample_count"]) == (9,13,13)
    assert pooled["metric_value"] == 9/13
    assert rows[-1]["metric_value"] is None
    assert len(pool_records([*rows,dict(rows[0],stage=2),dict(rows[0],experiment_id="another")])) == 3
    summary=summarize_gate(rows)
    assert len(summary) == 2 and {r["denominator"] for r in summary} == {3,10}


def test_score_chunk_offsets_and_sampling_denominators(core,tmp_path):
    m=KnownNetwork(core).train()
    q=m.model.layers[0].self_attn.q_proj
    q.is_in_cl_memory=types.MethodType(lambda self,i,x:core.is_in_gmm(self.cl_gmm[i],x,chunk_m=1),q)
    obs=attach_diagnostics(m,DiagnosticsConfig(enabled=True,level="extended",every_n_forwards=1,
        max_tokens_per_batch=2,save_token_traces=True),tmp_path,"chunks")
    obs.set_context(dataset="d",stage=1,epoch=0,mode="train")
    m(torch.tensor([[0,1,2,3]]),attention_mask=torch.tensor([[1,0,1,1]]))
    obs.close()
    # Flat positions 0,2 retained; scores [2,0] even though four scoring chunks
    # were computed, including chunks without any diagnostic positions.
    margin=find(records(tmp_path,"gmm_metrics"),"margin")
    assert margin["denominator"] == margin["sample_count"] == 2
    assert margin["metric_value"] == pytest.approx(1.,abs=3e-7)
    trace=find(records(tmp_path,"token_traces"),"gate_trace")
    assert trace["positions"] == [[0,0],[0,2]] and trace["decisions"] == [1,0]
    switch=find(records(tmp_path,"gate_metrics"),"switch_rate",scope="owner")
    assert switch["denominator"] == 0 and switch["metric_value"] is None  # gap is not adjacent


@pytest.mark.parametrize("condition",["no_adapters","unfitted","disabled","plain_lora"])
def test_collector_boundary_gate_semantics(core,tmp_path,condition):
    m=KnownNetwork(core,adapters=0 if condition=="no_adapters" else 1).eval()
    q=m.model.layers[0].self_attn.q_proj
    if condition=="unfitted":
        q.cl_gmm[0].params_vectorized["pos"]=None
    elif condition=="disabled":
        q.is_disabled[0]=True
    elif condition=="plain_lora":
        q.is_cl_lora=False
    obs=attach_diagnostics(m,DiagnosticsConfig(enabled=True,level="extended",every_n_forwards=1),tmp_path,"boundary")
    obs.set_context(stage=0,epoch=0,dataset="d",mode="eval")
    m(torch.tensor([[0,1,2,3]]),attention_mask=torch.ones(1,4))
    obs.close()
    gate=records(tmp_path,"gate_metrics")
    closed=find(gate,"all_closed_rate",adapter=None,scope="owner")
    assert closed["metric_value"] == (0 if condition=="plain_lora" else 1)
    if condition!="no_adapters":
        assert find(gate,"activation_rate",scope="owner")["metric_value"] == (1 if condition=="plain_lora" else 0)
        assert find(gate,"disabled_rate")["metric_value"] == (1 if condition=="disabled" else 0)
        assert find(records(tmp_path,"gmm_metrics"),"score_observed_rate")["metric_value"] == 0


def test_actual_collector_component_occupancy_entropy_oracle(core,tmp_path):
    m=KnownNetwork(core,adapters=1).eval()
    clf=m.model.layers[0].self_attn.q_proj.cl_gmm[0]
    priors=np.array([.25,.75],np.float32)
    means=np.array([[0.,0.],[2.,1.]],np.float32)
    covs=np.array([[1.,1.],[2.,.5]],np.float32)
    clf.params_vectorized["pos"]=(torch.log(torch.from_numpy(priors)),torch.from_numpy(means),torch.from_numpy(covs))
    obs=attach_diagnostics(m,DiagnosticsConfig(enabled=True,level="extended",every_n_forwards=1),tmp_path,"components")
    obs.set_context(dataset="known",stage=0,epoch=1,mode="eval")
    m(torch.tensor([[0,1,2,3]]),attention_mask=torch.ones(1,4))
    obs.close()
    x=m.inputs.numpy().astype(np.float64)
    weighted=np.column_stack([priors[j]/(2*math.pi*math.sqrt(float(covs[j].prod()))) *
        np.exp(-.5*(((x-means[j])**2)/covs[j]).sum(-1)) for j in range(2)])
    resp=weighted/weighted.sum(1)[:,None]
    rows=records(tmp_path,"gmm_metrics")
    occupancy=find(rows,"positive_component_occupancy")
    np.testing.assert_allclose(occupancy["posterior_sum"],resp.sum(0),atol=1e-7,rtol=0)
    assert occupancy["hard_counts"] == np.bincount(resp.argmax(1),minlength=2).tolist()
    entropy=-(resp*np.log(resp)).sum(1)
    assert find(rows,"component_entropy")["metric_value"] == pytest.approx(entropy.mean(),abs=1e-7)


def test_histogram_sufficient_statistics_nonfinite_known_answer():
    result=score_distribution([-2.,-.1,0.,.1,2.,np.nan,np.inf],[-1.,0.,1.],.1)
    assert result["histogram"] == [1,1,2,1]
    assert result["count"] == 5 and result["nonfinite"] == 2 and result["near_zero"] == 3
    assert result["sum"] == pytest.approx(0.,abs=1e-15)
    assert result["sum_sq"] == pytest.approx(8.02)

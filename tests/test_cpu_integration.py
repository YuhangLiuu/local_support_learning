"""Four-mode, multistage numerical audit using actual CPU LSL/Qwen kernels."""
import copy
import json
import random
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cpu_lsl_fixture import KernelQwenCPU, bind_imports, load_core
from diagnostics.collector import attach_diagnostics
from diagnostics.config import DiagnosticsConfig


@pytest.fixture
def core(monkeypatch):
    result = load_core()
    bind_imports(monkeypatch, result)
    return result


def batch():
    ids = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                        [0, 1, 2, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]])
    return ids, ids != 0


def track_gates(model):
    events = []
    for name, m in model.named_modules():
        if not isinstance(m, model.core.ConditionalLoRALinear):
            continue
        method, smoother = m.is_in_cl_memory, m._temporal_smoothing_filter
        def gate(self, i, x, method=method, name=name):
            out = method(i, x)
            events.append((name, "raw", i, out.detach().clone()))
            return out
        def smooth(self, mask, i, pad, method=smoother, name=name):
            out = method(mask, i, pad)
            events.append((name, "smoothed", i, out.detach().clone()))
            return out
        m.is_in_cl_memory = types.MethodType(gate, m)
        m._temporal_smoothing_filter = types.MethodType(smooth, m)
    return events


def tensors_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if left[key] is None:
            assert right[key] is None
        else:
            assert torch.equal(left[key], right[key]), key


def ema(model):
    return {(name, i): v.detach().clone() for name, m in model.named_modules()
            if isinstance(m, model.core.ConditionalLoRALinear) for i,v in m._smoothing_state.items()}


def internal_state(model):
    values={}
    for name,m in model.named_modules():
        if not isinstance(m,model.core.ConditionalLoRALinear):
            continue
        values[(name,"prefix")]=None if m.prefix_mask is None else m.prefix_mask.clone()
        values[(name,"hits")]=torch.tensor([m.hit_count,m.token_count,m.seq_hit_count,m.seq_count])
        for i,x in enumerate(m._pending_x):
            values[(name,"pending",i)]=x.clone()
        for i,clf in enumerate(m.cl_gmm):
            for which,params in clf.params_vectorized.items():
                for j,x in enumerate(params):
                    values[(name,"gmm",i,which,j)]=x.clone()
    return values


@pytest.mark.parametrize("seed", [0, 42, 2026])
@pytest.mark.parametrize("training", [True, False])
def test_four_modes_three_stages_real_decoder_kernels(core, tmp_path, seed, training):
    models, observers, events = [], [], []
    for mode in ("baseline", "off", "basic", "extended"):
        torch.manual_seed(seed)
        model = KernelQwenCPU(load_core(True) if mode == "baseline" else core)
        for stage in range(3):
            model.add_stage(stage)
        model.train(training)
        model.core.set_cl_accum_state(model,training)
        for module in model.modules():
            if isinstance(module,model.core.ConditionalLoRALinear):
                module.is_hit_rate_tracking=True
        models.append(model)
        obs = attach_diagnostics(model, DiagnosticsConfig(enabled=mode in {"basic", "extended"},
            level=mode if mode in {"basic", "extended"} else "basic", every_n_forwards=1,
            max_tokens_per_batch=11), tmp_path / mode, "four-mode")
        observers.append(obs)
        if obs:
            obs.set_context(dataset="fixed", stage=2, epoch=1, mode="train" if training else "eval")
        events.append(track_gates(model))
    for model in models[1:]:
        tensors_equal(models[0].state_dict(), model.state_dict())
    ids, pad = batch()
    rng = torch.get_rng_state().clone()
    numpy_before, python_before = np.random.get_state(), random.getstate()
    results = []
    for model, obs in zip(models, observers):
        torch.set_rng_state(rng)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        logits = model(ids, attention_mask=pad).logits
        labels = ids[:, 1:].clone().masked_fill(~pad[:, 1:], -100)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 32), labels.reshape(-1))
        loss.backward()
        grads = {n: None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()}
        assert any(v is not None and v.abs().max() > 0 for v in grads.values())
        optimizer.step()
        results.append((logits.detach(), loss.detach(), grads, copy.deepcopy(model.state_dict()),
                        ema(model), torch.get_rng_state().clone(),internal_state(model)))
        if obs:
            assert not obs.pending and not obs.score_bindings and obs.current_ticket is None
            assert not obs.active and not obs.last_snapshot
    for actual in results[1:]:
        expected = results[0]
        for i in (0, 1, 5):
            assert torch.equal(expected[i], actual[i])
        for i in (2, 3, 4, 6):
            tensors_equal(expected[i], actual[i])
    for event in events[1:]:
        assert len(event) == len(events[0])
        assert len(event) > 0
        for a, b in zip(events[0], event):
            assert a[:3] == b[:3] and torch.equal(a[3], b[3])
    assert np.array_equal(numpy_before[1], np.random.get_state()[1])
    assert numpy_before[0] == np.random.get_state()[0]
    assert numpy_before[2:] == np.random.get_state()[2:]
    assert python_before == random.getstate()
    for obs in observers:
        if obs:
            obs.close()
    assert not (tmp_path / "off").exists() and not (tmp_path / "baseline").exists()
    for mode in ("basic", "extended"):
        records = [json.loads(line) for line in (tmp_path/mode/"gate_metrics.rank0.jsonl").read_text().splitlines()]
        for scope, expected_count in (("owner", 8*3), ("matrix", 14*3)):
            rates = [r for r in records if r["metric_name"] == "activation_rate" and r.get("statistic_scope") == scope]
            assert len(rates) == expected_count
            assert all(r["denominator"] == 11 and r["sample_count"] == 11 for r in rates)
            assert all((r["stage"], r["epoch"], r["dataset"]) == (2, 1, "fixed") for r in rates)


def test_stage_growth_freezing_and_cl_state_roundtrip(core, tmp_path):
    torch.manual_seed(12)
    model = KernelQwenCPU(core)
    obs = attach_diagnostics(model, DiagnosticsConfig(enabled=True, level="extended", every_n_forwards=1),
                             tmp_path/"before", "stages")
    ids, pad = batch()
    for stage in range(3):
        old = {n:p.clone() for n,p in model.named_parameters()}
        model.add_stage(stage)
        model.train()
        obs.set_context(dataset=f"task{stage}", stage=stage, epoch=stage+1, mode="train")
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
        model(ids, attention_mask=pad).logits.square().mean().backward()
        opt.step()
        for n,p in model.named_parameters():
            if n in old:
                assert torch.equal(old[n],p), n
                assert not p.requires_grad and p.grad is None
        opt.zero_grad(set_to_none=True)
    model.eval()
    obs.set_context(dataset="restored", stage=2, epoch=3, mode="eval")
    expected = model(ids, attention_mask=pad).logits.detach()
    obs.close()
    core.save_cl_state(model, str(tmp_path/"cl.pt"))
    torch.manual_seed(12)
    restored = KernelQwenCPU(core).eval()  # same frozen random base, no adapters
    core.load_cl_state(restored, str(tmp_path/"cl.pt"))
    obs2 = attach_diagnostics(restored, DiagnosticsConfig(enabled=True, level="extended", every_n_forwards=1),
                              tmp_path/"after", "stages")
    obs2.set_context(dataset="restored", stage=2, epoch=3, mode="eval")
    actual = restored(ids, attention_mask=pad).logits.detach()
    assert torch.equal(expected,actual)
    obs2.close()
    for family in ("gate_metrics", "gmm_metrics", "adapter_metrics"):
        read = lambda path: [json.loads(l) for l in path.read_text().splitlines()]
        before = [r for r in read(tmp_path/"before"/f"{family}.rank0.jsonl") if r["dataset"] == "restored"]
        after = read(tmp_path/"after"/f"{family}.rank0.jsonl")
        for records in (before,after):
            for r in records:
                r.pop("forward_id", None)
        assert before == after
    rows = [json.loads(l) for l in (tmp_path/"before"/"gate_metrics.rank0.jsonl").read_text().splitlines()]
    for stage in range(3):
        assert {r["adapter_id"] for r in rows if r["stage"] == stage and r["metric_name"] == "activation_rate"} == set(range(stage+1))


def test_forward_exception_clears_hooks_bindings_graph_and_can_retry(core, tmp_path):
    model = KernelQwenCPU(core).train()
    model.add_stage(0); model.add_stage(1)
    obs = attach_diagnostics(model, DiagnosticsConfig(enabled=True, level="extended", every_n_forwards=1),
                             tmp_path, "exception")
    obs.set_context(stage=1, epoch=0, dataset="d", mode="train")
    ids,pad = batch()
    q = model.model.layers[0].self_attn.q_proj
    method = q.forward
    def fail(*args,**kwargs):
        method(*args,**kwargs)
        raise RuntimeError("after real q forward")
    q.forward = fail
    with pytest.raises(RuntimeError, match="after real"):
        model(ids, attention_mask=pad)
    assert obs.pending == [] and not obs.score_bindings and obs.current_ticket is None and not obs.active
    assert not any(hasattr(clf,"_lsl_diagnostics") for m in model.modules()
                   if isinstance(m,core.ConditionalLoRALinear) for clf in m.cl_gmm)
    q.forward = method
    model(ids, attention_mask=pad).logits.sum().backward()
    obs.close(); obs.close()
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert not any(hasattr(m,"_lsl_diagnostics") for m in model.modules())
    assert not hasattr(model,"_lsl_observer")
    # Reattach after removing both root hooks and module annotations.
    second = attach_diagnostics(model, DiagnosticsConfig(enabled=True), tmp_path, "exception")
    second.close()


def test_empty_valid_tokens_and_sample_budget_actual_prefix(core, tmp_path):
    model = KernelQwenCPU(core).eval(); model.add_stage(0)
    obs = attach_diagnostics(model, DiagnosticsConfig(enabled=True, every_n_forwards=1,
        max_tokens_per_batch=3), tmp_path,"empty")
    obs.set_context(dataset="d",stage=0,epoch=0,mode="eval")
    model(torch.tensor([[0,1,2]]), attention_mask=torch.tensor([[0,1,1]]))
    assert obs.sampled == 0 and not obs.totals and not obs.pending
    ids,pad = batch()
    model(ids, attention_mask=pad)
    np.testing.assert_array_equal(np.flatnonzero(obs.valid), [4,5,6])
    obs.close()
    rows = [json.loads(l) for l in (tmp_path/"gate_metrics.rank0.jsonl").read_text().splitlines()]
    assert all(r["denominator"] == 3 for r in rows if r["metric_name"] == "activation_rate")


@pytest.mark.parametrize("level", ["off", "basic", "extended"])
def test_actual_prefill_decode_dynamic_cache_gate_ema_rng(core,tmp_path,level):
    from transformers.cache_utils import DynamicCache
    torch.manual_seed(19)
    baseline=KernelQwenCPU(load_core(True));baseline.add_stage(0);baseline.add_stage(1);baseline.eval()
    torch.manual_seed(19)
    candidate=KernelQwenCPU(core);candidate.add_stage(0);candidate.add_stage(1);candidate.eval()
    obs=attach_diagnostics(candidate,DiagnosticsConfig(enabled=level!="off",level="basic" if level=="off" else level,
        every_n_forwards=1),tmp_path,"cache")
    if obs:
        obs.set_context(dataset="fixed",stage=1,epoch=0,mode="eval")
    cache_a,cache_b=DynamicCache(),DynamicCache()
    events_a,events_b=track_gates(baseline),track_gates(candidate)
    rng=torch.get_rng_state().clone()
    inputs=[(torch.tensor([[0,1,2,3,4],[1,2,5,6,7]]),torch.tensor([[0,1,1,1,1],[1,1,1,1,1]])),
            (torch.tensor([[8],[9]]),torch.tensor([[0,1,1,1,1,1],[1,1,1,1,1,1]])),
            (torch.tensor([[10],[11]]),torch.tensor([[0,1,1,1,1,1,1],[1,1,1,1,1,1,1]]))]
    with torch.no_grad():
        for ids,pad in inputs:
            a=baseline(ids,attention_mask=pad,use_cache=True,past_key_values=cache_a).logits
            b=candidate(ids,attention_mask=pad,use_cache=True,past_key_values=cache_b).logits
            assert torch.equal(a,b)
            tensors_equal(ema(baseline),ema(candidate))
            assert cache_a.get_seq_length() == cache_b.get_seq_length()
            for left,right in zip(cache_a.layers,cache_b.layers):
                assert torch.equal(left.keys,right.keys) and torch.equal(left.values,right.values)
    assert torch.equal(rng,torch.get_rng_state())
    assert len(events_a) == len(events_b)
    assert all(a[:3] == b[:3] and torch.equal(a[3],b[3]) for a,b in zip(events_a,events_b))
    if obs:
        obs.close()


def test_diagnostics_do_not_add_torch_matmuls_or_forward_calls(core,monkeypatch,tmp_path):
    # Extended's CPU responsibility calculations are summaries; all torch model,
    # GMM scoring and LoRA F.linear operations must occur exactly as in baseline.
    counts=[]
    for mode in ("baseline","off","basic","extended"):
        c=load_core(mode=="baseline")
        if mode!="baseline":
            bind_imports(monkeypatch,c)
        torch.manual_seed(33)
        m=KernelQwenCPU(c);m.add_stage(0);m.add_stage(1);m.train()
        obs=attach_diagnostics(m,DiagnosticsConfig(enabled=mode in {"basic","extended"},
            level=mode if mode in {"basic","extended"} else "basic",every_n_forwards=1),tmp_path/mode,"count")
        if obs:
            obs.set_context(dataset="d",stage=1,epoch=0,mode="train")
        n={"linear":0,"gmm_score":0,"root_forward":0}
        real_linear=F.linear
        def linear(*args,**kwargs):
            n["linear"]+=1
            return real_linear(*args,**kwargs)
        real_score=c.GMMClassifierGPU.predict.__globals__["_diag_gmm_logprob"]
        def score(*args,**kwargs):
            n["gmm_score"]+=1
            return real_score(*args,**kwargs)
        def forward_hook(*args):
            n["root_forward"]+=1
        handle=m.register_forward_hook(forward_hook)
        with monkeypatch.context() as scoped:
            scoped.setattr(F,"linear",linear)
            scoped.setitem(c.GMMClassifierGPU.predict.__globals__,"_diag_gmm_logprob",score)
            ids,pad=batch();m(ids,attention_mask=pad).logits.sum().backward()
        handle.remove()
        if obs:
            obs.close()
        counts.append(n)
    assert all(n==counts[0] for n in counts)
    assert counts[0]["root_forward"] == 1 and counts[0]["gmm_score"] == 28


def test_probe_runner_real_decoder_layers_multistage_and_state_purity(core,tmp_path):
    from diagnostics.probe_runner import ProbeRunner,isolated_state
    torch.manual_seed(52)
    m=KernelQwenCPU(core);m.add_stage(0);m.add_stage(1);m.train()
    m.model.layers[0].self_attn.eval()  # intentionally mixed mode tree
    ids,pad=batch()
    q=m.model.layers[0].self_attn.q_proj
    q._smoothing_state[0]=torch.tensor([.7,.2])
    q.lora_A[-1].grad=torch.ones_like(q.lora_A[-1])
    obs=attach_diagnostics(m,DiagnosticsConfig(enabled=True,level="probe",every_n_forwards=1,
        max_tokens_per_batch=4),tmp_path,"real-probe")
    runner=ProbeRunner(m)
    for stage in (1,2):
        if stage==2:
            m.add_stage(stage)
        initial=copy.deepcopy(m.state_dict())
        gradients={n:None if p.grad is None else p.grad.clone() for n,p in m.named_parameters()}
        modes={n:module.training for n,module in m.named_modules()}
        original_ema=ema(m)
        rng=torch.get_rng_state().clone()
        data=dict(input_ids=ids,attention_mask=pad,labels=ids.masked_fill(~pad,-100))
        result=runner.run("fixed",["a","b"],data,{"ids":"constant"},stage,epoch=1)
        tensors_equal(initial,m.state_dict())
        tensors_equal(gradients,{n:p.grad for n,p in m.named_parameters()})
        tensors_equal(original_ema,ema(m))
        assert modes == {n:module.training for n,module in m.named_modules()}
        assert torch.equal(rng,torch.get_rng_state())
        assert len(result["sampled_layer_outputs"]["full"]) == 2
        assert all(arr.shape == (4,64) for arr in result["sampled_layer_outputs"]["full"].values())
        assert all(not module._forward_hooks for module in m.model.layers)
        # Independent baseline-only forward on the same checkpoint. No adapters
        # and no cached state can influence these sampled base logits.
        with isolated_state(m),torch.no_grad():
            m.eval()
            for module in m.modules():
                if isinstance(module,core.ConditionalLoRALinear):
                    module.is_disabled=[True]*module.num_loras
            base=m(ids,attention_mask=pad).logits
        expected=base[torch.as_tensor(obs.valid)].float().cpu()
        assert torch.equal(expected,result["sampled_logits"]["base"])
        base_np=expected.numpy().astype(np.float64)
        full_np=result["sampled_logits"]["full"].numpy().astype(np.float64)
        def log_probs(v):
            v=v-v.max(-1,keepdims=True)
            return v-np.log(np.exp(v).sum(-1,keepdims=True))
        logbase,logfull=log_probs(base_np),log_probs(full_np)
        kl=(np.exp(logbase)*(logbase-logfull)).sum(-1).mean()
        rows=[json.loads(l) for l in (tmp_path/"probe_metrics.rank0.jsonl").read_text().splitlines()]
        recorded=[r for r in rows if r["stage"]==stage and r["metric_name"]=="kl_base_to_lsl"]
        assert len(recorded)==1
        assert recorded[0]["metric_value"] == pytest.approx(kl,abs=3e-7)
        with isolated_state(m),torch.no_grad():
            m.eval()
            full=m(ids,attention_mask=pad).logits
        assert torch.equal(full[torch.as_tensor(obs.valid)],result["sampled_logits"]["full"])
        shifted=log_probs(full[:,:-1].numpy().astype(np.float64))
        labels=ids[:,1:].numpy();keep=pad[:,1:].numpy()
        expected_nll=-np.take_along_axis(shifted,labels[...,None],axis=-1)[...,0][keep].mean()
        nll=[r for r in rows if r["stage"]==stage and r["mode"]=="probe_full" and r["metric_name"]=="answer_nll"]
        assert len(nll)==1 and nll[0]["denominator"]==int(keep.sum())
        assert nll[0]["metric_value"] == pytest.approx(expected_nll,abs=1e-6)
    drift=[json.loads(l) for l in (tmp_path/"drift_metrics.rank0.jsonl").read_text().splitlines()]
    assert all(r["stage"]==2 and r["reference_stage"]==1 for r in drift)
    assert not any(r["adapter_id"]==2 for r in drift)  # not present in historical checkpoint
    assert {r["adapter_id"] for r in drift} == {None,0,1}
    obs.close()


def test_checkpoint_recompute_does_not_collect_extra_forward_or_retain_graph(core,tmp_path):
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(74)
    m=KernelQwenCPU(core);m.add_stage(0);m.add_stage(1);m.train()
    # Only wrap decoder execution in test-owned activation checkpointing.
    # The original production gradient checkpointing mechanism stays untouched.
    for layer in m.model.layers:
        method=layer.forward
        def wrapped(*args,method=method,**kwargs):
            return checkpoint(method,*args,use_reentrant=False,**kwargs)
        layer.forward=wrapped
    obs=attach_diagnostics(m,DiagnosticsConfig(enabled=True,level="extended",every_n_forwards=1),tmp_path,"graph")
    obs.set_context(stage=1,epoch=0,dataset="d",mode="train")
    original_consume=obs._consume
    def inspect(value):
        if isinstance(value,torch.Tensor):
            assert value.grad_fn is None and not value.requires_grad
        elif isinstance(value,dict):
            for key,item in value.items():
                if key!="module":
                    inspect(item)
        elif isinstance(value,(tuple,list)):
            for item in value:
                inspect(item)
    def consume():
        inspect(obs.pending)
        original_consume()
    obs._consume=consume
    ids,pad=batch()
    before_ids,before_pad=ids.clone(),pad.clone()
    for _ in range(3):
        m.zero_grad(set_to_none=True)
        m(ids,attention_mask=pad).logits.square().mean().backward()
        assert not obs.pending and not obs.score_bindings and not obs.last_snapshot
    assert obs.calls == obs.sampled == obs.sequence == 3  # backward recompute not observed
    assert torch.equal(ids,before_ids) and torch.equal(pad,before_pad)
    obs.close()


@pytest.mark.parametrize("training",[True,False])
def test_failed_forward_has_same_original_algorithm_state_in_all_four_modes(core,tmp_path,training):
    states=[]
    for mode in ("baseline","off","basic","extended"):
        torch.manual_seed(82)
        m=KernelQwenCPU(load_core(True) if mode=="baseline" else core)
        m.add_stage(0);m.add_stage(1);m.train(training)
        m.core.set_cl_accum_state(m,training)
        obs=attach_diagnostics(m,DiagnosticsConfig(enabled=mode in {"basic","extended"},
            level=mode if mode in {"basic","extended"} else "basic",every_n_forwards=1),tmp_path/mode,"failed")
        if obs:
            obs.set_context(stage=1,epoch=0,dataset="d",mode="train" if training else "eval")
        q=m.model.layers[0].self_attn.q_proj
        original=q.forward
        def fail(*args,method=original,**kwargs):
            method(*args,**kwargs)
            raise RuntimeError("same failure after q")
        q.forward=fail
        ids,pad=batch()
        with pytest.raises(RuntimeError,match="same failure"):
            m(ids,attention_mask=pad)
        states.append((torch.get_rng_state().clone(),ema(m),internal_state(m),copy.deepcopy(m.state_dict())))
        if obs:
            assert not obs.pending and not obs.score_bindings and obs.current_ticket is None
            obs.close()
    for actual in states[1:]:
        assert torch.equal(actual[0],states[0][0])
        for i in (1,2,3):
            tensors_equal(actual[i],states[0][i])


def test_upstream_single_token_prefill_without_ema_fails_in_both_versions():
    # Record a baseline limitation, do not repair the published smoothing rule.
    for baseline in (True,False):
        c=load_core(baseline)
        q=c.ConditionalLoRALinear(torch.nn.Linear(2,2),True,"gmm",smoothing_alpha=.3).eval()
        q.add_lora(1,1,gmm_kwargs=dict(pos_kwargs={"n_components":1},neg_kwargs={"n_components":1}))
        with pytest.raises(KeyError) as error:
            q(torch.ones(1,1,2),torch.ones(1,1))
        assert error.value.args == (0,)

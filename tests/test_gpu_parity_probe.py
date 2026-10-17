import json
import random
from pathlib import Path

import pytest
import torch

from scripts.gpu_parity_test import build_synthetic_probe, synthetic_probe_stats


def test_probe_is_deterministic_and_seed_sensitive():
    first=build_synthetic_probe(123,128)
    second=build_synthetic_probe(123,128)
    other=build_synthetic_probe(456,128)
    assert first==second and first!=other
    assert len(set(first))==128


def test_probe_has_variable_lengths_and_no_padding_claim():
    rows=build_synthetic_probe(123,64)
    lengths={len(row.split()) for row in rows}
    assert len(lengths)>10
    # Probe construction itself contains no padded IDs; fit_cl_memory's attention
    # mask is the authority for valid token counting.
    assert all(row.strip() for row in rows)


def test_actual_activation_stats_reject_low_diversity_and_nonfinite():
    with pytest.raises(ValueError,match='diversity'):
        synthetic_probe_stats(torch.zeros(2048,256),32)
    with pytest.raises(ValueError,match='NaN/Inf'):
        synthetic_probe_stats(torch.tensor([[float('nan'),0.],[1.,2.]]),2)
    stats=synthetic_probe_stats(torch.arange(32*8,dtype=torch.float32).reshape(32,8),32)
    assert stats['unique_activation_rows']==32 and stats['duplicate_ratio']==0
    assert stats['valid_tokens']==32 and stats['gmm_components']==32


def test_activation_stats_use_rows_and_report_variance():
    x=torch.cat([torch.arange(8,dtype=torch.float32).repeat(4,1),
                 torch.arange(4,dtype=torch.float32).reshape(4,1)],dim=1)
    stats=synthetic_probe_stats(x,2)
    assert stats['unique_activation_rows']==4
    assert stats['near_zero_variance_dims']>=1


def test_k2_and_k32_accept_diverse_rows():
    x=torch.randn(64,256,generator=torch.Generator().manual_seed(123))
    assert synthetic_probe_stats(x,2)['unique_activation_rows']>=2
    assert synthetic_probe_stats(x,32)['unique_activation_rows']>=32


def test_off_basic_extended_share_the_same_probe_pool():
    # The worker starts a fresh process per mode, but the construction is pure
    # seed -> text and therefore identical in each mode.
    assert build_synthetic_probe(123,256)==build_synthetic_probe(123,256)


from types import SimpleNamespace
from scripts.gpu_parity_test import parity_fit_config, fit_outcome, compare_parity, validate_gmm_receipt


@pytest.mark.parametrize('policy',[True,False])
def test_fit_policy_comes_from_config(policy):
    fit=parity_fit_config(SimpleNamespace(gmm_skip_on_fail=policy),SimpleNamespace(fit_tokens=2048))
    assert fit.gmm_skip_on_fail is policy
    # Execute a controlled fit call, including the strict failure path.
    def controlled_fit(config):
        if not config.gmm_skip_on_fail:raise RuntimeError('owner failed')
        return 3/4
    if policy:assert fit_outcome(0,'neg',controlled_fit(fit))==[0,'neg',.75]
    else:
        with pytest.raises(RuntimeError,match='owner failed'):controlled_fit(fit)


@pytest.mark.parametrize('ratio',[0.,-1.,1.1,float('nan'),float('inf')])
def test_invalid_or_all_disabled_fit_fails(ratio):
    with pytest.raises(ValueError):fit_outcome(0,'neg',ratio)


@pytest.mark.parametrize('difference',['ratio','owner','parameters',None])
def test_modes_compare_actual_owner_state_and_parameters(difference):
    import copy
    a=dict(gmm_active_ratios=[[0,'neg',.75]],gmm_disabled={'a':True,'b':False},
           gmm={'b':torch.tensor([1.,2.])})
    b=copy.deepcopy(a)
    if difference=='ratio':b['gmm_active_ratios'][0][2]=.5
    if difference=='owner':b['gmm_disabled']={'a':False,'b':True}
    if difference=='parameters':b['gmm']['b'][0]=3.
    errors=dict(max_abs=0.,max_rel=0.,bitwise=True)
    if difference:
        with pytest.raises(AssertionError):compare_parity(a,b,errors)
    else:
        compare_parity(a,b,errors)
        assert errors==dict(max_abs=0.,max_rel=0.,bitwise=True)


def test_receipt_requires_policy_and_dynamic_positive_ratios():
    config=SimpleNamespace(gmm_skip_on_fail=True,activate_cl_method=True)
    receipt=dict(gmm_skip_on_fail=True,gmm_active_ratios=[fit_outcome(0,'neg',.75),fit_outcome(0,'pos',.5),fit_outcome(1,'pos',1.)])
    validate_gmm_receipt(json.loads(json.dumps(receipt)),config)
    receipt['gmm_skip_on_fail']=False
    with pytest.raises(ValueError,match='policy'):validate_gmm_receipt(receipt,config)


@pytest.mark.parametrize('wrapped',[True,False])
def test_offline_reproducer_uses_production_classifier(tmp_path,monkeypatch,wrapped):
    import sys
    import types
    from scripts.reproduce_gmm_failure import run
    calls=[]
    class Classifier:
        def __init__(self,pos,neg,device):
            assert pos==neg and pos['covariance_type']=='diag' and device=='cpu'
            calls.append(pos.copy())
        def fit_neg(self,x):assert x.ndim==2 and torch.isfinite(x).all()
    monkeypatch.setitem(sys.modules,'cl_memory',types.SimpleNamespace(GMMClassifierGPU=Classifier))
    x=torch.arange(64,dtype=torch.float32).reshape(16,4)
    path=tmp_path/'input.pt';torch.save({'X':x,'k':2} if wrapped else x,path)
    result=run(path,2,seeds=(7,8))
    assert [c['random_state'] for c in calls]==[7,8,123]
    assert all(r['status']=='fit_ok' for r in result['runs'])
    assert result['diverse_control']=='fit_ok'


@pytest.mark.parametrize('policy',[True,False])
def test_real_layer_fit_failure_policy(monkeypatch,policy):
    # Execute the project's actual fit_layer_gmms with tiny real LoRA owners.
    # Only the deliberately failing/succeeding EM methods are controlled here.
    import logging
    from cpu_lsl_fixture import load_core,bind_imports,definitions,source
    core=load_core();bind_imports(monkeypatch,core)
    ns=dict(torch=torch,logger=logging.getLogger(__name__),tqdm=lambda x,**kw:x)
    definitions(source('cl_memory.py',False),ns,{'fit_layer_gmms'})
    layer=torch.nn.Module();layer.self_attn=SimpleNamespace(layer_idx=0)
    for name,fail in [('good',False),('bad',True)]:
        owner=core.ConditionalLoRALinear(torch.nn.Linear(4,4),is_cl_lora=True,cl_memory_type='gmm')
        owner.is_disabled=[False];owner.accum_lora_idx=0;owner._pending_x=[torch.ones(8,4)]
        def fit(x,fail=fail):
            if fail:raise IndexError('controlled empty component')
        owner.cl_gmm=[SimpleNamespace(fit_neg=fit)]
        layer.add_module(name,owner)
    if policy:
        ns['fit_layer_gmms'](layer,'neg',skip_on_fail=policy)
        assert layer.good.is_disabled==[False] and layer.bad.is_disabled==[True]
        assert layer.bad._neg_fit_failed is True
        ratio=sum(not m.is_disabled[0] for m in (layer.good,layer.bad))/2
        assert fit_outcome(0,'neg',ratio)==[0,'neg',.5]
    else:
        with pytest.raises(IndexError,match='empty component'):
            ns['fit_layer_gmms'](layer,'neg',skip_on_fail=policy)

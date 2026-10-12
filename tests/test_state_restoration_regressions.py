"""Exercise exception restoration with actual mutable LSL state and shared owners."""
import copy

import pytest
import torch

from cpu_lsl_fixture import KernelQwenCPU, bind_imports, load_core
from diagnostics.collector import attach_diagnostics
from diagnostics.config import DiagnosticsConfig
from diagnostics.probe_runner import isolated_state


def test_isolation_restores_mutated_lists_and_preserves_shared_aliases(monkeypatch):
    core=load_core();bind_imports(monkeypatch,core)
    m=KernelQwenCPU(core);m.add_stage(0)
    q=m.model.layers[0].self_attn.q_proj
    k=m.model.layers[0].self_attn.k_proj
    original_disabled=q.is_disabled
    original_pending=q._pending_x
    tensor=torch.ones(1,8)
    q._pending_x.append(tensor)
    with pytest.raises(RuntimeError):
        with isolated_state(m):
            q.is_disabled[0]=True
            q._pending_x.append(torch.zeros(1,8))
            raise RuntimeError("failure after list mutation")
    assert q.is_disabled is k.is_disabled is original_disabled
    assert q.is_disabled == [False]
    assert q._pending_x is original_pending
    assert len(q._pending_x) == 1 and q._pending_x[0] is tensor


def test_isolation_restores_observer_sampling_budget_on_failure(monkeypatch,tmp_path):
    core=load_core();bind_imports(monkeypatch,core)
    m=KernelQwenCPU(core);m.add_stage(0)
    obs=attach_diagnostics(m,DiagnosticsConfig(enabled=True),tmp_path,"restore")
    obs.set_context(stage=0,epoch=0,mode="train",dataset="d")
    before=copy.deepcopy(obs.budgets)
    with pytest.raises(RuntimeError):
        with isolated_state(m):
            obs.set_context(stage=0,epoch=0,mode="probe_full",dataset="p")
            obs.calls=3;obs.sampled=1
            obs.set_context(stage=0,epoch=0,mode="probe_base",dataset="p")
            raise RuntimeError("probe failed")
    assert obs.context == dict(stage=0,epoch=0,mode="train",dataset="d")
    assert obs.budgets == before and obs.calls == obs.sampled == 0
    obs.close()

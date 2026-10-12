"""Actual source classes via AST: no model, dataset, CUDA or Pomegranate required.

The baseline source is read from the immutable Git tag. This exercises actual
baseline/research Linear and GMM score arithmetic, not a duplicated mock of it.
EM/FlashAttention/end-to-end lm-eval remain explicit integration gates.
"""

import ast
import copy
import math
from pathlib import Path
import random
import subprocess
import sys
import types
from typing import Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
import torch.nn.functional as F

from diagnostics.collector import attach_diagnostics
from diagnostics.config import DiagnosticsConfig

ROOT = Path(__file__).resolve().parents[1]


def extracted(source, names, namespace):
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<actual-lsl-source>", "exec"), namespace)
    return namespace


def load_classes(baseline=False):
    def read(file):
        if baseline:
            return subprocess.check_output(["git", "show", f"baseline-lsl-original:{file}"], cwd=ROOT, text=True, encoding="utf-8")
        return (ROOT / file).read_text(encoding="utf-8")
    namespace = dict(torch=torch, np=np, nn=nn, F=F, math=math, Optional=Optional)
    extracted(read("cl_memory.py"), {"GMMClassifierGPU", "_diag_gmm_logprob", "is_in_gmm"}, namespace)
    classifier = namespace["GMMClassifierGPU"]
    def init(self, pos_kwargs, neg_kwargs, device="cpu"):
        self.device = device
        self.params_vectorized = {}
        for which, k in (("pos", 2), ("neg", 3)):
            self.params_vectorized[which] = (torch.full((k,), -math.log(k)),
                torch.full((k, 4), .2 if which == "pos" else -.3), torch.ones(k, 4))
        self.pos_gmm = self.neg_gmm = None
    classifier.__init__ = init  # construction only: original predict / is_in_gmm run unchanged
    extracted(read("modeling_qwen2.py"), {"ConditionalLoRALinear"}, namespace)
    return namespace["ConditionalLoRALinear"], classifier, namespace["_diag_gmm_logprob"]


class TinyModel(nn.Module):
    def __init__(self, linear_class):
        super().__init__()
        self.model = nn.Module()
        self.model._shared_prefix_ids = None
        self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        self.embedding = nn.Embedding(16, 4)
        self.dropout = nn.Dropout(.2)
        for layer in self.model.layers:
            layer.self_attn = nn.Module()
            q = linear_class(nn.Linear(4, 4), True, "gmm", smoothing_alpha=.2)
            q.add_lora(2, 4, gmm_kwargs=dict(pos_kwargs={}, neg_kwargs={}))
            q.add_lora(2, 4, gmm_kwargs=dict(pos_kwargs={}, neg_kwargs={}))
            for b in q.lora_B:
                with torch.no_grad():
                    b.fill_(.05)
            layer.self_attn.q_proj = q
        self.head = nn.Linear(4, 16)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        x = self.dropout(self.embedding(input_ids))
        for layer in self.model.layers:
            x = x + layer.self_attn.q_proj(x, attention_mask)
        return types.SimpleNamespace(logits=self.head(x))


@pytest.fixture
def classes(monkeypatch):
    cls, clf, logprob = load_classes()
    # Only replace the import used by the observer's isinstance checks.
    monkeypatch.setitem(sys.modules, "modeling_qwen2", types.SimpleNamespace(ConditionalLoRALinear=cls))
    monkeypatch.setitem(sys.modules, "cl_memory", types.SimpleNamespace(
        drop_shared_prefix_mask=lambda ids, mask, prefix: mask.bool()))
    return cls, clf, logprob


@pytest.mark.parametrize("level", ["basic", "extended"])
@pytest.mark.parametrize("training", [True, False])
def test_actual_baseline_vs_observed_forward_grad_step_rng_ema(classes, tmp_path, level, training):
    cls, _, _ = classes
    baseline_cls, _, _ = load_classes(baseline=True)
    torch.manual_seed(42)
    original = TinyModel(baseline_cls)
    candidate = TinyModel(cls)
    candidate.load_state_dict(original.state_dict())
    original.train(training); candidate.train(training)
    collector = attach_diagnostics(candidate, DiagnosticsConfig(enabled=True, level=level,
        every_n_forwards=1, max_tokens_per_batch=32), tmp_path, "consistency")
    collector.set_context(dataset="synthetic", stage=1, epoch=0, mode="train" if training else "eval")
    consume = collector._consume
    def inspect_and_consume():
        def inspect_tensors(value):
            if isinstance(value, torch.Tensor):
                assert not value.requires_grad and value.grad_fn is None
            elif isinstance(value, dict):
                for k,v in value.items():
                    if k != "module":
                        inspect_tensors(v)
            elif isinstance(value, (tuple, list)):
                for v in value:
                    inspect_tensors(v)
        inspect_tensors(collector.pending)
        consume()
    collector._consume = inspect_and_consume
    ids = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]])
    pad = ids != 0
    initial_rng = torch.get_rng_state().clone()
    original_opt = torch.optim.AdamW(original.parameters(), lr=1e-3)
    candidate_opt = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
    torch.set_rng_state(initial_rng)
    expected = original(ids, attention_mask=pad).logits
    loss_a = F.cross_entropy(expected.reshape(-1, 16), ids.reshape(-1))
    loss_a.backward()
    rng_after = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    actual = candidate(ids, attention_mask=pad).logits
    loss_b = F.cross_entropy(actual.reshape(-1, 16), ids.reshape(-1))
    loss_b.backward()
    assert torch.equal(expected, actual)
    assert torch.equal(loss_a, loss_b)
    assert torch.equal(rng_after, torch.get_rng_state())
    for a, b in zip(original.parameters(), candidate.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)
    original_opt.step(); candidate_opt.step()
    for a, b in zip(original.parameters(), candidate.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(original.model.layers, candidate.model.layers):
        ma, mb = a.self_attn.q_proj, b.self_attn.q_proj
        assert ma._smoothing_state.keys() == mb._smoothing_state.keys()
        for i in ma._smoothing_state:
            assert torch.equal(ma._smoothing_state[i], mb._smoothing_state[i])
        for ca, cb in zip(ma.cl_gmm, mb.cl_gmm):
            for which in ("pos", "neg"):
                for pa, pb in zip(ca.params_vectorized[which], cb.params_vectorized[which]):
                    assert torch.equal(pa, pb)
            samples = torch.zeros(3, 4)
            assert torch.equal(ca.predict(samples), cb.predict(samples))
    assert collector.pending == [] and collector.current_ticket is None
    assert collector.score_bindings == []
    assert collector.last_snapshot == {}  # normal basic/extended do not retain token snapshots
    collector.close()


def test_disabled_attach_has_no_hooks_or_attributes(classes, tmp_path):
    cls, _, _ = classes
    model = TinyModel(cls)
    rng = torch.get_rng_state().clone()
    assert attach_diagnostics(model, DiagnosticsConfig(), tmp_path, "off") is None
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert not hasattr(model, "_lsl_observer")
    assert torch.equal(rng, torch.get_rng_state())
    assert list(tmp_path.iterdir()) == []


def test_probe_exception_restores_states_and_gradients(classes, tmp_path):
    from diagnostics.probe_runner import isolated_state
    cls, _, _ = classes
    model = TinyModel(cls).train()
    q = model.model.layers[0].self_attn.q_proj
    q._smoothing_state[0] = torch.tensor([.2])
    q.lora_A[-1].grad = torch.ones_like(q.lora_A[-1])
    torch_rng, np_rng, py_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    with pytest.raises(RuntimeError):
        with isolated_state(model):
            model.eval(); q._smoothing_state = {}; q.is_disabled = [True, True]
            torch.rand(3); np.random.rand(2); random.random()
            raise RuntimeError("probe failure")
    assert model.training and q.training
    assert q.is_disabled == [False, False]
    assert torch.equal(q._smoothing_state[0], torch.tensor([.2]))
    assert torch.equal(q.lora_A[-1].grad, torch.ones_like(q.lora_A[-1]))
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert np.array_equal(np_rng[1], np.random.get_state()[1])
    assert py_rng == random.getstate()


def test_score_equivalence_and_gate_decisions(classes):
    _, clf, _ = classes
    _, baseline_clf, _ = load_classes(True)
    a, b = clf({}, {}), baseline_clf({}, {})
    x = torch.linspace(-2, 2, 40).reshape(10, 4)
    assert torch.equal(a.predict(x), b.predict(x))
    assert torch.equal(a.predict(x) > 0, b.predict(x) > 0)


def test_shared_owner_not_counted_as_three_gmms(classes, tmp_path):
    import json
    cls, _, _ = classes
    model = TinyModel(cls).eval()
    owner = model.model.layers[0].self_attn.q_proj
    # Sibling need not be called in TinyModel: the test calls it within root forward.
    sibling = copy.deepcopy(owner)
    sibling.cl_gmm, sibling.is_disabled = owner.cl_gmm, owner.is_disabled
    sibling.is_cl_memory_owner = False
    model.model.layers[0].self_attn.k_proj = sibling
    original_forward = model.forward
    def forward(input_ids, attention_mask=None):
        result = original_forward(input_ids, attention_mask)
        sibling(model.embedding(input_ids), attention_mask)
        return result
    model.forward = forward
    obs = attach_diagnostics(model, DiagnosticsConfig(enabled=True, every_n_forwards=1), tmp_path, "owners")
    obs.set_context(dataset="d", stage=0, epoch=0, mode="eval")
    model(torch.tensor([[1,2,3]]), attention_mask=torch.ones(1,3))
    obs.close()
    rows = [json.loads(line) for line in (tmp_path / "gate_metrics.rank0.jsonl").read_text().splitlines()]
    matrix = [r for r in rows if r["metric_name"] == "activation_rate" and r.get("statistic_scope") == "matrix"]
    owners = [r for r in rows if r["metric_name"] == "activation_rate" and r.get("statistic_scope") == "owner"]
    assert len(matrix) == 6 and len(owners) == 4  # 3 matrices vs 2 owners, 2 adapters each


def test_independent_probe_modes_restore_and_compare(classes, tmp_path):
    from diagnostics.probe_runner import ProbeRunner
    cls, _, _ = classes
    model = TinyModel(cls).train()
    obs = attach_diagnostics(model, DiagnosticsConfig(enabled=True, level="probe"), tmp_path, "probe")
    batch = dict(input_ids=torch.tensor([[1,2,3]]), attention_mask=torch.ones(1,3,dtype=torch.long))
    rng = torch.get_rng_state().clone()
    state = copy.deepcopy(model.state_dict())
    runner = ProbeRunner(model)
    result = runner.run("fixed", ["sample-1"], batch, {"rule":"fixed-v1"}, stage=1, epoch=0)
    assert set(result["sampled_logits"]) == {"full", "base", "all_gated_no_smoothing", "current_forced"}
    assert model.training and torch.equal(rng, torch.get_rng_state())
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])
    assert obs.context is None and obs.pending == []
    runner.run("fixed", ["sample-1"], batch, {"rule":"fixed-v1"}, stage=2, epoch=0)
    assert (tmp_path / "drift_metrics.rank0.jsonl").exists()
    obs.close()


def test_pomegranate_em_consistency_when_available(tmp_path):
    pytest.importorskip("pomegranate")
    from pomegranate.gmm import GeneralMixtureModel
    from pomegranate.distributions import Normal
    namespace = dict(torch=torch, np=np, Optional=Optional, GeneralMixtureModel=GeneralMixtureModel, Normal=Normal)
    current_source = (ROOT / "cl_memory.py").read_text(encoding="utf-8")
    original_source = subprocess.check_output(["git", "show", "baseline-lsl-original:cl_memory.py"], cwd=ROOT, text=True, encoding="utf-8")
    names = {"GMMClassifierGPU", "_diag_gmm_logprob"}
    current = extracted(current_source, names, dict(namespace))["GMMClassifierGPU"]
    original = extracted(original_source, names, dict(namespace))["GMMClassifierGPU"]
    kwargs = dict(n_components=2, covariance_type="diag", random_state=0)
    x = torch.randn(100, 4, generator=torch.Generator().manual_seed(7))
    a, b = original(kwargs, kwargs, "cpu"), current(kwargs, kwargs, "cpu")
    a.fit(x, x+2); b.fit(x, x+2)
    for which in ("pos", "neg"):
        for left,right in zip(a.params_vectorized[which], b.params_vectorized[which]):
            assert torch.equal(left,right)
    assert torch.equal(a.predict(x), b.predict(x))


def test_prefill_decode_gate_and_ema_bitwise(classes, tmp_path):
    cls, _, _ = classes
    baseline_cls, _, _ = load_classes(True)
    torch.manual_seed(8)
    a, b = TinyModel(baseline_cls).eval(), TinyModel(cls).eval()
    b.load_state_dict(a.state_dict())
    gates_a, gates_b = [], []
    def track(model, target):
        for layer in model.model.layers:
            m = layer.self_attn.q_proj
            method = m.is_in_cl_memory
            def tracked(self, i, x, method=method):
                out = method(i, x)
                target.append(out.clone())
                return out
            m.is_in_cl_memory = types.MethodType(tracked, m)
    track(a, gates_a); track(b, gates_b)
    obs = attach_diagnostics(b, DiagnosticsConfig(enabled=True, every_n_forwards=1), tmp_path, "decode")
    obs.set_context(dataset="synthetic", stage=1, epoch=0, mode="eval")
    for ids in (torch.tensor([[1,2,3], [0,4,5]]), torch.tensor([[6], [7]]), torch.tensor([[8], [9]])):
        mask = ids != 0
        assert torch.equal(a(ids, attention_mask=mask).logits, b(ids, attention_mask=mask).logits)
    assert len(gates_a) == len(gates_b)
    assert all(torch.equal(left, right) for left,right in zip(gates_a, gates_b))
    for la, lb in zip(a.model.layers, b.model.layers):
        for i in la.self_attn.q_proj._smoothing_state:
            assert torch.equal(la.self_attn.q_proj._smoothing_state[i], lb.self_attn.q_proj._smoothing_state[i])
    obs.close()


def test_context_budget_survives_nested_eval(classes, tmp_path):
    from diagnostics.integration import observation_context
    cls, _, _ = classes
    model = TinyModel(cls)
    obs = attach_diagnostics(model, DiagnosticsConfig(enabled=True, every_n_forwards=1,
        max_batches_per_context=1), tmp_path, "budget")
    obs.set_context(dataset="d", stage=0, epoch=0, mode="train")
    ids = torch.tensor([[1,2]])
    model(ids)
    assert obs.sampled == 1
    with observation_context(model, mode="eval"):
        model(ids)
        assert obs.sampled == 1
    sequence = obs.sequence
    model(ids)
    assert obs.sequence == sequence  # train context cap was not reset by eval
    obs.close()

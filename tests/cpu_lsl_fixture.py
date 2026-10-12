"""Execute real Git-tag/current LSL kernels in a dependency-limited CPU harness.

Only fitted-parameter containers and import/decorator infrastructure are replaced.
No replacement for gate, smoothing, LoRA, attention, MLP, RMSNorm, sharing or
CL serialization arithmetic. This is NOT a full PreTrainedModel or EM test.
"""
import ast
import math
import os
from pathlib import Path
import subprocess
import types

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


class ParameterNormal(nn.Module):
    """Serialization-only container, deliberately has no fit/scoring methods."""
    def __init__(self):
        super().__init__()
        self._initialized = False

    def _reset_cache(self):
        pass


class ParameterMixture(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.distributions = nn.ModuleList([ParameterNormal() for _ in range(k)])
        self._initialized = False


def source(file, baseline):
    if baseline:
        return subprocess.check_output(["git", "show", f"baseline-lsl-original:{file}"],
                                       cwd=ROOT, text=True, encoding="utf-8")
    return (ROOT / file).read_text(encoding="utf-8")


def definitions(text, namespace, names=None, before=None):
    nodes = []
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name == before:
                break
            if names is None or node.name in names:
                nodes.append(node)
    # Postponed annotations permit unused HF typing imports to remain unresolved.
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, "<real-lsl-cpu-kernels>", "exec"), namespace)


def load_core(baseline=False):
    ns = dict(torch=torch, nn=nn, F=F, np=np, math=math, os=os,
              dist=torch.distributed, ACT2FN={"silu": F.silu},
              GradientCheckpointingLayer=nn.Module,
              deprecate_kwarg=lambda *a, **k: lambda f: f,
              use_kernel_forward_from_hub=lambda *a, **k: lambda c: c)
    definitions(source("cl_memory.py", baseline), ns,
                {"GMMClassifierGPU", "_diag_gmm_logprob", "is_in_gmm", "gmm_state", "drop_shared_prefix_mask"})
    ns["GMMClassifierGPU"].init_gmm = staticmethod(lambda kwargs, device="cpu":
                                                 ParameterMixture(kwargs["n_components"]).to(device))
    ns["LORA_DEFAULT_TARGET_MODULES"] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    ns["CL_MEMORY_NON_OWNERS"] = ("k_proj", "v_proj", "up_proj")
    definitions(source("modeling_qwen2.py", baseline), ns, before="Qwen2PreTrainedModel")
    return types.SimpleNamespace(**ns)


def assign_fit(clf, dim, positive_shift=0., negative_shift=2., components=2):
    """Deterministic fitted parameters; real load/vectorize/predict functions run."""
    for which, shift in (("pos", positive_shift), ("neg", negative_shift)):
        means = [torch.full((dim,), shift + i*.25) for i in range(components)]
        covs = [torch.full((dim,), 1. + i*.5) for i in range(components)]
        clf.load_gmm_state(which, dict(priors=torch.full((components,), 1/components), means=means, covs=covs))


class KernelQwenCPU(nn.Module):
    """Actual two Qwen decoder blocks, small test-owned embedding/head/RoPE/mask.

    The outer HF model cannot import in this environment. Do not call this an
    end-to-end Qwen2ForCausalLM test or production loss/evaluation test.
    """
    def __init__(self, core):
        super().__init__()
        cfg = types.SimpleNamespace(hidden_size=64, intermediate_size=128, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, attention_dropout=.15, hidden_act="silu",
            rms_norm_eps=1e-6, layer_types=["full_attention"]*2, _attn_implementation="eager")
        self.model = nn.Module()
        self.model._shared_prefix_ids = torch.tensor([1, 2])
        self.model.layers = nn.ModuleList([core.Qwen2DecoderLayer(cfg, i) for i in range(2)])
        self.embedding = nn.Embedding(32, 64)
        self.head = nn.Linear(64, 32, bias=False)
        self.core = core
        core.apply_conditional_lora_wrapper(self, is_cl_lora=True, jl_k=8, jl_seed=17,
                                           cl_memory_type="gmm", smoothing_alpha=.3)

    def add_stage(self, stage):
        self.core.add_lora(self, r=4, alpha=8, gmm_kwargs=dict(
            pos_kwargs=dict(n_components=2, covariance_type="diag", random_state=0),
            neg_kwargs=dict(n_components=2, covariance_type="diag", random_state=0)))
        for m in self.modules():
            if isinstance(m, self.core.ConditionalLoRALinear):
                with torch.no_grad():
                    # Nonzero B ensures gradients/gate/update comparisons are meaningful.
                    m.lora_B[-1].copy_(torch.linspace(-.025, .025, m.lora_B[-1].numel()).reshape_as(m.lora_B[-1]))
                if m.is_cl_memory_owner:
                    assign_fit(m.cl_gmm[-1], 8, positive_shift=stage*.1)

    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
        pad = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        pad_current = pad[:, -input_ids.shape[1]:]
        self.core.set_shared_prefix_mask(self.model, input_ids, pad_current)
        x = self.embedding(input_ids)
        length = input_ids.shape[1]
        previous = 0 if past_key_values is None else past_key_values.get_seq_length()
        pos = torch.arange(previous, previous+length)
        inv = 1. / (10000. ** (torch.arange(0, 16, 2).float()/16))
        phase = pos[:, None] * inv[None]
        phase = torch.cat((phase, phase), -1)[None].expand(input_ids.shape[0], -1, -1)
        blocked = torch.arange(previous+length)[None, :] > pos[:, None]
        blocked = blocked[None, None] | ~pad[:, None, None, :].bool()
        causal = torch.zeros_like(blocked, dtype=x.dtype).masked_fill(blocked, torch.finfo(x.dtype).min)
        for layer in self.model.layers:
            x = layer(x, attention_mask=causal, position_embeddings=(phase.cos(), phase.sin()),
                      token_pad_mask=pad_current, past_key_values=past_key_values,
                      cache_position=pos, use_cache=use_cache)
        return types.SimpleNamespace(logits=self.head(x))


def bind_imports(monkeypatch, core):
    monkeypatch.setitem(__import__("sys").modules, "modeling_qwen2", core)
    monkeypatch.setitem(__import__("sys").modules, "cl_memory", core)

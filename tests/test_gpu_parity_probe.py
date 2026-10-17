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

"""Pure NumPy metrics, also used by lightweight dependency-free tests.

All denominators concern sampled valid positions. Sequence means a sampled row
within one forward, NOT a reconstructed stream across cached decode calls.
"""

import numpy as np


def valid_positions(pad, prefix_keep=None, budget=None):
    keep = np.asarray(pad, dtype=bool).copy()
    if prefix_keep is not None:
        if np.shape(prefix_keep) != keep.shape:
            raise ValueError("prefix/padding shape mismatch")
        keep &= prefix_keep
    if budget is not None:
        ids = np.flatnonzero(keep)
        keep.flat[ids[budget:]] = False
    return keep


def gate_counts(raw, applied, valid):
    raw, applied, valid = (np.asarray(a, dtype=bool) for a in (raw, applied, valid))
    if raw.shape != applied.shape or raw.shape != valid.shape:
        raise ValueError("gate shapes must agree")
    pairs = valid[:, 1:] & valid[:, :-1]
    return {
        "activation_rate": (int((applied & valid).sum()), int(valid.sum())),
        "raw_activation_rate": (int((raw & valid).sum()), int(valid.sum())),
        "sequence_activation_rate": (int((applied & valid).any(1).sum()), int(valid.any(1).sum())),
        "raw_applied_disagreement": (int(((raw != applied) & valid).sum()), int(valid.sum())),
        "switch_rate": (int(((applied[:, 1:] != applied[:, :-1]) & pairs).sum()), int(pairs.sum())),
    }


def coactivation_counts(masks, valid):
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[1:] != valid.shape:
        raise ValueError("masks must be [adapter,batch,token]")
    n = masks.sum(0)
    return {
        "coactivation_rate": (int(((n > 1) & valid).sum()), int(valid.sum())),
        "all_closed_rate": (int(((n == 0) & valid).sum()), int(valid.sum())),
        "sequence_coactivation_rate": (int(((n > 1) & valid).any(1).sum()), int(valid.any(1).sum())),
        "sequence_all_closed_rate": (int((~((n > 0) & valid).any(1) & valid.any(1)).sum()), int(valid.any(1).sum())),
    }


def score_distribution(values, edges, near_zero):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    hist, bins = np.histogram(finite, bins=[-np.inf, *edges, np.inf])
    return {
        "sum": float(finite.sum()), "sum_sq": float((finite ** 2).sum()),
        "count": int(finite.size), "nonfinite": int(values.size - finite.size),
        "near_zero": int((np.abs(finite) <= near_zero).sum()),
        "histogram": hist.tolist(), "edges": list(edges),
    }


def diag_gmm_scores(x, log_priors, means, covs, responsibilities=False):
    """Diagnostic fp64 CPU rescore; authoritative decisions come from original GPU gate."""
    x = np.asarray(x, dtype=np.float64)
    means, covs = np.asarray(means, dtype=float), np.asarray(covs, dtype=float)
    comp = -.5 * (((x[:, None] - means[None]) ** 2 / covs).sum(-1)
                  + np.log(2 * np.pi * covs).sum(-1)) + log_priors
    peak = comp.max(1, keepdims=True)
    weights = np.exp(comp - peak)
    normalizer = weights.sum(1, keepdims=True)
    score = (peak + np.log(normalizer)).ravel()
    if responsibilities:
        return score, weights / normalizer
    return score


def adjacent_disagreement(left, right, valid):
    return int(((left != right) & valid).sum()), int(valid.sum())

"""Function-impact proxies, not statements of causal conflict."""

import numpy as np


def update_statistics(update, base):
    un = np.linalg.norm(update, axis=-1)
    bn = np.linalg.norm(base, axis=-1)
    return {"update_l2": un, "update_base_norm_ratio": un / np.maximum(bn, 1e-12)}


def cosine_values(left, right):
    denom = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    keep = denom > 1e-12
    return (left * right).sum(-1)[keep] / denom[keep]


def correlation_sums(x, y):
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    return dict(n=int(x.size), sx=float(x.sum()), sy=float(y.sum()),
                sxx=float((x*x).sum()), syy=float((y*y).sum()), sxy=float((x*y).sum()))

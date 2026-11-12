"""Scoreboard metrics for a router's failure probabilities.

y = 1 means "this tier failed", p = the router's predicted probability of that.
- auc:   can the router RANK hard prompts above easy ones? (0.5 = random, 1.0 = perfect)
- brier: average squared gap between p and the 0/1 outcome (lower is better)
- ece:   do the probabilities mean what they say? (0 = perfectly honest)
"""

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def _check(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"y and p must have the same shape, got {y.shape} and {p.shape}")
    if p.size and (p.min() < 0.0 or p.max() > 1.0):
        raise ValueError("probabilities must be between 0 and 1")
    return y, p


def auc(y, p):
    y, p = _check(y, p)
    return float(roc_auc_score(y, p))


def brier(y, p):
    y, p = _check(y, p)
    return float(brier_score_loss(y, p))


def reliability_table(y, p, n_bins=10):
    """One row per non-empty probability bin: what was predicted vs. what happened."""
    y, p = _check(y, p)
    # p = 1.0 would land in bin n_bins, so clamp it into the last bin.
    bin_of = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        in_bin = bin_of == b
        if in_bin.any():
            rows.append(
                {
                    "low": b / n_bins,
                    "high": (b + 1) / n_bins,
                    "count": int(in_bin.sum()),
                    "mean_pred": float(p[in_bin].mean()),
                    "frac_fail": float(y[in_bin].mean()),
                }
            )
    return rows


def ece(y, p, n_bins=10):
    """Expected Calibration Error: bin-weighted average of |predicted - actual|."""
    y, p = _check(y, p)
    rows = reliability_table(y, p, n_bins)
    n = len(y)
    return float(sum(r["count"] / n * abs(r["frac_fail"] - r["mean_pred"]) for r in rows))

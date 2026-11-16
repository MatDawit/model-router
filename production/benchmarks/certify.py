"""Choose routing cutoffs WITH a statistical guarantee on the quality drop.

Steps: order all candidate cutoff pairs from safest to riskiest (using only the
router's own predictions, never the calibration outcomes), climb that ladder with
fixed-sequence testing, then pick the certified pair that saves the most money.
"""

import numpy as np

from benchmarks.pareto import evaluate_routing, route, routing_losses
from core.router.conformal import empirical_bernstein_upper_bound, fixed_sequence_certify


def build_ladder(p_small, p_mid, taus_small, taus_mid):
    """All (tau_small, tau_mid) pairs, ordered from safest to riskiest.

    "Risk" here is the router's PREDICTED failure rate for the tiers it would use,
    so the order depends on prompts and predictions only, not on any true outcome.
    """
    p_small, p_mid = np.asarray(p_small, dtype=float), np.asarray(p_mid, dtype=float)
    rungs = []
    for tau_small in taus_small:
        for tau_mid in taus_mid:
            tier = route(p_small, p_mid, tau_small, tau_mid)
            predicted_risk = np.where(tier == 0, p_small, np.where(tier == 1, p_mid, 0.0)).mean()
            rungs.append((float(predicted_risk), float(tau_small), float(tau_mid)))
    rungs.sort()  # lowest predicted risk first; ties broken by the cutoffs themselves
    return [(tau_small, tau_mid) for _, tau_small, tau_mid in rungs]


def certify_and_pick(df, p_small, p_mid, taus_small, taus_mid, epsilon=0.02, alpha=0.05, **models):
    """Return the certified cutoff pair with the biggest savings, or None if none qualifies.

    With probability at least 1 - alpha, the TRUE average quality drop of the returned
    pair is at most `epsilon`. `df` is the calibration set.
    """
    ladder = build_ladder(p_small, p_mid, taus_small, taus_mid)
    tiers = [route(p_small, p_mid, ts, tm) for ts, tm in ladder]
    losses = [routing_losses(df, tier, **models) for tier in tiers]

    certified = fixed_sequence_certify(losses, epsilon=epsilon, alpha=alpha)
    if not certified:
        return None

    # Every certified rung is valid at the same time, so choosing the cheapest is free.
    results = [evaluate_routing(df, tiers[i], **models) for i in certified]
    best = max(range(len(certified)), key=lambda k: results[k]["savings"])
    i = certified[best]
    return {
        "tau_small": ladder[i][0],
        "tau_mid": ladder[i][1],
        "upper_bound": empirical_bernstein_upper_bound(losses[i], alpha=alpha),
        "n_certified": len(certified),
        "n_candidates": len(ladder),
        **results[best],
    }

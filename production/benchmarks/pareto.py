"""Turn failure probabilities into routing decisions, then measure real cost and quality.

RouterBench recorded every model's score and cost on every prompt, so we can
"replay" any routing policy: look up what the chosen tier actually scored and cost.
"""

import numpy as np
import pandas as pd

from benchmarks.data_loaders.routerbench import FRONTIER, MID, SMALL


def route(p_small, p_mid, tau_small, tau_mid=None):
    """Pick a tier per prompt: 0 = small, 1 = mid, 2 = frontier.

    Use the cheapest tier whose predicted failure probability is below its tau.
    """
    tau_mid = tau_small if tau_mid is None else tau_mid
    p_small, p_mid = np.asarray(p_small, dtype=float), np.asarray(p_mid, dtype=float)
    tier = np.full(len(p_small), 2)           # default: frontier
    tier = np.where(p_mid < tau_mid, 1, tier)  # mid if it looks safe
    tier = np.where(p_small < tau_small, 0, tier)  # small wins when it looks safe
    return tier


def evaluate_routing(df, tier, small=SMALL, mid=MID, frontier=FRONTIER):
    """Replay a routing decision on the recorded scores and costs."""
    models = [small, mid, frontier]
    scores = np.stack([df[m].to_numpy() for m in models], axis=1)
    costs = np.stack([df[f"{m}|total_cost"].to_numpy() for m in models], axis=1)
    rows = np.arange(len(df))
    tier = np.asarray(tier)

    quality = scores[rows, tier].mean()
    frontier_quality = scores[:, 2].mean()
    cost = costs[rows, tier].sum()
    frontier_cost = costs[:, 2].sum()
    return {
        "quality": float(quality),
        "quality_drop": float(frontier_quality - quality),  # in score points, vs always-frontier
        "savings": float(1.0 - cost / frontier_cost),
        "pct_small": float((tier == 0).mean()),
        "pct_mid": float((tier == 1).mean()),
        "pct_frontier": float((tier == 2).mean()),
    }


def sweep(df, p_small, p_mid, taus, **models):
    """Evaluate the routing rule at each tau (one shared tau for both tiers)."""
    out = []
    for tau in taus:
        result = evaluate_routing(df, route(p_small, p_mid, tau), **models)
        out.append({"tau": float(tau), **result})
    return pd.DataFrame(out)


def sweep_grid(df, p_small, p_mid, taus_small, taus_mid, **models):
    """Try every (tau_small, tau_mid) pair: two separate dials, one per tier."""
    out = []
    for tau_small in taus_small:
        for tau_mid in taus_mid:
            tier = route(p_small, p_mid, tau_small, tau_mid)
            out.append(
                {"tau_small": float(tau_small), "tau_mid": float(tau_mid), **evaluate_routing(df, tier, **models)}
            )
    return pd.DataFrame(out)


def best_under_budget(curve, epsilon=0.02):
    """The row with the biggest savings whose quality drop is at most `epsilon`.

    If no row stays within the budget, return the row with the smallest drop.
    """
    ok = curve[curve["quality_drop"] <= epsilon + 1e-9]
    if ok.empty:
        return curve.loc[curve["quality_drop"].idxmin()]
    return ok.loc[ok["savings"].idxmax()]

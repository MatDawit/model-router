import numpy as np
import pandas as pd
import pytest

from benchmarks.certify import build_ladder, certify_and_pick, ladder_report
from benchmarks.pareto import routing_losses

MODELS = dict(small="S", mid="M", frontier="F")


def _world(n=4000):
    """Half the prompts are easy (small model fine), half are hard (small fails).
    The router is well informed: it gives easy prompts 0.05 and hard prompts 0.95."""
    easy = np.arange(n) < n // 2
    df = pd.DataFrame(
        {
            "S": np.where(easy, 1.0, 0.0),
            "M": 1.0,
            "F": 1.0,
            "S|total_cost": 1.0,
            "M|total_cost": 10.0,
            "F|total_cost": 100.0,
        }
    )
    p_small = np.where(easy, 0.05, 0.95)
    p_mid = np.full(n, 0.05)
    return df, p_small, p_mid


def test_losses_are_frontier_minus_chosen_tier():
    df = pd.DataFrame({"S": [1.0, 0.0], "M": [1.0, 1.0], "F": [1.0, 1.0],
                       "S|total_cost": [1, 1], "M|total_cost": [1, 1], "F|total_cost": [1, 1]})
    assert routing_losses(df, [2, 2], **MODELS).tolist() == [0.0, 0.0]
    assert routing_losses(df, [0, 0], **MODELS).tolist() == [0.0, 1.0]


def test_ladder_puts_the_safest_rung_first():
    _, p_small, p_mid = _world()
    ladder = build_ladder(p_small, p_mid, [0.1, 0.99], [0.5])
    assert ladder[0] == (0.1, 0.5)   # cautious about small
    assert ladder[-1] == (0.99, 0.5)  # trusts small even on hard prompts


def test_picks_the_cheapest_certified_pair_and_avoids_the_risky_one():
    df, p_small, p_mid = _world()
    pick = certify_and_pick(df, p_small, p_mid, [0.1, 0.5, 0.99], [0.1, 0.5], **MODELS)
    assert pick is not None
    assert pick["tau_small"] < 0.99          # never the rung that sends hard prompts to small
    assert pick["upper_bound"] <= 0.02
    assert pick["quality_drop"] == pytest.approx(0.0)
    assert pick["savings"] > 0.9


def test_ladder_report_marks_rungs_after_the_first_failure_as_not_reached():
    df, p_small, p_mid = _world()
    rows = ladder_report(df, p_small, p_mid, [0.1, 0.5, 0.99], [0.1, 0.5], **MODELS)
    assert len(rows) == 6
    assert rows[0]["reached"] is True
    first_fail = next(i for i, r in enumerate(rows) if not r["passes_alone"])
    assert all(not r["reached"] for r in rows[first_fail + 1:])  # nothing past a failure is tested


def test_returns_none_when_nothing_can_be_certified():
    df = pd.DataFrame({"S": 0.0, "M": 0.0, "F": 1.0,
                       "S|total_cost": 1.0, "M|total_cost": 10.0, "F|total_cost": 100.0}, index=range(3000))
    overconfident = np.full(3000, 0.01)  # the router wrongly thinks everything is safe
    assert certify_and_pick(df, overconfident, overconfident, [0.5], [0.5], **MODELS) is None

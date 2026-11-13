import numpy as np
import pandas as pd
import pytest

from benchmarks.pareto import best_under_budget, evaluate_routing, route, sweep

MODELS = dict(small="S", mid="M", frontier="F")


def _df():
    # Two prompts. S fails the second one. Costs: S=1, M=10, F=100 per prompt.
    return pd.DataFrame(
        {
            "S": [1.0, 0.0],
            "M": [1.0, 1.0],
            "F": [1.0, 1.0],
            "S|total_cost": [1.0, 1.0],
            "M|total_cost": [10.0, 10.0],
            "F|total_cost": [100.0, 100.0],
        }
    )


def test_route_picks_the_cheapest_tier_that_looks_safe():
    tier = route([0.1, 0.9, 0.9], [0.1, 0.1, 0.9], tau_small=0.5)
    assert tier.tolist() == [0, 1, 2]


def test_always_frontier_saves_nothing_and_loses_nothing():
    res = evaluate_routing(_df(), [2, 2], **MODELS)
    assert res["savings"] == pytest.approx(0.0)
    assert res["quality_drop"] == pytest.approx(0.0)
    assert res["pct_frontier"] == 1.0


def test_smart_routing_saves_money_without_losing_quality():
    # First prompt -> small (cost 1), second -> mid (cost 10): 11 of 200.
    res = evaluate_routing(_df(), [0, 1], **MODELS)
    assert res["savings"] == pytest.approx(1 - 11 / 200)
    assert res["quality_drop"] == pytest.approx(0.0)


def test_sending_a_failing_prompt_to_small_costs_quality():
    res = evaluate_routing(_df(), [0, 0], **MODELS)
    assert res["quality_drop"] == pytest.approx(0.5)
    assert res["savings"] == pytest.approx(1 - 2 / 200)


def test_sweep_returns_one_row_per_tau():
    curve = sweep(_df(), [0.1, 0.9], [0.1, 0.1], [0.3, 0.5, 0.95], **MODELS)
    assert curve["tau"].tolist() == [0.3, 0.5, 0.95]


def test_best_under_budget_picks_largest_savings_within_epsilon():
    curve = pd.DataFrame(
        {"tau": [0.1, 0.5, 0.9], "quality_drop": [0.0, 0.01, 0.2], "savings": [0.1, 0.4, 0.9]}
    )
    assert best_under_budget(curve, epsilon=0.02)["tau"] == 0.5


def test_best_under_budget_falls_back_to_the_smallest_drop():
    curve = pd.DataFrame({"tau": [0.1, 0.9], "quality_drop": [0.05, 0.3], "savings": [0.2, 0.9]})
    assert best_under_budget(curve, epsilon=0.02)["tau"] == 0.1

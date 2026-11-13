import numpy as np
import pytest

from benchmarks.metrics import per_group_auc, weighted_mean_auc


def test_skips_small_and_single_outcome_groups_and_scores_the_rest():
    y_a = np.array([0] * 60 + [1] * 60)           # group A: perfectly ranked
    p_a = np.array([0.1] * 60 + [0.9] * 60)
    y_b = np.zeros(150, dtype=int)                # group B: only one outcome -> skipped
    p_b = np.full(150, 0.5)
    y_c = np.array([0, 1] * 5)                    # group C: too few rows -> skipped
    p_c = np.full(10, 0.5)

    groups = ["A"] * 120 + ["B"] * 150 + ["C"] * 10
    table = per_group_auc(groups, np.r_[y_a, y_b, y_c], np.r_[p_a, p_b, p_c], min_rows=100)

    assert table["group"].tolist() == ["A"]
    assert table["auc"].tolist() == [1.0]


def test_weighted_mean_uses_group_sizes():
    rng = np.random.default_rng(0)
    # Group "big" is perfectly ranked; group "small" is constant (AUC 0.5).
    y = np.r_[np.array([0] * 100 + [1] * 100), rng.integers(0, 2, 100)]
    y[-1], y[-2] = 0, 1  # guarantee both outcomes appear
    p = np.r_[np.array([0.1] * 100 + [0.9] * 100), np.full(100, 0.5)]
    groups = ["big"] * 200 + ["small"] * 100
    table = per_group_auc(groups, y, p, min_rows=50)
    expected = (1.0 * 200 + 0.5 * 100) / 300
    assert weighted_mean_auc(table) == pytest.approx(expected)

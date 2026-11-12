import pytest

from benchmarks.metrics import auc, brier, ece, reliability_table


def test_auc_is_one_for_perfect_ranking():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_auc_is_half_when_every_score_is_the_same():
    assert auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_ece_is_zero_when_probabilities_match_reality():
    # Half of the prompts fail, and the router says 0.5 for all of them.
    assert ece([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0, abs=1e-12)


def test_ece_is_large_when_overconfident():
    # Router says 0.9 everywhere but nothing fails: it is off by 0.9.
    assert ece([0, 0, 0, 0], [0.9, 0.9, 0.9, 0.9]) == pytest.approx(0.9)


def test_probability_of_exactly_one_is_counted():
    rows = reliability_table([1, 1], [1.0, 1.0])
    assert rows[-1]["count"] == 2


def test_brier_of_a_perfect_forecast_is_zero():
    assert brier([0, 1], [0.0, 1.0]) == 0.0


def test_probabilities_outside_zero_one_are_rejected():
    with pytest.raises(ValueError):
        ece([0, 1], [0.5, 1.5])

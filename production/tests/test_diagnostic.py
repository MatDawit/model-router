import pandas as pd
import pytest

from benchmarks.baselines.diagnostic import benchmark_rate_probs


def _train():
    return pd.DataFrame(
        {"eval_name": ["A"] * 50 + ["B"] * 50, "fails": [1] * 50 + [0] * 50}
    )


def test_hard_benchmark_gets_a_higher_probability_than_easy_one():
    other = pd.DataFrame({"eval_name": ["A", "B"]})
    p = benchmark_rate_probs(_train(), other, "fails")
    assert p[0] > 0.8 and p[1] < 0.2


def test_unseen_benchmark_gets_the_overall_rate():
    other = pd.DataFrame({"eval_name": ["Z"]})
    p = benchmark_rate_probs(_train(), other, "fails")
    assert p[0] == pytest.approx(0.5)


def test_probabilities_stay_between_zero_and_one():
    other = pd.DataFrame({"eval_name": ["A", "B", "Z"]})
    p = benchmark_rate_probs(_train(), other, "fails")
    assert p.min() >= 0.0 and p.max() <= 1.0

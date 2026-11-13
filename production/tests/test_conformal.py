import numpy as np
import pytest

from core.router.conformal import empirical_bernstein_upper_bound as bound
from core.router.conformal import fixed_sequence_certify


def test_all_zero_losses_still_get_a_small_safety_margin():
    # No spread at all, so only the small-sample term remains: 7*2*ln(40)/(3*7295).
    assert bound(np.zeros(7296)) == pytest.approx(0.00236, abs=1e-5)


def test_bound_is_always_above_the_measured_mean():
    x = np.random.default_rng(0).choice([-1.0, 0.0, 1.0], size=500, p=[0.1, 0.7, 0.2])
    assert bound(x) > x.mean()


def test_doubling_the_data_shrinks_the_margin_by_roughly_30_percent():
    pattern = np.tile([0, 1, -1, 0, 0, 0, 0, 0, 0, 0], 600).astype(float)  # mean exactly 0
    margin_n = bound(pattern[:2000])
    margin_2n = bound(pattern[:4000])
    assert 0.5 < margin_2n / margin_n < 0.72


def test_the_bound_really_holds_at_least_95_percent_of_the_time():
    rng = np.random.default_rng(1)
    values, probs = np.array([-1.0, 0.0, 1.0]), np.array([0.05, 0.75, 0.20])
    true_mean = float((values * probs).sum())  # 0.15
    misses = sum(
        bound(rng.choice(values, size=800, p=probs)) < true_mean for _ in range(1000)
    )
    assert misses / 1000 <= 0.05


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        bound([0.0])  # too few
    with pytest.raises(ValueError):
        bound([0.0, 2.0])  # outside [-1, 1]
    with pytest.raises(ValueError):
        bound([0.0, 0.1], alpha=1.5)


def _losses(n, n_ones):
    x = np.zeros(n)
    x[:n_ones] = 1.0
    return x


def test_certifies_up_the_ladder_and_stops_at_the_first_failure():
    safe = _losses(3000, 0)       # no loss at all
    fine = _losses(3000, 15)      # mean 0.005, comfortably inside the 2-point budget
    too_risky = _losses(3000, 300)  # mean 0.10, far over budget
    safe_again = _losses(3000, 0)   # would pass, but it comes AFTER the failure
    assert fixed_sequence_certify([safe, fine, too_risky, safe_again]) == [0, 1]


def test_empty_ladder_certifies_nothing():
    assert fixed_sequence_certify([]) == []


def test_a_wrong_certification_happens_at_most_5_percent_of_the_time():
    rng = np.random.default_rng(2)
    values, probs = np.array([-1.0, 0.0, 1.0]), np.array([0.0, 0.97, 0.03])  # true mean 0.03 > 0.02
    wrong = 0
    for _ in range(1000):
        ladder = [rng.choice(values, size=800, p=probs) for _ in range(5)]  # every rung is truly bad
        if fixed_sequence_certify(ladder):
            wrong += 1
    assert wrong / 1000 <= 0.05

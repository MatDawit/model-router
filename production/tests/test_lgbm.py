import numpy as np
from sklearn.linear_model import LogisticRegression

from benchmarks.metrics import auc
from core.router.models.lgbm import LGBMRouter


def _xor_data(n=3000, seed=0):
    """Fails only when exactly ONE of two switches is on. No straight line can express that."""
    rng = np.random.default_rng(seed)
    switches = rng.integers(0, 2, size=(n, 2)).astype(float)
    noise = rng.normal(size=(n, 5))
    y = (switches[:, 0] != switches[:, 1]).astype(int)
    return np.hstack([switches, noise]), y


def test_finds_a_combination_pattern_that_a_straight_line_cannot():
    x, y = _xor_data()
    x_train, y_train, x_test, y_test = x[:2200], y[:2200], x[2200:], y[2200:]

    straight_line = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    lgbm = LGBMRouter().fit(x_train, y_train)

    auc_line = auc(y_test, straight_line.predict_proba(x_test)[:, 1])
    auc_trees = auc(y_test, lgbm.predict_proba(x_test))
    assert auc_trees > 0.9
    assert auc_trees > auc_line + 0.3


def test_probabilities_are_valid():
    x, y = _xor_data(n=1000)
    p = LGBMRouter().fit(x, y).predict_proba(x)
    assert p.shape == (1000,)
    assert p.min() >= 0.0 and p.max() <= 1.0

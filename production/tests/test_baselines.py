import numpy as np
import pytest

from benchmarks.baselines.simple import BaseRateBaseline, LengthBaseline
from benchmarks.metrics import auc, ece


def _length_driven_data():
    # Long prompts fail, short prompts do not.
    texts = ["x" * 10] * 50 + ["x" * 2000] * 50
    y = np.array([0] * 50 + [1] * 50)
    return texts, y


def test_base_rate_is_calibrated_but_cannot_rank():
    texts = ["a"] * 6 + ["bb"] * 4
    y = np.array([1] * 6 + [0] * 4)
    p = BaseRateBaseline().fit(texts, y).predict_proba(texts)
    assert ece(y, p) == pytest.approx(0.0, abs=1e-9)
    assert auc(y, p) == 0.5


def test_length_baseline_finds_a_length_signal():
    texts, y = _length_driven_data()
    p = LengthBaseline().fit(texts, y).predict_proba(texts)
    assert auc(y, p) > 0.95


def test_length_baseline_returns_valid_probabilities():
    texts, y = _length_driven_data()
    p = LengthBaseline().fit(texts, y).predict_proba(texts)
    assert p.shape == (100,)
    assert p.min() >= 0.0 and p.max() <= 1.0

"""The two simplest possible routers. Every smarter model must beat these.

Shared interface (later models follow it too):
    model.fit(texts, y)          # texts: list of prompts, y: 0/1 "tier failed"
    model.predict_proba(texts)   # returns P(tier fails) per prompt
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class BaseRateBaseline:
    """Says the overall failure rate for every prompt.

    Perfectly calibrated (ECE ~ 0) but useless for routing (AUC = 0.5),
    because it never tells one prompt from another.
    """

    def fit(self, texts, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, texts):
        return np.full(len(list(texts)), self.rate_)


class LengthBaseline:
    """Predicts failure from prompt length alone (log scale) with logistic regression."""

    @staticmethod
    def _x(texts):
        # log1p because lengths are heavy-tailed: a few prompts are huge.
        return np.log1p([[len(t)] for t in texts])

    def fit(self, texts, y):
        self.model_ = LogisticRegression().fit(self._x(list(texts)), y)
        return self

    def predict_proba(self, texts):
        return self.model_.predict_proba(self._x(list(texts)))[:, 1]

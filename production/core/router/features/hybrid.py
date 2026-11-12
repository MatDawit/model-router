"""Hybrid features: 384 dense embedding dims + 12 scaled structural signals = 396."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from core.router.features.dense import embed
from core.router.features.structural import structural_matrix


class HybridFeatures:
    def __init__(self, encoder):
        self.encoder = encoder
        self.scaler = StandardScaler()
        self._fitted = False

    @staticmethod
    def _structural(texts):
        # Lengths and counts are heavy-tailed, so compress with log1p before scaling.
        return np.log1p(structural_matrix(texts))

    def fit(self, texts):
        """Learn the scaling from TRAIN prompts only, so no test information leaks in."""
        self.scaler.fit(self._structural(list(texts)))
        self._fitted = True
        return self

    def transform(self, texts, dense=None):
        """Build the (n, 396) matrix. Pass `dense` to reuse embeddings you already computed."""
        if not self._fitted:
            raise RuntimeError("call fit() on the training prompts first")
        texts = list(texts)
        if dense is None:
            dense = embed(texts, self.encoder)
        if len(dense) != len(texts):
            raise ValueError(f"got {len(dense)} embeddings for {len(texts)} prompts")
        scaled = self.scaler.transform(self._structural(texts))
        return np.hstack([dense, scaled]).astype(np.float32)

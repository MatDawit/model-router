import numpy as np
import pytest

from core.router.features.hybrid import HybridFeatures


class FakeEncoder:
    """Stands in for the real sentence encoder so tests are fast and need no download."""

    def encode(self, texts, **kwargs):
        return np.tile(np.arange(384, dtype=float), (len(texts), 1))


TRAIN = ["short", "a somewhat longer prompt here", "What is 2+2?", "x " * 50, "你好"]


def test_output_is_396_dimensional():
    feats = HybridFeatures(FakeEncoder()).fit(TRAIN)
    x = feats.transform(TRAIN)
    assert x.shape == (len(TRAIN), 396)
    assert x.dtype == np.float32


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        HybridFeatures(FakeEncoder()).transform(TRAIN)


def test_scaler_learns_only_from_the_fit_prompts():
    feats = HybridFeatures(FakeEncoder()).fit(TRAIN)
    feats.transform(["a completely different prompt"])
    assert feats.scaler.n_samples_seen_ == len(TRAIN)


def test_training_structural_columns_are_centered():
    x = HybridFeatures(FakeEncoder()).fit(TRAIN).transform(TRAIN)
    assert np.allclose(x[:, 384:].mean(axis=0), 0.0, atol=1e-5)


def test_precomputed_embeddings_are_used_as_given():
    feats = HybridFeatures(FakeEncoder()).fit(TRAIN)
    dense = np.full((len(TRAIN), 384), 7.0, dtype=np.float32)
    x = feats.transform(TRAIN, dense=dense)
    assert np.all(x[:, :384] == 7.0)


def test_wrong_number_of_embeddings_raises():
    feats = HybridFeatures(FakeEncoder()).fit(TRAIN)
    with pytest.raises(ValueError):
        feats.transform(TRAIN, dense=np.zeros((2, 384)))

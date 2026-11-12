"""Dense semantic embeddings (384 numbers per prompt) from a sentence encoder."""

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_encoder(name=DEFAULT_MODEL, device=None):
    """Load the sentence encoder. Imported lazily because it is slow and heavy."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=device)


def embed(texts, encoder, batch_size=64, show_progress=False):
    """Return an (n_prompts, 384) float32 array of unit-length embeddings."""
    vectors = encoder.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return np.asarray(vectors, dtype=np.float32)

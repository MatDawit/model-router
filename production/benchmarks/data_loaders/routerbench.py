"""Load RouterBench and build "did the cheaper tier fail?" labels.

RouterBench (Martian) ships as a pandas pickle. Each row is one prompt, and
every model's score on it is stored in a column named after the model.
"""

import pandas as pd

# Default RouterBench tiers (old models; the OpenRouter tiers live elsewhere).
SMALL = "mistralai/mistral-7b-chat"
MID = "zero-one-ai/Yi-34B-Chat"
FRONTIER = "gpt-4-1106-preview"

# Columns that are NOT model scores. `oracle_model_to_route_to` is built from
# the answers, so it must never be used as a router feature (data leakage).
NON_MODEL_COLUMNS = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}

# Scores like 0.8 - 0.1 are not exact in floating point (0.8 - 0.1 = 0.7000000000000001),
# so we compare against delta with a tiny cushion.
EPS = 1e-9


def load_routerbench(path):
    """Read the RouterBench pickle and return it as a DataFrame."""
    df = pd.read_pickle(path)
    missing = {"prompt", "eval_name"} - set(df.columns)
    if missing:
        raise ValueError(f"RouterBench file is missing columns: {sorted(missing)}")
    return df


def make_labels(df, small=SMALL, mid=MID, frontier=FRONTIER, delta=0.1):
    """Add `small_fails` and `mid_fails` (1 = that tier fell short of the frontier).

    A tier "fails" when its score is lower than the frontier's score by MORE
    than `delta`. A drop of exactly `delta` is tolerated.
    """
    for model in (small, mid, frontier):
        if model not in df.columns:
            raise KeyError(f"Model column not found in data: {model}")

    out = df.copy()  # never change the caller's DataFrame
    out["small_fails"] = ((df[frontier] - df[small]) > delta + EPS).astype(int)
    out["mid_fails"] = ((df[frontier] - df[mid]) > delta + EPS).astype(int)
    return out

"""Split RouterBench into train / calibration / test sets.

- train: the router learns from it.
- cal:   held out; used to calibrate probabilities and choose the threshold tau.
- test:  touched once at the end for the numbers we report.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def add_split_column(df, fractions=(0.6, 0.2, 0.2), min_group=20, seed=42):
    """Return a de-duplicated copy of `df` with a `split` column: train / cal / test.

    Each benchmark (`eval_name`) is spread across the three sets in the same
    proportions. Benchmarks with fewer than `min_group` rows are pooled into one
    "rare" bucket for balancing, because tiny groups cannot be cut three ways.
    """
    train_f, cal_f, test_f = fractions
    if abs(train_f + cal_f + test_f - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    # The same prompt in two sets would let the router "remember" the answer.
    out = df.drop_duplicates(subset="prompt").reset_index(drop=True)

    # Label used only to balance the split: the benchmark name, or "rare".
    counts = out["eval_name"].value_counts()
    stratum = out["eval_name"].where(out["eval_name"].map(counts) >= min_group, "rare")

    # Cut 1: train vs. everything else.
    train_idx, rest_idx = train_test_split(
        out.index, test_size=1.0 - train_f, stratify=stratum, random_state=seed
    )
    # Cut 2: split the rest into calibration and test.
    cal_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_f / (cal_f + test_f),
        stratify=stratum.loc[rest_idx],
        random_state=seed,
    )

    out["split"] = "train"
    out.loc[cal_idx, "split"] = "cal"
    out.loc[test_idx, "split"] = "test"
    return out

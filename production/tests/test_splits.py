import pandas as pd
import pytest

from benchmarks.data_loaders.splits import add_split_column


def _frame():
    rows = []
    for name, n in (("big", 200), ("medium", 100), ("tiny", 5)):
        rows += [{"prompt": f"{name}-{i}", "eval_name": name} for i in range(n)]
    rows.append({"prompt": "big-0", "eval_name": "big"})  # a duplicate prompt
    return pd.DataFrame(rows)


def test_duplicates_are_removed():
    out = add_split_column(_frame())
    assert out["prompt"].is_unique


def test_sizes_are_close_to_60_20_20():
    out = add_split_column(_frame())
    share = out["split"].value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.6, abs=0.03)
    assert share["cal"] == pytest.approx(0.2, abs=0.03)
    assert share["test"] == pytest.approx(0.2, abs=0.03)


def test_every_big_benchmark_appears_in_every_split():
    out = add_split_column(_frame())
    for name in ("big", "medium"):
        assert set(out.loc[out["eval_name"] == name, "split"]) == {"train", "cal", "test"}


def test_tiny_benchmark_does_not_crash_the_split():
    out = add_split_column(_frame())  # "tiny" has only 5 rows
    assert (out["eval_name"] == "tiny").sum() == 5


def test_same_seed_gives_same_split():
    a = add_split_column(_frame(), seed=7)
    b = add_split_column(_frame(), seed=7)
    assert a["split"].tolist() == b["split"].tolist()


def test_bad_fractions_raise():
    with pytest.raises(ValueError):
        add_split_column(_frame(), fractions=(0.5, 0.2, 0.2))


def test_input_frame_is_not_modified():
    df = _frame()
    add_split_column(df)
    assert "split" not in df.columns

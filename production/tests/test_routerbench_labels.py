import pandas as pd

from benchmarks.data_loaders.routerbench import make_labels


def _frame(frontier, small, mid):
    return pd.DataFrame({"F": frontier, "S": small, "M": mid})


def test_drop_bigger_than_delta_is_a_failure():
    df = _frame([0.8], [0.5], [0.8])
    out = make_labels(df, small="S", mid="M", frontier="F", delta=0.1)
    assert out["small_fails"].tolist() == [1]
    assert out["mid_fails"].tolist() == [0]


def test_drop_exactly_delta_is_tolerated():
    # 0.8 - 0.7 is exactly the tolerance; floating point must not flip it to a failure.
    df = _frame([0.8], [0.7], [0.7])
    out = make_labels(df, small="S", mid="M", frontier="F", delta=0.1)
    assert out["small_fails"].tolist() == [0]


def test_cheap_model_beating_frontier_is_not_a_failure():
    df = _frame([0.0], [1.0], [1.0])
    out = make_labels(df, small="S", mid="M", frontier="F")
    assert out["small_fails"].tolist() == [0]


def test_input_frame_is_not_modified():
    df = _frame([0.8], [0.1], [0.1])
    make_labels(df, small="S", mid="M", frontier="F")
    assert "small_fails" not in df.columns

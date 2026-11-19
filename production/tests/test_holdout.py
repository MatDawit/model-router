import pandas as pd

from benchmarks.run_holdout import holdout_parts


def _data():
    names = ["math"] * 6 + ["trivia"] * 6
    splits = ["train", "train", "train", "cal", "test", "test"] * 2
    return pd.DataFrame({"eval_name": names, "split": splits})


def test_the_held_out_benchmark_never_appears_in_train_cal_or_id_test():
    parts = holdout_parts(_data(), "math")
    for key in ("train", "cal", "id_test"):
        assert "math" not in set(parts[key]["eval_name"])


def test_ood_is_every_row_of_the_held_out_benchmark():
    parts = holdout_parts(_data(), "math")
    assert len(parts["ood"]) == 6
    assert set(parts["ood"]["eval_name"]) == {"math"}


def test_seen_rows_keep_their_original_split():
    parts = holdout_parts(_data(), "math")
    assert len(parts["train"]) == 3 and len(parts["cal"]) == 1 and len(parts["id_test"]) == 2

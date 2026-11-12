"""Score every router on the calibration set.

    python -m benchmarks.run_scoreboard --path routerbench_0shot.pkl
    python -m benchmarks.run_scoreboard --path routerbench_0shot.pkl --embed   # adds embeddings (use a GPU)
"""

import argparse

import pandas as pd
from sklearn.linear_model import LogisticRegression

from benchmarks.baselines.diagnostic import benchmark_rate_probs
from benchmarks.baselines.simple import BaseRateBaseline, LengthBaseline
from benchmarks.data_loaders.routerbench import load_routerbench, make_labels
from benchmarks.data_loaders.splits import add_split_column
from benchmarks.metrics import auc, brier, ece

TARGETS = ("small_fails", "mid_fails")


def _row(name, target, y, p):
    return {"model": name, "target": target, "auc": auc(y, p), "brier": brier(y, p), "ece": ece(y, p)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="path to routerbench_0shot.pkl")
    parser.add_argument("--embed", action="store_true", help="also score the 396-dim hybrid model")
    args = parser.parse_args()

    data = add_split_column(make_labels(load_routerbench(args.path)))
    train = data[data["split"] == "train"]
    cal = data[data["split"] == "cal"]
    train_texts, cal_texts = train["prompt"].tolist(), cal["prompt"].tolist()

    rows = []
    for target in TARGETS:
        for name, model in (("base_rate", BaseRateBaseline()), ("length", LengthBaseline())):
            model.fit(train_texts, train[target].values)
            rows.append(_row(name, target, cal[target].values, model.predict_proba(cal_texts)))
        # Cheating diagnostic (reads the benchmark name): how much signal is just "task type"?
        rows.append(_row("benchmark_id*", target, cal[target].values, benchmark_rate_probs(train, cal, target)))

    if args.embed:
        from core.router.features.dense import embed, load_encoder
        from core.router.features.hybrid import HybridFeatures

        encoder = load_encoder()
        dense = embed(data["prompt"].tolist(), encoder, show_progress=True)  # embed once, reuse
        feats = HybridFeatures(encoder).fit(train_texts)
        x_train = feats.transform(train_texts, dense=dense[train.index])
        x_cal = feats.transform(cal_texts, dense=dense[cal.index])
        for target in TARGETS:
            model = LogisticRegression(max_iter=2000).fit(x_train, train[target].values)
            rows.append(_row("hybrid_logreg", target, cal[target].values, model.predict_proba(x_cal)[:, 1]))

    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

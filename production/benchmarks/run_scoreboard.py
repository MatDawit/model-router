"""Score every router on the calibration set: quality of the probabilities AND real savings.

    python -m benchmarks.run_scoreboard --path routerbench_0shot.pkl
    python -m benchmarks.run_scoreboard --path routerbench_0shot.pkl --embed   # adds embeddings (use a GPU)
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from benchmarks.baselines.diagnostic import benchmark_rate_probs
from benchmarks.baselines.simple import BaseRateBaseline, LengthBaseline
from benchmarks.data_loaders.routerbench import load_routerbench, make_labels
from benchmarks.data_loaders.splits import add_split_column
from benchmarks.metrics import auc, brier, ece, per_group_auc, weighted_mean_auc
from benchmarks.pareto import best_under_budget, evaluate_routing, route, sweep

TARGETS = ("small_fails", "mid_fails")
TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def _row(name, target, y, p):
    return {"model": name, "target": target, "auc": auc(y, p), "brier": brier(y, p), "ece": ece(y, p)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="path to routerbench_0shot.pkl")
    parser.add_argument("--embed", action="store_true", help="also score the 396-dim hybrid model")
    parser.add_argument("--epsilon", type=float, default=0.02, help="allowed quality drop vs always-frontier")
    parser.add_argument("--out", default=None, help="optional CSV path for the full savings curves")
    args = parser.parse_args()

    data = add_split_column(make_labels(load_routerbench(args.path)))
    train = data[data["split"] == "train"]
    cal = data[data["split"] == "cal"]
    train_texts, cal_texts = train["prompt"].tolist(), cal["prompt"].tolist()

    preds = {}  # preds[model name][target] = predicted failure probabilities on cal

    def record(name, target, p):
        preds.setdefault(name, {})[target] = np.asarray(p)

    for target in TARGETS:
        for name, model in (("base_rate", BaseRateBaseline()), ("length", LengthBaseline())):
            model.fit(train_texts, train[target].values)
            record(name, target, model.predict_proba(cal_texts))
        # Cheating diagnostic (reads the benchmark name): how much signal is just "task type"?
        record("benchmark_id*", target, benchmark_rate_probs(train, cal, target))

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
            record("hybrid_logreg", target, model.predict_proba(x_cal)[:, 1])

    # ---- 1. Quality of the probabilities -------------------------------------------------
    rows = [_row(m, t, cal[t].values, p) for m, d in preds.items() for t, p in d.items()]
    print("\n== 1. Scoreboard on the calibration set ==")
    print(pd.DataFrame(rows).round(4).sort_values(["target", "auc"]).to_string(index=False))

    # ---- 2. Real savings (step A) ----------------------------------------------------------
    print(f"\n== 2. Best savings with quality drop <= {args.epsilon} (vs always-frontier) ==")
    summary, curves = [], []
    for name, d in preds.items():
        curve = sweep(cal, d["small_fails"], d["mid_fails"], TAUS)
        curves.append(curve.assign(model=name))
        best = best_under_budget(curve, args.epsilon)
        summary.append({"policy": name, **best.drop(["quality"]).to_dict()})

    references = {
        "ORACLE (hindsight)": route(cal["small_fails"], cal["mid_fails"], 0.5),
        "always small": np.zeros(len(cal), dtype=int),
        "always mid": np.ones(len(cal), dtype=int),
        "always frontier": np.full(len(cal), 2),
    }
    for name, tier in references.items():
        res = evaluate_routing(cal, tier)
        summary.append({"policy": name, "tau": np.nan, **{k: v for k, v in res.items() if k != "quality"}})
    print(pd.DataFrame(summary).round(4).to_string(index=False))

    # ---- 3. Within-benchmark skill (step B) --------------------------------------------------
    print("\n== 3. AUC INSIDE each benchmark (0.5 = no skill beyond task type) ==")
    for name in ("length", "hybrid_logreg"):
        if name not in preds:
            continue
        for target in TARGETS:
            table = per_group_auc(cal["eval_name"], cal[target], preds[name][target])
            print(f"\n{name} / {target}: weighted within-benchmark AUC = {weighted_mean_auc(table):.4f}")
            print(table.head(8).round(4).to_string(index=False))

    if args.out:
        pd.concat(curves).to_csv(args.out, index=False)
        print(f"\nFull curves written to {args.out}")


if __name__ == "__main__":
    main()

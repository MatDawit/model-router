"""End-to-end run: train the router, pick cutoffs on the calibration set, check ONCE on the test set.

    python -m benchmarks.run_guarantee --path routerbench_0shot.pkl      (use a GPU runtime)

For each allowed quality drop (epsilon) it compares two ways to pick cutoffs:
  naive      : best savings on the calibration set, no safety margin
  certified  : best savings among cutoffs with a statistical guarantee
and then measures both on the untouched test set.
"""

import argparse

import numpy as np
import pandas as pd

from benchmarks.certify import certify_and_pick, ladder_report
from benchmarks.data_loaders.routerbench import load_routerbench, make_labels
from benchmarks.data_loaders.splits import add_split_column
from benchmarks.pareto import best_under_budget, evaluate_routing, route, sweep_grid
from core.router.features.dense import embed, load_encoder
from core.router.features.hybrid import HybridFeatures
from core.router.models.lgbm import LGBMRouter

TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)
TARGETS = ("small_fails", "mid_fails")


def _on_test(test, p_test, tau_small, tau_mid):
    """Replay one pair of cutoffs on the test set."""
    tier = route(p_test["small_fails"], p_test["mid_fails"], tau_small, tau_mid)
    return evaluate_routing(test, tier)


def _print_ladder_summary(cal, p_cal, eps, alpha):
    """Show where the climb stopped and whether it left safe rungs unused."""
    rows = pd.DataFrame(ladder_report(cal, p_cal["small_fails"], p_cal["mid_fails"], TAUS, TAUS,
                                      epsilon=eps, alpha=alpha))
    certified = rows[rows["reached"] & rows["passes_alone"]]
    blocked = rows[~rows["reached"] & rows["passes_alone"]]  # fine on their own, never tested
    stop = rows[rows["reached"] & ~rows["passes_alone"]]      # the rung where the climb ended
    print(f"\n== Ladder diagnostic, epsilon={eps} ({len(rows)} rungs) ==")
    print(f"certified rungs: {len(certified)}, best savings among them: "
          f"{certified['savings'].max() if len(certified) else 0.0:.4f}")
    print(f"blocked rungs that would pass alone: {len(blocked)}, best savings among them: "
          f"{blocked['savings'].max() if len(blocked) else 0.0:.4f}")
    cols = ["rung", "tau_small", "tau_mid", "measured_drop", "cushion", "upper_bound",
            "passes_alone", "reached", "savings"]
    if len(stop):
        first = int(stop["rung"].iloc[0])
        print(f"climb stopped at rung {first}; rungs around the stop:")
        print(rows.loc[max(first - 3, 0): first + 6, cols].round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-ladder", action="store_true",
                        help="print where the certification climb stopped and why")
    parser.add_argument("--path", required=True, help="path to routerbench_0shot.pkl")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.05])
    parser.add_argument("--alpha", type=float, default=0.05, help="1 - confidence (0.05 = 95%%)")
    parser.add_argument("--out", default=None, help="optional CSV path for the results table")
    args = parser.parse_args()

    data = add_split_column(make_labels(load_routerbench(args.path)))
    train, cal, test = (data[data["split"] == s] for s in ("train", "cal", "test"))

    # 1. Turn prompts into numbers (embeddings computed once), fit scaling on TRAIN only.
    encoder = load_encoder()
    dense = embed(data["prompt"].tolist(), encoder, show_progress=True)
    feats = HybridFeatures(encoder).fit(train["prompt"].tolist())
    x = {
        name: feats.transform(part["prompt"].tolist(), dense=dense[part.index])
        for name, part in (("train", train), ("cal", cal), ("test", test))
    }

    # 2. Train one LightGBM per tier on TRAIN, then predict failure chances for cal and test.
    p_cal, p_test = {}, {}
    for target in TARGETS:
        model = LGBMRouter().fit(x["train"], train[target].values)
        p_cal[target] = model.predict_proba(x["cal"])
        p_test[target] = model.predict_proba(x["test"])

    # 3. For each allowed quality drop, pick cutoffs on CAL both ways and check on TEST.
    curve = sweep_grid(cal, p_cal["small_fails"], p_cal["mid_fails"], TAUS, TAUS)
    rows = []
    for eps in args.epsilons:
        naive = best_under_budget(curve, eps)
        res = _on_test(test, p_test, naive["tau_small"], naive["tau_mid"])
        rows.append(
            {"method": "naive", "epsilon": eps, "tau_small": naive["tau_small"], "tau_mid": naive["tau_mid"],
             "cal_savings": naive["savings"], "test_savings": res["savings"],
             "test_quality_drop": res["quality_drop"], "within_budget": res["quality_drop"] <= eps}
        )

        pick = certify_and_pick(cal, p_cal["small_fails"], p_cal["mid_fails"], TAUS, TAUS,
                                epsilon=eps, alpha=args.alpha)
        if pick is None:  # nothing could be guaranteed: fall back to always-frontier (zero savings)
            rows.append({"method": "certified", "epsilon": eps, "tau_small": np.nan, "tau_mid": np.nan,
                         "cal_savings": 0.0, "test_savings": 0.0, "test_quality_drop": 0.0, "within_budget": True})
        else:
            res = _on_test(test, p_test, pick["tau_small"], pick["tau_mid"])
            rows.append(
                {"method": "certified", "epsilon": eps, "tau_small": pick["tau_small"], "tau_mid": pick["tau_mid"],
                 "cal_savings": pick["savings"], "test_savings": res["savings"],
                 "test_quality_drop": res["quality_drop"], "within_budget": res["quality_drop"] <= eps}
            )

    table = pd.DataFrame(rows)
    print(f"\n== Cutoffs chosen on CAL, checked once on TEST (confidence {1 - args.alpha:.0%}) ==")
    print(table.round(4).to_string(index=False))
    print("\nReading guide: 'within_budget' = the test quality drop stayed at or under epsilon.")
    print("certified with all-zero savings means no cutoff pair could be guaranteed, so use always-frontier.")
    if args.out:
        table.to_csv(args.out, index=False)

    if args.show_ladder:
        for eps in args.epsilons:
            _print_ladder_summary(cal, p_cal, eps, args.alpha)


if __name__ == "__main__":
    main()

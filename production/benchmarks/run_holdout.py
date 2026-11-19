"""New-subject test: does the guarantee survive a benchmark the router never saw?

    python -m benchmarks.run_holdout --path routerbench_0shot.pkl      (use a GPU runtime)

For each benchmark (eval_name) in turn, we REMOVE it from the train and calibration
piles, train the router and choose cutoffs without it, then measure the result on:
  id_test : the usual test pile (benchmarks the router DID see)
  ood     : every row of the held-out benchmark (a subject it never saw)
"ood" = out of distribution, i.e. questions unlike the training data.
"""

import argparse

import numpy as np
import pandas as pd

from benchmarks.certify import certify_and_pick
from benchmarks.data_loaders.routerbench import load_routerbench, make_labels
from benchmarks.data_loaders.splits import add_split_column
from benchmarks.pareto import best_under_budget, sweep_grid
from benchmarks.run_guarantee import TARGETS, TAUS, _on_test
from core.router.features.dense import embed, load_encoder
from core.router.features.hybrid import HybridFeatures
from core.router.models.lgbm import LGBMRouter


def holdout_parts(data, held_name):
    """Cut `data` (already split) into train / cal / id_test / ood for one held-out benchmark."""
    held = data["eval_name"] == held_name
    seen = data[~held]
    return {
        "train": seen[seen["split"] == "train"],
        "cal": seen[seen["split"] == "cal"],
        "id_test": seen[seen["split"] == "test"],
        "ood": data[held],  # ALL rows of the unseen benchmark, whatever their split label
    }


def _choose(method, cal, p_cal, curve, epsilon, alpha):
    """Return (tau_small, tau_mid) for the chosen method, or None if nothing can be guaranteed."""
    if method == "naive":
        best = best_under_budget(curve, epsilon)
        return best["tau_small"], best["tau_mid"]
    pick = certify_and_pick(cal, p_cal["small_fails"], p_cal["mid_fails"], TAUS, TAUS,
                            epsilon=epsilon, alpha=alpha)
    return None if pick is None else (pick["tau_small"], pick["tau_mid"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="path to routerbench_0shot.pkl")
    parser.add_argument("--epsilon", type=float, default=0.02, help="allowed quality drop (score points)")
    parser.add_argument("--alpha", type=float, default=0.05, help="1 - confidence (0.05 = 95%%)")
    parser.add_argument("--min-rows", type=int, default=300, help="skip benchmarks smaller than this")
    parser.add_argument("--only", nargs="+", default=None, help="hold out just these eval_name values")
    parser.add_argument("--out", default=None, help="optional CSV path for the results table")
    args = parser.parse_args()

    data = add_split_column(make_labels(load_routerbench(args.path)))
    counts = data["eval_name"].value_counts()
    print("Benchmarks (eval_name) and row counts:")
    print(counts.to_string())
    names = args.only or [n for n, c in counts.items() if c >= args.min_rows]

    encoder = load_encoder()
    dense = embed(data["prompt"].tolist(), encoder, show_progress=True)  # once; rows align with data.index

    rows = []
    for name in names:
        parts = holdout_parts(data, name)
        feats = HybridFeatures(encoder).fit(parts["train"]["prompt"].tolist())
        x = {k: feats.transform(v["prompt"].tolist(), dense=dense[v.index]) for k, v in parts.items()}

        p = {k: {} for k in ("cal", "id_test", "ood")}
        for target in TARGETS:
            model = LGBMRouter().fit(x["train"], parts["train"][target].values)
            for k in p:
                p[k][target] = model.predict_proba(x[k])

        curve = sweep_grid(parts["cal"], p["cal"]["small_fails"], p["cal"]["mid_fails"], TAUS, TAUS)
        for method in ("naive", "certified"):
            cutoffs = _choose(method, parts["cal"], p["cal"], curve, args.epsilon, args.alpha)
            if cutoffs is None:  # nothing guaranteed: always-frontier, zero savings and zero drop
                id_res = ood_res = {"savings": 0.0, "quality_drop": 0.0}
                cutoffs = (np.nan, np.nan)
            else:
                id_res = _on_test(parts["id_test"], p["id_test"], *cutoffs)
                ood_res = _on_test(parts["ood"], p["ood"], *cutoffs)
            rows.append(
                {"held_out": name, "n_ood": len(parts["ood"]), "method": method,
                 "tau_small": cutoffs[0], "tau_mid": cutoffs[1],
                 "id_savings": id_res["savings"], "id_drop": id_res["quality_drop"],
                 "ood_savings": ood_res["savings"], "ood_drop": ood_res["quality_drop"],
                 "ood_within_budget": ood_res["quality_drop"] <= args.epsilon}
            )

    table = pd.DataFrame(rows)
    print(f"\n== Held-out benchmark test (epsilon={args.epsilon}, confidence {1 - args.alpha:.0%}) ==")
    print(table.round(4).to_string(index=False))
    print("\nReading guide: id_* = benchmarks the router saw; ood_* = the held-out benchmark.")
    print("ood_within_budget False = the promise broke on a subject the router never saw.")
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

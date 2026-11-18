"""MLP fair-multiple model (PyTorch), run through the same pipeline as 07/08.

1. Tune on the development years (2011-2017) only, oof scheme, using dev R^2
   (the same rule as the other models; see README "Model selection rule").
2. Score every year (2011-2025) for all schemes with the chosen setting.
3. Add "mlp" rows to data/scores.csv and results/fit_metrics.csv, save the grid
   to results/tuning_mlp.csv and the setting to results/best_params.json.

    python src/11_mlp.py           # full run (~10-20 min on a laptop)
    python src/11_mlp.py --quick   # one setting, dev years only (sanity check)
"""
import itertools
import sys
import time

import numpy as np
import pandas as pd

from mlp import TorchMLP
from preprocess import load_panel, DEV_YEARS, RESULTS
from scoring import score_model, fit_metrics, save_scores, save_params
from splits import load_folds, SCHEMES

GRID = {
    "hidden": [(32,), (64, 32)],
    "dropout": [0.0, 0.2],
    "weight_decay": [1e-4, 1e-2],
}


def grid_points(grid):
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def dev_r2(df, fold, params):
    preds = score_model(df, fold, lambda: TorchMLP(**params), "oof", years=DEV_YEARS)
    return fit_metrics(df, preds)["r2"].mean()


def main():
    RESULTS.mkdir(exist_ok=True)
    df = load_panel()
    fold = load_folds(df)

    if "--quick" in sys.argv:
        t0 = time.time()
        p = {"hidden": (64, 32), "dropout": 0.2, "weight_decay": 1e-4}
        print(f"quick check {p}: dev R2 = {dev_r2(df, fold, p):.3f} ({time.time() - t0:.0f}s)")
        return

    # ---- 1. tuning on 2011-2017 --------------------------------------------
    rows = []
    for params in grid_points(GRID):
        t0 = time.time()
        r2 = dev_r2(df, fold, params)
        rows.append({**{k: str(v) if k == "hidden" else v for k, v in params.items()},
                     "dev_r2": r2})
        print(f"mlp {params} dev R2={r2:.3f} ({time.time() - t0:.0f}s)", flush=True)
    tuning = pd.DataFrame(rows)
    tuning.to_csv(RESULTS / "tuning_mlp.csv", index=False)
    best_row = tuning.loc[tuning["dev_r2"].idxmax()]
    best = {"hidden": list(eval(best_row["hidden"])), "dropout": float(best_row["dropout"]),
            "weight_decay": float(best_row["weight_decay"])}
    save_params("mlp", best)
    print(f"mlp best: {best}", flush=True)

    # ---- 2. score all years, all schemes --------------------------------------
    preds = pd.concat([score_model(df, fold, lambda: TorchMLP(**best), s) for s in SCHEMES],
                      ignore_index=True)
    save_scores(preds, "mlp")
    m = fit_metrics(df, preds).assign(model="mlp")
    path = RESULTS / "fit_metrics.csv"
    old = pd.read_csv(path)
    pd.concat([old[old["model"] != "mlp"], m], ignore_index=True).to_csv(path, index=False)

    allm = pd.read_csv(path)
    allm["period"] = np.where(allm["year"] <= 2017, "dev 2011-17", "test 2018-25")
    print("\nMean within-year R^2 (oof)")
    print(allm[allm["scheme"] == "oof"].pivot_table(index="model", columns="period",
                                                     values="r2").round(3).to_string())


if __name__ == "__main__":
    main()

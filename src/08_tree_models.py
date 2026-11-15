"""Random Forest and XGBoost fair-multiple models.

1. Tune hyperparameters on the development years (2011-2017) only, using the
   main `oof` scheme. The chosen setting is then used for every scheme.
2. Score every year (2011-2025) for all schemes.
3. Add rows to data/scores.csv and results/fit_metrics.csv; save the tuning
   grid to results/tuning_<model>.csv (dev R^2 for each setting, useful for
   the fit-vs-signal variation) and the chosen setting to results/best_params.json.

    python src/08_tree_models.py            # rf and xgb
    python src/08_tree_models.py rf         # only one model
"""
import itertools
import sys
import time

import numpy as np
import pandas as pd

from preprocess import load_panel, DEV_YEARS, RESULTS
from scoring import score_model, fit_metrics, save_scores, save_params
from splits import load_folds, SCHEMES
from models import MODELS

def grid_points(grid):
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def run(name, df, fold):
    factory, grid = MODELS[name]

    # ---- 1. tuning on 2011-2017 (oof scheme) -----------------------------
    rows = []
    for params in grid_points(grid):
        t0 = time.time()
        preds = score_model(df, fold, lambda p=params: factory(**p), "oof", years=DEV_YEARS)
        r2 = fit_metrics(df, preds)["r2"].mean()
        rows.append({**params, "dev_r2": r2})
        print(f"{name} {params} dev R2={r2:.3f} ({time.time() - t0:.0f}s)", flush=True)
    tuning = pd.DataFrame(rows)
    tuning.to_csv(RESULTS / f"tuning_{name}.csv", index=False)
    best = tuning.loc[tuning["dev_r2"].idxmax(), list(grid)].to_dict()
    best = {k: (int(v) if float(v).is_integer() and k != "max_features" else float(v))
            for k, v in best.items()}
    save_params(name, best)
    print(f"{name} best: {best}", flush=True)

    # ---- 2. score all years ----------------------------------------------
    preds = pd.concat([score_model(df, fold, lambda: factory(**best), s) for s in SCHEMES],
                      ignore_index=True)
    save_scores(preds, name)
    m = fit_metrics(df, preds).assign(model=name)
    path = RESULTS / "fit_metrics.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[old["model"] != name]
    pd.concat([old, m], ignore_index=True).to_csv(path, index=False)
    return m


def main():
    RESULTS.mkdir(exist_ok=True)
    names = sys.argv[1:] or list(MODELS)
    df = load_panel()
    fold = load_folds(df)
    for name in names:
        run(name, df, fold)

    m = pd.read_csv(RESULTS / "fit_metrics.csv")
    m["period"] = np.where(m["year"] <= 2017, "dev 2011-17", "test 2018-25")
    print("\nMean within-year R^2")
    print(m.pivot_table(index=["model", "scheme"], columns="period", values="r2").round(3).to_string())


if __name__ == "__main__":
    main()

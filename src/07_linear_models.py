"""Linear, Ridge and Lasso fair-multiple models.

1. Tune alpha for each scheme using the development years (2011-2017) only.
2. With the chosen alpha, score every year (2011-2025).
3. Write data/scores.csv, results/fit_metrics.csv, results/tuning_linear.csv,
   results/best_params.json.

    python src/07_linear_models.py
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from preprocess import load_panel, DEV_YEARS, RESULTS
from scoring import score_model, fit_metrics, save_scores, save_params
from splits import load_folds, SCHEMES

warnings.filterwarnings("ignore", category=UserWarning)

GRIDS = {
    "ridge": (Ridge, np.logspace(-2, 4, 13)),
    "lasso": (Lasso, np.logspace(-4, 0, 9)),
}


def dev_r2(df, fold, make_model, scheme):
    preds = score_model(df, fold, make_model, scheme, years=DEV_YEARS)
    return fit_metrics(df, preds)["r2"].mean()


def main():
    RESULTS.mkdir(exist_ok=True)
    df = load_panel()
    fold = load_folds(df)

    # ---- 1. tuning on 2011-2017 -------------------------------------------
    tuning, best = [], {}
    for name, (cls, grid) in GRIDS.items():
        best[name] = {}
        for scheme in SCHEMES:
            scores = []
            for a in grid:
                make = (lambda a=a: cls(alpha=a, max_iter=50_000)) if cls is Lasso \
                    else (lambda a=a: cls(alpha=a))
                r2 = dev_r2(df, fold, make, scheme)
                scores.append(r2)
                tuning.append({"model": name, "scheme": scheme, "alpha": a, "dev_r2": r2})
            best[name][scheme] = float(grid[int(np.argmax(scores))])
            print(f"{name:6s} {scheme:17s} best alpha={best[name][scheme]:.4g} "
                  f"dev R2={max(scores):.3f}")
        save_params(name, best[name])
    pd.DataFrame(tuning).to_csv(RESULTS / "tuning_linear.csv", index=False)

    # ---- 2. score all years with the chosen settings ----------------------
    metrics = []
    for name in ["linear", "ridge", "lasso"]:
        preds_all = []
        for scheme in SCHEMES:
            if name == "linear":
                make = LinearRegression
            elif name == "ridge":
                make = lambda a=best["ridge"][scheme]: Ridge(alpha=a)
            else:
                make = lambda a=best["lasso"][scheme]: Lasso(alpha=a, max_iter=50_000)
            preds_all.append(score_model(df, fold, make, scheme))
        preds = pd.concat(preds_all, ignore_index=True)
        save_scores(preds, name)
        metrics.append(fit_metrics(df, preds).assign(model=name))

    m = pd.concat(metrics, ignore_index=True)
    path = RESULTS / "fit_metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        m = pd.concat([old[~old["model"].isin(m["model"].unique())], m], ignore_index=True)
    m.to_csv(path, index=False)

    m["period"] = np.where(m["year"] <= 2017, "dev 2011-17", "test 2018-25")
    print("\nMean within-year R^2")
    print(m.pivot_table(index=["model", "scheme"], columns="period", values="r2").round(3).to_string())


if __name__ == "__main__":
    main()

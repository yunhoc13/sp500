"""Price-free feature experiment.

Question: do the tree models lose their signal because some features carry
the stock price? log_mcap (= price x shares), volatility and beta are all
computed from the price, and the target log(EV/EBITDA) contains the price too,
so a model can partly "see" how cheap a stock is and explain it away.

Here every model is rebuilt with FEATURES_NOPX (preprocess.py): size is log
revenue instead of log market cap, and volatility / beta are dropped. Tuning
follows exactly the same rules as 07 / 08 (development years 2011-2017 only).
Rows are added to data/scores.csv as <model>_nopx (ridge_nopx, rf_nopx,
xgb_nopx), so 10_backtest.py compares them automatically.

This is an analysis of the accuracy-vs-signal trade-off, not a model
selection step: the final-model rule (highest dev R^2, base features) is
unchanged.

    python src/12_price_free.py              # ridge, rf, xgb (xgb needs xgboost)
    python src/12_price_free.py ridge rf     # a subset
"""
import importlib.util
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from preprocess import load_panel, FEATURES_NOPX, DEV_YEARS, RESULTS
from scoring import score_model, fit_metrics, save_scores, save_params
from splits import load_folds, SCHEMES
from models import MODELS

import itertools

RIDGE_GRID = np.logspace(-2, 4, 13)
SUFFIX = "_nopx"


def dev_r2(df, fold, make, scheme):
    preds = score_model(df, fold, make, scheme, years=DEV_YEARS, features=FEATURES_NOPX)
    return fit_metrics(df, preds)["r2"].mean()


def run_ridge(df, fold):
    """Ridge: alpha tuned per scheme (same as 07_linear_models.py)."""
    best, tuning, preds = {}, [], []
    for scheme in SCHEMES:
        scores = [dev_r2(df, fold, lambda a=a: Ridge(alpha=a), scheme) for a in RIDGE_GRID]
        tuning += [{"scheme": scheme, "alpha": a, "dev_r2": r} for a, r in zip(RIDGE_GRID, scores)]
        best[scheme] = float(RIDGE_GRID[int(np.argmax(scores))])
        print(f"ridge{SUFFIX} {scheme:17s} alpha={best[scheme]:.4g} dev R2={max(scores):.3f}", flush=True)
        preds.append(score_model(df, fold, lambda a=best[scheme]: Ridge(alpha=a), scheme,
                                 features=FEATURES_NOPX))
    pd.DataFrame(tuning).to_csv(RESULTS / f"tuning_ridge{SUFFIX}.csv", index=False)
    save_params(f"ridge{SUFFIX}", best)
    return pd.concat(preds, ignore_index=True)


def run_tree(name, df, fold):
    """RF / XGBoost: tuned on the oof scheme, same grid as 08_tree_models.py."""
    factory, grid = MODELS[name]
    keys = list(grid)
    rows = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        p = dict(zip(keys, vals))
        r2 = dev_r2(df, fold, lambda p=p: factory(**p), "oof")
        rows.append({**p, "dev_r2": r2})
        print(f"{name}{SUFFIX} {p} dev R2={r2:.3f}", flush=True)
    tuning = pd.DataFrame(rows)
    tuning.to_csv(RESULTS / f"tuning_{name}{SUFFIX}.csv", index=False)
    best = tuning.loc[tuning["dev_r2"].idxmax(), keys].to_dict()
    best = {k: (int(v) if float(v).is_integer() and k != "max_features" else float(v))
            for k, v in best.items()}
    save_params(f"{name}{SUFFIX}", best)
    print(f"{name}{SUFFIX} best: {best}", flush=True)
    return pd.concat([score_model(df, fold, lambda: factory(**best), s, features=FEATURES_NOPX)
                      for s in SCHEMES], ignore_index=True)


def main():
    names = sys.argv[1:] or ["ridge", "rf", "xgb"]
    if "xgb" in names and importlib.util.find_spec("xgboost") is None:
        print("xgboost is not installed here: skipping xgb (run this script where it is)")
        names = [n for n in names if n != "xgb"]
    df = load_panel()
    fold = load_folds(df)
    metrics = []
    for name in names:
        preds = run_ridge(df, fold) if name == "ridge" else run_tree(name, df, fold)
        save_scores(preds, name + SUFFIX)
        metrics.append(fit_metrics(df, preds).assign(model=name + SUFFIX))

    path = RESULTS / "fit_metrics.csv"
    m = pd.concat(metrics, ignore_index=True)
    old = pd.read_csv(path)
    m = pd.concat([old[~old["model"].isin(m["model"].unique())], m], ignore_index=True)
    m.to_csv(path, index=False)

    m["period"] = np.where(m["year"] <= 2017, "dev 2011-17", "test 2018-25")
    show = m[m["scheme"].isin(["oof", "walkforward"])]
    print("\nMean within-year R^2 (base vs price-free)")
    print(show.pivot_table(index=["model", "scheme"], columns="period", values="r2").round(3).to_string())


if __name__ == "__main__":
    main()

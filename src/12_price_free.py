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

Other feature sets (preprocess.FEATURE_SETS) can be run the same way to
separate the two price channels; rows are saved as <model>_<set>:
    novol    keep log_mcap, drop volatility and beta
    revsize  size from log revenue, keep volatility and beta
To keep data/scores.csv small and conflict-free, these go to separate files:
data/scores_ablation.csv, results/fit_metrics_ablation.csv,
results/best_params_ablation.json. Every run then rebuilds
results/ablation_summary.csv: one row per model x feature set (base, nopx,
novol, revsize) with fit, spread, IC, the spread without 2020, and the
volatility / return of the cheapest group in 2020.

This is an analysis of the accuracy-vs-signal trade-off, not a model
selection step: the final-model rule (highest dev R^2, base features) is
unchanged.

    python src/12_price_free.py                      # nopx: ridge, rf, xgb (xgb needs xgboost)
    python src/12_price_free.py ridge rf             # a subset of models
    python src/12_price_free.py --set novol          # another feature set
    python src/12_price_free.py xgb --set revsize
    python src/12_price_free.py --summary            # only rebuild the summary table
"""
import importlib.util
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from preprocess import load_panel, FEATURE_SETS, DEV_YEARS, TEST_YEARS, RESULTS, DATA
from scoring import score_model, fit_metrics, save_scores, save_params, SCORES_PATH
from splits import load_folds, SCHEMES
from models import MODELS

import itertools

RIDGE_GRID = np.logspace(-2, 4, 13)
SET = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "nopx"
FEATS = FEATURE_SETS[SET]
SUFFIX = f"_{SET}"
MAIN_FILES = SET == "nopx"          # nopx lives in the main files (already used by backtests)
SCORES_OUT = SCORES_PATH if MAIN_FILES else DATA / "scores_ablation.csv"
METRICS_OUT = RESULTS / ("fit_metrics.csv" if MAIN_FILES else "fit_metrics_ablation.csv")
PARAMS_OUT = RESULTS / ("best_params.json" if MAIN_FILES else "best_params_ablation.json")
SUMMARY_OUT = RESULTS / "ablation_summary.csv"
BASE_MODELS = ["ridge", "rf", "xgb"]


def dev_r2(df, fold, make, scheme):
    preds = score_model(df, fold, make, scheme, years=DEV_YEARS, features=FEATS)
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
                                 features=FEATS))
    pd.DataFrame(tuning).to_csv(RESULTS / f"tuning_ridge{SUFFIX}.csv", index=False)
    save_params(f"ridge{SUFFIX}", best, PARAMS_OUT)
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
    save_params(f"{name}{SUFFIX}", best, PARAMS_OUT)
    print(f"{name}{SUFFIX} best: {best}", flush=True)
    return pd.concat([score_model(df, fold, lambda: factory(**best), s, features=FEATS)
                      for s in SCHEMES], ignore_index=True)


def summary():
    """One row per model x feature set, from the main and ablation files."""
    import importlib
    bt = importlib.import_module("10_backtest")
    panel = pd.read_csv(DATA / "panel.csv")
    sc = [pd.read_csv(SCORES_PATH)]
    fm = [pd.read_csv(RESULTS / "fit_metrics.csv")]
    if (DATA / "scores_ablation.csv").exists():
        sc.append(pd.read_csv(DATA / "scores_ablation.csv"))
        fm.append(pd.read_csv(RESULTS / "fit_metrics_ablation.csv"))
    sc, fm = pd.concat(sc), pd.concat(fm)
    names = {m: (m, "base") for m in BASE_MODELS}
    names.update({f"{m}_{s}": (m, s) for m in BASE_MODELS for s in FEATURE_SETS if s != "base"})
    sc = sc[sc["model"].isin(names) & (sc["scheme"] == "oof")]
    fm = fm[fm["model"].isin(names) & (fm["scheme"] == "oof")]

    d = sc.merge(panel[["cik", "year", bt.RET, "volatility"]], on=["cik", "year"])
    d = d.dropna(subset=["mispricing", bt.RET])
    d["g5"] = bt.assign_groups(d, 5)
    yearly = bt.yearly_stats(d, 5)
    test, dev = bt.summarize(yearly, TEST_YEARS), bt.summarize(yearly, DEV_YEARS)
    no2020 = bt.leave_one_year_out(yearly, TEST_YEARS)["without 2020"]
    c20 = d[(d["g5"] == 1) & (d["year"] == 2020)].groupby("model")
    rows = []
    for (model, scheme), t in test.iterrows():
        base, fs = names[model]
        f = fm[fm["model"] == model]
        rows.append({
            "model": base, "feature_set": fs, "features": "+".join(FEATURE_SETS[fs]),
            "r2_dev": f.loc[f["year"].isin(DEV_YEARS), "r2"].mean(),
            "r2_test": f.loc[f["year"].isin(TEST_YEARS), "r2"].mean(),
            "spread_dev": dev.loc[(model, scheme), "spread_avg"],
            "spread_test": t["spread_avg"],
            "spread_test_without_2020": no2020.loc[(model, scheme)],
            "ic_test": t["ic_avg"],
            "cheapest_2020_volatility": c20["volatility"].mean().get(model),
            "cheapest_2020_ret_vs_spy": c20[bt.RET].mean().get(model),
        })
    order = {s: i for i, s in enumerate(FEATURE_SETS)}
    out = pd.DataFrame(rows).sort_values(["model", "feature_set"], key=lambda c: c.map(
        lambda v: order.get(v, BASE_MODELS.index(v) if v in BASE_MODELS else 99)))
    out.round(4).to_csv(SUMMARY_OUT, index=False)
    print(f"\nsaved {SUMMARY_OUT}")
    print(out.drop(columns="features").round(3).to_string(index=False))


def main():
    args = [a for i, a in enumerate(sys.argv[1:], 1)
            if not a.startswith("--") and sys.argv[i - 1] != "--set"]
    if "--summary" in sys.argv:
        summary()
        return
    names = args or ["ridge", "rf", "xgb"]
    print(f"feature set '{SET}': {FEATS}")
    if "xgb" in names and importlib.util.find_spec("xgboost") is None:
        print("xgboost is not installed here: skipping xgb (run this script where it is)")
        names = [n for n in names if n != "xgb"]
    df = load_panel()
    fold = load_folds(df)
    metrics = []
    for name in names:
        preds = run_ridge(df, fold) if name == "ridge" else run_tree(name, df, fold)
        save_scores(preds, name + SUFFIX, SCORES_OUT)
        metrics.append(fit_metrics(df, preds).assign(model=name + SUFFIX))

    m = pd.concat(metrics, ignore_index=True)
    if METRICS_OUT.exists():
        old = pd.read_csv(METRICS_OUT)
        m = pd.concat([old[~old["model"].isin(m["model"].unique())], m], ignore_index=True)
    m.to_csv(METRICS_OUT, index=False)

    m["period"] = np.where(m["year"] <= 2017, "dev 2011-17", "test 2018-25")
    show = m[m["scheme"].isin(["oof", "walkforward"])]
    print(f"\nMean within-year R^2 (base vs {SET})")
    print(show.pivot_table(index=["model", "scheme"], columns="period", values="r2").round(3).to_string())
    summary()


if __name__ == "__main__":
    main()

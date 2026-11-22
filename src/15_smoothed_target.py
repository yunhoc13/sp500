"""Step 15 (analysis): train on a SMOOTHED target so the model learns less of
the market's temporary mispricing.

Problem: the target (this year's log EV/EBITDA) already contains the mispricing
we want to find. A flexible model can learn "Energy is cheap this year" as fair
value, so the 2020 Energy stocks look fairly priced instead of undervalued.

Experiment: for TRAINING rows only, replace the target by the company's average
year-demeaned log(EV/EBITDA) over the current and up to two previous years
(only past and present information, no look-ahead):

    d[i, s]       = log(EV/EBITDA)[i, s] - median over companies in year s
    smooth[i, s]  = mean of d[i, s-2 .. s] (years the company is in the panel)
    target[i, s]  = median in year s + smooth[i, s]

A one-year dip (e.g. 2020) is then partly averaged away in what the model learns.
Mispricing is still measured against the ACTUAL current multiple:

    mispricing = actual log(EV/EBITDA) - prediction

Same features, folds, preprocessing and hyperparameters as the original models
(results/best_params.json, chosen on 2011-2017); no re-tuning. The test period had
already been seen, so this is reported as an explanation of the accuracy-vs-signal
trade-off and does not change the final model.

Outputs: data/scores_smoothed.csv          scores for <model>_smooth
         results/smoothed_target_comparison.csv
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from preprocess import load_panel, TARGET, YEAR, GROUP, DATA, RESULTS
from scoring import score_model, fit_metrics
from splits import load_folds
from models import make_rf, make_xgb

WINDOW = 3                          # current year + 2 previous years
SCHEMES = ["oof", "walkforward"]
DEV = range(2011, 2018)
TEST = range(2018, 2026)
RET = "fwd_excess_ret_12m"


def smoothed_target(df):
    d = df[TARGET] - df.groupby(YEAR)[TARGET].transform("median")
    wide = d.groupby([df[GROUP], df[YEAR]]).first().unstack(YEAR)
    sm = wide.T.rolling(WINDOW, min_periods=1).mean().T   # past + current only
    smooth = sm.stack().rename("smooth")
    out = df[[GROUP, YEAR]].join(smooth, on=[GROUP, YEAR])["smooth"]
    return df.groupby(YEAR)[TARGET].transform("median") + out


def make_models():
    bp = json.load(open(RESULTS / "best_params.json"))
    models = {
        "ridge": lambda s: (lambda: Ridge(alpha=bp["ridge"][s])),
        "rf": lambda s: (lambda: make_rf(**bp["rf"])),
        "xgb": lambda s: (lambda: make_xgb(**bp["xgb"])),
    }
    if "--mlp" in sys.argv:
        from mlp import TorchMLP
        models["mlp"] = lambda s: (lambda: TorchMLP(**bp["mlp"]))
    return models


def backtest(scores, panel, years):
    """Same definitions as 10_backtest.py: 5 groups, year-averaged."""
    x = scores.merge(panel[["cik", "year", RET]], on=["cik", "year"])
    x = x[x["year"].isin(years)]
    x["g"] = np.ceil(x.groupby(["model", "scheme", "year"])["mispricing"]
                     .rank(pct=True, method="first") * 5).astype(int)
    by = x.groupby(["model", "scheme", "year", "g"])[RET].mean().unstack("g")
    y = pd.DataFrame({"cheapest": by[1], "spread": by[1] - by[5]})
    y["ic"] = x.groupby(["model", "scheme", "year"]).apply(
        lambda v: -v["mispricing"].rank().corr(v[RET].rank()), include_groups=False)
    g = y.groupby(["model", "scheme"])
    no20 = y[y.index.get_level_values("year") != 2020].groupby(["model", "scheme"])
    return pd.DataFrame({"cheapest_avg": g["cheapest"].mean(), "spread_avg": g["spread"].mean(),
                         "spread_without_2020": no20["spread"].mean(), "ic_avg": g["ic"].mean()})


def main():
    df = load_panel()
    fold = load_folds(df)
    actual = df[TARGET].copy()
    df_s = df.copy()
    df_s[TARGET] = smoothed_target(df)
    print("correlation of smoothed and actual target:", round(df_s[TARGET].corr(actual), 3))

    preds = []
    for name, factory in make_models().items():
        for s in SCHEMES:
            p = score_model(df_s, fold, factory(s), s)
            p["mispricing"] = actual.loc[p["row"]].to_numpy() - p["pred_log_ev_ebitda"]
            preds.append(p.assign(model=f"{name}_smooth"))
            print(f"done {name} {s}", flush=True)
    new = pd.concat(preds, ignore_index=True)
    new[["ticker", "year", "model", "scheme", "pred_log_ev_ebitda", "mispricing", "cik", "fold"]] \
        .to_csv(DATA / "scores_smoothed.csv", index=False)

    # ---- compare with the original models (R^2 against the ACTUAL multiple)
    old = pd.read_csv(DATA / "scores.csv")
    base_models = [m.replace("_smooth", "") for m in new["model"].unique()]
    old = old[old["model"].isin(base_models) & old["scheme"].isin(SCHEMES)]
    both = pd.concat([old, new.drop(columns="row")], ignore_index=True)
    panel = pd.read_csv(DATA / "panel.csv")

    rows = []
    for (m, s), g in both.groupby(["model", "scheme"]):
        g = g.assign(row=g.set_index(["cik", "year"]).index.map(
            pd.Series(df.index, index=pd.MultiIndex.from_frame(df[["cik", "year"]])))).copy()
        g = g.assign(scheme=s)
        fm = fit_metrics(df, g)
        rows.append({"model": m, "scheme": s,
                     "r2_dev": fm[fm["year"].isin(DEV)]["r2"].mean(),
                     "r2_test": fm[fm["year"].isin(TEST)]["r2"].mean()})
    r2 = pd.DataFrame(rows).set_index(["model", "scheme"])
    test = backtest(both, panel, TEST).add_prefix("test_")
    dev = backtest(both, panel, DEV)[["spread_avg", "ic_avg"]].add_prefix("dev_")
    out = r2.join(test).join(dev).sort_index()
    out.round(4).to_csv(RESULTS / "smoothed_target_comparison.csv")
    pd.set_option("display.width", 200)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()

"""Fit-vs-signal curve: how does the investment signal change as a model fits
the fair multiple more closely?

For each model, capacity is stepped from very simple to very flexible
(Ridge: alpha from large to small; RF: min_samples_leaf from large to small;
XGBoost: max_depth from 1 to 6). For every setting we score all years with
the main `oof` scheme and record, for the development (2011-17) and test
(2018-25) periods:
    fit    = mean within-year out-of-fold R^2 of log(EV/EBITDA)
    signal = 5-group spread (cheapest minus most expensive, 12-month return
             vs SPY) and IC, computed with the same functions as 10_backtest.py
Both feature sets are run: base features and the price-free set (nopx).

This is an analysis, not model selection: nothing here changes the final
model. Results are appended as they finish, so an interrupted run resumes.

Outputs
    results/fit_signal_curve.csv
    figures/fit_signal_curve.png

    python src/13_fit_signal.py               # ridge, rf, xgb (xgb needs xgboost)
    python src/13_fit_signal.py --plot        # only redraw the figure
"""
import importlib
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from preprocess import load_panel, FEATURE_SETS, RESULTS, DEV_YEARS, TEST_YEARS
from scoring import score_model, fit_metrics
from splits import load_folds
from models import make_rf, make_xgb

bt = importlib.import_module("10_backtest")   # reuse the exact backtest metrics

OUT = RESULTS / "fit_signal_curve.csv"
FIG = RESULTS.parent / "figures" / "fit_signal_curve.png"
N_GROUPS = 5

# (model, capacity parameter, values ordered simple -> flexible, factory)
CURVES = [
    ("ridge", "alpha", [10000, 3000, 1000, 300, 100, 30, 10, 1],
     lambda v: Ridge(alpha=v)),
    ("rf", "min_samples_leaf", [300, 150, 80, 40, 20, 10, 5, 3],
     lambda v: make_rf(min_samples_leaf=v, max_features=0.33)),
    ("xgb", "max_depth", [1, 2, 3, 4, 6],
     lambda v: make_xgb(max_depth=v, n_estimators=200, learning_rate=0.02)),
]


def signal(preds, panel):
    """Spread and IC per period, using 10_backtest.py's own functions."""
    d = preds.merge(panel[["cik", "year", bt.RET]], on=["cik", "year"], how="inner")
    d = d.assign(model="m", scheme="s").dropna(subset=["mispricing", bt.RET])
    d[f"g{N_GROUPS}"] = bt.assign_groups(d, N_GROUPS)
    yearly = bt.yearly_stats(d, N_GROUPS)
    out = {}
    for label, years in (("dev", DEV_YEARS), ("test", TEST_YEARS)):
        s = bt.summarize(yearly, years).iloc[0]
        out[f"spread_{label}"] = s["spread_avg"]
        out[f"ic_{label}"] = s["ic_avg"]
    return out


def run(models):
    df = load_panel()
    fold = load_folds(df)
    panel = pd.read_csv(RESULTS.parent / "data" / "panel.csv")
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(
        columns=["model", "features", "param", "value"])
    for name, param, values, factory in CURVES:
        if name not in models:
            continue
        for fs, feats in FEATURE_SETS.items():
            for v in values:
                hit = done[(done["model"] == name) & (done["features"] == fs)
                           & (done["value"].astype(float) == float(v))]
                if len(hit):
                    continue
                t0 = time.time()
                preds = score_model(df, fold, lambda v=v: factory(v), "oof", features=feats)
                m = fit_metrics(df, preds)
                row = {"model": name, "features": fs, "param": param, "value": v,
                       "r2_dev": m.loc[m["year"].isin(DEV_YEARS), "r2"].mean(),
                       "r2_test": m.loc[m["year"].isin(TEST_YEARS), "r2"].mean(),
                       **signal(preds, panel)}
                done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
                done.to_csv(OUT, index=False)
                print(f"{name:5s} {fs:4s} {param}={v:<6} R2 test={row['r2_test']:.3f} "
                      f"spread test={row['spread_test']:+.3f} ({time.time() - t0:.0f}s)", flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.read_csv(OUT)
    colors = {"ridge": "#2a78d6", "rf": "#eb6834", "xgb": "#1baf7a"}   # categorical slots 1-3
    names = {"ridge": "Ridge", "rf": "Random Forest", "xgb": "XGBoost"}
    fs_style = {"base": dict(ls="-", mfc=None), "nopx": dict(ls="--", mfc="#fcfcfb")}
    fs_name = {"base": "base features", "nopx": "price-free features"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="#fcfcfb")
    for ax, period, title in ((axes[0], "test", "Test period 2018-2025"),
                              (axes[1], "dev", "Development period 2011-2017")):
        ax.set_facecolor("#fcfcfb")
        for (name, fs), g in d.groupby(["model", "features"], sort=False):
            st = fs_style[fs]
            ax.plot(g[f"r2_{period}"], g[f"spread_{period}"] * 100, ls=st["ls"], lw=2,
                    marker="o", ms=6, color=colors[name],
                    mfc=st["mfc"] or colors[name], mec=colors[name],
                    label=f"{names[name]}, {fs_name[fs]}")
        ax.axhline(0, color="#c9c8c2", lw=1)
        ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
        ax.set_xlabel("Fit: mean within-year out-of-fold R²", fontsize=9, color="#52514e")
        ax.set_ylabel("Signal: 5-group spread vs SPY (% per year)", fontsize=9, color="#52514e")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c8c2")
        ax.tick_params(colors="#52514e", labelsize=8)
        ax.grid(True, color="#ecebe7", lw=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle("More fit, less signal? Each point is one capacity setting "
                 "(simple → flexible along each line)", x=0.01, ha="left",
                 fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=200)
    plt.close(fig)
    print(f"saved {FIG}")


def main():
    if "--plot" not in sys.argv:
        models = [a for a in sys.argv[1:] if not a.startswith("--")] or ["ridge", "rf", "xgb"]
        if "xgb" in models and importlib.util.find_spec("xgboost") is None:
            print("xgboost is not installed here: skipping xgb")
            models = [m for m in models if m != "xgb"]
        run(models)
    plot()


if __name__ == "__main__":
    main()

"""SHAP explanations for a tree model's fair-multiple predictions.

For every (year, fold) of the main `oof` scheme, the fold model is refit
exactly as in 08_tree_models.py and its test companies are explained with
TreeSHAP. SHAP values are in log(EV/EBITDA) units: +0.10 means that feature
raised the company's fair multiple by roughly 10%.

Outputs
    results/shap_values_<model>.csv        one row per company-year
    results/shap_importance_<model>.csv    mean |SHAP| per feature (all / dev / test)
    results/figures/shap_importance_<model>.png
    results/top10_<model>_<year>.csv       demo: 10 most undervalued companies,
                                           why, and what happened next

    python src/09_shap.py                  # model with the best dev R^2, demo year 2024
    python src/09_shap.py xgb 2023
"""
import json
import sys

import numpy as np
import pandas as pd

from preprocess import (load_panel, Preprocessor, sector_levels, FEATURES, TARGET,
                        DATA, RESULTS, DEV_YEARS)
from splits import load_folds, iter_splits
from models import MODELS

LABELS = {
    "log_mcap": "Size (log market cap)", "rev_growth": "Revenue growth",
    "eps_growth": "EPS growth", "op_margin": "Operating margin", "roe": "ROE",
    "debt_to_equity": "Debt / equity", "volatility": "Volatility",
    "beta": "Beta", "sector": "Sector",
}
PCT = {"rev_growth", "eps_growth", "op_margin", "roe", "volatility"}


def best_model_name():
    m = pd.read_csv(RESULTS / "fit_metrics.csv")
    m = m[(m["scheme"] == "oof") & (m["year"].isin(DEV_YEARS)) & (m["model"].isin(MODELS))]
    return m.groupby("model")["r2"].mean().idxmax()


def explain(df, fold, name):
    import shap
    factory = MODELS[name][0]
    params = json.loads((RESULTS / "best_params.json").read_text())[name]
    sectors = sector_levels(df)
    out = []
    for t, f, tr, te in iter_splits(df, fold, "oof"):
        prep = Preprocessor(sectors).fit(df.loc[tr])
        model = factory(**params)
        model.fit(prep.transform_X(df.loc[tr]), prep.target(df.loc[tr]))
        X_te = prep.transform_X(df.loc[te])
        ex = shap.TreeExplainer(model)
        sv = np.asarray(ex.shap_values(X_te))
        base = float(np.ravel(ex.expected_value)[0])
        names = prep.feature_names
        vals = pd.DataFrame(sv, columns=names, index=te)
        sec_cols = [c for c in names if c.startswith("sector_")]
        vals["sector"] = vals[sec_cols].sum(axis=1)
        vals = vals.drop(columns=sec_cols)
        vals.insert(0, "base", base)
        vals.insert(0, "fold", f)
        vals["pred_log_ev_ebitda"] = prep.predict_level(df, te, sv.sum(axis=1) + base)
        out.append(vals)
        print(f"  {name} {t} fold {f}: {len(te)} explained", flush=True)
    res = pd.concat(out)
    res.insert(0, "year", df.loc[res.index, "year"])
    res.insert(0, "ticker", df.loc[res.index, "ticker"])
    return res


def importance(shap_df):
    cols = FEATURES + ["sector"]
    a = shap_df[cols].abs()
    imp = pd.DataFrame({
        "all_years": a.mean(),
        "dev_2011_17": a[shap_df["year"] <= 2017].mean(),
        "test_2018_25": a[shap_df["year"] >= 2018].mean(),
    }).sort_values("all_years", ascending=False)
    imp.index.name = "feature"
    return imp


def plot_importance(imp, name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = imp["all_years"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.barh([LABELS[i] for i in s.index], s.to_numpy(), height=0.6, color="#2a78d6")
    for y, v in enumerate(s.to_numpy()):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=8, color="#52514e")
    ax.set_xlabel("Mean |SHAP value| (log EV/EBITDA)", color="#52514e", fontsize=9)
    ax.set_title(f"What drives the fair EV/EBITDA multiple ({name.upper()}, all years)",
                 loc="left", fontsize=10, color="#0b0b0b")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(colors="#52514e", labelsize=8, length=0)
    ax.xaxis.grid(True, color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fmt_value(feat, v):
    return f"{v:+.0%}" if feat in PCT else f"{v:.2f}"


def top10(df_raw, shap_df, year, k=10):
    """Most undervalued companies in `year` and the features that raised their fair value."""
    s = shap_df[shap_df["year"] == year].copy()
    s["actual"] = df_raw.loc[s.index, TARGET]
    s["mispricing"] = s["actual"] - s["pred_log_ev_ebitda"]
    s = s.nsmallest(k, "mispricing")
    rows = []
    for i, r in s.iterrows():
        contrib = r[FEATURES + ["sector"]].sort_values(ascending=False)
        reasons = []
        for feat, c in contrib.head(3).items():
            if c <= 0:
                break
            val = df_raw.loc[i, "sector"] if feat == "sector" else fmt_value(feat, df_raw.loc[i, feat])
            reasons.append(f"{LABELS[feat]} {val} ({c:+.2f})")
        rows.append({
            "ticker": r["ticker"], "sector": df_raw.loc[i, "sector"],
            "actual_ev_ebitda": round(float(np.exp(r["actual"])), 1),
            "fair_ev_ebitda": round(float(np.exp(r["pred_log_ev_ebitda"])), 1),
            "discount_vs_fair": f"{np.exp(r['mispricing']) - 1:.0%}",
            "why_fair_value_is_higher": "; ".join(reasons),
            "fwd_ret_12m": f"{df_raw.loc[i, 'fwd_ret_12m']:.1%}",
            "fwd_excess_ret_12m_vs_spy": f"{df_raw.loc[i, 'fwd_excess_ret_12m']:.1%}",
        })
    return pd.DataFrame(rows)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else best_model_name()
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    print(f"SHAP for {name}, demo year {year}", flush=True)
    df = load_panel()
    fold = load_folds(df)
    df_raw = pd.read_csv(DATA / "panel.csv").loc[df.index]   # unwinsorized values for display

    shap_df = explain(df, fold, name)
    shap_df.to_csv(RESULTS / f"shap_values_{name}.csv", index=False)

    imp = importance(shap_df)
    imp.round(4).to_csv(RESULTS / f"shap_importance_{name}.csv")
    (RESULTS / "figures").mkdir(exist_ok=True)
    plot_importance(imp, name, RESULTS / "figures" / f"shap_importance_{name}.png")
    print(imp.round(3).to_string())

    t10 = top10(df_raw, shap_df, year)
    t10.to_csv(RESULTS / f"top10_{name}_{year}.csv", index=False)
    print(t10.to_string(index=False))


if __name__ == "__main__":
    main()

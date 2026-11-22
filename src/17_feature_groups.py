"""Feature-group ablation (the proposal's "remove one feature group at a time").

For each model, one group of inputs is removed and the model is scored again
with the main `oof` scheme. Hyperparameters are the ones already chosen for the
base model on 2011-2017 (results/best_params.json), with no re-tuning, so every
row differs from the base row only by the missing group.

Groups (proposal: growth, profitability, risk, size, sector; risk is split into
leverage and market risk because the two behave very differently):
    growth         revenue growth, EPS growth
    profitability  operating margin, ROE
    leverage       debt-to-equity
    market_risk    volatility, beta
    size           log market cap
    sector         sector one-hots

Reported for development (2011-17) and test (2018-25): mean within-year
out-of-fold R^2, and, with the same functions as 10_backtest.py, the 5-group
spread, the spread without 2020 and the IC. Like the other analyses in steps
12-15, this explains the model, it does not change the final model.

Output: results/feature_group_ablation.csv (rows are appended, so a run resumes)

    python src/17_feature_groups.py               # ridge, rf, xgb (xgb needs xgboost)
    python src/17_feature_groups.py xgb           # one model
    python src/17_feature_groups.py --mlp         # also the MLP (needs torch)
"""
import importlib
import importlib.util
import json
import sys
import time

import pandas as pd
from sklearn.linear_model import Ridge

from preprocess import load_panel, FEATURES, RESULTS, DATA, DEV_YEARS, TEST_YEARS
from scoring import score_model, fit_metrics
from splits import load_folds
from models import make_rf, make_xgb

bt = importlib.import_module("10_backtest")
OUT = RESULTS / "feature_group_ablation.csv"
N_GROUPS = 5

GROUPS = {
    "none": [],
    "growth": ["rev_growth", "eps_growth"],
    "profitability": ["op_margin", "roe"],
    "leverage": ["debt_to_equity"],
    "market_risk": ["volatility", "beta"],
    "size": ["log_mcap"],
    "sector": ["<sector one-hots>"],
}


def factories():
    bp = json.loads((RESULTS / "best_params.json").read_text())
    f = {"ridge": lambda: Ridge(alpha=bp["ridge"]["oof"]),
         "rf": lambda: make_rf(**bp["rf"]),
         "xgb": lambda: make_xgb(**bp["xgb"])}
    if "mlp" in bp:
        def mlp():
            from mlp import TorchMLP
            p = dict(bp["mlp"]); p["hidden"] = tuple(p["hidden"])
            return TorchMLP(**p)
        f["mlp"] = mlp
    return f


def evaluate(df, preds, panel):
    m = fit_metrics(df, preds)
    d = preds.merge(panel[["cik", "year", bt.RET]], on=["cik", "year"])
    d = d.assign(model="m", scheme="s").dropna(subset=["mispricing", bt.RET])
    d[f"g{N_GROUPS}"] = bt.assign_groups(d, N_GROUPS)
    yearly = bt.yearly_stats(d, N_GROUPS)
    test, dev = bt.summarize(yearly, TEST_YEARS).iloc[0], bt.summarize(yearly, DEV_YEARS).iloc[0]
    return {"r2_dev": m.loc[m["year"].isin(DEV_YEARS), "r2"].mean(),
            "r2_test": m.loc[m["year"].isin(TEST_YEARS), "r2"].mean(),
            "spread_dev": dev["spread_avg"], "spread_test": test["spread_avg"],
            "spread_test_without_2020": bt.leave_one_year_out(yearly, TEST_YEARS)["without 2020"].iloc[0],
            "ic_test": test["ic_avg"]}


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or ["ridge", "rf", "xgb"]
    if "--mlp" in sys.argv and "mlp" not in names:
        names.append("mlp")
    if "xgb" in names and importlib.util.find_spec("xgboost") is None:
        print("xgboost is not installed here: skipping xgb"); names.remove("xgb")
    if "mlp" in names and importlib.util.find_spec("torch") is None:
        print("torch is not installed here: skipping mlp"); names.remove("mlp")
    fac = factories()
    df = load_panel()
    fold = load_folds(df)
    panel = pd.read_csv(DATA / "panel.csv")
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["model", "dropped_group"])

    for name in names:
        for group, cols in GROUPS.items():
            if ((done["model"] == name) & (done["dropped_group"] == group)).any():
                continue
            t0 = time.time()
            feats = [c for c in FEATURES if c not in cols]
            preds = score_model(df, fold, fac[name], "oof", features=feats,
                                use_sector=(group != "sector"))
            row = {"model": name, "dropped_group": group,
                   "dropped_features": "+".join(cols) or "-", **evaluate(df, preds, panel)}
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(OUT, index=False)
            print(f"{name:5s} drop {group:13s} R2 test={row['r2_test']:.3f} "
                  f"spread test={row['spread_test']:+.3f} ({time.time() - t0:.0f}s)", flush=True)

    out = pd.read_csv(OUT)
    print("\nTest R^2 / spread by dropped group")
    print(out.pivot_table(index="dropped_group", columns="model",
                          values=["r2_test", "spread_test"]).round(3).to_string())


if __name__ == "__main__":
    main()

"""Demo: the final model's most undervalued S&P 500 stocks on one formation date,
with SHAP explanations of why each one looks cheap.

The final model is Random Forest (highest development-period R^2, see README
"Model selection rule"), with its tuned setting from results/best_params.json.
Scoring follows the main `oof` rules exactly: a company is scored by a model
trained on years <= the demo year that never saw any of that company's rows.

Current-year demo (needs data/panel_2026.csv):
    same columns as data/panel.csv, one row per S&P 500 member on April 1, 2026,
    year = 2026, using only 10-Ks filed before that date; fwd_ret_12m and
    fwd_excess_ret_12m may be empty (the future is unknown). Companies that are
    new to the panel get a fold from a separate seeded shuffle (splits.py), so
    the existing fold assignment does not change.

Back-test demo (works now): any year already in panel.csv, e.g. 2025, where
the realized 12-month returns are shown next to each pick.

Outputs
    results/top<k>_<model>_<year>.csv

    python src/16_demo_top20.py                 # 2026, top 20, rf (needs panel_2026.csv + shap)
    python src/16_demo_top20.py 2025            # a year already in the panel
    python src/16_demo_top20.py 2025 --k 10 --model xgb
"""
import importlib
import sys

import pandas as pd

from preprocess import load_panel, DATA, RESULTS
from splits import load_folds

shap09 = importlib.import_module("09_shap")   # reuse explain() and the table builder


def arg(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main():
    pos = [a for i, a in enumerate(sys.argv[1:], 1)
           if not a.startswith("--") and not sys.argv[i - 1].startswith("--")]
    year = int(pos[0]) if pos else 2026
    k = arg("--k", 20)
    name = arg("--model", "rf")

    panel = pd.read_csv(DATA / "panel.csv")
    extra = None
    if year not in set(panel["year"]):
        extra = DATA / f"panel_{year}.csv"
        if not extra.exists():
            sys.exit(f"{extra} not found. Build it with the same columns as panel.csv "
                     f"(year={year}, forward returns may be empty), or run a year already "
                     f"in the panel, e.g. `python src/16_demo_top20.py 2025`.")
    print(f"Top {k} undervalued stocks, {name}, April 1 {year}", flush=True)

    df = load_panel(extra=extra)
    fold = load_folds(df, allow_new=True)
    n_year = int((df["year"] == year).sum())
    print(f"{n_year} companies scored in {year}", flush=True)

    shap_df = shap09.explain(df, fold, name, years=[year])
    raw = panel if extra is None else pd.concat([panel, pd.read_csv(extra)], ignore_index=True)
    table = shap09.top10(raw, shap_df, year, k=k)
    out = RESULTS / f"top{k}_{name}_{year}.csv"
    table.to_csv(out, index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(table.to_string(index=False))
    rets = pd.to_numeric(table["fwd_excess_ret_12m_vs_spy"].str.rstrip("%"), errors="coerce")
    if rets.notna().any():
        print(f"\nAverage 12-month return vs SPY of these {k}: {rets.mean():+.1f}% "
              f"({(rets > 0).sum()} of {rets.notna().sum()} beat SPY)")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

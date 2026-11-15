"""Step 10: backtest every model's mispricing score against simple baselines.

Question: if we had bought the cheapest stocks on each April 1, would they have
beaten SPY over the next 12 months, and do ML scores beat simple value rules?

Inputs : data/panel.csv   (forward returns, features)
         data/scores.csv  (ticker, cik, year, model, scheme, mispricing, ...)
Outputs: results/backtest_yearly_g5.csv / _g10.csv   per model, scheme, year
         results/backtest_summary.csv                 dev and test periods, 5 and 10 groups
         results/backtest_leave_one_year_out.csv      test spread with each year removed
         results/backtest_group_returns.csv           avg return by group (test period)
         results/backtest_cheapest_profile.csv        risk / growth profile by group
         results/backtest_cheapest_sectors.csv        sector mix of the cheapest group
         results/backtest_2020_cheapest.csv           stocks in the 2020 cheapest group
         figures/backtest_groups.png                  bar chart of group returns

    python src/10_backtest.py

Conventions
-----------
mispricing = actual log(EV/EBITDA) - predicted; NEGATIVE = undervalued.
Group 1 = cheapest (most negative mispricing), group N = most expensive.
spread = group 1 return - group N return (positive = cheap beat expensive).
IC = Spearman correlation between -mispricing and forward excess return
     (positive = the score ranked future returns correctly).
Every year counts equally: we average within a year first, then across years.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS, FIGURES = ROOT / "data", ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

DEV_YEARS = list(range(2011, 2018))
TEST_YEARS = list(range(2018, 2026))
KEYS = ["model", "scheme"]
RET = "fwd_excess_ret_12m"
# Main comparison shown in the chart and the profile tables (if present in scores)
MAIN = [("cheap", "baseline"), ("sector_cheap", "baseline"), ("ridge", "oof"),
        ("rf", "oof"), ("xgb", "oof"), ("mlp", "oof")]


# ---------------------------------------------------------------- load
def load():
    panel = pd.read_csv(DATA / "panel.csv")
    scores = pd.read_csv(DATA / "scores.csv")
    cols = ["ticker", "cik", "year", "model", "scheme", "mispricing"]

    base = panel[["ticker", "cik", "year", "sector", "log_ev_ebitda"]]
    year_med = base.groupby("year")["log_ev_ebitda"].transform("median")
    sec_med = base.groupby(["year", "sector"])["log_ev_ebitda"].transform("median")
    baselines = pd.concat([
        base.assign(model="cheap", mispricing=base["log_ev_ebitda"] - year_med),
        base.assign(model="sector_cheap", mispricing=base["log_ev_ebitda"] - sec_med),
    ]).assign(scheme="baseline")

    allsc = pd.concat([scores[cols], baselines[cols]], ignore_index=True)
    # inner join: only company-years present in the current panel are evaluated
    allsc = allsc.merge(panel[["cik", "year", RET]], on=["cik", "year"], how="inner")
    n_lost = len(scores) - (allsc["scheme"] != "baseline").sum()
    if n_lost:
        print(f"note: {n_lost} score rows are not in the current panel.csv and were skipped "
              "(scores built from an older panel?)")
    return panel, allsc.dropna(subset=["mispricing", RET])


def assign_groups(df, n):
    rank = df.groupby(KEYS + ["year"])["mispricing"].rank(pct=True, method="first")
    return np.ceil(rank * n).astype(int)


# ---------------------------------------------------------------- metrics
def yearly_stats(df, n):
    g = f"g{n}"
    by_group = df.groupby(KEYS + ["year", g])[RET].mean().unstack(g)
    out = pd.DataFrame(index=by_group.index)
    out["n_stocks"] = df.groupby(KEYS + ["year"]).size()
    out["cheapest"] = by_group[1]
    out["most_expensive"] = by_group[n]
    out["spread"] = out["cheapest"] - out["most_expensive"]
    out["ic"] = df.groupby(KEYS + ["year"]).apply(
        lambda x: -x["mispricing"].rank().corr(x[RET].rank()), include_groups=False)
    return out


def summarize(yearly, years):
    y = yearly[yearly.index.get_level_values("year").isin(years)]
    g = y.groupby(KEYS)
    n = g.size()
    s = pd.DataFrame({
        "n_years": n,
        "cheapest_avg": g["cheapest"].mean(),
        "cheapest_hit": g["cheapest"].apply(lambda v: (v > 0).mean()),
        "spread_avg": g["spread"].mean(),
        "spread_median": g["spread"].median(),
        "spread_t": g["spread"].mean() / (g["spread"].std() / np.sqrt(n)),
        "spread_pos_years": g["spread"].apply(lambda v: (v > 0).sum()),
        "ic_avg": g["ic"].mean(),
        "ic_std": g["ic"].std(),
    })
    s["ic_t"] = s["ic_avg"] / (s["ic_std"] / np.sqrt(n))
    return s


def leave_one_year_out(yearly, years):
    y = yearly[yearly.index.get_level_values("year").isin(years)].reset_index()
    out = {f"all {years[0]}-{years[-1]}": y.groupby(KEYS)["spread"].mean()}
    for drop in years:
        out[f"without {drop}"] = y[y["year"] != drop].groupby(KEYS)["spread"].mean()
    return pd.DataFrame(out)


# ---------------------------------------------------------------- run
def main():
    panel, df = load()
    for n in (5, 10):
        df[f"g{n}"] = assign_groups(df, n)
    present = [m for m in MAIN if m in set(map(tuple, df[KEYS].drop_duplicates().to_numpy()))]

    yearly = {n: yearly_stats(df, n) for n in (5, 10)}
    for n, y in yearly.items():
        y.round(4).to_csv(RESULTS / f"backtest_yearly_g{n}.csv")

    summ = []
    for n, y in yearly.items():
        for period, years in (("test", TEST_YEARS), ("dev", DEV_YEARS)):
            summ.append(summarize(y, years).assign(groups=n, period=period))
    summary = pd.concat(summ).reset_index().set_index(KEYS + ["period", "groups"]).sort_index()
    summary.round(4).to_csv(RESULTS / "backtest_summary.csv")

    loo = leave_one_year_out(yearly[5], TEST_YEARS)
    loo.round(4).to_csv(RESULTS / "backtest_leave_one_year_out.csv")

    test = df[df["year"].isin(TEST_YEARS)]
    group_ret = (test.groupby(KEYS + ["year", "g5"])[RET].mean()
                     .groupby(KEYS + ["g5"]).mean().unstack("g5"))
    group_ret.round(4).to_csv(RESULTS / "backtest_group_returns.csv")

    feats = ["beta", "volatility", "rev_growth", "op_margin", "debt_to_equity"]
    t = test.merge(panel[["cik", "year", "sector"] + feats], on=["cik", "year"])
    t = t.set_index(KEYS).loc[present].reset_index()
    t.groupby(KEYS + ["g5"])[feats].mean().round(3).to_csv(RESULTS / "backtest_cheapest_profile.csv")
    c1 = t[t["g5"] == 1]
    sectors = (pd.crosstab([c1["model"], c1["scheme"]], c1["sector"], normalize="index") * 100).round(1)
    allmix = (panel[panel["year"].isin(TEST_YEARS)]["sector"].value_counts(normalize=True) * 100).round(1)
    sectors.loc[("all stocks", ""), :] = allmix
    sectors.to_csv(RESULTS / "backtest_cheapest_sectors.csv")

    y20 = df[(df["year"] == 2020) & (df["g5"] == 1)].merge(
        panel[["cik", "year", "sector", "fwd_ret_12m"]], on=["cik", "year"])
    y20 = y20.set_index(KEYS).loc[present].reset_index()
    y20.sort_values(KEYS + [RET], ascending=[True, True, False])[
        KEYS + ["ticker", "sector", "mispricing", "fwd_ret_12m", RET]].round(4).to_csv(
        RESULTS / "backtest_2020_cheapest.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        group_ret.loc[present].T.plot(kind="bar", ax=ax)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Group (1 = cheapest, 5 = most expensive)")
        ax.set_ylabel("Avg 12-month return vs SPY")
        ax.set_title(f"Test period {TEST_YEARS[0]}-{TEST_YEARS[-1]}: average return by valuation group")
        ax.legend(title="model / scheme", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "backtest_groups.png", dpi=150)
    except ImportError:
        print("matplotlib not installed: skipped the chart")

    # ---- console report
    pd.set_option("display.width", 160)
    cols = ["cheapest_avg", "spread_avg", "spread_median", "spread_t", "spread_pos_years",
            "ic_avg", "ic_t"]
    test5 = summary.xs(("test", 5), level=("period", "groups"))[cols]
    print(f"\nTest period {TEST_YEARS[0]}-{TEST_YEARS[-1]}, 5 groups (sorted by spread):")
    print(test5.sort_values("spread_avg", ascending=False).round(3).to_string())
    print("\nSpread with each test year removed (main models):")
    print(loo.loc[present].round(3).to_string())
    print(f"\nSaved results/backtest_*.csv and figures/backtest_groups.png")


if __name__ == "__main__":
    main()

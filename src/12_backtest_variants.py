"""Step 12: ways to USE the mispricing score better, and portfolio performance.

These are analysis experiments. The test period (2018-2025) had already been seen
when they were designed, so they are reported for every model and are NOT used to
change the final model (see README "Model selection rule").

Variants (each applied to every model and baseline):
  base           rank by mispricing within the year (same as 10_backtest.py)
  sector_neutral rank within each year x sector, so every sector supplies its own
                 cheapest stocks (removes sector bets such as "all of Energy in 2020")
  momentum       base groups, but the cheapest group keeps only stocks whose past
                 12-1 month return was >= 0 (avoid "falling knives" / value traps)
  combo          average of the model's rank and the simple cheap rank
                 (cheap on raw multiple AND cheap relative to fundamentals)

Momentum (12-1): return from 12 months to 1 month before April 1, from Yahoo
adjusted prices. Only information available before the formation date is used.

Portfolio performance: the cheapest group held for 12 months, rebalanced every
April 1 (equal weight), versus SPY. Annual returns, so 8 test observations.

Inputs : data/panel.csv, data/scores.csv,
         data/raw/prices_adjclose.csv.gz (only to build data/momentum.csv once)
Outputs: data/momentum.csv
         results/variants_summary.csv        test + dev metrics per variant
         results/variants_yearly.csv         yearly spread per variant
         results/portfolio_performance.csv   cumulative, annualized, Sharpe, drawdown
         results/portfolio_yearly.csv        yearly returns of each portfolio
         figures/cumulative_returns.png      growth of $1, test period
         figures/variants_spread.png         spread with and without 2020

    python src/12_backtest_variants.py
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
RET = "fwd_excess_ret_12m"
N = 5
KEYS = ["model", "scheme"]
MAIN = [("cheap", "baseline"), ("sector_cheap", "baseline"), ("ridge", "oof"),
        ("rf", "oof"), ("xgb", "oof"), ("mlp", "oof")]
SCHEMES = ["baseline", "oof", "walkforward"]


# ---------------------------------------------------------------- momentum
def momentum(panel):
    path = DATA / "momentum.csv"
    if path.exists():
        return pd.read_csv(path)
    adj = pd.read_csv(DATA / "raw" / "prices_adjclose.csv.gz", index_col=0, parse_dates=True)
    rows = []
    for r in panel[["ticker", "cik", "year"]].itertuples(index=False):
        sym = r.ticker.replace(".", "-")
        if sym not in adj.columns:
            continue
        F = pd.Timestamp(year=r.year, month=4, day=1)
        s = adj[sym].dropna()
        p_end = s.loc[:F - pd.DateOffset(months=1)]
        p_start = s.loc[:F - pd.DateOffset(months=12)]
        # require prices within a week of each date
        if len(p_end) and len(p_start) and \
                (F - pd.DateOffset(months=1) - p_end.index[-1]).days <= 7 and \
                (F - pd.DateOffset(months=12) - p_start.index[-1]).days <= 7:
            rows.append((r.cik, r.year, p_end.iloc[-1] / p_start.iloc[-1] - 1))
    mom = pd.DataFrame(rows, columns=["cik", "year", "mom_12_1"])
    mom.to_csv(path, index=False)
    return mom


# ---------------------------------------------------------------- data
def load():
    panel = pd.read_csv(DATA / "panel.csv")
    scores = pd.read_csv(DATA / "scores.csv")
    base = panel[["cik", "year", "sector", "log_ev_ebitda"]]
    year_med = base.groupby("year")["log_ev_ebitda"].transform("median")
    sec_med = base.groupby(["year", "sector"])["log_ev_ebitda"].transform("median")
    baselines = pd.concat([
        base.assign(model="cheap", mispricing=base["log_ev_ebitda"] - year_med),
        base.assign(model="sector_cheap", mispricing=base["log_ev_ebitda"] - sec_med),
    ]).assign(scheme="baseline")
    cols = ["cik", "year", "model", "scheme", "mispricing"]
    df = pd.concat([scores[cols], baselines[cols]], ignore_index=True)
    df = df[df["scheme"].isin(SCHEMES)]
    df = df.merge(panel[["cik", "year", "ticker", "sector", "log_ev_ebitda",
                         "fwd_ret_12m", RET]], on=["cik", "year"], how="inner")
    df = df.merge(momentum(panel), on=["cik", "year"], how="left")
    df["spy_ret"] = df["fwd_ret_12m"] - df[RET]
    # simple cheap score (same for every model), for the combo variant
    df["cheap_score"] = df["log_ev_ebitda"] - df.groupby(KEYS + ["year"])["log_ev_ebitda"].transform("median")
    return panel, df.dropna(subset=["mispricing", RET])


def groups(score, by, n=N):
    return np.ceil(score.groupby(by).rank(pct=True, method="first") * n).astype(int)


def add_variants(df):
    """Return one long frame: rows = stock x model x scheme x variant, with group g."""
    yk = KEYS + ["year"]
    out = []
    b = df.assign(variant="base", score=df["mispricing"])
    b["g"] = groups(b["score"], [b[k] for k in yk])
    out.append(b)

    s = df.assign(variant="sector_neutral", score=df["mispricing"])
    s["g"] = groups(s["score"], [s[k] for k in yk + ["sector"]])
    out.append(s)

    # momentum: same groups as base, then drop negative-momentum stocks from group 1
    m = b.assign(variant="momentum")
    m = m[~((m["g"] == 1) & (m["mom_12_1"] < 0))]
    out.append(m)

    r_model = df.groupby(yk)["mispricing"].rank(pct=True)
    r_cheap = df.groupby(yk)["cheap_score"].rank(pct=True)
    c = df.assign(variant="combo", score=(r_model + r_cheap) / 2)
    c["g"] = groups(c["score"], [c[k] for k in yk])
    out.append(c)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- metrics
def yearly(v):
    k = ["variant"] + KEYS + ["year"]
    by_g = v.groupby(k + ["g"])[RET].mean().unstack("g")
    y = pd.DataFrame({"cheapest": by_g[1], "most_expensive": by_g[N]})
    y["spread"] = y["cheapest"] - y["most_expensive"]
    y["n_cheapest"] = v[v["g"] == 1].groupby(k).size()
    # IC on the ranking score (not defined for the momentum filter)
    r = v[v["variant"] != "momentum"]
    y["ic"] = r.groupby(k).apply(lambda x: -x["score"].rank().corr(x[RET].rank()),
                                 include_groups=False)
    return y


def summarize(y, years):
    t = y[y.index.get_level_values("year").isin(years)]
    g = t.groupby(["variant"] + KEYS)
    n = g.size()
    s = pd.DataFrame({
        "cheapest_avg": g["cheapest"].mean(),
        "spread_avg": g["spread"].mean(),
        "spread_median": g["spread"].median(),
        "spread_t": g["spread"].mean() / (g["spread"].std() / np.sqrt(n)),
        "spread_pos_years": g["spread"].apply(lambda x: (x > 0).sum()),
        "ic_avg": g["ic"].mean(),
        "avg_n_cheapest": g["n_cheapest"].mean(),
    })
    no20 = t[t.index.get_level_values("year") != 2020].groupby(["variant"] + KEYS)
    s["spread_without_2020"] = no20["spread"].mean()
    s["cheapest_without_2020"] = no20["cheapest"].mean()
    return s


def portfolio(v, years):
    """Equal-weight cheapest group, rebalanced each April 1, vs SPY."""
    k = ["variant"] + KEYS + ["year"]
    t = v[v["year"].isin(years)]
    p = t[t["g"] == 1].groupby(k).agg(port=("fwd_ret_12m", "mean"), spy=("spy_ret", "first"))
    p["expensive"] = t[t["g"] == N].groupby(k)["fwd_ret_12m"].mean()
    p["long_short"] = p["port"] - p["expensive"]
    rows = []
    for key, g in p.groupby(["variant"] + KEYS):
        g = g.sort_index()
        r, spy, ex = g["port"], g["spy"], g["port"] - g["spy"]
        wealth = (1 + r).cumprod()
        dd = (wealth / wealth.cummax().clip(lower=1) - 1).min()
        n = len(r)
        rows.append(dict(zip(["variant"] + KEYS, key), years=n,
                         cumulative=wealth.iloc[-1] - 1,
                         spy_cumulative=(1 + spy).prod() - 1,
                         ann_return=wealth.iloc[-1] ** (1 / n) - 1,
                         spy_ann_return=(1 + spy).prod() ** (1 / n) - 1,
                         ann_vol=r.std(), sharpe_rf0=r.mean() / r.std(),
                         spy_sharpe_rf0=spy.mean() / spy.std(),
                         info_ratio=ex.mean() / ex.std(),
                         max_drawdown_annual=dd,
                         beat_spy_years=int((ex > 0).sum()),
                         long_short_avg=g["long_short"].mean()))
    return pd.DataFrame(rows).set_index(["variant"] + KEYS), p


# ---------------------------------------------------------------- run
def main():
    panel, df = load()
    print(f"momentum available for {df['mom_12_1'].notna().mean():.1%} of rows")
    v = add_variants(df)
    y = yearly(v)
    y.round(4).to_csv(RESULTS / "variants_yearly.csv")

    summ = pd.concat([summarize(y, TEST_YEARS).assign(period="test"),
                      summarize(y, DEV_YEARS).assign(period="dev")])
    summ = summ.reset_index().set_index(["period", "variant"] + KEYS).sort_index()
    summ.round(4).to_csv(RESULTS / "variants_summary.csv")

    perf, pyear = portfolio(v, TEST_YEARS)
    perf.round(4).to_csv(RESULTS / "portfolio_performance.csv")
    pyear.round(4).to_csv(RESULTS / "portfolio_yearly.csv")

    # ---- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        spy = pyear.xs(("base", "cheap", "baseline"), level=["variant"] + KEYS)["spy"]
        yrs = [TEST_YEARS[0]] + [yr + 1 for yr in TEST_YEARS]
        ax.plot(yrs, np.r_[1, (1 + spy).cumprod()], color="black", lw=2.5, label="SPY")
        for (m, s) in MAIN:
            r = pyear.xs(("base", m, s), level=["variant"] + KEYS)["port"]
            ax.plot(yrs, np.r_[1, (1 + r).cumprod()], lw=1.5, label=f"{m} ({s})")
        ax.set_xlabel("April 1 of year")
        ax.set_ylabel("Growth of $1")
        ax.set_title("Cheapest group (equal weight, rebalanced each April 1) vs SPY, 2018-2026")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "cumulative_returns.png", dpi=150)

        t = summ.loc["test"]
        main_idx = [(var, m, s) for var in ["base", "sector_neutral", "momentum", "combo"]
                    for (m, s) in MAIN if (var, m, s) in t.index]
        tt = t.loc[main_idx, ["spread_avg", "spread_without_2020"]]
        piv = tt["spread_avg"].unstack("variant")[["base", "sector_neutral", "momentum", "combo"]]
        piv20 = tt["spread_without_2020"].unstack("variant")[piv.columns]
        order = [f"{m} ({s})" for m, s in MAIN]
        piv.index = piv20.index = [f"{m} ({s})" for m, s in piv.index]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        piv.loc[order].plot(kind="bar", ax=axes[0], legend=False)
        piv20.loc[order].plot(kind="bar", ax=axes[1], legend=False)
        axes[0].set_title("Spread, test 2018-2025")
        axes[1].set_title("Spread, test without 2020")
        for a in axes:
            a.axhline(0, color="gray", lw=0.8)
            a.set_xlabel("")
            a.tick_params(axis="x", labelrotation=30)
            for lab in a.get_xticklabels():
                lab.set_ha("right")
        axes[0].set_ylabel("Cheapest - most expensive group\n(avg 12-month return)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, title="variant")
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(FIGURES / "variants_spread.png", dpi=150)
    except ImportError:
        print("matplotlib not installed: skipped the charts")

    pd.set_option("display.width", 180)
    cols = ["cheapest_avg", "spread_avg", "spread_t", "spread_without_2020", "ic_avg"]
    t = summ.loc["test"]
    print("\nTest 2018-2025 (oof models + baselines):")
    for var in ["base", "sector_neutral", "momentum", "combo"]:
        print(f"\n[{var}]")
        print(t.loc[var].loc[[k for k in MAIN if k in t.loc[var].index], cols].round(3).to_string())
    print("\nPortfolio performance, cheapest group (base), test period:")
    print(perf.loc["base"].loc[[k for k in MAIN if k in perf.loc["base"].index]][
        ["cumulative", "spy_cumulative", "ann_return", "spy_ann_return", "sharpe_rf0",
         "spy_sharpe_rf0", "max_drawdown_annual", "beat_spy_years"]].round(3).to_string())


if __name__ == "__main__":
    main()

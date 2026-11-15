"""Step 5: merge everything into the company x year panel.

Inputs : data/membership.csv, data/ticker_cik.csv, data/fundamentals.csv,
         data/raw/sec/submissions/*.json, data/raw/prices_*.csv.gz, data/raw/splits.csv
Outputs: data/panel_full.csv   every member-year we could build (features may be NaN)
         data/panel.csv        rows with all features present -> use this for modeling
         data/data_report.txt  how many rows were dropped at each step (for the paper)

One row = one company on one formation date (April 1 of `year`).
"""
import json

import numpy as np
import pandas as pd

from config import DATA, RAW, RAW_SEC, BENCHMARK, formation_date, yahoo_symbol

FEATURES = ["log_mcap", "rev_growth", "eps_growth", "op_margin", "roe",
            "debt_to_equity", "volatility", "beta"]
OPTIONAL = ["gross_margin"]   # many service companies report no COGS; kept in the panel, not required
WINSORIZE = ["rev_growth", "eps_growth", "op_margin", "gross_margin", "roe",
             "debt_to_equity"]

# SIC code -> broad sector (first match wins). Approximate GICS-like groups.
SECTOR_RANGES = [
    ("Financials", 6000, 6799),
    ("Energy", 1300, 1399), ("Energy", 2900, 2999),
    ("Health Care", 2830, 2836), ("Health Care", 3840, 3851), ("Health Care", 8000, 8099),
    ("Technology", 3570, 3579), ("Technology", 3600, 3699), ("Technology", 3810, 3829),
    ("Technology", 7370, 7379),
    ("Communication", 4800, 4899), ("Communication", 2700, 2799), ("Communication", 7800, 7899),
    ("Utilities", 4900, 4999),
    ("Materials", 1000, 1299), ("Materials", 1400, 1499), ("Materials", 2600, 2699),
    ("Materials", 2800, 2899), ("Materials", 3300, 3399),
    ("Consumer", 2000, 2399), ("Consumer", 2500, 2599), ("Consumer", 3140, 3199),
    ("Consumer", 3710, 3716), ("Consumer", 3940, 3949), ("Consumer", 5200, 5999),
    ("Consumer", 7000, 7099), ("Consumer", 7900, 7999),
    ("Industrials", 1500, 1799), ("Industrials", 3400, 3569), ("Industrials", 3580, 3599),
    ("Industrials", 3700, 3799), ("Industrials", 4000, 4799), ("Industrials", 5000, 5199),
    ("Industrials", 7380, 7389), ("Industrials", 8700, 8799),
]


def sector_of(sic):
    if pd.isna(sic):
        return "Other"
    sic = int(sic)
    for name, lo, hi in SECTOR_RANGES:
        if lo <= sic <= hi:
            return name
    return "Other"


report = []
stages = {}          # per-year row counts at each step (for the coverage table)
def log(msg, frame, short):
    report.append(f"{msg:<60}{len(frame):>8}")
    stages[short] = frame.groupby("year").size()
    print(report[-1])


# ---------- load ----------------------------------------------------------
members = pd.read_csv(DATA / "membership.csv")
tc = pd.read_csv(DATA / "ticker_cik.csv")
fund = pd.read_csv(DATA / "fundamentals.csv", parse_dates=["period_end", "available_date"])
close = pd.read_csv(RAW / "prices_close.csv.gz", index_col=0, parse_dates=True)
adj = pd.read_csv(RAW / "prices_adjclose.csv.gz", index_col=0, parse_dates=True)
splits = pd.read_csv(RAW / "splits.csv", parse_dates=["date"])

sic = {}
for f in (RAW_SEC / "submissions").glob("CIK*.json"):
    meta = json.loads(f.read_text() or "{}")
    s = meta.get("sic")
    sic[int(f.stem[3:])] = int(s) if s not in (None, "") else np.nan

rets = adj.pct_change(fill_method=None)
spy_ret = rets[BENCHMARK]
split_by_sym = {s: g.set_index("date")["ratio"] for s, g in splits.groupby("symbol")}
fund_by_cik = {c: g.sort_values("period_end") for c, g in fund.groupby("cik")}


def split_factor(sym, after, upto=None):
    """Product of split ratios with after < date <= upto."""
    s = split_by_sym.get(sym)
    if s is None:
        return 1.0
    m = s.index > after
    if upto is not None:
        m &= s.index <= upto
    return float(np.prod(s[m].values)) if m.any() else 1.0


def last_price(series, date, max_gap_days=7):
    s = series.loc[:date].dropna()
    if s.empty or (date - s.index[-1]).days > max_gap_days:
        return None, None
    return s.index[-1], s.iloc[-1]


# ---------- build ---------------------------------------------------------
df = members.merge(tc[["ticker", "cik"]], on="ticker", how="left")
log("Member-years (S&P 500, April 1, 2011-2025)", df, "members")
df = df[df.cik.notna()].copy()
df["cik"] = df.cik.astype(int)
log("  after ticker -> CIK mapping", df, "cik")
df = df.sort_values(["year", "cik", "ticker"]).drop_duplicates(["year", "cik"])
log("  after removing share-class duplicates (GOOG/GOOGL)", df, "dedup")
df["sic"] = df.cik.map(sic)
df["sector"] = df.sic.map(sector_of)
df = df[df.sector != "Financials"]
log("  after excluding financials (SIC 6000-6799)", df, "nonfin")

rows = []
for r in df.itertuples(index=False):
    F = formation_date(r.year)
    sym = yahoo_symbol(r.ticker)
    g = fund_by_cik.get(r.cik)
    if g is None:
        continue
    # latest 10-K public before F, covering a fiscal year that ended within 18 months
    cur = g[(g.available_date < F) & (g.period_end >= F - pd.DateOffset(months=18))]
    if cur.empty:
        continue
    cur = cur.iloc[-1]
    prev = g[(g.period_end >= cur.period_end - pd.Timedelta(days=400)) &
             (g.period_end <= cur.period_end - pd.Timedelta(days=330))]
    prev = prev.iloc[-1] if len(prev) else None

    rec = dict(year=r.year, ticker=r.ticker, cik=r.cik, sector=r.sector,
               sic=r.sic, period_end=cur.period_end.date(),
               available_date=cur.available_date.date(),
               revenue=cur.revenue, interest_expense=cur.interest_expense,
               net_income=cur.net_income,
               shares_fixed=bool(cur.shares_fixed),
               op_income_fixed=bool(cur.op_income_fixed or cur.op_income_fallback),
               debt_negative=bool(cur.debt_negative), hold_truncated=False)

    # ---- price and market cap
    if sym in close.columns:
        p_date, p_close = last_price(close[sym], F)
    else:
        p_date, p_close = None, None
    if p_date is not None:
        raw_price = p_close * split_factor(sym, p_date)           # undo later splits
        # 10-K share counts are already restated for splits before the filing
        # date, so only splits between the filing and the price date apply
        shares = cur.shares * split_factor(sym, cur.available_date, p_date)
        rec["price"] = raw_price
        rec["mcap"] = raw_price * shares
        # volatility and beta: prior 252 trading days of daily returns
        pos = rets.index.get_loc(p_date)
        win = rets.iloc[max(0, pos - 251): pos + 1]
        x, y = win[BENCHMARK], win[sym]
        ok = x.notna() & y.notna()
        if ok.sum() >= 200:
            rec["volatility"] = y[ok].std() * np.sqrt(252)
            rec["beta"] = np.cov(y[ok], x[ok])[0, 1] / x[ok].var()
        # forward 12-month return (stock and SPY), from F to next April 1
        F1 = formation_date(r.year + 1)
        a0_date, a0 = last_price(adj[sym], F)
        a1_date, a1 = last_price(adj[sym], F1)
        s0_date, s0 = last_price(adj[BENCHMARK], F)
        s1_date, s1 = last_price(adj[BENCHMARK], F1)
        if a0 is not None and a1 is None:
            # stopped trading during the holding year (acquired / delisted):
            # use the last available price, and compare with SPY to that date
            after = adj[sym].loc[F:F1].dropna()
            if len(after):
                a1_date, a1 = after.index[-1], after.iloc[-1]
                s1_date, s1 = last_price(adj[BENCHMARK], a1_date)
                rec["hold_truncated"] = True
        if None not in (a0, a1, s0, s1):
            rec["fwd_ret_12m"] = a1 / a0 - 1
            rec["fwd_excess_ret_12m"] = (a1 / a0) - (s1 / s0)

    # ---- fundamentals-based fields
    rev, op, da = cur.revenue, cur.op_income, cur.da
    rec["ebitda"] = op + da if pd.notna(op) and pd.notna(da) else np.nan
    rec["debt"], rec["cash"] = cur.debt, cur.cash if pd.notna(cur.cash) else 0.0
    rec["debt_missing"] = bool(cur.debt_missing)
    if rev and rev > 0:
        rec["op_margin"] = op / rev if pd.notna(op) else np.nan
        rec["gross_margin"] = cur.gross_profit / rev if pd.notna(cur.gross_profit) else np.nan
    if pd.notna(cur.equity) and cur.equity > 0:
        rec["roe"] = cur.net_income / cur.equity
        rec["debt_to_equity"] = cur.debt / cur.equity
    if prev is not None:
        if pd.notna(prev.revenue) and prev.revenue > 0 and pd.notna(rev):
            rec["rev_growth"] = rev / prev.revenue - 1
        if pd.notna(prev.eps) and prev.eps != 0 and pd.notna(cur.eps):
            rec["eps_growth"] = (cur.eps - prev.eps) / abs(prev.eps)
    rows.append(rec)

panel = pd.DataFrame(rows)
log("  with a 10-K filed before April 1 (point-in-time)", panel, "10k")
panel = panel[panel.get("mcap").notna()] if "mcap" in panel else panel.iloc[0:0]
log("  with a price on the formation date (yfinance)", panel, "price")

panel["ev"] = panel.mcap + panel.debt - panel.cash
panel = panel[(panel.ebitda > 0) & (panel.ev > 0)].copy()
log("  with positive EBITDA and EV", panel, "ebitda>0")

# ---- data-quality filters (rows that survive the checks in step 3 but are
#      still implausible). Dropped rows are listed in data/dropped_rows.csv.
panel["ebitda_margin"] = panel.ebitda / panel.revenue
ir = panel.interest_expense.abs() / panel.debt.where(panel.debt > 0)
panel["debt_suspect"] = (panel.debt < 0) | (ir > 0.20)    # interest > 20% of debt
panel["pe"] = panel.mcap / panel.net_income.where(panel.net_income > 0)
checks = {
    "no debt data (debt tag missing, EV would be understated)": lambda p: p.debt_missing,
    "implausible market cap (< $1B or > $5T)": lambda p: ~p.mcap.between(1e9, 5e12),
    "debt negative or too small for its interest expense": lambda p: p.debt_suspect,
    "EBITDA < 2% of revenue (EV/EBITDA not meaningful)": lambda p: ~(p.ebitda_margin >= 0.02),
}
dropped = []
for why, rule in checks.items():
    mask = rule(panel)
    d = panel[mask].assign(drop_reason=why)
    dropped.append(d)
    panel = panel[~mask]
    log(f"  after dropping: {why}", panel, why.split(" ")[0])
# P/E below 3 is NOT dropped: it is real for crashed stocks (e.g. airlines on
# 2020-04-01). It is flagged and listed in data/review_low_pe.csv to be checked
# by hand; only rows confirmed as data errors should be fixed at the source.
panel["pe_low"] = panel.pe < 3
panel.loc[panel.pe_low, ["year", "ticker", "price", "mcap", "net_income", "pe"]].to_csv(
    DATA / "review_low_pe.csv", index=False)
pd.concat(dropped)[["year", "ticker", "drop_reason", "price", "mcap", "pe", "ebitda",
                    "revenue", "debt", "interest_expense"]].to_csv(DATA / "dropped_rows.csv", index=False)
panel["log_ev_ebitda"] = np.log(panel.ev / panel.ebitda)
panel["log_mcap"] = np.log(panel.mcap)

# clip extreme ratios at the 1st/99th percentile within each year
for c in WINSORIZE:
    if c in panel:
        lo = panel.groupby("year")[c].transform(lambda s: s.quantile(0.01))
        hi = panel.groupby("year")[c].transform(lambda s: s.quantile(0.99))
        panel[c] = panel[c].clip(lo, hi)

cols = (["year", "ticker", "cik", "sector", "sic", "period_end", "available_date",
         "price", "mcap", "ev", "ebitda", "revenue", "net_income", "pe", "debt", "cash",
         "debt_missing", "shares_fixed", "op_income_fixed", "hold_truncated", "pe_low"]
        + FEATURES + OPTIONAL + ["log_ev_ebitda", "fwd_ret_12m", "fwd_excess_ret_12m"])
for c in cols:
    if c not in panel:
        panel[c] = np.nan
panel = panel[cols].sort_values(["year", "ticker"])
panel.to_csv(DATA / "panel_full.csv", index=False)

complete = panel.dropna(subset=FEATURES + ["log_ev_ebitda"])
log("  with all features present -> panel.csv", complete, "final")
report.append(f"{'  ... of which have a forward 12-month return':<60}"
              f"{complete.fwd_excess_ret_12m.notna().sum():>8}")
report.append(f"{'  ... of which stopped trading during the holding year':<60}"
              f"{int(complete.hold_truncated.sum()):>8}")
complete.to_csv(DATA / "panel.csv", index=False)

# coverage per year: how many index members survive each step
cov = pd.DataFrame(stages).fillna(0).astype(int)
cov["kept_%"] = (100 * cov["final"] / cov["members"]).round(1)
report.append("\nCoverage per year (rows remaining after each step):")
report.append(cov.to_string())
report.append("\nFlags in panel.csv (share of rows):")
for c in ["shares_fixed", "op_income_fixed", "hold_truncated", "pe_low"]:
    report.append(f"  {c:<20}{complete[c].mean():.3f}")

report.append("\nMissing rate per feature (panel_full.csv):")
report.append(panel[FEATURES + OPTIONAL].isna().mean().round(3).to_string())
report.append("\nRows per year (panel.csv):")
report.append(complete.groupby("year").size().to_string())
report.append("\nRows per sector (panel.csv):")
report.append(complete.sector.value_counts().to_string())
(DATA / "data_report.txt").write_text("\n".join(report))
print("\n".join(report[report.index("\nCoverage per year (rows remaining after each step):"):]))
print("\nDone. data/panel.csv is ready for modeling.")

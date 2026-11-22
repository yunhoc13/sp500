"""Step 6: data for the current-year demo (e.g. April 1, 2026).

Builds what 05_build_panel.py --demo <year> needs, WITHOUT changing any file
used for the 2011-2025 panel:

  1. data/membership_<year>.csv    S&P 500 members on April 1 of <year>
  2. data/ticker_cik_<year>.csv    ticker -> CIK. Tickers already in
                                   ticker_cik.csv keep their mapping; only new
                                   tickers are looked up in the SEC list.
     data/raw/sec/facts|submissions/CIK*.json  downloaded for NEW companies only
  3. data/raw/prices_close_<year>_new.csv.gz, prices_adjclose_<year>_new.csv.gz,
     splits_<year>_new.csv         prices and full split history for symbols
                                   that are not in the existing price files

Then:
    python src/03_fundamentals.py            # re-run: adds the new companies
    python src/05_build_panel.py --demo 2026 # -> data/panel_2026.csv
    python src/16_demo_top20.py              # Top 20 + SHAP

    python src/06_demo_data.py 2026
"""
import json
import sys
import time

import pandas as pd
import requests
import yfinance as yf

from config import (DATA, RAW, RAW_SEC, USER_AGENT, SEC_TICKERS_URL, SEC_FACTS_URL,
                    SEC_SUBMISSIONS_URL, PRICE_START, PRICE_END, formation_date, yahoo_symbol)
from tags import ALL_TAGS

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
F = formation_date(YEAR)
if "your_email" in USER_AGENT:
    raise SystemExit("Edit USER_AGENT in src/config.py first (your name + email).")
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})


def get(url, tries=5):
    for i in range(tries):
        r = session.get(url, timeout=60)
        time.sleep(0.12)                      # SEC: at most 10 requests/second
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        time.sleep(2 ** i)
    r.raise_for_status()


# ---- 1. members on April 1 -------------------------------------------------
hist = pd.read_csv(RAW / "sp500_historical_components.csv", parse_dates=["date"]).sort_values("date")
snap = hist[hist.date <= F]
if snap.empty or hist.date.max() < F:
    raise SystemExit(f"Membership history ends {hist.date.max().date()}; delete "
                     "data/raw/sp500_historical_components.csv and re-run 01_membership.py")
last = snap.iloc[-1]
tickers = sorted(set(t.strip() for t in last.tickers.split(",") if t.strip()))
pd.DataFrame({"year": YEAR, "formation_date": F.date(), "ticker": tickers}).to_csv(
    DATA / f"membership_{YEAR}.csv", index=False)
print(f"1. {YEAR}: snapshot {last.date.date()}, {len(tickers)} members")

# ---- 2. ticker -> CIK and SEC data for new companies ------------------------
old_tc = pd.read_csv(DATA / "ticker_cik.csv")
known = old_tc[old_tc.ticker.isin(tickers)]
new_t = [t for t in tickers if t not in set(old_tc.ticker)]
sec = get(SEC_TICKERS_URL).json()
sec_map = {}
for row in sec.values():
    sec_map.setdefault(row["ticker"].upper(), (row["cik_str"], row["title"]))
recs = []
for t in new_t:
    hit = sec_map.get(t.upper().replace(".", "-")) or sec_map.get(t.upper())
    recs.append((t, hit[0] if hit else None, hit[1] if hit else None))
new_tc = pd.DataFrame(recs, columns=["ticker", "cik", "sec_title"])
tc = pd.concat([known, new_tc], ignore_index=True)
tc["cik"] = tc["cik"].astype("Int64")
tc.to_csv(DATA / f"ticker_cik_{YEAR}.csv", index=False)
print(f"2. new tickers: {len(new_t)} ({', '.join(new_t)}); "
      f"unmapped overall: {tc.cik.isna().sum()}")

n_dl = 0
for cik in sorted(tc.cik.dropna().astype(int).unique()):
    f_facts = RAW_SEC / "facts" / f"CIK{cik:010d}.json"
    f_sub = RAW_SEC / "submissions" / f"CIK{cik:010d}.json"
    if not f_facts.exists():
        r = get(SEC_FACTS_URL.format(cik=cik))
        facts = {} if r is None else r.json().get("facts", {})
        trimmed = {ns: {tag: facts[ns][tag]["units"]
                        for tag in ALL_TAGS if tag in facts.get(ns, {})}
                   for ns in ("us-gaap", "dei")}
        trimmed["_tags"] = ALL_TAGS
        f_facts.write_text(json.dumps(trimmed) if r is not None else "{}")
        n_dl += 1
    if not f_sub.exists():
        r = get(SEC_SUBMISSIONS_URL.format(cik=cik))
        meta = {} if r is None else {k: r.json().get(k) for k in
                                     ("cik", "name", "sic", "sicDescription", "tickers")}
        f_sub.write_text(json.dumps(meta))
print(f"   downloaded SEC data for {n_dl} new companies")

# ---- 3. prices for symbols not in the existing price files ------------------
have = set(pd.read_csv(RAW / "prices_adjclose.csv.gz", nrows=0).columns)
need = sorted({yahoo_symbol(t) for t in tickers} - have)
print(f"3. downloading prices for {len(need)} symbols ...")
close, adj, splits = pd.DataFrame(), pd.DataFrame(), []
if need:
    df = yf.download(need, start=PRICE_START, end=PRICE_END, auto_adjust=False,
                     progress=False, threads=False, group_by="column")
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_product([df.columns, need])
    close, adj = df["Close"], df["Adj Close"]
    for sym in need:                      # full split history, up to today
        s = yf.Ticker(sym).splits
        if len(s):
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            splits.append(pd.DataFrame({"date": s.index, "symbol": sym, "ratio": s.values}))
        time.sleep(0.3)
close.to_csv(RAW / f"prices_close_{YEAR}_new.csv.gz")
adj.to_csv(RAW / f"prices_adjclose_{YEAR}_new.csv.gz")
(pd.concat(splits) if splits else pd.DataFrame(columns=["date", "symbol", "ratio"])).to_csv(
    RAW / f"splits_{YEAR}_new.csv", index=False)
got = adj.notna().any() if len(adj) else pd.Series(dtype=bool)
print(f"   prices found for {int(got.sum())} / {len(need)}; missing: "
      f"{', '.join(got[~got].index) if len(got) else '-'}")
print(f"\nNext: python src/03_fundamentals.py && python src/05_build_panel.py --demo {YEAR}")

"""Step 1: who was in the S&P 500 on each formation date (April 1).

Input : fja05680/sp500 historical components CSV (downloaded automatically)
Output: data/membership.csv  (year, formation_date, ticker)
"""
import io
import sys

import pandas as pd
import requests

from config import DATA, RAW, YEARS, FJA_URL, formation_date

src = RAW / "sp500_historical_components.csv"
if not src.exists():
    print("Downloading fja05680/sp500 ...")
    r = requests.get(FJA_URL, timeout=60)
    r.raise_for_status()
    src.write_bytes(r.content)

hist = pd.read_csv(src, parse_dates=["date"]).sort_values("date")
print(f"History rows: {len(hist)}  ({hist.date.min().date()} to {hist.date.max().date()})")

rows = []
for y in YEARS:
    f = formation_date(y)
    snap = hist[hist.date <= f]
    if snap.empty:
        sys.exit(f"No membership data on or before {f.date()}")
    last = snap.iloc[-1]
    tickers = sorted(set(t.strip() for t in last.tickers.split(",") if t.strip()))
    rows += [(y, f.date(), t) for t in tickers]
    print(f"{y}: snapshot {last.date.date()}  members={len(tickers)}")

out = pd.DataFrame(rows, columns=["year", "formation_date", "ticker"])
out.to_csv(DATA / "membership.csv", index=False)
print(f"\nSaved {len(out)} rows, {out.ticker.nunique()} unique tickers -> data/membership.csv")

"""Step 4c: download each symbol's COMPLETE split history (up to today).

Why: Yahoo adjusts past prices for every split up to TODAY, but step 4 only
recorded splits inside the price window (to PRICE_END). A split after that
date (e.g. KLAC 10:1 in 2026) left our historical prices 10x too low.
This replaces data/raw/splits.csv with the full history for every symbol.

Run after step 4 / 4b. Takes ~5-10 minutes. Then re-run step 5.
"""
import time

import pandas as pd
import yfinance as yf

from config import DATA, RAW

cov = pd.read_csv(DATA / "price_coverage.csv")
cov = cov.rename(columns={cov.columns[0]: "symbol"})
symbols = cov.loc[cov.has_data, "symbol"].tolist()
old = pd.read_csv(RAW / "splits.csv")
print(f"Fetching full split history for {len(symbols)} symbols ...")

rows, failed = [], []
for n, sym in enumerate(symbols, 1):
    for attempt in range(3):
        try:
            s = yf.Ticker(sym).splits
            break
        except Exception:
            s = None
            time.sleep(2)
    if s is None:
        failed.append(sym)
    elif len(s):
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        rows.append(pd.DataFrame({"date": s.index, "symbol": sym, "ratio": s.values}))
    time.sleep(0.3)
    if n % 50 == 0:
        print(f"  {n}/{len(symbols)}")

new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "symbol", "ratio"])
# keep any split from the old file that the new download missed (failed symbols)
old["date"] = pd.to_datetime(old["date"])
keep_old = old[old.symbol.isin(failed)]
splits = pd.concat([new, keep_old]).drop_duplicates(["date", "symbol"]).sort_values(["symbol", "date"])
splits.to_csv(RAW / "splits.csv", index=False)

added = splits.merge(old, on=["date", "symbol"], how="left", indicator=True)
added = added[added["_merge"] == "left_only"]
print(f"\nSplits before: {len(old)}  after: {len(splits)}  newly found: {len(added)}")
if len(added):
    print(added[["date", "symbol", "ratio_x"]].rename(columns={"ratio_x": "ratio"}).to_string(index=False))
if failed:
    print("Failed (kept old split data):", ", ".join(failed))
print("Next: python src/05_build_panel.py")

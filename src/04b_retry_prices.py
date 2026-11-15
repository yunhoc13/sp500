"""Step 4b: retry only the symbols that came back empty in step 4.

Some failures are temporary (Yahoo glitches, 'database is locked' from
parallel downloads). This retries each missing symbol one at a time.
Symbols that still fail are genuinely unavailable (mostly delisted).
"""
import time

import pandas as pd
import yfinance as yf

from config import DATA, RAW, PRICE_START, PRICE_END

close = pd.read_csv(RAW / "prices_close.csv.gz", index_col=0, parse_dates=True)
adj = pd.read_csv(RAW / "prices_adjclose.csv.gz", index_col=0, parse_dates=True)
splits = pd.read_csv(RAW / "splits.csv", parse_dates=["date"])
cov = pd.read_csv(DATA / "price_coverage.csv")
cov = cov.rename(columns={cov.columns[0]: "symbol"})   # first column holds the ticker
missing = cov.loc[~cov.has_data, "symbol"].tolist()
print(f"Retrying {len(missing)} symbols one at a time ...")

recovered = []
for n, sym in enumerate(missing, 1):
    for attempt in range(2):
        try:
            df = yf.download(sym, start=PRICE_START, end=PRICE_END, auto_adjust=False,
                             actions=True, progress=False, threads=False)
        except Exception:
            df = pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty and df["Adj Close"].notna().any():
            close[sym] = df["Close"]
            adj[sym] = df["Adj Close"]
            if "Stock Splits" in df:
                s = df.loc[df["Stock Splits"] > 0, "Stock Splits"]
                splits = pd.concat([splits, pd.DataFrame(
                    {"date": s.index, "symbol": sym, "ratio": s.values})])
            recovered.append(sym)
            break
        time.sleep(1.5)
    time.sleep(0.5)
    if n % 25 == 0:
        print(f"  {n}/{len(missing)}  recovered so far: {len(recovered)}")

close.sort_index().to_csv(RAW / "prices_close.csv.gz")
adj.sort_index().to_csv(RAW / "prices_adjclose.csv.gz")
splits.drop_duplicates(["date", "symbol"]).to_csv(RAW / "splits.csv", index=False)
cov["has_data"] = cov["has_data"] | cov["symbol"].isin(recovered)
cov.to_csv(DATA / "price_coverage.csv", index=False)
print(f"\nRecovered {len(recovered)}: {', '.join(recovered) or '-'}")
print(f"Symbols with data now: {cov.has_data.sum()} / {len(cov)}")
print("Still missing (report as a limitation):",
      ", ".join(cov.loc[~cov.has_data, "symbol"]))

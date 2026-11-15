"""Step 4: download daily prices for every ticker that was ever a member, plus SPY.

Input : data/membership.csv
Output: data/raw/prices_close.csv.gz      split-adjusted close (for market cap)
        data/raw/prices_adjclose.csv.gz   split + dividend adjusted (for returns)
        data/raw/splits.csv               stock splits (to undo split adjustment)
        data/price_coverage.csv           which tickers came back empty

Takes ~5-15 minutes. Safe to re-run (it starts over).
"""
import time

import pandas as pd
import yfinance as yf

from config import DATA, RAW, PRICE_START, PRICE_END, BENCHMARK, yahoo_symbol

members = pd.read_csv(DATA / "membership.csv")
symbols = sorted({yahoo_symbol(t) for t in members.ticker} | {BENCHMARK})
print(f"Downloading {len(symbols)} symbols, {PRICE_START} to {PRICE_END} ...")

close, adj, splits = [], [], []
CHUNK = 100
for i in range(0, len(symbols), CHUNK):
    batch = symbols[i:i + CHUNK]
    df = yf.download(batch, start=PRICE_START, end=PRICE_END, auto_adjust=False,
                     actions=True, group_by="column", progress=False, threads=False)
    if df.empty:
        continue
    if not isinstance(df.columns, pd.MultiIndex):        # single-symbol batch
        df.columns = pd.MultiIndex.from_product([df.columns, batch])
    close.append(df["Close"])
    adj.append(df["Adj Close"])
    if "Stock Splits" in df.columns.get_level_values(0):
        s = df["Stock Splits"].stack()
        s = s[s > 0].rename("ratio").reset_index()
        s.columns = ["date", "symbol", "ratio"]
        splits.append(s)
    print(f"  {min(i + CHUNK, len(symbols))}/{len(symbols)}")
    time.sleep(1)

close = pd.concat(close, axis=1).sort_index()
adj = pd.concat(adj, axis=1).sort_index()
close = close.loc[:, ~close.columns.duplicated()]
adj = adj.loc[:, ~adj.columns.duplicated()]
close.to_csv(RAW / "prices_close.csv.gz")
adj.to_csv(RAW / "prices_adjclose.csv.gz")
sp = pd.concat(splits) if splits else pd.DataFrame(columns=["date", "symbol", "ratio"])
sp.to_csv(RAW / "splits.csv", index=False)

cov = adj.notna().sum().rename("n_days").reindex(symbols).fillna(0).astype(int)
cov.index.name = "symbol"
cov = cov.reset_index()
cov["has_data"] = cov.n_days > 0
cov.to_csv(DATA / "price_coverage.csv", index=False)
print(f"\nSymbols with data: {cov.has_data.sum()} / {len(cov)}"
      f"  (missing ones are mostly delisted companies)")
if BENCHMARK not in adj.columns or adj[BENCHMARK].isna().all():
    raise SystemExit(f"{BENCHMARK} failed to download; re-run this step.")
print("Next: python src/05_build_panel.py")

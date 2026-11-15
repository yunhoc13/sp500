"""Shared settings for the data pipeline.

EDIT USER_AGENT before running 02_sec_download.py: the SEC requires every
request to identify who is sending it (name + email).
"""
from pathlib import Path

# ---- EDIT THIS ----------------------------------------------------------
USER_AGENT = "Your Name your_email@bu.edu"   # each person sets their own
# -------------------------------------------------------------------------

# One formation date per year: April 1. By then most companies with a
# December fiscal year have filed their 10-K.
YEARS = list(range(2011, 2026))          # 2011 ... 2025 (15 formation dates)
FORMATION_MONTH, FORMATION_DAY = 4, 1

# Price history needed: 1 year before the first formation date (volatility,
# beta) through 1 year after the last one (forward returns).
PRICE_START = "2009-06-01"
PRICE_END = "2026-05-01"
BENCHMARK = "SPY"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_SEC = RAW / "sec"
for d in (DATA, RAW, RAW_SEC / "facts", RAW_SEC / "submissions"):
    d.mkdir(parents=True, exist_ok=True)

FJA_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
           "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def formation_date(year):
    import pandas as pd
    return pd.Timestamp(year=year, month=FORMATION_MONTH, day=FORMATION_DAY)


def yahoo_symbol(ticker):
    """fja05680 / SEC use BRK.B; Yahoo uses BRK-B."""
    return ticker.replace(".", "-")

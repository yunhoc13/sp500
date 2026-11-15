# S&P 500 Undervaluation Detection (DS340 Final Project)

**Team:** Yunho Choi, Yeongchan Jang · Boston University, DS 340 (Fall 2026)

We train machine learning models to learn the "fair" EV/EBITDA multiple of S&P 500
companies from their fundamentals, flag stocks that trade well below that fair value
as undervalued, and backtest whether those stocks outperformed the S&P 500 (SPY)
over the next 12 months.

## Status

| Step | Description | Status |
|---|---|---|
| 1 | Historical S&P 500 membership (April 1, 2011-2025) | ✅ `data/membership.csv` |
| 2 | SEC financial statements (10-K, point-in-time) | ⏳ in progress |
| 3 | Daily prices (yfinance) | ⏳ |
| 4 | Final company x year panel (`panel.csv`) | ⏳ |
| 5 | Models: Linear/Ridge/Lasso -> RF -> XGBoost -> MLP, SHAP | ⏳ |
| 6 | Decile backtest vs SPY and simple value baselines | ⏳ |

## Data so far: `data/membership.csv`

Which companies were in the S&P 500 on each formation date (April 1 of each year).

| Column | Meaning |
|---|---|
| `year` | Formation year (2011-2025) |
| `formation_date` | April 1 of that year |
| `ticker` | Ticker of a company in the index on that date |

- 7,534 rows, 792 unique tickers, about 497-506 tickers per year
  (some companies list two share classes, e.g. GOOG and GOOGL).
- **Source:** [fja05680/sp500](https://github.com/fja05680/sp500), historical
  components file (updated 2026-09-07). For each year we take the most recent
  snapshot on or before April 1.
- **Why historical membership:** using today's list would only include survivors
  and inflate backtest results (survivorship bias).
- **Known gaps:** 2011-2014 have 497 tickers (source gaps); some older companies
  appear under later tickers (e.g. Yahoo as `AABA`).

## Why April 1?

Most S&P 500 companies have a December fiscal year and must file their 10-K within
60 days. By April 1 nearly every company's latest annual report is public, so all
companies are compared on equally fresh, already-public information.

## Planned data sources

| Data | Source |
|---|---|
| Ticker -> SEC company ID (CIK) | SEC `company_tickers.json` |
| Annual financials + filing dates | SEC companyfacts API |
| Industry codes (SIC) | SEC submissions API |
| Daily prices, SPY benchmark | Yahoo Finance via yfinance |

Data is for academic use in this course project only.

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
| 2-5 | SEC financials + prices -> company x year panel | ✅ `data/panel.csv` (3,858 rows) |
| 6 | Models: Linear/Ridge/Lasso -> RF -> XGBoost -> MLP, out-of-fold scores, SHAP | ⏳ |
| 7 | Decile backtest vs SPY and simple value baselines | ⏳ |

## Main dataset: `data/panel.csv`

One row = one company on one formation date (April 1 of `year`), using only
information that was public on that date.

| Column | Meaning |
|---|---|
| `year, ticker, cik, sector, sic` | identifiers; sector is derived from the SEC SIC code |
| `period_end, available_date` | fiscal year used, and when that 10-K became public (< April 1) |
| `price, mcap, ev, ebitda, revenue, debt, cash` | inputs to the target. **Never use as features** (they reveal y) |
| `debt_missing` | no debt tag found, debt set to 0 (9.6% of rows); use for robustness checks |
| `shares_fixed` | share count replaced by net income / EPS because the filed value had the wrong scale (2.4%) |
| `op_income_fixed` | operating income taken as pre-tax income + interest (tag missing or mis-tagged) (18%) |
| `hold_truncated` | stopped trading during the holding year; return measured to the last price (0 rows, see below) |
| `log_mcap, rev_growth, eps_growth, op_margin, roe, debt_to_equity, volatility, beta` | **features X** |
| `gross_margin` | optional feature (missing for ~26% of firms, mostly services) |
| `log_ev_ebitda` | **target y** |
| `fwd_ret_12m, fwd_excess_ret_12m` | next-12-month return, and minus SPY. **Evaluation only, never a feature** |

Rows per year: 174 (2011) to 331 (2025). How rows were filtered at each step is in
`data/data_report.txt`:

| Step | Rows |
|---|---|
| S&P 500 member-years, April 1 2011-2025 | 7,534 |
| matched to an SEC company ID (CIK) | 6,226 |
| one share class per company | 6,189 |
| excluding financials (SIC 6000-6799) | 4,983 |
| 10-K filed before April 1 | 4,787 |
| price available on the formation date | 4,732 |
| positive EBITDA and EV | 4,537 |
| plausible market cap ($1B-$5T) | 4,532 |
| debt not negative, interest < 20% of debt | 4,466 |
| EBITDA at least 2% of revenue | 4,300 |
| all features present (**panel.csv**) | **3,858** |

Rows removed by the three quality checks are listed in `data/dropped_rows.csv` with the
reason. A per-year version of this table (share of index members kept: 35% in 2011,
66% in 2025) is in `data_report.txt`.

## Data quality fixes

Raw XBRL filings contain errors, so the pipeline checks and corrects them:

- **Point-in-time values:** each number is taken as first reported (not later restatements),
  and the fiscal year is used only if its 10-K was filed before April 1.
- **Share counts in the wrong scale** (e.g. 747 instead of 752 million): replaced by
  net income / EPS when the two disagree by more than 1.5x.
- **Mis-tagged operating income:** replaced by pre-tax income + interest when it is far off.
- **Stock splits:** share counts in a 10-K are already restated for splits before the filing
  date, so only splits after the filing are applied.
- **Check:** market cap / net income matches price / EPS within 0.88-1.05 for 98% of rows.

## Data sources

| Data | Source | Notes |
|---|---|---|
| S&P 500 membership | [fja05680/sp500](https://github.com/fja05680/sp500) | Compiled from *Trading Evolved* + index changes |
| Ticker -> CIK | SEC `company_tickers.json` | Current tickers only; renamed/acquired firms drop out |
| Annual financials (10-K) | SEC companyfacts API | Each number kept as **first reported** (point-in-time) |
| Industry (SIC) | SEC submissions API | Sector feature, financials exclusion |
| Daily prices, SPY | Yahoo Finance via yfinance | 199 of 793 symbols unavailable (mostly delisted) |

**Known limitations:** companies that were later acquired or delisted are
under-represented (no CIK match or no Yahoo prices), which can bias backtest
results upward (survivorship bias). Earlier years have fewer rows for this reason.
Yahoo removes the entire price history of delisted symbols, so no stock in the
panel stops trading during its holding year (`hold_truncated` = 0); delisted firms
are missing from the start rather than ending with a partial return. This means a
"-100% for delisted stocks" sensitivity analysis is not possible with this data.

## Why April 1?

Most S&P 500 companies have a December fiscal year and must file their 10-K within
60 days. By April 1 nearly every company's latest annual report is public, so all
companies are compared on equally fresh, already-public information.

## Rebuilding the data

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy requests yfinance
# set USER_AGENT in src/config.py to your name + email (SEC requirement)
python src/01_membership.py      # S&P 500 membership
python src/02_sec_download.py    # SEC financial facts + SIC codes (~5-10 min)
python src/03_fundamentals.py    # one row per company fiscal year
python src/04_prices.py          # daily prices (~10-15 min)
python src/04b_retry_prices.py   # retry symbols that failed
python src/05_build_panel.py     # -> data/panel.csv, data/data_report.txt
```

Data is for academic use in this course project only.

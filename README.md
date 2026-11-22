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
| 2-5 | SEC financials + prices -> company x year panel | ✅ `data/panel.csv` (3,488 rows) |
| 6 | Models: Linear/Ridge/Lasso, RF, XGBoost, PyTorch MLP (`data/scores.csv`), SHAP for RF | ✅ |
| 7 | Backtest vs SPY and simple value baselines (`src/10_backtest.py`) | ✅ results in `results/backtest_*.csv` |

## Main dataset: `data/panel.csv`

One row = one company on one formation date (April 1 of `year`), using only
information that was public on that date.

| Column | Meaning |
|---|---|
| `year, ticker, cik, sector, sic` | identifiers; sector is derived from the SEC SIC code |
| `period_end, available_date` | fiscal year used, and when that 10-K became public (< April 1) |
| `price, mcap, ev, ebitda, revenue, net_income, pe, debt, cash` | inputs to the target and checks. **Never use as features** (they reveal y) |
| `debt_missing` | always False in panel.csv: rows with no debt tag are excluded (they are in `panel_full.csv`) |
| `shares_fixed` | share count replaced by net income / EPS because the filed value had the wrong scale (2.4%) |
| `op_income_fixed` | operating income taken as pre-tax income + interest (tag missing or mis-tagged) (18%) |
| `hold_truncated` | stopped trading during the holding year; return measured to the last price (0 rows, see below) |
| `pe_low` | P/E below 3. Flagged, not dropped: all 13 cases were checked and are real (2020 crash, one-off gains). See `data/review_low_pe.csv` |
| `log_mcap, rev_growth, eps_growth, op_margin, roe, debt_to_equity, volatility, beta` | **features X** |
| `gross_margin` | optional feature (missing for ~26% of firms, mostly services) |
| `log_ev_ebitda` | **target y** |
| `fwd_ret_12m, fwd_excess_ret_12m` | next-12-month return, and minus SPY. **Evaluation only, never a feature** |

Rows per year: 153 (2011) to 300 (2025). How rows were filtered at each step is in
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
| debt data available (no missing debt tag) | 4,098 |
| plausible market cap ($1B-$5T) | 4,096 |
| debt not negative, interest < 20% of debt | 4,030 |
| EBITDA at least 2% of revenue | 3,877 |
| all features present (**panel.csv**) | **3,488** |

Rows removed by the three quality checks are listed in `data/dropped_rows.csv` with the
reason. A per-year version of this table is in `data_report.txt`.

## Data quality fixes

Raw XBRL filings contain errors, so the pipeline checks and corrects them:

- **Point-in-time values:** each number is taken as first reported (not later restatements),
  and the fiscal year is used only if its 10-K was filed before April 1.
- **Share counts in the wrong scale** (e.g. 747 instead of 752 million): replaced by
  net income / EPS when the two disagree by more than 1.5x.
- **Mis-tagged operating income:** replaced by pre-tax income + interest when it is far off.
- **Stock splits:** share counts in a 10-K are already restated for splits before the filing
  date, so only splits after the filing are applied. Yahoo adjusts past prices for every
  split up to today, so `04c_full_splits.py` downloads each symbol's complete split history
  (e.g. KLAC 10:1 and CTVA 6.665x in 2026 had made earlier prices 10x / 6.7x too low).
- **Missing debt:** companies whose debt tags we cannot read (e.g. Ford, AES) would get
  debt = 0 and look artificially cheap, and they cluster in the cheapest decile. They are
  excluded from panel.csv (kept in panel_full.csv for a robustness check).
- **Check:** market cap / net income matches price / EPS within 0.88-1.05 for 98% of rows.

## Backtest: `src/10_backtest.py`

Each April 1, stocks are ranked by `mispricing` within each model and scheme and split
into 5 (and 10) groups; group 1 is the cheapest. For each group we take the average
`fwd_excess_ret_12m` (12-month return minus SPY), averaging within a year first so every
year counts equally. Two baselines use no model: `cheap` (log EV/EBITDA minus the year
median) and `sector_cheap` (minus the same year's sector median).

Reported for the test period (2018-2025) and the development period (2011-2017):
cheapest-group return, spread (cheapest minus most expensive), share of positive years,
t-statistic of the spread, and the information coefficient (Spearman correlation of
-mispricing with the forward return) with its mean, std and t-statistic. Robustness: the
test-period spread with each year left out, the risk/growth profile and sector mix of the
cheapest group, and the stocks in the 2020 cheapest group.

```bash
python src/10_backtest.py   # -> results/backtest_*.csv, figures/backtest_groups.png
```

## Model selection rule (decided 2026-11-16)

The final model is the one with the highest **development-period (2011-2017) out-of-fold
R^2** for predicting log(EV/EBITDA), because predicting the fair multiple is the model's
task. Backtest metrics (spread, IC) are reported for **every** model and are not used to
choose. Reason: in 2011-2017 the IC is negative for every model and baseline, so picking
the "best" dev IC would select the least-bad model largely by chance. This rule was set
after the test-period results had been seen, and is recorded here so the choice is not
tuned to them. Current choice: Random Forest (dev R^2 0.174 vs XGBoost 0.142 and MLP 0.128;
re-checked 2026-11-22 after adding the MLP, whose test R^2 0.257 is slightly higher
but is not used for selection).

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
python src/04c_full_splits.py    # full split history (prices are adjusted up to today)
python src/05_build_panel.py     # -> data/panel.csv, data/data_report.txt
```

### Current-year demo (April 1, 2026)

Builds `data/panel_2026.csv` (301 companies, forward returns empty) without changing
any file used for 2011-2025: members on April 1, 2026, SEC data and prices for the
20 companies that joined the index after April 2025, then the same panel rules.

```bash
python src/06_demo_data.py 2026            # members, new companies' SEC data + prices
python src/03_fundamentals.py              # re-run: adds the new companies
python src/05_build_panel.py --demo 2026   # -> data/panel_2026.csv
python src/16_demo_top20.py                # final model (RF) Top 20 + SHAP -> results/top20_rf_2026.csv
```

Checked: `05_build_panel.py` without `--demo` still reproduces `panel.csv` exactly, and
`--demo 2025` reproduces the 2025 rows of `panel.csv`. The Top 20 is an illustration of
what the model looks at, not an investment recommendation (the backtest signal is weak).

## Use of AI tools

Following the course policy, we disclose AI-generated code. The following files were
written with help from Claude (Anthropic): the data pipeline (`config.py`, `tags.py`,
`01`-`06`), the backtests (`10_backtest.py`, `14_backtest_variants.py`), the smoothed-target experiment (`15_smoothed_target.py`) and the MLP (`mlp.py`, `11_mlp.py`). The team chose the
research design, checked every step against the raw data (e.g. share counts, stock
splits, P/E consistency) and fixed the problems found. Data sources are listed above.

Data is for academic use in this course project only.

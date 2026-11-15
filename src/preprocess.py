"""Shared preprocessing for every fair-multiple model (linear, trees, MLP).

Usage
-----
    from preprocess import load_panel, Preprocessor

    df = load_panel()                       # panel.csv + per-year feature winsorizing
    prep = Preprocessor().fit(df.loc[train_idx])
    X_tr = prep.transform_X(df.loc[train_idx])
    y_tr = prep.target(df.loc[train_idx])   # winsorized, year-demeaned log(EV/EBITDA)
    model.fit(X_tr, y_tr)
    pred = prep.predict_level(df, test_idx, model.predict(prep.transform_X(df.loc[test_idx])))

Design
------
1. Features are winsorized at the 1st/99th percentile *within each year*
   (done once in load_panel). This uses only features, never prices or
   returns, and every feature value was public on the formation date.
2. Sector is one-hot encoded with a fixed category list (first one dropped),
   so every train/test split produces the same columns.
3. StandardScaler is fit on the training rows only.
4. The target is winsorized per year using cutoffs from the training rows
   only, then the year's median (training rows) is subtracted. Models learn
   "how far above/below this year's typical multiple" a company should trade.
5. To turn a model output back into a log multiple for test row i in year t,
   we add the median of year t's log(EV/EBITDA) over all *other* companies
   (leave-one-out). Row i's own price never enters its own prediction. This
   level is (almost) the same for every company in a year, so it does not
   change any within-year ranking; it only puts predictions on the original
   scale for R^2 / RMSE and the mispricing column.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

FEATURES = ["log_mcap", "rev_growth", "eps_growth", "op_margin", "roe",
            "debt_to_equity", "volatility", "beta"]
SECTOR = "sector"
TARGET = "log_ev_ebitda"
GROUP = "cik"            # company identifier used for folds
YEAR = "year"
WINSOR_Q = 0.01
LEVEL = "year_level_loo"   # helper column added by load_panel, never a feature

DEV_YEARS = list(range(2011, 2018))    # model selection / tuning only
TEST_YEARS = list(range(2018, 2026))   # looked at once, at the end
ALL_YEARS = DEV_YEARS + TEST_YEARS

# Columns that must never be model inputs (prices, target ingredients,
# future returns, data-quality flags).
FORBIDDEN = ["fwd_ret_12m", "fwd_excess_ret_12m", "price", "mcap", "ev",
             "ebitda", "revenue", "debt", "cash", TARGET,
             "debt_missing", "shares_fixed", "op_income_fixed", "hold_truncated",
             LEVEL]


def winsorize_by_year(df, cols, q=WINSOR_Q):
    """Clip each column at its q / 1-q quantile within each year."""
    out = df.copy()
    for c in cols:
        g = out.groupby(YEAR)[c]
        lo, hi = g.transform(lambda s: s.quantile(q)), g.transform(lambda s: s.quantile(1 - q))
        out[c] = out[c].clip(lo, hi)
    return out


def load_panel(path=DATA / "panel.csv"):
    """Read panel.csv and winsorize the features within each year."""
    df = pd.read_csv(path)
    assert not set(FEATURES) & set(FORBIDDEN)
    df = df.dropna(subset=FEATURES + [TARGET, SECTOR]).reset_index(drop=True)
    df = winsorize_by_year(df, FEATURES)
    df[LEVEL] = loo_year_median(df)
    return df


def loo_year_median(df):
    """For each row: median log(EV/EBITDA) of the same year, excluding itself.

    Used only to put predictions back on the original scale (see note 5).
    Never a model input.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, rows in df.groupby(YEAR):
        y = rows[TARGET].to_numpy(float)
        for j, i in enumerate(rows.index):
            out[i] = np.median(np.delete(y, j))
    return out


def sector_levels(df):
    return sorted(df[SECTOR].unique())


class Preprocessor:
    """Fit on training rows; transform any rows the same way."""

    def __init__(self, sectors=None):
        self.sectors = sectors

    def fit(self, train):
        if self.sectors is None:
            self.sectors = sector_levels(train)
        self.scaler = StandardScaler().fit(train[FEATURES].to_numpy(float))
        g = train.groupby(YEAR)[TARGET]
        self.y_lo = g.quantile(WINSOR_Q).to_dict()
        self.y_hi = g.quantile(1 - WINSOR_Q).to_dict()
        y_w = self._winsor_y(train)
        self.y_level = y_w.groupby(train[YEAR]).median().to_dict()
        return self

    # ----- features -------------------------------------------------------
    @property
    def feature_names(self):
        return FEATURES + [f"sector_{s}" for s in self.sectors[1:]]

    def transform_X(self, rows):
        num = self.scaler.transform(rows[FEATURES].to_numpy(float))
        sec = rows[SECTOR].to_numpy()
        onehot = np.column_stack([(sec == s).astype(float) for s in self.sectors[1:]])
        return np.hstack([num, onehot])

    # ----- target ---------------------------------------------------------
    def _winsor_y(self, rows):
        lo = rows[YEAR].map(self.y_lo)
        hi = rows[YEAR].map(self.y_hi)
        return rows[TARGET].clip(lo, hi)

    def target(self, train):
        """Winsorized, year-demeaned target for training rows."""
        return (self._winsor_y(train) - train[YEAR].map(self.y_level)).to_numpy(float)

    # ----- back to the original scale -------------------------------------
    @staticmethod
    def predict_level(df, idx, model_output):
        """Model output (year-demeaned) -> predicted log(EV/EBITDA)."""
        return np.asarray(model_output, float) + df.loc[idx, LEVEL].to_numpy(float)

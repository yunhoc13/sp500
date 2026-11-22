"""Shared fold assignment and train/test splits for every model.

Folds are assigned per company (cik), K=5, seed 42, and saved to
data/folds.csv so that everyone uses exactly the same split.

    python src/splits.py          # (re)creates data/folds.csv

Schemes (scores for year t, fold k; "self" = companies in fold k):

    oof               train on years <= t, companies not in fold k   (main)
    walkforward       train on years <  t, companies not in fold k
    walkforward_self  train on years <  t, all companies (own past rows kept)
    oof_year          train on year == t, companies not in fold k

In every scheme except walkforward_self, a company's rows from ANY year are
excluded when it is being scored, so a model cannot simply recall its past
multiple. walkforward_self keeps them on purpose, as a variation.
walkforward schemes have no scores for the first year (no past data).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import DATA, GROUP, YEAR, ALL_YEARS

K = 5
SEED = 42
FOLDS_PATH = DATA / "folds.csv"
SCHEMES = ["oof", "walkforward", "walkforward_self", "oof_year"]


def make_folds(df, k=K, seed=SEED):
    ciks = np.sort(df[GROUP].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(ciks)
    return pd.DataFrame({GROUP: ciks, "fold": np.arange(len(ciks)) % k}).sort_values(GROUP)


def load_folds(df, allow_new=False):
    """Return a Series fold (index aligned with df).

    allow_new: companies missing from folds.csv (e.g. firms that joined the
    index in a new demo year) get a fold from a separate seeded shuffle, so the
    existing assignment never changes. Off by default.
    """
    folds = pd.read_csv(FOLDS_PATH)
    fold = df[GROUP].map(folds.set_index(GROUP)["fold"])
    if fold.isna().any():
        if not allow_new:
            raise ValueError(f"{fold.isna().sum()} rows have a cik missing from folds.csv")
        new = np.sort(df.loc[fold.isna(), GROUP].unique())
        np.random.default_rng(SEED + 1).shuffle(new)
        fold = fold.fillna(df[GROUP].map(dict(zip(new, np.arange(len(new)) % K))))
    return fold.astype(int)


def iter_splits(df, fold, scheme, years=ALL_YEARS, k=K):
    """Yield (year, fold_id, train_index, test_index)."""
    if scheme not in SCHEMES:
        raise ValueError(scheme)
    yr = df[YEAR]
    for t in years:
        for f in range(k):
            test = df.index[(yr == t) & (fold == f)]
            if scheme == "oof":
                train = df.index[(yr <= t) & (fold != f)]
            elif scheme == "walkforward":
                train = df.index[(yr < t) & (fold != f)]
            elif scheme == "walkforward_self":
                train = df.index[yr < t]
            else:  # oof_year
                train = df.index[(yr == t) & (fold != f)]
            if len(test) == 0 or len(train) == 0:
                continue
            yield t, f, train, test


if __name__ == "__main__":
    from preprocess import load_panel
    df = load_panel()
    folds = make_folds(df)
    folds.to_csv(FOLDS_PATH, index=False)
    print(f"wrote {FOLDS_PATH}: {len(folds)} companies")
    print(folds["fold"].value_counts().sort_index().to_string())
    print(pd.crosstab(df[YEAR], load_folds(df)).to_string())

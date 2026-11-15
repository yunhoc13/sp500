"""Run any sklearn-style model through the shared splits and collect scores.

    from scoring import score_model, fit_metrics, save_scores
    preds = score_model(df, fold, lambda: Ridge(alpha=10), "oof")
"""
import json

import numpy as np
import pandas as pd

from preprocess import (Preprocessor, sector_levels, TARGET, YEAR, GROUP,
                        RESULTS, DATA, WINSOR_Q, ALL_YEARS)
from splits import iter_splits

SCORES_PATH = DATA / "scores.csv"
SCORE_COLS = ["ticker", "year", "model", "scheme", "pred_log_ev_ebitda",
              "mispricing", "cik", "fold"]


def score_model(df, fold, make_model, scheme, years=ALL_YEARS):
    """Out-of-sample prediction for every (year, fold) test block.

    make_model: function returning a fresh, unfitted model with fit/predict.
    Returns one row per scored company-year.
    """
    sectors = sector_levels(df)
    out = []
    for t, f, tr, te in iter_splits(df, fold, scheme, years):
        prep = Preprocessor(sectors).fit(df.loc[tr])
        model = make_model()
        model.fit(prep.transform_X(df.loc[tr]), prep.target(df.loc[tr]))
        raw = model.predict(prep.transform_X(df.loc[te]))
        pred = prep.predict_level(df, te, raw)
        out.append(pd.DataFrame({
            "row": te, "ticker": df.loc[te, "ticker"].to_numpy(),
            "cik": df.loc[te, GROUP].to_numpy(), "year": t, "fold": f,
            "pred_log_ev_ebitda": pred,
            "mispricing": df.loc[te, TARGET].to_numpy() - pred,
        }))
    res = pd.concat(out, ignore_index=True)
    res["scheme"] = scheme
    return res


def eval_target(df):
    """Target winsorized within each year (evaluation only)."""
    g = df.groupby(YEAR)[TARGET]
    return df[TARGET].clip(g.transform(lambda s: s.quantile(WINSOR_Q)),
                           g.transform(lambda s: s.quantile(1 - WINSOR_Q)))


def fit_metrics(df, preds):
    """Within-year R^2 and RMSE of predicted vs actual (winsorized) log multiple.

    R^2 is measured against each year's own mean, so it shows how much of the
    cross-sectional spread in multiples the fundamentals explain.
    """
    y = eval_target(df)
    p = preds.assign(y=y.loc[preds["row"]].to_numpy())
    rows = []
    for (yr, sch), g in p.groupby([YEAR, "scheme"]):
        err = g["y"] - g["pred_log_ev_ebitda"]
        sst = ((g["y"] - g["y"].mean()) ** 2).sum()
        rows.append({"year": yr, "scheme": sch, "n": len(g),
                     "r2": 1 - (err ** 2).sum() / sst,
                     "rmse": np.sqrt((err ** 2).mean())})
    return pd.DataFrame(rows)


def save_scores(preds, model_name):
    """Add/replace this model's rows in data/scores.csv."""
    new = preds.assign(model=model_name)[SCORE_COLS]
    if SCORES_PATH.exists():
        old = pd.read_csv(SCORES_PATH)
        new = pd.concat([old[old["model"] != model_name], new], ignore_index=True)
    new.sort_values(["model", "scheme", "year", "ticker"]).to_csv(SCORES_PATH, index=False)


def save_params(model_name, params):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "best_params.json"
    allp = json.loads(path.read_text()) if path.exists() else {}
    allp[model_name] = params
    path.write_text(json.dumps(allp, indent=2))

"""Step 3: turn each company's XBRL facts into one row per fiscal year.

Input : data/raw/sec/facts/CIK*.json   (from step 2)
Output: data/fundamentals.csv
        one row = (cik, fiscal year end), with the date it became public.

Point-in-time rule: a fiscal year's numbers are reported again in the next
year's 10-K as comparatives. We keep the FIRST filing date for each number,
because that is when the market actually saw it.
"""
import json

import numpy as np
import pandas as pd

from config import DATA, RAW_SEC
from tags import CONCEPTS

ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT"}


def concept_series(facts, kind, unit, tags):
    """Return DataFrame(end, start, val, filed) for one concept, best tag per year."""
    frames = []
    for rank, tag in enumerate(tags):
        entries = facts.get("us-gaap", {}).get(tag, {}).get(unit, [])
        if not entries:
            continue
        df = pd.DataFrame(entries)
        if "form" not in df or "end" not in df:
            continue
        df = df[df["form"].isin(ANNUAL_FORMS)].copy()
        if df.empty:
            continue
        df["end"] = pd.to_datetime(df["end"])
        df["filed"] = pd.to_datetime(df["filed"])
        if kind == "duration":
            if "start" not in df:
                continue
            df["start"] = pd.to_datetime(df["start"])
            days = (df["end"] - df["start"]).dt.days
            df = df[(days >= 350) & (days <= 380)]   # full fiscal year only
        else:
            df = df[df.get("start", pd.Series(index=df.index, dtype=object)).isna()]
        df["rank"] = rank
        frames.append(df[["end", "val", "filed", "rank"]])
    if not frames:
        return pd.DataFrame(columns=["end", "val", "filed"])
    allf = pd.concat(frames)
    # for each year-end: the value as FIRST reported (earliest filing), using tag
    # priority only to break ties. Companies sometimes switch tags (e.g. ASC 606
    # revenue in 2018); picking by tag first would take the later restated value.
    allf = allf.sort_values(["end", "filed", "rank"])
    return allf.groupby("end", as_index=False).first()[["end", "val", "filed"]]


def company_rows(cik, facts):
    dur = {c: v for c, v in CONCEPTS.items() if v[0] == "duration"}
    inst = {c: v for c, v in CONCEPTS.items() if v[0] == "instant"}

    # fiscal years = year-ends where operating income or revenue exists
    parts = {}
    for c, (kind, unit, tags) in dur.items():
        s = concept_series(facts, kind, unit, tags)
        parts[c] = s.set_index("end")
    base_ends = sorted(set(parts["op_income"].index) | set(parts["revenue"].index)
                       | set(parts["pretax_income"].index))
    if not base_ends:
        return None
    out = pd.DataFrame({"period_end": pd.to_datetime(base_ends)})
    filed_cols = []
    for c, s in parts.items():
        out[c] = out["period_end"].map(s["val"]) if len(s) else np.nan
        out[f"{c}_filed"] = out["period_end"].map(s["filed"]) if len(s) else pd.NaT
        filed_cols.append(f"{c}_filed")

    # balance-sheet items: value on (or within 7 days of) the year-end
    for c, (kind, unit, tags) in inst.items():
        s = concept_series(facts, kind, unit, tags).sort_values("end")
        if s.empty:
            out[c], out[f"{c}_filed"] = np.nan, pd.NaT
        else:
            m = pd.merge_asof(out[["period_end"]].sort_values("period_end"),
                              s.rename(columns={"end": "period_end"}),
                              on="period_end", direction="nearest",
                              tolerance=pd.Timedelta("7D"))
            out[c] = m["val"].values
            out[f"{c}_filed"] = m["filed"].values
        filed_cols.append(f"{c}_filed")

    # date the numbers we use were all public = the latest of their first filings
    used = ["revenue_filed", "op_income_filed", "da_filed", "shares_filed",
            "equity_filed", "cash_filed"]
    out["available_date"] = out[used].max(axis=1)
    out["cik"] = cik
    return out.drop(columns=filed_cols)


rows = []
files = sorted((RAW_SEC / "facts").glob("CIK*.json"))
for i, f in enumerate(files, 1):
    facts = json.loads(f.read_text() or "{}")
    if not facts:
        continue
    df = company_rows(int(f.stem[3:]), facts)
    if df is not None:
        rows.append(df)
    if i % 100 == 0:
        print(f"  {i}/{len(files)} files parsed")

fund = pd.concat(rows, ignore_index=True)

# operating income fallback: pre-tax income + interest expense
fund["op_income_fallback"] = fund["op_income"].isna() & fund["pretax_income"].notna()
fund["op_income"] = fund["op_income"].fillna(
    fund["pretax_income"] + fund["interest_expense"].fillna(0))

# D&A fallback, gross profit fallback, total debt
da_parts = fund[["depreciation", "amortization"]].sum(axis=1, min_count=1)
fund["da"] = fund["da"].fillna(da_parts.where(fund["depreciation"].notna()))
fund["gross_profit"] = fund["gross_profit"].fillna(fund["revenue"] - fund["cogs"])
parts = fund[["debt_noncurrent", "debt_current"]]
split_sum = parts.sum(axis=1, min_count=1)
fund["debt"] = fund["debt_total"].fillna(split_sum)
fund["debt_missing"] = fund["debt"].isna() & fund["short_term_borrowings"].isna()
fund["debt"] = fund["debt"].fillna(0) + fund["short_term_borrowings"].fillna(0)

cols = ["cik", "period_end", "available_date", "revenue", "gross_profit",
        "op_income", "da", "net_income", "eps", "shares", "equity", "cash",
        "debt", "debt_missing", "op_income_fallback"]
fund = fund[cols].sort_values(["cik", "period_end"])
fund.to_csv(DATA / "fundamentals.csv", index=False)

print(f"\nSaved {len(fund)} company-years for {fund.cik.nunique()} companies -> data/fundamentals.csv")
print("Share of rows with each item:")
print(fund[cols[3:-2]].notna().mean().round(3).to_string())
print(f"rows with no debt tag (debt set to 0): {fund.debt_missing.mean():.3f}")
print(f"rows using pre-tax + interest for operating income: {fund.op_income_fallback.mean():.3f}")
print("Next: python src/04_prices.py")

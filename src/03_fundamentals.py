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

    # available_date = filing date of the original 10-K for that fiscal year
    # (earliest first-filing among the core income items). Any item that only
    # appeared in a LATER filing was not public then -> treated as missing.
    core = ["revenue_filed", "op_income_filed", "net_income_filed", "pretax_income_filed"]
    out["available_date"] = out[core].min(axis=1)
    for fc in filed_cols:
        late = out[fc] > out["available_date"] + pd.Timedelta(days=45)
        out.loc[late, fc[:-len("_filed")]] = np.nan
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

# ---- data-quality fixes ---------------------------------------------------
# (a) operating income: fall back to pre-tax income + interest (EBIT) when the
#     tag is missing, or when the tagged value is far from EBIT (mis-tagged line)
ebit_alt = fund["pretax_income"] + fund["interest_expense"].fillna(0)
fund["op_income_fallback"] = fund["op_income"].isna() & ebit_alt.notna()
far = (fund["op_income"].notna() & ebit_alt.notna() &
       ((fund["op_income"] - ebit_alt).abs() >
        0.5 * ebit_alt.abs() + 0.02 * fund["revenue"].abs()))
fund["op_income_fixed"] = far
fund.loc[far | fund["op_income_fallback"], "op_income"] = ebit_alt

# (b) shares: some filers tag share counts in the wrong scale (e.g. 747 instead
#     of 752 million). Check against net income / EPS and replace when off.
implied = fund["net_income"] / fund["eps"]
reliable = (fund["eps"].abs() >= 0.25) & (np.sign(fund["net_income"]) == np.sign(fund["eps"]))
implied = implied.where(reliable)
ratio = fund["shares"] / implied
bad = implied.notna() & (fund["shares"].isna() | (ratio < 0.67) | (ratio > 1.5))
fund["shares_fixed"] = bad
fund.loc[bad, "shares"] = implied

# D&A fallback, gross profit fallback, total debt
da_parts = fund[["depreciation", "amortization"]].sum(axis=1, min_count=1)
fund["da"] = fund["da"].fillna(da_parts.where(fund["depreciation"].notna()))
fund["gross_profit"] = fund["gross_profit"].fillna(fund["revenue"] - fund["cogs"])
debt_cols = ["debt_total", "debt_noncurrent", "debt_current", "short_term_borrowings"]
fund["debt_negative"] = (fund[debt_cols] < 0).any(axis=1)
parts = fund[["debt_noncurrent", "debt_current"]]
split_sum = parts.sum(axis=1, min_count=1)
fund["debt"] = fund["debt_total"].fillna(split_sum)
fund["debt_missing"] = fund["debt"].isna() & fund["short_term_borrowings"].isna()
fund["debt"] = fund["debt"].fillna(0) + fund["short_term_borrowings"].fillna(0)

cols = ["cik", "period_end", "available_date", "revenue", "gross_profit",
        "op_income", "da", "net_income", "eps", "shares", "equity", "cash",
        "debt", "interest_expense",
        "debt_missing", "debt_negative", "op_income_fallback", "op_income_fixed",
        "shares_fixed"]
fund = fund[cols].sort_values(["cik", "period_end"])
fund.to_csv(DATA / "fundamentals.csv", index=False)

print(f"\nSaved {len(fund)} company-years for {fund.cik.nunique()} companies -> data/fundamentals.csv")
print("Share of rows with each item:")
print(fund[cols[3:14]].notna().mean().round(3).to_string())
print("Data-quality flags (share of company-years):")
for c in cols[14:]:
    print(f"  {c:<20}{fund[c].mean():.3f}")
print("Next: python src/04_prices.py")

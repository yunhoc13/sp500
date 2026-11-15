"""Step 2: map tickers to SEC CIKs and download each company's XBRL facts.

Inputs : data/membership.csv
Outputs: data/ticker_cik.csv               (ticker -> cik, or unmapped)
         data/raw/sec/facts/CIK*.json      (only the XBRL tags we use, trimmed)
         data/raw/sec/submissions/CIK*.json (company name, SIC industry code)

Safe to re-run: files already downloaded are skipped.
SEC rules: a User-Agent with your name/email, and at most 10 requests/second.
"""
import json
import time

import pandas as pd
import requests

from config import (DATA, RAW_SEC, USER_AGENT, SEC_TICKERS_URL, SEC_FACTS_URL,
                    SEC_SUBMISSIONS_URL)
from tags import ALL_TAGS

if "your_email" in USER_AGENT:
    raise SystemExit("Edit USER_AGENT in src/config.py first (your name + email).")

HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
session = requests.Session()
session.headers.update(HEADERS)


def get(url, tries=5):
    for i in range(tries):
        r = session.get(url, timeout=60)
        time.sleep(0.12)                      # stay under 10 requests/second
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        wait = 2 ** i
        print(f"  HTTP {r.status_code} for {url} -> retry in {wait}s")
        time.sleep(wait)
    r.raise_for_status()


# ---- 2a. ticker -> CIK ----------------------------------------------------
members = pd.read_csv(DATA / "membership.csv")
tickers = sorted(members.ticker.unique())

sec = get(SEC_TICKERS_URL).json()
sec_map = {}
for row in sec.values():
    sec_map.setdefault(row["ticker"].upper(), (row["cik_str"], row["title"]))

recs = []
for t in tickers:
    key = t.upper().replace(".", "-")          # SEC uses BRK-B
    hit = sec_map.get(key) or sec_map.get(t.upper())
    recs.append((t, hit[0] if hit else None, hit[1] if hit else None))
tc = pd.DataFrame(recs, columns=["ticker", "cik", "sec_title"])
tc["cik"] = tc["cik"].astype("Int64")
tc.to_csv(DATA / "ticker_cik.csv", index=False)
print(f"Tickers: {len(tc)}  mapped: {tc.cik.notna().sum()}  unmapped: {tc.cik.isna().sum()}")
print("(Unmapped = renamed, acquired or delisted companies; reported as a limitation.)")

# ---- 2b. download facts + submissions per CIK -----------------------------
ciks = sorted(tc.cik.dropna().astype(int).unique())
for n, cik in enumerate(ciks, 1):
    f_facts = RAW_SEC / "facts" / f"CIK{cik:010d}.json"
    f_sub = RAW_SEC / "submissions" / f"CIK{cik:010d}.json"

    stale = False
    if f_facts.exists():                 # re-download if saved with an older tag list
        saved = json.loads(f_facts.read_text() or "{}")
        stale = bool(saved) and not set(ALL_TAGS) <= set(saved.get("_tags", []))
    if not f_facts.exists() or stale:
        r = get(SEC_FACTS_URL.format(cik=cik))
        if r is None:
            f_facts.write_text("{}")
        else:
            facts = r.json().get("facts", {})
            trimmed = {ns: {tag: facts[ns][tag]["units"]
                            for tag in ALL_TAGS if tag in facts.get(ns, {})}
                       for ns in ("us-gaap", "dei")}
            trimmed["_tags"] = ALL_TAGS
            f_facts.write_text(json.dumps(trimmed))

    if not f_sub.exists():
        r = get(SEC_SUBMISSIONS_URL.format(cik=cik))
        meta = {} if r is None else {k: r.json().get(k) for k in
                                     ("cik", "name", "sic", "sicDescription", "tickers")}
        f_sub.write_text(json.dumps(meta))

    if n % 50 == 0 or n == len(ciks):
        print(f"  {n}/{len(ciks)} companies done")

print("Done. Next: python src/03_fundamentals.py")

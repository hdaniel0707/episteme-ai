"""Scratch script: search manugram_reph_sci.parquet for any mention of ICLR.

Scans a handful of likely columns (venue, title, query, domain, prompt) for
the substring "iclr", case-insensitive, and prints the row id, which column
matched, and the matching value (truncated if long).

Run:
    uv run python _find_iclr.py
"""

import pandas as pd

PATH = "data/parquet/manugram_reph_sci.parquet"
SEARCH_COLS = ["venue", "title", "query", "domain", "prompt"]

df = pd.read_parquet(PATH)
print(f"Loaded {len(df):,} rows, columns: {list(df.columns)}\n")

found_any = False
for col in SEARCH_COLS:
    if col not in df.columns:
        print(f"(no {col!r} column)")
        continue
    mask = df[col].astype("string").str.contains("iclr", case=False, na=False)
    matches = df[mask]
    if matches.empty:
        continue
    found_any = True
    id_col = "id" if "id" in df.columns else None
    for _, row in matches.drop_duplicates(subset=id_col or col).iterrows():
        value = str(row[col])
        snippet = value if len(value) <= 200 else value[:200] + "..."
        rid = row[id_col] if id_col else "(no id column)"
        print(f"id={rid}  column={col}\n  {snippet}\n")

if not found_any:
    print("No 'iclr' matches found in any of:", SEARCH_COLS)

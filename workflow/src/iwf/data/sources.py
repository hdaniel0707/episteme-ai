"""Phase-0 corpus sources: the epai SQLite DB + OpenAlex gap-filling.

The epai store is the PRIMARY source of papers and is read-only. ``assemble`` loads it
into a dataframe (null-safe re-implementation of ``epai.utils.sqlite_utils.load_db`` —
that loader hard-requires an embedding JSON on every row, which the extended DB does not
guarantee), then optionally gap-fills OpenAlex impact fields (fwci,
citation_normalized_percentile, topic domain/subfield) by title+year match, cached and
resume-safe under workflow/data/cache/.

DB selection (first hit wins):
  * ``EPAI_DB`` in .env.workflow — absolute path, or a name under store/data/sql/,
    or the literal ``extended`` for the workflow-owned workflow/data/sql/extended.db.
  * newest ``*.db`` under store/data/sql/ otherwise.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import pandas.errors
import requests

from workflow.src.iwf import config, paths
from workflow.src.iwf.data.identity import PAPER_ID, ensure_paper_id, normalize_doi

OPENALEX_WORKS = "https://api.openalex.org/works"

# columns the OpenAlex enrichment can add (NaN when unmatched)
OPENALEX_COLS = ["fwci", "citation_normalized_percentile", "is_in_top_1_percent",
                 "is_in_top_10_percent", "doi", "oa_id", "oa_domain", "subfield"]


# --------------------------------------------------------------------- sqlite loading

def resolve_db_path() -> Path:
    """The SQLite DB to read (see module docstring for the selection order)."""
    name = config.get_str("EPAI_DB", "")
    if name:
        name_lower = name.strip().lower()
        if name_lower == "extended":
            return paths.SQL_DIR / "extended.db"
        if name_lower == "training":
            return paths.SQL_DIR / "training.db"
        p = Path(name)
        if not p.is_absolute():
            p = paths.EPAI_SQL_DIR / name
        return p
    candidates = sorted(paths.EPAI_SQL_DIR.glob("*.db"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .db file found under {paths.EPAI_SQL_DIR}.")
    return candidates[0]


def _json_or(value, default):
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def load_sqlite(path: str | Path, year_from: int | None = None,
                year_to: int | None = None) -> pd.DataFrame:
    """papers + author_count + best_author_h_index from one epai-schema SQLite file.

    Fully defensive: gracefully handles missing columns, NULL data, and missing tables.
    Skips rows with NULL title/abstract (unusable for training). Never crashes on
    bad schema or missing fields.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SQLite DB not found: {path}")
    conditions, params = [], []
    if year_from is not None:
        conditions.append("year >= ?")
        params.append(int(year_from))
    if year_to is not None:
        conditions.append("year <= ?")
        params.append(int(year_to))
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    conn = sqlite3.connect(path)
    try:
        papers = pd.read_sql(f"SELECT * FROM papers {where}", conn, params=params or None)
        # Try to load author info; gracefully skip if tables don't exist
        author_counts = None
        best_h = None
        try:
            author_counts = pd.read_sql(
                "SELECT paper_id, COUNT(*) AS author_count FROM paper_authors GROUP BY paper_id",
                conn)
        except (sqlite3.OperationalError, pandas.errors.DatabaseError):
            pass  # paper_authors table doesn't exist
        try:
            best_h = pd.read_sql(
                """
                SELECT pa.paper_id, MAX(a.h_index) AS best_author_h_index
                FROM paper_authors pa JOIN authors a ON pa.author_id = a.author_id
                WHERE a.h_index IS NOT NULL GROUP BY pa.paper_id
                """, conn)
        except (sqlite3.OperationalError, pandas.errors.DatabaseError):
            pass  # author tables don't exist
    finally:
        conn.close()

    df = papers
    if author_counts is not None:
        df = df.merge(author_counts, on="paper_id", how="left")
    else:
        df["author_count"] = None
    if best_h is not None:
        df = df.merge(best_h, on="paper_id", how="left")
    else:
        df["best_author_h_index"] = None
    df["source_db"] = path.stem

    # Safe JSON parsing: NULL/empty values become defaults without crashing
    for col, default in (("embedding", None), ("queries", []), ("domains", [])):
        if col in df.columns:
            df[col] = df[col].map(lambda v, d=default: _json_or(v, d))

    # Skip rows missing critical fields: title and abstract are required for training
    before = len(df)
    df = df[
        (df.get("title", pd.Series()).notna()) &
        (df.get("title", pd.Series()).astype(str).str.strip() != "") &
        (df.get("abstract", pd.Series()).notna()) &
        (df.get("abstract", pd.Series()).astype(str).str.strip() != "")
    ].reset_index(drop=True)
    after = len(df)
    if before > after:
        print(f"[sources] dropped {before - after} rows missing title/abstract")

    return df


# --------------------------------------------------------------------- OpenAlex enrich

class _OpenAlexQuotaExhausted(Exception):
    """Raised when OpenAlex reports a long retry-after (daily quota, not a burst
    limit) — retrying per-row would burn hours for zero benefit."""


# a 429 with retry-after beyond this is the daily quota, not a transient burst limit
_QUOTA_RETRY_AFTER_S = 60


def _oa_get(params: dict) -> dict | None:
    mailto = config.get_str("OPENALEX_MAILTO", "")
    if mailto:
        params = {**params, "mailto": mailto}
    for attempt in range(3):
        try:
            r = requests.get(OPENALEX_WORKS, params=params, timeout=30)
            if r.status_code == 429:
                retry_after = int(r.headers.get("retry-after", 0) or 0)
                if retry_after > _QUOTA_RETRY_AFTER_S:
                    raise _OpenAlexQuotaExhausted(
                        f"OpenAlex quota exhausted (retry-after={retry_after}s)")
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.ok:
                return r.json()
            return None
        except requests.RequestException:
            time.sleep(1.0)
    return None


def openalex_fields(work: dict) -> dict:
    """Flatten the impact-relevant fields of one OpenAlex work object."""
    cnp = work.get("citation_normalized_percentile") or {}
    topic = work.get("primary_topic") or {}
    return {
        "fwci": work.get("fwci"),
        "citation_normalized_percentile": cnp.get("value"),
        "is_in_top_1_percent": cnp.get("is_in_top_1_percent"),
        "is_in_top_10_percent": cnp.get("is_in_top_10_percent"),
        "doi": normalize_doi(work.get("doi")),
        "oa_id": work.get("id"),
        "oa_domain": (topic.get("domain") or {}).get("display_name"),
        "subfield": (topic.get("subfield") or {}).get("display_name"),
    }


def _match_one(title: str, year) -> dict | None:
    """Best-effort OpenAlex lookup of one paper by title (+year when known)."""
    # commas and colons are OpenAlex filter syntax — strip them from the title term
    title = re.sub(r"[,:]", " ", (title or "")).strip()
    if not title:
        return None
    filt = f"title.search:{title[:300]}"
    try:
        y = int(year)
        if y > 0:
            filt += f",publication_year:{y}"
    except (TypeError, ValueError):
        pass
    data = _oa_get({"filter": filt, "per-page": 1,
                    "select": "id,doi,title,fwci,citation_normalized_percentile,primary_topic"})
    results = (data or {}).get("results") or []
    return results[0] if results else None


def enrich_with_openalex(df: pd.DataFrame) -> pd.DataFrame:
    """Add OPENALEX_COLS by title+year match; cached per paper_id (resume-safe)."""
    from tqdm import tqdm

    cache_path = paths.CACHE_DIR / "openalex_enrich.json"
    cache: dict[str, dict | None] = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}

    rows, dirty = [], 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="OpenAlex enrich"):
        pid = str(row[PAPER_ID])
        if pid in cache:
            rows.append(cache[pid] or {})
            continue
        try:
            work = _match_one(str(row.get("title") or ""), row.get("year"))
        except _OpenAlexQuotaExhausted as e:
            # quota is dead for hours — retrying the remaining rows would just
            # replay this for every one of them; fall back to unenriched instead
            print(f"[sources] {e}; leaving {len(df) - len(rows)} rows unenriched")
            rows.extend({} for _ in range(len(df) - len(rows)))
            break
        fields = openalex_fields(work) if work else None
        cache[pid] = fields
        rows.append(fields or {})
        dirty += 1
        if dirty % 50 == 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        time.sleep(0.12)  # ~8 req/s, inside the OpenAlex polite-pool limit
    if dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    extra = pd.DataFrame(rows, index=df.index)
    for col in OPENALEX_COLS:
        if col not in extra.columns:
            extra[col] = None
        # never clobber a value the DB already carries (extended DB rows have them)
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), extra[col])
        else:
            df[col] = extra[col]
    n = df["fwci"].notna().sum() if "fwci" in df.columns else 0
    print(f"[sources] OpenAlex enrichment: fwci on {n}/{len(df)} rows")
    return df


# --------------------------------------------------------------------- public entry

def assemble(year_from: int | None = None, year_to: int | None = None,
             max_papers: int | None = None, enrich_openalex: bool = True,
             db_path: str | Path | None = None) -> pd.DataFrame:
    """Load the corpus dataframe for phase 0 (see module docstring).

    Fully defensive: skips unusable rows (missing title/abstract), filters out
    non-human papers, and gracefully handles enrichment failures. Never crashes
    on bad data.
    """
    path = Path(db_path) if db_path else resolve_db_path()
    df = load_sqlite(path, year_from=year_from, year_to=year_to)
    print(f"[sources] {len(df)} papers from {path.name}")

    # only human-written papers may enter the impact-training corpus
    if "generated_by" in df.columns:
        before = len(df)
        df = df[df["generated_by"].fillna("human") == "human"].reset_index(drop=True)
        if len(df) != before:
            print(f"[sources] dropped {before - len(df)} non-human rows")

    # ensure paper_id exists; skip any row that fails
    try:
        df = ensure_paper_id(df)
    except Exception as e:
        print(f"[sources] ensure_paper_id failed ({e}); dropping problematic rows")
        # retry with filter: keep only rows that don't cause errors
        usable = []
        for idx, row in df.iterrows():
            try:
                test_id = row.get(PAPER_ID, "")
                if not str(test_id).strip():
                    # try to generate one
                    from workflow.src.iwf.data.identity import stable_text_id
                    _ = stable_text_id(str(row.get("title", "")),
                                      str(row.get("abstract", "")),
                                      salt=str(row.get("year", "")))
                usable.append(idx)
            except Exception:
                continue
        df = df.loc[usable].reset_index(drop=True)
        print(f"[sources] kept {len(df)} rows with valid paper_ids")

    if max_papers is not None and len(df) > int(max_papers):
        df = df.head(int(max_papers)).reset_index(drop=True)
        print(f"[sources] capped corpus to max_papers={max_papers}")

    if enrich_openalex:
        try:
            df = enrich_with_openalex(df)
        except Exception as e:  # enrichment is best-effort; targets have fallbacks
            print(f"[sources] OpenAlex enrichment skipped ({e})")

    # Final sanity check: drop any remaining rows with NULL title/abstract
    before = len(df)
    df = df[
        (df.get("title", pd.Series()).notna()) &
        (df.get("title", pd.Series()).astype(str).str.strip() != "") &
        (df.get("abstract", pd.Series()).notna()) &
        (df.get("abstract", pd.Series()).astype(str).str.strip() != "")
    ].reset_index(drop=True)
    if len(df) < before:
        print(f"[sources] final pass: dropped {before - len(df)} rows with NULL/empty text")

    return df

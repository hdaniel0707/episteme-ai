"""Paper identity: the ONE join key + id/DOI normalization helpers.

Every artifact of the pipeline is keyed by ``PAPER_ID`` so re-runs can merge cached
columns (embeddings, external scores) onto a fresh dataframe by KEY, never by row order.
"""

from __future__ import annotations

import hashlib

import pandas as pd

PAPER_ID = "paper_id"

_DOI_PREFIXES = (
    "https://doi.org/", "http://doi.org/",
    "https://dx.doi.org/", "http://dx.doi.org/",
    "doi.org/", "doi:",
)


def normalize_doi(doi: str | None) -> str | None:
    """Lower-case bare DOI ('10.x/...') from any of the common URL/prefix forms."""
    if doi is None:
        return None
    s = str(doi).strip()
    if not s:
        return None
    low = s.lower()
    for pre in _DOI_PREFIXES:
        if low.startswith(pre):
            s = s[len(pre):]
            low = s.lower()
    return low or None


def stable_text_id(title: str | None, abstract: str | None, salt: str = "") -> str:
    """Deterministic id from the paper's text (used when no upstream id exists)."""
    payload = f"{(title or '').strip()}\x1f{(abstract or '').strip()}\x1f{salt}"
    return "T" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def ensure_paper_id(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a non-null, unique ``paper_id`` column (copy; input untouched).

    Rows that already carry an id keep it; missing ids are derived from the text hash,
    and residual collisions get a positional salt so the column is strictly unique.
    """
    df = df.copy()
    if PAPER_ID not in df.columns:
        df[PAPER_ID] = None
    ids = df[PAPER_ID].astype("object").where(df[PAPER_ID].notna(), None)

    seen: set[str] = set()
    out: list[str] = []
    for pos, (pid, title, abstract) in enumerate(
        zip(ids, df.get("title", pd.Series([None] * len(df), index=df.index)),
            df.get("abstract", pd.Series([None] * len(df), index=df.index)))
    ):
        cand = str(pid) if pid not in (None, "") else stable_text_id(title, abstract)
        if cand in seen:  # duplicate text or duplicate upstream id -> positional salt
            cand = stable_text_id(title, abstract, salt=str(pos))
        seen.add(cand)
        out.append(cand)
    df[PAPER_ID] = out
    return df


def keyed_merge(left: pd.DataFrame, right: pd.DataFrame,
                cols: list[str]) -> pd.DataFrame:
    """Attach ``cols`` from ``right`` onto ``left`` joined by ``PAPER_ID`` (row order of
    ``left`` preserved; a column already on ``left`` is replaced by the joined values)."""
    if PAPER_ID not in left.columns or PAPER_ID not in right.columns:
        raise KeyError(f"keyed_merge needs {PAPER_ID!r} on both frames.")
    take = [c for c in cols if c in right.columns]
    slim = right[[PAPER_ID, *take]].drop_duplicates(subset=[PAPER_ID], keep="last")
    out = left.drop(columns=[c for c in take if c in left.columns])
    return out.merge(slim, on=PAPER_ID, how="left")

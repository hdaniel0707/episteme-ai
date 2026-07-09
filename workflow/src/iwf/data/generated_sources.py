"""LLM-generated paper sources (MIRAI / ADP) for main_predict_generated.

Both loaders return a dataframe with at least: paper_id, title, abstract, source.
Generated papers have no citations, so no target columns are expected here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from workflow.src.iwf import config
from workflow.src.iwf.data.identity import ensure_paper_id

REQUIRED = ("title", "abstract")

# accepted spellings for the two text columns in ad-hoc exports
_ALIASES = {
    "title": ("title", "paper_title", "name"),
    "abstract": ("abstract", "paper_abstract", "summary", "abstract_text"),
}


def _normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    for want, alts in _ALIASES.items():
        if want in df.columns:
            continue
        for alt in alts:
            if alt in cols:
                df = df.rename(columns={cols[alt]: want})
                break
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"Generated-paper source {source!r} is missing columns {missing}; "
                       f"have {list(df.columns)}.")
    df = df[df["title"].notna() | df["abstract"].notna()].reset_index(drop=True)
    if "source" not in df.columns:
        df["source"] = source
    return ensure_paper_id(df)


def load_from_file(path: str | Path, source: str = "mirai") -> pd.DataFrame:
    """Load a local export (.parquet / .csv / .jsonl|.json) of generated papers."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GENERATED_FILE not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(data if isinstance(data, list) else data.get("papers", []))
    else:
        raise ValueError(f"Unsupported generated-paper file type {suffix!r} "
                         "(use .parquet/.csv/.jsonl/.json).")
    print(f"[generated] {len(df)} rows from {path.name}")
    return _normalize(df, source)


def load_from_api(source: str = "mirai") -> pd.DataFrame:
    """Fetch generated papers from the configured API; empty frame when not configured.

    Expects ``<SOURCE>_API_URL`` (+ optional ``<SOURCE>_API_KEY`` bearer token) in the
    repo-root .env; the endpoint must return a JSON list (or {'papers': [...]}) of
    objects with title/abstract fields.
    """
    prefix = (source or "mirai").upper()
    url = config.secret(f"{prefix}_API_URL")
    if not url:
        print(f"[generated] {prefix}_API_URL not configured - skipping the API source.")
        return pd.DataFrame(columns=[*REQUIRED, "source"])
    headers = {}
    key = config.secret(f"{prefix}_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[generated] API fetch failed ({e}) - skipping.")
        return pd.DataFrame(columns=[*REQUIRED, "source"])
    df = pd.DataFrame(data if isinstance(data, list) else data.get("papers", []))
    print(f"[generated] {len(df)} rows from the {source} API")
    return _normalize(df, source)

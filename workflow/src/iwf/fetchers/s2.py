"""Semantic Scholar gap-filler (plan §4.4) — thin, workflow-owned (epai is not imported).

Fills ``influential_citation_count`` and the SPECTER ``embedding`` for extended-DB rows
that came from OpenAlex, using the S2 batch endpoint keyed by DOI (100 papers/request).
API key from SEMANTIC_SCHOLAR_API_KEY (or the legacy alias S2_API_KEY) in the repo-root
.env (optional; the public tier also works, just slower and on a shared anonymous pool).
"""

from __future__ import annotations

import json
import sqlite3
import time

import requests

from workflow.src.iwf import config

BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "externalIds,influentialCitationCount,embedding"
BATCH = 100


def _headers() -> dict:
    # .env ships the key as SEMANTIC_SCHOLAR_API_KEY; S2_API_KEY kept as a legacy alias.
    key = config.api_key("SEMANTIC_SCHOLAR_API_KEY", "S2_API_KEY")
    return {"x-api-key": key} if key else {}


def fill_from_s2(conn: sqlite3.Connection, max_papers: int | None = None) -> int:
    """Gap-fill influential_citation_count + embedding by DOI; returns rows updated."""
    rows = conn.execute(
        """
        SELECT paper_id, doi FROM papers
        WHERE doi IS NOT NULL
          AND (influential_citation_count IS NULL OR embedding IS NULL)
          AND source IN ('openalex', 'uefiscdi_journal')
        """).fetchall()
    if max_papers:
        rows = rows[:max_papers]
    print(f"[s2] gap-filling {len(rows)} papers by DOI")
    updated = 0
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        ids = [f"DOI:{doi}" for _, doi in chunk]
        by_doi = {doi.lower(): pid for pid, doi in chunk}
        try:
            r = requests.post(BATCH_URL, params={"fields": FIELDS},
                              json={"ids": ids}, headers=_headers(), timeout=60)
            if r.status_code == 429:
                time.sleep(5)
                continue
            if not r.ok:
                print(f"[s2] HTTP {r.status_code}; stopping gap-fill")
                break
            papers = r.json()
        except (requests.RequestException, ValueError) as e:
            print(f"[s2] batch failed ({e}); continuing")
            continue
        for p in papers or []:
            if not p:
                continue
            doi = ((p.get("externalIds") or {}).get("DOI") or "").lower()
            pid = by_doi.get(doi)
            if not pid:
                continue
            emb = p.get("embedding")
            conn.execute(
                """
                UPDATE papers SET
                    influential_citation_count =
                        COALESCE(influential_citation_count, ?),
                    embedding = COALESCE(embedding, ?)
                WHERE paper_id = ?
                """,
                (p.get("influentialCitationCount"),
                 json.dumps(emb["vector"]) if emb and emb.get("vector") else None,
                 pid))
            updated += 1
        conn.commit()
        time.sleep(1.1 if not _headers() else 0.15)
    print(f"[s2] updated {updated} rows")
    return updated

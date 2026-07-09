"""Enrich incomplete paper metadata by searching OpenAlex and S2 by title.

Finds papers with missing critical fields (DOI, FWCI, citations, embeddings)
and enriches them via API lookups.
"""

import re
import sqlite3
import time
from typing import Optional

import requests

from workflow.src.iwf import paths, config
from workflow.src.iwf.data.identity import normalize_doi


OPENALEX_WORKS = "https://api.openalex.org/works"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"


def _mailto() -> dict:
    """Get mailto param for OpenAlex API (for polite pool)."""
    m = config.get_str("OPENALEX_MAILTO", "")
    return {"mailto": m} if m else {}


def _normalize_title(title: str) -> str:
    """Normalize title for comparison: remove punctuation, lowercase, collapse spaces."""
    if not title:
        return ""
    t = str(title).lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)  # Remove punctuation
    t = re.sub(r"\s+", " ", t)  # Collapse spaces
    return t


def search_openalex_by_title(title: str, year: Optional[int] = None) -> Optional[dict]:
    """Search OpenAlex by title, return best match with metadata.

    Returns: {doi, fwci, citation_count, citation_normalized_percentile, ...}
    """
    if not title or len(str(title).strip()) < 3:
        return None

    # OpenAlex filter syntax: commas and colons must be removed
    title_clean = re.sub(r"[,:]", " ", str(title)[:300]).strip()
    filt = f"title.search:{title_clean}"

    if year:
        try:
            y = int(year)
            if y > 1900 and y < 2100:
                filt += f",publication_year:{y}"
        except (ValueError, TypeError):
            pass

    try:
        for attempt in range(3):
            try:
                r = requests.get(
                    OPENALEX_WORKS,
                    params={**_mailto(), "filter": filt, "per-page": 1,
                            "select": "id,doi,title,fwci,citation_normalized_percentile,"
                                     "cited_by_count,publication_year"},
                    timeout=10
                )
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if r.ok:
                    results = r.json().get("results", [])
                    if results:
                        work = results[0]
                        # Verify title similarity (simple check)
                        oa_title = work.get("title", "")
                        if _normalize_title(oa_title) in _normalize_title(title) or \
                           _normalize_title(title) in _normalize_title(oa_title):
                            return {
                                "doi": normalize_doi(work.get("doi")),
                                "fwci": work.get("fwci"),
                                "citation_count": work.get("cited_by_count"),
                                "citation_normalized_percentile": work.get("citation_normalized_percentile"),
                                "oa_id": work.get("id"),
                            }
                    return None
                return None
            except requests.RequestException:
                time.sleep(1)
        return None
    except Exception as e:
        return None


def search_s2_by_title(title: str, year: Optional[int] = None) -> Optional[dict]:
    """Search S2 by title, return best match with embeddings and citations.

    Returns: {s2_id, embedding_vector, influentialCitationCount, ...}
    """
    if not title or len(str(title).strip()) < 3:
        return None

    try:
        for attempt in range(3):
            try:
                # Note: S2 API may require auth token; this is public endpoint
                params = {"query": str(title)[:300], "limit": 1}
                if year:
                    try:
                        y = int(year)
                        if y > 1900 and y < 2100:
                            params["year"] = y
                    except (ValueError, TypeError):
                        pass

                r = requests.get(
                    S2_API,
                    params=params,
                    timeout=10
                )
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if r.ok:
                    data = r.json()
                    results = data.get("data", [])
                    if results:
                        paper = results[0]
                        # Verify title similarity
                        s2_title = paper.get("title", "")
                        if _normalize_title(s2_title) in _normalize_title(title) or \
                           _normalize_title(title) in _normalize_title(s2_title):
                            embedding = paper.get("embedding")
                            return {
                                "s2_id": paper.get("paperId"),
                                "embedding": embedding,
                                "influential_citation_count": paper.get("influentialCitationCount"),
                                "venue": paper.get("venue"),
                            }
                    return None
                return None
            except requests.RequestException:
                time.sleep(1)
        return None
    except Exception as e:
        return None


def enrich_incomplete_papers(conn: sqlite3.Connection) -> int:
    """Find papers with missing metadata and enrich via API searches.

    Searches for papers missing: DOI, FWCI, citations, or embeddings.
    """
    # Find papers needing enrichment
    papers = conn.execute("""
    SELECT paper_id, title, year
    FROM papers
    WHERE (doi IS NULL OR fwci IS NULL OR citation_count IS NULL)
    AND title IS NOT NULL AND title != ''
    ORDER BY RANDOM()
    LIMIT 5000
    """).fetchall()

    if not papers:
        print("[enrich] No papers needing enrichment")
        return 0

    print(f"[enrich] Found {len(papers)} papers to enrich")

    enriched = 0
    for idx, (paper_id, title, year) in enumerate(papers):
        try:
            # Search OpenAlex first (better for citation metrics)
            oa_data = search_openalex_by_title(title, year)
            if oa_data:
                updates = {}
                if oa_data.get("doi") and not conn.execute(
                    "SELECT doi FROM papers WHERE paper_id = ? AND doi IS NOT NULL",
                    (paper_id,)
                ).fetchone():
                    updates["doi"] = oa_data["doi"]
                if oa_data.get("fwci"):
                    updates["fwci"] = oa_data["fwci"]
                if oa_data.get("citation_count"):
                    updates["citation_count"] = oa_data["citation_count"]
                if oa_data.get("citation_normalized_percentile"):
                    updates["citation_normalized_percentile"] = oa_data["citation_normalized_percentile"]

                if updates:
                    set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
                    conn.execute(
                        f"UPDATE papers SET {set_clause} WHERE paper_id = ?",
                        list(updates.values()) + [paper_id]
                    )
                    enriched += 1

            # Search S2 for embeddings if still missing
            if not conn.execute(
                "SELECT embedding FROM papers WHERE paper_id = ? AND embedding IS NOT NULL",
                (paper_id,)
            ).fetchone():
                s2_data = search_s2_by_title(title, year)
                if s2_data and s2_data.get("embedding"):
                    try:
                        import json
                        embedding_json = json.dumps(s2_data["embedding"])
                        conn.execute(
                            "UPDATE papers SET embedding = ? WHERE paper_id = ?",
                            (embedding_json, paper_id)
                        )
                        enriched += 1
                    except Exception:
                        pass

            if (idx + 1) % 100 == 0:
                conn.commit()
                print(f"[enrich] Processed {idx + 1}/{len(papers)}, enriched {enriched}")
                time.sleep(0.5)  # Rate limiting

        except Exception as e:
            continue

    conn.commit()
    print(f"[enrich] Complete: enriched {enriched} papers")
    return enriched


if __name__ == "__main__":
    conn = sqlite3.connect(paths.SQL_DIR / "extended.db")
    count = enrich_incomplete_papers(conn)
    conn.close()
    print(f"[enrich] Total enriched: {count}")

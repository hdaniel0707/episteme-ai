"""Crossref fetcher — cursor-paginated /works API for journal articles.

Query filters: type:journal-article, optional year range, optional DOI presence.
Resume-safe via cursor in fetch_state table.
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Iterator

import requests

from workflow.src.iwf import config
from workflow.src.iwf.data.identity import normalize_doi
from workflow.src.iwf.fetchers.base import BaseFetcher, QuotaExhausted, get_cursor, set_cursor

CROSSREF_WORKS = "https://api.crossref.org/works"


class CrossrefFetcher(BaseFetcher):
    """Crossref /works cursor-paginated crawler."""

    name = "crossref"

    def __init__(self, mailto: str | None = None,
                 year_from: int | None = None, year_to: int | None = None):
        self.mailto = mailto or config.get_str("OPENALEX_MAILTO", "")
        self.year_from = year_from
        self.year_to = year_to

    def rows(self, conn: sqlite3.Connection) -> Iterator[dict]:
        """Yield normalized paper dicts from Crossref /works."""
        # Build filter string
        filters = ["type:journal-article"]
        if self.year_from and self.year_to:
            filters.append(f"from-pub-date:{self.year_from}-01-01")
            filters.append(f"until-pub-date:{self.year_to}-12-31")

        params = {
            "rows": 200,
            "filter": ",".join(filters),
        }
        if self.mailto:
            params["mailto"] = self.mailto

        cursor = get_cursor(conn, self.name) or "*"
        fetched = 0

        while cursor:
            params["cursor"] = cursor
            try:
                r = requests.get(CROSSREF_WORKS, params=params, timeout=60)
                if r.status_code == 429:
                    retry_after = int(r.headers.get("retry-after", 0) or 0)
                    if retry_after > 60:
                        raise QuotaExhausted(
                            f"Crossref quota exhausted (retry-after={retry_after}s)")
                    time.sleep(3.0)
                    continue
                if not r.ok:
                    print(f"[crossref] HTTP {r.status_code}: {r.text[:200]}")
                    break

                data = r.json()
                for item in data.get("message", {}).get("items", []):
                    fetched += 1
                    yield self._work_to_row(item)

                # Get next cursor
                cursor = data.get("message", {}).get("next-cursor")
                set_cursor(conn, self.name, cursor, fetched)
                time.sleep(1.0)  # Polite rate limit

            except QuotaExhausted:
                raise
            except requests.RequestException as e:
                print(f"[crossref] request error: {e}")
                break

    def _work_to_row(self, work: dict) -> dict:
        """Normalize Crossref work object to our schema."""
        doi = work.get("DOI")
        if not doi:
            return {}  # Skip rows without DOI

        paper_id = f"CR:{normalize_doi(doi)}"
        title = work.get("title")
        if isinstance(title, list) and title:
            title = title[0]

        abstract_raw = work.get("abstract") or ""
        # Strip JATS/HTML tags
        abstract = re.sub(r"<[^>]+>", " ", abstract_raw).strip()
        abstract = abstract if abstract else None

        year = None
        issued = work.get("issued", {}).get("date-parts")
        if issued and isinstance(issued, list) and len(issued) > 0:
            date_parts = issued[0]
            if isinstance(date_parts, list) and len(date_parts) > 0:
                year = date_parts[0]

        venue = work.get("container-title")
        if isinstance(venue, list) and venue:
            venue = venue[0]

        citation_count = work.get("is-referenced-by-count")

        # Authors
        authors = []
        for a in work.get("author", []):
            if not isinstance(a, dict):
                continue
            orcid = a.get("ORCID")
            given = a.get("given", "")
            family = a.get("family", "")
            if orcid or family:
                author_id = f"CR:{orcid if orcid else given + '_' + family}"
                name = f"{given} {family}".strip()
                authors.append({"author_id": author_id, "name": name})

        return {
            "paper_id": paper_id,
            "doi": normalize_doi(doi),
            "title": title,
            "abstract": abstract,
            "year": year,
            "venue": venue,
            "citation_count": citation_count,
            "authors": authors,
            "source": "crossref",
        }

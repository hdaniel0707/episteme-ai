"""arXiv fetcher — Atom API for papers in CS and neighboring fields.

Queries by category (cs.*, stat.*, physics.*, etc.), paginated by start/max_results.
Resume-safe via cursor in fetch_state table (cursor = "category:start").
"""

from __future__ import annotations

import sqlite3
import time
import xml.etree.ElementTree as ET
from typing import Iterator

import requests

from workflow.src.iwf.data.identity import normalize_doi
from workflow.src.iwf.fetchers.base import BaseFetcher, get_cursor, set_cursor

ARXIV_API = "http://export.arxiv.org/api/query"

# Atom XML namespace
ATOM_NS = "{http://www.w3.org/2005/Atom}"


class ArxivFetcher(BaseFetcher):
    """arXiv API fetcher for CS, statistics, physics, and related fields."""

    name = "arxiv"

    def __init__(self, categories: list[str] | None = None, delay_s: float = 3.0):
        self.categories = categories or ["cs", "stat", "physics", "q-bio", "econ", "math"]
        self.delay_s = delay_s

    def rows(self, conn: sqlite3.Connection) -> Iterator[dict]:
        """Yield normalized paper dicts from arXiv Atom feed."""
        for category in self.categories:
            state_key = f"{self.name}:{category}"
            cursor_str = get_cursor(conn, state_key) or f"{category}:0"
            # Parse cursor format: "category:start"
            parts = cursor_str.rsplit(":", 1)
            start = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

            fetched = 0
            max_results = 100

            while True:
                params = {
                    "search_query": f"cat:{category}*",
                    "start": start,
                    "max_results": max_results,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }

                try:
                    r = requests.get(ARXIV_API, params=params, timeout=60)
                    if not r.ok:
                        print(f"[arxiv] HTTP {r.status_code}")
                        break

                    # Parse Atom XML
                    try:
                        root = ET.fromstring(r.content)
                    except ET.ParseError as e:
                        print(f"[arxiv] XML parse error: {e}")
                        break

                    entries = root.findall(f"{ATOM_NS}entry")
                    if not entries:
                        # No more results for this category
                        break

                    for entry in entries:
                        fetched += 1
                        row = self._entry_to_row(entry, category)
                        if row:
                            yield row

                    # Update cursor for resume
                    next_start = start + len(entries)
                    set_cursor(conn, state_key, f"{category}:{next_start}", fetched)

                    # Stop if we got fewer results than requested (last page)
                    if len(entries) < max_results:
                        break

                    start = next_start
                    time.sleep(self.delay_s)

                except requests.RequestException as e:
                    print(f"[arxiv] request error: {e}")
                    break

    def _entry_to_row(self, entry: ET.Element, category: str) -> dict | None:
        """Convert Atom entry to normalized paper dict."""
        # ID: typically "http://arxiv.org/abs/2312.12345v1" → "2312.12345"
        id_elem = entry.find(f"{ATOM_NS}id")
        if id_elem is None or not id_elem.text:
            return None
        entry_id = id_elem.text.strip()
        # Extract arXiv ID from URL
        arxiv_id = entry_id.split("/abs/")[-1].split("v")[0]

        title_elem = entry.find(f"{ATOM_NS}title")
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else None

        summary_elem = entry.find(f"{ATOM_NS}summary")
        abstract = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else None

        # Year from published date
        year = None
        published_elem = entry.find(f"{ATOM_NS}published")
        if published_elem is not None and published_elem.text:
            try:
                year = int(published_elem.text[:4])
            except ValueError:
                pass

        # Primary category
        venue = f"arXiv:{category}"
        for tag in entry.findall(f"{ATOM_NS}category"):
            term = tag.get("term", "")
            if term.startswith(category):
                venue = f"arXiv:{term}"
                break

        # DOI if available
        doi = None
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "doi":
                doi = normalize_doi(link.get("href"))
                break

        return {
            "paper_id": f"ARX:{arxiv_id}",
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "year": year,
            "venue": venue,
            "citation_count": None,  # arXiv doesn't provide citation counts
            "domains": [category],
            "source": "arxiv",
        }

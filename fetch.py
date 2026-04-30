#!/usr/bin/env python3
"""
Fetch NeurIPS / ICLR / ICML papers (2020-2024) from the Semantic Scholar API
and persist them to an SQLite database.

Usage:
    python fetch.py [--db papers.db] [--api-key <key>] [--limit N]

Set env var S2_API_KEY to avoid the unauthenticated rate limit (100 req/5 min).
Use --limit to cap the total number of papers saved (useful for testing).
Omit --limit or pass 0 to fetch everything.
"""

import argparse
import os
import time

import sqlite3
from pathlib import Path

from tqdm import tqdm

from dotenv import load_dotenv
from utils.env_utils import doublecheck_env, doublecheck_pkgs
from utils.sqlite_utils import init_db, save_paper
from utils.s2_utils import _session, iter_search

load_dotenv()

# Check and print results
doublecheck_env(".env")  # check environmental variables
doublecheck_pkgs(pyproject_path="pyproject.toml", verbose=True)  # check packages


PAPER_FIELDS = (
    "paperId,title,abstract,embedding,"
    "authors,citationCount,influentialCitationCount,year,venue"
)

REQUEST_DELAY = 1.5  # seconds between paginated requests

# VENUES = ["NeurIPS", "ICLR", "ICML"]
VENUES = ["NeurIPS"]
YEARS = list(range(2020, 2024))

# Substrings to match against the raw venue string (case-insensitive)
VENUE_ALIASES: dict[str, set[str]] = {
    "NeurIPS": {"neurips", "nips", "neural information processing"},
    "ICLR": {"iclr", "international conference on learning representations"},
    "ICML": {"icml", "international conference on machine learning"},
}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def venue_matches(raw: str | None, target: str) -> bool:
    if not raw:
        return False
    lower = raw.lower()
    return any(alias in lower for alias in VENUE_ALIASES[target])


# ---------------------------------------------------------------------------
# Main fetch logic
# ---------------------------------------------------------------------------


def fetch_by_venue_year(
    conn: sqlite3.Connection, api_key: str | None, limit: int = 0
) -> None:
    """Fetch papers. limit=0 means no cap (fetch all)."""
    total_saved = 0
    session = _session(api_key)

    for venue in VENUES:
        for year in YEARS:
            if limit and total_saved >= limit:
                break

            query = f"{venue} {year}"
            saved = skipped = 0
            papers = iter_search(query, session, PAPER_FIELDS)
            for paper in tqdm(papers, desc=f"{venue} {year}", unit="paper"):
                if limit and total_saved >= limit:
                    break
                if paper.get("year") != year:
                    skipped += 1
                    continue
                if not venue_matches(paper.get("venue"), venue):
                    skipped += 1
                    continue
                save_paper(conn, paper)
                saved += 1
                total_saved += 1

            conn.commit()
            tqdm.write(f"  [{venue} {year}] saved {saved}  (skipped {skipped})")
            time.sleep(REQUEST_DELAY)
        else:
            continue
        break

    print(f"\nTotal papers saved: {total_saved}")


def fetch_by_topic_year(
    conn: sqlite3.Connection,
    api_key: str | None,
    topic: str,
    years: list[int] | None = None,
    limit: int = 0,
) -> None:
    """Fetch papers matching *topic* for each year in *years*, with no venue filter.

    years defaults to the module-level YEARS list.
    limit=0 means no cap (fetch all).
    """
    if years is None:
        years = YEARS

    total_saved = 0
    session = _session(api_key)

    for year in years:
        if limit and total_saved >= limit:
            break

        query = f"{topic} {year}"
        saved = skipped = 0
        papers = iter_search(query, session, PAPER_FIELDS)
        for paper in tqdm(papers, desc=f"{topic} {year}", unit="paper"):
            if limit and total_saved >= limit:
                break
            if paper.get("year") != year:
                skipped += 1
                continue
            save_paper(conn, paper)
            saved += 1
            total_saved += 1

        conn.commit()
        tqdm.write(f"  [{topic} {year}] saved {saved}  (skipped {skipped})")
        time.sleep(REQUEST_DELAY)

    print(f"\nTotal papers saved: {total_saved}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch ML conference papers into SQLite."
    )
    parser.add_argument("--db", default="topic02.db", help="SQLite database file")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Stop after saving N papers total (0 = fetch all)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

    Path(args.db).unlink(missing_ok=True)
    conn = sqlite3.connect(args.db)
    init_db(conn)

    try:
        # fetch_by_venue_year(conn, api_key, limit=args.limit)
        fetch_by_topic_year(
            conn=conn,
            api_key=api_key,
            topic="sim2real",
            years=[2022, 2023],
            limit=10,
        )
    finally:
        conn.close()

    print(f"\nDone. Database written to {args.db!r}")


if __name__ == "__main__":
    main()

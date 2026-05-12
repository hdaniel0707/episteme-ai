#!/usr/bin/env python3
import argparse
import os
import time
from itertools import islice

import sqlite3
import yaml
from pathlib import Path

from tqdm import tqdm

from dotenv import load_dotenv
from aip.utils.env_utils import doublecheck_env, doublecheck_pkgs
from aip.utils.sqlite_utils import init_db, save_paper
from aip.utils.s2_utils import _session, iter_search

load_dotenv()

# Check and print results
doublecheck_env(".env")  # check environmental variables
doublecheck_pkgs(pyproject_path="pyproject.toml", verbose=True)  # check packages


PAPER_FIELDS = (
    "paperId,title,abstract,embedding,"
    "authors.authorId,authors.name,authors.hIndex,"
    "authors.citationCount,authors.paperCount,authors.affiliations,"
    "citationCount,influentialCitationCount,year,venue"
)

BASE_DIR = Path(__file__).parent.parent
DB_DIR = BASE_DIR / "data"
DB_PATH = DB_DIR / "sql" / "X_0512_B.db"
CFG_PATH = BASE_DIR / "cfg" / "fetch" / "hd" / "fetch_cfg.yaml"

with open(CFG_PATH) as f:
    QUERIES_DOMAINS: dict[str, list[str]] = yaml.safe_load(f)["queries_domains"]
QUERIES = [q for qs in QUERIES_DOMAINS.values() for q in qs]
TOPIC_LIMIT = 1


def fetch(
    conn: sqlite3.Connection,
    api_key: str | None,
    query: str,
    domain: str,
    limit: int = 10,
) -> None:
    session = _session(api_key)
    saved = skipped = 0
    papers = iter_search(query, session, PAPER_FIELDS)

    for paper in tqdm(
        islice(papers, limit),
        total=limit,
        desc=f"{query}",
        unit="paper",
        leave=False,
    ):
        save_paper(conn, paper, query, domain)
        saved += 1

    conn.commit()
    tqdm.write(f"  [{query}] saved {saved}  (skipped {skipped})")
    time.sleep(1.5)


def main() -> None:

    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(DB_PATH).unlink(missing_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn, delete_if_exsist=True)

    try:
        domain_bar = tqdm(QUERIES_DOMAINS.items(), desc="domains")
        for domain, queries in domain_bar:
            domain_bar.set_description(f"domain: {domain}")
            for query in tqdm(queries, desc="queries", leave=False):
                query_limit = max(1, TOPIC_LIMIT // len(queries))
                fetch(
                    conn=conn,
                    api_key=api_key,
                    query=query,
                    domain=domain,
                    limit=query_limit,
                )
    finally:
        conn.close()

    print(f"\nDone. Database written to {DB_PATH!r}")


if __name__ == "__main__":
    main()

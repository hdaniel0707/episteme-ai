#!/usr/bin/env python3
import argparse
import os
import time
from itertools import islice

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

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "data"
DB_PATH = DB_DIR / "sql" / "0504_B.db"
QUERY = "curriculum learning"


def fetch(
    conn: sqlite3.Connection,
    api_key: str | None,
    query: str,
    limit: int = 10,
) -> None:
    session = _session(api_key)
    saved = skipped = 0
    papers = iter_search(query, session, PAPER_FIELDS)

    for paper in tqdm(
        islice(papers, limit), total=limit, desc=f"{query}", unit="paper"
    ):
        save_paper(conn, paper)
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
        fetch(
            conn=conn,
            api_key=api_key,
            query=QUERY,
            limit=500,
        )
    finally:
        conn.close()

    print(f"\nDone. Database written to {DB_PATH!r}")


if __name__ == "__main__":
    main()

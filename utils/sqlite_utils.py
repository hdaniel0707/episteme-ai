import sqlite3
import json
from pathlib import Path
import pandas as pd


def init_db(conn: sqlite3.Connection, delete_if_exsist=False) -> None:
    if delete_if_exsist:
        conn.executescript("""
            DROP TABLE IF EXISTS paper_authors;
            DROP TABLE IF EXISTS papers;
            DROP TABLE IF EXISTS authors;
        """)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id                    TEXT PRIMARY KEY,
            title                       TEXT,
            abstract                    TEXT,
            embedding                   TEXT,          -- JSON array (SPECTER vector)
            citation_count              INTEGER,
            influential_citation_count  INTEGER,
            year                        INTEGER,
            venue                       TEXT
        );
        CREATE TABLE IF NOT EXISTS authors (
            author_id   TEXT PRIMARY KEY,
            name        TEXT
        );
        CREATE TABLE IF NOT EXISTS paper_authors (
            paper_id    TEXT NOT NULL REFERENCES papers(paper_id),
            author_id   TEXT NOT NULL REFERENCES authors(author_id),
            PRIMARY KEY (paper_id, author_id)
        );
        CREATE INDEX IF NOT EXISTS idx_papers_year  ON papers(year);
        CREATE INDEX IF NOT EXISTS idx_papers_venue ON papers(venue);
    """)
    conn.commit()


def save_paper(conn: sqlite3.Connection, paper: dict) -> None:
    emb = paper.get("embedding")
    embedding_json = json.dumps(emb["vector"]) if emb else None

    conn.execute(
        """
        INSERT OR REPLACE INTO papers
          (paper_id, title, abstract, embedding,
           citation_count, influential_citation_count, year, venue)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper["paperId"],
            paper.get("title"),
            paper.get("abstract"),
            embedding_json,
            paper.get("citationCount"),
            paper.get("influentialCitationCount"),
            paper.get("year"),
            paper.get("venue"),
        ),
    )

    for author in paper.get("authors") or []:
        aid = author.get("authorId")
        if not aid:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO authors (author_id, name) VALUES (?, ?)",
            (aid, author.get("name")),
        )
        conn.execute(
            "INSERT OR IGNORE INTO paper_authors (paper_id, author_id) VALUES (?, ?)",
            (paper["paperId"], aid),
        )


def load_db(path: Path) -> pd.DataFrame:
    """Load papers + per-paper author count from one SQLite database file."""
    conn = sqlite3.connect(path)
    papers = pd.read_sql(
        """
        SELECT paper_id, title, abstract, embedding,
               citation_count, influential_citation_count, year, venue
        FROM papers
        """,
        conn,
    )
    author_counts = pd.read_sql(
        "SELECT paper_id, COUNT(*) AS author_count FROM paper_authors GROUP BY paper_id",
        conn,
    )
    conn.close()
    df = papers.merge(author_counts, on="paper_id", how="left")
    df["source_db"] = Path(path).stem
    df = df[df["embedding"].notna()].reset_index(drop=True)
    df["embedding"] = df["embedding"].map(json.loads)
    return df

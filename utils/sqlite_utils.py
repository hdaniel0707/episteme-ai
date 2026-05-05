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
            author_id       TEXT PRIMARY KEY,
            name            TEXT,
            h_index         INTEGER,
            citation_count  INTEGER,
            paper_count     INTEGER,
            affiliations    TEXT   -- JSON array of strings
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
        affiliations = author.get("affiliations")
        affiliations_json = (
            json.dumps(affiliations) if affiliations is not None else None
        )
        conn.execute(
            """
            INSERT INTO authors (author_id, name, h_index, citation_count, paper_count, affiliations)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(author_id) DO UPDATE SET
                name           = COALESCE(excluded.name,           name),
                h_index        = COALESCE(excluded.h_index,        h_index),
                citation_count = COALESCE(excluded.citation_count, citation_count),
                paper_count    = COALESCE(excluded.paper_count,    paper_count),
                affiliations   = COALESCE(excluded.affiliations,   affiliations)
            """,
            (
                aid,
                author.get("name"),
                author.get("hIndex"),
                author.get("citationCount"),
                author.get("paperCount"),
                affiliations_json,
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO paper_authors (paper_id, author_id) VALUES (?, ?)",
            (paper["paperId"], aid),
        )


def load_db(
    path: Path,
    year_from: int | None = None,
    year_to: int | None = None,
) -> pd.DataFrame:
    """Load papers + per-paper author count and best author h-index from one SQLite database file."""
    conditions = []
    params: list[int] = []
    if year_from is not None:
        conditions.append("year >= ?")
        params.append(year_from)
    if year_to is not None:
        conditions.append("year <= ?")
        params.append(year_to)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    conn = sqlite3.connect(path)
    papers = pd.read_sql(
        f"""
        SELECT paper_id, title, abstract, embedding,
               citation_count, influential_citation_count, year, venue
        FROM papers
        {where}
        """,
        conn,
        params=params or None,
    )
    author_counts = pd.read_sql(
        "SELECT paper_id, COUNT(*) AS author_count FROM paper_authors GROUP BY paper_id",
        conn,
    )
    best_h_index = pd.read_sql(
        """
        SELECT pa.paper_id, MAX(a.h_index) AS best_author_h_index
        FROM paper_authors pa
        JOIN authors a ON pa.author_id = a.author_id
        WHERE a.h_index IS NOT NULL
        GROUP BY pa.paper_id
        """,
        conn,
    )
    conn.close()
    df = papers.merge(author_counts, on="paper_id", how="left")
    df = df.merge(best_h_index, on="paper_id", how="left")
    df["source_db"] = Path(path).stem
    df = df[
        df["embedding"].notna() & df["best_author_h_index"].notna() & df["year"].notna()
    ].reset_index(drop=True)
    df["embedding"] = df["embedding"].map(json.loads)
    return df

import sqlite3
import json


def init_db(conn: sqlite3.Connection) -> None:
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

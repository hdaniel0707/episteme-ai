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
            venue                       TEXT,
            queries                     TEXT,          -- JSON array of search queries
            domains                     TEXT,          -- JSON array of research domains
            generated_by                TEXT DEFAULT 'human'
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


def save_paper(
    conn: sqlite3.Connection,
    paper: dict,
    query: str | None = None,
    domain: str | None = None,
) -> None:
    emb = paper.get("embedding")
    embedding_json = json.dumps(emb["vector"]) if emb else None

    paper_id = paper["paperId"]
    row = conn.execute(
        "SELECT queries, domains FROM papers WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    if row and row[0]:
        existing_queries = json.loads(row[0])
        if query and query not in existing_queries:
            existing_queries.append(query)
        queries_json = json.dumps(existing_queries)
    else:
        queries_json = json.dumps([query]) if query else None

    if row and row[1]:
        existing_domains = json.loads(row[1])
        if domain and domain not in existing_domains:
            existing_domains.append(domain)
        domains_json = json.dumps(existing_domains)
    else:
        domains_json = json.dumps([domain]) if domain else None

    conn.execute(
        """
        INSERT INTO papers
          (paper_id, title, abstract, embedding,
           citation_count, influential_citation_count, year, venue, queries, domains)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            title                      = COALESCE(excluded.title,                      title),
            abstract                   = COALESCE(excluded.abstract,                   abstract),
            embedding                  = COALESCE(excluded.embedding,                  embedding),
            citation_count             = COALESCE(excluded.citation_count,             citation_count),
            influential_citation_count = COALESCE(excluded.influential_citation_count, influential_citation_count),
            year                       = COALESCE(excluded.year,                       year),
            venue                      = COALESCE(excluded.venue,                      venue),
            queries                    = excluded.queries,
            domains                    = excluded.domains
        """,
        (
            paper_id,
            paper.get("title"),
            paper.get("abstract"),
            embedding_json,
            paper.get("citationCount"),
            paper.get("influentialCitationCount"),
            paper.get("year"),
            paper.get("venue"),
            queries_json,
            domains_json,
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
    require_fields: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> pd.DataFrame:
    """Load papers + per-paper author count and best author h-index from one SQLite database file."""
    if require_fields is None:
        require_fields = ["embedding", "best_author_h_index", "year"]
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
        SELECT * FROM papers
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
    if require_fields:
        mask = pd.Series(True, index=df.index)
        for field in require_fields:
            mask &= df[field].notna()
        df = df[mask].reset_index(drop=True)
    df["embedding"] = df["embedding"].map(json.loads)
    df["queries"] = df["queries"].map(lambda v: json.loads(v) if v else [])
    df["domains"] = df["domains"].map(lambda v: json.loads(v) if v else [])
    return df

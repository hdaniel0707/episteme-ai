"""Fetcher base: the extended DB schema + normalized-dict insertion with dedup/resume.

The extended DB (workflow/data/sql/extended.db by default) uses the professor's exact
``papers/authors/paper_authors`` schema (DDL mirrored from epai/utils/sqlite_utils.py —
copied, not imported, so the schema here can gain extra nullable columns without touching
the base repo) plus impact/provenance columns. Real records only: every row comes from an
API response; rows without an abstract are skipped.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from workflow.src.iwf import paths
from workflow.src.iwf.data.identity import normalize_doi


class QuotaExhausted(Exception):
    """Raised when a fetcher's quota is exhausted (HTTP 429 with long retry-after).

    Retrying per-row would burn hours for zero benefit; better to checkpoint and
    resume later."""


# The one dict contract every fetcher produces (missing keys are stored as NULL).
NORMALIZED_KEYS = (
    "paper_id", "doi", "title", "abstract", "year", "venue", "venue_issn",
    "citation_count", "influential_citation_count", "fwci",
    "citation_normalized_percentile", "is_in_top_1_percent", "is_in_top_10_percent",
    "domains", "subfield", "authors", "source", "venue_ais", "venue_zone", "oa_id",
)

_SCHEMA = """
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
    generated_by                TEXT DEFAULT 'human',
    -- extension columns (all nullable; absent in the professor's DBs)
    doi                         TEXT,
    source                      TEXT,          -- s2_original | openalex | uefiscdi_journal | s2
    fwci                        REAL,
    citation_normalized_percentile REAL,
    is_in_top_1_percent         INTEGER,
    is_in_top_10_percent        INTEGER,
    subfield                    TEXT,
    venue_issn                  TEXT,
    venue_ais                   REAL,
    venue_zone                  TEXT,
    oa_id                       TEXT
);
CREATE TABLE IF NOT EXISTS authors (
    author_id       TEXT PRIMARY KEY,
    name            TEXT,
    h_index         INTEGER,
    citation_count  INTEGER,
    paper_count     INTEGER,
    affiliations    TEXT
);
CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id    TEXT NOT NULL REFERENCES papers(paper_id),
    author_id   TEXT NOT NULL REFERENCES authors(author_id),
    PRIMARY KEY (paper_id, author_id)
);
CREATE TABLE IF NOT EXISTS fetch_state (
    source      TEXT PRIMARY KEY,
    cursor      TEXT,
    fetched     INTEGER DEFAULT 0,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_papers_year   ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_venue  ON papers(venue);
CREATE INDEX IF NOT EXISTS idx_papers_doi    ON papers(doi);
CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);
"""


def raw_db_path() -> Path:
    return paths.SQL_DIR / "raw.db"


def extended_db_path() -> Path:
    return paths.SQL_DIR / "extended.db"


def open_db(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path) if path else extended_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _title_hash(title: str | None, year) -> str | None:
    t = re.sub(r"\W+", " ", str(title or "")).strip().lower()
    if not t:
        return None
    return hashlib.sha1(f"{t}|{year}".encode("utf-8")).hexdigest()


class Deduper:
    """In-memory dedup over paper_id, DOI and (normalized title, year)."""

    def __init__(self, conn: sqlite3.Connection):
        self.ids = {r[0] for r in conn.execute("SELECT paper_id FROM papers")}
        self.dois = {r[0] for r in
                     conn.execute("SELECT doi FROM papers WHERE doi IS NOT NULL")}
        self.title_hashes = {
            _title_hash(r[0], r[1])
            for r in conn.execute("SELECT title, year FROM papers WHERE title IS NOT NULL")
        } - {None}

    def seen(self, row: dict) -> bool:
        pid = str(row.get("paper_id"))
        doi = normalize_doi(row.get("doi"))
        # OpenAlex papers (OA: prefix) are globally unique; don't use title+year for dedup
        # For other sources, use stricter matching (paper_id + DOI only, skip title+year)
        if pid.startswith("OA:"):
            return pid in self.ids or (doi is not None and doi in self.dois)
        # For non-OpenAlex sources, also check title+year to catch duplicates from other sources
        th = _title_hash(row.get("title"), row.get("year"))
        return (pid in self.ids
                or (doi is not None and doi in self.dois)
                or (th is not None and th in self.title_hashes))

    def add(self, row: dict) -> None:
        self.ids.add(str(row.get("paper_id")))
        doi = normalize_doi(row.get("doi"))
        if doi:
            self.dois.add(doi)
        th = _title_hash(row.get("title"), row.get("year"))
        if th:
            self.title_hashes.add(th)


def _normalize_domains(v) -> list:
    """Coerce a fetcher's ``domains`` value (list) OR a copied row's already-JSON-encoded
    TEXT column (str) into a plain Python list, so re-serializing is always idempotent
    and never double-encodes (which previously produced values like '"[\\"Physics\\"]"')."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        s = v
        for _ in range(2):  # unwrap at most one accidental extra encoding layer
            try:
                decoded = json.loads(s)
            except (json.JSONDecodeError, TypeError):
                return [s] if s else []
            if isinstance(decoded, list):
                return decoded
            if isinstance(decoded, str):
                s = decoded
                continue
            break
        return [s] if s else []
    return []


def insert_paper(conn: sqlite3.Connection, row: dict) -> None:
    """Insert one normalized dict (paper + authors + links). No commit here."""
    conn.execute(
        """
        INSERT OR IGNORE INTO papers
          (paper_id, title, abstract, citation_count, influential_citation_count,
           year, venue, domains, generated_by, doi, source, fwci,
           citation_normalized_percentile, is_in_top_1_percent, is_in_top_10_percent,
           subfield, venue_issn, venue_ais, venue_zone, oa_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'human', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(row["paper_id"]), row.get("title"), row.get("abstract"),
            row.get("citation_count"), row.get("influential_citation_count"),
            row.get("year"), row.get("venue"),
            json.dumps(_normalize_domains(row.get("domains"))),
            normalize_doi(row.get("doi")), row.get("source"),
            row.get("fwci"), row.get("citation_normalized_percentile"),
            _int_or_none(row.get("is_in_top_1_percent")),
            _int_or_none(row.get("is_in_top_10_percent")),
            row.get("subfield"), row.get("venue_issn"),
            row.get("venue_ais"), row.get("venue_zone"), row.get("oa_id"),
        ),
    )
    for author in row.get("authors") or []:
        aid = author.get("author_id")
        if not aid:
            continue
        conn.execute(
            """
            INSERT INTO authors (author_id, name, h_index, citation_count, paper_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(author_id) DO UPDATE SET
                name           = COALESCE(excluded.name,           name),
                h_index        = COALESCE(excluded.h_index,        h_index),
                citation_count = COALESCE(excluded.citation_count, citation_count),
                paper_count    = COALESCE(excluded.paper_count,    paper_count)
            """,
            (str(aid), author.get("name"), author.get("h_index"),
             author.get("citation_count"), author.get("paper_count")),
        )
        conn.execute(
            "INSERT OR IGNORE INTO paper_authors (paper_id, author_id) VALUES (?, ?)",
            (str(row["paper_id"]), str(aid)),
        )


def _int_or_none(v):
    if v is None:
        return None
    return int(bool(v))


def get_cursor(conn: sqlite3.Connection, source: str) -> str | None:
    r = conn.execute("SELECT cursor FROM fetch_state WHERE source = ?", (source,)).fetchone()
    return r[0] if r else None


def set_cursor(conn: sqlite3.Connection, source: str, cursor: str | None,
               fetched: int) -> None:
    conn.execute(
        """
        INSERT INTO fetch_state (source, cursor, fetched, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(source) DO UPDATE SET cursor = excluded.cursor,
            fetched = excluded.fetched, updated_at = CURRENT_TIMESTAMP
        """, (source, cursor, int(fetched)))
    conn.commit()


class BaseFetcher(ABC):
    """A fetcher yields normalized dicts; ``run`` inserts them with dedup + progress."""

    name = "base"

    @abstractmethod
    def rows(self, conn: sqlite3.Connection) -> Iterator[dict]:
        ...

    def run(self, conn: sqlite3.Connection, limit: int | None = None,
            keep_incomplete: bool = False) -> int:
        from tqdm import tqdm

        dedup = Deduper(conn)
        inserted = 0
        bar = tqdm(desc=f"[{self.name}]", unit="paper", total=limit)
        try:
            for row in self.rows(conn):
                if limit is not None and inserted >= limit:
                    break
                if not keep_incomplete and (not row.get("abstract") or not str(row.get("abstract")).strip()):
                    continue  # real, usable records only
                if not row.get("paper_id") or dedup.seen(row):
                    continue
                insert_paper(conn, row)
                dedup.add(row)
                inserted += 1
                bar.update(1)
                if inserted % 500 == 0:
                    conn.commit()
        except QuotaExhausted as e:
            set_cursor(conn, self.name, None, inserted)
            conn.commit()
            print(f"[{self.name}] quota exhausted — checkpointed, resume later")
            bar.close()
            print(f"[{self.name}] inserted {inserted} papers (quota hit)")
            return inserted
        conn.commit()
        bar.close()
        print(f"[{self.name}] inserted {inserted} new papers")
        return inserted

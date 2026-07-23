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
from difflib import SequenceMatcher
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


def _title_key(title: str | None) -> str:
    """Compact normalized title for fuzzy dedup: lowercase, drop spaces/underscores/
    hyphens, then keep only alphanumerics. '(Deep-Learning)_v2' -> 'deeplearningv2'."""
    t = str(title or "").lower()
    t = re.sub(r"[\s_\-]+", "", t)
    return re.sub(r"[^a-z0-9]+", "", t)


def _block_key(norm: str) -> tuple:
    """Blocking key so fuzzy comparison stays ~O(n): only titles sharing a length
    bucket AND the same 12-char prefix are ever compared (avoids an O(n^2) sweep)."""
    return (len(norm) // 8, norm[:12])


def _year_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _author_key(name) -> str:
    """Last-name key for author confirmation (lowercased alphanumerics of the last token)."""
    toks = re.sub(r"[^\w\s]", " ", str(name or "")).lower().split()
    return re.sub(r"[^a-z0-9]+", "", toks[-1]) if toks else ""


def _incoming_author_key(row: dict) -> str:
    """First-author key from a normalized row dict (authors list of {name} or an
    'author'/'authors' string), or '' when no author information is available."""
    authors = row.get("authors")
    if isinstance(authors, (list, tuple)) and authors:
        first = authors[0]
        name = first.get("name") if isinstance(first, dict) else first
        return _author_key(name)
    if isinstance(authors, str) and authors.strip():
        return _author_key(authors.split(",")[0])
    if row.get("author"):
        return _author_key(str(row["author"]).split(",")[0])
    return ""


class Deduper:
    """In-memory dedup over paper_id, DOI and (normalized title, year).

    ``fuzzy=True`` additionally enables near-duplicate detection across sources
    (``find_duplicate``): titles are normalized (spaces/_/- stripped) and compared with
    a character-similarity ratio; a match is confirmed by publication year and, when
    available, first-author last name. The threshold defaults to
    ``TITLE_SIMILARITY_THRESHOLD`` in .env.workflow (0.85). Comparison is blocked by
    title length+prefix so it stays roughly linear even on ~100k-row merges.
    """

    def __init__(self, conn: sqlite3.Connection, fuzzy: bool = False,
                 threshold: float | None = None):
        self.ids = {r[0] for r in conn.execute("SELECT paper_id FROM papers")}
        self.dois = {r[0] for r in
                     conn.execute("SELECT doi FROM papers WHERE doi IS NOT NULL")}
        self.title_hashes = {
            _title_hash(r[0], r[1])
            for r in conn.execute("SELECT title, year FROM papers WHERE title IS NOT NULL")
        } - {None}
        self.fuzzy = fuzzy
        self.blocks: dict[tuple, list] = {}
        self.first_author: dict[str, str] = {}
        if fuzzy:
            if threshold is None:
                from workflow.src.iwf import config
                threshold = config.get_float("TITLE_SIMILARITY_THRESHOLD", 0.85)
            self.threshold = float(threshold)
            for pid, title, year in conn.execute(
                    "SELECT paper_id, title, year FROM papers WHERE title IS NOT NULL"):
                norm = _title_key(title)
                if norm:
                    self.blocks.setdefault(_block_key(norm), []).append(
                        (norm, pid, _year_int(year)))
            try:
                for pid, name in conn.execute(
                        "SELECT pa.paper_id, a.name FROM paper_authors pa "
                        "JOIN authors a ON a.author_id = pa.author_id"):
                    self.first_author.setdefault(pid, _author_key(name))
            except sqlite3.OperationalError:
                pass
        else:
            self.threshold = 1.0

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

    def seen_exact(self, row: dict) -> bool:
        """Same-record check by identity only (paper_id or DOI) — used by the merge paths
        so that title-based near-duplicates fall through to ``find_duplicate`` instead."""
        pid = str(row.get("paper_id"))
        doi = normalize_doi(row.get("doi"))
        return pid in self.ids or (doi is not None and doi in self.dois)

    def find_duplicate(self, row: dict) -> str | None:
        """Return the paper_id of an existing near-duplicate (fuzzy title + year/author
        confirmation), or None. No-op unless the Deduper was built with fuzzy=True."""
        if not self.fuzzy:
            return None
        norm = _title_key(row.get("title"))
        if not norm or len(norm) < 6:
            return None
        yr = _year_int(row.get("year"))
        incoming_author = _incoming_author_key(row)
        len_gate = max(6, int(len(norm) * (1.0 - self.threshold)) + 1)
        for cnorm, cpid, cyear in self.blocks.get(_block_key(norm), ()):  # blocked candidates
            if abs(len(cnorm) - len(norm)) > len_gate:
                continue
            if SequenceMatcher(None, norm, cnorm).ratio() < self.threshold:
                continue
            if yr is not None and cyear is not None and yr != cyear:
                continue                                   # same title, different year -> keep both
            cauthor = self.first_author.get(cpid, "")
            if incoming_author and cauthor and incoming_author != cauthor:
                continue                                   # authors disagree -> not a duplicate
            return cpid
        return None

    def add(self, row: dict) -> None:
        pid = str(row.get("paper_id"))
        self.ids.add(pid)
        doi = normalize_doi(row.get("doi"))
        if doi:
            self.dois.add(doi)
        th = _title_hash(row.get("title"), row.get("year"))
        if th:
            self.title_hashes.add(th)
        if self.fuzzy:
            norm = _title_key(row.get("title"))
            if norm:
                self.blocks.setdefault(_block_key(norm), []).append(
                    (norm, pid, _year_int(row.get("year"))))
            ak = _incoming_author_key(row)
            if ak:
                self.first_author.setdefault(pid, ak)


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


# Columns that a duplicate row may contribute to an existing paper (COALESCE-fill only —
# an existing non-null value is never overwritten; the incoming row only fills gaps).
_MERGEABLE_COLS = (
    "abstract", "citation_count", "influential_citation_count", "year", "venue",
    "doi", "fwci", "citation_normalized_percentile", "is_in_top_1_percent",
    "is_in_top_10_percent", "subfield", "venue_issn", "venue_ais", "venue_zone",
    "oa_id", "embedding",
)


def merge_paper_metadata(conn: sqlite3.Connection, existing_pid: str, row: dict) -> None:
    """Gap-fill an existing paper's NULL columns from a duplicate row (no overwrite).

    Used by the DB-merge paths when ``Deduper.find_duplicate`` matches a near-duplicate:
    instead of dropping the incoming row (old INSERT OR IGNORE behaviour) or inserting a
    second copy, its metadata is folded into the row already stored. ``domains`` is filled
    only when the existing value is empty. No commit here.
    """
    sets: list[str] = []
    vals: list = []
    for col in _MERGEABLE_COLS:
        val = row.get(col)
        if col == "doi":
            val = normalize_doi(val)
        elif col in ("is_in_top_1_percent", "is_in_top_10_percent"):
            val = _int_or_none(val)
        if val is None or val == "":
            continue
        sets.append(f"{col} = COALESCE({col}, ?)")
        vals.append(val)
    doms = _normalize_domains(row.get("domains"))
    if doms:
        sets.append("domains = CASE WHEN domains IS NULL OR TRIM(COALESCE(domains,'')) "
                    "IN ('', '[]') THEN ? ELSE domains END")
        vals.append(json.dumps(doms))
    if not sets:
        return
    vals.append(str(existing_pid))
    conn.execute(f"UPDATE papers SET {', '.join(sets)} WHERE paper_id = ?", vals)


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

    # Why the last run() stopped: limit | quota | stalled | exhausted. The daily-reset
    # orchestrator reads it to decide whether waiting could yield more papers.
    stop_reason: str = "exhausted"

    def run(self, conn: sqlite3.Connection, limit: int | None = None,
            keep_incomplete: bool = False, stall_limit: int | None = None) -> int:
        """Insert normalized rows with dedup, resume, and an anti-waste stall guard."""
        from tqdm import tqdm

        dedup = Deduper(conn)
        inserted = 0
        since_last_insert = 0
        self.stop_reason = "exhausted"
        bar = tqdm(desc=f"[{self.name}]", unit="paper", total=limit)
        try:
            for row in self.rows(conn):
                if limit is not None and inserted >= limit:
                    self.stop_reason = "limit"
                    break
                usable = keep_incomplete or (row.get("abstract") and
                                             str(row.get("abstract")).strip())
                if usable and row.get("paper_id") and not dedup.seen(row):
                    insert_paper(conn, row)
                    dedup.add(row)
                    inserted += 1
                    since_last_insert = 0
                    bar.update(1)
                    if inserted % 500 == 0:
                        conn.commit()
                    continue
                since_last_insert += 1
                if stall_limit and since_last_insert >= stall_limit:
                    self.stop_reason = "stalled"
                    print(f"[{self.name}] {since_last_insert} consecutive already-known/"
                          f"unusable rows — query mined out, stopping this session")
                    break
        except QuotaExhausted:
            set_cursor(conn, self.name, None, inserted)
            conn.commit()
            self.stop_reason = "quota"
            bar.close()
            print(f"[{self.name}] rate/daily limit hit — checkpointed at {inserted} papers")
            return inserted
        conn.commit()
        bar.close()
        print(f"[{self.name}] inserted {inserted} new papers ({self.stop_reason})")
        return inserted

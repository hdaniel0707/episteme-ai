"""Merge all source databases and PDF data into extended.db (cumulative, never overwrites).

Workflow:
  1. Merge all source DB files into extended.db with deduplication
  2. Import journal data from UEFISCDI PDFs
  3. Create extended.db as authoritative cumulative dataset

Filters out papers with NULL title/abstract, deduplicates by paper_id/DOI/title+year.

NOTE: This standalone script is NOT wired into the `main_extend_db` pipeline. The
authoritative merge used by `python -m workflow.src.main_extend_db extend` is
``main_extend_db.copy_legacy_dbs`` (+ the inline raw.db copy in ``cmd_extend``), which now
does fuzzy-title dedup and metadata-merge via ``base.Deduper(fuzzy=True)`` /
``base.merge_paper_metadata``. Prefer that path; keep this module only for ad-hoc one-off
merges.
"""

import sqlite3
from pathlib import Path

from workflow.src.iwf import paths
from workflow.src.iwf.data.identity import stable_text_id
from workflow.src.iwf.fetchers.base import Deduper, insert_paper


def merge_all_databases(conn: sqlite3.Connection) -> int:
    """Merge all source databases into extended.db. Returns total papers merged."""
    source_db_dir = paths.EPAI_SQL_DIR
    all_db_files = sorted(source_db_dir.glob("*.db"))

    dedup = Deduper(conn)
    total_copied = 0
    total_skipped = 0

    for db_file in all_db_files:
        if "extended" in db_file.name:
            continue  # skip extended.db itself

        print(f"\n[merge] Processing {db_file.name}...")
        src = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True)
        copied = 0
        skipped = 0

        try:
            papers = src.execute("SELECT * FROM papers").fetchall()
            cols = [d[0] for d in src.execute("SELECT * FROM papers LIMIT 1").description]

            for values in papers:
                try:
                    row = dict(zip(cols, values))
                    # Filter: require title + abstract
                    title = str(row.get("title") or "").strip()
                    abstract = str(row.get("abstract") or "").strip()
                    if not title or not abstract:
                        skipped += 1
                        continue

                    # Get or generate paper_id
                    pid = str(row.get("paper_id") or "").strip()
                    if not pid:
                        year = row.get("year")
                        pid = stable_text_id(title, abstract, salt=str(year or ""))

                    # Skip if already seen
                    if dedup.seen({"paper_id": pid, "title": title, "doi": row.get("doi"), "year": row.get("year")}):
                        skipped += 1
                        continue

                    # Insert paper
                    normalized_row = {
                        "paper_id": pid,
                        "title": title,
                        "abstract": abstract,
                        "year": row.get("year"),
                        "venue": row.get("venue"),
                        "citation_count": row.get("citation_count"),
                        "influential_citation_count": row.get("influential_citation_count"),
                        "doi": row.get("doi"),
                        "domains": row.get("domains"),
                        "subfield": row.get("subfield"),
                        "source": f"s2_original_{db_file.stem}",  # tag with DB name
                    }
                    insert_paper(conn, normalized_row)
                    dedup.add(normalized_row)
                    copied += 1

                except Exception as e:
                    skipped += 1
                    continue

            conn.commit()
            print(f"  OK: {copied} papers merged, {skipped} skipped")
            total_copied += copied
            total_skipped += skipped

        finally:
            src.close()

    print(f"\n[merge] Total: {total_copied} papers merged, {total_skipped} skipped")
    return total_copied


if __name__ == "__main__":
    extended_db_path = paths.SQL_DIR / "extended.db"

    # Open or create extended.db (cumulative, never delete)
    conn = sqlite3.connect(extended_db_path)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS papers (
        paper_id TEXT PRIMARY KEY,
        title TEXT, abstract TEXT, embedding TEXT,
        citation_count INTEGER, influential_citation_count INTEGER,
        year INTEGER, venue TEXT, queries TEXT, domains TEXT,
        generated_by TEXT DEFAULT 'human',
        doi TEXT, source TEXT, fwci REAL,
        citation_normalized_percentile REAL,
        is_in_top_1_percent INTEGER, is_in_top_10_percent INTEGER,
        subfield TEXT, venue_issn TEXT, venue_ais REAL, venue_zone TEXT, oa_id TEXT
    );
    CREATE TABLE IF NOT EXISTS authors (
        author_id TEXT PRIMARY KEY, name TEXT, h_index INTEGER,
        citation_count INTEGER, paper_count INTEGER, affiliations TEXT
    );
    CREATE TABLE IF NOT EXISTS paper_authors (
        paper_id TEXT NOT NULL, author_id TEXT NOT NULL,
        PRIMARY KEY (paper_id, author_id)
    );
    CREATE TABLE IF NOT EXISTS fetch_state (
        source TEXT PRIMARY KEY,
        cursor TEXT,
        fetched INTEGER DEFAULT 0,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()

    # Step 1: Merge all source databases
    print("[merge_workflow] Step 1: Merging all source databases...")
    papers_merged = merge_all_databases(conn)
    print(f"[merge_workflow] Merged {papers_merged} papers\n")

    # Step 2: Extract and import journal data from PDFs
    print("[merge_workflow] Step 2: Extracting journal data from PDFs...")
    try:
        from workflow.src.iwf.fetchers.uefiscdi_pdf import import_journal_data
        journals_imported = import_journal_data(conn)
        print(f"[merge_workflow] Imported {journals_imported} journals from PDFs\n")
    except ImportError:
        print("[merge_workflow] WARNING: pdfplumber not available, skipping PDF import")
        print("               Run 'uv sync' to install dependencies\n")
    except Exception as e:
        print(f"[merge_workflow] WARNING: PDF import failed: {e}\n")

    conn.close()

    # Final report
    final_conn = sqlite3.connect(extended_db_path)
    count = final_conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    final_conn.close()
    print(f"[merge_workflow] SUCCESS: Extended DB ready")
    print(f"                 Total papers: {count}")
    print(f"                 Location: {extended_db_path}")

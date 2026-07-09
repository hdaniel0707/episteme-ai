"""Complete pipeline to build workflow-owned extended DB from all available sources.

    python -m workflow.src.main_extend_db --sources copy,openalex,s2 --target-rows 100000

Unified workflow (each phase skippable via --sources; all resume-safe):
  1. copy     — merge ALL source databases + extract journal data from PDFs into extended.db
                 (cumulative, never deletes; includes all previous extended.db data)
  2. uefiscdi — seed red/yellow-zone journal papers via OpenAlex ISSN queries
  3. openalex — domain-balanced fill up to --target-rows from OpenAlex API
  4. s2       — gap-fill influential_citation_count + SPECTER embedding by DOI
  5. report   — rows per source / domain / year (printed + extended_db_report.png)

Result: extended.db with only papers that have title + abstract (usable for embeddings).

Point the pipeline at the result with ``EPAI_DB=extended`` in workflow/.env.workflow.
Config defaults come from workflow/config.yaml section 7 (fetch_* keys).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
from pathlib import Path
from contextlib import contextmanager

from workflow.src.iwf import paths
from workflow.src.iwf.config import load_config_yaml
from workflow.src.iwf.fetchers import base as fbase
from workflow.src.iwf.fetchers.openalex import OpenAlexFetcher, fill_author_h_index
from workflow.src.iwf.fetchers.s2 import fill_from_s2
from workflow.src.iwf.fetchers.uefiscdi import UefiscdiSeedFetcher, print_download_hints


@contextmanager
def phase_timeout(phase_name: str, timeout_seconds: int):
    """Context manager with timeout for each phase (works on Windows + Unix).

    If phase exceeds timeout, logs warning and continues to next phase.
    Note: Timeout is advisory (doesn't forcefully kill the phase).
    """
    start_time = time.time()
    timeout_event = threading.Event()

    def timeout_monitor():
        """Background thread that warns if timeout exceeded."""
        time.sleep(timeout_seconds)
        if not timeout_event.is_set():
            elapsed = time.time() - start_time
            print(f"\n[extend_db] WARNING: {phase_name} exceeded {timeout_seconds}s timeout")
            print(f"[extend_db] ({elapsed:.1f}s elapsed) - Phase may be hanging, but continuing...")

    monitor = threading.Thread(target=timeout_monitor, daemon=True)
    monitor.start()

    try:
        yield
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[extend_db] ERROR in {phase_name}: {type(e).__name__}: {e}")
        print(f"[extend_db] ({elapsed:.1f}s) Continuing to next phase")
    finally:
        timeout_event.set()
        elapsed = time.time() - start_time
        if elapsed < timeout_seconds:
            print(f"[extend_db] {phase_name} completed in {elapsed:.1f}s")


# --------------------------------------------------------------------- phase 1: copy

def copy_prof_db(conn: sqlite3.Connection, src_db: Path) -> int:
    """Copy papers/authors/links from the professor's DB (read-only) into extended.db.

    Fully defensive: skips any row missing title/abstract (unusable for training),
    handles missing columns/tables gracefully. Never crashes on bad schema or NULL data.
    """
    if not src_db.is_file():
        print(f"[copy] source DB not found ({src_db}); skipping")
        return 0
    src = sqlite3.connect(f"file:{src_db.as_posix()}?mode=ro", uri=True)
    dedup = fbase.Deduper(conn)
    copied = 0
    skipped = 0
    try:
        papers = src.execute("SELECT * FROM papers").fetchall()
        cols = [d[0] for d in src.execute("SELECT * FROM papers LIMIT 1").description]
        for values in papers:
            try:
                row = dict(zip(cols, values))
                # Skip rows missing usable content (title + abstract required for training)
                title = str(row.get("title") or "").strip()
                abstract = str(row.get("abstract") or "").strip()
                if not title or not abstract:
                    skipped += 1
                    continue
                # paper_id may not exist; generate if needed
                pid = str(row.get("paper_id") or "").strip()
                if not pid:
                    from workflow.src.iwf.data.identity import stable_text_id
                    year = row.get("year")
                    pid = stable_text_id(title, abstract, salt=str(year or ""))
                if dedup.seen({"paper_id": pid, "title": title, "year": row.get("year")}):
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO papers
                      (paper_id, title, abstract, embedding, citation_count,
                       influential_citation_count, year, venue, queries, domains,
                       generated_by, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 's2_original')
                    """,
                    (pid, title, abstract,
                     row.get("embedding"), row.get("citation_count"),
                     row.get("influential_citation_count"), row.get("year"),
                     row.get("venue"), row.get("queries"), row.get("domains"),
                     row.get("generated_by", "human")))
                dedup.add({"paper_id": pid, "title": title, "year": row.get("year")})
                copied += 1
            except Exception as e:
                skipped += 1
                continue  # skip any row that fails for any reason
        # authors + links: try to copy if tables exist; skip gracefully if not
        try:
            conn.commit()
            conn.execute("ATTACH DATABASE ? AS src", (str(src_db),))
            try:
                src.execute("SELECT 1 FROM authors LIMIT 1")  # check table exists
                src.execute("SELECT 1 FROM paper_authors LIMIT 1")  # check both exist
                conn.execute("""INSERT OR IGNORE INTO authors
                                (author_id, name, h_index, citation_count, paper_count,
                                 affiliations)
                                SELECT author_id, name, h_index, citation_count, paper_count,
                                       affiliations FROM src.authors WHERE author_id IS NOT NULL""")
                conn.execute("""INSERT OR IGNORE INTO paper_authors
                                SELECT paper_id, author_id FROM src.paper_authors
                                WHERE paper_id IS NOT NULL AND author_id IS NOT NULL""")
                conn.commit()
            except sqlite3.OperationalError:
                pass  # tables don't exist in source; skip author sync
        finally:
            try:
                conn.execute("DETACH DATABASE src")
            except sqlite3.OperationalError:
                pass
    finally:
        src.close()
    conn.commit()
    print(f"[copy] {copied} papers copied, {skipped} skipped (unusable/bad data) from {src_db.name}")
    return copied


# --------------------------------------------------------------------- phase 5: report

def report(conn: sqlite3.Connection, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    by_source = pd.read_sql(
        "SELECT COALESCE(source,'?') source, COUNT(*) n FROM papers GROUP BY source", conn)
    by_year = pd.read_sql(
        "SELECT year, COUNT(*) n FROM papers WHERE year IS NOT NULL GROUP BY year "
        "ORDER BY year", conn)
    with_abs = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL AND abstract != ''"
    ).fetchone()[0]
    with_fwci = conn.execute("SELECT COUNT(*) FROM papers WHERE fwci IS NOT NULL"
                             ).fetchone()[0]
    print(f"\n[report] extended DB: {total} papers  |  abstract: {with_abs}  |  "
          f"fwci: {with_fwci}")
    print(by_source.to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(by_source["source"], by_source["n"], color="tab:blue", alpha=0.85)
    ax1.set_title("papers per source")
    ax1.tick_params(axis="x", rotation=20)
    ax2.bar(by_year["year"], by_year["n"], color="tab:green", alpha=0.85)
    ax2.set_title("papers per year")
    fig.suptitle(f"extended.db — {total} papers")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[report] figure -> {out_png}")


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", default="copy,uefiscdi,openalex,s2",
                    help="comma list of phases: copy,uefiscdi,openalex,s2 (report always runs)")
    ap.add_argument("--target-rows", type=int, default=None,
                    help="stop the openalex fill when the DB reaches this many papers "
                         "(default: fetch_target_rows from config.yaml, else 100000)")
    ap.add_argument("--db", default=None, help="extended DB path (default workflow/data/sql/extended.db)")
    ap.add_argument("--uefiscdi-hints", action="store_true",
                    help="print where to download the journal lists and exit")
    args = ap.parse_args()

    if args.uefiscdi_hints:
        print_download_hints()
        return 0

    cfg = load_config_yaml()
    year_from = cfg.get("fetch_year_from") or cfg.get("year_from")
    year_to = cfg.get("fetch_year_to") or cfg.get("year_to")
    target_rows = int(args.target_rows or cfg.get("fetch_target_rows") or 100000)
    domain_filters = list(cfg.get("fetch_domains") or [""])
    per_journal_cap = int(cfg.get("fetch_per_journal_cap") or 300)

    sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
    paths.ensure_dirs()
    conn = fbase.open_db(args.db)
    print(f"[extend_db] db={args.db or fbase.extended_db_path()}  target={target_rows}  "
          f"years={year_from}-{year_to}  phases={sorted(sources)}")

    try:
        # Phase 1: Merge databases and extract PDFs (timeout: 30 minutes)
        if "copy" in sources:
            print("[extend_db] Phase 1/5: Merge all databases + extract PDFs (timeout: 1800s)")
            with phase_timeout("copy", timeout_seconds=1800):
                from workflow.src.iwf.fetchers.merge_all_databases import merge_all_databases
                papers_merged = merge_all_databases(conn)
                print(f"[extend_db] Merged {papers_merged} papers from all databases")

                # Also import journal data from PDFs if available
                try:
                    from workflow.src.iwf.fetchers.uefiscdi_pdf import import_journal_data
                    journals_imported = import_journal_data(conn)
                    print(f"[extend_db] Imported {journals_imported} journals from PDFs")
                except ImportError:
                    print("[extend_db] PDF extraction skipped (pdfplumber not installed)")

        # Phase 1.5: Enrich incomplete paper metadata (timeout: 30 minutes)
        # This runs automatically after copy to fill in missing DOI/FWCI/embeddings
        print("[extend_db] Phase 1.5/5: Enrich incomplete papers by title lookup (timeout: 1800s)")
        with phase_timeout("enrich", timeout_seconds=1800):
            try:
                from workflow.src.iwf.fetchers.enrich_metadata import enrich_incomplete_papers
                enriched = enrich_incomplete_papers(conn)
                print(f"[extend_db] Enriched {enriched} papers with missing metadata")
            except Exception as e:
                print(f"[extend_db] Metadata enrichment skipped: {e}")

        # Phase 2: UEFISCDI journal seeding (timeout: 10 minutes)
        if "uefiscdi" in sources:
            print("[extend_db] Phase 2/5: UEFISCDI journal seeding (timeout: 600s)")
            with phase_timeout("uefiscdi", timeout_seconds=600):
                try:
                    UefiscdiSeedFetcher(year_from, year_to,
                                        per_journal_cap=per_journal_cap).run(conn)
                except FileNotFoundError as e:
                    print(f"[uefiscdi] skipped: {e}")
                    print_download_hints()

        # Phase 3: OpenAlex fill (timeout: 60 minutes)
        if "openalex" in sources:
            print("[extend_db] Phase 3/5: OpenAlex API fill (timeout: 3600s)")
            with phase_timeout("openalex", timeout_seconds=3600):
                have = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
                remaining = max(0, target_rows - have)
                if remaining:
                    cap = max(1, remaining // max(1, len(domain_filters)))
                    OpenAlexFetcher(year_from, year_to, domain_filters,
                                    per_filter_cap=cap).run(conn, limit=remaining)
                    fill_author_h_index(conn)
                else:
                    print(f"[openalex] target of {target_rows} already reached ({have}); skipping")

        # Phase 4: S2 gap-fill (timeout: 30 minutes)
        if "s2" in sources:
            print("[extend_db] Phase 4/5: S2 gap-fill embeddings + citations (timeout: 1800s)")
            with phase_timeout("s2", timeout_seconds=1800):
                fill_from_s2(conn)

        # Phase 5: Report (timeout: 5 minutes)
        print("[extend_db] Phase 5/5: Generate report (timeout: 300s)")
        with phase_timeout("report", timeout_seconds=300):
            report(conn, paths.DATA_DIR / "extended_db_report.png")
        state = {r[0]: {"cursor": bool(r[1]), "fetched": r[2]} for r in
                 conn.execute("SELECT source, cursor, fetched FROM fetch_state")}
        if state:
            print(f"[extend_db] fetch state: {json.dumps(state)}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

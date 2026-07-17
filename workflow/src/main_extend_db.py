"""Extended DB pipeline: bootstrap (raw.db) and extend (extended.db -> training.db).

Bootstrap phase — fresh start, fetch from every free source, KEEP incomplete rows:
    python -m workflow.src.main_extend_db bootstrap --target-rows 100000
    sources: openalex, crossref, arxiv, s2   ->  workflow/data/sql/raw.db

Extend phase — build the two DBs from raw.db + legacy DBs:
    python -m workflow.src.main_extend_db extend --target-rows 100000
    steps (each skippable via --sources):
      copy      raw.db + every legacy store/data/sql/*.db  -> extended.db (dedup only,
                incomplete rows KEPT; provenance in the `source` column)
      pdf       UEFISCDI/JCR PDF extraction -> journal_data table (AIS/quartiles)
      uefiscdi  red/yellow-zone journal paper seeding (OpenAlex ISSN queries)
      openalex  domain-balanced fill toward --target-rows
      s2        influential-citation + SPECTER gap-fill by DOI
      fill      metadata gap-fill of incomplete rows (OpenAlex+S2 title search) —
                runs BEFORE training.db derivation to maximise usable rows
    then always: report + derive training.db (strict: title+abstract+year+citations).

Legacy (backward compat):
    python -m workflow.src.main_extend_db  # behaves as extend

Point the pipeline at the result with ``EPAI_DB=training`` (or ``extended``) in
workflow/.env.workflow.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import threading
from contextlib import contextmanager
from pathlib import Path

from workflow.src.iwf import paths
from workflow.src.iwf.config import load_config_yaml
from workflow.src.iwf.fetchers import base as fbase
from workflow.src.iwf.fetchers.openalex import OpenAlexFetcher, fill_author_h_index
from workflow.src.iwf.fetchers.s2 import fill_from_s2
from workflow.src.iwf.fetchers.uefiscdi import UefiscdiSeedFetcher, print_download_hints
from workflow.src.iwf.fetchers.crossref import CrossrefFetcher
from workflow.src.iwf.fetchers.arxiv import ArxivFetcher


@contextmanager
def phase_timeout(phase_name: str, timeout_seconds: int):
    """Context manager with advisory timeout for each phase."""
    start_time = time.time()
    timeout_event = threading.Event()

    def timeout_monitor():
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


def report(conn: sqlite3.Connection, out_png: Path) -> None:
    """Generate and save a report figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from workflow.src.iwf.evaluation import palette

    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    by_source = pd.read_sql(
        "SELECT COALESCE(source,'?') source, COUNT(*) n FROM papers GROUP BY source", conn)
    by_year = pd.read_sql(
        "SELECT year, COUNT(*) n FROM papers WHERE year IS NOT NULL GROUP BY year "
        "ORDER BY year", conn)
    with_abs = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL AND abstract != ''"
    ).fetchone()[0]

    print(f"\n[report] DB: {total} papers  |  with abstract: {with_abs}")
    print(by_source.to_string(index=False))

    palette.apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(by_source["source"], by_source["n"], color=palette.DUSK_BLUE, alpha=0.85)
    ax1.set_title("papers per source")
    ax1.tick_params(axis="x", rotation=20)
    ax2.bar(by_year["year"], by_year["n"], color=palette.OLIVE, alpha=0.85)
    ax2.set_title("papers per year")
    fig.suptitle(f"DB - {total} papers")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[report] figure: {out_png}")


def derive_training(ext_path: Path, out_path: Path) -> tuple[int, int]:
    """Create training.db: strict filter (title+abstract+year+citations) from extended.db.

    Rebuilt from scratch every time (pure function of extended.db, so idempotent).
    Reads extended.db via ATTACH; only the column intersection is copied, so an
    extended.db built by an older schema version still derives cleanly.
    """
    out_path = Path(out_path)
    out_path.unlink(missing_ok=True)          # always a fresh derivation
    out_conn = fbase.open_db(out_path)
    out_conn.execute("ATTACH DATABASE ? AS ext", (str(ext_path),))

    n_ext = out_conn.execute("SELECT COUNT(*) FROM ext.papers").fetchone()[0]

    ext_cols = [r[1] for r in out_conn.execute("PRAGMA ext.table_info(papers)")]
    tr_cols = [r[1] for r in out_conn.execute("PRAGMA table_info(papers)")]
    cols = ", ".join(c for c in tr_cols if c in ext_cols)
    inserted = out_conn.execute(f"""
        INSERT OR IGNORE INTO papers ({cols})
        SELECT {cols} FROM ext.papers
        WHERE title IS NOT NULL AND TRIM(title) != ''
          AND abstract IS NOT NULL AND TRIM(abstract) != ''
          AND year IS NOT NULL
          AND citation_count IS NOT NULL
    """).rowcount

    try:
        a_ext = [r[1] for r in out_conn.execute("PRAGMA ext.table_info(authors)")]
        a_tr = [r[1] for r in out_conn.execute("PRAGMA table_info(authors)")]
        a_cols = ", ".join(c for c in a_tr if c in a_ext)
        if a_cols:
            out_conn.execute(f"INSERT OR IGNORE INTO authors ({a_cols}) "
                             f"SELECT {a_cols} FROM ext.authors")
        out_conn.execute("""
            INSERT OR IGNORE INTO paper_authors (paper_id, author_id)
            SELECT pa.paper_id, pa.author_id FROM ext.paper_authors pa
            WHERE pa.paper_id IN (SELECT paper_id FROM papers)
        """)
    except sqlite3.OperationalError:
        pass  # author tables may not exist in older extended.db files

    out_conn.commit()
    out_conn.execute("DETACH DATABASE ext")
    out_conn.close()

    print(f"[derive] extended.db: {n_ext} papers -> training.db: {inserted} papers (strict filter)")
    return n_ext, inserted


def copy_legacy_dbs(conn: sqlite3.Connection) -> int:
    """Copy papers/authors/links from every legacy DB under store/data/sql into
    extended.db — KEEPING incomplete rows (the huge archive stores everything that
    is not a duplicate; completeness filtering happens only in derive_training).

    Rows lacking BOTH title and abstract are skipped (no identity can be derived).
    Provenance label: source = 's2_original_<dbname>'.
    """
    from workflow.src.iwf.data.identity import stable_text_id

    total = 0
    legacy = sorted(paths.EPAI_SQL_DIR.glob("*.db")) if paths.EPAI_SQL_DIR.is_dir() else []
    if not legacy:
        print(f"[copy] no legacy DBs under {paths.EPAI_SQL_DIR}; skipping")
        return 0
    for src_db in legacy:
        src = sqlite3.connect(f"file:{src_db.as_posix()}?mode=ro", uri=True)
        dedup = fbase.Deduper(conn)
        copied = skipped = 0
        try:
            try:
                cur = src.execute("SELECT * FROM papers")
            except sqlite3.OperationalError:
                print(f"[copy] {src_db.name}: no papers table; skipping")
                continue
            cols = [d[0] for d in cur.description]
            for values in cur:
                try:
                    row = dict(zip(cols, values))
                    title = str(row.get("title") or "").strip()
                    abstract = str(row.get("abstract") or "").strip()
                    if not title and not abstract:
                        skipped += 1          # no identity derivable — truly unusable
                        continue
                    pid = str(row.get("paper_id") or "").strip()
                    if not pid:
                        pid = stable_text_id(title, abstract, salt=str(row.get("year") or ""))
                    if dedup.seen({"paper_id": pid, "title": title,
                                   "year": row.get("year"), "doi": row.get("doi")}):
                        continue
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO papers
                          (paper_id, title, abstract, embedding, citation_count,
                           influential_citation_count, year, venue, queries, domains,
                           generated_by, doi, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (pid, title or None, abstract or None,
                         row.get("embedding"), row.get("citation_count"),
                         row.get("influential_citation_count"), row.get("year"),
                         row.get("venue"), row.get("queries"), row.get("domains"),
                         row.get("generated_by", "human"), row.get("doi"),
                         f"s2_original_{src_db.stem}"))
                    dedup.add({"paper_id": pid, "title": title,
                               "year": row.get("year"), "doi": row.get("doi")})
                    copied += 1
                except Exception:
                    skipped += 1
                    continue
            conn.commit()
            # authors + links (attach for a fast bulk copy; skip if tables absent)
            try:
                conn.execute("ATTACH DATABASE ? AS src", (str(src_db),))
                try:
                    src.execute("SELECT 1 FROM authors LIMIT 1")
                    src.execute("SELECT 1 FROM paper_authors LIMIT 1")
                    conn.execute("""INSERT OR IGNORE INTO authors
                        (author_id, name, h_index, citation_count, paper_count, affiliations)
                        SELECT author_id, name, h_index, citation_count, paper_count,
                               affiliations FROM src.authors WHERE author_id IS NOT NULL""")
                    conn.execute("""INSERT OR IGNORE INTO paper_authors
                        SELECT paper_id, author_id FROM src.paper_authors
                        WHERE paper_id IS NOT NULL AND author_id IS NOT NULL""")
                    conn.commit()
                except sqlite3.OperationalError:
                    pass
                finally:
                    try:
                        conn.execute("DETACH DATABASE src")
                    except sqlite3.OperationalError:
                        pass
            except sqlite3.OperationalError:
                pass
        finally:
            src.close()
        print(f"[copy] {src_db.name}: {copied} copied, {skipped} skipped (no identity)")
        total += copied
    conn.commit()
    return total


def cmd_bootstrap(args) -> int:
    """Bootstrap phase: fresh raw.db from all sources."""
    if fbase.raw_db_path().exists():
        have = sqlite3.connect(fbase.raw_db_path()).execute(
            "SELECT COUNT(*) FROM papers"
        ).fetchone()[0]
        if have > 0 and not args.fresh:
            print(f"[bootstrap] raw.db exists with {have} papers; use --fresh to rebuild", file=sys.stderr)
            return 1

    cfg = load_config_yaml()
    year_from = cfg.get("fetch_year_from")
    year_to = cfg.get("fetch_year_to")
    target_rows = args.target_rows or cfg.get("fetch_target_rows") or 50000

    sources = {s.strip().lower() for s in (args.sources or "openalex,crossref,arxiv,s2").split(",")}
    per_source = max(1, target_rows // len(sources))

    paths.ensure_dirs()
    conn = fbase.open_db(fbase.raw_db_path())
    print(f"[bootstrap] raw.db: target={target_rows}, per_source={per_source}, sources={sorted(sources)}")

    try:
        if "openalex" in sources:
            print("[bootstrap] OpenAlex -> raw.db (timeout: 3600s)")
            with phase_timeout("openalex", 3600):
                OpenAlexFetcher(year_from, year_to).run(conn, limit=per_source, keep_incomplete=True)
                fill_author_h_index(conn)

        if "crossref" in sources:
            print("[bootstrap] Crossref -> raw.db (timeout: 3600s)")
            with phase_timeout("crossref", 3600):
                CrossrefFetcher(year_from=year_from, year_to=year_to).run(
                    conn, limit=per_source, keep_incomplete=True)

        if "arxiv" in sources:
            print("[bootstrap] arXiv -> raw.db (timeout: 3600s)")
            with phase_timeout("arxiv", 3600):
                ArxivFetcher().run(conn, limit=per_source, keep_incomplete=True)

        if "s2" in sources:
            print("[bootstrap] S2 bulk -> raw.db (timeout: 1800s)")
            with phase_timeout("s2", 1800):
                fill_from_s2(conn)

        # Report
        print("[bootstrap] Report (timeout: 300s)")
        with phase_timeout("report", 300):
            report(conn, paths.DATA_DIR / "raw_db_report.png")

    finally:
        conn.close()

    return 0


def cmd_extend(args) -> int:
    """Extend phase: raw.db + legacy -> extended.db -> training.db."""
    cfg = load_config_yaml()
    year_from = cfg.get("fetch_year_from")
    year_to = cfg.get("fetch_year_to")
    target_rows = args.target_rows or cfg.get("fetch_target_rows") or 100000
    domain_filters = list(cfg.get("fetch_domains") or [""])
    per_journal_cap = int(cfg.get("fetch_per_journal_cap") or 300)

    sources = {s.strip().lower()
               for s in (args.sources or "copy,pdf,uefiscdi,openalex,s2,fill").split(",")}
    paths.ensure_dirs()

    # Open/create extended.db
    ext_path = fbase.extended_db_path()
    conn = fbase.open_db(ext_path)
    print(f"[extend] extended.db={ext_path}, target={target_rows}, sources={sorted(sources)}")

    try:
        # Phase 1: Copy from raw.db + legacy DBs (without title/abstract filter)
        if "copy" in sources:
            print("[extend] Copy raw.db + legacy DBs (timeout: 1800s)")
            with phase_timeout("copy", 1800):
                # Copy from raw.db if it exists
                if fbase.raw_db_path().exists():
                    raw_conn = sqlite3.connect(fbase.raw_db_path())
                    raw_conn.row_factory = sqlite3.Row
                    dedup = fbase.Deduper(conn)

                    # author_id -> {name,h_index,citation_count,paper_count} + paper_id -> [author_id,...]
                    authors_by_id: dict[str, dict] = {}
                    for a in raw_conn.execute(
                        "SELECT author_id, name, h_index, citation_count, paper_count FROM authors"
                    ):
                        authors_by_id[a["author_id"]] = dict(a)
                    paper_author_ids: dict[str, list[str]] = {}
                    for pa in raw_conn.execute("SELECT paper_id, author_id FROM paper_authors"):
                        paper_author_ids.setdefault(pa["paper_id"], []).append(pa["author_id"])

                    n_copied = 0
                    for row in raw_conn.execute("SELECT * FROM papers"):
                        pid = row["paper_id"]
                        if not dedup.seen({"paper_id": pid}):
                            row_dict = dict(row)
                            row_dict["authors"] = [
                                authors_by_id[aid] for aid in paper_author_ids.get(pid, [])
                                if aid in authors_by_id
                            ]
                            fbase.insert_paper(conn, row_dict)
                            dedup.add({"paper_id": pid})
                            n_copied += 1
                    raw_conn.close()
                    print(f"[extend] Copied {n_copied} papers (+ author records) from raw.db to extended.db")

                # Copy from every legacy/professor DB (kept incomplete; dedup only)
                copy_legacy_dbs(conn)

        # Phase 1b: PDF journal-data extraction (UEFISCDI/JCR lists -> journal_data
        # table; feeds venue_ais + the uefiscdi journal seeding below)
        if "pdf" in sources:
            print("[extend] PDF journal extraction (timeout: 600s)")
            with phase_timeout("pdf", 600):
                from workflow.src.iwf.fetchers.uefiscdi_pdf import import_journal_data
                n_pdf = import_journal_data(conn)
                print(f"[pdf] {n_pdf} journal rows imported from workflow/data/pdf/")

        # Phase 2: Enrichment (openalex, s2, uefiscdi)
        if "uefiscdi" in sources:
            print("[extend] UEFISCDI seeding (timeout: 600s)")
            with phase_timeout("uefiscdi", 600):
                try:
                    UefiscdiSeedFetcher(year_from, year_to, per_journal_cap).run(conn)
                except FileNotFoundError as e:
                    print(f"[uefiscdi] skipped: {e}")

        if "openalex" in sources:
            print("[extend] OpenAlex fill (timeout: 3600s)")
            with phase_timeout("openalex", 3600):
                have = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
                remaining = max(0, target_rows - have)
                if remaining:
                    cap = max(1, remaining // max(1, len(domain_filters)))
                    OpenAlexFetcher(year_from, year_to, domain_filters,
                                    per_filter_cap=cap).run(conn, limit=remaining)
                    fill_author_h_index(conn)

        if "s2" in sources:
            print("[extend] S2 gap-fill (timeout: 1800s)")
            with phase_timeout("s2", 1800):
                fill_from_s2(conn)

        # Phase 2b: metadata gap-fill — BEFORE deriving training.db, so as many
        # incomplete archive rows as possible become training-usable (missing
        # DOI/citations/year filled via OpenAlex + S2 title search).
        if "fill" in sources:
            print("[extend] Gap-fill incomplete papers (timeout: 3600s)")
            with phase_timeout("fill", 3600):
                from workflow.src.iwf.fetchers.enrich_metadata import enrich_incomplete_papers
                n_filled = enrich_incomplete_papers(conn)
                print(f"[fill] {n_filled} papers enriched")

        # Phase 3: Report
        print("[extend] Report (timeout: 300s)")
        with phase_timeout("report", 300):
            report(conn, paths.DATA_DIR / "extended_db_report.png")

    finally:
        conn.close()

    # Phase 4: Derive training.db
    print("[extend] Derive training.db (timeout: 300s)")
    with phase_timeout("training", 300):
        n_ext, n_train = derive_training(ext_path, fbase.extended_db_path().parent / "training.db")

    # Phase 5: Build the local holdout benchmark from the fresh training DB, so the
    # training session later needs no network for benchmark evaluation.
    print("[extend] Build holdout benchmark (timeout: 600s)")
    with phase_timeout("benchmark", 600):
        from workflow.src.iwf.evaluation.benchmarks import build_holdout_benchmark
        build_holdout_benchmark(fbase.extended_db_path().parent / "training.db")

    print(f"\n[extend] complete: extended={n_ext}, training={n_train}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = ap.add_subparsers(dest="cmd", help="subcommand")

    # bootstrap subcommand
    bootstrap_ap = subparsers.add_parser("bootstrap", help="Create raw.db from all sources (fresh start)")
    bootstrap_ap.add_argument("--target-rows", type=int, default=None, help="target total rows (default: 50000)")
    bootstrap_ap.add_argument("--sources", default="openalex,crossref,arxiv,s2", help="comma-separated sources")
    bootstrap_ap.add_argument("--fresh", action="store_true", help="overwrite existing raw.db")

    # extend subcommand
    extend_ap = subparsers.add_parser("extend", help="Create extended.db and training.db")
    extend_ap.add_argument("--target-rows", type=int, default=None, help="target rows in extended.db")
    extend_ap.add_argument("--sources", default="copy,openalex,s2", help="comma-separated sources")

    args = ap.parse_args()

    if args.cmd == "bootstrap":
        return cmd_bootstrap(args)
    elif args.cmd == "extend":
        return cmd_extend(args)
    else:
        # Legacy: no subcommand defaults to extend
        return cmd_extend(argparse.Namespace(
            target_rows=None, sources="copy,openalex,s2", fresh=False
        ))


if __name__ == "__main__":
    sys.exit(main())

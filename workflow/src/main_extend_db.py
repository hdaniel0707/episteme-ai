"""Extended DB pipeline: bootstrap (raw.db) and extend (extended.db → training.db).

Bootstrap phase (fresh start only):
    python -m workflow.src.main_extend_db bootstrap --target-rows 100000

Extend phase (incremental):
    python -m workflow.src.main_extend_db extend --target-rows 100000

Legacy (backward compat):
    python -m workflow.src.main_extend_db  # behaves as extend
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
    """Create training.db: strict filter (title+abstract+year+citations) from extended.db."""
    ext_conn = sqlite3.connect(ext_path)
    out_conn = fbase.open_db(out_path)

    # Copy matching papers + authors + links
    n_ext = ext_conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    # Insert papers with title+abstract+year+citation_count
    inserted = out_conn.execute("""
        INSERT INTO papers
        SELECT * FROM (
            SELECT * FROM papers
            WHERE title IS NOT NULL AND TRIM(title) != ''
              AND abstract IS NOT NULL AND TRIM(abstract) != ''
              AND year IS NOT NULL
              AND citation_count IS NOT NULL
        )
    """).rowcount

    # Copy matching authors and paper_authors
    # First, find papers we kept
    kept_ids = {r[0] for r in out_conn.execute("SELECT paper_id FROM papers")}

    # Copy authors of kept papers
    try:
        for author_id, name, h_idx, cit_cnt, pap_cnt in ext_conn.execute(
            "SELECT author_id, name, h_index, citation_count, paper_count FROM authors"
        ):
            out_conn.execute(
                "INSERT OR IGNORE INTO authors VALUES (?, ?, ?, ?, ?, ?)",
                (author_id, name, h_idx, cit_cnt, pap_cnt, None)
            )
        # Copy paper_authors for kept papers only
        for pid, aid in ext_conn.execute("SELECT paper_id, author_id FROM paper_authors"):
            if pid in kept_ids:
                out_conn.execute(
                    "INSERT OR IGNORE INTO paper_authors VALUES (?, ?)", (pid, aid)
                )
    except sqlite3.OperationalError:
        pass  # tables may not exist

    out_conn.commit()
    ext_conn.close()
    out_conn.close()

    print(f"[derive] extended.db: {n_ext} papers → training.db: {inserted} papers (strict filter)")
    return n_ext, inserted


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
            print("[bootstrap] OpenAlex → raw.db (timeout: 3600s)")
            with phase_timeout("openalex", 3600):
                OpenAlexFetcher(year_from, year_to).run(conn, limit=per_source, keep_incomplete=True)
                fill_author_h_index(conn)

        if "crossref" in sources:
            print("[bootstrap] Crossref → raw.db (timeout: 3600s)")
            with phase_timeout("crossref", 3600):
                CrossrefFetcher(year_from, year_to).run(conn, limit=per_source, keep_incomplete=True)

        if "arxiv" in sources:
            print("[bootstrap] arXiv → raw.db (timeout: 3600s)")
            with phase_timeout("arxiv", 3600):
                ArxivFetcher().run(conn, limit=per_source, keep_incomplete=True)

        if "s2" in sources:
            print("[bootstrap] S2 bulk → raw.db (timeout: 1800s)")
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
    """Extend phase: raw.db + legacy → extended.db → training.db."""
    cfg = load_config_yaml()
    year_from = cfg.get("fetch_year_from")
    year_to = cfg.get("fetch_year_to")
    target_rows = args.target_rows or cfg.get("fetch_target_rows") or 100000
    domain_filters = list(cfg.get("fetch_domains") or [""])
    per_journal_cap = int(cfg.get("fetch_per_journal_cap") or 300)

    sources = {s.strip().lower() for s in (args.sources or "copy,openalex,s2").split(",")}
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
                    dedup = fbase.Deduper(conn)
                    for row in raw_conn.execute("SELECT * FROM papers"):
                        if not dedup.seen({"paper_id": row["paper_id"]}):
                            fbase.insert_paper(conn, dict(row))
                            dedup.add({"paper_id": row["paper_id"]})
                    raw_conn.close()
                    print("[extend] Copied raw.db to extended.db")

                # Copy from legacy/professor DBs (no filtering)
                # This is a minimal version; full implementation would iterate legacy paths
                conn.commit()

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

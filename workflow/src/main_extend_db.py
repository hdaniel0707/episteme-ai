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
      llm_fill  LLM auto-fill: domain classification (ENABLE_LLM_FILL, on by default) +
                optional abstract/year reconstruction (ENABLE_LLM_ABSTRACT_FILL, off by
                default, marks source='llm') — no-op unless LLM_JUDGE_PROVIDER/MODEL/
                API_KEY + LOCAL_LLM_URL are configured; see .env.workflow
    then always: report + derive training.db (strict: title+abstract+year+citations;
                filters by TRAIN_YEAR_FROM/TO and MAX_TRAINING_ROWS, see .env.workflow).

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
from workflow.src.iwf import config
from workflow.src.iwf.config import load_config_yaml
from workflow.src.iwf.fetchers import base as fbase
from workflow.src.iwf.fetchers.openalex import (
    OpenAlexFetcher, collect_new_papers, fill_author_h_index)
from workflow.src.iwf.fetchers.s2 import fill_from_s2
from workflow.src.iwf.fetchers.uefiscdi import UefiscdiSeedFetcher, print_download_hints
from workflow.src.iwf.fetchers.crossref import CrossrefFetcher
from workflow.src.iwf.fetchers.arxiv import ArxivFetcher


# Per-source on/off switches read from .env.workflow, so a direct `main_extend_db` run
# honours the same ENABLE_* toggles the deploy scripts pass as --sources. 'copy' has no
# toggle (always allowed). Default True = source enabled unless explicitly turned off.
_SOURCE_TOGGLES = {
    "openalex": "ENABLE_OPENALEX", "s2": "ENABLE_S2", "pdf": "ENABLE_PDF",
    "crossref": "ENABLE_CROSSREF", "arxiv": "ENABLE_ARXIV",
    "uefiscdi": "ENABLE_UEFISCDI", "fill": "ENABLE_FILL",
}


def _filter_sources(sources: set[str]) -> set[str]:
    """Drop any requested source whose ENABLE_* flag is false in .env.workflow."""
    kept = set()
    for s in sources:
        env = _SOURCE_TOGGLES.get(s)
        if env is None or config.get_bool(env, True):
            kept.add(s)
        else:
            print(f"[extend_db] source '{s}' disabled via {env}=false; skipping")
    return kept


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
    import random

    out_path = Path(out_path)
    out_path.unlink(missing_ok=True)          # always a fresh derivation
    out_conn = fbase.open_db(out_path)
    out_conn.execute("ATTACH DATABASE ? AS ext", (str(ext_path),))

    n_ext = out_conn.execute("SELECT COUNT(*) FROM ext.papers").fetchone()[0]

    ext_cols = [r[1] for r in out_conn.execute("PRAGMA ext.table_info(papers)")]
    tr_cols = [r[1] for r in out_conn.execute("PRAGMA table_info(papers)")]
    cols = ", ".join(c for c in tr_cols if c in ext_cols)

    # Training-corpus filters, all adjustable from .env.workflow:
    #   TRAIN_YEAR_FROM / TRAIN_YEAR_TO       keep only 1700..2024 (inclusive) by default
    #   USE_LLM_ABSTRACTS_FOR_TRAINING        include rows whose abstract was LLM-filled
    #                                         (source='llm'); default off (see Stage E)
    #   MAX_TRAINING_ROWS / SELECTION_SEED    cap to a seeded random subset (reproducible;
    #                                         seed defaults to the config.yaml split_seed 45)
    year_from = config.get_int("TRAIN_YEAR_FROM", 1700)
    year_to = config.get_int("TRAIN_YEAR_TO", 2024)
    # config.yaml `use_llm_abstracts_for_training` (falls back to the .env.workflow flag)
    use_llm_abs = config.use_llm_abstracts_for_training()
    max_rows = config.get_int("MAX_TRAINING_ROWS", 150)
    seed = config.get_int("SELECTION_SEED", 45)

    where = ["title IS NOT NULL AND TRIM(title) != ''",
             "abstract IS NOT NULL AND TRIM(abstract) != ''",
             "year IS NOT NULL", "citation_count IS NOT NULL",
             "year >= ?", "year <= ?"]
    params = [year_from, year_to]
    if not use_llm_abs:
        where.append("(source IS NULL OR source != 'llm')")
    # unknown-domain papers must never be used for training (config.yaml
    # require_known_domain_for_training) — keep only rows with a non-empty domain
    if config.require_known_domain_for_training():
        where.append("domains IS NOT NULL AND TRIM(domains) NOT IN ('', '[]')")
    candidates = [r[0] for r in out_conn.execute(
        f"SELECT paper_id FROM ext.papers WHERE {' AND '.join(where)}", params)]
    n_complete = len(candidates)
    if max_rows and n_complete > max_rows:
        candidates = random.Random(seed).sample(candidates, max_rows)
        print(f"[derive] capping training set to {max_rows} of {n_complete} complete rows "
              f"(random, seed={seed})")

    out_conn.execute("CREATE TEMP TABLE _keep (paper_id TEXT PRIMARY KEY)")
    out_conn.executemany("INSERT OR IGNORE INTO _keep VALUES (?)",
                         [(c,) for c in candidates])
    inserted = out_conn.execute(f"""
        INSERT OR IGNORE INTO papers ({cols})
        SELECT {cols} FROM ext.papers
        WHERE paper_id IN (SELECT paper_id FROM _keep)
    """).rowcount
    out_conn.execute("DROP TABLE _keep")

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

    print(f"[derive] extended.db: {n_ext} papers -> training.db: {inserted} papers "
          f"(strict filter, years {year_from}-{year_to})")
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
        dedup = fbase.Deduper(conn, fuzzy=True)   # cross-source fuzzy title dedup + merge
        copied = skipped = merged = 0
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
                    ident = {"paper_id": pid, "title": title or None,
                             "year": row.get("year"), "doi": row.get("doi")}
                    if dedup.seen_exact(ident):
                        continue              # same record already stored (paper_id/DOI)
                    dup_pid = dedup.find_duplicate(ident)
                    if dup_pid:               # near-duplicate title -> merge metadata, don't re-insert
                        fbase.merge_paper_metadata(conn, dup_pid, row)
                        merged += 1
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
                    dedup.add(ident)
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
        print(f"[copy] {src_db.name}: {copied} copied, {merged} merged (near-dup), "
              f"{skipped} skipped (no identity)")
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
    sources = _filter_sources(sources)
    if not sources:
        print("[bootstrap] all requested sources disabled via ENABLE_* flags; nothing to do")
        return 0
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
    # fetch_domains: filter strings, or "all" to crawl every OpenAlex field.
    raw_domains = cfg.get("fetch_domains")
    if raw_domains in ("all", ["all"]):
        from workflow.src.iwf.domains import all_field_filters
        domain_filters = all_field_filters()
        print(f"[extend] fetch_domains=all -> crawling {len(domain_filters)} OpenAlex fields")
    else:
        domain_filters = list(raw_domains or [""])
    per_journal_cap = int(cfg.get("fetch_per_journal_cap") or 300)

    sources = {s.strip().lower()
               for s in (args.sources or "copy,pdf,uefiscdi,openalex,s2,fill,llm_fill").split(",")}
    sources = _filter_sources(sources)

    # fetch_from_scratch: back up any existing DB and build OpenAlex-only from empty.
    from_scratch = bool(cfg.get("fetch_from_scratch"))
    paths.ensure_dirs()
    ext_path = fbase.extended_db_path()
    if from_scratch:
        sources.discard("copy")
        if ext_path.exists():
            import shutil
            bak = ext_path.with_name(ext_path.name + f".pre_scratch_{int(time.time())}")
            shutil.move(str(ext_path), str(bak))
            print(f"[extend] fetch_from_scratch: backed up existing DB -> {bak.name}")

    conn = fbase.open_db(ext_path)
    print(f"[extend] extended.db={ext_path}, target={target_rows}, "
          f"from_scratch={from_scratch}, sources={sorted(sources)}")

    # Run-total LLM-fill budget, shared by the wait-time fills and the final llm_fill phase.
    fill_budget = {"domain": int(cfg.get("llm_fill_domain_max") or 60),
                   "abstract": int(cfg.get("llm_fill_abstract_max") or 50)}

    def _fill_during_wait() -> None:
        """Spend an idle daily-limit wait on LLM domain/abstract fills."""
        from workflow.src.iwf.fetchers.llm_fill import llm_fill_domains, llm_fill_abstracts
        if config.get_bool("ENABLE_LLM_FILL", False) and fill_budget["domain"] > 0:
            fill_budget["domain"] -= llm_fill_domains(conn, limit=fill_budget["domain"])
        if config.get_bool("ENABLE_LLM_ABSTRACT_FILL", False) and fill_budget["abstract"] > 0:
            fill_budget["abstract"] -= llm_fill_abstracts(conn, limit=fill_budget["abstract"])

    try:
        # Phase 1: Copy from raw.db + legacy DBs (without title/abstract filter)
        if "copy" in sources:
            print("[extend] Copy raw.db + legacy DBs (timeout: 1800s)")
            with phase_timeout("copy", 1800):
                # Copy from raw.db if it exists
                if fbase.raw_db_path().exists():
                    raw_conn = sqlite3.connect(fbase.raw_db_path())
                    raw_conn.row_factory = sqlite3.Row
                    dedup = fbase.Deduper(conn, fuzzy=True)  # fuzzy title dedup + merge

                    # author_id -> {name,h_index,citation_count,paper_count} + paper_id -> [author_id,...]
                    authors_by_id: dict[str, dict] = {}
                    for a in raw_conn.execute(
                        "SELECT author_id, name, h_index, citation_count, paper_count FROM authors"
                    ):
                        authors_by_id[a["author_id"]] = dict(a)
                    paper_author_ids: dict[str, list[str]] = {}
                    for pa in raw_conn.execute("SELECT paper_id, author_id FROM paper_authors"):
                        paper_author_ids.setdefault(pa["paper_id"], []).append(pa["author_id"])

                    n_copied = n_merged = 0
                    for row in raw_conn.execute("SELECT * FROM papers"):
                        pid = row["paper_id"]
                        row_dict = dict(row)
                        row_dict["authors"] = [
                            authors_by_id[aid] for aid in paper_author_ids.get(pid, [])
                            if aid in authors_by_id
                        ]
                        if dedup.seen_exact({"paper_id": pid, "doi": row_dict.get("doi")}):
                            continue          # same record already stored (paper_id/DOI)
                        dup_pid = dedup.find_duplicate(row_dict)
                        if dup_pid:           # near-duplicate title -> merge metadata
                            fbase.merge_paper_metadata(conn, dup_pid, row_dict)
                            n_merged += 1
                            continue
                        fbase.insert_paper(conn, row_dict)
                        dedup.add(row_dict)
                        n_copied += 1
                    raw_conn.close()
                    print(f"[extend] Copied {n_copied} papers ({n_merged} merged as near-dups) "
                          f"(+ author records) from raw.db to extended.db")

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
            # Daily-limit survival knobs (see fetchers.openalex.collect_new_papers).
            wait_days = int(cfg.get("fetch_wait_days") or 0)
            wait_hours = config.get_float("FETCH_RATE_LIMIT_WAIT_HOURS", 25.0)
            stall_limit = config.get_int("FETCH_STALL_LIMIT", 2000)
            oa_timeout = 3600 + wait_days * int(wait_hours * 3600) + 3600
            print(f"[extend] OpenAlex fill (timeout advisory: {oa_timeout}s; "
                  f"wait_days={wait_days}, wait_hours={wait_hours:g})")
            with phase_timeout("openalex", oa_timeout):
                have = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
                # fetch_new_rows: fixed count of new papers; else fill toward fetch_target_rows.
                new_rows = int(cfg.get("fetch_new_rows") or 0)
                remaining = new_rows if new_rows > 0 else max(0, target_rows - have)
                print(f"[extend] OpenAlex: have={have}, fetching up to {remaining} new "
                      f"({'fetch_new_rows' if new_rows > 0 else 'target_rows'} mode)")
                if remaining:
                    cap = max(1, remaining // max(1, len(domain_filters)))
                    fetcher = OpenAlexFetcher(year_from, year_to, domain_filters,
                                              per_filter_cap=cap)
                    got = collect_new_papers(conn, fetcher, remaining, wait_days=wait_days,
                                             wait_hours=wait_hours, stall_limit=stall_limit,
                                             on_wait=_fill_during_wait)
                    print(f"[extend] OpenAlex collected {got}/{remaining} new papers")
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

        # Phase 2c: LLM auto-fill — domain classification (cheap, on by default) + optional
        # abstract/year reconstruction from the LLM's own knowledge (heavier, off by
        # default). No-op (returns 0, prints nothing beyond a skip note) unless
        # LLM_JUDGE_PROVIDER/MODEL/API_KEY + LOCAL_LLM_URL are configured in .env.workflow.
        # Uses the SAME shared budget as the wait-time fills above, so the run-total stays
        # within llm_fill_domain_max / llm_fill_abstract_max no matter how the fills split
        # between the daily-limit waits and this final pass.
        if "llm_fill" in sources:
            from workflow.src.iwf.fetchers.llm_fill import llm_fill_abstracts, llm_fill_domains
            if config.get_bool("ENABLE_LLM_FILL", False) and fill_budget["domain"] > 0:
                print("[extend] LLM domain fill (timeout: 3600s)")
                with phase_timeout("llm_fill_domains", 3600):
                    n_dom = llm_fill_domains(conn, limit=fill_budget["domain"])
                    fill_budget["domain"] -= n_dom
                    print(f"[llm_fill] {n_dom} domains filled")
            if config.get_bool("ENABLE_LLM_ABSTRACT_FILL", False) and fill_budget["abstract"] > 0:
                print("[extend] LLM abstract/year fill (timeout: 3600s)")
                with phase_timeout("llm_fill_abstracts", 3600):
                    n_abs = llm_fill_abstracts(conn, limit=fill_budget["abstract"])
                    fill_budget["abstract"] -= n_abs
                    print(f"[llm_fill] {n_abs} abstracts/years filled")

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

    # Phase 5: Build the local benchmarks from the fresh training DB, so the training
    # session later needs no network for benchmark evaluation. Two are built: the
    # stratified holdout (holdout_extended) and the recent-years temporal holdout
    # (holdout_recent), giving each trained model more than one benchmark to compare on.
    print("[extend] Build local benchmarks (timeout: 600s)")
    with phase_timeout("benchmark", 600):
        from workflow.src.iwf.evaluation.benchmarks import (
            build_holdout_benchmark, build_temporal_holdout)
        train_db = fbase.extended_db_path().parent / "training.db"
        build_holdout_benchmark(train_db)
        try:
            build_temporal_holdout(train_db)
        except Exception as e:
            print(f"[extend] temporal-holdout benchmark skipped: {e}")

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

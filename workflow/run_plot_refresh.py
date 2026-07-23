"""Manual plot-refresh utility — regenerates every plot from already-computed data,
without re-running data collection (fetch/bootstrap/extend) or training.

Deliberately placed at the workflow root next to run_pipeline.py / run_all_workflows.py
even though workflow/CLAUDE.md reserves that level for the two full-pipeline entry points:
this is a companion "replay only the plotting/analysis phase" tool, not a training entry
point, and its whole purpose is to be easy to find alongside the other run scripts.

It contains NO plotting logic of its own — it only imports and calls the exact same
functions the real pipeline calls (``data_analysis.analyse``, the four cross-run summary
functions, and ``workflow_results.finalize_workflow`` — the very function
``run_pipeline.py``'s phase 7 calls). So it can never drift from how the workflow itself
produces these figures: if those functions change, this script picks up the change
automatically the next time it runs.

    python -m workflow.run_plot_refresh

Prompts interactively for which database to use for the dataset-analysis plots (every
available database is listed first; press Enter to accept the default, "extended.db", or
type another name/path — an invalid answer re-prompts instead of aborting). Then
regenerates, in order:

  1. workflow/data/analysis/*             (from the chosen database)
  2. workflow/results/<run>/aggregate/*   (for every run folder found under results/)
  3. workflow/results/summary/*           (the cross-run comparison figures)

Needs only workflow/data/ + workflow/results/ (+ workflow/models/ for any run that used a
non-identity encoder) — no fetch/bootstrap/extend, no training.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Same UTF-8 guard as run_pipeline.py: keeps a legacy (cp1252) Windows console from
# aborting a run on the arrow/box glyphs the base parquet IO sometimes prints.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from workflow.src.main_analyse_dataset import resolve_db_path
from workflow.src.iwf import paths, workflow_results
from workflow.src.iwf.config import load_settings

DEFAULT_DB = "extended"


def _list_databases() -> None:
    """Print every database resolve_db_path() can reach, so the prompt below is an
    informed choice instead of a guess."""
    print("Available databases:")
    workflow_dbs = sorted(paths.SQL_DIR.glob("*.db")) if paths.SQL_DIR.is_dir() else []
    if workflow_dbs:
        print("  workflow/data/sql/  (type the bare name, or 'extended' / 'training' / 'raw'):")
        for p in workflow_dbs:
            print(f"    - {p.name}")
    else:
        print(f"  workflow/data/sql/  -- none found ({paths.SQL_DIR})")

    store_dbs = sorted(paths.EPAI_SQL_DIR.glob("*.db")) if paths.EPAI_SQL_DIR.is_dir() else []
    if store_dbs:
        print("  store/data/sql/  (read-only; type the FULL PATH shown to select one):")
        for p in store_dbs:
            print(f"    - {p}")
    print()


def _prompt_for_db() -> Path:
    """Interactive database prompt: Enter accepts DEFAULT_DB ('extended' -> extended.db);
    anything else is resolved with the SAME resolve_db_path() the real
    main_analyse_dataset CLI uses (--db), and a bad answer re-prompts instead of aborting."""
    _list_databases()
    while True:
        raw = input(f"Database to analyse [{DEFAULT_DB}]: ").strip()
        db_spec = raw or DEFAULT_DB
        db_path = resolve_db_path(db_spec)
        if db_path.is_file():
            return db_path
        print(f"  [!] not found: {db_path}  -- try again (or Ctrl+C to abort)\n")


def _prompt_use_benchmark_cache() -> bool:
    """Interactive prompt for the benchmark-embedding cache (see benchmarks.py's
    _embed_benchmark_with_cache): the holdout_extended/holdout_recent texts are frozen, so
    re-embedding them from scratch (~3 min/run on GPU) for every run folder in this refresh
    is pure waste once it has been done once. Enter accepts the fast default (reuse the
    cache when present, transparently falling back to a full recompute per run/benchmark
    if the cache is missing, incomplete, or unreadable); 'n' forces every run to recompute,
    matching the pre-cache behaviour exactly."""
    print("\nBenchmark scoring re-embeds the same frozen benchmark texts (e.g. "
          "holdout_extended) for every run folder — reusing a cached embedding instead of "
          "recomputing it saves ~minutes per run, after the first run pays for it once.")
    while True:
        raw = input("Reuse cached benchmark embeddings when available? [Y/n]: ").strip().lower()
        if raw in ("", "y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  [!] please answer y or n\n")


def _reconstruct_settings(run_config: dict):
    """Rebuild the exact Settings a run was trained with, from its saved run_config.json
    snapshot (written by workflow_results.write_run_config). That snapshot already holds
    every resolved Settings field (Settings.to_dict() == dataclasses.asdict(self), not just
    the overridden subset), so feeding it to load_settings() as overrides reproduces the
    run's configuration exactly, regardless of whatever is currently in config.yaml."""
    overrides = {k: v for k, v in run_config.items() if k != "folder"}
    return load_settings(None, overrides=overrides)


def refresh_dataset_analysis(db_path: Path) -> None:
    from workflow.src.iwf.evaluation.data_analysis import analyse

    out_dir = paths.DATA_DIR / "analysis"
    print(f"\n[refresh] dataset analysis: {db_path} -> {out_dir}")
    result = analyse(db_path, out_dir)
    print(f"[refresh] dataset analysis done: {result['n_papers']:,} papers, "
          f"{result['n_training_complete']:,} training-ready")


def refresh_run_plots() -> list[str]:
    """Re-run finalize_workflow — the exact phase-7 function run_pipeline.py calls — for
    every results/<run>/ folder found. Never retrains: it only reads the already-written
    seed_*/result.json + seed_*/model/model.joblib and regenerates the plots from them."""
    done: list[str] = []
    run_dirs = sorted(d for d in paths.RESULTS_DIR.glob("*_[0-9]*") if d.is_dir())
    if not run_dirs:
        print(f"[refresh] no run folders found under {paths.RESULTS_DIR}")
        return done
    for wf_dir in run_dirs:
        rc_path = wf_dir / "run_config.json"
        if not rc_path.is_file():
            print(f"[refresh] {wf_dir.name}: no run_config.json, skipping")
            continue
        try:
            rc = json.loads(rc_path.read_text(encoding="utf-8"))
            wf_type = rc.get("workflow_type")
            s = _reconstruct_settings(rc)
            print(f"\n[refresh] === {wf_dir.name} ({wf_type}) ===")
            workflow_results.finalize_workflow(wf_type, s, wf_dir)
            done.append(wf_dir.name)
        except Exception as e:
            print(f"[refresh] {wf_dir.name}: FAILED ({e})", file=sys.stderr)
    return done


def refresh_summaries() -> None:
    """The cross-run summary figures — the same calls run_all_workflows.py makes after
    its training queue finishes. Each is independently best-effort, matching the
    original's own try/except-per-figure resilience."""
    print(f"\n[refresh] cross-run summary plots -> {paths.RESULTS_DIR / 'summary'}")
    workflow_results.cross_workflow_comparison()
    try:
        from workflow.src.iwf.evaluation.baseline_compare import baseline_leaderboard
        baseline_leaderboard(paths.RESULTS_DIR)
    except Exception as e:
        print(f"[refresh] baseline leaderboard skipped ({e})", file=sys.stderr)
    try:
        from workflow.src.iwf.evaluation.nf_analysis import nf_ablation
        nf_ablation(paths.RESULTS_DIR)
    except Exception as e:
        print(f"[refresh] nf ablation skipped ({e})", file=sys.stderr)
    try:
        from workflow.src.iwf.evaluation.benchmarks import benchmark_overview
        benchmark_overview(paths.RESULTS_DIR)
    except Exception as e:
        print(f"[refresh] benchmark overview skipped ({e})", file=sys.stderr)
    try:
        from workflow.src.iwf.evaluation.cross_domain import cross_domain_matrix
        cross_domain_matrix(paths.RESULTS_DIR)
    except Exception as e:
        print(f"[refresh] cross-domain matrix skipped ({e})", file=sys.stderr)
    try:
        from workflow.src.iwf.evaluation.embedder_summary import generate_embedder_summary
        generate_embedder_summary(paths.RESULTS_DIR)
    except Exception as e:
        print(f"[refresh] embedder-performance summary skipped ({e})", file=sys.stderr)


def main() -> int:
    print("=" * 70)
    print("run_plot_refresh — regenerate every plot from existing data/ + results/")
    print("(no data collection, no training)")
    print("=" * 70)

    try:
        db_path = _prompt_for_db()
        use_cache = _prompt_use_benchmark_cache()
    except (KeyboardInterrupt, EOFError):
        print("\n[refresh] aborted.")
        return 1

    # See benchmarks._force_benchmark_embed(): same FORCE_* env-var convention as
    # FORCE_FETCH/FORCE_EMBED in .env.workflow, just set for this process instead of read
    # from the file, so a 'n' answer here forces every run to recompute (old behaviour).
    os.environ["FORCE_BENCHMARK_EMBED"] = "false" if use_cache else "true"

    refresh_dataset_analysis(db_path)
    done = refresh_run_plots()
    refresh_summaries()

    print(f"\n[refresh] complete. {len(done)} run folder(s) refreshed: "
          f"{', '.join(done) or '(none)'}")
    print(f"[refresh] see {paths.DATA_DIR / 'analysis'} and {paths.RESULTS_DIR / 'summary'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

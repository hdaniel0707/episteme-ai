"""Multi-training runner — executes every entry of the ``runs:`` queue in config.yaml.

    python -m workflow.run_all_workflows                    # workflow/config.yaml
    python -m workflow.run_all_workflows --config other.yaml

ALWAYS wipes workflow/results/ first (after a short countdown), so every queue session
starts fresh; a single `python -m workflow.run_pipeline` never wipes anything.

Each ``runs:`` entry names one training setup: a ``workflow`` (or a list of workflows to
run with the same parameters) plus any overrides of the top-level config.yaml keys. Every
run gets its own auto-incremented folder (``emb_clf_1``, ``emb_clf_2``, …) and log file.

Expensive shared phases are computed once and reused between runs whose relevant
parameters match: fetch/prepare are keyed on the data+target+split configuration, and
embed/encode/reduce additionally on the embedder+encoder configuration. The label phase
is always re-run (it is cheap and depends on the per-run class configuration).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("UMAP_OPEN_BROWSER", "false")   # never pop browsers mid-benchmark

from workflow import run_pipeline                              # noqa: E402
from workflow.src.iwf import paths, workflow_results          # noqa: E402
from workflow.src.iwf.config import (                          # noqa: E402
    WORKFLOWS, Settings, is_text_workflow, load_config_yaml, load_settings,
)
from workflow.src.iwf.progress import MultiWorkflowProgress    # noqa: E402

COUNTDOWN_SECONDS = 3


# ===================================================================== runs queue

def expand_runs(cfg: dict, config_path: str | None) -> list[tuple[str, str, Settings]]:
    """Expand config.yaml's ``runs:`` into (workflow_type, name, Settings) triples."""
    runs = cfg.get("runs")
    if not runs:
        raise ValueError(
            "config.yaml has no (or an empty) 'runs:' list — add at least one entry "
            "(see the commented TEMPLATE block in workflow/config.yaml)."
        )
    expanded: list[tuple[str, str, Settings]] = []
    for i, entry in enumerate(runs):
        if not isinstance(entry, dict):
            raise ValueError(f"runs[{i}] must be a mapping (got {type(entry).__name__}).")
        name = str(entry.get("name") or f"run{i + 1}")
        wfs = entry.get("workflow") or cfg.get("workflow") or "emb_clf"
        if isinstance(wfs, str):
            wfs = [wfs]
        overrides = {k: v for k, v in entry.items() if k not in ("name", "workflow")}
        for wf in wfs:
            wf = str(wf).lower()
            if wf not in WORKFLOWS:
                raise ValueError(
                    f"runs entry {name!r}: workflow={wf!r} is invalid; "
                    f"choose from {list(WORKFLOWS)}."
                )
            s = load_settings(config_path, overrides={**overrides, "workflow": wf})
            s.validate()
            expanded.append((wf, name, s))
    return expanded


def _data_key(s: Settings) -> tuple:
    """Everything that determines the fetch/prepare outputs (phases 0-1)."""
    return (s.dataset_id, s.year_from, s.year_to, s.max_papers, s.target_kind,
            s.target_min_refset, s.split_seed, tuple(s.split_ratios),
            tuple(s.metadata_fields), s.domain_holdout)


def _embed_key(s: Settings, wf: str) -> tuple | None:
    """Everything that additionally determines embed/encode/reduce (phases 2-4).
    None for text workflows, which have no embed/encode phases at all."""
    if is_text_workflow(wf):
        return None
    return (*_data_key(s), tuple(s.embedders), s.embed_for_encoder, s.encoder_kind,
            tuple(s.dim_grid), s.dim_recon_target, s.dim_tolerance,
            s.encoder_max_epochs, s.encoder_patience, s.activation)


# ===================================================================== orchestrator

def _reset_results() -> None:
    """Wipe workflow/results so every run_all_workflows session starts fresh (a session's
    figures/aggregates must never mix with a previous session's folders)."""
    print(f"[run_all] wiping {paths.RESULTS_DIR} for a fresh session "
          "(single `run_pipeline` runs never wipe).")
    for k in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"[run_all]   wiping results in {k}... (Ctrl+C to abort)", flush=True)
        time.sleep(1)
    if paths.RESULTS_DIR.exists():
        shutil.rmtree(paths.RESULTS_DIR, ignore_errors=True)
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run every training in config.yaml's runs queue.")
    ap.add_argument("--config", default=None,
                    help="alternate config.yaml (default: workflow/config.yaml)")
    args = ap.parse_args()

    cfg = load_config_yaml(args.config)
    all_runs = expand_runs(cfg, args.config)
    n_total = len(all_runs)

    _reset_results()   # ALWAYS start from an empty results/ — essential fresh restart
    print(f"[run_all] {n_total} runs from {len(cfg.get('runs', []))} config entries.")
    if args.config:
        print(f"[run_all] config: {args.config}")
    print()

    run_labels = [f"{wf}:{name}" for wf, name, _ in all_runs]
    progress = MultiWorkflowProgress(
        run_labels,
        labels={rl: rl for rl in run_labels},
    ).start()

    data_done: set[tuple] = set()
    embed_done: set[tuple] = set()
    failures: dict[str, str] = {}
    logs: dict[str, Path] = {}
    wf_dirs: dict[str, Path] = {}

    try:
        for i, (wf_type, name, s) in enumerate(all_runs):
            run_id = run_labels[i]
            wf_dir = workflow_results.next_workflow_dir(wf_type)
            wf_dir.mkdir(parents=True, exist_ok=True)
            wf_dirs[run_id] = wf_dir
            log_file = paths.RESULTS_DIR / f"{wf_dir.name}.log"
            logs[run_id] = log_file

            dk, ek = _data_key(s), _embed_key(s, wf_type)
            skip: set[int] = set()
            if dk in data_done:
                skip |= {0, 1}
            if ek is not None and ek in embed_done:
                skip |= {2, 3, 4}

            try:
                with open(log_file, "w", encoding="utf-8") as lf:
                    old_out, old_err = sys.stdout, sys.stderr
                    sys.stdout = lf
                    sys.stderr = lf
                    try:
                        run_pipeline.run_workflow(
                            wf_type, s, progress=progress, shared=skip,
                            wf_dir=wf_dir, progress_id=run_id,
                        )
                    finally:
                        sys.stdout = old_out
                        sys.stderr = old_err

                data_done.add(dk)
                if ek is not None:
                    embed_done.add(ek)

            except SystemExit as e:
                failures[run_id] = f"phase exited with code {e.code}"
                progress.error(run_id, failures[run_id])
            except Exception as e:
                failures[run_id] = f"{type(e).__name__}: {e}"
                progress.error(run_id, str(e))
                print(f"[run_all] {run_id} failed: {e}", file=sys.stderr)
                traceback.print_exc()
    finally:
        progress.stop()

    comparison = workflow_results.cross_workflow_comparison()

    # cross-run baseline leaderboard + novelty/feasibility ablation (both best-effort)
    try:
        from workflow.src.iwf.evaluation.baseline_compare import baseline_leaderboard
        baseline_leaderboard(paths.RESULTS_DIR)
    except Exception as e:
        print(f"[run_all] baseline leaderboard skipped ({e})", file=sys.stderr)
    try:
        from workflow.src.iwf.evaluation.nf_analysis import nf_ablation
        nf_ablation(paths.RESULTS_DIR)
    except Exception as e:
        print(f"[run_all] nf ablation figure skipped ({e})", file=sys.stderr)
    try:
        from workflow.src.iwf.evaluation.benchmarks import benchmark_overview
        benchmark_overview(paths.RESULTS_DIR)
    except Exception as e:
        print(f"[run_all] benchmark overview skipped ({e})", file=sys.stderr)
    try:
        from workflow.src.iwf.evaluation.cross_domain import cross_domain_matrix
        cross_domain_matrix(paths.RESULTS_DIR)
    except Exception as e:
        print(f"[run_all] cross-domain matrix skipped ({e})", file=sys.stderr)
    try:
        from workflow.src.iwf.evaluation.embedder_summary import generate_embedder_summary
        generate_embedder_summary(paths.RESULTS_DIR)
    except Exception as e:
        print(f"[run_all] embedder-performance summary skipped ({e})", file=sys.stderr)

    print(f"\n[run_all] summary ({n_total} runs):")
    print(f"  {'FOLDER':<25s} {'CONFIG':<25s} {'STATUS':<10s} {'METRIC'}")
    print(f"  {'-' * 25} {'-' * 25} {'-' * 10} {'-' * 20}")
    for run_id, (wf_type, name, _) in zip(run_labels, all_runs):
        d = wf_dirs.get(run_id)
        folder = d.name if d else "?"
        c = comparison.get(folder) if d else None
        if run_id in failures:
            print(f"  {folder:<25s} {name:<25s} {'FAILED':<10s} {failures[run_id][:40]}")
        elif c:
            primary = c.get("primary_metric", "")
            best = c.get(f"best_{primary}")
            mean = c.get(f"mean_{primary}")
            val = f"{primary}: mean={mean:.4f} best={best:.4f}" if best is not None else ""
            print(f"  {folder:<25s} {name:<25s} {'OK':<10s} {val}")
        else:
            print(f"  {folder:<25s} {name:<25s} {'OK':<10s} (no aggregate)")

    if failures:
        print(f"\n[run_all] {len(failures)}/{n_total} run(s) failed.", file=sys.stderr)
        for run_id in sorted(failures):
            log = logs.get(run_id)
            if log and log.is_file():
                print(f"  {run_id} -> {log}")
        return 1
    print(f"\n[run_all] all {n_total} runs completed.")
    print(f"[run_all] comparison -> {paths.RESULTS_DIR / 'summary' / 'cross_workflow_comparison.png'}")
    print(f"[run_all] per-run logs -> {paths.RESULTS_DIR}/*.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())

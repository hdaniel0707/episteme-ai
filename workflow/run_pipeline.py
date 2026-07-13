"""Single-workflow orchestrator — runs the workflow configured in ``config.yaml``.

A run is organised around a workflow type (the pipeline topology). Each topology runs the
fixed phase list in ``config.WORKFLOWS`` and writes to its own auto-incremented folder
``workflow/results/<wf_type>_N/``.

    python -m workflow.run_pipeline                  # run `workflow:` from config.yaml
    python -m workflow.run_pipeline --from 2 --to 5  # only a phase range (reuse outputs)
    python -m workflow.run_pipeline --force          # ignore skip-if-output-exists shortcuts

Phases (ids used by --from/--to):
    0 fetch | 1 prepare | 2 embed | 3 encoder | 4 reduce | 5 label | 6 train | 7 finalize

The tfm_* workflows skip 2-4 entirely (they fine-tune on raw text and never touch the
encoder artifact). ``workflow: all`` runs every topology once, computing the shared
phases (fetch/prepare/embed/encode/reduce) once and reusing them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The base epai parquet IO prints arrow glyphs; force UTF-8 so a legacy (cp1252) Windows
# console never aborts a run on those prints (override of a base-repo behaviour).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from workflow.src import (
    main_embed, main_encoder, main_fetch, main_label, main_prepare, main_reduce,
    main_train, main_train_tfm,
)
from workflow.src.iwf import paths, workflow_results
from workflow.src.iwf.config import (
    WORKFLOWS, WORKFLOW_TYPES, Settings, get_bool, is_text_workflow, load_settings,
)

PHASE_NAMES = {
    0: "fetch", 1: "prepare", 2: "embed", 3: "encoder", 4: "reduce", 5: "label",
    6: "train", 7: "finalize",
}

# Phases whose output is shared across workflows in an ``all`` run / benchmark (computed
# once, reused). The label phase (5) is deliberately NOT shared: it is cheap and depends
# on the per-run class configuration, so skipping it could leave stale class labels.
SHARED_PHASES = {0, 1, 2, 3, 4}


def _run_phase(num: int, wf: str, s: Settings, force: bool, prog_cb,
               wf_dir: Path | None) -> None:
    """Execute one phase for workflow ``wf``."""
    if num == 0:
        out = paths.raw_parquet(s.dataset_id)
        if out.is_file() and not force and not get_bool("FORCE_FETCH", False):
            print(f"[run] skip phase 0 fetch: reusing {out.name} "
                  "(FORCE_FETCH=true / --force to rebuild)")
            return
        main_fetch.main(s)
    elif num == 1:
        main_prepare.main(s)
    elif num == 2:
        main_embed.main(s)
    elif num == 3:
        main_encoder.main(s)
    elif num == 4:
        main_reduce.main(s)
    elif num == 5:
        main_label.main(s)
    elif num == 6:
        if is_text_workflow(wf):
            main_train_tfm.main(s, workflow_type=wf, progress=prog_cb, wf_dir=wf_dir)
        else:
            main_train.main(s, workflow_type=wf, wf_dir=wf_dir)
    elif num == 7:
        workflow_results.finalize_workflow(wf, s, wf_dir=wf_dir)
    else:
        raise ValueError(f"Unknown phase number {num}.")


def run_workflow(wf_type: str, settings: Settings, *, start: int = 0, end: int = 7,
                 force: bool = False, progress=None, shared: set | None = None,
                 wf_dir: Path | None = None, progress_id: str | None = None) -> Path:
    """Run a single workflow topology end-to-end. Returns the output directory path.

    ``shared`` is a mutable set of already-computed SHARED_PHASES (pass the same set to
    consecutive calls to reuse fetch/prepare/embed/encode outputs). ``progress_id`` keys
    the live progress display when several runs of the same type are active.
    """
    wf_type = wf_type.lower()
    if wf_type not in WORKFLOWS:
        raise ValueError(f"Unknown workflow type {wf_type!r}; choose from {list(WORKFLOWS)}.")
    pid = progress_id or wf_type
    spec = WORKFLOWS[wf_type]
    phases = spec["phases"]

    # Per-workflow copy so an `all` run / queued runs can never leak state between
    # topologies; the leakage bypass is a property of the workflow type, not a user knob.
    settings = settings.copy()
    settings.workflow_type = wf_type
    settings.bypass_leakage_guard = spec["features"] == "all_meta"

    if wf_dir is None:
        wf_dir = workflow_results.next_workflow_dir(wf_type)
        wf_dir.mkdir(parents=True, exist_ok=True)

    active = [p for p in phases if start <= p <= end]
    print(f"\n{'#' * 70}\n# WORKFLOW {wf_type}  -> {wf_dir.name}  "
          f"phases={[PHASE_NAMES[p] for p in active]}\n{'#' * 70}")
    if progress is not None:
        progress.update(pid, phase=(active[0] if active else 0, len(phases)), percent=0.0)

    n = len(active)
    for i, num in enumerate(active):
        if shared is not None and num in SHARED_PHASES:
            if num in shared:
                print(f"[run] skip phase {num} {PHASE_NAMES[num]}: shared output "
                      "already computed.")
                _advance(progress, pid, num, phases, i, n)
                continue
            shared.add(num)
        print(f"\n{'=' * 70}\n  PHASE {num}: {PHASE_NAMES[num]}  ({wf_dir.name})\n{'=' * 70}")
        prog_cb = _make_tfm_callback(progress, pid, num, phases) if num == 6 else None
        _run_phase(num, wf_type, settings, force, prog_cb, wf_dir)
        _advance(progress, pid, num, phases, i, n)

    if progress is not None:
        progress.done(pid, metric=_read_primary(wf_dir))

    return wf_dir


def _advance(progress, pid, num, phases, i, n) -> None:
    if progress is None:
        return
    progress.update(pid, phase=(num, max(phases)), percent=100.0 * (i + 1) / max(1, n))


def _make_tfm_callback(progress, pid, num, phases):
    if progress is None:
        return None

    def cb(info: dict) -> None:
        seed = info.get("seed", 0)
        n_runs = info.get("n_runs", 0)
        ep, eps = info.get("epoch", 0), info.get("epochs", 1)
        seed_frac = (seed + ep / max(1, eps)) / max(1, n_runs)
        progress.update(pid, phase=(num, max(phases)), percent=100.0 * seed_frac,
                        seed=(seed + 1, n_runs),
                        metric=("val_metric", float(info.get("val_metric", 0.0))))

    return cb


def _read_primary(wf_dir: Path):
    """Read the primary metric from a workflow's aggregate summary, if present."""
    agg = wf_dir / "aggregate" / "metrics_summary.json"
    if not agg.is_file():
        return None
    import json
    data = json.loads(agg.read_text(encoding="utf-8"))
    primary = data.get("primary_metric", "spearman")
    best = data.get(f"best_{primary}")
    return (f"best_{primary}", float(best)) if best is not None else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None,
                    help="alternate config.yaml (default: workflow/config.yaml)")
    ap.add_argument("--from", dest="start", type=int, default=0,
                    help="first phase id to run (0 fetch .. 7 finalize)")
    ap.add_argument("--to", dest="end", type=int, default=7,
                    help="last phase id to run")
    ap.add_argument("--force", action="store_true",
                    help="ignore skip-if-output-exists shortcuts")
    args = ap.parse_args()

    s = load_settings(args.config)
    s.validate()

    wf = s.workflow_type.lower()
    workflows = list(WORKFLOW_TYPES) if wf == "all" else [wf]
    print(f"[run] workflow={wf}  -> {workflows}  target={s.target_kind}  "
          f"encoder={s.encoder_kind}")

    shared: set | None = set() if wf == "all" else None
    wf_dirs: list[Path] = []
    for w in workflows:
        d = run_workflow(w, s, start=args.start, end=args.end, force=args.force,
                         shared=shared)
        wf_dirs.append(d)

    workflow_results.cross_workflow_comparison()
    print(f"\n[run] pipeline finished. Results in: {', '.join(d.name for d in wf_dirs)}")


if __name__ == "__main__":
    main()

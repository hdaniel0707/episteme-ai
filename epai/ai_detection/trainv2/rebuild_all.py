#!/usr/bin/env python3
r"""Rebuild every trainv2 result and calibrated config, in dependency order.

Runs the steps from the "Recommended full-rebuild order" in this folder's
README.md. The steps are independent, so they run concurrently, bounded by
--jobs and by --heavy-jobs (the memory / compute heavy ones).

Each step runs as ``<this interpreter> <script> ...`` from the repo root -- so
launch this orchestrator with ``uv run python`` and every step inherits that
venv with no further ``uv`` invocation of its own.

    uv run python epai/ai_detection/trainv2/rebuild_all.py
    uv run python epai/ai_detection/trainv2/rebuild_all.py --jobs 1      # serial, live output
    uv run python epai/ai_detection/trainv2/rebuild_all.py --dry-run
    uv run python epai/ai_detection/trainv2/rebuild_all.py --only run_lodo_performance
    uv run python epai/ai_detection/trainv2/rebuild_all.py --skip-youden

With --jobs > 1 each step's output is captured to
``results/trainv2/_rebuild_logs/<step>.log`` and only the tail is echoed on
failure; with --jobs 1 the steps stream straight to the console. A step whose
dependency did not succeed is skipped. Exit status is non-zero if any step
failed or was skipped.

The MG-Reph family is NOT part of this rebuild -- every step here uses the
default splits (``splits.DEFAULT``), which excludes it. Run MG-Reph experiments
separately (``--splits mg_reph`` / ``mg_reph_sci`` / ``mg_reph_gb``; see
``split_manugram_reph.py``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
DIR = "epai/ai_detection/trainv2"
PQ = "data/parquet"
LOG_DIR = BASE / "results" / "trainv2" / "_rebuild_logs"

CALIB = [
    "calibrate_judge_baseline",
    "calibrate_binox_baseline",
    "calibrate_fastdetect_baseline",
    "calibrate_detector_baseline",
]


@dataclass
class Task:
    name: str
    argv: list[str]
    after: set[str] = field(default_factory=set)
    heavy: bool = False  # RAM- or compute-hungry: throttled by --heavy-jobs
    youden: bool = False  # optional per-domain Youden step (--skip-youden)


TASKS: list[Task] = [
    # A -- baseline calibration -> cfg/ai_detection/trainv2/*_baseline_calibrated.yaml
    *[Task(n, [f"{DIR}/{n}.py", "--corpora", "all"]) for n in CALIB],
    # C / D / E -- zero-shot analyses
    Task("run_embedding_effectiveness",
         [f"{DIR}/run_embedding_effectiveness.py"], heavy=True),
    Task("run_zeroshot_performance", [f"{DIR}/run_zeroshot_performance.py"]),
    Task("run_best_zeroshot_performance", [f"{DIR}/run_best_zeroshot_performance.py"]),
    Task("run_judge_by_generator", [f"{DIR}/run_judge_by_generator.py"]),
    # F -- trained detectors
    Task("run_in_domain_performance",
         [f"{DIR}/run_in_domain_performance.py"], heavy=True),
    Task("run_lodo_performance", [f"{DIR}/run_lodo_performance.py"], heavy=True),
    # B -- optional per-domain Youden calibration
    Task("run_judge_youden_calibration",
         [f"{DIR}/run_judge_youden_calibration.py", f"{PQ}/manugram_sci.parquet"],
         youden=True),
    Task("run_binox_youden_calibration",
         [f"{DIR}/run_binox_youden_calibration.py", f"{PQ}/manugram_sci.parquet"],
         youden=True),
    Task("run_fastdetect_youden_calibration",
         [f"{DIR}/run_fastdetect_youden_calibration.py", f"{PQ}/manugram_sci.parquet"],
         youden=True),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cmd(task: Task) -> list[str]:
    return [sys.executable, *task.argv]


def _tail(path: Path, n: int = 40) -> None:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return
    print(f"      -- last {min(n, len(lines))} lines of {path.relative_to(BASE)} --")
    for ln in lines[-n:]:
        print(f"      {ln}")


def _launch(task: Task, capture: bool):
    cmd = _cmd(task)
    start = time.time()
    if capture:
        fh = open(LOG_DIR / f"{task.name}.log", "w")
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=BASE, stdout=fh, stderr=subprocess.STDOUT)
        return proc, fh, start
    print(f"\n{'=' * 72}\n$ {' '.join(cmd)}\n{'=' * 72}")
    return subprocess.Popen(cmd, cwd=BASE), None, start


def _waves(tasks: list[Task]) -> list[list[Task]]:
    """Topological levels -- an upper bound on parallelism (--heavy-jobs may
    still serialise the heavy steps within a wave)."""
    names = {t.name for t in tasks}
    done: set[str] = set()
    remaining = list(tasks)
    waves: list[list[Task]] = []
    while remaining:
        w = [t for t in remaining if (t.after & names) <= done]
        waves.append(w)
        done |= {t.name for t in w}
        remaining = [t for t in remaining if t not in w]
    return waves


def _dry_run(tasks: list[Task], jobs: int, heavy_jobs: int) -> int:
    print(f"plan -- {len(tasks)} steps, jobs={jobs}, heavy-jobs={heavy_jobs}")
    print(f"interpreter: {sys.executable}\n")
    for i, wave in enumerate(_waves(tasks), 1):
        after = " (after the previous wave)" if i > 1 else ""
        print(f"wave {i}  (up to {jobs} parallel){after}:")
        for t in wave:
            tag = " [heavy]" if t.heavy else ""
            print(f"    {t.name:34}{tag:9}  $ {' '.join(_cmd(t))}")
    print("\nheavy steps run at most --heavy-jobs at a time regardless of wave.")
    return 0


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------


def run(tasks: list[Task], jobs: int, heavy_jobs: int) -> int:
    bn = {t.name: t for t in tasks}
    capture = jobs > 1
    if capture:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    pending = list(tasks)
    running: dict[str, tuple] = {}  # name -> (proc, log_fh|None, start)
    result: dict[str, str] = {}  # name -> "ok" | "fail" | "skip"
    t0 = time.time()

    print(f"rebuilding {len(tasks)} steps  (jobs={jobs}, heavy-jobs={heavy_jobs}, "
          f"{'captured logs' if capture else 'live output'})\n")

    try:
        while pending or running:
            changed = False

            for name in list(running):
                proc, fh, start = running[name]
                if proc.poll() is None:
                    continue
                rc = proc.returncode
                if fh:
                    fh.close()
                del running[name]
                result[name] = "ok" if rc == 0 else "fail"
                dur = f"{time.time() - start:.0f}s"
                if rc == 0:
                    print(f"  ✓ {name}  ({dur})")
                else:
                    print(f"  ✗ {name}  ({dur}, exit {rc})")
                    if capture:
                        _tail(LOG_DIR / f"{name}.log")
                changed = True

            ok = {n for n, r in result.items() if r == "ok"}
            bad = {n for n, r in result.items() if r in ("fail", "skip")}
            for t in list(pending):
                blockers = t.after & bad
                if blockers:
                    pending.remove(t)
                    result[t.name] = "skip"
                    print(f"  – {t.name}  (skipped: needs {', '.join(sorted(blockers))})")
                    changed = True

            for t in list(pending):
                if len(running) >= jobs:
                    break
                if not t.after <= ok:
                    continue
                if t.heavy and sum(bn[n].heavy for n in running) >= heavy_jobs:
                    continue
                pending.remove(t)
                running[t.name] = _launch(t, capture)
                print(f"  ▶ {t.name}{'  [heavy]' if t.heavy else ''}")
                changed = True

            if not changed:
                if pending and not running:
                    for t in pending:
                        result[t.name] = "skip"
                    print("  ! unschedulable: " + ", ".join(t.name for t in pending))
                    break
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\ninterrupted -- terminating running steps")
        for _, (proc, fh, _s) in running.items():
            proc.terminate()
            if fh:
                fh.close()
        return 130

    failed = [n for n, r in result.items() if r == "fail"]
    skipped = [n for n, r in result.items() if r == "skip"]
    n_ok = sum(r == "ok" for r in result.values())
    print(f"\n{'=' * 72}")
    print(f"done in {time.time() - t0:.0f}s -- {n_ok} ok, {len(failed)} failed, "
          f"{len(skipped)} skipped")
    if failed:
        print(f"  failed : {', '.join(failed)}")
    if skipped:
        print(f"  skipped: {', '.join(skipped)}")
    if capture:
        print(f"  logs   : {LOG_DIR.relative_to(BASE)}/")
    return 1 if (failed or skipped) else 0


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-j", "--jobs", type=int, default=min(4, os.cpu_count() or 1),
                    help="max steps at once (default: %(default)s; 1 = serial, live output)")
    ap.add_argument("--heavy-jobs", type=int, default=1,
                    help="max heavy steps at once -- run_embedding_effectiveness "
                         "(reads full parquets), run_in_domain_performance, "
                         "run_lodo_performance (default: 1)")
    ap.add_argument("--only", nargs="+", metavar="STEP",
                    help="run only these steps, plus any they depend on")
    ap.add_argument("--skip", nargs="+", metavar="STEP", default=[],
                    help="exclude these steps")
    ap.add_argument("--skip-youden", action="store_true",
                    help="drop the 3 optional per-domain Youden steps")
    ap.add_argument("--list", action="store_true", help="list step names and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args()

    by_name = {t.name: t for t in TASKS}

    if args.list:
        for t in TASKS:
            tags = " ".join(x for x, on in (("heavy", t.heavy), ("youden", t.youden)) if on)
            print(f"  {t.name:34} {tags}")
        return 0

    for n in (args.only or []) + args.skip:
        if n not in by_name:
            raise SystemExit(f"unknown step {n!r} -- run with --list to see them")
    if args.jobs < 1 or args.heavy_jobs < 1:
        raise SystemExit("--jobs and --heavy-jobs must be >= 1")

    if args.only:
        chosen: set[str] = set()
        stack = list(args.only)
        while stack:
            n = stack.pop()
            if n in chosen:
                continue
            chosen.add(n)
            stack += by_name[n].after
    else:
        chosen = {t.name for t in TASKS}
        if args.skip_youden:
            chosen -= {t.name for t in TASKS if t.youden}
    chosen -= set(args.skip)
    if not chosen:
        raise SystemExit("nothing selected")

    selected = [t for t in TASKS if t.name in chosen]
    missing = {d for t in selected for d in t.after if d not in chosen}
    if missing:
        raise SystemExit(
            "selected steps depend on unselected steps: " + ", ".join(sorted(missing))
            + "\nadd them back, or drop the step that needs them"
        )

    for t in selected:
        if not (BASE / t.argv[0]).exists():
            raise SystemExit(f"missing script: {t.argv[0]}")
        for a in t.argv[1:]:
            if a.endswith(".parquet") and not (BASE / a).exists():
                raise SystemExit(f"missing parquet: {a}")

    if args.dry_run:
        return _dry_run(selected, args.jobs, args.heavy_jobs)
    return run(selected, args.jobs, args.heavy_jobs)


if __name__ == "__main__":
    sys.exit(main())

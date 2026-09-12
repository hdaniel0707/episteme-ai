#!/usr/bin/env python3
r"""Run the A -> D -> F trainv2 steps back to back, archived under one folder.

Steps, strictly in order:

    A. Baseline calibration    calibrate_judge_baseline.py      --corpora all
                                calibrate_binox_baseline.py      --corpora all
                                calibrate_fastdetect_baseline.py --corpora all
                                calibrate_detector_baseline.py   --corpora all
    D. Zero-shot performance   run_zeroshot_performance.py
                                run_best_zeroshot_performance.py
    F. Trained detection       run_in_domain_performance.py
                                run_lodo_performance.py

Every step's own output flags (``-o`` / ``--csv`` / ``--fig-dir`` for the
calibration scripts, ``--out-dir`` for the rest) are redirected into
``results/trainv2/train_pipeline_<YYYYMMDD_HHMM>/<step>/`` -- nothing lands in
a script's usual default location (including the live
``cfg/ai_detection/trainv2/*_baseline_calibrated.yaml``), so one run of this
pipeline never overwrites another's, or the shared config.

Runs sequentially, each step inheriting stdout/stderr live, and stops at the
first failure unless ``--keep-going`` is given.

``--no-training`` drops the F steps (``run_in_domain_performance.py`` /
``run_lodo_performance.py``) -- by far the slowest part (each trains several
detectors over N seeds) and the least useful ones to re-run just to check that
A/D still work, e.g. after a splits.py / plots.py change.

    uv run python epai/ai_detection/trainv2/run_train_pipeline.py
    uv run python epai/ai_detection/trainv2/run_train_pipeline.py --no-training
    uv run python epai/ai_detection/trainv2/run_train_pipeline.py --keep-going
    uv run python epai/ai_detection/trainv2/run_train_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
DIR = "epai/ai_detection/trainv2"
RESULTS = BASE / "results" / "trainv2"


@dataclass
class Step:
    name: str
    script: str
    args: list[str] = field(default_factory=list)
    #: output flag -> filename under the step's pipeline subfolder ("" = the
    #: subfolder itself, for flags that take a directory).
    out_flags: dict[str, str] = field(default_factory=dict)
    #: F-group trained-detector steps -- slow (multi-seed training), dropped
    #: by --no-training.
    training: bool = False


STEPS: list[Step] = [
    # A -- baseline calibration
    Step("calibrate_judge_baseline", "calibrate_judge_baseline.py", ["--corpora", "all"],
         {"-o": "judge_baseline_calibrated.yaml", "--csv": "thresholds.csv", "--fig-dir": ""}),
    Step("calibrate_binox_baseline", "calibrate_binox_baseline.py", ["--corpora", "all"],
         {"-o": "binox_baseline_calibrated.yaml", "--csv": "thresholds.csv", "--fig-dir": ""}),
    Step("calibrate_fastdetect_baseline", "calibrate_fastdetect_baseline.py", ["--corpora", "all"],
         {"-o": "fastdetect_baseline_calibrated.yaml", "--csv": "thresholds.csv", "--fig-dir": ""}),
    Step("calibrate_detector_baseline", "calibrate_detector_baseline.py", ["--corpora", "all"],
         {"-o": "detector_baseline_calibrated.yaml", "--csv": "thresholds.csv", "--fig-dir": ""}),
    # D -- zero-shot performance
    Step("run_zeroshot_performance", "run_zeroshot_performance.py", [], {"--out-dir": ""}),
    Step("run_best_zeroshot_performance", "run_best_zeroshot_performance.py", [], {"--out-dir": ""}),
    # F -- trained detection
    Step("run_in_domain_performance", "run_in_domain_performance.py", [], {"--out-dir": ""}, training=True),
    Step("run_lodo_performance", "run_lodo_performance.py", [], {"--out-dir": ""}, training=True),
]


def _cmd(step: Step, pipeline_dir: Path) -> list[str]:
    step_dir = pipeline_dir / step.name
    argv = [sys.executable, f"{DIR}/{step.script}", *step.args]
    for flag, filename in step.out_flags.items():
        argv += [flag, str(step_dir / filename if filename else step_dir)]
    return argv


def run(steps: list[Step], pipeline_dir: Path, keep_going: bool) -> int:
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    print(f"train pipeline -> {pipeline_dir.relative_to(BASE)}\n")

    ran: list[str] = []
    failed: list[str] = []
    t0 = time.time()
    for i, step in enumerate(steps, 1):
        (pipeline_dir / step.name).mkdir(parents=True, exist_ok=True)
        cmd = _cmd(step, pipeline_dir)
        print(f"{'=' * 72}\n[{i}/{len(steps)}] {step.name}\n$ {' '.join(cmd)}\n{'=' * 72}")
        start = time.time()
        rc = subprocess.run(cmd, cwd=BASE).returncode
        ran.append(step.name)
        dur = f"{time.time() - start:.0f}s"
        if rc == 0:
            print(f"  ✓ {step.name}  ({dur})\n")
        else:
            print(f"  ✗ {step.name}  ({dur}, exit {rc})\n")
            failed.append(step.name)
            if not keep_going:
                break

    skipped = [s.name for s in steps if s.name not in ran]
    print(f"{'=' * 72}")
    print(f"done in {time.time() - t0:.0f}s -- {len(ran) - len(failed)} ok, "
          f"{len(failed)} failed, {len(skipped)} skipped")
    if failed:
        print(f"  failed : {', '.join(failed)}")
    if skipped:
        print(f"  skipped: {', '.join(skipped)}")
    print(f"  results: {pipeline_dir.relative_to(BASE)}/")
    return 1 if (failed or skipped) else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--keep-going", action="store_true",
                     help="run every step even after a failure (default: stop at the first)")
    ap.add_argument("--no-training", action="store_true",
                     help="drop the F steps (run_in_domain_performance.py / "
                          "run_lodo_performance.py) -- the slow, multi-seed "
                          "trained-detector steps -- to just verify A/D quickly")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = ap.parse_args()

    steps = [s for s in STEPS if not (args.no_training and s.training)]
    pipeline_dir = RESULTS / f"train_pipeline_{datetime.now():%Y%m%d_%H%M}"

    if args.dry_run:
        print(f"train pipeline -> {pipeline_dir.relative_to(BASE)}  (dry run)\n")
        for i, step in enumerate(steps, 1):
            print(f"[{i}/{len(steps)}] $ {' '.join(_cmd(step, pipeline_dir))}")
        return 0

    return run(steps, pipeline_dir, args.keep_going)


if __name__ == "__main__":
    sys.exit(main())

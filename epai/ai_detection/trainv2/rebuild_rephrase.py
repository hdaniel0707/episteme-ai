#!/usr/bin/env python3
r"""Rebuild every MG-Reph result, in dependency order.

The rephrase counterpart of ``rebuild_all.py`` -- runs the MG-Reph pipeline:
``split_manugram_reph.py`` (a no-op once the two halves exist), then the five
standalone rephrase scripts, each writing to its own ``results/trainv2/rephrase_*``
directory:

    score_histograms       plot_score_histograms.py          -> rephrase_score_histograms/
    zeroshot               rephrase_zeroshot_performance.py   -> rephrase_zeroshot_performance/
    best_zeroshot          rephrase_best_zeroshot_performance.py -> rephrase_best_zeroshot_performance/
    in_domain    [heavy]   rephrase_in_domain_performance.py  -> rephrase_in_domain_performance/
    lodo         [heavy]   rephrase_lodo_performance.py       -> rephrase_lodo_performance/

Every step (except the split) counts rephrased text as HUMAN ground truth and
uses only gpt56luna -- no gemma. Independent steps run concurrently (bounded by
--jobs and --heavy-jobs); the split runs first. Launch with ``uv run python`` so
the steps inherit that venv.

    uv run python epai/ai_detection/trainv2/rebuild_rephrase.py
    uv run python epai/ai_detection/trainv2/rebuild_rephrase.py --dry-run
    uv run python epai/ai_detection/trainv2/rebuild_rephrase.py --jobs 1        # serial, live output
    uv run python epai/ai_detection/trainv2/rebuild_rephrase.py --only in_domain lodo

With --jobs > 1 each step is logged to ``results/trainv2/_rebuild_logs/<step>.log``
(tail shown on failure); a step whose dependency failed is skipped; exit is
non-zero if anything failed or was skipped.
"""

from __future__ import annotations

import argparse
import os
import sys

from epai.ai_detection.trainv2.rebuild_all import Task, _dry_run
from epai.ai_detection.trainv2.rebuild_all import run as run_scheduler

DIR = "epai/ai_detection/trainv2"

_SPLIT = "split_manugram_reph"

TASKS: list[Task] = [
    Task(_SPLIT, [f"{DIR}/split_manugram_reph.py"]),
    Task("score_histograms", [f"{DIR}/plot_score_histograms.py"], after={_SPLIT}),
    Task("zeroshot", [f"{DIR}/rephrase_zeroshot_performance.py"], after={_SPLIT}),
    Task("best_zeroshot", [f"{DIR}/rephrase_best_zeroshot_performance.py"], after={_SPLIT}),
    Task("in_domain", [f"{DIR}/rephrase_in_domain_performance.py"], after={_SPLIT}, heavy=True),
    Task("lodo", [f"{DIR}/rephrase_lodo_performance.py"], after={_SPLIT}, heavy=True),
]


def main() -> int:
    names = [t.name for t in TASKS]
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-j", "--jobs", type=int, default=min(4, os.cpu_count() or 1),
                    help="max steps at once (default: %(default)s; 1 = serial, live output)")
    ap.add_argument("--heavy-jobs", type=int, default=1,
                    help="max heavy steps at once (in_domain, lodo) (default: 1)")
    ap.add_argument("--only", nargs="+", metavar="STEP",
                    help=f"run only these steps, plus what they depend on (of: {', '.join(names)})")
    ap.add_argument("--list", action="store_true", help="list step names and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args()

    by_name = {t.name: t for t in TASKS}
    if args.list:
        for t in TASKS:
            print(f"  {t.name:18}{'  [heavy]' if t.heavy else ''}  {' '.join(t.argv)}")
        return 0

    for n in args.only or []:
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
        selected = [t for t in TASKS if t.name in chosen]
    else:
        selected = list(TASKS)

    for t in selected:
        script = os.path.join(*t.argv[0].split("/"))
        if not os.path.exists(script):
            raise SystemExit(f"missing script: {t.argv[0]}")

    if args.dry_run:
        return _dry_run(selected, args.jobs, args.heavy_jobs)
    return run_scheduler(selected, args.jobs, args.heavy_jobs)


if __name__ == "__main__":
    sys.exit(main())

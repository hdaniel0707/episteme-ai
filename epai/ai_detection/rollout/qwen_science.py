r"""qwen3.8:27b into the science_v3f corpus — generate, append, score.

``llama3.3:70b`` already generated for this paper set and its rows are kept, so
qwen3.8:27b is added *alongside* it (a 4th generator), not in its place.

  generate  epai/science_data/generate/generate_ai.py --cfg
            cfg/science_data/science_v3.yaml. The checkpoint is keyed by
            (paper, label), so only the ~3000 qwen3827b:0811A calls are issued;
            the other generators are read from the shard. ``--skip-gen
            llama3.3:70b`` keeps the run off that endpoint entirely — its rows
            are still exported from cache.
  append    epai/ai_detection/prepare/append_generator_rows.py — the qwen rows
            into data/parquet/science_v3f_multi_0811A.parquet (text folded the
            same way, suite columns Null). ~54 v3f papers were backfilled during
            dedup and have no qwen row; that is reported, not fatal.
  score     run_holdout_pipeline.py --from embed — emb / Binoculars /
            Fast-DetectGPT / detector / judges over the Null (qwen) rows, and
            the new qwen38-27b judge column over every row. ``--skip-judge
            llama33-70b`` while that endpoint is down.

    python epai/ai_detection/rollout/qwen_science.py --gpu 0 --yes \
        --skip-gen llama3.3:70b --skip-judge llama33-70b

    python epai/ai_detection/rollout/qwen_science.py --from score --gpu 0 --yes
    python epai/ai_detection/rollout/qwen_science.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from epai.ai_detection.rollout._common import (
    BASE_DIR, DIM, RESET, YELLOW, holdout_score, rel, resolve_judges, run, run_steps,
)

CFG = "cfg/science_data/science_v3.yaml"
GENERATE_SCRIPT = "epai/science_data/generate/generate_ai.py"
APPEND_SCRIPT = "epai/ai_detection/prepare/append_generator_rows.py"

GEN_OUT = BASE_DIR / "data/science/science_v3_multi_0811A.parquet"
TARGET = BASE_DIR / "data/parquet/science_v3f_multi_0811A.parquet"
LABEL = "qwen3827b:0811A"

STEPS = ["generate", "append", "score"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--from", dest="start_from", choices=STEPS, metavar="STEP",
                    help=f"start here instead of at the start: {', '.join(STEPS)}")
    ap.add_argument("--skip-gen", nargs="+", default=[], metavar="MODEL",
                    help="generators to leave alone (substring), e.g. llama3.3:70b")
    ap.add_argument("--skip-judge", nargs="+", default=[], metavar="JUDGE",
                    help="suite judges to leave out (substring), e.g. llama33-70b")
    ap.add_argument("--gpu", default=None, metavar="ID")
    ap.add_argument("--cpu-threads", type=int, default=None, metavar="N")
    ap.add_argument("--per-band", type=int, default=20000,
                    help="cue-vector coverage cap; must exceed the largest "
                         "domain's row count (default 20000)")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    steps = STEPS[STEPS.index(args.start_from):] if args.start_from else list(STEPS)
    judges = resolve_judges(args.skip_judge)

    print(f"{DIM}  corpus   {rel(TARGET)}")
    print(f"  steps    {' → '.join(steps)}")
    print(f"  judges   {', '.join(judges)}{RESET}")

    def step_generate() -> int:
        cmd = [sys.executable, BASE_DIR / GENERATE_SCRIPT, "--cfg", CFG]
        for model in args.skip_gen:
            cmd += ["--skip-generator", model]
        return run(cmd, args.dry_run)

    def step_append() -> int:
        if not args.dry_run and not GEN_OUT.is_file():
            print(f"{YELLOW}{rel(GEN_OUT)} not found — run the generate step first.{RESET}")
            return 1
        cmd = [sys.executable, BASE_DIR / APPEND_SCRIPT,
               "--source", GEN_OUT, "--target", TARGET,
               "--label", LABEL, "--whitespace", "flat"]
        if args.dry_run:
            cmd.append("--dry-run")
        return run(cmd, args.dry_run)

    def step_score() -> int:
        return holdout_score(TARGET, judges, gpu=args.gpu,
                             cpu_threads=args.cpu_threads, per_band=args.per_band,
                             yes=args.yes, dry_run=args.dry_run)

    run_steps(steps, {"generate": step_generate, "append": step_append,
                      "score": step_score},
              "epai/ai_detection/rollout/qwen_science.py")


if __name__ == "__main__":
    main()

r"""The science-3 Ghostbuster experiment, with gemma4:31b in place of llama3.3:70b.

``run_science3_full.py`` builds ``ghostbuster_multi_sci3.parquet`` from
gpt56luna + gpt-oss:120b + llama3.3:70b. The llama3.3:70b leg stalled on the
GenAI4Science endpoint and never finished, so this runs the same experiment with
gemma4:31b as the third generator instead. llama was never generated here —
nothing to keep — so it is a replacement, not an addition. When the llama
endpoint recovers, add it with ``append_generator_rows.py`` rather than
rebuilding.

  gptoss120b      run_full_pipeline.py --cfg cfg/ghostbuster/gb_gptoss120b.yaml
  gemma431b       run_full_pipeline.py --cfg cfg/ghostbuster/gb_gemma4_31b.yaml
                  (generate → … → standardize → score → embed)
  verify_gpt56luna   the reused gpt56luna corpus is clean before it is folded in
  combine         build_ghostbuster_multi.py → ghostbuster_multi_sci3.parquet
                  (human + gpt56luna + gpt-oss:120b + gemma4:31b)
  verify_combined
  score           run_holdout_pipeline.py --from embed — Fast-DetectGPT,
                  detector and every suite judge (emb / Binoculars are already
                  in from the per-generator pipelines). ``--skip-judge
                  llama33-70b`` while that endpoint is down.

``--skip-gen`` drops a generator leg by name (``--skip-gen gptoss`` when
ghostbuster_gptoss120b is already built). ``--from`` resumes at a step.

    python epai/ai_detection/rollout/gemma_ghostbuster_new.py --gpu 0 --yes \
        --skip-judge llama33-70b

    python epai/ai_detection/rollout/gemma_ghostbuster_new.py --from combine --gpu 0 --yes
    python epai/ai_detection/rollout/gemma_ghostbuster_new.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from epai.ai_detection.rollout._common import (
    BASE_DIR, DIM, RESET, YELLOW, holdout_score, match_skip, rel, resolve_judges,
    run, run_steps, verify_corpus,
)

RUN_FULL_PIPELINE = "epai/ghostbuster/run_full_pipeline.py"
COMBINE_SCRIPT = "epai/ai_detection/prepare/build_ghostbuster_multi.py"

PARQUET_DIR = BASE_DIR / "data/parquet"
GPT56LUNA = PARQUET_DIR / "ghostbuster_gpt56luna.parquet"
COMBINED = PARQUET_DIR / "ghostbuster_multi_sci3.parquet"
DEFAULT_REFERENCE = PARQUET_DIR / "science_v3f_multi_0811A.parquet"

# step key -> (cfg file, generator leg name for --skip-gen matching)
GEN_LEGS = {
    "gptoss120b": ("gb_gptoss120b.yaml", "gpt-oss:120b gptoss120b"),
    "gemma431b": ("gb_gemma4_31b.yaml", "gemma4:31b gemma431b"),
}
STEPS = ["gptoss120b", "gemma431b", "verify_gpt56luna", "combine",
         "verify_combined", "score"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--from", dest="start_from", choices=STEPS, metavar="STEP",
                    help=f"start here: {', '.join(STEPS)}")
    ap.add_argument("--skip-gen", nargs="+", default=[], metavar="MODEL",
                    help="generator legs to skip (substring), e.g. gptoss")
    ap.add_argument("--skip-judge", nargs="+", default=[], metavar="JUDGE",
                    help="suite judges to leave out (substring), e.g. llama33-70b")
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                    help="parquet to diff the combined corpus against in "
                         f"verify_combined (default {rel(DEFAULT_REFERENCE)})")
    ap.add_argument("--gpu", default=None, metavar="ID")
    ap.add_argument("--cpu-threads", type=int, default=None, metavar="N")
    ap.add_argument("--gen-limit", type=int, default=None,
                    help="passed through to run_full_pipeline.py")
    ap.add_argument("--per-band", type=int, default=20000)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    steps = STEPS[STEPS.index(args.start_from):] if args.start_from else list(STEPS)
    steps = [s for s in steps
             if not (s in GEN_LEGS and match_skip(GEN_LEGS[s][1], args.skip_gen))]
    judges = resolve_judges(args.skip_judge)

    print(f"{DIM}  combined {rel(COMBINED)}")
    print(f"  steps    {' → '.join(steps)}")
    print(f"  judges   {', '.join(judges)}{RESET}")

    def full_pipeline(cfg_file: str):
        cmd = [sys.executable, BASE_DIR / RUN_FULL_PIPELINE,
               "--cfg", f"cfg/ghostbuster/{cfg_file}", "--yes"]
        if args.gpu is not None:
            cmd += ["--gpu", args.gpu]
        if args.cpu_threads:
            cmd += ["--cpu-threads", args.cpu_threads]
        if args.gen_limit is not None:
            cmd += ["--gen-limit", args.gen_limit]
        if args.dry_run:
            cmd.append("--dry-run")
        return run(cmd, args.dry_run)

    def step_combine() -> int:
        cmd = [sys.executable, BASE_DIR / COMBINE_SCRIPT, "-o", COMBINED,
               "--source", "ghostbuster_gpt56luna", "human",
               "--source", "ghostbuster_gpt56luna", "ai",
               "--source", "ghostbuster_gptoss120b", "ai",
               "--source", "ghostbuster_gemma431b", "ai"]
        return run(cmd, args.dry_run)

    reference = args.reference if args.reference.is_file() else None
    runners = {
        "gptoss120b": lambda: full_pipeline(GEN_LEGS["gptoss120b"][0]),
        "gemma431b": lambda: full_pipeline(GEN_LEGS["gemma431b"][0]),
        "verify_gpt56luna": lambda: verify_corpus(GPT56LUNA, dry_run=args.dry_run),
        "combine": step_combine,
        "verify_combined": lambda: verify_corpus(
            COMBINED, reference=reference, dry_run=args.dry_run),
        "score": lambda: holdout_score(
            COMBINED, judges, gpu=args.gpu, cpu_threads=args.cpu_threads,
            per_band=args.per_band, yes=args.yes, dry_run=args.dry_run),
    }

    run_steps(steps, runners, "epai/ai_detection/rollout/gemma_ghostbuster_new.py")


if __name__ == "__main__":
    main()

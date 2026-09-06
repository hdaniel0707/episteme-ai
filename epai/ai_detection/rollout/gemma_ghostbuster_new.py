r"""The 4-generator Ghostbuster corpus — the essay/wp/reuter mirror of science_v3f.

Builds ``ghostbuster_v3f_multi.parquet`` from the same four generators
``science_v3f_multi_0811A`` carries, applied to the Ghostbuster prompts instead
of the science abstracts:

    human + gpt56luna + gpt-oss:120b + llama3.3:70b + gemma4:31b

then scores it with every ``cfg/detector_suite.yaml`` judge (gpt56luna,
qwen3-14b, gpt-oss:120b, llama33-70b, gemma4-31b).

  gptoss120b        run_full_pipeline.py --cfg cfg/ghostbuster/gb_gptoss120b.yaml
  gemma431b         run_full_pipeline.py --cfg cfg/ghostbuster/gb_gemma4_31b.yaml
  llama3370b        run_full_pipeline.py --cfg cfg/ghostbuster/gb_llama33_70b.yaml
                    (generate → … → standardize → score → embed)
  verify_gpt56luna  the reused gpt56luna corpus is clean before it is folded in
  combine           build_ghostbuster_multi.py → ghostbuster_v3f_multi.parquet
  verify_combined   diffed against science_v3f_multi_0811A by default
  score             run_holdout_pipeline.py --from embed — Fast-DetectGPT,
                    detector and every suite judge (emb / Binoculars are already
                    in from the per-generator pipelines).

Every generator leg is resumable: run_full_pipeline.py's generate stage skips
any .txt that already exists, so a leg that was partly built by an earlier run
(the old ``run_science3_full.py`` / ``gemma_ghostbuster_new.py`` sci3 attempt)
only pays for its gaps. The llama3.3:70b leg's earlier partial output was
generated BEFORE ``generate.py``'s ``MAX_OUTPUT_TOKENS`` cap and should be wiped
first — see the README / the restart notes — so a clean run under the cap
replaces it.

``--skip-gen`` drops a generator leg by name (``--skip-gen gptoss`` when
ghostbuster_gptoss120b is already built and verified). ``--from`` resumes at a
step. ``--skip-judge llama33-70b`` leaves a judge out if its endpoint is down.

    python epai/ai_detection/rollout/gemma_ghostbuster_new.py --gpu 0 --yes

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
COMBINED = PARQUET_DIR / "ghostbuster_v3f_multi.parquet"
DEFAULT_REFERENCE = PARQUET_DIR / "science_v3f_multi_0811A.parquet"

# step key -> (cfg file, generator leg name for --skip-gen matching)
GEN_LEGS = {
    "gptoss120b": ("gb_gptoss120b.yaml", "gpt-oss:120b gptoss120b"),
    "gemma431b": ("gb_gemma4_31b.yaml", "gemma4:31b gemma431b"),
    "llama3370b": ("gb_llama33_70b.yaml", "llama3.3:70b llama3370b llama33-70b"),
}
# The three per-generator legs stand alone (each writes its own parquet), so
# their order among themselves does not matter; llama last only because it is
# the one most likely to need a --skip-gen.
STEPS = ["gptoss120b", "gemma431b", "llama3370b", "verify_gpt56luna", "combine",
         "verify_combined", "score"]

# build_ghostbuster_multi.py --source args: the human class once (from
# gpt56luna), then one ai class per generator.
COMBINE_SOURCES = [
    ("ghostbuster_gpt56luna", "human"),
    ("ghostbuster_gpt56luna", "ai"),
    ("ghostbuster_gptoss120b", "ai"),
    ("ghostbuster_gemma431b", "ai"),
    ("ghostbuster_llama3370b", "ai"),
]


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

    combine_sources = [s for s in COMBINE_SOURCES
                       if not match_skip(s[0], args.skip_gen)]

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
        cmd = [sys.executable, BASE_DIR / COMBINE_SCRIPT, "-o", COMBINED]
        for stem, which in combine_sources:
            cmd += ["--source", stem, which]
        return run(cmd, args.dry_run)

    reference = args.reference if args.reference.is_file() else None
    runners = {
        "gptoss120b": lambda: full_pipeline(GEN_LEGS["gptoss120b"][0]),
        "gemma431b": lambda: full_pipeline(GEN_LEGS["gemma431b"][0]),
        "llama3370b": lambda: full_pipeline(GEN_LEGS["llama3370b"][0]),
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

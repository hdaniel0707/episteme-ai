r"""The qwen38-27b judge column for the existing Ghostbuster corpora.

These corpora are already generated, embedded and Binoculars-scored — the only
thing a new suite judge needs from them is the judge column itself, so this runs
``add_llm_judge_scores.py`` and nothing else. It runs every judge the suite
lists (``--skip-judge`` drops one), and ``add_llm_judge_scores.py`` skips rows
that already have a score, so a corpus that already carries ``gpt56luna`` /
``qwen3-14b`` only pays for the missing judges.

  ghostbuster_gpt35t    original gpt-3.5-turbo generations (raw text)
  ghostbuster_gpt35ts   the same, standardize_text.py already run
  ghostbuster_gpt56luna the new-generator baseline; needs gptoss/qwen38 added

``ghostbuster_gpt54mini`` is deliberately not here — it is not part of any
future corpus.

    python epai/ai_detection/rollout/qwen_ghostbuster_original.py --yes \
        --skip-judge llama33-70b

    python epai/ai_detection/rollout/qwen_ghostbuster_original.py --dry-run \
        --only ghostbuster_gpt56luna
"""

from __future__ import annotations

import argparse
from pathlib import Path

from epai.ai_detection.rollout._common import (
    BASE_DIR, GREEN, RED, RESET, YELLOW, add_judges, banner, rel, resolve_judges,
)

CORPORA = [
    "ghostbuster_gpt35t",
    "ghostbuster_gpt35ts",
    "ghostbuster_gpt56luna",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="+", default=None, metavar="STEM",
                    choices=CORPORA, help="run only these of %(choices)s")
    ap.add_argument("--skip-judge", nargs="+", default=[], metavar="JUDGE",
                    help="suite judges to leave out (substring), e.g. llama33-70b")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent API calls; 4 is the GenAI4Science cap (default)")
    ap.add_argument("--cpu-threads", type=int, default=None, metavar="N")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    judges = resolve_judges(args.skip_judge)
    stems = args.only or CORPORA
    print(f"  judges   {', '.join(judges)}")

    failures: list[str] = []
    for stem in stems:
        parquet = BASE_DIR / "data/parquet" / f"{stem}.parquet"
        banner(stem, rel(parquet))
        if not args.dry_run and not parquet.is_file():
            print(f"{YELLOW}not on disk — skipped{RESET}")
            continue
        code = add_judges(parquet, judges, workers=args.workers,
                          cpu_threads=args.cpu_threads, yes=args.yes,
                          dry_run=args.dry_run)
        if code != 0:
            failures.append(stem)

    if failures:
        print(f"\n{RED}failed: {', '.join(failures)}{RESET}\n  Re-run — filled "
              f"rows are skipped, so it only retries the gaps.")
        raise SystemExit(1)
    if not args.dry_run:
        print(f"\n{GREEN}Done.{RESET}")


if __name__ == "__main__":
    main()

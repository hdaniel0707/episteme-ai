r"""The qwen38-27b judge column for the Denario corpus.

``denario_hc`` is eval-only and already fully scored; a new suite judge just
needs its column. Runs every judge the suite lists (``--skip-judge`` drops one);
``add_llm_judge_scores.py`` skips rows that already have a score.

    python epai/ai_detection/rollout/qwen_denario.py --yes --skip-judge llama33-70b
    python epai/ai_detection/rollout/qwen_denario.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from epai.ai_detection.rollout._common import (
    BASE_DIR, GREEN, RESET, YELLOW, add_judges, banner, rel, resolve_judges,
)

CORPUS = "denario_hc"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--skip-judge", nargs="+", default=[], metavar="JUDGE",
                    help="suite judges to leave out (substring), e.g. llama33-70b")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent API calls; 4 is the GenAI4Science cap (default)")
    ap.add_argument("--cpu-threads", type=int, default=None, metavar="N")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    judges = resolve_judges(args.skip_judge)
    parquet = BASE_DIR / "data/parquet" / f"{CORPUS}.parquet"

    banner(CORPUS, rel(parquet))
    print(f"  judges   {', '.join(judges)}")
    if not args.dry_run and not parquet.is_file():
        raise SystemExit(f"{YELLOW}not on disk: {rel(parquet)}{RESET}")

    code = add_judges(parquet, judges, workers=args.workers,
                      cpu_threads=args.cpu_threads, yes=args.yes,
                      dry_run=args.dry_run)
    if code != 0:
        raise SystemExit(code)
    if not args.dry_run:
        print(f"\n{GREEN}Done.{RESET}")


if __name__ == "__main__":
    main()

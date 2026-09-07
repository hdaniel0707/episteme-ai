r"""Merge the four per-generator Ghostbuster corpora into ghostbuster_v3f_multi.parquet.

The union step of ``gemma_ghostbuster_new.py``, pulled out so it can be run and
re-run on its own. It NEVER writes to the four input parquets — they are opened
read-only by ``build_ghostbuster_multi.py`` — and the single output file is
overwritten in place on every run, so this is safe to repeat at any point.

Sources (from ``gemma_ghostbuster_new.COMBINE_SOURCES``):

    human            <- ghostbuster_gpt56luna    (human rows, once)
    gpt56luna:0701A  <- ghostbuster_gpt56luna    (ai rows)
    gpt-oss:120b...  <- ghostbuster_gptoss120b   (ai rows)
    gemma4:31b...    <- ghostbuster_gemma431b    (ai rows)
    llama3.3:70b...  <- ghostbuster_llama3370b   (ai rows)

The merge only concatenates and NaN-fills — it scores nothing and validates
nothing. Blank rows, runaway generations and unfilled suite columns all pass
straight through; ``gb_v3f_clean.py`` then ``fill_suite.py`` deal with those.

    uv run python epai/ai_detection/rollout/gb_v3f_merge.py
    uv run python epai/ai_detection/rollout/gb_v3f_merge.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from epai.ai_detection.rollout._common import BASE_DIR, GREEN, RESET, rel, run
from epai.ai_detection.rollout.gemma_ghostbuster_new import (
    COMBINE_SCRIPT, COMBINE_SOURCES, COMBINED,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-o", "--out", type=Path, default=COMBINED,
                    help=f"output parquet (default {rel(COMBINED)})")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"  merge  {', '.join(f'{s}:{w}' for s, w in COMBINE_SOURCES)}")
    print(f"  ->     {rel(args.out)}")

    cmd = [sys.executable, BASE_DIR / COMBINE_SCRIPT, "-o", args.out]
    for stem, which in COMBINE_SOURCES:
        cmd += ["--source", stem, which]

    code = run(cmd, args.dry_run)
    if code != 0:
        raise SystemExit(code)

    if not args.dry_run:
        print(f"\n{GREEN}merged -> {rel(args.out)}{RESET}  "
              f"(inputs untouched)\n  next: gb_v3f_clean.py")


if __name__ == "__main__":
    main()

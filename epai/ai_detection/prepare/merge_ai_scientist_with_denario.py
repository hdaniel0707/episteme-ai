r"""Merge the scored ``ai_scientist_hc.parquet`` into ``manugram_den.parquet``
(the current, in-use ``mg_den`` corpus -- despite the filename, Denario's own
``denario_hc.parquet``/``denario_detect.parquet`` no longer exist anywhere on
disk; they were superseded by this file in the MANUGRAM rename. See
trainv2's MODEL_BIAS.md and ``epai/science_data_denario/exclude.py``).

Run ``check_ai_scientist_overlap.py`` BEFORE this (and before the scoring
pipeline) -- this script does not re-check for duplicate papers, only for
duplicate ids, which is a different and much cheaper thing.

WHY A SEPARATE SCRIPT, NOT ``pd.concat`` BY HAND
--------------------------------------------------------------------------------
The two files' column sets are close but NOT identical, and getting that
silently wrong is a real risk here:

* ``manugram_den.parquet`` carries ``llm_judge_score_sonnet5_v3`` and
  ``llm_judge_score_gemma4_31b_v3_lp`` -- both intentionally NOT produced by
  ``run_ai_scientist_hc_pipeline.py`` (matching ``ghostbuster_orig.parquet``'s
  own precedent: sonnet5 was retired from the suite, and the gemma logprob
  judge is a separate, not-run-by-default script). These come out NULL on
  every ai_scientist row after the merge -- reported, not hidden.
* Neither file has an ``ai_author`` column today (``build_denario_hc.py``'s
  narrow ``OUT_COLUMNS`` drops the source-fetch script's own ``ai_author`` /
  ``source_url`` columns when it builds the raw hc file). This script derives
  one for BOTH sides from ``generated_by`` (identical to it here -- Denario's
  machine rows are literally the string ``'denario'``, exactly what
  ``ai_author`` would say anyway), so the merged file has one column that
  reads "which system wrote this" for every row without a lookup.
* A plain ``pd.concat`` still does the right thing for a genuinely one-sided
  column (fills the other side with Null) -- the reason for this script is
  making that fact VISIBLE (a printed list) rather than something you find by
  noticing a column is 50% null three analyses later.

Run:
    uv run python epai/ai_detection/prepare/merge_ai_scientist_with_denario.py
    uv run python epai/ai_detection/prepare/merge_ai_scientist_with_denario.py \
        --denario data/parquet/manugram_den.parquet \
        --ai-scientist data/parquet/ai_scientist_hc.parquet \
        -o data/parquet/manugram_den_ext.parquet

Output is a NEW file by default (``manugram_den_ext.parquet``), never
``manugram_den.parquet`` in place -- point ``splits.py``'s ``mg_den`` entry at
it yourself once you've looked at the report below, the same
review-before-repoint pattern used for ``gb_orig`` -> ``ghostbuster_orig_stand``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"

DEFAULT_DENARIO = PARQUET_DIR / "manugram_den.parquet"
DEFAULT_AI_SCIENTIST = PARQUET_DIR / "ai_scientist_hc.parquet"
DEFAULT_OUT = PARQUET_DIR / "manugram_den_ext.parquet"

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
RESET = "\033[0m"

REQUIRED_COLS = ["id", "uid", "domain", "text", "generated_by"]


def _check_required(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{RED}{name} is missing required column(s) {missing}{RESET}")


def merge(denario: pd.DataFrame, ai_scientist: pd.DataFrame) -> pd.DataFrame:
    _check_required(denario, "--denario")
    _check_required(ai_scientist, "--ai-scientist")

    dupe_ids = set(denario["id"]) & set(ai_scientist["id"])
    if dupe_ids:
        raise SystemExit(
            f"{RED}{len(dupe_ids)} id(s) appear in BOTH files -- concatenating "
            f"would silently merge two different papers into one group: "
            f"{sorted(dupe_ids)[:10]}{RESET}\n"
            f"Run check_ai_scientist_overlap.py to see what actually collided."
        )
    dupe_uids = set(denario["uid"]) & set(ai_scientist["uid"])
    if dupe_uids:
        raise SystemExit(f"{RED}{len(dupe_uids)} uid(s) appear in BOTH files: "
                          f"{sorted(dupe_uids)[:10]}{RESET}")

    denario = denario.copy()
    ai_scientist = ai_scientist.copy()
    denario["ai_author"] = denario["generated_by"]
    ai_scientist["ai_author"] = ai_scientist["generated_by"]

    den_cols, ai_cols = set(denario.columns), set(ai_scientist.columns)
    only_den = sorted(den_cols - ai_cols)
    only_ai = sorted(ai_cols - den_cols)
    if only_den:
        print(f"{YELLOW}Denario-only column(s) -- NULL on every ai_scientist row "
              f"after the merge:{RESET}")
        for c in only_den:
            print(f"  {c}")
    if only_ai:
        print(f"{YELLOW}ai_scientist-only column(s) -- NULL on every denario row "
              f"after the merge:{RESET}")
        for c in only_ai:
            print(f"  {c}")

    out = pd.concat([denario, ai_scientist], ignore_index=True, sort=False)

    # `id` is a GROUP key, not a row key -- a human row and its matched AI
    # counterpart legitimately share one `id` (both here and in gb_orig/mg_gb/
    # mg_sci), so duplicate `id` WITHIN a file is normal and expected. What
    # would be a real bug is the same `id` spanning the TWO SOURCE FILES
    # (silently merging an ai_scientist group into a denario one) -- already
    # ruled out above. `uid` is a per-ROW key everywhere, though, so that one
    # genuinely must stay unique post-merge.
    dupe_uids_after = int(out["uid"].duplicated().sum())
    if dupe_uids_after:
        raise SystemExit(
            f"{RED}post-merge: {dupe_uids_after} duplicate uid(s) -- "
            f"not written.{RESET}"
        )
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--denario", type=Path, default=DEFAULT_DENARIO)
    ap.add_argument("--ai-scientist", type=Path, default=DEFAULT_AI_SCIENTIST)
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the report, write nothing")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    for p, name in ((args.denario, "--denario"), (args.ai_scientist, "--ai-scientist")):
        if not p.is_file():
            raise SystemExit(f"{RED}not found: {p} ({name}){RESET}")

    denario = pd.read_parquet(args.denario)
    ai_scientist = pd.read_parquet(args.ai_scientist)

    print(f"denario:      {len(denario)} rows, {len(denario.columns)} columns  "
          f"({args.denario.name})")
    print(f"ai_scientist: {len(ai_scientist)} rows, {len(ai_scientist.columns)} "
          f"columns  ({args.ai_scientist.name})")
    print()

    out = merge(denario, ai_scientist)

    print(f"\n{GREEN}{len(out)} rows, {len(out.columns)} columns "
          f"({len(denario)} + {len(ai_scientist)}){RESET}")
    print(f"  generated_by: {out['generated_by'].value_counts().to_dict()}")
    print(f"  ai_author:    {out['ai_author'].value_counts().to_dict()}")
    print(f"  domain:       {out['domain'].value_counts().to_dict()}")

    if args.dry_run:
        print(f"\n{YELLOW}--dry-run: nothing written to {args.out}{RESET}")
        return

    from epai.utils.parquet_utils import save_parquet
    save_parquet(out, args.out)
    print(
        f"\n{DIM}Not wired into splits.py yet -- this is a new file, "
        f"{args.out.name}, not a replacement for manugram_den.parquet. "
        f"Review the column report above, then point splits.py's mg_den "
        f"entry at it yourself once satisfied.{RESET}"
    )


if __name__ == "__main__":
    main()

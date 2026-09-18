r"""One-off: rename the ``ensemble_score`` column to ``fusionjudge_score`` in
place, across all stems that carry it.

Pure rename, no recomputation -- ``fusionjudge_score`` is the same FusionJudge
mean (of ``llm_judge_score_gpt56luna_v3`` and ``logjudge_ord2_gpt56luna_v3``)
that ``score_ensemble_gpt56luna_v3.py`` (since renamed
``score_fusionjudge_gpt56luna_v3.py``) already wrote under its old name;
values are untouched, only the column label changes. Covers the same 6 stems
``score_fusionjudge_gpt56luna_v3.py --only ...`` can target (its 4 default
``CORE_STEMS`` plus the 2 opt-in ``REPHRASE_STEMS``); a stem missing the file
or the column is reported and skipped, not an error.

Prompts for a typed confirmation before writing anything (``--yes`` skips the
prompt for non-interactive use). ``--dry-run`` only prints the plan.

Run:
    uv run python epai/ai_detection/score/rename_ensemble_to_fusionjudge.py --dry-run
    uv run python epai/ai_detection/score/rename_ensemble_to_fusionjudge.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from epai.utils.parquet_utils import load_parquet, save_parquet

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"

STEMS = [
    "ghostbuster_orig", "manugram_gb", "manugram_sci", "manugram_aiscientist",
    "manugram_reph_gb", "manugram_reph_sci",
]

OLD_COL = "ensemble_score"
NEW_COL = "fusionjudge_score"


def plan() -> list[tuple[str, Path, str]]:
    """One ``(stem, path, status)`` per stem in :data:`STEMS`, where
    ``status`` is ``"rename"``, ``"no file"``, ``"no ensemble_score"``, or
    ``"fusionjudge_score already present"``."""
    rows = []
    for stem in STEMS:
        p = PARQUET_DIR / f"{stem}.parquet"
        if not p.is_file():
            rows.append((stem, p, "no file"))
            continue
        names = set(pq.read_schema(p).names)
        if NEW_COL in names:
            rows.append((stem, p, "fusionjudge_score already present"))
        elif OLD_COL not in names:
            rows.append((stem, p, "no ensemble_score"))
        else:
            rows.append((stem, p, "rename"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, write nothing, no prompt")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt (non-interactive use)")
    args = ap.parse_args()

    rows = plan()
    to_rename = [(stem, p) for stem, p, status in rows if status == "rename"]

    print(f"{'stem':<22} status")
    print("-" * 50)
    for stem, _, status in rows:
        print(f"{stem:<22} {status}")

    if not to_rename:
        print("\nnothing to rename")
        return

    print(f"\n{len(to_rename)} file(s) will have {OLD_COL!r} renamed to {NEW_COL!r} in place:")
    for stem, p in to_rename:
        print(f"  {p}")

    if args.dry_run:
        print("\n[dry run] -- nothing written")
        return

    if not args.yes:
        reply = input(f"\nType 'rename' to proceed with {len(to_rename)} file(s): ")
        if reply.strip().lower() != "rename":
            print("aborted -- nothing written")
            return

    for stem, p in to_rename:
        df = load_parquet(p)
        df = df.rename(columns={OLD_COL: NEW_COL})
        save_parquet(df, p)

    print(f"\ndone -- renamed {OLD_COL!r} -> {NEW_COL!r} in {len(to_rename)} file(s)")


if __name__ == "__main__":
    main()

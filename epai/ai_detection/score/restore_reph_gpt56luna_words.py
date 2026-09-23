r"""Restore ``llm_judge_logprobs_gpt56luna_v3_words`` values that
``merge_captured_logprobs.py`` silently dropped from the reph corpora.

That merge folds in the rephrase-only capture (``REPHRASE_STEMS``, words mode,
only the two rephrased classes) by left-joining its sidecar onto the full
file and reassigning the *whole* column to the join result. Since the sidecar
only covers the two rephrased classes, every other row's pre-existing value in
that column -- e.g. the non-rephrased AI class, copied over from the core
``manugram_gb``/``manugram_sci`` files, which already had it -- was silently
replaced with empty instead of being left alone.

Those values are not gone: ``data/parquet/backup04/`` was written before that
merge ran and still has them. This script only ever fills a currently-empty
cell in ``llm_judge_logprobs_gpt56luna_v3_words`` from the matching row (by
``uid``) in the backup; it never overwrites a cell that already holds a value,
and never touches any other column. No LLM calls, no recomputation.

    uv run python epai/ai_detection/score/restore_reph_gpt56luna_words.py

Prints a full before/after breakdown per file per ``generated_by`` class and
asks for confirmation before writing each file. Pass ``--yes`` to skip the
per-file prompt on a re-run you've already reviewed once.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
BACKUP_DIR = PARQUET_DIR / "backup04"
SAFETY_BACKUP_DIR = PARQUET_DIR / "backup_restore_reph_words"

COLUMN = "llm_judge_logprobs_gpt56luna_v3_words"
STEMS = ["manugram_reph_gb", "manugram_reph_sci"]


def _nonempty(x) -> bool:
    if x is None:
        return False
    try:
        return len(x) > 0
    except TypeError:
        return False


def _confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N] ").strip().lower() == "y"


def analyse(stem: str) -> dict | None:
    cur_path = PARQUET_DIR / f"{stem}.parquet"
    bak_path = BACKUP_DIR / f"{stem}.parquet"
    print(f"\n{'=' * 74}\n{stem}")

    if not cur_path.is_file():
        print(f"  MISSING current file: {cur_path} -- skip")
        return None
    if not bak_path.is_file():
        print(f"  MISSING backup file: {bak_path} -- skip")
        return None

    cur = pd.read_parquet(cur_path)
    bak = pd.read_parquet(bak_path)
    print(f"  current: {len(cur)} rows  ({cur_path})")
    print(f"  backup : {len(bak)} rows  ({bak_path})")

    for name, df in (("current", cur), ("backup", bak)):
        if COLUMN not in df.columns:
            print(f"  {name} file has no column {COLUMN!r} -- abort, not touching it")
            return None
        if "uid" not in df.columns or "generated_by" not in df.columns:
            print(f"  {name} file is missing uid/generated_by -- abort")
            return None
        if not df["uid"].is_unique:
            print(f"  {name} file: uid is not unique -- abort")
            return None

    cur_uids = set(cur["uid"])
    bak_uids = set(bak["uid"])
    if cur_uids != bak_uids:
        only_cur = list(cur_uids - bak_uids)[:3]
        only_bak = list(bak_uids - cur_uids)[:3]
        print(f"  uid sets differ (current-only e.g. {only_cur}, "
              f"backup-only e.g. {only_bak}) -- abort, will not guess alignment")
        return None
    print(f"  uid sets match exactly ({len(cur_uids)} rows)")

    merged = cur[["uid", "generated_by", COLUMN]].merge(
        bak[["uid", COLUMN]], on="uid", suffixes=("_cur", "_bak")
    )
    merged["cur_nonempty"] = merged[f"{COLUMN}_cur"].map(_nonempty)
    merged["bak_nonempty"] = merged[f"{COLUMN}_bak"].map(_nonempty)
    merged["restore"] = (~merged["cur_nonempty"]) & merged["bak_nonempty"]
    merged["both_filled"] = merged["cur_nonempty"] & merged["bak_nonempty"]

    print(f"\n  {COLUMN!r} coverage, current vs backup, by generated_by:")
    summary = merged.groupby("generated_by").agg(
        n=("uid", "size"),
        cur_filled=("cur_nonempty", "sum"),
        bak_filled=("bak_nonempty", "sum"),
        to_restore=("restore", "sum"),
        both_filled=("both_filled", "sum"),
    )
    print(summary.to_string())

    n_both = int(merged["both_filled"].sum())
    if n_both:
        print(f"\n  NOTE: {n_both} row(s) are non-empty in BOTH current and "
              f"backup -- left untouched either way (no data-loss risk here), "
              f"informational only.")

    n_restore = int(merged["restore"].sum())
    if n_restore == 0:
        print("\n  Nothing to restore for this file.")
        return None

    restore_uids = set(merged.loc[merged["restore"], "uid"])
    print(f"\n  Will restore {n_restore} row(s) for {stem!r}.")
    return {"stem": stem, "cur_path": cur_path, "cur": cur, "bak": bak,
            "restore_uids": restore_uids}


def apply_fix(plan: dict) -> None:
    stem = plan["stem"]
    cur: pd.DataFrame = plan["cur"]
    bak: pd.DataFrame = plan["bak"]
    restore_uids = plan["restore_uids"]
    cur_path: Path = plan["cur_path"]

    backup_map = dict(zip(bak["uid"], bak[COLUMN]))
    before_other_cols = cur.drop(columns=[COLUMN])

    fixed = cur.copy()
    mask = fixed["uid"].isin(restore_uids)
    fixed.loc[mask, COLUMN] = fixed.loc[mask, "uid"].map(backup_map)

    # Paranoia check: this assignment must never touch a column other than
    # COLUMN. If it did, refuse to write rather than risk a silent corruption.
    if not before_other_cols.equals(fixed.drop(columns=[COLUMN])):
        print("  ABORT: a column other than the target column changed -- "
              "this should be impossible, refusing to write.")
        sys.exit(1)

    n_before = int(cur[COLUMN].map(_nonempty).sum())
    n_after = int(fixed[COLUMN].map(_nonempty).sum())
    print(f"  {COLUMN}: {n_before} -> {n_after} filled (+{n_after - n_before})")

    SAFETY_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    safety_path = SAFETY_BACKUP_DIR / f"{stem}.parquet"
    if safety_path.exists():
        print(f"  safety backup already present at {safety_path} -- keeping "
              f"the first pre-fix copy, not overwriting it")
    else:
        cur.to_parquet(safety_path, index=False)
        print(f"  wrote pre-fix safety copy -> {safety_path}")

    fixed.to_parquet(cur_path, index=False)
    print(f"  wrote fixed file -> {cur_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="+", choices=STEMS, default=STEMS)
    ap.add_argument("--yes", action="store_true",
                     help="skip the interactive per-file confirmation "
                          "(still prints the full analysis first)")
    args = ap.parse_args()

    print(f"Restoring dropped {COLUMN!r} values from {BACKUP_DIR} into the "
          "live reph parquets.\nThis only ever fills a currently-empty cell "
          "from the pre-merge backup; it never overwrites a filled cell, and "
          "never touches any other column.")

    plans = [p for stem in args.only if (p := analyse(stem)) is not None]

    if not plans:
        print("\nNothing to do.")
        return

    print(f"\n{'=' * 74}")
    for plan in plans:
        if not args.yes and not _confirm(
            f"Apply the fix to {plan['stem']}.parquet now?"
        ):
            print(f"  skipped {plan['stem']}")
            continue
        apply_fix(plan)

    print("\nDone. Re-running this script afterwards should report 0 rows "
          "to restore for every file it touched.")


if __name__ == "__main__":
    main()

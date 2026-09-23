r"""Backfill ``llm_judge_logprobs_*`` columns for the ``human`` rows of the
MG-Reph files from their core-corpus source.

``rephrase_build.py``'s ``extract`` step pulled the ``human`` rows into
``manugram_reph_gb.parquet`` / ``manugram_reph_sci.parquet`` on 2026-09-09,
"with no separate carry step" -- whatever columns the core corpus
(``manugram_gb.parquet`` / ``manugram_sci.parquet``) had at that moment came
along for free. The four logprob-window columns weren't added to the core
corpora until the capture-and-merge on 2026-09-10 -- a day later -- so the
``human`` snapshot in the reph files never picked them up. The AI/rewrite
duplicate row (``gpt56luna:0701A``/``0811A``) didn't have this problem because
``reph_add_rewrite.py`` pulled it in later (2026-09-16), after the core
corpora already had the columns.

The core corpora have full logprob-window coverage for ``human``, and every
``human`` uid in the reph files is confirmed (not assumed) to also exist as a
``human`` row in the matching core file. So this is a pure uid-matched column
copy, no LLM calls, no recomputation: it only ever fills a currently-empty
cell in one of the four target columns, for ``human`` rows only, from the
value already sitting in the core file's matching row. It never overwrites a
cell that already has a value, and never touches any other row or column.

    uv run python epai/ai_detection/score/restore_reph_human_logprobs.py

After this, the derived columns are stale for the newly-filled rows and need
their own (separate, deterministic, no-API) recompute:

    uv run python epai/ai_detection/score/score_logjudge_ord2.py \
        --only manugram_reph_gb manugram_reph_sci --overwrite
    uv run python epai/ai_detection/score/score_fusionjudge_gpt56luna_v3.py \
        --only manugram_reph_gb manugram_reph_sci --overwrite

Prints a full before/after coverage breakdown and asks for confirmation
before writing each file. Pass ``--yes`` to skip the per-file prompt on a
re-run you've already reviewed once.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
SAFETY_BACKUP_DIR = PARQUET_DIR / "backup_restore_reph_human_logprobs"

HUMAN_LABEL = "human"
COLUMNS = [
    "llm_judge_logprobs_gpt56luna_v3_words",
    "llm_judge_logprobs_gpt56luna_v3_scale",
    "llm_judge_logprobs_gemma4_31b_v3_words",
    "llm_judge_logprobs_gemma4_31b_v3_scale",
]
#: reph stem -> its core-corpus source.
STEM_PAIRS = {
    "manugram_reph_sci": "manugram_sci",
    "manugram_reph_gb": "manugram_gb",
}


def _nonempty(x) -> bool:
    if x is None:
        return False
    try:
        return len(x) > 0
    except TypeError:
        return False


def _confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N] ").strip().lower() == "y"


def analyse(reph_stem: str, core_stem: str) -> dict | None:
    reph_path = PARQUET_DIR / f"{reph_stem}.parquet"
    core_path = PARQUET_DIR / f"{core_stem}.parquet"
    print(f"\n{'=' * 74}\n{reph_stem}  (source: {core_stem})")

    if not reph_path.is_file():
        print(f"  MISSING reph file: {reph_path} -- skip")
        return None
    if not core_path.is_file():
        print(f"  MISSING core file: {core_path} -- skip")
        return None

    reph = pd.read_parquet(reph_path)
    for c in ("uid", "generated_by", *COLUMNS):
        if c not in reph.columns:
            print(f"  reph file has no column {c!r} -- abort, not touching it")
            return None
    if not reph["uid"].is_unique:
        print("  reph file: uid is not unique -- abort")
        return None

    need = ["uid", "generated_by", *COLUMNS]
    core = pd.read_parquet(core_path, columns=need)
    for c in COLUMNS:
        if c not in core.columns:
            print(f"  core file has no column {c!r} -- abort")
            return None
    core_human = core[core["generated_by"] == HUMAN_LABEL]
    if not core_human["uid"].is_unique:
        print("  core file: human uid is not unique -- abort")
        return None

    reph_human = reph[reph["generated_by"] == HUMAN_LABEL]
    reph_human_uids = set(reph_human["uid"])
    core_human_uids = set(core_human["uid"])
    missing = reph_human_uids - core_human_uids
    if missing:
        print(f"  {len(missing)} reph human uid(s) have no match in the core "
              f"file's human rows (e.g. {list(missing)[:3]}) -- abort, will "
              f"not guess")
        return None
    print(f"  {len(reph_human_uids)} human uid(s) in {reph_stem}, all found "
          f"in {core_stem}'s human rows")

    core_lookup = core_human.set_index("uid")
    merged = reph_human[["uid", *COLUMNS]].merge(
        core_human[["uid", *COLUMNS]], on="uid", suffixes=("_cur", "_core")
    )

    restore_uids_by_col: dict[str, set] = {}
    rows = []
    for c in COLUMNS:
        cur_nonempty = merged[f"{c}_cur"].map(_nonempty)
        core_nonempty = merged[f"{c}_core"].map(_nonempty)
        restore = (~cur_nonempty) & core_nonempty
        both = cur_nonempty & core_nonempty
        restore_uids_by_col[c] = set(merged.loc[restore, "uid"])
        rows.append({
            "column": c, "n": len(merged),
            "cur_filled": int(cur_nonempty.sum()),
            "core_filled": int(core_nonempty.sum()),
            "to_restore": int(restore.sum()),
            "both_filled": int(both.sum()),
        })

    print("\n  human-row coverage, current reph vs core, per column:")
    print(pd.DataFrame(rows).to_string(index=False))

    n_both_total = sum(r["both_filled"] for r in rows)
    if n_both_total:
        print(f"\n  NOTE: {n_both_total} (column, row) cell(s) across all "
              f"columns are non-empty in BOTH -- left untouched either way, "
              f"informational only.")

    if not any(restore_uids_by_col.values()):
        print("\n  Nothing to restore for this file.")
        return None

    n_restore_total = sum(len(v) for v in restore_uids_by_col.values())
    n_cols_touched = sum(1 for v in restore_uids_by_col.values() if v)
    print(f"\n  Will restore {n_restore_total} cell(s) across "
          f"{n_cols_touched} column(s) for {reph_stem!r}.")
    return {
        "reph_stem": reph_stem,
        "reph_path": reph_path,
        "reph": reph,
        "core_lookup": core_lookup,
        "restore_uids_by_col": restore_uids_by_col,
    }


def apply_fix(plan: dict) -> None:
    reph_stem = plan["reph_stem"]
    reph_path: Path = plan["reph_path"]
    reph: pd.DataFrame = plan["reph"]
    core_lookup: pd.DataFrame = plan["core_lookup"]
    restore_uids_by_col: dict[str, set] = plan["restore_uids_by_col"]

    other_cols = [c for c in reph.columns if c not in COLUMNS]
    before_other = reph[other_cols]

    fixed = reph.copy()
    for c, uids in restore_uids_by_col.items():
        if not uids:
            continue
        backup_map = core_lookup[c].to_dict()
        mask = fixed["uid"].isin(uids)
        fixed.loc[mask, c] = fixed.loc[mask, "uid"].map(backup_map)
        n_before = int(reph[c].map(_nonempty).sum())
        n_after = int(fixed[c].map(_nonempty).sum())
        print(f"  {c}: {n_before} -> {n_after} filled (+{n_after - n_before})")

    # Paranoia check: this must never touch a column other than COLUMNS. If
    # it did, refuse to write rather than risk a silent corruption.
    if not before_other.equals(fixed[other_cols]):
        print("  ABORT: a column other than the target columns changed -- "
              "this should be impossible, refusing to write.")
        sys.exit(1)

    SAFETY_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    safety_path = SAFETY_BACKUP_DIR / f"{reph_stem}.parquet"
    if safety_path.exists():
        print(f"  safety backup already present at {safety_path} -- keeping "
              f"the first pre-fix copy, not overwriting it")
    else:
        reph.to_parquet(safety_path, index=False)
        print(f"  wrote pre-fix safety copy -> {safety_path}")

    fixed.to_parquet(reph_path, index=False)
    print(f"  wrote fixed file -> {reph_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="+", choices=sorted(STEM_PAIRS), default=None,
                     metavar="STEM")
    ap.add_argument("--yes", action="store_true",
                     help="skip the interactive per-file confirmation "
                          "(still prints the full analysis first)")
    args = ap.parse_args()

    stems = args.only or sorted(STEM_PAIRS)
    print(f"Backfilling {', '.join(COLUMNS)}\nfor human rows of "
          f"{', '.join(stems)} from their core-corpus source.\nThis only "
          f"ever fills a currently-empty cell from the core file; it never "
          f"overwrites a filled cell, and never touches any other row or "
          f"column.")

    plans = [p for stem in stems
              if (p := analyse(stem, STEM_PAIRS[stem])) is not None]

    if not plans:
        print("\nNothing to do.")
        return

    print(f"\n{'=' * 74}")
    for plan in plans:
        if not args.yes and not _confirm(
            f"Apply the fix to {plan['reph_stem']}.parquet now?"
        ):
            print(f"  skipped {plan['reph_stem']}")
            continue
        apply_fix(plan)

    print("\nDone. Now re-run (deterministic, no API):\n"
          "  uv run python epai/ai_detection/score/score_logjudge_ord2.py "
          "--only manugram_reph_gb manugram_reph_sci --overwrite\n"
          "  uv run python epai/ai_detection/score/score_fusionjudge_gpt56luna_v3.py "
          "--only manugram_reph_gb manugram_reph_sci --overwrite\n"
          "so the derived columns pick up the newly-filled human rows.")


if __name__ == "__main__":
    main()

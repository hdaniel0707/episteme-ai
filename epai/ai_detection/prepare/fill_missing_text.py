"""Backfill missing ``text`` values in a scored Parquet from a source Parquet.

Some rows in a ``*_binox`` Parquet (the *target*) have an empty ``text`` column
and therefore an empty binoculars score. This script looks those rows up by key
in a second *source* Parquet (which has good ``text``) and pastes the source
text into the empty cells of the target.

Only empty ``text`` cells are filled, and each fill is double-checked to confirm
the target cell really was empty before writing. Score columns (e.g.
``binoculars_*``) are never touched, so the binoculars score stays null for the
freshly filled rows — it can be recomputed later.

Before writing anything the script prints exactly what it plans to do and asks
for confirmation. Nothing is saved unless you type ``yes``.

Run:
    uv run python epai/prepare/fill_missing_text.py data/parquet/reuter2_binox0.parquet \
        data/parquet/reuter_gpt54mini.parquet \
        -o data/parquet/reuter_gpt54mini_binox0_filled.parquet
    
    uv run python epai/prepare/fill_missing_text.py data/parquet/essay2_binox0.parquet \
        data/parquet/essay_gpt54mini.parquet \
        -o data/parquet/essay_gpt54mini_binox0_filled.parquet

    uv run python epai/prepare/fill_missing_text.py data/parquet/wp2_binox0.parquet \
        data/parquet/wp_gpt54mini.parquet \
        -o data/parquet/wp_gpt54mini_binox0_filled.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.parquet_utils import load_parquet, save_parquet

BASE_DIR = Path(__file__).parent.parent.parent.parent

DEFAULT_KEYS = ["id", "generated_by"]
DEFAULT_TEXT_COL = "text"


def _is_empty(value) -> bool:
    """True for NaN/None, empty/whitespace string, or an empty container."""
    if value is None:
        return True
    if np.isscalar(value) or isinstance(value, float):
        try:
            if pd.isna(value):
                return True
        except (TypeError, ValueError):
            pass
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict, np.ndarray)):
        return len(value) == 0
    return False


def _check_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required column(s): {missing}. "
                       f"Available columns: {list(df.columns)}")


def _key_series(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """A single hashable key per row, built by tupling the key columns."""
    return df[keys].apply(lambda r: tuple(r), axis=1)


def plan_fills(
    target: pd.DataFrame,
    source: pd.DataFrame,
    keys: list[str],
    text_col: str,
) -> tuple[list[dict], list[dict]]:
    """Work out which target rows can be filled.

    Returns (fillable, unresolved) where each item is a dict describing a target
    row whose ``text`` is empty. ``fillable`` rows have exactly one matching,
    non-empty source row; ``unresolved`` rows do not (no match, empty source
    text, or ambiguous duplicate keys) and will be left untouched.
    """
    # Map source key -> source text, but only for non-empty, unambiguous keys.
    src_text: dict[tuple, object] = {}
    src_dup_keys: set[tuple] = set()
    src_keys = _key_series(source, keys)
    seen: set[tuple] = set()
    for key, value in zip(src_keys, source[text_col]):
        if _is_empty(value):
            continue
        if key in seen:
            src_dup_keys.add(key)
        seen.add(key)
        # Keep the first non-empty occurrence; duplicates flagged above.
        src_text.setdefault(key, value)

    tgt_keys = _key_series(target, keys)

    fillable: list[dict] = []
    unresolved: list[dict] = []
    for pos, (idx, key, cur) in enumerate(zip(target.index, tgt_keys, target[text_col])):
        if not _is_empty(cur):
            continue  # target already has text — never overwrite
        record = {"index": idx, "key": key}
        if key in src_dup_keys:
            record["reason"] = "ambiguous: multiple non-empty source rows for this key"
            unresolved.append(record)
        elif key in src_text:
            record["text"] = src_text[key]
            fillable.append(record)
        else:
            record["reason"] = "no matching non-empty source row"
            unresolved.append(record)
    return fillable, unresolved


def _preview(value: object, width: int = 70) -> str:
    s = str(value).replace("\n", " ")
    return s[:width] + ("…" if len(s) > width else "")


def print_plan(
    target_path: Path,
    source_path: Path,
    out_path: Path,
    target: pd.DataFrame,
    keys: list[str],
    text_col: str,
    fillable: list[dict],
    unresolved: list[dict],
) -> None:
    n_empty = int(target[text_col].apply(_is_empty).sum())
    print("=" * 74)
    print("Plan: backfill missing text")
    print(f"  target (to be updated) : {target_path}")
    print(f"  source (text donor)    : {source_path}")
    print(f"  output                 : {out_path}")
    print(f"  match keys             : {keys}")
    print(f"  text column            : {text_col!r}")
    print("=" * 74)
    print(f"Target rows                        : {len(target)}")
    print(f"Target rows with empty {text_col!r:12}: {n_empty}")
    print(f"  -> will be filled from source    : {len(fillable)}")
    print(f"  -> cannot be filled (untouched)  : {len(unresolved)}")
    print("Score columns                      : left unchanged (stay null)")

    if fillable:
        print("\nRows to fill (double-checked empty in target):")
        for r in fillable[:20]:
            print(f"  key={r['key']}  <- {_preview(r['text'])}")
        if len(fillable) > 20:
            print(f"  … and {len(fillable) - 20} more")

    if unresolved:
        print("\nEmpty rows that will be LEFT AS-IS:")
        for r in unresolved[:20]:
            print(f"  key={r['key']}  ({r['reason']})")
        if len(unresolved) > 20:
            print(f"  … and {len(unresolved) - 20} more")
    print()


def apply_fills(
    target: pd.DataFrame,
    text_col: str,
    fillable: list[dict],
) -> int:
    """Write the planned text into the target. Re-verifies each cell is empty."""
    n = 0
    for r in fillable:
        idx = r["index"]
        current = target.at[idx, text_col]
        if not _is_empty(current):
            # Safety net: never overwrite a non-empty cell, even if the plan
            # said otherwise. Skip and warn.
            print(f"  ! skipped index {idx}: cell is not empty anymore "
                  f"({_preview(current)})")
            continue
        target.at[idx, text_col] = r["text"]
        n += 1
    return n


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", type=Path,
                    help="Parquet with empty text/scores to be filled (updated).")
    ap.add_argument("source", type=Path,
                    help="Parquet providing the correct text (read-only).")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output path (default: overwrite the target in place).")
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS,
                    help=f"Key columns for matching (default: {DEFAULT_KEYS}).")
    ap.add_argument("--text-col", default=DEFAULT_TEXT_COL,
                    help=f"Text column to fill (default: {DEFAULT_TEXT_COL!r}).")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the confirmation prompt and write immediately.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or args.target

    target = load_parquet(args.target)
    source = load_parquet(args.source)

    _check_columns(target, args.keys + [args.text_col], "target")
    _check_columns(source, args.keys + [args.text_col], "source")

    fillable, unresolved = plan_fills(target, source, args.keys, args.text_col)

    print_plan(args.target, args.source, out_path, target,
               args.keys, args.text_col, fillable, unresolved)

    if not fillable:
        print("Nothing to fill — no changes made.")
        return

    if not args.yes:
        answer = input(f"Fill {len(fillable)} row(s) and write to {out_path}? "
                       "Type 'yes' to proceed: ").strip().lower()
        if answer != "yes":
            print("Aborted — no changes made.")
            return

    n_written = apply_fills(target, args.text_col, fillable)
    print(f"\nFilled {n_written} row(s).")
    save_parquet(target, out_path)


if __name__ == "__main__":
    main()

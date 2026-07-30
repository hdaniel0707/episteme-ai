"""Delete columns from a Parquet file, after showing exactly what would go.

Companion to ``add_llm_judge_scores.py``, and defaulted for it: with no column
names given it targets that script's default judge column
(``llm_judge_score_gpt54_mini``), which is the column to drop when a file was
scored under the *old* convention where 1.0 meant AI. The judge now writes
P(human) -- 1.0 ⇒ human, 0.0 ⇒ machine -- into a column of the same name, so
stale values cannot be told apart from fresh ones and re-judging means clearing
the column first.

Nothing is written until the plan is printed and confirmed. The plan lists, per
targeted column, its dtype, how many values it holds, its range, and -- when the
file has a ``generated_by`` column -- its mean per generator. That last table is
what identifies the convention a judge column was written under: under P(human)
the ``human`` rows have the HIGHER mean, under the old P(AI) the LOWER one. Read
it before typing 'yes'.

Requested columns that the file does not have are reported and skipped, so the
same command can be pointed at a whole set of parquets where only some carry the
column; ``--strict`` turns a missing column into an error instead. Every other
column, and the row order, is left exactly as it was.

The write is in place unless ``--out`` is given. These parquets are tracked by
git, so an unwanted delete can be undone with ``git checkout -- <file>`` as long
as the file was committed.

Run:
    # the default judge column, in place, with a confirmation prompt
    uv run python epai/ai_detection/llm_judge/delete_parquet_columns.py data/parquet/wp_binox0.parquet

    # look, decide nothing
    uv run python epai/ai_detection/llm_judge/delete_parquet_columns.py data/parquet/wp_binox0.parquet --dry-run

    # specific columns
    uv run python epai/ai_detection/llm_judge/delete_parquet_columns.py data/parquet/wp_binox0.parquet \
        llm_judge_score_gpt54_mini llm_judge_score_haiku45

    # into a copy, leaving the input untouched
    uv run python epai/ai_detection/llm_judge/delete_parquet_columns.py data/parquet/wp_binox0.parquet \
        -o data/parquet/wp_binox0_nojudge.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from epai.ai_detection.llm_judge.add_llm_judge_scores import (
    COL_PREFIX,
    DEFAULT_MODEL,
    GENERATED_BY_COL,
    score_column,
)
from epai.utils.parquet_utils import load_parquet, save_parquet

# With no columns on the command line, the target is the column
# add_llm_judge_scores.py writes by default -- the one that has to be cleared
# before a file scored under the old P(AI) convention can be re-judged.
DEFAULT_COLUMNS = [score_column(DEFAULT_MODEL)]

# Non-numeric columns are previewed by example rather than by range.
EXAMPLE_VALUES = 3
EXAMPLE_CHARS = 60

RULE = 78


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquet", type=Path, help="Parquet to delete columns from.")
    ap.add_argument(
        "columns",
        nargs="*",
        default=None,
        metavar="COLUMN",
        help="Columns to delete (default: %s)." % " ".join(DEFAULT_COLUMNS),
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output path (default: overwrite the input in place).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Error out if a requested column is not in the file. Default: "
        "report it and delete the ones that are there.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and stop. Nothing is asked and nothing is written.",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt (the delete is destructive; the "
        "prompt is the default).",
    )
    return ap.parse_args()


def resolve_columns(requested: list[str] | None) -> list[str]:
    """The columns to delete, de-duplicated, order preserved."""
    values = list(DEFAULT_COLUMNS) if not requested else list(requested)
    seen: set[str] = set()
    return [c for c in values if not (c in seen or seen.add(c))]


def _example_values(series: pd.Series) -> str:
    """A few real values from a non-numeric column, truncated for printing."""
    present = series.dropna().astype(str).head(EXAMPLE_VALUES).tolist()
    if not present:
        return "(no values)"
    return ", ".join(
        repr(v[:EXAMPLE_CHARS] + "…" if len(v) > EXAMPLE_CHARS else v[:EXAMPLE_CHARS])
        for v in present
    )


def describe_column(df: pd.DataFrame, col: str) -> list[str]:
    """What the file currently holds in ``col``, as printable lines.

    Enough detail to recognise the column before it is thrown away: how much of
    it is filled in, and either its numeric range or a few of its values.
    """
    series = df[col]
    n_null = int(series.isna().sum())
    lines = [
        f"    dtype       : {series.dtype}",
        f"    values      : {len(series) - n_null} of {len(series)} rows "
        f"({n_null} null)",
    ]

    values = series.dropna()
    if is_numeric_dtype(series) and not values.empty:
        lines.append(
            f"    range       : min {values.min():.4f}   "
            f"mean {values.mean():.4f}   max {values.max():.4f}"
        )
        lines.extend(_means_per_generator(df, col))
    else:
        lines.append(f"    examples    : {_example_values(series)}")
    return lines


def _means_per_generator(df: pd.DataFrame, col: str) -> list[str]:
    """Mean of ``col`` per ``generated_by`` value, which reveals its direction.

    For a judge column this is the check that matters: P(human) puts the
    ``human`` rows at the high end, the old P(AI) convention at the low end.
    """
    if GENERATED_BY_COL not in df.columns:
        return []

    lines = [f"    mean per {GENERATED_BY_COL}:"]
    groups = df.groupby(df[GENERATED_BY_COL].astype(str), sort=True)[col]
    for name, group in groups:
        present = group.dropna()
        if present.empty:
            lines.append(f"      {name:<14}: (no values, n=0)")
        else:
            lines.append(
                f"      {name:<14}: {present.mean():.4f}  (n={len(present)})"
            )

    if col.startswith(COL_PREFIX):
        lines.append(
            "      ^ judge column: 'human' highest ⇒ P(human), the current "
            "convention;"
        )
        lines.append(
            "        'human' lowest ⇒ the old P(AI) scores, which need this "
            "delete."
        )
    return lines


def print_plan(
    path: Path,
    out_path: Path,
    df: pd.DataFrame,
    targets: list[str],
    missing: list[str],
) -> None:
    """Everything the run would do, before it is allowed to do any of it."""
    remaining = [c for c in df.columns if c not in set(targets)]

    print("=" * RULE)
    print("Parquet")
    print(f"  file          : {path}")
    print(f"  rows          : {len(df)}")
    print(f"  columns       : {len(df.columns)}")
    print()
    print(f"Columns to DELETE ({len(targets)})")
    for col in targets:
        print(f"  - {col}")
        for line in describe_column(df, col):
            print(line)
    if missing:
        print()
        print(f"Not in the file, skipped ({len(missing)})")
        for col in missing:
            print(f"  - {col}")
    print()
    print(f"Columns kept ({len(remaining)}), unchanged")
    for col in remaining:
        print(f"  · {col}")
    print()
    print("Write")
    print(f"  output        : {out_path}")
    print(
        "  mode          : "
        + (
            "IN PLACE — the input file is overwritten"
            if out_path == path
            else "new file — the input is left untouched"
        )
    )
    print("=" * RULE)


def confirm(targets: list[str], out_path: Path) -> bool:
    """Ask before anything is written. Anything but 'yes' aborts."""
    print()
    print(
        f"This deletes {len(targets)} column(s) and their values: "
        + ", ".join(targets)
    )
    answer = input(f"Proceed and write {out_path}? Type 'yes': ").strip().lower()
    return answer == "yes"


def main() -> None:
    args = parse_args()
    out_path = args.out or args.parquet
    requested = resolve_columns(args.columns)

    df = load_parquet(args.parquet)

    present = set(df.columns)
    targets = [c for c in requested if c in present]
    missing = [c for c in requested if c not in present]

    if missing and args.strict:
        raise SystemExit(
            f"{args.parquet}: column(s) {missing} not in the file (--strict). "
            f"It has: {list(df.columns)}"
        )
    if not targets:
        print(
            f"None of {requested} is in {args.parquet} — nothing to delete.\n"
            f"The file has: {list(df.columns)}"
        )
        return
    if len(targets) == len(df.columns):
        raise SystemExit(
            f"{args.parquet}: that would delete every column and leave an empty "
            "file. Refusing."
        )

    print_plan(args.parquet, out_path, df, targets, missing)

    if args.dry_run:
        print("\n--dry-run: nothing was written.")
        return
    if not args.yes and not confirm(targets, out_path):
        print("Aborted — no changes made.")
        return

    save_parquet(df.drop(columns=targets), out_path)
    print(f"Deleted {len(targets)} column(s): {', '.join(targets)}")


if __name__ == "__main__":
    main()

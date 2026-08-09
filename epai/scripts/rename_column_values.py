"""Rename *values* inside one column of a Parquet file, interactively.

The column itself keeps its name — only the labels stored in it change. The
motivating case is ``generated_by``, where a corpus was written with a
placeholder generator id (``gpt_plain``) that should become the canonical
``<model_slug>:<prompt_config>`` label produced by
:func:`epai.science_data.config.generated_by_label` (``gpt54mini:0701A``).
Those two are the defaults, so the common case is just:

    uv run python epai/scripts/rename_column_values.py data/parquet/essay.parquet

Nothing is written until you type ``yes`` at the prompt: the script first
prints the current value distribution, the exact rows it would touch, and the
resulting distribution. ``--dry-run`` skips the prompt and never writes;
``--yes`` skips the prompt and does write (for pipelines).

By default the file is rewritten in place, atomically — the new frame goes to a
temporary file in the same directory and is moved over the original only after
a successful write, so an interrupted run cannot leave a half-written parquet.
Pass ``--output`` to write elsewhere and leave the input untouched.

Examples:
    # defaults: generated_by, "gpt_plain" -> "gpt54mini:0701A", in place
    uv run python epai/scripts/rename_column_values.py data/parquet/ghostbuster_gpt54mini.parquet

    uv run python epai/scripts/rename_column_values.py data/parquet/ghostbuster_gpt35t.parquet \
            --old gpt \
            --new gpt35t:0000A

    # look, don't touch
    uv run python epai/scripts/rename_column_values.py data/parquet/essay.parquet --dry-run

    # several old labels collapsing onto one new label
    uv run python epai/scripts/rename_column_values.py data/parquet/essay.parquet \
        --column generated_by \
        --old gpt_plain gpt_plain_v2 \
        --new gpt54mini:0701A

    # pairwise: each old gets its own new (lists must be the same length)
    uv run python epai/scripts/rename_column_values.py data/parquet/essay.parquet \
        --old gpt_plain claude_plain \
        --new gpt54mini:0701A claude45haiku:0701A

    # to a new file, no prompt
    uv run python epai/scripts/rename_column_values.py data/parquet/essay.parquet \
        --output data/parquet/essay_relabelled.parquet --yes
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

DEFAULT_COLUMN = "generated_by"
DEFAULT_OLD = ["gpt_plain"]
DEFAULT_NEW = ["gpt54mini:0701A"]


def build_mapping(old: list[str], new: list[str]) -> dict[str, str]:
    """Pair ``old`` values with ``new`` ones.

    A single ``new`` value is broadcast over every ``old`` value (several
    labels collapsing onto one); otherwise the two lists are zipped and must
    have the same length.
    """
    if len(new) == 1:
        new = new * len(old)
    if len(old) != len(new):
        raise ValueError(
            f"--old has {len(old)} value(s) and --new has {len(new)}. Give one "
            "--new value (applied to all) or one per --old value."
        )
    mapping: dict[str, str] = {}
    for o, n in zip(old, new):
        if o in mapping and mapping[o] != n:
            raise ValueError(
                f"Old value {o!r} is mapped to both {mapping[o]!r} and {n!r}."
            )
        mapping[o] = n
    return mapping


def as_labels(series: pd.Series) -> pd.Series:
    """The column as plain objects, so categorical columns compare normally."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.astype(object)
    return series


def value_counts_table(series: pd.Series, top: int) -> str:
    """Value distribution, most frequent first, with nulls counted separately."""
    counts = series.value_counts(dropna=False)
    total = len(series)
    shown = counts.head(top)
    labels = ["<null>" if pd.isna(v) else repr(v) for v in shown.index]
    width = max((len(label) for label in labels), default=10)
    lines = []
    for label, n in zip(labels, shown):
        pct = (n / total * 100) if total else 0.0
        lines.append(f"    {label:<{width}}  {n:>9,}  {pct:5.1f}%")
    if len(counts) > top:
        lines.append(f"    ... and {len(counts) - top:,} more distinct value(s)")
    return "\n".join(lines)


def report_current(df: pd.DataFrame, column: str, path: Path, top: int) -> None:
    series = as_labels(df[column])
    print(f"File     {path}")
    print(f"Rows     {len(df):,}    Columns  {len(df.columns)}")
    print(f"Column   {column!r}  (dtype {df[column].dtype})")
    print(f"Distinct {series.nunique(dropna=True):,}    Nulls  {series.isna().sum():,}")
    print("\nCurrent distribution:")
    print(value_counts_table(series, top))


def report_plan(
    series: pd.Series, mapping: dict[str, str], top: int
) -> tuple[dict[str, int], list[str]]:
    """Print the per-value plan. Returns (rows affected per old value, warnings)."""
    counts = series.value_counts(dropna=False)
    existing = set(counts.index)
    affected: dict[str, int] = {}
    warnings: list[str] = []

    print("\nPlanned changes:")
    for old, new in mapping.items():
        n = int(counts.get(old, 0))
        affected[old] = n
        note = ""
        if n == 0:
            note = "  <- NOT PRESENT, nothing to do"
            warnings.append(f"{old!r} does not occur in the column")
        elif new in existing and new != old:
            merged = int(counts.get(new, 0))
            note = f"  <- MERGES into existing {new!r} ({merged:,} rows already)"
            warnings.append(
                f"{new!r} already exists ({merged:,} rows); {old!r} will be merged into it"
            )
        elif new == old:
            note = "  <- unchanged (old == new)"
        print(f"    {old!r} -> {new!r}   {n:>9,} row(s){note}")

    total = sum(affected.values())
    print(f"\n    total rows to change: {total:,}")
    return affected, warnings


def apply_mapping(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Replace values per ``mapping``; everything else is left as it is."""
    labels = as_labels(series)
    replaced = labels.map(lambda v: mapping.get(v, v) if not pd.isna(v) else v)
    if isinstance(series.dtype, pd.CategoricalDtype):
        return replaced.astype("category")
    # `map` always hands back object dtype; put an extension string column back
    # on its own dtype so the rewritten parquet keeps the schema it had. A
    # non-string column replaced by string labels legitimately becomes object,
    # and the caller reports that dtype change.
    if series.dtype != object and pd.api.types.is_string_dtype(series.dtype):
        return replaced.astype(series.dtype)
    return replaced


def write_parquet(df: pd.DataFrame, target: Path) -> None:
    """Write ``df`` to ``target`` atomically (temp file in the same dir + rename)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def confirm(prompt: str) -> bool:
    """Ask for an explicit ``yes``. Anything else — including EOF — is a no."""
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        print("\nNo input available (stdin is not a terminal). Aborting.")
        return False
    return answer in {"yes", "y"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename values inside one column of a Parquet file. Prints the "
            "current distribution and the planned changes, then asks before writing."
        )
    )
    parser.add_argument("parquet_file", help="Path to the Parquet file.")
    parser.add_argument(
        "-c",
        "--column",
        default=DEFAULT_COLUMN,
        help=f"Column whose values are renamed (default: {DEFAULT_COLUMN}).",
    )
    parser.add_argument(
        "--old",
        nargs="+",
        default=DEFAULT_OLD,
        metavar="VALUE",
        help=f"Value(s) to replace (default: {' '.join(DEFAULT_OLD)}).",
    )
    parser.add_argument(
        "--new",
        nargs="+",
        default=DEFAULT_NEW,
        metavar="VALUE",
        help=(
            "Replacement value(s). One value is applied to every --old value; "
            f"otherwise give one per --old value (default: {' '.join(DEFAULT_NEW)})."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write here instead of overwriting the input file.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Report only; never write and never prompt.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt (for non-interactive use).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="How many distinct values to show per distribution (default: 30).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    path = Path(args.parquet_file)
    if not path.is_file():
        print(f"Parquet file not found: {path}", file=sys.stderr)
        return 1

    try:
        mapping = build_mapping(list(args.old), list(args.new))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    df = pd.read_parquet(path)
    if args.column not in df.columns:
        print(
            f"Column {args.column!r} not found in {path}.\n"
            f"Available columns: {list(df.columns)}",
            file=sys.stderr,
        )
        return 2

    report_current(df, args.column, path, args.top)
    before = as_labels(df[args.column])
    affected, warnings = report_plan(before, mapping, args.top)

    target = Path(args.output) if args.output else path
    print(f"\nWould write to  {target}" + ("  (in place)" if target == path else ""))
    if args.output and target.exists():
        print("    WARNING: output file exists and would be overwritten.")
    for w in warnings:
        print(f"    WARNING: {w}")

    if sum(affected.values()) == 0:
        print("\nNothing to change. No file written.")
        return 0

    after = apply_mapping(df[args.column], mapping)
    print("\nResulting distribution:")
    print(value_counts_table(as_labels(after), args.top))
    if after.dtype != df[args.column].dtype:
        print(f"\n    NOTE: column dtype changes {df[args.column].dtype} -> {after.dtype}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    if not args.yes and not confirm("\nApply these changes? type 'yes' to confirm: "):
        print("Aborted. No file written.")
        return 1

    df[args.column] = after
    write_parquet(df, target)

    print("\nSummary")
    print(f"    file     {target}")
    print(f"    column   {args.column!r}")
    for old, new in mapping.items():
        n = affected[old]
        verdict = f"{n:,} row(s) changed" if n else "no rows (value absent)"
        print(f"    {old!r} -> {new!r}: {verdict}")
    print(f"    total    {sum(affected.values()):,} of {len(df):,} row(s) changed")
    if target == path:
        print("    written in place")
    else:
        print(f"    input {path} left unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

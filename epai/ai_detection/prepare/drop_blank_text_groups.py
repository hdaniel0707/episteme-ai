r"""Drop rows whose text is blank -- and every row paired with them -- into a new parquet.

    .venv/bin/python epai/ai_detection/prepare/drop_blank_text_groups.py \
        data/parquet/ghostbuster_multi_orig.parquet

A blank ``text`` cell (null, empty, or whitespace-only) is unusable: every
scorer and embedder downstream writes ``null`` for that row, so one blank
document shows up as a partial-fill ERROR on ~30 columns at once. The
Ghostbuster essay release ships six such rows -- seeded human input, never model
output, so nothing downstream can regenerate them.

This removes those rows AND every other row sharing their ``--group-col`` value
(a group is unpaired once any member is gone -- the human original and its
machine rewrites go together), writes the result to ``<stem>_cleaned.parquet``,
and never touches the input. The Arrow schema is preserved exactly (a pyarrow
filter, not a pandas round-trip).

It does NOT fix anything else the completeness checker flags. A judge column run
on only some arms, or a deliberately un-standardised arm (``gpt35t``), are not
bad rows -- see the checker's ``--allow-unstandardized``.

    # see what would go, write nothing
    .venv/bin/python epai/ai_detection/prepare/drop_blank_text_groups.py \
        data/parquet/ghostbuster_multi_orig.parquet --dry-run
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _blank_mask(col: pa.ChunkedArray) -> pa.ChunkedArray:
    """True where the cell is null, empty, or whitespace-only."""
    is_null = pc.is_null(col)
    stripped = pc.utf8_trim_whitespace(col)
    is_empty = pc.fill_null(pc.equal(stripped, ""), False)
    return pc.or_(is_null, is_empty)


def _counts(table: pa.Table, mask: pa.ChunkedArray, col: str) -> Counter:
    if col not in table.column_names:
        return Counter()
    vals = table.column(col).filter(mask).to_pylist()
    return Counter(str(v) for v in vals)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("parquet", type=Path, help="the corpus to clean (read-only)")
    ap.add_argument("--text-col", default="text",
                    help="the column whose blank cells mark a bad row (default: text)")
    ap.add_argument("--group-col", default="id",
                    help="rows sharing this value are one pair group and go together "
                         "(default: id)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output path (default: <stem>_cleaned.parquet next to the input)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the output file if it already exists")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt (for non-interactive use)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be removed and exit without writing")
    args = ap.parse_args()

    src: Path = args.parquet
    if not src.is_file():
        sys.exit(f"{RED}not a file:{RESET} {src}")

    out: Path = args.out or src.with_name(f"{src.stem}_cleaned{src.suffix}")
    if out.resolve() == src.resolve():
        sys.exit(f"{RED}refusing to write over the input:{RESET} {out}")
    if out.exists() and not args.force and not args.dry_run:
        sys.exit(f"{RED}output exists{RESET} (pass --force to overwrite): {out}")

    table = pq.read_table(src)
    n = table.num_rows
    print(f"{BOLD}{src}{RESET}  ({n} rows, {table.num_columns} columns)")

    for col in (args.text_col, args.group_col):
        if col not in table.column_names:
            sys.exit(f"{RED}no {col!r} column in this parquet{RESET}")

    bad = _blank_mask(table.column(args.text_col))
    n_bad = pc.sum(pc.cast(bad, pa.int64())).as_py() or 0
    if n_bad == 0:
        print(f"{GREEN}nothing to clean{RESET} -- no blank {args.text_col!r} cells.")
        return

    bad_groups = pc.unique(table.column(args.group_col).filter(bad))
    drop_mask = pc.is_in(table.column(args.group_col), value_set=bad_groups)
    n_drop = pc.sum(pc.cast(drop_mask, pa.int64())).as_py()
    keep = pc.invert(drop_mask)
    n_keep = n - n_drop

    print(f"\n{BOLD}blank {args.text_col!r} rows{RESET}: {n_bad}")
    for kind_col in ("generated_by", "domain"):
        c = _counts(table, bad, kind_col)
        if c:
            print(f"  by {kind_col}: {dict(c)}")

    print(f"\n{BOLD}pair groups to drop{RESET} ({args.group_col}): {len(bad_groups)}")
    gl = sorted(str(v) for v in bad_groups.to_pylist())
    print("  " + ", ".join(gl if len(gl) <= 30 else gl[:30] + ["..."]))

    print(f"\n{BOLD}rows{RESET}: drop {RED}{n_drop}{RESET}  keep {GREEN}{n_keep}{RESET}  "
          f"(of {n})")
    for kind_col in ("generated_by", "domain"):
        c = _counts(table, drop_mask, kind_col)
        if c:
            print(f"  dropped by {kind_col}: {dict(c)}")

    if args.dry_run:
        print(f"\n{DIM}--dry-run: nothing written. Would write -> {out}{RESET}")
        return

    print(f"\nwrite {GREEN}{n_keep}{RESET} rows -> {BOLD}{out}{RESET}")
    if not args.yes:
        reply = input(f"{YELLOW}Proceed?{RESET} [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            sys.exit("aborted.")

    cleaned = table.filter(keep)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(cleaned, out)
    back = pq.ParquetFile(out).metadata.num_rows
    print(f"{GREEN}wrote{RESET} {out}  ({back} rows, {cleaned.num_columns} columns)")
    if back != n_keep:
        sys.exit(f"{RED}row count mismatch{RESET}: expected {n_keep}, file has {back}")


if __name__ == "__main__":
    main()

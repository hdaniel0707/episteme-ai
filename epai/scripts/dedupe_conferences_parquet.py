"""Drop duplicate rows from the conferences parquet, keeping one per group.

A duplicate group is rows that share the same ``(title, abstract)`` — the same
definition ``plot_conferences.py`` uses to count them. Rows with a null title or
abstract are excluded from the check (a null is not a real duplicate of every
other null). Within a group, the first row (current file order) is kept and the
rest are dropped.

Prints the plan — row counts, group sizes, example groups — then asks for an
explicit ``yes`` before writing anything. ``--dry-run`` prints the plan and
stops; ``--yes`` skips the prompt for non-interactive use.

    .venv/bin/python epai/scripts/dedupe_conferences_parquet.py --dry-run

    .venv/bin/python epai/scripts/dedupe_conferences_parquet.py

    .venv/bin/python epai/scripts/dedupe_conferences_parquet.py \
        --parquet data/parquet/conferences.parquet \
        --out data/parquet/conferences_dedup.parquet
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd

from epai.utils.parquet_utils import load_parquet

DEFAULT_PARQUET = Path("data/parquet/conferences.parquet")
DUP_KEY = ["title", "abstract"]

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--parquet", type=Path, default=DEFAULT_PARQUET, help="input parquet (default: %(default)s)"
    )
    ap.add_argument(
        "-o", "--out", type=Path, default=None,
        help="output path (default: overwrite --parquet in place)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop; write nothing")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument(
        "--list", type=int, default=5, metavar="N",
        help="example duplicate groups to print (default: %(default)s)",
    )
    return ap.parse_args()


def confirm(prompt: str) -> bool:
    """Ask for an explicit ``yes``. Anything else — including EOF — is a no."""
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        print("\nNo input available (stdin is not a terminal). Aborting.")
        return False
    return answer in {"yes", "y"}


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


def main() -> int:
    args = parse_args()
    df = load_parquet(args.parquet)

    content = df.dropna(subset=DUP_KEY)
    n_null = len(df) - len(content)
    dup_mask = content.duplicated(subset=DUP_KEY, keep=False)
    dup_rows = content.loc[dup_mask]
    to_drop = content.loc[content.duplicated(subset=DUP_KEY, keep="first")].index
    groups = dup_rows.groupby(DUP_KEY, sort=False)

    print("-" * 78)
    print(f"{BOLD}{args.parquet}{RESET}")
    print(f"  rows                : {len(df)}")
    print(f"  excluded (null title/abstract, never counted as duplicates): {n_null}")
    print(f"  duplicate rows      : {len(dup_rows)}")
    print(f"  duplicate groups    : {groups.ngroups}")
    print(f"  rows to drop        : {len(to_drop)}")
    print(f"  rows after dedupe   : {len(df) - len(to_drop)}")

    if groups.ngroups:
        cols = [c for c in ("id", "conference_id", "conference_year") if c in df.columns] + ["title"]
        print(f"\n  example groups (up to {args.list}):")
        for i, (_, grp) in enumerate(groups):
            if i >= args.list:
                break
            title = grp["title"].iloc[0]
            title = title if len(title) <= 70 else title[:67] + "..."
            print(f"    {BOLD}{title}{RESET}  x{len(grp)}")
            for _, row in grp[cols[:-1]].iterrows():
                marker = "keep" if row.name not in to_drop else "drop"
                print(f"      [{marker}] " + "  ".join(f"{c}={row[c]}" for c in cols[:-1]))

    if not len(to_drop):
        print(f"\n{GREEN}No duplicates found. Nothing written.{RESET}")
        return 0

    if args.dry_run:
        print("\n--dry-run: nothing was written.")
        return 0

    target = args.out or args.parquet
    print(f"\n  write               : {target}" + ("  (in place)" if target == args.parquet else ""))
    if not args.yes and not confirm(
        f"\n{YELLOW}Drop {len(to_drop)} duplicate row(s) and write {target}?{RESET} "
        "type 'yes' to confirm: "
    ):
        print("Aborted. No file written.")
        return 1

    out_df = df.drop(index=to_drop)
    write_parquet(out_df, target)
    print(f"\n{GREEN}Done — {len(to_drop)} duplicate row(s) dropped, "
          f"{len(out_df)} rows written to {target}.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

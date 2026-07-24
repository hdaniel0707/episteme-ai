"""Compare two Parquet files (e.g. two ``*_binox`` files).

Reports, to both the terminal and a log file:
  - row counts of each
  - column counts, shared columns, and columns unique to each file
  - dtype mismatches on shared columns
  - per-column missing/empty counts side by side (shared columns)

"Missing/empty" means: NaN/None, empty string ``""``, or an empty
list/array/container (relevant for columns like ``emb_specter2``).

Output log saved to results/parquet_compare/<stem_a>__vs__<stem_b>/compare.log

Usage:
    uv run python epai/analyse/analyse_parquet_compare.py \
        data/parquet/reuter_binox0.parquet data/parquet/reuter_binox1.parquet
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "parquet_compare"


class Tee:
    """Collect printed lines so they can be echoed to stdout and a log file."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def print(self, line: str = "") -> None:
        print(line)
        self._lines.append(line)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._lines) + "\n")


def _is_empty(value) -> bool:
    """True for NaN/None, empty string, or an empty list/array/container."""
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


def _n_missing(series: pd.Series) -> int:
    return int(series.apply(_is_empty).sum())


def compare(df_a: pd.DataFrame, df_b: pd.DataFrame, path_a: Path, path_b: Path, out: Tee) -> None:
    name_a, name_b = path_a.stem, path_b.stem

    out.print("=" * 70)
    out.print("Parquet comparison")
    out.print(f"  A: {path_a}")
    out.print(f"  B: {path_b}")
    out.print(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    out.print("=" * 70)

    # --- rows ---
    out.print("\nRow counts:")
    out.print(f"  A ({name_a}): {len(df_a)}")
    out.print(f"  B ({name_b}): {len(df_b)}")
    out.print(f"  difference (A - B): {len(df_a) - len(df_b)}")

    # --- columns ---
    cols_a, cols_b = list(df_a.columns), list(df_b.columns)
    set_a, set_b = set(cols_a), set(cols_b)
    shared = [c for c in cols_a if c in set_b]  # keep A's order
    only_a = [c for c in cols_a if c not in set_b]
    only_b = [c for c in cols_b if c not in set_a]

    out.print("\nColumn counts:")
    out.print(f"  A ({name_a}): {len(cols_a)}")
    out.print(f"  B ({name_b}): {len(cols_b)}")
    out.print(f"  shared: {len(shared)}")

    out.print(f"\nColumns only in A ({name_a}): {only_a if only_a else '(none)'}")
    out.print(f"Columns only in B ({name_b}): {only_b if only_b else '(none)'}")

    # --- dtype mismatches on shared columns ---
    out.print("\nDtype mismatches on shared columns:")
    mismatches = [
        (c, str(df_a[c].dtype), str(df_b[c].dtype))
        for c in shared
        if df_a[c].dtype != df_b[c].dtype
    ]
    if mismatches:
        for c, da, db in mismatches:
            out.print(f"  - {c}: A={da}  B={db}")
    else:
        out.print("  (none — all shared columns have matching dtypes)")

    # --- missing/empty side by side on shared columns ---
    out.print("\nMissing / empty per shared column (A | B):")
    if shared:
        width = max(len(c) for c in shared)
        for c in shared:
            ma, mb = _n_missing(df_a[c]), _n_missing(df_b[c])
            flag = "  <-- differs" if ma != mb else ""
            out.print(f"  {c:<{width}}  {ma:>8} | {mb:>8}{flag}")
    else:
        out.print("  (no shared columns)")

    out.print("\n" + "=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parquet_a", help="Path to the first Parquet file (A).")
    parser.add_argument("parquet_b", help="Path to the second Parquet file (B).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_a, path_b = Path(args.parquet_a), Path(args.parquet_b)
    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)

    out = Tee()
    compare(df_a, df_b, path_a, path_b, out)

    log_path = RESULTS_DIR / f"{path_a.stem}__vs__{path_b.stem}" / "compare.log"
    out.save(log_path)
    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()

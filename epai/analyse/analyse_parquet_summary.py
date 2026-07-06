"""Summary statistics for a single Parquet file (e.g. a ``*_binox`` file).

Reports, to both the terminal and a log file:
  - row count
  - column count + names & dtypes
  - per-column missing/empty counts and *where* (row indices) they occur

"Missing/empty" means: NaN/None, empty string ``""``, or an empty
list/array/container (relevant for columns like ``emb_specter2``).

Output log saved to results/parquet_summary/<parquet_stem>/summary.log

Usage:
    uv run python epai/analyse/analyse_parquet_summary.py data/parquet/reuter_binox.parquet
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "parquet_summary"

# How many example row indices to list per column before truncating.
MAX_INDICES_SHOWN = 20


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
    # Scalars: use pd.isna, but guard against arrays (pd.isna returns an array).
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


def _empty_mask(series: pd.Series) -> pd.Series:
    """Boolean mask marking missing/empty entries in a column."""
    # Fast path for the common nullable case, then refine with the element-wise
    # check to also catch empty strings / empty containers.
    return series.apply(_is_empty)


def summarise(df: pd.DataFrame, path: Path, out: Tee) -> None:
    out.print("=" * 70)
    out.print(f"Parquet summary: {path}")
    out.print(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    out.print("=" * 70)

    n_rows, n_cols = df.shape
    out.print(f"\nRows:    {n_rows}")
    out.print(f"Columns: {n_cols}")

    out.print("\nColumns (name : dtype):")
    for col in df.columns:
        out.print(f"  - {col} : {df[col].dtype}")

    out.print("\nMissing / empty values per column:")
    any_missing = False
    for col in df.columns:
        mask = _empty_mask(df[col])
        n_missing = int(mask.sum())
        if n_missing == 0:
            continue
        any_missing = True
        idx = df.index[mask].tolist()
        pct = 100.0 * n_missing / n_rows if n_rows else 0.0
        shown = idx[:MAX_INDICES_SHOWN]
        more = "" if len(idx) <= MAX_INDICES_SHOWN else f" ... (+{len(idx) - MAX_INDICES_SHOWN} more)"
        out.print(f"  - {col}: {n_missing} missing ({pct:.2f}%) at rows {shown}{more}")

    if not any_missing:
        out.print("  (none — no missing/empty values in any column)")

    out.print("\n" + "=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parquet_file", help="Path to the Parquet file to summarise.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.parquet_file)
    df = pd.read_parquet(path)

    out = Tee()
    summarise(df, path, out)

    log_path = RESULTS_DIR / path.stem / "summary.log"
    out.save(log_path)
    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()

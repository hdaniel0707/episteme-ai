"""Rename a single column in a Parquet file, leaving all values unchanged.

Example:
    uv run python epai/utils/rename_parquet_column.py \
        data/parquet/reuter_binox.parquet \
        binoculars_falcon_7b_x_falcon_7b_instruct \
        binoculars_score \
        --output data/parquet/reuter_binox1.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def rename_parquet_column(
    parquet_file: str | Path,
    column_name: str,
    new_column_name: str,
    output_file: str | Path | None = None,
) -> Path:
    """Rename ``column_name`` to ``new_column_name`` in a Parquet file.

    Values are left untouched. Writes back to ``parquet_file`` in place unless
    ``output_file`` is given.

    Returns the path that was written.
    """
    parquet_file = Path(parquet_file)
    output_file = Path(output_file) if output_file is not None else parquet_file

    df = pd.read_parquet(parquet_file)

    if column_name not in df.columns:
        raise KeyError(
            f"Column {column_name!r} not found in {parquet_file}. "
            f"Available columns: {list(df.columns)}"
        )
    if new_column_name in df.columns and new_column_name != column_name:
        raise ValueError(
            f"Column {new_column_name!r} already exists in {parquet_file}."
        )

    df = df.rename(columns={column_name: new_column_name})
    df.to_parquet(output_file, index=False)
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename a column in a Parquet file, keeping values unchanged."
    )
    parser.add_argument("parquet_file", help="Path to the input Parquet file.")
    parser.add_argument("column_name", help="Existing column name to rename.")
    parser.add_argument("new_column_name", help="New column name.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output path. Defaults to overwriting the input file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = rename_parquet_column(
        args.parquet_file,
        args.column_name,
        args.new_column_name,
        args.output,
    )
    print(f"Renamed {args.column_name!r} -> {args.new_column_name!r} in {written}")


if __name__ == "__main__":
    main()

"""
Merge two or more Parquet datasets (e.g. the ones in data/parquet/) into a single
Parquet file. Columns are unioned across inputs; any column missing from a given
input is filled with Null in the result.

Run:
    python -m epai.prepare.merge_parquets data/parquet/0515_A_500_AI_260.parquet \
        data/parquet/denario_papers.parquet -o data/parquet/merged.parquet

    # default output is data/parquet/merged.parquet
    python -m epai.prepare.merge_parquets a.parquet b.parquet c.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from epai.utils.parquet_utils import load_parquet, save_parquet

BASE_DIR = Path(__file__).parent.parent
INPUTS = [
    BASE_DIR / "data" / "parquet" / "0515_A_500_AI_260.parquet",
    BASE_DIR / "data" / "parquet" / "denario_papers.parquet",
]
DEFAULT_OUT = BASE_DIR / "data" / "parquet" / "merged.parquet"


def merge_parquets(paths: list[Path]) -> pd.DataFrame:
    frames = [load_parquet(p) for p in paths]
    # Preserve the column order of the first frame, then append any new columns.
    columns: list[str] = []
    for frame in frames:
        for col in frame.columns:
            if col not in columns:
                columns.append(col)
    merged = pd.concat(frames, ignore_index=True)
    return merged.reindex(columns=columns)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge 2+ parquet datasets into one.")
    ap.add_argument(
        "-o", "--out", type=Path, default=DEFAULT_OUT, help="output parquet path"
    )
    args = ap.parse_args()

    df = merge_parquets(INPUTS)
    save_parquet(df, args.out)


if __name__ == "__main__":
    main()

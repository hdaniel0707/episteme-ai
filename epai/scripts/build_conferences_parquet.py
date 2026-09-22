"""Build a parquet from the raw OpenAlex conference-abstracts CSV.

Reads ``data/conferences/oalex_abstracts_2020-2025.csv`` (columns: id,
conference_id, conference_year, title, abstract) and writes it straight to
parquet, with ``conference_year`` coerced to int.

    .venv/bin/python epai/scripts/build_conferences_parquet.py

    .venv/bin/python epai/scripts/build_conferences_parquet.py \
        --input data/conferences/oalex_abstracts_2020-2025.csv \
        --output data/parquet/conferences.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epai.utils.parquet_utils import save_parquet

DEFAULT_INPUT = Path("data/conferences/oalex_abstracts_2020-2025.csv")
DEFAULT_OUTPUT = Path("data/parquet/conferences.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="raw CSV path (default: %(default)s)"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="parquet destination (default: %(default)s)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["conference_year"] = df["conference_year"].astype(int)
    save_parquet(df, args.output)


if __name__ == "__main__":
    main()

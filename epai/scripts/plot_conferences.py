"""Plot the year distribution of the conferences parquet, and report duplicates.

Reads the parquet built by ``build_conferences_parquet.py``, optionally filters
rows to one or more ``conference_id`` values, then:

  - plots a bar chart of row counts per ``conference_year`` to
    ``results/conferences/<tag>/year_distribution.png``
  - reports duplicate rows: exact (title, abstract) matches (rows with a null
    title or abstract are excluded from this check, since every null is not a
    real duplicate of every other null)

Both the plot and the duplicate check run on the filtered rows only.

    .venv/bin/python epai/scripts/plot_conferences.py --conference ICLR

    .venv/bin/python epai/scripts/plot_conferences.py --conference ICLR ICML

    .venv/bin/python epai/scripts/plot_conferences.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from epai.utils.parquet_utils import load_parquet

DEFAULT_PARQUET = Path("data/parquet/conferences.parquet")
RESULTS_DIR = Path("results/conferences")

# dataviz skill: sequential magnitude gets one hue, blue, from cfg/colors.yaml's
# validated ramp (step 450).
BAR_COLOR = "#2a78d6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet", type=Path, default=DEFAULT_PARQUET, help="input parquet (default: %(default)s)"
    )
    parser.add_argument(
        "--conference",
        nargs="+",
        default=None,
        metavar="CONFERENCE_ID",
        help="filter to one or more conference_id values (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_parquet(args.parquet)

    if args.conference:
        requested = {c.upper() for c in args.conference}
        available = set(df["conference_id"].unique())
        unknown = requested - available
        if unknown:
            raise SystemExit(
                f"Unknown conference_id(s): {sorted(unknown)}\nAvailable: {sorted(available)}"
            )
        df = df[df["conference_id"].isin(requested)]
        tag = "_".join(sorted(requested)).replace(" ", "-")
    else:
        tag = "all"

    print(f"{len(df)} rows after filtering (conference={tag})")

    # ------------------------------------------------------------- duplicates --
    content = df.dropna(subset=["title", "abstract"])
    is_dup = content.duplicated(subset=["title", "abstract"], keep=False)
    n_dup_rows = int(is_dup.sum())
    n_dup_groups = int(content.loc[is_dup].duplicated(subset=["title", "abstract"]).sum())
    print(
        f"{n_dup_rows} rows share a (title, abstract) with another row, "
        f"across {n_dup_groups} duplicate group(s) "
        f"({len(df) - len(content)} rows excluded: null title/abstract)"
    )

    # ----------------------------------------------------------------- plot --
    per_year = df["conference_year"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(per_year.index.astype(str), per_year.values, color=BAR_COLOR, width=0.6)
    for x, y in zip(per_year.index.astype(str), per_year.values):
        ax.text(x, y, f"{y}", ha="center", va="bottom", fontsize=9, color="#52514e")
    ax.set_xlabel("Year")
    ax.set_ylabel("Papers")
    ax.set_title(f"Year distribution — {tag}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "year_distribution.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot → {out_path}")


if __name__ == "__main__":
    main()

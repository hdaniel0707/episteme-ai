"""Plot the conferences parquet: per-year mean judge score and score-distribution
histograms, and report duplicates.

Reads the parquet built by ``build_conferences_parquet.py`` and scored by
``run_conferences_judge_pipeline.py``, optionally filters rows to one or more
``conference_id`` values, then:

  - plots the number of abstracts per year (bar chart) to
    ``results/conferences/<tag>/year_abstract_counts.png``.
  - plots the mean score per year for all 3 judge-score columns
    (``llm_judge_score_gpt56luna_v3``, ``logjudge_ord2_gpt56luna_v3``,
    ``fusionjudge_score``) on one chart, to
    ``results/conferences/<tag>/year_mean_scores.png``. Each line carries a
    shaded +/-1 SEM band (standard error of the mean, per year) in its own
    color, behind the mean.
  - for each of the 3 score columns, plots the score distribution per year as
    overlaid step histograms (light -> dark blue, one step per year) to
    ``results/conferences/<tag>/year_distribution_<column>.png``. Each
    column's score is P(human) in [0, 1].
  - for each of the 3 score columns, plots a finer-grained distribution
    comparing only the earliest and latest year (2020 vs. 2025) to
    ``results/conferences/<tag>/year_distribution_first_vs_last_<column>.png``.
  - reports duplicate rows: exact (title, abstract) matches (rows with a null
    title or abstract are excluded from this check, since every null is not a
    real duplicate of every other null)

Every plot and the duplicate check run on the filtered rows only.

    .venv/bin/python epai/scripts/plot_conferences.py --conference ICLR

    .venv/bin/python epai/scripts/plot_conferences.py --conference ICLR ICML

    .venv/bin/python epai/scripts/plot_conferences.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epai.utils.parquet_utils import load_parquet

DEFAULT_PARQUET = Path("data/parquet/conferences.parquet")
RESULTS_DIR = Path("results/conferences")

GRID_COLOR = "#e5e4e0"

# (column, short label used in plot title / filename), in the fixed
# categorical order (dataviz skill: slots 1/2/3 -- blue/orange/aqua), never
# reordered or cycled across plots.
SCORE_COLUMNS = [
    ("llm_judge_score_gpt56luna_v3", "LLM judge (parsed)"),
    ("logjudge_ord2_gpt56luna_v3", "LLM judge (ord2)"),
    ("fusionjudge_score", "Fusion judge"),
]

# Canonical name/color per judge, used on the mean-scores chart only (the
# paper's own convention -- see trainv2/run_best_zeroshot_performance.py's
# _ATHM_BARS_COLOR: BaseJudge gets its own steel blue rather than sharing
# LogJudge's blue, since here -- like that figure -- all three appear as
# persistent series rather than a hollow/filled "ours" pairing).
CANONICAL_NAME = {
    "llm_judge_score_gpt56luna_v3": "BaseJudge",
    "logjudge_ord2_gpt56luna_v3": "LogJudge",
    "fusionjudge_score": "FusionJudge",
}
SERIES_COLOR = {
    "llm_judge_score_gpt56luna_v3": "#4f77a5",  # BaseJudge
    "logjudge_ord2_gpt56luna_v3": "#1d6fe0",  # LogJudge
    "fusionjudge_score": "#4a3aa7",  # FusionJudge
}

# dataviz skill: sequential magnitude gets one hue, blue, light -> dark
# (validated ramp, steps 100-700). Used here to encode year as an ordinal
# position rather than a magnitude.
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def year_ramp(n: int) -> list[str]:
    """``n`` evenly spaced colors from :data:`BLUE_RAMP`, light -> dark."""
    idx = np.linspace(0, len(BLUE_RAMP) - 1, n).round().astype(int)
    return [BLUE_RAMP[i] for i in idx]


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
    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    years = sorted(df["conference_year"].dropna().unique())
    x = [str(y) for y in years]

    # -------------------------------------------------------------- counts --
    counts = df["conference_year"].value_counts().reindex(years, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, counts.values, color=SERIES_COLOR["llm_judge_score_gpt56luna_v3"], width=0.6)
    for xi, c in zip(x, counts.values):
        ax.text(xi, c, f"{c}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Year")
    ax.set_ylabel("Abstracts")
    ax.set_title(f"Abstracts per year — {tag}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_path = out_dir / "year_abstract_counts.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {out_path}")

    # -------------------------------------------------------------- means --
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in SCORE_COLUMNS:
        if col not in df.columns:
            print(f"(skip {col}: column not in parquet)")
            continue
        color = SERIES_COLOR[col]
        g = df.dropna(subset=[col]).groupby("conference_year")[col]
        means = g.mean().reindex(years)
        sem = (g.std().reindex(years) / np.sqrt(g.count().reindex(years)))
        ax.fill_between(x, (means - sem).values, (means + sem).values, color=color, alpha=0.25, linewidth=0)
        ax.plot(x, means.values, color=color, marker="o", linewidth=2, label=CANONICAL_NAME[col])

    ax.set_xlabel("Year")
    ax.set_ylabel("Score  P(human)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Mean judge score per year (±1 SEM) — {tag}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()

    out_path = out_dir / "year_mean_scores.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {out_path}")

    # -------------------------------------------------------- distributions --
    edges = np.linspace(0, 1, 41)
    mids = (edges[:-1] + edges[1:]) / 2
    colors = year_ramp(len(years))

    for col, label in SCORE_COLUMNS:
        if col not in df.columns:
            continue
        scored = df.dropna(subset=[col])

        fig, ax = plt.subplots(figsize=(8, 5))
        for y, color in zip(years, colors):
            vals = scored.loc[scored["conference_year"] == y, col].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            density = np.histogram(vals, bins=edges, density=True)[0]
            ax.step(mids, density, where="mid", color=color, linewidth=1.8, label=str(y))

        ax.set_xlabel("Score  P(human)")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.set_title(f"{label} — score distribution by year — {tag}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(title="Year", frameon=False, loc="upper left")
        fig.tight_layout()

        out_path = out_dir / f"year_distribution_{col}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot → {out_path}")

    # ----------------------------------------------- first vs. last, finer --
    first_year, last_year = years[0], years[-1]
    fine_edges = np.linspace(0, 1, 101)
    fine_mids = (fine_edges[:-1] + fine_edges[1:]) / 2
    first_color, last_color = "#008300", "#eb6834"  # slot 6 green, slot 2 orange

    for col, label in SCORE_COLUMNS:
        if col not in df.columns:
            continue
        scored = df.dropna(subset=[col])

        fig, ax = plt.subplots(figsize=(8, 5))
        for y, color in ((first_year, first_color), (last_year, last_color)):
            vals = scored.loc[scored["conference_year"] == y, col].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            density = np.histogram(vals, bins=fine_edges, density=True)[0]
            ax.fill_between(fine_mids, density, step="mid", color=color, alpha=0.45, linewidth=0)
            ax.step(fine_mids, density, where="mid", color=color, linewidth=1.5, label=str(y))

        ax.set_xlabel("Score  P(human)")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.set_title(f"{label} — {first_year} vs. {last_year} score distribution — {tag}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(title="Year", frameon=False, loc="upper right")
        fig.tight_layout()

        out_path = out_dir / f"year_distribution_first_vs_last_{col}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot → {out_path}")


if __name__ == "__main__":
    main()

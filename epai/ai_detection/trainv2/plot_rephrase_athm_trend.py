r"""AT-HM trend plot for the rephrase-robustness analysis: one slope per
main-table scorer (RADAR excluded, see ``rephrase_common.py``) from its
classic (without-rephrase) AT-HM to its rephrased (with-rephrase) AT-HM,
pooled over MG-GB and MG-Sci (the ``mean`` corpus row of
``run_rephrase_transfer.py``'s ``hm_delta.csv``).

Hollow marker = classic, filled marker = rephrased, joined by a line, all in
that scorer's own colour (``run_best_zeroshot_performance._ATHM_BARS_COLOR``,
the same palette Figure~\ref{fig:general-results-scatter}'s AT-HM bars use
and :mod:`plot_rephrase_center_of_mass` reuses) -- a downward slope means
rephrasing degrades that scorer, a flat one means it doesn't. Replaces
``tab:rephrase`` (the numeric table) as this analysis's second figure; each
point is labelled with its AT-HM value since no table backs the numbers
anymore.

Output (under ``--out-dir``, default ``results/trainv2/rephrase_transfer/``):
    athm_trend.pdf

Run:
    uv run python epai/ai_detection/trainv2/plot_rephrase_athm_trend.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epai.ai_detection.trainv2.rephrase_common import MAIN_TABLE_SCORERS
from epai.ai_detection.trainv2.run_best_zeroshot_performance import (
    _ATHM_BARS_COLOR,
    _LEGEND_LABEL,
)
from epai.ai_detection.trainv2.run_rephrase_transfer import MEAN_CORPUS

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CSV = BASE_DIR / "results" / "trainv2" / "rephrase_transfer" / "hm_delta.csv"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "rephrase_transfer"

X_CLASSIC, X_REPHRASED = 0, 1
X_TICKS = [X_CLASSIC, X_REPHRASED]
X_LABELS = ["Classic\n(w/o rephrase)", "Rephrased\n(w/ rephrase)"]


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    for r in rows:
        color = r["color"]
        ax.plot([X_CLASSIC, X_REPHRASED], [r["athm_without"], r["athm_with"]],
                color=color, lw=1.6, zorder=1, label=r["label"])
        ax.scatter([X_CLASSIC], [r["athm_without"]], facecolors="none",
                   edgecolors=color, s=55, linewidths=1.6, zorder=2)
        ax.scatter([X_REPHRASED], [r["athm_with"]], facecolors=color,
                   edgecolors=color, s=55, zorder=2)
        ax.annotate(f"{r['athm_without']:.3f}", (X_CLASSIC, r["athm_without"]),
                    xytext=(-6, 4), textcoords="offset points", fontsize=7,
                    ha="right", color=color)
        ax.annotate(f"{r['athm_with']:.3f}", (X_REPHRASED, r["athm_with"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=7,
                    ha="left", color=color)

    ax.set_xticks(X_TICKS)
    ax.set_xticklabels(X_LABELS)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("AT-HM")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8,
              frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[(df.corpus == MEAN_CORPUS) & (df.fpr == args.fpr)]

    rows = []
    for algo, column, _ in MAIN_TABLE_SCORERS:
        r = df[df.scorer == column].iloc[0]
        rows.append({
            "athm_without": float(r["athm_without"]), "athm_with": float(r["athm_with"]),
            "color": _ATHM_BARS_COLOR[algo], "label": _LEGEND_LABEL[algo],
        })

    plot(rows, args.out_dir / "athm_trend.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

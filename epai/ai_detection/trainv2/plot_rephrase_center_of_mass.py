r"""Center-of-mass plot for the rephrase-robustness analysis: one small
panel per main-table scorer (RADAR excluded, see ``rephrase_common.py``),
showing where each of the four classes' mean score sits, pooled over MG-GB
and MG-Sci (the ``mean`` corpus row of ``run_rephrase_transfer.py``'s
``center_of_mass.csv``).

Two horizontal dumbbells per panel -- Human -> Rephrased Human, and
Machine -> Rephrased Machine -- hollow marker at the original mean, filled
at the rephrased one, joined by a line, all in that scorer's own colour
(``run_best_zeroshot_performance._ATHM_BARS_COLOR``, the same palette
Figure~\ref{fig:general-results-scatter}'s AT-HM bars use, so a scorer's
colour here matches every other figure in the paper). A robust scorer's two
dumbbells stay short; a scorer that fails under rephrasing shows a long
Human dumbbell (judge family) or a long Machine one (perplexity family).

Output (under ``--out-dir``, default ``results/trainv2/rephrase_transfer/``):
    center_of_mass.pdf

Run:
    uv run python epai/ai_detection/trainv2/plot_rephrase_center_of_mass.py
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
DEFAULT_CSV = BASE_DIR / "results" / "trainv2" / "rephrase_transfer" / "center_of_mass.csv"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "rephrase_transfer"

#: (row label, original class, rephrased class), bottom to top.
PAIRS = [("Machine", "ai", "ai_rephrased"), ("Human", "human", "human_rephrased")]


def plot(means: dict[str, dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(MAIN_TABLE_SCORERS)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6), squeeze=False)
    axes = axes[0]

    for ax, (algo, column, machine_is_high) in zip(axes, MAIN_TABLE_SCORERS):
        color = _ATHM_BARS_COLOR[algo]
        m = means[column]
        for y, (_, orig_cls, reph_cls) in enumerate(PAIRS):
            x0, x1 = m[orig_cls], m[reph_cls]
            ax.plot([x0, x1], [y, y], color=color, lw=3.2, zorder=1)
            ax.scatter([x0], [y], facecolors="none", edgecolors=color, s=220,
                       linewidths=3.2, zorder=2)
            ax.scatter([x1], [y], facecolors=color, edgecolors=color, s=220, zorder=2)
        ax.set_yticks(range(len(PAIRS)))
        ax.set_yticklabels([label for label, _, _ in PAIRS], fontsize=20)
        ax.set_ylim(-0.6, len(PAIRS) - 0.4)
        arrow = r"$\uparrow$AI" if machine_is_high else r"$\downarrow$AI"
        ax.set_xlabel(f"Score ({arrow})", fontsize=20)
        ax.set_title(_LEGEND_LABEL[algo], fontsize=23)
        ax.grid(True, axis="x", lw=0.4, alpha=0.4)
        ax.tick_params(labelsize=18)

    hollow = plt.Line2D([0], [0], marker="o", color="0.3", linestyle="-",
                        markerfacecolor="none", markersize=16, label="original")
    filled = plt.Line2D([0], [0], marker="o", color="0.3", linestyle="-",
                        markerfacecolor="0.3", markersize=16, label="rephrased")
    fig.legend(handles=[hollow, filled], loc="lower center", ncol=2,
               fontsize=20, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df.corpus == MEAN_CORPUS]

    means: dict[str, dict[str, float]] = {}
    for _, column, _ in MAIN_TABLE_SCORERS:
        sub = df[df.scorer == column]
        means[column] = {r["class"]: r["mean"] for _, r in sub.iterrows()}

    plot(means, args.out_dir / "center_of_mass.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

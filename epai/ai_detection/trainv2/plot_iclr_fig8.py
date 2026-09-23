r"""ICLR 2027 paper, Fig. 8 (fig:judge_generator_athm) -- judge x generator
AT-HM heatmap, parsed readout, redrawn to read at the ~0.45-linewidth slot it
has in the paper: no colorbar (every cell carries its value), the "Judge" /
"Generator" axis names moved into the empty top-left corner, short generator
labels (full model names are on the rows), larger type.

Cells: AT-HM of human vs. ONE generator's text on one split
(run_judge_by_generator.compute, oracle TPR@1%FPR). "Mean" column: AT-HM of
the judge's mean AUROC and mean TPR over all its cells (the Fig. 2
convention). Outlined cells: judge and generator are the same model.

Outputs (results/trainv2/iclr_fig8/):
    judge_generator_athm.{pdf,png}   same name as in latex figs/
    matrix.csv

Run:
    uv run python epai/ai_detection/trainv2/plot_iclr_fig8.py
    uv run python epai/ai_detection/trainv2/plot_iclr_fig8.py --recompute
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from epai.ai_detection.trainv2.core import goodness
from epai.ai_detection.trainv2.plots import _INK, _MUTED
from epai.ai_detection.trainv2.run_judge_by_generator import compute
from epai.ai_detection.trainv2.splits import display as split_display
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
OUT_DIR = BASE_DIR / "results" / "trainv2" / "iclr_fig8"
FPR = 0.01
SPLITS = ["gb_orig", "mg_gb", "mg_sci"]

# ---------------------------------------------------------------- design knobs
FIGSIZE = (5.0, 3.1)
FONT_CELL = 12
FONT_ROW = 12
FONT_COL = 11.5
COL_ROTATION = 40
FONT_GROUP = 11.5
FONT_CORNER = 11.5
CMAP = "Blues"
DROP_LEADING_ZERO = True     # ".92" instead of "0.92" -- saves cell width
OUTLINE_LW = 2.0

# judge column -> row label (row order = mean AT-HM, computed)
JUDGES = {
    "llm_judge_score_gpt56luna_v3": "gpt-5.6-luna",
    "llm_judge_score_gemma4_31b_v3": "gemma-4-31b",
    "llm_judge_score_llama33_70b_v3": "llama-3.3-70b",
    "llm_judge_score_gptoss_120b_v3": "gpt-oss-120b",
}
# gen_model -> short column label (full names are in the row labels)
GEN_LABEL = {
    "gpt35t": "gpt-3.5t",
    "gemma431b": "gemma",
    "gpt56luna": "luna",
    "gptoss120b": "gpt-oss",
    "llama3370b": "llama",
}


def _fmt(v: float) -> str:
    s = f"{v:.2f}"
    return s[1:] if DROP_LEADING_ZERO and s.startswith("0") else s


def plot(rows: pd.DataFrame, path_stem: Path) -> None:
    rows = rows.assign(athm=[goodness(a, t) for a, t in zip(rows.auroc, rows.tpr_at_fpr)])
    judge_of = {c.removeprefix("llm_judge_score_").replace("_v3", ""): lbl
                for c, lbl in JUDGES.items()}
    mean = {j: goodness(d.auroc.mean(), d.tpr_at_fpr.mean())
            for j, d in rows.groupby("judge")}
    judges = sorted(judge_of, key=lambda j: -mean[j])

    # Column order per split: the same-family generator goes at its judge's
    # row position, so self-family cells (outlined) fall on the diagonal;
    # any generator with no matching judge is appended after, alphabetically.
    cols = []
    for s in SPLITS:
        gens = sorted(rows[rows.split == s].gen_model.unique())
        self_map = {r.judge: r.gen_model for r in
                    rows[(rows.split == s) & rows.self_family].itertuples()}
        diagonal = [self_map[j] for j in judges if j in self_map]
        leftover = sorted(set(gens) - set(diagonal))
        cols += [(s, g) for g in diagonal + leftover]
    by = rows.set_index(["split", "judge", "gen_model"])
    M = np.array([[by.at[(s, j, g), "athm"] for s, g in cols] + [mean[j]] for j in judges])
    selfc = [[bool(by.at[(s, j, g), "self_family"]) for s, g in cols] + [False] for j in judges]
    n_r, n_c = M.shape

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    # the Mean column sits one half-cell apart from the per-split cells
    xs = np.array([*range(n_c - 1), n_c - 1 + 0.25])
    cmap = plt.get_cmap(CMAP)
    for i in range(n_r):
        for k in range(n_c):
            v = M[i, k]
            ax.add_patch(Rectangle((xs[k] - 0.5, i - 0.5), 1, 1, facecolor=cmap(v),
                                   edgecolor="white", lw=1.2))
            ax.text(xs[k], i, _fmt(v), ha="center", va="center", fontsize=FONT_CELL,
                    color="white" if v > 0.55 else _INK,
                    fontweight="bold" if k == n_c - 1 else "normal")
            if selfc[i][k]:
                ax.add_patch(Rectangle((xs[k] - 0.46, i - 0.46), 0.92, 0.92, fill=False,
                                       edgecolor=_INK, lw=OUTLINE_LW, zorder=3))

    ax.set_xlim(-0.5, xs[-1] + 0.5)
    ax.set_ylim(n_r - 0.5, -0.5)
    ax.set_aspect("auto")
    ax.set_yticks(range(n_r))
    ax.set_yticklabels([judge_of[j] for j in judges], fontsize=FONT_ROW, color=_INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([GEN_LABEL.get(g, g) for _, g in cols] + ["Mean"],
                       fontsize=FONT_COL, color=_INK, rotation=COL_ROTATION,
                       ha="right", rotation_mode="anchor")
    ax.tick_params(length=0, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # split brackets above the columns
    for s in SPLITS:
        ks = [k for k, (ss, _) in enumerate(cols) if ss == s]
        lo, hi = ks[0] - 0.42, ks[-1] + 0.42
        y = -0.62
        ax.plot([lo, lo, hi, hi], [y + 0.08, y, y, y + 0.08], color=_MUTED, lw=1.0,
                clip_on=False)
        ax.text((lo + hi) / 2, y - 0.08, split_display(s), ha="center", va="bottom",
                fontsize=FONT_GROUP, color=_INK, fontweight="bold")

    # axis names in the dead corners instead of rotated side labels
    ax.text(-0.12, 1.0, "Generator →\nJudge ↓", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=FONT_CORNER, color=_MUTED, style="italic",
            multialignment="right", linespacing=1.3)

    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = path_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=200, facecolor="white")
        print(f"wrote {p}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--recompute", action="store_true",
                    help="rescore the matrix from the parquets")
    args = ap.parse_args()

    csv_path = OUT_DIR / "matrix.csv"
    if args.recompute or not csv_path.exists():
        rows = compute(resolve_splits(SPLITS, PARQUET_DIR), list(JUDGES), FPR, "human")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")
    plot(pd.read_csv(csv_path), OUT_DIR / "judge_generator_athm")


if __name__ == "__main__":
    main()

r"""ICLR 2027 paper, Fig. 7 (fig:rephrase) -- AUROC vs. TPR@1%FPR before and
after rephrasing, redrawn like Fig. 2/3 (plot_iclr_fig2_fig3.py), larger
fonts. To save height, the legend (detectors only) sits INSIDE the empty
top-left of the axes, with the split shapes as its last rows; "chance" is
written next to its marker.

    before (hollow)  MG-Reph-GB / MG-Reph-Sci originals only: human vs.
                     gpt-5.6-luna text (no other generator, so the arrow
                     isolates rephrasing), cached in rephrase_before.csv
    after  (filled)  MG-Reph-GB / MG-Reph-Sci: human + rephrased human vs.
                     gpt-5.6-luna text + its rephrase (rephrase_common's four
                     classes), cached in rephrase_after.csv

Per split: one small marker (shape = split); big circle = mean of the two
splits' AUROC and TPR, AT-HM taken off that mean pair (the Fig. 2
convention). Hollow -> filled mean joined by a line. RADAR is left out (its
AT-HM floors to 0 on MG-GB/MG-Sci, see rephrase_common.MAIN_TABLE_SCORERS).

Outputs (results/trainv2/iclr_fig7/):
    rephrase_best_zeroshot_1pct.{pdf,png}   same name as in latex figs/
    rephrase_before.csv, rephrase_after.csv

Run:
    uv run python epai/ai_detection/trainv2/plot_iclr_fig7.py
    uv run python epai/ai_detection/trainv2/plot_iclr_fig7.py --recompute
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from epai.ai_detection.trainv2.core import goodness
from epai.ai_detection.trainv2.plots import _INK, _MUTED, _SURFACE, _distort, _goodness_field, _tint
from epai.ai_detection.trainv2.rephrase_common import (
    AI_REPHRASE_GEN,
    CORPUS_SOURCES,
    GEN_COL,
    HUMAN,
    PARQUET_DIR,
    REPHRASE_GEN,
)
from epai.ai_detection.trainv2.run_best_zeroshot_performance import _CHANCE_GRAY
from epai.ai_detection.trainv2.splits import display as split_display
from epai.ai_detection.trainv2.splits import marker as split_marker
from epai.utils.cd_core import auroc_score, oriented_scores, roc_point_at_fpr

BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / "results" / "trainv2" / "iclr_fig7"
FPR = 0.01
SPLITS = ["mg_gb", "mg_sci"]

# ---------------------------------------------------------------- design knobs
FIGSIZE = (5.4, 3.9)
FONT_LEGEND = 12
FONT_LABEL = 14
FONT_TICK = 12
FONT_ISO = 10
LEGEND_ALPHA = 0.85          # white box behind the in-axes legend
MARKER_SPLIT = 42
MARKER_MEAN = 8.5
KX = KY = 1.7
X_LABEL = "AUROC (power-warped)"
Y_LABEL = f"TPR@{FPR * 100:g}%FPR (power-warped)"
# iso-AT-HM contour label -> (AUROC, TPR) it is placed nearest to; kept out of
# the top-left, where the legend sits. Nudge these if a label lands badly.
CONTOUR_LABEL_AT = {0.2: (0.56, 0.25), 0.4: (0.75, 0.33), 0.6: (0.88, 0.62), 0.8: (0.93, 0.90)}

# score column -> (legend name, colour, machine_is_high, bold)
DETECTORS = {
    "fusionjudge_score": ("FusionJudge (ours)", "#4a3aa7", False, True),
    "logjudge_ord2_gpt56luna_v3": ("LogJudge (ours)", "#1d6fe0", False, True),
    "llm_judge_score_gpt56luna_v3": ("BaseJudge", "#4f77a5", False, False),
    "fastdetect_crit_llama31_8b": ("Fast-DetectGPT", "#eda100", True, False),
    "binoculars_score_llama31_8b": ("Binoculars", "#1baf7a", False, False),
}
SHOW_ATHM = True             # "0.74 → 0.37" after each legend name


def compute(phase: str) -> pd.DataFrame:
    """``before``: human vs. gpt-5.6-luna originals only; ``after``: adds both
    rephrased classes. Both read from the MG-Reph parquet, so the two phases
    share the same documents and the same (single) generator."""
    rows = []
    for split, src in zip(SPLITS, CORPUS_SOURCES.values()):
        df = pd.read_parquet(PARQUET_DIR / src["reph"], columns=[GEN_COL, *DETECTORS])
        keep = {HUMAN, src["rewrite_gen"]}
        if phase == "after":
            keep |= {REPHRASE_GEN, AI_REPHRASE_GEN}
        df = df[df[GEN_COL].isin(keep)]
        for col, (_, _, machine_is_high, _) in DETECTORS.items():
            sub = df[[GEN_COL, col]].dropna()
            y = (~sub[GEN_COL].isin([HUMAN, REPHRASE_GEN])).to_numpy().astype(int)
            s = oriented_scores(sub[col].to_numpy(dtype=float), machine_is_high)
            tpr, _ = roc_point_at_fpr(y, s, FPR)
            rows.append({"column": col, "split": split, "n": len(sub),
                         "auroc": float(auroc_score(y, s)), "tpr_at_fpr": float(tpr)})
    return pd.DataFrame(rows)


def plot(before: pd.DataFrame, after: pd.DataFrame, path_stem: Path) -> None:
    def pts(df, col):
        d = df[df["column"] == col].set_index("split")
        return {s: (d.at[s, "auroc"], d.at[s, "tpr_at_fpr"]) for s in SPLITS}

    data = {col: {"before": pts(before, col), "after": pts(after, col)} for col in DETECTORS}
    means = {col: {k: tuple(np.mean(list(v.values()), axis=0)) for k, v in d.items()}
             for col, d in data.items()}

    all_auroc = [a for d in data.values() for v in d.values() for a, _ in v.values()]
    x_lo = min(0.45, min(all_auroc) - 0.12)
    x_hi, y_lo, y_hi = 1.02, -0.25, 1.04

    def wx(v):
        return _distort(v, x_lo, x_hi, KX)

    def wy(v):
        return _distort(v, y_lo, y_hi, KY)

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.axvline(wx(0.5), color=_CHANCE_GRAY, lw=1.0, ls=":", alpha=0.6, zorder=0)
    ax.axhline(wy(FPR), color=_CHANCE_GRAY, lw=1.0, ls=":", alpha=0.6, zorder=0)
    ax.scatter([wx(0.5)], [wy(FPR)], s=120, color=_CHANCE_GRAY, marker="x", lw=1.8,
               alpha=0.6, zorder=0)

    for col, (_, color, _, _) in DETECTORS.items():
        for phase, hollow in (("before", True), ("after", False)):
            for s, (a, t) in data[col][phase].items():
                ax.scatter([wx(a)], [wy(t)], s=MARKER_SPLIT, marker=split_marker(s),
                           facecolors="none" if hollow else _tint(color),
                           edgecolors=_tint(color, 0.55) if hollow else _INK,
                           linewidths=0.6 if hollow else 0.8, zorder=3)
        (xb, yb), (xa, ya) = means[col]["before"], means[col]["after"]
        ax.plot([wx(xb), wx(xa)], [wy(yb), wy(ya)], color=color, lw=1.4, alpha=0.8, zorder=4)
        ax.plot([wx(xb)], [wy(yb)], "o", ms=MARKER_MEAN, mfc="none", mec=color, mew=1.5, zorder=5)
        ax.plot([wx(xa)], [wy(ya)], "o", ms=MARKER_MEAN, mfc=color, mec=_INK, mew=1.1, zorder=5)

    GX, GY = np.meshgrid(np.linspace(x_lo, 1.0, 200), np.linspace(-0.04, 1.0, 200))
    GZ = _goodness_field(GX, GY)
    ax.contourf(wx(GX), wy(GY), GZ, levels=np.linspace(0, 1, 11), cmap="Greys",
                alpha=0.30, zorder=-5)
    cs = ax.contour(wx(GX), wy(GY), GZ, levels=[0.2, 0.4, 0.6, 0.8],
                    colors="#8a8a8a", linewidths=0.6, alpha=0.8, zorder=-4)
    ax.clabel(cs, fmt=lambda v: f"{v:.1f}", fontsize=FONT_ISO, colors="#6a6a6a", inline=True,
              manual=[(wx(a), wy(t)) for a, t in CONTOUR_LABEL_AT.values()])
    ax.annotate("chance", (wx(0.5), wy(FPR)), xytext=(7, 4), textcoords="offset points",
                ha="left", va="bottom", fontsize=FONT_ISO, color=_CHANCE_GRAY)

    ax.set_xlim(wx(x_lo), wx(x_hi))
    ax.set_ylim(wy(y_lo), wy(y_hi))
    ax.set_xlabel(X_LABEL, fontsize=FONT_LABEL, color=_INK)
    ax.set_ylabel(Y_LABEL, fontsize=FONT_LABEL, color=_INK)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.tick_params(labelsize=FONT_TICK, colors=_MUTED)
    x_ticks = [t for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0) if t >= x_lo]
    ax.set_xticks([wx(t) for t in x_ticks])
    ax.set_xticklabels([f"{t:g}" for t in x_ticks])
    y_ticks = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
    ax.set_yticks([wy(t) for t in y_ticks])
    ax.set_yticklabels([f"{t:g}" for t in y_ticks])

    # Legend in the empty top-left: one entry per detector (hollow -> filled
    # pair, before -> after AT-HM), ranked by before-AT-HM; then split shapes.
    athm = {col: {k: goodness(*v, "hmean") for k, v in m.items()} for col, m in means.items()}
    ranked = sorted(DETECTORS, key=lambda c: -athm[c]["before"])
    handles, labels = [], []
    for col in ranked:
        name, color, _, _ = DETECTORS[col]
        handles.append((plt.Line2D([], [], marker="o", ls="", ms=7, mfc="none", mec=color, mew=1.3),
                        plt.Line2D([], [], marker="o", ls="", ms=7, mfc=color, mec=_INK, mew=0.8)))
        b, a = athm[col]["before"], athm[col]["after"]
        # Drop the leading zero (same convention as plot_iclr_fig8.py's
        # DROP_LEADING_ZERO) and the "(ours)" tag (redundant with the bold
        # weight already marking those entries) -- keeps the box narrow.
        fmt = lambda v: f"{v:.2f}"[1:]
        short_name = name.replace(" (ours)", "")
        labels.append(f"{short_name} {fmt(b)} → {fmt(a)}" if SHOW_ATHM else short_name)
    for s in SPLITS:
        handles.append(plt.Line2D([], [], marker=split_marker(s), ls="", ms=7, color="#8a8a8a"))
        labels.append(split_display(s))
    legend = ax.legend(handles, labels, loc="upper left", fontsize=FONT_LEGEND,
                       frameon=True, framealpha=LEGEND_ALPHA, facecolor="white",
                       edgecolor="none", borderaxespad=0.3, handlelength=1.2,
                       handletextpad=0.3, labelspacing=0.3,
                       handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)})
    bold = {labels[i] for i, col in enumerate(ranked) if DETECTORS[col][3]}
    for txt in legend.get_texts():
        if txt.get_text() in bold:
            txt.set_fontweight("bold")

    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = path_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=200, facecolor="white")
        print(f"wrote {p}")
    plt.close(fig)

    for col in ranked:
        print(f"  {DETECTORS[col][0]:18s} AT-HM {athm[col]['before']:.3f} -> {athm[col]['after']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--recompute", action="store_true",
                    help="rescore the before/after points from the MG-Reph parquets")
    args = ap.parse_args()

    frames = {}
    for phase in ("before", "after"):
        csv = OUT_DIR / f"rephrase_{phase}.csv"
        if args.recompute or not csv.exists():
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            compute(phase).to_csv(csv, index=False)
            print(f"wrote {csv}")
        frames[phase] = pd.read_csv(csv)
    plot(frames["before"], frames["after"], OUT_DIR / "rephrase_best_zeroshot_1pct")


if __name__ == "__main__":
    main()

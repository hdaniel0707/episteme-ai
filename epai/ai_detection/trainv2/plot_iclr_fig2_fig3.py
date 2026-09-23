r"""ICLR 2027 paper, Fig. 2 and Fig. 3 -- the two side-by-side AUROC-vs-TPR@1%FPR
scatters, redrawn for a half-column slot: larger fonts, and both figures at
the same size so the two panels line up in the paper. To save height the
legend (detectors, then split shapes) sits INSIDE the empty top-left of the
axes and "chance" is written next to its marker.

    Fig. 2  (fig:main-results)   best zero-shot detectors (LogJudge, FusionJudge,
                                 BaseJudge, Fast-DetectGPT, Binoculars, RADAR)
    Fig. 3  (fig:judge-backbone) the five parsed-judge (BaseJudge) backbones

Same marks and encoding as :func:`plots.zeroshot_scatter` (colour = detector,
marker shape = split, big circle = four-split mean, grey AT-HM iso-lines, both
axes power-warped); only the layout/typography differs, and every design knob
sits in the constants block below.

Inputs:
    Fig. 2  results/trainv2/best_zeroshot_performance/performance.csv
            (written by run_best_zeroshot_performance.py)
    Fig. 3  results/trainv2/iclr_fig2_fig3/judge_backbone_performance.csv --
            scored here from the core parquets on first run (same compute() as
            run_zeroshot_performance.py), reused afterwards; --recompute rescores.

Outputs (results/trainv2/iclr_fig2_fig3/):
    best_zeroshot_1pct.{pdf,png}    (Fig. 2, same name as in latex figs/)
    judge_backbone_1pct.{pdf,png}   (Fig. 3, same name as in latex figs/)

Run:
    uv run python epai/ai_detection/trainv2/plot_iclr_fig2_fig3.py
    uv run python epai/ai_detection/trainv2/plot_iclr_fig2_fig3.py --recompute
    uv run python epai/ai_detection/trainv2/plot_iclr_fig2_fig3.py --gemma
        Fig. 2 + gemma-4-31b's BaseJudge / LogJudge / FusionJudge ->
        best_zeroshot_gemma_1pct.{pdf,png}, scores cached in
        gemma_judge_performance.csv (needs score_fusionjudge_gemma4_31b_v3.py first)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from epai.ai_detection.trainv2.core import goodness, rows_at_fpr
from epai.ai_detection.trainv2.detector_names import is_judge_family
from epai.ai_detection.trainv2.plots import (
    _INK,
    _MUTED,
    _SURFACE,
    _distort,
    _goodness_field,
    _tint,
    _variant_palette_hollow,
)
from epai.ai_detection.trainv2.run_best_zeroshot_performance import (
    _CHANCE_GRAY,
    _COLOR_OVERRIDE,
    _EMPHASIZE,
    _LEGEND_LABEL,
    _LIGHT_BORDER,
)
from epai.ai_detection.trainv2.run_zeroshot_performance import CSV_FIELDS, FAMILIES, compute
from epai.ai_detection.trainv2.splits import DEFAULT as CORE_SPLITS
from epai.ai_detection.trainv2.splits import display as split_display
from epai.ai_detection.trainv2.splits import marker as split_marker
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
FIG2_CSV = BASE_DIR / "results" / "trainv2" / "best_zeroshot_performance" / "performance.csv"
OUT_DIR = BASE_DIR / "results" / "trainv2" / "iclr_fig2_fig3"
FPR = 0.01

# ---------------------------------------------------------------- design knobs
# Both figures share one size so the two half-width panels scale identically.
# In the paper each is shown at ~2.7 in wide, so fonts render at roughly
# FONT * 2.7 / FIGSIZE[0] pt.
FIGSIZE = (4.8, 4.0)
FONT_LEGEND = 10
FONT_LABEL = 11
FONT_TICK = 9.5
FONT_ISO = 8                 # AT-HM iso-line labels
LEGEND_ALPHA = 0.85          # white box behind the in-axes legend
MARKER_SPLIT = 42            # per-split points (scatter area, pt^2)
MARKER_MEAN = 8.5            # four-split mean circle (marker size, pt)
KX = KY = 1.7                # power-warp exponent per axis
X_LABEL = "AUROC (power-warped)"
Y_LABEL = f"TPR@{FPR * 100:g}%FPR (power-warped)"
# iso-AT-HM contour label -> (AUROC, TPR) it is placed nearest to; kept out of
# the top-left, where the legend sits. Nudge these if a label lands badly.
CONTOUR_LABEL_AT = {0.2: (0.80, 0.13), 0.4: (0.75, 0.33), 0.6: (0.84, 0.66), 0.8: (0.93, 0.90)}

# Fig. 3: parsed-judge columns -> legend name and colour. gpt-5.6-luna keeps
# the blue it has in Fig. 2 (BaseJudge there is the same score column).
FIG3_JUDGES = {
    "llm_judge_score_gpt56luna_v3": ("gpt-5.6-luna", "#1d6fe0"),
    "llm_judge_score_gemma4_31b_v3": ("gemma-4-31b", "#eb6834"),
    "llm_judge_score_llama33_70b_v3": ("llama-3.3-70b", "#1baf7a"),
    "llm_judge_score_gptoss_120b_v3": ("gpt-oss-120b", "#e87ba4"),
    "llm_judge_score_qwen3_14b_v3": ("qwen3-14b", "#eda100"),
}
SHOW_ATHM = True             # append the mean AT-HM (".74") to each legend name


def scatter_inset_legend(
    rows: list[dict], path_stem: Path, *, key: str,
    color: dict[str, str], label: dict[str, str],
    variant_suffix: str = "_ord2",
    light_border: frozenset[str] = frozenset(),
    emphasize: frozenset[str] = frozenset(),
    figsize: tuple[float, float] = FIGSIZE,
) -> None:
    rows = [r for r in rows if r["split"] == "mean" or r["split"] in CORE_SPLITS]
    names = sorted({r[key] for r in rows})
    splits = [s for s in CORE_SPLITS if s in {r["split"] for r in rows}]
    base_of, paired_base, _ = _variant_palette_hollow(names, suffix=variant_suffix)
    hue = {v: color[v] for v in names}

    aurocs = [r["auroc"] for r in rows if np.isfinite(r["auroc"])]
    x_lo = min(0.45, min(aurocs) - 0.12)
    x_hi, y_lo, y_hi = 1.02, -0.25, 1.04

    def wx(v):
        return _distort(v, x_lo, x_hi, KX)

    def wy(v):
        return _distort(v, y_lo, y_hi, KY)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # chance: AUROC 0.5, oracle TPR = the FPR level
    ax.axvline(wx(0.5), color=_CHANCE_GRAY, lw=1.0, ls=":", alpha=0.6, zorder=0)
    ax.axhline(wy(FPR), color=_CHANCE_GRAY, lw=1.0, ls=":", alpha=0.6, zorder=0)
    ax.scatter([wx(0.5)], [wy(FPR)], s=120, color=_CHANCE_GRAY, marker="x", lw=1.8,
               alpha=0.6, zorder=0)

    for r in rows:
        if r["split"] == "mean" or not np.isfinite(r["tpr_at_fpr"]):
            continue
        v = r[key]
        hollow = v in paired_base
        if hollow:
            edge, lw = _tint(hue[v], 0.55), 0.6
        elif v in light_border:
            edge, lw = _INK, 0.5
        else:
            edge, lw = _INK, 1.0
        ax.scatter([wx(r["auroc"])], [wy(r["tpr_at_fpr"])], s=MARKER_SPLIT,
                   marker=split_marker(r["split"]),
                   facecolors="none" if hollow else _tint(hue[v]),
                   edgecolors=edge, linewidths=lw, zorder=3.5 if v in emphasize else 3)

    mean_xy = {r[key]: (r["auroc"], r["tpr_at_fpr"]) for r in rows
               if r["split"] == "mean" and np.isfinite(r["tpr_at_fpr"])}
    for v, b in base_of.items():
        if v in mean_xy and b in mean_xy:
            (xv, yv), (xb, yb) = mean_xy[v], mean_xy[b]
            ax.plot([wx(xb), wx(xv)], [wy(yb), wy(yv)], color=hue[b], lw=1.1,
                    alpha=0.6, zorder=4)
    for v, (x0, y0) in mean_xy.items():
        hollow = v in paired_base
        if hollow:
            mec, mew = hue[v], 1.2
        elif v in light_border:
            mec, mew = _INK, 0.7
        else:
            mec, mew = _INK, 1.3
        ax.plot([wx(x0)], [wy(y0)], "o", ms=MARKER_MEAN,
                mfc="none" if hollow else hue[v], mec=mec, mew=mew,
                zorder=5.5 if v in emphasize else 5)

    # AT-HM background field + iso-lines, warped like the points
    GX, GY = np.meshgrid(np.linspace(x_lo, 1.0, 200), np.linspace(-0.04, 1.0, 200))
    GZ = _goodness_field(GX, GY)
    ax.contourf(wx(GX), wy(GY), GZ, levels=np.linspace(0, 1, 11), cmap="Greys",
                alpha=0.30, zorder=-5)
    cs = ax.contour(wx(GX), wy(GY), GZ, levels=[0.2, 0.4, 0.6, 0.8],
                    colors="#8a8a8a", linewidths=0.6, alpha=0.8, zorder=-4)
    ax.clabel(cs, fmt=lambda v: f"{v:.1f}", fontsize=FONT_ISO, colors="#6a6a6a",
              inline=True, manual=[(wx(a), wy(t)) for a, t in CONTOUR_LABEL_AT.values()])
    # below the marker: the space beside it is crowded by near-chance points
    ax.annotate("chance", (wx(0.5), wy(FPR)), xytext=(0, -8), textcoords="offset points",
                ha="center", va="top", fontsize=FONT_ISO, color=_CHANCE_GRAY)

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

    # Legend in the empty top-left, one column: detectors ranked by mean
    # AT-HM, then the split shapes.
    def at_hm(v):
        return goodness(*mean_xy[v], "hmean") if v in mean_xy else -1.0

    ranked = sorted(names, key=lambda v: -at_hm(v))
    det = [plt.Line2D([], [], marker="o", ls="", ms=7,
                      mfc="none" if v in paired_base else hue[v],
                      mec=hue[v] if v in paired_base else "#222222", mew=1.0,
                      label=f"{label[v]} {at_hm(v):.2f}".replace(" 0.", " .")
                      if SHOW_ATHM else label[v]) for v in ranked]
    shp = [plt.Line2D([], [], marker=split_marker(s), ls="", ms=7, color="#8a8a8a",
                      label=split_display(s)) for s in splits]
    legend = ax.legend(handles=det + shp, loc="upper left", fontsize=FONT_LEGEND,
                       frameon=True, framealpha=LEGEND_ALPHA, facecolor="white",
                       edgecolor="none", borderaxespad=0.3, handlelength=1.0,
                       handletextpad=0.4, labelspacing=0.3)
    for v, txt in zip(ranked, legend.get_texts()):
        if is_judge_family(v):
            txt.set_fontweight("bold")

    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = path_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=200, facecolor="white")
        print(f"wrote {p}")
    plt.close(fig)


def fig2() -> None:
    rows = rows_at_fpr(pd.read_csv(FIG2_CSV).to_dict("records"), FPR)
    scatter_inset_legend(
        rows, OUT_DIR / "best_zeroshot_1pct", key="detector",
        color=_COLOR_OVERRIDE, label=_LEGEND_LABEL,
        light_border=_LIGHT_BORDER, emphasize=_EMPHASIZE,
    )


#: Fig. 2 + gemma variant: gemma-4-31b's three judge readouts next to
#: gpt-5.6-luna's. Orange = gemma (its Fig. 3 colour), darker for FusionJudge.
GEMMA_COLS = ["llm_judge_score_gemma4_31b_v3", "logjudge_ord2_gemma4_31b_v3",
              "fusionjudge_gemma4_31b_v3_score"]
GEMMA_COLOR = {"judge/gemma4_31b": "#eb6834", "judge/gemma4_31b_ord2": "#eb6834",
               "fusionjudge/gemma4_31b_v3": "#a8431a"}
GEMMA_LABEL = {
    "fusionjudge/gpt56luna_v3": "FusionJudge (luna)",
    "judge/gpt56luna_ord2": "LogJudge (luna)",
    "judge/gpt56luna": "BaseJudge (luna)",
    "fusionjudge/gemma4_31b_v3": "FusionJudge (gemma)",
    "judge/gemma4_31b_ord2": "LogJudge (gemma)",
    "judge/gemma4_31b": "BaseJudge (gemma)",
}


def fig2_gemma(recompute: bool) -> None:
    """Fig. 2 plus the three gemma-4-31b judge detectors, written under a new
    name (best_zeroshot_gemma_1pct) -- the paper's Fig. 2 is left untouched."""
    csv_path = OUT_DIR / "gemma_judge_performance.csv"
    if recompute or not csv_path.exists():
        splits = resolve_splits(list(CORE_SPLITS), PARQUET_DIR)
        fams = {k: FAMILIES[k] for k in ("judge", "fusionjudge")}
        rows = compute(splits, fams, GEMMA_COLS, [FPR], "human")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")
    gemma = pd.read_csv(csv_path)
    gemma.insert(0, "detector", gemma["family"] + "/" + gemma["version"])
    base = pd.read_csv(FIG2_CSV)
    rows = rows_at_fpr(pd.concat([base, gemma], ignore_index=True).to_dict("records"), FPR)
    scatter_inset_legend(
        rows, OUT_DIR / "best_zeroshot_gemma_1pct", key="detector",
        color={**_COLOR_OVERRIDE, **GEMMA_COLOR}, label={**_LEGEND_LABEL, **GEMMA_LABEL},
        light_border=_LIGHT_BORDER, emphasize=_EMPHASIZE | frozenset(GEMMA_COLOR),
        figsize=(6.6, 5.6),  # 9 detectors: the half-column FIGSIZE clips the legend
    )
    for r in rows:
        if r["split"] == "mean":
            print(f"  {r['detector']:28s} AUROC {r['auroc']:.3f}  TPR {r['tpr_at_fpr']:.3f}  "
                  f"AT-HM {goodness(r['auroc'], r['tpr_at_fpr'], 'hmean'):.3f}")


def fig3(recompute: bool) -> None:
    csv_path = OUT_DIR / "judge_backbone_performance.csv"
    if recompute or not csv_path.exists():
        splits = resolve_splits(list(CORE_SPLITS), PARQUET_DIR)
        rows = compute(splits, {"judge": FAMILIES["judge"]}, list(FIG3_JUDGES),
                       [FPR], "human")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")
    rows = rows_at_fpr(pd.read_csv(csv_path).to_dict("records"), FPR)

    by_col = {r["column"]: r for r in rows if r["split"] == "mean"}
    color, label = {}, {}
    for col, (name, hex_) in FIG3_JUDGES.items():
        v = by_col[col]["version"]
        color[v] = hex_
        label[v] = name
    scatter_inset_legend(rows, OUT_DIR / "judge_backbone_1pct", key="version",
                       color=color, label=label)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--recompute", action="store_true",
                    help="rescore Fig. 3's judge columns from the parquets")
    ap.add_argument("--gemma", action="store_true",
                    help="only draw Fig. 2 + the gemma-4-31b judges "
                         "(best_zeroshot_gemma_1pct); Fig. 2/3 are not touched")
    args = ap.parse_args()
    if args.gemma:
        fig2_gemma(args.recompute)
        return
    fig2()
    fig3(args.recompute)


if __name__ == "__main__":
    main()

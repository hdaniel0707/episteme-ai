#!/usr/bin/env python3
r"""GB-orig vs. GB-orig-stand: does standardizing features change anything?

Same rows (human vs. ``gpt35t:0000A``, wp/reuter/essay) under two feature
encodings -- raw text/features (``gb_orig``) vs. standardized
(``gb_orig_stand``). Two comparisons:

    1. the best zero-shot picks (``cfg/ai_detection/trainv2/best_zeroshot.yaml``),
       scored on the full corpus -- same as ``run_best_zeroshot_performance.py``.
    2. one trained detector -- plain logistic regression on ``emb_roberta``
       (``_trained.TRAINED_SPECS['roberta_logistic']``) -- grouped 70/30, N
       seeds. The same seeds + test_size on both splits mean the same
       document ids land in train vs. test either side, so it's a paired
       comparison, not two independent ones.

Prints one table per comparison: gb_orig | gb_orig_stand | delta
(gb_orig_stand - gb_orig), for AUROC and oracle TPR@ each ``--fpr`` level.
Both tables are written to CSV, and plotted as diverging bar charts (one bar
per detector, blue = GB-orig-stand reads higher, red = GB-orig reads higher,
grey = within noise) -- the delta is the whole story here, not the absolute
level, so that is what is drawn:

    auroc_delta.png   one panel, all detectors (AUROC is FPR-independent)
    tpr_delta.png     two panels, one per --fpr level

    uv run python epai/ai_detection/trainv2/compare_gb_orig_stand.py
    uv run python epai/ai_detection/trainv2/compare_gb_orig_stand.py --fpr 0.01 0.05 --seeds 0 1 2 3 4
    uv run python epai/ai_detection/trainv2/compare_gb_orig_stand.py --replot results/trainv2/compare_gb_orig_stand
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from epai.ai_detection.trainv2._trained import TRAINED_SPECS, eval_on_fprs, fit, load_features
from epai.ai_detection.trainv2.core import grouped_split
from epai.ai_detection.trainv2.run_best_zeroshot_performance import (
    DEFAULT_CONFIG,
    _detector,
    _load_columns,
)
from epai.ai_detection.trainv2.run_zeroshot_performance import FAMILIES
from epai.ai_detection.trainv2.run_zeroshot_performance import compute as zeroshot_compute
from epai.ai_detection.trainv2.splits import SPLITS

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "compare_gb_orig_stand"

SPLIT_A, SPLIT_B = "gb_orig", "gb_orig_stand"
DEFAULT_FPRS = [0.01, 0.05]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
TRAINED_KEY = "roberta_logistic"


def zeroshot_diff(fprs: list[float], human_label: str) -> pd.DataFrame:
    """Best zero-shot picks, scored on the full corpus of each split."""
    splits = [(SPLIT_A, PARQUET_DIR / SPLITS[SPLIT_A]),
              (SPLIT_B, PARQUET_DIR / SPLITS[SPLIT_B])]
    cols = _load_columns(DEFAULT_CONFIG)
    rows = zeroshot_compute(splits, FAMILIES, cols, fprs, human_label)
    rows = [r for r in rows if r["split"] in (SPLIT_A, SPLIT_B)]  # drop the pooled 'mean'
    for r in rows:
        r["detector"] = _detector(r["family"], r["version"])

    out = []
    for (det, fpr), g in pd.DataFrame(rows).groupby(["detector", "fpr"]):
        a = g[g["split"] == SPLIT_A].iloc[0]
        b = g[g["split"] == SPLIT_B].iloc[0]
        out.append({
            "detector": det, "fpr": fpr,
            "auroc_gb_orig": a["auroc"], "auroc_gb_orig_stand": b["auroc"],
            "auroc_delta": b["auroc"] - a["auroc"],
            "tpr_gb_orig": a["tpr_at_fpr"], "tpr_gb_orig_stand": b["tpr_at_fpr"],
            "tpr_delta": b["tpr_at_fpr"] - a["tpr_at_fpr"],
        })
    return pd.DataFrame(out).sort_values(["fpr", "detector"]).reset_index(drop=True)


def trained_diff(seeds: list[int], test_size: float, fprs: list[float],
                 group_col: str, human_label: str) -> pd.DataFrame:
    """``roberta_logistic``, grouped 70/30, N seeds -- same seeds on both
    splits, so the same held-out document ids score each side."""
    spec = TRAINED_SPECS[TRAINED_KEY]
    by_split: dict[str, dict] = {}
    for split in (SPLIT_A, SPLIT_B):
        X, y, Z, groups = load_features(PARQUET_DIR / SPLITS[split], group_col, human_label)
        aurocs: list[float] = []
        tprs = {f: [] for f in fprs}
        for seed in seeds:
            tr, te = grouped_split(y, groups, test_size, seed)
            clf = fit(spec, X, y, Z, groups, tr, seed)
            au, per = eval_on_fprs(clf, spec, X[te], y[te], Z[te], fprs)
            aurocs.append(au)
            for f in fprs:
                tprs[f].append(per[f][0])
        by_split[split] = {"auroc": aurocs, "tpr": tprs}

    out = []
    for f in fprs:
        a, b = by_split[SPLIT_A], by_split[SPLIT_B]
        out.append({
            "detector": f"trained/{spec['label']}", "fpr": f,
            "auroc_gb_orig": np.mean(a["auroc"]), "auroc_gb_orig_stand": np.mean(b["auroc"]),
            "auroc_delta": np.mean(b["auroc"]) - np.mean(a["auroc"]),
            "auroc_std_gb_orig": np.std(a["auroc"]), "auroc_std_gb_orig_stand": np.std(b["auroc"]),
            "tpr_gb_orig": np.mean(a["tpr"][f]), "tpr_gb_orig_stand": np.mean(b["tpr"][f]),
            "tpr_delta": np.mean(b["tpr"][f]) - np.mean(a["tpr"][f]),
            "tpr_std_gb_orig": np.std(a["tpr"][f]), "tpr_std_gb_orig_stand": np.std(b["tpr"][f]),
        })
    return pd.DataFrame(out)


def _print(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ---------------------------------------------------------------------------
# Plots -- diverging bars, one per detector: blue = GB-orig-stand reads
# higher, red = GB-orig reads higher, grey = within noise of zero. Polarity
# is the whole point of this script, so it drives the colour, not detector
# identity (cd_core / plots.py use per-split colour for a different job --
# comparing many splits at once -- this compares exactly two).
# ---------------------------------------------------------------------------

_BLUE = "#2a78d6"    # GB-orig-stand higher
_RED = "#e34948"     # GB-orig higher
_NEUTRAL = "#c3c2b7"  # |delta| < eps -- no material difference
_INK = "#0b0b0b"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_SURFACE = "#fcfcfb"

AUROC_EPS = 0.001  # smaller than this, call it noise
TPR_EPS = 0.005

_TITLE = ("GB-orig vs. GB-orig-stand -- {metric} delta\n"
          "blue = GB-orig-stand higher, red = GB-orig higher, grey = within noise")


def _bar_color(delta: float, eps: float) -> str:
    if not np.isfinite(delta) or abs(delta) < eps:
        return _NEUTRAL
    return _BLUE if delta > 0 else _RED


def _diverging_bars(ax, labels: list[str], deltas: list[float], eps: float,
                    xerr: list[float] | None = None) -> None:
    y = list(range(len(labels)))
    colors = [_bar_color(d, eps) for d in deltas]
    ax.barh(y, deltas, height=0.62, color=colors, edgecolor="#222222",
            linewidth=0.4, zorder=3,
            xerr=xerr, error_kw=dict(ecolor="#222222", elinewidth=0.8, capsize=2.0))
    ax.axvline(0, color=_MUTED, lw=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=_INK)
    ax.invert_yaxis()  # first row (largest |delta|, by convention) on top
    span = max((abs(d) for d in deltas if np.isfinite(d)), default=eps)
    pad = span * 0.35 + eps
    ax.set_xlim(-span - pad, span + pad)
    for d, yi in zip(deltas, y):
        if not np.isfinite(d):
            continue
        ax.text(d + (pad * 0.15 if d >= 0 else -pad * 0.15), yi,
                f"{d:+.4f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=7.5, color=_INK)
    ax.grid(axis="x", lw=0.4, alpha=0.6, color=_GRID, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, colors=_MUTED)
    ax.set_facecolor(_SURFACE)


def plot_auroc_delta(zs: pd.DataFrame, tr: pd.DataFrame, path: Path) -> None:
    """One panel: AUROC is FPR-independent, so one row per detector (the
    zero-shot picks repeat identically across --fpr rows; take the first)."""
    z = zs.drop_duplicates("detector")[["detector", "auroc_delta"]].copy()
    z["xerr"] = 0.0
    t = tr.drop_duplicates("detector")[["detector", "auroc_delta"]].copy()
    t["xerr"] = tr.drop_duplicates("detector")["auroc_std_gb_orig_stand"].to_numpy()
    df = pd.concat([z, t], ignore_index=True)
    df = df.reindex(df["auroc_delta"].abs().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(df) + 1.2))
    _diverging_bars(ax, df["detector"].tolist(), df["auroc_delta"].tolist(),
                    AUROC_EPS, df["xerr"].tolist())
    ax.set_xlabel("AUROC delta  (GB-orig-stand − GB-orig)", fontsize=9, color=_INK)
    ax.set_title(_TITLE.format(metric="AUROC"), fontsize=10.5, color=_INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_tpr_delta(zs: pd.DataFrame, tr: pd.DataFrame, path: Path,
                   fprs: list[float]) -> None:
    """One panel per --fpr level, same detector order in both (fixed by the
    first level's |delta| ranking) so the two panels stay comparable."""
    order: list[str] | None = None
    fig, axes = plt.subplots(1, len(fprs), figsize=(7.0 * len(fprs), 5.0), squeeze=False)
    axes = axes[0]
    for ax, fpr in zip(axes, fprs):
        z = zs[np.isclose(zs["fpr"], fpr)][["detector", "tpr_delta"]].copy()
        z["xerr"] = 0.0
        t = tr[np.isclose(tr["fpr"], fpr)][["detector", "tpr_delta"]].copy()
        t["xerr"] = tr[np.isclose(tr["fpr"], fpr)]["tpr_std_gb_orig_stand"].to_numpy()
        df = pd.concat([z, t], ignore_index=True)
        if order is None:
            df = df.reindex(df["tpr_delta"].abs().sort_values(ascending=False).index)
            order = df["detector"].tolist()
        else:
            df = df.set_index("detector").loc[order].reset_index()
        _diverging_bars(ax, df["detector"].tolist(), df["tpr_delta"].tolist(),
                        TPR_EPS, df["xerr"].tolist())
        ax.set_xlabel(f"oracle TPR@{fpr * 100:g}%FPR delta", fontsize=9, color=_INK)
    fig.suptitle(_TITLE.format(metric="oracle TPR"), fontsize=10.5, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS,
                    help=f"oracle FPR level(s) (default: {DEFAULT_FPRS})")
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                    help=f"seeds for the trained-detector split (default: {DEFAULT_SEEDS})")
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true", help="skip the PNGs (CSVs still written)")
    ap.add_argument("--replot", type=Path, default=None, metavar="DIR",
                    help="skip scoring; redraw the PNGs from "
                         "<DIR>/{zeroshot,trained}_diff.csv")
    args = ap.parse_args()

    if args.replot is not None:
        zs = pd.read_csv(args.replot / "zeroshot_diff.csv")
        tr = pd.read_csv(args.replot / "trained_diff.csv")
        plot_auroc_delta(zs, tr, args.replot / "auroc_delta.png")
        plot_tpr_delta(zs, tr, args.replot / "tpr_delta.png", sorted(zs["fpr"].unique()))
        print(f"replotted from {args.replot}")
        return

    zs = zeroshot_diff(args.fpr, args.human_label)
    tr = trained_diff(args.seeds, args.test_size, args.fpr, args.group_col, args.human_label)

    _print(zs, "best zero-shot picks -- GB-orig vs. GB-orig-stand (full corpus)")
    _print(tr, f"trained roberta-logistic -- GB-orig vs. GB-orig-stand "
               f"(grouped 70/30, seeds={args.seeds})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zs.to_csv(args.out_dir / "zeroshot_diff.csv", index=False)
    tr.to_csv(args.out_dir / "trained_diff.csv", index=False)
    print(f"\nwrote {args.out_dir / 'zeroshot_diff.csv'}")
    print(f"wrote {args.out_dir / 'trained_diff.csv'}")

    if not args.no_plots:
        plot_auroc_delta(zs, tr, args.out_dir / "auroc_delta.png")
        plot_tpr_delta(zs, tr, args.out_dir / "tpr_delta.png", args.fpr)


if __name__ == "__main__":
    main()

r"""One figure: 3 scenarios x 5 algorithms, for the training-based comparison.

A companion to ``plot_sentinel_comparison.py``'s per-regime scatters (that
script, and ``plots.py``'s main figures, are untouched -- this is a new,
separate script for this specific comparison). Reads ``run_sentinel_
comparison.py``'s ``performance.csv``, which must have been produced WITH
zero-shot rows (the default; see that script's ``--zs-columns`` /
``--no-zeroshot``).

Same visual language as ``plots.zeroshot_scatter`` (AUROC vs. oracle
TPR@1%FPR, power-warped axes, grey AT-HM contour field in the background),
but a different mark, per the request this was built for:
  * one point per (algorithm, scenario) -- no per-corpus light points;
  * no error bars drawn (the SD is already in the CSV -- just not drawn here);
  * colour = algorithm;
  * a TRAINED algorithm gets one marker per scenario (shape = scenario) and a
    same-colour line through its 3 scenario points in a fixed order
    (In-Domain -> Near-OOD -> Far-OOD), so the degradation path from
    in-domain to increasingly distant OOD reads as one coloured line;
  * a ZERO-SHOT algorithm gets ONE star marker instead, not three overlapping
    scenario shapes -- see "Fixed eval sets" below for why its reading really
    is identical across all three scenarios, not merely similar.

Algorithms (fixed; raw ``detector`` ids as ``run_sentinel_comparison.py``
writes them):
  trained/roberta·logistic       roberta-logistic (this run's trained probe)
  trained/roberta·sentinel_mlp   roberta-sentinel (this run's trained MLP)
  judge/gpt56luna_ord2           LogJudge (gpt-5.6-luna) -- zero-shot
  fusionjudge/gpt56luna_v3       FusionJudge (gpt-5.6-luna) -- zero-shot
  binox/llama31-8b               Binoculars (llama-3.1-8b) -- zero-shot,
                                  the strongest backbone in this paper's
                                  own General Results table

Fixed eval sets (see ``run_sentinel_comparison.py``'s module docstring,
"Design: a fixed evaluation set per scenario", for the full derivation):
each corpus (MG-GB, MG-Sci) has ONE fixed grouped 70/30 split, drawn once and
reused everywhere; only the TRAINED half's training set narrows across
scenarios (full 70% -> 70% minus one domain -> reused for Far-OOD's cross-
corpus score). A zero-shot pick is scored once per fixed 30% set and that
number is looked up everywhere -- so with only two fixed eval sets total
(plus MG-AI-Sci) and every scenario averaging over both corpus orientations,
a zero-shot algorithm's In-Domain, Near-OOD, and Far-OOD readings are the
exact same number, not merely close. That's why it gets a single star rather
than three markers that would otherwise sit exactly on top of each other.

Scenarios (each a filter on performance.csv's ``level == "split"`` rows, then
a plain per-detector mean of ``auroc`` / ``tpr_at_fpr`` over the matching
rows -- see ``run_sentinel_comparison.py``'s module docstring for exactly
which eval targets each regime's rows cover, trained or zero-shot). No
``train_split`` filter is needed for any of the three: every regime already
carries both corpus orientations (MG-GB primary and MG-Sci primary) under
the same ``regime`` value, so a plain regime match IS the "mean of the 2"
average the article's prose describes:
  in_domain     regime == "in_domain"
  near_ood      regime == "near_ood"
  far_ood       regime == "far_ood"

Writes a PNG next to the CSV (quick preview) AND a PDF into the article's
``figs/`` directory by default (``--figs-dir ''`` to skip the PDF), named
``sentinel_scenarios_<fpr-tag>`` either way -- matching how every other figure
this article's ``iclr2027_conference.tex`` includes lives in ``figs/`` and is
referenced there as ``\includegraphics{figs/sentinel_scenarios_1pct}``.
``--k`` (default 2.0) is the power-warp exponent on both axes -- see
``DEFAULT_K``. No title is drawn (this is meant to be dropped straight into
the article with its own caption).

Run:
    uv run python epai/ai_detection/trainv2/plot_sentinel_scenarios.py \
        results/trainv2/sentinel_comparison_<timestamp>/performance.csv
    uv run python epai/ai_detection/trainv2/plot_sentinel_scenarios.py \
        results/trainv2/sentinel_comparison_<timestamp>/performance.csv --k 3.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the Agg backend is chosen)

from epai.ai_detection.trainv2.detector_names import display_detector  # noqa: E402
from epai.ai_detection.trainv2.plots import (  # noqa: E402
    _INK,
    _MUTED,
    _PALETTE,
    _SURFACE,
    _distort,
    _goodness_field,
)

#: fixed algorithm order -> also the colour-assignment order.
ALGORITHMS = [
    "trained/roberta·logistic",
    "trained/roberta·sentinel_mlp",
    "judge/gpt56luna_ord2",
    "fusionjudge/gpt56luna_v3",
    "binox/llama31-8b",
]


def _is_trained(algo: str) -> bool:
    return algo.startswith("trained/")


#: fixed scenario order -> the order the connecting line is drawn in
#: (trained algorithms only -- see the module docstring).
SCENARIOS = ["in_domain", "near_ood", "far_ood"]
#: legend text only -- plain, no parentheses, no "trained on X" framing (that
#: framing is only true for the trained detectors; a zero-shot pick has no
#: source corpus, it's just scored on that scenario's eval rows too). The
#: full definition of each scenario -- which corpus is trained on / which
#: eval targets are averaged in -- belongs in the figure caption, not here.
SCENARIO_LABEL = {
    "in_domain": "In-Domain",
    "near_ood": "Near-OOD",
    "far_ood": "Far-OOD",
}
SCENARIO_MARKER = {"in_domain": "o", "near_ood": "^", "far_ood": "s"}
#: the zero-shot marker -- one per algorithm, not one per scenario. See the
#: module docstring's "Fixed eval sets" section for why a single marker is
#: correct here rather than an approximation.
ZS_MARKER = "*"

#: drawn on top of every other algorithm's line/markers wherever they overlap
#: (a higher zorder, not a legend-order change -- see :func:`_zorder`).
FRONT_ALGO = "trained/roberta·sentinel_mlp"


def _zorder(base: float, algo: str) -> float:
    return base + (0.5 if algo == FRONT_ALGO else 0.0)


def _scenario_mask(df: pd.DataFrame, scenario: str) -> pd.Series:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    return (df["level"] == "split") & (df["regime"] == scenario)


def _points(df: pd.DataFrame, fpr: float) -> dict[tuple[str, str], tuple[float, float]]:
    """``(algorithm, scenario) -> (auroc_mean, tpr_at_fpr_mean)`` -- the plain
    mean over that scenario's matching rows for that detector (both corpus
    orientations pooled together -- see the module docstring). No weighting,
    no error bars (those live in the CSV's ``_std`` columns, not here)."""
    sub = df[df["fpr"] == fpr]
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for algo in ALGORITHMS:
        for scen in SCENARIOS:
            rows = sub[_scenario_mask(sub, scen) & (sub["detector"] == algo)]
            rows = rows[np.isfinite(rows["auroc"]) & np.isfinite(rows["tpr_at_fpr"])]
            if rows.empty:
                continue
            out[(algo, scen)] = (float(rows["auroc"].mean()), float(rows["tpr_at_fpr"].mean()))
    return out


def _zs_point(points: dict, algo: str) -> tuple[float, float] | None:
    """A zero-shot algorithm's single (identical-across-scenarios) point --
    whichever scenario is present first, since by construction they all
    agree exactly (see the module docstring)."""
    for scen in SCENARIOS:
        if (algo, scen) in points:
            return points[(algo, scen)]
    return None


#: the power-warp exponent (plots._distort) -- k > 1 compresses the mostly-
#: empty low end and stretches the crowded near-1.0 cluster; higher k pulls
#: the in-domain cluster and the OOD points further apart on the page. Same
#: default as plots.zeroshot_scatter's main figure.
DEFAULT_K = 1.7


def plot(df: pd.DataFrame, fpr: float, paths: list[Path],
        k: float = DEFAULT_K) -> None:
    points = _points(df, fpr)
    hue = {algo: _PALETTE[i % len(_PALETTE)] for i, algo in enumerate(ALGORITHMS)}

    aurocs = [v[0] for v in points.values()]
    x_lo = min(0.45, (min(aurocs) - 0.12) if aurocs else 0.45)
    x_hi, y_lo, y_hi = 1.02, -0.25, 1.04
    kx = ky = k

    def wx(v):
        return _distort(v, x_lo, x_hi, kx)

    def wy(v):
        return _distort(v, y_lo, y_hi, ky)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Chance point, same convention as plots.zeroshot_scatter.
    ax.axvline(wx(0.5), color="#b5495b", lw=1.0, ls=":", alpha=0.6, zorder=0)
    ax.axhline(wy(fpr), color="#b5495b", lw=1.0, ls=":", alpha=0.6, zorder=0)
    ax.scatter([wx(0.5)], [wy(fpr)], s=100, color="#b5495b", marker="x", lw=1.6,
               alpha=0.6, zorder=0)

    # AT-HM background field, same convention as plots.zeroshot_scatter.
    gx = np.linspace(x_lo, 1.0, 200)
    gy = np.linspace(-0.04, 1.0, 200)
    GX, GY = np.meshgrid(gx, gy)
    GZ = _goodness_field(GX, GY, alpha=fpr)
    ax.contourf(wx(GX), wy(GY), GZ, levels=np.linspace(0, 1, 11), cmap="Greys",
                alpha=0.30, zorder=-5)
    cs = ax.contour(wx(GX), wy(GY), GZ, levels=[0.2, 0.4, 0.6, 0.8],
                     colors="#8a8a8a", linewidths=0.6, alpha=0.8, zorder=-4)
    # Manual label anchors: a point toward the upper-left end of each level's
    # longest segment, but pulled in from the grid's own edge by a margin --
    # automatic placement put the labels where the data-point cluster covers
    # them (contourf/contour draw well behind the markers, zorder -5/-4 vs.
    # 5); the true left-most point sits right on the grid boundary, off the
    # visible figure. Keeping only "interior" points and taking the left-most
    # of those lands the label upper-left of centre without touching the edge.
    x_grid_lo, x_grid_hi = wx(x_lo), wx(1.0)
    y_grid_lo, y_grid_hi = wy(-0.04), wy(1.0)
    mx, my = (x_grid_hi - x_grid_lo) * 0.08, (y_grid_hi - y_grid_lo) * 0.08
    manual_pos = []
    for seg_list in cs.allsegs:
        segs = [s for s in seg_list if len(s)]
        if not segs:
            continue
        longest = max(segs, key=len)
        interior = longest[
            (longest[:, 0] > x_grid_lo + mx) & (longest[:, 0] < x_grid_hi - mx)
            & (longest[:, 1] > y_grid_lo + my) & (longest[:, 1] < y_grid_hi - my)
        ]
        cand = interior if len(interior) else longest
        manual_pos.append(tuple(cand[np.argmin(cand[:, 0])]))
    ax.clabel(cs, manual=manual_pos or None,
             fmt=lambda v: f"{v:.1f}", fontsize=6.5, colors="#6a6a6a", inline=True)

    # One same-colour line per TRAINED algorithm, through its scenario points
    # in SCENARIOS order -- the "degradation path" from in-domain to far OOD.
    # Zero-shot algorithms have only one point (see the module docstring), so
    # no line for them. FRONT_ALGO draws (and so overlaps) on top of every
    # other algorithm's line/markers, regardless of ALGORITHMS' own (legend)
    # order.
    for algo in ALGORITHMS:
        if not _is_trained(algo):
            continue
        xs = [points[(algo, s)][0] for s in SCENARIOS if (algo, s) in points]
        ys = [points[(algo, s)][1] for s in SCENARIOS if (algo, s) in points]
        if len(xs) >= 2:
            ax.plot([wx(x) for x in xs], [wy(y) for y in ys],
                    color=hue[algo], lw=1.3, alpha=0.75, zorder=_zorder(2, algo))

    for algo in ALGORITHMS:
        if _is_trained(algo):
            for scen in SCENARIOS:
                if (algo, scen) not in points:
                    continue
                x, y = points[(algo, scen)]
                ax.scatter([wx(x)], [wy(y)], s=90, marker=SCENARIO_MARKER[scen],
                           facecolors=hue[algo], edgecolors="#222222", linewidths=0.9,
                           zorder=_zorder(5, algo))
        else:
            p = _zs_point(points, algo)
            if p is None:
                continue
            x, y = p
            ax.scatter([wx(x)], [wy(y)], s=210, marker=ZS_MARKER,
                       facecolors=hue[algo], edgecolors="#222222", linewidths=0.9,
                       zorder=_zorder(6, algo))

    ax.set_xlabel(f"AUROC\n(power warp, k={kx:g})", fontsize=9, color=_INK)
    ax.set_ylabel(f"TPR@{fpr * 100:g}%FPR\n(power warp, k={ky:g})", fontsize=9, color=_INK)
    ax.set_xlim(wx(x_lo), wx(x_hi))
    ax.set_ylim(wy(y_lo), wy(y_hi))
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.tick_params(labelsize=8, colors=_MUTED)

    x_ticks = [t for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0) if t >= x_lo]
    ax.set_xticks([wx(t) for t in x_ticks])
    ax.set_xticklabels([f"{t:g}" for t in x_ticks])
    y_ticks = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0)
    ax.set_yticks([wy(t) for t in y_ticks])
    ax.set_yticklabels([f"{t:g}" for t in y_ticks])

    algo_handles = [
        plt.Line2D([], [], marker=("o" if _is_trained(a) else ZS_MARKER),
                  ls=("-" if _is_trained(a) else ""), ms=(8 if _is_trained(a) else 12),
                  color=hue[a], mfc=hue[a], mec="#222222", mew=0.9, label=display_detector(a))
        for a in ALGORITHMS
    ]
    scen_handles = [plt.Line2D([], [], marker=SCENARIO_MARKER[s], ls="", ms=8,
                               color="#6a6a6a", label=f"{SCENARIO_LABEL[s]} (trained)")
                    for s in SCENARIOS]
    zs_handle = plt.Line2D([], [], marker=ZS_MARKER, ls="", ms=12, color="#6a6a6a",
                           label="zero-shot (all scenarios)")
    chance_handle = plt.Line2D([], [], marker="x", ls="", color="#b5495b",
                               label=f"chance (0.5, {fpr * 100:g}%)")
    legend = ax.legend(
        handles=[*algo_handles, plt.Line2D([], [], ls="", label=""),
                *scen_handles, zs_handle, chance_handle],
        fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 0.80),
        framealpha=0.9, borderaxespad=0.0,
    )

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight",
                    bbox_extra_artists=(legend,))
        print(f"wrote {path}")
    plt.close(fig)


#: the article's figure directory (matches every other figs/<name>.pdf
#: referenced there via \includegraphics{figs/<name>}, e.g. best_zeroshot_1pct).
BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_FIGS_DIR = BASE_DIR / "latex" / "_ICLR_2027__AI_Detect" / "figs"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="run_sentinel_comparison.py's performance.csv")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: the CSV's own directory (PNG, for quick preview)")
    ap.add_argument("--figs-dir", type=Path, default=DEFAULT_FIGS_DIR,
                    help=f"where the article-ready PDF is also written (default: "
                         f"{DEFAULT_FIGS_DIR}); pass '' to skip the PDF entirely")
    ap.add_argument("--k", type=float, default=DEFAULT_K,
                    help=f"power-warp exponent on both axes (default: {DEFAULT_K})")
    args = ap.parse_args()
    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = Path(args.figs_dir) if str(args.figs_dir) else None

    df = pd.read_csv(args.csv)
    if "kind" not in df.columns:
        raise SystemExit(
            f"{args.csv}: no 'kind' column -- this CSV predates zero-shot rows. "
            f"Re-run run_sentinel_comparison.py (zero-shot is on by default) to "
            f"regenerate it.")
    missing = [a for a in ALGORITHMS if a not in set(df["detector"])]
    if missing:
        raise SystemExit(
            f"performance.csv is missing detector(s) {missing} -- for the zero-shot "
            f"picks, re-run run_sentinel_comparison.py with those columns available "
            f"(see --zs-columns / --no-zeroshot).")

    for fpr in sorted(df["fpr"].unique()):
        tag = f"{fpr * 100:g}pct"
        name = f"sentinel_scenarios_{tag}"
        paths = [out_dir / f"{name}.png"]
        if figs_dir is not None:
            paths.append(figs_dir / f"{name}.pdf")
        plot(df, fpr, paths, k=args.k)


if __name__ == "__main__":
    main()

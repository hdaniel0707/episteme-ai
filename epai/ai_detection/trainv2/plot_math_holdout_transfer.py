r"""One figure: 3 training-corpus scenarios x 5 algorithms, for the
MG-Sci domain-holdout transfer comparison (``run_math_holdout_transfer.py``).

A direct structural copy of ``plot_aisci_transfer.py`` with the scenario
labels swapped for this experiment's three training-corpus choices -- see
that script's module docstring for the shared visual language (colour =
algorithm; a TRAINED algorithm gets one marker per training corpus joined by
a line; a ZERO-SHOT algorithm gets ONE star, since it has no training corpus
and the held-out domain's rows are the one fixed eval target throughout).

Which MG-Sci domain was held out (mathematics / computer_science /
sociology, ``--holdout-domain`` in ``run_math_holdout_transfer.py``) is read
straight off the CSV's own ``train_split`` values (see
:func:`scenario_layout`) rather than passed as a flag here -- one run's CSV
is fully self-describing, so this script never needs to be told separately
which domain that particular run held out.

Writes a PNG next to the CSV (quick preview) AND a PDF into the article's
``figs/`` directory by default (``--figs-dir ''`` to skip the PDF), named
``math_holdout_transfer_<code>_<fpr-tag>`` either way.

Run:
    uv run python epai/ai_detection/trainv2/plot_math_holdout_transfer.py \
        results/trainv2/math_holdout_transfer_<code>_<timestamp>/performance.csv
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

#: fixed algorithm order -> also the colour-assignment order. Same five as
#: plot_sentinel_scenarios.py / plot_aisci_transfer.py's ALGORITHMS, for a
#: consistent colour mapping across all three figures.
ALGORITHMS = [
    "trained/roberta·logistic",
    "trained/roberta·sentinel_mlp",
    "judge/gpt56luna_ord2",
    "fusionjudge/gpt56luna_v3",
    "binox/llama31-8b",
]


def _is_trained(algo: str) -> bool:
    return algo.startswith("trained/")


#: short code -> the readable domain name in the legend/axis labels --
#: matches run_math_holdout_transfer.py's DOMAIN_CODE, inverted.
_CODE_READABLE = {"math": "math", "cs": "computer science", "soc": "sociology"}
ZS_MARKER = "*"
FRONT_ALGO = "trained/roberta·sentinel_mlp"


def scenario_layout(df: pd.DataFrame) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """``(scenarios, scenario_label, scenario_marker)`` derived from the
    CSV's own ``train_split`` values -- the ex-domain scenario's exact key
    (``mg_sci_ex_<code>``) depends on which MG-Sci domain
    ``run_math_holdout_transfer.py --holdout-domain`` held out for this
    particular run, so it's read off the data rather than hardcoded."""
    ex_candidates = sorted(s for s in df["train_split"].unique()
                           if s not in ("mg_gb", "gb_orig"))
    if len(ex_candidates) != 1:
        raise SystemExit(
            f"expected exactly one 'mg_sci_ex_<code>' scenario in train_split, found "
            f"{ex_candidates} -- was this CSV produced by run_math_holdout_transfer.py?")
    ex_key = ex_candidates[0]
    code = ex_key.removeprefix("mg_sci_ex_")
    readable = _CODE_READABLE.get(code, code)
    scenarios = [ex_key, "mg_gb", "gb_orig"]
    labels = {
        ex_key: f"Trained on MG-Sci (excl. {readable})",
        "mg_gb": "Trained on MG-GB",
        "gb_orig": "Trained on GB-orig",
    }
    markers = {ex_key: "o", "mg_gb": "^", "gb_orig": "s"}
    return scenarios, labels, markers


def _zorder(base: float, algo: str) -> float:
    return base + (0.5 if algo == FRONT_ALGO else 0.0)


def _points(df: pd.DataFrame, fpr: float,
           scenarios: list[str]) -> dict[tuple[str, str], tuple[float, float]]:
    """``(algorithm, train_split) -> (auroc, tpr_at_fpr)`` -- one row per
    cell already, no pooling needed (every scenario here is one training
    corpus against the single fixed held-out eval set)."""
    sub = df[df["fpr"] == fpr]
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for algo in ALGORITHMS:
        for scen in scenarios:
            rows = sub[(sub["detector"] == algo) & (sub["train_split"] == scen)]
            rows = rows[np.isfinite(rows["auroc"]) & np.isfinite(rows["tpr_at_fpr"])]
            if rows.empty:
                continue
            out[(algo, scen)] = (float(rows["auroc"].mean()), float(rows["tpr_at_fpr"].mean()))
    return out


def _zs_point(points: dict, algo: str, scenarios: list[str]) -> tuple[float, float] | None:
    for scen in scenarios:
        if (algo, scen) in points:
            return points[(algo, scen)]
    return None


DEFAULT_K = 1.7


def plot(df: pd.DataFrame, fpr: float, paths: list[Path],
        k: float = DEFAULT_K) -> None:
    scenarios, scenario_label, scenario_marker = scenario_layout(df)
    points = _points(df, fpr, scenarios)
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

    ax.axvline(wx(0.5), color="#b5495b", lw=1.0, ls=":", alpha=0.6, zorder=0)
    ax.axhline(wy(fpr), color="#b5495b", lw=1.0, ls=":", alpha=0.6, zorder=0)
    ax.scatter([wx(0.5)], [wy(fpr)], s=100, color="#b5495b", marker="x", lw=1.6,
               alpha=0.6, zorder=0)

    gx = np.linspace(x_lo, 1.0, 200)
    gy = np.linspace(-0.04, 1.0, 200)
    GX, GY = np.meshgrid(gx, gy)
    GZ = _goodness_field(GX, GY, alpha=fpr)
    ax.contourf(wx(GX), wy(GY), GZ, levels=np.linspace(0, 1, 11), cmap="Greys",
                alpha=0.30, zorder=-5)
    cs = ax.contour(wx(GX), wy(GY), GZ, levels=[0.2, 0.4, 0.6, 0.8],
                     colors="#8a8a8a", linewidths=0.6, alpha=0.8, zorder=-4)
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

    for algo in ALGORITHMS:
        if not _is_trained(algo):
            continue
        xs = [points[(algo, s)][0] for s in scenarios if (algo, s) in points]
        ys = [points[(algo, s)][1] for s in scenarios if (algo, s) in points]
        if len(xs) >= 2:
            ax.plot([wx(x) for x in xs], [wy(y) for y in ys],
                    color=hue[algo], lw=1.3, alpha=0.75, zorder=_zorder(2, algo))

    for algo in ALGORITHMS:
        if _is_trained(algo):
            for scen in scenarios:
                if (algo, scen) not in points:
                    continue
                x, y = points[(algo, scen)]
                ax.scatter([wx(x)], [wy(y)], s=90, marker=scenario_marker[scen],
                           facecolors=hue[algo], edgecolors="#222222", linewidths=0.9,
                           zorder=_zorder(5, algo))
        else:
            p = _zs_point(points, algo, scenarios)
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
    scen_handles = [plt.Line2D([], [], marker=scenario_marker[s], ls="", ms=8,
                               color="#6a6a6a", label=scenario_label[s])
                    for s in scenarios]
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


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_FIGS_DIR = BASE_DIR / "latex" / "_ICLR_2027__AI_Detect" / "figs"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="run_math_holdout_transfer.py's performance.csv")
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
    missing = [a for a in ALGORITHMS if a not in set(df["detector"])]
    if missing:
        raise SystemExit(
            f"performance.csv is missing detector(s) {missing} -- for the zero-shot "
            f"picks, re-run run_math_holdout_transfer.py with those columns available "
            f"(see --zs-columns / --no-zeroshot).")

    ex_key = scenario_layout(df)[0][0]
    code = ex_key.removeprefix("mg_sci_ex_")
    for fpr in sorted(df["fpr"].unique()):
        tag = f"{fpr * 100:g}pct"
        name = f"math_holdout_transfer_{code}_{tag}"
        paths = [out_dir / f"{name}.png"]
        if figs_dir is not None:
            paths.append(figs_dir / f"{name}.pdf")
        plot(df, fpr, paths, k=args.k)


if __name__ == "__main__":
    main()

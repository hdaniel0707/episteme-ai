r"""Publication figure: is MANUGRAM harder for Binoculars than GB-orig?

One figure, two rows, four columns:

    row 1   Binoculars, ORIGINAL backbone   (falcon-7b, 2023 -- the paper's
            own pair, column ``binoculars_score``)
    row 2   Binoculars, BEST backbone       (llama31-8b -- the empirically
            best Binoculars pair on MANUGRAM, per
            ``cfg/ai_detection/trainv2/best_zeroshot.yaml``, column
            ``binoculars_score_llama31_8b``)
    cols    the four CORE splits, GB-orig -> MG-GB -> MG-Sci -> MG-AI-Sci
            (``splits.DEFAULT`` -- deliberately NOT including the pooled
            "All Benchmarks" column the baseline-calibration plots show)

Each panel is a human-vs-machine score-density histogram with the 1%-FPR
threshold marked, in the same visual language as
``calibrate_common._plot_column`` (same colors, same threshold-line style) --
this is that plot's figure, not a new design, just: two fixed columns instead
of every column, two rows instead of one, the pooled "All Benchmarks" column
dropped, and everything sized for a paper column rather than a diagnostic
sweep.

Thresholds are fit FRESH per (pair, corpus) at ``--fpr-target`` (default 1%),
exactly like ``calibrate_*_baseline.py --corpora`` does -- not read back from
the calibrated YAML, which is a single threshold pooled over MG-Sci alone and
would not be the right cut for GB-orig or MG-AI-Sci.

Run:
    uv run python epai/ai_detection/trainv2/plot_binox_difficulty.py

    uv run python epai/ai_detection/trainv2/plot_binox_difficulty.py \
        --corpora gb_orig mg_gb mg_sci mg_aisci \
        --top-col binoculars_score --top-label "Falcon-7B (original, 2023)" \
        --bottom-col binoculars_score_llama31_8b --bottom-label "Llama-3.1-8B (best)" \
        -o results/trainv2/binox_difficulty/binox_vs_manugram.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.cd_core import auroc_score, oriented_scores, rates_at_threshold, roc_point_at_fpr

from .calibrate_common import PARQUET_DIR
from .splits import DEFAULT, SPLITS, display

GEN_COL = "generated_by"

_HUMAN_COLOR = "#1c6fb3"   # human = blue   (matches calibrate_common)
_MACHINE_COLOR = "#d92d20"  # AI = red
_FPR_COLOR = "#7048e8"     # 1%-FPR threshold = violet

# Binoculars reads LOW = machine (see cd_core.oriented_scores docstring).
MACHINE_IS_HIGH = False


def _resolve(names: list[str]) -> list[tuple[str, Path]]:
    """Split keys -> ``[(key, parquet_path), ...]``, in ``DEFAULT`` order,
    with NO pooled 'all' -- 'all' is not a key in ``splits.SPLITS`` at all,
    it only exists as a pseudo-corpus ``calibrate_common._multi`` builds."""
    chosen = list(DEFAULT) if not names else [k for k in SPLITS if k in names]
    unknown = [n for n in names if n not in SPLITS] if names else []
    if unknown:
        raise SystemExit(f"unknown split(s) {unknown}; choose from {list(SPLITS)}")
    return [(k, PARQUET_DIR / SPLITS[k]) for k in chosen]


def _fit(col: str, corpus: str, df: pd.DataFrame, fpr_target: float,
          human_label: str) -> dict | None:
    """AUROC + 1%-FPR operating point for one (column, corpus), fit fresh on
    that corpus's own rows -- same statistic ``calibrate_common._row`` reports,
    trimmed to just what this figure draws."""
    if col not in df.columns:
        return None
    sub = df[[col, GEN_COL]].dropna(subset=[col])
    raw = sub[col].to_numpy(dtype=float)
    y = (sub[GEN_COL].astype(str) != human_label).to_numpy().astype(int)
    if np.unique(y).size < 2 or len(y) < 10:
        return None

    oriented = oriented_scores(raw, MACHINE_IS_HIGH)
    tpr, thr_or = roc_point_at_fpr(y, oriented, fpr_target)
    _, fpr = rates_at_threshold(y, oriented, thr_or)
    thr_raw = float(thr_or) if MACHINE_IS_HIGH else -float(thr_or)

    return {
        "column": col, "corpus": corpus, "raw": raw, "y": y,
        "n": int(len(y)), "n_human": int((y == 0).sum()), "n_machine": int((y == 1).sum()),
        "auroc": float(auroc_score(y, oriented)),
        "threshold": thr_raw, "tpr": float(tpr), "fpr": float(fpr),
    }


def _panel(ax, r: dict, edges: np.ndarray, mids: np.ndarray,
            fpr_target: float, show_ylabel: bool) -> None:
    raw, y = r["raw"], r["y"]
    h = np.histogram(raw[y == 0], bins=edges, density=True)[0]
    m = np.histogram(raw[y == 1], bins=edges, density=True)[0]

    ax.step(mids, h, where="mid", color=_HUMAN_COLOR, lw=1.8, label="human")
    ax.fill_between(mids, h, step="mid", color=_HUMAN_COLOR, alpha=0.15)
    ax.step(mids, m, where="mid", color=_MACHINE_COLOR, lw=1.8, label="AI")
    ax.fill_between(mids, m, step="mid", color=_MACHINE_COLOR, alpha=0.15)

    if np.isfinite(r["threshold"]):
        ax.axvline(r["threshold"], color=_FPR_COLOR, lw=1.8, ls="--",
                   label=f"{fpr_target * 100:g}% FPR cut")

    ax.set_title(
        f"{display(r['corpus'])} (n={r['n']})\n"
        f"AUROC {r['auroc']:.3f}   TPR@{fpr_target * 100:g}%FPR {r['tpr']:.2f}",
        fontsize=11,
    )
    ax.set_xlim(edges[0], edges[-1])
    ax.set_yticks([])
    ax.grid(True, lw=0.4, alpha=0.35)
    ax.tick_params(axis="x", labelsize=9)
    if show_ylabel:
        ax.set_ylabel("density", fontsize=11)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--corpora", nargs="+", default=None, metavar="NAME",
                    help="which splits to plot as columns, left to right "
                         "(default: the four core splits, gb_orig mg_gb "
                         "mg_sci mg_aisci -- \"All Benchmarks\" is never included)")
    ap.add_argument("--top-col", default="binoculars_score",
                    help="score column for the top row (default: "
                         "binoculars_score = falcon-7b, the original pair)")
    ap.add_argument("--top-label", default="Falcon-7B",
                    help="row label for the top row")
    ap.add_argument("--bottom-col", default="binoculars_score_llama31_8b",
                    help="score column for the bottom row (default: "
                         "binoculars_score_llama31_8b, the empirically best "
                         "pair per cfg/ai_detection/trainv2/best_zeroshot.yaml)")
    ap.add_argument("--bottom-label", default="Llama-3.1-8B",
                    help="row label for the bottom row")
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("-o", "--out", type=Path,
                    default=Path("results/trainv2/binox_difficulty/binox_vs_manugram.pdf"),
                    help="figure path; also written as a sibling .png at 300 dpi")
    args = ap.parse_args()

    corpora = _resolve(args.corpora or [])
    frames = {name: pd.read_parquet(path) for name, path in corpora}

    rows: dict[str, dict[str, dict]] = {args.top_col: {}, args.bottom_col: {}}
    print(f"{'row':<10}{'corpus':<12}{'n_hum':>7}{'n_mac':>7}{'auroc':>8}"
          f"{'thr@fpr':>11}{'tpr':>6}{'fpr':>7}")
    for col, tag in ((args.top_col, "top"), (args.bottom_col, "bottom")):
        for name, df in frames.items():
            r = _fit(col, name, df, args.fpr_target, args.human_label)
            if r is None:
                print(f"{tag:<10}{name:<12}  -- skipped: column absent, single "
                      f"class, or n<10 --")
                continue
            print(f"{tag:<10}{name:<12}{r['n_human']:>7}{r['n_machine']:>7}"
                  f"{r['auroc']:>8.3f}{r['threshold']:>11.4g}{r['tpr']:>6.2f}"
                  f"{r['fpr']:>7.3f}")
            rows[col][name] = r
    print()

    for col, tag in ((args.top_col, "top"), (args.bottom_col, "bottom")):
        missing = [n for n, _ in corpora if n not in rows[col]]
        if missing:
            raise SystemExit(f"{tag} row ({col}): no usable data for {missing} "
                              f"-- cannot draw a complete grid")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,   # embed as real text, not curves
        "ps.fonttype": 42,
    })

    names = [n for n, _ in corpora]
    ncols = len(names)
    fig, axes = plt.subplots(2, ncols, figsize=(3.6 * ncols, 6.4), squeeze=False)

    for col, row_idx, row_label in (
        (args.top_col, 0, args.top_label), (args.bottom_col, 1, args.bottom_label),
    ):
        all_raw = np.concatenate([rows[col][n]["raw"] for n in names])
        lo, hi = np.percentile(all_raw, [0.5, 99.5])
        thrs = [rows[col][n]["threshold"] for n in names
                if np.isfinite(rows[col][n]["threshold"])]
        if thrs:
            lo, hi = min(lo, *thrs), max(hi, *thrs)
        pad = 0.03 * (hi - lo) or 0.01
        edges = np.linspace(lo - pad, hi + pad, args.bins + 1)
        mids = (edges[:-1] + edges[1:]) / 2

        for j, name in enumerate(names):
            ax = axes[row_idx][j]
            _panel(ax, rows[col][name], edges, mids, args.fpr_target,
                   show_ylabel=(j == 0))
            if row_idx == 1:
                ax.set_xlabel("Binoculars score", fontsize=11)
        axes[row_idx][0].annotate(
            row_label, xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-axes[row_idx][0].yaxis.labelpad - 38, 0),
            textcoords="offset points", ha="right", va="center",
            fontsize=11, fontweight="bold", rotation=90,
        )

    # One shared legend for the whole figure, outside every axes -- anchored
    # just above the top-left panel so it never overlaps a histogram.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.03, 1.0),
               ncol=len(labels), frameon=False, fontsize=9)

    fig.tight_layout(rect=(0.03, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()

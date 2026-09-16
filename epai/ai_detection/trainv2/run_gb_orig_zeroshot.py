r"""Zero-shot detector oracle AUROC / TPR@FPR on GB-orig alone -- the exact
corpus (Ghostbuster's News / Creative-Writing / Student-Essay set, Verma et
al. 2023) the Binoculars paper (Hans et al. 2024) reports TPR@0.01%FPR on.

Why the literal 0.01% reading is degenerate here: the paper's number is a
TRANSPORTED cutoff (a threshold fit on a much larger, separate human corpus,
then applied to Ghostbuster's eval set) -- not an oracle threshold fit on
GB-orig's own human rows. Every trainv2 zero-shot script (including this one)
reports the ORACLE reading instead: the threshold is fit on the split's own
ROC, which needs ``n_human * fpr >= 1`` to land anywhere but the trivial
"flag nobody" point. GB-orig has ~2,992 human rows, so at 0.01% FPR
(``n_human * fpr`` = 0.3) every detector's literal oracle TPR is exactly 0.0
-- there is no data point between "flag nobody" and the next achievable step,
~1/2992 = 0.033% FPR. Both readings are written to the CSV (the literal one
for the paper comparison, flagged ``reachable=False``); the plot uses
``tpr_eff`` -- the oracle TPR at the *finest FPR this file can actually
resolve* (``core.eff_fpr_level``, the same fallback every other trainv2
zero-shot script uses for an unreachable target, e.g. MG-AI-Sci at 1%).

Single combined scatter (every zero-shot column, every family, one point
each -- GB-orig is one split, so there's no per-split/mean spread to draw
the way ``run_zeroshot_performance.py``'s multi-split scatters do). The
legend carries each detector's AUROC and TPR@eff_fpr numbers directly,
not just its name.

compute -> CSV -> plot:

    results/trainv2/gb_orig_zeroshot/
        performance.csv
        gb_orig_zeroshot.png

Run:
    uv run python epai/ai_detection/trainv2/run_gb_orig_zeroshot.py
    uv run python epai/ai_detection/trainv2/run_gb_orig_zeroshot.py --fpr 0.0001 0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.ai_detection.trainv2.plots import _INK, _MUTED, _PALETTE, _SURFACE
from epai.ai_detection.trainv2.run_best_zeroshot_performance import _detector
from epai.ai_detection.trainv2.run_zeroshot_performance import (
    CSV_FIELDS,
    FAMILIES,
    compute,
)
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "gb_orig_zeroshot"
SPLIT_KEY = "gb_orig"
DEFAULT_FPRS = [0.0001]

OUT_FIELDS = ["detector", *CSV_FIELDS]


def plot(rows: list[dict], fpr: float, path: Path) -> None:
    """One combined AUROC-vs-oracle-TPR scatter, every zero-shot column as one
    point coloured by family; the legend carries each detector's AUROC and
    TPR@eff_fpr numbers, not just its name."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if np.isfinite(r["auroc"]) and np.isfinite(r["tpr_eff"])]
    if not rows:
        print("plot: nothing finite to draw -- skipped")
        return
    fam_colour = {f: _PALETTE[i % len(_PALETTE)] for i, f in enumerate(FAMILIES)}

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.axvline(0.5, color="#b5495b", lw=1.0, ls=":", alpha=0.6, zorder=0)

    handles = []
    for r in sorted(rows, key=lambda r: -r["auroc"]):
        x, y = r["auroc"], r["tpr_eff"]
        c = fam_colour.get(r["family"], _MUTED)
        ax.scatter([x], [y], s=70, color=c, edgecolors="#222222", lw=0.8, zorder=3)
        handles.append(plt.Line2D(
            [], [], marker="o", ls="", ms=7, color=c,
            label=f"{r['detector']} — AUROC {x:.3f}, "
                  f"TPR@{r['eff_fpr'] * 100:.3g}%FPR {y:.3f}"))

    n_h = rows[0]["n_human"]
    ax.set_xlabel("AUROC (machine = positive)", fontsize=9, color=_INK)
    ax.set_ylabel(f"oracle TPR @ effective FPR floor (finest achievable at "
                  f"n_human={n_h}; target was {fpr * 100:g}%FPR)",
                  fontsize=9, color=_INK)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.tick_params(labelsize=8, colors=_MUTED)
    ax.legend(handles=handles, fontsize=7.5, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), framealpha=0.9, borderaxespad=0.0,
              title="ranked by AUROC")
    ax.set_title(f"Zero-shot detectors on GB-orig — AUROC vs. oracle TPR\n"
                 f"(target {fpr * 100:g}%FPR unreachable at n_human={n_h}; "
                 f"Binoculars reports this at a transported 0.01%FPR cutoff, "
                 f"not an oracle fit on GB-orig alone)",
                 fontsize=10, color=_INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS,
                    metavar="LEVEL",
                    help=f"oracle FPR level(s) to record in the CSV; the plot "
                         f"always uses the first level's effective floor "
                         f"(default: {DEFAULT_FPRS})")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV")
    args = ap.parse_args()

    if args.replot is not None:
        df = pd.read_csv(args.replot)
        fpr = args.fpr[0]
        rows = df[np.isclose(df["fpr"], fpr)].to_dict("records")
        plot(rows, fpr, args.replot.parent / "gb_orig_zeroshot.png")
        print(f"replotted from {args.replot}")
        return

    splits = resolve_splits([SPLIT_KEY], PARQUET_DIR)
    path = splits[0][1]
    if not path.exists():
        raise SystemExit(f"parquet not found: {path}")

    print(f"split : {SPLIT_KEY} ({path.name})")
    print(f"metric: AUROC (oriented), oracle TPR@"
          f"{[f'{f * 100:g}%' for f in args.fpr]}FPR\n")

    rows = compute(splits, FAMILIES, None, args.fpr, args.human_label)
    rows = [r for r in rows if r["split"] == SPLIT_KEY]  # drop the redundant 'mean' dupes
    for r in rows:
        r["detector"] = _detector(r["family"], r["version"])
    rows.sort(key=lambda r: (float(r["fpr"]), -r["auroc"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "performance.csv"
    pd.DataFrame(rows, columns=OUT_FIELDS).to_csv(csv_path, index=False)

    print(pd.DataFrame(rows, columns=["detector", "fpr", "n_human", "auroc",
                                      "tpr_at_fpr", "reachable", "tpr_eff",
                                      "eff_fpr"]).to_string(index=False))
    print(f"\nwrote {csv_path}")
    first_fpr_rows = [r for r in rows if float(r["fpr"]) == args.fpr[0]]
    if first_fpr_rows and not any(r["reachable"] for r in first_fpr_rows):
        print(f"\nnote: {args.fpr[0] * 100:g}%FPR is unreachable on GB-orig "
              f"(n_human={rows[0]['n_human']}, needs >= {1 / args.fpr[0]:.0f}) -- "
              f"tpr_at_fpr reads 0.0 for every detector (the 'flag nobody' "
              f"point); tpr_eff / eff_fpr is the finest level this file can "
              f"actually resolve, and is what the plot uses.")

    if not args.no_plots:
        plot(first_fpr_rows, args.fpr[0], args.out_dir / "gb_orig_zeroshot.png")


if __name__ == "__main__":
    main()

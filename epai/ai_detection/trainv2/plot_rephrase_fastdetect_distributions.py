r"""Fast-DetectGPT score-distribution figure for the rephrase-robustness
paragraph: one panel per corpus (MG-GB, MG-Sci), each an overlaid density
histogram of four classes -- human, human_rephrased, ai, ai_rephrased.

Single detector (``fastdetect_crit_llama31_8b``, the same backbone the main
results table uses) and two panels only, no pooled "MG-All" third panel --
this is scoped to exactly the figure asked for, not the general multi-family
grid the (now deleted) ``plot_score_histograms.py`` drew. Colours: human
blue, human_rephrased dark blue, ai red, ai_rephrased dark red
(``rephrase_common.CLASS_COLORS``).

Also fits the three human-vs-other Youden cuts and AUROCs
(human|human_rephrased, human|ai, human|ai_rephrased) per corpus and writes
them to ``summary.csv`` alongside per-class n/mean/percentiles -- not
plotted, just recorded for the write-up.

Output (under ``--out-dir``, default
``results/trainv2/rephrase_fastdetect_distributions/``):
    fastdetect_distributions.pdf   2 panels, MG-GB / MG-Sci
    summary.csv                    one row per (corpus, class)

Run:
    uv run python epai/ai_detection/trainv2/plot_rephrase_fastdetect_distributions.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.cd_core import auroc_score, oriented_scores, youden_threshold

from epai.ai_detection.trainv2.rephrase_common import (
    CLASS_COLORS,
    CORPUS_ORDER,
    CORPUS_SOURCES,
    load_classed_frame,
)

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "rephrase_fastdetect_distributions"

COLUMN = "fastdetect_crit_llama31_8b"
MACHINE_IS_HIGH = True
XLABEL = "Fast-DetectGPT criterion"

CSV_FIELDS = [
    "corpus", "class", "n", "mean", "p1", "p50", "p99",
    "cut_human_rephrased", "cut_ai", "cut_ai_rephrased",
    "auroc_human_rephrased", "auroc_ai", "auroc_ai_rephrased",
]


def corpus_stats(sub: pd.DataFrame) -> dict:
    """Per-class percentiles + the human-vs-other Youden cuts/AUROCs for one
    corpus's rows."""
    vals = {cls: sub.loc[sub.cls == cls, COLUMN].to_numpy(dtype=float) for cls in CLASS_COLORS}
    vals = {cls: v[np.isfinite(v)] for cls, v in vals.items()}
    per_class = {
        cls: {
            "n": int(v.size),
            "mean": float(v.mean()) if v.size else float("nan"),
            "p1": float(np.percentile(v, 1)) if v.size else float("nan"),
            "p50": float(np.median(v)) if v.size else float("nan"),
            "p99": float(np.percentile(v, 99)) if v.size else float("nan"),
        }
        for cls, v in vals.items()
    }
    cuts, aurocs = {}, {}
    h = vals["human"]
    for other in (c for c in CLASS_COLORS if c != "human"):
        o = vals[other]
        if h.size < 10 or o.size < 10:
            cuts[other] = aurocs[other] = float("nan")
            continue
        y = np.concatenate([np.zeros(h.size), np.ones(o.size)])
        oriented = oriented_scores(np.concatenate([h, o]), MACHINE_IS_HIGH)
        thr_or = youden_threshold(y, oriented)
        cuts[other] = float(thr_or if MACHINE_IS_HIGH else -thr_or)
        aurocs[other] = float(auroc_score(y, oriented))
    return {"per_class": per_class, "cuts": cuts, "aurocs": aurocs}


def plot(df: pd.DataFrame, stats: dict[str, dict], path: Path, bins: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(CORPUS_ORDER), figsize=(4.7 * len(CORPUS_ORDER), 4.3),
                             squeeze=False)
    axes = axes[0]

    raw_all = df[COLUMN].to_numpy(dtype=float)
    raw_all = raw_all[np.isfinite(raw_all)]
    lo, hi = np.percentile(raw_all, [0.5, 99.5])
    pad = 0.03 * (hi - lo) or 0.01
    lo, hi = lo - pad, hi + pad
    edges = np.linspace(lo, hi, bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2

    for ax, corpus in zip(axes, CORPUS_ORDER):
        s = stats[corpus]
        sub = df[df.corpus == corpus]
        for cls, colour in CLASS_COLORS.items():
            v = sub.loc[sub.cls == cls, COLUMN].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            dens = np.histogram(v, bins=edges, density=True)[0]
            ax.step(mids, dens, where="mid", color=colour, lw=1.6, label=cls)
            ax.fill_between(mids, dens, step="mid", color=colour, alpha=0.13)

        a_hr, a_ai, a_air = (s["aurocs"]["human_rephrased"], s["aurocs"]["ai"],
                            s["aurocs"]["ai_rephrased"])
        title = f"{CORPUS_SOURCES[corpus]['display']}  (n={len(sub)})"
        if np.isfinite(a_hr) and np.isfinite(a_ai) and np.isfinite(a_air):
            title += (f"\nAUROC  h|hr {a_hr:.3f}   h|ai {a_ai:.3f}"
                      f"   h|air {a_air:.3f}")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(XLABEL)
        ax.set_xlim(lo, hi)
        ax.grid(True, lw=0.4, alpha=0.4)

    axes[0].set_ylabel("density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc="lower center",
               ncol=len(CLASS_COLORS), bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bins", type=int, default=45)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    df = load_classed_frame([COLUMN])

    stats = {corpus: corpus_stats(df[df.corpus == corpus]) for corpus in CORPUS_ORDER}
    plot(df, stats, args.out_dir / "fastdetect_distributions.pdf", args.bins)

    rows: list[dict] = []
    for corpus, s in stats.items():
        for cls, pc in s["per_class"].items():
            row = {"corpus": corpus, "class": cls, **pc}
            if cls == "human":
                row.update({
                    "cut_human_rephrased": s["cuts"]["human_rephrased"],
                    "cut_ai": s["cuts"]["ai"],
                    "cut_ai_rephrased": s["cuts"]["ai_rephrased"],
                    "auroc_human_rephrased": s["aurocs"]["human_rephrased"],
                    "auroc_ai": s["aurocs"]["ai"],
                    "auroc_ai_rephrased": s["aurocs"]["ai_rephrased"],
                })
            rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "summary.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

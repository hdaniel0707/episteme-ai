r"""Plots for ``run_sentinel_comparison.py``'s ``performance.csv``, reusing
``plots.zeroshot_scatter`` unchanged -- the same design as the paper's main
General Results figure (``fig:general-results-scatter`` / ``best_zeroshot_
1pct.png``): colour = detector, marker shape = MANUGRAM split, a bold circle
with error bars = that detector's mean, a grey AT-HM contour field behind
everything, both axes power-warped toward 1.0.

``zeroshot_scatter`` expects "detector x corpus" rows plus one ``split ==
"mean"`` row per detector; ``run_sentinel_comparison.py``'s CSV doesn't carry
that mean row (its own "overall" reading is folded into per-regime
aggregation instead), so :func:`_with_mean_rows` adds a synthetic one here --
unweighted mean of AUROC / TPR@FPR over that detector's real-split rows, SD =
mean of the per-split SDs (matches ``run_in_domain_performance._overall_
cell``'s convention: the typical per-corpus spread, not the spread across
corpora).

One figure per regime:

  in_domain_<fpr>.png   in-domain read on MG-GB / MG-Sci (own split) +
                        MG-AI-Sci (eval-only OOD)
  lodo_<fpr>.png        LODO ("close" OOD): mean-over-held-out-domains on
                        MG-GB / MG-Sci + MG-AI-Sci (eval-only OOD)
  far_ood_<fpr>.png     "far" OOD: cross-corpus transfer. A far-OOD point
                        depends on BOTH which corpus a model was trained on
                        and which it's scored on, so here the plotted
                        "detector" key is "<detector> (train=<CORPUS>)" --
                        the 4 resulting series (2 detectors x 2 training
                        corpora) each get their own colour/legend entry, and
                        "split" is the eval corpus.

Run:
    uv run python epai/ai_detection/trainv2/plot_sentinel_comparison.py \
        results/trainv2/sentinel_comparison_<timestamp>/performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.ai_detection.trainv2.plots import zeroshot_scatter
from epai.ai_detection.trainv2.splits import display as split_display

#: registry order for the marker legend -- matches splits.MARKER's shapes
#: (mg_gb = "v", mg_sci = "s", mg_aisci = "D"), same family the paper's main
#: scatter uses.
EVAL_ORDER = ["mg_gb", "mg_sci", "mg_aisci"]


def _std_mean(vals) -> float:
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _with_mean_rows(rows: list[dict], key: str) -> list[dict]:
    """``rows`` plus one synthetic ``split == "mean"`` row per ``key`` value --
    see the module docstring for the convention."""
    out = list(rows)
    for k in sorted({r[key] for r in rows}):
        sub = [r for r in rows if r[key] == k and np.isfinite(r["auroc"])]
        if not sub:
            continue
        out.append({
            key: k, "split": "mean",
            "auroc": float(np.mean([r["auroc"] for r in sub])),
            "auroc_std": _std_mean([r["auroc_std"] for r in sub]),
            "tpr_at_fpr": float(np.mean([r["tpr_at_fpr"] for r in sub])),
            "tpr_at_fpr_std": _std_mean([r["tpr_at_fpr_std"] for r in sub]),
        })
    return out


def _base_rows(sub: pd.DataFrame, detector_of) -> list[dict]:
    return [{
        "detector": detector_of(r), "split": r.eval_split,
        "auroc": r.auroc, "auroc_std": r.auroc_std,
        "tpr_at_fpr": r.tpr_at_fpr, "tpr_at_fpr_std": r.tpr_at_fpr_std,
    } for r in sub.itertuples()]


def _plot(df: pd.DataFrame, regime: str, fpr: float, out_path: Path, title: str,
          detector_of=lambda r: r.detector) -> None:
    sub = df[(df["regime"] == regime) & (df["level"] == "split") & (df["fpr"] == fpr)]
    rows = _with_mean_rows(_base_rows(sub, detector_of), "detector")
    zeroshot_scatter(rows, {"splits": EVAL_ORDER, "fpr": fpr}, out_path,
                     key="detector", title=title)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="run_sentinel_comparison.py's performance.csv")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: the CSV's own directory")
    args = ap.parse_args()
    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "kind" in df.columns:
        # run_sentinel_comparison.py now also emits kind="zeroshot" rows (for
        # plot_sentinel_scenarios.py); these per-regime scatters stay exactly
        # what they were -- the two trained detectors only.
        df = df[df["kind"] == "trained"]
    for fpr in sorted(df["fpr"].unique()):
        tag = f"{fpr * 100:g}pct"
        _plot(df, "in_domain", fpr, out_dir / f"in_domain_{tag}.png",
             f"In-domain (grouped 70/30) — RoBERTa-Sentinel vs. roberta·logistic\n"
             f"AUROC vs. oracle TPR@{fpr * 100:g}%FPR, MG-AI-Sci eval-only OOD")
        _plot(df, "lodo", fpr, out_dir / f"lodo_{tag}.png",
             f"LODO (\"close\" OOD) — RoBERTa-Sentinel vs. roberta·logistic\n"
             f"AUROC vs. oracle TPR@{fpr * 100:g}%FPR, MG-AI-Sci eval-only OOD")
        _plot(df, "far_ood", fpr, out_dir / f"far_ood_{tag}.png",
             f"Far OOD (train on the full corpus, score on the others)\n"
             f"AUROC vs. oracle TPR@{fpr * 100:g}%FPR",
             detector_of=lambda r: f"{r.detector} (train={split_display(r.train_split)})")
        print(f"wrote {out_dir}/{{in_domain,lodo,far_ood}}_{tag}.png")


if __name__ == "__main__":
    main()

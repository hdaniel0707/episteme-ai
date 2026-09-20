r"""Rephrase-robustness numbers for the 6 main-table scorers
(``rephrase_common.MAIN_TABLE_SCORERS``: FusionJudge, LogJudge, BaseJudge,
Fast-DetectGPT, Binoculars, RADAR), on MG-GB, MG-Sci, and their pool.

Three corpus rows per scorer: ``mg_gb``, ``mg_sci``, and ``mean`` --
MG-GB+MG-Sci rows concatenated and recomputed from scratch (not the average
of the two corpora's own numbers), the same convention
``build_general_results_table.py`` uses for its "Mean" column (a pooled
AUROC/TPR pair, then AT-HM off that pair -- never the mean of two AT-HMs).

Two CSVs, no plotting -- ``build_rephrase_table.py`` is the companion table
script:

``center_of_mass.csv`` -- one row per (corpus, scorer, class): n and the raw
score mean ("center of mass") for each of the four classes (human,
human_rephrased, ai, ai_rephrased). Raw, not orientation-flipped, so it
reads directly off each scorer's own native scale; ``machine_is_high`` is
carried alongside so a table/plot can tell which direction is "more AI-like".

``hm_delta.csv`` -- one row per (corpus, scorer): AUROC, oracle TPR@1%FPR,
and AT-HM (``core.goodness``, the paper's composite metric everywhere else)
for human-vs-machine WITHOUT rephrase (human vs. ai) and WITH rephrase
folded in as its own class (human+human_rephrased vs. ai+ai_rephrased), plus
the with-minus-without deltas. Both legs are computed here (not read off an
existing zero-shot-performance CSV) so the pair is always consistent -- same
code path, same oriented-score convention, same FPR level.

Run:
    uv run python epai/ai_detection/trainv2/run_rephrase_transfer.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.cd_core import auroc_score, oriented_scores, roc_point_at_fpr

from epai.ai_detection.trainv2.core import goodness
from epai.ai_detection.trainv2.rephrase_common import (
    CORPUS_ORDER,
    MAIN_TABLE_SCORERS,
    load_classed_frame,
)

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "rephrase_transfer"
DEFAULT_FPR = 0.01

MEAN_CORPUS = "mean"
OUTPUT_CORPORA = [*CORPUS_ORDER, MEAN_CORPUS]

COM_FIELDS = ["corpus", "scorer", "machine_is_high", "class", "n", "mean"]
DELTA_FIELDS = [
    "corpus", "scorer", "fpr",
    "auroc_without", "tpr_without", "athm_without",
    "auroc_with", "tpr_with", "athm_with",
    "delta_auroc", "delta_tpr", "delta_athm",
]


def _corpus_slice(df: pd.DataFrame, corpus: str) -> pd.DataFrame:
    """MG-GB / MG-Sci rows, or (``mean``) the two pooled -- ``df`` already
    holds both, so pooling is just "don't filter"."""
    return df if corpus == MEAN_CORPUS else df[df.corpus == corpus]


def center_of_mass_rows(df: pd.DataFrame, scorer: str, machine_is_high: bool) -> list[dict]:
    rows = []
    for corpus in OUTPUT_CORPORA:
        sub = _corpus_slice(df, corpus)
        for cls in ("human", "human_rephrased", "ai", "ai_rephrased"):
            v = sub.loc[sub.cls == cls, scorer].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            rows.append({
                "corpus": corpus, "scorer": scorer, "machine_is_high": machine_is_high,
                "class": cls, "n": int(v.size),
                "mean": float(v.mean()) if v.size else float("nan"),
            })
    return rows


def _auroc_tpr(sub: pd.DataFrame, scorer: str, human_classes: set, ai_classes: set,
              machine_is_high: bool, fpr: float) -> tuple[float, float]:
    mask = sub.cls.isin(human_classes | ai_classes)
    s = sub.loc[mask]
    y = s.cls.isin(ai_classes).to_numpy(dtype=int)
    raw = s[scorer].to_numpy(dtype=float)
    finite = np.isfinite(raw)
    y, raw = y[finite], raw[finite]
    oriented = oriented_scores(raw, machine_is_high)
    auroc = auroc_score(y, oriented)
    tpr, _ = roc_point_at_fpr(y, oriented, fpr)
    return auroc, tpr


def hm_delta_rows(df: pd.DataFrame, scorer: str, machine_is_high: bool, fpr: float) -> list[dict]:
    rows = []
    for corpus in OUTPUT_CORPORA:
        sub = _corpus_slice(df, corpus)
        auroc_wo, tpr_wo = _auroc_tpr(sub, scorer, {"human"}, {"ai"}, machine_is_high, fpr)
        auroc_w, tpr_w = _auroc_tpr(
            sub, scorer, {"human", "human_rephrased"}, {"ai", "ai_rephrased"},
            machine_is_high, fpr,
        )
        athm_wo = goodness(auroc_wo, tpr_wo, alpha=fpr)
        athm_w = goodness(auroc_w, tpr_w, alpha=fpr)
        rows.append({
            "corpus": corpus, "scorer": scorer, "fpr": fpr,
            "auroc_without": auroc_wo, "tpr_without": tpr_wo, "athm_without": athm_wo,
            "auroc_with": auroc_w, "tpr_with": tpr_w, "athm_with": athm_w,
            "delta_auroc": auroc_w - auroc_wo, "delta_tpr": tpr_w - tpr_wo,
            "delta_athm": athm_w - athm_wo,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fpr", type=float, default=DEFAULT_FPR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    columns = [col for _, col, _ in MAIN_TABLE_SCORERS]
    df = load_classed_frame(columns)

    com_rows, delta_rows = [], []
    for _, column, machine_is_high in MAIN_TABLE_SCORERS:
        com_rows += center_of_mass_rows(df, column, machine_is_high)
        delta_rows += hm_delta_rows(df, column, machine_is_high, args.fpr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    com_path = args.out_dir / "center_of_mass.csv"
    delta_path = args.out_dir / "hm_delta.csv"
    pd.DataFrame(com_rows, columns=COM_FIELDS).to_csv(com_path, index=False)
    pd.DataFrame(delta_rows, columns=DELTA_FIELDS).to_csv(delta_path, index=False)
    print(f"wrote {com_path}")
    print(f"wrote {delta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

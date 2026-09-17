r"""Per-domain breakdown for the hand-picked General Results detectors
(``run_best_zeroshot_performance.py``'s config), for every MANUGRAM core
split whose parquet actually carries a ``domain`` column with more than one
value -- essay/reuter/wp for GB-orig and MG-GB, computer_science/mathematics/
sociology for MG-Sci. MG-AI-Sci's ten AI-Scientist-source "domains" (8-40
rows each) are scored too, but read them as a floor even more than the
split-level MG-AI-Sci row already is -- some have single-digit human counts.

Same oriented-AUROC + oracle-TPR@1%FPR scoring as run_zeroshot_performance.py
/ run_best_zeroshot_performance.py (``eff_fpr``/``tpr_eff`` cover the
1%-FPR-unreachable-at-this-n case exactly as it already does at the split
level), just filtered to each domain subset before scoring -- independent of,
and cross-checkable against, the split-level performance.csv.

compute -> CSV:
    results/trainv2/best_zeroshot_performance/performance_by_domain.csv

Run:
    .venv/bin/python epai/ai_detection/trainv2/run_best_zeroshot_domains.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from epai.ai_detection.trainv2.core import eff_fpr_level
from epai.ai_detection.trainv2.run_best_zeroshot_performance import (
    DEFAULT_CONFIG,
    _detector,
    _load_columns,
)
from epai.ai_detection.trainv2.run_zeroshot_performance import FAMILIES, GEN_COL
from epai.ai_detection.trainv2.splits import resolve as resolve_splits
from epai.utils.cd_core import (
    auroc_score,
    oriented_scores,
    rates_at_threshold,
    roc_point_at_fpr,
)

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT = (BASE_DIR / "results" / "trainv2" / "best_zeroshot_performance"
               / "performance_by_domain.csv")
FPR = 0.01
HUMAN_LABEL = "human"

FIELDS = ["detector", "family", "version", "column", "split", "domain", "fpr",
          "n", "n_human", "n_machine", "auroc", "tpr_at_fpr", "fpr_at_fpr",
          "reachable", "tpr_eff", "eff_fpr"]


def _family_for(col: str) -> tuple[str, dict] | None:
    for fam, spec in FAMILIES.items():
        if col.startswith(spec["prefix"]):
            return fam, spec
    return None


def compute_domains(splits: list[tuple[str, Path]], cols: list[str],
                     fpr: float) -> list[dict]:
    rows: list[dict] = []
    for split_key, path in splits:
        names = pq.read_schema(path).names
        if "domain" not in names:
            continue
        want = {GEN_COL, "domain"} | {c for c in cols if c in names}
        df = pd.read_parquet(path, columns=sorted(want))
        if df["domain"].nunique() < 2:
            continue  # a single "domain" isn't a breakdown

        for domain, ddf in df.groupby("domain"):
            for col in cols:
                if col not in ddf.columns:
                    continue
                fam_spec = _family_for(col)
                if fam_spec is None:
                    continue
                fam, spec = fam_spec
                sub = ddf[[col, GEN_COL]].dropna(subset=[col])
                raw = sub[col].to_numpy(dtype=float)
                y = (sub[GEN_COL].astype(str) != HUMAN_LABEL).to_numpy().astype(int)
                if np.unique(y).size < 2 or len(y) < 10:
                    continue
                oriented = oriented_scores(raw, spec["machine_is_high"])
                auroc_v = float(auroc_score(y, oriented))
                n_h = int((y == 0).sum())
                tpr, thr = roc_point_at_fpr(y, oriented, fpr)
                _, fpr_act = rates_at_threshold(y, oriented, thr)
                eff = eff_fpr_level(n_h, fpr)
                tpr_eff, _ = roc_point_at_fpr(y, oriented, eff)
                rows.append({
                    "detector": _detector(fam, spec["label"](col)),
                    "family": fam, "version": spec["label"](col), "column": col,
                    "split": split_key, "domain": str(domain), "fpr": fpr,
                    "n": int(len(y)), "n_human": n_h,
                    "n_machine": int((y == 1).sum()),
                    "auroc": auroc_v, "tpr_at_fpr": float(tpr),
                    "fpr_at_fpr": float(fpr_act),
                    "reachable": bool(n_h * fpr >= 1.0 and np.isfinite(thr)),
                    "tpr_eff": float(tpr_eff), "eff_fpr": float(eff),
                })
    return rows


def main() -> None:
    cols = _load_columns(DEFAULT_CONFIG)
    splits = resolve_splits([], PARQUET_DIR)  # the four core splits
    rows = compute_domains(splits, cols, FPR)
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=FIELDS).to_csv(DEFAULT_OUT, index=False)
    print(f"wrote {DEFAULT_OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

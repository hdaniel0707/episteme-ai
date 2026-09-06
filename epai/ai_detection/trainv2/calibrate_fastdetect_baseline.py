r"""Calibrate one fixed, 1%-FPR threshold per Fast-DetectGPT pair into a baseline config.

    uv run python epai/ai_detection/trainv2/calibrate_fastdetect_baseline.py \
        data/parquet/science_v3f_multi_0811A.parquet \
        -o cfg/ai_detection/trainv2/fastdetect_baseline_calibrated.yaml

Writes a ``fastdetect_baseline`` block -- the SAME config shape
``epai.utils.cd_core.BASELINE_KINDS`` already reads for a zero-shot score
baseline (``column``/``threshold``/``label`` entries under ``columns:``), so
the result is usable both by ``run_fastdetect_performance.py`` (this
experiment) and, unchanged, as a real baseline block in a
``run_clf_cross_domain.py`` training config.

Only ``fastdetect_crit_*`` columns are calibrated -- the ``_prob_``/``_ntok_``
columns are not scores (see ``cfg/detector_suite.yaml``). A HIGH criterion
means machine here, the opposite direction from Binoculars and the LLM judge
over the same model pairs.

ONE dataset, POOLED over every domain ('all'), fixes ONE threshold per pair --
not per domain, not per dataset -- because the point of calibrating is to
apply the SAME cutoff to every corpus a later comparison looks at
(``run_fastdetect_performance.py``), not to refit a new one per corpus. Fit it
on a single, large, already-fully-scored reference corpus: the default,
``science_v3f_multi_0811A.parquet``, carries 3000 human rows, so the 1% cut
sits on ~30 of them. A small corpus is a bad calibration source for the same
reason cd_core's own docs call out -- ``denario_hc.parquet``'s 80 human rows
would place a 1%-FPR cut on well under one row, which is not a threshold, it
is noise.

This is the Fast-DetectGPT sibling of ``calibrate_judge_baseline.py`` /
``calibrate_binox_baseline.py``.

Run:
    uv run python epai/ai_detection/trainv2/calibrate_fastdetect_baseline.py
    uv run python epai/ai_detection/trainv2/calibrate_fastdetect_baseline.py \
        data/parquet/science_v3f_multi_0811A.parquet --fpr-target 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from epai.utils.cd_core import auroc_score, oriented_scores, rates_at_threshold, roc_point_at_fpr

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = BASE_DIR / "data" / "parquet" / "science_v3f_multi_0811A.parquet"
DEFAULT_OUT = BASE_DIR / "cfg" / "ai_detection" / "trainv2" / "fastdetect_baseline_calibrated.yaml"

FASTDETECT_PREFIX = "fastdetect_crit_"
GEN_COL = "generated_by"
# Fast-DetectGPT: a HIGH criterion means machine. Fixed, not a flag -- this
# script is Fast-DetectGPT-only, and matches cd_core's own fastdetect
# BaselineKind.
MACHINE_IS_HIGH = True


def _pair_label(column: str) -> str:
    """``fastdetect_crit_qwen25_7b`` -> ``qwen25-7b`` -- matches the pair keys
    ``cfg/detector_suite.yaml`` and ``run_holdout_pipeline.py`` use (and the
    labels ``calibrate_binox_baseline.py`` writes for the same model pairs)."""
    return column.removeprefix(FASTDETECT_PREFIX).replace("_", "-")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET,
                     help=f"calibration source, pooled over domains "
                          f"(default: {DEFAULT_DATASET.relative_to(BASE_DIR)})")
    ap.add_argument("--cols", nargs="+", default=None,
                     help="Fast-DetectGPT columns (default: every fastdetect_crit_* column)")
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--human-label", default="human")
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    if GEN_COL not in df.columns:
        raise SystemExit(f"{args.dataset}: no {GEN_COL!r} column")
    cols = args.cols or sorted(c for c in df.columns if c.startswith(FASTDETECT_PREFIX))
    if not cols:
        raise SystemExit(f"no {FASTDETECT_PREFIX}* column in {args.dataset}")

    print(f"calibrating on {args.dataset}  ({len(df)} rows, "
          f"target FPR {args.fpr_target:g})\n")
    print(f"{'pair':<24}{'n':>7}{'auroc':>8}{'threshold':>11}{'tpr':>8}{'fpr':>8}")

    entries: list[dict] = []
    for col in cols:
        sub = df[[col, GEN_COL]].dropna(subset=[col])
        raw = sub[col].to_numpy(dtype=float)
        y = (sub[GEN_COL].astype(str) != args.human_label).to_numpy().astype(int)
        if np.unique(y).size < 2 or len(y) < 10:
            print(f"{col:<24}  -- skipped: single class or n<10 --")
            continue

        oriented = oriented_scores(raw, MACHINE_IS_HIGH)
        tpr, thr_or = roc_point_at_fpr(y, oriented, args.fpr_target)
        _, fpr_actual = rates_at_threshold(y, oriented, thr_or)
        # oriented = raw here (MACHINE_IS_HIGH is True), so the oriented
        # threshold IS the raw threshold a caller applies to the raw column.
        thr_raw = thr_or if MACHINE_IS_HIGH else -thr_or

        print(f"{col:<24}{len(y):>7}{auroc_score(y, oriented):>8.3f}"
              f"{thr_raw:>11.3g}{tpr:>8.3f}{fpr_actual:>8.3f}")
        entries.append({
            "column": col,
            "threshold": round(float(thr_raw), 6),
            "label": _pair_label(col),
        })

    if not entries:
        raise SystemExit("\nnothing calibrated -- every Fast-DetectGPT column was skipped")

    cfg = {"fastdetect_baseline": {"columns": entries}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Calibrated by calibrate_fastdetect_baseline.py on {args.dataset.name}\n"
        f"# target FPR {args.fpr_target:g}, pooled over every domain ('all').\n"
        f"# Read by cd_core.baseline_entries() / run_fastdetect_performance.py.\n"
    )
    args.out.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

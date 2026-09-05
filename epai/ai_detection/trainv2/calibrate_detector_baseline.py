r"""Calibrate one fixed, 1%-FPR threshold per pretrained detector into a baseline config.

    uv run python epai/ai_detection/trainv2/calibrate_detector_baseline.py \
        data/parquet/science_v3_multi_0811A.parquet \
        -o cfg/ai_detection/trainv2/detector_baseline_calibrated.yaml

Writes a ``detector_baseline`` block -- the SAME config shape
``epai.utils.cd_core.BASELINE_KINDS`` already reads for a zero-shot score
baseline (``column``/``threshold``/``label`` entries under ``columns:``), so
the result is usable both by ``run_detector_performance.py`` (this
experiment) and, unchanged, as a real baseline block in a
``run_clf_cross_domain.py`` training config.

Only ``detector_score_*`` columns are calibrated -- ``detector_ntok_*`` is a
token count, not a score. A HIGH score means machine here (RADAR emits
P(AI)), the same direction as Fast-DetectGPT.

In this repo's corpora there is currently only one column in this family,
``detector_score_radar`` -- ``pangram_roberta`` is gated on the Hub and never
scored -- but the script reads whatever ``detector_score_*`` columns are
actually present, so a newly-unlocked detector needs no change here.

ONE dataset, POOLED over every domain ('all'), fixes ONE threshold per
detector -- not per domain, not per dataset -- for the same reason as its
Binoculars / Fast-DetectGPT / judge siblings: apply the SAME cutoff to every
corpus a later comparison looks at, not a fresh one per corpus. Fit it on a
single, large, already-fully-scored reference corpus: the default,
``science_v3_multi_0811A.parquet``, carries 6000 human rows, so the 1% cut
sits on ~60 of them.

This is the pretrained-detector sibling of ``calibrate_judge_baseline.py`` /
``calibrate_binox_baseline.py`` / ``calibrate_fastdetect_baseline.py``.

Run:
    uv run python epai/ai_detection/trainv2/calibrate_detector_baseline.py
    uv run python epai/ai_detection/trainv2/calibrate_detector_baseline.py \
        data/parquet/science_v3_multi_0811A.parquet --fpr-target 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from epai.utils.cd_core import auroc_score, oriented_scores, rates_at_threshold, roc_point_at_fpr

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = BASE_DIR / "data" / "parquet" / "science_v3_multi_0811A.parquet"
DEFAULT_OUT = BASE_DIR / "cfg" / "ai_detection" / "trainv2" / "detector_baseline_calibrated.yaml"

DETECTOR_PREFIX = "detector_score_"
GEN_COL = "generated_by"
# The pretrained detector emits P(AI): a HIGH score means machine. Fixed, not
# a flag -- this script is detector-only, and matches cd_core's own detector
# BaselineKind.
MACHINE_IS_HIGH = True


def _detector_label(column: str) -> str:
    """``detector_score_radar`` -> ``radar``."""
    return column.removeprefix(DETECTOR_PREFIX)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET,
                     help=f"calibration source, pooled over domains "
                          f"(default: {DEFAULT_DATASET.relative_to(BASE_DIR)})")
    ap.add_argument("--cols", nargs="+", default=None,
                     help="detector columns (default: every detector_score_* column)")
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--human-label", default="human")
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    if GEN_COL not in df.columns:
        raise SystemExit(f"{args.dataset}: no {GEN_COL!r} column")
    cols = args.cols or sorted(c for c in df.columns if c.startswith(DETECTOR_PREFIX))
    if not cols:
        raise SystemExit(f"no {DETECTOR_PREFIX}* column in {args.dataset}")

    print(f"calibrating on {args.dataset}  ({len(df)} rows, "
          f"target FPR {args.fpr_target:g})\n")
    print(f"{'detector':<20}{'n':>7}{'auroc':>8}{'threshold':>11}{'tpr':>8}{'fpr':>8}")

    entries: list[dict] = []
    for col in cols:
        sub = df[[col, GEN_COL]].dropna(subset=[col])
        raw = sub[col].to_numpy(dtype=float)
        y = (sub[GEN_COL].astype(str) != args.human_label).to_numpy().astype(int)
        if np.unique(y).size < 2 or len(y) < 10:
            print(f"{col:<20}  -- skipped: single class or n<10 --")
            continue

        oriented = oriented_scores(raw, MACHINE_IS_HIGH)
        tpr, thr_or = roc_point_at_fpr(y, oriented, args.fpr_target)
        _, fpr_actual = rates_at_threshold(y, oriented, thr_or)
        # oriented = raw here (MACHINE_IS_HIGH is True), so the oriented
        # threshold IS the raw threshold a caller applies to the raw column.
        thr_raw = thr_or if MACHINE_IS_HIGH else -thr_or

        print(f"{col:<20}{len(y):>7}{auroc_score(y, oriented):>8.3f}"
              f"{thr_raw:>11.3g}{tpr:>8.3f}{fpr_actual:>8.3f}")
        entries.append({
            "column": col,
            "threshold": round(float(thr_raw), 6),
            "label": _detector_label(col),
        })

    if not entries:
        raise SystemExit("\nnothing calibrated -- every detector column was skipped")

    cfg = {"detector_baseline": {"columns": entries}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Calibrated by calibrate_detector_baseline.py on {args.dataset.name}\n"
        f"# target FPR {args.fpr_target:g}, pooled over every domain ('all').\n"
        f"# Read by cd_core.baseline_entries() / run_detector_performance.py.\n"
    )
    args.out.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

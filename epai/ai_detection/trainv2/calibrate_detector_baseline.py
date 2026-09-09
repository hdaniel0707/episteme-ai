r"""Calibrate one fixed, 1%-FPR threshold per pretrained detector into a baseline config.

    uv run python epai/ai_detection/trainv2/calibrate_detector_baseline.py \
        data/parquet/manugram_sci.parquet \
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

ONE dataset, POOLED over every domain ('all'), fixes ONE threshold per
detector -- not per domain, not per dataset -- for the same reason as its
Binoculars / Fast-DetectGPT / judge siblings: apply the SAME cutoff to every
corpus a later comparison looks at, not a fresh one per corpus.

``--corpora`` calibrates on each MANUGRAM split (``gb_orig``,
``mg_gb``, ``mg_sci``, ``mg_den``) AND on all of them
pooled ('all'), prints a per-corpus AUROC / threshold / rate / percentile
table, drops one distribution PNG per detector column, and writes the YAML
from the pooled 'all' row:

    uv run python epai/ai_detection/trainv2/calibrate_detector_baseline.py --corpora all

``--generators gpt56luna`` restricts the machine class to gpt-5.6-luna rows and
writes to ``results/trainv2/baseline_calibration_[flag=gpt56luna]/`` (YAML there
too, not ``cfg/``):

    uv run python epai/ai_detection/trainv2/calibrate_detector_baseline.py --corpora all --generators gpt56luna

Run:
    uv run python epai/ai_detection/trainv2/calibrate_detector_baseline.py
    uv run python epai/ai_detection/trainv2/calibrate_detector_baseline.py \
        data/parquet/manugram_sci.parquet --fpr-target 0.05
"""

from __future__ import annotations

from epai.ai_detection.trainv2.calibrate_common import DEFAULT_OUT_DIR, PARQUET_DIR, run

DETECTOR_PREFIX = "detector_score_"


def _detector_label(column: str) -> str:
    """``detector_score_radar`` -> ``radar``."""
    return column.removeprefix(DETECTOR_PREFIX)


if __name__ == "__main__":
    run(
        script="calibrate_detector_baseline.py",
        doc=__doc__,
        prefix=DETECTOR_PREFIX,
        # RADAR emits P(AI): a HIGH score means machine. Matches cd_core's own
        # detector BaselineKind.
        machine_is_high=True,
        block_key="detector_baseline",
        label_fn=_detector_label,
        family="detector",
        xlabel="detector score  P(AI)",
        default_dataset=PARQUET_DIR / "manugram_sci.parquet",
        default_out=DEFAULT_OUT_DIR / "detector_baseline_calibrated.yaml",
    )

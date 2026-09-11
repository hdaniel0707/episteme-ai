r"""Calibrate one fixed, 1%-FPR threshold per Fast-DetectGPT pair into a baseline config.

    uv run python epai/ai_detection/trainv2/calibrate_fastdetect_baseline.py \
        data/parquet/manugram_sci.parquet \
        -o cfg/ai_detection/trainv2/fastdetect_baseline_calibrated.yaml

Writes a ``fastdetect_baseline`` block -- the SAME config shape
``epai.utils.cd_core.BASELINE_KINDS`` already reads for a zero-shot score
baseline (``column``/``threshold``/``label`` entries under ``columns:``), so
the result drops in unchanged as a real baseline block in a
``run_clf_cross_domain.py`` training config.

Only ``fastdetect_crit_*`` columns are calibrated -- the ``_prob_``/``_ntok_``
columns are not scores (see ``cfg/detector_suite.yaml``). A HIGH criterion
means machine here, the opposite direction from Binoculars and the LLM judge
over the same model pairs.

ONE dataset, POOLED over every domain ('all'), fixes ONE threshold per pair --
not per domain, not per dataset -- because the point of calibrating is to
apply the SAME cutoff to every corpus a later comparison looks at, not to
refit a new one per corpus. A small corpus is a bad calibration source:
``manugram_aiscientist.parquet``'s 100 human rows would place a 1%-FPR cut on
right around one row, which is still too noisy to trust as a threshold.

``--corpora`` calibrates on each MANUGRAM split (``gb_orig``,
``mg_gb``, ``mg_sci``, ``mg_aisci``) AND on all of them
pooled ('all'), prints a per-corpus AUROC / threshold / rate / percentile
table, drops one distribution PNG per pair, and writes the YAML from the
pooled 'all' row:

    uv run python epai/ai_detection/trainv2/calibrate_fastdetect_baseline.py --corpora all

``--generators gpt56luna`` restricts the machine class to gpt-5.6-luna rows and
writes to ``results/trainv2/baseline_calibration_[flag=gpt56luna]/`` (YAML there
too, not ``cfg/``):

    uv run python epai/ai_detection/trainv2/calibrate_fastdetect_baseline.py --corpora all --generators gpt56luna

Run:
    uv run python epai/ai_detection/trainv2/calibrate_fastdetect_baseline.py
    uv run python epai/ai_detection/trainv2/calibrate_fastdetect_baseline.py \
        data/parquet/manugram_sci.parquet --fpr-target 0.05
"""

from __future__ import annotations

from epai.ai_detection.trainv2.calibrate_common import DEFAULT_OUT_DIR, PARQUET_DIR, run

FASTDETECT_PREFIX = "fastdetect_crit_"


def _pair_label(column: str) -> str:
    """``fastdetect_crit_qwen25_7b`` -> ``qwen25-7b`` -- matches the pair keys
    ``cfg/detector_suite.yaml`` and ``run_holdout_pipeline.py`` use (and the
    labels ``calibrate_binox_baseline.py`` writes for the same model pairs)."""
    return column.removeprefix(FASTDETECT_PREFIX).replace("_", "-")


if __name__ == "__main__":
    run(
        script="calibrate_fastdetect_baseline.py",
        doc=__doc__,
        prefix=FASTDETECT_PREFIX,
        # Fast-DetectGPT: a HIGH criterion means machine. Matches cd_core's
        # own fastdetect BaselineKind.
        machine_is_high=True,
        block_key="fastdetect_baseline",
        label_fn=_pair_label,
        family="pair",
        xlabel="Fast-DetectGPT criterion",
        default_dataset=PARQUET_DIR / "manugram_sci.parquet",
        default_out=DEFAULT_OUT_DIR / "fastdetect_baseline_calibrated.yaml",
    )

r"""Calibrate one fixed, 1%-FPR threshold per LLM judge into a baseline config.

    uv run python epai/ai_detection/trainv2/calibrate_judge_baseline.py \
        data/parquet/manugram_sci.parquet \
        -o cfg/ai_detection/trainv2/judge_baseline_calibrated.yaml

Writes an ``llm_judge_baseline`` block -- the SAME config shape
``epai.utils.cd_core.BASELINE_KINDS`` already reads for a zero-shot score
baseline (``column``/``threshold``/``label`` entries under ``columns:``), so
the result drops in unchanged as a real baseline block in a
``run_clf_cross_domain.py`` training config.

ONE dataset, POOLED over every domain ('all'), fixes ONE threshold per judge
-- not per domain, not per dataset -- because the point of calibrating is to
apply the SAME cutoff to every corpus a later comparison looks at, not to
refit a new one per corpus. A small
corpus is a bad calibration source: ``manugram_aiscientist.parquet``'s 100 human rows
would place a 1%-FPR cut on right around one row, which is still too noisy to
trust as a threshold.

``--corpora`` calibrates on each MANUGRAM split (``gb_orig``,
``mg_gb``, ``mg_sci``, ``mg_aisci``) AND on all of them
pooled ('all'), prints a per-corpus AUROC / threshold / rate / percentile
table, drops one distribution PNG per judge, and writes the YAML from the
pooled 'all' row:

    uv run python epai/ai_detection/trainv2/calibrate_judge_baseline.py --corpora all

``--generators gpt56luna`` restricts the machine class to gpt-5.6-luna rows and
writes to ``results/trainv2/baseline_calibration_[flag=gpt56luna]/`` (YAML there
too, not ``cfg/``):

    uv run python epai/ai_detection/trainv2/calibrate_judge_baseline.py --corpora all --generators gpt56luna

Run:
    uv run python epai/ai_detection/trainv2/calibrate_judge_baseline.py
    uv run python epai/ai_detection/trainv2/calibrate_judge_baseline.py \
        data/parquet/manugram_sci.parquet --fpr-target 0.05
"""

from __future__ import annotations

from epai.ai_detection.trainv2.calibrate_common import DEFAULT_OUT_DIR, PARQUET_DIR, run

JUDGE_PREFIX = "llm_judge_score_"

# sonnet5 is scored on MG-GB / MG-AI-Sci but not on MG-Sci,
# and is not part of the trainv2 judge suite -- keep it out of the calibrated
# baseline. Pass it explicitly via --cols to calibrate it anyway.
JUDGE_EXCLUDE = frozenset({"llm_judge_score_sonnet5_v3"})


# backbone column slug -> the paper's own spelling (see iclr2027_conference.tex)
_BACKBONE_DISPLAY = {
    "gpt56luna": "gpt-5.6-luna",
    "gemma4_31b": "gemma-4-31b",
    "llama33_70b": "llama-3.3-70b",
    "gptoss_120b": "gpt-oss-120b",
    "qwen3_14b": "qwen3-14b",
}


def _judge_label(column: str) -> str:
    """``llm_judge_score_gpt56luna_v3`` -> ``BaseJudge (gpt-5.6-luna)``;
    ``llm_judge_score_gpt56luna_v3_wlp`` -> ``LogJudge (gpt-5.6-luna)`` --
    ``wlp`` is the windowed-logprob readout, identical in formula to the
    paper's LogJudge (``score_logprob_windows.py``'s ``t0000``/``ord2``).
    Any other backbone or readout variant (cand03/cand10, slp, sbc,
    logjudge_t0000/t0001) falls back to the plain slug -- those aren't one of
    the paper's two named readouts, so they get no canonical name."""
    slug = column.removeprefix(JUDGE_PREFIX).replace("_v3", "")
    if slug.endswith("_wlp"):
        backbone, family = slug[: -len("_wlp")], "LogJudge"
    else:
        backbone, family = slug, "BaseJudge"
    pretty = _BACKBONE_DISPLAY.get(backbone)
    return f"{family} ({pretty})" if pretty else slug


if __name__ == "__main__":
    run(
        script="calibrate_judge_baseline.py",
        doc=__doc__,
        prefix=JUDGE_PREFIX,
        # The judge writes P(human): a LOW score means machine. Matches
        # cd_core's own llm_judge BaselineKind.
        machine_is_high=False,
        block_key="llm_judge_baseline",
        label_fn=_judge_label,
        family="judge",
        xlabel="P(human)",  # the column-specific label already says "judge"
        default_dataset=PARQUET_DIR / "manugram_sci.parquet",
        default_out=DEFAULT_OUT_DIR / "judge_baseline_calibrated.yaml",
        exclude_cols=JUDGE_EXCLUDE,
    )

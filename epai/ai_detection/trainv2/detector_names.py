"""Canonical display names for the trainv2 General Results detectors, shared
by plots.py's AUROC-vs-TPR scatter and build_general_results_table.py's
LaTeX tables -- one lookup, so the same raw detector id always prints the
same way in the figure and every table variant.

Convention: ``<family>/<LLM version-size>`` (e.g. ``binoculars/llama-3.1-8b``),
matching how the paper's prose spells these backbones elsewhere
(``gpt-5.6-luna``, ``gemma-4-31b``, ``llama-3.3-70b``). The raw ids are the
column-safe versions (``gpt56luna``, ``gemma4_31b``, ``llama31-8b``), each
cleaned up by hand below rather than by a generic regex, since the three
don't sanitize the same way. RADAR is the one exception to the pattern --
it has no backbone LLM to name, so it's just ``radar``.

``judge/<x>_ord2`` -> ``logjudge/<x>``: the ``ord2`` readout
(``score_logjudge_ord2.py``, the paper's Eq. logjudge_simple, exactly) is
the paper's LogJudge contribution, so it gets that family name instead of
the generic ``judge/`` one. ``fusionjudge/<x>`` (``score_fusionjudge_
gpt56luna_v3.py``, the mean of LogJudge and the parsed-answer judge) is our
other contribution. Both get the bold treatment (:func:`is_judge_family`);
the plain parsed-answer judge baseline (``judge/<x>``, no suffix) does not.
"""

from __future__ import annotations

#: raw ``detector`` id (run_best_zeroshot_performance.py's ``_detector()``,
#: "<family-abbr>/<version>") -> display name.
DETECTOR_DISPLAY: dict[str, str] = {
    "judge/gpt56luna_ord2": "logjudge/gpt-5.6-luna",
    "judge/gpt56luna": "judge/gpt-5.6-luna",
    "judge/gemma4_31b_ord2": "logjudge/gemma-4-31b",
    "judge/gemma4_31b": "judge/gemma-4-31b",
    "fusionjudge/gpt56luna_v3": "fusionjudge/gpt-5.6-luna",
    "binox/llama31-8b": "binoculars/llama-3.1-8b",
    "fastdetect/llama31-8b": "fastdetect/llama-3.1-8b",
    "detector/radar": "radar",
}


def display_detector(name: str) -> str:
    """``<x>_reph`` (``rephrase_best_zeroshot_performance.py``'s baseline-vs-
    rephrase shift figure) displays as its base's name with a suffix -- the
    hollow/filled marker already carries the baseline/rephrase distinction,
    this just keeps the legend text unambiguous out of sorted order."""
    if name.endswith("_reph"):
        return f"{display_detector(name[: -len('_reph')])} (rephrase)"
    return DETECTOR_DISPLAY.get(name, name)


def is_judge_family(name: str) -> bool:
    """True only for our two zero-shot contributions -- LogJudge (the
    ``_ord2`` logprob readout) and FusionJudge (the mean of LogJudge and the
    parsed-answer judge) -- the bold-in-figure-and-table rule. The plain
    parsed-answer judge baseline is deliberately NOT bolded. Keyed on the RAW
    id (before :func:`display_detector`)."""
    return name.endswith("_ord2") or name.startswith("fusionjudge/")

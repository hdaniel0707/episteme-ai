r"""Shared data loading for the MG-Reph (rephrase-robustness) analysis
scripts: :mod:`plot_rephrase_fastdetect_distributions` and
:mod:`run_rephrase_transfer`.

Both corpora (MG-GB, MG-Sci) are assembled the same way: the ``human`` and
``ai`` (i.e. original, non-rephrased) classes come from the full MANUGRAM
parquet -- much larger N, and the source of every other results-section
number -- while ``human_rephrased`` and ``ai_rephrased`` only exist in the
dedicated ``*_reph_*`` parquet. That parquet's own human/ai rows are
duplicates of the same documents (same ``id``, suite scores copied whole),
so they are not read from there. This assembly is lifted from the (now
deleted) ``trainv2/plot_score_histograms.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"

GEN_COL = "generated_by"
HUMAN = "human"
REPHRASE_GEN = "gpt56luna:0908R"
AI_REPHRASE_GEN = "gpt56luna:0916R"

#: corpus key -> display name, source parquet, and the machine-generation id.
CORPUS_SOURCES: dict[str, dict] = {
    "mg_gb": {"display": "MG-GB", "full": "manugram_gb.parquet",
              "rewrite_gen": "gpt56luna:0701A", "reph": "manugram_reph_gb.parquet"},
    "mg_sci": {"display": "MG-Sci", "full": "manugram_sci.parquet",
               "rewrite_gen": "gpt56luna:0811A", "reph": "manugram_reph_sci.parquet"},
}
CORPUS_ORDER = list(CORPUS_SOURCES)

#: class -> colour, in legend/draw order. human=blue, human_rephrased=dark
#: blue, ai=red, ai_rephrased=dark red -- unchanged from the deleted
#: histogram script.
CLASS_COLORS: dict[str, str] = {
    "human": "#1c6fb3", "human_rephrased": "#0a2f5c",
    "ai": "#d92d20", "ai_rephrased": "#7a1710",
}
CLASS_ORDER = list(CLASS_COLORS)


def load_classed_frame(columns: list[str]) -> pd.DataFrame:
    """One row per document per corpus, columns ``[corpus, cls, *columns]``.

    ``cls`` is one of ``human``, ``human_rephrased``, ``ai``, ``ai_rephrased``.
    """
    need = [GEN_COL, *columns]
    parts = []
    for key, c in CORPUS_SOURCES.items():
        full = pd.read_parquet(PARQUET_DIR / c["full"], columns=need)
        reph = pd.read_parquet(PARQUET_DIR / c["reph"], columns=need)
        picks = {
            "human": full[full[GEN_COL] == HUMAN],
            "human_rephrased": reph[reph[GEN_COL] == REPHRASE_GEN],
            "ai": full[full[GEN_COL] == c["rewrite_gen"]],
            "ai_rephrased": reph[reph[GEN_COL] == AI_REPHRASE_GEN],
        }
        for cls, rows in picks.items():
            if rows.empty:
                raise SystemExit(f"{key}: class {cls!r} has no rows")
            parts.append(rows[columns].assign(corpus=key, cls=cls))
    return pd.concat(parts, ignore_index=True)[["corpus", "cls", *columns]]


#: 5 of the 6 scorers from the main results table (``tab:general-results``),
#: as (algo key, score column, machine_is_high). RADAR is left out here on
#: purpose -- its AUROC is below chance on MG-GB/MG-Sci, so its AT-HM floors
#: to exactly 0 in every rephrase reading (same floor noted in
#: ``sec:results_training_comparison``) and it carries no information. Algo
#: keys and orientation match ``build_general_results_table.TABLE_LABEL``
#: and ``run_zeroshot_performance.FAMILIES`` exactly, so
#: ``build_general_results_table._table_label(algo)`` gives the identical
#: display name/citation used in the main table.
MAIN_TABLE_SCORERS: list[tuple[str, str, bool]] = [
    ("fusionjudge/gpt56luna_v3", "fusionjudge_score", False),
    ("judge/gpt56luna_ord2", "logjudge_ord2_gpt56luna_v3", False),
    ("judge/gpt56luna", "llm_judge_score_gpt56luna_v3", False),
    ("fastdetect/llama31-8b", "fastdetect_crit_llama31_8b", True),
    ("binox/llama31-8b", "binoculars_score_llama31_8b", False),
]

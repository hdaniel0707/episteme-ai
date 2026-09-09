"""Shared feature loading for the rephrase trained-detector scripts
(``rephrase_in_domain_performance.py`` / ``rephrase_lodo_performance.py``).

Same features and detectors as :mod:`._trained` (``emb_roberta`` + the
``gpt56luna`` judge cue vector -- no gemma anywhere), but the rows are the
three-class rephrase set and the label counts **rephrase as human**:

    y = 1   rewritten   the full gpt56luna generation ('...:0701A' GB, '...:0811A' Sci)
    y = 0   human       generated_by == 'human'
    y = 0   rephrase    generated_by == 'gpt56luna:0908R'   (from the *_reph_* parquet)

A document's human / rephrase / full-generation rows all share one ``id``, so a
grouped split keeps the three together -- exactly the discipline
:mod:`._trained` applies to (human, rewrite) pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epai.ai_detection.trainv2._trained import AUX_COL, EMB_COL, _stack
from epai.ai_detection.trainv2.data import _is_missing
from epai.ai_detection.trainv2.plot_score_histograms import CORPUS_SOURCES, PARQUET_DIR

GEN_COL = "generated_by"
DOMAIN_COL = "domain"
GROUP_COL = "id"
HUMAN = "human"
REPHRASE_GEN = "gpt56luna:0908R"

#: the rephrase training corpora, in the fixed left-to-right order
REPH_SPLITS = list(CORPUS_SOURCES)  # ["mg_gb", "mg_sci"]


def load_reph_features(corpus_key: str):
    """``(X_emb, y, Z_cue, domain, groups)`` for one rephrase corpus.

    ``human`` / ``rewritten`` come from the full parquet, ``rephrase`` from the
    rephrase half; ``y`` counts rephrase as human. A group is dropped whole if
    any of its rows is missing either feature; a group must sit in one domain.
    """
    c = CORPUS_SOURCES[corpus_key]
    cols = [GROUP_COL, DOMAIN_COL, GEN_COL, EMB_COL, AUX_COL]
    full = pd.read_parquet(PARQUET_DIR / c["full"], columns=cols)
    reph = pd.read_parquet(PARQUET_DIR / c["reph"], columns=cols)
    picks = pd.concat(
        [
            full[full[GEN_COL] == HUMAN].assign(_cls="human"),
            reph[reph[GEN_COL] == REPHRASE_GEN].assign(_cls="rephrase"),
            full[full[GEN_COL] == c["rewrite_gen"]].assign(_cls="rewritten"),
        ],
        ignore_index=True,
    )

    bad = picks[EMB_COL].map(_is_missing) | picks[AUX_COL].map(_is_missing)
    if bad.any():
        picks = picks[~picks[GROUP_COL].isin(picks.loc[bad, GROUP_COL].unique())]
    picks = picks.reset_index(drop=True)

    span = picks.groupby(GROUP_COL)[DOMAIN_COL].nunique()
    if (span > 1).any():
        raise SystemExit(
            f"{corpus_key}: {int((span > 1).sum())} group(s) span more than one domain"
        )

    y = (picks["_cls"] == "rewritten").to_numpy().astype(int)
    return (
        _stack(picks[EMB_COL]),
        y,
        _stack(picks[AUX_COL]),
        picks[DOMAIN_COL].astype(str).to_numpy(),
        picks[GROUP_COL].to_numpy(),
    )

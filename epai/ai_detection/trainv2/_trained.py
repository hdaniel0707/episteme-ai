"""The trained-detector configs shared by ``run_in_domain_performance.py`` and
``run_lodo_performance.py``.

One plain linear probe on the RoBERTa embedding, plus the two RoBERTa + judge
cue-vector hybrids from the old ``run_trained_vs_zeroshot_performance.py``,
recovered unchanged: concatenation fusion (``fusion_mlp``) and a learned gate
on the embedding branch (``gated_mlp``). Both read the 11-dim
``llm_judge_cuevec_gpt56luna_v3`` column as the auxiliary input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from epai.algs import build_algorithm
from epai.ai_detection.trainv2.core import auroc, eff_fpr_level, tpr_at_fpr
from epai.ai_detection.trainv2.data import GEN_COL, _is_missing

EMB_COL = "emb_roberta"
AUX_COL = "llm_judge_cuevec_gpt56luna_v3"

#: key -> spec. ``label`` is the detector name on the plot / CSV (after
#: ``trained/``); ``aux`` = the algorithm also takes the cue vector.
TRAINED_SPECS: dict[str, dict] = {
    "roberta_logistic": {
        "label": "roberta·logistic", "alg": "logistic", "aux": False,
        "params": {"C": 1.0, "max_iter": 1000},
    },
    "roberta_fusion_cuevec": {
        "label": "roberta⊕cuevec", "alg": "fusion_mlp", "aux": True,
        "params": {"pre_dims": [256, 16], "post_dims": [32], "aux_dims": [32]},
    },
    "roberta_gated_cuevec": {
        "label": "roberta+gated", "alg": "gated_mlp", "aux": True,
        "params": {"emb_dims": [256, 64], "aux_dims": [32], "gate": "sigmoid"},
    },
}


def _stack(s: pd.Series) -> np.ndarray:
    return np.asarray(np.stack(s.to_numpy()), dtype=np.float32)


def load_features(path: Path, group_col: str, human_label: str):
    """``(X_emb, y, Z_cue, groups)``. A group is dropped whole if ANY of its
    rows is missing EITHER feature, so every spec (aux or not) trains and
    scores on the identical rows."""
    df = pd.read_parquet(path, columns=[group_col, GEN_COL, EMB_COL, AUX_COL])
    bad = df[EMB_COL].map(_is_missing) | df[AUX_COL].map(_is_missing)
    if bad.any():
        df = df[~df[group_col].isin(df.loc[bad, group_col].unique())]
    df = df.reset_index(drop=True)
    y = (df[GEN_COL].to_numpy() != human_label).astype(int)
    return _stack(df[EMB_COL]), y, _stack(df[AUX_COL]), df[group_col].to_numpy()


def fit(spec: dict, X, y, Z, groups, idx, seed: int):
    """Fit ``spec`` on the rows selected by ``idx`` (int array or bool mask)."""
    clf = build_algorithm(spec["alg"], seed, **spec["params"])
    clf.set_fit_groups(groups[idx])
    if spec["aux"]:
        clf.fit(X[idx], y[idx], Z[idx])
    else:
        clf.fit(X[idx], y[idx])
    return clf


def eval_on(clf, spec: dict, X, y, Z,
            fpr: float) -> tuple[float, float, float]:
    """``(auroc, tpr@fpr, tpr@eff_fpr)`` on the rows passed in (already the eval
    subset). ``decision_score`` is already oriented high = machine. The third
    value reads the operating point at the smallest FPR the ROC grid can hit
    (``eff_fpr_level``) -- a real number even on a split too small for 1% FPR."""
    s = clf.decision_score(X, Z) if spec["aux"] else clf.decision_score(X)
    n_h = int((np.asarray(y) == 0).sum())
    return (float(auroc(y, s)), float(tpr_at_fpr(y, s, fpr)),
            float(tpr_at_fpr(y, s, eff_fpr_level(n_h, fpr))))

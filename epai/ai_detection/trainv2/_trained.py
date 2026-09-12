"""The trained-detector configs shared by ``run_in_domain_performance.py`` and
``run_lodo_performance.py``.

One plain linear probe on the RoBERTa embedding. The two RoBERTa + judge
cue-vector hybrids that used to live here (concatenation fusion via
``fusion_mlp`` and a learned gate via ``gated_mlp``) were removed when the
multi-embedding architectures below took over the "beyond one linear probe"
role -- see ``embedding_ideas.md``. ``AUX_COL`` / the aux-column machinery
stays (``_reph_trained.py`` still uses it for the rephrase benchmark), but no
entry in ``TRAINED_SPECS`` sets ``aux: True`` any more.

``MULTI_EMB_COLS`` / ``MULTI_EMB_SPECS`` / ``load_multi_features`` /
``load_multi_ood`` / ``with_emb_dims`` below are the multi-embedding
counterpart: four architectures (score-level ensemble, early fusion, an
N-branch gate, attention fusion -- one new :mod:`epai.algs` module each) that
all read the SAME concatenation of a diversity-picked embedding subset instead
of ``EMB_COL`` alone. They are deliberately a separate dict from
``TRAINED_SPECS``: ``run_dr_transfer.py`` and ``run_robust_performance.py``
also import ``TRAINED_SPECS`` directly for their own (unrelated) comparisons,
and wiring the new architectures into every one of those would be a much
bigger change than "add them to the in-domain/LODO runners" -- only
``run_in_domain_performance.py`` / ``run_lodo_performance.py`` import
``MULTI_EMB_SPECS``.
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
        "embeddings": (EMB_COL,),
        "params": {"C": 1.0, "max_iter": 1000},
    },
}

# ---------------------------------------------------------------------------
# Multi-embedding architectures (embedding_ideas.md, Stage 1)
# ---------------------------------------------------------------------------

#: Diversity-picked embeddings for the multi-embedding architectures: one
#: general-purpose LM embedding (also ``EMB_COL``, the single-embedding
#: baseline above), one style embedding (``styledistance`` scored highest
#: in-domain among the style family in
#: ``results/trainv2/embedding_effectiveness/summary.csv`` and is the family
#: hypothesized in embedding_ideas.md to transfer better across domain
#: shift), one scientific-domain embedding (``specter2`` -- relevant given
#: MG-Sci / MG-AI-Sci is scientific text). ``EMB_COL`` first is relied on by
#: the two runner scripts: it means the single-embedding baseline's column is
#: a prefix of this tuple, so loading ``MULTI_EMB_COLS`` once per split can
#: also serve ``roberta_logistic`` on identical rows (see the loaders below).
MULTI_EMB_COLS: tuple[str, ...] = ("emb_roberta", "emb_styledistance", "emb_specter2")
assert MULTI_EMB_COLS[0] == EMB_COL

#: key -> spec, same shape as ``TRAINED_SPECS``. Every entry reads the full
#: ``MULTI_EMB_COLS`` concatenation; ``emb_dims`` (the block widths, in that
#: order) is NOT hand-written here -- :func:`with_emb_dims` injects it from
#: the actual loaded columns before ``build_algorithm``, since it is a
#: property of the data, not a config choice.
MULTI_EMB_SPECS: dict[str, dict] = {
    "emb_stack_avg": {
        "label": "roberta+style+sci·stack(avg)", "alg": "emb_stacking",
        "aux": False, "embeddings": MULTI_EMB_COLS,
        "params": {"meta": "average"},
    },
    "emb_stack_logistic": {
        "label": "roberta+style+sci·stack(LR)", "alg": "emb_stacking",
        "aux": False, "embeddings": MULTI_EMB_COLS,
        "params": {"meta": "logistic", "cv": 5},
    },
    "emb_early_fusion": {
        "label": "roberta+style+sci·fusion(PCA)", "alg": "emb_early_fusion",
        "aux": False, "embeddings": MULTI_EMB_COLS,
        "params": {"pca_dim": 64, "emb_dropout": 0.1, "hidden_dims": []},
    },
    "emb_gated_n": {
        "label": "roberta+style+sci·gated(N)", "alg": "emb_gated_n",
        "aux": False, "embeddings": MULTI_EMB_COLS,
        "params": {"branch_dims": [256, 64], "gate": "sigmoid"},
    },
    "emb_attn_fusion": {
        "label": "roberta+style+sci·attention", "alg": "emb_attn_fusion",
        "aux": False, "embeddings": MULTI_EMB_COLS,
        "params": {"d_model": 64, "n_heads": 4, "n_layers": 1, "ff_dim": 128},
    },
}


def with_emb_dims(spec: dict, dims: dict[str, int],
                  embeddings: tuple[str, ...] = MULTI_EMB_COLS) -> dict:
    """``spec`` with ``emb_dims`` (block widths, in ``embeddings`` order)
    merged into its params. The width of a given embedding column is a
    property of the data (which model produced it), not a config choice, so
    it is computed from whatever :func:`load_multi_features` /
    :func:`load_multi_ood` actually loaded rather than hard-coded into
    ``MULTI_EMB_SPECS``."""
    return {**spec, "params": {**spec["params"], "emb_dims": [dims[c] for c in embeddings]}}


def _missing_mask(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    """Rows missing ANY of ``cols``. Shared by the two loaders below so a group
    is dropped whole if it lacks any embedding any multi-embedding spec needs
    -- the multi-embedding analogue of ``load_features``'s single-column
    missingness check."""
    bad = pd.Series(False, index=df.index)
    for c in cols:
        bad = bad | df[c].map(_is_missing)
    return bad


def load_multi_features(
    path: Path, group_col: str, human_label: str,
    embeddings: tuple[str, ...] = MULTI_EMB_COLS, *,
    domain_col: str | None = None, score_cols: tuple[str, ...] = (),
):
    """``(X, y, groups[, domains], dims[, raw])`` -- the multi-embedding
    counterpart of :func:`load_features`. ``X`` concatenates ``embeddings``'s
    columns in the given order; ``dims`` is ``{column: width}`` for
    :func:`with_emb_dims`. A group is dropped whole if any of its rows is
    missing ANY of ``embeddings``, so every ``MULTI_EMB_SPECS`` entry (which
    all read the same ``embeddings``) trains and scores on identical rows.

    With ``domain_col`` (for ``run_lodo_performance.py``), also returns the
    domain per surviving row and raises if any group spans more than one
    domain -- the same LODO invariant ``run_lodo_performance._load`` checks.
    """
    extra = [domain_col] if domain_col else []
    cols = list(dict.fromkeys([group_col, GEN_COL, *extra, *embeddings, *score_cols]))
    df = pd.read_parquet(path, columns=cols)
    bad = _missing_mask(df, embeddings)
    if bad.any():
        df = df[~df[group_col].isin(df.loc[bad, group_col].unique())]
    df = df.reset_index(drop=True)
    if domain_col:
        span = df.groupby(group_col)[domain_col].nunique()
        if (span > 1).any():
            raise SystemExit(
                f"{path.name}: {int((span > 1).sum())} group(s) span more "
                f"than one domain — LODO would split them"
            )
    y = (df[GEN_COL].to_numpy() != human_label).astype(int)
    blocks = [_stack(df[e]) for e in embeddings]
    X = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)
    dims = {e: b.shape[1] for e, b in zip(embeddings, blocks)}
    out = [X, y, df[group_col].to_numpy()]
    if domain_col:
        out.append(df[domain_col].astype(str).to_numpy())
    out.append(dims)
    if score_cols:
        out.append({c: df[c].to_numpy(dtype=float) for c in score_cols})
    return tuple(out)


def load_multi_ood(
    path: Path, human_label: str, embeddings: tuple[str, ...] = MULTI_EMB_COLS, *,
    score_cols: tuple[str, ...] = (),
):
    """``(X, y, n_human, dims[, raw])`` -- full-corpus eval set (never trained
    on), the multi-embedding counterpart of ``run_lodo_performance._load_ood``
    / the OOD branch of ``compute_trained``."""
    cols = list(dict.fromkeys([GEN_COL, *embeddings, *score_cols]))
    df = pd.read_parquet(path, columns=cols)
    bad = _missing_mask(df, embeddings)
    df = df[~bad].reset_index(drop=True)
    y = (df[GEN_COL].to_numpy() != human_label).astype(int)
    blocks = [_stack(df[e]) for e in embeddings]
    X = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)
    dims = {e: b.shape[1] for e, b in zip(embeddings, blocks)}
    out = [X, y, int((y == 0).sum()), dims]
    if score_cols:
        out.append({c: df[c].to_numpy(dtype=float) for c in score_cols})
    return tuple(out)


def _stack(s: pd.Series) -> np.ndarray:
    return np.asarray(np.stack(s.to_numpy()), dtype=np.float32)


def load_features(path: Path, group_col: str, human_label: str,
                  *, score_cols: tuple[str, ...] = ()):
    """``(X_emb, y, Z_cue, groups)``. A group is dropped whole if ANY of its
    rows is missing EITHER feature, so every spec (aux or not) trains and
    scores on the identical rows.

    With ``score_cols`` (raw zero-shot columns, all assumed present in the
    parquet), a 5th element is returned: ``{col: float_array}`` aligned to the
    same surviving rows -- so a zero-shot pick can be scored on the trained
    models' exact eval partition. NaNs are left in place."""
    cols = list(dict.fromkeys([group_col, GEN_COL, EMB_COL, AUX_COL, *score_cols]))
    df = pd.read_parquet(path, columns=cols)
    bad = df[EMB_COL].map(_is_missing) | df[AUX_COL].map(_is_missing)
    if bad.any():
        df = df[~df[group_col].isin(df.loc[bad, group_col].unique())]
    df = df.reset_index(drop=True)
    y = (df[GEN_COL].to_numpy() != human_label).astype(int)
    feats = (_stack(df[EMB_COL]), y, _stack(df[AUX_COL]), df[group_col].to_numpy())
    if score_cols:
        return (*feats, {c: df[c].to_numpy(dtype=float) for c in score_cols})
    return feats


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
    au, per = eval_on_fprs(clf, spec, X, y, Z, (fpr,))
    return (au, *per[fpr])


def eval_on_fprs(clf, spec: dict, X, y, Z,
                 fprs) -> tuple[float, dict]:
    """``(auroc, {fpr: (tpr@fpr, tpr@eff_fpr)})`` -- the ``eval_on`` reading at
    several FPR levels at once, ``decision_score`` computed a single time.
    AUROC is threshold-free, so it is shared across the levels."""
    s = clf.decision_score(X, Z) if spec["aux"] else clf.decision_score(X)
    n_h = int((np.asarray(y) == 0).sum())
    per = {f: (float(tpr_at_fpr(y, s, f)),
               float(tpr_at_fpr(y, s, eff_fpr_level(n_h, f)))) for f in fprs}
    return float(auroc(y, s)), per

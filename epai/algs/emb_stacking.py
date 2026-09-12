"""Score-level ensemble over several embeddings: one logistic probe per
embedding, combined post-hoc.

Architecture #1 of ``epai/ai_detection/trainv2/embedding_ideas.md``: instead
of concatenating several embeddings into one design matrix (see
:mod:`.emb_early_fusion`), fit one small, independent probe per embedding and
combine their *scores*. Far lower effective capacity than a single probe over
the concatenation -- the same reason a random-forest-style ensemble overfits
less than one big tree -- which is the point, given each embedding adds
hundreds of dimensions against training splits of a few thousand rows.

Two combination modes (``meta``):

``average``
    Unweighted mean of the per-embedding probes' machine-probability, each
    probe fit once on the full training split. The "naive ensembling" control:
    no second-stage fitting, so it cannot itself overfit.

``logistic``
    A logistic-regression meta-model over the per-embedding probes' scores
    (Wolpert, "Stacked Generalization", *Neural Networks* 5(2), 1992). Fit on
    OUT-OF-FOLD predictions, never on the same rows the base probes were fit
    on: a base probe here is capable of ~99% in-domain AUROC (see
    ``results/trainv2/embedding_effectiveness/summary.csv``), so an in-sample
    meta-fit would just learn "trust whichever probe overfit hardest" rather
    than which embedding actually generalizes. The OOF split is
    ``GroupKFold`` -- a document and its rewrites share a group id and must
    never straddle a fold, exactly the discipline
    :func:`epai.ai_detection.trainv2.core.grouped_split` applies to the outer
    train/test cut. (This is why this class hand-rolls the OOF loop instead of
    handing embeddings to :class:`sklearn.ensemble.StackingClassifier`: its
    public ``fit(X, y)`` has no way to pass ``groups`` through to the internal
    ``cross_val_predict``.) The final per-embedding probes used at predict
    time are refit on the FULL training split, same as ``average`` -- only the
    meta-model's training data is held out.

Config (``spec["params"]``)::

    {"emb_dims": [768, 1024, 768], "meta": "logistic", "cv": 5,
     "base_C": 1.0, "base_max_iter": 1000}

``emb_dims`` is not meant to be hand-written: the trainv2 loader computes it
from the actual concatenated embedding columns and injects it before
``build_algorithm`` (see ``epai/ai_detection/trainv2/_trained.py``).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import ClfAlgorithm

META_MODES = ("average", "logistic")


def _blocks(emb_dims: list[int]) -> list[tuple[int, int]]:
    """Cumulative ``(start, end)`` column ranges for each embedding block."""
    starts = np.cumsum([0, *emb_dims[:-1]])
    return [(int(s), int(s + d)) for s, d in zip(starts, emb_dims)]


def _base_probe(C: float, max_iter: int, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=max_iter, random_state=seed),
    )


class EmbStackingAlg(ClfAlgorithm):
    """Per-embedding logistic probes, combined by averaging or a stacked meta-LR.

    Architecture params:
      * ``emb_dims`` -- column widths of each embedding block, in the order they
        were concatenated into ``X`` (required; see the module docstring);
      * ``meta`` -- ``"average"`` or ``"logistic"`` (default ``"logistic"``);
      * ``cv`` -- number of ``GroupKFold`` folds for the OOF meta-training data,
        only read when ``meta == "logistic"`` (default 5);
      * ``base_C`` / ``base_max_iter`` -- hyper-parameters of each per-embedding
        probe (defaults 1.0 / 1000, matching :class:`epai.algs.logistic.LogisticAlg`);
      * ``meta_C`` / ``meta_max_iter`` -- hyper-parameters of the meta-LR
        (defaults 1.0 / 1000), only read when ``meta == "logistic"``.
    """

    name = "emb_stacking"
    PARAMS = (
        "emb_dims", "meta", "cv", "base_C", "base_max_iter", "meta_C", "meta_max_iter",
    )

    def _emb_dims(self) -> list[int]:
        dims = self.params.get("emb_dims")
        if not dims:
            raise ValueError(f"{self.name}: 'emb_dims' is required (block widths)")
        return [int(d) for d in dims]

    def _meta(self) -> str:
        meta = self.params.get("meta", "logistic")
        if meta not in META_MODES:
            raise ValueError(f"meta must be one of {META_MODES}, got {meta!r}")
        return meta

    def validate_params(self) -> None:
        self._emb_dims()
        self._meta()

    def resolved_params(self) -> dict:
        return {
            "emb_dims": self._emb_dims(), "meta": self._meta(),
            "cv": int(self.params.get("cv", 5)),
            "base_C": float(self.params.get("base_C", 1.0)),
            "base_max_iter": int(self.params.get("base_max_iter", 1000)),
            "meta_C": float(self.params.get("meta_C", 1.0)),
            "meta_max_iter": int(self.params.get("meta_max_iter", 1000)),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EmbStackingAlg":
        p = self.resolved_params()
        blocks = _blocks(p["emb_dims"])
        if sum(p["emb_dims"]) != X.shape[1]:
            raise ValueError(
                f"{self.name}: emb_dims {p['emb_dims']} sums to "
                f"{sum(p['emb_dims'])}, X has {X.shape[1]} columns"
            )

        # Final per-embedding probes, fit on the FULL training split -- used at
        # predict time regardless of `meta`.
        self.probes_ = [
            _base_probe(p["base_C"], p["base_max_iter"], self.seed).fit(
                X[:, s:e], y
            )
            for s, e in blocks
        ]

        if p["meta"] == "average":
            self.meta_ = None
            return self

        groups = self.fit_groups_
        if groups is None:
            raise ValueError(
                f"{self.name} meta='logistic' needs group ids; call "
                f"set_fit_groups before fit"
            )
        n_splits = min(p["cv"], len(set(groups)))
        if n_splits < 2:
            # Too few groups for an OOF split (a tiny fold, e.g. a single LODO
            # domain) -- fall back to average rather than a meta-fit that
            # cannot be honestly cross-validated.
            self.meta_ = None
            return self

        oof = np.zeros((len(y), len(blocks)), dtype=np.float64)
        gkf = GroupKFold(n_splits=n_splits)
        for tr, va in gkf.split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue  # degenerate fold; leave this fold's rows at 0
            for j, (s, e) in enumerate(blocks):
                probe = _base_probe(p["base_C"], p["base_max_iter"], self.seed)
                probe.fit(X[tr, s:e], y[tr])
                oof[va, j] = probe.predict_proba(X[va, s:e])[:, 1]

        self.meta_ = LogisticRegression(
            C=p["meta_C"], max_iter=p["meta_max_iter"], random_state=self.seed
        ).fit(oof, y)
        return self

    def _probe_proba(self, X: np.ndarray) -> np.ndarray:
        dims = self._emb_dims()
        blocks = _blocks(dims)
        return np.column_stack(
            [probe.predict_proba(X[:, s:e])[:, 1]
             for probe, (s, e) in zip(self.probes_, blocks)]
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_score(X) >= 0.5).astype(int)

    def decision_score(self, X: np.ndarray) -> np.ndarray:
        proba = self._probe_proba(X)
        if self.meta_ is None:
            return proba.mean(axis=1)
        return self.meta_.predict_proba(proba)[:, 1]

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

"""Distilling the ``gpt56luna`` judge's continuous score into an ``emb_roberta``
linear probe, instead of (or alongside) the ground-truth label -- a pilot for
whether a soft judge target generalizes to MG-AI-Sci better than GT does. See
``epai/ai_detection/trainv2/run_judge_distillation_performance.py`` for the
experiment this feeds and the numbers it is measured against
(``trained/roberta·logistic`` and the zero-shot ``llm_judge_score_gpt56luna_v3_wlp``
point, both already in ``in_domain_performance.csv``).

Two arms, both taking the judge's continuous P(human) score as the auxiliary
column (``uses_aux = True``) rather than the usual cue vector:

* :class:`JudgeDistillRegressAlg` -- pure distillation. Ignores the GT label
  entirely; a Ridge regression on standardised ``emb_roberta`` fit to predict
  the judge score. ``decision_score`` negates the prediction (the judge score
  is P(human), so higher predicted P(human) should rank as LESS machine-like).
* :class:`JudgeDistillHybridAlg` -- hybrid. A linear probe (single
  ``Linear(in_dim, 2)``, no hidden layer, matching ``roberta·logistic``'s
  capacity) trained on cross-entropy against the GT label PLUS a weighted MSE
  term pulling the model's own machine-probability toward ``1 - judge_score``.
  Its own small training loop (not :func:`epai.algs._torch_common.train_torch_model`,
  which hard-codes ``CrossEntropyLoss`` with no second loss term), but reuses
  :func:`epai.algs._torch_common.split_train_val` for a real held-out split
  (grouped on ``fit_groups_``), early-stopped on validation AUROC (the GT
  label, since that's what every downstream metric reads) -- ``val_frac`` /
  ``patience``, same semantics as ``robust_mlp``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ._torch_common import FeatureNormalizer, logit_margin, split_train_val
from .base import ClfAlgorithm


class JudgeDistillRegressAlg(ClfAlgorithm):
    """Pure distillation: regress standardised ``emb_roberta`` onto the
    auxiliary column (assumed a continuous P(human) judge score), ignoring
    the GT label ``y`` given to ``fit`` entirely. Params: ``alpha`` (Ridge L2
    strength, default 1.0)."""

    name = "judge_distill_regress"
    uses_aux = True
    PARAMS = ("alpha",)

    def resolved_params(self) -> dict:
        return {"alpha": float(self.params.get("alpha", 1.0))}

    def fit(self, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None = None) -> "JudgeDistillRegressAlg":
        if Z is None:
            raise ValueError(f"{self.name} needs the judge score as its auxiliary column")
        target = np.asarray(Z, dtype=np.float64).reshape(-1)
        self.model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(self.params.get("alpha", 1.0)), random_state=self.seed),
        )
        self.model.fit(X, target)
        return self

    def predict(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        return (self.decision_score(X) > 0).astype(int)

    def decision_score(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        """Negated predicted P(human) -- higher = more machine-like, matching
        every other algorithm's orientation."""
        return -np.asarray(self.model.predict(X), dtype=np.float64)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)


class _HybridLinear(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.head = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class JudgeDistillHybridAlg(ClfAlgorithm):
    """Hybrid distillation: a linear probe trained on
    ``CE(logits, y) + lambda_distill * MSE(P(machine), 1 - judge_score)``.

    Params: ``lambda_distill`` (weight of the soft-target term, default 0.5),
    ``epochs`` (max epochs, default 100), ``batch_size`` (default 64), ``lr``
    (default 1e-3), ``weight_decay`` (default 1e-4), ``normalize`` (default
    ``"standard"`` -- see :class:`epai.algs._torch_common.FeatureNormalizer`),
    ``class_weight`` (``"none"`` default or ``"balanced"`` -- sklearn-style
    inverse-frequency weights on the CE(y) term only; the MSE term against the
    judge score is unaffected, since it isn't a class label), ``val_frac``
    (default 0.15, grouped on ``fit_groups_``) and ``patience`` (default 10)
    -- early stopping on validation AUROC (see :func:`epai.algs._torch_common.split_train_val`);
    ``val_frac: 0`` disables it and trains the full ``epochs`` like before.
    """

    name = "judge_distill_hybrid"
    uses_aux = True
    N_CLASSES = 2
    PARAMS = ("lambda_distill", "epochs", "batch_size", "lr", "weight_decay",
             "normalize", "class_weight", "val_frac", "patience")

    def resolved_params(self) -> dict:
        p = self.params
        return {
            "lambda_distill": float(p.get("lambda_distill", 0.5)),
            "epochs": int(p.get("epochs", 100)),
            "batch_size": int(p.get("batch_size", 64)),
            "lr": float(p.get("lr", 1e-3)),
            "weight_decay": float(p.get("weight_decay", 1e-4)),
            "normalize": p.get("normalize", "standard"),
            "class_weight": p.get("class_weight", "none"),
            "val_frac": float(p.get("val_frac", 0.15)),
            "patience": int(p.get("patience", 10)),
        }

    def fit(self, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None = None) -> "JudgeDistillHybridAlg":
        if Z is None:
            raise ValueError(f"{self.name} needs the judge score as its auxiliary column")
        rp = self.resolved_params()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.normalizer = FeatureNormalizer(rp["normalize"])
        Xs = self.normalizer.fit_transform(X)

        y_arr = np.asarray(y, dtype=np.int64)
        n = len(y_arr)
        train_idx, val_idx, _fallback = split_train_val(
            n, rp["val_frac"], self.seed, self.fit_groups_, y_arr)
        use_val = len(val_idx) > 0

        self.model = _HybridLinear(Xs.shape[1], self.N_CLASSES).to(self.device)
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=rp["lr"], weight_decay=rp["weight_decay"]
        )

        X_t = torch.from_numpy(np.ascontiguousarray(Xs)).to(self.device)
        y_t = torch.from_numpy(y_arr).to(self.device)
        # judge score is P(human); the soft target for the model's own
        # P(machine) is its complement.
        p_machine_target = torch.from_numpy(
            (1.0 - np.asarray(Z, dtype=np.float64).reshape(-1)).astype(np.float32)
        ).to(self.device)

        tr = torch.from_numpy(train_idx).to(self.device)
        X_train, y_train, z_train = X_t[tr], y_t[tr], p_machine_target[tr]
        if use_val:
            va = torch.from_numpy(val_idx).to(self.device)
            X_val, y_val = X_t[va], y_t[va]

        loader = DataLoader(
            TensorDataset(X_train, y_train, z_train),
            batch_size=rp["batch_size"], shuffle=True,
        )
        lam = rp["lambda_distill"]

        class_weight = None
        if rp["class_weight"] == "balanced":
            # sklearn's "balanced" convention: n_samples / (n_classes * count),
            # computed from the TRAINING rows only.
            counts = torch.bincount(y_train, minlength=2).float().clamp(min=1.0)
            class_weight = (y_train.numel() / (2.0 * counts)).to(self.device)

        best_score, best_state, no_improve = -float("inf"), None, 0
        for _epoch in range(rp["epochs"]):
            self.model.train()
            for xb, yb, zb in loader:
                opt.zero_grad()
                logits = self.model(xb)
                p_machine = F.softmax(logits, dim=1)[:, 1]
                loss = (F.cross_entropy(logits, yb, weight=class_weight)
                       + lam * F.mse_loss(p_machine, zb))
                loss.backward()
                opt.step()
            if use_val:
                self.model.eval()
                with torch.no_grad():
                    score = float(roc_auc_score(
                        y_val.cpu().numpy(), logit_margin(self.model(X_val))))
                if score > best_score + 1e-5:
                    best_score, best_state, no_improve = score, copy.deepcopy(
                        self.model.state_dict()), 0
                else:
                    no_improve += 1
                    if rp["patience"] and no_improve >= rp["patience"]:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _logits(self, X: np.ndarray) -> torch.Tensor:
        Xs = self.normalizer.transform(X)
        X_t = torch.from_numpy(Xs).to(self.device)
        self.model.eval()
        return self.model(X_t)

    def predict(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        return self._logits(X).argmax(dim=1).cpu().numpy()

    def decision_score(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        """The machine-minus-human logit margin. See :func:`logit_margin`."""
        return logit_margin(self._logits(X))

    def save(self, path: Path) -> None:
        self.model.to("cpu")
        self.device = torch.device("cpu")
        joblib.dump(self, path)

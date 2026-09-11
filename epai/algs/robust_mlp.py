"""A single-branch ``emb_roberta`` classifier exposing a tail-focused loss
(TopPush) and a domain-invariance penalty (CORAL / V-REx) on top of the
shared torch training harness in :mod:`epai.algs._torch_common`.

These are the "improving SOTA" ablation arms from
``epai/ai_detection/trainv2/improving_sota_ideas.md`` (#1 tail-aware
objective, #2 domain-invariance regularizer). Architecture-identical to
:class:`epai.algs.simple_mlp.SimpleMLPAlg` -- an empty ``hidden_dims`` is the
same linear probe -- so a run can isolate the loss/regularizer's effect from
the architecture's. The new hyper-parameters (``loss_kind``, ``push_frac``,
``push_weight``, ``push_margin``, ``domain_penalty``,
``domain_penalty_weight``) live on the shared ``TrainHParams`` and default to
plain cross-entropy with no domain penalty, so every other algorithm in this
package is unaffected.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from ._torch_common import (
    TRAIN_HPARAM_NAMES,
    FeatureNormalizer,
    TrainHParams,
    hparams_dict,
    logit_margin,
    mlp_layers,
    parse_dims,
    read_hparams,
    train_torch_model,
)
from .base import ClfAlgorithm


class _RobustMLP(nn.Module):
    """``hidden(x) -> features``, ``head(features) -> logits``. An empty
    ``hidden_dims`` makes ``hidden`` the identity, so ``features`` is (the
    optionally input-dropped) ``x`` itself -- CORAL then aligns the raw
    embedding's covariance across domains, which is the simplest faithful
    reading of the method when the head is a linear probe."""

    def __init__(
        self, in_dim: int, hidden_dims: list[int], n_classes: int, hp: TrainHParams
    ):
        super().__init__()
        layers, prev = mlp_layers(in_dim, hidden_dims, hp)
        self.hidden = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.hidden(x)
        logits = self.head(feats)
        return (logits, feats) if return_features else logits


class RobustMLPAlg(ClfAlgorithm):
    """See module docstring. ``domains`` (set via
    :meth:`~epai.algs.base.ClfAlgorithm.set_fit_domains`) are only read when
    ``domain_penalty != "none"``.

    Architecture params:
      * ``hidden_dims`` -- layer widths (default ``[]``: a linear probe, the
        arm every ablation in ``improving_sota_ideas.md`` uses).

    Every other hyper-parameter is documented on
    :class:`epai.algs._torch_common.TrainHParams`.
    """

    name = "robust_mlp"
    N_CLASSES = 2
    PARAMS = ("hidden_dims",) + TRAIN_HPARAM_NAMES

    def _hidden_dims(self) -> list[int]:
        return parse_dims(self.params.get("hidden_dims", []), "hidden_dims")

    def validate_params(self) -> None:
        read_hparams(self.params)  # ranges, enums, and the mixup/push/domain guard
        self._hidden_dims()

    def resolved_params(self) -> dict:
        return {"hidden_dims": self._hidden_dims(), **hparams_dict(read_hparams(self.params))}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RobustMLPAlg":
        hp = read_hparams(self.params)
        hidden_dims = self._hidden_dims()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalizer = FeatureNormalizer(hp.normalize)
        Xs = self.normalizer.fit_transform(X)

        self.model = _RobustMLP(Xs.shape[1], hidden_dims, self.N_CLASSES, hp).to(self.device)
        self.history_ = train_torch_model(
            self.model,
            [Xs],
            y,
            hp,
            self.device,
            self.seed,
            groups=self.fit_groups_,
            domains=self.fit_domains_,
        )
        return self

    @torch.no_grad()
    def _logits(self, X: np.ndarray) -> torch.Tensor:
        Xs = self.normalizer.transform(X)
        X_t = torch.from_numpy(Xs).to(self.device)
        self.model.eval()
        return self.model(X_t)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._logits(X).argmax(dim=1).cpu().numpy()

    def decision_score(self, X: np.ndarray) -> np.ndarray:
        """The machine-minus-human logit margin. See :func:`logit_margin`."""
        return logit_margin(self._logits(X))

    def save(self, path: Path) -> None:
        self.model.to("cpu")
        self.device = torch.device("cpu")
        joblib.dump(self, path)

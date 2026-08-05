"""A small feed-forward classifier trained from scratch in PyTorch."""

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
    mlp_layers,
    parse_dims,
    read_hparams,
    train_torch_model,
)
from .base import ClfAlgorithm


class _MLP(nn.Module):
    """Plain MLP: ``[input dropout]`` then a hidden stack, then a logit head.

    An empty ``hidden_dims`` collapses this to a linear probe on the features --
    kept reachable on purpose; see :func:`mlp_layers`.
    """

    def __init__(
        self, in_dim: int, hidden_dims: list[int], n_classes: int, hp: TrainHParams
    ):
        super().__init__()
        layers, prev = mlp_layers(in_dim, hidden_dims, hp)
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleMLPAlg(ClfAlgorithm):
    """A custom PyTorch MLP: normalise, then train an MLP under cross-entropy.

    Runs on CUDA when available, otherwise CPU. Architecture params:
      * ``hidden_dims`` -- layer widths (default ``[256, 64]``; ``[]`` gives a
        linear probe).

    Every other hyper-parameter is shared with :mod:`binox_mlp` and documented on
    :class:`epai.algs._torch_common.TrainHParams`. In brief:
      * ``normalize`` -- ``none`` | ``standard`` | ``l2``, combined with ``+``
        (default ``standard``). ``standard`` carries the *source* corpus's mean
        and scale onto every target; ``l2`` is per-row and carries nothing;
      * ``val_frac`` -- hold this fraction of the training rows out (grouped) and
        early-stop on it, restoring the best weights (default 0, which instead
        early-stops on the training loss -- the old behaviour);
      * ``val_metric`` -- ``loss`` or ``accuracy`` (default ``loss``);
      * ``dropout`` / ``input_dropout`` -- between hidden layers / on the raw
        feature vector (defaults 0.1 / 0.0);
      * ``norm_layer`` -- ``none`` or ``layernorm`` (default ``none``);
      * ``label_smoothing``, ``mixup_alpha``, ``grad_clip`` -- default 0, off;
      * ``epochs``, ``batch_size``, ``lr``, ``weight_decay``, ``patience`` --
        defaults 100, 64, 1e-3, 1e-4, 10.

    The defaults reproduce this class's behaviour before the shared trainer was
    introduced, so existing configs are unaffected.

    ``PARAMS`` is ``hidden_dims`` plus every field of ``TrainHParams``, which is
    exactly the space ``sweep_clf_cross_domain.py`` searches -- so a winning trial's
    params paste straight into a ``clf_cd_algs`` config entry, and a name that does
    not belong there is a preflight error rather than a silently ignored key.
    """

    name = "simple_mlp"
    N_CLASSES = 2
    PARAMS = ("hidden_dims",) + TRAIN_HPARAM_NAMES

    def _hidden_dims(self) -> list[int]:
        """The layer widths, parsed and validated. ``[]`` is a linear probe."""
        return parse_dims(self.params.get("hidden_dims", [256, 64]), "hidden_dims")

    def validate_params(self) -> None:
        read_hparams(self.params)  # ranges and enums
        self._hidden_dims()

    def resolved_params(self) -> dict:
        return {"hidden_dims": self._hidden_dims(), **hparams_dict(read_hparams(self.params))}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleMLPAlg":
        hp = read_hparams(self.params)
        hidden_dims = self._hidden_dims()

        # Seed torch and numpy so a run is reproducible for a given seed.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalizer = FeatureNormalizer(hp.normalize)
        Xs = self.normalizer.fit_transform(X)

        self.model = _MLP(Xs.shape[1], hidden_dims, self.N_CLASSES, hp).to(self.device)
        self.history_ = train_torch_model(
            self.model,
            [Xs],
            y,
            hp,
            self.device,
            self.seed,
            groups=self.fit_groups_,
        )
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.normalizer.transform(X)
        X_t = torch.from_numpy(Xs).to(self.device)
        self.model.eval()
        logits = self.model(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    def save(self, path: Path) -> None:
        # Move to CPU first so the pickle can be loaded on a machine without a GPU.
        self.model.to("cpu")
        self.device = torch.device("cpu")
        joblib.dump(self, path)

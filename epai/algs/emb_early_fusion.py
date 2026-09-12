"""Early fusion: concatenate several embeddings (PCA-reduced) into one probe,
with modality (embedding) dropout as the overfitting control.

Architecture #2 of ``epai/ai_detection/trainv2/embedding_ideas.md``. Naively
concatenating N embeddings can reach thousands of dimensions against training
splits of a few thousand rows -- exactly the overfitting risk the notes flag.
Two mitigations, both fit on the training split only and reused unchanged on
every target corpus, like every other normalizer in this package:

``pca_dim``
    Each embedding block is standardised and PCA-reduced to this width BEFORE
    concatenation, so every embedding contributes the same number of columns
    regardless of its native width (RoBERTa, SciBERT, LUAR, ... all differ).

``emb_dropout``
    Modality dropout (Neverova et al., "ModDrop", *TPAMI* 2016): during
    training, each embedding's whole (already PCA-reduced) block is zeroed for
    a given row with probability ``emb_dropout``, independently per row and
    per block. Forces the probe not to lean on any one embedding being
    present -- an overfitting control and a mild robustness regularizer at
    once. Off at eval time, like ordinary dropout.

What is left AFTER PCA + concatenation is an ordinary probe over the shared
training harness (:mod:`epai.algs._torch_common`) -- ``hidden_dims: []`` (the
default) makes it a linear probe, the honest "early fusion" control the notes
ask for; a non-empty ``hidden_dims`` is available for the same reason
:class:`epai.algs.simple_mlp.SimpleMLPAlg` is.

Config (``spec["params"]``)::

    {"emb_dims": [768, 1024, 768], "pca_dim": 64, "emb_dropout": 0.1,
     "hidden_dims": []}

``emb_dims`` -- the RAW, pre-PCA block widths -- is injected by the trainv2
loader, not hand-written; see ``epai/ai_detection/trainv2/_trained.py``.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


def _blocks(emb_dims: list[int]) -> list[tuple[int, int]]:
    starts = np.cumsum([0, *emb_dims[:-1]])
    return [(int(s), int(s + d)) for s, d in zip(starts, emb_dims)]


class _EmbeddingDropout(nn.Module):
    """Zeroes each row's each ``block_size``-wide chunk independently, w.p. ``p``.

    Training-mode only (identity in ``eval()``), like :class:`nn.Dropout`. Not
    rescaled by ``1 / (1 - p)``: unlike feature dropout, a *modality* going
    missing is meant to be a real, undamped signal the downstream probe has to
    tolerate, not an expectation-preserving perturbation.
    """

    def __init__(self, n_blocks: int, block_size: int, p: float):
        super().__init__()
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return x
        mask = (torch.rand(x.shape[0], self.n_blocks, device=x.device) >= self.p).float()
        mask = mask.repeat_interleave(self.block_size, dim=1)
        return x * mask


class _EarlyFusionNet(nn.Module):
    def __init__(
        self, n_blocks: int, block_size: int, emb_dropout: float,
        hidden_dims: list[int], n_classes: int, hp: TrainHParams,
    ):
        super().__init__()
        self.drop = _EmbeddingDropout(n_blocks, block_size, emb_dropout)
        layers, prev = mlp_layers(n_blocks * block_size, hidden_dims, hp)
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.drop(x))


class EmbEarlyFusionAlg(ClfAlgorithm):
    """Concatenated, PCA-reduced multi-embedding probe with embedding dropout.

    Architecture params:
      * ``emb_dims`` -- RAW column widths of each embedding block, in
        concatenation order (required);
      * ``pca_dim`` -- per-block PCA target width (default 64); clipped down to
        ``min(pca_dim, block_width, n_train_rows - 1)`` if a block or the
        training split is too small to support it;
      * ``emb_dropout`` -- probability of zeroing a whole (reduced) block for a
        given row during training (default 0.1);
      * ``hidden_dims`` -- probe widths after the dropout layer (default
        ``[]``: a linear probe over the concatenated, reduced blocks).

    Every other hyper-parameter is shared with :mod:`simple_mlp` and documented
    on :class:`epai.algs._torch_common.TrainHParams`.
    """

    name = "emb_early_fusion"
    N_CLASSES = 2
    PARAMS = ("emb_dims", "pca_dim", "emb_dropout", "hidden_dims") + TRAIN_HPARAM_NAMES

    def _emb_dims(self) -> list[int]:
        dims = self.params.get("emb_dims")
        if not dims:
            raise ValueError(f"{self.name}: 'emb_dims' is required (block widths)")
        return [int(d) for d in dims]

    def _pca_dim(self) -> int:
        v = int(self.params.get("pca_dim", 64))
        if v < 1:
            raise ValueError(f"pca_dim must be >= 1, got {v}")
        return v

    def _emb_dropout(self) -> float:
        v = float(self.params.get("emb_dropout", 0.1))
        if not 0.0 <= v < 1.0:
            raise ValueError(f"emb_dropout must be in [0, 1), got {v}")
        return v

    def _hidden_dims(self) -> list[int]:
        return parse_dims(self.params.get("hidden_dims", []), "hidden_dims")

    def validate_params(self) -> None:
        read_hparams(self.params)
        self._emb_dims()
        self._pca_dim()
        self._emb_dropout()
        self._hidden_dims()

    def resolved_params(self) -> dict:
        return {
            "emb_dims": self._emb_dims(), "pca_dim": self._pca_dim(),
            "emb_dropout": self._emb_dropout(), "hidden_dims": self._hidden_dims(),
            **hparams_dict(read_hparams(self.params)),
        }

    def _reduce_fit(self, X: np.ndarray) -> np.ndarray:
        """Fit per-block scaler+PCA on the training split; returns the reduced,
        concatenated matrix. Stores ``self.block_scalers_`` / ``self.block_pcas_``
        / ``self.pca_dim_`` (the actually-used, possibly clipped, width)."""
        dims = self._emb_dims()
        blocks = _blocks(dims)
        if sum(dims) != X.shape[1]:
            raise ValueError(
                f"{self.name}: emb_dims {dims} sums to {sum(dims)}, X has "
                f"{X.shape[1]} columns"
            )
        requested = self._pca_dim()
        # A PCA can extract at most min(n_samples, n_features) components.
        self.pca_dim_ = max(1, min(requested, X.shape[0] - 1, *dims))
        self.block_scalers_: list[StandardScaler] = []
        self.block_pcas_: list[PCA] = []
        reduced = []
        for s, e in blocks:
            scaler = StandardScaler().fit(X[:, s:e])
            pca = PCA(n_components=self.pca_dim_, random_state=self.seed)
            reduced.append(pca.fit_transform(scaler.transform(X[:, s:e])))
            self.block_scalers_.append(scaler)
            self.block_pcas_.append(pca)
        return np.concatenate(reduced, axis=1).astype(np.float32)

    def _reduce_transform(self, X: np.ndarray) -> np.ndarray:
        dims = self._emb_dims()
        blocks = _blocks(dims)
        reduced = [
            pca.transform(scaler.transform(X[:, s:e]))
            for (s, e), scaler, pca in zip(blocks, self.block_scalers_, self.block_pcas_)
        ]
        return np.concatenate(reduced, axis=1).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EmbEarlyFusionAlg":
        hp = read_hparams(self.params)
        n_blocks = len(self._emb_dims())
        emb_dropout = self._emb_dropout()
        hidden_dims = self._hidden_dims()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Xr = self._reduce_fit(X)
        self.normalizer = FeatureNormalizer(hp.normalize)
        Xs = self.normalizer.fit_transform(Xr)

        self.model = _EarlyFusionNet(
            n_blocks, self.pca_dim_, emb_dropout, hidden_dims, self.N_CLASSES, hp
        ).to(self.device)
        self.history_ = train_torch_model(
            self.model, [Xs], y, hp, self.device, self.seed, groups=self.fit_groups_,
        )
        return self

    @torch.no_grad()
    def _logits(self, X: np.ndarray) -> torch.Tensor:
        Xs = self.normalizer.transform(self._reduce_transform(X))
        X_t = torch.from_numpy(Xs).to(self.device)
        self.model.eval()
        return self.model(X_t)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._logits(X).argmax(dim=1).cpu().numpy()

    def decision_score(self, X: np.ndarray) -> np.ndarray:
        return logit_margin(self._logits(X))

    def save(self, path: Path) -> None:
        self.model.to("cpu")
        self.device = torch.device("cpu")
        joblib.dump(self, path)

"""Attention fusion: each embedding is one "token," self-attention learns how
to combine them.

Architecture #4 (the stretch goal) of
``epai/ai_detection/trainv2/embedding_ideas.md``. Each embedding block is
linearly projected to a shared width ``d_model`` (they arrive at different
native widths -- RoBERTa, SciBERT, LUAR, ... -- so a per-branch projection is
unavoidable before they can share one sequence), optionally joined by a
learned CLS token, and run through a small ``nn.TransformerEncoder``. This is
explicitly the highest-capacity, highest-overfitting-risk architecture of the
four -- a handful of "tokens" (one per embedding, N is small: 3-12) is a very
short sequence for self-attention to do anything more expressive than the
additive gate in :mod:`.emb_gated_n`, so treat a win here as informative and a
loss as expected on this little data, not as evidence attention fusion is
wrong in general. (Cross-modal attention framing: Lu et al., "ViLBERT",
*NeurIPS* 2019, though the modalities here are several text embeddings, not
vision+text.)

Config (``spec["params"]``)::

    {"emb_dims": [768, 1024, 768], "d_model": 64, "n_heads": 4, "n_layers": 1,
     "ff_dim": 128, "pool": "cls"}

``emb_dims`` is injected by the trainv2 loader, not hand-written; see
``epai/ai_detection/trainv2/_trained.py``.
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
    hidden_block,
    logit_margin,
    parse_dims,
    read_hparams,
    train_torch_model,
)
from .base import ClfAlgorithm

POOL_MODES = ("cls", "mean")


def _blocks(emb_dims: list[int]) -> list[tuple[int, int]]:
    starts = np.cumsum([0, *emb_dims[:-1]])
    return [(int(s), int(s + d)) for s, d in zip(starts, emb_dims)]


def parse_pool(mode: str) -> str:
    if mode not in POOL_MODES:
        raise ValueError(f"pool must be one of {POOL_MODES}, got {mode!r}")
    return mode


class _AttnFusionNet(nn.Module):
    def __init__(
        self, emb_dims: list[int], d_model: int, n_heads: int, n_layers: int,
        ff_dim: int, pool: str, post_dims: list[int], n_classes: int, hp: TrainHParams,
    ):
        super().__init__()
        self.blocks = _blocks(emb_dims)
        self.pool = pool
        self.proj = nn.ModuleList([nn.Linear(d, d_model) for d in emb_dims])
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model)) if pool == "cls" else None

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=hp.dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        head: list[nn.Module] = []
        prev = d_model
        for h in post_dims:
            head += hidden_block(prev, h, hp.dropout, hp.norm_layer)
            prev = h
        head.append(nn.Linear(prev, n_classes))
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack(
            [proj(x[:, s:e]) for proj, (s, e) in zip(self.proj, self.blocks)], dim=1
        )  # (batch, n_blocks, d_model)
        if self.cls is not None:
            tokens = torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1)
        encoded = self.encoder(tokens)
        pooled = encoded[:, 0, :] if self.pool == "cls" else encoded.mean(dim=1)
        return self.head(pooled)


class EmbAttentionFusionAlg(ClfAlgorithm):
    """Self-attention over embeddings-as-tokens.

    Architecture params:
      * ``emb_dims`` -- column widths of each embedding block, in concatenation
        order (required);
      * ``d_model`` -- shared token width every embedding is projected to
        (default 64; must be divisible by ``n_heads``);
      * ``n_heads`` -- attention heads (default 4);
      * ``n_layers`` -- encoder layers (default 1);
      * ``ff_dim`` -- feed-forward width inside each encoder layer (default 128);
      * ``pool`` -- ``"cls"`` (a learned CLS token) or ``"mean"`` (average over
        the embedding tokens) (default ``"cls"``);
      * ``post_dims`` -- classifier head widths after pooling (default ``[]``:
        one linear layer on the pooled vector).

    Every other hyper-parameter is shared with :mod:`simple_mlp` and documented
    on :class:`epai.algs._torch_common.TrainHParams` (``dropout`` is read
    inside each encoder layer as well as between any ``post_dims`` layers).
    """

    name = "emb_attn_fusion"
    N_CLASSES = 2
    PARAMS = (
        "emb_dims", "d_model", "n_heads", "n_layers", "ff_dim", "pool", "post_dims",
    ) + TRAIN_HPARAM_NAMES

    def _emb_dims(self) -> list[int]:
        dims = self.params.get("emb_dims")
        if not dims:
            raise ValueError(f"{self.name}: 'emb_dims' is required (block widths)")
        return [int(d) for d in dims]

    def _arch(self) -> dict:
        d_model = int(self.params.get("d_model", 64))
        n_heads = int(self.params.get("n_heads", 4))
        n_layers = int(self.params.get("n_layers", 1))
        ff_dim = int(self.params.get("ff_dim", 128))
        if d_model < 1 or n_heads < 1 or n_layers < 1 or ff_dim < 1:
            raise ValueError("d_model, n_heads, n_layers, ff_dim must all be >= 1")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        return {
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
            "ff_dim": ff_dim, "pool": parse_pool(self.params.get("pool", "cls")),
            "post_dims": parse_dims(self.params.get("post_dims", []), "post_dims"),
        }

    def validate_params(self) -> None:
        read_hparams(self.params)
        self._emb_dims()
        self._arch()

    def resolved_params(self) -> dict:
        return {
            "emb_dims": self._emb_dims(), **self._arch(),
            **hparams_dict(read_hparams(self.params)),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EmbAttentionFusionAlg":
        hp = read_hparams(self.params)
        emb_dims = self._emb_dims()
        arch = self._arch()
        if sum(emb_dims) != X.shape[1]:
            raise ValueError(
                f"{self.name}: emb_dims {emb_dims} sums to {sum(emb_dims)}, X "
                f"has {X.shape[1]} columns"
            )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.normalizer = FeatureNormalizer(hp.normalize)
        Xs = self.normalizer.fit_transform(X)

        self.model = _AttnFusionNet(
            emb_dims, arch["d_model"], arch["n_heads"], arch["n_layers"],
            arch["ff_dim"], arch["pool"], arch["post_dims"], self.N_CLASSES, hp,
        ).to(self.device)
        self.history_ = train_torch_model(
            self.model, [Xs], y, hp, self.device, self.seed, groups=self.fit_groups_,
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
        return logit_margin(self._logits(X))

    def save(self, path: Path) -> None:
        self.model.to("cpu")
        self.device = torch.device("cpu")
        joblib.dump(self, path)

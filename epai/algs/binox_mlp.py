"""A late-fusion MLP: the embedding is compressed, then the Binoculars score joins.

Sibling of :mod:`simple_mlp`. The difference is *where* the Binoculars score
enters the network. Concatenated at the input it has to survive the trunk's
compression of the embedding, and whatever is not linearly recoverable from the
bottleneck is gone; concatenated *after* the trunk it reaches the classifier
head on a path the compression never touches.
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
    hidden_block,
    hparams_dict,
    logit_margin,
    mlp_layers,
    parse_dims,
    parse_normalize,
    read_hparams,
    train_torch_model,
)
from .base import ClfAlgorithm


class _BinoxMLP(nn.Module):
    """Trunk over the embedding → concat the score → head → logits.

    ``pre_dims`` sizes the trunk, ``post_dims`` the head, both as
    (Linear → [LayerNorm] → ReLU → Dropout) blocks. An empty ``pre_dims`` leaves
    the trunk as a pass-through, which is exactly early fusion (the score
    concatenated at the input) -- the control this architecture is meant to be
    measured against.

    Dropout is applied to the trunk's output but never to the score: zeroing the
    score on a fraction of the steps would teach the model not to depend on it,
    which is the opposite of the point. ``input_dropout`` likewise reaches only
    the embedding, at the very front of the trunk.
    """

    def __init__(
        self,
        in_dim: int,
        aux_dim: int,
        pre_dims: list[int],
        post_dims: list[int],
        score_hidden: int,
        n_classes: int,
        hp: TrainHParams,
    ):
        super().__init__()

        layers, trunk_dim = mlp_layers(in_dim, pre_dims, hp)
        self.trunk = nn.Sequential(*layers)

        # A bare Linear on the score would be pointless -- an affine function of
        # one scalar that the next layer's weights simply absorb. With the ReLU
        # it becomes a set of hinges, i.e. the head can learn a non-monotonic
        # response to the score. Off by default; the zero-shot baseline's single
        # threshold already assumes monotonicity.
        if score_hidden:
            self.score = nn.Sequential(nn.Linear(aux_dim, score_hidden), nn.ReLU())
            score_dim = score_hidden
        else:
            self.score = nn.Identity()
            score_dim = aux_dim

        head: list[nn.Module] = []
        prev = trunk_dim + score_dim
        for h in post_dims:
            head += hidden_block(prev, h, hp.dropout, hp.norm_layer)
            prev = h
        head.append(nn.Linear(prev, n_classes))
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.trunk(x), self.score(z)], dim=1))


class BinoxMLPAlg(ClfAlgorithm):
    """A hybrid MLP over an embedding plus the zero-shot Binoculars score.

    Takes the auxiliary column the config's ``binoculars`` key names, as a
    second matrix ``Z`` alongside the design matrix ``X`` (see ``uses_aux``).
    Runs on CUDA when available, otherwise CPU. Architecture params:
      * ``pre_dims`` -- trunk widths, applied to the embedding before the score
        is concatenated (default ``[256, 64, 8]``; ``[]`` gives early fusion);
      * ``post_dims`` -- head widths, applied after the concatenation
        (default ``[16]``);
      * ``score_hidden`` -- project the score through ``Linear → ReLU`` to this
        width before concatenating (default 0, i.e. use the raw scalar);
      * ``aux_normalize`` -- how the score matrix is normalised (default
        ``standard``; see below).

    Every other hyper-parameter is shared with :mod:`simple_mlp` and documented
    on :class:`epai.algs._torch_common.TrainHParams` -- ``normalize``,
    ``val_frac``, ``dropout``, ``input_dropout``, ``norm_layer``,
    ``label_smoothing``, ``mixup_alpha``, ``epochs``, ``batch_size``, ``lr``,
    ``weight_decay``, ``patience``. Defaults reproduce this class's behaviour
    before the shared trainer existed.

    ``X`` and ``Z`` are normalised separately, both fitted on the *source*
    corpus's training split. The score's distribution shifts between corpora, so
    under the default ``standard`` it arrives standardised by statistics that are
    not the target's -- deliberately, since the zero-shot baseline calibrates its
    threshold on the source split too, which keeps the two comparable. ``Z`` has
    its own ``aux_normalize`` rather than following ``normalize`` because it is a
    single column: a per-row ``l2`` on one column collapses it to its sign, which
    the normaliser rejects outright rather than doing quietly.
    """

    name = "binox_mlp"
    uses_aux = True
    N_CLASSES = 2
    PARAMS = (
        "pre_dims",
        "post_dims",
        "score_hidden",
        "aux_normalize",
    ) + TRAIN_HPARAM_NAMES

    def _arch(self) -> dict:
        """This class's own params, parsed and validated. Shared by fit and preflight."""
        p = self.params
        try:
            score_hidden = int(p.get("score_hidden", 0))
        except (TypeError, ValueError):
            raise ValueError(
                f"score_hidden must be an int (0 = raw scalar), got "
                f"{p.get('score_hidden')!r}"
            ) from None
        if score_hidden < 0:
            raise ValueError(f"score_hidden must be >= 0 (0 disables), got {score_hidden}")
        aux_normalize = p.get("aux_normalize", "standard")
        parse_normalize(aux_normalize)  # raises on an unknown step
        return {
            "pre_dims": parse_dims(p.get("pre_dims", [256, 64, 8]), "pre_dims"),
            "post_dims": parse_dims(p.get("post_dims", [16]), "post_dims"),
            "score_hidden": score_hidden,
            "aux_normalize": aux_normalize,
        }

    def validate_params(self) -> None:
        read_hparams(self.params)  # ranges and enums
        self._arch()

    def resolved_params(self) -> dict:
        return {**self._arch(), **hparams_dict(read_hparams(self.params))}

    def _clean_aux(self, Z: np.ndarray) -> np.ndarray:
        """Replace non-finite scores with the training median of their column.

        A failed Binoculars computation leaves a NaN or an inf, and a single one
        of those would take the scaler's mean and std with it.
        """
        Z = np.asarray(Z, dtype=np.float64).reshape(len(Z), -1)
        bad = ~np.isfinite(Z)
        if bad.any():
            Z = np.where(bad, self.aux_fill_, Z)
        return Z

    def fit(
        self, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None = None
    ) -> "BinoxMLPAlg":
        if Z is None:
            raise ValueError(
                f"{type(self).__name__} needs the auxiliary Binoculars column; "
                f"set 'binoculars' on this algorithm's config entry"
            )

        p = self.params
        hp = read_hparams(p)
        arch = self._arch()
        pre_dims, post_dims = arch["pre_dims"], arch["post_dims"]
        score_hidden = arch["score_hidden"]

        # Seed torch and numpy so a run is reproducible for a given seed.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Median over the finite training values only, so _clean_aux has a fill
        # value that predict() can reuse unchanged on every target corpus.
        Z_raw = np.asarray(Z, dtype=np.float64).reshape(len(Z), -1)
        finite = np.where(np.isfinite(Z_raw), Z_raw, np.nan)
        self.aux_fill_ = np.nan_to_num(np.nanmedian(finite, axis=0), nan=0.0)

        self.normalizer = FeatureNormalizer(hp.normalize)
        self.aux_normalizer = FeatureNormalizer(arch["aux_normalize"])
        Xs = self.normalizer.fit_transform(X)
        Zs = self.aux_normalizer.fit_transform(self._clean_aux(Z))

        self.model = _BinoxMLP(
            Xs.shape[1],
            Zs.shape[1],
            pre_dims,
            post_dims,
            score_hidden,
            self.N_CLASSES,
            hp,
        ).to(self.device)
        self.history_ = train_torch_model(
            self.model,
            [Xs, Zs],
            y,
            hp,
            self.device,
            self.seed,
            groups=self.fit_groups_,
        )
        return self

    @torch.no_grad()
    def _logits(self, X: np.ndarray, Z: np.ndarray | None) -> torch.Tensor:
        if Z is None:
            raise ValueError(
                f"{type(self).__name__} needs the auxiliary Binoculars column at "
                f"predict time as well"
            )
        Xs = self.normalizer.transform(X)
        Zs = self.aux_normalizer.transform(self._clean_aux(Z))
        X_t = torch.from_numpy(Xs).to(self.device)
        Z_t = torch.from_numpy(Zs).to(self.device)
        self.model.eval()
        return self.model(X_t, Z_t)

    def predict(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        return self._logits(X, Z).argmax(dim=1).cpu().numpy()

    def decision_score(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        """The machine-minus-human logit margin. See :func:`logit_margin`."""
        return logit_margin(self._logits(X, Z))

    def save(self, path: Path) -> None:
        # Move to CPU first so the pickle can be loaded on a machine without a GPU.
        self.model.to("cpu")
        self.device = torch.device("cpu")
        joblib.dump(self, path)

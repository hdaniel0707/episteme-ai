"""A late-fusion MLP: the embedding is compressed, then a feature vector joins.

The successor to :mod:`binox_mlp`, which is the same architecture written when
the second input was a single Binoculars score. Nothing here is specific to a
score: the auxiliary input is a column of any width, and the case this exists
for is a *vector* one -- ``llm_judge_cuevec_gpt56luna_v3``, the eleven bits
recording which of the judge's closed-vocabulary cues it cited for a document
(see ``epai/ai_detection/score/add_cue_features.py``).

WHY NOT JUST CONCATENATE
------------------------
Stacking the eleven cue bits onto the 512-dim embedding and training
:mod:`simple_mlp` on the result is one line of config, and it is the control
this class must be measured against. The reason to expect more from late fusion
is NOT that a concatenating MLP cannot represent this one -- it can; its first
layer could learn to pass the eleven through untouched. It is that nothing makes
it do so:

  * eleven sparse binary dimensions receive a negligible share of the first
    layer's gradient next to 512 dense, correlated ones, and weight decay shrinks
    them on the same schedule as everything else;
  * ``input_dropout`` zeroes cue bits, and losing one of eleven informative bits
    is destructive where losing one of 512 redundant dimensions is not.

So the fusion point is a *prior*, not a capacity argument. ``pre_dims`` caps how
much of the embedding -- and therefore how much corpus-specific detail -- can
reach the classifier head at all, while the auxiliary vector reaches it on a
path the compression, the input dropout and (by default) the dropout never
touch. That asymmetry is the hypothesis: the embedding scores ~98% in-domain and
at or below chance across domains, so a head forced to lean on the cues is being
pushed towards the part of the input that has a chance of transferring.

WHAT THE ARCHITECTURE DOES NOT DO
---------------------------------
``aux_dims`` is described as "upsampling" the eleven bits, and it is worth being
precise about what that buys: nothing informational. Eleven bits are eleven bits
after a ``Linear -> ReLU`` to 32 units. What the branch gains is a basis of
hinges over cue *combinations*, so the head can express "``even_rhythm`` counts
for more when ``specifics`` is absent" -- which a linear read of the eleven
cannot. Expect ``pre_dims`` to decide the result and ``aux_dims`` to trim it.

READ IT AGAINST ITS CONTROLS
----------------------------
A number from this class means little on its own. Run it beside, on the same
rows and the same seeds: the embedding alone, the cue vector alone (under
``logistic``, whose eleven coefficients are readable), the two concatenated, and
this class with ``pre_dims: []`` -- which removes the compression and leaves
only the branch asymmetry, separating "the bottleneck did it" from "keeping the
cues out of the trunk did it".

Config::

    algorithms:
      - name: fusion_mlp
        id: fusion_cuevec
        label: "LUAR ⊕ judge cues (late fusion)"
        aux: llm_judge_cuevec_gpt56luna_v3
        params: {pre_dims: [256, 16], post_dims: [32], aux_dims: [32]}

with ``feature_sets: [emb_luar_crud]``. The auxiliary column is named on the
algorithm entry, never in the feature set: that is what keeps it out of ``X``.
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
    aux_fill_values,
    aux_layers,
    clean_aux,
    hidden_block,
    hparams_dict,
    logit_margin,
    mlp_layers,
    parse_aux_fill,
    parse_dims,
    parse_normalize,
    read_hparams,
    train_torch_model,
)
from .base import ClfAlgorithm

#: Shared hyper-parameters this class defaults differently from
#: :class:`~epai.algs._torch_common.TrainHParams`, which holds the values
#: :mod:`simple_mlp` and :mod:`binox_mlp` had before the shared trainer existed
#: and must keep for their old configs. This class is new, so it starts from what
#: a cross-domain run actually wants rather than from that history:
#:
#:   * ``val_frac`` / ``val_metric`` -- a real grouped held-out split, stopped on
#:     AUROC, because these runs are reported at a low-FPR operating point and
#:     stopping on the training loss regularises nothing;
#:   * ``weight_decay`` -- an order of magnitude up, aimed at the trunk: the
#:     embedding path is the one that should be squeezed;
#:   * ``input_dropout`` / ``dropout`` / ``norm_layer`` -- ordinary regularisation
#:     of that path, none of which reaches the auxiliary branch;
#:   * ``label_smoothing`` -- a little, against the over-confidence a 98%
#:     in-domain fit produces.
#:
#: ``mixup_alpha`` stays 0 and should be left there for a multi-hot auxiliary
#: column: mixup interpolates the two inputs together, and a half-cited cue is a
#: row that cannot occur.
HP_DEFAULTS = {
    "dropout": 0.2,
    "input_dropout": 0.1,
    "norm_layer": "layernorm",
    "weight_decay": 1e-3,
    "val_frac": 0.15,
    "val_metric": "auroc",
    "label_smoothing": 0.05,
}


class _FusionMLP(nn.Module):
    """Trunk over the embedding → concat the auxiliary branch → head → logits.

    ``pre_dims`` sizes the trunk, ``post_dims`` the head, both as
    (Linear → [LayerNorm] → ReLU → Dropout) blocks; ``aux_dims`` sizes the
    auxiliary branch, which is (Linear → ReLU → [Dropout]) blocks with no
    LayerNorm and no input dropout (see :func:`aux_layers` for why).

    An empty ``pre_dims`` leaves the trunk as a pass-through: the embedding
    reaches the head uncompressed, which is the control that isolates the
    bottleneck's contribution. It is *not* quite early fusion -- the auxiliary
    vector still bypasses ``input_dropout`` and still has its own branch -- and
    the honest early-fusion control is ``simple_mlp`` on the stacked columns.

    Dropout reaches the trunk's output but the auxiliary branch only through
    ``aux_dropout``, which defaults to 0: dropping the auxiliary signal on a
    fraction of the steps teaches the model not to depend on it, which is the
    opposite of the point.
    """

    def __init__(
        self,
        in_dim: int,
        aux_dim: int,
        pre_dims: list[int],
        post_dims: list[int],
        aux_dims: list[int],
        aux_dropout: float,
        n_classes: int,
        hp: TrainHParams,
    ):
        super().__init__()

        layers, trunk_dim = mlp_layers(in_dim, pre_dims, hp)
        self.trunk = nn.Sequential(*layers)

        aux_stack, aux_out = aux_layers(aux_dim, aux_dims, aux_dropout)
        self.aux = nn.Sequential(*aux_stack)

        head: list[nn.Module] = []
        prev = trunk_dim + aux_out
        for h in post_dims:
            head += hidden_block(prev, h, hp.dropout, hp.norm_layer)
            prev = h
        head.append(nn.Linear(prev, n_classes))
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.trunk(x), self.aux(z)], dim=1))


class FusionMLPAlg(ClfAlgorithm):
    """Late-fusion MLP over an embedding plus an auxiliary feature column.

    Takes the column the config entry's ``aux`` key names as a second matrix
    ``Z`` alongside the design matrix ``X`` (see ``uses_aux``), of any width.
    Runs on CUDA when available, otherwise CPU. Architecture params:

      * ``pre_dims`` -- trunk widths, applied to the embedding before the
        auxiliary branch is concatenated (default ``[256, 16]``; ``[]`` leaves
        the embedding uncompressed);
      * ``post_dims`` -- head widths, applied after the concatenation
        (default ``[32]``);
      * ``aux_dims`` -- auxiliary branch widths (default ``[32]``; ``[]`` passes
        the raw column through);
      * ``aux_dropout`` -- dropout inside that branch (default 0.0, i.e. the
        auxiliary path is not regularised at all);
      * ``aux_normalize`` -- how ``Z`` is normalised (default ``none``);
      * ``aux_fill`` -- what replaces a non-finite auxiliary entry
        (``zero`` | ``median``, default ``zero``).

    Every other hyper-parameter is shared with :mod:`simple_mlp` and documented
    on :class:`epai.algs._torch_common.TrainHParams`. This class overrides seven
    of their defaults; see ``HP_DEFAULTS`` above for which and why.

    WHY ``aux_normalize`` DEFAULTS TO ``none``
    ------------------------------------------
    :mod:`binox_mlp` standardises its auxiliary column, which is right for a
    continuous score whose scale is arbitrary. It is wrong for a multi-hot cue
    vector, and quietly so. ``standard`` learns each cue's *citation rate on the
    source corpus* and carries it to every target: a cue cited on 3% of essay
    rows is multiplied by ~5.7 and re-centred, and base rates are precisely what
    shifts between domains. That injects source-corpus statistics into the one
    input whose whole claim is that it means the same thing everywhere -- so the
    default leaves 0/1 as 0/1, and ``standard`` is available as the control that
    measures what the choice was worth.

    ``X`` and ``Z`` are normalised separately, both fitted on the source corpus's
    training split, and both fits are reused unchanged on every target.
    """

    name = "fusion_mlp"
    uses_aux = True
    N_CLASSES = 2
    PARAMS = (
        "pre_dims",
        "post_dims",
        "aux_dims",
        "aux_dropout",
        "aux_normalize",
        "aux_fill",
    ) + TRAIN_HPARAM_NAMES

    def _hp(self) -> TrainHParams:
        """The shared hyper-parameters, this class's defaults applied first."""
        return read_hparams(self.params, **HP_DEFAULTS)

    def _arch(self) -> dict:
        """This class's own params, parsed and validated. Shared by fit and preflight."""
        p = self.params
        try:
            aux_dropout = float(p.get("aux_dropout", 0.0))
        except (TypeError, ValueError):
            raise ValueError(
                f"aux_dropout must be a float in [0, 1), got {p.get('aux_dropout')!r}"
            ) from None
        if not 0.0 <= aux_dropout < 1.0:
            raise ValueError(f"aux_dropout must be in [0, 1), got {aux_dropout}")
        aux_normalize = p.get("aux_normalize", "none")
        parse_normalize(aux_normalize)  # raises on an unknown step
        return {
            "pre_dims": parse_dims(p.get("pre_dims", [256, 16]), "pre_dims"),
            "post_dims": parse_dims(p.get("post_dims", [32]), "post_dims"),
            "aux_dims": parse_dims(p.get("aux_dims", [32]), "aux_dims"),
            "aux_dropout": aux_dropout,
            "aux_normalize": aux_normalize,
            "aux_fill": parse_aux_fill(p.get("aux_fill", "zero")),
        }

    def validate_params(self) -> None:
        self._hp()  # ranges and enums
        self._arch()

    def resolved_params(self) -> dict:
        return {**self._arch(), **hparams_dict(self._hp())}

    def fit(
        self, X: np.ndarray, y: np.ndarray, Z: np.ndarray | None = None
    ) -> "FusionMLPAlg":
        if Z is None:
            raise ValueError(
                f"{type(self).__name__} needs an auxiliary column; set 'aux' on "
                f"this algorithm's config entry"
            )

        hp = self._hp()
        arch = self._arch()

        # Seed torch and numpy so a run is reproducible for a given seed.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Fitted on the training rows only, so predict() has a fill value it can
        # reuse unchanged on every target corpus.
        self.aux_fill_ = aux_fill_values(Z, arch["aux_fill"])

        self.normalizer = FeatureNormalizer(hp.normalize)
        self.aux_normalizer = FeatureNormalizer(arch["aux_normalize"])
        Xs = self.normalizer.fit_transform(X)
        Zs = self.aux_normalizer.fit_transform(clean_aux(Z, self.aux_fill_))

        self.model = _FusionMLP(
            Xs.shape[1],
            Zs.shape[1],
            arch["pre_dims"],
            arch["post_dims"],
            arch["aux_dims"],
            arch["aux_dropout"],
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
                f"{type(self).__name__} needs its auxiliary column at predict "
                f"time as well"
            )
        Xs = self.normalizer.transform(X)
        Zs = self.aux_normalizer.transform(clean_aux(Z, self.aux_fill_))
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

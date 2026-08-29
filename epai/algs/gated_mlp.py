"""Two branches, added as logits, with a learned gate on the embedding's half.

    logit = f_aux(z) + alpha * g_emb(x)

Both branches emit a full set of class logits and the network's output is their
sum, so the decision margin decomposes exactly::

    margin(x, z) = margin_aux(z) + alpha * margin_emb(x)

Nothing else in this package has that property. :mod:`fusion_mlp` concatenates
its two paths and lets a head mix them, which is strictly more expressive and
completely opaque: once a run's transfer matrix collapses off-domain there is no
way to ask *which input did that*. Here there is. :meth:`GatedMLPAlg.branch_scores`
returns both terms per row, so the same fitted model can be scored on every
target corpus and each branch's transfer read separately -- and ``alpha``, one
number, says how much of the embedding branch the fit chose to use at all.

WHAT THIS IS FOR
----------------
The finding this package is stuck on is a classifier that scores ~98% in-domain
on ``emb_luar_crud`` and at or below chance across domains, with the sign of the
error flipping. Two questions follow, and a concatenating model answers neither:

  * does the auxiliary input (the judge's eleven cue bits, say) transfer flat
    while the embedding inverts? ``margin_aux`` on the target corpus answers it
    directly, from the *fused* model rather than from a separately trained one;
  * is the embedding branch actively harmful off-domain, rather than merely
    uninformative? Score ``margin_aux`` alone against ``margin_aux + alpha *
    margin_emb`` on the target and the difference is the damage, in AUROC.

READ ``alpha`` FOR WHAT IT IS
-----------------------------
It is fitted on the source corpus, and so is the validation split that early
stopping reads. A 512-dim embedding can fit an in-domain split almost perfectly,
so the gradient has every reason to push ``alpha`` up regardless of what happens
off-domain: a high ``alpha`` is NOT evidence that the embedding branch will
transfer. It is a measurement of how much the model leaned on it, which is
exactly what you need to interpret the transfer numbers -- a model that
collapsed off-domain with ``alpha`` near zero collapsed for some other reason.

Config::

    algorithms:
      - name: gated_mlp
        id: gated_cuevec
        label: "judge cues + gated LUAR"
        aux: llm_judge_cuevec_gpt56luna_v3
        params: {emb_dims: [256, 64], aux_dims: [32], gate: sigmoid}

with ``feature_sets: [emb_luar_crud]``. Reading the gate and the two branches
back off a saved fit::

    alg = joblib.load("gated_cuevec/models/0/essay_gpt56luna.joblib")
    alg.gate_                       # the learned alpha, as a float
    parts = alg.branch_scores(X, Z) # {'aux', 'emb', 'gated_emb', 'total', 'alpha'}
    roc_auc_score(y, parts["aux"])  # what the cues alone do on THIS corpus
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._torch_common import (
    TRAIN_HPARAM_NAMES,
    FeatureNormalizer,
    TrainHParams,
    aux_fill_values,
    aux_layers,
    clean_aux,
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
from .fusion_mlp import HP_DEFAULTS

#: How the learned gate is parameterised. ``alpha`` is a single scalar in every
#: case; what differs is the range it is confined to.
#:
#: ``sigmoid``
#:     ``alpha = sigmoid(a)``, in (0, 1). The default: the auxiliary branch is
#:     the reference at weight 1 and the embedding branch can at most match it,
#:     which makes the number readable as "how much of the embedding was used"
#:     rather than as an arbitrary scale.
#: ``softplus``
#:     ``alpha = softplus(a)``, non-negative and unbounded. Use when the
#:     embedding is expected to dominate and you do not want the ceiling.
#: ``free``
#:     ``alpha = a``, any real. Kept reachable for completeness; note that the
#:     sign is not identifiable (the embedding branch's own head can flip its
#:     weights and the product is unchanged), so a negative ``alpha`` is not
#:     evidence of an inverted branch.
GATE_MODES = ("sigmoid", "softplus", "free")


def parse_gate(mode: str) -> str:
    """Validate a ``gate`` setting. Returns it unchanged."""
    if mode not in GATE_MODES:
        raise ValueError(f"gate must be one of {sorted(GATE_MODES)}, got {mode!r}")
    return mode


class _GatedMLP(nn.Module):
    """``f_aux(z) + alpha * g_emb(x)``, both branches ending in a logit head.

    The two branches are the ones :mod:`fusion_mlp` uses, with the same
    asymmetry: the embedding path gets ``input_dropout``, ``dropout`` and
    optional LayerNorm, the auxiliary path gets none of them unless
    ``aux_dropout`` is set (see :func:`aux_layers`).

    The gate is one scalar parameter and is optimised with everything else,
    which means AdamW's weight decay applies to it too -- a weak pull of the
    pre-activation towards 0, i.e. towards ``alpha = 0.5`` under ``sigmoid``.
    That is deliberate slack rather than an oversight: the pull is tiny beside
    the loss gradient on a scalar the whole network's output scale depends on.
    """

    def __init__(
        self,
        in_dim: int,
        aux_dim: int,
        emb_dims: list[int],
        aux_dims: list[int],
        aux_dropout: float,
        n_classes: int,
        hp: TrainHParams,
        gate: str,
        gate_init: float,
    ):
        super().__init__()
        self.gate_mode = gate

        emb_stack, emb_out = mlp_layers(in_dim, emb_dims, hp)
        emb_stack.append(nn.Linear(emb_out, n_classes))
        self.emb = nn.Sequential(*emb_stack)

        aux_stack, aux_out = aux_layers(aux_dim, aux_dims, aux_dropout)
        aux_stack.append(nn.Linear(aux_out, n_classes))
        self.aux = nn.Sequential(*aux_stack)

        self.gate_raw = nn.Parameter(torch.tensor(float(gate_init)))

    def alpha(self) -> torch.Tensor:
        """The gate, squashed into this mode's range. A 0-dim tensor."""
        if self.gate_mode == "sigmoid":
            return torch.sigmoid(self.gate_raw)
        if self.gate_mode == "softplus":
            return F.softplus(self.gate_raw)
        return self.gate_raw

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.aux(z) + self.alpha() * self.emb(x)


class GatedMLPAlg(ClfAlgorithm):
    """Additive-logit fusion of an embedding and an auxiliary column, with a gate.

    Takes the column the config entry's ``aux`` key names as a second matrix
    ``Z`` alongside the design matrix ``X`` (see ``uses_aux``), of any width.
    Runs on CUDA when available, otherwise CPU. Architecture params:

      * ``emb_dims`` -- hidden widths of the embedding branch before its logit
        head (default ``[256, 64]``; ``[]`` makes it a linear probe);
      * ``aux_dims`` -- hidden widths of the auxiliary branch (default ``[32]``;
        ``[]`` makes it a linear read of the raw column);
      * ``aux_dropout`` -- dropout inside the auxiliary branch (default 0.0);
      * ``gate`` -- how ``alpha`` is parameterised (``sigmoid`` | ``softplus`` |
        ``free``, default ``sigmoid``; see ``GATE_MODES``);
      * ``gate_init`` -- the gate's pre-activation at initialisation (default
        0.0, i.e. ``alpha = 0.5`` under ``sigmoid``, ``0.693`` under
        ``softplus``);
      * ``aux_normalize`` -- how ``Z`` is normalised (default ``none``; see
        :class:`~epai.algs.fusion_mlp.FusionMLPAlg` for why not ``standard``);
      * ``aux_fill`` -- what replaces a non-finite auxiliary entry
        (``zero`` | ``median``, default ``zero``).

    Every other hyper-parameter is shared with :mod:`simple_mlp` and documented
    on :class:`epai.algs._torch_common.TrainHParams`; this class starts from the
    same overridden defaults as :mod:`fusion_mlp` (``HP_DEFAULTS`` there).

    ``emb_dims`` defaults wider than :mod:`fusion_mlp`'s ``pre_dims`` on purpose.
    There the bottleneck IS the regularizer -- it is the only thing limiting how
    much embedding reaches the head. Here the gate does that job explicitly, so
    the branch is left able to fit, and how much of it survives into the sum is
    a fitted quantity you can read rather than an architectural constant you
    chose in advance.

    After ``fit`` the learned gate is on ``self.gate_`` (a float) and the branch
    decomposition is available through :meth:`branch_scores`.
    """

    name = "gated_mlp"
    uses_aux = True
    N_CLASSES = 2
    PARAMS = (
        "emb_dims",
        "aux_dims",
        "aux_dropout",
        "gate",
        "gate_init",
        "aux_normalize",
        "aux_fill",
    ) + TRAIN_HPARAM_NAMES

    def _hp(self) -> TrainHParams:
        """The shared hyper-parameters, this family's defaults applied first."""
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
        try:
            gate_init = float(p.get("gate_init", 0.0))
        except (TypeError, ValueError):
            raise ValueError(
                f"gate_init must be a float, got {p.get('gate_init')!r}"
            ) from None
        if not np.isfinite(gate_init):
            raise ValueError(f"gate_init must be finite, got {gate_init}")
        aux_normalize = p.get("aux_normalize", "none")
        parse_normalize(aux_normalize)  # raises on an unknown step
        return {
            "emb_dims": parse_dims(p.get("emb_dims", [256, 64]), "emb_dims"),
            "aux_dims": parse_dims(p.get("aux_dims", [32]), "aux_dims"),
            "aux_dropout": aux_dropout,
            "gate": parse_gate(p.get("gate", "sigmoid")),
            "gate_init": gate_init,
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
    ) -> "GatedMLPAlg":
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

        self.model = _GatedMLP(
            Xs.shape[1],
            Zs.shape[1],
            arch["emb_dims"],
            arch["aux_dims"],
            arch["aux_dropout"],
            self.N_CLASSES,
            hp,
            arch["gate"],
            arch["gate_init"],
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
        # Recorded on the fitted object, not only in the weights: this is the
        # number the whole architecture exists to expose, and a results
        # directory should not need torch loaded to read it.
        with torch.no_grad():
            self.gate_ = float(self.model.alpha().item())
        return self

    def _inputs(self, X: np.ndarray, Z: np.ndarray | None) -> tuple:
        """Normalise both inputs and put them on the device, as ``fit`` did."""
        if Z is None:
            raise ValueError(
                f"{type(self).__name__} needs its auxiliary column at predict "
                f"time as well"
            )
        Xs = self.normalizer.transform(X)
        Zs = self.aux_normalizer.transform(clean_aux(Z, self.aux_fill_))
        return (
            torch.from_numpy(Xs).to(self.device),
            torch.from_numpy(Zs).to(self.device),
        )

    @torch.no_grad()
    def _logits(self, X: np.ndarray, Z: np.ndarray | None) -> torch.Tensor:
        X_t, Z_t = self._inputs(X, Z)
        self.model.eval()
        return self.model(X_t, Z_t)

    def predict(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        return self._logits(X, Z).argmax(dim=1).cpu().numpy()

    def decision_score(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        """The machine-minus-human logit margin. See :func:`logit_margin`."""
        return logit_margin(self._logits(X, Z))

    @torch.no_grad()
    def branch_scores(self, X: np.ndarray, Z: np.ndarray | None = None) -> dict:
        """Each branch's contribution to the decision score, per row.

        Returns ``{'aux', 'emb', 'gated_emb', 'total', 'alpha'}``, where the
        first four are margins in the sense of :func:`logit_margin` and
        ``total`` is what :meth:`decision_score` returns. By construction
        ``total == aux + gated_emb`` and ``gated_emb == alpha * emb`` (to float
        precision), which is what makes a per-branch AUROC on a *target* corpus
        meaningful: the two numbers are terms of the same sum, not two models.

        Not part of the :class:`ClfAlgorithm` contract and not called by the
        runners -- this is for reading a saved fit afterwards.
        """
        X_t, Z_t = self._inputs(X, Z)
        self.model.eval()
        aux = logit_margin(self.model.aux(Z_t))
        emb = logit_margin(self.model.emb(X_t))
        alpha = float(self.model.alpha().item())
        return {
            "aux": aux,
            "emb": emb,
            "gated_emb": alpha * emb,
            "total": aux + alpha * emb,
            "alpha": alpha,
        }

    def save(self, path: Path) -> None:
        # Move to CPU first so the pickle can be loaded on a machine without a GPU.
        self.model.to("cpu")
        self.device = torch.device("cpu")
        joblib.dump(self, path)

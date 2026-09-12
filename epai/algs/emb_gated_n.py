"""N-branch generalization of :mod:`.gated_mlp`: one branch per embedding, each
with its own learned gate, summed as logits.

    logit = sum_i alpha_i * branch_i(x_i)

Architecture #3 of ``epai/ai_detection/trainv2/embedding_ideas.md``.
:mod:`.gated_mlp` gates exactly two inputs of different KINDS (an embedding vs.
a judge cue vector); this gates N inputs of the SAME kind (several
embeddings), so every branch has an identical architecture and only its gate
differs. The additive-logit structure is what :mod:`.gated_mlp`'s docstring
calls out as its whole point: the margin decomposes exactly,

    margin(x_1, ..., x_N) = sum_i alpha_i * margin_i(x_i)

so :meth:`branch_scores` can read, on any target corpus, which embedding a
fitted model actually leaned on -- ``alpha_i`` near zero for the branch that
collapses off-domain is a different finding than every branch collapsing
together.

READ THE ALPHAS FOR WHAT THEY ARE (see :mod:`.gated_mlp`'s docstring in full):
each is fit on the source corpus, together with the validation split early
stopping reads, so a high alpha is evidence of how much the model LEANED ON
that branch in-domain, not evidence that the branch transfers.

Config (``spec["params"]``)::

    {"emb_dims": [768, 1024, 768], "branch_dims": [256, 64], "gate": "sigmoid"}

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
    logit_margin,
    mlp_layers,
    parse_dims,
    read_hparams,
    train_torch_model,
)
from .base import ClfAlgorithm
from .gated_mlp import parse_gate


def _blocks(emb_dims: list[int]) -> list[tuple[int, int]]:
    starts = np.cumsum([0, *emb_dims[:-1]])
    return [(int(s), int(s + d)) for s, d in zip(starts, emb_dims)]


class _GatedNMLP(nn.Module):
    def __init__(
        self, emb_dims: list[int], branch_dims: list[int], n_classes: int,
        hp: TrainHParams, gate: str, gate_init: float,
    ):
        super().__init__()
        self.blocks = _blocks(emb_dims)
        self.gate_mode = gate

        self.branches = nn.ModuleList()
        for d in emb_dims:
            layers, out = mlp_layers(d, branch_dims, hp)
            layers.append(nn.Linear(out, n_classes))
            self.branches.append(nn.Sequential(*layers))

        self.gate_raw = nn.Parameter(torch.full((len(emb_dims),), float(gate_init)))

    def alpha(self) -> torch.Tensor:
        """The N gates, squashed into this mode's range. Shape ``(n_branches,)``."""
        if self.gate_mode == "sigmoid":
            return torch.sigmoid(self.gate_raw)
        if self.gate_mode == "softplus":
            return nn.functional.softplus(self.gate_raw)
        return self.gate_raw

    def branch_logits(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [branch(x[:, s:e]) for branch, (s, e) in zip(self.branches, self.blocks)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha()
        logits = self.branch_logits(x)
        return sum(a * lg for a, lg in zip(alpha, logits))


class EmbGatedNAlg(ClfAlgorithm):
    """Additive-logit fusion of N embedding branches, each with its own gate.

    Architecture params:
      * ``emb_dims`` -- column widths of each embedding block, in concatenation
        order (required);
      * ``branch_dims`` -- hidden widths of every branch before its logit head
        (default ``[256, 64]``; ``[]`` makes every branch a linear probe);
      * ``gate`` -- how each ``alpha_i`` is parameterised (``sigmoid`` |
        ``softplus`` | ``free``, default ``sigmoid``; see
        :data:`epai.algs.gated_mlp.GATE_MODES`);
      * ``gate_init`` -- every gate's pre-activation at initialisation
        (default 0.0).

    Every other hyper-parameter is shared with :mod:`simple_mlp` and documented
    on :class:`epai.algs._torch_common.TrainHParams`.

    After ``fit``, ``self.gates_`` holds the fitted alphas (one float per
    embedding, in ``emb_dims`` order) and :meth:`branch_scores` decomposes any
    matrix's decision score per branch, exactly like
    :meth:`epai.algs.gated_mlp.GatedMLPAlg.branch_scores`.
    """

    name = "emb_gated_n"
    N_CLASSES = 2
    PARAMS = ("emb_dims", "branch_dims", "gate", "gate_init") + TRAIN_HPARAM_NAMES

    def _emb_dims(self) -> list[int]:
        dims = self.params.get("emb_dims")
        if not dims:
            raise ValueError(f"{self.name}: 'emb_dims' is required (block widths)")
        return [int(d) for d in dims]

    def _branch_dims(self) -> list[int]:
        return parse_dims(self.params.get("branch_dims", [256, 64]), "branch_dims")

    def _gate(self) -> str:
        return parse_gate(self.params.get("gate", "sigmoid"))

    def _gate_init(self) -> float:
        v = float(self.params.get("gate_init", 0.0))
        if not np.isfinite(v):
            raise ValueError(f"gate_init must be finite, got {v}")
        return v

    def validate_params(self) -> None:
        read_hparams(self.params)
        self._emb_dims()
        self._branch_dims()
        self._gate()
        self._gate_init()

    def resolved_params(self) -> dict:
        return {
            "emb_dims": self._emb_dims(), "branch_dims": self._branch_dims(),
            "gate": self._gate(), "gate_init": self._gate_init(),
            **hparams_dict(read_hparams(self.params)),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EmbGatedNAlg":
        hp = read_hparams(self.params)
        emb_dims = self._emb_dims()
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

        self.model = _GatedNMLP(
            emb_dims, self._branch_dims(), self.N_CLASSES, hp,
            self._gate(), self._gate_init(),
        ).to(self.device)
        self.history_ = train_torch_model(
            self.model, [Xs], y, hp, self.device, self.seed, groups=self.fit_groups_,
        )
        with torch.no_grad():
            self.gates_ = [float(a) for a in self.model.alpha().cpu().numpy()]
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

    @torch.no_grad()
    def branch_scores(self, X: np.ndarray) -> dict:
        """Each branch's contribution to the decision score, per row.

        Returns ``{'branches': [margin_0, ...], 'gated': [alpha_0*margin_0,
        ...], 'total': ..., 'alphas': [...]}``. ``total`` equals
        :meth:`decision_score` and equals ``sum(gated)`` to float precision.
        Not part of the :class:`ClfAlgorithm` contract -- for reading a saved
        fit afterwards.
        """
        Xs = self.normalizer.transform(X)
        X_t = torch.from_numpy(Xs).to(self.device)
        self.model.eval()
        alpha = self.model.alpha()
        branch_logits = self.model.branch_logits(X_t)
        branches = [logit_margin(lg) for lg in branch_logits]
        alphas = [float(a) for a in alpha.cpu().numpy()]
        gated = [a * m for a, m in zip(alphas, branches)]
        return {
            "branches": branches, "gated": gated,
            "total": sum(gated), "alphas": alphas,
        }

    def save(self, path: Path) -> None:
        self.model.to("cpu")
        self.device = torch.device("cpu")
        joblib.dump(self, path)

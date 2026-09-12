"""Pluggable binary classifiers for the cross-domain algorithm comparison.

Each algorithm is a :class:`ClfAlgorithm` subclass in its own module. The
``ALGORITHMS`` registry maps the config ``name`` to its class, and
``build_algorithm`` instantiates one with a seed and its config ``params``. Add
a new algorithm by writing a subclass and adding it to the registry below.
"""

from __future__ import annotations

from .base import ClfAlgorithm
from .binox_mlp import BinoxMLPAlg
from .emb_attention_fusion import EmbAttentionFusionAlg
from .emb_early_fusion import EmbEarlyFusionAlg
from .emb_gated_n import EmbGatedNAlg
from .emb_stacking import EmbStackingAlg
from .fusion_mlp import FusionMLPAlg
from .gated_mlp import GatedMLPAlg
from .gradient_boosting import GradientBoostingAlg
from .judge_distill import JudgeDistillHybridAlg, JudgeDistillRegressAlg
from .logistic import LogisticAlg
from .robust_mlp import RobustMLPAlg
from .svm import SVMAlg
from .simple_mlp import SimpleMLPAlg

ALGORITHMS: dict[str, type[ClfAlgorithm]] = {
    LogisticAlg.name: LogisticAlg,
    SVMAlg.name: SVMAlg,
    GradientBoostingAlg.name: GradientBoostingAlg,
    SimpleMLPAlg.name: SimpleMLPAlg,
    BinoxMLPAlg.name: BinoxMLPAlg,
    FusionMLPAlg.name: FusionMLPAlg,
    GatedMLPAlg.name: GatedMLPAlg,
    RobustMLPAlg.name: RobustMLPAlg,
    EmbStackingAlg.name: EmbStackingAlg,
    EmbEarlyFusionAlg.name: EmbEarlyFusionAlg,
    EmbGatedNAlg.name: EmbGatedNAlg,
    EmbAttentionFusionAlg.name: EmbAttentionFusionAlg,
    JudgeDistillRegressAlg.name: JudgeDistillRegressAlg,
    JudgeDistillHybridAlg.name: JudgeDistillHybridAlg,
}


def build_algorithm(name: str, seed: int = 0, **params) -> ClfAlgorithm:
    """Instantiate a registered algorithm by name with its hyper-parameters."""
    try:
        cls = ALGORITHMS[name]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm {name!r}. Choose from: {sorted(ALGORITHMS)}"
        ) from None
    return cls(seed=seed, **params)


__all__ = [
    "ALGORITHMS",
    "BinoxMLPAlg",
    "ClfAlgorithm",
    "EmbAttentionFusionAlg",
    "EmbEarlyFusionAlg",
    "EmbGatedNAlg",
    "EmbStackingAlg",
    "FusionMLPAlg",
    "GatedMLPAlg",
    "GradientBoostingAlg",
    "JudgeDistillHybridAlg",
    "JudgeDistillRegressAlg",
    "LogisticAlg",
    "RobustMLPAlg",
    "SVMAlg",
    "SimpleMLPAlg",
    "build_algorithm",
]

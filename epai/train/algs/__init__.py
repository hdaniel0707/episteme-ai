"""Pluggable binary classifiers for the cross-domain algorithm comparison.

Each algorithm is a :class:`ClfAlgorithm` subclass in its own module. The
``ALGORITHMS`` registry maps the config ``name`` to its class, and
``build_algorithm`` instantiates one with a seed and its config ``params``. Add
a new algorithm by writing a subclass and adding it to the registry below.
"""

from __future__ import annotations

from .base import ClfAlgorithm
from .gradient_boosting import GradientBoostingAlg
from .logistic import LogisticAlg
from .svm import SVMAlg
from .torch_mlp import TorchMLPAlg

ALGORITHMS: dict[str, type[ClfAlgorithm]] = {
    LogisticAlg.name: LogisticAlg,
    SVMAlg.name: SVMAlg,
    GradientBoostingAlg.name: GradientBoostingAlg,
    TorchMLPAlg.name: TorchMLPAlg,
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
    "ClfAlgorithm",
    "GradientBoostingAlg",
    "LogisticAlg",
    "SVMAlg",
    "TorchMLPAlg",
    "build_algorithm",
]

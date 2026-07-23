"""Abstract base for the pluggable cross-domain classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np


class ClfAlgorithm(ABC):
    """A binary human-vs-machine classifier for the cross-domain comparison.

    One instance is fit on a single corpus's training split and then asked to
    predict on that corpus's held-out split (in-domain) and on every other
    corpus (transfer). Implementations own their own preprocessing -- feature
    scaling, for example -- so the runner only ever hands them the raw stacked
    design matrix and calls ``fit`` / ``predict``.

    To add an algorithm:
      * subclass this and set the class attribute ``name`` (the key a config
        selects it by and the folder its results land in);
      * implement ``fit(X, y)`` and ``predict(X)``;
      * register it in ``epai/train/algs/__init__.py``;
      * optionally override ``save`` if the default joblib pickle is unsuitable.

    ``params`` captures whatever keyword arguments the config passed, so an
    algorithm is fully described by ``(name, seed, params)`` and each subclass
    reads its own hyper-parameters out of ``self.params`` with sensible
    defaults.
    """

    #: Registry key / output-folder name. Overridden by every concrete subclass.
    name: str = "base"

    def __init__(self, seed: int = 0, **params):
        self.seed = int(seed)
        self.params = dict(params)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClfAlgorithm":
        """Fit on one corpus's training split. Returns ``self``."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class labels (0 = human, 1 = machine), one per row of ``X``."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Persist the fitted algorithm. Default: joblib-pickle the whole object.

        Best-effort and overridable; the runner wraps the call so a
        serialization failure never aborts a long run.
        """
        joblib.dump(self, path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(seed={self.seed}, params={self.params})"

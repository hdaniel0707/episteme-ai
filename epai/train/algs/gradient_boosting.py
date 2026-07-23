"""Gradient-boosted decision trees."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .base import ClfAlgorithm


class GradientBoostingAlg(ClfAlgorithm):
    """Gradient-boosted trees. No scaling -- trees are scale-invariant.

    Params (all optional):
      * ``n_estimators`` -- number of boosting stages (default 100);
      * ``learning_rate`` -- shrinkage (default 0.1);
      * ``max_depth`` -- per-tree depth (default 3).
    """

    name = "gradient_boosting"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingAlg":
        self.model = GradientBoostingClassifier(
            n_estimators=self.params.get("n_estimators", 100),
            learning_rate=self.params.get("learning_rate", 0.1),
            max_depth=self.params.get("max_depth", 3),
            random_state=self.seed,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

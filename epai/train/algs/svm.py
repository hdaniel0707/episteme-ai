"""Support-vector classifier on standardised features."""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .base import ClfAlgorithm


class SVMAlg(ClfAlgorithm):
    """SVC with a StandardScaler front-end (SVMs are scale-sensitive).

    Params (all optional):
      * ``C`` -- regularisation (default 1.0);
      * ``kernel`` -- 'rbf' (default), 'linear', 'poly', 'sigmoid';
      * ``gamma`` -- 'scale' (default), 'auto', or a float.
    """

    name = "svm"
    PARAMS = ("C", "kernel", "gamma")

    def resolved_params(self) -> dict:
        return {
            "C": self.params.get("C", 1.0),
            "kernel": self.params.get("kernel", "rbf"),
            "gamma": self.params.get("gamma", "scale"),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMAlg":
        self.model = make_pipeline(
            StandardScaler(),
            SVC(
                C=self.params.get("C", 1.0),
                kernel=self.params.get("kernel", "rbf"),
                gamma=self.params.get("gamma", "scale"),
                random_state=self.seed,
            ),
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

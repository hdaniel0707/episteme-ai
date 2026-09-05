"""scikit-learn LogisticRegression/LinearRegression baselines — same interface as the torch
heads."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


class SklearnLogisticRegressionClassifier:
    """Multinomial logistic regression baseline for the ordinal impact classes."""

    task = "clf"
    net_ = None

    def __init__(self, *, n_classes: int, seed: int = 42,
                 class_weight: str = "effective", max_iter: int = 2000):
        self.n_classes = int(n_classes)
        self.seed = int(seed)
        sk_weight = "balanced" if (class_weight or "").lower() == "balanced" else None
        self.model_ = LogisticRegression(max_iter=int(max_iter), random_state=self.seed,
                                         class_weight=sk_weight)
        self.history_: dict = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        """Fit on TRAIN; val is accepted for interface parity but unused (no early stop)."""
        self.model_.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Full [N, n_classes] proba matrix even if a class was absent from TRAIN."""
        proba = np.zeros((len(X), self.n_classes))
        raw = self.model_.predict_proba(X)
        for j, c in enumerate(self.model_.classes_):
            proba[:, int(c)] = raw[:, j]
        return proba


class SklearnLinearRegressionBaseline:
    """Plain (unregularized) linear regression baseline for the impact score."""

    task = "reg"
    net_ = None

    def __init__(self, *, seed: int = 42):
        self.seed = int(seed)
        self.model_ = LinearRegression()
        self.history_: dict = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        """Fit on TRAIN; val is accepted for interface parity but unused (no early stop)."""
        self.model_.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

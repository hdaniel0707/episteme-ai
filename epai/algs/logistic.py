"""L2 logistic regression on standardised features."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import ClfAlgorithm


class LogisticAlg(ClfAlgorithm):
    """Logistic regression with a StandardScaler front-end.

    Defaults to L2 (sklearn's own default), so ``penalty`` is left unset -- in
    sklearn >= 1.8 passing it explicitly raises a deprecation warning. Params
    (all optional):
      * ``C`` -- inverse regularisation strength (default 1.0); use a very large
        value (e.g. 1e9) to approximate an unregularised fit;
      * ``max_iter`` -- solver iterations (default 1000);
      * ``l1_ratio`` -- elastic-net mix, 0 = pure L2, 1 = pure L1 (needs a
        compatible ``solver``, e.g. 'saga'), forwarded only when given;
      * ``solver`` -- forwarded only when given;
      * ``class_weight`` -- ``None`` (default, every row weighted equally) or
        ``"balanced"`` (sklearn's inverse-frequency weighting -- reweights the
        log-loss so a 4:1 machine:human split, e.g. MG-GB/MG-Sci, does not
        just get more say because it has more rows).
    """

    name = "logistic"
    PARAMS = ("C", "max_iter", "l1_ratio", "solver", "class_weight")

    def resolved_params(self) -> dict:
        out = {"C": self.params.get("C", 1.0), "max_iter": self.params.get("max_iter", 1000),
              "class_weight": self.params.get("class_weight", None)}
        # Only reported when set, mirroring fit: sklearn's own defaults apply
        # otherwise and naming them here would misreport what ran.
        out.update({k: self.params[k] for k in ("l1_ratio", "solver") if k in self.params})
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticAlg":
        kwargs = dict(
            C=self.params.get("C", 1.0),
            max_iter=self.params.get("max_iter", 1000),
            class_weight=self.params.get("class_weight", None),
            random_state=self.seed,
        )
        # Forward only when set: the defaults already give L2, and naming
        # 'penalty' (or an incompatible solver) trips sklearn 1.8 deprecations.
        for key in ("l1_ratio", "solver"):
            if key in self.params:
                kwargs[key] = self.params[key]
        self.model = make_pipeline(StandardScaler(), LogisticRegression(**kwargs))
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

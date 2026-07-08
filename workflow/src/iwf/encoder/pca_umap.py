"""Linear / manifold / identity encoders - swap-in alternatives to the autoencoder.

PCA and UMAP reuse epai.utils.dimred_utils conceptually but keep a fitted, reusable
transform (epai's helper is fit_transform-only), so the SAME encoder maps train, test and
generated rows consistently.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

from workflow.src.iwf.encoder.base import Encoder


class PCAEncoder(Encoder):
    kind = "pca"

    def __init__(self, latent_dim: int, seed: int = 0):
        self.latent_dim = latent_dim
        self.seed = seed
        self.model_: PCA | None = None

    def fit(self, X: np.ndarray) -> "PCAEncoder":
        self.model_ = PCA(n_components=self.latent_dim, random_state=self.seed).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model_.transform(X).astype(np.float32)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_, path)

    def load(self, path: Path) -> "PCAEncoder":
        self.model_ = joblib.load(path)
        self.latent_dim = self.model_.n_components_
        return self


class UMAPEncoder(Encoder):
    kind = "umap"

    def __init__(self, latent_dim: int, seed: int = 0, n_neighbors: int = 15, min_dist: float = 0.1):
        self.latent_dim = latent_dim
        self.seed = seed
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.model_ = None

    def fit(self, X: np.ndarray) -> "UMAPEncoder":
        import umap

        self.model_ = umap.UMAP(
            n_components=self.latent_dim, random_state=self.seed,
            n_neighbors=self.n_neighbors, min_dist=self.min_dist,
        ).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.transform(X), dtype=np.float32)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_, path)

    def load(self, path: Path) -> "UMAPEncoder":
        self.model_ = joblib.load(path)
        return self


class IdentityEncoder(Encoder):
    """No reduction - passes the full embedding through (k* = input dim)."""

    kind = "identity"

    def __init__(self):
        self.latent_dim = -1

    def fit(self, X: np.ndarray) -> "IdentityEncoder":
        self.latent_dim = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"latent_dim": self.latent_dim}, path)

    def load(self, path: Path) -> "IdentityEncoder":
        self.latent_dim = joblib.load(path)["latent_dim"]
        return self

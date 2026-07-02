"""Encoder interface (plan section 4.3): swap the reducer with one config key.

All encoders share fit / transform / save / load. The autoencoder is UNSUPERVISED, so it
is fit on train+val (never test); see plan section 4.6.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Encoder(ABC):
    kind: str = "base"
    latent_dim: int

    @abstractmethod
    def fit(self, X: np.ndarray) -> "Encoder":
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @abstractmethod
    def load(self, path: Path) -> "Encoder":
        ...


def get_encoder(kind: str, latent_dim: int, input_dim: int | None = None,
                max_epochs: int = 200, seed: int = 42, activation: str = "gelu",
                patience: int = 25) -> Encoder:
    from workflow.src.iwf.encoder.autoencoder import AutoencoderEncoder, EpaiAutoencoderEncoder
    from workflow.src.iwf.encoder.pca_umap import PCAEncoder, UMAPEncoder, IdentityEncoder

    kind = (kind or "autoencoder").lower()
    if kind == "autoencoder":
        return AutoencoderEncoder(latent_dim=latent_dim, input_dim=input_dim,
                                  max_epochs=max_epochs, seed=seed, activation=activation,
                                  patience=patience)
    if kind == "epai_autoencoder":
        return EpaiAutoencoderEncoder(latent_dim=latent_dim, input_dim=input_dim,
                                      max_epochs=max_epochs, seed=seed, patience=patience)
    if kind == "pca":
        return PCAEncoder(latent_dim=latent_dim, seed=seed)
    if kind == "umap":
        return UMAPEncoder(latent_dim=latent_dim, seed=seed)
    if kind == "identity":
        return IdentityEncoder()
    raise ValueError(f"Unknown encoder kind {kind!r}.")


def load_saved_encoder(dataset_id: str) -> tuple["Encoder", dict]:
    """Reconstruct the frozen encoder saved by Phase 3 from its meta json + weights."""
    import json

    from workflow.src.iwf import paths

    meta = json.loads(paths.encoder_meta_path(dataset_id).read_text(encoding="utf-8"))
    enc = get_encoder(meta["kind"], latent_dim=meta["k_star"], input_dim=meta["input_dim"])
    enc.load(paths.encoder_path(dataset_id))
    return enc, meta

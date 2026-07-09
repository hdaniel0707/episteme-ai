"""Novelty from embedding geometry (plan section 4.5b; upgraded per plan §3.1).

Two label-free (leakage-safe) novelty variants over the reduced latent space Z:

  * spherical (default) — Shibayama-style geodesic novelty: points are L2-normalized
    onto the unit hypersphere and novelty_i = mean arccos(cosine sim) to the k nearest
    neighbours, percentile-ranked to a uniform [0,1] scale.
  * cosine_max (legacy)  — 1 - max cosine similarity to the k nearest neighbours.

``novelty_feasibility`` is kept for backward compatibility (legacy cosine geometry for
both values); new code should call ``compute_novelty`` + ``features.feasibility``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

NOVELTY_KINDS = ("spherical", "cosine_max")


def _knn_sims(Z: np.ndarray, k: int) -> np.ndarray:
    """Cosine similarities to the k nearest neighbours (self excluded), shape (n, k)."""
    Zn = normalize(Z)  # cosine similarity == dot product after L2 normalize
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1).fit(Zn)
    dist, _ = nn.kneighbors(Zn)
    return 1.0 - dist[:, 1:]  # drop self; cosine distance -> similarity


def percentile_01(values: np.ndarray) -> np.ndarray:
    """Rank-transform to a uniform [0,1] scale (ties averaged)."""
    v = np.asarray(values, dtype=np.float64)
    return (rankdata(v, method="average") / len(v)).astype(np.float32)


def novelty_spherical(Z: np.ndarray, k: int = 10) -> np.ndarray:
    """Geodesic-distance novelty in [0,1] (percentile of the mean arccos-sim to kNN)."""
    n = len(Z)
    if n <= 2:
        return np.zeros(n, dtype=np.float32)
    k = min(k, n - 1)
    sims = _knn_sims(np.asarray(Z, dtype=np.float32), k)
    geo = np.arccos(np.clip(sims, -1.0, 1.0))          # geodesic distance in [0, pi]
    return percentile_01(geo.mean(axis=1))


def novelty_cosine_max(Z: np.ndarray, k: int = 10) -> np.ndarray:
    """Legacy novelty = 1 - max cosine similarity to the kNN (raw scale)."""
    n = len(Z)
    if n <= 2:
        return np.zeros(n, dtype=np.float32)
    k = min(k, n - 1)
    sims = _knn_sims(np.asarray(Z, dtype=np.float32), k)
    return (1.0 - sims.max(axis=1)).astype(np.float32)


def compute_novelty(Z: np.ndarray, k: int = 10, kind: str = "spherical") -> np.ndarray:
    key = (kind or "spherical").strip().lower()
    if key == "spherical":
        return novelty_spherical(Z, k)
    if key == "cosine_max":
        return novelty_cosine_max(Z, k)
    raise ValueError(f"Unknown novelty_kind {kind!r}. Choose from {NOVELTY_KINDS}.")


def novelty_feasibility(Z: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Legacy pair: (1 - max-sim novelty, mean-sim density feasibility) — kept so older
    callers/tests keep working; the pipeline now uses compute_novelty + feasibility."""
    n = len(Z)
    if n <= 2:
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    k = min(k, n - 1)
    sims = _knn_sims(np.asarray(Z, dtype=np.float32), k)
    novelty = 1.0 - sims.max(axis=1)
    feasibility = sims.mean(axis=1)
    return novelty.astype(np.float32), feasibility.astype(np.float32)

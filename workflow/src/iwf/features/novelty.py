"""Novelty & feasibility from embedding geometry (plan section 4.5b).

Both are unsupervised (no labels), so they are leakage-safe and computed on all rows:
  * novelty     = 1 - max cosine similarity to the k nearest neighbours
                  (distinctiveness; larger => more unique direction).
  * feasibility = mean cosine similarity to the k nearest neighbours
                  (embeddedness/density; larger => sits in a well-populated region).
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def novelty_feasibility(Z: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return (novelty, feasibility) arrays for rows of Z (cosine-based)."""
    n = len(Z)
    if n <= 2:
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    k = min(k, n - 1)
    Zn = normalize(Z)  # cosine similarity == dot product after L2 normalize
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1).fit(Zn)
    dist, _ = nn.kneighbors(Zn)
    sims = 1.0 - dist[:, 1:]  # drop self; cosine distance -> similarity
    novelty = 1.0 - sims.max(axis=1)
    feasibility = sims.mean(axis=1)
    return novelty.astype(np.float32), feasibility.astype(np.float32)

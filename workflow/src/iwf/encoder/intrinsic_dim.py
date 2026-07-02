"""Intrinsic-dimension estimators (plan section 4.3): theoretical lower-bound anchors.

TwoNN (Facco et al. 2017) and the Levina-Bickel MLE (NIPS 2004). Two independent
estimators agreeing is a sanity check on the chosen latent size k.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def twonn(X: np.ndarray, discard_fraction: float = 0.1) -> float:
    """TwoNN estimate from the ratio of 2nd / 1st nearest-neighbour distances."""
    n = len(X)
    if n < 10:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=3, n_jobs=-1).fit(X)
    dist, _ = nn.kneighbors(X)
    r1, r2 = dist[:, 1], dist[:, 2]
    ok = r1 > 0
    mu = np.sort((r2[ok] / r1[ok]))
    mu = mu[mu > 1]
    m = len(mu)
    keep = int(m * (1 - discard_fraction))
    mu = mu[:keep]
    if len(mu) < 2:
        return float("nan")
    f = np.arange(1, len(mu) + 1) / m
    x = np.log(mu)
    y = -np.log(1 - f[: len(mu)])
    d = float(np.dot(x, y) / np.dot(x, x))  # slope through origin
    return d


def mle_levina_bickel(X: np.ndarray, k: int = 10) -> float:
    """Levina-Bickel maximum-likelihood intrinsic-dimension estimate."""
    n = len(X)
    if n <= k + 1:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
    dist, _ = nn.kneighbors(X)
    dist = dist[:, 1:]  # drop self
    # duplicate points give zero NN distance -> divide-by-zero; the inf/nan is sanitized
    # to 0.0 below, so just suppress the (expected) warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(dist[:, -1][:, None] / dist[:, :-1])
    logs = np.where(np.isfinite(logs), logs, 0.0)
    m_inv = logs.sum(axis=1) / (k - 1)
    m_inv = m_inv[m_inv > 0]
    if len(m_inv) == 0:
        return float("nan")
    return float(1.0 / np.mean(m_inv))


def estimate_intrinsic_dim(X: np.ndarray) -> dict:
    return {"twonn": twonn(X), "mle": mle_levina_bickel(X)}

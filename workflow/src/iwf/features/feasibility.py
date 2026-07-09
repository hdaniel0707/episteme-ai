"""Feasibility composite from embedding geometry + publication years (plan §3.2).

Meaning: how plausible it is that the paper's claims can be realized and tested. The
computable, label-free proxy follows the literature-supported-claim tradition
(Matter-of-Fact 2025; replicability-prediction work): claims close to well-established
prior work are feasible, claims far from any precedent are risky. Three components over
the reduced latent space Z (all percentile-ranked to [0,1] before weighting):

  * PRECEDENT (w=0.5) — mean cosine similarity to the k nearest neighbours among papers
                        STRICTLY OLDER than the paper ("prior art exists"). Papers of the
                        earliest year fall back to the global-kNN density value.
  * DENSITY   (w=0.3) — mean cosine similarity to the global k nearest neighbours
                        (a well-populated region = methods and a community exist).
  * MATURITY  (w=0.2) — fraction of the global kNN that is older than the paper
                        (building on settled ground rather than a co-emerging cluster).

The weighted sum is percentile-ranked again, so the final feasibility is uniform [0,1]
and comparable across corpora. Label-free ⇒ leakage-safe ⇒ computed on all rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from workflow.src.iwf.features.novelty import percentile_01

DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)


def feasibility(Z: np.ndarray, years=None, k: int = 10,
                weights: tuple[float, float, float] = DEFAULT_WEIGHTS) -> np.ndarray:
    """Composite feasibility in uniform [0,1] for the rows of Z (see module docstring).

    ``years`` may be None or contain NaN — those rows use density for the precedent
    component and 0.5 (neutral) for maturity.
    """
    n = len(Z)
    if n <= 2:
        return np.zeros(n, dtype=np.float32)
    k = int(min(k, n - 1))
    Zn = normalize(np.asarray(Z, dtype=np.float32))

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1).fit(Zn)
    dist, idx = nn.kneighbors(Zn)
    sims = 1.0 - dist[:, 1:]
    neigh = idx[:, 1:]
    density = sims.mean(axis=1)

    if years is not None:
        yr = pd.to_numeric(pd.Series(np.asarray(years).ravel()), errors="coerce"
                           ).to_numpy(dtype=np.float64)
    else:
        yr = np.full(n, np.nan)
    has_year = np.isfinite(yr)

    # MATURITY: share of global neighbours strictly older than the paper
    maturity = np.full(n, 0.5)
    if has_year.any():
        neigh_years = np.where(np.isfinite(yr), yr, np.inf)[neigh]  # unknown = "not older"
        older = neigh_years < yr[:, None]
        maturity[has_year] = older[has_year].mean(axis=1)

    # PRECEDENT: kNN similarity restricted to strictly-older papers. One kNN index per
    # distinct year (fitted on everything older), so the cost stays ~O(#years * n log n).
    precedent = density.copy()  # fallback for rows without a usable older pool
    if has_year.any():
        for y in np.unique(yr[has_year]):
            rows = np.where(has_year & (yr == y))[0]
            pool = np.where(has_year & (yr < y))[0]
            if len(pool) < 2:
                continue  # earliest year(s): keep the density fallback
            kk = min(k, len(pool))
            nn_old = NearestNeighbors(n_neighbors=kk, metric="cosine",
                                      n_jobs=-1).fit(Zn[pool])
            d_old, _ = nn_old.kneighbors(Zn[rows])
            precedent[rows] = (1.0 - d_old).mean(axis=1)

    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    composite = (w[0] * percentile_01(precedent) + w[1] * percentile_01(density)
                 + w[2] * percentile_01(maturity))
    return percentile_01(composite)

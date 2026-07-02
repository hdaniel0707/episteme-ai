"""Latent-dimension selection (plan section 4.3).

For autoencoder encoders the rule is RECONSTRUCTION-driven: train an AE at each candidate k
and pick the SMALLEST k whose reconstruction R^2 >= ``recon_target`` (default 0.98 = "98% of
the original embedding is recreated"). If no k reaches the target, keep the k with the best
reconstruction (so the dim does not drop when compression would lose too much).

For encoders without reconstruction (pca/umap) it falls back to the original probe rule:
smallest k whose downstream val metric is within a tolerance band of the full-embedding
baseline. The encoder is fit on train+val (unsupervised); the probe trains on train z and is
scored on val z (plan section 4.6).
"""

from __future__ import annotations

import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from workflow.src.iwf.encoder.base import get_encoder
from workflow.src.iwf.encoder.intrinsic_dim import estimate_intrinsic_dim
from workflow.src.iwf.parallel import run_jobs, torch_cuda_available


def _probe_njobs() -> int:
    """-1 (all cores) when run sequentially; the worker thread budget when run in parallel,
    so the probe's BLAS/OvR work doesn't oversubscribe the CPU across concurrent workers."""
    try:
        return int(os.getenv("WF_TORCH_THREADS", "") or 0) or -1
    except ValueError:
        return -1


def _probe_metric(z_tr, y_tr, z_va, y_va, task: str) -> float:
    sc = StandardScaler().fit(z_tr)
    z_tr, z_va = sc.transform(z_tr), sc.transform(z_va)
    if task == "clf":
        clf = LogisticRegression(max_iter=1000, n_jobs=_probe_njobs()).fit(z_tr, y_tr)
        return float(f1_score(y_va, clf.predict(z_va), average="macro"))
    reg = Ridge().fit(z_tr, y_tr)
    rho = spearmanr(y_va, reg.predict(z_va)).correlation
    return float(rho if rho == rho else 0.0)  # NaN -> 0


def _eval_one_dim(payload: dict) -> dict:
    """Fit the encoder at one candidate dim and score it. Module-level so it can run in a
    worker process; returns the curve entry {k, [recon_r2], metric}."""
    X = payload["X"]
    k = payload["k"]
    task = payload["task"]
    enc = get_encoder(payload["encoder_kind"], latent_dim=k, input_dim=X.shape[1],
                      max_epochs=payload["max_epochs"], seed=payload["seed"],
                      activation=payload["activation"], patience=payload["patience"])
    enc.fit(X[payload["fit_mask"]])
    entry: dict = {"k": k}
    if hasattr(enc, "reconstruction_stats"):
        entry["recon_r2"] = enc.reconstruction_stats(X[payload["fit_mask"]])["reconstruction_r2"]
    z_tr = enc.transform(X[payload["train_mask"]])
    z_va = enc.transform(X[payload["val_mask"]])
    entry["metric"] = _probe_metric(z_tr, payload["y_tr"], z_va, payload["y_va"], task)
    recon = f"recon_r2={entry['recon_r2']:.4f} " if "recon_r2" in entry else ""
    print(f"  [dim] k={k}: {recon}probe={entry['metric']:.4f}")
    return entry


def pca_variance_anchor(X: np.ndarray, retained: float = 0.99) -> int:
    from sklearn.decomposition import PCA

    p = PCA().fit(X)
    cum = np.cumsum(p.explained_variance_ratio_)
    return int(np.searchsorted(cum, retained) + 1)


def select_dim(
    X: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    y: np.ndarray,
    candidate_dims: list[int],
    task: str = "reg",
    encoder_kind: str = "autoencoder",
    tolerance: float = 0.01,
    max_epochs: int = 200,
    seed: int = 42,
    activation: str = "gelu",
    patience: int = 25,
    recon_target: float = 0.98,
) -> dict:
    """Run the full dim search and return a report dict including ``k_star``."""
    fit_mask = train_mask | val_mask  # encoder sees train+val (unsupervised)
    X_fit = X[fit_mask]
    y_tr, y_va = y[train_mask], y[val_mask]

    # Full-embedding baseline (no compression).
    baseline = _probe_metric(X[train_mask], y_tr, X[val_mask], y_va, task)

    anchors = estimate_intrinsic_dim(X_fit)
    try:
        anchors["pca_99"] = pca_variance_anchor(X_fit, 0.99)
    except Exception:
        anchors["pca_99"] = None

    input_dim = X.shape[1]
    dims = sorted({k for k in candidate_dims if 0 < k < input_dim})

    # Each candidate dim is an independent encoder fit + probe, so they can run in parallel.
    payloads = [
        {"X": X, "fit_mask": fit_mask, "train_mask": train_mask, "val_mask": val_mask,
         "y_tr": y_tr, "y_va": y_va, "k": k, "task": task, "encoder_kind": encoder_kind,
         "max_epochs": max_epochs, "seed": seed, "activation": activation, "patience": patience}
        for k in dims
    ]
    # autoencoders use the GPU (vram cap matters); pca/umap are CPU-only.
    use_cuda = torch_cuda_available() and encoder_kind in ("autoencoder", "epai_autoencoder")
    curve = [c for c in run_jobs(_eval_one_dim, payloads, use_cuda=use_cuda,
                                 vram_per_worker_mib=1600,
                                 label=f"{encoder_kind} dim search") if c is not None]
    have_recon = any("recon_r2" in c for c in curve)

    if have_recon:
        # smallest k that recreates >= recon_target of the embedding; else best reconstruction
        ok = [c for c in curve if c.get("recon_r2", -1.0) >= recon_target]
        if ok:
            k_star = min(c["k"] for c in ok)
        elif curve:
            k_star = max(curve, key=lambda c: c.get("recon_r2", -1.0))["k"]
        else:
            k_star = input_dim
        criterion = f"reconstruction_r2>={recon_target}"
    else:
        # probe rule: smallest k within tolerance of the full-embedding baseline
        within = [c for c in curve if c["metric"] >= baseline - tolerance]
        if any(c["metric"] > baseline for c in curve):
            k_star = max(curve, key=lambda c: c["metric"])["k"]
        elif within:
            k_star = min(c["k"] for c in within)
        elif curve:
            k_star = max(curve, key=lambda c: c["metric"])["k"]
        else:
            k_star = min(dims) if dims else input_dim
        criterion = f"probe_within_{tolerance}_of_baseline"

    return {
        "task": task,
        "criterion": criterion,
        "baseline_metric": baseline,
        "recon_target": recon_target,
        "tolerance": tolerance,
        "anchors": anchors,
        "curve": curve,
        "k_star": int(k_star),
    }

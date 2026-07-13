"""Parquet IO + the ONE shared train/val/test split (plan section 4.6).

The split is computed once (seeded, stratified) in main_prepare and stored as a
``split`` column; every later stage only ever reads it back through ``split_masks``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_COL = "split"
SPLIT_NAMES = ("train", "val", "test")


def load_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet not found: {path} — run the earlier phase first.")
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[dataset] saved {len(df)} rows -> {path}")
    return path


def make_split(n: int, seed: int, ratios: list[float],
               stratify: np.ndarray | None = None) -> np.ndarray:
    """Return an (n,) array of 'train'/'val'/'test' labels.

    Stratified per class when ``stratify`` is given; a stratum too small to receive a
    deterministic share still gets its rows assigned via the shuffled global remainder,
    so degenerate label distributions never crash (they just fall back gracefully).
    """
    ratios = [float(r) for r in ratios]
    if len(ratios) != 3:
        raise ValueError(f"split_ratios must have 3 entries (train/val/test), got {ratios}.")
    total = sum(ratios)
    ratios = [r / total for r in ratios]

    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=object)

    if stratify is not None and len(stratify) == n:
        strata = pd.Series(np.asarray(stratify, dtype=object)).fillna("__nan__").to_numpy()
        groups = pd.unique(strata)
    else:
        strata = np.zeros(n, dtype=object)
        groups = np.array([0], dtype=object)

    for g in groups:
        idx = np.where(strata == g)[0]
        idx = idx[rng.permutation(len(idx))]
        m = len(idx)
        n_train = int(round(m * ratios[0]))
        n_val = int(round(m * ratios[1]))
        # keep the counts consistent for tiny strata
        n_train = min(n_train, m)
        n_val = min(n_val, m - n_train)
        out[idx[:n_train]] = "train"
        out[idx[n_train:n_train + n_val]] = "val"
        out[idx[n_train + n_val:]] = "test"

    # tiny-corpus safety: guarantee every split name exists when n allows it
    if n >= 3:
        for name in SPLIT_NAMES:
            if not (out == name).any():
                donor = "train" if name != "train" else "test"
                cand = np.where(out == donor)[0]
                if len(cand) > 1:
                    out[cand[-1]] = name
    return out.astype(str)


def add_split_column(df: pd.DataFrame, seed: int, ratios: list[float],
                     stratify_col: str | None = None, domain_holdout: str = "") -> pd.DataFrame:
    """Return a copy of ``df`` with the seeded (optionally stratified) ``split`` column.

    If domain_holdout is set, that domain is excluded from training set.
    """
    df = df.copy()
    stratify = None
    if stratify_col and stratify_col in df.columns:
        stratify = df[stratify_col].to_numpy()
    df[SPLIT_COL] = make_split(len(df), seed, ratios, stratify=stratify)

    # LODO: exclude domain_holdout from training
    if domain_holdout and "domains" in df.columns:
        from workflow.src.iwf.evaluation import plots
        domains = plots._get_domain(df)
        holdout_mask = domains == domain_holdout
        train_mask = df[SPLIT_COL] == "train"
        n_moved = (holdout_mask & train_mask).sum()
        df.loc[holdout_mask & train_mask, SPLIT_COL] = "test"
        if n_moved > 0:
            print(f"[dataset] LODO: moved {n_moved} papers from domain '{domain_holdout}' "
                  f"out of training set to test")

    return df


def split_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks {'train','val','test'} over the rows of ``df``."""
    if SPLIT_COL not in df.columns:
        raise KeyError("No 'split' column — run main_prepare first.")
    col = df[SPLIT_COL].astype(str).to_numpy()
    return {name: col == name for name in SPLIT_NAMES}

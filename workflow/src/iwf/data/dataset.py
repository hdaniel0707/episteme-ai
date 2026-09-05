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


def _sanitize_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all-numeric object columns (mixed bool/int) to float for pyarrow."""
    for col in df.columns:
        if df[col].dtype != object:
            continue
        nonnull = df[col].dropna()
        if nonnull.empty:
            continue
        if all(isinstance(v, (bool, int, float)) and not isinstance(v, str)
               for v in nonnull):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _sanitize_object_columns(df)
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


def _force_exact_counts(labels: np.ndarray, wanted: dict[str, int], seed: int) -> None:
    """Trade rows with 'train' in place until each split has exactly its wanted count.

    make_split allocates per stratum and rounds, so with 3+ strata the holdout can land a
    row or two off (e.g. 248/252 instead of 250/250). Corrections follow one seeded order,
    so the result stays identical across calls with the same n/seed/stratify.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    for name, want in wanted.items():
        have = [i for i in order if labels[i] == name]
        if len(have) > want:
            for i in have[want:]:
                labels[i] = "train"
        elif len(have) < want:
            spare = [i for i in order if labels[i] == "train"]
            for i in spare[:want - len(have)]:
                labels[i] = name


def make_fixed_holdout_split(n: int, val_n: int, test_n: int, train_n: int,
                             holdout_seed: int, train_select_seed: int,
                             stratify: np.ndarray | None = None) -> np.ndarray:
    """Return an (n,) 'train'/'val'/'test'/'unused' label array where val/test are a FIXED
    holdout of exactly val_n/test_n rows, chosen the same way every call given the same
    n/holdout_seed/stratify - so a data-size sweep (train_n varying, val_n/test_n constant)
    always evaluates on the identical held-out rows. train is a seeded prefix of size
    train_n from whatever remains (train_n<=0 = use everything remaining); the SAME
    train_select_seed reused across a sweep makes a smaller train_n a strict subset of a
    larger one. Rows past train_n are labelled 'unused' (excluded from every split)."""
    if val_n < 0 or test_n < 0 or val_n + test_n > n:
        raise ValueError(f"val_n+test_n ({val_n + test_n}) must be between 0 and n ({n}).")
    holdout_ratios = [max(0, n - val_n - test_n) / n, val_n / n, test_n / n]
    # object dtype, not make_split's fixed-width <U5 - "unused" would else be truncated
    labels = make_split(n, holdout_seed, holdout_ratios, stratify=stratify).astype(object)
    _force_exact_counts(labels, {"val": val_n, "test": test_n}, holdout_seed)

    remaining_idx = np.where(labels == "train")[0]
    rng = np.random.default_rng(train_select_seed)
    order = remaining_idx[rng.permutation(len(remaining_idx))]
    keep_n = len(order) if train_n <= 0 else min(int(train_n), len(order))
    labels[order[keep_n:]] = "unused"
    return labels.astype(str)


def add_split_column(df: pd.DataFrame, seed: int, ratios: list[float],
                     stratify_col: str | None = None, domain_holdout: str = "",
                     val_n: int = 0, test_n: int = 0, train_n: int = 0,
                     holdout_seed: int = 0, train_select_seed: int = 0) -> pd.DataFrame:
    """Return a copy of ``df`` with the seeded (optionally stratified) ``split`` column.

    Default behaviour (unchanged): proportional ``ratios`` split. When ``val_n`` AND
    ``test_n`` are both > 0, uses the FIXED-HOLDOUT strategy instead (see
    make_fixed_holdout_split) so val/test stay identical across a train-size sweep.

    If domain_holdout is set, that domain is excluded from training set.
    """
    df = df.copy()
    stratify = None
    if stratify_col and stratify_col in df.columns:
        stratify = df[stratify_col].to_numpy()
    if val_n > 0 and test_n > 0:
        df[SPLIT_COL] = make_fixed_holdout_split(
            len(df), val_n, test_n, train_n, holdout_seed or seed,
            train_select_seed or seed, stratify=stratify)
    else:
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

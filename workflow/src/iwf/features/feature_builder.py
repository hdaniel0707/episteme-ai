"""Assemble model inputs and own the neutral-zero + mask contract (plan section 6.1).

Input layout (columns):  [ z_0..z_{k-1}, <scalar features...>, <scalar>_missing... ]
  * z is the reduced latent vector (never missing).
  * scalar features = metadata + novelty/feasibility + external-impact scores.
  * for every scalar feature an extra binary ``*_missing`` column is appended.

NeutralScaler standardizes the VALUE block on TRAIN only (nan-aware) so that 0.0 is the
neutral mean, then fills any missing value with 0.0. The mask block passes through
unscaled (0/1). This keeps every stage runnable on incomplete rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Z_COL = "z"

# Candidate scalar feature columns, in a fixed order (only those present are used).
# meta_year is intentionally absent: year is a target input, not a feature (see below).
METADATA_COLS = ["meta_h_index", "meta_author_count", "meta_venue_freq"]
EXTERNAL_COLS = ["naip", "forecite", "cited_potential"]

# Leakage guard (plan §4.5): every impact target is a deterministic function of these
# columns, so feeding any of them as a feature would let the model reconstruct the target
# and bypass the title+abstract embedding entirely. They are NEVER used as model inputs.
RESERVED_TARGET_INPUTS = {
    "citation_count", "influential_citation_count", "year", "oa_year",
    "domains", "domain", "subfield",
    "fwci", "citation_normalized_percentile",
    "is_in_top_1_percent", "is_in_top_10_percent",
    "meta_year",
}


def _is_reserved(col: str) -> bool:
    return (
        col in RESERVED_TARGET_INPUTS
        or col == "target"
        or col == "strat_label"
        or col.startswith("target_")
    )


def drop_reserved(cols: list[str], bypass_guard: bool = False) -> list[str]:
    """Strip reserved target-input columns from a feature list (with a warning).

    When ``bypass_guard=True`` the check is skipped — used by the emb_all_meta_* workflows
    that intentionally include leakage data to measure the performance upper bound.
    """
    if bypass_guard:
        return list(cols)
    kept = [c for c in cols if not _is_reserved(c)]
    dropped = [c for c in cols if _is_reserved(c)]
    if dropped:
        print(f"[feature_builder] leakage guard - excluded target-input columns: {dropped}")
    return kept


def all_scalar_columns(df: pd.DataFrame) -> list[str]:
    """All numeric scalar columns in ``df`` that are not the target / label / z / split.

    Used by the emb_all_meta_* workflows: includes citation_count, year, fwci etc.
    (the leakage guard is NOT applied — this is intentional for upper-bound testing).
    """
    ALWAYS_EXCLUDE = {
        "target", "strat_label", "class_label", "class_label_id",
        Z_COL, "split", "paper_id", "doi", "pmid", "corpus_id",
        "title", "abstract", "authors",
    }
    EXCLUDE_PREFIXES = ("specter2_", "scibert_", "scincl_", "e5_")
    result = []
    for col in df.columns:
        if col in ALWAYS_EXCLUDE:
            continue
        if any(col.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        try:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                result.append(col)
        except Exception:
            continue
    return result


def scalar_feature_columns(
    df: pd.DataFrame, use_metadata: bool, use_novelty: bool,
    use_feasibility: bool, external: list[str],
) -> list[str]:
    """Which scalar feature columns to fuse (those present in df), in fixed order."""
    cols: list[str] = []
    if use_metadata:
        cols += [c for c in METADATA_COLS if c in df.columns]
    if use_novelty and "novelty" in df.columns:
        cols.append("novelty")
    if use_feasibility and "feasibility" in df.columns:
        cols.append("feasibility")
    cols += [c for c in EXTERNAL_COLS if c in external and c in df.columns]
    return drop_reserved(cols)


def _z_matrix(df: pd.DataFrame) -> np.ndarray:
    if Z_COL not in df.columns:
        raise KeyError("Missing 'z' column - run main_reduce first.")
    return np.array(df[Z_COL].tolist(), dtype=np.float32)


def build_xy(
    df: pd.DataFrame,
    task: str,
    scalar_cols: list[str] | None = None,
    target_col: str = "target",
    label_col: str = "class_label_id",
    bypass_guard: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, list[str], int]:
    """Return (X, y, feature_names, n_value_cols).

    ``n_value_cols`` is the number of leading columns NeutralScaler should standardize
    (z dims + scalar features); the remaining trailing columns are the 0/1 mask bits.
    y is None when neither target nor label column is present (e.g. generated papers).
    ``bypass_guard``: when True, reserved target-input columns are not filtered out.
    """
    scalar_cols = drop_reserved(scalar_cols or [], bypass_guard=bypass_guard)
    Z = _z_matrix(df)
    names = [f"z[{i}]" for i in range(Z.shape[1])]
    blocks = [Z]
    mask_blocks = []

    for col in scalar_cols:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        blocks.append(vals.reshape(-1, 1))
        mask_blocks.append(np.isnan(vals).astype(np.float32).reshape(-1, 1))
        names.append(col)

    n_value_cols = Z.shape[1] + len(scalar_cols)
    if mask_blocks:
        names += [f"{c}_missing" for c in scalar_cols]
        X = np.hstack(blocks + mask_blocks).astype(np.float32)
    else:
        X = np.hstack(blocks).astype(np.float32)

    y = None
    if task == "clf" and label_col in df.columns:
        y = df[label_col].to_numpy()
    elif task == "reg" and target_col in df.columns:
        y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float32)
    return X, y, names, n_value_cols


class NeutralScaler:
    """Nan-aware standardizer: value block -> z-score with NaN->0; mask block passthrough."""

    def __init__(self, n_value_cols: int):
        self.n_value_cols = n_value_cols
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "NeutralScaler":
        v = X[:, : self.n_value_cols].astype(np.float64)
        self.mean_ = np.nanmean(v, axis=0)
        std = np.nanstd(v, axis=0)
        std[std == 0] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32).copy()
        v = X[:, : self.n_value_cols].astype(np.float64)
        v = (v - self.mean_) / self.std_
        v = np.nan_to_num(v, nan=0.0)  # missing -> neutral 0
        X[:, : self.n_value_cols] = v.astype(np.float32)
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

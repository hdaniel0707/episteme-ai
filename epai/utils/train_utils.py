import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm
from epai.utils.config_utils import YELLOW, RESET

CLF_TARGET_COL = "generated_by"


def is_missing(value) -> bool:
    """True if a cell is null, empty, or holds a non-finite (NaN/inf) vector entry."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return True
        return arr.size == 0 or not np.isfinite(arr).all()
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def drop_missing_rows(
    df: pd.DataFrame, columns: list[str], group_col: str | None = None
) -> pd.DataFrame:
    """Return a copy of df without rows missing any of `columns`.

    With `group_col`, a missing value takes down its whole group rather than the
    single offending row. In the Ghostbuster corpora a handful of human source
    documents are empty files, so they have no embedding; dropping only those
    rows would leave their machine rewrites behind as orphaned half-groups and
    skew the class balance. Dropping the whole group keeps every group intact
    and the classes even.

    Operates purely in memory: the parquet file on disk is never rewritten.
    """
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in dataframe: {missing_cols}")
    if group_col is not None and group_col not in df.columns:
        raise KeyError(f"group_col {group_col!r} not found in dataframe.")

    mask = pd.Series(False, index=df.index)
    per_col: dict[str, int] = {}
    for col in columns:
        col_mask = df[col].map(is_missing)
        n_col = int(col_mask.sum())
        if n_col:
            per_col[col] = n_col
        mask |= col_mask

    n_rows = int(mask.sum())
    if n_rows and group_col is not None:
        bad_groups = df.loc[mask, group_col].unique()
        mask = mask | df[group_col].isin(bad_groups)

    n_dropped = int(mask.sum())
    if n_dropped:
        detail = ", ".join(f"{c}={n}" for c, n in per_col.items())
        extra = ""
        if group_col is not None and n_dropped > n_rows:
            extra = (
                f" — {n_rows} row(s) were missing, dropping {n_dropped} to keep "
                f"{group_col!r} groups intact"
            )
        tqdm.write(
            f"{YELLOW}  ⚠️  Skipping {n_dropped}/{len(df)} rows with missing values "
            f"({detail}){extra}. The parquet file is left untouched.{RESET}"
        )

    clean = df[~mask].reset_index(drop=True)
    if clean.empty:
        raise ValueError(
            f"All {len(df)} rows were dropped because of missing values in {columns}."
        )
    return clean


def stack_features(df, features: list[str]) -> tuple[np.ndarray, list[str]]:
    """Horizontally stack every column in `features` into one design matrix.

    Vector-valued columns (e.g. embeddings) contribute one dimension per entry;
    scalar columns contribute a single column. Column order follows `features`.
    """
    if not features:
        raise ValueError("No features given.")

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Feature columns not found in dataframe: {missing}")

    parts: list[np.ndarray] = []
    labels: list[str] = []
    for col in features:
        sample = df[col].iloc[0]
        arr = np.array(df[col].tolist(), dtype=np.float32)
        if hasattr(sample, "__len__"):
            parts.append(arr)
            labels.extend(f"{col}[{i}]" for i in range(len(sample)))
        else:
            parts.append(arr.reshape(-1, 1))
            labels.append(col)
    return np.hstack(parts), labels


def build_features(df, features: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, labels = stack_features(df, features)
    y = np.log1p(np.array(df["citation_count"].tolist(), dtype=np.float32))
    return X, y, labels


def build_model(ml_alg: str, base_seed: int):
    if ml_alg == "gradient_boosting":
        return GradientBoostingRegressor(random_state=base_seed)
    if ml_alg == "linear":
        return LinearRegression()
    if ml_alg == "svm":
        return SVR()
    if ml_alg == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(516, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            warm_start=True,
            random_state=base_seed,
        )
    raise ValueError(
        f"Unknown ml_alg: {ml_alg!r}. Choose from: gradient_boosting, linear, svm, mlp"
    )


def build_clf_features(
    df,
    features: list[str],
    drop_classes: list[str] | None = None,
    group_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, np.ndarray | None, list[str]]:
    """Build the classification design matrix, targets, and (optional) split groups.

    Every column in `features` is used, not just the first one. When `group_col`
    is given, its values are returned as `groups` so that rows sharing a group
    (e.g. the human original and every LLM rewrite of the same source document,
    which share one `id`) can be kept in the same train/val/test bucket.
    """
    if drop_classes:
        df = df[~df[CLF_TARGET_COL].isin(drop_classes)].reset_index(drop=True)
        tqdm.write(f"  Dropped classes: {drop_classes}")

    X, labels = stack_features(df, features)
    le = LabelEncoder()
    y = le.fit_transform(df[CLF_TARGET_COL].tolist())

    groups = None
    if group_col is not None:
        if group_col not in df.columns:
            raise KeyError(
                f"group_col {group_col!r} not found in dataframe. "
                f"Available columns: {list(df.columns)}"
            )
        groups = df[group_col].to_numpy()

    return X, y, le, groups, labels


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    groups: np.ndarray | None = None,
    test_size: float = 0.30,
) -> tuple[np.ndarray, ...]:
    """Split into train / val / test (70 / 15 / 15 by default).

    With `groups`, a GroupShuffleSplit keeps every row of a group on one side of
    each cut, so a document and its rewrites can never straddle train and test.
    Without `groups`, the split is a plain stratified random split of the rows —
    which leaks whenever rows are paraphrases of one another.

    Note that grouping and stratification cannot both be enforced exactly; the
    grouped path therefore does not stratify, and the caller should check the
    reported per-split class balance.
    """
    if groups is None:
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, tmp_idx = next(outer.split(X, y, groups))

    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    rel_val_idx, rel_test_idx = next(
        inner.split(X[tmp_idx], y[tmp_idx], groups[tmp_idx])
    )
    val_idx, test_idx = tmp_idx[rel_val_idx], tmp_idx[rel_test_idx]

    g_train, g_val, g_test = (
        set(groups[train_idx]),
        set(groups[val_idx]),
        set(groups[test_idx]),
    )
    overlap = (g_train & g_val) | (g_train & g_test) | (g_val & g_test)
    if overlap:
        raise AssertionError(
            f"Group leakage: {len(overlap)} group(s) appear in more than one split, "
            f"e.g. {sorted(overlap)[:5]}"
        )

    return (
        X[train_idx],
        X[val_idx],
        X[test_idx],
        y[train_idx],
        y[val_idx],
        y[test_idx],
    )


# The classifier factory that used to live here (`build_clf_model`, selected by a
# config's `model.ml_alg`) is gone. Classifiers are now `ClfAlgorithm` subclasses
# in `epai/algs`, built by `build_algorithm(name, seed, **params)`: they own their
# own preprocessing and declare which hyper-parameters they read, so a config
# either takes effect or fails in preflight instead of silently training a
# default model. `logistic`, `svm` and `gradient_boosting` carry the same
# defaults the factory did; the sklearn `mlp` it also offered was unused and was
# dropped rather than ported (see `simple_mlp` for the torch replacement).

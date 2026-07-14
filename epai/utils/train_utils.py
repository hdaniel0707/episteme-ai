import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm
from epai.train.autoencoder import AERegressor
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


def drop_missing_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy of df without rows missing any of `columns`.

    Operates purely in memory: the parquet file on disk is never rewritten.
    """
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in dataframe: {missing_cols}")

    mask = pd.Series(False, index=df.index)
    per_col: dict[str, int] = {}
    for col in columns:
        col_mask = df[col].map(is_missing)
        n_col = int(col_mask.sum())
        if n_col:
            per_col[col] = n_col
        mask |= col_mask

    n_dropped = int(mask.sum())
    if n_dropped:
        detail = ", ".join(f"{c}={n}" for c, n in per_col.items())
        tqdm.write(
            f"{YELLOW}  ⚠️  Skipping {n_dropped}/{len(df)} rows with missing values "
            f"({detail}). The parquet file is left untouched.{RESET}"
        )

    clean = df[~mask].reset_index(drop=True)
    if clean.empty:
        raise ValueError(
            f"All {len(df)} rows were dropped because of missing values in {columns}."
        )
    return clean


def build_features(df, features: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
    X = np.hstack(parts)
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
    if ml_alg == "ae":
        return AERegressor(random_state=base_seed)
    raise ValueError(
        f"Unknown ml_alg: {ml_alg!r}. Choose from: gradient_boosting, linear, svm, mlp"
    )


def build_clf_features(
    df, features: list[str], drop_classes: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    if drop_classes:
        df = df[~df[CLF_TARGET_COL].isin(drop_classes)].reset_index(drop=True)
        tqdm.write(f"  Dropped classes: {drop_classes}")
    X = np.array(df[features[0]].tolist(), dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df[CLF_TARGET_COL].tolist())
    return X, y, le


def build_clf_model(ml_alg: str, seed: int):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    if ml_alg == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    if ml_alg == "logistic":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if ml_alg == "svm":
        return SVC(random_state=seed)
    if ml_alg == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(516, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=seed,
        )
    raise ValueError(
        f"Unknown ml_alg: {ml_alg!r}. Choose from: gradient_boosting, logistic, svm, mlp"
    )

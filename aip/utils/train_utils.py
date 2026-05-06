import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


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
            max_iter=100,
            warm_start=True,
            random_state=base_seed,
        )
    raise ValueError(
        f"Unknown ml_alg: {ml_alg!r}. Choose from: gradient_boosting, linear, svm, mlp"
    )

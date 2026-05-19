import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from aip.autoencoder import AERegressor


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
        df = df[~df["generated_by"].isin(drop_classes)].reset_index(drop=True)
        print(f"Dropped classes: {drop_classes}")
    X = np.array(df[features[0]].tolist(), dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["generated_by"].tolist())
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

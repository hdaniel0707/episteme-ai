import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from utils.parquet_utils import load_parquet

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "data"
PAQUET_PATH = DB_DIR / "paquet" / "0505_A"
RUN_ID = "0505_A4"
MODELS_DIR = BASE_DIR / "results" / RUN_ID / "models"
METRICS_DIR = BASE_DIR / "results" / RUN_ID / "metrics"

N_ESTIMATORS = 300
LEARNING_RATE = 0.05
MAX_DEPTH = 4
SUBSAMPLE = 0.8
SEED = 42
N_RUNS = 3


def build_features(
    df, features: list[str] = ["embedding"]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    parts = []
    labels = []
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


def train_run(X, y, seed: int):
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    train_losses = [
        mean_squared_error(y_train, p) for p in model.staged_predict(X_train)
    ]
    val_losses = [mean_squared_error(y_val, p) for p in model.staged_predict(X_val)]

    metrics = {}
    for split, X_s, y_s in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_pred = model.predict(X_s)
        metrics[f"{split}_rmse"] = np.sqrt(mean_squared_error(y_s, y_pred))
        metrics[f"{split}_r2"] = r2_score(y_s, y_pred)

    metrics["train_losses"] = [float(v) for v in train_losses]
    metrics["val_losses"] = [float(v) for v in val_losses]
    metrics["y_test"] = y_test.tolist()
    metrics["y_pred"] = model.predict(X_test).tolist()
    metrics["feature_importances"] = model.feature_importances_.tolist()

    return (
        model,
        scaler,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        train_losses,
        val_losses,
        metrics,
    )


if __name__ == "__main__":
    df = load_parquet(PAQUET_PATH)

    # FEATURES = ["embedding", "year", "best_author_h_index"]
    FEATURES = ["best_author_h_index"]
    X, y, feature_labels = build_features(df, FEATURES)
    print(f"X: {X.shape}  y: {y.shape}")

    all_metrics = []
    model = scaler = train_losses = val_losses = X_test = y_test = None
    for run in range(N_RUNS):
        seed = SEED + run
        (
            model,
            scaler,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            train_losses,
            val_losses,
            metrics,
        ) = train_run(X, y, seed)
        metrics["seed"] = seed
        metrics["feature_labels"] = feature_labels
        all_metrics.append(metrics)
        print(
            f"Run {run+1:2d}/{N_RUNS}  seed={seed}  train RMSE={metrics['train_rmse']:.4f}  R²={metrics['train_r2']:.4f}  val RMSE={metrics['val_rmse']:.4f}  R²={metrics['val_r2']:.4f}  test RMSE={metrics['test_rmse']:.4f}  R²={metrics['test_r2']:.4f}"
        )

    metric_keys = [
        "train_rmse",
        "train_r2",
        "val_rmse",
        "val_r2",
        "test_rmse",
        "test_r2",
    ]
    summary = {}
    print("\n--- Summary over {} runs ---".format(N_RUNS))
    for key in metric_keys:
        vals = np.array([m[key] for m in all_metrics])
        q1, q3 = np.percentile(vals, [25, 75])
        iqm_mask = (vals >= q1) & (vals <= q3)
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "iqm": float(np.mean(vals[iqm_mask])),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
        print(
            f"{key:12s}  mean={summary[key]['mean']:.4f}  std={summary[key]['std']:.4f}  min={summary[key]['min']:.4f}  max={summary[key]['max']:.4f}"
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "runs.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(METRICS_DIR / "summary.json", "w") as f:
        json.dump({"n_runs": N_RUNS, "metrics": summary}, f, indent=2)
    print(f"\nMetrics saved to {METRICS_DIR}")

    # Save model + scaler from the last run
    if model is not None and N_RUNS > 0:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS_DIR / "model.joblib")
        joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
        print(f"Model saved to {MODELS_DIR}")

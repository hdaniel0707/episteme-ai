from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from utils.parquet_utils import load_parquet

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "data"
PAQUET_PATH = DB_DIR / "paquet" / "0504_B"
FIG_DIR = BASE_DIR / "results" / "figs" / "0504_B"

N_ESTIMATORS = 300
LEARNING_RATE = 0.05
MAX_DEPTH = 4
SUBSAMPLE = 0.8
SEED = 42


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


def plot_loss_curves(train_losses, val_losses, out: Path) -> None:
    iters = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, train_losses, label="Train MSE")
    ax.plot(iters, val_losses, label="Val MSE")
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("MSE  (log1p scale)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_actual_vs_predicted(y_true, y_pred, split: str, out: Path) -> None:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.4)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual  log1p(citations)")
    ax.set_ylabel("Predicted  log1p(citations)")
    ax.set_title(f"{split}  RMSE={rmse:.3f}  R²={r2:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_feature_importances(
    importances: np.ndarray, labels: list[str], top_k: int, out: Path
) -> None:
    top_k = min(top_k, len(importances))
    top_idx = np.argsort(importances)[-top_k:][::-1]
    top_labels = [labels[i] for i in top_idx]
    top_values = importances[top_idx]

    groups = [lbl.split("[")[0] for lbl in top_labels]
    unique_groups = list(dict.fromkeys(groups))
    palette = ["steelblue", "tomato", "seagreen", "mediumpurple", "darkorange"]
    group_color = {g: palette[i % len(palette)] for i, g in enumerate(unique_groups)}
    colors = [group_color[g] for g in groups]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(top_k), top_values, color=colors)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(top_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Feature importance")
    ax.set_title(f"Top {top_k} features by importance")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=group_color[g], label=g) for g in unique_groups])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    df = load_parquet(PAQUET_PATH)

    FEATURES = ["embedding"]
    X, y, feature_labels = build_features(df, FEATURES)
    print(f"X: {X.shape}  y: {y.shape}")

    # 70 / 15 / 15 split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED
    )
    print(f"Train {len(X_train)}  Val {len(X_val)}  Test {len(X_test)}")

    # Standardise (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train
    model = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        random_state=SEED,
    )
    model.fit(X_train, y_train)

    # Loss curves via staged predictions
    train_losses = [
        mean_squared_error(y_train, p) for p in model.staged_predict(X_train)
    ]
    val_losses = [mean_squared_error(y_val, p) for p in model.staged_predict(X_val)]

    # Final metrics
    for split, X_s, y_s in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        y_pred = model.predict(X_s)
        rmse = np.sqrt(mean_squared_error(y_s, y_pred))
        r2 = r2_score(y_s, y_pred)
        print(f"{split:4s}  RMSE={rmse:.4f}  R²={r2:.4f}")

    # Plots
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_loss_curves(
        train_losses,
        val_losses,
        FIG_DIR / "loss_curves.png",
    )
    plot_actual_vs_predicted(
        y_test,
        model.predict(X_test),
        "Test set",
        FIG_DIR / "actual_vs_predicted.png",
    )
    plot_feature_importances(
        model.feature_importances_,
        feature_labels,
        20,
        FIG_DIR / "feature_importances.png",
    )

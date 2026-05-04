"""
Citation count prediction pipeline.

Steps:
  1. Load all *.db (SQLite) files → merge into one DataFrame → save as Parquet
  2. Engineer features: SPECTER embeddings (PCA-reduced) + paper metadata
  3. Train a Random Forest regressor; evaluate with cross-validation + hold-out split
  4. Save result figures to a results directory and the trained pipeline to disk
"""

import argparse
import glob
import json
import sqlite3
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── Constants ────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 768
DEFAULT_N_PCA = 32   # reduced before RF; capped to fold train size at runtime
META_COLS = ["year", "author_count", "abstract_length", "title_length"]

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


# ── 1. Data loading ──────────────────────────────────────────────────────────

def load_db(path: str | Path) -> pd.DataFrame:
    """Load papers + per-paper author count from one SQLite database file."""
    conn = sqlite3.connect(path)
    papers = pd.read_sql(
        """
        SELECT paper_id, title, abstract, embedding,
               citation_count, influential_citation_count, year, venue
        FROM papers
        """,
        conn,
    )
    author_counts = pd.read_sql(
        "SELECT paper_id, COUNT(*) AS author_count FROM paper_authors GROUP BY paper_id",
        conn,
    )
    conn.close()
    df = papers.merge(author_counts, on="paper_id", how="left")
    df["source_db"] = Path(path).stem
    return df


def load_all_dbs(pattern: str = "*.db") -> pd.DataFrame:
    """Glob for .db files, load each one, and concatenate."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No .db files found matching '{pattern}'")
    frames = [load_db(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} papers from {len(paths)} file(s): {paths}")
    return df


def save_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows → {path}")


def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows ← {path}")
    return df


# ── 2. Feature engineering ───────────────────────────────────────────────────

def parse_embeddings(series: pd.Series) -> np.ndarray:
    """Decode JSON-serialised embedding strings to a (n, 768) float32 matrix."""
    vectors = [
        json.loads(v) if isinstance(v, str) else [0.0] * EMBEDDING_DIM
        for v in series
    ]
    return np.array(vectors, dtype=np.float32)


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, y):
      X — (n_samples, EMBEDDING_DIM + len(META_COLS)) float array
      y — (n_samples,) citation_count float array
    """
    emb = parse_embeddings(df["embedding"])
    meta = np.column_stack([
        df["year"].fillna(0).astype(float).values,
        df["author_count"].fillna(0).astype(float).values,
        df["abstract"].fillna("").str.len().astype(float).values,
        df["title"].fillna("").str.len().astype(float).values,
    ])
    X = np.hstack([emb, meta])
    y = df["citation_count"].values.astype(float)
    return X, y


# ── 3. Pipeline ──────────────────────────────────────────────────────────────

def build_pipeline(n_pca: int) -> Pipeline:
    """
    ColumnTransformer applies PCA+scaling to the embedding block and standard
    scaling to the metadata block, then feeds everything into Random Forest.
    """
    emb_idx  = list(range(EMBEDDING_DIM))
    meta_idx = list(range(EMBEDDING_DIM, EMBEDDING_DIM + len(META_COLS)))
    preprocessor = ColumnTransformer([
        ("emb_pca", Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=n_pca, random_state=42)),
        ]), emb_idx),
        ("meta_scale", StandardScaler(), meta_idx),
    ])
    return Pipeline([
        ("prep",  preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])


# ── 4. Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2":   r2_score(y_true, y_pred),
    }


def print_metrics(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    m = compute_metrics(y_true, y_pred)
    print(f"\n{label}")
    print(f"  MAE  : {m['MAE']:.2f}")
    print(f"  RMSE : {m['RMSE']:.2f}")
    print(f"  R²   : {m['R2']:.3f}")


# ── 5. Figures ───────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_oob_curve(prep, X_tr: np.ndarray, y_tr: np.ndarray, results_dir: Path) -> None:
    """
    OOB RMSE vs number of trees — closest analogue to a training-loss curve for
    Random Forest (OOB samples act as an internal validation set during fitting).
    """
    X_t = prep.transform(X_tr)
    tree_counts = list(range(10, 310, 10))
    oob_rmse = []

    rf = RandomForestRegressor(
        warm_start=True, oob_score=True,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    for n in tree_counts:
        rf.n_estimators = n
        rf.fit(X_t, y_tr)
        # samples never left out of bag yet have oob_prediction_ == 0; exclude them
        mask = rf.oob_prediction_ != 0
        rmse = mean_squared_error(y_tr[mask], rf.oob_prediction_[mask]) ** 0.5
        oob_rmse.append(rmse)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tree_counts, oob_rmse, "o-", color="steelblue", markersize=4, linewidth=1.5)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("OOB RMSE (citations)")
    ax.set_title("OOB Error Curve  (training-loss proxy for Random Forest)")
    fig.tight_layout()
    _save(fig, results_dir / "oob_error_curve.png")


def plot_metrics_summary(
    y_tr: np.ndarray, y_pred_tr: np.ndarray,
    y: np.ndarray,    y_cv: np.ndarray,
    y_te: np.ndarray, y_pred_te: np.ndarray,
    results_dir: Path,
) -> None:
    """Grouped bar chart: MAE / RMSE / R² for train, CV-validation, and test."""
    splits = ["Train", "CV (valid)", "Test"]
    pairs  = [(y_tr, y_pred_tr), (y, y_cv), (y_te, y_pred_te)]
    # {metric: [train_val, cv_val, test_val]}
    metrics = {k: [compute_metrics(yt, yp)[k] for yt, yp in pairs]
               for k in ("MAE", "RMSE", "R2")}

    x      = np.arange(len(splits))
    colors = ["#4c72b0", "#dd8452", "#55a868"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, (metric, values), color in zip(axes, metrics.items(), colors):
        bars = ax.bar(x, values, width=0.55, color=color, edgecolor="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        # annotate bar values; handle negative bars (possible for R²)
        ymin, ymax = ax.get_ylim()
        offset = (ymax - ymin) * 0.025
        for bar, val in zip(bars, values):
            if val >= 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, val - offset,
                        f"{val:.2f}", ha="center", va="top", fontsize=9)
    fig.suptitle("Performance Summary — Train / CV (Validation) / Test", fontsize=12)
    _save(fig, results_dir / "metrics_summary.png")


def plot_actual_vs_predicted(
    y_tr: np.ndarray, y_pred_tr: np.ndarray,
    y: np.ndarray,    y_cv: np.ndarray,
    y_te: np.ndarray, y_pred_te: np.ndarray,
    results_dir: Path,
) -> None:
    """Three-panel scatter: train, cross-validation (valid), hold-out test."""
    panels = [
        ("Train",      y_tr,  y_pred_tr),
        ("CV (valid)", y,     y_cv),
        ("Test",       y_te,  y_pred_te),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, (label, y_true, y_pred) in zip(axes, panels):
        m  = compute_metrics(y_true, y_pred)
        lo = min(y_true.min(), y_pred.min()) * 0.85
        hi = max(y_true.max(), y_pred.max()) * 1.10
        ax.scatter(y_true, y_pred, alpha=0.75, edgecolors="k", linewidths=0.4, zorder=3)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="Perfect prediction")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual citation count")
        ax.set_ylabel("Predicted citation count")
        ax.set_title(f"{label}\nRMSE={m['RMSE']:.1f}   R²={m['R2']:.2f}")
        ax.legend(fontsize=8)
    fig.suptitle("Actual vs Predicted Citation Count", fontsize=12)
    _save(fig, results_dir / "actual_vs_predicted.png")


def plot_residuals(
    y_tr: np.ndarray, y_pred_tr: np.ndarray,
    y: np.ndarray,    y_cv: np.ndarray,
    y_te: np.ndarray, y_pred_te: np.ndarray,
    results_dir: Path,
) -> None:
    """Residual histogram (top) and residuals-vs-predicted (bottom) for each split."""
    panels = [
        ("Train",      y_tr,  y_pred_tr),
        ("CV (valid)", y,     y_cv),
        ("Test",       y_te,  y_pred_te),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for col, (label, y_true, y_pred) in enumerate(panels):
        resid = y_true - y_pred
        # histogram
        axes[0, col].hist(resid, bins="auto", edgecolor="k", color="steelblue", alpha=0.8)
        axes[0, col].axvline(0, color="red", linestyle="--", linewidth=1.2)
        axes[0, col].set_title(f"{label}  —  Residual Distribution")
        axes[0, col].set_xlabel("Residual (actual − predicted)")
        axes[0, col].set_ylabel("Count")
        # residuals vs predicted
        axes[1, col].scatter(y_pred, resid, alpha=0.75, edgecolors="k", linewidths=0.4)
        axes[1, col].axhline(0, color="red", linestyle="--", linewidth=1.2)
        axes[1, col].set_title(f"{label}  —  Residuals vs Predicted")
        axes[1, col].set_xlabel("Predicted citation count")
        axes[1, col].set_ylabel("Residual")
    _save(fig, results_dir / "residuals.png")


def plot_feature_importances(
    pipeline: Pipeline, n_pca: int, results_dir: Path, top_n: int = 20,
) -> None:
    """Horizontal bar chart of top-N Random Forest feature importances."""
    rf           = pipeline.named_steps["model"]
    importances  = rf.feature_importances_
    feat_names   = [f"PCA_{i + 1}" for i in range(n_pca)] + META_COLS
    top_n        = min(top_n, len(feat_names))
    top_idx      = np.argsort(importances)[-top_n:]  # ascending; last = most important

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.38)))
    ax.barh(
        [feat_names[i] for i in top_idx],
        importances[top_idx],
        color="steelblue", edgecolor="k", linewidth=0.4,
    )
    ax.set_xlabel("Importance (mean impurity decrease)")
    ax.set_title(f"Top-{top_n} Feature Importances")
    fig.tight_layout()
    _save(fig, results_dir / "feature_importances.png")


# ── 6. Main ──────────────────────────────────────────────────────────────────

def main(db_pattern: str, parquet_path: str, model_path: str, results_dir: str) -> None:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- load & persist to parquet ------------------------------------------
    df = load_all_dbs(db_pattern)
    save_parquet(df, parquet_path)
    df = load_parquet(parquet_path)

    # --- features -----------------------------------------------------------
    X, y = build_features(df)
    n_samples = len(y)
    print(f"\nDataset : {n_samples} samples, {X.shape[1]} raw features")
    print(f"Target  : citation_count  min={y.min():.0f}  "
          f"max={y.max():.0f}  mean={y.mean():.1f}  std={y.std():.1f}")

    # cap PCA to the smallest CV training-fold size
    n_folds = min(5, n_samples)
    fold_train_size = n_samples - (n_samples // n_folds)
    n_pca = min(DEFAULT_N_PCA, fold_train_size - 1)
    if n_pca < DEFAULT_N_PCA:
        warnings.warn(
            f"Only {n_samples} samples — PCA components capped to {n_pca}. "
            "Collect more data for reliable generalisation.",
            UserWarning, stacklevel=2,
        )

    pipeline = build_pipeline(n_pca)
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # --- cross-validation (validation metrics on full dataset) --------------
    y_cv = cross_val_predict(pipeline, X, y, cv=cv)
    print_metrics(f"{n_folds}-fold cross-validation (validation)", y, y_cv)

    # --- hold-out train / test split ----------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_tr, y_tr)
    y_pred_tr = pipeline.predict(X_tr)
    y_pred_te = pipeline.predict(X_te)
    print_metrics("Train (80 %)", y_tr, y_pred_tr)
    print_metrics("Test  (20 %)", y_te, y_pred_te)

    # --- figures ------------------------------------------------------------
    print(f"\nSaving figures → {out}/")
    prep = pipeline.named_steps["prep"]
    plot_oob_curve(prep, X_tr, y_tr, out)
    plot_metrics_summary(y_tr, y_pred_tr, y, y_cv, y_te, y_pred_te, out)
    plot_actual_vs_predicted(y_tr, y_pred_tr, y, y_cv, y_te, y_pred_te, out)
    plot_residuals(y_tr, y_pred_tr, y, y_cv, y_te, y_pred_te, out)
    plot_feature_importances(pipeline, n_pca, out)

    # --- model --------------------------------------------------------------
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved → {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a citation-count predictor")
    parser.add_argument("--db",      default="*.db",           help="Glob pattern for input .db files")
    parser.add_argument("--parquet", default="papers.parquet", help="Parquet output path")
    parser.add_argument("--model",   default="model.joblib",   help="Saved model output path")
    parser.add_argument("--results", default="results2",        help="Directory for output figures")
    args = parser.parse_args()
    main(args.db, args.parquet, args.model, args.results)

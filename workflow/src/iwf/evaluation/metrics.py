"""Metric helpers — the single implementation used by every workflow.

Both the MLP heads (training.experiment) and the text-transformer trainer report
through these two functions, so all seeds/workflows produce the same metric keys
and the aggregation/plot layer never needs per-workflow special cases.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return {"rmse": 0.0, "mse": 0.0, "mae": 0.0, "r2": 0.0, "spearman": 0.0,
                "n": int(len(y_true))}
    rho = spearmanr(y_true, y_pred).correlation if np.std(y_pred) > 0 else 0.0
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho if rho == rho else 0.0),
        "n": int(len(y_true)),
    }


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict:
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels,
                                   zero_division=0)),
        "per_class_f1": [float(v) for v in
                         f1_score(y_true, y_pred, labels=labels, average=None,
                                  zero_division=0)],
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def generated_vs_human(human_scores: np.ndarray, generated: dict[str, np.ndarray]) -> dict:
    """Summary stats of predicted impact for the human test split vs each generated source."""
    def describe(a: np.ndarray) -> dict:
        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return {"n": 0}
        return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
                "median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
                "p90": float(np.percentile(a, 90))}

    out = {"human_test": describe(human_scores)}
    for name, arr in generated.items():
        out[name] = describe(arr)
    return out


# ===================================================================== benchmark metrics

def ndcg_at_frac(y_true: np.ndarray, y_score: np.ndarray, frac: float = 0.10) -> float:
    """NDCG@k where k = max(1, frac * len(y)).

    Assumes y_true is already binary (0/1) or normalized [0,1] gains.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) < 2:
        return float('nan')

    k = max(1, int(np.ceil(len(y_true) * frac)))
    # Sort by score descending and take top-k
    top_k_indices = np.argsort(-y_score)[:k]
    gains = y_true[top_k_indices]

    # Ideal DCG: best possible ranking (all 1s first)
    ideal_gains = np.sort(y_true)[-k:][::-1]

    # DCG calculation
    def dcg(gains):
        return np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))

    actual_dcg = dcg(gains)
    ideal_dcg = dcg(ideal_gains)

    if ideal_dcg == 0:
        return float('nan')
    return float(actual_dcg / ideal_dcg)


def top_frac_auc(y_true: np.ndarray, y_score: np.ndarray, frac: float = 0.10) -> float:
    """Binary AUC where label = (y_true >= quantile(1-frac))."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) < 2:
        return float('nan')

    threshold = np.quantile(y_true, 1 - frac)
    label = (y_true >= threshold).astype(int)

    # If single-class, return nan
    if len(np.unique(label)) < 2:
        return float('nan')

    try:
        return float(roc_auc_score(label, y_score))
    except ValueError:
        return float('nan')


def pairwise_accuracy(score_a: np.ndarray, score_b: np.ndarray,
                      label: np.ndarray) -> float:
    """Fraction of pairs where (score_a > score_b) == label.

    label 1 = paper A is more impactful; label 0 = paper B is more impactful.
    Ties (score_a == score_b) count as 0.5 correct when label == 0 or 1 accordingly.
    """
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    label = np.asarray(label, dtype=float)

    if len(label) == 0:
        return float('nan')

    pred = score_a > score_b  # True if A > B
    correct = pred == label.astype(bool)
    # Ties: score_a == score_b
    ties = score_a == score_b
    # For ties, count as 0.5 correct (neutral)
    correct = correct.astype(float)
    correct[ties] = 0.5

    return float(np.mean(correct))

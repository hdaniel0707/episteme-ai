import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_seed_results(run_dir: Path) -> list[dict]:
    seed_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit())
    results = []
    for sd in seed_dirs:
        result_path = sd / "results" / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                results.append(json.load(f))
    return results


def load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Feature label reconstruction
# ---------------------------------------------------------------------------

_KNOWN_SCALARS = {
    "year",
    "best_author_h_index",
    "h_index",
    "citation_count",
    "author_count",
    "page_count",
}


def reconstruct_feature_labels(cfg: dict, n_features: int) -> list[str]:
    """
    Rebuild feature labels from config and total feature count.
    Scalar features (1-dim each) are identified by name; the remainder
    is attributed to the first array feature (typically 'embedding').
    """
    feature_names: list[str] = cfg.get("features", [])
    if not feature_names:
        return [str(i) for i in range(n_features)]

    scalar_names = [f for f in feature_names if f in _KNOWN_SCALARS]
    array_names = [f for f in feature_names if f not in _KNOWN_SCALARS]
    n_scalars = len(scalar_names)
    n_array_total = n_features - n_scalars

    labels: list[str] = []
    if array_names:
        # Distribute array dims evenly across array features
        per_array = n_array_total // len(array_names)
        remainder = n_array_total % len(array_names)
        for idx, name in enumerate(array_names):
            dim = per_array + (1 if idx < remainder else 0)
            labels.extend(f"{name}[{i}]" for i in range(dim))
    labels.extend(scalar_names)
    return labels


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _ci_band(ax, x, values: np.ndarray, label: str, color: str, alpha: float = 0.2):
    """Plot mean ± std band for a (n_seeds, n_steps) array."""
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha)


def plot_loss_curves(seed_results: list[dict], run_dir: Path, fig_dir: Path) -> None:
    train_losses = [r["train_losses"] for r in seed_results if r.get("train_losses")]
    val_losses = [r["val_losses"] for r in seed_results if r.get("val_losses")]

    if not train_losses:
        return

    # Trim to shortest length in case seeds differ
    min_len = min(len(t) for t in train_losses + val_losses)
    train_arr = np.array([t[:min_len] for t in train_losses])
    val_arr = np.array([v[:min_len] for v in val_losses])
    x = np.arange(1, min_len + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    _ci_band(ax, x, train_arr, label="train (mean ± std)", color="steelblue")
    _ci_band(ax, x, val_arr, label="val (mean ± std)", color="coral")
    ax.set_xlabel("Epochs / Boosting stage")
    ax.set_ylabel("Loss")
    ax.set_title(f"{run_dir.name} — loss curves ({len(seed_results)} seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_curves.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(
    seed_results: list[dict],
    cfg: dict,
    run_dir: Path,
    fig_dir: Path,
    top_k: int = 20,
) -> None:
    fi_lists = [
        r["feature_importances"] for r in seed_results if r.get("feature_importances")
    ]
    if not fi_lists:
        return

    n_features = len(fi_lists[0])
    fi_arr = np.array(fi_lists)  # (n_seeds, n_features)
    labels = reconstruct_feature_labels(cfg, n_features)

    mean_imp = fi_arr.mean(axis=0)
    std_imp = fi_arr.std(axis=0)

    top_idx = np.argsort(mean_imp)[::-1][:top_k]
    top_means = mean_imp[top_idx]
    top_stds = std_imp[top_idx]
    top_labels = [labels[i] if i < len(labels) else str(i) for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, max(4, top_k * 0.35)))
    y_pos = np.arange(top_k)
    ax.barh(
        y_pos,
        top_means[::-1],
        xerr=top_stds[::-1],
        color="steelblue",
        alpha=0.8,
        error_kw={"elinewidth": 1.2, "capsize": 3},
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels[::-1], fontsize=8)
    ax.set_xlabel("Feature importance (mean ± std)")
    ax.set_title(f"{run_dir.name} — top {top_k} features ({len(fi_lists)} seeds)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=150)
    plt.close(fig)


def plot_metrics_summary(
    seed_results: list[dict], run_dir: Path, fig_dir: Path
) -> None:
    metric_keys = [
        "train_rmse",
        "val_rmse",
        "test_rmse",
        "train_r2",
        "val_r2",
        "test_r2",
    ]
    data = {k: [r[k] for r in seed_results if k in r] for k in metric_keys}
    data = {k: v for k, v in data.items() if v}
    if not data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rmse_keys = ["train_rmse", "val_rmse", "test_rmse"]
    r2_keys = ["train_r2", "val_r2", "test_r2"]

    for ax, keys, ylabel in zip(axes, [rmse_keys, r2_keys], ["RMSE", "R²"]):
        vals = [data[k] for k in keys if k in data]
        labels = [k.replace("_rmse", "").replace("_r2", "") for k in keys if k in data]
        means = [np.mean(v) for v in vals]
        stds = [np.std(v) for v in vals]
        x = np.arange(len(labels))
        ax.bar(
            x,
            means,
            yerr=stds,
            color=["steelblue", "coral", "seagreen"],
            alpha=0.8,
            error_kw={"elinewidth": 1.5, "capsize": 4},
        )
        for xi, seed_vals in enumerate(vals):
            ax.scatter(
                [xi] * len(seed_vals),
                seed_vals,
                color="black",
                zorder=3,
                s=20,
                alpha=0.6,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} across splits")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{run_dir.name} — metrics ({len(seed_results)} seeds)")
    fig.tight_layout()
    fig.savefig(fig_dir / "metrics_summary.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def analyse_run(run_dir: Path) -> None:
    seed_results = load_seed_results(run_dir)
    if not seed_results:
        print(f"  [SKIP] {run_dir.name}: no seed results found")
        return

    cfg = load_config(run_dir)
    fig_dir = run_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_curves(seed_results, run_dir, fig_dir)
    plot_feature_importance(seed_results, cfg, run_dir, fig_dir)
    plot_metrics_summary(seed_results, run_dir, fig_dir)

    saved = sorted(p.name for p in fig_dir.glob("*.png"))
    print(f"  [OK]   {run_dir.name}: {len(seed_results)} seeds → {saved}")


def run_individual_analysis(experiment_id: str):
    EXP_DIR = RESULTS_DIR / experiment_id
    print(f"Analysing: {EXP_DIR}")

    run_dirs = sorted(p for p in EXP_DIR.iterdir() if p.is_dir())
    for run_dir in run_dirs:
        analyse_run(run_dir)


def main() -> None:
    run_individual_analysis("exp_20260506_203629")


if __name__ == "__main__":
    main()

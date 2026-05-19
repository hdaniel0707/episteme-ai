import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "reg"

_KNOWN_SCALARS = {
    "year",
    "best_author_h_index",
    "h_index",
    "citation_count",
    "author_count",
    "page_count",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_seed_results(alg_dir: Path) -> list[dict]:
    seed_dirs = sorted(p for p in alg_dir.iterdir() if p.is_dir() and p.name.isdigit())
    results = []
    for sd in seed_dirs:
        result_path = sd / "results" / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                results.append(json.load(f))
    return results


def load_config(alg_dir: Path) -> dict:
    cfg_path = alg_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def reconstruct_feature_labels(cfg: dict, n_features: int) -> list[str]:
    feature_names: list[str] = cfg.get("features", [])
    if not feature_names:
        return [str(i) for i in range(n_features)]

    scalar_names = [f for f in feature_names if f in _KNOWN_SCALARS]
    array_names = [f for f in feature_names if f not in _KNOWN_SCALARS]
    n_array_total = n_features - len(scalar_names)

    labels: list[str] = []
    if array_names:
        per_array = n_array_total // len(array_names)
        remainder = n_array_total % len(array_names)
        for idx, name in enumerate(array_names):
            dim = per_array + (1 if idx < remainder else 0)
            labels.extend(f"{name}[{i}]" for i in range(dim))
    labels.extend(scalar_names)
    return labels


def get_alg_label(cfg: dict, alg_dir: Path) -> str:
    return cfg.get("alg_label") or cfg.get("alg_id") or alg_dir.name


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_results_comparison(alg_data: list[dict], fig_dir: Path) -> None:
    """Grouped bar chart: splits on x-axis, one bar per run per split."""
    metric_groups = [
        (["train_rmse", "val_rmse", "test_rmse"], "RMSE"),
        (["train_r2", "val_r2", "test_r2"], "R²"),
    ]
    split_labels = ["train", "val", "test"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (keys, ylabel) in zip(axes, metric_groups):
        n_splits = len(split_labels)
        n_runs = len(alg_data)
        width = 0.8 / n_runs
        x = np.arange(n_splits)

        for alg_idx, rd in enumerate(alg_data):
            means, stds = [], []
            for key in keys:
                vals = [r[key] for r in rd["seed_results"] if key in r]
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals) if vals else 0.0)

            offset = (alg_idx - (n_runs - 1) / 2) * width
            ax.bar(
                x + offset,
                means,
                width=width * 0.9,
                yerr=stds,
                label=rd["label"],
                color=colors[alg_idx % len(colors)],
                alpha=0.8,
                error_kw={"elinewidth": 1.2, "capsize": 3},
            )
            # scatter individual seed values
            for xi, key in enumerate(keys):
                seed_vals = [r[key] for r in rd["seed_results"] if key in r]
                ax.scatter(
                    [x[xi] + offset] * len(seed_vals),
                    seed_vals,
                    color=colors[alg_idx % len(colors)],
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=3,
                    s=18,
                    alpha=0.8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(split_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by split")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    n_seeds = "/".join(str(len(rd["seed_results"])) for rd in alg_data)
    fig.suptitle(f"Results comparison — {len(alg_data)} algs ({n_seeds} seeds)")
    fig.tight_layout()
    fig.savefig(fig_dir / "results_comparison.png", dpi=150)
    plt.close(fig)


def plot_loss_curves_comparison(alg_data: list[dict], fig_dir: Path) -> None:
    """Train and val loss curves for each run overlaid with CI bands."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, loss_key, title in zip(
        axes,
        ["train_losses", "val_losses"],
        ["Train loss", "Val loss"],
    ):
        for alg_idx, rd in enumerate(alg_data):
            losses = [r[loss_key] for r in rd["seed_results"] if r.get(loss_key)]
            if not losses:
                continue
            min_len = min(len(l) for l in losses)
            arr = np.array([l[:min_len] for l in losses])
            x = np.arange(1, min_len + 1)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            color = colors[alg_idx % len(colors)]
            ax.plot(x, mean, color=color, label=rd["label"])
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_xlabel("Epochs / Boosting stage")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Loss curves comparison")
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_curves_comparison.png", dpi=150)
    plt.close(fig)


def plot_feature_importance_comparison(
    alg_data: list[dict], fig_dir: Path, top_k: int = 20
) -> None:
    """
    If all runs share the same feature count: single grouped horizontal bar
    chart for the top-k features (ranked by global mean importance).
    Otherwise: one subplot per run showing its own top-k.
    """
    fi_data = []
    for rd in alg_data:
        fi_lists = [
            r["feature_importances"]
            for r in rd["seed_results"]
            if r.get("feature_importances")
        ]
        if not fi_lists:
            fi_data.append(None)
            continue
        fi_arr = np.array(fi_lists)
        labels = reconstruct_feature_labels(rd["cfg"], fi_arr.shape[1])
        fi_data.append(
            {
                "mean": fi_arr.mean(axis=0),
                "std": fi_arr.std(axis=0),
                "labels": labels,
                "alg_label": rd["label"],
            }
        )

    valid = [d for d in fi_data if d is not None]
    if not valid:
        return

    n_runs = len(valid)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    same_size = len({len(d["mean"]) for d in valid}) == 1

    if same_size and n_runs > 1:
        # Grouped horizontal bar: top-k by global mean importance
        global_mean = np.mean([d["mean"] for d in valid], axis=0)
        top_idx = np.argsort(global_mean)[::-1][:top_k]
        feature_labels = [
            valid[0]["labels"][i] if i < len(valid[0]["labels"]) else str(i)
            for i in top_idx
        ]

        width = 0.8 / n_runs
        y_pos = np.arange(len(top_idx))

        fig, ax = plt.subplots(figsize=(11, max(5, len(top_idx) * 0.40)))
        for alg_idx, d in enumerate(valid):
            means = d["mean"][top_idx]
            stds = d["std"][top_idx]
            offset = (alg_idx - (n_runs - 1) / 2) * width
            ax.barh(
                y_pos + offset,
                means[::-1],
                height=width * 0.9,
                xerr=stds[::-1],
                label=d["alg_label"],
                color=colors[alg_idx % len(colors)],
                alpha=0.8,
                error_kw={"elinewidth": 1.0, "capsize": 2},
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels[::-1], fontsize=8)
        ax.set_xlabel("Feature importance (mean ± std)")
        ax.set_title(f"Top {len(top_idx)} feature importances — {n_runs} runs")
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
    else:
        # Different feature spaces — one subplot per run
        ncols = min(n_runs, 3)
        nrows = (n_runs + ncols - 1) // ncols
        fig, axes_grid = plt.subplots(
            nrows,
            ncols,
            figsize=(7 * ncols, max(4, top_k * 0.35) * nrows),
        )
        axes_flat = np.array(axes_grid).flatten() if n_runs > 1 else [axes_grid]

        for ax, d in zip(axes_flat, valid):
            k = min(top_k, len(d["mean"]))
            top_idx = np.argsort(d["mean"])[::-1][:k]
            top_means = d["mean"][top_idx]
            top_stds = d["std"][top_idx]
            top_labels = [
                d["labels"][i] if i < len(d["labels"]) else str(i) for i in top_idx
            ]
            y_pos = np.arange(k)
            ax.barh(
                y_pos,
                top_means[::-1],
                xerr=top_stds[::-1],
                color="steelblue",
                alpha=0.8,
                error_kw={"elinewidth": 1.0, "capsize": 2},
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_labels[::-1], fontsize=8)
            ax.set_xlabel("Feature importance")
            ax.set_title(d["alg_label"])
            ax.grid(True, axis="x", alpha=0.3)

        for ax in axes_flat[len(valid) :]:
            ax.set_visible(False)

        fig.suptitle(f"Feature importance — top {top_k} per run")
        fig.tight_layout()

    fig.savefig(fig_dir / "feature_importance_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_compared_analysis(res_dir: Path, experiment_id: str) -> None:
    exp_dir = res_dir / experiment_id
    print(f"Comparing runs in: {exp_dir}")

    alg_dirs = sorted(p for p in exp_dir.iterdir() if p.is_dir() and p.name != "figs")
    alg_data = []
    for alg_dir in alg_dirs:
        seed_results = load_seed_results(alg_dir)
        if not seed_results:
            print(f"  [SKIP] {alg_dir.name}: no seed results found")
            continue
        cfg = load_config(alg_dir)
        label = get_alg_label(cfg, alg_dir)
        alg_data.append(
            {
                "label": label,
                "alg_dir": alg_dir,
                "seed_results": seed_results,
                "cfg": cfg,
            }
        )
        print(f"  [OK]   {alg_dir.name}: {len(seed_results)} seeds, label={label!r}")

    if len(alg_data) < 2:
        print("Need at least 2 runs with results to compare.")
        return

    fig_dir = exp_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_results_comparison(alg_data, fig_dir)
    plot_loss_curves_comparison(alg_data, fig_dir)
    plot_feature_importance_comparison(alg_data, fig_dir, top_k=20)

    saved = sorted(p.name for p in fig_dir.glob("*.png"))
    print(f"Saved {len(saved)} figures to {fig_dir}: {saved}")


def main() -> None:
    run_compared_analysis(RESULTS_DIR, "exp_20260506_203629")


if __name__ == "__main__":
    main()

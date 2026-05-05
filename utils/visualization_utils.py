from pathlib import Path

import numpy as np
from scipy import stats
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def plot_pca_2d_cat(
    X2: np.ndarray,
    labels: np.ndarray | list | None = None,
    names: np.ndarray | list | None = None,
    scores: np.ndarray | list | None = None,
    title: str = "",
    save_path: Path | None = None,
):
    X2 = np.asarray(X2)
    n = len(X2)

    names = (
        np.array([f"point {i}" for i in range(n)])
        if names is None
        else np.asarray(names)
    )
    labels = (
        np.array(["default"] * n) if labels is None else np.asarray(labels, dtype=str)
    )

    fig = go.Figure()

    # Get unique folder names (labels)
    unique_labels = sorted(np.unique(labels).tolist())

    scores_arr = np.asarray(scores) if scores is not None else None
    # Normalize scores to a reasonable marker size range (6–20px)
    if scores_arr is not None:
        s_min, s_max = scores_arr.min(), scores_arr.max()
        denom = s_max - s_min if s_max != s_min else 1.0
        normalized = 6 + 14 * (scores_arr - s_min) / denom
    else:
        normalized = None

    for label in unique_labels:
        mask = labels == label

        sub_x = X2[mask, 0]
        sub_y = X2[mask, 1]
        sub_names = names[mask].tolist()
        sub_sizes = normalized[mask].tolist() if normalized is not None else 10
        sub_scores = scores_arr[mask].tolist() if scores_arr is not None else None

        score_hover = "Score: %{customdata:.3f}<br>" if sub_scores is not None else ""

        fig.add_trace(
            go.Scatter(
                x=sub_x,
                y=sub_y,
                name=label,  # This creates the legend entry
                mode="markers",
                marker=dict(size=sub_sizes, opacity=0.8),
                text=sub_names,
                customdata=sub_scores,
                hovertemplate=(
                    "<b>Group: " + label + "</b><br>"
                    "Title: %{text}<br>" + score_hover + "Dim 1: %{x:.3f}<br>"
                    "Dim 2: %{y:.3f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        # Ensures 1 unit on X = 1 unit on Y (important for PCA)
        xaxis=dict(scaleanchor="y", scaleratio=1),
        margin=dict(l=40, r=20, t=40, b=40),
        legend_title_text="Groups",
        hovermode="closest",
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Interactive plot saved to {save_path}")

    fig.show()


def plot_pca_3d_cat(
    X3: np.ndarray,
    labels: np.ndarray | list | None = None,
    names: np.ndarray | list | None = None,
    scores: np.ndarray | list | None = None,
    title: str = "",
    save_path: Path | None = None,
):
    X3 = np.asarray(X3)
    n = len(X3)

    names = (
        np.array([f"point {i}" for i in range(n)])
        if names is None
        else np.asarray(names)
    )
    labels = (
        np.array(["default"] * n) if labels is None else np.asarray(labels, dtype=str)
    )

    fig = go.Figure()

    unique_labels = sorted(np.unique(labels).tolist())

    scores_arr = np.asarray(scores) if scores is not None else None
    # Normalize scores to a reasonable marker size range (3–12px)
    if scores_arr is not None:
        s_min, s_max = scores_arr.min(), scores_arr.max()
        denom = s_max - s_min if s_max != s_min else 1.0
        normalized = 3 + 9 * (scores_arr - s_min) / denom
    else:
        normalized = None

    for label in unique_labels:
        mask = labels == label

        sub_x = X3[mask, 0]
        sub_y = X3[mask, 1]
        sub_z = X3[mask, 2]
        sub_names = names[mask].tolist()
        sub_sizes = normalized[mask].tolist() if normalized is not None else 5
        sub_scores = scores_arr[mask].tolist() if scores_arr is not None else None

        score_hover = "Score: %{customdata:.3f}<br>" if sub_scores is not None else ""

        fig.add_trace(
            go.Scatter3d(
                x=sub_x,
                y=sub_y,
                z=sub_z,
                name=label,
                mode="markers",
                marker=dict(
                    size=sub_sizes,
                    opacity=0.8,
                    line=dict(width=0.5, color="white"),
                ),
                text=sub_names,
                customdata=sub_scores,
                hovertemplate=(
                    "<b>Group: " + label + "</b><br>"
                    "Title: %{text}<br>" + score_hover + "Dim 1: %{x:.3f}<br>"
                    "Dim 2: %{y:.3f}<br>"
                    "Dim 3: %{z:.3f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
            zaxis_title="Dim 3",
            # This keeps the aspect ratio proportional
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),  # 3D plots benefit from more space
        legend_title_text="Groups",
        hovermode="closest",
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Interactive plot saved to {save_path}")

    fig.show()


def plot_citation_year_correlation(
    years: np.ndarray | list,
    citations: np.ndarray | list,
    names: np.ndarray | list | None = None,
    title: str = "Citation Count vs Year",
    log1p_text: bool = False,
    save_path: Path | None = None,
):
    years = np.asarray(years)
    citations = np.asarray(citations)
    n = len(years)

    names_arr = (
        np.array([f"point {i}" for i in range(n)])
        if names is None
        else np.asarray(names)
    )

    # per-year stats for the trend line
    unique_years = np.unique(years)
    median_citations = np.array(
        [np.median(citations[years == y]) for y in unique_years]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=years.tolist(),
            y=citations.tolist(),
            mode="markers",
            name="Articles",
            text=names_arr.tolist(),
            marker=dict(size=6, opacity=0.5, color="steelblue"),
            hovertemplate=(
                "Title: %{text}<br>"
                "Year: %{x}<br>"
                "Citations: %{y}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=unique_years.tolist(),
            y=median_citations.tolist(),
            mode="lines+markers",
            name="Median per year",
            line=dict(color="firebrick", width=2),
            marker=dict(size=7),
            hovertemplate="Year: %{x}<br>Median citations: %{y:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="log1p(Citation Count)" if log1p_text else "Citation Count",
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="closest",
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Interactive plot saved to {save_path}")

    fig.show()


def plot_hindex_citation_correlation(
    h_indices: np.ndarray | list,
    citations: np.ndarray | list,
    names: np.ndarray | list | None = None,
    title: str = "Citation Count vs Best Author H-Index",
    log1p_text: bool = False,
    save_path: Path | None = None,
):
    h_indices = np.asarray(h_indices, dtype=float)
    citations = np.asarray(citations, dtype=float)

    valid = ~(np.isnan(h_indices) | np.isnan(citations))
    h_valid = h_indices[valid]
    c_valid = citations[valid]
    names_arr = (
        np.array([f"point {i}" for i in range(len(h_indices))])
        if names is None
        else np.asarray(names)
    )
    names_valid = names_arr[valid]

    unique_h = np.unique(h_valid)
    median_citations = np.array([np.median(c_valid[h_valid == h]) for h in unique_h])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=h_valid.tolist(),
            y=c_valid.tolist(),
            mode="markers",
            name="Articles",
            text=names_valid.tolist(),
            marker=dict(size=6, opacity=0.5, color="steelblue"),
            hovertemplate=(
                "Title: %{text}<br>"
                "Best Author H-Index: %{x}<br>"
                "Citations: %{y}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=unique_h.tolist(),
            y=median_citations.tolist(),
            mode="lines+markers",
            name="Median per h-index",
            line=dict(color="firebrick", width=2),
            marker=dict(size=7),
            hovertemplate="H-Index: %{x}<br>Median citations: %{y:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Best Author H-Index",
        yaxis_title="log1p(Citation Count)" if log1p_text else "Citation Count",
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="closest",
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Interactive plot saved to {save_path}")

    fig.show()


def plot_year_histogram(
    years: np.ndarray | list,
    title: str = "Articles by Year",
    save_path: Path | None = None,
):
    years = np.asarray(years)
    unique, counts = np.unique(years, return_counts=True)

    fig = go.Figure(
        go.Bar(
            x=unique.tolist(),
            y=counts.tolist(),
            hovertemplate="Year: %{x}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Number of Articles",
        xaxis=dict(type="category"),
        margin=dict(l=40, r=20, t=40, b=40),
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Interactive plot saved to {save_path}")

    fig.show()


########################################################################
# TRAIN
########################################################################


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


def plot_loss_curves_ci(
    all_train_losses: list[list[float]],
    all_val_losses: list[list[float]],
    out: Path,
    ci: float = 0.95,
) -> None:
    from scipy import stats

    train = np.array(all_train_losses)
    val = np.array(all_val_losses)
    n = train.shape[0]
    iters = np.arange(1, train.shape[1] + 1)

    t_crit = stats.t.ppf((1 + ci) / 2, df=n - 1)

    def ci_bounds(arr):
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(n)
        return mean, mean - t_crit * se, mean + t_crit * se

    tr_mean, tr_lo, tr_hi = ci_bounds(train)
    vl_mean, vl_lo, vl_hi = ci_bounds(val)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, tr_mean, color="steelblue", label="Train MSE (mean)")
    ax.fill_between(
        iters,
        tr_lo,
        tr_hi,
        alpha=0.25,
        color="steelblue",
        label=f"Train {int(ci*100)}% CI",
    )
    ax.plot(iters, vl_mean, color="tomato", label="Val MSE (mean)")
    ax.fill_between(
        iters, vl_lo, vl_hi, alpha=0.25, color="tomato", label=f"Val {int(ci*100)}% CI"
    )
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("MSE  (log1p scale)")
    ax.set_title(f"Training & Validation Loss  (n={n} seeds, {int(ci*100)}% CI)")
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


#############################################################
# COMPARE
#############################################################

METRICS = ["test_rmse", "test_r2", "val_rmse", "val_r2"]
METRIC_LABELS = {
    "test_rmse": "Test RMSE",
    "test_r2": "Test R²",
    "val_rmse": "Val RMSE",
    "val_r2": "Val R²",
}


def plot_test_metrics_ci(
    comparison: dict, out_dir: Path, labels: list[str] | None = None
) -> None:
    run_ids = comparison["run_ids"]
    display_labels = labels if labels is not None else run_ids
    ci = comparison["ci"]
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, METRICS):
        means, lo_errs, hi_errs = [], [], []
        for run_id in run_ids:
            s = comparison["runs"][run_id].get(metric, {})
            mean = s.get("mean", 0.0)
            means.append(mean)
            lo_errs.append(mean - s.get("ci_low", mean))
            hi_errs.append(s.get("ci_high", mean) - mean)

        x = np.arange(len(run_ids))
        bar_colors = [colors[i % len(colors)] for i in range(len(run_ids))]
        bars = ax.bar(
            x,
            means,
            color=bar_colors,
            alpha=0.75,
            yerr=[lo_errs, hi_errs],
            capsize=6,
            ecolor="black",
            error_kw={"lw": 1.5},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=15, ha="right", fontsize=9)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel(METRIC_LABELS[metric])

        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(lo_errs) * 0.05,
                f"{mean:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(f"Test & Val Metrics Comparison  ({int(ci * 100)}% CI)", fontsize=13)
    fig.tight_layout()
    out = out_dir / "test_metrics_ci.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_loss_curves_comparison(
    run_ids: list[str],
    runs_data: dict[str, list[dict]],
    out_dir: Path,
    ci: float,
    labels: list[str] | None = None,
) -> None:
    display_labels = labels if labels is not None else run_ids
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, run_id in enumerate(run_ids):
        runs = runs_data[run_id]
        color = colors[i % len(colors)]
        label = display_labels[i]

        all_train = np.array([r["train_losses"] for r in runs if "train_losses" in r])
        all_val = np.array([r["val_losses"] for r in runs if "val_losses" in r])
        if all_train.size == 0 or all_val.size == 0:
            print(f"  Skipping loss curves for '{run_id}': no loss data")
            continue

        n = all_train.shape[0]
        iters = np.arange(1, all_train.shape[1] + 1)
        t_crit = stats.t.ppf((1 + ci) / 2, df=max(n - 1, 1))

        for ax, arr, split in zip(axes, [all_train, all_val], ["Train", "Val"]):
            mean = arr.mean(axis=0)
            se = arr.std(axis=0, ddof=1) / np.sqrt(n)
            lo, hi = mean - t_crit * se, mean + t_crit * se
            ax.plot(iters, mean, color=color, label=label)
            ax.fill_between(iters, lo, hi, alpha=0.2, color=color)

    for ax, split in zip(axes, ["Train", "Val"]):
        ax.set_title(f"{split} Loss  ({int(ci * 100)}% CI)")
        ax.set_xlabel("Boosting iteration")
        ax.set_ylabel("MSE  (log1p scale)")
        ax.legend()

    fig.suptitle("Train & Validation Loss Comparison", fontsize=13)
    fig.tight_layout()
    out = out_dir / "loss_curves_comparison_ci.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

from pathlib import Path
import textwrap

import numpy as np
from scipy import stats
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.ticker import MaxNLocator
import json
from sklearn.metrics import mean_squared_error, r2_score

# Consistent colors per LLM family are defined in cfg/colors.yaml and loaded via
# color_utils so every plotting script shares one source of truth.
from epai.utils.color_utils import llm_color as _llm_color

# The reported metrics and how each is meant to be read (its reference level, its
# direction, the caption it needs) live with the code that computes them, so a
# figure can never disagree with the number it is drawing. cd_core imports
# nothing from here, so this direction of the dependency is the safe one.
from epai.utils.cd_core import (
    BY_SEED_SUFFIX,
    CD_CHANCE_RATE,
    SCORE_DIST_KEY,
    MetricSpec,
    cell_metric_specs,
    level_tag,
    HEADLINE_METRICS,
    domain_mask,
    headline_metric_specs,
    present_headline_specs,
    present_metric_specs,
    split_diag,
    transported_fpr_key,
)


def plot_pca_2d_cat(
    X2: np.ndarray,
    labels: np.ndarray | list | None = None,
    names: np.ndarray | list | None = None,
    scores: np.ndarray | list | None = None,
    title: str = "",
    save_path: Path | None = None,
    show_centroids: bool = False,
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

    centroids = []  # (label, cx, cy, color) collected to draw on top after the loop

    for label in unique_labels:
        mask = labels == label

        sub_x = X2[mask, 0]
        sub_y = X2[mask, 1]
        sub_names = names[mask].tolist()
        sub_sizes = normalized[mask].tolist() if normalized is not None else 10
        sub_scores = scores_arr[mask].tolist() if scores_arr is not None else None

        score_hover = "Score: %{customdata:.3f}<br>" if sub_scores is not None else ""

        color = _llm_color(label)
        marker_kw = dict(size=sub_sizes, opacity=0.8)
        if color is not None:
            marker_kw["color"] = color

        fig.add_trace(
            go.Scatter(
                x=sub_x,
                y=sub_y,
                name=label,  # This creates the legend entry
                mode="markers",
                marker=marker_kw,
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

        if show_centroids:
            centroids.append(
                (label, float(np.mean(sub_x)), float(np.mean(sub_y)), color)
            )

    # Draw the centroid stars after every group's points so all stars sit on
    # top of all points, not just the points of their own group.
    for label, cx, cy, color in centroids:
        centroid_kw = dict(size=20, symbol="star", line=dict(width=1, color="black"))
        if color is not None:
            centroid_kw["color"] = color
        fig.add_trace(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers",
                marker=centroid_kw,
                showlegend=False,
                hovertemplate=(
                    "<b>Centroid: " + label + "</b><br>"
                    "Dim 1: %{x:.3f}<br>"
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
        html_path = Path(save_path).with_suffix(".html")
        fig.write_html(html_path)
        png_path = Path(save_path).with_suffix(".png")
        fig.write_image(str(png_path), scale=3)
        print(f"Interactive plot saved to {html_path}")
        print(f"Image saved to {png_path}")

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
        html_path = Path(save_path).with_suffix(".html")
        fig.write_html(html_path)
        png_path = Path(save_path).with_suffix(".png")
        fig.write_image(str(png_path), scale=3)
        print(f"Interactive plot saved to {html_path}")
        print(f"Image saved to {png_path}")

    fig.show()


def plot_citation_year_correlation(
    years: np.ndarray | list,
    citations: np.ndarray | list,
    names: np.ndarray | list | None = None,
    title: str = "Citation Count vs Year",
    log1p_text: bool = False,
    central_tendency: str = "mean",
    smooth_mean: bool = True,
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

    agg_fn = np.mean if central_tendency == "mean" else np.median
    agg_label = central_tendency.capitalize()

    unique_years = np.unique(years)
    agg_citations = np.array([agg_fn(citations[years == y]) for y in unique_years])

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
            y=agg_citations.tolist(),
            mode="lines+markers",
            name=f"{agg_label} per year",
            line=dict(color="salmon", width=2),
            marker=dict(size=7),
            hovertemplate=f"Year: %{{x}}<br>{agg_label} citations: %{{y:.1f}}<extra></extra>",
        )
    )

    if smooth_mean and len(unique_years) >= 4:
        window = max(3, len(unique_years) // 5)
        pad = window // 2
        padded = np.pad(agg_citations, pad, mode="edge")
        ma = np.convolve(padded, np.ones(window) / window, mode="valid")
        fig.add_trace(
            go.Scatter(
                x=unique_years.tolist(),
                y=ma.tolist(),
                mode="lines",
                name=f"Smooth {agg_label}",
                line=dict(color="firebrick", width=4),
                hovertemplate=f"Year: %{{x}}<br>Smooth {agg_label}: %{{y:.1f}}<extra></extra>",
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
        html_path = Path(save_path).with_suffix(".html")
        fig.write_html(html_path)
        png_path = Path(save_path).with_suffix(".png")
        fig.write_image(str(png_path), scale=3)
        print(f"Interactive plot saved to {html_path}")
        print(f"Image saved to {png_path}")

    fig.show()


def plot_hindex_citation_correlation(
    h_indices: np.ndarray | list,
    citations: np.ndarray | list,
    names: np.ndarray | list | None = None,
    title: str = "Citation Count vs Best Author H-Index",
    log1p_text: bool = False,
    central_tendency: str = "mean",
    smooth_mean: bool = True,
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

    agg_fn = np.mean if central_tendency == "mean" else np.median
    agg_label = central_tendency.capitalize()

    unique_h = np.unique(h_valid)
    agg_citations = np.array([agg_fn(c_valid[h_valid == h]) for h in unique_h])

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
            y=agg_citations.tolist(),
            mode="lines+markers",
            name=f"{agg_label} per h-index",
            line=dict(color="salmon", width=2),
            marker=dict(size=7),
            hovertemplate=f"H-Index: %{{x}}<br>{agg_label} citations: %{{y:.1f}}<extra></extra>",
        )
    )

    if smooth_mean and len(unique_h) >= 4:
        window = max(3, len(unique_h) // 5)
        pad = window // 2
        padded = np.pad(agg_citations, pad, mode="edge")
        ma = np.convolve(padded, np.ones(window) / window, mode="valid")
        fig.add_trace(
            go.Scatter(
                x=unique_h.tolist(),
                y=ma.tolist(),
                mode="lines",
                name=f"Smooth {agg_label}",
                line=dict(color="firebrick", width=4),
                hovertemplate=f"H-Index: %{{x}}<br>Smooth {agg_label}: %{{y:.1f}}<extra></extra>",
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
        html_path = Path(save_path).with_suffix(".html")
        fig.write_html(html_path)
        png_path = Path(save_path).with_suffix(".png")
        fig.write_image(str(png_path), scale=3)
        print(f"Interactive plot saved to {html_path}")
        print(f"Image saved to {png_path}")

    fig.show()


def plot_histogram(
    values: np.ndarray | list,
    title: str = "Histogram",
    x_axis_title: str = "Value",
    y_axis_title: str = "Count",
    bins: int | None = None,
    save_path: Path | None = None,
):
    values = np.asarray(values)

    if bins is not None:
        counts, edges = np.histogram(values, bins=bins)
        bin_labels = [f"{edges[i]:.4g}–{edges[i+1]:.4g}" for i in range(len(counts))]
        x = bin_labels
        y = counts.tolist()
        hovertemplate = f"{x_axis_title}: %{{x}}<br>Count: %{{y}}<extra></extra>"
        xaxis_type = "category"
    else:
        unique, counts = np.unique(values, return_counts=True)
        x = unique.tolist()
        y = counts.tolist()
        hovertemplate = f"{x_axis_title}: %{{x}}<br>Count: %{{y}}<extra></extra>"
        xaxis_type = "category"

    fig = go.Figure(go.Bar(x=x, y=y, hovertemplate=hovertemplate))

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        xaxis=dict(type=xaxis_type),
        margin=dict(l=40, r=20, t=40, b=40),
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        html_path = Path(save_path).with_suffix(".html")
        fig.write_html(html_path)
        png_path = Path(save_path).with_suffix(".png")
        fig.write_image(str(png_path), scale=3)
        print(f"Interactive plot saved to {html_path}")
        print(f"Image saved to {png_path}")

    fig.show()


def plot_categorical_distribution(
    values: np.ndarray | list,
    title: str = "Categorical Distribution",
    x_axis_title: str = "Category",
    y_axis_title: str = "Count",
    save_path: Path | None = None,
):
    values = np.asarray(values, dtype=str)
    unique, counts = np.unique(values, return_counts=True)
    order = np.argsort(counts)[::-1]
    unique = unique[order]
    counts = counts[order]

    fig = go.Figure(
        go.Bar(
            x=unique.tolist(),
            y=counts.tolist(),
            hovertemplate=f"{x_axis_title}: %{{x}}<br>Count: %{{y}}<extra></extra>",
        )
    )

    total = int(counts.sum())
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>n = {total:,}</sup>"),
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        xaxis=dict(type="category"),
        margin=dict(l=40, r=20, t=40, b=120),
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        html_path = Path(save_path).with_suffix(".html")
        fig.write_html(html_path)
        png_path = Path(save_path).with_suffix(".png")
        fig.write_image(str(png_path), scale=3)
        print(f"Interactive plot saved to {html_path}")
        print(f"Image saved to {png_path}")

    fig.show()


def plot_human_vs_llm_distance(
    embeddings: np.ndarray,
    paper_ids: np.ndarray | list,
    labels: np.ndarray | list,
    title: str = "Human vs LLM Abstract Similarity",
    save_path: Path | None = None,
):
    """Bar chart of mean cosine distance between human and each LLM embedding, per paper."""
    from collections import defaultdict

    embeddings = np.asarray(embeddings, dtype=float)
    paper_ids = np.asarray(paper_ids)
    labels = np.asarray(labels, dtype=str)

    pair_distances: dict[str, list[float]] = defaultdict(list)

    for pid in np.unique(paper_ids):
        mask = paper_ids == pid
        embs = embeddings[mask]
        lbls = labels[mask]

        human_mask = lbls == "human"
        if not human_mask.any():
            continue
        human_emb = embs[human_mask][0]
        h_norm = np.linalg.norm(human_emb)
        if h_norm == 0:
            continue
        human_unit = human_emb / h_norm

        for emb, lbl in zip(embs, lbls):
            if lbl == "human":
                continue
            n = np.linalg.norm(emb)
            if n == 0:
                continue
            dist = 1.0 - float(np.dot(emb / n, human_unit))
            pair_distances[f"human vs {lbl}"].append(dist)

    pair_labels = sorted(pair_distances.keys())
    means = [float(np.mean(pair_distances[lbl])) for lbl in pair_labels]
    stds = [float(np.std(pair_distances[lbl])) for lbl in pair_labels]

    # Color each bar by the LLM family ("human vs <llm-label>" → strip prefix)
    bar_colors = [
        _llm_color(lbl.removeprefix("human vs ")) or "#636efa" for lbl in pair_labels
    ]

    fig = go.Figure(
        go.Bar(
            x=pair_labels,
            y=means,
            marker_color=bar_colors,
            error_y=dict(type="data", array=stds, visible=True),
            hovertemplate="%{x}<br>Mean cosine dist: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Pair",
        yaxis_title="Mean cosine distance (human vs LLM)",
        xaxis=dict(type="category"),
        margin=dict(l=40, r=20, t=40, b=100),
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        html_path = Path(save_path).with_suffix(".html")
        fig.write_html(html_path)
        png_path = Path(save_path).with_suffix(".png")
        fig.write_image(str(png_path), scale=3)
        print(f"Interactive plot saved to {html_path}")
        print(f"Image saved to {png_path}")

    fig.show()


########################################################################
# TRAIN - CLASSIFICATION
########################################################################


def _save_clf_alg_figures(
    out_dir: Path,
    label: str,
    results: list[dict],
    class_names: list[str],
    splits: list[str],
    show_random: bool,
    tick_fs: int,
) -> None:
    """Save accuracy / confusion-matrix / F1 figures for a single algorithm into
    its own experiment subfolder (out_dir), one panel per split."""
    from sklearn.metrics import confusion_matrix as compute_cm

    nc = len(class_names)
    n_splits = len(splits)
    split_colors = ["steelblue", "orange", "seagreen"]

    # Accuracy — one bar per split.
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(n_splits)
    means = [float(np.mean([r[f"{s}_accuracy"] for r in results])) for s in splits]
    stds = [float(np.std([r[f"{s}_accuracy"] for r in results])) for s in splits]
    bars = ax.bar(x, means, 0.6, yerr=stds, capsize=4, color=split_colors[:n_splits])
    ax.bar_label(
        bars, fmt="%.3f", label_type="edge", fontsize=11, padding=2, color="black"
    )
    if show_random:
        ax.axhline(1 / nc, linestyle="--", color="red", linewidth=1.2, label="random")
        ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{label} – Accuracy")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy.png", dpi=150)
    plt.close(fig)

    # Confusion matrices — one per split.
    fig, axes = plt.subplots(1, n_splits, figsize=(5 * n_splits, 4.5))
    axes = np.atleast_1d(axes)
    for j, split in enumerate(splits):
        ax = axes[j]
        cms = []
        for r in results:
            yt = r.get(f"y_{split}")
            yp = r.get(f"y_{split}_pred")
            if yt is not None and yp is not None:
                cms.append(compute_cm(yt, yp, labels=range(nc)))
        if not cms:
            ax.text(0.5, 0.5, "no predictions", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(split)
            continue
        avg_cm = np.mean(cms, axis=0)
        norm_cm = avg_cm / avg_cm.sum(axis=1, keepdims=True)
        im = ax.imshow(norm_cm, vmin=0, vmax=1, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(nc))
        ax.set_yticks(range(nc))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=tick_fs)
        ax.set_yticklabels(class_names, fontsize=tick_fs)
        for r_idx in range(nc):
            for c_idx in range(nc):
                ax.text(
                    c_idx,
                    r_idx,
                    f"{avg_cm[r_idx, c_idx]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=tick_fs,
                    color="white" if norm_cm[r_idx, c_idx] > 0.5 else "black",
                )
        ax.set_title(split)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle(f"{label} – Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # Per-class F1 — one panel per split.
    fig, axes = plt.subplots(1, n_splits, figsize=(max(6, nc) * n_splits * 0.6, 4))
    axes = np.atleast_1d(axes)
    for j, split in enumerate(splits):
        ax = axes[j]
        f1s = {c: [] for c in class_names}
        for r in results:
            report = r.get(f"{split}_report", {})
            for c in class_names:
                if c in report:
                    f1s[c].append(report[c]["f1-score"])
        means = [float(np.mean(f1s[c])) if f1s[c] else 0.0 for c in class_names]
        stds = [float(np.std(f1s[c])) if f1s[c] else 0.0 for c in class_names]
        bars = ax.bar(range(nc), means, yerr=stds, capsize=4, color="steelblue")
        ax.bar_label(
            bars, fmt="%.3f", label_type="edge", fontsize=11, padding=2, color="black"
        )
        ax.axhline(1 / nc, linestyle="--", color="red", linewidth=1.2, label="random")
        ax.set_xticks(range(nc))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=tick_fs)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("F1 Score")
        ax.set_title(split)
        ax.legend(fontsize=7)
    fig.suptitle(f"{label} – F1 Score")
    fig.tight_layout()
    fig.savefig(out_dir / "f1_scores.png", dpi=150)
    plt.close(fig)


def plot_clf_results(exp_dir: Path, show_random: bool = True) -> None:
    from sklearn.metrics import confusion_matrix as compute_cm

    alg_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
    if not alg_dirs:
        print(f"No algorithm directories found in {exp_dir}")
        return

    # Each config can drop different classes, so every alg dir has its own
    # LabelEncoder / label_mapping.json — load them per-alg instead of assuming
    # all models share the first one's classes.
    alg_class_names: dict[str, list[str]] = {}
    for d in alg_dirs:
        with open(d / "label_mapping.json") as f:
            label_mapping = json.load(f)
        alg_class_names[d.name] = [
            label_mapping[str(i)] for i in range(len(label_mapping))
        ]
    max_classes = max(len(v) for v in alg_class_names.values())

    alg_names = [d.name for d in alg_dirs]
    alg_labels = []
    for d in alg_dirs:
        cfg_path = d / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            alg_labels.append(cfg.get("alg_label", d.name))
        else:
            alg_labels.append(d.name)
    data: dict = {}
    for alg_dir in alg_dirs:
        seeds = sorted(
            [d.name for d in alg_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=int,
        )
        data[alg_dir.name] = []
        for seed in seeds:
            result_path = alg_dir / seed / "results" / "result.json"
            if result_path.exists():
                with open(result_path) as f:
                    data[alg_dir.name].append(json.load(f))

    splits = ["train", "val", "test"]
    n_models = len(alg_names)
    n_splits = len(splits)
    tick_fs = max(6, 10 - max_classes // 3)

    # Figure 1: Accuracy
    fig, ax = plt.subplots(figsize=(max(8, n_models * 2.5), 5))
    x = np.arange(n_models)
    w = 0.25
    for j, (split, color) in enumerate(
        zip(splits, ["steelblue", "orange", "seagreen"])
    ):
        means, stds = [], []
        for mn in alg_names:
            accs = [r[f"{split}_accuracy"] for r in data[mn]]
            means.append(float(np.mean(accs)))
            stds.append(float(np.std(accs)))
        bars = ax.bar(x + j * w, means, w, label=split, color=color, yerr=stds, capsize=4)
        ax.bar_label(
            bars, fmt="%.3f", label_type="edge", fontsize=11, padding=2, color="black"
        )
    if show_random:
        ax.axhline(
            1 / max_classes, linestyle="--", color="red", linewidth=1.2, label="random"
        )
    ax.set_xticks(x + w)
    ax.set_xticklabels(alg_labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Model and Split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(exp_dir / "accuracy.png", dpi=150)
    plt.close(fig)

    # Figure 2: Confusion matrices
    fig, axes = plt.subplots(n_models, n_splits, figsize=(5 * n_splits, 4.5 * n_models))
    axes = np.array(axes).reshape(n_models, n_splits)
    for i, mn in enumerate(alg_names):
        cn = alg_class_names[mn]
        nc = len(cn)
        for j, split in enumerate(splits):
            ax = axes[i, j]
            cms = []
            for r in data[mn]:
                yt = r.get(f"y_{split}")
                yp = r.get(f"y_{split}_pred")
                if yt is not None and yp is not None:
                    cms.append(compute_cm(yt, yp, labels=range(nc)))
            if not cms:
                ax.text(
                    0.5,
                    0.5,
                    "no predictions",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{alg_labels[i]} – {split}")
                continue
            avg_cm = np.mean(cms, axis=0)
            norm_cm = avg_cm / avg_cm.sum(axis=1, keepdims=True)
            im = ax.imshow(norm_cm, vmin=0, vmax=1, cmap="Blues")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(nc))
            ax.set_yticks(range(nc))
            ax.set_xticklabels(cn, rotation=45, ha="right", fontsize=tick_fs)
            ax.set_yticklabels(cn, fontsize=tick_fs)
            for r_idx in range(nc):
                for c_idx in range(nc):
                    ax.text(
                        c_idx,
                        r_idx,
                        f"{avg_cm[r_idx, c_idx]:.0f}",
                        ha="center",
                        va="center",
                        fontsize=tick_fs,
                        color="white" if norm_cm[r_idx, c_idx] > 0.5 else "black",
                    )
            ax.set_title(f"{alg_labels[i]} – {split}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(exp_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # Figure 3: Per-class F1
    fig, axes = plt.subplots(
        n_models, n_splits, figsize=(max(6, max_classes) * n_splits * 0.6, 4 * n_models)
    )
    axes = np.array(axes).reshape(n_models, n_splits)
    for i, mn in enumerate(alg_names):
        cn = alg_class_names[mn]
        nc = len(cn)
        for j, split in enumerate(splits):
            ax = axes[i, j]
            f1s = {c: [] for c in cn}
            for r in data[mn]:
                for c in cn:
                    report = r.get(f"{split}_report", {})
                    if c in report:
                        f1s[c].append(report[c]["f1-score"])
            means = [float(np.mean(f1s[c])) if f1s[c] else 0.0 for c in cn]
            stds = [float(np.std(f1s[c])) if f1s[c] else 0.0 for c in cn]
            bars = ax.bar(range(nc), means, yerr=stds, capsize=4, color="steelblue")
            ax.bar_label(
                bars, fmt="%.3f", label_type="edge", fontsize=11, padding=2, color="black"
            )
            ax.axhline(
                1 / nc,
                linestyle="--",
                color="red",
                linewidth=1.2,
                label="random",
            )
            ax.set_xticks(range(nc))
            ax.set_xticklabels(cn, rotation=45, ha="right", fontsize=tick_fs)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("F1 Score")
            ax.set_title(f"{alg_labels[i]} – {split}")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(exp_dir / "f1_scores.png", dpi=150)
    plt.close(fig)

    # Also save each algorithm's figures separately in its own subfolder.
    for i, mn in enumerate(alg_names):
        _save_clf_alg_figures(
            alg_dirs[i],
            alg_labels[i],
            data[mn],
            alg_class_names[mn],
            splits,
            show_random,
            tick_fs,
        )

    print(f"Plots saved to: {exp_dir}")


########################################################################
# TRAIN - CROSS-DOMAIN CLASSIFICATION
########################################################################

# Chance level of the binarised human-vs-machine problem. Both classes are
# balanced, so 0.50 is the meaningful midpoint every RATE figure is anchored to.
# TPR at a fixed FPR is not a rate metric in that sense -- its chance level is
# the FPR itself -- so each metric carries its own reference on its MetricSpec
# and only accuracy / macro F1 / AUROC use this one.
CD_CHANCE = CD_CHANCE_RATE

# Sequential ramps for the metrics whose reference sits near zero. A diverging
# map anchored at 1% would be white at one end of the scale and give a working
# detector the same colour as a broken one; "more ink = more of the thing" reads
# correctly there, with the reference drawn on the colour bar instead.
#
# Two of them, because the direction is not the same for every metric: more TPR
# is good and more FPR is bad, so a realized-FPR matrix drawn in the same blue as
# a TPR one would paint a threshold that blew up into 40% false accusations in
# exactly the colour that means "working" everywhere else on the page.
CD_SEQ_CMAP = "Blues"  # higher is better
CD_SEQ_CMAP_BAD = "Reds"  # higher is worse


def _cd_matrix(summary: dict, names: list[str], key: str) -> np.ndarray:
    """Source x target matrix of one metric; rows = trained on, cols = tested on."""
    return np.array(
        [[summary[s][t][key] for t in names] for s in names], dtype=float
    )


def _plot_cd_matrix(
    ax,
    mat: np.ndarray,
    std: np.ndarray,
    counts: np.ndarray,
    labels: list[str],
    title: str,
    cbar_label: str,
    fig,
    *,
    reference: float = CD_CHANCE,
    diverging: bool = True,
    higher_is_better: bool = True,
    extra: np.ndarray | None = None,
    extra_fmt: str = "",
    row_labels: list[str] | None = None,
    in_domain: np.ndarray | None = None,
) -> None:
    """One annotated source x target heatmap.

    ``diverging`` anchors a red/white/blue map at ``reference``: red = below,
    white = at it, blue = above. It is the right encoding for a metric whose
    reference sits mid-scale (accuracy, macro F1, AUROC at 0.5) and the wrong one
    for a metric whose reference is near zero, where a sequential ramp is used
    and the reference is marked on the colour bar only.

    ``extra`` adds a second annotated quantity to every cell -- the realized FPR
    beside a transported TPR, which must be read together: a TPR of 0.98 means
    nothing without knowing whether the threshold that bought it cost 1% or 40%
    false positives.

    The in-domain cells are boxed, because they answer a different question
    (a held-out split of what the row trained on) than the transfer ones. WHICH
    cells those are comes from ``in_domain``, never from the diagonal: a
    ``mode: lodo`` matrix boxes everything except it. ``row_labels`` names what
    each row trained on, which in that mode is not the corpus the row is keyed
    by. Both default to the ``cross`` reading.
    """
    n = len(labels)
    row_labels = list(row_labels) if row_labels is not None else list(labels)
    in_domain = (
        np.eye(n, dtype=bool) if in_domain is None else np.asarray(in_domain, dtype=bool)
    )
    if diverging:
        # Symmetric about the reference so the white midpoint lands on it, and
        # reversed for a lower-is-better metric so red always means "worse".
        half = max(reference, 1.0 - reference)
        im = ax.imshow(
            mat,
            vmin=reference - half,
            vmax=reference + half,
            cmap="RdBu" if higher_is_better else "RdBu_r",
        )
    else:
        im = ax.imshow(
            mat, vmin=0.0, vmax=1.0,
            cmap=CD_SEQ_CMAP if higher_is_better else CD_SEQ_CMAP_BAD,
        )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.ax.axhline(reference, color="black", linewidth=1.2, linestyle="--")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("TESTED ON  (test set)", fontweight="bold", labelpad=10)
    ax.set_ylabel("TRAINED ON  (train set)", fontweight="bold", labelpad=10)
    ax.set_title(title, pad=12)

    # 2px surface gap between cells so adjacent fills never touch.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            value = mat[i, j]
            # Ink stays readable on both poles of the diverging ramp, and on the
            # dark end of the sequential one.
            if not np.isfinite(value):
                color = "black"
            elif diverging:
                color = "white" if abs(value - reference) > 0.32 else "black"
            else:
                color = "white" if value > 0.55 else "black"
            text = (
                "n/a" if not np.isfinite(value) else f"{value*100:.1f}"
            ) + f"\n±{std[i, j]*100:.1f}"
            if extra is not None:
                text += (
                    f"\n{extra_fmt}n/a"
                    if not np.isfinite(extra[i, j])
                    else f"\n{extra_fmt}{extra[i, j]*100:.1f}"
                )
            text += f"\nn={int(counts[i, j])}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                linespacing=1.4,
                color=color,
            )
            if in_domain[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="black", linewidth=2.5,
                    )
                )


#: Characters per caption line, and the height one line occupies. The captions
#: under these figures explain what a metric claims, which is not optional
#: reading here -- a transported TPR is actively misleading without it -- so they
#: are wrapped to fit rather than trimmed to a length that happens to render.
CD_NOTE_WIDTH = 135
CD_NOTE_LINE_IN = 7.5 * 1.35 / 72  # one line at fontsize 7.5
CD_NOTE_MARGIN_IN = 0.10


def _wrap_note(*paragraphs: str) -> str:
    """Wrap caption paragraphs to ``CD_NOTE_WIDTH`` and join them with newlines."""
    lines: list[str] = []
    for para in paragraphs:
        if para:
            lines.extend(textwrap.wrap(para, CD_NOTE_WIDTH))
    return "\n".join(lines)


def _note_in(note: str) -> float:
    """Height in INCHES of the caption strip for ``note``.

    Separate from :func:`_note_strip` because a figure whose height is composed
    from its parts needs the inches before there is a figure to take a fraction
    of.
    """
    n_lines = note.count("\n") + 1 if note else 0
    return n_lines * CD_NOTE_LINE_IN + CD_NOTE_MARGIN_IN


def _note_strip(note: str, fig) -> float:
    """Figure-fraction height to reserve below the axes for ``note``."""
    return min(0.45, _note_in(note) / fig.get_figheight())


def _level_of(key: str) -> float:
    """The FPR level baked into a metric key: ``tpr_at_srcfpr_0.01`` -> ``0.01``."""
    return float(key.rsplit("_", 1)[-1])


def _metric_bar_positions(specs: list[MetricSpec]) -> np.ndarray:
    """x positions for one grouped bar per metric."""
    return np.arange(len(specs), dtype=float)


def _save_clf_cd_figures(out_dir: Path, payload: dict) -> None:
    """Save the transfer matrices and the in-domain vs transfer summary bars."""
    names: list[str] = payload["dataset_names"]
    label_of: dict = payload.get("dataset_labels", {})
    labels = [label_of.get(n, n) for n in names]
    summary = payload["summary"]
    cfg = payload["config"]
    alg_label = payload.get("alg_label", payload.get("alg_id", out_dir.name))

    # Which cells are in-domain, and what each row actually trained on. Under
    # `mode: lodo` neither is the diagonal / the corpus name, and a payload
    # written before `mode` existed falls back to exactly what it meant.
    in_domain = domain_mask(summary, names)
    lodo = bool(in_domain.size) and not in_domain.diagonal().any()
    src_label_of: dict = payload.get("source_labels") or label_of
    row_labels = [src_label_of.get(n, n) for n in names]

    n_runs = cfg["model"]["n_runs"]
    # No "chance = 50%" here any more: this subtitle is shared by every metric
    # figure, and chance is 50% for only three of them. Each figure states its own
    # reference on its colour bar and in its caption instead.
    subtitle = (
        f"{cfg['model']['ml_alg']} · {', '.join(cfg['features'])} · "
        f"{n_runs} seeds · grouped by {cfg['data']['group_col']!r}"
    )

    acc = _cd_matrix(summary, names, "accuracy_mean")
    counts = _cd_matrix(summary, names, "n")
    # Only the plotted metrics. The summary still HOLDS every key -- macro F1,
    # the target-calibrated TPR, the realized FPR, the 5% level -- and the
    # printed tables still report them; they simply do not each get a matrix.
    plotted = set(CD_PLOT_METRICS)
    specs = [m for m in present_metric_specs(summary, names) if m.key in plotted]

    size = max(7.0, 1.6 * len(names) + 3.0)

    # One matrix figure per plotted metric. Separate figures rather than panels
    # of one because each is a full n x n table that deserves its own
    # annotation, and because the caption each needs is different -- a
    # transported TPR means nothing without the (b) explanation, and accuracy
    # means nothing with it.
    for spec in specs:
        mat = _cd_matrix(summary, names, f"{spec.key}_mean")
        std = _cd_matrix(summary, names, f"{spec.key}_std")
        # A transported TPR is drawn with the FPR it actually cost, in the same
        # cell. The pair is the measurement; either half alone is misleading.
        extra_key = (
            transported_fpr_key(_level_of(spec.key))
            if spec.key.startswith("tpr_at_srcfpr_")
            else None
        )
        extra = (
            _cd_matrix(summary, names, f"{extra_key}_mean")
            if extra_key and f"{extra_key}_mean" in summary[names[0]][names[0]]
            else None
        )

        fig, ax = plt.subplots(figsize=(size, size * 0.85))
        _plot_cd_matrix(
            ax, mat, std, counts, labels,
            f"{alg_label} — "
            f"{'leave-one-domain-out' if lodo else 'cross-domain'}, "
            f"{spec.label} (%)\n{subtitle}",
            f"{spec.label} (dashed = {spec.reference*100:g}%)", fig,
            reference=spec.reference,
            diverging=spec.diverging,
            higher_is_better=spec.higher_is_better,
            extra=extra,
            extra_fmt="FPR ",
            row_labels=row_labels,
            in_domain=in_domain,
        )
        note = _wrap_note(
            (
                "Each row trained on every corpus BUT one. Boxed cells = "
                "in-domain: a held-out grouped split of that training pool. The "
                "one unboxed cell in each row is the corpus left out — the "
                "transfer number, and here it is the diagonal."
                if lodo else
                "Boxed diagonal = in-domain (held-out grouped test split of the "
                "SAME corpus). Off-diagonal = transfer to an unseen corpus."
            )
            + " Cells show mean ± std over seeds"
            + (", then the realized FPR" if extra is not None else "")
            + "; n = test rows. Target rows whose group was seen in training are "
            "excluded.",
            spec.note,
        )
        # The caption is a variable number of lines, so the strip reserved for it
        # is sized from the line count rather than from a constant that happens
        # to fit the longest one.
        strip = _note_strip(note, fig)
        fig.text(0.01, 0.01, note, fontsize=7.5, color="dimgray", va="bottom")
        fig.tight_layout(rect=(0, strip, 1, 1))
        fig.savefig(out_dir / f"cross_domain_{spec.filename}.png", dpi=150)
        plt.close(fig)

    # In-domain vs transfer, per train set and per test set. Accuracy only: this
    # figure asks "which corpus fails", which is a different question from "how
    # does this variant rank", and one metric per panel is what makes it legible.
    #
    # Driven off the mask rather than the diagonal, so under `lodo` the left
    # panel contrasts each pool's held-out split against the corpus it left out,
    # and the right panel reads "how well is this corpus detected by the model
    # that never saw it" — which is the LODO number, one per corpus.
    off_mask = ~in_domain
    with np.errstate(invalid="ignore"):
        diag = np.array([
            np.nanmean(acc[i][in_domain[i]]) if in_domain[i].any() else np.nan
            for i in range(len(names))
        ])  # trained here, tested on rows of the same pool
        out_mean = np.array([
            np.nanmean(acc[i][off_mask[i]]) if off_mask[i].any() else np.nan
            for i in range(len(names))
        ])  # trained here, tested on a corpus the pool did not hold
        in_mean = np.array([
            np.nanmean(acc[:, j][off_mask[:, j]]) if off_mask[:, j].any() else np.nan
            for j in range(len(names))
        ])  # trained elsewhere, tested here

    fig, axes = plt.subplots(1, 2, figsize=(max(11.0, 2.0 * len(names) + 6), 5.2))
    x = np.arange(len(names))
    w = 0.38

    b1 = axes[0].bar(
        x - w / 2, diag, w, color="steelblue",
        label=(
            "in-domain (held-out split of the pool)" if lodo
            else "in-domain (tested on itself)"
        ),
    )
    b2 = axes[0].bar(
        x + w / 2, out_mean, w, color="orange",
        label=(
            "transfer (the corpus left out)" if lodo
            else "transfer out (mean of other test sets)"
        ),
    )
    axes[0].bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    axes[0].bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(row_labels, rotation=25, ha="right")
    axes[0].set_xlabel("TRAINED ON  (train set)", fontweight="bold")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(
        "Per training pool: what does the held-out corpus cost?" if lodo
        else "Per train set: does it survive leaving its corpus?"
    )

    b3 = axes[1].bar(x, in_mean, w * 1.6, color="seagreen")
    axes[1].bar_label(b3, fmt="%.3f", fontsize=8, padding=2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_xlabel("TESTED ON  (test set)", fontweight="bold")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(
        "Per test set: the model that never saw it" if lodo
        else "Per test set: transfer in (mean of models trained elsewhere)"
    )

    for ax in axes:
        ax.axhline(CD_CHANCE, linestyle="--", color="red", linewidth=1.2, label="chance")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"{alg_label} — in-domain vs transfer"
        + (" (leave-one-domain-out)" if lodo else "")
        + f"\n{subtitle}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "cross_domain_summary.png", dpi=150)
    plt.close(fig)

    # The mechanism behind the transported TPR, where the payload kept the raw
    # scores to draw it from. Silently skipped where it did not.
    plot_cd_score_hist(out_dir, payload)


#: Human vs machine on the figures that draw the two CLASSES apart, rather than
#: a variant's summary numbers. Blue is human, red is machine, and the pairing is
#: fixed -- it is the convention a reader of detection figures already has, and
#: these panels are meant to be read at a glance.
#:
#: It deliberately does not match the dumbbell palette (steelblue = in-domain,
#: orange = transfer): those hues answer "which side of the matrix", these answer
#: "which class", and the two questions never appear on one axis.
CD_HUMAN_COLOR = "#3d6fb4"
CD_MACHINE_COLOR = "#c5392f"
#: The transported cut. Near-black: both hues on the panel are spoken for, and
#: red here means "machine", not "reference".
CD_CUT_COLOR = "0.12"
#: Bins per panel, shared across the grid.
CD_SCORE_BINS = 56


def _score_dist_range(
    cells: dict, thresholds: dict, names: list[str]
) -> tuple[float, float]:
    """x limits shared by every panel, so the panels can be compared at all.

    The 0.3 / 99.7 percentiles of the pooled scores, then widened to hold every
    threshold with room to spare. Scores outside the window are CLIPPED into the
    end bins by the caller rather than dropped -- what the figure is read for is
    the share of each class either side of the cut, and a dropped tail is exactly
    the mass a 1% operating point lives in.
    """
    pooled = np.concatenate(
        [
            np.asarray(cells[s][t]["scores"], dtype=float)
            for s in names
            for t in names
            if t in cells.get(s, {})
        ]
        or [np.array([0.0, 1.0])]
    )
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(pooled, [0.3, 99.7])
    cuts = [v for v in thresholds.values() if np.isfinite(v)]
    if cuts:
        lo, hi = min(lo, min(cuts)), max(hi, max(cuts))
    pad = 0.06 * max(hi - lo, 1e-9)
    return float(lo - pad), float(hi + pad)


def plot_cd_score_hist(out_dir: Path, payload: dict) -> None:
    """Where the transported cut falls in each corpus's two score distributions.

    This is the figure behind ``tpr_at_srcfpr_<level>``. Every other figure
    reports that number; this one shows the mechanism that produces it, because
    no aggregate can:

      * The cut is placed on the row's **calibration slice** (see
        ``cd_core.carve_calibration``), a grouped cut of the training pool that
        is disjoint from every panel drawn in the row -- so that exactly
        ``level`` of the HUMAN scores in that slice fall at or right of it.
        That is the whole definition of the operating point, and it is a
        construction on blue alone: the machine scores have no say in where the
        line goes. The calibration slice itself is never drawn as a panel.
      * On every panel of that row -- in-domain (tinted) and transfer alike --
        the same line is drawn again, unmoved. Nothing is re-fitted. What
        changes is the corpus under it.

    So a row reads left to right as one question: does the line still mean what
    it meant when it was placed? A detector whose score distributions merely
    shift -- ordering intact, AUROC intact -- shows up here as red mass sliding
    across a stationary line, which is the failure the transported TPR exists to
    catch and the one that accuracy and AUROC both miss.

    Bars are each class's share of ITSELF, not counts: the cells have very
    different row counts (a held-out split against a whole corpus), and only the
    share makes panels comparable. It also makes the two numbers printed on each
    panel readable off the picture -- red area right of the line is the TPR, blue
    area right of it is the realized FPR.

    Needs ``payload[SCORE_DIST_KEY]``, written by the runners for one seed. A
    payload from before that existed simply gets no figure: the scores cannot be
    recovered from a matrix.
    """
    dist = payload.get(SCORE_DIST_KEY) or {}
    cells = dist.get("cells") or {}
    thresholds = dist.get("thresholds") or {}
    names: list[str] = payload["dataset_names"]
    if not cells or not thresholds:
        return

    label_of: dict = payload.get("dataset_labels", {})
    labels = [label_of.get(n, n) for n in names]
    src_label_of: dict = payload.get("source_labels") or label_of
    row_labels = [src_label_of.get(n, n) for n in names]
    alg_label = payload.get("alg_label", payload.get("alg_id", out_dir.name))
    level = float(dist.get("level", 0.01))
    pct = f"{level * 100:g}%"
    n = len(names)

    # It is calibrated on the SOURCE's calibration slice (see
    # cd_core.carve_calibration) -- a grouped cut of the training pool that is
    # disjoint from every panel drawn here, in-domain or transfer alike, under
    # `cross` and `lodo` both. No panel is where the cut "came from" any more;
    # the mask below only says which panels are in-domain (tinted) so the cut
    # can be read against a same-corpus distribution as well as a transferred
    # one.
    in_domain = domain_mask(payload.get("summary") or {}, names)
    lodo = bool(in_domain.size) and not in_domain.diagonal().any()

    lo, hi = _score_dist_range(cells, thresholds, names)
    edges = np.linspace(lo, hi, CD_SCORE_BINS + 1)

    note = _wrap_note(
        (
            f"Each row's cut is fit on a calibration slice held out from every "
            f"panel — the tinted panels are the in-domain ones — where {pct} of "
            f"human scores fall right of it, then drawn unmoved onto the corpus "
            f"that row left out (the untinted panel)."
            if lodo else
            f"Each row's cut is fit on a calibration slice held out from every "
            f"panel, including the diagonal, where {pct} of human scores fall "
            f"right of it, then drawn unmoved across the row."
        )
        + " Bars are each class's share of itself: red right of the line is the "
        f"TPR, blue right of it the realized FPR. Seed {dist.get('seed', 0)}."
    )

    panel = 2.35
    title_in, legend_in = 0.72, 0.30
    note_in = _note_in(note)
    fig_w = max(9.0, panel * n + 2.0)
    fig_h = title_in + panel * n + 0.55 + legend_in + note_in
    fig, axes = plt.subplots(
        n, n, figsize=(fig_w, fig_h), sharex=True, sharey=True, squeeze=False
    )

    for i, source in enumerate(names):
        cut = float(thresholds.get(source, np.nan))
        for j, target in enumerate(names):
            ax = axes[i][j]
            cell = (cells.get(source) or {}).get(target)
            if not cell:
                ax.set_axis_off()
                continue
            scores = np.asarray(cell["scores"], dtype=float)
            y = np.asarray(cell["y"], dtype=int)

            # The panels that defined the cut. Tinted and named, so a reader
            # scanning a row can see where the line came from -- the whole
            # figure is the difference between those panels and the rest.
            if in_domain[i, j]:
                ax.set_facecolor("#eef2f7")
                ax.text(0.025, 0.94,
                        "cut set on these" if lodo else "cut set here",
                        transform=ax.transAxes,
                        ha="left", va="top", fontsize=8, style="italic",
                        color="0.35")
            # Everything at or right of the line is called machine. A faint band
            # rather than an arrow: it has to be legible under the bars.
            if np.isfinite(cut):
                ax.axvspan(cut, hi, color="0.90", zorder=0)

            for cls, color in ((0, CD_HUMAN_COLOR), (1, CD_MACHINE_COLOR)):
                values = scores[y == cls]
                if not values.size:
                    continue
                ax.hist(
                    np.clip(values, lo, hi), bins=edges,
                    weights=np.full(values.size, 1.0 / values.size),
                    color=color, alpha=0.55, edgecolor=color, linewidth=0.6,
                    zorder=2,
                )
            if np.isfinite(cut):
                ax.axvline(cut, color=CD_CUT_COLOR, linewidth=1.6, zorder=4)

            # Recomputed from the stored scores, not read off the summary, so the
            # numbers printed here cannot disagree with the bars beside them.
            # Unclipped: clipping is a drawing device and must not move a rate.
            for cls, color, name in (
                (1, CD_MACHINE_COLOR, "TPR"), (0, CD_HUMAN_COLOR, "FPR")
            ):
                values = scores[y == cls]
                rate = (
                    float((values >= cut).mean())
                    if values.size and np.isfinite(cut) else float("nan")
                )
                ax.text(
                    0.975, 0.94 if cls == 1 else 0.80,
                    f"{name} {rate:.3f}" if np.isfinite(rate) else f"{name}  n/a",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, fontfamily="monospace", color=color,
                    fontweight="bold" if cls == 1 else "normal",
                )

            if i == 0:
                ax.set_title(labels[j], fontsize=9.5, pad=6)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=9.5, fontweight="bold")
            ax.tick_params(labelsize=7.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    axes[0][0].set_xlim(lo, hi)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    handles = [
        Patch(facecolor=CD_HUMAN_COLOR, alpha=0.55, edgecolor=CD_HUMAN_COLOR,
              label="human"),
        Patch(facecolor=CD_MACHINE_COLOR, alpha=0.55, edgecolor=CD_MACHINE_COLOR,
              label="AI"),
        Line2D([], [], color=CD_CUT_COLOR, linewidth=1.6,
               label=f"cut at {pct} source FPR"),
        Patch(facecolor="0.90", edgecolor="none", label="called machine"),
    ]

    fig.suptitle(
        f"{alg_label} — score distributions and the transported {pct} FPR cut\n"
        "rows: CALIBRATED ON  ·  columns: TESTED ON"
        + ("  (leave-one-domain-out)" if lodo else "")
    )
    # Placed by hand: supxlabel's default y lands in the caption strip, and every
    # strip on this figure is already known in inches.
    fig.supxlabel(
        "decision score  (higher = more machine-like)", fontsize=9.5,
        y=(note_in + legend_in + 0.16) / fig_h,
    )
    fig.text(0.01, CD_NOTE_MARGIN_IN / fig_h, note, fontsize=7.5, color="dimgray",
             va="bottom")
    fig.subplots_adjust(
        top=1.0 - title_in / fig_h,
        bottom=(0.55 + legend_in + note_in) / fig_h,
        left=0.075, right=0.985, hspace=0.22, wspace=0.10,
    )
    fig.legend(
        handles=handles, fontsize=8.5, loc="lower center",
        bbox_to_anchor=(0.5, (note_in + 0.04) / fig_h),
        bbox_transform=fig.transFigure, ncol=4, frameon=False,
    )
    fig.savefig(
        out_dir / f"cross_domain_score_hist_{level_tag(level)}.png", dpi=150
    )
    plt.close(fig)


def plot_clf_cd_result(alg_dir: Path) -> None:
    """Plot one cross-domain result directory from its ``cross_domain.json``.

    Split out of ``plot_clf_cd_results`` so a run that produces one matrix per
    embedding can plot each one as soon as it finishes, instead of only at the
    end.
    """
    result_path = alg_dir / "cross_domain.json"
    if not result_path.exists():
        print(f"No cross_domain.json in {alg_dir}")
        return
    with open(result_path) as f:
        payload = json.load(f)
    _save_clf_cd_figures(alg_dir, payload)


def plot_clf_cd_results(exp_dir: Path) -> None:
    """Plot every cross-domain config in an experiment directory."""
    alg_dirs = sorted(
        d for d in exp_dir.iterdir() if d.is_dir() and (d / "cross_domain.json").exists()
    )
    if not alg_dirs:
        print(f"No cross-domain results found in {exp_dir}")
        return

    for alg_dir in alg_dirs:
        plot_clf_cd_result(alg_dir)

    print(f"Plots saved to: {exp_dir}")


# The hues every cross-domain figure is built from, and the only ones.
#
# Hue says WHICH SIDE of the matrix a number came from -- ● transfer
# (off-diagonal), ○ in-domain (diagonal) -- and nothing else. It does not also
# encode what produced the row: every variant's transfer dot is the same orange,
# a zero-shot baseline's included, because the axis is a comparison and a row
# drawn in its own colour reads as a different kind of measurement rather than a
# competitor on the same scale.
#
# What marks a baseline is the ★ on its row label, in one place. The old figure
# had it three ways at once -- an oversized ★ marker, a crimson hue, and the
# label mark -- with crimson also meaning "reference line" a few inches away.
CD_IN_DOMAIN_COLOR = "steelblue"
CD_TRANSFER_COLOR = "orange"
# The ★ on a baseline's row label and in the legend. Same orange as the dot it
# stands for, so the mark and the mark's data point read as one thing.
CD_BASELINE_COLOR = CD_TRANSFER_COLOR

# Zero-shot baseline families. The hatch is what tells them apart wherever a
# figure draws bars rather than markers; on the marker figures they are told
# apart by the family name on the legend's ★ entry.
CD_BASELINE_STYLES: dict[str, dict] = {
    "binoculars": {"label": "Binoculars baseline", "color": CD_BASELINE_COLOR,
                   "hatch": "/"},
    "llm_judge": {"label": "LLM Judge baseline", "color": CD_BASELINE_COLOR,
                  "hatch": "\\"},
}
# A baseline whose family cannot be told (an older payload, a third detector).
CD_BASELINE_FALLBACK = {
    "label": "zero-shot baseline", "color": CD_BASELINE_COLOR, "hatch": "-"
}
CD_HATCH_LINEWIDTH = 0.8  # bar outline *and* hatch stroke weight
# The hatch is only a "this is a baseline" flag, so it is drawn in translucent
# black: it sits behind the numbers instead of competing with them, and the
# family is carried by the hatch pattern plus the coloured ★ on the row label.
CD_HATCH_COLOR = (0.0, 0.0, 0.0, 0.35)
# Items trained on the shared feature set, and hybrids, are the same kind of
# thing (a trained classifier), so they share a colour and are told apart by
# marker shape -- ● vs ◆ -- not by hue, which is reserved for the baselines.
CD_ITEM_COLOR = "seagreen"
CD_HYBRID_COLOR = CD_ITEM_COLOR


def _cd_baseline_style(row: dict) -> dict:
    """Colour / hatch / legend label for one baseline row.

    ``baseline_kind`` is written by the run scripts; the label prefix is the
    fallback so comparison JSONs written before that key exist still plot with
    the right family instead of dropping to the generic style.
    """
    kind = row.get("baseline_kind")
    if kind not in CD_BASELINE_STYLES:
        label = str(row.get("label", ""))
        kind = next(
            (k for k in CD_BASELINE_STYLES if label.startswith(k)), None
        )
    return CD_BASELINE_STYLES.get(kind, CD_BASELINE_FALLBACK)


def _cd_baseline_legend_styles(order: list[dict], is_baseline) -> list[dict]:
    """The distinct baseline styles present, in the order the families are listed.

    Deduplicated so a family that appears on several rows still contributes
    exactly one legend entry, and stable so the left and right panels of a figure
    list the same families in the same order.
    """
    seen: dict[str, dict] = {}
    for row, baseline in zip(order, is_baseline):
        if not baseline:
            continue
        style = _cd_baseline_style(row)
        seen.setdefault(style["label"], style)
    ordering = [s["label"] for s in CD_BASELINE_STYLES.values()]
    return sorted(
        seen.values(),
        key=lambda s: ordering.index(s["label"]) if s["label"] in ordering else len(ordering),
    )


def _cd_row_mark(row: dict) -> str:
    """Prefix for a comparison row: ★ zero-shot baseline, ◆ hybrid, else none.

    A row that is neither was trained on exactly the shared feature set, which
    is the only case where its rank can be read straight off the figure.
    """
    if row.get("baseline"):
        return "★ "
    if row.get("hybrid"):
        return "◆ "
    return ""


def _num(value, default: float = float("nan")) -> float:
    """One aggregate as a float; a missing or unparseable value becomes NaN."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ensure_row_aggregates(rows: list[dict]) -> list[dict]:
    """Fill each row's per-metric aggregates from its matrix where they are absent.

    Comparison payloads written before the score metrics existed carry the
    accuracy and macro-F1 matrices but only the *unqualified* accuracy
    aggregates (``transfer_mean``, ``in_domain_mean``, ...). Recomputing from the
    matrix -- which is present, and is what those aggregates were derived from in
    the first place -- lets an old ``variant_comparison.json`` draw its accuracy
    and macro-F1 comparisons through exactly the same metric-driven path as a new
    one, instead of silently producing no figures at all.

    Rows are copied, so a replot never mutates the payload it was handed.
    """
    out = []
    for row in rows:
        row = dict(row)
        for spec in cell_metric_specs():
            if f"in_domain_{spec.key}_mean" in row or spec.key not in row:
                continue
            mat = np.asarray(row[spec.key], dtype=float)
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or not mat.size:
                continue
            diag, off = split_diag(mat)
            with np.errstate(invalid="ignore"):
                row[f"in_domain_{spec.key}_mean"] = float(np.nanmean(diag))
                row[f"in_domain_{spec.key}_std"] = float(np.nanstd(diag))
                row[f"transfer_{spec.key}_mean"] = (
                    float(np.nanmean(off)) if off.size else float("nan")
                )
                row[f"transfer_{spec.key}_std"] = (
                    float(np.nanstd(off)) if off.size else float("nan")
                )
                row[f"transfer_{spec.key}_worst"] = (
                    (float(np.nanmin(off)) if spec.higher_is_better
                     else float(np.nanmax(off)))
                    if off.size and np.isfinite(off).any() else float("nan")
                )
        out.append(row)
    return out


#: The metrics every cross-domain figure is drawn for, in reporting order, and
#: the only ones. This is a PLOTTING filter, not a measurement one: a run still
#: computes and writes every key ``cell_metric_specs()`` defines -- the realized
#: FPR, the 5% operating point -- into its JSONs and its printed tables. They
#: simply do not each get a figure. Nine metrics x two figure kinds plus a grid
#: was fifty-odd files per run, which buries the ones a run is actually read on.
#:
#: It is now exactly ``cd_core.HEADLINE_METRICS``, which is also what the
#: in-domain tables and figures report, so a variant is drawn under the same five
#: metrics whichever of the three training designs produced it. Read that
#: constant for why those five. Trimming the figure set again means trimming this
#: tuple, which no longer changes what any table says.
#:
#: One consequence worth knowing before wondering why two figures look identical:
#: on this binarised task the classes are near-balanced, so ``macro_f1`` tracks
#: ``accuracy`` closely and their two matrices will often agree to a decimal.
#: That is a property of the corpus, not a duplicated figure.
#:
#: A transported TPR is half a measurement: it is bought at a realized FPR, and
#: that FPR is not plotted. It is not lost either -- it is annotated inside the
#: cells of the per-variant TPR matrix (see :func:`_save_clf_cd_figures`), which
#: is where the pair has to be read together anyway.
CD_PLOT_METRICS: tuple[str, ...] = HEADLINE_METRICS

#: Fraction of the axis span kept clear on the right for the per-row numbers.
CD_GUTTER_FRAC = 0.26

#: The one spread these figures draw: ±1 SD of the aggregate over seeds.
#:
#: A Student-t interval on the mean used to be drawn beside it, and it was the
#: same number twice: at n seeds the interval's half-width is
#: ``t(0.975, n-1)/sqrt(n)`` x the SD, which is 1.24x at n=5. Two marks for one
#: quantity is cognitive load carrying no information, so the SD stands alone --
#: it answers "how far does this move if I re-seed", which is the question a
#: reader of a seed-averaged figure actually has, and it does not invite the
#: "the intervals overlap, so it is not significant" reading, which is not a
#: valid inference here.
#:
#: Thin and opaque rather than thick and translucent: a 5pt bar reads as the
#: datum and drowns the marker that IS the datum.
CD_SD_LINEWIDTH = 2.6

#: Marker size of a mean. One size and one shape for every row: a baseline used
#: to draw an oversized ★ here and now draws the same ● as everything else.
CD_MARKER_SIZE = 8.0


def _comparison_specs(rows: list[dict]) -> list[MetricSpec]:
    """The plotted metrics (:data:`CD_PLOT_METRICS`) these rows can be drawn under."""
    by_key = {m.key: m for m in cell_metric_specs()}
    return [
        by_key[key]
        for key in CD_PLOT_METRICS
        if key in by_key
        and any(np.isfinite(_num(r.get(f"transfer_{key}_mean"))) for r in rows)
    ]


def _finite_or_none(value) -> float | None:
    """One aggregate as a hashable value, with every non-finite mapped to None."""
    number = _num(value)
    return float(number) if np.isfinite(number) else None


def _row_metric_signature(row: dict, spec: MetricSpec) -> tuple:
    """Everything a comparison figure draws for one row under one metric.

    Two rows with the same signature produce the same bar, the same scatter point
    and the same matrix line, so drawing both says nothing the first did not.
    The mark and the baseline family are part of it: a ★ row and a plain row that
    happened to tie are still two different claims and must not merge.
    """
    matrix = row.get(spec.key)
    cells: tuple = ()
    if matrix is not None:
        try:
            flat = np.asarray(matrix, dtype=float).ravel()
        except (TypeError, ValueError):
            flat = np.array([], dtype=float)
        cells = tuple(_finite_or_none(v) for v in flat.tolist())
    aggregates = tuple(
        _finite_or_none(row.get(f"{where}_{spec.key}_{stat}"))
        for where in ("in_domain", "transfer")
        for stat in ("mean", "std", "worst")
    )
    return (
        _cd_row_mark(row),
        _cd_baseline_style(row)["label"] if row.get("baseline") else "",
        cells,
        aggregates,
    )


def _collapsed_label(rows: list[dict]) -> str:
    """Row label for a merged group.

    A score baseline family collapses to the family's own name -- ``binoculars``,
    not a list of the rows that merged -- but only when every merged row reads the
    same score column. The row *is* the Binoculars score, which is the whole
    content of these metrics; what was configured on top of it is a fact about
    the config and belongs in the caption, not in the name of a row it does not
    distinguish.

    Two *different* columns of one family are a different matter: Binoculars run
    with two scoring pairs is two detectors, and them coming out bit-identical is
    a finding, not a duplicate to be tidied away under the family name. Those keep
    every member's label, as anything else that merges does.
    """
    kinds = {r.get("baseline_kind") for r in rows}
    columns = {r.get("column") for r in rows}
    kind = kinds.pop() if len(kinds) == 1 else None
    if kind and len(columns) == 1:
        return str(kind)
    return " = ".join(str(r.get("label", "")) for r in rows)


def _collapse_identical_rows(rows: list[dict], spec: MetricSpec) -> list[dict]:
    """Rows that would draw identically under ``spec``, merged into one.

    The case this exists for is two rows that are the SAME score on the same
    seeds and splits. A threshold enters exactly one metric family -- the
    confusion matrix, so accuracy and macro F1. AUROC sweeps every threshold; a
    target-calibrated TPR@FPR reads its cutoff off the tested cell's own ROC; a
    transported TPR/FPR reads it off the source split's ROC at the nominal level.
    None of the three ever sees a declared cutoff, so any two rows over one score
    column come out bit-identical there, and drawing them twice reads as two
    references agreeing where there is only one. (A baseline family is one row per
    score column, so this catches an algorithm entered twice under two ids -- and
    two columns that really do hold the same score.)

    Equality is exact and taken from what the figure actually draws
    (:func:`_row_metric_signature`), not from the rows being baselines, so this
    also catches an algorithm entered twice under two ids -- and never merges
    rows that merely round to the same value.

    The merge is per metric, so a family that draws as one row on AUROC still
    draws as several on accuracy -- which is the only place a declared cutoff
    shows up, and therefore the only figure that can tell those rows apart.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(_row_metric_signature(row, spec), []).append(row)
    if all(len(g) == 1 for g in groups.values()):
        return rows

    out: list[dict] = []
    for group in groups.values():
        if len(group) == 1:
            out.append(group[0])
            continue
        merged = dict(group[0])
        # The label is the whole disclosure: a merged row prints every member's
        # name joined by " = ", so the rows that vanished are named where the
        # reader is already looking, not in a caption under the figure.
        merged["label"] = _collapsed_label(group)
        out.append(merged)
    return out


def _draw_cd_comparison_set(
    exp_dir: Path,
    rows: list[dict],
    alg_label: str,
    *,
    noun: str,
    subtitle: str,
    row_label,
) -> None:
    """One ranked figure per plotted metric, written flat into ``exp_dir``.

    ``<noun>_comparison_<metric>.png`` for each of :data:`CD_PLOT_METRICS` the
    payload actually holds. Three files, no sub-directory. The per-metric split
    is what makes a ranking disagreement findable: a variant can top the accuracy
    comparison and sit mid-table on TPR at 1% source FPR, and with only one of
    them there is nowhere that shows up.
    """
    rows = _ensure_row_aggregates(rows)
    specs = _comparison_specs(rows)
    if not specs:
        print(f"No comparable metrics in {exp_dir}")
        return

    for spec in specs:
        # Collapsed per metric, because whether two rows coincide is a property
        # of the metric: the score baselines are one row on AUROC and every
        # TPR@FPR reading, and separate rows on accuracy.
        _plot_cd_comparison(
            exp_dir, _collapse_identical_rows(rows, spec), alg_label,
            spec=spec, noun=noun, subtitle=subtitle, row_label=row_label,
        )
    print(
        f"Comparison plots saved to: {exp_dir} "
        f"({', '.join(s.short for s in specs)})"
    )



def _cd_gutter(
    ax,
    values: np.ndarray,
    spec: MetricSpec,
    *,
    gutter_frac: float = CD_GUTTER_FRAC,
    nbins: int = 6,
) -> float:
    """Set data-driven x limits with room on the right for the numbers.

    The old figure hardcoded ``0..1.18``. On AUROC that is the bug the whole
    panel suffers from: every trained classifier lives between 0.96 and 0.98, so
    the entire result -- and every error bar on it -- is three pixels wide. The
    limits follow the data instead, and the numbers move into a reserved gutter
    on the right rather than floating at the end of a bar.

    Two things it then has to get right, and used not to:

      * **The metric's ceiling is a real edge.** Every metric drawn here is a
        rate in [0, 1], so padding past 1.0 invents axis the data cannot reach --
        the within-dataset AUROC figure was ticking 1.005 and 1.020, values AUROC
        does not have. The padded limits are clamped into ``[spec.floor,
        spec.ceiling]`` wherever the data itself stays inside them.
      * **Perfect has to be labelled.** ``MaxNLocator`` keeps only the ticks
        falling inside the range, and on the suite AUROC figure that dropped
        every tick above 0.90: the in-domain circles sat at 0.999 with nothing to
        read them against, which is how a figure of near-perfect in-domain scores
        managed not to show where perfect was. The ceiling is forced into the
        tick list whenever the axis reaches it.

    Ticks still stop at the gutter -- one inside it would advertise a scale where
    there is only text -- and a faint grid is drawn from them, because over
    twenty rows the eye cannot carry a marker's position to the axis unaided.

    ``values`` is everything that must fit, the reference line included where it
    belongs -- the caller decides, because on the right half of a broken axis it
    does not (the reference sits in the stretch that was cut out).

    ``gutter_frac`` is the share of the axis kept clear on the right for the
    per-row numbers. Pass 0.0 on a panel that carries no numbers, so it spends
    its whole width on data. ``nbins`` is the tick budget -- lower it on a narrow
    panel, where six ticks collide and the forced ceiling tick makes the last
    gap tighter still.

    Returns the x of the gutter's left edge, so the caller can rule it off.
    """
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not finite.size:
        finite = np.asarray([spec.reference], dtype=float)
    lo, hi = float(finite.min()), float(finite.max())
    pad = max(0.015, 0.05 * (hi - lo))
    lo = max(lo - pad, spec.floor) if lo >= spec.floor else lo - pad
    hi = min(hi + pad, spec.ceiling) if hi <= spec.ceiling else hi + pad
    span = max(hi - lo, 1e-6)
    ax.set_xlim(lo, hi + gutter_frac * span)

    ticks = [t for t in MaxNLocator(nbins=nbins).tick_values(lo, hi) if lo <= t <= hi]
    if hi >= spec.ceiling and not any(abs(t - spec.ceiling) < 1e-9 for t in ticks):
        # Drop only what the ceiling would sit on top of, then pin it: an
        # unlabelled ceiling is the failure this branch exists for.
        step = (
            (max(ticks) - min(ticks)) / (len(ticks) - 1) if len(ticks) > 1 else span
        )
        ticks = [t for t in ticks if spec.ceiling - t > 0.55 * step] + [spec.ceiling]
    ax.set_xticks(sorted(ticks))
    _cd_xgrid(ax)
    return hi


def _cd_xgrid(ax) -> None:
    """A faint rule down from every x tick, behind the data.

    Not decoration: these figures put twenty rows against an axis whose
    interesting span is a few hundredths wide, and without a grid the reader has
    to carry a marker's position across empty paper to read it.
    """
    ax.grid(axis="x", color="0.92", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _cd_zebra(ax, y: np.ndarray, idx: int) -> None:
    """Alternating background band, so a long row list stays trackable."""
    if idx % 2 == 1:
        ax.axhspan(y[idx] - 0.5, y[idx] + 0.5, color=CD_SUITE_BAND_COLOR, zorder=0)


def _cd_by_seed(source: dict, stem: str) -> list[float]:
    """One ``<stem>_by_seed`` list off a comparison row or a per-dataset cell."""
    values = (source or {}).get(f"{stem}{BY_SEED_SUFFIX}")
    if not isinstance(values, list):
        return []
    return [_num(v) for v in values]


def _cd_seed_stats(source: dict, stem: str) -> tuple:
    """``(mean, sd, n)`` over the seeds behind one aggregate.

    ``n`` is the number of seeds, and those n numbers are the only datapoints any
    of these figures has. Each is one seed's whole aggregate recomputed -- that
    seed's mean over the diagonal, or over the off-diagonal -- NOT the spread of
    the individual cells around that mean, which is what the payload's ``_std``
    holds and what the old whisker showed. The two differ by about a factor of
    ten here, and only this one is an uncertainty on the number the run is ranked
    by.

    ``sd`` is the sample SD (``ddof=1``). It is NaN with fewer than two seeds --
    a payload written before the per-seed record existed cannot have one
    recovered from it -- and legitimately 0.0 when the seeds agree exactly, which
    is a fact about the variant and not a missing measurement. The two cases draw
    differently; see :func:`_cd_draw_estimate`.

    Falls back to the stored ``<stem>_mean`` for the marker, so a row without a
    per-seed record still plots -- it simply plots without a bar.
    """
    values = np.array(
        [v for v in _cd_by_seed(source, stem) if np.isfinite(v)], dtype=float
    )
    stored = _num((source or {}).get(f"{stem}_mean"))
    if values.size < 2:
        return (stored if np.isfinite(stored) else float("nan"),
                float("nan"), int(values.size))
    return float(values.mean()), float(values.std(ddof=1)), int(values.size)


def _cd_draw_estimate(
    ax,
    y: float,
    stats_tuple: tuple,
    *,
    color: str,
    filled: bool,
    scale: float = 1.0,
    zorder: int = 5,
) -> None:
    """One mean and its ±1 SD bar over seeds, on row ``y``.

    Shape is fixed: ``●`` filled for a transfer mean, ``○`` open for an in-domain
    one, on every row whatever produced it. ``color`` is the only thing that
    varies, and it is what says trained / baseline.

    ``scale`` shrinks the whole mark for a subordinate row -- the per-dataset
    lines under a block's mean line -- so the summary reads as the summary
    without needing a different symbol for it.

    Three cases, drawn apart on purpose:

      * **sd > 0** -- the bar.
      * **sd == 0** with seeds behind it -- a short upright tick at the mean.
        The seeds agreed exactly, which happens and means something: a zero-shot
        score does not depend on the seed, so a threshold-free metric on a fixed
        evaluation set comes out identical every time. Drawn rather than left
        blank so it cannot be misread as the third case.
      * **sd is NaN** -- fewer than two seeds on record. Marker alone.
    """
    mean, sd, _n = stats_tuple
    if not np.isfinite(mean):
        return
    if np.isfinite(sd) and sd > 0:
        ax.plot([mean - sd, mean + sd], [y, y], color=color,
                linewidth=CD_SD_LINEWIDTH * scale, solid_capstyle="butt",
                zorder=zorder - 1)
    elif np.isfinite(sd):
        ax.plot([mean, mean], [y - 0.22 * scale, y + 0.22 * scale], color=color,
                linewidth=CD_SD_LINEWIDTH * scale, solid_capstyle="butt",
                zorder=zorder - 1)
    ax.plot(
        mean, y, marker="o", markersize=CD_MARKER_SIZE * scale,
        markerfacecolor=color if filled else "white",
        markeredgecolor="black" if filled else color,
        markeredgewidth=0.7 * scale if filled else 1.8 * scale,
        zorder=zorder,
    )


def _cd_transfer_color(row: dict) -> str:
    """The hue of one row's transfer end: baseline family, or the trained orange."""
    return (
        _cd_baseline_style(row)["color"] if row.get("baseline")
        else CD_TRANSFER_COLOR
    )


def _cd_seed_legend(order: list[dict], n_seeds: int) -> list:
    """Legend entries shared by the ranked figure and the cross-dataset one.

    "Off-diagonal" is the right name for the transfer side under ``mode: cross``
    and the exactly wrong one under ``mode: lodo``, where the transfer cell IS
    the diagonal. The rows carry their mode, so the legend says which side each
    dot came from in the terms of the run that produced it.
    """
    from matplotlib.lines import Line2D

    lodo = any(r.get("mode") == "lodo" for r in order)

    def mark(color: str, filled: bool, label: str) -> Line2D:
        return Line2D(
            [], [], linestyle="none", marker="o", markersize=CD_MARKER_SIZE,
            markerfacecolor=color if filled else "white",
            markeredgecolor="black" if filled else color,
            markeredgewidth=0.8 if filled else 1.8, label=label,
        )

    handles = [
        mark(
            CD_TRANSFER_COLOR, True,
            "transfer (the corpus left out)" if lodo else "transfer (off-diagonal)",
        ),
        mark(
            CD_IN_DOMAIN_COLOR, False,
            "in-domain (held-out split of the training pool)" if lodo
            else "in-domain (diagonal)",
        ),
    ]
    # The ★, not a second dot: a baseline's transfer dot is the same orange as
    # everyone else's, so a marker entry here would just repeat the first one.
    # What needs explaining is the mark on its row label.
    is_baseline = [bool(r.get("baseline")) for r in order]
    handles += [
        Line2D([], [], linestyle="none", marker="*", color=style["color"],
               markeredgecolor="black", markeredgewidth=0.4, markersize=11,
               label=f"★ {style['label']} (zero-shot)")
        for style in _cd_baseline_legend_styles(order, is_baseline)
    ]
    handles.append(
        Line2D([], [], color="0.45", linewidth=CD_SD_LINEWIDTH,
               solid_capstyle="butt",
               label=f"±1 SD over {n_seeds} seeds" if n_seeds else "±1 SD over seeds")
    )
    if any(r.get("hybrid") for r in order):
        handles.append(
            Line2D([], [], linestyle="none", marker="D", color=CD_HYBRID_COLOR,
                   markersize=6, label="hybrid (extra features)")
        )
    return handles


def _cd_seed_note(n_seeds: int) -> str:
    """The caption both figures carry: one line, what the bar is.

    It says what the bar IS and nothing else. Everything the caption used to
    explain is either already visible or lives where it belongs: merged rows
    name themselves in their own label (``_collapsed_label``), a broken axis
    carries its break marks, and what a metric claims is on the per-variant
    matrix for that metric, which is the figure that has to carry it -- a
    transported TPR is misleading without its (b) note, and that note is on the
    figure where the transported TPR's cells actually are.
    """
    seeds = f"{n_seeds} seeds" if n_seeds else "the seeds"
    return (
        f"Bar = ±1 SD over {seeds}, of each seed's own mean. "
        "A tick instead of a bar: the seeds agreed exactly."
    )


def _cd_axis_break(main: np.ndarray, other: np.ndarray) -> tuple | None:
    """``((lo0, hi0), (lo1, hi1))`` when the rows want a broken x axis, else None.

    The case this exists for is AUROC with a zero-shot baseline on the figure.
    Every trained item sits between 0.966 and 0.975 and the baseline sits at
    0.317, so one axis spanning both gives the entire trained result about one
    percent of the width -- and its error bars rather less than a pixel, which is
    how the figure came to look as though the run had no variance in it.

    Both facts are worth keeping: how far below the baseline is, and what
    separates the items above it. A break keeps both at the cost of one
    convention every reader already knows. It is only taken when it earns
    itself -- when the items would otherwise be squeezed into under a fifth of
    the axis and there is a real empty gap to break across, rather than a
    continuum someone might read across.
    """
    main = main[np.isfinite(main)]
    other = other[np.isfinite(other)]
    if main.size == 0 or other.size == 0:
        return None
    full = max(float(max(main.max(), other.max()) - min(main.min(), other.min())), 1e-9)
    if float(np.ptp(main)) > 0.20 * full:
        return None
    # The gap has to be on one side: a baseline inside the cluster's range is not
    # an outlier and there is nothing to break across.
    if other.max() < main.min():
        gap = float(main.min() - other.max())
        segments = ((other.min(), other.max()), (main.min(), main.max()))
    elif other.min() > main.max():
        gap = float(other.min() - main.max())
        segments = ((main.min(), main.max()), (other.min(), other.max()))
    else:
        return None
    return segments if gap > 0.40 * full else None


def _cd_pad(lo: float, hi: float, frac: float = 0.08) -> tuple[float, float]:
    """A segment widened enough that its extreme markers are not on the spine."""
    pad = max(0.008, frac * max(hi - lo, 1e-9))
    return lo - pad, hi + pad


def _cd_draw_break_marks(left, right) -> None:
    """The two slashes that say the axis is cut, one pair on each facing spine."""
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=7, linestyle="none",
              color="0.35", mec="0.35", mew=1.0, clip_on=False)
    left.plot([1, 1], [0, 1], transform=left.transAxes, **kw)
    right.plot([0, 0], [0, 1], transform=right.transAxes, **kw)


def _plot_cd_ranked(
    out_dir: Path,
    order: list[dict],
    alg_label: str,
    *,
    spec: MetricSpec,
    noun: str,
    slug: str,
    subtitle: str,
    row_label,
) -> None:
    """One metric, one panel: every item's two means with their seed error bars.

    A row is a dumbbell -- ``●`` its transfer (off-diagonal) mean, ``○`` its
    in-domain (diagonal) mean, the link between them the generalisation gap --
    and each end carries ±1 SD of that mean over seeds.

    What is deliberately NOT here is the per-cell detail. An earlier version drew
    every source->target cell on the row, coloured by target, beside a paired
    per-cell difference against the best item. It was correct and it was too much
    to read. Every one of those numbers is still on the per-variant matrices this
    run writes; this figure answers the one question ahead of them -- how the
    items rank, and whether the gaps between them survive the seeds.

    The x axis follows the data, stops at the metric's ceiling, and breaks where
    a zero-shot baseline would otherwise flatten it (see :func:`_cd_axis_break`).
    It has to: the seed spread on AUROC is a few thousandths, and on an axis
    padded out to 1.18 for bar labels it would be invisible -- which is how the
    original figure came to look as though it had no variance at all.
    """
    key = spec.key
    y = np.arange(len(order))
    in_stats = [_cd_seed_stats(r, f"in_domain_{key}") for r in order]
    tr_stats = [_cd_seed_stats(r, f"transfer_{key}") for r in order]
    n_seeds = max((s[2] for s in tr_stats), default=0)

    def bounds(stats_list, mask) -> np.ndarray:
        return np.array(
            [
                b
                for keep, (mean, sd, _n) in zip(mask, stats_list)
                if keep
                for b in (mean, mean - sd, mean + sd)
            ],
            dtype=float,
        )

    plain = [not r.get("baseline") for r in order]
    base = [bool(r.get("baseline")) for r in order]
    main_vals = np.concatenate([bounds(in_stats, plain), bounds(tr_stats, plain)])
    base_vals = np.concatenate([bounds(in_stats, base), bounds(tr_stats, base)])
    all_vals = np.concatenate([main_vals, base_vals, [spec.reference]])
    segments = _cd_axis_break(main_vals, np.concatenate([base_vals, [spec.reference]]))

    note = _cd_seed_note(n_seeds)
    handles = _cd_seed_legend(order, n_seeds)
    plot_in = max(3.4, 0.46 * len(order) + 1.9)
    legend_in = _legend_strip_in(len(handles), ncol_max=4)
    note_in = _note_in(note)
    # The x-label is placed by hand rather than by set_xlabel, because a broken
    # axis is two axes and would otherwise get two labels for one scale. That
    # means reserving its strip here too.
    xlabel_in = 0.52
    title_in = 0.62  # the two-line suptitle
    fig_h = title_in + plot_in + xlabel_in + legend_in + note_in
    # Room for the row labels, which hang outside the axes as offset boxes and so
    # are invisible to any automatic layout. Sized to the longest one actually
    # drawn rather than to a guess, at roughly the width of a character at the
    # size _draw_cd_row_mark_label uses.
    label_in = min(
        5.0,
        max(1.1, 0.18 + 0.072 * max((len(row_label(r)) for r in order), default=0)),
    )

    fig_w = 13.0
    if segments:
        fig, axes = plt.subplots(
            1, 2, figsize=(fig_w, fig_h), sharey=True,
            gridspec_kw={"width_ratios": [1.0, 3.4], "wspace": 0.035},
        )
    else:
        fig, ax_one = plt.subplots(figsize=(fig_w, fig_h))
        axes = [ax_one]
    gutter_ax = axes[-1]

    def draw(ax) -> None:
        for idx, row in enumerate(order):
            _cd_zebra(ax, y, idx)
            in_mean, tr_mean = in_stats[idx][0], tr_stats[idx][0]
            if np.isfinite(in_mean) and np.isfinite(tr_mean):
                ax.plot([tr_mean, in_mean], [y[idx]] * 2, linestyle=":",
                        color=CD_SUITE_MEAN_COLOR, linewidth=1.2, zorder=2)
            _cd_draw_estimate(ax, y[idx], in_stats[idx], color=CD_IN_DOMAIN_COLOR,
                              filled=False, zorder=5)
            # Same ● whatever produced the row; the hue is what says zero-shot
            # baseline. A ★ here would make shape carry two meanings at once.
            _cd_draw_estimate(ax, y[idx], tr_stats[idx],
                              color=_cd_transfer_color(row), filled=True, zorder=7)
        ax.axvline(spec.reference, linestyle="--", color="red", linewidth=1.2,
                   zorder=1)

    for ax in axes:
        draw(ax)

    if segments:
        axes[0].set_xlim(*_cd_pad(*segments[0]))
        lo1, hi1 = _cd_pad(*segments[1])
        gutter_x = _cd_gutter(axes[1], np.array([lo1, hi1]), spec)
        axes[0].spines["right"].set_visible(False)
        axes[1].spines["left"].set_visible(False)
        axes[1].tick_params(left=False)
        axes[0].set_xticks(MaxNLocator(nbins=3).tick_values(*segments[0]))
        axes[0].set_xlim(*_cd_pad(*segments[0]))
        _cd_xgrid(axes[0])
        _cd_draw_break_marks(axes[0], axes[1])
    else:
        gutter_x = _cd_gutter(axes[0], all_vals, spec)
    gutter_ax.axvline(gutter_x, color="0.85", linewidth=0.8, zorder=1)

    for idx in range(len(order)):
        tr_mean, in_mean = tr_stats[idx][0], in_stats[idx][0]
        gutter_ax.text(
            0.995, y[idx],
            ("n/a" if not np.isfinite(tr_mean) else f"{tr_mean:.3f}")
            + " / "
            + ("n/a" if not np.isfinite(in_mean) else f"{in_mean:.3f}"),
            transform=gutter_ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=7.5, fontfamily="monospace",
        )
    gutter_ax.text(0.995, 0.995, "transfer / in-dom.", transform=gutter_ax.transAxes,
                   ha="right", va="top", fontsize=7.0, color="dimgray",
                   fontfamily="monospace")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([""] * len(order))
    for idx, row in enumerate(order):
        _draw_cd_row_mark_label(axes[0], row_label(row), row, y[idx])
    axes[0].set_ylim(-0.6, len(order) - 0.4)

    # No axes title: the figure title above already names the metric, and
    # repeating it over a single panel just pushes the plot down.
    fig.suptitle(
        f"{alg_label} — {noun} comparison, ranked by transfer: {spec.label}"
        f"\n{subtitle}"
    )
    fig.text(0.01, CD_NOTE_MARGIN_IN / fig_h, note, fontsize=7.5, color="dimgray",
             va="bottom")
    # subplots_adjust, not tight_layout: a broken axis carries break marks drawn
    # outside the axes with clip_on=False, and tight_layout refuses such a figure
    # ("Axes that are not compatible with tight_layout") -- it warns, does
    # nothing, and the strip below ends up printed over the plot. Every strip
    # here is already known in inches, so placing them is arithmetic.
    fig.subplots_adjust(
        top=1.0 - title_in / fig_h,
        bottom=(xlabel_in + legend_in + note_in) / fig_h,
        left=label_in / fig_w,
        right=0.985,
        wspace=0.035,
    )
    box = [ax.get_position() for ax in axes]
    centre = (box[0].x0 + box[-1].x1) / 2
    fig.legend(
        handles=handles, fontsize=7.5, loc="lower center",
        bbox_to_anchor=(centre, note_in / fig_h),
        bbox_transform=fig.transFigure,
        ncol=max(1, min(len(handles), 4)), frameon=False,
    )
    # One label under the pair, not one per segment: a break is still one scale.
    fig.text(
        centre, (note_in + legend_in) / fig_h,
        spec.label + ("" if spec.higher_is_better else "  (lower is better)")
        + f"    ·    reference {spec.reference*100:g}%",
        ha="center", va="bottom", fontsize=10,
    )
    fig.savefig(out_dir / f"{slug}_comparison_{spec.filename}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _plot_cd_comparison(
    out_dir: Path,
    rows: list[dict],
    alg_label: str,
    *,
    spec: MetricSpec,
    noun: str,
    subtitle: str,
    row_label,
) -> None:
    """Rank the items of one experiment under ONE metric, best transfer on top.

    Both the embedding and the algorithm comparison rank the same per-item schema
    (``label``, its in-domain / transfer aggregates and their per-seed record, a
    ``baseline`` flag), so only the wording and the output filename differ and
    those are the parameters:

      * ``spec`` -- which metric to draw: its label, its reference, its
        direction, its ceiling, and the key its aggregates are stored under. It
        also names the output file, ``<noun>_comparison_<metric>.png``.
      * ``noun`` -- what the varying axis is called ("feature set", "algorithm",
        "variant"); used in titles, filenames and legends.
      * ``subtitle`` -- the caller's one-line description (the sweeps differ:
        embeddings share an algorithm, algorithms share a feature set).
      * ``row_label(r)`` -- the y-axis label for item ``r``.

    Zero-shot baselines (``baseline: true``) draw the same ● in the same orange
    as every other row -- the figure is a comparison, and a row in its own hue
    reads as a different kind of measurement rather than a competitor on one
    scale. A ★ on the row label is what marks them. Hybrids (``hybrid: true``)
    carry a ◆ there for the same reason: they were trained on extra information
    (the Binoculars score on top of the shared feature set) and their rank should
    say so.

    This used to write a second figure beside the ranked one: every source->target
    cell of every item as a heatmap. It is gone. The per-variant matrices already
    hold every one of those numbers, one metric per file, annotated -- the
    comparison-level copy was the same table transposed, and having it meant every
    metric cost two files.
    """
    # Ranked by transfer: the point is which item survives leaving its corpus,
    # not which scores highest at home. y runs bottom-up, so ascending by
    # transfer puts the best row on top -- and "best" inverts for a realized FPR,
    # where the smallest value wins.
    order = sorted(
        rows,
        key=lambda r: _num(r.get(f"transfer_{spec.key}_mean")),
        reverse=not spec.higher_is_better,
    )
    _plot_cd_ranked(
        out_dir, order, alg_label, spec=spec, noun=noun,
        # The noun reads as prose in the titles ("per feature set"), so it is
        # only squashed where it has to be a filename.
        slug=noun.replace(" ", "_"),
        subtitle=subtitle, row_label=row_label,
    )


def plot_clf_cd_comparison(exp_dir: Path, payload: dict) -> None:
    """Compare the variants of one cross-domain experiment.

    ``payload`` is the ``variant_comparison.json`` written by
    ``epai/ai_detection/train/run_clf_cross_domain.py``: one entry per variant --
    a feature set crossed with an algorithm, plus any zero-shot baselines -- each
    holding its full source x target matrices and the in-domain / transfer means
    taken from them.

    Writes ``<noun>_comparison_<metric>.png`` for each of
    :data:`CD_PLOT_METRICS`, named after whichever axis the run actually varied
    (``payload["noun"]``: embedding, algorithm, or variant when both moved). The
    per-cell detail lives in the per-variant matrices beside them.
    """
    rows: list[dict] = payload.get("variants") or []
    if not rows:
        print(f"No results to compare in {exp_dir}")
        return

    # The feature width is worth a column only when it varies: a pure algorithm
    # sweep shares one feature set, so printing the same "(768d)" on every row
    # would be noise where on an embedding sweep it is half the point.
    show_dim = len({r.get("feature_dim") for r in rows}) > 1

    _draw_cd_comparison_set(
        exp_dir,
        rows,
        payload.get("alg_label", payload.get("alg_id", exp_dir.name)),
        noun=payload.get("noun", "variant"),
        subtitle=payload.get("subtitle", ""),
        row_label=lambda r: (
            f"{_cd_row_mark(r)}{r['label']}"
            + (f" ({r['feature_dim']}d)" if show_dim else "")
        ),
    )


def plot_clf_cd_emb_comparison(exp_dir: Path, payload: dict) -> None:
    """LEGACY: an ``embedding_comparison.json`` from before the runners merged.

    The embedding sweep is now one shape of :func:`plot_clf_cd_comparison`, which
    is what new runs write. This stays so ``replot_cross_domain.py`` can still
    redraw the result trees already in ``results/clf_cd_emb/``, whose payload
    keys its rows under ``embeddings`` and carries no ``subtitle``.
    """
    rows: list[dict] = payload["embeddings"]
    if not rows:
        print(f"No embedding results to compare in {exp_dir}")
        return

    names: list[str] = payload["dataset_names"]
    cfg = payload["config"]
    alg_label = payload.get("alg_label", payload.get("alg_id", exp_dir.name))
    subtitle = (
        f"{cfg['model']['ml_alg']} · {cfg['model']['n_runs']} seeds · "
        f"grouped by {cfg['data']['group_col']!r} · "
        f"{len(names)} corpora"
    )
    _draw_cd_comparison_set(
        exp_dir, rows, alg_label,
        noun="embedding", subtitle=subtitle,
        row_label=lambda r: (
            f"{_cd_row_mark(r)}{r['label']} ({r['feature_dim']}d)"
        ),
    )


def plot_clf_cd_alg_comparison(exp_dir: Path, payload: dict) -> None:
    """LEGACY: an ``algorithm_comparison.json`` from before the runners merged.

    The algorithm sweep is now one shape of :func:`plot_clf_cd_comparison`, which
    is what new runs write. This stays so ``replot_cross_domain.py`` can still
    redraw the result trees already in ``results/clf_cd_algs/``, whose payload
    keys its rows under ``algorithms`` and carries no ``subtitle``.
    """
    rows: list[dict] = payload["algorithms"]
    if not rows:
        print(f"No algorithm results to compare in {exp_dir}")
        return

    names: list[str] = payload["dataset_names"]
    cfg = payload["config"]
    alg_label = payload.get("alg_label", payload.get("alg_id", exp_dir.name))
    feats = ", ".join(cfg.get("features", []))
    if len(feats) > 60:
        feats = feats[:57] + "…"
    # Name the extra columns in the subtitle: the ◆ rows did not run on `feats`
    # alone, and the figure should say so without needing the JSON.
    aux = sorted({r["aux_column"] for r in rows if r.get("aux_column")})
    if aux:
        feats += f" (◆ also + {', '.join(aux)})"
    subtitle = (
        f"features: {feats} · {cfg['model']['n_runs']} seeds · "
        f"grouped by {cfg['data']['group_col']!r} · "
        f"{len(names)} corpora"
    )
    _draw_cd_comparison_set(
        exp_dir, rows, alg_label,
        noun="algorithm", subtitle=subtitle,
        row_label=lambda r: f"{_cd_row_mark(r)}{r['label']}",
    )



########################################################################
# TRAIN - CROSS-DATASET (SUITE) COMPARISON
########################################################################

# Written by epai/ai_detection/train/run_clf_cross_domain.py when its config
# declares `data.suites`. One level above plot_clf_cd_comparison: that one ranks
# the variants of ONE dataset by how well they leave their domain, this one puts
# those rankings side by side across every dataset the run covered.
#
# Nothing here draws a source x target cell. A cell belongs to one dataset's
# matrix and there is no meaningful transfer between, say, `essay` and
# `sociology` -- they are different corpora with different generators, and a
# pooled matrix over all twelve corpora would invent cells nobody trained for.
# What crosses datasets is the PAIR of summary numbers each matrix produces (its
# in-domain mean and its transfer mean), and that pair is what every figure below
# is built from.

#: The ends of a dumbbell are CD_IN_DOMAIN_COLOR / CD_TRANSFER_COLOR, defined
#: once with the rest of the palette above: a reader moving from a
#: within-dataset figure to this one must not re-learn them.
#:
#: The per-dataset connectors, and the mean's. The mean is the darker, heavier
#: one: it is an aggregate of the others and should not read as a sixth dataset.
CD_SUITE_LINE_COLOR = "0.72"
CD_SUITE_MEAN_COLOR = "0.35"
#: Alternating background band, so a block of dumbbells reads as one variant.
CD_SUITE_BAND_COLOR = "0.955"


def plot_clf_cd_suite_comparison(exp_dir: Path, payload: dict) -> None:
    """Compare the variants of a run across every dataset it covered.

    ``payload`` is the ``suite_comparison.json`` written by
    ``run_clf_cross_domain.py``: one entry per variant, each holding its
    per-dataset in-domain / transfer pair under ``per_suite`` and the unweighted
    aggregates over them at the top level (see :mod:`epai.utils.suite_utils`).

    One figure per plotted metric (:data:`CD_PLOT_METRICS`), written flat into
    ``exp_dir`` as ``suite_comparison_<metric>.png``: two panels sharing a y
    axis, the datasets on the left and their mean on the right, each row a
    dumbbell whose LENGTH is the generalisation gap. See
    :func:`_plot_suite_dumbbells` for why the two are drawn apart.
    """
    rows: list[dict] = payload.get("variants") or []
    suite_names: list[str] = payload.get("suite_names") or []
    if not rows or not suite_names:
        print(f"No cross-dataset results to compare in {exp_dir}")
        return

    specs = _comparison_specs(rows)
    if not specs:
        print(f"No comparable metrics in {exp_dir}")
        return

    show_dim = len({r.get("feature_dim") for r in rows}) > 1

    def row_label(r: dict) -> str:
        return (
            f"{_cd_row_mark(r)}{r['label']}"
            + (f" ({r['feature_dim']}d)" if show_dim else "")
        )

    for spec in specs:
        _plot_suite_dumbbells(
            exp_dir, rows, suite_names, payload.get("suite_labels", {}),
            payload.get("alg_label", payload.get("alg_id", exp_dir.name)),
            spec=spec, noun=payload.get("noun", "variant"),
            subtitle=payload.get("subtitle", ""), row_label=row_label,
        )
    print(
        f"Cross-dataset plots saved to: {exp_dir} "
        f"({', '.join(s.short for s in specs)})"
    )


def _suite_seed_stats(row: dict, suite: str | None, stem: str) -> tuple:
    """``(mean, sd, n)`` over seeds, for a dataset's line or for the mean line.

    ``suite`` None means the variant's own row -- the aggregate over every
    dataset, whose per-seed record was recombined across datasets when the
    payload was written (``suite_utils.mean_by_seed_across``), so entry ``i`` is
    still one seed and the bar means the same thing on both kinds of line.
    """
    source = row if suite is None else (row.get("per_suite") or {}).get(suite, {})
    return _cd_seed_stats(source, stem)


def _suite_row_order(rows: list[dict], spec: MetricSpec) -> list[dict]:
    """Variants best-transfer first, in the metric's own direction."""
    return sorted(
        rows,
        key=lambda r: _num(r.get(f"transfer_{spec.key}_mean")),
        reverse=spec.higher_is_better,
    )


def _draw_cd_row_mark_label(
    ax, label: str, row: dict, y: float, *, x_offset: float = -6.0
) -> None:
    """A row label whose ★ / ◆ is coloured and whose name is not.

    A tick label is a single Text artist and cannot hold two colours, so the
    label is drawn as a packed box right-aligned where the tick label would have
    been -- the same treatment the within-dataset ranked figure gives its rows,
    kept identical so the mark means the same thing in both.

    ``x_offset`` is where the box sits, in points left of the axis. The default
    puts it just outside; the cross-dataset figure pushes it further, past the
    dataset tick labels, so the two make two clean columns instead of colliding
    wherever a variant's name lands at the same height as a dataset's.
    """
    prefix = _cd_row_mark(row)
    mark = prefix.strip()
    parts = [
        TextArea(label[len(prefix):], textprops=dict(color="black", fontsize=8.5,
                                                     fontweight="bold"))
    ]
    if mark:
        mark_color = (
            _cd_baseline_style(row)["color"] if row.get("baseline")
            else CD_HYBRID_COLOR
        )
        parts.insert(0, TextArea(mark, textprops=dict(color=mark_color, fontsize=10)))
    ax.add_artist(
        AnnotationBbox(
            HPacker(children=parts, align="center", pad=0.0, sep=3.0),
            (0.0, y),
            xycoords=("axes fraction", "data"),
            xybox=(x_offset, 0.0),
            boxcoords="offset points",
            box_alignment=(1.0, 0.5),
            frameon=False,
            pad=0.0,
            annotation_clip=False,
        )
    )


def _plot_suite_dumbbells(
    out_dir: Path,
    rows: list[dict],
    suite_names: list[str],
    labels_of: dict,
    alg_label: str,
    *,
    spec: MetricSpec,
    noun: str,
    subtitle: str,
    row_label,
) -> None:
    """One metric, two panels: the datasets on the left, their mean on the right.

    A line is a dumbbell -- ``●`` transfer (off-diagonal), ``○`` in-domain
    (diagonal), the gap between them the distance the variant falls leaving its
    domain -- and each end carries ±1 SD over seeds.

    The split is the point. Both panels used to be one, with each variant's mean
    drawn as a heavier line at the top of its own block, and that asked the
    reader to do two incompatible things on one axis: read four datasets spread
    over the metric's whole range, and rank four variants whose means sit within
    two hundredths of each other. On a shared scale only the first is legible,
    which is how a figure meant to rank variants ended up unable to.

      * **Left, per dataset.** Where each variant lands in each dataset, and
        whether the datasets agree -- a block of parallel lines means the loss
        belongs to the variant, a block of scattered ones means it belongs to
        the dataset. This is the panel the outliers live on, so it takes the
        wide axis.
      * **Right, the mean.** The unweighted mean over datasets, one line per
        variant, on an axis fitted to just those means. This is the ranking, and
        it is readable because nothing else shares its scale.

    The two panels share their y axis, so a variant's block on the left and its
    summary line on the right sit at the same height, and the bold variant name
    in the left margin labels both. They do NOT share x: the right panel is a
    zoom, its own ticks say so, and its title repeats it.

    The mean's bar is not the datasets' spread pooled: the seeds are recombined
    across datasets first (``suite_utils.mean_by_seed_across``), so it is one
    seed's whole-suite score varying. The datasets disagreeing is the left panel.
    """
    order = _suite_row_order(rows, spec)
    key = spec.key
    n_suites = len(suite_names)

    # One block per variant: n_suites rows on the left, and on the right a single
    # line at the block's centre, which is also where the variant's name goes.
    block_gap = 1.2
    suite_y: list[list[float]] = []
    centre_y: list[float] = []
    cursor = 0.0
    for _ in order:
        ys = [cursor + i for i in range(n_suites)]
        suite_y.append(ys)
        centre_y.append(sum(ys) / len(ys))
        cursor += n_suites + block_gap

    stats_of = {
        (idx, suite, stem): _suite_seed_stats(row, suite, stem)
        for idx, row in enumerate(order)
        for suite in [None] + list(suite_names)
        for stem in (f"in_domain_{key}", f"transfer_{key}")
    }
    n_seeds = max((s[2] for s in stats_of.values()), default=0)

    from matplotlib.lines import Line2D

    handles = _cd_seed_legend(order, n_seeds) + [
        Line2D([], [], linestyle=":", color=CD_SUITE_LINE_COLOR, linewidth=1.6,
               label="gap: transfer → in-domain"),
        Line2D([], [], linestyle="--", color="red", linewidth=1.2,
               label=f"reference ({spec.reference*100:g}%)"),
    ]
    note = _cd_seed_note(n_seeds)

    # Two label columns in the left margin -- variant, then dataset -- so the
    # bold name can sit at its block's centre without landing on top of the tick
    # label of whichever dataset happens to be in the middle.
    suite_label_in = 0.072 * max(
        (len(str(labels_of.get(n, n))) for n in suite_names), default=0
    )
    variant_label_in = 0.075 * max((len(row_label(r)) for r in order), default=0)
    label_in = min(6.0, 0.30 + suite_label_in + variant_label_in)

    plot_in = max(4.0, 0.26 * (len(order) * (n_suites + block_gap)) + 1.2)
    # The two-line suptitle AND a row of panel titles under it.
    xlabel_in, title_in = 0.52, 1.15
    numbers_in = 1.00  # right margin holding the mean panel's printed values
    legend_in = _legend_strip_in(len(handles), ncol_max=3)
    note_in = _note_in(note)
    fig_h = title_in + plot_in + xlabel_in + legend_in + note_in
    fig_w = 13.5

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h), sharey=True,
        gridspec_kw={"width_ratios": [2.5, 1.0], "wspace": 0.045},
    )

    for idx, row in enumerate(order):
        transfer_color = _cd_transfer_color(row)
        if idx % 2 == 1:
            band = (suite_y[idx][0] - 0.6, suite_y[idx][-1] + 0.6)
            for ax in (ax_l, ax_r):
                ax.axhspan(*band, color=CD_SUITE_BAND_COLOR, zorder=0)

        # Left: one line per dataset. Right: the mean, at the block's centre.
        panels = [
            (ax_l, y, name, 0.78, CD_SUITE_LINE_COLOR, 1.2)
            for y, name in zip(suite_y[idx], suite_names)
        ] + [(ax_r, centre_y[idx], None, 1.0, CD_SUITE_MEAN_COLOR, 2.0)]

        for ax, y, suite, scale, line_color, line_w in panels:
            in_stats = stats_of[(idx, suite, f"in_domain_{key}")]
            tr_stats = stats_of[(idx, suite, f"transfer_{key}")]
            transfer, in_domain = tr_stats[0], in_stats[0]
            if not (np.isfinite(transfer) or np.isfinite(in_domain)):
                continue
            if np.isfinite(transfer) and np.isfinite(in_domain):
                # Dotted: a solid bar of this weight reads as an error bar, and
                # this is the opposite kind of thing -- a distance between two
                # estimates, not the spread of one.
                ax.plot([transfer, in_domain], [y, y], linestyle=":",
                        color=line_color, linewidth=line_w, zorder=2)
            _cd_draw_estimate(ax, y, in_stats, color=CD_IN_DOMAIN_COLOR,
                              filled=False, scale=scale, zorder=5)
            _cd_draw_estimate(ax, y, tr_stats, color=transfer_color,
                              filled=True, scale=scale, zorder=7)

        # Numbers on the mean only. Twenty rows of them was most of the ink on
        # this figure and none of its argument; a dataset's value is read off
        # its position, which is what the left panel is for.
        tr_mean = stats_of[(idx, None, f"transfer_{key}")][0]
        in_mean = stats_of[(idx, None, f"in_domain_{key}")][0]
        # Outside the axes, in reserved margin, not in a gutter cut out of the
        # scale: this panel is a zoom onto a couple of hundredths and cannot
        # afford to spend a quarter of its width on text.
        ax_r.text(
            1.03, centre_y[idx],
            (f"{tr_mean:.3f}" if np.isfinite(tr_mean) else "n/a")
            + (f" / {in_mean:.3f}" if np.isfinite(in_mean) else ""),
            transform=ax_r.get_yaxis_transform(), ha="left", va="center",
            fontsize=7.6, fontfamily="monospace", fontweight="bold",
            clip_on=False,
        )

    # Left panel holds every dataset value; right panel only the means, so it
    # gets its own scale. Neither is padded out to reach the other.
    def bounds(*, means: bool) -> list[float]:
        """Every drawn extremity on one panel: the mean row's, or the datasets'."""
        return [
            b
            for (_idx, suite, _stem), (mean, sd, _n) in stats_of.items()
            if (suite is None) is means
            for b in (mean, mean - sd, mean + sd)
        ]

    # A hair of room past the data so a marker sitting exactly on the ceiling is
    # not sliced in half by the spine. Neither panel reserves a text gutter: the
    # left carries no numbers and the right puts its own outside the axes.
    _cd_gutter(
        ax_l, np.array(bounds(means=False) + [spec.reference], dtype=float),
        spec, gutter_frac=0.035,
    )
    _cd_gutter(ax_r, np.array(bounds(means=True), dtype=float), spec,
               gutter_frac=0.05, nbins=4)
    ax_r.text(1.03, 1.0, "transfer / in-dom.", transform=ax_r.transAxes,
              ha="left", va="bottom", fontsize=7.0, color="dimgray",
              fontfamily="monospace", clip_on=False)

    # Only where it is on the panel: forcing the means' axis out to chance would
    # undo the zoom the panel exists for.
    for ax in (ax_l, ax_r):
        left, right = ax.get_xlim()
        if left <= spec.reference <= right:
            ax.axvline(spec.reference, linestyle="--", color="red", linewidth=1.2,
                       zorder=1)

    ax_l.set_yticks([y for ys in suite_y for y in ys])
    ax_l.set_yticklabels(
        [str(labels_of.get(n, n)) for _ in order for n in suite_names], fontsize=7.5
    )
    for idx, row in enumerate(order):
        _draw_cd_row_mark_label(
            ax_l, row_label(row), row, centre_y[idx],
            x_offset=-(suite_label_in * 72 + 12),
        )
    ax_l.set_ylim(cursor - block_gap, -1.0)  # inverted: best variant on top
    ax_r.tick_params(left=False)

    ax_l.set_title("per dataset", fontsize=9.5, pad=6)
    ax_r.set_title(
        f"mean over {n_suites} datasets, unweighted\n(axis fitted to the means)",
        fontsize=9.5, pad=6,
    )

    fig.suptitle(
        f"{alg_label} — {noun} across {n_suites} datasets, ranked by transfer: "
        f"{spec.label}\n{subtitle}"
    )
    fig.text(0.01, CD_NOTE_MARGIN_IN / fig_h, note, fontsize=7.5, color="dimgray",
             va="bottom")
    fig.subplots_adjust(
        top=1.0 - title_in / fig_h,
        bottom=(xlabel_in + legend_in + note_in) / fig_h,
        left=label_in / fig_w,
        right=1.0 - numbers_in / fig_w,
        wspace=0.045,
    )
    box = [ax.get_position() for ax in (ax_l, ax_r)]
    centre = (box[0].x0 + box[-1].x1) / 2
    fig.legend(
        handles=handles, fontsize=7.5, loc="lower center",
        bbox_to_anchor=(centre, note_in / fig_h), bbox_transform=fig.transFigure,
        ncol=max(1, min(len(handles), 3)), frameon=False,
    )
    fig.text(
        centre, (note_in + legend_in) / fig_h,
        spec.label + ("" if spec.higher_is_better else "  (lower is better)")
        + f"    ·    reference {spec.reference*100:g}%",
        ha="center", va="bottom", fontsize=10,
    )
    fig.savefig(out_dir / f"suite_comparison_{spec.filename}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


########################################################################
# TRAIN - IN-DOMAIN CLASSIFICATION (one corpus, held-out split)
########################################################################

# Written by epai/ai_detection/train/run_clf_in_domain.py. One corpus, one
# held-out split, so there is no matrix and no transfer axis -- the figures say
# what a variant scores at home and nothing about anywhere else. They reuse the
# cross-domain marks (★ baseline, ◆ hybrid) and the 50% chance line so the two
# figure sets can sit side by side without the symbols changing meaning.


def _id_rows(payload: dict) -> list[dict]:
    return payload.get("variants") or []


def plot_clf_id_result(out_dir: Path) -> None:
    """One variant's confusion matrix, from its ``in_domain.json``.

    Counts are the mean over seeds, so they are not integers; the annotation
    shows the row-normalised share as well, which is what actually reads -- 40
    false positives means nothing until you know whether the class had 50 rows
    or 5000.
    """
    result_path = out_dir / "in_domain.json"
    if not result_path.exists():
        print(f"No in_domain.json in {out_dir}")
        return
    with open(result_path) as f:
        payload = json.load(f)

    classes = payload.get("classes", ["human", "machine"])
    cm = np.array(payload["summary"]["confusion_mean"], dtype=float)
    s = payload["summary"]
    alg_label = payload.get("alg_label", out_dir.name)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    # Row-normalised: the colour answers "what share of this true class landed
    # here", which is comparable across an unbalanced corpus. The counts stay in
    # the text.
    rows_sum = cm.sum(axis=1, keepdims=True)
    shares = np.divide(cm, rows_sum, out=np.zeros_like(cm), where=rows_sum > 0)
    im = ax.imshow(shares, vmin=0.0, vmax=1.0, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Share of true class")

    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    # The confusion matrix is the held-out one; the train score is in the title
    # so the cell counts are read next to how much of the fit survived the split.
    gap = s.get("overfit_gap")
    gap_note = (
        f" · train {s['train_accuracy_mean']*100:.1f}% (gap {gap*100:+.1f})"
        if gap is not None
        else ""
    )
    ax.set_title(
        f"{alg_label}\nheld-out split · acc {s['accuracy_mean']*100:.1f}% "
        f"± {s['accuracy_std']*100:.1f}{gap_note}",
        fontsize=10,
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:.0f}\n{shares[i, j]*100:.1f}%",
                ha="center", va="center", fontsize=10,
                color="white" if shares[i, j] > 0.55 else "black",
            )
    fig.text(
        0.01, 0.01,
        f"Mean over {len(payload.get('per_seed', []))} seed(s); "
        f"{s['train_n']} train / {s['n']} test rows. In-domain only.",
        fontsize=7.5, color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_dir / "in_domain_confusion.png", dpi=150)
    plt.close(fig)

    _save_id_metric_figure(out_dir, payload)
    print(f"Plot saved to: {out_dir}")


def _save_id_metric_figure(out_dir: Path, payload: dict) -> None:
    """One variant's metrics on the fitted rows and the held-out ones.

    The same train-vs-test reading the confusion figure's title carries for
    accuracy, extended to every metric -- because the gap is not the same size
    for all of them. A model can lose almost nothing in accuracy across the split
    and lose a large share of its TPR at 1% FPR, since that operating point lives
    in the tail of the human distribution, which is exactly the part a fitted
    model has pushed furthest away on the rows it saw.

    The transported columns (``tprT`` / ``fprT``) have no train bar: the
    diagnostic ``train_res`` (see ``run_clf_in_domain.run_variant``) is scored
    without ``cal_thresholds``, so those keys are simply absent -- the fit
    rows are not the calibration slice a transported threshold comes from
    (``cd_core.carve_calibration``), so a train-side reading of them was never
    a meaningful third axis, just an unbuilt one.
    """
    s = payload["summary"]
    specs = [m for m in cell_metric_specs() if f"{m.key}_mean" in s]
    if not specs:
        return

    alg_label = payload.get("alg_label", out_dir.name)
    x = _metric_bar_positions(specs)
    test = np.array([float(s.get(f"{m.key}_mean", np.nan)) for m in specs])
    test_std = np.array([float(s.get(f"{m.key}_std", 0.0)) for m in specs])
    train = np.array([float(s.get(f"train_{m.key}_mean", np.nan)) for m in specs])
    train_std = np.array([float(s.get(f"train_{m.key}_std", 0.0)) for m in specs])

    fig, ax = plt.subplots(figsize=(max(5.5, 1.5 * len(specs) + 2.0), 6.4))
    w = 0.38
    b1 = ax.bar(x - w / 2, np.nan_to_num(train, nan=0.0), w, yerr=train_std,
                capsize=2, color=ID_TRAIN_COLOR, label="train (fitted on these rows)")
    b2 = ax.bar(x + w / 2, test, w, yerr=test_std, capsize=2,
                color=ID_TEST_COLOR, label="test (held out)")
    # Labelled from the real values so a missing train bar reads as blank rather
    # than as a confident 0.000.
    ax.bar_label(b1, labels=["" if not np.isfinite(v) else f"{v:.3f}" for v in train],
                 fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)

    for xi, spec in zip(x, specs):
        ax.plot([xi - w, xi + w], [spec.reference] * 2, color="red",
                linewidth=1.6, linestyle="--", zorder=4)
    ax.plot([], [], color="red", linewidth=1.6, linestyle="--",
            label="reference (chance, or the nominal FPR)")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            m.label.replace(" (", "\n(")
            + ("" if m.higher_is_better else "\n↓ lower is better")
            for m in specs
        ],
        fontsize=7.5,
    )
    ax.set_ylim(0, 1.34)
    ax.set_ylabel("Value")
    ax.legend(fontsize=8, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"{alg_label} — every metric, train vs held-out\n"
                 f"in-domain only · {len(payload.get('per_seed', []))} seed(s)")
    note = _wrap_note(
        "One corpus, split in two: both bars are the SAME data. This is an upper "
        "bound on a deployed detector, not an estimate of one.",
        "tpr@X (a) fits its threshold on the rows it scores. tprT@X / fprT@X (b) "
        "fit it to X% FPR on the TRAINING rows and carry it to the held-out ones — "
        "so they have no train bar, and fprT@X missing its nominal level here means "
        "the operating point is not stable even within one corpus."
        if any(m.key.startswith("tpr_at_") for m in specs) else "",
    )
    fig.text(0.01, 0.01, note, fontsize=7.5, color="dimgray", va="bottom")
    fig.tight_layout(rect=(0, _note_strip(note, fig), 1, 1))
    fig.savefig(out_dir / "in_domain_metrics.png", dpi=150)
    plt.close(fig)


# Train and test are the same measurement on two row sets, so they are one hue
# at two weights rather than two colours -- the eye should read "same thing,
# seen twice", and the gap between the bar ends is the quantity of interest.
ID_TRAIN_COLOR = "#a9c6e8"
ID_TEST_COLOR = "#2a78d6"

# Vertical space reserved below the axes, in INCHES -- the plotting area above
# scales with the number of variants, so an axes-relative offset does not stay
# clear of the x-label when it does. Sized to the content rather than to a worst
# case: a one-row legend gets one row of space. The legend part is shared with
# the cross-domain ranked figure, which composes its height the same way.
FIG_LEGEND_ROW_IN = 0.20  # one row of legend entries at fontsize 8
FIG_LEGEND_PAD_IN = 0.04
ID_FOOTNOTE_LINE_IN = 7.5 * 1.2 / 72  # one line at fontsize 7.5, 1.2 line spacing
ID_FOOTNOTE_LINES = 3
ID_MARGIN_IN = 0.04  # canvas edge to the footnote
FIG_GAP_IN = 0.07  # between x-label / legend / footnote


def _id_footnote_in() -> float:
    """Height of the footnote strip: the text, its bottom margin and one gap."""
    return ID_MARGIN_IN + ID_FOOTNOTE_LINES * ID_FOOTNOTE_LINE_IN + FIG_GAP_IN


def _legend_strip_in(n_entries: int, ncol_max: int = 3) -> float:
    """Height of the legend strip, for the number of entries actually drawn."""
    ncol = max(1, min(n_entries, ncol_max))
    rows = -(-n_entries // ncol)  # ceil
    return rows * FIG_LEGEND_ROW_IN + FIG_LEGEND_PAD_IN + FIG_GAP_IN


def _plot_split_composition(ax, composition: dict) -> None:
    """Train vs test share per value, for every ``report_by`` column.

    One row per (column, value), two bars: the share of the training rows and of
    the test rows that fall in it. Equal-length pairs mean the split is
    representative -- which is the only thing this panel is for, since
    ``GroupShuffleSplit`` groups but does not stratify and nothing else in the
    run would say.
    """
    labels, train, test = [], [], []
    for col, per_value in composition.items():
        for value, c in per_value.items():
            labels.append(f"{col}={value}")
            train.append(c["train_share"])
            test.append(c["test_share"])
    if not labels:
        return

    y = np.arange(len(labels))
    h = 0.38
    b1 = ax.barh(y + h / 2, train, h, color=ID_TRAIN_COLOR, label="train")
    b2 = ax.barh(y - h / 2, test, h, color=ID_TEST_COLOR, label="test")
    ax.bar_label(b1, fmt="%.1f%%", labels=[f"{v*100:.1f}%" for v in train],
                 fontsize=7, padding=2)
    ax.bar_label(b2, fmt="%.1f%%", labels=[f"{v*100:.1f}%" for v in test],
                 fontsize=7, padding=2)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(0.0, max(max(train + test) * 1.25, 0.1))
    ax.set_xlabel("Share of that half's rows")
    ax.set_title("Is the held-out half representative?", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", frameon=False)


def plot_clf_id_comparison(exp_dir: Path, payload: dict) -> None:
    """Rank the variants of one in-domain experiment, train against test.

    ``payload`` is the ``in_domain_comparison.json`` written by
    ``run_clf_in_domain.py``. The two bars per row are the SAME metric on the
    two row sets -- accuracy on the training rows and on the held-out ones -- so
    the distance between them is what the model failed to generalise even inside
    its own corpus. (They used to be accuracy and macro F1, which on a balanced
    binary task are the same number to three decimals: two identical bars saying
    one thing.)

    A second panel appears when the config sets ``report_by``: the composition of
    each half by that column, which is how you see whether a pooled run's test
    split actually resembles its training split.

    There is deliberately no in-domain-vs-transfer scatter here; with no transfer
    number there is nothing to plot against.
    """
    rows = _id_rows(payload)
    if not rows:
        print(f"No results to compare in {exp_dir}")
        return

    noun = payload.get("noun", "variant")
    slug = noun.replace(" ", "_")
    alg_label = payload.get("alg_label", payload.get("alg_id", exp_dir.name))
    order = sorted(rows, key=lambda r: r["accuracy_mean"])

    labels = [f"{_cd_row_mark(r)}{r['label']}" for r in order]
    test = np.array([r["accuracy_mean"] for r in order])
    test_std = np.array([r["accuracy_std"] for r in order])
    is_baseline = np.array([bool(r.get("baseline")) for r in order])
    # Payloads written before the train pass existed carry no train score. Drawing
    # a NaN bar would label it "nan"; drop the series instead so an older result
    # replots as what it is -- held-out accuracy alone.
    has_train = all(r.get("train_accuracy_mean") is not None for r in order)
    train = np.array([r.get("train_accuracy_mean", np.nan) for r in order])
    train_std = np.array([r.get("train_accuracy_std", 0.0) for r in order])

    # Merge the composition of every feature set: with one feature set (the usual
    # case) that is just its table; with several, identical splits collapse and
    # differing ones are worth seeing side by side.
    composition: dict = {}
    for _fs_id, per_col in (payload.get("split_composition") or {}).items():
        for col, per_value in per_col.items():
            composition.setdefault(col, {}).update(per_value)

    # The plotting area grows with the number of variants; the legend and the
    # footnote below it do not. Reserving that strip in INCHES rather than as a
    # fraction of the axes is the whole point -- a fraction that clears the
    # x-label on a tall chart lands on top of it on a short one.
    plot_h = max(2.6, 0.42 * len(order) + 1.0)
    if composition:
        n_comp = sum(len(v) for v in composition.values())
        plot_h = max(plot_h, 0.36 * n_comp + 1.0)

    # What the legend will hold: the bar series, the chance line, and one star per
    # baseline family present. Counted here rather than after plotting, so the
    # strip is sized to the rows that actually get drawn.
    n_legend = (2 if has_train else 1) + 1 + len(
        _cd_baseline_legend_styles(order, is_baseline)
    )
    legend_in, footnote_in = _legend_strip_in(n_legend), _id_footnote_in()
    fig_h = plot_h + legend_in + footnote_in

    if composition:
        fig, axes = plt.subplots(
            1, 2, figsize=(15.0, fig_h), gridspec_kw={"width_ratios": [1.45, 1.0]}
        )
        ax, ax_comp = axes
    else:
        fig, ax = plt.subplots(figsize=(9.5, fig_h))
        ax_comp = None

    y = np.arange(len(order))
    h = 0.38 if has_train else 0.55
    bars = []
    if has_train:
        b1 = ax.barh(y + h / 2, train, h, xerr=train_std, capsize=2,
                     color=ID_TRAIN_COLOR, label="train (fitted on these rows)")
        ax.bar_label(b1, fmt="%.3f", fontsize=7, padding=2)
        bars.append(b1)
    b2 = ax.barh(y - h / 2 if has_train else y, test, h, xerr=test_std, capsize=2,
                 color=ID_TEST_COLOR, label="test (held out)")
    ax.bar_label(b2, fmt="%.3f", fontsize=7, padding=2)
    bars.append(b2)
    # Baselines keep the cross-domain hatch, per family, so a starred row means
    # the same thing in both figure sets.
    for idx in np.nonzero(is_baseline)[0]:
        style = _cd_baseline_style(order[idx])
        for bar in (b[idx] for b in bars):
            bar.set_hatch(style["hatch"])
            bar.set_edgecolor(CD_HATCH_COLOR)
            bar.set_linewidth(CD_HATCH_LINEWIDTH)
    # The gap is the point of the panel, so it is written where it happens rather
    # than left to be eyeballed off two bar ends.
    for idx, r in enumerate(order):
        gap = r.get("overfit_gap")
        if gap is None or not np.isfinite(gap):
            continue
        ax.annotate(
            f"gap {gap*100:+.1f}", (max(train[idx], test[idx]), y[idx]),
            textcoords="offset points", xytext=(46, -3), fontsize=7,
            color="dimgray",
        )
    ax.axvline(CD_CHANCE, linestyle="--", color="red", linewidth=1.2, label="chance")
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(0.0, 1.22)
    ax.set_xlabel("Accuracy")
    ax.set_title("Ranked by held-out accuracy", fontsize=10)

    handles, _ = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D

    for style in _cd_baseline_legend_styles(order, is_baseline):
        handles.append(
            Line2D([], [], linestyle="none", marker="*", color=style["color"],
                   markersize=11, label=style["label"])
        )
    if ax_comp is not None:
        _plot_split_composition(ax_comp, composition)

    fig.suptitle(f"{alg_label} — {noun} comparison\n{payload.get('subtitle', '')}")
    # va="bottom" so the block grows UP from the anchor and stays inside the
    # reserved strip; the default baseline alignment would hang the 2nd and 3rd
    # lines off the bottom of the canvas.
    fig.text(
        0.01, ID_MARGIN_IN / fig_h, va="bottom",
        s="In-domain only: both bars are the SAME corpus, split in two. The gap is "
        "what did not generalise even within it.\nThe held-out score is an upper "
        "bound on a deployed detector, not an estimate of one — see "
        "run_clf_cross_domain.py for the transfer question.\n"
        "Error bars are the spread over seeds on overlapping splits, which "
        "understates the sampling error; they are not confidence intervals.",
        fontsize=7.5, color="dimgray",
    )
    # Lay the axes out inside what is left above the strip, THEN put the legend in
    # it: reading the axes position afterwards centres the legend under the panel
    # it describes rather than under the whole figure.
    fig.tight_layout(rect=(0, (legend_in + footnote_in) / fig_h, 1, 1))
    pos = ax.get_position()
    ax.legend(
        handles=handles, fontsize=8, loc="lower center",
        bbox_to_anchor=((pos.x0 + pos.x1) / 2, footnote_in / fig_h),
        bbox_transform=fig.transFigure,
        ncol=max(1, min(len(handles), 3)), frameon=False,
    )
    prev = plt.rcParams["hatch.linewidth"]
    plt.rcParams["hatch.linewidth"] = CD_HATCH_LINEWIDTH
    fig.savefig(exp_dir / f"{slug}_comparison.png", dpi=150, bbox_inches="tight")
    plt.rcParams["hatch.linewidth"] = prev
    plt.close(fig)

    _plot_id_comparison_metrics(
        exp_dir, list(reversed(order)), alg_label,
        noun=noun, slug=slug, subtitle=payload.get("subtitle", ""),
    )
    print(f"Comparison plot saved to: {exp_dir}")


#: Bar hues for the per-corpus columns of the in-domain corpus comparison. One
#: sequential family rather than a categorical one: the columns are the SAME
#: measurement on different corpora, not different measurements, and a
#: categorical palette would invite reading a hue as a kind. The pooled row is
#: told apart by its hatch instead -- it is not independent of the others, it is
#: the same documents again.
ID_CORPUS_COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c"]
ID_POOLED_HATCH = "//"


def plot_clf_id_corpus_comparison(exp_dir: Path, payload: dict) -> None:
    """Every variant's in-domain result per corpus, one figure per headline metric.

    ``payload`` is the ``corpus_comparison.json`` written by
    ``run_clf_in_domain.py`` when its config names several corpora. One figure is
    written per :data:`HEADLINE_METRICS` entry the rows carry, as
    ``in_domain_corpus_comparison_<metric>.png`` -- the same five metrics, in the
    same order, that the matrix runs draw, so an in-domain figure and a
    cross-domain one can be put side by side.

    The bars in one group are INDEPENDENT experiments -- a model trained and
    tested on that corpus alone -- so the group is a table, not a decomposition,
    and there is no transfer claim anywhere on these figures. The right panel is
    the train minus held-out accuracy gap, meaned over the corpora, repeated on
    every figure because it is the context a headline number is read in: a 99%
    means something different at a 0.4-point gap than at a 15-point one, whatever
    metric produced it.

    A pooled row (the whole file, from ``include_pooled``) is hatched: it is the
    same documents as the per-domain bars beside it, counted again, so its bar is
    a number to read and not evidence to add.
    """
    rows: list[dict] = payload.get("variants") or []
    names: list[str] = payload.get("corpus_names") or []
    if not rows or not names:
        print(f"No corpus comparison to draw in {exp_dir}")
        return

    specs = [
        spec for spec in headline_metric_specs()
        if any(f"{spec.key}_mean" in r for r in rows)
    ]
    if not specs:
        print(f"No headline metrics on the rows in {exp_dir}")
        return

    for spec in specs:
        _plot_id_corpus_metric(exp_dir, payload, rows, names, spec)


def _plot_id_corpus_metric(
    exp_dir: Path,
    payload: dict,
    rows: list[dict],
    names: list[str],
    spec: MetricSpec,
) -> None:
    """One corpus-comparison figure, for one metric. See the caller."""
    label_of: dict = payload.get("corpus_labels") or {}
    labels = [label_of.get(n, n) for n in names]
    alg_label = payload.get("alg_label", payload.get("alg_id", exp_dir.name))
    subtitle = payload.get("subtitle", "")
    noun = payload.get("noun", "variant")
    pooled = {
        n for n in names
        if str(label_of.get(n, n)).lower() in ("all", "all data") or n.lower() == "all"
    }

    # Ranked by THIS metric, so each figure answers its own question. A row the
    # metric is missing from sorts last rather than disappearing.
    order = sorted(
        rows,
        key=lambda r: (
            -_num(r.get(f"{spec.key}_mean"), -1.0)
            if spec.higher_is_better
            else _num(r.get(f"{spec.key}_mean"), 1e9),
            str(r.get("label", "")),
        ),
    )
    y = np.arange(len(order), dtype=float)
    n_bars = len(names)
    # The group fills 0.82 of a row whatever the corpus count, so a two-corpus
    # figure does not draw two very fat bars and a six-corpus one still leaves a
    # visible gutter between groups.
    h = 0.82 / n_bars

    fig_h = max(4.2, 0.46 * len(order) * n_bars / 3 + 2.6)
    fig, axes = plt.subplots(
        1, 2, figsize=(max(12.0, 1.1 * n_bars + 9.0), fig_h),
        gridspec_kw={"width_ratios": [3.0, 1.0]}, sharey=True,
    )

    per_corpus = [
        np.array(
            [
                _num(r.get("per_corpus", {}).get(name, {}).get(f"{spec.key}_mean"))
                for r in order
            ],
            dtype=float,
        )
        for name in names
    ]
    for k, name in enumerate(names):
        values = per_corpus[k]
        errs = np.array(
            [
                _num(r.get("per_corpus", {}).get(name, {}).get(f"{spec.key}_std"), 0.0)
                for r in order
            ],
            dtype=float,
        )
        offset = (n_bars - 1) / 2.0 - k
        bars = axes[0].barh(
            y + offset * h, np.nan_to_num(values, nan=0.0), h,
            xerr=np.nan_to_num(errs, nan=0.0),
            error_kw={"elinewidth": 0.8, "ecolor": "0.35"},
            color=ID_CORPUS_COLORS[k % len(ID_CORPUS_COLORS)],
            edgecolor="white", linewidth=0.4,
            hatch=ID_POOLED_HATCH if name in pooled else None,
            label=labels[k] + (" (pooled)" if name in pooled else ""),
        )
        axes[0].bar_label(
            bars, fontsize=7,
            labels=["" if not np.isfinite(v) else f"{v*100:.1f}" for v in values],
            padding=2,
        )

    # The mean over the columns, as a rule across the group rather than a bar: it
    # is a summary OF that group, and an extra bar would read as an extra corpus.
    # Drawn in data coordinates so it spans exactly the group whatever the corpus
    # count, which a fixed marker size would not.
    means = np.array([_num(r.get(f"{spec.key}_mean")) for r in order], dtype=float)
    for yi, m in zip(y, means):
        if np.isfinite(m):
            axes[0].plot([m, m], [yi - 0.44, yi + 0.44], color="black",
                         linewidth=1.5, zorder=5, solid_capstyle="butt")
    axes[0].plot([], [], color="black", linewidth=1.5, label="mean over corpora")

    # The reference is the metric's own, NOT 0.5 for everything: a random
    # detector scores 0.5 accuracy but only 1% TPR at 1% FPR, and drawing the
    # chance line at 50% on a TPR@1%FPR panel would make every working detector
    # look broken.
    axes[0].axvline(
        spec.reference, linestyle="--", color="red", linewidth=1.2,
        label=f"reference ({spec.reference*100:g}%)",
    )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [_cd_row_mark(r) + str(r.get("label", "")) for r in order], fontsize=9
    )
    axes[0].invert_yaxis()
    # Scaled to the data rather than pinned to [0.4, 1]: a transported TPR can
    # sit anywhere in [0, 1], and an axis that always starts at 0.4 either clips
    # it or flattens every difference worth seeing.
    finite = np.concatenate([v[np.isfinite(v)] for v in per_corpus] + [np.array([spec.reference])])
    lo, hi = _cd_pad(float(finite.min()), float(finite.max()))
    axes[0].set_xlim(max(spec.floor, lo), min(spec.ceiling + 0.02, hi + 0.02))
    axes[0].set_xlabel(
        f"{spec.label} — trained and tested on that corpus"
        + ("" if spec.higher_is_better else "   (lower is better)")
    )
    axes[0].set_title(f"In-domain {spec.label} per corpus", fontsize=10)
    _cd_xgrid(axes[0])

    gaps = np.array([_num(r.get("overfit_gap")) for r in order], dtype=float)
    gap_bars = axes[1].barh(
        y, np.nan_to_num(gaps, nan=0.0), 0.6, color="#d9a441", edgecolor="white",
        linewidth=0.4,
    )
    axes[1].bar_label(
        gap_bars, fontsize=7.5,
        labels=["" if not np.isfinite(v) else f"{v*100:.1f}" for v in gaps],
        padding=2,
    )
    axes[1].axvline(0.0, color="0.3", linewidth=1.0)
    axes[1].set_xlabel("train − held-out accuracy (points)")
    axes[1].set_title("Overfit gap, meaned over corpora", fontsize=10)
    _cd_xgrid(axes[1])

    fig.suptitle(
        f"{alg_label} — in-domain {spec.label}, every corpus\n{subtitle}"
    )
    note = _wrap_note(
        f"Each bar is a SEPARATE experiment: one {noun} trained and tested on "
        "that corpus alone, on a grouped held-out split. Nothing on this figure "
        "crosses corpora — no model is ever tested on a corpus it did not train "
        "on, so none of it is a transfer result. Error bars are ±1 SD over "
        "seeds, whose test halves overlap, so they understate the sampling "
        "error.",
        spec.note,
        (
            "The hatched bar is the whole file pooled: the same documents as the "
            "other bars in its group, counted again, so it is a number to read "
            "and not evidence to add to them."
            if pooled else ""
        ),
        "★ zero-shot baseline (no classifier trained) · ◆ hybrid: also saw an "
        "auxiliary column, so it is not ranked on equal information."
        if any(r.get("baseline") or r.get("hybrid") for r in order) else "",
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    # The caption is a variable number of lines and the legend a variable number
    # of rows, so the strip below the axes is measured rather than guessed -- and
    # clamped, because a very long caption on a short figure would otherwise ask
    # tight_layout for a negative plotting area.
    note_frac = _note_strip(note, fig)
    legend_frac = _legend_strip_in(len(handles)) / fig_h
    bottom = min(note_frac + legend_frac, 0.45)
    fig.legend(
        handles, legend_labels, fontsize=8, loc="lower center",
        ncol=min(len(handles), 6), frameon=False,
        bbox_to_anchor=(0.5, min(note_frac, bottom)),
    )
    fig.text(0.01, 0.01, note, fontsize=7.5, color="dimgray", va="bottom")
    fig.tight_layout(rect=(0, bottom, 1, 0.93))
    fig.savefig(
        exp_dir / f"in_domain_corpus_comparison_{spec.filename}.png", dpi=150
    )
    plt.close(fig)


def _plot_id_comparison_metrics(
    exp_dir: Path,
    order_desc: list[dict],
    alg_label: str,
    *,
    noun: str,
    slug: str,
    subtitle: str,
) -> None:
    """Every in-domain variant under every metric, held-out rows only.

    The in-domain counterpart of the cross-domain all-metric grid, and the same
    question: whether the accuracy ranking survives being asked at a low-FPR
    operating point. Only the held-out columns are drawn -- the train ones belong
    to the per-variant figure, where the pair is the point.
    """
    specs = [
        m for m in cell_metric_specs() if any(f"{m.key}_mean" in r for r in order_desc)
    ]
    if not specs or not order_desc:
        return

    mat = np.array(
        [[float(r.get(f"{m.key}_mean", np.nan)) for m in specs] for r in order_desc],
        dtype=float,
    )
    col_labels = [
        f"{m.short}\n({m.reference*100:g}%{'' if m.higher_is_better else ', ↓'})"
        for m in specs
    ]
    row_labels = [f"{_cd_row_mark(r)}{r['label']}" for r in order_desc]

    fig, ax = plt.subplots(
        figsize=(
            max(9.0, 1.05 * len(specs) + 4.5),
            max(4.5, 0.42 * len(order_desc) + 3.4),
        )
    )
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap=CD_SEQ_CMAP, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("Value")
    ax.set_xticks(range(len(specs)))
    ax.set_xticklabels(col_labels, fontsize=7.5, rotation=45, ha="right")
    ax.set_yticks(range(len(order_desc)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(specs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order_desc), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            ax.text(
                j, i, "n/a" if not np.isfinite(value) else f"{value*100:.1f}",
                ha="center", va="center", fontsize=7,
                color="white" if np.isfinite(value) and value > 0.55 else "black",
            )

    fig.suptitle(f"{alg_label} — every metric, per {noun} (held out)\n{subtitle}")
    note = _wrap_note(
        "Rows keep the held-out accuracy order, best first: a column that does not "
        "descend is a ranking that does not survive that metric.",
        "Each column carries its own reference; ↓ marks the columns where a higher "
        "value is worse. In-domain only — see run_clf_cross_domain.py for transfer.",
    )
    fig.text(0.01, 0.01, note, fontsize=7.5, color="dimgray", va="bottom")
    fig.tight_layout(rect=(0, _note_strip(note, fig), 1, 1))
    fig.savefig(exp_dir / f"{slug}_comparison_all.png", dpi=150)
    plt.close(fig)


########################################################################
# TRAIN - HYPER-PARAMETER SWEEP OVER CROSS-DOMAIN TRANSFER
########################################################################

# Written by epai/ai_detection/train/run_sweep_cross_domain.py. These figures
# describe the SEARCH -- what the space contains, what moved the objective, how
# much of the ranking is noise -- which the per-run cross-domain figures above
# cannot show: they see one model, not the two hundred that were rejected.
#
# Two hues, validated as a categorical pair on a light surface (worst-pair CVD
# ΔE 24.7, normal-vision 33.6, both >= 3:1 contrast). Blue is the condition the
# search optimised, orange the harder one it did not -- the same reading as the
# in-domain/transfer pair in the comparison figures above, so the two figure sets
# can sit side by side in a deck without the colours changing meaning.
SWEEP_DEV_COLOR = "#2a78d6"
SWEEP_HOLDOUT_COLOR = "#eb6834"
# Context marks: every trial, drawn behind the highlighted subset. Recessive
# enough to read as background, dark enough to still show the cloud's shape.
SWEEP_CONTEXT_COLOR = "#bdbdb8"
SWEEP_INK = "#0b0b0b"
SWEEP_INK_MUTED = "#52514e"
#: Panels per row in the per-axis marginal grid.
SWEEP_AXES_NCOLS = 4
#: An axis with at most this many distinct values is drawn as boxes per value;
#: anything wider is treated as continuous and binned.
SWEEP_DISCRETE_MAX = 6


def _sweep_scored(records: list[dict], stage: int | None = None) -> list[dict]:
    """Trials that produced an objective, optionally for one stage.

    Rejected trials carry ``metrics: None`` (or no ``accuracy`` at all, when the
    fit itself raised), so every figure below has to filter first rather than
    index into them.
    """
    rows = [r for r in records if r.get("objective") is not None and r.get("metrics")]
    if stage is not None:
        rows = [r for r in rows if int(r.get("stage", 1)) == stage]
    return rows


def _sweep_param_label(value) -> str:
    """One parameter value as a short axis tick. Mirrors the leaderboard's format."""
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(x) for x in value) + "]"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _sweep_varying_axes(rows: list[dict]) -> list[str]:
    """Search axes that actually took more than one value in these trials.

    Read off the records rather than the config's ``search`` block so this module
    never has to import the sweep script, and so an axis that was declared as a
    range but happened to draw one value is still dropped.
    """
    keys: dict[str, set] = {}
    for r in rows:
        for k, v in (r.get("params") or {}).items():
            keys.setdefault(k, set()).add(_sweep_param_label(v))
    return [k for k, values in keys.items() if len(values) > 1]


def _sweep_holdout_masks(
    names: list[str], holdout: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean source x target masks for (dev off-diagonal, holdout off-diagonal).

    A cell counts as holdout when *either* end is a held-out corpus: both
    ``dev -> holdout`` (a tuned model meeting an unseen corpus) and
    ``holdout -> dev`` (training on one) are honest, because the search tuned on
    neither. The diagonal is excluded from both -- an in-domain cell answers a
    different question and averaging it in would flatter every bar equally.
    """
    n = len(names)
    held = np.array([name in set(holdout) for name in names], dtype=bool)
    off = ~np.eye(n, dtype=bool)
    involves_holdout = held[:, None] | held[None, :]
    return off & ~involves_holdout, off & involves_holdout


def _sweep_order_labels(raw_of: dict[str, object]) -> list[str]:
    """Tick labels for one axis, ordered by the value they stand for.

    Numeric axes sort numerically (``32, 64, 128``, not the string order
    ``128, 32, 64``); layer-width lists sort by depth then total width, so an
    architecture axis reads from the linear probe outwards; anything else falls
    back to alphabetical.
    """
    labels = list(raw_of)
    values = [raw_of[label] for label in labels]

    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return sorted(labels, key=lambda label: float(raw_of[label]))
    if all(isinstance(v, (list, tuple)) for v in values):
        return sorted(
            labels,
            key=lambda label: (len(raw_of[label]), sum(raw_of[label]) if raw_of[label] else 0),
        )
    return sorted(labels)


def _sweep_span(values: np.ndarray, pad_frac: float = 0.07) -> tuple[float, float]:
    """Padded limits for one axis, from that axis's own data.

    Accuracies in a sweep are not spread over [0, 1]: in-domain routinely sits
    within a couple of points of 1.0 while transfer spreads over twenty. Limits
    taken from the union of both would leave the interesting axis unreadable.
    """
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-9:  # every trial identical on this axis
        return max(0.0, lo - 0.01), min(1.0, hi + 0.01)
    pad = (hi - lo) * pad_frac
    return max(0.0, lo - pad), min(1.0, hi + pad)


def _sweep_no_data(ax, message: str) -> None:
    """Leave a panel empty but explained, rather than silently blank or absent."""
    ax.text(
        0.5, 0.5, message, ha="center", va="center", fontsize=10,
        color=SWEEP_INK_MUTED, wrap=True, transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def _sweep_style_axis(ax) -> None:
    """Recessive grid and spines, so the marks carry the figure."""
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_sweep_holdout(
    out_dir: Path, comparison: dict, holdout: list[str]
) -> Path | None:
    """Transfer on the corpora the search tuned on, beside the corpora it never saw.

    The one figure that shows what the sweep actually bought. Every other number
    in a sweep is measured on the cells the objective was computed from, so a
    tuned model's margin there is part real and part the search having fitted
    those cells' noise. The held-out corpora took no part in the search at all,
    so their cells are the honest estimate -- and the distance between the two
    bars for the tuned row *is* the selection bias, drawn rather than argued.

    ``comparison`` is the ``algorithm_comparison.json`` of the ``--final`` run,
    which is the only run that scores the holdout. Returns None when there is no
    holdout to compare against, since then both bars would be the same cells.
    """
    rows: list[dict] = comparison.get("algorithms") or []
    names: list[str] = comparison["dataset_names"]
    if not rows:
        print(f"No algorithms in the final comparison under {out_dir}")
        return None
    if not holdout:
        print(
            f"{'holdout is empty'} — skipping the holdout figure: every cell is one "
            f"the search optimised, so there is nothing to compare against"
        )
        return None

    dev_mask, held_mask = _sweep_holdout_masks(names, holdout)
    if not held_mask.any():
        print(f"No off-diagonal holdout cells in {out_dir}; skipping holdout figure")
        return None

    # Cells rather than a single mean per bar: the mean is what gets quoted, but
    # a bar averaging two cells and a bar averaging fourteen are not the same
    # kind of claim, and the strip makes that visible instead of hiding it in an
    # error bar. It also sidesteps the trap that `transfer_std` in these payloads
    # is spread ACROSS CELLS, not across seeds.
    entries = []
    for row in rows:
        acc = np.asarray(row["accuracy"], dtype=float)
        entries.append(
            {
                "row": row,
                "dev": acc[dev_mask],
                "held": acc[held_mask],
            }
        )
    # Ranked by the honest number, not the optimised one: if the ordering changes
    # between the two bars, that reordering is the result.
    entries.sort(key=lambda e: float(np.mean(e["held"])))

    y = np.arange(len(entries))
    h = 0.38
    height = max(4.6, 0.62 * len(entries) + 2.6)
    fig, ax = plt.subplots(figsize=(12.5, height))

    dev_means = np.array([np.mean(e["dev"]) for e in entries])
    held_means = np.array([np.mean(e["held"]) for e in entries])
    b_dev = ax.barh(
        y + h / 2, dev_means, h, color=SWEEP_DEV_COLOR,
        edgecolor="white", linewidth=1.0,
        label=f"dev corpora — the search optimised these ({int(dev_mask.sum())} cells)",
    )
    b_held = ax.barh(
        y - h / 2, held_means, h, color=SWEEP_HOLDOUT_COLOR,
        edgecolor="white", linewidth=1.0,
        label=f"held out — first contact ({int(held_mask.sum())} cells)",
    )
    # Inside the bar, not past its end: the individual cell marks below sit around
    # the bar end by construction, so an edge label lands on top of them.
    ax.bar_label(b_dev, fmt="%.3f", fontsize=8, label_type="center", color="white")
    ax.bar_label(b_held, fmt="%.3f", fontsize=8, label_type="center", color="white")

    # Baselines keep the hatch and the ★ they carry everywhere else in this
    # module, so a zero-shot rule never reads as another tuned model.
    for idx, entry in enumerate(entries):
        if entry["row"].get("baseline"):
            style = _cd_baseline_style(entry["row"])
            for bar in (b_dev[idx], b_held[idx]):
                bar.set_hatch(style["hatch"])
                bar.set_edgecolor(CD_HATCH_COLOR)
                bar.set_linewidth(CD_HATCH_LINEWIDTH)

    for idx, entry in enumerate(entries):
        for offset, cells, color in (
            (h / 2, entry["dev"], SWEEP_DEV_COLOR),
            (-h / 2, entry["held"], SWEEP_HOLDOUT_COLOR),
        ):
            ax.scatter(
                cells,
                np.full(len(cells), idx + offset),
                s=16, facecolor="white", edgecolor=color,
                linewidth=1.2, zorder=4,
            )

    ax.axvline(
        CD_CHANCE, color=SWEEP_INK, linewidth=1.2, linestyle="--", zorder=1,
    )
    ax.annotate(
        "chance",
        xy=(CD_CHANCE, len(entries) - 0.45), xytext=(4, 0),
        textcoords="offset points", fontsize=8, color=SWEEP_INK_MUTED,
    )

    ax.set_yticks(y)
    ax.set_yticklabels([f"{_cd_row_mark(e['row'])}{e['row']['label']}" for e in entries])
    ax.set_xlabel("transfer accuracy (off-diagonal cells)")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(
        f"{comparison.get('alg_label', out_dir.name)}\n"
        f"what the sweep bought: tuned on {len(names) - len(holdout)} corpora, "
        f"measured on {len(holdout)} it never saw",
        fontsize=12, color=SWEEP_INK,
    )
    _sweep_style_axis(ax)

    # Legend and note below the axes: every bar here runs most of the width, so
    # there is no corner inside the plot a legend can occupy without covering data.
    fig.tight_layout()
    # Explicit proxies rather than the bar containers: the container's handle is
    # whichever bar happens to be first, and if that row is a baseline the legend
    # swatch inherits its hatch -- making the hatch look like part of what the
    # dev/holdout split means.
    fig.legend(
        handles=[
            Patch(facecolor=SWEEP_DEV_COLOR, edgecolor="white",
                  label=f"dev corpora — the search optimised these "
                        f"({int(dev_mask.sum())} cells)"),
            Patch(facecolor=SWEEP_HOLDOUT_COLOR, edgecolor="white",
                  label=f"held out — first contact ({int(held_mask.sum())} cells)"),
        ],
        loc="lower center", ncol=2, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.035),
    )
    fig.text(
        0.5, -0.075,
        f"held out: {', '.join(holdout)}  ·  hollow marks are the individual "
        f"source→target cells behind each mean  ·  ★ zero-shot baseline, "
        f"◆ hybrid (extra input)",
        ha="center", va="top", fontsize=8, color=SWEEP_INK_MUTED,
    )

    path = out_dir / "sweep_holdout.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sweep_tradeoff(
    out_dir: Path, records: list[dict], sweep_cfg: dict
) -> Path | None:
    """In-domain against transfer for every trial, split by normalisation.

    The scientific figure of the whole exercise: is transfer bought by giving up
    in-domain accuracy, or is the gap reducible for free? Each point is one
    hyper-parameter setting; the distance below the ``y = x`` line is that
    setting's ``gap``, so the shape of the cloud answers the question directly.

    Faceted rather than coloured. The panel axis is normally ``normalize``, which
    is the knob most likely to matter here -- a source-fitted scaler carries the
    training corpus's mean and scale onto every target, a per-row one cannot --
    and it has four values. Four categorical hues in a scatter cannot clear the
    colourblind separation floors on an all-pairs check, so the honest options
    are to fold two together or to facet; faceting loses nothing and makes the
    comparison a side-by-side rather than a legend lookup.
    """
    rows = _sweep_scored(records, stage=1)
    if not rows:
        print(f"No scored stage-1 trials in {out_dir}; skipping the trade-off figure")
        return None

    facet_key = "normalize" if any("normalize" in (r["params"] or {}) for r in rows) else None
    if facet_key is None:
        groups: list[tuple[str, list[dict]]] = [("all trials", rows)]
    else:
        buckets: dict[str, list[dict]] = {}
        for r in rows:
            buckets.setdefault(_sweep_param_label(r["params"].get(facet_key)), []).append(r)
        groups = sorted(buckets.items(), key=lambda kv: -float(np.mean(
            [x["metrics"]["transfer_mean"] for x in kv[1]]
        )))

    in_all = np.array([r["metrics"]["in_domain_mean"] for r in rows])
    tr_all = np.array([r["metrics"]["transfer_mean"] for r in rows])
    floor = float(sweep_cfg.get("objective", {}).get("min_in_domain", 0.0) or 0.0)

    # Each axis gets limits from its own data, shared across the facets. A single
    # pair of limits spanning both -- which would let y = x be drawn to scale --
    # is what the shape of these runs cannot afford: in-domain saturates near 1.0
    # while transfer spreads over twenty points, so a square panel spends almost
    # all of itself on empty space and collapses the cloud onto a vertical line.
    # The reference lines below are therefore drawn only when they fall in range,
    # and stated as a note when they do not.
    xlo, xhi = _sweep_span(in_all)
    ylo, yhi = _sweep_span(tr_all)
    draw_diagonal = max(xlo, ylo) < min(xhi, yhi)
    draw_floor = floor > 0 and xlo <= floor <= xhi

    fig, axes = plt.subplots(
        1, len(groups), figsize=(4.1 * len(groups) + 0.6, 4.9),
        sharex=True, sharey=True, squeeze=False,
    )
    best = max(rows, key=lambda r: r["objective"]) if sweep_cfg.get(
        "objective", {}
    ).get("direction", "max") == "max" else min(rows, key=lambda r: r["objective"])

    for ax, (label, group) in zip(axes[0], groups):
        # Every trial in grey behind this facet's subset: the comparison a reader
        # actually wants is "these against all of them", and repeating the full
        # cloud in each panel makes that a within-panel judgement.
        ax.scatter(
            in_all, tr_all, s=14, color=SWEEP_CONTEXT_COLOR,
            linewidth=0, zorder=2, label="all trials",
        )
        gx = np.array([r["metrics"]["in_domain_mean"] for r in group])
        gy = np.array([r["metrics"]["transfer_mean"] for r in group])
        ax.scatter(
            gx, gy, s=26, facecolor=SWEEP_DEV_COLOR, edgecolor="white",
            linewidth=0.8, zorder=3, label=f"{label}  (n={len(group)})",
        )
        # y = x: the vertical drop from this line is the in-domain/transfer gap.
        # Only when the two ranges actually overlap -- otherwise it is a line
        # along the panel edge that reads as a border.
        if draw_diagonal:
            ax.plot([max(xlo, ylo), min(xhi, yhi)], [max(xlo, ylo), min(xhi, yhi)],
                    color=SWEEP_INK, linewidth=1.0, linestyle="--", zorder=1)
        if draw_floor:
            ax.axvline(floor, color=SWEEP_HOLDOUT_COLOR, linewidth=1.2,
                       linestyle=":", zorder=1)
        if best in group:
            ax.scatter(
                [best["metrics"]["in_domain_mean"]],
                [best["metrics"]["transfer_mean"]],
                s=190, marker="*", facecolor=SWEEP_HOLDOUT_COLOR,
                edgecolor=SWEEP_INK, linewidth=0.7, zorder=5,
            )
        ax.set_title(
            f"{facet_key or 'trials'} = {label}\n"
            f"mean transfer {np.mean(gy):.3f}",
            fontsize=10, color=SWEEP_INK,
        )
        ax.set_xlabel("in-domain accuracy (diagonal)")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        _sweep_style_axis(ax)

    axes[0][0].set_ylabel("transfer accuracy (off-diagonal)")
    # Lay the panels out before anything is hung off the figure edges: the facet
    # titles are two lines each, so a suptitle placed inside the figure box lands
    # on top of them.
    fig.tight_layout()
    handles, _ = axes[0][0].get_legend_handles_labels()
    extra = [
        plt.Line2D([], [], color=SWEEP_HOLDOUT_COLOR, marker="*", linestyle="",
                   markersize=11, label="best trial"),
    ]
    if draw_diagonal:
        extra.insert(0, plt.Line2D([], [], color=SWEEP_INK, linestyle="--",
                                   label="y = x (no gap)"))
    if draw_floor:
        extra.append(
            plt.Line2D([], [], color=SWEEP_HOLDOUT_COLOR, linestyle=":",
                       label=f"in-domain floor ({floor:g})")
        )
    fig.legend(
        handles=handles[:1] + extra, loc="lower center", ncol=4, fontsize=9,
        frameon=False, bbox_to_anchor=(0.5, -0.10),
    )

    # The spans are the finding, not axis furniture: when in-domain is pinned and
    # transfer is not, there is no trade-off to tune and every point of the gap is
    # transfer's. Said in words because the reader cannot get it off axes whose
    # ranges differ by an order of magnitude.
    notes = [
        f"in-domain spans {in_all.min():.3f}–{in_all.max():.3f} "
        f"({(in_all.max() - in_all.min())*100:.1f} pts) · "
        f"transfer spans {tr_all.min():.3f}–{tr_all.max():.3f} "
        f"({(tr_all.max() - tr_all.min())*100:.1f} pts) · "
        f"gap {(in_all - tr_all).min()*100:.1f}–{(in_all - tr_all).max()*100:.1f} pts",
    ]
    if not draw_diagonal:
        notes.append(
            "note: the axes cover different ranges, so y = x is off-panel — "
            "in-domain is saturated and the whole spread is in transfer"
        )
    if floor > 0 and not draw_floor:
        notes.append(
            f"note: the in-domain floor ({floor:g}) never came near binding — "
            f"the lowest in-domain of any trial was {in_all.min():.3f}"
        )
    fig.text(
        0.5, -0.155, "\n".join(notes), ha="center", va="top",
        fontsize=8.5, color=SWEEP_INK_MUTED,
    )

    corpora = ", ".join(rows[0].get("dataset_names") or [])
    # Above the laid-out axes rather than inside them; bbox_inches="tight" pulls
    # it back into the saved image.
    fig.suptitle(
        f"{sweep_cfg.get('sweep_id', out_dir.name)} — in-domain vs transfer, "
        f"{len(rows)} settings\n"
        f"dev pool: {corpora} · stage 1, "
        f"{rows[0].get('n_runs', '?')} seed(s) per trial",
        fontsize=12, color=SWEEP_INK, y=1.10,
    )
    path = out_dir / "sweep_tradeoff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _sweep_axis_effect(rows: list[dict], key: str) -> tuple[float, list[tuple[str, np.ndarray]]]:
    """Group one axis's trials by value and return (effect size, groups).

    Effect size is the spread of the group means -- how far the objective moves
    between the best and worst setting of this axis alone. Deliberately the same
    statistic for a categorical axis and a binned continuous one, so the panels
    can be ordered against each other; a per-type measure (group spread here,
    a rank correlation there) would not be comparable.

    It is a MARGINAL, so it confounds: the mean for ``dropout`` in its top bin is
    taken over whatever the other ten axes happened to be doing in those trials.
    With a few hundred random draws over ten-odd axes that is suggestive, not an
    importance ranking, and the figure says so.
    """
    values = [_sweep_param_label(r["params"].get(key)) for r in rows]
    scores = np.array([r["objective"] for r in rows], dtype=float)

    # One raw value per tick label, so the ticks can be ordered by what they mean
    # rather than by how they render: sorting the labels as strings puts
    # batch_size at 128, 32, 64 and scatters hidden_dims arbitrarily.
    raw_of: dict[str, object] = {}
    for r in rows:
        raw_of.setdefault(_sweep_param_label(r["params"].get(key)), r["params"].get(key))
    distinct = _sweep_order_labels(raw_of)

    numeric = []
    for r in rows:
        v = r["params"].get(key)
        numeric.append(v if isinstance(v, (int, float)) and not isinstance(v, bool) else None)

    if len(distinct) > SWEEP_DISCRETE_MAX and all(v is not None for v in numeric):
        # Continuous: quantile bins, so each bin holds a comparable number of
        # trials instead of whatever a fixed-width grid happens to catch.
        arr = np.array(numeric, dtype=float)
        n_bins = min(5, max(2, len(rows) // 12))
        edges = np.quantile(arr, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        groups = []
        for i in range(len(edges) - 1):
            last = i == len(edges) - 2
            sel = (arr >= edges[i]) & (arr <= edges[i + 1] if last else arr < edges[i + 1])
            if sel.any():
                groups.append((f"{edges[i]:.3g}–{edges[i + 1]:.3g}", scores[sel]))
    else:
        groups = [
            (value, scores[np.array(values) == value])
            for value in distinct
        ]
        groups = [(label, s) for label, s in groups if len(s)]

    if len(groups) < 2:
        return 0.0, groups
    means = np.array([s.mean() for _, s in groups])
    return float(means.max() - means.min()), groups


def plot_sweep_axes(out_dir: Path, records: list[dict], sweep_cfg: dict) -> Path | None:
    """One panel per search axis, ordered by how far it moved the objective.

    The figure a reader of a sweep wants: of the knobs that were searched, which
    ones mattered. Boxes per value for a discrete axis, quantile bins for a
    continuous one, with the trials themselves jittered on top so a panel whose
    boxes rest on four points each cannot pass for a trend.
    """
    rows = _sweep_scored(records, stage=1)
    if not rows:
        print(f"No scored stage-1 trials in {out_dir}; skipping the per-axis figure")
        return None
    keys = _sweep_varying_axes(rows)
    if not keys:
        print(f"No varying search axis in {out_dir}; skipping the per-axis figure")
        return None

    metric = sweep_cfg.get("objective", {}).get("metric", "objective")
    effects = {key: _sweep_axis_effect(rows, key) for key in keys}
    keys = sorted(keys, key=lambda k: -effects[k][0])

    ncols = min(SWEEP_AXES_NCOLS, len(keys))
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), squeeze=False, sharey=True
    )
    flat = [ax for row in axes for ax in row]
    rng = np.random.default_rng(0)  # jitter only; fixed so the figure is stable

    for ax, key in zip(flat, keys):
        effect, groups = effects[key]
        positions = np.arange(len(groups))
        ax.boxplot(
            [s for _, s in groups], positions=positions, widths=0.55,
            showfliers=False, medianprops=dict(color=SWEEP_DEV_COLOR, linewidth=1.8),
            boxprops=dict(color=SWEEP_INK_MUTED, linewidth=0.9),
            whiskerprops=dict(color=SWEEP_INK_MUTED, linewidth=0.9),
            capprops=dict(color=SWEEP_INK_MUTED, linewidth=0.9),
        )
        for pos, (_, s) in zip(positions, groups):
            ax.scatter(
                pos + rng.uniform(-0.16, 0.16, len(s)), s,
                s=11, color=SWEEP_CONTEXT_COLOR, linewidth=0, zorder=1,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [label for label, _ in groups], rotation=30, ha="right", fontsize=8
        )
        ax.set_title(f"{key}   (Δ {effect*100:.2f} pts)", fontsize=10, color=SWEEP_INK)
        _sweep_style_axis(ax)

    for ax in flat[len(keys):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel(metric)

    fig.suptitle(
        f"{sweep_cfg.get('sweep_id', out_dir.name)} — {metric} per search axis, "
        f"{len(rows)} settings, panels ordered by effect\n"
        f"MARGINAL effects: each panel averages over every other axis, so read "
        f"these as suggestive, not as an importance ranking",
        fontsize=12, color=SWEEP_INK,
    )
    path = out_dir / "sweep_axes.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sweep_reliability(
    out_dir: Path, records: list[dict], sweep_cfg: dict
) -> Path | None:
    """Was the trial budget enough, and how much of the ranking was noise.

    Two panels, both about trusting the leaderboard rather than reading it:

    * **left** -- the running best against trial index. Random search draws
      independently, so this is the running maximum of i.i.d. samples: a curve
      still climbing at the last trial means the budget, not the space, was the
      limit. It says nothing about the search having *learned* anything.
    * **right** -- each survivor's stage-1 score (one seed) against its stage-2
      score (more seeds), with ``y = x``. Points falling below the line are the
      cost of ranking on one seed, and the mean drop is the amount of the stage-1
      margin that was seed noise. This is the same effect the corpus holdout
      exists to absorb, at a smaller scale and cheap to measure.
    """
    rows = _sweep_scored(records, stage=1)
    if not rows:
        print(f"No scored stage-1 trials in {out_dir}; skipping the reliability figure")
        return None

    metric = sweep_cfg.get("objective", {}).get("metric", "objective")
    maximise = sweep_cfg.get("objective", {}).get("direction", "max") == "max"
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))

    # --- left: running best -------------------------------------------------
    scores = np.array([r["objective"] for r in rows], dtype=float)
    order = np.arange(1, len(scores) + 1)
    running = np.maximum.accumulate(scores) if maximise else np.minimum.accumulate(scores)
    axes[0].scatter(
        order, scores, s=14, color=SWEEP_CONTEXT_COLOR, linewidth=0,
        zorder=2, label="each trial",
    )
    axes[0].plot(
        order, running, color=SWEEP_DEV_COLOR, linewidth=2.0, zorder=3,
        label=f"best so far ({'max' if maximise else 'min'})",
    )
    axes[0].set_xlabel("trial (order run)")
    axes[0].set_ylabel(metric)
    axes[0].set_title(
        f"Did {len(rows)} trials exhaust the space?\n"
        f"best {running[-1]:.4f}, first reached at trial "
        f"{int(np.argmax(running == running[-1]) + 1)}",
        fontsize=11, color=SWEEP_INK,
    )
    _sweep_style_axis(axes[0])
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.95)

    # --- right: one seed against several ------------------------------------
    stage2 = {r["trial_id"]: r for r in _sweep_scored(records, stage=2)}
    paired = [(r, stage2[r["trial_id"]]) for r in rows if r["trial_id"] in stage2]
    if not paired:
        _sweep_no_data(
            axes[1],
            "stage 2 has not run yet.\n\nOnce the survivors are re-run on more "
            "seeds, this panel shows how much\nof each trial's stage-1 margin "
            "was seed noise.",
        )
        axes[1].set_title("One seed vs several", fontsize=11, color=SWEEP_INK)
    else:
        s1 = np.array([a["objective"] for a, _ in paired], dtype=float)
        s2 = np.array([b["objective"] for _, b in paired], dtype=float)
        lo = min(s1.min(), s2.min()) - 0.01
        hi = max(s1.max(), s2.max()) + 0.01
        axes[1].plot([lo, hi], [lo, hi], color=SWEEP_INK, linewidth=1.0,
                     linestyle="--", zorder=1, label="y = x (seeds agree)")
        axes[1].scatter(
            s1, s2, s=34, facecolor=SWEEP_DEV_COLOR, edgecolor="white",
            linewidth=0.8, zorder=3, label=f"survivor (n={len(paired)})",
        )
        drop = float(np.mean(s1 - s2))
        # How many survivors got worse, not just the average: a mean drop is
        # consistent with half the trials rising, whereas "n of n below the line"
        # is the statement that ranking on one seed was systematically optimistic.
        n_below = int((s2 < s1).sum())
        axes[1].set_xlabel(f"{metric} — stage 1 ({paired[0][0].get('n_runs', '?')} seed)")
        axes[1].set_ylabel(f"{metric} — stage 2 ({paired[0][1].get('n_runs', '?')} seeds)")
        axes[1].set_xlim(lo, hi)
        axes[1].set_ylim(lo, hi)
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].set_title(
            f"One seed vs several\n"
            f"mean change on re-running {-drop:+.4f} · "
            f"{n_below} of {len(paired)} fell",
            fontsize=11, color=SWEEP_INK,
        )
        _sweep_style_axis(axes[1])
        axes[1].legend(loc="upper left", fontsize=9, framealpha=0.95)

    holdout = sweep_cfg.get("holdout") or []
    note = (
        f"holdout: {', '.join(holdout)}" if holdout
        else "holdout is EMPTY — the winning score is the one the search optimised"
    )
    # Panels first, then the suptitle above them -- see plot_sweep_tradeoff.
    fig.tight_layout()
    fig.suptitle(
        f"{sweep_cfg.get('sweep_id', out_dir.name)} — how much to trust the "
        f"leaderboard\n{note}",
        fontsize=12, color=SWEEP_INK, y=1.09,
    )
    path = out_dir / "sweep_reliability.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def sweep_axis_effects(records: list[dict]) -> list[tuple[str, float]]:
    """Each varying search axis with its marginal effect, strongest first.

    The ordering behind ``plot_sweep_axes``, exposed so a report can print the
    same ranking the figure draws instead of recomputing it differently. Same
    caveat as the figure: these are marginals over a random sample, so they
    suggest which knobs matter rather than measuring it.
    """
    rows = _sweep_scored(records, stage=1)
    if not rows:
        return []
    return sorted(
        ((key, _sweep_axis_effect(rows, key)[0]) for key in _sweep_varying_axes(rows)),
        key=lambda kv: -kv[1],
    )


def plot_sweep_results(
    out_dir: Path,
    records: list[dict],
    sweep_cfg: dict,
    comparison: dict | None = None,
) -> list[Path]:
    """Every sweep figure the data on disk supports. Returns the paths written.

    Degrades rather than fails: the three trial figures need only
    ``trials.jsonl``, while the holdout figure needs the ``--final`` run's
    comparison payload, so a search that has not been finalised yet still gets
    three of the four.
    """
    written: list[Path] = []
    for path in (
        plot_sweep_tradeoff(out_dir, records, sweep_cfg),
        plot_sweep_axes(out_dir, records, sweep_cfg),
        plot_sweep_reliability(out_dir, records, sweep_cfg),
        plot_sweep_holdout(out_dir, comparison, sweep_cfg.get("holdout") or [])
        if comparison else None,
    ):
        if path is not None:
            written.append(path)
    return written


########################################################################
# TRAIN - LEAVE-ONE-OUT (POOLED) TRANSFER
########################################################################

# The two folds written by epai/ai_detection/train/run_clf_leave_one_out.py.
LOO_MODES = [
    ("lodo", "Leave-one-domain-out"),
    ("logo", "Leave-one-generator-out"),
]


# Every bar in the leave-one-out figure is an accuracy, so they are all directly
# comparable. The two model bars are solid, the Binoculars baselines hatched, so a
# baseline never reads as another trained model.
LOO_MODEL_SPLITS = [
    ("model_in_domain_accuracy", "model in-domain", "lightsteelblue"),
    ("model_out_domain_accuracy", "model out-of-domain", "steelblue"),
]
LOO_BASELINE_COLORS = ["orange", "goldenrod", "peru", "darkkhaki", "tan"]
# In the gap figure the bar colour already encodes the sign of the gap, so the
# baselines are told apart by hatch instead.
LOO_GAP_HATCHES = [None, "//", "xx", "..", "\\\\"]


def _loo_subtitle(cfg: dict) -> str:
    return (
        f"{cfg['model']['ml_alg']} · {', '.join(cfg['features'])} · "
        f"{cfg['model']['n_runs']} seeds · train = pooled rest · "
        f"in-domain test = {cfg['model'].get('indomain_test_size', 0.20):.0%} of "
        f"training ids · chance = {CD_CHANCE*100:.0f}%"
    )


def _loo_baselines(payload: dict) -> list[tuple[str, str]]:
    """The Binoculars baseline variants to draw, as (key, legend label) pairs.

    Taken from the payload's declared order when present, else discovered from
    the per-slice results, so a fold that skipped a baseline (all scores missing)
    does not silently reorder the bars.
    """
    declared = payload.get("baselines") or []
    names = [b["name"] for b in declared if isinstance(b, dict) and "name" in b]
    labels = {b["name"]: b.get("label", b["name"]) for b in declared if "name" in b}

    for mode, _ in LOO_MODES:
        for agg in payload.get("results", {}).get(mode, {}).values():
            for name, entry in (agg.get("baselines") or {}).items():
                if name not in names:
                    names.append(name)
                labels.setdefault(name, entry.get("label", name))

    # Older payloads carry per-cutoff names such as `binoculars_vanilla_0.901531`,
    # which are too wide for a legend; the variant plus the cutoff is all the
    # label has to carry.
    def pretty(name: str) -> str:
        return labels.get(name, name).replace("binoculars_", "binoculars ").replace("_", " ")

    return [(name, pretty(name)) for name in names]


def _loo_is_current(payload: dict) -> bool:
    """True when the JSON has the in-domain / out-of-domain accuracies to plot."""
    for mode, _ in LOO_MODES:
        for agg in payload.get("results", {}).get(mode, {}).values():
            return "model_in_domain_accuracy" in agg
    return False


def _loo_bar_panel(ax, mode_rows: dict, title: str, baselines: list[tuple[str, str]]):
    """One grouped-bar panel: per held-out slice, four accuracies side by side.

    In-domain (the ceiling), out-of-domain (the transfer number), and one bar per
    Binoculars baseline on the same out-of-domain rows. A trailing 'mean' group
    averages the slices.

    Returns the handles/labels for a single figure-level legend: with two model
    bars plus a baseline each, a per-axes legend would sit on top of the bars.
    """
    helds = list(mode_rows.keys())

    def add_mean(vals):
        clean = [v for v in vals if v is not None]
        return vals + [float(np.mean(clean)) if clean else None]

    def series_for(getter):
        vals = [getter(mode_rows[h]) for h in helds]
        means = add_mean([v["mean"] if v else None for v in vals])
        stds = [(v["std"] or 0.0) if v else 0.0 for v in vals] + [0.0]
        return means, stds

    series = []
    for key, label, color in LOO_MODEL_SPLITS:
        means, stds = series_for(lambda agg, k=key: agg.get(k))
        series.append((label, means, stds, color, None))
    for i, (name, label) in enumerate(baselines):
        means, stds = series_for(
            lambda agg, n=name: (agg.get("baselines") or {}).get(n, {}).get(
                "out_domain_accuracy"
            )
        )
        series.append(
            (label, means, stds, LOO_BASELINE_COLORS[i % len(LOO_BASELINE_COLORS)], "//")
        )

    groups = helds + ["mean"]
    x = np.arange(len(groups))
    w = 0.84 / len(series)
    for i, (name, vals, errs, color, hatch) in enumerate(series):
        xpos = x + (i - (len(series) - 1) / 2) * w
        heights = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(
            xpos, heights, w, label=name, color=color,
            yerr=errs, capsize=2,
            hatch=hatch, edgecolor="black" if hatch else color,
            linewidth=1.0 if hatch else 0.0,
        )
        ax.bar_label(
            bars,
            labels=[f"{v*100:.0f}" if v is not None else "" for v in vals],
            fontsize=6.5 if len(series) > 4 else 7, padding=2,
        )

    ax.axhline(CD_CHANCE, linestyle="--", color="red", linewidth=1.2, label="chance")
    # Divider before the averaged 'mean' group so it doesn't read as a slice.
    ax.axvline(len(helds) - 0.5, color="lightgray", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            g
            if g == "mean"
            else f"{g}\n(in n={mode_rows[g]['in_domain_n']} · "
                 f"out n={mode_rows[g]['out_domain_n']})"
            for g in groups
        ],
        fontsize=7.5,
    )
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    return ax.get_legend_handles_labels()


def _save_loo_gap_figure(
    out_dir: Path,
    results: dict,
    alg_label: str,
    subtitle: str,
    baselines: list[tuple[str, str]],
) -> None:
    """Diverging bars: out-of-domain model accuracy minus each Binoculars baseline.

    This is the whole research question in one picture — blue where the pooled
    trained model beats the baseline on an unseen slice, red where it loses. Every
    number is an accuracy on the same held-out rows, so the gap is like-for-like.
    """
    slices: list[str] = []
    gaps: dict[str, list[float | None]] = {name: [] for name, _ in baselines}
    for mode, _ in LOO_MODES:
        tag = "LODO" if mode == "lodo" else "LOGO"
        for held, agg in results.get(mode, {}).items():
            model = agg.get("model_out_domain_accuracy", {}).get("mean")
            slices.append(f"{tag}: {held}")
            for name, _ in baselines:
                entry = (agg.get("baselines") or {}).get(name)
                bino = entry["out_domain_accuracy"]["mean"] if entry else None
                gaps[name].append(
                    (model - bino) * 100.0 if model is not None and bino is not None else None
                )
    if not slices or not baselines:
        return

    n_series = len(baselines)
    fig, ax = plt.subplots(
        figsize=(9.0, max(3.6, 0.42 * len(slices) * n_series + 2.2))
    )
    y = np.arange(len(slices))[::-1]  # first slice on top
    h = 0.84 / n_series
    handles = []
    for i, (name, label) in enumerate(baselines):
        vals = [g if g is not None else 0.0 for g in gaps[name]]
        hatch = LOO_GAP_HATCHES[i % len(LOO_GAP_HATCHES)]
        ypos = y - (i - (n_series - 1) / 2) * h
        bars = ax.barh(
            ypos, vals, h,
            color=["steelblue" if v >= 0 else "crimson" for v in vals],
            edgecolor="black", linewidth=0.5, hatch=hatch,
        )
        ax.bar_label(
            bars,
            labels=[f"{g:+.1f}" if g is not None else "" for g in gaps[name]],
            fontsize=7.5, padding=3,
        )
        # The bar colour carries the sign, not the series, so the legend key has
        # to be a neutral swatch showing only this baseline's hatch.
        handles.append(
            Patch(facecolor="whitesmoke", edgecolor="black", hatch=hatch,
                  label=f"vs {label}")
        )

    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(slices, fontsize=8)
    ax.set_xlabel(
        "out-of-domain model accuracy − Binoculars accuracy  (percentage points)"
    )
    ax.set_title(f"{alg_label} — does the trained model beat zero-shot OOD?\n{subtitle}")
    finite = [g for col in gaps.values() for g in col if g is not None]
    pad = max(5.0, max(abs(g) for g in finite) * 1.3) if finite else 5.0
    ax.set_xlim(-pad, pad)
    ax.legend(handles=handles, fontsize=7.5, loc="lower right")
    fig.text(
        0.01, 0.01,
        "Blue = the pooled trained model wins on the held-out slice; "
        "red = the Binoculars baseline wins.\nAll accuracies on the same "
        "out-of-domain rows, at the baseline's fixed cutoff.",
        fontsize=7.5, color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "leave_one_out_gap.png", dpi=150)
    plt.close(fig)


def _save_clf_loo_figures(out_dir: Path, payload: dict) -> None:
    """Save the leave-one-out bar figure and the model-vs-Binoculars gap figure."""
    results = payload["results"]
    cfg = payload["config"]
    alg_label = payload.get("alg_label", payload.get("alg_id", out_dir.name))
    subtitle = _loo_subtitle(cfg)

    present = [(mode, title) for mode, title in LOO_MODES if results.get(mode)]
    if not present:
        print(f"No leave-one-out results to plot in {out_dir}")
        return
    if not _loo_is_current(payload):
        print(
            f"Skipping {out_dir}: leave_one_out.json predates the in-domain / "
            f"out-of-domain split. Re-run run_clf_leave_one_out.py to regenerate it."
        )
        return

    baselines = _loo_baselines(payload)

    # Figure 1: grouped bars per fold (in-domain, out-of-domain, baselines).
    fig, axes = plt.subplots(
        1, len(present), figsize=(max(8.0, 7.0 * len(present)), 6.0), squeeze=False
    )
    keys: tuple = ([], [])
    for ax, (mode, title) in zip(axes[0], present):
        keys = _loo_bar_panel(ax, results[mode], title, baselines)

    # One legend for the whole figure: the panels share their series, and at four
    # or more bars per group a per-axes legend covers the bars it explains.
    fig.legend(
        keys[0], keys[1],
        loc="lower center", ncol=min(len(keys[0]), 6), fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, 0.055),
    )
    fig.suptitle(f"{alg_label} — leave-one-out transfer\n{subtitle}")
    note = (
        "In-domain = held-out documents of the training pool (same domains and "
        "generators as training) — the ceiling. Out-of-domain = the held-out slice."
    )
    if baselines:
        note += (
            "\nHatched = zero-shot Binoculars on the same out-of-domain rows, at "
            "a fixed cutoff that sees no labels at all."
        )
    fig.text(0.01, 0.005, note, fontsize=7.5, color="dimgray")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(out_dir / "leave_one_out.png", dpi=150)
    plt.close(fig)

    # Figure 2: the head-to-head gap, only meaningful with a Binoculars baseline.
    if baselines:
        _save_loo_gap_figure(out_dir, results, alg_label, subtitle, baselines)


def plot_clf_loo_result(alg_dir: Path) -> None:
    """Plot one leave-one-out result directory from its ``leave_one_out.json``."""
    result_path = alg_dir / "leave_one_out.json"
    if not result_path.exists():
        print(f"No leave_one_out.json in {alg_dir}")
        return
    with open(result_path) as f:
        payload = json.load(f)
    _save_clf_loo_figures(alg_dir, payload)


def plot_clf_loo_results(exp_dir: Path) -> None:
    """Plot every leave-one-out config in an experiment directory."""
    alg_dirs = sorted(
        d for d in exp_dir.iterdir()
        if d.is_dir() and (d / "leave_one_out.json").exists()
    )
    if not alg_dirs:
        print(f"No leave-one-out results found in {exp_dir}")
        return

    for alg_dir in alg_dirs:
        plot_clf_loo_result(alg_dir)

    print(f"Plots saved to: {exp_dir}")


########################################################################
# TRAIN - REGRESSION
########################################################################


def plot_results(results_dir: Path) -> None:
    print(results_dir)
    metrics_dir = results_dir / "metrics"
    fig_dir = results_dir / "figs" / "train"

    with open(metrics_dir / "runs.json") as f:
        runs = json.load(f)

    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Loss curves with 95% CI ---
    all_train = [r["train_losses"] for r in runs]
    all_val = [r["val_losses"] for r in runs]
    plot_loss_curves_ci(all_train, all_val, fig_dir / "loss_curves_ci.png", ci=0.95)

    # --- Per-seed actual vs predicted and feature importance ---
    for run in runs:
        seed = run["seed"]

        if "y_test" not in run or "y_pred" not in run:
            print(
                f"Seed {seed}: no predictions in runs.json — re-run train.py to generate them"
            )
            continue

        y_test = np.array(run["y_test"])
        y_pred = np.array(run["y_pred"])
        plot_actual_vs_predicted(
            y_test,
            y_pred,
            f"Test set  (seed={seed})",
            fig_dir / f"actual_vs_predicted_seed{seed}.png",
        )

        if "feature_importances" not in run or "feature_labels" not in run:
            print(
                f"Seed {seed}: no feature importances in runs.json — re-run train.py to generate them"
            )
            continue

        if run["feature_importances"]:
            importances = np.array(run["feature_importances"])
            labels = run["feature_labels"]
            plot_feature_importances(
                importances,
                labels,
                20,
                fig_dir / f"feature_importances_seed{seed}.png",
            )

    print(f"  Plots -> {fig_dir}")


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
    fig.savefig(out, dpi=300)
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
    fig.savefig(out, dpi=300)
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
    fig.savefig(out, dpi=300)
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
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


#############################################################
# COMPARE
#############################################################

METRICS = ["train_rmse", "train_r2", "val_rmse", "val_r2", "test_rmse", "test_r2"]
METRIC_LABELS = {
    "train_rmse": "Train RMSE",
    "train_r2": "Train R²",
    "val_rmse": "Val RMSE",
    "val_r2": "Val R²",
    "test_rmse": "Test RMSE",
    "test_r2": "Test R²",
}


def plot_metrics_ci(
    comparison: dict, out_dir: Path, labels: list[str] | None = None
) -> None:
    alg_ids = comparison["alg_ids"]
    display_labels = labels if labels is not None else alg_ids
    ci = comparison["ci"]
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    fig, axes = plt.subplots(3, 2, figsize=(16, 8))
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, METRICS):
        means, lo_errs, hi_errs = [], [], []
        for alg_id in alg_ids:
            s = comparison["runs"][alg_id].get(metric, {})
            mean = s.get("mean", 0.0)
            means.append(mean)
            lo_errs.append(mean - s.get("ci_low", mean))
            hi_errs.append(s.get("ci_high", mean) - mean)

        x = np.arange(len(alg_ids))
        bar_colors = [colors[i % len(colors)] for i in range(len(alg_ids))]
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
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def plot_loss_curves_comparison(
    alg_ids: list[str],
    runs_data: dict[str, list[dict]],
    out_dir: Path,
    ci: float,
    labels: list[str] | None = None,
) -> None:
    display_labels = labels if labels is not None else alg_ids
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, alg_id in enumerate(alg_ids):
        runs = runs_data[alg_id]
        color = colors[i % len(colors)]
        label = display_labels[i]

        all_train = np.array([r["train_losses"] for r in runs if "train_losses" in r])
        all_val = np.array([r["val_losses"] for r in runs if "val_losses" in r])
        if all_train.size == 0 or all_val.size == 0:
            print(f"  Skipping loss curves for '{alg_id}': no loss data")
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
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")

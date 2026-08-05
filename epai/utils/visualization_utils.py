from pathlib import Path

import numpy as np
from scipy import stats
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
import json
from sklearn.metrics import mean_squared_error, r2_score

# Consistent colors per LLM family are defined in cfg/colors.yaml and loaded via
# color_utils so every plotting script shares one source of truth.
from epai.utils.color_utils import llm_color as _llm_color


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
# balanced, so 0.50 is the meaningful midpoint every figure is anchored to.
CD_CHANCE = 0.5


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
) -> None:
    """One annotated source x target heatmap.

    Diverging map anchored at chance: red = below chance, white = chance,
    blue = above. The in-domain diagonal is boxed, because those cells answer a
    different question (held-out same-corpus) than the off-diagonal ones.
    """
    n = len(labels)
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="RdBu")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.ax.axhline(CD_CHANCE, color="black", linewidth=1.2, linestyle="--")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
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
            # Ink stays readable on both poles of the diverging ramp.
            strong = abs(mat[i, j] - CD_CHANCE) > 0.32
            color = "white" if strong else "black"
            ax.text(
                j,
                i,
                f"{mat[i, j]*100:.1f}\n±{std[i, j]*100:.1f}\nn={int(counts[i, j])}",
                ha="center",
                va="center",
                fontsize=8,
                linespacing=1.4,
                color=color,
            )
        ax.add_patch(
            plt.Rectangle(
                (i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.5
            )
        )


def _save_clf_cd_figures(out_dir: Path, payload: dict) -> None:
    """Save the transfer matrices and the in-domain vs transfer summary bars."""
    names: list[str] = payload["dataset_names"]
    label_of: dict = payload.get("dataset_labels", {})
    labels = [label_of.get(n, n) for n in names]
    summary = payload["summary"]
    cfg = payload["config"]
    alg_label = payload.get("alg_label", payload.get("alg_id", out_dir.name))

    n_runs = cfg["model"]["n_runs"]
    subtitle = (
        f"{cfg['model']['ml_alg']} · {', '.join(cfg['features'])} · "
        f"{n_runs} seeds · grouped by {cfg['data']['group_col']!r} · "
        f"chance = {CD_CHANCE*100:.0f}%"
    )

    acc = _cd_matrix(summary, names, "accuracy_mean")
    acc_std = _cd_matrix(summary, names, "accuracy_std")
    f1 = _cd_matrix(summary, names, "macro_f1_mean")
    f1_std = _cd_matrix(summary, names, "macro_f1_std")
    counts = _cd_matrix(summary, names, "n")

    size = max(7.0, 1.6 * len(names) + 3.0)

    # Figure 1: accuracy matrix.
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    _plot_cd_matrix(
        ax, acc, acc_std, counts, labels,
        f"{alg_label} — cross-domain accuracy (%)\n{subtitle}",
        "Accuracy (dashed = chance)", fig,
    )
    fig.text(
        0.01, 0.01,
        "Boxed diagonal = in-domain (held-out grouped test split of the SAME corpus). "
        "Off-diagonal = transfer to an unseen corpus.\n"
        "Cells show mean ± std over seeds; n = test rows. Target rows whose group was "
        "seen in training are excluded.",
        fontsize=7.5, color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "cross_domain_accuracy.png", dpi=150)
    plt.close(fig)

    # Figure 2: macro-F1 matrix — accuracy alone hides a model that predicts one
    # class on an unseen corpus.
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    _plot_cd_matrix(
        ax, f1, f1_std, counts, labels,
        f"{alg_label} — cross-domain macro F1 (%)\n{subtitle}",
        "Macro F1 (dashed = chance)", fig,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_dir / "cross_domain_macro_f1.png", dpi=150)
    plt.close(fig)

    # Figure 3: in-domain vs transfer, per train set and per test set.
    diag = np.diag(acc)
    off_mask = ~np.eye(len(names), dtype=bool)
    out_mean = np.array(
        [acc[i][off_mask[i]].mean() for i in range(len(names))]
    )  # trained here, tested elsewhere
    in_mean = np.array(
        [acc[:, j][off_mask[:, j]].mean() for j in range(len(names))]
    )  # trained elsewhere, tested here

    fig, axes = plt.subplots(1, 2, figsize=(max(11.0, 2.0 * len(names) + 6), 5.2))
    x = np.arange(len(names))
    w = 0.38

    b1 = axes[0].bar(x - w / 2, diag, w, label="in-domain (tested on itself)", color="steelblue")
    b2 = axes[0].bar(x + w / 2, out_mean, w, label="transfer out (mean of other test sets)", color="orange")
    axes[0].bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    axes[0].bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_xlabel("TRAINED ON  (train set)", fontweight="bold")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Per train set: does it survive leaving its corpus?")

    b3 = axes[1].bar(x, in_mean, w * 1.6, color="seagreen")
    axes[1].bar_label(b3, fmt="%.3f", fontsize=8, padding=2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_xlabel("TESTED ON  (test set)", fontweight="bold")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Per test set: transfer in (mean of models trained elsewhere)")

    for ax in axes:
        ax.axhline(CD_CHANCE, linestyle="--", color="red", linewidth=1.2, label="chance")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"{alg_label} — in-domain vs transfer\n{subtitle}")
    fig.tight_layout()
    fig.savefig(out_dir / "cross_domain_summary.png", dpi=150)
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


# Zero-shot baseline families, styled apart on the comparison figure. The two
# read their score in opposite directions and are not the same claim, so they get
# their own colour, hatch and legend entry rather than collapsing into one
# "baseline" item. The hatches are deliberately sparse: they mark a bar as a
# baseline without competing with the value printed on it.
CD_BASELINE_STYLES: dict[str, dict] = {
    "binoculars": {"label": "Binoculars baseline", "color": "crimson", "hatch": "/"},
    "llm_judge": {"label": "LLM Judge baseline", "color": "darkviolet", "hatch": "\\"},
}
# A baseline whose family cannot be told (an older payload, a third detector).
CD_BASELINE_FALLBACK = {
    "label": "zero-shot baseline", "color": "black", "hatch": "-"
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

    Deduplicated so a family with three variants (fitted plus two cutoffs) still
    contributes exactly one legend entry, and stable so the left and right panels
    of a figure list the same families in the same order.
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


def _plot_cd_comparison(
    exp_dir: Path,
    rows: list[dict],
    names: list[str],
    label_of: dict,
    alg_label: str,
    *,
    noun: str,
    subtitle: str,
    row_label,
) -> None:
    """Shared body for the embedding / algorithm cross-domain comparisons.

    Both comparisons rank the same per-item schema (``label``, ``accuracy``
    matrix, in-domain / transfer means, ``baseline`` flag) by transfer accuracy
    and draw the same two figures. Only the wording, the row label and the output
    filenames differ, so those are the parameters:

      * ``noun`` -- "embedding" or "algorithm"; used in titles, the scatter
        legend and the ``<noun>_comparison[_matrix].png`` filenames.
      * ``subtitle`` -- the caller's one-line description (the sweep differs:
        embeddings share an algorithm, algorithms share a feature set).
      * ``row_label(r)`` -- the y-axis label for item ``r``.

    Zero-shot baselines (``baseline: true``) are marked with a ★ and faintly
    hatched bars so they never read as just another item, and each *family* of
    baseline (``baseline_kind``: Binoculars, LLM judge) gets its own colour, hatch
    pattern and legend entry on both panels -- they are opposite-direction rules
    on different scores, so one merged "baseline" item would hide which is which.
    The colour is carried by the ★ and the legend only; the hatch itself is
    translucent black so it stays a background texture. Hybrids
    (``hybrid: true``) are
    marked more lightly -- a ◆ on the row label and their own scatter marker, but
    plain bars -- because they were trained on extra information (the Binoculars
    score on top of the shared feature set) and their rank should carry that
    caveat without the bars becoming unreadable.
    """
    # Ranked by transfer: the whole point is which item survives leaving its
    # corpus, not which scores highest at home.
    order = sorted(rows, key=lambda r: r["transfer_mean"])
    is_baseline = np.array([bool(r.get("baseline")) for r in order])
    is_hybrid = np.array([bool(r.get("hybrid")) for r in order])
    labels = [row_label(r) for r in order]
    in_domain = np.array([r["in_domain_mean"] for r in order])
    transfer = np.array([r["transfer_mean"] for r in order])
    transfer_std = np.array([r["transfer_std"] for r in order])
    in_domain_std = np.array([r["in_domain_std"] for r in order])
    transfer_min = np.array([r["transfer_min"] for r in order])

    # Figure 1: ranked bars + the in-domain/transfer scatter.
    height = max(5.0, 0.42 * len(order) + 2.4)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, height))

    y = np.arange(len(order))
    h = 0.38
    b1 = axes[0].barh(y + h / 2, in_domain, h, xerr=in_domain_std, capsize=2,
                      color="steelblue", label="in-domain (diagonal)")
    b2 = axes[0].barh(y - h / 2, transfer, h, xerr=transfer_std, capsize=2,
                      color="orange", label="transfer (off-diagonal)")
    axes[0].bar_label(b1, fmt="%.3f", fontsize=7, padding=2)
    axes[0].bar_label(b2, fmt="%.3f", fontsize=7, padding=2)
    # Hatch the baseline's bars so it never reads as just another item, in its
    # family's hatch pattern so the two baseline families are told apart at a
    # glance. The stroke is translucent black rather than the family colour: the
    # colour lives on the ★ and in the legend, and a faint hatch stays a
    # background texture instead of swallowing the bar. Hybrids are marked by the
    # ◆ on their row label and by their marker in the scatter, which is enough --
    # hatching their bars as well drowns out the numbers.
    for idx in np.nonzero(is_baseline)[0]:
        style = _cd_baseline_style(order[idx])
        for bar in (b1[idx], b2[idx]):
            bar.set_hatch(style["hatch"])
            bar.set_edgecolor(CD_HATCH_COLOR)
            bar.set_linewidth(CD_HATCH_LINEWIDTH)
    # Worst single transfer cell: a mean can hide one collapsed corpus pair.
    axes[0].scatter(transfer_min, y - h / 2, marker="|", s=90, color="black",
                    zorder=3, label="worst transfer cell")
    axes[0].axvline(CD_CHANCE, linestyle="--", color="red", linewidth=1.2, label="chance")
    axes[0].set_yticks(y)
    # Only the ★ / ◆ is coloured -- the name itself stays black, so the colour on
    # the row means "baseline family", not "this row is special in some way". A
    # tick label is a single Text artist and cannot hold two colours, so each row
    # label is drawn instead as a packed box (coloured mark + black name)
    # right-aligned on the axis where the tick label would have been.
    axes[0].set_yticklabels([""] * len(order))
    for idx, row in enumerate(order):
        prefix = _cd_row_mark(row)
        mark = prefix.strip()
        parts = [
            TextArea(labels[idx][len(prefix):],
                     textprops=dict(color="black", fontsize=8))
        ]
        if mark:
            mark_color = (
                _cd_baseline_style(row)["color"] if row.get("baseline")
                else CD_HYBRID_COLOR
            )
            parts.insert(
                0, TextArea(mark, textprops=dict(color=mark_color, fontsize=10))
            )
        axes[0].add_artist(
            AnnotationBbox(
                HPacker(children=parts, align="center", pad=0.0, sep=3.0),
                (0.0, y[idx]),
                xycoords=("axes fraction", "data"),
                xybox=(-6.0, 0.0),
                boxcoords="offset points",
                box_alignment=(1.0, 0.5),
                frameon=False,
                pad=0.0,
                annotation_clip=False,
            )
        )
    axes[0].set_xlim(0.0, 1.18)
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Ranked by transfer accuracy")
    handles, _ = axes[0].get_legend_handles_labels()
    # Star glyphs matching the ★ prefixes on the row labels: one entry per
    # baseline family present, in that family's colour, so the legend says which
    # baseline a starred row is rather than only that it is one.
    from matplotlib.lines import Line2D

    for style in _cd_baseline_legend_styles(order, is_baseline):
        handles.append(
            Line2D([], [], linestyle="none", marker="*", color=style["color"],
                   markersize=11, label=style["label"])
        )
    if is_hybrid.any():
        handles.append(
            Line2D([], [], linestyle="none", marker="D", color=CD_HYBRID_COLOR,
                   markersize=6, label="hybrid (extra features)")
        )
    # Legend below the axes so it never sits over the bars / worst-cell marks.
    axes[0].legend(
        handles=handles, fontsize=8, loc="upper center",
        bbox_to_anchor=(0.5, -0.12), ncol=min(len(handles), 3), frameon=False,
    )

    axes[1].plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1.0,
                 label="no loss under transfer")
    axes[1].axhline(CD_CHANCE, linestyle="--", color="red", linewidth=1.2, label="chance")
    axes[1].axvline(CD_CHANCE, linestyle="--", color="red", linewidth=1.2)
    item_pts = ~is_baseline & ~is_hybrid
    axes[1].scatter(in_domain[item_pts], transfer[item_pts], s=45,
                    color=CD_ITEM_COLOR, zorder=3, label=noun)
    hybrid_pts = is_hybrid & ~is_baseline
    # Same green as the plain items: a hybrid is still a trained classifier, so it
    # is set apart by the ◆ shape (and the ◆ on its row) rather than by hue.
    if hybrid_pts.any():
        axes[1].scatter(in_domain[hybrid_pts], transfer[hybrid_pts], s=85,
                        marker="D", color=CD_HYBRID_COLOR, edgecolor="black",
                        linewidth=0.6, zorder=4, label="hybrid (extra features)")
    # One star series per baseline family, matching the left panel's colours and
    # legend wording, so a star on the scatter and a starred row in the ranking
    # are recognisably the same thing.
    for style in _cd_baseline_legend_styles(order, is_baseline):
        pts = np.array(
            [
                bool(b) and _cd_baseline_style(r)["label"] == style["label"]
                for r, b in zip(order, is_baseline)
            ]
        )
        if pts.any():
            axes[1].scatter(in_domain[pts], transfer[pts], s=130,
                            marker="*", color=style["color"], edgecolor="black",
                            linewidth=0.6, zorder=4, label=style["label"])
    for r, x_val, y_val in zip(order, in_domain, transfer):
        axes[1].annotate(r["label"], (x_val, y_val), textcoords="offset points",
                         xytext=(5, 4), fontsize=7.5)
    lo = min(0.45, float(min(transfer.min(), in_domain.min())) - 0.05)
    axes[1].set_xlim(lo, 1.02)
    axes[1].set_ylim(lo, 1.02)
    axes[1].set_xlabel("In-domain accuracy (diagonal)")
    axes[1].set_ylabel("Transfer accuracy (off-diagonal)")
    axes[1].set_title("Does the in-domain score survive an unseen corpus?")
    n_leg1 = len(axes[1].get_legend_handles_labels()[0])
    axes[1].legend(
        fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=min(n_leg1, 3), frameon=False,
    )

    fig.suptitle(f"{alg_label} — {noun} comparison\n{subtitle}")
    # bbox_inches="tight" so the below-axes legends are included, not clipped.
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    # The hatch stroke weight is an rcParam read at draw time, not an artist
    # property, so it is set around this savefig only -- thin enough that the
    # hatching marks a baseline without fighting the value printed on the bar --
    # and restored so no other figure in the process inherits it.
    prev_hatch_lw = plt.rcParams["hatch.linewidth"]
    plt.rcParams["hatch.linewidth"] = CD_HATCH_LINEWIDTH
    fig.savefig(exp_dir / f"{noun}_comparison.png", dpi=150, bbox_inches="tight")
    plt.rcParams["hatch.linewidth"] = prev_hatch_lw
    plt.close(fig)

    # Figure 2: every cell of every item, best transfer at the top.
    order_desc = list(reversed(order))
    pairs = [(s, t) for s in names for t in names]
    pair_labels = [
        f"{label_of.get(s, s)}\n→ {label_of.get(t, t)}" for s, t in pairs
    ]
    mat = np.array(
        [
            [r["accuracy"][names.index(s)][names.index(t)] for s, t in pairs]
            for r in order_desc
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(max(9.0, 1.25 * len(pairs) + 3.0), max(4.5, 0.36 * len(order) + 3.0))
    )
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="RdBu", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Accuracy (dashed = chance)")
    cbar.ax.axhline(CD_CHANCE, color="black", linewidth=1.2, linestyle="--")

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pair_labels, fontsize=7.5, rotation=45, ha="right")
    ax.set_yticks(range(len(order_desc)))
    ax.set_yticklabels([row_label(r) for r in order_desc], fontsize=8)
    ax.set_xlabel("TRAINED ON → TESTED ON", fontweight="bold", labelpad=8)
    ax.set_title(f"{alg_label} — every transfer cell, per {noun}\n{subtitle}")

    ax.set_xticks(np.arange(-0.5, len(pairs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order_desc), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            strong = abs(mat[i, j] - CD_CHANCE) > 0.32
            ax.text(j, i, f"{mat[i, j]*100:.1f}", ha="center", va="center",
                    fontsize=7, color="white" if strong else "black")
    # Boxed columns are in-domain: they answer a different question from the rest.
    for j, (s, t) in enumerate(pairs):
        if s == t:
            ax.add_patch(
                plt.Rectangle((j - 0.5, -0.5), 1, mat.shape[0], fill=False,
                              edgecolor="black", linewidth=2.0)
            )

    note = (
        "Boxed columns = in-domain (held-out split of the same corpus); the rest "
        "are transfer to an unseen corpus.\nRows are ordered by mean transfer "
        "accuracy, best first. ★ marks a zero-shot baseline (no classifier trained)."
    )
    if is_hybrid.any():
        note += (
            "\n◆ marks a hybrid: trained on extra features beyond the shared set, "
            "so its row is not comparable on equal information."
        )
    fig.text(0.01, 0.01, note, fontsize=7.5, color="dimgray")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(exp_dir / f"{noun}_comparison_matrix.png", dpi=150)
    plt.close(fig)

    print(f"Comparison plots saved to: {exp_dir}")


def plot_clf_cd_emb_comparison(exp_dir: Path, payload: dict) -> None:
    """Compare the embeddings of one cross-domain experiment.

    ``payload`` is the ``embedding_comparison.json`` written by
    ``epai/ai_detection/train/run_clf_cross_domain_embeddings.py``: one entry per embedding,
    each holding its full source x target accuracy matrix and the in-domain /
    transfer means taken from it.

    Two figures:

      * ``embedding_comparison.png`` -- in-domain vs transfer per embedding,
        ranked, next to the same two numbers as a scatter. The scatter is what
        separates "this embedding is good" from "this embedding is good *at this
        corpus*": points near the diagonal keep their score off-domain, points
        far below it have learned the corpus.
      * ``embedding_comparison_matrix.png`` -- every source->target cell of every
        embedding in one heatmap, for the cases the means average away (one
        corpus pair that collapses while the rest hold up).
    """
    rows: list[dict] = payload["embeddings"]
    if not rows:
        print(f"No embedding results to compare in {exp_dir}")
        return

    names: list[str] = payload["dataset_names"]
    label_of: dict = payload.get("dataset_labels", {})
    cfg = payload["config"]
    alg_label = payload.get("alg_label", payload.get("alg_id", exp_dir.name))
    subtitle = (
        f"{cfg['model']['ml_alg']} · {cfg['model']['n_runs']} seeds · "
        f"grouped by {cfg['data']['group_col']!r} · "
        f"{len(names)} corpora · chance = {CD_CHANCE*100:.0f}%"
    )
    _plot_cd_comparison(
        exp_dir, rows, names, label_of, alg_label,
        noun="embedding", subtitle=subtitle,
        row_label=lambda r: (
            f"{_cd_row_mark(r)}{r['label']} ({r['feature_dim']}d)"
        ),
    )


def plot_clf_cd_alg_comparison(exp_dir: Path, payload: dict) -> None:
    """Compare the algorithms of one cross-domain experiment.

    ``payload`` is the ``algorithm_comparison.json`` written by
    ``epai/ai_detection/train/run_clf_cross_domain_algs.py``: one entry per algorithm, on a
    single shared feature set, each holding its source x target accuracy matrix.
    Draws the same two figures as the embedding comparison (``algorithm_comparison
    .png`` and ``algorithm_comparison_matrix.png``) ranked by transfer accuracy.
    """
    rows: list[dict] = payload["algorithms"]
    if not rows:
        print(f"No algorithm results to compare in {exp_dir}")
        return

    names: list[str] = payload["dataset_names"]
    label_of: dict = payload.get("dataset_labels", {})
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
        f"{len(names)} corpora · chance = {CD_CHANCE*100:.0f}%"
    )
    _plot_cd_comparison(
        exp_dir, rows, names, label_of, alg_label,
        noun="algorithm", subtitle=subtitle,
        row_label=lambda r: f"{_cd_row_mark(r)}{r['label']}",
    )


########################################################################
# TRAIN - HYPER-PARAMETER SWEEP OVER CROSS-DOMAIN TRANSFER
########################################################################

# Written by epai/ai_detection/train/sweep_clf_cross_domain.py. These figures
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

    # `binoculars_vanilla_0.901531` is too wide for a legend; the variant plus the
    # cutoff is all the label has to carry.
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
        "out-of-domain rows (fitted = threshold from the training pool, "
        "vanilla = fixed cutoff, fit to nothing).",
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
            "\nHatched = zero-shot Binoculars on the same out-of-domain rows; "
            "'fitted' calibrates its cutoff on the training pool, 'vanilla' uses a "
            "fixed cutoff and sees no labels at all."
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

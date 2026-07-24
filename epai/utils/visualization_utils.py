from pathlib import Path

import numpy as np
from scipy import stats
import plotly.graph_objects as go
import matplotlib.pyplot as plt
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

    Zero-shot baselines (``baseline: true``) are marked with a ★ and hatched
    bars so they never read as just another item. Hybrids (``hybrid: true``) are
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
    # Hatch the baseline's bars so it never reads as just another item. Hybrids
    # are marked by the ◆ on their row label and by their marker in the scatter,
    # which is enough -- hatching their bars as well drowns out the numbers.
    for idx in np.nonzero(is_baseline)[0]:
        for bar in (b1[idx], b2[idx]):
            bar.set_hatch("//")
            bar.set_edgecolor("black")
            bar.set_linewidth(1.2)
    # Worst single transfer cell: a mean can hide one collapsed corpus pair.
    axes[0].scatter(transfer_min, y - h / 2, marker="|", s=90, color="black",
                    zorder=3, label="worst transfer cell")
    axes[0].axvline(CD_CHANCE, linestyle="--", color="red", linewidth=1.2, label="chance")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlim(0.0, 1.18)
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Ranked by transfer accuracy")
    handles, _ = axes[0].get_legend_handles_labels()
    # Black glyphs matching the ★ / ◆ prefixes on the row labels, rather than the
    # bar hatch: the prefix is what a reader scans for, and a coloured marker
    # here would read as a second, different symbol.
    if is_baseline.any():
        from matplotlib.lines import Line2D
        handles.append(
            Line2D([], [], linestyle="none", marker="*", color="black",
                   markersize=11, label="zero-shot baseline")
        )
    if is_hybrid.any():
        from matplotlib.lines import Line2D
        handles.append(
            Line2D([], [], linestyle="none", marker="D", color="black",
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
    axes[1].scatter(in_domain[item_pts], transfer[item_pts], s=45, color="seagreen",
                    zorder=3, label=noun)
    hybrid_pts = is_hybrid & ~is_baseline
    if hybrid_pts.any():
        axes[1].scatter(in_domain[hybrid_pts], transfer[hybrid_pts], s=85,
                        marker="D", color="mediumpurple", edgecolor="black",
                        linewidth=0.6, zorder=4, label="hybrid (extra features)")
    if is_baseline.any():
        axes[1].scatter(in_domain[is_baseline], transfer[is_baseline], s=130,
                        marker="*", color="crimson", edgecolor="black",
                        linewidth=0.6, zorder=4, label="zero-shot baseline")
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
    fig.savefig(exp_dir / f"{noun}_comparison.png", dpi=150, bbox_inches="tight")
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
    ``epai/train/run_clf_cross_domain_embeddings.py``: one entry per embedding,
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
    ``epai/train/run_clf_cross_domain_algs.py``: one entry per algorithm, on a
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

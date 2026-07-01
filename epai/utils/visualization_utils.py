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


def plot_clf_results(exp_dir: Path, show_random: bool = True) -> None:
    from sklearn.metrics import confusion_matrix as compute_cm

    alg_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
    if not alg_dirs:
        print(f"No algorithm directories found in {exp_dir}")
        return

    with open(alg_dirs[0] / "label_mapping.json") as f:
        label_mapping = json.load(f)
    class_names = [label_mapping[str(i)] for i in range(len(label_mapping))]
    n_classes = len(class_names)

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
    tick_fs = max(6, 10 - n_classes // 3)

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
        ax.bar(x + j * w, means, w, label=split, color=color, yerr=stds, capsize=4)
    if show_random:
        ax.axhline(
            1 / n_classes, linestyle="--", color="red", linewidth=1.2, label="random"
        )
    ax.set_xticks(x + w)
    ax.set_xticklabels(alg_labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
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
        for j, split in enumerate(splits):
            ax = axes[i, j]
            cms = []
            for r in data[mn]:
                yt = r.get(f"y_{split}")
                yp = r.get(f"y_{split}_pred")
                if yt is not None and yp is not None:
                    cms.append(compute_cm(yt, yp))
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
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=tick_fs)
            ax.set_yticklabels(class_names, fontsize=tick_fs)
            for r_idx in range(n_classes):
                for c_idx in range(n_classes):
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
        n_models, n_splits, figsize=(max(6, n_classes) * n_splits * 0.6, 4 * n_models)
    )
    axes = np.array(axes).reshape(n_models, n_splits)
    for i, mn in enumerate(alg_names):
        for j, split in enumerate(splits):
            ax = axes[i, j]
            f1s = {c: [] for c in class_names}
            for r in data[mn]:
                for c in class_names:
                    report = r.get(f"{split}_report", {})
                    if c in report:
                        f1s[c].append(report[c]["f1-score"])
            means = [float(np.mean(f1s[c])) if f1s[c] else 0.0 for c in class_names]
            stds = [float(np.std(f1s[c])) if f1s[c] else 0.0 for c in class_names]
            ax.bar(range(n_classes), means, yerr=stds, capsize=4, color="steelblue")
            ax.axhline(
                1 / n_classes,
                linestyle="--",
                color="red",
                linewidth=1.2,
                label="random",
            )
            ax.set_xticks(range(n_classes))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=tick_fs)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("F1 Score")
            ax.set_title(f"{alg_labels[i]} – {split}")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(exp_dir / "f1_scores.png", dpi=150)
    plt.close(fig)

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

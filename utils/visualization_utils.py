from pathlib import Path

import numpy as np
import plotly.graph_objects as go


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
    labels = np.array(["default"] * n) if labels is None else np.asarray(labels)

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
        legend_title_text="Researchers",
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
    labels = np.array(["default"] * n) if labels is None else np.asarray(labels)

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
        legend_title_text="Subfolders",
        hovermode="closest",
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        print(f"Interactive plot saved to {save_path}")

    fig.show()

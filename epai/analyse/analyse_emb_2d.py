"""Plot text embeddings in 2D, coloured by human vs machine.

Reduces an embedding column to two dimensions (PCA / kernel PCA / UMAP, via
epai.utils.dimred_utils.reduce) and scatters it, with human text in blue and
machine-generated text in red.

This is the eyeball version of the classifier question: if the two colours form
visibly separate clouds, a linear model will trivially reach the ~98% seen
in-domain. If they overlap and the structure you can see is topical instead,
that score is corpus-specific rather than a general "AI-ness" signal.

Examples:
    # just human vs the plain gpt rewrite, UMAP
    uv run python epai/analyse/analyse_emb_2d.py data/parquet/essay_binox0.parquet \
        --drop-classes claude gpt_prompt1 gpt_prompt2 gpt_semantic gpt_writing \
        --method clf
    
    uv run python epai/analyse/analyse_emb_2d.py data/parquet/reuter_binox0.parquet \
        --drop-classes claude gpt_prompt1 gpt_prompt2 gpt_semantic gpt_writing

    uv run python epai/analyse/analyse_emb_2d.py data/parquet/wp_binox0.parquet \
        --drop-classes claude gpt_prompt1 gpt_prompt2 gpt_semantic gpt_writing
    
    uv run python epai/analyse/analyse_emb_2d.py data/parquet/essay_gpt54mini_binox0.parquet
    uv run python epai/analyse/analyse_emb_2d.py data/parquet/reuter_gpt54mini_binox0.parquet
    uv run python epai/analyse/analyse_emb_2d.py data/parquet/wp_gpt54mini_binox0.parquet


    # colour each generator separately instead of the binary view
    python -m epai.analyse.analyse_emb_2d data/parquet/wp_binox0.parquet --by-generator
"""

import argparse
import html
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from epai.utils.config_utils import RESET, YELLOW
from epai.utils.dimred_utils import DIM_RED_METHODS, reduce
from epai.utils.parquet_utils import load_parquet
from epai.utils.train_utils import CLF_TARGET_COL, drop_missing_rows

BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "analyse_emb_2d"

# Supervised projections: unlike the unsupervised methods in DIM_RED_METHODS,
# these use the labels, so they must be fit on train and plotted on test.
SUPERVISED_METHODS = ["lda", "clf"]
ALL_METHODS = DIM_RED_METHODS + SUPERVISED_METHODS

GROUP_COL = "id"  # rows sharing this stay on one side of the supervised split

HUMAN_LABEL = "human"
HUMAN_COLOR = "#2563eb"  # blue
MACHINE_COLOR = "#dc2626"  # red

HOVER_CHARS = 300  # characters of `text` shown in the interactive tooltip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scatter 2D-reduced embeddings, human (blue) vs machine (red).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "parquet",
        type=Path,
        help="Input parquet, e.g. data/parquet/essay_binox0.parquet",
    )
    p.add_argument(
        "--method",
        choices=ALL_METHODS,
        default="pca",
        help="Unsupervised (pca/kernel_pca/umap) or supervised (lda/clf). The "
        "supervised ones fit on a train split and plot only held-out rows.",
    )
    p.add_argument(
        "--feature",
        default="emb_specter2",
        help="Embedding column to reduce.",
    )
    p.add_argument(
        "--drop-classes",
        nargs="*",
        default=[],
        metavar="CLS",
        help="Generators to exclude, e.g. claude gpt_prompt1.",
    )
    p.add_argument(
        "--by-generator",
        action="store_true",
        help="Colour each generator separately instead of binary human/machine.",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Randomly subsample to N rows (UMAP on big files is slow).",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for --sample.")
    p.add_argument(
        "--no-html",
        action="store_true",
        help="Skip the interactive HTML and write only the PNG.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG; the HTML is written alongside it. "
        "Default: results/analyse_emb_2d/<stem>_<method>.png",
    )
    return p.parse_args()


def reduce_supervised(
    X: np.ndarray,
    generated_by: np.ndarray,
    groups: np.ndarray | None,
    method: str,
    seed: int,
    test_size: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    """Project onto a *supervised* axis: the direction that separates the classes.

    PCA and UMAP are unsupervised — they rank directions by variance, and in
    SPECTER2 space the variance is topic, so the human/machine direction never
    surfaces in the top two components. These two methods look for it directly:

        lda  -> Linear Discriminant Analysis: maximise between-class separation
                relative to within-class spread.
        clf  -> the weight vector of the same LogisticRegression that run_clf.py
                trains, i.e. the boundary the real classifier actually uses.

    Both are fit on a *training* split and only the held-out points are returned.
    That discipline is not optional: with 768 dimensions and ~1400 rows either
    model can separate even randomly permuted labels on the data it was fit on,
    so a plot fit on everything would look immaculate and mean nothing. Fitting
    on train and plotting test makes the overlap you see the honest error rate.

    x is the model's signed decision score, so the decision boundary sits at
    x = 0. y is the leading direction of what is left after the discriminant is
    projected out — a spread axis, carrying no class information by construction.

    Returns (X2 for the test rows, test row indices, axis labels).
    """
    y = (generated_by != HUMAN_LABEL).astype(int)

    if groups is None:
        print(
            f"{YELLOW}  ⚠️  no group column — falling back to a random row split. "
            f"A document and its rewrite may land on opposite sides.{RESET}"
        )
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
        )
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups))

        # A document and every rewrite of it share one group, so they must not
        # straddle the cut — otherwise the "held-out" rows are not held out and
        # the separation on show would be flattered.
        overlap = set(groups[train_idx]) & set(groups[test_idx])
        if overlap:
            raise AssertionError(
                f"Group leakage: {len(overlap)} group(s) in both splits, "
                f"e.g. {sorted(overlap)[:3]}"
            )
        print(
            f"  Split: grouped by '{GROUP_COL}' — "
            f"{len(set(groups[train_idx]))} train / {len(set(groups[test_idx]))} test "
            f"groups, no group in both"
        )

    scaler = StandardScaler().fit(X[train_idx])
    X_train, X_test = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])

    if method == "lda":
        model = LinearDiscriminantAnalysis()
        x_label = "LDA discriminant (decision score)"
    else:
        model = LogisticRegression(max_iter=1000, random_state=seed)
        x_label = "logistic w·x (decision score)"
    model.fit(X_train, y[train_idx])

    train_acc = accuracy_score(y[train_idx], model.predict(X_train))
    test_acc = accuracy_score(y[test_idx], model.predict(X_test))
    print(
        f"  {method.upper()} fit on {len(train_idx)} train rows "
        f"(train acc {train_acc*100:.1f}%), plotting {len(test_idx)} held-out rows "
        f"(test acc {test_acc*100:.1f}%)"
    )
    if train_acc - test_acc > 0.15:
        print(
            f"{YELLOW}  ⚠️  train >> test: the model is partly memorising. The "
            f"separation below is still honest (it is the test split), but do not "
            f"read the training fit as signal.{RESET}"
        )

    w = np.asarray(model.coef_[0], dtype=np.float64)
    w_hat = w / np.linalg.norm(w)

    xs = model.decision_function(X_test)

    # Remove the discriminant from the test points, then take the leading
    # direction of the remainder. Purely a spread axis so the points do not all
    # collapse onto one line; it is orthogonal to the discriminant by
    # construction, so it cannot smuggle class information back in.
    residual = X_test - np.outer(X_test @ w_hat, w_hat)
    ys = PCA(n_components=1, random_state=0).fit_transform(residual)[:, 0]

    X2 = np.column_stack([xs, ys])
    return X2, test_idx, (x_label, "PC1 of the residual (no class signal)")


def build_groups(
    generated_by: np.ndarray, by_generator: bool
) -> list[tuple[str, np.ndarray, str]]:
    """Return the (label, row-mask, hex colour) of each group to plot.

    Machine groups come first so the human cloud is drawn on top of them.
    Binary mode collapses every generator into one red "machine" group; the
    per-generator mode keeps human blue and gives each generator its own shade.
    """
    is_human = generated_by == HUMAN_LABEL

    if not by_generator:
        return [
            ("machine", ~is_human, MACHINE_COLOR),
            (HUMAN_LABEL, is_human, HUMAN_COLOR),
        ]

    generators = sorted(set(generated_by) - {HUMAN_LABEL})
    shades = plt.get_cmap("autumn")(np.linspace(0, 0.75, max(len(generators), 1)))
    groups = [
        (gen, generated_by == gen, mcolors.to_hex(shade))
        for gen, shade in zip(generators, shades)
    ]
    groups.append((HUMAN_LABEL, is_human, HUMAN_COLOR))
    return groups


def centroid(X2: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    return float(X2[mask, 0].mean()), float(X2[mask, 1].mean())


def plot_static(
    X2: np.ndarray,
    groups: list[tuple[str, np.ndarray, str]],
    title: str,
    method: str,
    out: Path,
    axis_labels: tuple[str, str] = ("1", "2"),
) -> None:
    """Matplotlib PNG: one scatter per group plus a star at each group mean."""
    fig, ax = plt.subplots(figsize=(9, 7.5))

    # For the supervised projections x is a signed decision score, so x = 0 is
    # literally the classifier's boundary: points on the wrong side are its errors.
    if method in SUPERVISED_METHODS:
        ax.axvline(0, color="#374151", linestyle="--", linewidth=1.2, zorder=1)
        ax.annotate(
            "decision boundary",
            xy=(0, ax.get_ylim()[1]), xytext=(4, -12),
            textcoords="offset points", rotation=90,
            fontsize=8, color="#374151", va="top",
        )

    for label, mask, color in groups:
        ax.scatter(X2[mask, 0], X2[mask, 1], c=color, s=8, alpha=0.45, linewidths=0)

    # Group means, drawn last so they sit above the cloud. The white edge keeps
    # the star legible where the two clouds overlap.
    for label, mask, color in groups:
        cx, cy = centroid(X2, mask)
        ax.scatter(
            cx, cy, marker="*", s=450, c=color,
            edgecolors="white", linewidths=1.2, zorder=5,
        )

    # The legend is built from proxy handles rather than the plotted artists:
    # the points are drawn at s=8 and the stars at s=450, so reusing them gives
    # a legend of invisible dots next to enormous stars. One star entry stands
    # for all of them — the star's colour already says which group it marks.
    handles = [
        plt.Line2D(
            [], [], linestyle="none", marker="o", markersize=7,
            markerfacecolor=color, markeredgecolor="none",
            label=f"{label} (n={int(mask.sum())})",
        )
        for label, mask, color in groups
    ]
    handles.append(
        plt.Line2D(
            [], [], linestyle="none", marker="*", markersize=15,
            markerfacecolor="#4b5563", markeredgecolor="white", markeredgewidth=1.0,
            label="group mean",
        )
    )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.legend(handles=handles, loc="best", frameon=True, fontsize=9, labelspacing=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def hover_text(row: pd.Series, feature: str) -> str:
    """One tooltip: the row's identity plus a readable snippet of its text."""
    parts = [f"<b>{html.escape(str(row['id']))}</b>",
             f"generated_by: {html.escape(str(row[CLF_TARGET_COL]))}"]

    for extra in ("author", "binoculars_score"):
        if extra in row.index and pd.notna(row[extra]):
            value = row[extra]
            if isinstance(value, float):
                value = f"{value:.4f}"
            parts.append(f"{extra}: {html.escape(str(value))}")

    text = str(row.get("text", "") or "")
    snippet = text[:HOVER_CHARS] + ("…" if len(text) > HOVER_CHARS else "")
    # Escape first, then insert <br> — plotly renders the tooltip as HTML, so a
    # stray "<" in the source text would otherwise eat the rest of the tooltip.
    snippet = html.escape(snippet).replace("\n", " ")
    wrapped = "<br>".join(
        snippet[i : i + 60] for i in range(0, len(snippet), 60)
    )
    parts.append(f"chars: {len(text)}<br><br>{wrapped}")
    return "<br>".join(parts)


def plot_interactive(
    X2: np.ndarray,
    groups: list[tuple[str, np.ndarray, str]],
    df: pd.DataFrame,
    feature: str,
    title: str,
    method: str,
    out: Path,
    axis_labels: tuple[str, str] = ("1", "2"),
) -> None:
    """Plotly HTML: same plot, but hovering a point shows the row behind it."""
    hovers = df.apply(hover_text, axis=1, args=(feature,)).to_numpy()
    fig = go.Figure()

    if method in SUPERVISED_METHODS:
        fig.add_vline(
            x=0, line=dict(color="#374151", width=1.2, dash="dash"),
            annotation_text="decision boundary", annotation_position="top",
        )

    for label, mask, color in groups:
        fig.add_trace(
            go.Scattergl(
                x=X2[mask, 0], y=X2[mask, 1],
                mode="markers",
                name=f"{label} (n={int(mask.sum())})",
                marker=dict(color=color, size=5, opacity=0.55),
                text=hovers[mask],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    for label, mask, color in groups:
        cx, cy = centroid(X2, mask)
        fig.add_trace(
            go.Scatter(
                x=[cx], y=[cy],
                mode="markers",
                name=f"{label} mean",
                marker=dict(
                    color=color, size=22, symbol="star",
                    line=dict(color="white", width=1.5),
                ),
                hovertemplate=f"<b>{label} mean</b><br>({cx:.3f}, {cy:.3f})<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=axis_labels[0],
        yaxis_title=axis_labels[1],
        template="plotly_white",
        hovermode="closest",
        width=1100,
        height=800,
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(out)


def main() -> None:
    args = parse_args()

    parquet_path = args.parquet if args.parquet.is_absolute() else BASE_DIR / args.parquet
    df = load_parquet(parquet_path)

    # Some human source files are empty and so have no embedding; those rows
    # would otherwise blow up the np.array() stack below.
    df = drop_missing_rows(df, [args.feature, CLF_TARGET_COL])

    if args.drop_classes:
        df = df[~df[CLF_TARGET_COL].isin(args.drop_classes)].reset_index(drop=True)
        print(f"  Dropped classes: {args.drop_classes}")

    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed).reset_index(drop=True)
        print(f"  Subsampled to {len(df)} rows")

    X = np.array(df[args.feature].tolist(), dtype=np.float32)
    generated_by = df[CLF_TARGET_COL].to_numpy()
    print(f"  {X.shape[0]} rows x {X.shape[1]} dims -> 2D via {args.method}")
    print(f"  Classes: {df[CLF_TARGET_COL].value_counts().to_dict()}")

    if args.method in SUPERVISED_METHODS:
        # A document and its rewrites share an id and must not straddle the
        # split, or the held-out points are not really held out.
        split_groups = df[GROUP_COL].to_numpy() if GROUP_COL in df.columns else None
        X2, keep_idx, axis_labels = reduce_supervised(
            X, generated_by, split_groups, args.method, args.seed
        )
        # Only the held-out rows are plotted, so the frame must follow.
        df = df.iloc[keep_idx].reset_index(drop=True)
        generated_by = generated_by[keep_idx]
    else:
        X2 = reduce(X, args.method, 2)
        axis_labels = (f"{args.method} 1", f"{args.method} 2")

    groups = build_groups(generated_by, args.by_generator)
    title = f"{parquet_path.stem} — {args.feature} ({args.method.upper()})"
    if args.method in SUPERVISED_METHODS:
        title += " — held-out rows only"

    out = args.out or RESULTS_DIR / f"{parquet_path.stem}_{args.method}.png"
    if not out.is_absolute():
        out = BASE_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)

    plot_static(X2, groups, title, args.method, out, axis_labels)
    print(f"\nSaved: {out}")

    for label, mask, _ in groups:
        cx, cy = centroid(X2, mask)
        print(f"  {label:<16} mean = ({cx:7.3f}, {cy:7.3f})")

    if not args.no_html:
        html_out = out.with_suffix(".html")
        plot_interactive(
            X2, groups, df, args.feature, title, args.method, html_out, axis_labels
        )
        print(f"Saved: {html_out}")


if __name__ == "__main__":
    main()

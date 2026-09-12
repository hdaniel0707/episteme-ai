r"""PCA scatter of one embedding across TWO corpora, in a shared projection --
built to look at whether ``emb_roberta`` actually occupies different regions
of space on MG-Sci vs. MG-AI-Sci, the diagnosis behind why no trained
detector (GT-labelled or judge-distilled -- see
``run_judge_distillation_performance.py``) closes the gap to the zero-shot
judge on MG-AI-Sci.

Unlike ``analyse_domain_pca.py`` (one file, domain = an internal column),
this fits ONE PCA on the union of two separate parquets, so the two corpora's
points land in the same coordinate frame and are directly comparable -- a
domain shift you can see rather than infer.

Colour is the class (human vs. every AI row pooled), FIXED across both
corpora -- blue human, red AI, the same two colours regardless of which file
a row comes from. Marker SHAPE is which corpus a row comes from. Four groups
result (dataset x class); each is its own trace in the HTML figure, so
clicking a legend entry toggles that one cloud (and its centroid) on or off,
same as ``analyse_domain_pca.py``'s interactive plots. A dashed line joins
each dataset's human/AI centroids -- the cross-corpus analogue of that
script's domain links.

Two outputs per dimensionality (2D, 3D): a static PNG and an interactive HTML
(plotly: click a legend entry to hide one group, or use the buttons to hide a
whole layer -- "Centroids + links" strips the clouds down to the geometry).

Usage:
    uv run python epai/ai_detection/analyse/analyse_cross_corpus_pca.py

    uv run python epai/ai_detection/analyse/analyse_cross_corpus_pca.py \
        data/parquet/manugram_gb.parquet data/parquet/ghostbuster_orig.parquet \
        --label-a MG-GB --label-b GB-orig

    uv run python epai/ai_detection/analyse/analyse_cross_corpus_pca.py \
        --no-html   # PNGs only, skip the (larger) interactive files
"""

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the Agg backend is chosen)
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (registers '3d')

BASE_DIR = Path(__file__).parent.parent.parent.parent
PARQUET_DIR = BASE_DIR / "data" / "parquet"
RESULTS_DIR = BASE_DIR / "results" / "analyse_cross_corpus_pca"

GEN_COL = "generated_by"
HUMAN_LABEL = "human"
DEFAULT_FEATURE = "emb_roberta"

DEFAULT_A = PARQUET_DIR / "manugram_sci.parquet"
DEFAULT_B = PARQUET_DIR / "manugram_aiscientist.parquet"
DEFAULT_LABEL_A = "MG-Sci"
DEFAULT_LABEL_B = "MG-AI-Sci"

HUMAN_COLOR = "#4e79a7"  # blue
AI_COLOR = "#e15759"     # red
LINK_COLOR = "#4b5563"   # neutral grey, belongs to neither class

# (matplotlib marker, plotly 2D symbol, plotly 3D symbol) per dataset --
# scatter3d has no triangles, so dataset B gets 'cross' there instead.
DATASET_MARKERS = (
    ("o", "circle", "circle"),
    ("^", "triangle-up", "cross"),
)
NORMALIZE_MODES = ("none", "l2", "standard")
HOVER_CHARS = 240


@dataclass
class Group:
    dataset: str
    klass: str  # "human" | "ai"
    color: str
    marker: str
    symbol2d: str
    symbol3d: str
    mask: np.ndarray
    centroid_pc: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def name(self) -> str:
        return f"{self.dataset} · {self.klass}"

    @property
    def n(self) -> int:
        return int(self.mask.sum())


def _load(path: Path, feature: str, label: str, sample: int | None, seed: int) -> pd.DataFrame:
    cols = [GEN_COL, feature]
    id_col = None
    for candidate in ("uid", "id"):
        if candidate not in cols and _has_column(path, candidate):
            id_col = candidate
            cols.append(candidate)
            break
    df = pd.read_parquet(path, columns=cols)
    df = df[df[feature].map(lambda v: v is not None and np.all(np.isfinite(v)))]
    df = df.reset_index(drop=True)
    if sample and sample < len(df):
        df = df.sample(sample, random_state=seed).reset_index(drop=True)
    df["_dataset"] = label
    df["_class"] = np.where(df[GEN_COL] == HUMAN_LABEL, HUMAN_LABEL, "ai")
    df["_id"] = df[id_col].astype(str) if id_col else ""
    return df


def _has_column(path: Path, col: str) -> bool:
    import pyarrow.parquet as pq
    return col in pq.read_schema(path).names


def normalize(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "l2":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms
    if mode == "standard":
        mean, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd == 0] = 1.0
        return (X - mean) / sd
    return X


def build_groups(df: pd.DataFrame, labels: tuple[str, str]) -> list[Group]:
    groups = []
    for dataset, markers in zip(labels, DATASET_MARKERS):
        for klass, color in ((HUMAN_LABEL, HUMAN_COLOR), ("ai", AI_COLOR)):
            mask = ((df["_dataset"] == dataset) & (df["_class"] == klass)).to_numpy()
            if mask.any():
                groups.append(Group(dataset, klass, color, *markers, mask))
    return groups


def dataset_links(groups: list[Group]) -> list[tuple[str, Group, Group]]:
    """(dataset, human group, ai group) for every link line to draw."""
    links = []
    for dataset in dict.fromkeys(g.dataset for g in groups):
        human = next((g for g in groups if g.dataset == dataset and g.klass == HUMAN_LABEL), None)
        ai = next((g for g in groups if g.dataset == dataset and g.klass == "ai"), None)
        if human is not None and ai is not None:
            links.append((dataset, human, ai))
    return links


# ---------------------------------------------------------------------------
# Static (matplotlib)
# ---------------------------------------------------------------------------


def plot_static(Z: np.ndarray, groups: list[Group], links, pc_labels: list[str],
                title: str, out: Path, three_d: bool) -> None:
    fig = plt.figure(figsize=(11, 9) if three_d else (10.5, 8))
    ax = fig.add_subplot(111, projection="3d") if three_d else fig.add_subplot(111)
    dims = 3 if three_d else 2

    for group in groups:
        coords = [Z[group.mask, i] for i in range(dims)]
        kwargs = dict(c=group.color, s=10, alpha=0.45, linewidths=0, marker=group.marker,
                     label=f"{group.name} (n={group.n})")
        if three_d:
            ax.scatter(*coords, depthshade=False, **kwargs)
        else:
            ax.scatter(*coords, **kwargs)

    for _, human, ai in links:
        coords = [[human.centroid_pc[i], ai.centroid_pc[i]] for i in range(dims)]
        ax.plot(*coords, color=LINK_COLOR, linewidth=1.6, linestyle="--", alpha=0.9, zorder=4)

    for group in groups:
        coords = [[group.centroid_pc[i]] for i in range(dims)]
        ax.scatter(*coords, marker="*", s=340, c=group.color, edgecolors="white",
                   linewidths=1.2, zorder=6, **({"depthshade": False} if three_d else {}))

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(pc_labels[0])
    ax.set_ylabel(pc_labels[1])
    if three_d:
        ax.set_zlabel(pc_labels[2])
        ax.view_init(elev=18, azim=-60)
    else:
        ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left" if three_d else "best", frameon=True, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interactive (plotly) -- same toggle model as analyse_domain_pca.py:
# legend click hides one group + its centroid; buttons hide a whole layer.
# ---------------------------------------------------------------------------


def hover_text(df: pd.DataFrame) -> np.ndarray:
    """One tooltip per row. Built from plain arrays, not ``itertuples`` --
    ``_dataset``/``_class``/``_id`` all start with an underscore, which
    ``namedtuple`` (what ``itertuples`` builds on) silently renames to a
    positional field, so ``row._dataset`` would not be the column at all."""
    ids = df["_id"].to_numpy()
    datasets = df["_dataset"].to_numpy()
    classes = df["_class"].to_numpy()
    out = []
    for uid, dataset, klass in zip(ids, datasets, classes):
        parts = []
        if uid:
            parts.append(f"<b>{html.escape(str(uid))}</b>")
        parts.append(f"dataset: {html.escape(str(dataset))}")
        parts.append(f"class: {html.escape(str(klass))}")
        out.append("<br>".join(parts))
    return np.array(out, dtype=object)


def plot_interactive(Z: np.ndarray, groups: list[Group], links, pc_labels: list[str],
                     hovers: np.ndarray, title: str, out: Path, three_d: bool,
                     plotly_js: object = True) -> None:
    fig = go.Figure()
    kinds: list[str] = []

    def coords(values: np.ndarray) -> dict:
        keys = ("x", "y", "z") if three_d else ("x", "y")
        return {k: values[:, i] for i, k in enumerate(keys)}

    Scatter = go.Scatter3d if three_d else go.Scattergl
    marker_kwargs = dict(size=2.5 if three_d else 5, opacity=0.55)

    def symbol(group: Group) -> str:
        return group.symbol3d if three_d else group.symbol2d

    for group in groups:
        fig.add_trace(Scatter(
            **coords(Z[group.mask]), mode="markers",
            name=f"{group.name} (n={group.n})", legendgroup=group.name,
            marker=dict(color=group.color, symbol=symbol(group), **marker_kwargs),
            text=hovers[group.mask], hovertemplate="%{text}<extra></extra>",
        ))
        kinds.append("points")

    for dataset, human, ai in links:
        line = np.vstack([human.centroid_pc, ai.centroid_pc])
        fig.add_trace(Scatter(
            **coords(line), mode="lines", name=f"{dataset} link", legendgroup=f"{dataset} link",
            line=dict(color=LINK_COLOR, width=4 if three_d else 2, dash="dash"),
            hovertemplate=f"<b>{dataset}</b><br>human ↔ ai<extra></extra>",
        ))
        kinds.append("links")

    for group in groups:
        point = group.centroid_pc[None, :]
        fig.add_trace(Scatter(
            **coords(point), mode="markers", name=f"{group.name} centroid",
            legendgroup=group.name, showlegend=False,
            marker=dict(color=group.color, symbol="diamond" if three_d else "star",
                       size=12 if three_d else 20, line=dict(color="white", width=1.5)),
            hovertemplate=(f"<b>{group.name} centroid</b><br>"
                           + "<br>".join(f"{pc_labels[i]}: {group.centroid_pc[i]:.3f}"
                                        for i in range(3 if three_d else 2))
                           + "<extra></extra>"),
        ))
        kinds.append("centroids")

    def visibility(*keep: str) -> list[bool]:
        return [k in keep for k in kinds]

    buttons = [
        dict(label="Everything", method="restyle",
             args=[{"visible": visibility("points", "links", "centroids")}]),
        dict(label="Points only", method="restyle",
             args=[{"visible": visibility("points")}]),
        dict(label="Centroids + links", method="restyle",
             args=[{"visible": visibility("links", "centroids")}]),
        dict(label="Points + centroids", method="restyle",
             args=[{"visible": visibility("points", "centroids")}]),
    ]

    layout = dict(
        title=title, template="plotly_white", hovermode="closest",
        width=1200, height=850,
        legend=dict(itemsizing="constant", groupclick="togglegroup"),
        updatemenus=[dict(type="buttons", direction="right", showactive=True,
                          x=0, y=1.09, xanchor="left", buttons=buttons)],
    )
    if three_d:
        layout["scene"] = dict(xaxis_title=pc_labels[0], yaxis_title=pc_labels[1],
                               zaxis_title=pc_labels[2])
    else:
        layout.update(xaxis_title=pc_labels[0], yaxis_title=pc_labels[1])
    fig.update_layout(**layout)
    fig.write_html(out, include_plotlyjs=plotly_js)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("parquet_a", type=Path, nargs="?", default=DEFAULT_A)
    ap.add_argument("parquet_b", type=Path, nargs="?", default=DEFAULT_B)
    ap.add_argument("--feature", default=DEFAULT_FEATURE,
                    help=f"embedding column to project (default: {DEFAULT_FEATURE})")
    ap.add_argument("--label-a", default=None, help=f"default: {DEFAULT_LABEL_A} "
                    "if parquet_a is the default file, else its stem")
    ap.add_argument("--label-b", default=None, help=f"default: {DEFAULT_LABEL_B} "
                    "if parquet_b is the default file, else its stem")
    ap.add_argument("--normalize", choices=NORMALIZE_MODES, default="none")
    ap.add_argument("--sample", type=int, default=None, metavar="N",
                    help="subsample EACH corpus to N rows before PCA/plotting")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-html", action="store_true",
                    help="write the PNGs but skip the interactive HTML")
    ap.add_argument("--plotly-cdn", action="store_true",
                    help="load plotly.js from a CDN instead of embedding it "
                    "(~3 MB smaller per HTML, needs network to open)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: results/analyse_cross_corpus_pca/<a>_vs_<b>_<feature>/")
    args = ap.parse_args()

    label_a = args.label_a or (DEFAULT_LABEL_A if args.parquet_a == DEFAULT_A else args.parquet_a.stem)
    label_b = args.label_b or (DEFAULT_LABEL_B if args.parquet_b == DEFAULT_B else args.parquet_b.stem)

    for p in (args.parquet_a, args.parquet_b):
        if not p.exists():
            raise SystemExit(f"parquet not found: {p}")

    out_dir = args.out_dir or RESULTS_DIR / f"{label_a}_vs_{label_b}_{args.feature}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_a = _load(args.parquet_a, args.feature, label_a, args.sample, args.seed)
    df_b = _load(args.parquet_b, args.feature, label_b, args.sample, args.seed)
    df = pd.concat([df_a, df_b], ignore_index=True)
    print(f"{label_a}: {len(df_a)} rows ({(df_a['_class'] == HUMAN_LABEL).sum()} human, "
          f"{(df_a['_class'] == 'ai').sum()} ai)")
    print(f"{label_b}: {len(df_b)} rows ({(df_b['_class'] == HUMAN_LABEL).sum()} human, "
          f"{(df_b['_class'] == 'ai').sum()} ai)")

    X = np.asarray(np.vstack(df[args.feature].to_numpy()), dtype=np.float64)
    X = normalize(X, args.normalize)

    n_components = int(min(10, X.shape[0], X.shape[1]))
    pca = PCA(n_components=n_components, random_state=0).fit(X)
    Z = pca.transform(X)
    evr = pca.explained_variance_ratio_
    print("\nPCA explained variance (fit on the pooled rows of both corpora):")
    for i, value in enumerate(evr[:5]):
        print(f"  PC{i + 1:<2} {100 * value:>6.2f}%  (cum {100 * evr[:i + 1].sum():>6.2f}%)")
    pc_labels = [f"PC{i + 1} ({100 * evr[i]:.1f}%)" for i in range(len(evr))]

    pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(evr))],
        "explained_variance_ratio": evr,
        "cumulative": np.cumsum(evr),
    }).to_csv(out_dir / "pca_variance.csv", index=False)

    groups = build_groups(df, (label_a, label_b))
    for group in groups:
        group.centroid_pc = Z[group.mask].mean(axis=0)[:3]
    links = dataset_links(groups)

    title = (f"{args.feature} — {label_a} vs. {label_b}"
             + (f" ({args.normalize})" if args.normalize != "none" else ""))
    hovers = hover_text(df) if not args.no_html else None

    for three_d in (False, True):
        if three_d and Z.shape[1] < 3:
            print("only 2 component(s) available — skipping the 3D figures")
            continue
        tag = "pca3d" if three_d else "pca2d"
        png = out_dir / f"{tag}.png"
        plot_static(Z, groups, links, pc_labels, title, png, three_d)
        print(f"wrote {png}")
        if args.no_html:
            continue
        html_out = out_dir / f"{tag}.html"
        plot_interactive(Z, groups, links, pc_labels, hovers, title, html_out, three_d,
                         plotly_js="cdn" if args.plotly_cdn else True)
        print(f"wrote {html_out}")
    print(f"wrote {out_dir / 'pca_variance.csv'}")


if __name__ == "__main__":
    main()

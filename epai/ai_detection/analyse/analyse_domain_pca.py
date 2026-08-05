r"""Describe a combined multi-domain parquet, and project one embedding into PCA space.

The companion to ``prepare/combine_domains.py``: that script builds one file out
of essay/reuter/wp, this one answers what is actually in it and what the chosen
embedding does with it. Two outputs, and the second is the point:

  statistics  counts, lengths, scores and embedding geometry per domain and
              class, printed and written to results/analyse_domain_pca/<run>/
  pictures    PCA in 2D and 3D, each as a PNG and an interactive HTML

Every group -- one (domain, class) combination -- gets its own colour, warm for
human and cold for machine, so the two questions the plot can answer stay
visually separate:

    does the embedding separate HUMAN from MACHINE?   warm vs cold
    does it separate the DOMAINS?                     hue within each temperature

A star marks each group's centroid, drawn above the cloud, and a line joins the
centroids belonging to one domain. That line is the thing to look at. If the
lines are short next to the distance between domains, the embedding is mostly
encoding topic, and the human/machine direction is a small correction on top of
a much larger domain signal -- which is the geometry behind a classifier that
scores ~98% in-domain and fails to transfer. If the lines are long and roughly
PARALLEL across domains, there is a shared human->machine direction and transfer
has something to stand on. If they point in different directions, a boundary
fitted on one domain is pointing the wrong way in another, and below-chance
transfer is the expected result rather than a surprise.

The same reading is available numerically, without the eyeball, in the centroid
geometry section: the between-domain / within-domain distance ratio, and the
cosine between each pair of domain human->machine vectors, both computed in the
FULL embedding space rather than in the 3 plotted components.

Usage:
    # defaults: emb_luar_crud, every plot and table
    uv run python epai/ai_detection/analyse/analyse_domain_pca.py \
        data/parquet/ghostbuster_gpt35t.parquet

    # another embedding
    uv run python epai/ai_detection/analyse/analyse_domain_pca.py \
        data/parquet/ghostbuster_gpt54mini.parquet --feature emb_openai_large

    # cosine-style embeddings: put every vector on the unit sphere first
    uv run python epai/ai_detection/analyse/analyse_domain_pca.py \
        data/parquet/ghostbuster_gpt35t.parquet --normalize l2

    # one colour per generator instead of the binary human/machine view
    uv run python epai/ai_detection/analyse/analyse_domain_pca.py \
        data/parquet/ghostbuster_gpt35t.parquet --by-generator

    # tables only, no figures
    uv run python epai/ai_detection/analyse/analyse_domain_pca.py \
        data/parquet/ghostbuster_gpt35t.parquet --no-plots
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the Agg backend is chosen)
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (registers '3d')

from epai.utils.config_utils import RESET, YELLOW  # noqa: E402
from epai.utils.parquet_utils import load_parquet  # noqa: E402
from epai.utils.train_utils import CLF_TARGET_COL, drop_missing_rows  # noqa: E402

BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "analyse_domain_pca"

DOMAIN_COL = "domain"
ID_COL = "id"
UID_COL = "uid"
TEXT_COL = "text"
HUMAN_LABEL = "human"
SCORE_PREFIXES = ("binoculars_score", "llm_judge_score")

DEFAULT_FEATURE = "emb_luar_crud"

# One sequential colormap per domain, sampled near its dark end. Warm families
# for human, cold for machine, indexed by the domain's position in sorted order
# so the pairing is stable across runs and across files.
WARM_CMAPS = ("Reds", "Oranges", "YlOrBr", "RdPu", "spring", "copper")
COLD_CMAPS = ("Blues", "GnBu", "Purples", "PuBu", "BuGn", "winter")
# Where in each colormap to sample. A single machine label per domain takes the
# midpoint; several generators spread over the range, dark to light.
SHADE_RANGE = (0.75, 0.45)

# The pair kind that answers "how far is human from machine, inside one domain".
# Named because both the distance table and its interpretation key off it.
WITHIN_CROSS = "within-domain, human vs machine"

LINK_COLOR = "#4b5563"  # neutral grey: the link belongs to neither class
HOVER_CHARS = 240
NORMALIZE_MODES = ("none", "l2", "standard")


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


@dataclass
class Group:
    """One (domain, label) cloud: its rows, its colour, its centroid."""

    domain: str
    label: str
    color: str
    mask: np.ndarray
    is_human: bool
    centroid_pc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    centroid_full: np.ndarray = field(default_factory=lambda: np.zeros(1))

    @property
    def name(self) -> str:
        return f"{self.domain} · {self.label}"

    @property
    def n(self) -> int:
        return int(self.mask.sum())


def shade(cmap_name: str, index: int, total: int) -> str:
    """Hex colour ``index`` of ``total`` from a sequential colormap."""
    lo, hi = SHADE_RANGE
    position = lo if total == 1 else lo + (hi - lo) * index / (total - 1)
    return mcolors.to_hex(plt.get_cmap(cmap_name)(position))


def build_groups(df: pd.DataFrame, by_generator: bool) -> list[Group]:
    """Machine groups first, human last, so the human cloud draws on top.

    Colour carries two facts at once: temperature is the class (warm human,
    cold machine) and the family within a temperature is the domain. That is
    the only way a single scatter can be read for both questions -- a palette
    keyed on the 6 groups alone would answer neither at a glance.
    """
    domains = sorted(df[DOMAIN_COL].astype(str).unique())
    labels = df[CLF_TARGET_COL].astype(str).to_numpy()
    domain_values = df[DOMAIN_COL].astype(str).to_numpy()

    machine_groups: list[Group] = []
    human_groups: list[Group] = []

    for d_index, domain in enumerate(domains):
        in_domain = domain_values == domain
        warm = WARM_CMAPS[d_index % len(WARM_CMAPS)]
        cold = COLD_CMAPS[d_index % len(COLD_CMAPS)]

        human_mask = in_domain & (labels == HUMAN_LABEL)
        if human_mask.any():
            human_groups.append(Group(domain, HUMAN_LABEL, shade(warm, 0, 1),
                                      human_mask, True))

        machine_labels = sorted(set(labels[in_domain & (labels != HUMAN_LABEL)]))
        if not by_generator:
            machine_mask = in_domain & (labels != HUMAN_LABEL)
            if machine_mask.any():
                name = machine_labels[0] if len(machine_labels) == 1 else "machine"
                machine_groups.append(Group(domain, name, shade(cold, 0, 1),
                                            machine_mask, False))
            continue
        for g_index, generator in enumerate(machine_labels):
            machine_groups.append(Group(
                domain, generator, shade(cold, g_index, len(machine_labels)),
                in_domain & (labels == generator), False,
            ))

    return machine_groups + human_groups


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class Report:
    """Print a line and keep it, so the console output is the saved file."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        print(line)
        self.lines.append(line)

    def rule(self, title: str = "") -> None:
        self("")
        self("=" * 88)
        if title:
            self(title)
            self("=" * 88)

    def frame(self, table: pd.DataFrame, floatfmt: str = "%.3f") -> None:
        text = table.to_string(float_format=lambda v: floatfmt % v)
        for line in text.splitlines():
            self("  " + line)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n")


# ---------------------------------------------------------------------------
# Corpus statistics
# ---------------------------------------------------------------------------


def counts_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rows per domain x generated_by, plus each domain's machine share."""
    table = (df.groupby([DOMAIN_COL, CLF_TARGET_COL]).size()
             .unstack(fill_value=0).sort_index())
    table["total"] = table.sum(axis=1)
    human = table[HUMAN_LABEL] if HUMAN_LABEL in table.columns else 0
    table["machine_%"] = 100.0 * (table["total"] - human) / table["total"]
    return table


def length_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean words and characters per domain and class, and the machine-human gap.

    Length is the cheapest shortcut a text classifier can take, and the one that
    breaks first across domains: a gap of opposite SIGN in two domains is enough
    on its own to produce good in-domain scores and below-chance transfer.
    """
    text = df[TEXT_COL].fillna("").astype(str) if TEXT_COL in df.columns else None
    if text is None:
        return pd.DataFrame()
    stats = pd.DataFrame({
        DOMAIN_COL: df[DOMAIN_COL],
        "class": np.where(df[CLF_TARGET_COL] == HUMAN_LABEL, HUMAN_LABEL, "machine"),
        "words": text.str.split().str.len(),
        "chars": text.str.len(),
    })
    table = stats.groupby([DOMAIN_COL, "class"])[["words", "chars"]].mean().unstack()
    for metric in ("words", "chars"):
        classes = table[metric].columns
        if HUMAN_LABEL in classes and "machine" in classes:
            table[(metric, "gap")] = (table[(metric, "machine")]
                                      - table[(metric, HUMAN_LABEL)])
    return table.sort_index(axis=1)


def score_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/sd of every scalar score, per domain and class, in long form."""
    cols = [c for c in df.columns if c.startswith(SCORE_PREFIXES)]
    if not cols:
        return pd.DataFrame()
    rows = []
    klass = np.where(df[CLF_TARGET_COL] == HUMAN_LABEL, HUMAN_LABEL, "machine")
    for col in cols:
        grouped = df.assign(_class=klass).groupby([DOMAIN_COL, "_class"])[col]
        for (domain, cls), values in grouped:
            rows.append({
                "score": col, DOMAIN_COL: domain, "class": cls,
                "n": int(values.notna().sum()),
                "mean": float(values.mean()), "sd": float(values.std()),
                "min": float(values.min()), "max": float(values.max()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Embedding geometry
# ---------------------------------------------------------------------------


def norm_table(X: np.ndarray, groups: list[Group]) -> pd.DataFrame:
    """L2 norm of the raw vectors per group.

    A norm that differs systematically between two groups is a one-number
    separator: any classifier can find it, and it usually reflects text length
    or a preprocessing difference rather than authorship.
    """
    rows = []
    for group in groups:
        norms = np.linalg.norm(X[group.mask], axis=1)
        rows.append({"group": group.name, "n": group.n,
                     "norm_mean": float(norms.mean()),
                     "norm_sd": float(norms.std()),
                     "norm_min": float(norms.min()),
                     "norm_max": float(norms.max())})
    return pd.DataFrame(rows).set_index("group")


def centroid_distances(groups: list[Group]) -> pd.DataFrame:
    """Every pairwise centroid distance in the FULL space, euclidean and cosine.

    Full space, not the 3 plotted components: the picture shows the projection
    that maximises variance, and if the class difference lives outside it the
    plot understates the separation. These numbers do not.
    """
    rows = []
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            diff = a.centroid_full - b.centroid_full
            denom = np.linalg.norm(a.centroid_full) * np.linalg.norm(b.centroid_full)
            cosine = float(a.centroid_full @ b.centroid_full / denom) if denom else np.nan
            same_domain, same_class = a.domain == b.domain, a.is_human == b.is_human
            rows.append({
                "a": a.name, "b": b.name,
                "kind": (WITHIN_CROSS if same_domain and not same_class else
                         "within-domain, same class" if same_domain else
                         "between-domain, same class" if same_class else
                         "between-domain, cross-class"),
                "euclidean": float(np.linalg.norm(diff)),
                "cosine": cosine,
            })
    return pd.DataFrame(rows)


def direction_table(groups: list[Group]) -> pd.DataFrame:
    """Cosine between the human->machine vectors of each pair of domains.

    This is the transfer question in one number per pair. A classifier trained
    on domain A applies A's direction to domain B; if the two directions are
    near-orthogonal (cosine ~ 0) that classifier is guessing, and if they are
    anti-parallel (cosine < 0) it is confidently backwards -- which is what
    below-chance transfer looks like from the inside.
    """
    vectors: dict[str, np.ndarray] = {}
    for domain in sorted({g.domain for g in groups}):
        humans = [g for g in groups if g.domain == domain and g.is_human]
        machines = [g for g in groups if g.domain == domain and not g.is_human]
        if not humans or not machines:
            continue
        machine_mean = np.mean([g.centroid_full for g in machines], axis=0)
        vectors[domain] = machine_mean - humans[0].centroid_full

    rows = []
    names = list(vectors)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va, vb = vectors[a], vectors[b]
            denom = np.linalg.norm(va) * np.linalg.norm(vb)
            rows.append({
                "domain_a": a, "domain_b": b,
                "cosine": float(va @ vb / denom) if denom else np.nan,
                "len_a": float(np.linalg.norm(va)),
                "len_b": float(np.linalg.norm(vb)),
            })
    return pd.DataFrame(rows)


def interpret(distances: pd.DataFrame, directions: pd.DataFrame, report: Report) -> dict:
    """Turn the two tables into the two numbers worth quoting."""
    within = distances.loc[distances["kind"] == WITHIN_CROSS, "euclidean"]
    between = distances.loc[distances["kind"].str.startswith("between-domain"),
                            "euclidean"]
    summary: dict[str, float] = {}
    if len(within) and len(between):
        ratio = float(between.mean() / within.mean()) if within.mean() else float("inf")
        summary["within_domain_mean"] = float(within.mean())
        summary["between_domain_mean"] = float(between.mean())
        summary["between_over_within"] = ratio
        report("")
        report(f"  human->machine centroid distance, within a domain : "
               f"{within.mean():.4f}")
        report(f"  centroid distance between domains                 : "
               f"{between.mean():.4f}")
        report(f"  ratio (between / within)                          : {ratio:.2f}")
        if ratio > 2:
            report(f"{YELLOW}  The domains are further apart than the classes are. "
                   f"In this embedding the\n  dominant axis is topic, so a boundary "
                   f"fitted in one domain sits inside another\n  domain's cloud "
                   f"rather than between its classes.{RESET}")
    if not directions.empty:
        summary["direction_cosine_mean"] = float(directions["cosine"].mean())
        summary["direction_cosine_min"] = float(directions["cosine"].min())
        report("")
        report("  cosine between the human->machine directions of two domains:")
        for _, row in directions.iterrows():
            verdict = ("aligned — a shared direction to transfer"
                       if row["cosine"] > 0.5 else
                       "weakly aligned" if row["cosine"] > 0.2 else
                       "near-orthogonal — nothing to transfer" if row["cosine"] > -0.1
                       else "OPPOSED — a transferred boundary points backwards")
            report(f"    {row['domain_a']:>8} vs {row['domain_b']:<8} "
                   f"{row['cosine']:>+6.3f}   {verdict}")
    return summary


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def normalize(X: np.ndarray, mode: str, report: Report) -> np.ndarray:
    if mode == "l2":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        report("  Normalisation: L2 (every vector on the unit sphere, so PCA sees "
               "direction only)")
        return X / norms
    if mode == "standard":
        mean, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd == 0] = 1.0
        report("  Normalisation: per-dimension standardisation")
        return (X - mean) / sd
    report("  Normalisation: none (PCA centres the data itself)")
    return X


def run_pca(X: np.ndarray, report: Report) -> tuple[np.ndarray, PCA, list[str]]:
    n_components = int(min(10, X.shape[0], X.shape[1]))
    pca = PCA(n_components=n_components, random_state=0).fit(X)
    Z = pca.transform(X)
    evr = pca.explained_variance_ratio_

    report("")
    report("  PCA explained variance")
    for i, value in enumerate(evr):
        bar = "#" * int(round(60 * value / max(evr[0], 1e-12)))
        report(f"    PC{i + 1:<2} {100 * value:>6.2f}%  "
               f"(cum {100 * evr[:i + 1].sum():>6.2f}%)  {bar}")
    if len(evr) >= 3 and evr[:3].sum() < 0.25:
        report(f"{YELLOW}  The first three components hold only "
               f"{100 * evr[:3].sum():.1f}% of the variance, so the scatter is a "
               f"thin slice\n  of this embedding. Read the centroid geometry below "
               f"(computed in the full space)\n  before trusting the picture.{RESET}")

    labels = [f"PC{i + 1} ({100 * evr[i]:.1f}%)" for i in range(len(evr))]
    return Z, pca, labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def domain_links(groups: list[Group]) -> list[tuple[str, Group, Group]]:
    """(domain, human group, machine group) for every link line to draw."""
    links = []
    for domain in sorted({g.domain for g in groups}):
        humans = [g for g in groups if g.domain == domain and g.is_human]
        machines = [g for g in groups if g.domain == domain and not g.is_human]
        for human in humans:
            for machine in machines:
                links.append((domain, human, machine))
    return links


def plot_static(
    Z: np.ndarray, groups: list[Group], links, labels: list[str],
    title: str, out: Path, three_d: bool,
) -> None:
    """Matplotlib PNG, 2D or 3D: clouds, link lines, then centroid stars on top."""
    fig = plt.figure(figsize=(11, 9) if three_d else (10.5, 8))
    ax = fig.add_subplot(111, projection="3d") if three_d else fig.add_subplot(111)

    for group in groups:
        coords = [Z[group.mask, i] for i in range(3 if three_d else 2)]
        kwargs = dict(c=group.color, s=7, alpha=0.35, linewidths=0)
        if three_d:
            ax.scatter(*coords, depthshade=False, **kwargs)
        else:
            ax.scatter(*coords, **kwargs)

    # Links under the stars but over the points: the line is context for the
    # two stars it joins, and it must not be lost inside the cloud.
    for _, human, machine in links:
        coords = [[human.centroid_pc[i], machine.centroid_pc[i]]
                  for i in range(3 if three_d else 2)]
        ax.plot(*coords, color=LINK_COLOR, linewidth=1.6, linestyle="--",
                alpha=0.9, zorder=4)

    for group in groups:
        coords = [[group.centroid_pc[i]] for i in range(3 if three_d else 2)]
        ax.scatter(*coords, marker="*", s=520, c=group.color,
                   edgecolors="white", linewidths=1.4, zorder=6,
                   **({"depthshade": False} if three_d else {}))

    handles = [
        plt.Line2D([], [], linestyle="none", marker="o", markersize=7,
                   markerfacecolor=group.color, markeredgecolor="none",
                   label=f"{group.name} (n={group.n})")
        for group in groups
    ]
    handles += [
        plt.Line2D([], [], linestyle="none", marker="*", markersize=15,
                   markerfacecolor="#9ca3af", markeredgecolor="white",
                   markeredgewidth=1.0, label="group centroid"),
        plt.Line2D([], [], linestyle="--", color=LINK_COLOR, linewidth=1.6,
                   label="human ↔ machine, same domain"),
    ]

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if three_d:
        ax.set_zlabel(labels[2])
        ax.view_init(elev=18, azim=-60)
    else:
        ax.spines[["top", "right"]].set_visible(False)
    # 'best' does a layout search the 3D axes do not support well; pin it there.
    ax.legend(handles=handles, loc="upper left" if three_d else "best",
              frameon=True, fontsize=8.5, labelspacing=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def hover_text(df: pd.DataFrame) -> np.ndarray:
    """One tooltip per row: identity, class, scores, and a snippet of the text.

    Built from a narrow copy of the frame: iterating rows of the full one would
    materialise every embedding column into a Series per row, which is minutes
    of work for a tooltip that shows none of it.
    """
    id_col = UID_COL if UID_COL in df.columns else ID_COL
    extras = [c for c in df.columns if c.startswith(SCORE_PREFIXES)]
    texts = (df[TEXT_COL].fillna("").astype(str) if TEXT_COL in df.columns
             else pd.Series([""] * len(df), index=df.index))
    narrow = df[[c for c in (id_col, DOMAIN_COL, CLF_TARGET_COL, *extras)
                 if c in df.columns]]

    out = []
    for (_, row), text in zip(narrow.iterrows(), texts):
        parts = [f"<b>{html.escape(str(row.get(id_col, '')))}</b>",
                 f"{DOMAIN_COL}: {html.escape(str(row[DOMAIN_COL]))}",
                 f"{CLF_TARGET_COL}: {html.escape(str(row[CLF_TARGET_COL]))}"]
        for col in extras:
            if pd.notna(row[col]):
                parts.append(f"{col}: {row[col]:.4f}" if isinstance(row[col], float)
                             else f"{col}: {html.escape(str(row[col]))}")
        snippet = html.escape(text[:HOVER_CHARS].replace("\n", " "))
        if len(text) > HOVER_CHARS:
            snippet += "…"
        wrapped = "<br>".join(snippet[i:i + 60] for i in range(0, len(snippet), 60))
        parts.append(f"chars: {len(text)}<br><br>{wrapped}")
        out.append("<br>".join(parts))
    return np.array(out, dtype=object)


def plot_interactive(
    Z: np.ndarray, groups: list[Group], links, labels: list[str],
    hovers: np.ndarray, title: str, out: Path, three_d: bool,
    plotly_js: object = True,
) -> None:
    """Plotly HTML: same figure, but every layer can be switched off.

    Two ways to switch, because they answer different questions. Clicking a
    legend entry hides one group (its points and its star together, via
    legendgroup) -- for looking at two clouds without the others in the way.
    The buttons hide a whole LAYER: 'centroids only' strips the clouds and
    leaves the skeleton of stars and links, which is where the domain-vs-class
    geometry is actually readable.
    """
    fig = go.Figure()
    kinds: list[str] = []  # layer of each trace, in order, for the buttons

    def coords(values: np.ndarray) -> dict:
        keys = ("x", "y", "z") if three_d else ("x", "y")
        return {k: values[:, i] for i, k in enumerate(keys)}

    Scatter = go.Scatter3d if three_d else go.Scattergl
    marker_kwargs = dict(size=2.5 if three_d else 5, opacity=0.55)

    for group in groups:
        fig.add_trace(Scatter(
            **coords(Z[group.mask]),
            mode="markers",
            name=f"{group.name} (n={group.n})",
            legendgroup=group.name,
            marker=dict(color=group.color, **marker_kwargs),
            text=hovers[group.mask],
            hovertemplate="%{text}<extra></extra>",
        ))
        kinds.append("points")

    seen_domains: set[str] = set()
    for domain, human, machine in links:
        line = np.vstack([human.centroid_pc, machine.centroid_pc])
        fig.add_trace(Scatter(
            **coords(line),
            mode="lines",
            name=f"{domain} link",
            legendgroup=f"{domain} link",
            # One legend entry per domain, however many generators it links.
            showlegend=domain not in seen_domains,
            line=dict(color=LINK_COLOR, width=4 if three_d else 2, dash="dash"),
            hovertemplate=(f"<b>{domain}</b><br>"
                           f"{human.label} ↔ {machine.label}<extra></extra>"),
        ))
        kinds.append("links")

    # Added last so they render above every cloud.
    for group in groups:
        point = group.centroid_pc[None, :]
        fig.add_trace(Scatter(
            **coords(point),
            mode="markers",
            name=f"{group.name} centroid",
            legendgroup=group.name,
            showlegend=False,
            marker=dict(
                color=group.color, symbol="diamond" if three_d else "star",
                size=12 if three_d else 20,
                line=dict(color="white", width=1.5),
            ),
            hovertemplate=(f"<b>{group.name} centroid</b><br>"
                           + "<br>".join(f"{labels[i]}: {group.centroid_pc[i]:.3f}"
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

    axis_titles = dict(zip(("xaxis_title", "yaxis_title"), labels[:2]))
    layout = dict(
        title=title,
        template="plotly_white",
        hovermode="closest",
        width=1200, height=850,
        legend=dict(itemsizing="constant"),
        updatemenus=[dict(type="buttons", direction="right", showactive=True,
                          x=0, y=1.09, xanchor="left", buttons=buttons)],
    )
    if three_d:
        layout["scene"] = dict(xaxis_title=labels[0], yaxis_title=labels[1],
                               zaxis_title=labels[2])
    else:
        layout.update(axis_titles)
    fig.update_layout(**layout)
    # plotly.js is embedded by default (~3 MB per file) so the page works with
    # no network; --plotly-cdn swaps it for a script tag when size matters.
    fig.write_html(out, include_plotlyjs=plotly_js)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("parquet", type=Path,
                   help="Combined parquet, e.g. data/parquet/ghostbuster_gpt35t.parquet")
    p.add_argument("--feature", default=DEFAULT_FEATURE,
                   help="Embedding column to analyse.")
    p.add_argument("--normalize", choices=NORMALIZE_MODES, default="none",
                   help="Vector preprocessing before PCA. 'l2' for embeddings "
                   "meant to be compared by cosine.")
    p.add_argument("--by-generator", action="store_true",
                   help="One colour per generator instead of one per domain and "
                   "class (still cold for machine, warm for human).")
    p.add_argument("--drop-classes", nargs="*", default=[], metavar="CLS",
                   help="generated_by labels to exclude before analysing.")
    p.add_argument("--sample", type=int, default=None, metavar="N",
                   help="Subsample to N rows before plotting (the 3D HTML gets "
                   "heavy past ~20k points).")
    p.add_argument("--seed", type=int, default=42, help="Seed for --sample.")
    p.add_argument("--no-plots", action="store_true",
                   help="Statistics only; skip the four figures.")
    p.add_argument("--no-html", action="store_true",
                   help="Write the PNGs but skip the interactive HTML.")
    p.add_argument("--plotly-cdn", action="store_true",
                   help="Load plotly.js from a CDN instead of embedding it: "
                   "~3 MB smaller per HTML, but needs a network connection to "
                   "open.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory. Default: "
                   "results/analyse_domain_pca/<parquet stem>_<feature>/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = Report()

    parquet_path = (args.parquet if args.parquet.is_absolute()
                    else BASE_DIR / args.parquet)
    out_dir = args.out_dir or RESULTS_DIR / f"{parquet_path.stem}_{args.feature}"
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report.rule(f"analyse_domain_pca.py — {datetime.now():%Y-%m-%d %H:%M:%S}")
    report(f"  Parquet : {parquet_path}")
    report(f"  Feature : {args.feature}")
    report(f"  Output  : {out_dir}")

    df = load_parquet(parquet_path)
    for col in (DOMAIN_COL, CLF_TARGET_COL):
        if col not in df.columns:
            raise SystemExit(
                f"{parquet_path.name}: no {col!r} column. This script expects the "
                "output of prepare/combine_domains.py."
            )
    if args.feature not in df.columns:
        available = sorted(c for c in df.columns if c.startswith("emb_"))
        raise SystemExit(f"{parquet_path.name}: no column {args.feature!r}. "
                         f"Embeddings present: {available}")

    df = drop_missing_rows(df, [args.feature, CLF_TARGET_COL])
    if args.drop_classes:
        df = df[~df[CLF_TARGET_COL].isin(args.drop_classes)].reset_index(drop=True)
        report(f"  Dropped classes: {args.drop_classes}")
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed).reset_index(drop=True)
        report(f"  Subsampled to {len(df)} rows (seed {args.seed})")
    df = df.reset_index(drop=True)

    X = np.asarray(np.vstack(df[args.feature].to_numpy()), dtype=np.float64)
    report(f"  Rows    : {len(df)}   embedding dim: {X.shape[1]}")

    # --- corpus ------------------------------------------------------------
    report.rule("CORPUS")
    counts = counts_table(df)
    report("  Rows per domain x generated_by")
    report.frame(counts, "%.1f")

    lengths = length_table(df)
    if not lengths.empty:
        report("")
        report("  Mean document length (gap = machine - human)")
        report.frame(lengths, "%.0f")

    scores = score_table(df)
    if not scores.empty:
        report("")
        report("  Scores")
        report.frame(scores.set_index(["score", DOMAIN_COL, "class"]))

    if ID_COL in df.columns:
        pairs = df.groupby(ID_COL)[CLF_TARGET_COL].nunique()
        report("")
        report(f"  Pairing: {len(pairs)} distinct {ID_COL!r}, "
               f"{int((pairs > 1).sum())} carrying more than one class")

    # --- embedding ---------------------------------------------------------
    report.rule(f"EMBEDDING — {args.feature}")
    X = normalize(X, args.normalize, report)
    groups = build_groups(df, args.by_generator)
    report(f"  Groups: {len(groups)} — " + ", ".join(f"{g.name} ({g.n})" for g in groups))

    norms = norm_table(X, groups)
    report("")
    report("  Vector norms per group")
    report.frame(norms)

    Z, pca, pc_labels = run_pca(X, report)
    for group in groups:
        group.centroid_pc = Z[group.mask].mean(axis=0)[:3]
        group.centroid_full = X[group.mask].mean(axis=0)

    centroids = pd.DataFrame(
        [{"group": g.name, "n": g.n,
          **{pc_labels[i].split()[0]: g.centroid_pc[i]
             for i in range(min(3, Z.shape[1]))}} for g in groups]
    ).set_index("group")
    report("")
    report("  Centroids in PCA space")
    report.frame(centroids)

    report.rule("CENTROID GEOMETRY (full embedding space)")
    distances = centroid_distances(groups)
    report.frame(distances.set_index(["a", "b"]))
    directions = direction_table(groups)
    summary = interpret(distances, directions, report)

    # --- save --------------------------------------------------------------
    report.rule("OUTPUT")
    tables = {
        "counts.csv": counts, "lengths.csv": lengths, "scores.csv": scores,
        "norms.csv": norms, "centroids_pca.csv": centroids,
        "centroid_distances.csv": distances, "class_directions.csv": directions,
        "pca_variance.csv": pd.DataFrame({
            "component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        }),
    }
    for name, table in tables.items():
        if table is None or table.empty:
            continue
        table.to_csv(out_dir / name)
        report(f"  {out_dir / name}")

    summary.update({
        "parquet": str(parquet_path), "feature": args.feature,
        "normalize": args.normalize, "rows": int(len(df)),
        "dim": int(X.shape[1]),
        "explained_variance_3": float(pca.explained_variance_ratio_[:3].sum()),
        "groups": {g.name: g.n for g in groups},
    })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    report(f"  {out_dir / 'summary.json'}")

    # --- figures -----------------------------------------------------------
    if not args.no_plots:
        links = domain_links(groups)
        title = (f"{parquet_path.stem} — {args.feature}"
                 + (f" ({args.normalize})" if args.normalize != "none" else ""))
        hovers = None
        for three_d in (False, True):
            if three_d and Z.shape[1] < 3:
                report(f"{YELLOW}  Only {Z.shape[1]} component(s) available — "
                       f"skipping the 3D figures.{RESET}")
                continue
            tag = "pca3d" if three_d else "pca2d"
            png = out_dir / f"{tag}.png"
            plot_static(Z, groups, links, pc_labels, title, png, three_d)
            report(f"  {png}")
            if args.no_html:
                continue
            if hovers is None:
                hovers = hover_text(df)
            html_out = out_dir / f"{tag}.html"
            plot_interactive(Z, groups, links, pc_labels, hovers, title,
                             html_out, three_d,
                             plotly_js="cdn" if args.plotly_cdn else True)
            report(f"  {html_out}")

    stats_path = out_dir / "stats.txt"
    report(f"  {stats_path}")
    report.save(stats_path)


if __name__ == "__main__":
    main()

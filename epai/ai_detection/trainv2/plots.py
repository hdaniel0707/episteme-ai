"""One figure family: a small-multiples grid of (AUROC, TPR@FPR) scatters.

Rows are embeddings, columns are algorithms, one figure per domain slice. Inside
each panel, one point per generator — the answer to "which generator does this
detector catch, and does it hold at a 1%-FPR operating point".

Deliberately plain: no gridlines behind the marks, no dual axes, fixed
categorical colours in slot order (the data-viz default palette — its first
three slots clear the all-pairs colour-vision gate, which is why three
generators is the comfortable case), and every point is directly labelled so
identity never rests on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from .core import POOLED

#: Categorical slots 1–3 of the validated default palette (all-pairs safe).
#: A 4th+ generator extends into the later slots and prints a note — past three,
#: colour alone is not reliable, but the direct labels still carry identity.
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7",
            "#008300", "#e34948"]
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_MUTED = "#8a8a86"


def _short(generator: str) -> str:
    """``gpt56luna:0811A`` -> ``gpt56luna``."""
    return generator.split(":", 1)[0]


def _emb_short(column: str) -> str:
    """``emb_luar_crud`` -> ``luar_crud``."""
    return column[4:] if column.startswith("emb_") else column


def _domain_title(domain: str) -> str:
    return "pooled (all domains)" if domain == POOLED else domain


def scatter_grids(summary_rows: list[dict], meta: dict, fig_dir: Path) -> list[Path]:
    """Write ``scatter_<domain>.png`` for every domain in ``summary_rows``."""
    fpr_pct = f"{meta['fpr'] * 100:g}"
    embeddings = sorted({r["embedding"] for r in summary_rows})
    algorithms = sorted({r["algorithm"] for r in summary_rows})
    generators = sorted({r["generator"] for r in summary_rows})
    domains = sorted({r["domain"] for r in summary_rows})
    colour = {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(generators)}

    out: list[Path] = []
    for domain in domains:
        rows = [r for r in summary_rows if r["domain"] == domain]
        path = fig_dir / f"scatter_{domain}.png"
        _one_grid(
            rows, embeddings, algorithms, colour, generators,
            xlim=_axis_range([r["auroc_mean"] for r in rows]),
            ylim=_axis_range([r["tpr_at_fpr_mean"] for r in rows], hard_lo=0.0),
            fpr_level=meta["fpr"], fpr_pct=fpr_pct,
            title=f"{meta['experiment']} — {_domain_title(domain)}",
            path=path,
        )
        out.append(path)
    return out


def _axis_range(values, *, hard_lo=None, min_span=0.05, ceil_cap=1.008):
    """Tight bounds around the data with a margin.

    In-domain the points cluster hard against 1.0, so a fixed 0.5–1.0 axis would
    stack them in a corner; this zooms to where the data actually is. A rate
    cannot exceed 1.0, so the top is capped just past it (whitespace there is the
    visible ceiling), and the bottom is clamped to ``hard_lo`` when given.
    """
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return (0.0 if hard_lo is None else hard_lo, 1.0)
    lo, hi = min(vals), max(vals)
    pad = max(hi - lo, min_span) * 0.30
    lo, hi = lo - pad, min(hi + pad, ceil_cap)
    if hard_lo is not None:
        lo = max(lo, hard_lo)
    return lo, hi


def _one_grid(rows, embeddings, algorithms, colour, generators, *,
              xlim, ylim, fpr_level, fpr_pct, title, path):
    nr, nc = len(embeddings), len(algorithms)
    fig, axes = plt.subplots(
        nr, nc, figsize=(3.3 * nc + 1.0, 3.0 * nr + 1.0),
        squeeze=False, sharex=True, sharey=True,
    )
    fig.patch.set_facecolor("white")
    stroke = [pe.withStroke(linewidth=2, foreground="white")]
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_cell.setdefault((r["embedding"], r["algorithm"]), []).append(r)

    for i, emb in enumerate(embeddings):
        for j, alg in enumerate(algorithms):
            ax = axes[i][j]
            ax.set_facecolor(_SURFACE)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if xlim[0] <= 0.5 <= xlim[1]:
                ax.axvline(0.5, color=_MUTED, lw=0.8, ls=":", zorder=0)
            if ylim[0] <= fpr_level <= ylim[1]:
                ax.axhline(fpr_level, color=_MUTED, lw=0.8, ls=":", zorder=0)

            cell = sorted(by_cell.get((emb, alg), []), key=lambda r: r["generator"])
            y0, y1 = ylim
            for k, r in enumerate(cell):
                g = r["generator"]
                yv = r["tpr_at_fpr_mean"]
                ax.errorbar(
                    r["auroc_mean"], yv,
                    xerr=r["auroc_std"], yerr=r["tpr_at_fpr_std"],
                    fmt="o", ms=8, mfc=colour[g], mec="white", mew=1.2,
                    ecolor=colour[g], elinewidth=1, capsize=2, alpha=0.95, zorder=3,
                )
                # Label to the right of the marker, clear of the vertical error
                # bar; stagger vertically by index so stacked points stay apart.
                dy = 7 - 12 * (k % 3)
                ax.annotate(
                    _short(g), (r["auroc_mean"], yv),
                    textcoords="offset points", xytext=(9, dy), ha="left", va="center",
                    fontsize=7, color=_INK, zorder=4, path_effects=stroke,
                )

            ax.set_title(f"{_emb_short(emb)} · {alg}", fontsize=9, color=_INK)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.tick_params(labelsize=7.5, colors=_MUTED)
            if i == nr - 1:
                ax.set_xlabel("AUROC", fontsize=8.5, color=_INK)
            if j == 0:
                ax.set_ylabel(f"TPR @ {fpr_pct}% FPR (oracle)", fontsize=8.5, color=_INK)

    # The legend anchor and the reserved bottom margin scale with figure height
    # rather than a fixed fraction, so a taller grid (more embeddings) doesn't
    # waste space and a short one still fits both rows below the panels. A
    # NEGATIVE bbox_to_anchor y (placing the legend below the canvas) combined
    # with a nearby fig.text reliably corrupts matplotlib's tight-bbox export
    # into wildly oversized text — keep both anchors positive and separated.
    height = fig.get_size_inches()[1]
    legend_y, caption_y, rect_bottom = 0.34 / height, 0.08 / height, 0.62 / height

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=7, color=colour[g], label=_short(g))
        for g in generators
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=min(len(generators), 6),
        frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, legend_y),
    )
    fig.suptitle(title, fontsize=11, color=_INK, y=0.995)
    fig.text(
        0.5, caption_y,
        "each point = one human-vs-generator training · error bars = ±1 SD over seeds "
        "· axes zoomed to the data",
        ha="center", fontsize=7, color=_MUTED,
    )
    fig.tight_layout(rect=(0, rect_bottom, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Embedding effectiveness across the SCRIBE splits
# (run_embedding_effectiveness.py). Two Cleveland dot plots, not scatters:
# with a dozen embeddings a 2-D scatter piles labels on top of each other
# right where the answer lives (near the ceiling). A ranked list scales to as
# many rows as there are embeddings and states the answer in the row order.
# Both read the summary CSV's records (dicts), incl. the split=="mean" rows.
# ---------------------------------------------------------------------------


def _emb_rows(summary_rows: list[dict], split: str) -> dict[str, dict]:
    return {r["embedding"]: r for r in summary_rows if r["split"] == split}


def emb_mean_ranking(summary_rows: list[dict], meta: dict, path: Path) -> None:
    """Embeddings ranked by mean AUROC across the SCRIBE splits, AUROC and
    TPR@FPR as two series on one 0-1 rate axis. ``xerr`` is the spread ACROSS
    splits (does it hold up on every benchmark), not seed noise."""
    mean = _emb_rows(summary_rows, "mean")
    order = sorted(mean, key=lambda e: mean[e]["auroc_mean"], reverse=True)
    n = len(order)
    y_of = {e: n - 1 - i for i, e in enumerate(order)}  # best on top
    auroc_c, tpr_c = _PALETTE[0], _PALETTE[1]

    fig, ax = plt.subplots(figsize=(6.6, 0.5 * n + 1.6))
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for e in order:
        r, y = mean[e], y_of[e]
        ax.plot([r["auroc_mean"], r["tpr_at_fpr_mean"]], [y + 0.16, y - 0.16],
                color=_MUTED, lw=0.8, zorder=1)
        ax.errorbar(r["auroc_mean"], y + 0.16, xerr=r["auroc_std"],
                    fmt="o", ms=7, color=auroc_c, ecolor=auroc_c,
                    elinewidth=1, capsize=2, zorder=3)
        ax.errorbar(r["tpr_at_fpr_mean"], y - 0.16, xerr=r["tpr_at_fpr_std"],
                    fmt="D", ms=6, color=tpr_c, ecolor=tpr_c,
                    elinewidth=1, capsize=2, zorder=3)

    ax.set_yticks([y_of[e] for e in order])
    ax.set_yticklabels([_emb_short(e) for e in order], fontsize=8.5, color=_INK)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == _emb_short(order[0]):
            lbl.set_fontweight("bold")
    ax.set_ylim(-0.6, n - 0.4)

    xvals = ([mean[e]["auroc_mean"] for e in order]
             + [mean[e]["tpr_at_fpr_mean"] for e in order])
    ax.set_xlim(*_axis_range(xvals))
    ax.set_xlabel("score", fontsize=9, color=_INK)
    ax.tick_params(labelsize=8, colors=_MUTED)

    fpr_pct = f"{meta['fpr'] * 100:g}"
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", ms=7, color=auroc_c, label="AUROC"),
        plt.Line2D([], [], marker="D", ls="", ms=6, color=tpr_c,
                   label=f"TPR @ {fpr_pct}% FPR (oracle)"),
    ], loc="upper left", frameon=False, fontsize=8)
    tr = int(round((1 - meta["test_size"]) * 100))
    ax.set_title(
        f"Embedding ranking — mean over {len(meta['splits'])} SCRIBE splits, "
        f"logistic regression\n{meta['n_seeds']} seeds, "
        f"{tr}/{100 - tr} grouped split",
        fontsize=10.5, color=_INK, loc="left",
    )
    fig.text(0.02, 0.008,
             "bars = spread across the SCRIBE splits (consistency, not seed noise)",
             fontsize=7, color=_MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def emb_by_split(summary_rows: list[dict], meta: dict, path: Path) -> None:
    """AUROC per embedding, a column per SCRIBE split plus a 'mean' column, in
    the mean ranking's row order — does the ranking hold on every benchmark.
    Error bars: seed SD in a split column, across-split SD in 'mean'."""
    from .splits import display as _split_display

    mean = _emb_rows(summary_rows, "mean")
    order = sorted(mean, key=lambda e: mean[e]["auroc_mean"], reverse=True)
    columns = list(meta["splits"]) + ["mean"]
    n = len(order)
    y_of = {e: n - 1 - i for i, e in enumerate(order)}
    by_key = {(r["embedding"], r["split"]): r for r in summary_rows}

    fig, axes = plt.subplots(
        1, len(columns), figsize=(2.3 * len(columns) + 1.8, 0.42 * n + 1.6),
        sharey=True, squeeze=False,
    )
    axes = list(axes[0])

    xvals = [by_key[(e, c)]["auroc_mean"]
             for e in order for c in columns if (e, c) in by_key]
    xlim = _axis_range(xvals)

    for ax, col in zip(axes, columns):
        ax.set_facecolor(_SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        is_mean = col == "mean"
        colour = _INK if is_mean else _PALETTE[0]
        for e in order:
            r = by_key.get((e, col))
            if r is None:
                continue
            ax.errorbar(r["auroc_mean"], y_of[e], xerr=r["auroc_std"],
                        fmt="D" if is_mean else "o", ms=6, color=colour,
                        ecolor=colour, elinewidth=1, capsize=2, zorder=3)
        ax.set_title(_split_display(col), fontsize=9, color=_INK,
                     fontweight="bold" if is_mean else "normal")
        ax.set_xlim(*xlim)
        ax.set_xlabel("AUROC", fontsize=8.5, color=_INK)
        ax.tick_params(labelsize=7.5, colors=_MUTED)

    axes[0].set_yticks([y_of[e] for e in order])
    axes[0].set_yticklabels([_emb_short(e) for e in order], fontsize=8, color=_INK)
    axes[0].set_ylim(-0.6, n - 0.4)

    fig.suptitle("Embedding AUROC by SCRIBE split (row order = mean ranking)",
                 fontsize=10.5, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Zero-shot detector performance (run_zeroshot_performance.py /
# run_best_zeroshot_performance.py). AUROC vs. oracle-TPR@1%FPR scatter:
# colour = detector (keyed by ``key`` -- 'version' within a family, 'detector'
# across families), marker shape = SCRIBE split (splits.MARKER), a bold filled
# circle = that detector's mean across splits. ``rows`` is the pre-filtered
# subset to draw; it includes the split=="mean" rows.
# ---------------------------------------------------------------------------


def _tint(hex_colour: str, t: float = 0.42) -> str:
    """A pale version of ``hex_colour`` for the small per-split points -- the
    shape already says which split; the colour only needs to whisper which
    detector, staying clear of that detector's solid mean circle."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"#{int(255 - (255 - r) * t):02x}{int(255 - (255 - g) * t):02x}" \
           f"{int(255 - (255 - b) * t):02x}"


def zeroshot_scatter(
    rows: list[dict], meta: dict, path: Path, *, key: str = "version", title: str = ""
) -> None:
    from .splits import display as _sd
    from .splits import marker as _split_marker

    names = sorted({r[key] for r in rows})
    splits = [s for s in meta["splits"] if s in {r["split"] for r in rows}]
    hue = {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(names)}
    marker = {s: _split_marker(s) for s in splits}
    fpr = meta["fpr"]

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Chance: an uninformative score is AUROC 0.5, and its oracle TPR at a
    # target FPR is that FPR -- a single point, not a diagonal (x is AUROC).
    ax.axvline(0.5, color="#b5495b", lw=1.0, ls=":", alpha=0.6, zorder=0)
    ax.axhline(fpr, color="#b5495b", lw=1.0, ls=":", alpha=0.6, zorder=0)
    ax.scatter([0.5], [fpr], s=120, color="#b5495b", marker="x", lw=1.8,
               alpha=0.6, zorder=0)

    for r in rows:
        if r["split"] == "mean" or not np.isfinite(r["tpr_at_fpr"]):
            continue
        ax.scatter([r["auroc"]], [r["tpr_at_fpr"]], s=62,
                   color=_tint(hue[r[key]]), edgecolors="#6a6a6a", lw=0.6,
                   marker=marker.get(r["split"], "P"), zorder=3)

    means = [next((r for r in rows if r[key] == v and r["split"] == "mean"), None)
             for v in names]
    x_hi = 1.02
    for v, m in zip(names, means):
        if m is None or not np.isfinite(m["tpr_at_fpr"]):
            continue
        ax.scatter([m["auroc"]], [m["tpr_at_fpr"]], s=165, color=hue[v],
                   edgecolors="#222222", lw=1.0, marker="o", zorder=5)
        # Label away from the right edge / the legend column: past 78% of the
        # x-range it goes to the marker's left instead.
        right = m["auroc"] > 0.45 + 0.78 * (x_hi - 0.45)
        ax.annotate(v, (m["auroc"], m["tpr_at_fpr"]), textcoords="offset points",
                    xytext=(-9 if right else 9, 4),
                    ha="right" if right else "left", fontsize=7.5,
                    color=_INK, zorder=6,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    aurocs = [r["auroc"] for r in rows if np.isfinite(r["auroc"])]
    x_lo = min(0.45, (min(aurocs) - 0.03) if aurocs else 0.45)
    ax.set_xlabel("AUROC (machine = positive)", fontsize=9, color=_INK)
    ax.set_ylabel(f"oracle TPR @ {fpr * 100:g}% FPR", fontsize=9, color=_INK)
    ax.set_xlim(x_lo, 1.02)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.tick_params(labelsize=8, colors=_MUTED)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=hue[v], label=v)
               for v in names]
    handles.append(plt.Line2D([], [], ls="", label=""))
    handles += [plt.Line2D([], [], marker=marker[s], ls="", ms=7,
                           color="#8a8a8a", label=_sd(s)) for s in splits]
    handles.append(plt.Line2D([], [], marker="x", ls="", color="#b5495b",
                              label=f"chance (0.5, {fpr * 100:g}%)"))
    ax.legend(handles=handles, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), framealpha=0.9, borderaxespad=0.0,
              title="● = mean across splits\nshape = split")

    ax.set_title(title or "zero-shot AUROC vs. oracle TPR@"
                 f"{fpr * 100:g}%FPR across the SCRIBE splits",
                 fontsize=10.5, color=_INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Judge x generator matrix (run_judge_by_generator.py). One heatmap panel per
# SCRIBE split, judges down the rows, generators across the columns; the cell
# is the oracle AUROC or TPR@1%FPR for human-vs-that-one-generator. '*' marks a
# judge scoring its own model family. Reads the long matrix CSV's records.
# ---------------------------------------------------------------------------

_METRIC_META = {
    "auroc": ("AUROC (machine = positive)", "coolwarm"),
    "tpr_at_fpr": ("oracle TPR @ {pct}% FPR", "viridis"),
}


def _cell_text_colour(metric: str, v: float) -> str:
    if not np.isfinite(v):
        return _MUTED
    mid = (metric == "auroc" and 0.28 < v < 0.72) or (metric != "auroc" and v < 0.55)
    return "white" if mid else "#111111"


def judge_generator_heatmaps(
    rows: list[dict], metric: str, meta: dict, path: Path
) -> None:
    """One heatmap panel per split + an 'overall' panel. Judge rows are ordered
    by each judge's grand mean of ``metric`` over every cell (all splits, all
    generators); a generator column is ordered, within its panel, by that
    generator's mean over the judges. Each panel carries a bottom 'mean' row
    (the generator means); the 'overall' panel's bottom cell is the grand
    mean."""
    from .splits import display as _sd

    label_tmpl, cmap = _METRIC_META[metric]
    label = label_tmpl.format(pct=f"{meta['fpr'] * 100:g}")

    split_keys = [s for s in meta["splits"] if s in {r["split"] for r in rows}]
    by = {(r["split"], r["judge"], r["gen_model"]): r for r in rows}
    all_judges = {r["judge"] for r in rows}

    def _mean(vals: list[float]) -> float:
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    overall = {j: _mean([r[metric] for r in rows if r["judge"] == j])
               for j in all_judges}
    judges = sorted(all_judges, key=lambda j: (np.isnan(overall[j]), -overall[j]))

    gens_of, gmean_of = {}, {}
    for s in split_keys:
        gens = {r["gen_model"] for r in rows if r["split"] == s}
        gmean_of[s] = {
            g: _mean([r[metric] for r in rows
                      if r["split"] == s and r["gen_model"] == g])
            for g in gens
        }
        gens_of[s] = sorted(gens, key=lambda g: (np.isnan(gmean_of[s][g]),
                                                 -gmean_of[s][g]))

    n_rows = len(judges) + 1  # + the 'mean' row
    ylabels = judges + ["mean"]
    widths = [max(len(gens_of[s]), 1) for s in split_keys] + [1]
    fig, axes = plt.subplots(
        1, len(split_keys) + 1, squeeze=False, layout="constrained",
        figsize=(1.15 * sum(widths) + 3.4, 0.5 * n_rows + 2.2),
        gridspec_kw={"width_ratios": widths},
    )
    axes = list(axes[0])

    def _draw(ax, col_labels, matrix, stars, first):
        im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(ylabels if first else [], fontsize=8)
        if first:
            ax.get_yticklabels()[0].set_fontweight("bold")   # top-ranked judge
        for i in range(n_rows):
            for k in range(len(col_labels)):
                v = matrix[i][k]
                txt = "–" if not np.isfinite(v) else f"{v:.2f}" + (
                    "*" if stars[i][k] else "")
                ax.text(k, i, txt, ha="center", va="center", fontsize=8,
                        color=_cell_text_colour(metric, v))
        ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="minor", length=0)
        ax.axhline(len(judges) - 0.5, color="#222222", lw=1.4)
        return im

    im = None
    for ax, s in zip(axes, split_keys):
        gens = gens_of[s]
        matrix = [[by.get((s, j, g), {}).get(metric, np.nan) for g in gens]
                  for j in judges]
        stars = [[bool(by.get((s, j, g), {}).get("self_family")) for g in gens]
                 for j in judges]
        matrix.append([gmean_of[s][g] for g in gens])
        stars.append([False] * len(gens))
        im = _draw(ax, gens, matrix, stars, first=ax is axes[0])
        ax.set_title(_sd(s), fontsize=9.5, color=_INK)

    ovr_col = [[overall[j]] for j in judges] + [[_mean(list(overall.values()))]]
    _draw(axes[-1], ["overall"], ovr_col, [[False]] * n_rows, first=False)
    axes[-1].set_title("overall", fontsize=9.5, color=_INK)

    fig.colorbar(im, ax=axes, shrink=0.85, aspect=30)
    fig.suptitle(
        f"{label}  —  judge × generator, oracle\n"
        f"rows ranked by the judge's overall mean · "
        f"* = judge and generator are the same model family",
        fontsize=10.5, color=_INK,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


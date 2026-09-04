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
# Embedding ranking (run_embedding_effectiveness.py) — a Cleveland dot plot,
# not a scatter: with a dozen embeddings, a 2-D scatter piles labels on top of
# each other right where the answer lives (in-domain, near the ceiling). A
# ranked list scales to as many rows as there are embeddings and states the
# answer in the row order alone.
# ---------------------------------------------------------------------------


def embedding_ranking_plot(ranking: list[dict], meta: dict, path: Path) -> None:
    """The answer figure: embeddings ranked by combined AUROC, both metrics shown.

    ``ranking`` is :func:`run_embedding_effectiveness.combine_ranking`'s output
    — already sorted best AUROC first. One row per embedding; AUROC and
    TPR@FPR are two series on the same 0-1 rate axis (one dot each, joined by a
    thin connector), not a dual axis.
    """
    n = len(ranking)
    y_of = {r["embedding"]: n - 1 - i for i, r in enumerate(ranking)}  # best on top
    auroc_c, tpr_c = _PALETTE[0], _PALETTE[1]

    fig, ax = plt.subplots(figsize=(6.6, 0.5 * n + 1.6))
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for r in ranking:
        y = y_of[r["embedding"]]
        ax.plot(
            [r["auroc_mean"], r["tpr_at_fpr_mean"]], [y + 0.16, y - 0.16],
            color=_MUTED, lw=0.8, zorder=1,
        )
        ax.errorbar(
            r["auroc_mean"], y + 0.16, xerr=r["auroc_spread"],
            fmt="o", ms=7, color=auroc_c, ecolor=auroc_c,
            elinewidth=1, capsize=2, zorder=3,
        )
        ax.errorbar(
            r["tpr_at_fpr_mean"], y - 0.16, xerr=r["tpr_at_fpr_spread"],
            fmt="D", ms=6, color=tpr_c, ecolor=tpr_c,
            elinewidth=1, capsize=2, zorder=3,
        )

    ax.set_yticks([y_of[r["embedding"]] for r in ranking])
    ax.set_yticklabels([_emb_short(r["embedding"]) for r in ranking], fontsize=8.5, color=_INK)
    winner = _emb_short(ranking[0]["embedding"])
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == winner:
            lbl.set_fontweight("bold")
    ax.set_ylim(-0.6, n - 0.4)

    xvals = [r["auroc_mean"] for r in ranking] + [r["tpr_at_fpr_mean"] for r in ranking]
    ax.set_xlim(*_axis_range(xvals))
    ax.set_xlabel("score", fontsize=9, color=_INK)
    ax.tick_params(labelsize=8, colors=_MUTED)

    fpr_pct = f"{meta['fpr'] * 100:g}"
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=7, color=auroc_c, label="AUROC"),
        plt.Line2D([], [], marker="D", ls="", ms=6, color=tpr_c,
                   label=f"TPR @ {fpr_pct}% FPR (oracle)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)

    n_gen = len(meta["generators"])
    n_dom = len(meta["ranking_domains"])
    ax.set_title(
        f"{meta['experiment']} — embedding ranking, {meta['algorithm']}\n"
        f"combined over {n_gen} generator(s) x {n_dom} domain(s)",
        fontsize=10.5, color=_INK, loc="left",
    )
    fig.text(
        0.02, 0.008,
        "bars = spread across generators/domains (consistency, not seed noise) "
        "· pooled excluded from the ranking",
        fontsize=7, color=_MUTED,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def by_generator_plot(
    summary_rows: list[dict], ranking: list[dict], meta: dict, path: Path
) -> None:
    """AUROC (pooled domain) generator by generator, in the combined ranking's row
    order — does the ranking hold up separately, or is it an artifact of averaging.
    """
    order = [r["embedding"] for r in ranking]
    generators = meta["generators"]
    n = len(order)
    y_of = {emb: n - 1 - i for i, emb in enumerate(order)}
    by_key = {(r["embedding"], r["generator"], r["domain"]): r for r in summary_rows}

    fig, axes = plt.subplots(
        1, len(generators), figsize=(2.5 * len(generators) + 1.8, 0.42 * n + 1.6),
        sharey=True,
    )
    axes = [axes] if len(generators) == 1 else list(axes)

    xvals = [
        by_key[(emb, g, POOLED)]["auroc_mean"]
        for emb in order for g in generators
        if (emb, g, POOLED) in by_key
    ]
    xlim = _axis_range(xvals)

    for j, g in enumerate(generators):
        ax = axes[j]
        ax.set_facecolor(_SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for emb in order:
            r = by_key.get((emb, g, POOLED))
            if r is None:
                continue
            ax.errorbar(
                r["auroc_mean"], y_of[emb], xerr=r["auroc_std"],
                fmt="o", ms=6, color=_PALETTE[0], ecolor=_PALETTE[0],
                elinewidth=1, capsize=2, zorder=3,
            )
        ax.set_title(_short(g), fontsize=9, color=_INK)
        ax.set_xlim(*xlim)
        ax.set_xlabel("AUROC", fontsize=8.5, color=_INK)
        ax.tick_params(labelsize=7.5, colors=_MUTED)

    axes[0].set_yticks([y_of[e] for e in order])
    axes[0].set_yticklabels([_emb_short(e) for e in order], fontsize=8, color=_INK)
    axes[0].set_ylim(-0.6, n - 0.4)

    fig.suptitle(
        f"{meta['experiment']} — AUROC by generator, pooled domain "
        "(same row order as the combined ranking)",
        fontsize=10.5, color=_INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)

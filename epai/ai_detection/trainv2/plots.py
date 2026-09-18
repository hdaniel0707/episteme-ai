"""The trainv2 figure family: AUROC vs. TPR@1%FPR scatters and a couple of
Cleveland dot plots, all reading a run's CSV records (dicts).

Deliberately plain: no gridlines behind the marks, no dual axes, fixed
categorical colours in slot order (the data-viz default palette — its first
three slots clear the all-pairs colour-vision gate).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import goodness as _goodness
from .detector_names import display_detector as _display_detector
from .detector_names import is_judge_family

_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7",
            "#008300", "#e34948"]
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_MUTED = "#8a8a86"


def _emb_short(column: str) -> str:
    """``emb_luar_crud`` -> ``luar_crud``."""
    return column[4:] if column.startswith("emb_") else column


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



# ---------------------------------------------------------------------------
# Embedding effectiveness across the MANUGRAM splits
# (run_embedding_effectiveness.py). Two Cleveland dot plots, not scatters:
# with a dozen embeddings a 2-D scatter piles labels on top of each other
# right where the answer lives (near the ceiling). A ranked list scales to as
# many rows as there are embeddings and states the answer in the row order.
# Both read the summary CSV's records (dicts), incl. the split=="mean" rows.
# ---------------------------------------------------------------------------


def _emb_rows(summary_rows: list[dict], split: str) -> dict[str, dict]:
    return {r["embedding"]: r for r in summary_rows if r["split"] == split}


def emb_mean_ranking(summary_rows: list[dict], meta: dict, path: Path) -> None:
    """Embeddings ranked by mean AUROC across the MANUGRAM splits, AUROC and
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
        f"Embedding ranking — mean over {len(meta['splits'])} MANUGRAM splits, "
        f"logistic regression\n{meta['n_seeds']} seeds, "
        f"{tr}/{100 - tr} grouped split",
        fontsize=10.5, color=_INK, loc="left",
    )
    fig.text(0.02, 0.008,
             "bars = spread across the MANUGRAM splits (consistency, not seed noise)",
             fontsize=7, color=_MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def emb_by_split(summary_rows: list[dict], meta: dict, path: Path) -> None:
    """AUROC per embedding, a column per MANUGRAM split plus a 'mean' column, in
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

    fig.suptitle("Embedding AUROC by MANUGRAM split (row order = mean ranking)",
                 fontsize=10.5, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Zero-shot detector performance (run_zeroshot_performance.py /
# run_best_zeroshot_performance.py). AUROC vs. oracle-TPR@1%FPR scatter:
# colour = detector (keyed by ``key`` -- 'version' within a family, 'detector'
# across families), marker shape = MANUGRAM split (splits.MARKER), a bold
# marker = that detector's mean across splits (filled = a ``_wlp``-style
# variant, hollow/same hue = its base). A grey Greys contourf/contour field
# behind everything shows the AT-HM region each point falls in. Both axes use
# an anchored power-law warp (:func:`_distort`) that compresses the mostly-
# empty low end and stretches out the crowded near-1.0 cluster. ``rows`` is
# the pre-filtered subset to draw; it includes the split=="mean" rows.
# ---------------------------------------------------------------------------


def _tint(hex_colour: str, t: float = 0.42) -> str:
    """A pale version of ``hex_colour`` for the small per-split points -- the
    shape already says which split; the colour only needs to whisper which
    detector, staying clear of that detector's solid mean circle."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"#{int(255 - (255 - r) * t):02x}{int(255 - (255 - g) * t):02x}" \
           f"{int(255 - (255 - b) * t):02x}"


def _shade(hex_colour: str, f: float = 0.72) -> str:
    """A darker version of ``hex_colour`` (channels x ``f``, ``f`` < 1) -- a
    base judge that has a ``_wlp`` twin draws in this darker shade; the ``_wlp``
    keeps the plain hue."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"#{int(r * f):02x}{int(g * f):02x}{int(b * f):02x}"


def _f(v):
    """``float(v)`` if finite, else ``None`` -- for reading maybe-blank CSV cells."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _goodness_field(auroc, tpr, alpha: float = 0.01, how: str = "hmean"):
    """Vectorised twin of ``core.goodness`` for a background grid (arrays, not
    a single point) -- same G/T rescaling, same combine rule."""
    g = np.clip(2.0 * auroc - 1.0, 0.0, None)
    t = np.clip((tpr - alpha) / (1.0 - alpha), 0.0, None)
    if how == "amean":
        return (g + t) / 2.0
    if how == "gmean":
        return np.sqrt(g * t)
    denom = g + t
    with np.errstate(invalid="ignore", divide="ignore"):
        hm = 2.0 * g * t / denom
    return np.where(denom > 0, hm, 0.0)


def _variant_palette(names, suffix: str = "_wlp"):
    """Colour rule shared by :func:`denario_scatter` and :func:`goodness_bars`:
    a detector ``<x><suffix>`` is a readout variant of ``<x>`` -- the pair
    shares a hue, and the *base* draws one shade darker so the variant reads
    as the lighter twin. Returns ``(base_of, dcol)`` where ``base_of[variant]
    = base`` and ``dcol(name)`` is that name's display colour. (The main
    AUROC-vs-TPR scatter, :func:`zeroshot_scatter`, uses the hollow-base
    scheme below instead.)"""
    base_of = {n: n[: -len(suffix)] for n in names
               if n.endswith(suffix) and n[: -len(suffix)] in names}
    paired_base = set(base_of.values())
    hue: dict[str, str] = {}
    for v in names:  # non-variants take palette slots in order
        if v not in base_of:
            hue[v] = _PALETTE[len(hue) % len(_PALETTE)]
    for v, b in base_of.items():  # variants inherit their base's colour
        hue[v] = hue[b]

    def dcol(v: str) -> str:
        return _shade(hue[v]) if v in paired_base else hue[v]

    return base_of, dcol


def _variant_palette_hollow(names, suffix: str = "_wlp"):
    """Colour rule for :func:`zeroshot_scatter`: like :func:`_variant_palette`,
    a detector ``<x><suffix>`` shares a hue with its base ``<x>``, but the
    base keeps that *same* hue instead of a darker shade and is drawn hollow
    (unfilled) instead -- "same colour, filled vs. outline" reads more
    clearly than "two adjacent but different colours" once there are several
    such pairs on one plot. Returns ``(base_of, paired_base, hue)``."""
    base_of = {n: n[: -len(suffix)] for n in names
               if n.endswith(suffix) and n[: -len(suffix)] in names}
    paired_base = set(base_of.values())
    hue: dict[str, str] = {}
    for v in names:
        if v not in base_of:
            hue[v] = _PALETTE[len(hue) % len(_PALETTE)]
    for v, b in base_of.items():
        hue[v] = hue[b]
    return base_of, paired_base, hue


def _distort(v, lo: float, hi: float, k: float):
    """Anchored power-law warp on ``[lo, hi]``: endpoints stay fixed, ``k >
    1`` compresses values near ``lo`` and stretches values near ``hi`` (the
    derivative shrinks to 0 near ``lo`` and grows to ``k`` near ``hi``).
    Works on a scalar or an array (used for both point coordinates and the
    AT-HM background grid)."""
    u = np.clip((np.asarray(v, dtype=float) - lo) / (hi - lo), 0.0, None)
    return lo + u ** k * (hi - lo)


def zeroshot_scatter(
    rows: list[dict], meta: dict, path: Path | list[Path], *, key: str = "version",
    title: str = "", variant_suffix: str = "_wlp", figsize: tuple[float, float] = (9.6, 4.6),
    kx: float = 1.7, ky: float = 1.7,
    color_override: dict[str, str] | None = None,
    light_border: frozenset[str] = frozenset(),
    emphasize: frozenset[str] = frozenset(),
    chance_color: str = "#b5495b",
    label_override: dict[str, str] | None = None,
    show_scores: bool = True,
) -> None:
    """Both axes use a nonlinear "expand near 1" warp (:func:`_distort`,
    anchored power law) instead of a linear scale: it compresses the mostly-
    empty region near the low end and stretches out the crowded top-right
    cluster. The AT-HM background field is warped the same way so its
    contours still line up with the (distorted) points. Tick positions are
    chosen by hand and placed at their warped location, since evenly-spaced
    real values no longer land evenly spaced on the page.

    ``color_override`` replaces the auto-assigned hue for the given raw
    ``key`` values (any name absent from it keeps its automatic palette
    slot) -- for a figure that needs specific series to carry a fixed,
    semantic colour rather than whatever slot they land on. Every marker is
    the same size and gets SOME border; ``light_border`` only thins that
    (non-hollow) name's edge instead of the default bold one -- both stay
    black, so a de-emphasised series still reads as "the same kind of mark,
    less shouty" rather than a different, flatter kind of mark. ``emphasize``
    draws those names' markers one zorder step above everything else, so
    they sit visibly on top of any overlap at the same point. The intended
    combination is baselines with a thin black border in their own
    full-saturation hues (``light_border``, left out of ``emphasize``)
    against a bold-bordered, accent-coloured "ours" drawn on top
    (``emphasize``, left out of ``light_border``). ``chance_color`` recolours
    the chance cross/dotted reference lines. ``label_override`` replaces a
    name's whole legend string outright (e.g. dropping the backbone suffix
    down to a short official name, or adding a "(ours)" tag); ``show_scores``
    (default on) toggles the auto "— AT-HM x.xx" suffix for every name NOT
    in ``label_override``. All six default to the original look (bold
    border, no z-order boost, maroon chance marker, full ``family/backbone``
    names with their AT-HM score), so every other caller of this function is
    unaffected."""
    from .splits import PLOT_EXCLUDE
    from .splits import display as _sd
    from .splits import marker as _split_marker

    excluded = PLOT_EXCLUDE.get("scatter", frozenset())
    rows = [r for r in rows if r["split"] == "mean" or r["split"] not in excluded]

    names = sorted({r[key] for r in rows})
    splits = [s for s in meta["splits"] if s in {r["split"] for r in rows}]

    # A ``variant_suffix`` detector is a readout variant of its base (default
    # ``_wlp``, a judge's logprob-window score; ``run_dr_transfer.py`` passes
    # ``_dr`` for its domain-randomized twin): shared hue, the two means
    # joined by a connector. The variant draws as a filled circle; its base
    # draws hollow in the same hue (see :func:`_variant_palette_hollow`).
    base_of, paired_base, hue = _variant_palette_hollow(names, suffix=variant_suffix)
    if color_override:
        hue.update({k: v for k, v in color_override.items() if k in hue})

    marker = {s: _split_marker(s) for s in splits}
    fpr = meta["fpr"]

    # A bigger low-end pad than a linear scale would need -- the warp is
    # anchored at (x_lo, y_lo) and compresses hardest right there, so a small
    # pad gets squeezed down to almost nothing, leaving no room for a
    # marker's own radius and clipping it against the axis edge.
    aurocs = [r["auroc"] for r in rows if np.isfinite(r["auroc"])]
    x_lo = min(0.45, (min(aurocs) - 0.12) if aurocs else 0.45)
    x_hi, y_lo, y_hi = 1.02, -0.25, 1.04

    def wx(v):
        return _distort(v, x_lo, x_hi, kx)

    def wy(v):
        return _distort(v, y_lo, y_hi, ky)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Chance: an uninformative score is AUROC 0.5, and its oracle TPR at a
    # target FPR is that FPR -- a single point, not a diagonal (x is AUROC).
    ax.axvline(wx(0.5), color=chance_color, lw=1.0, ls=":", alpha=0.6, zorder=0)
    ax.axhline(wy(fpr), color=chance_color, lw=1.0, ls=":", alpha=0.6, zorder=0)
    ax.scatter([wx(0.5)], [wy(fpr)], s=120, color=chance_color, marker="x", lw=1.8,
               alpha=0.6, zorder=0)

    def _err(r: dict) -> tuple[float | None, float | None]:
        """SD, when the CSV carries a finite one (trained rows). Only ever
        drawn on the mean marker -- the light per-split points stay bare."""
        xe, ye = r.get("auroc_std"), r.get("tpr_at_fpr_std")
        try:
            xe, ye = float(xe), float(ye)
        except (TypeError, ValueError):
            return None, None
        return (xe if np.isfinite(xe) else None, ye if np.isfinite(ye) else None)

    # Light per-split points: bare markers, no error bars, no labels -- the
    # legend carries every name. A paired base draws hollow, same hue as its
    # variant, at the same size/edge-weight as a filled point.
    for r in rows:
        if r["split"] == "mean" or not np.isfinite(r["tpr_at_fpr"]):
            continue
        hollow = r[key] in paired_base
        light = r[key] in light_border and not hollow
        big = r[key] in emphasize
        col = hue[r[key]]
        if hollow:
            edge, lw = _tint(col, 0.55), 0.6
        elif light:
            edge, lw = _INK, 0.5
        else:
            edge, lw = _INK, 1.0
        ax.scatter([wx(r["auroc"])], [wy(r["tpr_at_fpr"])], s=48,
                   marker=marker.get(r["split"], "P"),
                   facecolors="none" if hollow else _tint(col),
                   edgecolors=edge, linewidths=lw, zorder=3.5 if big else 3)

    means = [(v, next((r for r in rows if r[key] == v and r["split"] == "mean"), None))
             for v in names]
    means = [(v, m) for v, m in means if m is not None and np.isfinite(m["tpr_at_fpr"])]
    mean_xy = {v: (m["auroc"], m["tpr_at_fpr"]) for v, m in means}

    # base -> wlp connectors, drawn behind the mean markers
    for v, b in base_of.items():
        if v in mean_xy and b in mean_xy:
            (xv, yv), (xb, yb) = mean_xy[v], mean_xy[b]
            ax.plot([wx(xb), wx(xv)], [wy(yb), wy(yv)], color=hue[b], lw=1.1,
                    alpha=0.6, zorder=4)

    for v, m in means:
        xe, ye = _err(m)
        hollow = v in paired_base
        light = v in light_border and not hollow
        big = v in emphasize
        x0, y0 = m["auroc"], m["tpr_at_fpr"]
        if hollow:
            mec, mew = hue[v], 1.2
        elif light:
            mec, mew = _INK, 0.7
        else:
            mec, mew = _INK, 1.3
        # Asymmetric error bars: (mean - err) and (mean + err) each warp by a
        # different amount since the transform isn't linear.
        xerr = [[wx(x0) - wx(x0 - xe)], [wx(x0 + xe) - wx(x0)]] if xe is not None else None
        yerr = [[wy(y0) - wy(y0 - ye)], [wy(y0 + ye) - wy(y0)]] if ye is not None else None
        ax.errorbar([wx(x0)], [wy(y0)], xerr=xerr, yerr=yerr,
                    fmt="o", ms=9, mfc="none" if hollow else hue[v], mec=mec, mew=mew,
                    ecolor="#222222", elinewidth=1.0, capsize=2.0,
                    zorder=5.5 if big else 5)

    ax.set_xlabel(f"AUROC\n(power warp, k={kx:g})", fontsize=9, color=_INK)
    ax.set_ylabel(f"TPR@{fpr * 100:g}%FPR\n(power warp, k={ky:g})", fontsize=9,
                  color=_INK)
    ax.set_xlim(wx(x_lo), wx(x_hi))
    ax.set_ylim(wy(y_lo), wy(y_hi))

    # AT-HM background field: shows which region of (AUROC, TPR) space each
    # point's combined score falls in, so the trade-off reads directly off the
    # plot rather than only through the separate goodness bar chart. Drawn
    # deep behind everything else (points, chance cross, grid). Computed on a
    # real-value grid capped at 1.0 (not the axis's padded x_hi/y_hi -- AUROC
    # and TPR can't exceed 1), then warped to display coordinates so the
    # shading lines up with the (distorted) points.
    gx = np.linspace(x_lo, 1.0, 200)
    gy = np.linspace(-0.04, 1.0, 200)
    GX, GY = np.meshgrid(gx, gy)
    GZ = _goodness_field(GX, GY)
    ax.contourf(wx(GX), wy(GY), GZ, levels=np.linspace(0, 1, 11), cmap="Greys",
                alpha=0.30, zorder=-5)
    cs = ax.contour(wx(GX), wy(GY), GZ, levels=[0.2, 0.4, 0.6, 0.8],
                     colors="#8a8a8a", linewidths=0.6, alpha=0.8, zorder=-4)
    ax.clabel(cs, fmt=lambda v: f"{v:.1f}", fontsize=6.5, colors="#6a6a6a",
              inline=True)

    ax.grid(True, lw=0.4, alpha=0.4)
    ax.tick_params(labelsize=8, colors=_MUTED)

    # Hand-picked tick values, placed at their warped position -- evenly
    # spaced real values no longer land evenly spaced once warped, and the
    # extra room near 1.0 is exactly where finer ticks earn their keep.
    x_ticks = [t for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0) if t >= x_lo]
    ax.set_xticks([wx(t) for t in x_ticks])
    ax.set_xticklabels([f"{t:g}" for t in x_ticks])
    y_ticks = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0)
    ax.set_yticks([wy(t) for t in y_ticks])
    ax.set_yticklabels([f"{t:g}" for t in y_ticks])

    # Colour key: sorted by mean AT-HM (descending) with the value printed, so
    # the legend order matches the goodness bar chart's ranking; a base and
    # its _wlp variant no longer sit next to each other, but the combined
    # score is now the thing being read off, not the pairing.
    def _at_hm(v: str) -> float | None:
        xy = mean_xy.get(v)
        return _goodness(*xy, "hmean") if xy is not None else None

    ranked = sorted(names, key=lambda v: (_at_hm(v) is None, -(_at_hm(v) or 0.0)))
    handles = []
    for v in ranked:
        hollow = v in paired_base
        if label_override and v in label_override:
            label = label_override[v]
        else:
            s = _at_hm(v)
            dv = _display_detector(v)
            label = f"{dv} — AT-HM {s:.2f}" if (show_scores and s is not None) else dv
        handles.append(plt.Line2D(
            [], [], marker="o", ls="", ms=7,
            mfc="none" if hollow else hue[v],
            mec=hue[v] if hollow else "#222222", mew=1.0, label=label))
    handles.append(plt.Line2D([], [], ls="", label=""))
    handles += [plt.Line2D([], [], marker=marker[s], ls="", ms=7,
                           color="#8a8a8a", label=_sd(s)) for s in splits]
    handles.append(plt.Line2D([], [], marker="x", ls="", color=chance_color,
                              label=f"chance (0.5, {fpr * 100:g}%)"))
    legend = ax.legend(
        handles=handles, fontsize=8, loc="upper left",
        bbox_to_anchor=(1.01, 0.95), framealpha=0.9, borderaxespad=0.0,
    )
    # Our two contributions (LogJudge, FusionJudge) in bold, same rule as
    # build_general_results_table.py's tables -- zip stops at len(ranked), so
    # this never touches the split-shape/chance entries below.
    for v, txt in zip(ranked, legend.get_texts()):
        if is_judge_family(v):
            txt.set_fontweight("bold")

    if title:
        ax.set_title(title, fontsize=10.5, color=_INK)
    for p in [path] if isinstance(path, Path) else path:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, facecolor="white", bbox_inches="tight",
                    bbox_extra_artists=(legend,))
        print(f"wrote {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# The eval-only OOD corpus on its own (run_in_domain_performance.py /
# run_lodo_performance.py, alongside their main scatter). A dedicated zoom on
# MG-AI-Sci -- the one point the main scatters bury against the ceiling: every
# trained detector (marker carries the SD across the source models) and every
# zero-shot pick (a filled circle with no error bar), one point each. Reads the
# same CSV records; keeps the full 0-1 TPR axis so the drop from the in-domain
# ceiling stays visible.
# ---------------------------------------------------------------------------


def denario_scatter(
    rows: list[dict], meta: dict, path: Path, *, title: str = "",
    variant_suffix: str = "_wlp",
) -> None:
    """AUROC vs. oracle TPR@FPR on the ``meta["ood"]`` corpus alone (MG-AI-Sci by
    default). Built for the single-corpus default; with several OOD corpora each
    detector gets one point per corpus, colour-coded by detector only."""
    fpr = meta["fpr"]
    ood_keys = set(meta.get("ood") or ["mg_aisci"])

    def _f(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    pts = [r for r in rows
           if r.get("split") in ood_keys
           and _f(r.get("auroc")) is not None
           and _f(r.get("tpr_at_fpr")) is not None]
    trained = [r for r in pts if r.get("kind") == "trained"]
    zshot = [r for r in pts if r.get("kind") == "zeroshot"]
    if not trained and not zshot:
        print(f"denario_scatter: no OOD rows for {sorted(ood_keys)} — skipped")
        return

    # Same colour rule as the main scatter (over the identical row set), incl.
    # the base -> variant pairing.
    names = sorted({r["detector"] for r in rows})
    base_of, dcol = _variant_palette(names, suffix=variant_suffix)

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Chance: AUROC 0.5, oracle TPR at the target FPR is that FPR -- one point.
    ax.axvline(0.5, color="#b5495b", lw=1.0, ls=":", alpha=0.6, zorder=0)
    ax.axhline(fpr, color="#b5495b", lw=1.0, ls=":", alpha=0.6, zorder=0)
    ax.scatter([0.5], [fpr], s=120, color="#b5495b", marker="x", lw=1.8,
               alpha=0.6, zorder=0)

    aurocs = [_f(r["auroc"]) for r in trained + zshot]
    x_lo, x_hi = max(0.0, min(aurocs) - 0.05), 1.01

    # No in-plot point labels: on MG-AI-Sci every detector piles into the same
    # corner, so a name next to each mark is an unreadable stack. The legend
    # carries the colour -> detector key instead.
    xy_of = {r["detector"]: (_f(r["auroc"]), _f(r["tpr_at_fpr"])) for r in pts}
    for v, b in base_of.items():  # base -> _wlp connector
        if v in xy_of and b in xy_of and None not in (*xy_of[v], *xy_of[b]):
            (xv, yv), (xb, yb) = xy_of[v], xy_of[b]
            ax.plot([xb, xv], [yb, yv], color=dcol(b), lw=1.1, alpha=0.55, zorder=2)

    for r in zshot:  # filled circle, no error bar -- a reference point
        ax.scatter([_f(r["auroc"])], [_f(r["tpr_at_fpr"])], s=38, marker="o",
                   color=dcol(r["detector"]), edgecolors="#222222",
                   linewidths=1.0, zorder=4)

    for r in trained:
        ax.errorbar([_f(r["auroc"])], [_f(r["tpr_at_fpr"])],
                    xerr=_f(r.get("auroc_std")), yerr=_f(r.get("tpr_at_fpr_std")),
                    fmt="o", ms=7, mfc=dcol(r["detector"]), mec="#222222",
                    mew=1.0, ecolor="#222222", elinewidth=1.0, capsize=2.0,
                    zorder=5)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.tick_params(labelsize=8, colors=_MUTED)
    ax.set_xlabel("AUROC (machine = positive)", fontsize=9, color=_INK)
    ax.set_ylabel(f"oracle TPR @ {fpr * 100:g}% FPR", fontsize=9, color=_INK)

    # One coloured entry per detector, then the trained / zero-shot key (both
    # filled circles; trained carries the SD error bar) + chance.
    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, color=dcol(v), label=v)
               for v in names]
    handles.append(plt.Line2D([], [], ls="", label=""))
    handles += [
        plt.Line2D([], [], marker="o", ls="", ms=8, mfc="#8a8a8a",
                   mec="#222222", label="trained  (bars = SD across source models)"),
        plt.Line2D([], [], marker="o", ls="", ms=8, mfc="#8a8a8a",
                   mec="#222222", label="zero-shot  (no error bar)"),
        plt.Line2D([], [], marker="x", ls="", color="#b5495b",
                   label=f"chance  (0.5, {fpr * 100:g}%)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), framealpha=0.9, borderaxespad=0.0,
              title="colour = detector")

    ood = "/".join(sorted(ood_keys))
    ax.set_title(title or f"{ood} (eval-only OOD) — AUROC vs. oracle "
                 f"TPR@{fpr * 100:g}%FPR", fontsize=10.5, color=_INK)
    fig.text(0.02, 0.008,
             f"{ood}: never trained on. Few human rows -> the {fpr * 100:g}%-FPR "
             f"grid point floors at FPR=0, so TPR here is a lower bound, not an "
             f"estimate.", fontsize=7, color=_MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Goodness bars (run_zeroshot_performance / run_best_zeroshot_performance /
# run_in_domain_performance / run_lodo_performance). One grouped bar chart:
# x = dataset, a bar per detector, height = a single "goodness" score that
# folds ranking (Gini = 2*AUROC - 1) and the operating point (oracle
# TPR@1%FPR, or at the smallest reachable FPR on a split too small for 1%)
# into one number. Default combine is the harmonic mean -- weakest-link, like
# F1: a detector useless on either axis scores ~0. Reads the run's CSV
# records; uses `tpr_eff` / `eff_fpr` so a split too small to reach the target
# FPR exactly (MG-AI-Sci, 100 humans -- 1% sits right at the n_human=100 floor
# of core.eff_fpr_level) still gets a real bar at its own reachable FPR,
# annotated.
# ---------------------------------------------------------------------------

_COMBINE_LABEL = {"hmean": "harmonic mean", "gmean": "geometric mean",
                  "amean": "arithmetic mean"}
_COMBINE_EXPR = {"hmean": "HM", "gmean": "GM", "amean": "mean"}


def goodness_bars(
    rows: list[dict], meta: dict, path: Path, *, key: str = "version",
    title: str = "", combine: str = "hmean", variant_suffix: str = "_wlp",
) -> None:
    from .splits import PLOT_EXCLUDE
    from .splits import display as _sd

    excluded = PLOT_EXCLUDE.get("goodness", frozenset())
    rows = [r for r in rows if r.get("kind") != "trained"
            or r.get("split") is not None] if rows else []
    rows = [r for r in rows if r.get("split") == "mean" or r.get("split") not in excluded]
    # dataset groups: the per-split rows in the run's order, then the pooled
    # 'mean' (shown last).
    present = {r["split"] for r in rows}
    # dict.fromkeys dedupes while keeping order -- meta["splits"] may already
    # carry the OOD split (in_domain / lodo concatenate it in).
    order = list(dict.fromkeys(
        list(meta.get("splits", [])) + list(meta.get("ood", []))))
    groups = [s for s in order if s in present]
    for s in sorted(present):  # any split not named in meta, e.g. mg_aisci
        if s not in groups and s != "mean":
            groups.append(s)
    if "mean" in present:
        groups.append("mean")
    if not groups:
        print("goodness_bars: no rows — skipped")
        return

    names = sorted({r[key] for r in rows})
    base_of, dcol = _variant_palette(names, suffix=variant_suffix)

    def _cell(det: str, split: str) -> dict | None:
        return next((r for r in rows if r[key] == det and r["split"] == split), None)

    def _tpr(c: dict) -> float:  # eff-FPR TPR, or the plain 1%-FPR one (old CSV)
        t = _f(c.get("tpr_eff"))
        return (t if t is not None else _f(c.get("tpr_at_fpr")) or 0.0)

    def _score(c: dict) -> float | None:
        # the runner stores a per-seed / per-fold goodness mean when it has
        # replicates (in_domain / lodo, trained AND zero-shot); fall back to
        # recomputing from AUROC + eff-FPR TPR for the CSVs that don't.
        stored = _f(c.get("goodness"))
        if stored is not None:
            return stored
        return None if _f(c.get("auroc")) is None else _goodness(
            c["auroc"], _tpr(c), combine)

    def _score_std(c: dict) -> float | None:
        sd = _f(c.get("goodness_std"))
        return sd if sd is not None and sd > 0 else None

    # rank detectors by their pooled-'mean' score, else the mean over groups
    def _rank(det: str) -> float:
        m = _cell(det, "mean")
        if m is not None and _score(m) is not None:
            return _score(m)
        vals = [s for g in groups if (c := _cell(det, g)) is not None
                and (s := _score(c)) is not None]
        return float(np.mean(vals)) if vals else -1.0

    dets = sorted(names, key=_rank, reverse=True)
    n = len(dets)
    bw = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(max(1.5, 0.26 * n) * len(groups) + 2.6, 5.0))
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    loosened: dict[str, float] = {}
    drew_err = False
    for gi, g in enumerate(groups):
        for di, det in enumerate(dets):
            c = _cell(det, g)
            score = None if c is None else _score(c)
            if score is None:
                continue
            x = gi + (di - (n - 1) / 2) * bw
            sd = _score_std(c)
            drew_err = drew_err or sd is not None
            ax.bar(x, score, bw * 0.92, color=dcol(det), edgecolor="#222222",
                   linewidth=0.4, zorder=3,
                   yerr=sd, error_kw=dict(ecolor="#222222", elinewidth=0.9,
                                          capsize=2.0))
            ef = _f(c.get("eff_fpr"))
            if ef is not None and ef > meta["fpr"] + 1e-6:
                loosened[g] = ef

    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_xticks(range(len(groups)))
    xlabels = []
    for g in groups:
        lab = _sd(g)
        if g in loosened:
            lab += f"\n@ {loosened[g] * 100:.2f}% FPR"
        xlabels.append(lab)
    ax.set_xticklabels(xlabels, fontsize=8.5, color=_INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(
        f"goodness = {_COMBINE_EXPR[combine]}(2·AUROC−1,  oracle TPR@"
        f"{meta['fpr'] * 100:g}%FPR)", fontsize=9, color=_INK)
    ax.tick_params(axis="y", labelsize=8, colors=_MUTED)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    for gi in range(1, len(groups)):
        ax.axvline(gi - 0.5, color="#dddddd", lw=0.8, zorder=0)

    handles = [plt.Line2D([], [], marker="s", ls="", ms=8, color=dcol(d), label=d)
               for d in dets]
    ax.legend(handles=handles, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), framealpha=0.9, borderaxespad=0.0,
              title=f"bars ranked by {_sd('mean')}\n{_COMBINE_LABEL[combine]}")

    ax.set_title(title or f"detector goodness across the MANUGRAM splits "
                 f"({_COMBINE_LABEL[combine]} of Gini and oracle "
                 f"TPR@{meta['fpr'] * 100:g}%FPR)", fontsize=10.5, color=_INK)
    notes = []
    if drew_err:
        notes.append("error bar = SD of the per-seed / per-fold goodness "
                     "(zero-shot on the same eval rows as the trained models); "
                     "the 'mean' group's bar is the mean per-split SD.")
    if loosened:
        notes.append("@ n.nn% FPR: 1%-FPR unreachable on that few humans; TPR "
                     "read at the smallest achievable FPR (one human flagged).")
    for i, note in enumerate(notes):
        fig.text(0.02, 0.008 + 0.018 * (len(notes) - 1 - i), note,
                 fontsize=7, color=_MUTED)
    fig.tight_layout(rect=(0, 0.03 + 0.02 * (len(notes) - 1) if notes else 0, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Judge x generator matrix (run_judge_by_generator.py). One heatmap panel per
# MANUGRAM split, judges down the rows, generators across the columns; the cell
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


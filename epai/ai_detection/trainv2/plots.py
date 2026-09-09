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
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

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
# across families), marker shape = MANUGRAM split (splits.MARKER), a bold filled
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

    def _err(r: dict) -> tuple[float | None, float | None]:
        """SD, when the CSV carries a finite one (trained rows). Only ever
        drawn on the mean marker -- the light per-split points stay bare."""
        xe, ye = r.get("auroc_std"), r.get("tpr_at_fpr_std")
        try:
            xe, ye = float(xe), float(ye)
        except (TypeError, ValueError):
            return None, None
        return (xe if np.isfinite(xe) else None, ye if np.isfinite(ye) else None)

    # Light per-split points: bare markers, no error bars.
    for r in rows:
        if r["split"] == "mean" or not np.isfinite(r["tpr_at_fpr"]):
            continue
        ax.scatter([r["auroc"]], [r["tpr_at_fpr"]], s=42,
                   marker=marker.get(r["split"], "P"),
                   color=_tint(hue[r[key]]), edgecolors="#6a6a6a", lw=0.6,
                   zorder=3)

    means = [(v, next((r for r in rows if r[key] == v and r["split"] == "mean"), None))
             for v in names]
    means = [(v, m) for v, m in means if m is not None and np.isfinite(m["tpr_at_fpr"])]
    x_hi = 1.02
    # Stagger labels of means that sit close together (the trained detectors
    # pile up near the ceiling): sort by position, push each label further out.
    order = sorted(range(len(means)),
                   key=lambda i: (means[i][1]["auroc"], means[i][1]["tpr_at_fpr"]))
    dy_of: dict[int, int] = {}
    for rank, i in enumerate(order):
        prev = order[rank - 1] if rank else None
        close = prev is not None and (
            abs(means[i][1]["auroc"] - means[prev][1]["auroc"]) < 0.03
            and abs(means[i][1]["tpr_at_fpr"] - means[prev][1]["tpr_at_fpr"]) < 0.06)
        dy_of[i] = (dy_of.get(prev, -2) + 12) if close else 4

    for i, (v, m) in enumerate(means):
        xe, ye = _err(m)
        ax.errorbar([m["auroc"]], [m["tpr_at_fpr"]], xerr=xe, yerr=ye,
                    fmt="o", ms=12, mfc=hue[v], mec="#222222", mew=1.0,
                    ecolor="#222222", elinewidth=1.0, capsize=2.5, zorder=5)
        right = m["auroc"] > 0.45 + 0.78 * (x_hi - 0.45)
        ax.annotate(v, (m["auroc"], m["tpr_at_fpr"]), textcoords="offset points",
                    xytext=(-9 if right else 9, dy_of[i]),
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
                 f"{fpr * 100:g}%FPR across the MANUGRAM splits",
                 fontsize=10.5, color=_INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# The eval-only OOD corpus on its own (run_in_domain_performance.py /
# run_lodo_performance.py, alongside their main scatter). A dedicated zoom on
# MG-Den -- the one point the main scatters bury against the ceiling: every
# trained detector (marker carries the SD across the source models) and every
# zero-shot pick (open diamond, reference only), one point each. Reads the same
# CSV records; keeps the full 0-1 TPR axis so the drop from the in-domain
# ceiling stays visible.
# ---------------------------------------------------------------------------


def denario_scatter(
    rows: list[dict], meta: dict, path: Path, *, title: str = ""
) -> None:
    """AUROC vs. oracle TPR@FPR on the ``meta["ood"]`` corpus alone (MG-Den by
    default). Built for the single-corpus default; with several OOD corpora each
    detector gets one point per corpus, colour-coded by detector only."""
    fpr = meta["fpr"]
    ood_keys = set(meta.get("ood") or ["mg_den"])

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

    # Same colour per detector as the main scatter: identical assignment rule
    # (sorted detector names -> palette slots) over the identical row set.
    names = sorted({r["detector"] for r in rows})
    hue = {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(names)}

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

    # No in-plot point labels: on MG-Den every detector piles into the same
    # corner, so a name next to each mark is an unreadable stack. The legend
    # carries the colour -> detector key instead.
    for r in zshot:
        ax.scatter([_f(r["auroc"])], [_f(r["tpr_at_fpr"])], s=80, marker="D",
                   facecolors="none", edgecolors=hue[r["detector"]],
                   linewidths=1.5, zorder=3)

    for r in trained:
        ax.errorbar([_f(r["auroc"])], [_f(r["tpr_at_fpr"])],
                    xerr=_f(r.get("auroc_std")), yerr=_f(r.get("tpr_at_fpr_std")),
                    fmt="o", ms=12, mfc=hue[r["detector"]], mec="#222222",
                    mew=1.0, ecolor="#222222", elinewidth=1.0, capsize=2.5,
                    zorder=5)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.tick_params(labelsize=8, colors=_MUTED)
    ax.set_xlabel("AUROC (machine = positive)", fontsize=9, color=_INK)
    ax.set_ylabel(f"oracle TPR @ {fpr * 100:g}% FPR", fontsize=9, color=_INK)

    # One coloured entry per detector, exactly as the main scatter's legend,
    # then the marker-shape key (fill = trained, outline = zero-shot) + chance.
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=hue[v], label=v)
               for v in names]
    handles.append(plt.Line2D([], [], ls="", label=""))
    handles += [
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc="#8a8a8a",
                   mec="#222222", label="trained  (bars = SD across source models)"),
        plt.Line2D([], [], marker="D", ls="", ms=9, mfc="none", mec="#8a8a8a",
                   label="zero-shot pick  (reference)"),
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
# Domain randomization -> MG-Den transfer (run_dr_transfer.py). Per training
# regime a panel; per detector a base (filled circle) and its domain-randomized
# twin (hollow ring, same colour) joined by a faint connector, so the eye reads
# the base -> twin shift. Zero-shot picks are open diamonds (reference). Reads
# the run_dr_transfer performance.csv records (dicts). Colour rule matches
# zeroshot_scatter / denario_scatter so the figures agree.
# ---------------------------------------------------------------------------


def dr_scatter(rows: list[dict], meta: dict, path: Path, *, title: str = "") -> None:
    fpr = meta["fpr"]

    def _f(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    regimes = [rg for rg in ("in_domain", "lodo")
               if any(r.get("regime") == rg for r in rows)]
    if not regimes:
        print("dr_scatter: no in_domain / lodo rows — skipped")
        return
    panel_title = {"in_domain": "in-domain 70/30 -> MG-Den",
                   "lodo": "LODO -> MG-Den"}

    names = sorted({r["detector"] for r in rows})
    hue = {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(names)}

    ood = [r for r in rows if r.get("role") == "ood"]
    aurocs = [_f(r["auroc"]) for r in ood if _f(r["auroc"]) is not None]
    x_lo = max(0.0, min(aurocs) - 0.05) if aurocs else 0.80
    x_hi = 1.01

    fig, axes = plt.subplots(1, len(regimes), figsize=(6.4 * len(regimes), 5.6),
                             sharex=True, sharey=True, squeeze=False)
    axes = list(axes[0])

    for ax, regime in zip(axes, regimes):
        ax.set_facecolor(_SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.axvline(0.5, color="#b5495b", lw=1.0, ls=":", alpha=0.6, zorder=0)
        ax.axhline(fpr, color="#b5495b", lw=1.0, ls=":", alpha=0.6, zorder=0)
        ax.scatter([0.5], [fpr], s=120, color="#b5495b", marker="x", lw=1.8,
                   alpha=0.6, zorder=0)

        pr = [r for r in ood if r.get("regime") == regime]
        by_det: dict[str, dict[str, dict]] = {}
        for r in pr:
            if r.get("kind") == "trained":
                by_det.setdefault(r["detector"], {})[r["variant"]] = r

        for v, pair in by_det.items():
            b, d = pair.get("base"), pair.get("dr")
            if b and d:
                ax.plot([_f(b["auroc"]), _f(d["auroc"])],
                        [_f(b["tpr_at_fpr"]), _f(d["tpr_at_fpr"])],
                        color=hue[v], lw=1.0, alpha=0.45, zorder=2)
            if b:
                ax.errorbar([_f(b["auroc"])], [_f(b["tpr_at_fpr"])],
                            xerr=_f(b.get("auroc_std")), yerr=_f(b.get("tpr_at_fpr_std")),
                            fmt="o", ms=12, mfc=hue[v], mec="#222222", mew=1.0,
                            ecolor="#222222", elinewidth=1.0, capsize=2.5, zorder=5)
            if d:
                ax.errorbar([_f(d["auroc"])], [_f(d["tpr_at_fpr"])],
                            xerr=_f(d.get("auroc_std")), yerr=_f(d.get("tpr_at_fpr_std")),
                            fmt="o", ms=12, mfc="none", mec=hue[v], mew=2.4,
                            ecolor=hue[v], elinewidth=1.0, capsize=2.5, zorder=6)

        for r in pr:
            if r.get("variant") == "ref":
                ax.scatter([_f(r["auroc"])], [_f(r["tpr_at_fpr"])], s=80,
                           marker="D", facecolors="none",
                           edgecolors=hue[r["detector"]], linewidths=1.5, zorder=3)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.tick_params(labelsize=8, colors=_MUTED)
        ax.set_xlabel("AUROC (machine = positive)", fontsize=9, color=_INK)
        ax.set_title(panel_title[regime], fontsize=9.5, color=_INK)
    axes[0].set_ylabel(f"oracle TPR @ {fpr * 100:g}% FPR", fontsize=9, color=_INK)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=hue[v], label=v)
               for v in names]
    handles.append(plt.Line2D([], [], ls="", label=""))
    handles += [
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc="#8a8a8a", mec="#222222",
                   label="base  (embeddings unperturbed)"),
        plt.Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec="#8a8a8a",
                   mew=2.2, label="domain-randomized twin"),
        plt.Line2D([], [], marker="D", ls="", ms=9, mfc="none", mec="#8a8a8a",
                   label="zero-shot pick  (reference)"),
        plt.Line2D([], [], marker="x", ls="", color="#b5495b",
                   label=f"chance  (0.5, {fpr * 100:g}%)"),
    ]
    axes[-1].legend(handles=handles, fontsize=8, loc="upper left",
                    bbox_to_anchor=(1.01, 1.0), framealpha=0.9, borderaxespad=0.0,
                    title=f"colour = detector\n{meta.get('dr', '')}")

    fig.suptitle(title or "domain randomization -> MG-Den transfer",
                 fontsize=10.5, color=_INK)
    fig.text(0.02, 0.008,
             f"MG-Den: never trained on, 80 human rows -> the {fpr * 100:g}%-FPR "
             "grid point floors at FPR=0. Read the base -> twin delta, not the "
             "absolute TPR.", fontsize=7, color=_MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
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


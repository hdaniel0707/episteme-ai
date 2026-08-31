r"""Figures for the eval-only corpora. Separate files, separate axes, separate claim.

Everything here draws ``holdout_<name>.json`` (see
``epai/utils/holdout_core.py``) and nothing else. It adds no panel, no bar and no
series to any existing figure: a Denario result is an extra page beside the main
results, never a column inside them, which is the whole point of the eval-only
block.

The style helpers are imported from ``visualization_utils`` rather than copied,
private names included. That is deliberate: these figures are meant to be read
against the in-domain and transfer figures, and a caption strip, a grid or a
baseline hatch that drifted from theirs would make two figures of the same
quantity look like two different measurements.

WHAT THE FIGURE HAS TO SAY, BEYOND THE BARS
--------------------------------------------
Three things, and each is drawn rather than left to the caption:

  the ceiling            on an all-machine corpus a detector that answers
                         "machine" to everything scores 100%. That line is on
                         the axis, labelled, so the bar next to it is never read
                         as a quality ranking.
  the sampling error     82 documents. The error bar on every bar is the Wilson
                         interval of that rate on that n -- NOT the spread over
                         seeds, which resamples the source's threshold and not
                         these rows.
  the missing FPR        no human row means no realized false-positive rate, so
                         nothing on the page says what a detection cost. It is
                         the honest limit of the exercise and it is written
                         under every figure.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from epai.utils.holdout_core import holdout_metric_specs, rate_ci
from epai.utils.cd_core import MetricSpec
from epai.utils.visualization_utils import (
    ID_CORPUS_COLORS,
    CD_HATCH_COLOR,
    CD_HATCH_LINEWIDTH,
    _cd_baseline_legend_styles,
    _cd_baseline_style,
    _cd_pad,
    _cd_row_mark,
    _cd_xgrid,
    _legend_strip_in,
    _note_strip,
    _num,
    _wrap_note,
)

#: Colour of the "answer machine to everything" ceiling. Grey, not red: it is
#: not a failure line, it is the value the metric cannot exceed, and painting it
#: like a chance line would say the opposite of what it means.
HOLDOUT_CEILING_COLOR = "0.45"

#: Colour of the declared-cutoff panel -- the off-the-shelf reading, which only
#: the zero-shot baselines have.
HOLDOUT_DECLARED_COLOR = "#8c7fb8"


def plot_holdout_comparison(out_dir: Path, payload: dict) -> None:
    """Every detector on one eval-only corpus, one figure per reportable metric.

    Written as ``holdout_<corpus>_<metric>.png``. One figure per metric rather
    than one figure with a metric axis, matching the in-domain corpus comparison
    beside it, so the two can be laid side by side per metric.
    """
    rows: list[dict] = payload.get("variants") or []
    sources: list[str] = payload.get("source_names") or []
    if not rows or not sources:
        print(f"No holdout comparison to draw in {out_dir}")
        return

    specs = [
        spec for spec in holdout_metric_specs()
        if any(f"{spec.key}_mean" in r for r in rows)
    ]
    if not specs:
        print(f"No holdout metrics on the rows in {out_dir}")
        return

    for spec in specs:
        _plot_holdout_metric(out_dir, payload, rows, sources, spec)
    print(f"Holdout plots saved to: {out_dir}")


def _row_ci(row: dict, source: str, key: str, n: int) -> tuple[float, float]:
    """Wilson interval of one cell's rate, as (lower error, upper error) lengths.

    Read off the stored interval where the run wrote one, recomputed from the
    rate and ``n`` where it did not -- an older payload, or a replot of one.
    Returned as *lengths* because that is what matplotlib's asymmetric ``xerr``
    takes, and a rate at 0.99 has a very asymmetric one.
    """
    cell = (row.get("per_source") or {}).get(source) or {}
    mean = _num(cell.get(f"{key}_mean"))
    if not np.isfinite(mean):
        return 0.0, 0.0
    lo = _num(cell.get(f"{key}_ci_lo"))
    hi = _num(cell.get(f"{key}_ci_hi"))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo, hi = rate_ci(mean, int(cell.get("n", n) or n))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 0.0, 0.0
    return max(0.0, mean - lo), max(0.0, hi - mean)


def _plot_holdout_metric(
    out_dir: Path,
    payload: dict,
    rows: list[dict],
    sources: list[str],
    spec: MetricSpec,
) -> None:
    """One holdout figure, for one metric. See the caller."""
    label_of: dict = payload.get("source_labels") or {}
    labels = [label_of.get(n, n) for n in sources]
    alg_label = payload.get("alg_label", payload.get("alg_id", out_dir.name))
    holdout_label = payload.get("holdout_label", payload.get("holdout", "holdout"))
    single = payload.get("single_class")
    n_rows_corpus = payload.get("n") or 0

    # Ranked by this metric, so each figure answers its own question. A detector
    # the metric is missing from sorts last rather than vanishing.
    order = sorted(
        rows,
        key=lambda r: (
            -_num(r.get(f"{spec.key}_mean"), -1.0)
            if spec.higher_is_better
            else _num(r.get(f"{spec.key}_mean"), 1e9),
            str(r.get("label", "")),
        ),
    )
    y = np.arange(len(order), dtype=float)
    n_bars = len(sources)
    h = 0.82 / n_bars

    # The declared-cutoff panel only exists if some row carries one, which means
    # only when the run had zero-shot baselines in it.
    declared = np.array(
        [_num(r.get("declared_accuracy_mean")) for r in order], dtype=float
    )
    has_declared = bool(np.isfinite(declared).any())

    fig_h = max(4.2, 0.46 * len(order) * n_bars / 3 + 2.8)
    if has_declared:
        fig, axes = plt.subplots(
            1, 2, figsize=(max(12.0, 1.1 * n_bars + 9.0), fig_h),
            gridspec_kw={"width_ratios": [3.0, 1.0]}, sharey=True,
        )
        ax, ax_dec = axes
    else:
        fig, ax = plt.subplots(figsize=(max(10.0, 1.1 * n_bars + 8.0), fig_h))
        ax_dec = None

    per_source = [
        np.array(
            [
                _num(r.get("per_source", {}).get(name, {}).get(f"{spec.key}_mean"))
                for r in order
            ],
            dtype=float,
        )
        for name in sources
    ]

    for k, name in enumerate(sources):
        values = per_source[k]
        # Wilson on n, per cell. NOT the seed spread: the seeds move the source's
        # threshold, they do not resample these documents, so their agreement is
        # not this number's error bar. See holdout_core.aggregate_holdout.
        errs = np.array(
            [_row_ci(r, name, spec.key, n_rows_corpus) for r in order], dtype=float
        ).T if len(order) else np.zeros((2, 0))
        offset = (n_bars - 1) / 2.0 - k
        bars = ax.barh(
            y + offset * h, np.nan_to_num(values, nan=0.0), h,
            xerr=errs, error_kw={"elinewidth": 0.8, "ecolor": "0.35", "capsize": 1.5},
            color=ID_CORPUS_COLORS[k % len(ID_CORPUS_COLORS)],
            edgecolor="white", linewidth=0.4,
            label=f"threshold from {labels[k]}",
        )
        # Baselines keep the family hatch they carry everywhere else, so a ★ row
        # means the same thing on this page as on the transfer figures.
        for idx, r in enumerate(order):
            if r.get("baseline"):
                style = _cd_baseline_style(r)
                bars[idx].set_hatch(style["hatch"])
                bars[idx].set_edgecolor(CD_HATCH_COLOR)
                bars[idx].set_linewidth(CD_HATCH_LINEWIDTH)
        ax.bar_label(
            bars, fontsize=7,
            labels=["" if not np.isfinite(v) else f"{v*100:.1f}" for v in values],
            padding=2,
        )

    # The mean over the threshold sources, as a rule across its own group rather
    # than an extra bar -- it summarises the group, it is not another source.
    means = np.array([_num(r.get(f"{spec.key}_mean")) for r in order], dtype=float)
    for yi, m in zip(y, means):
        if np.isfinite(m):
            ax.plot([m, m], [yi - 0.44, yi + 0.44], color="black", linewidth=1.5,
                    zorder=5, solid_capstyle="butt")
    ax.plot([], [], color="black", linewidth=1.5, label="mean over sources")

    # The metric's own reference, exactly as the main figures draw it: 1% for a
    # TPR at 1% FPR, 50% for accuracy. Never 50% for everything.
    ax.axvline(spec.reference, linestyle="--", color="red", linewidth=1.2,
               label=f"reference ({spec.reference*100:g}%)")
    if single == "machine":
        # The ceiling, and the reason this axis cannot rank detectors on its own.
        ax.axvline(1.0, linestyle=":", color=HOLDOUT_CEILING_COLOR, linewidth=1.4,
                   label='ceiling: always answer "machine" = 100%')

    ax.set_yticks(y)
    ax.set_yticklabels(
        [_cd_row_mark(r) + str(r.get("label", "")) for r in order], fontsize=9
    )
    ax.invert_yaxis()
    finite = np.concatenate(
        [v[np.isfinite(v)] for v in per_source] + [np.array([spec.reference])]
    )
    lo, hi = _cd_pad(float(finite.min()), float(finite.max()))
    ax.set_xlim(max(spec.floor, lo), min(spec.ceiling + 0.02, max(hi + 0.02, 1.02)))
    ax.set_xlabel(
        f"{spec.label} on {holdout_label}"
        + ("  —  detection rate" if single == "machine" else "")
    )
    ax.set_title(
        f"{holdout_label}: thresholds transported from each training corpus",
        fontsize=10,
    )
    _cd_xgrid(ax)

    if ax_dec is not None:
        dec_bars = ax_dec.barh(
            y, np.nan_to_num(declared, nan=0.0), 0.6,
            color=HOLDOUT_DECLARED_COLOR, edgecolor="white", linewidth=0.4,
        )
        ax_dec.bar_label(
            dec_bars, fontsize=7.5,
            labels=["" if not np.isfinite(v) else f"{v*100:.1f}" for v in declared],
            padding=2,
        )
        if single == "machine":
            ax_dec.axvline(1.0, linestyle=":", color=HOLDOUT_CEILING_COLOR,
                           linewidth=1.2)
        ax_dec.set_xlim(0.0, 1.05)
        ax_dec.set_xlabel("share flagged machine")
        ax_dec.set_title("At the DECLARED cutoff\n(zero-shot, saw no data)",
                         fontsize=9)
        _cd_xgrid(ax_dec)

    fig.suptitle(f"{alg_label} — {holdout_label}, {spec.label}\n{payload.get('subtitle', '')}")

    note = _wrap_note(
        f"EVAL-ONLY corpus: no model was trained on it, no threshold was fitted "
        f"on it, and no number on this page enters the in-domain or transfer "
        f"results. Every bar is the SAME {n_rows_corpus} documents scored by the "
        f"same detector, under a threshold calibrated on a different training "
        f"corpus — so the spread across a group is what the choice of "
        f"calibration corpus costs, not a property of these documents.",
        spec.note,
        (
            "This corpus holds machine text only, so accuracy IS recall and the "
            "ceiling is 100%: a detector that answers \"machine\" to everything "
            "sits on the dotted line. There is no human row here, so there is no "
            "realized false-positive rate — nothing on this figure says what a "
            "detection cost, and these rates therefore cannot be ranked against "
            "each other on their own. Read a bar beside the same detector's "
            "transfer cells, where the price is visible."
            if single == "machine" else ""
        ),
        (
            "The confound this design cannot remove: these documents are also a "
            "different GENRE from every training corpus, so a low rate is not "
            "separable into \"missed this generator\" and \"reads this kind of "
            "writing as human\"."
        ),
        "Error bars are the 95% Wilson interval of the rate on n documents — the "
        "sampling error. They are NOT the spread over seeds: the seeds refit the "
        "source's threshold, they do not resample these rows.",
        "★ zero-shot baseline (no classifier trained)"
        if any(r.get("baseline") for r in order) else "",
    )

    handles, legend_labels = ax.get_legend_handles_labels()
    for style in _cd_baseline_legend_styles(
        order, [bool(r.get("baseline")) for r in order]
    ):
        handles.append(
            Line2D([], [], linestyle="none", marker="*", color=style["color"],
                   markersize=11, label=style["label"])
        )
        legend_labels.append(style["label"])

    note_frac = _note_strip(note, fig)
    legend_frac = _legend_strip_in(len(handles)) / fig_h
    bottom = min(note_frac + legend_frac, 0.45)
    fig.legend(
        handles, legend_labels, fontsize=8, loc="lower center",
        ncol=min(len(handles), 5), frameon=False,
        bbox_to_anchor=(0.5, min(note_frac, bottom)),
    )
    fig.text(0.01, 0.01, note, fontsize=7.5, color="dimgray", va="bottom")
    fig.tight_layout(rect=(0, bottom, 1, 0.92))

    prev = plt.rcParams["hatch.linewidth"]
    plt.rcParams["hatch.linewidth"] = CD_HATCH_LINEWIDTH
    fig.savefig(
        out_dir / f"holdout_{payload.get('holdout', 'set')}_{spec.filename}.png",
        dpi=150,
    )
    plt.rcParams["hatch.linewidth"] = prev
    plt.close(fig)


def plot_holdouts(out_dir: Path) -> None:
    """Draw every ``holdout_*.json`` sitting in ``out_dir``.

    What the replot scripts call: the JSON is the record and the figures are
    derived from it, so a caption or a colour can be changed without re-running
    a single fit.
    """
    for path in sorted(out_dir.glob("holdout_*.json")):
        with open(path) as f:
            payload = json.load(f)
        plot_holdout_comparison(out_dir, payload)


__all__ = ["plot_holdout_comparison", "plot_holdouts"]

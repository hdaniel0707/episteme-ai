r"""Figures for the eval-only corpora. Separate files, separate axes, separate claim.

Everything here draws ``holdout_<name>.json`` (see
``epai/utils/holdout_core.py``) and nothing else. It adds no panel, no bar and no
series to any existing figure: a Denario result is an extra page beside the main
results, never a column inside them, which is the whole point of the eval-only
block.

The style helpers are imported from ``visualization_utils`` rather than copied,
private names included. That is deliberate: these figures are meant to be read
against the in-domain and transfer figures, and a caption strip, a grid or a
row mark that drifted from theirs would make two figures of the same
quantity look like two different measurements.

WHAT THE FIGURE HAS TO SAY, BEYOND THE BARS
--------------------------------------------
Three things, and each is drawn rather than left to the caption:

  the ceiling            on an all-machine corpus a detector that answers
                         "machine" to everything scores 100%. That line is on
                         the axis, labelled, so the bar next to it is never read
                         as a quality ranking.
  the sampling error     80 documents. The error bar on every bar is the Wilson
                         interval of that rate on that n -- NOT the spread over
                         seeds, which resamples the source's threshold and not
                         these rows.
  the missing FPR        no human row means no realized false-positive rate, so
                         nothing on the page says what a detection cost. It is
                         the honest limit of the exercise and it is written
                         under every figure.

ONE BAR PER DETECTOR, AND WHICH ONE
------------------------------------
This used to draw one bar per threshold SOURCE -- four bars a row on the
in-domain figure -- plus a separate right-hand panel for the baselines' declared
cutoff. Both are gone, and for the same reason: with the baselines now read at
their declared cutoff by default (``cd_core.baseline_fits_threshold``) a ❄ row's
value does not depend on the source at all, so three quarters of its bars were
redraws of one number, and the declared panel was a second copy of the bar
already on the main axis.

What one bar means depends on what the run had:

  ``primary_source``     an in-domain run trains a pooled row on the whole file
  present                (``include_pooled``), and that is the row to show when
                         showing one -- the model that saw everything available,
                         rather than one domain picked for us.
  absent                 a cross-domain or lodo run has no such row: every model
                         left something out. The bar is then the unweighted mean
                         over the sources, and the grey whisker behind it spans
                         the lowest and highest source, so what the choice of
                         calibration corpus cost is still on the page rather
                         than only in the JSON.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from epai.utils.holdout_core import holdout_metric_specs, rate_ci
from epai.utils.cd_core import (
    MetricSpec,
    TRAINED_MARK,
    baseline_was_fitted,
    zeroshot_legend_text,
)
from epai.utils.visualization_utils import (
    CD_TRAINED_LEGEND,
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

#: Bar fills, by what produced the row. This is the one figure in the set where
#: hue may carry the zero-shot / trained split: there is a single bar per row and
#: no second series for a hue to be confused with, so the fill can repeat what
#: the row mark says instead of competing with it. Muted versions of the mark
#: colours -- the glyph should stay the more saturated of the two.
HOLDOUT_ZEROSHOT_COLOR = "#7fa8cc"
HOLDOUT_TRAINED_COLOR = "#e5b06a"

#: The min-max-over-sources whisker on a run with no pooled row. Grey and behind
#: the Wilson bar: it is a spread over CALIBRATION CHOICES, not an uncertainty
#: about these documents, and the two must not read as one interval.
HOLDOUT_SPREAD_COLOR = "0.55"


def plot_holdout_comparison(out_dir: Path, payload: dict) -> None:
    """Every detector on one eval-only corpus, one figure per reportable metric.

    Written as ``holdout_<corpus>_<metric>.png``. One figure per metric rather
    than one figure with a metric axis, matching the in-domain corpus comparison
    beside it, so the two can be laid side by side per metric.

    A figure holds only the rows that CARRY its metric. That matters now that a
    zero-shot baseline reports no ``tpr_at_srcfpr_*``: the alternative is a
    tpr@srcFPR figure whose ❄ rows are all empty bars, which reads as "these
    detectors caught nothing" when what happened is that the number does not
    exist for them. A row without the metric is off the page, and the caption
    says how many were dropped and why.
    """
    rows: list[dict] = payload.get("variants") or []
    sources: list[str] = payload.get("source_names") or []
    if not rows or not sources:
        print(f"No holdout comparison to draw in {out_dir}")
        return

    drawn = False
    for spec in holdout_metric_specs():
        present = [r for r in rows if f"{spec.key}_mean" in r]
        if not present:
            continue
        _plot_holdout_metric(
            out_dir, payload, present, sources, spec,
            n_omitted=len(rows) - len(present),
        )
        drawn = True
    if not drawn:
        print(f"No holdout metrics on the rows in {out_dir}")
        return
    print(f"Holdout plots saved to: {out_dir}")

def _bar_value(
    row: dict, sources: list[str], key: str, primary: str | None
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """The one number this row's bar shows, its Wilson error, and its spread.

    Returns ``(value, (ci_lo_len, ci_hi_len), (src_min, src_max))``. The spread
    is ``(nan, nan)`` when there is nothing to show for it -- a primary source
    was named, or fewer than two sources reported the metric -- and the caller
    draws no whisker then.

    ``primary``, when given and present on the row, IS the value: one source's
    cell, read with the interval that cell carries. Otherwise the value is the
    unweighted mean over the sources, which is the row aggregate
    ``holdout_core._finish_holdout_row`` already wrote, and its interval is
    recomputed from that mean because no single cell's stored interval belongs
    to it.

    Both intervals are Wilson on n -- the sampling error of a rate on this many
    documents -- never the seed spread. The seeds refit the source's threshold;
    they do not resample these rows.
    """
    per_source = row.get("per_source") or {}
    cells = [per_source.get(name) or {} for name in sources]
    values = np.array(
        [_num(c.get(f"{key}_mean")) for c in cells], dtype=float
    )
    finite = values[np.isfinite(values)]

    if primary is not None and f"{key}_mean" in (per_source.get(primary) or {}):
        cell = per_source[primary]
        value = _num(cell.get(f"{key}_mean"))
        n = int(cell.get("n", 0) or 0)
        lo, hi = _num(cell.get(f"{key}_ci_lo")), _num(cell.get(f"{key}_ci_hi"))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            lo, hi = rate_ci(value, n)
        return value, _err_lengths(value, lo, hi), (float("nan"), float("nan"))

    value = _num(row.get(f"{key}_mean"))
    if not np.isfinite(value):
        return value, (0.0, 0.0), (float("nan"), float("nan"))
    # One n for the corpus. `_finish_holdout_row` writes a LIST when the sources
    # did not all score the same rows (a partly-filled score column), and there
    # is no single binomial n then -- so there is no interval either, rather
    # than one computed off whichever n happened to be first.
    n = row.get("n")
    lo, hi = rate_ci(value, n) if isinstance(n, int) else (float("nan"), float("nan"))
    spread = (
        (float(finite.min()), float(finite.max()))
        if finite.size > 1 else (float("nan"), float("nan"))
    )
    return value, _err_lengths(value, lo, hi), spread


def _err_lengths(value: float, lo: float, hi: float) -> tuple[float, float]:
    """An interval as the two *lengths* matplotlib's asymmetric ``xerr`` wants.

    A rate near the ceiling has a very asymmetric Wilson interval, and passing
    it as endpoints would silently draw it as a symmetric one.
    """
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(value)):
        return 0.0, 0.0
    return max(0.0, value - lo), max(0.0, hi - value)


def _primary_source(payload: dict, sources: list[str]) -> str | None:
    """The one threshold source to draw, or None to average them.

    ``primary_source`` is written by ``holdout_core.holdout_comparison``. The
    fallback exists for payloads written before it did: the pooled row is called
    ``all`` unless a config renamed it, and a run that has one is exactly a run
    whose figure should show it.
    """
    if "primary_source" in payload:
        name = payload["primary_source"]
        return name if name in sources else None
    labels = payload.get("source_labels") or {}
    return next(
        (
            n for n in sources
            if n.lower() == "all" or str(labels.get(n, n)).lower() in ("all", "all data")
        ),
        None,
    )


def _plot_holdout_metric(
    out_dir: Path,
    payload: dict,
    rows: list[dict],
    sources: list[str],
    spec: MetricSpec,
    n_omitted: int = 0,
) -> None:
    """One holdout figure, for one metric. See the caller.

    ``rows`` are the detectors that carry the metric; ``n_omitted`` counts the
    ones that did not, which the caption has to account for rather than leave
    as a silently shorter figure.
    """
    label_of: dict = payload.get("source_labels") or {}
    alg_label = payload.get("alg_label", payload.get("alg_id", out_dir.name))
    holdout_label = payload.get("holdout_label", payload.get("holdout", "holdout"))
    single = payload.get("single_class")
    n_rows_corpus = payload.get("n") or 0
    primary = _primary_source(payload, sources)

    # Ranked by this metric, so each figure answers its own question. Every row
    # here carries it -- the caller dropped the ones that did not.
    order = sorted(
        rows,
        key=lambda r: (
            -_num(r.get(f"{spec.key}_mean"), -1.0)
            if spec.higher_is_better
            else _num(r.get(f"{spec.key}_mean"), 1e9),
            str(r.get("label", "")),
        ),
    )

    drawn = [_bar_value(r, sources, spec.key, primary) for r in order]
    values = np.array([d[0] for d in drawn], dtype=float)
    errs = np.array([d[1] for d in drawn], dtype=float).T if drawn else np.zeros((2, 0))
    spread = np.array([d[2] for d in drawn], dtype=float)

    y = np.arange(len(order), dtype=float)
    fig_h = max(4.0, 0.42 * len(order) + 2.6)
    fig, ax = plt.subplots(figsize=(11.0, fig_h))

    colors = [
        HOLDOUT_ZEROSHOT_COLOR if r.get("baseline") else HOLDOUT_TRAINED_COLOR
        for r in order
    ]
    # Behind the bars, and drawn first so a bar never hides its own whisker's
    # near end: the range the metric took across the threshold sources. Only on
    # a run with no pooled row -- where one bar is a mean over corpora that each
    # left something out, and the reader is owed the range that mean covers.
    for yi, (lo, hi) in zip(y, spread):
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.plot([lo, hi], [yi, yi], color=HOLDOUT_SPREAD_COLOR, linewidth=1.0,
                    solid_capstyle="butt", zorder=1)
            for x in (lo, hi):
                ax.plot([x, x], [yi - 0.16, yi + 0.16], color=HOLDOUT_SPREAD_COLOR,
                        linewidth=1.0, solid_capstyle="butt", zorder=1)

    bars = ax.barh(
        y, np.nan_to_num(values, nan=0.0), 0.62,
        xerr=errs, error_kw={"elinewidth": 0.9, "ecolor": "0.25", "capsize": 2.0},
        color=colors, edgecolor="white", linewidth=0.4, zorder=2,
    )
    ax.bar_label(
        bars, fontsize=7.5,
        labels=["" if not np.isfinite(v) else f"{v*100:.1f}" for v in values],
        padding=2,
    )

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

    finite = values[np.isfinite(values)]
    bounds = np.concatenate(
        [finite, spread[np.isfinite(spread)], np.array([spec.reference])]
    )
    lo, hi = _cd_pad(float(bounds.min()), float(bounds.max()))
    ax.set_xlim(max(spec.floor, lo), min(spec.ceiling + 0.02, max(hi + 0.02, 1.02)))
    ax.set_xlabel(
        f"{spec.label} on {holdout_label}"
        + ("  —  detection rate" if single == "machine" else "")
    )
    ax.set_title(_holdout_axis_title(primary, label_of, sources), fontsize=10)
    _cd_xgrid(ax)

    fig.suptitle(f"{alg_label} — {holdout_label}, {spec.label}\n{payload.get('subtitle', '')}")

    note = _wrap_note(
        f"EVAL-ONLY corpus: no model was trained on it, no threshold was fitted "
        f"on it, and no number on this page enters the in-domain or transfer "
        f"results. Every bar is the SAME {n_rows_corpus} documents scored by the "
        f"same detector.",
        _holdout_bar_note(primary, label_of, sources),
        (
            f"{n_omitted} detector(s) are not on this figure: they report no "
            f"{spec.short}. A zero-shot row has no tpr@srcFPR by construction — "
            f"that number is a threshold placed at a nominal FPR on human rows "
            f"from a training corpus, and placing one is the thing a ❄ row does "
            f"not do. They are on the accuracy figure beside this one."
            if n_omitted else ""
        ),
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
        _holdout_mark_note(order),
    )

    handles, legend_labels = ax.get_legend_handles_labels()
    # Patches, not the blank handles the other figures use: here the bar FILL
    # also carries the zero-shot / trained split, so one legend entry can
    # explain the glyph and the colour together instead of leaving a reader to
    # guess that the two encodings mean the same thing.
    if len({bool(r.get("baseline")) for r in order}) == 2:
        for color, text in (
            (HOLDOUT_ZEROSHOT_COLOR, zeroshot_legend_text(order)),
            (HOLDOUT_TRAINED_COLOR, CD_TRAINED_LEGEND),
        ):
            handles.append(Patch(facecolor=color, edgecolor="white", label=text))
            legend_labels.append(text)
    if np.isfinite(spread).any():
        handles.append(
            Line2D([], [], color=HOLDOUT_SPREAD_COLOR, linewidth=1.0,
                   label="range over threshold sources")
        )
        legend_labels.append("range over threshold sources")

    note_frac = _note_strip(note, fig)
    legend_frac = _legend_strip_in(len(handles)) / fig_h
    bottom = min(note_frac + legend_frac, 0.45)
    fig.legend(
        handles, legend_labels, fontsize=8, loc="lower center",
        ncol=min(len(handles), 4), frameon=False,
        bbox_to_anchor=(0.5, min(note_frac, bottom)),
    )
    fig.text(0.01, 0.01, note, fontsize=7.5, color="dimgray", va="bottom")
    fig.tight_layout(rect=(0, bottom, 1, 0.92))

    fig.savefig(
        out_dir / f"holdout_{payload.get('holdout', 'set')}_{spec.filename}.png",
        dpi=150,
    )
    plt.close(fig)


def _holdout_mark_note(order: list[dict]) -> str:
    """The caption line explaining ❄ / ☀ on this figure, or "" with no baselines.

    The zero-shot half is not one sentence: a ❄ row read at its declared cutoff
    reports the SAME number whichever corpus supplied the threshold, which is
    the strongest thing this figure can say about it. A ❄ row from a run that
    refit its cutoff -- or from a result produced before that was optional --
    does not, and saying it did would be a false claim about a real number.
    """
    if not any(r.get("baseline") for r in order):
        return ""
    if any(baseline_was_fitted(r) for r in order):
        tail = (
            " — so on this figure a ❄ row's number DOES depend on which corpus "
            "supplied its threshold, exactly as a trained row's does."
        )
    else:
        tail = (
            ", so a ❄ row's number does not depend on a training corpus at all: "
            "it is the same number whichever one supplied the threshold."
        )
    return f"{zeroshot_legend_text(order)}{tail}  {TRAINED_MARK} trained."


def _holdout_axis_title(
    primary: str | None, label_of: dict, sources: list[str]
) -> str:
    """What the single bar is, in the axis title. Two runs, two sentences."""
    if primary is not None:
        return (
            f"threshold transported from the model trained on "
            f"{label_of.get(primary, primary)}"
        )
    return (
        f"mean over the {len(sources)} training corpora a threshold was "
        f"transported from"
    )


def _holdout_bar_note(
    primary: str | None, label_of: dict, sources: list[str]
) -> str:
    """The caption sentence that says what one bar covers, and what it hides."""
    others = [label_of.get(n, n) for n in sources if n != primary]
    if primary is not None:
        return (
            f"One bar per detector, under the threshold the model trained on "
            f"{label_of.get(primary, primary)} produced — the pooled row, the "
            f"one that saw everything in the file. The other "
            f"{len(others)} training corpora ({', '.join(others)}) scored these "
            f"same documents too and are in the JSON; they are left off the "
            f"figure because a per-corpus threshold is a worse answer to "
            f"\"what would this detector catch\" than the pooled one, not "
            f"because they agreed."
        )
    return (
        f"One bar per detector: the unweighted mean over the {len(sources)} "
        f"corpora a threshold was transported from ({', '.join(others)}), each "
        f"of which left something out — there is no pooled model here to prefer. "
        f"The grey whisker is the lowest and highest of those {len(sources)} "
        f"numbers, so it is what the choice of calibration corpus COSTS, not an "
        f"uncertainty about these documents."
    )


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

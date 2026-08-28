"""Draw the memorisation probe: the guided/unguided delta, and its context.

``probe_memorisation.py`` answers one question -- does the judge already know
these documents -- and prints the answer as two contrasts at the bottom of a
log. Those two contrasts are the whole finding, and they are exactly the kind of
number that is unconvincing as text: "+0.40 longest_run on human rows, improved
on 12% of rows" is either nothing or everything depending on the spread behind
it, and the log cannot show the spread. This script draws it.

It reads only what the probe wrote. No Parquet is opened, because the probe
never wrote to one: it is a read-only experiment whose entire output lives under
``results/llm_judge_memorisation/``.

WHAT IT READS
-------------
Each positional argument is one of three things:

``a run directory``
    ``results/llm_judge_memorisation/<parquet>+<model>/`` — the directory one
    probe run wrote. Everything below is computed from ``completions.csv``, the
    per-completion frame, and nothing here reads the run's own
    ``memorisation_summary.csv``: the two are written together and the
    completions carry strictly more, including the pairing that the aggregate
    has already summed away.

``the results root``
    ``results/llm_judge_memorisation/`` — every run directory beneath it, which
    is how the corpora are compared against each other in one call.

``a completions.csv``
    Named directly, for a run whose directory has been moved.

THE FIGURES, AND WHAT EACH ONE IS FOR
-------------------------------------
``memorisation_guided_delta.png``
    The headline, and the only figure that answers the question the probe was
    built to ask. Per document, guided minus unguided, drawn as a distribution
    with the mean and its 95% interval on top -- machine rows beside human rows,
    because machine rows cannot be memorised and so the machine violin is what
    "no effect" looks like measured on this sample size. Both metrics are shown:
    ``longest_run`` is the one to believe, ``lcs_f1`` is beside it because a
    delta that appears in the fuzzy metric and not in the sharp one is a
    paraphrase effect rather than recall.

``memorisation_paired_scatter.png``
    The same delta undrawn: one point per document, unguided on x, guided on y,
    the diagonal for "the corpus name changed nothing". A mean delta near zero
    is produced both by a cloud sitting on the diagonal and by a cloud scattered
    hard on both sides of it, and only the first of those is evidence of
    nothing happening.

``memorisation_overlap_distribution.png``
    The raw ``longest_run`` distribution per class and probe, as a survival
    curve, with the verbatim threshold marked. The threshold is the claim that
    matters -- fluency buys paraphrase, it does not buy fifteen identical tokens
    in a row -- so what is worth seeing is the shape of the tail, and a
    histogram of a quantity whose interesting values are rare hides it.

``memorisation_by_domain.png``
    Mean overlap per domain, class and probe. The human halves of these corpora
    are not equally exposed: reddit WritingPrompts and Reuters newswire have
    been mirrored across the web for a decade, so if any of this is recall
    rather than predictability, it has a domain in which it should be strongest.

WHERE THE OUTPUTS GO
--------------------
``results/llm_judge_memorisation_figures/<run>[__<run>...]/``, one directory per
call, named after every run it drew. ``--out`` overrides it.

Alongside the figures:
  - ``memorisation_plot_summary.csv``  — per (run, class, metric): the paired
    delta, its spread, its 95% interval and the share of documents it improved
  - ``memorisation_plot_summary.log``  — the same in words, with the reading of
    each contrast spelled out and the sample sizes it rests on

HOW TO READ THE RESULT, IN ONE PARAGRAPH
----------------------------------------
A delta near zero on both classes is the clean outcome: naming the corpus bought
the model nothing, so it has no instance-level memory of it, and the in-domain
detection numbers are not explained away by recall. A clear positive delta on
human rows *only* is the finding the probe exists to detect. A positive delta on
both classes is not memorisation -- machine rows cannot be memorised -- it is
the provenance sentence making the continuation task easier in general, and it
invalidates the contrast rather than answering it.

Usage:
    # one run
    uv run python epai/ai_detection/score/plot_memorisation.py \
        results/llm_judge_memorisation/ghostbuster_gpt35t+gpt56luna

    # every corpus probed so far, side by side
    uv run python epai/ai_detection/score/plot_memorisation.py \
        results/llm_judge_memorisation

    # one domain of one run
    uv run python epai/ai_detection/score/plot_memorisation.py \
        results/llm_judge_memorisation/ghostbuster_gpt56luna+gpt56luna \
        --domains wp
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# The probe's own vocabulary: the two arm names and the run threshold. Imported
# rather than restated so that renaming an arm, or moving the threshold off 10
# tokens, cannot leave a figure captioned with the old one.
from epai.ai_detection.score.probe_memorisation import (
    PROBE_GUIDED,
    PROBE_UNGUIDED,
    PROBES,
    VERBATIM_RUN,
)

BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_ROOT = BASE_DIR / "results"
REPORT_DIR = RESULTS_ROOT / "llm_judge_memorisation_figures"

COMPLETIONS_CSV = "completions.csv"

HUMAN_CLASS = "human"
MACHINE_CLASS = "machine"

# Same two colors as CD_HUMAN_COLOR / CD_MACHINE_COLOR in
# epai/utils/visualization_utils.py, so a human distribution is the same blue
# here as in the classifier figures. Copied rather than imported: that module
# pulls in plotly, scipy and sklearn, and this script needs two hex strings.
HUMAN_COLOR = "#3d6fb4"
MACHINE_COLOR = "#c5392f"
CLASS_COLOR = {HUMAN_CLASS: HUMAN_COLOR, MACHINE_CLASS: MACHINE_COLOR}
CLASSES = (HUMAN_CLASS, MACHINE_CLASS)

# The unguided arm is drawn hollow and the guided arm filled, everywhere, so the
# two arms are the same pair of marks in every figure.
PROBE_ALPHA = {PROBE_UNGUIDED: 0.35, PROBE_GUIDED: 0.85}

# Both overlap metrics, with the reading each one needs attached to it. The
# order is the order they are drawn in, sharp measure first: `longest_run` is
# the one the probe's conclusion rests on and `lcs_f1` is its cross-check, and a
# figure that puts the fuzzy one first invites the opposite reading.
METRICS: tuple[tuple[str, str, str], ...] = (
    ("longest_run", "longest identical token run",
     "the sharp one — fluency does not buy a long identical run"),
    ("lcs_f1", "LCS F1 against the true continuation",
     "the fuzzy one — machine rows lead here for predictability alone"),
)

BOOTSTRAP = 2000  # resamples behind the 95% interval on the delta
RNG_SEED = 0  # the interval is decoration on a mean; it must not move per run
DPI = 150


class Tee:
    """Collect printed lines so they can be echoed to stdout and a log file."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def print(self, line: str = "") -> None:
        print(line)
        self._lines.append(line)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._lines) + "\n")


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """One probe run: its completions, and the paired deltas derived from them.

    ``deltas`` is keyed by (class, metric) and holds one value per *document*,
    not per completion: the contrast is guided minus unguided on the same row,
    so a row completed under only one arm contributes nothing and is dropped
    here rather than being averaged into a difference of two different samples.
    """

    label: str
    origin: Path
    frame: pd.DataFrame  # per completion
    deltas: dict[tuple[str, str], np.ndarray]
    paired: dict[tuple[str, str], pd.DataFrame]  # (class, metric) -> unguided/guided

    def n(self, klass: str) -> int:
        return int((self.frame["class"] == klass).sum())


def _class_of(generated_by: pd.Series) -> np.ndarray:
    """``human`` against everything else — the only split the probe makes.

    The AI half carries a ``<model_slug>:<prompt_id>`` label and the human half
    the literal ``human``. The distinction is not cosmetic here: the machine
    rows are the control arm precisely because their provenance makes them
    impossible to have been memorised.
    """
    return np.where(generated_by.astype(str) == HUMAN_CLASS,
                    HUMAN_CLASS, MACHINE_CLASS)


def _pair(frame: pd.DataFrame, klass: str, metric: str) -> pd.DataFrame:
    """The rows of one class completed under *both* arms, one row per document.

    Returned wide, with a column per arm, which is what both the delta and the
    paired scatter need. ``aggfunc="mean"`` collapses a document completed twice
    under the same arm; that is not the normal case, and averaging it is the
    only treatment that keeps the row paired.
    """
    block = frame[frame["class"] == klass]
    if block.empty or not set(PROBES).issubset(set(block["probe"].unique())):
        return pd.DataFrame()
    wide = block.pivot_table(index="row", columns="probe", values=metric,
                             aggfunc="mean").dropna()
    missing = [arm for arm in PROBES if arm not in wide.columns]
    return pd.DataFrame() if missing else wide


def load_run(path: Path, domains: list[str] | None, out: Tee) -> Run | None:
    """One run from its ``completions.csv``, with the paired deltas computed."""
    csv = path if path.is_file() else path / COMPLETIONS_CSV
    if not csv.is_file():
        out.print(f"  [skip] {path}: no {COMPLETIONS_CSV}")
        return None

    frame = pd.read_csv(csv)
    label = (path.parent if path.is_file() else path).name
    if domains:
        if "domain" not in frame.columns:
            out.print(f"  [skip] {label}: --domains given but the file has no "
                      "'domain' column")
            return None
        frame = frame[frame["domain"].astype(str).isin(domains)]
        label = f"{label}+{'-'.join(domains)}"
    if frame.empty:
        out.print(f"  [skip] {label}: no completions"
                  + (f" in domains {domains}" if domains else ""))
        return None

    frame = frame.copy()
    frame["class"] = _class_of(frame["generated_by"])
    for metric, _, _ in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=[m for m, _, _ in METRICS])

    deltas: dict[tuple[str, str], np.ndarray] = {}
    paired: dict[tuple[str, str], pd.DataFrame] = {}
    for klass in CLASSES:
        for metric, _, _ in METRICS:
            wide = _pair(frame, klass, metric)
            if wide.empty:
                continue
            paired[(klass, metric)] = wide
            deltas[(klass, metric)] = (
                wide[PROBE_GUIDED] - wide[PROBE_UNGUIDED]
            ).to_numpy(dtype=float)

    arms = sorted(frame["probe"].astype(str).unique())
    out.print(f"  [read] {label}: {len(frame)} completions "
              f"(human {int((frame['class'] == HUMAN_CLASS).sum())}, "
              f"machine {int((frame['class'] == MACHINE_CLASS).sum())}), "
              f"arms {arms}")
    if not deltas:
        out.print("         no document was completed under both arms — the "
                  "delta figures will be empty for this run")
    return Run(label=label, origin=csv, frame=frame, deltas=deltas, paired=paired)


def resolve(path: Path) -> Path:
    """A path as typed, or the same path under the repo root.

    The probe writes ``results/...`` relative to the working directory, so the
    paths in this project's notes are repo-relative; accepting them from
    anywhere costs one fallback and saves the reader a cd.
    """
    if path.exists() or path.is_absolute():
        return path
    candidate = BASE_DIR / path
    return candidate if candidate.exists() else path


def load_runs(paths: list[Path], domains: list[str] | None, out: Tee) -> list[Run]:
    """Every run named on the command line, in the order it was named."""
    runs: list[Run] = []
    for raw in paths:
        path = resolve(raw)
        if path.is_file():
            run = load_run(path, domains, out)
            if run is not None:
                runs.append(run)
        elif path.is_dir() and (path / COMPLETIONS_CSV).is_file():
            run = load_run(path, domains, out)
            if run is not None:
                runs.append(run)
        elif path.is_dir():
            children = sorted(d for d in path.iterdir()
                              if d.is_dir() and (d / COMPLETIONS_CSV).is_file())
            if not children:
                out.print(f"  [skip] {path}: neither a run directory nor a "
                          "directory of run directories")
            for child in children:
                run = load_run(child, domains, out)
                if run is not None:
                    runs.append(run)
        else:
            raise SystemExit(
                f"{raw} is not a run directory, a results root, or a "
                f"{COMPLETIONS_CSV}."
            )
    if not runs:
        raise SystemExit("Nothing to plot: no run carried scored completions.")
    return runs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def bootstrap_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of a paired delta.

    A t interval would do as well on these sample sizes; the bootstrap is used
    because ``longest_run`` is an integer-valued count with a long right tail
    and a handful of documents supplying most of it, which is the shape a
    normal-theory interval flatters. Seeded, so the same CSV draws the same bar
    twice.
    """
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(RNG_SEED)
    means = rng.choice(values, size=(BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    tail = 100.0 * (1.0 - level) / 2.0
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def _survival(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x, and the share of values at least x — the tail, drawn as a step."""
    ordered = np.sort(values[np.isfinite(values)])
    if len(ordered) == 0:
        return np.array([]), np.array([])
    share = 1.0 - np.arange(len(ordered)) / len(ordered)
    return ordered, 100.0 * share


def _save(fig, fig_dir: Path, fname: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / fname, dpi=DPI)
    plt.close(fig)
    print(f"  [OK] {fname}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_guided_delta(runs: list[Run], fig_dir: Path, out: Tee) -> None:
    """Guided minus unguided per document: the probe's answer, with its spread.

    One violin per (run, class), one panel per metric, zero drawn as a solid
    line because zero is the null this whole experiment is arranged around. The
    mean and its 95% interval are overlaid on the violin rather than replacing
    it: the mean is the number the log reports, and the violin is what says
    whether that mean is a shift or an artefact of three documents.

    The machine violin is the control and belongs in the same panel as the human
    one. Machine text cannot have been memorised, so whatever width and offset
    the machine violin has is what this sample size produces from no effect at
    all, and the human violin is only interesting insofar as it differs from it.
    """
    if not any(run.deltas for run in runs):
        out.print("  [skip] memorisation_guided_delta.png: no run has documents "
                  "completed under both arms")
        return

    fig, axes = plt.subplots(1, len(METRICS),
                             figsize=(1.5 + 4.6 * len(METRICS), 5.4),
                             squeeze=False)
    width = 0.34
    for ax, (metric, name, reading) in zip(axes[0], METRICS):
        ticks: list[float] = []
        labels: list[str] = []
        for i, run in enumerate(runs):
            for klass, offset in zip(CLASSES, (-width / 1.7, width / 1.7)):
                values = run.deltas.get((klass, metric))
                position = i + offset
                ticks.append(position)
                labels.append(f"{klass}\nn={0 if values is None else len(values)}")
                if values is None or len(values) < 2:
                    continue
                # A kernel density estimate needs the values to have some
                # spread; a narrow slice where every document moved by exactly
                # the same amount (usually zero) would otherwise raise on a
                # singular covariance. Draw the bar alone in that case.
                if float(np.ptp(values)) > 0:
                    parts = ax.violinplot([values], positions=[position],
                                          widths=width, showextrema=False)
                    for body in parts["bodies"]:
                        body.set_facecolor(CLASS_COLOR[klass])
                        body.set_edgecolor("0.15")
                        body.set_alpha(0.55)
                mean = float(np.mean(values))
                low, high = bootstrap_ci(values)
                ax.vlines(position, low, high, color="0.1", linewidth=2.0, zorder=3)
                ax.plot([position], [mean], marker="o", color="0.1",
                        markersize=6, zorder=4)
                ax.text(position, mean, f"  {mean:+.2f}", fontsize=8, va="center",
                        ha="left", fontweight="bold")
        ax.axhline(0.0, color="black", linewidth=1.2)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlim(-0.6, len(runs) - 0.4)
        for i, run in enumerate(runs):
            ax.text(i, 1.005, run.label, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8)
            if i:
                ax.axvline(i - 0.5, color="0.85", linewidth=1, zorder=0)
        ax.set_ylabel(f"guided − unguided   ({name})")
        ax.set_title(reading, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Did naming the corpus help?  Above zero on human rows and not "
                 "on machine rows is memorisation;\nthe machine violin is what "
                 "no effect looks like at this sample size", fontsize=11)
    fig.legend(handles=[
        Line2D([], [], color=CLASS_COLOR[k], linewidth=8, alpha=0.55,
               label=f"{k} rows" + (" (control: cannot be memorised)"
                                    if k == MACHINE_CLASS else ""))
        for k in CLASSES
    ] + [Line2D([], [], color="0.1", marker="o", linewidth=2,
                label="mean, 95% bootstrap interval")],
        loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    _save(fig, fig_dir, "memorisation_guided_delta.png")


def plot_paired_scatter(runs: list[Run], fig_dir: Path, out: Tee) -> None:
    """The same documents under both arms, one point each, against the diagonal.

    This is the figure that distinguishes "nothing happened" from "a lot
    happened in both directions and cancelled": the first is a cloud lying along
    the diagonal, the second is a cloud spread across it, and both produce the
    same mean delta. Drawn on ``longest_run`` because that is the metric the
    conclusion rests on; the counts are integers, so the points are jittered a
    little to keep a stack of identical pairs from reading as one document.
    """
    usable = [r for r in runs if any((k, "longest_run") in r.deltas for k in CLASSES)]
    if not usable:
        out.print("  [skip] memorisation_paired_scatter.png: no paired documents")
        return

    rng = np.random.default_rng(RNG_SEED)
    fig, axes = plt.subplots(1, len(usable), figsize=(1.0 + 4.2 * len(usable), 4.8),
                             sharex=True, sharey=True, squeeze=False)
    for ax, run in zip(axes[0], usable):
        top = 1.0
        for klass in CLASSES:
            wide = run.paired.get((klass, "longest_run"))
            if wide is None or wide.empty:
                continue
            x = wide[PROBE_UNGUIDED].to_numpy(dtype=float)
            y = wide[PROBE_GUIDED].to_numpy(dtype=float)
            top = max(top, float(np.max(x)), float(np.max(y)))
            jitter = rng.uniform(-0.16, 0.16, size=(2, len(x)))
            improved = 100.0 * float(np.mean(y > x))
            ax.scatter(x + jitter[0], y + jitter[1], s=22, alpha=0.6,
                       color=CLASS_COLOR[klass], edgecolor="none",
                       label=f"{klass} (n={len(x)}, {improved:.0f}% above)")
        limit = top * 1.08 + 0.4
        ax.plot([0, limit], [0, limit], color="black", linestyle="--",
                linewidth=1, alpha=0.8)
        ax.axhline(VERBATIM_RUN, color="0.45", linewidth=0.8, linestyle=":")
        ax.axvline(VERBATIM_RUN, color="0.45", linewidth=0.8, linestyle=":")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("unguided  (longest identical token run)")
        ax.set_title(run.label, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0][0].set_ylabel("guided  (longest identical token run)")
    fig.suptitle("Same document, both arms — points above the diagonal are the "
                 f"ones the corpus name helped\n(dotted: the {VERBATIM_RUN}-token "
                 "verbatim threshold)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, fig_dir, "memorisation_paired_scatter.png")


def plot_overlap_distribution(runs: list[Run], fig_dir: Path) -> None:
    """The tail of ``longest_run``, per class and arm, against the threshold.

    A survival curve rather than a histogram, because the claim lives in the
    tail: almost every document overlaps for two or three tokens and nothing
    follows from that, while a document overlapping for the threshold length is
    recall and one is enough to be worth finding. The share of rows past the
    threshold is printed in the legend, since that is the number the log
    reports as ``>=run``.
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(1.0 + 4.2 * len(runs), 4.6),
                             sharex=True, sharey=True, squeeze=False)
    for ax, run in zip(axes[0], runs):
        for klass in CLASSES:
            for probe in PROBES:
                block = run.frame[(run.frame["class"] == klass)
                                  & (run.frame["probe"] == probe)]
                if block.empty:
                    continue
                x, share = _survival(block["longest_run"].to_numpy(dtype=float))
                beyond = 100.0 * float(np.mean(
                    block["longest_run"].to_numpy(dtype=float) >= VERBATIM_RUN))
                ax.step(x, share, where="post", color=CLASS_COLOR[klass],
                        alpha=PROBE_ALPHA[probe],
                        linewidth=2.0 if probe == PROBE_GUIDED else 1.6,
                        linestyle="-" if probe == PROBE_GUIDED else "--",
                        label=f"{klass} / {probe}  (n={len(block)}, "
                              f"{beyond:.0f}% ≥ {VERBATIM_RUN})")
        ax.axvline(VERBATIM_RUN, color="0.2", linestyle=":", linewidth=1.4)
        ax.text(VERBATIM_RUN, 101, f" verbatim ≥ {VERBATIM_RUN}", fontsize=8,
                va="bottom", ha="left", color="0.2")
        ax.set_yscale("log")
        ax.set_ylim(0.5, 130)
        ax.set_xlabel("longest identical token run")
        ax.set_title(run.label, fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper right")
    axes[0][0].set_ylabel("completions at least this long (%, log scale)")
    fig.suptitle("How far the overlap goes — the tail is the claim, so it is "
                 "drawn on a log axis", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, fig_dir, "memorisation_overlap_distribution.png")


def plot_by_domain(runs: list[Run], fig_dir: Path, out: Tee) -> None:
    """Mean overlap per domain, class and arm — where recall would show first.

    The human halves are not equally exposed to a pretraining crawl, so a
    memorisation effect has a domain it should be largest in. Both metrics are
    drawn; the ``lcs_f1`` panel is where the machine rows lead for reasons that
    have nothing to do with memory, which is easier to accept once it is visible
    beside the sharp metric rather than asserted in a caption.
    """
    usable = [r for r in runs if "domain" in r.frame.columns
              and r.frame["domain"].nunique() > 0]
    if not usable:
        out.print("  [skip] memorisation_by_domain.png: no run carries a domain "
                  "column")
        return

    domains = sorted({str(d) for r in usable for d in r.frame["domain"].unique()})
    fig, axes = plt.subplots(len(METRICS), len(usable),
                             figsize=(1.2 + 4.0 * len(usable), 3.2 * len(METRICS)),
                             sharex=True, squeeze=False)
    x = np.arange(len(domains), dtype=float)
    for col, run in enumerate(usable):
        for row, (metric, name, _) in enumerate(METRICS):
            ax = axes[row][col]
            bars = [(klass, probe) for klass in CLASSES for probe in PROBES]
            width = 0.8 / len(bars)
            for i, (klass, probe) in enumerate(bars):
                means, counts = [], []
                for domain in domains:
                    block = run.frame[(run.frame["domain"].astype(str) == domain)
                                      & (run.frame["class"] == klass)
                                      & (run.frame["probe"] == probe)]
                    means.append(float(block[metric].mean()) if not block.empty
                                 else np.nan)
                    counts.append(len(block))
                ax.bar(x + (i - (len(bars) - 1) / 2) * width, means,
                       width=width * 0.9, color=CLASS_COLOR[klass],
                       alpha=PROBE_ALPHA[probe], edgecolor="0.15", linewidth=0.5,
                       label=f"{klass} / {probe}" if row == 0 else None)
            if metric == "longest_run":
                ax.axhline(VERBATIM_RUN, color="0.2", linestyle=":", linewidth=1.2)
            ax.set_xticks(x)
            ax.set_xticklabels(domains, fontsize=9)
            ax.set_ylabel(f"mean {name}" if col == 0 else "")
            ax.grid(True, axis="y", alpha=0.3)
            if row == 0:
                ax.set_title(run.label, fontsize=9)
                ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Overlap per domain — a memorisation effect should be largest "
                 "in the most-mirrored human corpus", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, fig_dir, "memorisation_by_domain.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def summary_frame(runs: list[Run]) -> pd.DataFrame:
    """One row per (run, class, metric): the delta, its spread and its interval."""
    records: list[dict] = []
    for run in runs:
        for klass in CLASSES:
            for metric, _, _ in METRICS:
                values = run.deltas.get((klass, metric))
                wide = run.paired.get((klass, metric))
                if values is None or len(values) == 0:
                    continue
                low, high = bootstrap_ci(values)
                records.append({
                    "run": run.label,
                    "class": klass,
                    "metric": metric,
                    "n_documents": len(values),
                    "n_completions": run.n(klass),
                    "mean_unguided": round(float(wide[PROBE_UNGUIDED].mean()), 4),
                    "mean_guided": round(float(wide[PROBE_GUIDED].mean()), 4),
                    "delta_mean": round(float(np.mean(values)), 4),
                    "delta_sd": round(float(np.std(values, ddof=1)), 4)
                    if len(values) > 1 else float("nan"),
                    "delta_ci_low": round(low, 4),
                    "delta_ci_high": round(high, 4),
                    "pct_improved": round(100.0 * float(np.mean(values > 0)), 1),
                })
    return pd.DataFrame(records)


def report(runs: list[Run], summary: pd.DataFrame, out: Tee) -> None:
    """The contrasts in words, and the one sentence that reads them."""
    out.print()
    out.print("=" * 86)
    out.print(f"Guided minus unguided, paired on the same documents "
              f"(verbatim threshold {VERBATIM_RUN} tokens)")
    out.print("-" * 86)
    if summary.empty:
        out.print("  no run had documents completed under both arms.")
    else:
        out.print(f"{'run':<34}{'class':<9}{'metric':<13}{'n':>5}"
                  f"{'delta':>9}{'95% CI':>19}{'improved':>10}")
        out.print("-" * 86)
        for _, row in summary.iterrows():
            out.print(
                f"{row['run'][:33]:<34}{row['class']:<9}{row['metric']:<13}"
                f"{row['n_documents']:>5}{row['delta_mean']:>+9.3f}"
                f"{f'[{row.delta_ci_low:+.3f}, {row.delta_ci_high:+.3f}]':>19}"
                f"{row['pct_improved']:>9.0f}%"
            )

    out.print()
    out.print("-" * 86)
    out.print("  Reading it: a delta near zero on BOTH classes is the clean")
    out.print("  result — naming the corpus bought nothing, so there is no")
    out.print("  instance-level memory of it. A clear positive delta on human")
    out.print("  rows only is the finding the probe exists to detect. A positive")
    out.print("  delta on both is the provenance sentence making the task easier")
    out.print("  in general, which invalidates the contrast rather than answering")
    out.print("  it — machine rows cannot be memorised.")

    out.print()
    out.print("=" * 86)
    out.print("Verbatim runs (the unguided arm, where nothing was hinted)")
    out.print("-" * 86)
    for run in runs:
        unguided = run.frame[run.frame["probe"] == PROBE_UNGUIDED]
        if unguided.empty:
            out.print(f"  {run.label}: no unguided completions")
            continue
        for klass in CLASSES:
            block = unguided[unguided["class"] == klass]
            if block.empty:
                continue
            runs_len = block["longest_run"].to_numpy(dtype=float)
            beyond = int(np.sum(runs_len >= VERBATIM_RUN))
            out.print(f"  {run.label[:40]:<42}{klass:<9}"
                      f"mean run {runs_len.mean():>5.1f}   "
                      f"max {runs_len.max():>4.0f}   "
                      f"≥{VERBATIM_RUN}: {beyond} of {len(block)}")
    out.print("=" * 86)


def run_dir_name(runs: list[Run]) -> str:
    """Named after every run drawn, in the order given, deduplicated."""
    return "__".join(dict.fromkeys(r.label for r in runs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "runs", nargs="+", type=Path,
        help="Run directories under results/llm_judge_memorisation/, that "
             "results root itself, or a completions.csv named directly. They "
             "may be mixed in one call.",
    )
    ap.add_argument(
        "-o", "--out", type=Path, default=None,
        help=f"Where the figures go. Default: {REPORT_DIR}/<run>[__<run>...]/",
    )
    ap.add_argument(
        "--domains", nargs="+", default=None, metavar="DOMAIN",
        help="Restrict every run to these domains before anything is computed.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Tee()
    out.print("Reading runs ...")
    runs = load_runs(args.runs, args.domains, out)

    fig_dir = args.out or (REPORT_DIR / run_dir_name(runs))
    fig_dir.mkdir(parents=True, exist_ok=True)
    print()
    plot_guided_delta(runs, fig_dir, out)
    plot_paired_scatter(runs, fig_dir, out)
    plot_overlap_distribution(runs, fig_dir)
    plot_by_domain(runs, fig_dir, out)

    summary = summary_frame(runs)
    report(runs, summary, out)
    if not summary.empty:
        summary.to_csv(fig_dir / "memorisation_plot_summary.csv", index=False)
    out.save(fig_dir / "memorisation_plot_summary.log")
    print(f"\nAll outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()

"""Draw what the explaining judge said: cue tables as figures, not as logs.

``add_llm_judge_reasons.py`` already prints the cue table it computes and writes
it to ``cue_frequencies.csv``. That table is eleven rows for one corpus, which
is readable; it is eleven rows times four corpora times two classes once the
same judge has explained every parquet in ``data/parquet``, which is not. This
script is the reading end: it takes the rationale output of one or more runs and
turns it into figures whose point is legible without counting columns.

It never writes to a Parquet and never touches a score. Every input is opened
read-only.

WHAT IT READS
-------------
Each positional argument is one of three things, and the script works out which:

``a .parquet``
    The richest input, because the rationale columns are per row: the cue lists,
    the free-text ``other`` field, the domain and the ``generated_by`` label are
    all still attached to the document they came from. Everything below is
    available from a parquet, including the three figures that cannot be
    rebuilt from the CSV (per-domain lift, cue load, cue-to-score shift). One
    source is produced per judge stem found in the file, so a parquet explained
    by two judges yields two series and they are drawn side by side.

``a run directory``
    ``results/llm_judge_reasons/<parquet>+<judge>/`` — the directory the run
    itself wrote. ``cue_frequencies.csv`` carries the aggregate table and
    ``other_evidence.csv`` the out-of-vocabulary rows; the explained-row totals
    are read from ``cue_summary.log``, and derived from the counts and
    percentages in the CSV if that log is missing. Enough for the class-level
    figures, not enough for the per-row ones, which are then skipped by name
    rather than silently.

``the results root``
    ``results/llm_judge_reasons/`` — every run directory beneath it, which is
    the one-argument way to see all four corpora at once.

Mixing them in one invocation is allowed and is the intended use: a parquet for
the run being examined, run directories for the ones being compared against.

THE FIGURES, AND WHAT EACH ONE IS FOR
-------------------------------------
``cue_lift.png``
    The headline. One bar per cue, per source, showing lift = machine% minus
    human%: how much more often the judge cited that cue on machine rows than on
    human rows. The prompt assigns every cue a side, so each bar has an expected
    sign -- human-side cues negative, machine-side cues positive -- and the plot
    is drawn around that expectation: the panel is shaded by side and any bar
    whose sign disagrees with its own side is hatched. A hatched bar is a cue
    the judge is using backwards from the way the prompt describes it, which is
    the single most useful thing this output can tell you and is invisible in a
    sorted table.

``cue_frequencies.png``
    The rates the lift is a difference of. A cue can have near-zero lift because
    the judge never cites it (2% against 3%) or because it cites it constantly
    on both classes (91% against 94%), and those two are opposite findings about
    the prompt. Only the raw pair of percentages separates them.

``cue_by_domain.png``   (parquet sources only)
    Lift per cue per domain, as a heatmap. A cue that separates the classes in
    ``essay`` and not in ``reuter`` is a cue whose usefulness is domain-bound,
    which is the shape of the transfer problem this project keeps meeting: the
    in-domain number is carried by evidence that does not exist in the next
    domain.

``cue_load.png``   (parquet sources only)
    How many cues the judge cited per row, human rows against machine rows, with
    the share of rows it cited none for called out. Zero cues is the judge
    declining to name evidence while still returning a rationale, and a class
    that draws systematically fewer cues than the other is being judged on less.

``other_evidence.png``
    How often the judge reported evidence the closed vocabulary has no name for,
    split by class. The cue counts are trustworthy exactly because the
    vocabulary is enum-constrained, and that is also their blind spot: if this
    bar is high, the tidy table above it is describing a fraction of what the
    judge is actually reading, and the text of those rows is in
    ``other_evidence.csv``.

``cue_score_shift.png``   (parquet sources with the matching score column)
    For each cue, the judge's own mean P(human) on the rows that cited it
    against the rows that did not. The cue table says what the judge *reported*;
    this says whether reporting it went with a different number. A cue with a
    large citation rate and no score shift is decoration.

WHERE THE OUTPUTS GO
--------------------
``results/llm_judge_reason_figures/<source>[__<source>...]/``, one directory per
run, named after every source it drew -- ``ghostbuster_gpt35t+gpt56luna_v3``,
the same spelling the rationale runs use for their own directories, so the
figures sit under a name that says which corpus and which judge they are about.
``--out`` overrides it.

Alongside the figures:
  - ``cue_plot_summary.csv``  — every cue rate and lift, one row per
    (source, cue), which is the four-corpus table the per-run CSVs are slices of
  - ``cue_plot_summary.log``  — what was read, what was skipped and why, and the
    cues whose sign disagrees with the side the prompt put them on

Usage:
    # one corpus, from the parquet: every figure, including the per-row ones
    uv run python epai/ai_detection/score/plot_llm_judge_reasons.py \
        data/parquet/ghostbuster_gpt56luna.parquet

    # the three ghostbuster generations side by side, one bar each per cue
    uv run python epai/ai_detection/score/plot_llm_judge_reasons.py \
        data/parquet/ghostbuster_gpt35t.parquet \
        data/parquet/ghostbuster_gpt54mini.parquet \
        data/parquet/ghostbuster_gpt56luna.parquet \
        data/parquet/science_v3_gpt56luna_0811A.parquet

    # everything that has been explained so far, from the reports alone
    uv run python epai/ai_detection/score/plot_llm_judge_reasons.py \
        results/llm_judge_reasons

    # one domain of one corpus, and one judge where the file holds several
    uv run python epai/ai_detection/score/plot_llm_judge_reasons.py \
        data/parquet/ghostbuster_gpt56luna.parquet \
        --domains wp --judge gpt56luna_v3
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# The vocabulary, the column spellings and the cell parsers all come from the
# script that wrote the data. Restating any of them here would be a second
# spelling of one thing: a prompt version that grows a cue would then show up in
# the parquet and not in the figure, which is the failure this import prevents.
from epai.ai_detection.score.add_llm_judge_reasons import (
    HUMAN_CUES,
    HUMAN_CUES_PREFIX,
    MACHINE_CUES,
    MACHINE_CUES_PREFIX,
    OTHER_PREFIX,
    REASON_PREFIX,
    _as_cue_list,
    _as_other,
    _as_reason,
)
from epai.ai_detection.score.add_llm_judge_scores import (
    COL_PREFIX as SCORE_PREFIX,
    DOMAIN_COL,
    GENERATED_BY_COL,
)
from epai.utils.parquet_utils import load_parquet

BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_ROOT = BASE_DIR / "results"
REPORT_DIR = RESULTS_ROOT / "llm_judge_reason_figures"

# The files one rationale run leaves behind; a directory holding the first is a
# run directory, and that is how a positional argument is classified.
FREQ_CSV = "cue_frequencies.csv"
OTHER_CSV = "other_evidence.csv"
SUMMARY_LOG = "cue_summary.log"

HUMAN_CLASS = "human"
MACHINE_CLASS = "machine"

# Same two colors as CD_HUMAN_COLOR / CD_MACHINE_COLOR in
# epai/utils/visualization_utils.py, so a human bar is the same blue here as in
# the classifier figures. Copied rather than imported: that module pulls in
# plotly, scipy and sklearn, and this script needs two hex strings from it.
HUMAN_COLOR = "#3d6fb4"
MACHINE_COLOR = "#c5392f"
SIDE_COLOR = {HUMAN_CLASS: HUMAN_COLOR, MACHINE_CLASS: MACHINE_COLOR}

# Sign disagreement is marked with a hatch rather than a different color: the
# color already carries which side the prompt put the cue on, and a bar needs to
# say both things at once.
MISFIRE_HATCH = "////"

DPI = 150

# Cues in one fixed order everywhere -- human side first, machine side second,
# each in the order the prompt lists them. Sorting by value instead would put
# every figure of every source in a different order and make the comparison
# figures unreadable.
CUE_ORDER: list[tuple[str, str]] = (
    [(cue, HUMAN_CLASS) for cue in HUMAN_CUES]
    + [(cue, MACHINE_CLASS) for cue in MACHINE_CUES]
)
CUE_SIDE = dict(CUE_ORDER)
CUES = [cue for cue, _ in CUE_ORDER]


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
# One source: a corpus explained by one judge, however it was loaded
# ---------------------------------------------------------------------------


@dataclass
class Source:
    """Everything the figures need about one (corpus, judge) pair.

    ``rows`` is the per-row frame and is None for a source loaded from a run
    directory: the CSV is already aggregated over rows, and there is no honest
    way to invent the rows back. Every figure that needs it checks for it and
    says which sources it had to leave out, because a per-domain panel missing
    one corpus without saying so reads as that corpus having no domains.
    """

    label: str  # legend/title name: "<corpus stem>+<judge stem>"
    stem: str  # judge stem, e.g. "gpt56luna_v3"
    origin: Path
    n_human: int  # explained human rows
    n_machine: int  # explained machine rows
    freq: pd.DataFrame  # cue, side, human_n, human_pct, machine_n, machine_pct, lift_pct
    rows: pd.DataFrame | None = None  # per-row, parquet sources only
    other: pd.DataFrame | None = None  # rows carrying out-of-vocabulary evidence

    @property
    def n_explained(self) -> int:
        return self.n_human + self.n_machine

    def lift(self) -> np.ndarray:
        """Lift per cue, in CUE_ORDER, NaN where the source has no such cue."""
        by_cue = self.freq.set_index("cue")["lift_pct"]
        return np.array([float(by_cue.get(cue, np.nan)) for cue in CUES])

    def rates(self, klass: str) -> np.ndarray:
        """Citation rate per cue for one class, in CUE_ORDER, as percentages."""
        by_cue = self.freq.set_index("cue")[f"{klass}_pct"]
        return np.array([float(by_cue.get(cue, np.nan)) for cue in CUES])


def _class_of(generated_by: pd.Series) -> np.ndarray:
    """``human`` against everything else, which is what every split here is.

    The AI half of a corpus carries a ``<model_slug>:<prompt_id>`` label and the
    human half carries the literal ``human``; no figure below separates two
    generators within one file, so the two-way split is the whole mapping.
    """
    return np.where(generated_by.astype(str) == HUMAN_CLASS,
                    HUMAN_CLASS, MACHINE_CLASS)


def _freq_table(rows: pd.DataFrame, n_human: int, n_machine: int) -> pd.DataFrame:
    """The cue table, computed exactly as ``cue_report`` computes it.

    Same denominators (explained rows of that class, not all rows of it) and
    the same definition of lift, so a figure drawn from a parquet and a figure
    drawn from that run's own CSV are the same numbers.
    """
    records: list[dict] = []
    is_human = (rows["class"] == HUMAN_CLASS).to_numpy(dtype=bool)
    for cue, side in CUE_ORDER:
        flags = rows[f"cue_{cue}"].to_numpy(dtype=bool)
        h = int((flags & is_human).sum())
        m = int((flags & ~is_human).sum())
        h_pct = 100.0 * h / n_human if n_human else float("nan")
        m_pct = 100.0 * m / n_machine if n_machine else float("nan")
        records.append({
            "cue": cue, "side": side,
            "human_n": h, "human_pct": round(h_pct, 2),
            "machine_n": m, "machine_pct": round(m_pct, 2),
            "lift_pct": round(m_pct - h_pct, 2),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Loading: parquet
# ---------------------------------------------------------------------------


def judge_stems(df: pd.DataFrame) -> list[str]:
    """Judge stems the file has rationale columns for, in column order."""
    return [c[len(REASON_PREFIX):] for c in df.columns
            if c.startswith(REASON_PREFIX)]


def _score_column(df: pd.DataFrame, stem: str) -> str | None:
    """The score column of the judge that wrote ``stem``'s rationales, if any.

    Exact match first: the rationale stem is ``<model>_<prompt>`` and the parsed
    score column is that stem under the score prefix. A logprob run appends a
    mode suffix, so a prefix match is accepted as a fallback -- but only when it
    is unambiguous, since picking one of two scoring modes at random would put a
    number in the figure that no caption could name.
    """
    exact = f"{SCORE_PREFIX}{stem}"
    if exact in df.columns:
        return exact
    candidates = [c for c in df.columns if c.startswith(exact)]
    return candidates[0] if len(candidates) == 1 else None


def source_from_parquet(
    df: pd.DataFrame, path: Path, stem: str, domains: list[str] | None, out: Tee
) -> Source | None:
    """One source per judge stem in a parquet, built from the explained rows.

    Only rows carrying a rationale are counted. A row the run never reached is
    not evidence of anything -- the sample is drawn by band, so the unexplained
    rows are systematically the ones nobody asked about, and letting them into a
    denominator would divide every rate by the size of the corpus instead of the
    size of the sample.
    """
    reason_col = f"{REASON_PREFIX}{stem}"
    human_col = f"{HUMAN_CUES_PREFIX}{stem}"
    machine_col = f"{MACHINE_CUES_PREFIX}{stem}"
    other_col = f"{OTHER_PREFIX}{stem}"

    if domains:
        if DOMAIN_COL not in df.columns:
            out.print(f"  [skip] {path.name}: --domains given but the file has "
                      f"no {DOMAIN_COL!r} column")
            return None
        df = df[df[DOMAIN_COL].astype(str).isin(domains)]

    explained = np.array([_as_reason(v) is not None for v in df[reason_col]])
    if not explained.any():
        out.print(f"  [skip] {path.name} + {stem}: no explained rows"
                  + (f" in domains {domains}" if domains else ""))
        return None
    df = df[explained]

    empty: list[list[str]] = [[] for _ in range(len(df))]
    human_cues = ([_as_cue_list(v) or [] for v in df[human_col]]
                  if human_col in df.columns else empty)
    machine_cues = ([_as_cue_list(v) or [] for v in df[machine_col]]
                    if machine_col in df.columns else empty)
    cited = [set(h) | set(m) for h, m in zip(human_cues, machine_cues)]

    rows = pd.DataFrame({
        "domain": (df[DOMAIN_COL].astype(str).to_numpy()
                   if DOMAIN_COL in df.columns else np.array(["all"] * len(df))),
        "generated_by": (df[GENERATED_BY_COL].astype(str).to_numpy()
                         if GENERATED_BY_COL in df.columns
                         else np.array([HUMAN_CLASS] * len(df))),
    })
    rows["class"] = _class_of(rows["generated_by"])
    for cue in CUES:
        rows[f"cue_{cue}"] = [cue in c for c in cited]
    rows["n_cues"] = [len(c) for c in cited]

    others = ([_as_other(v) for v in df[other_col]] if other_col in df.columns
              else [None] * len(df))
    rows["has_other"] = [o is not None for o in others]

    score_col = _score_column(df, stem)
    rows["score"] = (pd.to_numeric(df[score_col], errors="coerce").to_numpy()
                     if score_col else np.nan)

    n_human = int((rows["class"] == HUMAN_CLASS).sum())
    n_machine = int((rows["class"] == MACHINE_CLASS).sum())
    other_rows = pd.DataFrame({
        "domain": rows.loc[rows["has_other"], "domain"].to_numpy(),
        "generated_by": rows.loc[rows["has_other"], "generated_by"].to_numpy(),
        "other_evidence": [o for o in others if o is not None],
    })

    label = f"{path.stem}+{stem}" + (f"+{'-'.join(domains)}" if domains else "")
    out.print(f"  [read] {label}: {n_human + n_machine} explained rows "
              f"(human {n_human}, machine {n_machine})"
              + ("" if score_col else "  — no matching score column"))
    return Source(
        label=label,
        stem=stem,
        origin=path,
        n_human=n_human,
        n_machine=n_machine,
        freq=_freq_table(rows, n_human, n_machine),
        rows=rows,
        other=other_rows if not other_rows.empty else None,
    )


# ---------------------------------------------------------------------------
# Loading: a run directory written by add_llm_judge_reasons.py
# ---------------------------------------------------------------------------


_EXPLAINED_RE = re.compile(
    r"explained rows\s*:\s*(\d+)\s*\(human\s+(\d+),\s*machine\s+(\d+)\)"
)


def _totals_from_log(directory: Path) -> tuple[int, int] | None:
    """Explained human/machine rows as the run itself reported them."""
    log = directory / SUMMARY_LOG
    if not log.is_file():
        return None
    match = _EXPLAINED_RE.search(log.read_text(errors="replace"))
    return (int(match.group(2)), int(match.group(3))) if match else None


def _totals_from_freq(freq: pd.DataFrame) -> tuple[int, int]:
    """Explained rows recovered from the counts and percentages in the CSV.

    ``n = 100 * count / pct`` for any cue the class actually cited, taken from
    the largest count so the rounding of ``pct`` to two decimals matters least.
    This is the fallback for a run directory whose log has been moved; it is
    exact for every realistic sample size, and it is only ever used for
    denominators that the log would otherwise have stated outright.
    """
    totals: list[int] = []
    for klass in (HUMAN_CLASS, MACHINE_CLASS):
        counts = freq[f"{klass}_n"].to_numpy(dtype=float)
        pcts = freq[f"{klass}_pct"].to_numpy(dtype=float)
        usable = (counts > 0) & (pcts > 0)
        if not usable.any():
            totals.append(0)
            continue
        best = int(np.argmax(np.where(usable, counts, -1)))
        totals.append(int(round(100.0 * counts[best] / pcts[best])))
    return totals[0], totals[1]


def source_from_report(directory: Path, out: Tee) -> Source | None:
    """One source from a ``results/llm_judge_reasons/<parquet>+<judge>/`` run."""
    freq_path = directory / FREQ_CSV
    if not freq_path.is_file():
        out.print(f"  [skip] {directory}: no {FREQ_CSV}")
        return None
    freq = pd.read_csv(freq_path)
    if "lift_pct" not in freq.columns:  # written before lift was tabulated
        freq["lift_pct"] = freq["machine_pct"] - freq["human_pct"]
    known_side = freq["cue"].map(CUE_SIDE)
    freq["side"] = (known_side.fillna(freq["side"]) if "side" in freq.columns
                    else known_side)

    totals = _totals_from_log(directory)
    derived = ""
    if totals is None:
        totals = _totals_from_freq(freq)
        derived = "  (totals derived from the CSV — no summary log)"
    n_human, n_machine = totals

    other = None
    other_path = directory / OTHER_CSV
    if other_path.is_file():
        other = pd.read_csv(other_path)

    # The directory is named "<parquet stem>+<judge stem>" by the run; the judge
    # stem is what the figures need, and it is the part after the last '+'.
    label = directory.name
    stem = label.rsplit("+", 1)[-1]
    out.print(f"  [read] {label}: {n_human + n_machine} explained rows "
              f"(human {n_human}, machine {n_machine}){derived}")
    return Source(
        label=label,
        stem=stem,
        origin=directory,
        n_human=n_human,
        n_machine=n_machine,
        freq=freq,
        rows=None,
        other=other,
    )


def resolve(path: Path) -> Path:
    """A path as typed, or the same path under the repo root.

    The rationale runs write ``results/...`` relative to the working directory,
    so the paths in this project's notes are repo-relative; accepting them from
    anywhere costs one fallback and saves the reader a cd.
    """
    if path.exists() or path.is_absolute():
        return path
    candidate = BASE_DIR / path
    return candidate if candidate.exists() else path


def load_sources(
    paths: list[Path], judge: str | None, domains: list[str] | None, out: Tee
) -> list[Source]:
    """Every source named on the command line, in the order it was named."""
    sources: list[Source] = []
    for raw in paths:
        path = resolve(raw)
        if path.is_file() and path.suffix == ".parquet":
            df = load_parquet(path)
            df_stems = judge_stems(df)
            wanted = [s for s in df_stems if judge is None or s == judge]
            if not wanted:
                out.print(f"  [skip] {path.name}: no {REASON_PREFIX}* columns"
                          + (f" for judge {judge!r}" if judge else "")
                          + (f" (it has: {df_stems})" if df_stems else ""))
                continue
            for stem in wanted:
                source = source_from_parquet(df, path, stem, domains, out)
                if source is not None:
                    sources.append(source)
        elif path.is_dir() and (path / FREQ_CSV).is_file():
            source = source_from_report(path, out)
            if source is not None:
                sources.append(source)
        elif path.is_dir():
            children = sorted(d for d in path.iterdir()
                              if d.is_dir() and (d / FREQ_CSV).is_file())
            if not children:
                out.print(f"  [skip] {path}: neither a run directory nor a "
                          "directory of run directories")
            for child in children:
                source = source_from_report(child, out)
                if source is not None:
                    sources.append(source)
        else:
            raise SystemExit(
                f"{raw} is not a parquet, a run directory, or the results root."
            )
    if not sources:
        raise SystemExit("Nothing to plot: no source carried explained rows.")
    return sources


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _source_colors(sources: list[Source]) -> list:
    """One color per source when several are drawn, tab10 in argument order."""
    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]
    return [palette[i % len(palette)] for i in range(len(sources))]


def _cue_yaxis(ax) -> np.ndarray:
    """Cues down the y-axis in CUE_ORDER, split by the side they belong to.

    Returns the y positions. The separating line and the two side labels are
    drawn here so every cue-keyed figure agrees about which block is which.
    """
    y = np.arange(len(CUES), dtype=float)
    ax.set_yticks(y)
    ax.set_yticklabels(CUES, fontsize=9)
    for tick, cue in zip(ax.get_yticklabels(), CUES):
        tick.set_color(SIDE_COLOR[CUE_SIDE[cue]])
    ax.set_ylim(len(CUES) - 0.5, -0.5)  # first cue at the top
    ax.axhline(len(HUMAN_CUES) - 0.5, color="0.4", linewidth=1, linestyle=":")
    return y


def _save(fig, fig_dir: Path, fname: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / fname, dpi=DPI)
    plt.close(fig)
    print(f"  [OK] {fname}")


def _no_rows_note(sources: list[Source]) -> str:
    """The names of the sources a per-row figure had to leave out."""
    missing = [s.label for s in sources if s.rows is None]
    return (f"not shown (aggregate CSV only, no per-row data): "
            f"{', '.join(missing)}") if missing else ""


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_cue_lift(
    sources: list[Source], fig_dir: Path
) -> list[tuple[str, str, float]]:
    """Lift per cue, with the prompt's expected sign drawn into the panel.

    The x-axis is signed on purpose and the halves are shaded by side: a bar
    that reaches into the shading of its own color is the cue behaving as the
    prompt says, and a bar crossing zero into the other side's shading is not.
    Those crossings are hatched so they survive being read at thumbnail size --
    they are the reason to look at this figure at all.

    Returns the (source, cue, lift) triples it hatched, so the closing report
    can name them in words rather than leaving them to be spotted in the png.
    """
    colors = _source_colors(sources)
    height = 0.8 / len(sources)
    fig, ax = plt.subplots(figsize=(9, 0.46 * len(CUES) * max(1, len(sources) * 0.55) + 2.6))
    y = _cue_yaxis(ax)

    misfires: list[tuple[str, str, float]] = []
    for i, (source, color) in enumerate(zip(sources, colors)):
        lift = source.lift()
        offset = (i - (len(sources) - 1) / 2) * height
        face = color if len(sources) > 1 else [
            SIDE_COLOR[CUE_SIDE[cue]] for cue in CUES
        ]
        bars = ax.barh(y + offset, lift, height=height * 0.92, color=face,
                       edgecolor="0.15", linewidth=0.6,
                       label=source.label if len(sources) > 1 else None)
        for bar, cue, value in zip(bars, CUES, lift):
            if np.isnan(value):
                continue
            expected_positive = CUE_SIDE[cue] == MACHINE_CLASS
            if (value > 0) != expected_positive and value != 0:
                bar.set_hatch(MISFIRE_HATCH)
                misfires.append((source.label, cue, float(value)))

    spans = [float(np.nanmax(np.abs(lift))) for lift in
             (s.lift() for s in sources) if np.isfinite(lift).any()]
    span = max(spans + [1.0])
    limit = 1.18 * span
    ax.set_xlim(-limit, limit)
    ax.axvspan(-limit, 0, color=HUMAN_COLOR, alpha=0.06, zorder=0)
    ax.axvspan(0, limit, color=MACHINE_COLOR, alpha=0.06, zorder=0)
    ax.axvline(0, color="black", linewidth=1)
    ax.text(-limit * 0.97, -0.42, "cited more on HUMAN rows", fontsize=8,
            color=HUMAN_COLOR, ha="left", va="center", fontweight="bold")
    ax.text(limit * 0.97, -0.42, "cited more on MACHINE rows", fontsize=8,
            color=MACHINE_COLOR, ha="right", va="center", fontweight="bold")
    ax.set_xlabel("lift = machine% − human%  (percentage points)")
    ax.set_title("Which cues actually separate the two classes\n"
                 "hatched = the judge cites this cue against the side the "
                 "prompt assigns it", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    if len(sources) > 1:
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, fig_dir, "cue_lift.png")
    return misfires


def plot_cue_frequencies(sources: list[Source], fig_dir: Path) -> None:
    """The two citation rates the lift is a difference of, one panel per source.

    Kept beside the lift figure rather than folded into it because the same lift
    is produced by a cue nobody cites and a cue everybody cites, and the
    difference between those two decides whether the prompt bullet is worth
    keeping.
    """
    fig, axes = plt.subplots(
        1, len(sources), figsize=(1.0 + 4.2 * len(sources), 0.42 * len(CUES) + 2.4),
        sharey=True, squeeze=False,
    )
    for ax, source in zip(axes[0], sources):
        y = _cue_yaxis(ax)
        ax.barh(y - 0.19, source.rates(HUMAN_CLASS), height=0.36,
                color=HUMAN_COLOR, edgecolor="0.15", linewidth=0.5,
                label=f"human rows (n={source.n_human})")
        ax.barh(y + 0.19, source.rates(MACHINE_CLASS), height=0.36,
                color=MACHINE_COLOR, edgecolor="0.15", linewidth=0.5,
                label=f"machine rows (n={source.n_machine})")
        ax.set_xlim(0, 105)
        ax.set_xlabel("rows citing the cue (%)")
        ax.set_title(source.label, fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("How often each cue was cited, per class", fontsize=12)
    fig.tight_layout()
    _save(fig, fig_dir, "cue_frequencies.png")


def _domain_n(rows: pd.DataFrame, domain: str, klass: str) -> int:
    """Explained rows of one class in one domain — the tick label's denominator.

    Worth showing on the axis: a domain the sample only reached twice produces
    a lift of +/-100 as easily as a domain it reached two hundred times, and the
    heatmap cell looks identical either way.
    """
    return int(((rows["domain"] == domain) & (rows["class"] == klass)).sum())


def plot_cue_by_domain(sources: list[Source], fig_dir: Path, out: Tee) -> None:
    """Lift per cue per domain: does the evidence survive changing domain?

    Drawn only for parquet sources, since a run's CSV has already summed over
    the domains. Blue is a cue marking human rows in that domain, red a cue
    marking machine rows, and a row of the heatmap that changes color across
    domains is a cue whose direction is a property of the domain rather than of
    the class.
    """
    usable = [s for s in sources if s.rows is not None
              and s.rows["domain"].nunique() > 1]
    if not usable:
        note = _no_rows_note(sources)
        out.print("  [skip] cue_by_domain.png: needs parquet sources with more "
                  "than one domain" + (f"; {note}" if note else ""))
        return

    fig, axes = plt.subplots(
        1, len(usable), figsize=(1.4 + 3.4 * len(usable), 0.42 * len(CUES) + 2.8),
        sharey=True, squeeze=False,
    )
    for ax, source in zip(axes[0], usable):
        rows = source.rows
        domains = sorted(rows["domain"].unique())
        matrix = np.full((len(CUES), len(domains)), np.nan)
        for j, domain in enumerate(domains):
            block = rows[rows["domain"] == domain]
            human = block[block["class"] == HUMAN_CLASS]
            machine = block[block["class"] == MACHINE_CLASS]
            if human.empty or machine.empty:
                continue
            for i, cue in enumerate(CUES):
                matrix[i, j] = (100.0 * machine[f"cue_{cue}"].mean()
                                - 100.0 * human[f"cue_{cue}"].mean())
        limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit,
                          aspect="auto")
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(
            [f"{d}\nh={_domain_n(rows, d, HUMAN_CLASS)}"
             f" m={_domain_n(rows, d, MACHINE_CLASS)}" for d in domains],
            fontsize=8,
        )
        _cue_yaxis(ax)
        ax.set_ylim(len(CUES) - 0.5, -0.5)
        for i in range(len(CUES)):
            for j in range(len(domains)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:+.0f}", ha="center",
                            va="center", fontsize=7, color="0.1")
        ax.set_title(source.label, fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03,
                     label="lift (pp)")
    fig.suptitle("Cue lift per domain — red marks machine rows, blue marks human",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, fig_dir, "cue_by_domain.png")


def plot_cue_load(sources: list[Source], fig_dir: Path, out: Tee) -> None:
    """How many cues the judge cited per row, and how often it cited none.

    The zero bar is the one to read: a rationale with no cue attached is the
    judge writing prose it declined to ground in the vocabulary, and if that
    happens far more on one class than the other, the two classes were not
    judged on the same amount of evidence.
    """
    usable = [s for s in sources if s.rows is not None]
    if not usable:
        out.print("  [skip] cue_load.png: needs at least one parquet source")
        return

    fig, axes = plt.subplots(
        1, len(usable), figsize=(1.0 + 4.0 * len(usable), 4.4),
        sharey=True, squeeze=False,
    )
    top = max(int(s.rows["n_cues"].max()) for s in usable)
    bins = np.arange(0, top + 2) - 0.5
    for ax, source in zip(axes[0], usable):
        for klass, color, offset in (
            (HUMAN_CLASS, HUMAN_COLOR, -0.19), (MACHINE_CLASS, MACHINE_COLOR, 0.19)
        ):
            block = source.rows[source.rows["class"] == klass]
            if block.empty:
                continue
            counts, _ = np.histogram(block["n_cues"], bins=bins)
            share = 100.0 * counts / len(block)
            ax.bar(np.arange(len(counts)) + offset, share, width=0.36,
                   color=color, edgecolor="0.15", linewidth=0.5,
                   label=f"{klass} (n={len(block)}, "
                         f"{100.0 * (block['n_cues'] == 0).mean():.0f}% with none)")
        ax.set_xticks(range(top + 1))
        ax.set_xlabel("cues cited on the row")
        ax.set_title(source.label, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("rows of that class (%)")
    fig.suptitle("Cue load per row — the zero bar is the judge naming no evidence",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, fig_dir, "cue_load.png")


def plot_other_evidence(sources: list[Source], fig_dir: Path, out: Tee) -> None:
    """How often the closed vocabulary had no name for what the judge saw.

    Two panels: the rate per class, which every source can report, and the rate
    per domain, which needs the per-row data because the CSV holds only the rows
    that *had* other evidence and not the rows that did not.
    """
    have_any = [s for s in sources if s.other is not None or s.rows is not None]
    if not have_any:
        out.print("  [skip] other_evidence.png: no source carries the "
                  "out-of-vocabulary field")
        return

    per_domain = [s for s in sources if s.rows is not None
                  and s.rows["domain"].nunique() > 1]
    ncols = 2 if per_domain else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols + 1.0, 4.4),
                             squeeze=False)

    ax = axes[0][0]
    x = np.arange(len(sources), dtype=float)
    for klass, color, offset in (
        (HUMAN_CLASS, HUMAN_COLOR, -0.19), (MACHINE_CLASS, MACHINE_COLOR, 0.19)
    ):
        rates = []
        for source in sources:
            total = source.n_human if klass == HUMAN_CLASS else source.n_machine
            if source.rows is not None:
                block = source.rows[source.rows["class"] == klass]
                rate = 100.0 * block["has_other"].mean() if len(block) else np.nan
            elif (source.other is not None and total
                  and "generated_by" in source.other.columns):
                hits = _class_of(source.other["generated_by"]) == klass
                rate = 100.0 * int(np.sum(hits)) / total
            else:
                rate = np.nan
            rates.append(rate)
        ax.bar(x + offset, rates, width=0.36, color=color, edgecolor="0.15",
               linewidth=0.5, label=klass)
    ax.set_xticks(x)
    ax.set_xticklabels([s.label for s in sources], fontsize=8, rotation=20,
                       ha="right")
    ax.set_ylabel("explained rows with evidence outside the vocabulary (%)")
    ax.set_title("Out-of-vocabulary evidence, per class", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    if per_domain:
        ax = axes[0][1]
        domains = sorted({d for s in per_domain for d in s.rows["domain"].unique()})
        width = 0.8 / len(per_domain)
        colors = _source_colors(per_domain)
        for i, (source, color) in enumerate(zip(per_domain, colors)):
            rates = [
                100.0 * source.rows.loc[source.rows["domain"] == d, "has_other"].mean()
                if (source.rows["domain"] == d).any() else np.nan
                for d in domains
            ]
            ax.bar(np.arange(len(domains)) + (i - (len(per_domain) - 1) / 2) * width,
                   rates, width=width * 0.92, color=color, edgecolor="0.15",
                   linewidth=0.5, label=source.label)
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels(domains, fontsize=9)
        ax.set_title("Same rate, per domain", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    else:
        note = _no_rows_note(sources)
        if note:
            out.print(f"  [note] other_evidence.png: per-domain panel {note}")

    fig.suptitle("Where the cue table is blind — high bars mean the counts "
                 "above describe only part of the evidence", fontsize=11)
    fig.tight_layout()
    _save(fig, fig_dir, "other_evidence.png")


def plot_cue_score_shift(sources: list[Source], fig_dir: Path, out: Tee) -> None:
    """The judge's own P(human) on rows citing a cue against rows that did not.

    The cue table reports what the judge *said*; this reports whether saying it
    went with a different number. A cue whose two dots sit on top of each other
    was cited without moving the score, however often it was cited -- and since
    the score is P(human), a machine-side cue earns its bullet by pulling the
    dot to the left.
    """
    usable = [s for s in sources
              if s.rows is not None and s.rows["score"].notna().any()]
    if not usable:
        out.print("  [skip] cue_score_shift.png: no parquet source has the "
                  "judge's own score column")
        return

    fig, axes = plt.subplots(
        1, len(usable), figsize=(1.0 + 4.2 * len(usable), 0.42 * len(CUES) + 2.4),
        sharey=True, squeeze=False,
    )
    for ax, source in zip(axes[0], usable):
        rows = source.rows[source.rows["score"].notna()]
        y = _cue_yaxis(ax)
        for i, cue in enumerate(CUES):
            cited = rows[rows[f"cue_{cue}"]]
            not_cited = rows[~rows[f"cue_{cue}"]]
            if cited.empty or not_cited.empty:
                continue
            a = float(not_cited["score"].mean())
            b = float(cited["score"].mean())
            ax.plot([a, b], [i, i], color="0.55", linewidth=1.6, zorder=1)
            ax.scatter([a], [i], color="0.55", s=26, zorder=2)
            ax.scatter([b], [i], color=SIDE_COLOR[CUE_SIDE[cue]], s=42, zorder=3)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("mean judge score  P(human)")
        ax.set_title(f"{source.label}  (n={len(rows)})", fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(handles=[
            Line2D([], [], marker="o", linestyle="", color="0.55",
                   markersize=6, label="rows NOT citing the cue"),
            Line2D([], [], marker="o", linestyle="", color=HUMAN_COLOR,
                   markersize=8, label="rows citing it (human-side cue)"),
            Line2D([], [], marker="o", linestyle="", color=MACHINE_COLOR,
                   markersize=8, label="rows citing it (machine-side cue)"),
        ], fontsize=7, loc="lower right")
    fig.suptitle("Does citing the cue go with a different score?  "
                 "(dashed line = the 0.5 decision threshold)", fontsize=11)
    fig.tight_layout()
    _save(fig, fig_dir, "cue_score_shift.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def summary_frame(sources: list[Source]) -> pd.DataFrame:
    """Every rate and lift of every source in one table, one row per cue."""
    frames = []
    for source in sources:
        frame = source.freq.copy()
        frame.insert(0, "source", source.label)
        frame.insert(1, "judge", source.stem)
        frame["n_human"] = source.n_human
        frame["n_machine"] = source.n_machine
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def report(sources: list[Source], misfires: list[tuple[str, str, float]],
           out: Tee) -> None:
    """The two things worth saying in words beside the figures."""
    out.print()
    out.print("=" * 78)
    out.print("Sources drawn")
    out.print("-" * 78)
    for source in sources:
        detail = "per-row" if source.rows is not None else "aggregate CSV"
        out.print(f"  {source.label:<44}{source.n_explained:>5} rows  ({detail})")

    out.print()
    out.print("=" * 78)
    out.print("Cues used against the side the prompt assigns them")
    out.print("-" * 78)
    if not misfires:
        out.print("  none — every cue's lift has the sign its side predicts.")
    else:
        for label, cue, value in misfires:
            side = CUE_SIDE[cue]
            direction = ("cited MORE on machine rows" if value > 0
                         else "cited MORE on human rows")
            out.print(f"  {label:<40}{cue:<22}{value:>+7.0f}pp  "
                      f"({side}-side cue, {direction})")
        out.print()
        out.print("  A cue here is not necessarily wrong — it is a bullet of the")
        out.print("  prompt the judge is reading in the opposite direction, which")
        out.print("  is either a fact about these corpora or a fact about the")
        out.print("  wording, and the rationales in other_evidence.csv are where")
        out.print("  the difference shows.")
    out.print("=" * 78)


def run_dir_name(sources: list[Source]) -> str:
    """Named after every source drawn, in the order given, deduplicated."""
    return "__".join(dict.fromkeys(s.label for s in sources))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "sources", nargs="+", type=Path,
        help="Parquets explained by add_llm_judge_reasons.py, run directories "
             "under results/llm_judge_reasons/, or that results root itself. "
             "They may be mixed in one call.",
    )
    ap.add_argument(
        "-o", "--out", type=Path, default=None,
        help=f"Where the figures go. Default: {REPORT_DIR}/<source>[__<source>...]/",
    )
    ap.add_argument(
        "--judge", default=None, metavar="STEM",
        help="Only this judge stem from a parquet holding several, e.g. "
             "gpt56luna_v3. Ignored for run directories, which hold one judge "
             "each by construction.",
    )
    ap.add_argument(
        "--domains", nargs="+", default=None, metavar="DOMAIN",
        help="Restrict parquet sources to these domains before anything is "
             "counted. Run directories cannot be narrowed this way — their CSV "
             "is already summed over domains.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Tee()
    out.print("Reading sources ...")
    sources = load_sources(args.sources, args.judge, args.domains, out)

    fig_dir = args.out or (REPORT_DIR / run_dir_name(sources))
    fig_dir.mkdir(parents=True, exist_ok=True)
    print()
    misfires = plot_cue_lift(sources, fig_dir)
    plot_cue_frequencies(sources, fig_dir)
    plot_cue_by_domain(sources, fig_dir, out)
    plot_cue_load(sources, fig_dir, out)
    plot_other_evidence(sources, fig_dir, out)
    plot_cue_score_shift(sources, fig_dir, out)

    report(sources, misfires, out)
    summary_frame(sources).to_csv(fig_dir / "cue_plot_summary.csv", index=False)
    out.save(fig_dir / "cue_plot_summary.log")
    print(f"\nAll outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()

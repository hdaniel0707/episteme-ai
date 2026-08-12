"""Compare human vs AI detector scores *within* parquets — one violin per file.

Works off any per-row detector score in the parquet: ``--metric binox`` reads the
``binoculars_*`` columns, ``--metric llm_judge`` the ``llm_judge_score_*`` columns
written by ``epai/ai_detection/llm_judge/add_llm_judge_scores.py``. The two differ
only in which columns are read: both are *low* for machine text (Binoculars by
construction, the judge because it reports P(human), 1.0 ⇒ human). Everything
below — pairing, violins, histograms, ROC/PR, confusion — is the same either way,
so the two baselines stay directly comparable.

Every ``id`` in a ``*_binox*`` parquet appears once per ``generated_by`` group
(``human``, ``gpt``, ``claude``, ...). For each parquet this script pairs, per
``id``, the ``human`` row against a chosen generator's row (default ``gpt``) and
plots the per-id score differences. Multiple parquets are shown side by side —
one violin each — and every parquet can be given its own target generator.

Pairing & distance
-------------------
For a target generator ``g`` the ``human`` and ``g`` subsets are merged on ``id``
(unique within each generator). For each matched id the score vector across the
metric's columns is taken for human and for ``g``; the pairwise distance is the
Euclidean distance ``|human - g|`` (abs diff for the single-column case), and the
signed difference ``human - g`` keeps direction. MSE = mean over all valid element
pairs of ``(human - g)**2``. Ids present for only one side are reported.

Invalid pairs
-------------
If *either* element of a pair is null or not a real number, the pair is skipped,
recorded, and reported in the closing summary statistics. This is the normal case
for ``--metric llm_judge``: the judge only scores the generators it was asked for,
so every unjudged row simply drops out of the comparison.

Targets
-------
Each parquet is given as ``PATH:TARGET``, where ``TARGET`` is the
``generated_by`` value to pair against ``human``. It is required, and it is one
format: ``<model_slug>:<prompt_id>`` — ``gpt35t:0000A``, ``gpt56luna:0807A`` —
the label :mod:`epai.utils.label_utils` defines and every corpus this repo
writes now stores, ghostbuster and science alike.

Nothing else is accepted. The bare Ghostbuster-era names (``gpt``, ``gpt_plain``,
``gpt_prompt1``) are deprecated: a corpus still carrying them was built before
the convention and should be relabelled with
``epai/scripts/rename_column_values.py`` rather than analysed under two naming
schemes. A target is matched against the file EXACTLY — no case folding, no
prefixes, no guessing which generator was meant — because every near-miss here
ends as a figure titled with a model that did not write the text.

The path/target split is on the first ':', so the target keeps the colon of its
own that the format requires: ``x.parquet:gpt56luna:0807A`` is the file
``x.parquet`` and the generator ``gpt56luna:0807A``.

Where the outputs go
--------------------
results/<metric>_human_vs_ai/<file>+<generator>[__<file>+<generator>...][+<domains>]/

One directory per run, named after every (file, generator) pair it compared —
``binox_human_vs_ai/ghostbuster_gpt35t+gpt35t_0000A/`` — with ``--domains``
appended when the run was narrowed to some of them. Both halves are needed to
keep runs apart: the same file is analysed against several of its generators,
and the same model wrote the AI half of several files. The generator is spelled
with '_' instead of ':', the directory spelling of a label (label_utils.py).

Contents:
  - human_vs_ai_distance_violin.png    — |human − g| per parquet, MSE annotated
  - human_vs_ai_signed_diff_violin.png — signed human − g (which side scores higher)
  - human_vs_ai_score_distributions.png — per parquet: human vs AI score histograms
  - human_vs_ai_roc.png                — ROC per parquet overlaid, AUC in legend
  - human_vs_ai_pr.png                 — precision–recall per parquet overlaid, AP in legend
  - human_vs_ai_confusion.png          — per-parquet 2×2 confusion matrix at best threshold
  - human_vs_ai_summary.csv            — per-parquet dist stats + AUC/AP/acc/F1/confusion
  - human_vs_ai_summary.log            — full text summary incl. skipped-pair details

Classification framing
----------------------
Two classes: positive = AI (target), negative = human. The decision rule follows
the metric's direction: "predict AI when score <= threshold" for Binoculars (low
⇒ machine, decision score = -score), "score >= threshold" for the LLM judge. The
operating threshold is Youden's J optimum on the ROC curve. Classes are balanced
(paired), so the PR chance baseline is 0.5.

Usage:
    # every domain of a combined ghostbuster parquet, one violin each
    uv run python epai/ai_detection/analyse/analyse_score_human_vs_ai.py \
        data/parquet/ghostbuster_gpt35t.parquet:gpt35t:0000A
    
    uv run python epai/ai_detection/analyse/analyse_score_human_vs_ai.py \
        data/parquet/ghostbuster_gpt54mini.parquet:gpt54mini:0701A

    uv run python epai/ai_detection/analyse/analyse_score_human_vs_ai.py \
        data/parquet/ghostbuster_gpt56luna.parquet:gpt56luna:0701A

    # the science corpora read exactly the same way
    uv run python epai/ai_detection/analyse/analyse_score_human_vs_ai.py \
        data/parquet/science_v1_gpt56luna_0807A.parquet:gpt56luna:0807A

    # three generations of the same corpus side by side, one violin per
    # (file, domain) — each file names the model that wrote its AI half
    uv run python epai/ai_detection/analyse/analyse_score_human_vs_ai.py \
        data/parquet/ghostbuster_gpt35t.parquet:gpt35t:0000A \
        data/parquet/ghostbuster_gpt54mini.parquet:gpt54mini:0701A \
        data/parquet/ghostbuster_gpt56luna.parquet:gpt56luna:0701A

    # one domain, LLM-judge scores instead of Binoculars
    uv run python epai/ai_detection/analyse/analyse_score_human_vs_ai.py \
        data/parquet/ghostbuster_gpt35t.parquet:gpt35t:0000A --domains wp \
        --metric llm_judge --score-cols llm_judge_score_gpt54_mini
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epai.utils.corpus_utils import CorpusSpec, read_corpus, specs_from_paths
from epai.utils.label_utils import (
    GENERATOR_LABEL_RE,
    check_generator_label,
    dir_name_from_label,
)
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_ROOT = BASE_DIR / "results"

ID_COL = "id"
GROUP_COL = "generated_by"
HUMAN_GROUP = "human"


@dataclass(frozen=True)
class Metric:
    """One detector score: which columns hold it and which way means 'AI'.

    Everything downstream is written against this, so adding a third detector is
    one entry here rather than a fork of the script.
    """

    key: str  # --metric value; also names the results directory
    prefix: str  # score columns are the ones starting with this
    label: str  # axis label / legend wording
    ai_is_high: bool  # True when a *high* score means machine-written
    hist_range: tuple[float, float] | None = None  # fixed histogram x-range

    @property
    def results_dir(self) -> Path:
        return RESULTS_ROOT / f"{self.key}_human_vs_ai"

    @property
    def rule(self) -> str:
        """The decision rule as it reads in a title, e.g. 'score>=0.5 ⇒ AI'."""
        return "score≥thr ⇒ AI" if self.ai_is_high else "score≤thr ⇒ AI"


METRICS: dict[str, Metric] = {
    "binox": Metric(
        key="binox",
        prefix="binoculars_",
        label="Binoculars score",
        ai_is_high=False,  # Binoculars: low ⇒ machine
    ),
    "llm_judge": Metric(
        key="llm_judge",
        prefix="llm_judge_score_",
        label="LLM-judge P(human)",
        ai_is_high=False,  # the judge reports P(human), so low ⇒ machine
        hist_range=(0.0, 1.0),  # a probability: bin the whole range, always
    ),
}
DEFAULT_METRIC = "binox"

# Per parquet, how many skipped pairs are listed individually in the log.
SKIPPED_DETAIL_LIMIT = 20


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
# Which generator to compare against
# ---------------------------------------------------------------------------


def _generators(df: pd.DataFrame) -> list[str]:
    """The AI generators present, sorted. ``human`` is the baseline, not a target."""
    values = df[GROUP_COL].dropna().astype(str).unique()
    return sorted(v for v in values if v != HUMAN_GROUP)


def check_target(target: str, spec: str) -> str:
    """The target as typed, once it is a ``<model_slug>:<prompt_id>`` label.

    Checked before any file is opened, because the two ways of getting a target
    wrong are indistinguishable once the parquet is loaded: a corpus that does
    not hold ``gpt:000A`` and a target that could never be any corpus's label
    both surface as "no such generator". Naming the format here — while the
    parquet has not been read, let alone five of them — says which of the two it
    was.
    """
    try:
        return check_generator_label(target, f"target in {spec!r}")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def find_target(target: str, df: pd.DataFrame, source: str) -> pd.DataFrame:
    """The rows generated by ``target``, matched exactly, or a clear exit.

    Exactly — no case folding, no prefix, no fallback to the file's only
    generator. This comparison names a model in the title of every figure it
    writes, and there is no spelling of "nearly the label you asked for" that is
    worth putting there.
    """
    rows = df[df[GROUP_COL].astype(str) == target]
    if not rows.empty:
        return rows

    available = _generators(df)
    deprecated = [g for g in available if not GENERATOR_LABEL_RE.match(g)]
    if deprecated:
        # Ghostbuster-era labels: the corpus predates the convention, so no
        # target in the accepted format can ever match it.
        hint = (
            f" {deprecated} predate the <model_slug>:<prompt_id> convention — "
            "relabel the corpus with epai/scripts/rename_column_values.py "
            "(--old/--new) and analyse it under the label it deserves."
        )
    elif available:
        hint = f" This corpus was generated by {available}."
    else:
        hint = (
            f" This corpus has no AI rows at all: every '{GROUP_COL}' value is "
            f"'{HUMAN_GROUP}'."
        )
    raise SystemExit(f"generator {target!r} not found in {source}.{hint}")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _score_columns(
    df: pd.DataFrame, metric: Metric, explicit: list[str] | None, path: Path
) -> list[str]:
    """The metric's score columns in ``df``, or the explicitly requested ones."""
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise SystemExit(
                f"{path.name}: --score-cols {missing} not in the file "
                f"(it has: {sorted(c for c in df.columns if c.startswith(metric.prefix))})"
            )
        return list(explicit)

    cols = sorted(c for c in df.columns if c.startswith(metric.prefix))
    if not cols:
        raise SystemExit(
            f"No '{metric.prefix}*' columns found in {path.name} for "
            f"--metric {metric.key}."
        )
    return cols


def _usable_columns(
    cols: list[str],
    human: pd.DataFrame,
    tgt: pd.DataFrame,
    path: Path,
    explicit: bool,
) -> list[str]:
    """Keep only score columns that carry numbers on *both* sides.

    A parquet can hold a judge column that was never filled in for this run (two
    judges configured, one of them scored), and a single all-null column would
    otherwise invalidate every pair and leave an empty plot with no explanation.
    """
    keep = [
        c for c in cols
        if pd.to_numeric(human[c], errors="coerce").notna().any()
        and pd.to_numeric(tgt[c], errors="coerce").notna().any()
    ]
    dropped = [c for c in cols if c not in keep]
    if not keep:
        raise SystemExit(
            f"{path.name}: none of {cols} has scores for both 'human' and the "
            "target generator — was this parquet scored for both?"
        )
    if dropped:
        if explicit:
            raise SystemExit(
                f"{path.name}: --score-cols {dropped} has no scores on one of "
                "the two sides."
            )
        print(f"  [note] {path.name}: ignoring {dropped} (unscored on one side)")
    return keep


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce the given columns to floats; non-numeric / null become NaN."""
    return df[cols].apply(pd.to_numeric, errors="coerce")


class PairResult:
    """Human-vs-<target> comparison for one parquet file."""

    def __init__(
        self, parquet: str, stem: str, target: str, cols: list[str], metric: Metric
    ) -> None:
        self.parquet = parquet  # corpus name (domain, or the stem), for labels
        self.stem = stem  # file the corpus was cut from, for the run directory
        self.target = target
        self.cols = cols
        self.metric = metric
        self.n_pairs = 0  # ids present for both human and target
        self.n_only_human = 0  # ids with a human row but no target row
        self.n_only_target = 0  # ids with a target row but no human row
        self.distances = np.empty(0)  # Euclidean / |human-g| — magnitude only
        self.signed_diffs = np.empty(0)  # human - g (mean over cols) — keeps sign
        self.human_scores = np.empty(0)  # human score per valid pair (mean over cols)
        self.target_scores = np.empty(0)  # target score per valid pair (mean over cols)
        self.mse = float("nan")
        self.skipped: list[dict] = []  # id + raw values of skipped pairs

        # --- classification (human vs AI, positive = AI, cut in metric direction) ---
        self.has_clf = False
        self.roc_fpr = np.empty(0)
        self.roc_tpr = np.empty(0)
        self.roc_auc = float("nan")
        self.pr_precision = np.empty(0)
        self.pr_recall = np.empty(0)
        self.avg_precision = float("nan")
        self.clf_threshold = float("nan")  # score cut, read in the metric's direction
        self.accuracy = float("nan")
        self.f1 = float("nan")
        self.cm = np.zeros((2, 2), int)  # [[TN, FP], [FN, TP]] over [human, AI]

    @property
    def corpus(self) -> str:
        """File and slice — ``ghostbuster_gpt35t/essay``, or just the stem.

        What the summary calls this comparison. The corpus name alone is the
        domain (``essay``), which says nothing about which of the three
        ghostbuster corpora it came from.
        """
        return self.parquet if self.parquet == self.stem else f"{self.stem}/{self.parquet}"

    @property
    def label(self) -> str:
        return f"{self.parquet}\nvs {self.target}"

    @property
    def n_valid(self) -> int:
        return int(self.distances.size)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped)


def compute_for_spec(
    spec: CorpusSpec,
    target: str,
    metric: Metric,
    score_cols: list[str] | None,
    out: Tee,
) -> PairResult:
    """Load one corpus and compute human-vs-``target`` per-id differences.

    The lookup is per corpus, not per file: ``--domains`` slices one parquet
    into several corpora, and a generator can be missing from one domain of a
    file that holds it in the others.
    """
    path = spec.path
    if not path.exists():
        raise SystemExit(f"Missing parquet: {path}")

    df = read_corpus(spec, quiet=True).df
    for col in (ID_COL, GROUP_COL):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' missing from {path.name}.")
    cols = _score_columns(df, metric, score_cols, path)

    # Compared as strings, like every other value test against a corpus column
    # (see read_corpus) — a categorical `generated_by` matches the same way.
    human = df[df[GROUP_COL].astype(str) == HUMAN_GROUP]
    if human.empty:
        raise SystemExit(f"No '{HUMAN_GROUP}' rows found in {path.name}.")
    tgt = find_target(target, df, spec.source)

    cols = _usable_columns(cols, human, tgt, path, bool(score_cols))

    # id must be unique within each generator subset to pair unambiguously.
    for grp_label, sub in ((HUMAN_GROUP, human), (target, tgt)):
        dup = sub.duplicated(subset=[ID_COL]).sum()
        if dup:
            raise SystemExit(
                f"[{path.name}] id is not unique within '{grp_label}' rows "
                f"({dup} duplicates) — cannot pair unambiguously."
            )

    left = pd.concat(
        [human[[ID_COL]], _to_numeric(human, cols).add_suffix("_h")], axis=1
    )
    right = pd.concat(
        [tgt[[ID_COL]], _to_numeric(tgt, cols).add_suffix("_t")], axis=1
    )
    merged = left.merge(right, on=ID_COL, how="outer", indicator=True)

    res = PairResult(spec.name, path.stem, target, cols, metric)
    res.n_only_human = int((merged["_merge"] == "left_only").sum())
    res.n_only_target = int((merged["_merge"] == "right_only").sum())
    paired = merged[merged["_merge"] == "both"].reset_index(drop=True)
    res.n_pairs = len(paired)

    a = paired[[f"{c}_h" for c in cols]].to_numpy(float)  # human
    b = paired[[f"{c}_t" for c in cols]].to_numpy(float)  # target

    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    for i in np.nonzero(~valid)[0]:
        res.skipped.append(
            {
                "id": paired.loc[i, ID_COL],
                "human": a[i].tolist(),
                target: b[i].tolist(),
            }
        )

    av, bv = a[valid], b[valid]
    diff = av - bv  # (n_valid, k)
    res.distances = np.sqrt((diff**2).sum(axis=1))  # Euclidean; |human-g| for k=1
    res.signed_diffs = diff.mean(axis=1)  # signed; == human - g for k=1
    res.human_scores = av.mean(axis=1)  # == human score for k=1
    res.target_scores = bv.mean(axis=1)  # == target score for k=1
    res.mse = float((diff**2).mean()) if diff.size else float("nan")

    compute_classification(res)
    return res


def compute_classification(res: PairResult) -> None:
    """Score human-vs-AI classification from the paired detector scores.

    Positive class = AI (the target generator). sklearn wants a decision score
    that is *higher* for the positive class, so a metric where a low score means
    AI (Binoculars) is fed as ``-score`` and its threshold flipped back at the
    end; a metric where a high score means AI (the LLM judge) goes in as is. The
    operating threshold is Youden's J optimum on the ROC curve.
    """
    if res.human_scores.size == 0 or res.target_scores.size == 0:
        return

    scores = np.concatenate([res.human_scores, res.target_scores])
    y = np.concatenate(
        [np.zeros(res.human_scores.size, int), np.ones(res.target_scores.size, int)]
    )
    sign = 1.0 if res.metric.ai_is_high else -1.0
    decision = sign * scores  # higher ⇒ more AI-like

    res.roc_fpr, res.roc_tpr, thr = roc_curve(y, decision)
    res.roc_auc = float(roc_auc_score(y, decision))
    res.pr_precision, res.pr_recall, _ = precision_recall_curve(y, decision)
    res.avg_precision = float(average_precision_score(y, decision))

    # Youden's J optimal operating point → threshold on the raw score.
    j = res.roc_tpr - res.roc_fpr
    best_decision_thr = thr[int(np.argmax(j))]
    res.clf_threshold = float(sign * best_decision_thr)
    y_pred = (
        (scores >= res.clf_threshold) if res.metric.ai_is_high
        else (scores <= res.clf_threshold)
    ).astype(int)
    res.accuracy = float((y_pred == y).mean())
    res.f1 = float(f1_score(y, y_pred, zero_division=0))
    res.cm = confusion_matrix(y, y_pred, labels=[0, 1])
    res.has_clf = True


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _violin_figure(
    usable: list[PairResult],
    data: list[np.ndarray],
    *,
    ylabel: str,
    title: str,
    annotate,
    zero_line: bool,
    fname: str,
    fig_dir: Path,
) -> None:
    """Draw one violin per parquet and save it.

    ``annotate(r)`` returns the short text placed above each violin.
    """
    positions = list(range(1, len(usable) + 1))
    fig, ax = plt.subplots(figsize=(max(6, 2.4 * len(usable) + 2), 6))
    parts = ax.violinplot(
        data, positions=positions, showmeans=True, showextrema=True, widths=0.8
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.6)
    if "cmeans" in parts:
        parts["cmeans"].set_color("#d1495b")

    if zero_line:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    labels = []
    for x, d, r in zip(positions, data, usable):
        top = float(d.max())
        pad = 0.01 * (abs(top) or 1) + 0.005
        ax.text(
            x, top + pad, annotate(r),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
        labels.append(
            f"{r.label}\nn={r.n_valid}"
            + (f"  (skipped {r.n_skipped})" if r.n_skipped else "")
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.margins(y=0.12)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  [OK] {fname}")


def plot_violins(results: list[PairResult], fig_dir: Path) -> None:
    usable = [r for r in results if r.n_valid > 0]
    if not usable:
        print("  [skip] no valid pairs for any parquet — nothing to plot")
        return
    label = usable[0].metric.label

    # Figure 1 — magnitude of human/AI score gap, annotated with MSE.
    _violin_figure(
        usable,
        [r.distances for r in usable],
        ylabel="Pairwise distance  |human − AI|",
        title=f"Human vs AI {label} distance (magnitude) per parquet",
        annotate=lambda r: f"MSE={r.mse:.4g}",
        zero_line=False,
        fname="human_vs_ai_distance_violin.png",
        fig_dir=fig_dir,
    )

    # Figure 2 — signed difference, to see which side scores higher.
    _violin_figure(
        usable,
        [r.signed_diffs for r in usable],
        ylabel="Signed difference  (human − AI)",
        title=f"Human vs AI {label} signed difference per parquet  (>0 ⇒ human larger)",
        annotate=lambda r: f"mean={r.signed_diffs.mean():+.4g}",
        zero_line=True,
        fname="human_vs_ai_signed_diff_violin.png",
        fig_dir=fig_dir,
    )


def plot_distributions(results: list[PairResult], fig_dir: Path) -> None:
    """Per parquet, overlay the human and AI score histograms.

    The two histograms share one set of bins so the overlap is readable — that
    overlap *is* the detector's error region, and how far apart the two humps sit
    is the same information the ROC curve reports as AUC.
    """
    usable = [r for r in results if r.n_valid > 0]
    if not usable:
        print("  [skip] no valid pairs for any parquet — no distributions to plot")
        return
    metric = usable[0].metric

    ncols = min(3, len(usable))
    nrows = int(np.ceil(len(usable) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False
    )
    flat = axes.flatten()

    for ax, r in zip(flat, usable):
        both = np.concatenate([r.human_scores, r.target_scores])
        if metric.hist_range is not None:
            # A bounded score (a probability): always show the whole range, so
            # panels stay comparable and a pile-up at 0 or 1 stays visible.
            lo, hi = metric.hist_range
        else:
            lo, hi = np.percentile(both, [1, 99])
            if lo == hi:  # degenerate range guard
                lo, hi = both.min(), both.max() or 1.0
        bins = np.linspace(lo, hi, 40)
        ax.hist(
            r.human_scores, bins=bins, color="#4c72b0", alpha=0.6, density=True,
            label=f"human (n={r.human_scores.size})",
        )
        ax.hist(
            r.target_scores, bins=bins, color="#d1495b", alpha=0.6, density=True,
            label=f"{r.target} (n={r.target_scores.size})",
        )
        ax.axvline(r.human_scores.mean(), color="#4c72b0", linestyle="--", linewidth=1.2)
        ax.axvline(r.target_scores.mean(), color="#d1495b", linestyle="--", linewidth=1.2)
        if r.has_clf:
            ax.axvline(
                r.clf_threshold, color="black", linestyle=":", linewidth=1.2,
                label=f"threshold={r.clf_threshold:.3g}",
            )
        title = f"{r.parquet}\nhuman vs {r.target}"
        if r.has_clf:
            title += f"   AUC={r.roc_auc:.3f}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(metric.label)
        ax.set_ylabel("Density")
        if metric.hist_range is not None:
            ax.set_xlim(*metric.hist_range)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    for ax in flat[len(usable):]:  # hide unused axes
        ax.axis("off")

    fig.suptitle(f"Human vs AI {metric.label} distributions", fontsize=13)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fname = "human_vs_ai_score_distributions.png"
    fig.savefig(fig_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  [OK] {fname}")


def plot_roc(results: list[PairResult], fig_dir: Path) -> None:
    """Overlay the human-vs-AI ROC curve of every parquet, AUC in the legend."""
    usable = [r for r in results if r.has_clf]
    if not usable:
        print("  [skip] no classifiable parquet — no ROC to plot")
        return
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, r in enumerate(usable):
        ax.plot(
            r.roc_fpr, r.roc_tpr, color=colors[i % len(colors)], linewidth=2,
            label=f"{r.parquet} vs {r.target}  (AUC={r.roc_auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="chance (AUC=0.5)")
    ax.set_xlabel("False positive rate  (human classified as AI)")
    ax.set_ylabel("True positive rate  (AI detected)")
    ax.set_title(f"Human vs AI detection ROC ({usable[0].metric.label})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "human_vs_ai_roc.png", dpi=150)
    plt.close(fig)
    print("  [OK] human_vs_ai_roc.png")


def plot_pr(results: list[PairResult], fig_dir: Path) -> None:
    """Overlay the human-vs-AI precision–recall curve of every parquet, AP in legend."""
    usable = [r for r in results if r.has_clf]
    if not usable:
        print("  [skip] no classifiable parquet — no PR curve to plot")
        return
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, r in enumerate(usable):
        ax.plot(
            r.pr_recall, r.pr_precision, color=colors[i % len(colors)], linewidth=2,
            label=f"{r.parquet} vs {r.target}  (AP={r.avg_precision:.3f})",
        )
    # Classes are balanced (paired), so chance precision = 0.5.
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1, alpha=0.6, label="chance (AP=0.5)")
    ax.set_xlabel("Recall  (AI detected)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Human vs AI detection precision–recall ({usable[0].metric.label})")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "human_vs_ai_pr.png", dpi=150)
    plt.close(fig)
    print("  [OK] human_vs_ai_pr.png")


def plot_confusion(results: list[PairResult], fig_dir: Path) -> None:
    """Per parquet, the 2×2 confusion matrix at its Youden-optimal threshold."""
    usable = [r for r in results if r.has_clf]
    if not usable:
        print("  [skip] no classifiable parquet — no confusion matrices to plot")
        return

    ncols = min(3, len(usable))
    nrows = int(np.ceil(len(usable) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows), squeeze=False
    )
    flat = axes.flatten()
    ticks = ["human", "AI"]

    for ax, r in zip(flat, usable):
        cm = r.cm
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        thresh = cm.max() / 2 if cm.max() else 0
        for row in range(2):
            for col in range(2):
                ax.text(
                    col, row, f"{cm[row, col]}", ha="center", va="center",
                    fontsize=12,
                    color="white" if cm[row, col] > thresh else "black",
                )
        ax.set_xticks([0, 1], labels=ticks)
        ax.set_yticks([0, 1], labels=ticks)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        cmp_sign = "≥" if r.metric.ai_is_high else "≤"
        ax.set_title(
            f"{r.parquet} vs {r.target}\n"
            f"score{cmp_sign}{r.clf_threshold:.3f}⇒AI  acc={r.accuracy:.3f}  F1={r.f1:.3f}",
            fontsize=9,
        )

    for ax in flat[len(usable):]:  # hide unused axes
        ax.axis("off")

    fig.suptitle("Human vs AI confusion matrices (Youden-optimal threshold)", fontsize=13)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "human_vs_ai_confusion.png", dpi=150)
    plt.close(fig)
    print("  [OK] human_vs_ai_confusion.png")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _dist_stats(d: np.ndarray) -> dict:
    if d.size == 0:
        return {k: float("nan") for k in ("mean", "std", "min", "median", "max")}
    return {
        "mean": float(d.mean()),
        "std": float(d.std()),
        "min": float(d.min()),
        "median": float(np.median(d)),
        "max": float(d.max()),
    }


def report(results: list[PairResult], fig_dir: Path, out: Tee) -> None:
    metric = results[0].metric
    out.print("=" * 70)
    out.print(f"Human vs AI {metric.label} comparison (per parquet)")
    out.print(f"metric: {metric.key}  ({metric.rule})")
    out.print(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    out.print("=" * 70)

    rows = []
    for r in results:
        s = _dist_stats(r.distances)
        out.print(f"\n[{r.corpus}: human vs {r.target}]  columns: {r.cols}")
        out.print(
            f"  matched pairs={r.n_pairs}  valid={r.n_valid}  skipped={r.n_skipped}"
            f"  (unmatched: only human={r.n_only_human}, only {r.target}={r.n_only_target})"
        )
        out.print(
            f"  distance  mean={s['mean']:.4g}  std={s['std']:.4g}  "
            f"min={s['min']:.4g}  median={s['median']:.4g}  max={s['max']:.4g}"
        )
        mean_signed = float(r.signed_diffs.mean()) if r.n_valid else float("nan")
        larger = (
            "human larger" if mean_signed > 0
            else f"{r.target} larger" if mean_signed < 0
            else "equal"
        )
        out.print(f"  mean signed diff (human - {r.target})={mean_signed:+.4g}  → {larger}")
        out.print(f"  MSE={r.mse:.6g}")
        tn, fp, fn, tp = r.cm.ravel()
        out.print(
            f"  classification (AI positive): ROC-AUC={r.roc_auc:.3f}  "
            f"AP={r.avg_precision:.3f}  acc={r.accuracy:.3f}  F1={r.f1:.3f}"
        )
        out.print(
            f"    threshold: score{'>=' if r.metric.ai_is_high else '<='}"
            f"{r.clf_threshold:.4g} ⇒ AI   "
            f"confusion  TP={tp} FP={fp} FN={fn} TN={tn}"
        )
        if r.n_valid:
            out.print(
                f"  mean score  human={r.human_scores.mean():.4g}  "
                f"{r.target}={r.target_scores.mean():.4g}"
            )
        rows.append(
            {
                "file": r.stem,  # which parquet; 'parquet' below is the slice
                "parquet": r.parquet,
                "metric": r.metric.key,
                "target": r.target,
                "columns": ",".join(r.cols),
                "n_pairs": r.n_pairs,
                "n_valid": r.n_valid,
                "n_skipped": r.n_skipped,
                "n_only_human": r.n_only_human,
                "n_only_target": r.n_only_target,
                "dist_mean": s["mean"],
                "dist_std": s["std"],
                "dist_min": s["min"],
                "dist_median": s["median"],
                "dist_max": s["max"],
                "mean_signed_diff": mean_signed,
                "mean_human": float(r.human_scores.mean()) if r.n_valid else float("nan"),
                "mean_target": float(r.target_scores.mean()) if r.n_valid else float("nan"),
                "mse": r.mse,
                "roc_auc": r.roc_auc,
                "avg_precision": r.avg_precision,
                "clf_threshold": r.clf_threshold,
                "accuracy": r.accuracy,
                "f1": r.f1,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }
        )

    # --- skipped-pair detail ---
    out.print("\n" + "-" * 70)
    out.print("Skipped pairs (element null / non-numeric in human or target):")
    total_skipped = sum(r.n_skipped for r in results)
    if total_skipped == 0:
        out.print("  (none — every pair had numeric scores on both sides)")
    else:
        for r in results:
            if not r.skipped:
                continue
            out.print(f"\n  [{r.corpus}: human vs {r.target}] {r.n_skipped} skipped:")
            # An unscored generator skips *every* pair (normal for llm_judge),
            # which would otherwise dump thousands of identical lines.
            for sk in r.skipped[:SKIPPED_DETAIL_LIMIT]:
                out.print(
                    f"    id={sk['id']}  human={sk['human']}  "
                    f"{r.target}={sk[r.target]}"
                )
            if r.n_skipped > SKIPPED_DETAIL_LIMIT:
                out.print(
                    f"    ... and {r.n_skipped - SKIPPED_DETAIL_LIMIT} more "
                    "(listing capped)"
                )
        out.print(f"\n  TOTAL skipped across parquets: {total_skipped}")

    out.print("\n" + "=" * 70)

    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_path = fig_dir / "human_vs_ai_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  [OK] {csv_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_dir_name(results: list[PairResult], domains: list[str] | None) -> str:
    """The directory this run writes to, named after everything it compared.

    Every corpus of the run contributes ``<file stem>+<generator>``, deduped,
    in the order given — the two things that decide what the figures show. The
    name has to carry BOTH: the file alone collides as soon as one corpus is
    analysed against two of its generators, and the generator alone collides
    when the same model wrote the AI half of two corpora (``gpt56luna:0701A``
    is in ghostbuster_gpt56luna, ``gpt56luna:0807A`` in science_v1).

    It also cannot be named after the CORPORA, which is what it used to be: a
    combined parquet is split per domain, so three separate runs over
    ghostbuster_gpt35t / _gpt54mini / _gpt56luna each produced corpora called
    essay, reuter and wp, and all three wrote over one ``essay__reuter__wp``
    directory — the figures of the last run standing, under a name that claimed
    nothing about which model they were about.

    ``--domains`` is appended when it was given, since a run over one domain is
    a different comparison from the same file's full run, and would otherwise
    land on top of it. The generator is spelled with '_' rather than ':' — the
    directory spelling of a label, per epai/utils/label_utils.py.
    """
    parts = list(dict.fromkeys(
        f"{r.stem}+{dir_name_from_label(r.target)}" for r in results
    ))
    name = "__".join(parts)
    if domains:
        name += "+" + "-".join(dict.fromkeys(domains))
    return name


def _parse_spec(spec: str) -> tuple[Path, str]:
    """Split a ``PATH:TARGET`` spec into the file and the generator to compare.

    There is no default target. Which generator a corpus holds is a property of
    that corpus — ``gpt35t:0000A``, ``gpt54mini:0701A``, ``gpt56luna:0807A`` are
    all in ``data/parquet`` right now — so any name defaulted here would be
    wrong for most files it was applied to, and wrong in the quietest possible
    way: a comparison against a generator the corpus does not contain.

    The split is on the FIRST ':', so everything after it is the target — the
    label format contains a colon of its own (``gpt56luna:0807A``, model plus
    prompt id), and splitting on the last ':' would move the prompt id out of
    the target and into the path.

    A leading Windows drive letter (``C:\\corpora\\x.parquet``) is not a target
    separator and is skipped before the split.
    """
    head, sep, tail = spec.partition(":")
    if sep and len(head) == 1 and head.isalpha() and tail[:1] in ("\\", "/"):
        drive, head = head, ""
        head, sep, tail = tail.partition(":")
        head = f"{drive}:{head}"
    if not (sep and head and tail):
        raise SystemExit(
            f"{spec!r} names no generator. A spec is 'PATH:TARGET', where TARGET "
            "is the <model_slug>:<prompt_id> label to compare against human — "
            "e.g. data/parquet/ghostbuster_gpt35t.parquet:gpt35t:0000A"
        )
    return Path(head), check_target(tail, spec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "specs",
        nargs="+",
        help="One or more parquet specs 'PATH:TARGET', one violin per (file, "
        "domain). TARGET is required and is the generated_by label to compare "
        "against human, in the one accepted format <model_slug>:<prompt_id> — "
        "e.g. data/parquet/ghostbuster_gpt35t.parquet:gpt35t:0000A.",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        metavar="DOMAIN",
        help="Domains to take from each parquet spec — one violin per (file, "
        "domain). Default: every domain a combined file holds; a file with no "
        "domain column becomes a single violin.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=sorted(METRICS),
        help="Which detector score to analyse (default: %(default)s). 'binox' "
        "reads binoculars_* (low ⇒ AI), 'llm_judge' reads llm_judge_score_* "
        "(also low ⇒ AI: the judge reports P(human)).",
    )
    parser.add_argument(
        "--score-cols",
        nargs="+",
        default=None,
        metavar="COL",
        help="Use exactly these score columns instead of every column matching "
        "the metric's prefix — e.g. one judge model when the parquet holds two.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [_parse_spec(s) for s in args.specs]
    metric = METRICS[args.metric]

    out = Tee()
    results: list[PairResult] = []
    for path, target in specs:
        for corpus in specs_from_paths([path], args.domains, base_dir=BASE_DIR):
            print(f"Comparing human vs '{target}' in {corpus.source} "
                  f"({metric.key}) ...")
            results.append(
                compute_for_spec(corpus, target, metric, args.score_cols, out)
            )

    fig_dir = metric.results_dir / run_dir_name(results, args.domains)
    print()
    plot_violins(results, fig_dir)
    plot_distributions(results, fig_dir)
    plot_roc(results, fig_dir)
    plot_pr(results, fig_dir)
    plot_confusion(results, fig_dir)
    report(results, fig_dir, out)

    out.save(fig_dir / "human_vs_ai_summary.log")
    print(f"\nAll outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()

r"""Turn the judge's cue citations into a design-matrix column.

``add_llm_judge_reasons.py`` writes what the judge *saw*: two lists per row,
drawn from a closed vocabulary of eleven cues (five human-side, six
machine-side). Those lists are read today by ``plot_llm_judge_reasons.py``,
which counts them -- "``even_rhythm`` cited on 78% of machine rows and 12% of
human rows". A count is a description. This script asks the next question:

    Is that description enough to CLASSIFY on?

It answers it by making the cue lists trainable. Each row's two lists become one
fixed-width multi-hot vector, written as an ordinary feature column beside
``emb_luar_crud``, so the classifier configs can train on it with no new code
path -- one line in ``feature_sets`` and the same logistic regression, the same
source x target matrix, the same figures.

WHAT THE VECTOR IS
------------------
Eleven dimensions, in the order the prompt lists them (never sorted by value:
the column has to mean the same thing in every parquet it is written to)::

    0..4    human cues     specifics commitment asymmetry idiom errors
    5..10   machine cues   generic_vocabulary even_rhythm scaffolding hedging
                           coverage assistant_register

1.0 if the judge cited that cue for that row, 0.0 if it did not. With
``--with-other`` a twelfth dimension is appended, 1.0 when the row's free-text
escape hatch held something the vocabulary had no word for -- not a cue, a flag
that the judge went outside the vocabulary at all.

Counts (how many human cues, how many machine cues, their difference) are
deliberately NOT extra dimensions. Every one of them is a fixed linear
combination of the eleven, so a linear model can already form them and adding
them buys nothing but collinearity.

WHY MULTI-HOT AND NOT THE SCORE
-------------------------------
``llm_judge_score_*`` is already in the parquet and already a baseline row in
every config. This column is not a cheaper version of it -- it is the judge's
evidence with its verdict removed. The prompt that produced the cues is under
explicit instruction not to score (see EXPLAIN_BLOCK in the reasons script), so
what a classifier fitted on this column learns is how the eleven cues combine
into a decision, which is exactly the thing the judge's own number hides. Two
outcomes are worth the run:

    PREDICTION  eleven bits will not reach the embedding's in-domain accuracy;
                nothing with eleven bits could. The number to watch is TRANSFER.
                The embeddings score ~98% in-domain and below chance across
                domains, which is the finding this project is stuck on -- and a
                cue vector cannot encode a punctuation habit or a domain's house
                style, so if it transfers FLAT where the embedding collapses,
                the eleven bits are carrying something the 512-dim vector is not.
                A flat 70% beats a 98% that inverts.

    AND IF IT DOES NOT  a cue vector that transfers as badly as the embedding
                says the judge's evidence is domain-bound too, which is a real
                result about the prompt and not about the classifier.

COVERAGE IS THE CATCH, AND IT IS NOT SMALL
------------------------------------------
The reasons script explains a SAMPLE of rows by default (three bands, 100 rows
each) because a rationale costs output tokens on every row. A row nobody asked
about has no cue lists, gets ``None`` here, and is then dropped by the training
loader -- group-aware, so its partner rows go with it. That drop is silent in
the config and loud in this script's report, which is the reason the report
exists: run this, read the coverage line, and only then decide whether that
parquet can carry a cue variant at all.

As of this writing only ``ghostbuster_gpt56luna`` is explained end to end
(5994 of 6000 rows). The other three corpora hold ~350-400 explained rows each,
which will not survive the group-aware drop, and a cue feature set pointed at
them aborts the preflight rather than quietly training on the remnant. Explain
them first if you want the four-way comparison:

    uv run python epai/ai_detection/score/add_llm_judge_reasons.py \
        data/parquet/ghostbuster_gpt35t.parquet --bands all

WHAT IS NOT MISSING
-------------------
A row the judge DID look at and found no cues in is an all-zeros vector, not a
gap. That distinction is the whole reliability of the column: ``is_missing`` in
``train_utils`` treats an empty vector as absent and an all-zero one as present,
so "explained, nothing found" stays in the training set carrying its real
information (this document showed the judge nothing) instead of vanishing into
the same bucket as "never asked". Coverage is decided by the rationale column,
which the judge always fills, never by the cue lists, which it legitimately may
not.

Usage:
    # the fully-explained corpus, written in place
    uv run python epai/ai_detection/score/add_cue_features.py \
        data/parquet/ghostbuster_gpt56luna.parquet

    # look first: coverage, per-cue rates and the drop it would cause
    uv run python epai/ai_detection/score/add_cue_features.py \
        data/parquet/ghostbuster_gpt56luna.parquet --dry-run

    # every corpus in one go (each written in place)
    uv run python epai/ai_detection/score/add_cue_features.py \
        data/parquet/ghostbuster_gpt35t.parquet \
        data/parquet/ghostbuster_gpt54mini.parquet \
        data/parquet/ghostbuster_gpt56luna.parquet \
        data/parquet/science_v3_gpt56luna_0811A.parquet

    # a twelfth dimension for the free-text escape hatch
    uv run python epai/ai_detection/score/add_cue_features.py \
        data/parquet/ghostbuster_gpt56luna.parquet --with-other

    # one judge of a file explained by several, to a new file
    uv run python epai/ai_detection/score/add_cue_features.py \
        data/parquet/ghostbuster_gpt56luna.parquet \
        --judge gpt56luna_v3 -o data/parquet/gpt56luna_cues.parquet

    # rebuild a column that is already there
    uv run python epai/ai_detection/score/add_cue_features.py \
        data/parquet/ghostbuster_gpt56luna.parquet --overwrite

Then train on it -- the column name the run prints goes straight into a config::

    feature_sets:
      - emb_luar_crud
      - llm_judge_cuevec_gpt56luna_v3
      - [emb_luar_crud, llm_judge_cuevec_gpt56luna_v3]

Report: ``results/llm_judge_cue_features/<parquet stem>+<judge stem>/``,
holding ``report.txt`` (the transcript) and ``report.json`` (the measurements).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The vocabulary, the column spellings and the cell parsers all come from the
# script that wrote the data -- restating any of them here would be a second
# spelling of one thing, and a prompt version that grows a cue would then reach
# the parquet without reaching the design matrix.
from epai.ai_detection.score.add_llm_judge_reasons import (
    HUMAN_CUES,
    HUMAN_CUES_PREFIX,
    MACHINE_CUES,
    MACHINE_CUES_PREFIX,
    OTHER_PREFIX,
    REASON_PREFIX,
    Tee,
    _as_cue_list,
    _as_other,
    _as_reason,
)
from epai.ai_detection.score.add_llm_judge_scores import (
    DOMAIN_COL,
    GENERATED_BY_COL,
)
from epai.utils.parquet_utils import load_parquet, save_parquet

BASE_DIR = Path(__file__).parent.parent.parent.parent
REPORT_DIR = BASE_DIR / "results" / "llm_judge_cue_features"

# The column this script writes. Under ``llm_judge_`` so truncate_text.py drops
# it with the rest of the judge's output when a corpus is re-cut (these are
# judgements of the untruncated text), and deliberately NOT under
# ``llm_judge_score_``, which cd_core collects as zero-shot baselines -- a
# feature column swept as if it were a threshold would be nonsense.
# ``cuevec`` rather than ``cues`` so a glob for the list columns the reasons
# script writes cannot pick this one up as a third list.
CUE_FEATURE_PREFIX = "llm_judge_cuevec_"

HUMAN_CLASS = "human"

# Dimension order, fixed here and nowhere else. Human side first, machine side
# second, each in the order its prompt lists it. Sorting by anything -- name,
# frequency, lift -- would make dimension 3 mean a different cue in every
# parquet, and a model trained on one file would be silently mis-indexed on the
# next.
CUE_DIMS: list[tuple[str, str]] = (
    [(HUMAN_CLASS, cue) for cue in HUMAN_CUES]
    + [("machine", cue) for cue in MACHINE_CUES]
)
OTHER_DIM = ("other", "off_vocabulary")


def feature_column(stem: str) -> str:
    """The column one judge stem's cue vector is written to."""
    return f"{CUE_FEATURE_PREFIX}{stem}"


def dim_names(with_other: bool) -> list[str]:
    """The vector's dimensions, in index order, as ``side:cue`` strings."""
    dims = [f"{side}:{cue}" for side, cue in CUE_DIMS]
    if with_other:
        dims.append(f"{OTHER_DIM[0]}:{OTHER_DIM[1]}")
    return dims


def judge_stems(df: pd.DataFrame) -> list[str]:
    """Judge stems the file has rationale columns for, in column order.

    Same rule as ``plot_llm_judge_reasons.judge_stems``, restated rather than
    imported: that module pulls in matplotlib for one two-line function.
    """
    return [c[len(REASON_PREFIX):] for c in df.columns
            if c.startswith(REASON_PREFIX)]


def resolve_stem(df: pd.DataFrame, judge: str | None, path: Path) -> str | None:
    """The one judge stem to build from, or None with a reason printed."""
    stems = judge_stems(df)
    if not stems:
        print(f"  [skip] {path.name}: no {REASON_PREFIX}* column — run "
              f"add_llm_judge_reasons.py on it first")
        return None
    if judge is not None:
        if judge not in stems:
            print(f"  [skip] {path.name}: no rationales from judge {judge!r} "
                  f"(it has: {stems})")
            return None
        return judge
    if len(stems) > 1:
        print(f"  [skip] {path.name}: explained by several judges {stems}; "
              f"name one with --judge")
        return None
    return stems[0]


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------


def build_vectors(
    df: pd.DataFrame, stem: str, with_other: bool
) -> tuple[list, np.ndarray]:
    """One multi-hot vector per row, and the mask of rows that got one.

    A row is covered when the judge answered for it at all, which is what the
    rationale column records. Cue lists are NOT the test: an explained row with
    two empty lists is a real observation (the judge found nothing) and must
    reach the design matrix as zeros, while a row nobody asked about must reach
    it as ``None`` so the loader drops it and its whole group with it.
    """
    reason_col = f"{REASON_PREFIX}{stem}"
    human_col = f"{HUMAN_CUES_PREFIX}{stem}"
    machine_col = f"{MACHINE_CUES_PREFIX}{stem}"
    other_col = f"{OTHER_PREFIX}{stem}"

    reasons = [_as_reason(v) for v in df[reason_col]]
    human_cited = [set(_as_cue_list(v) or []) for v in df[human_col]]
    machine_cited = [set(_as_cue_list(v) or []) for v in df[machine_col]]
    others = (
        [_as_other(v) for v in df[other_col]]
        if other_col in df.columns
        else [None] * len(df)
    )

    # A rationale that failed to come back but whose cue lists did is still an
    # answer about the document, so it counts as covered; the reverse (a
    # rationale with no cues) is the ordinary "found nothing" case.
    covered = np.array([
        reason is not None or bool(h) or bool(m)
        for reason, h, m in zip(reasons, human_cited, machine_cited)
    ])

    width = len(CUE_DIMS) + (1 if with_other else 0)
    values: list = [None] * len(df)
    for i in np.flatnonzero(covered):
        vec = np.zeros(width, dtype=np.float32)
        for j, (side, cue) in enumerate(CUE_DIMS):
            cited = human_cited[i] if side == HUMAN_CLASS else machine_cited[i]
            if cue in cited:
                vec[j] = 1.0
        if with_other and others[i] is not None:
            vec[-1] = 1.0
        values[int(i)] = vec

    return values, covered


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _jsonable(value: float, digits: int) -> float | None:
    """A rounded number, or ``None`` where a class was empty.

    ``json.dumps`` writes a NaN as the bare token ``NaN``, which no strict JSON
    reader accepts -- and report.json exists to be read by something other than
    this script.
    """
    return None if not np.isfinite(value) else round(float(value), digits)


def _class_mask(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(is_human, is_machine) over every row, by ``generated_by``."""
    if GENERATED_BY_COL not in df.columns:
        allrows = np.ones(len(df), dtype=bool)
        return allrows, np.zeros(len(df), dtype=bool)
    gen = df[GENERATED_BY_COL].astype(str).to_numpy()
    return gen == HUMAN_CLASS, gen != HUMAN_CLASS


def _group_loss(df: pd.DataFrame, covered: np.ndarray, group_col: str) -> dict:
    """What the training loader's group-aware drop will actually cost.

    The number a coverage percentage hides. ``load_prepared`` drops a row with a
    missing feature AND every other row sharing its group, so a corpus explained
    one row at a time loses both halves of every pair it touched -- a file can
    read 6% covered and train on 0 rows. Reported before anything is written,
    because it decides whether the column is worth having in this parquet.
    """
    if group_col not in df.columns:
        return {"group_col": None}
    groups = df[group_col].to_numpy()
    bad_groups = set(groups[~covered].tolist())
    survives = np.array([g not in bad_groups for g in groups])
    return {
        "group_col": group_col,
        "n_groups": int(pd.unique(groups).size),
        "n_groups_intact": int(pd.unique(groups[survives]).size) if survives.any() else 0,
        "n_rows_after_group_drop": int(survives.sum()),
    }


def report_file(
    path: Path,
    df: pd.DataFrame,
    stem: str,
    column: str,
    values: list,
    covered: np.ndarray,
    with_other: bool,
    group_col: str,
    out: Tee,
) -> dict:
    """Print, and return, everything that decides whether this column is usable."""
    n = len(df)
    is_human, is_machine = _class_mask(df)
    names = dim_names(with_other)

    out.print("=" * 78)
    out.print(f"{path.name}  +  judge {stem}")
    out.print("=" * 78)
    out.print(f"column          {column}")
    out.print(f"dimensions      {len(names)}  ({len(HUMAN_CUES)} human + "
              f"{len(MACHINE_CUES)} machine"
              + (" + 1 off-vocabulary" if with_other else "") + ")")

    n_cov = int(covered.sum())
    pct = 100.0 * n_cov / n if n else float("nan")
    out.print(f"explained rows  {n_cov} of {n}  ({pct:.1f}%)")
    out.print(f"  human         {int((covered & is_human).sum())} of {int(is_human.sum())}")
    out.print(f"  machine       {int((covered & is_machine).sum())} of {int(is_machine.sum())}")

    loss = _group_loss(df, covered, group_col)
    if loss.get("group_col"):
        out.print(
            f"after the group-aware drop on {group_col!r}: "
            f"{loss['n_rows_after_group_drop']} row(s) usable, "
            f"{loss['n_groups_intact']} of {loss['n_groups']} group(s) intact"
        )
        if loss["n_rows_after_group_drop"] < 50:
            out.print(
                "  ^ below the configs' default min_rows=50 — a feature set naming "
                "this column will ABORT the preflight for this corpus, not train "
                "on the remnant. Explain the rest of the file first "
                "(add_llm_judge_reasons.py --bands all)."
            )

    # Per-dimension rates on the explained rows only: the denominator has to be
    # what the classifier will actually see, or a sampled file reads as a file
    # whose judge never cites anything.
    cov_h, cov_m = covered & is_human, covered & is_machine
    nh, nm = int(cov_h.sum()), int(cov_m.sum())
    stacked = np.zeros((n, len(names)), dtype=np.float32)
    for i in np.flatnonzero(covered):
        stacked[i] = values[int(i)]

    out.print("")
    out.print(f"{'dim':<4}{'name':<28}{'human':>14}{'machine':>14}{'lift':>9}")
    out.print("-" * 78)
    dims: list[dict] = []
    for j, name in enumerate(names):
        h = float(stacked[cov_h, j].mean()) if nh else float("nan")
        m = float(stacked[cov_m, j].mean()) if nm else float("nan")
        lift = 100.0 * (m - h)
        out.print(f"{j:<4}{name:<28}{100*h:>13.0f}%{100*m:>13.0f}%{lift:>+8.0f}%")
        dims.append({
            "index": j, "name": name,
            "human_rate": _jsonable(h, 4), "machine_rate": _jsonable(m, 4),
            "lift_pct": _jsonable(lift, 2),
        })

    # An all-zero explained row is legal but carries nothing, and a column made
    # mostly of them cannot separate anything however good the cues are. Load
    # differing between the classes is worth seeing for the opposite reason:
    # a side the judge writes fewer cues about is being judged on less evidence,
    # which is a separability the classifier can find without any cue mattering.
    load = stacked.sum(axis=1)
    zero = int((load[covered] == 0).sum()) if n_cov else 0
    out.print("")
    if not n_cov:
        out.print("all-zero rows   n/a — nothing explained")
    else:
        out.print(f"all-zero rows   {zero} of {n_cov} explained "
                  f"({100.0 * zero / n_cov:.1f}%)")
        h_load = float(load[cov_h].mean()) if nh else float("nan")
        m_load = float(load[cov_m].mean()) if nm else float("nan")
        out.print(f"cues per row    {float(load[covered].mean()):.2f} mean "
                  f"(human {h_load:.2f}, machine {m_load:.2f})")

    if DOMAIN_COL in df.columns:
        out.print("")
        out.print("explained rows by domain:")
        for domain in sorted(df[DOMAIN_COL].astype(str).unique()):
            in_dom = (df[DOMAIN_COL].astype(str) == domain).to_numpy()
            out.print(f"  {domain:<20}{int((covered & in_dom).sum()):>6} of "
                      f"{int(in_dom.sum()):>6}")

    return {
        "parquet": str(path),
        "judge": stem,
        "column": column,
        "n_rows": n,
        "n_explained": n_cov,
        "explained_pct": _jsonable(pct, 2),
        "n_explained_human": int(cov_h.sum()),
        "n_explained_machine": int(cov_m.sum()),
        "dim_names": names,
        "dims": dims,
        "n_all_zero": zero,
        "group_drop": loss,
    }


def save_report(out: Tee, payload: dict, path: Path, stem: str) -> Path:
    """Transcript and measurements, where the other analyse scripts put theirs."""
    directory = REPORT_DIR / f"{path.stem}+{stem}"
    directory.mkdir(parents=True, exist_ok=True)
    out.save(directory / "report.txt")
    (directory / "report.json").write_text(json.dumps(payload, indent=2) + "\n")
    return directory


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquets", type=Path, nargs="+",
                    help="Parquet(s) to add the cue feature column to.")
    ap.add_argument("--judge", default=None, metavar="STEM",
                    help="Judge stem to build from, e.g. gpt56luna_v3. Required "
                         "only for a file explained by more than one judge.")
    ap.add_argument("--with-other", action="store_true",
                    help="Append a 12th dimension: 1.0 when the free-text "
                         "escape hatch held something off-vocabulary.")
    ap.add_argument("--group-col", default="id",
                    help="Group column the training loader will drop by "
                         "(default: id). Reporting only; nothing is dropped here.")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output path (default: overwrite the input in place). "
                         "Only valid with a single input.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Rebuild the feature column if it is already present.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report only; write no parquet.")
    ap.add_argument("--no-save", action="store_true",
                    help="Do not write the report under results/.")
    args = ap.parse_args()
    if args.out is not None and len(args.parquets) > 1:
        ap.error("-o/--out takes a single input parquet")
    return args


def main() -> None:
    args = parse_args()
    failures: list[str] = []

    for path in args.parquets:
        if not path.is_file():
            print(f"  [skip] {path}: not a file")
            failures.append(str(path))
            continue

        df = load_parquet(path)
        stem = resolve_stem(df, args.judge, path)
        if stem is None:
            failures.append(str(path))
            continue

        column = feature_column(stem)
        if column in df.columns and not args.overwrite:
            print(f"  [skip] {path.name}: {column} already present — "
                  f"pass --overwrite to rebuild it")
            failures.append(str(path))
            continue

        # One transcript per file rather than one per run: the report directory
        # is named after the (parquet, judge) pair, so a re-run replaces its own
        # verdict, and a four-file run leaves four reports that each read on
        # their own instead of four copies of a concatenation.
        out = Tee()
        values, covered = build_vectors(df, stem, args.with_other)
        payload = report_file(
            path, df, stem, column, values, covered,
            args.with_other, args.group_col, out,
        )

        if args.dry_run:
            out.print("")
            out.print("--dry-run: no parquet written.")
        else:
            df[column] = pd.Series(values, index=df.index, dtype="object")
            target = args.out or path
            save_parquet(df, target)
            out.print("")
            out.print(f"wrote {column} → {target}")

        if not args.no_save:
            print(f"Report → {save_report(out, payload, path, stem)}")
        print("")

    if failures:
        print(f"\n{len(failures)} file(s) skipped: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()

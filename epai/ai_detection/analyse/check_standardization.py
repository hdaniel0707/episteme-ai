r"""Is this corpus standardised, and is it standardised the SAME WAY as that one?

``audit_corpus.py`` asks whether a corpus is fit to train on — blank documents,
duplicate uids, null scores, broken pairs. This script asks the question one
step to the side: **is the corpus in the agreed FORM**, and, when more than one
file is named, **do the files agree with each other**. A corpus can be perfectly
fit to train on and still be built to a different convention than the corpus it
is about to be compared against, and that difference lands in the results as if
it were a finding about detectors.

It does not re-check what ``audit_corpus.py`` checks. Form, not fitness: the set
and width of the ``emb_*`` columns rather than the health of the vectors, the
spelling of ``generated_by`` rather than the class balance, whether ``text`` is
a fixpoint of :func:`~epai.utils.text_standardize.standardize_text` rather than
whether it is long enough to be a document. Run both.

WHY THIS IS WORTH A SCRIPT
--------------------------
The corpora in ``data/parquet/`` were built over several months by two pipelines
whose stages were added at different times, and they have drifted:

  * ``text_raw`` exists in the corpora built after ``standardize_text.py`` became
    a stage and not in the ones built before it — so "was this text standardised"
    is currently unanswerable from three of the files.
  * the ``emb_*`` set runs from three columns (``science_v1``) to fifteen
    (``aidetect_01``), and a cross-suite comparison silently reduces to the
    intersection.
  * ``llm_judge_score_gpt54_mini`` is in three files out of eight.
  * ``prompt`` is arrow ``string`` in the science corpora and ``str``/object
    everywhere else, which is the dtype that makes a ``pd.concat`` of two of them
    raise rather than union.

Each of those is invisible in a row count and expensive to discover downstream.

WHAT "STANDARDISED" MEANS HERE — the characteristics checked
------------------------------------------------------------
Five groups, each of which can be checked from the file alone:

  schema     the columns ``combine_domains.py`` promises to write, in the order
             ``order_columns`` puts them, with nothing extra carried along, and
             one arrow type per column name.
  identity   ``uid`` unique (an error -- it is the row key), ``uid ==
             "{id}:{generated_by}"`` (only a warning: nothing parses a uid, and
             a deliberate relabel legitimately leaves it stale), ``domain``
             present, lower-case and aliased (never ``reut``).
  labels     ``generated_by`` is ``human`` or ``<model_slug>:<prompt_id>``
             (:data:`~epai.utils.label_utils.GENERATOR_LABEL_RE`), with none of
             the prompt-variant labels ``combine_domains.py`` drops by default.
  text       ``text`` is a FIXPOINT of ``standardize_text`` — the authoritative
             test, since the stage is idempotent, so a text it would still
             change is a text the stage never saw. Residual typographic
             characters are then broken down per domain and per class, because
             an artifact that is symmetric across the classes is a nuisance and
             one that is not is a label. Blank rows are split by class too: a
             blank MACHINE row is always an error (a failed generation filed as
             machine-written), while blank HUMAN rows are an error only above
             ``--max-blank-human-frac`` of the human class, since the upstream
             release ships a handful that nothing can refill.
  features   which score and ``emb_*`` columns exist and how wide the vectors
             are — the axis a suite comparison is actually run over.

WHITESPACE MODE IS INFERRED, NOT ASSUMED
----------------------------------------
``standardize_text`` has two conventions and the pipelines legitimately use
both: ``lines`` for the ghostbuster corpora (paragraphs are content),
``flat`` for the science ones (an abstract is one paragraph, and
``rules.clean_abstract`` already ends in that collapse). ``flat`` text is also
``lines``-conformant, so the script reports the TIGHTEST mode a file satisfies
and only complains when a file satisfies neither. ``--whitespace`` pins the mode
when you want a file held to one specific convention rather than asked which it
meets.

READING THE OUTPUT
------------------
Every finding is an ERROR (the file is not standardised, or two files are not
comparable) or a WARN (a difference worth knowing that does not by itself
invalidate a comparison). The per-file verdict is STANDARDISED / NOT
STANDARDISED; with two or more files there is a closing comparability section
whose verdict is COMPARABLE / NOT COMPARABLE, and that one is the point of
naming several files at once.

Nothing is modified. Every run is SAVED as well as printed -- ``report.txt``,
the transcript exactly as it appeared with the colours stripped, and
``report.json``, every measurement behind it -- under
``results/standardization/<inputs>/``. The directory is a function of the input
file names, so re-running after a fix replaces that verdict rather than leaving
two reports and no way to tell which is current. ``--no-save`` prints only.

Usage:
    # one file: is it standardised?
    uv run python epai/ai_detection/analyse/check_standardization.py \
        data/parquet/ghostbuster_gpt56luna.parquet

    # the suite: are these four built the same way?
    uv run python epai/ai_detection/analyse/check_standardization.py \
        data/parquet/ghostbuster_gpt35t.parquet \
        data/parquet/ghostbuster_gpt35ts.parquet \
        data/parquet/ghostbuster_gpt54mini.parquet \
        data/parquet/ghostbuster_gpt56luna.parquet \
        data/parquet/science_v3_gpt56luna_0811A.parquet

    # hold every file to the 'flat' convention rather than asking which it meets
    uv run python epai/ai_detection/analyse/check_standardization.py \
        data/science/*.parquet --whitespace flat

    # the science corpora carry paper metadata on purpose: allow it
    uv run python epai/ai_detection/analyse/check_standardization.py \
        data/parquet/science_v3_gpt56luna_0811A.parquet \
        --extra-cols doi title year venue fwci prompt author_ids author_count \
                     citation_count language fields_of_study query \
                     also_matched_domains

    # gate a pipeline on it; the report is saved either way
    uv run python epai/ai_detection/analyse/check_standardization.py \
        data/parquet/ghostbuster_gpt56luna.parquet --fail-on error

    # somewhere specific, or nowhere
    uv run python epai/ai_detection/analyse/check_standardization.py \
        data/parquet/*.parquet --out-dir results/standardization/before_cleanup
    uv run python epai/ai_detection/analyse/check_standardization.py \
        data/parquet/aidetect_01.parquet --no-save

WHAT TO DO WITH A "NOT COMPARABLE" VERDICT
------------------------------------------
The emb_* and score sets are the axis a suite comparison is run over, and the
comparability section names exactly which columns are not shared. To cut a set
of corpora down to the columns they all have::

    uv run python epai/scripts/delete_parquet_columns.py \
        data/parquet/ghostbuster_gpt35t.parquet \
        data/parquet/ghostbuster_gpt54mini.parquet \
        data/parquet/ghostbuster_gpt56luna.parquet \
        --intersect --dry-run

which computes the same set difference this script reports and deletes it, after
printing the plan and asking. Re-run this script afterwards to confirm.

Name the corpora explicitly there rather than globbing a directory. The
intersection is only as rich as the POOREST file in it, so sweeping in one
corpus that was never embedded reduces every other corpus to its schema --
which is why that script refuses an empty intersection outright.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

# Every name below comes from the script that WRITES the thing being checked, so
# a conformance check cannot disagree with the pipeline about what conformance
# is. The one convention this module owns is TIGHTEST_FIRST.
from epai.ai_detection.prepare.combine_domains import (
    DOMAIN_ALIASES,
    DOMAIN_COL,
    DEFAULT_DROP_LABELS,
    EMB_PREFIX,
    ID_COL,
    OPTIONAL_PREFIXES,
    SCORE_PREFIXES,
    TARGET_COL,
    TEXT_COL,
    UID_COL,
    domains_of,
    order_columns,
)
from epai.ai_detection.prepare.standardize_text import RAW_SUFFIX
from epai.utils.label_utils import GENERATOR_LABEL_RE, LABEL_SEP
from epai.utils.text_standardize import (
    WHITESPACE_MODES,
    count_words,
    standardize_text,
    substitution_counts,
)

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

RULE = 92
MAX_LISTED = 8

# Where a run is saved when --out-dir says nothing. Under results/ beside the
# other analyse scripts' output, in a directory named for the files checked, so
# two runs over two different suites do not overwrite each other and a run over
# the same suite does -- the report is a statement about the files as they are
# now, not a log to accumulate.
DEFAULT_OUT_DIR = Path("results") / "standardization"

# Beyond this many inputs the directory name is a stem plus a count. Eight
# corpora would otherwise produce a 300-character path.
MAX_STEMS_IN_NAME = 3

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# The columns a standardised corpus must have. ``combine_domains.py`` requires
# only (id, text, generated_by) because it is building the other two; a file it
# has finished with has all five, and a file without them is a per-domain
# parquet from an earlier stage rather than a corpus.
STANDARD_COLS = (UID_COL, ID_COL, DOMAIN_COL, TEXT_COL, TARGET_COL)

RAW_TEXT_COL = f"{TEXT_COL}{RAW_SUFFIX}"

# The text column, its pre-standardisation copy, and the identity columns: the
# only ones this script reads row-wise. Everything else is inspected through the
# footer, which is what keeps a 400 MB corpus with fifteen embeddings cheap to
# check — loading the frame would cost more than every check in it.
ROW_COLS = (UID_COL, ID_COL, DOMAIN_COL, TEXT_COL, RAW_TEXT_COL, TARGET_COL)

# Whitespace modes ordered tightest first. ``flat`` collapses every run of
# whitespace to one space, so a flat-conformant text is trivially also
# lines-conformant; reporting the first mode that holds therefore reports the
# strongest true statement about the file rather than the weakest.
TIGHTEST_FIRST = ("flat", "lines")

# ``generated_by`` values that are neither ``human`` nor of the current
# convention, but are not mistakes: they are the generator names the original
# Ghostbuster release shipped, and a corpus built from those trees has to keep
# calling them what they are called there. Flagged as a WARN, never an ERROR --
# see label_utils.label_from_dir_name, which passes exactly these through.
# ``gpt`` is added by hand because it is the only one of them combine_domains.py
# does NOT drop by default -- it is the plain generation, the one the corpus is
# usually built on -- but it is still a name of the old convention.
LEGACY_LABELS = frozenset(DEFAULT_DROP_LABELS) | {"gpt"}

HUMAN_LABEL = "human"

# A machine/human mean word count further apart than this is a length tell: the
# ratio, not the difference, because the corpora differ in absolute length by
# an order of magnitude. 1.25 is chosen to sit above the spread the ghostbuster
# generators produce when they size a request against the human partner and
# below the gap that made the 200w/10w truncated variants necessary.
LENGTH_RATIO_WARN = 1.25

# Share of the HUMAN class that may be blank before it stops being upstream
# noise and starts being a corpus problem.
#
# There is a known floor here that no pipeline can lift: the Ghostbuster essay
# release ships six empty human documents, and nothing downstream can refill
# them -- fix_empty_generations.py regenerates from a prompt, and a human
# document has none. Six of 3000 human rows is 0.20%, so a threshold at 0.5%
# clears the documented case with room to spare while still catching a real
# hole. Above it the provenance argument stops working: a human class that is
# 5% empty is not "the upstream release ships a few", it is a corpus whose human
# side went missing somewhere, and the fact that the rows are human does not
# make that expected.
#
# Measured against the human class, not the whole file, because that is the
# question -- "how much of the human side is missing" does not have the machine
# rows in its denominator.
BLANK_HUMAN_ERROR_FRAC = 0.005


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


class Report:
    """Findings for one file, or for the comparison across files.

    Deliberately the same shape as ``audit_corpus.Report`` -- the two scripts are
    meant to be read one after the other, and a reader should not have to learn
    two verdict formats. Not imported from there: that module loads numpy and
    the whole summary_parquet stack for checks this one does not make.
    """

    def __init__(self) -> None:
        self.findings: list[dict] = []

    def error(self, section: str, message: str, hint: str = "", rows=None) -> None:
        self._add("ERROR", section, message, hint, rows)

    def warn(self, section: str, message: str, hint: str = "", rows=None) -> None:
        self._add("WARN", section, message, hint, rows)

    def _add(self, level, section, message, hint, rows) -> None:
        self.findings.append({
            "level": level,
            "section": section,
            "message": message,
            "hint": hint,
            "rows": list(rows) if rows is not None else [],
        })

    def counts(self) -> tuple[int, int]:
        levels = Counter(f["level"] for f in self.findings)
        return levels["ERROR"], levels["WARN"]

    @property
    def ok(self) -> bool:
        return not any(f["level"] == "ERROR" for f in self.findings)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print()
    print("=" * RULE)
    print(title)
    print("=" * RULE)


def pct(n: int, total: int) -> str:
    return f"{n} ({100.0 * n / total:.2f}%)" if total else f"{n}"


def listing(values, limit: int = MAX_LISTED) -> str:
    values = list(values)
    if not values:
        return "- (none)"
    shown = ", ".join(str(v) for v in values[:limit])
    more = len(values) - limit
    return f"{shown}{f', ... (+{more} more)' if more > 0 else ''}"


def fmt_size(path: Path) -> str:
    mib = path.stat().st_size / 1024**2
    return f"{mib:.1f} MiB" if mib < 1024 else f"{mib / 1024:.2f} GiB"


def is_blank(value) -> bool:
    """True for a cell holding no text: None, NaN, or whitespace only."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):  # an array is never a blank text
        pass
    return str(value).strip() == ""


# ---------------------------------------------------------------------------
# Reading -- footer first, then the few columns that need rows
# ---------------------------------------------------------------------------


def read_footer(path: Path) -> dict:
    """Column names, arrow types and row count, without reading a single row."""
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    schema = handle.schema_arrow
    return {
        "handle": handle,
        "columns": list(schema.names),
        "types": {name: str(schema.field(name).type) for name in schema.names},
        "n_rows": int(handle.metadata.num_rows),
    }


def read_rows(path: Path, columns: list[str]) -> pd.DataFrame:
    """The named columns only.

    Never the ``emb_*`` ones: fifteen 768-wide float columns are the whole size
    of these files, and nothing this script asks needs a vector -- only how many
    of them there are and how wide they are, which the footer and one row group
    answer.
    """
    return pd.read_parquet(path, columns=columns)


def embedding_widths(footer: dict, emb_cols: list[str]) -> dict[str, int | None]:
    """Width of each ``emb_*`` column, from the first row group only.

    One row group rather than the column: the width is a property of the
    embedder, so the first vector that exists answers it, and reading the rest
    would materialise hundreds of megabytes to re-confirm the same number. A
    column whose first row group is entirely null reports ``None`` -- that is a
    finding for ``audit_corpus.py``, not for this script, so it is reported and
    not judged.
    """
    handle = footer["handle"]
    widths: dict[str, int | None] = {}
    if handle.metadata.num_row_groups == 0:
        return {col: None for col in emb_cols}
    table = handle.read_row_group(0, columns=emb_cols)
    for col in emb_cols:
        width = None
        for value in table.column(col).to_pylist():
            if value is None:
                continue
            width = len(value)
            break
        widths[col] = width
    return widths


# ---------------------------------------------------------------------------
# Section 1 -- schema
# ---------------------------------------------------------------------------


def known_column(col: str, extra: tuple[str, ...]) -> bool:
    if col in STANDARD_COLS or col == RAW_TEXT_COL or col in extra:
        return True
    return any(col.startswith(p) for p in OPTIONAL_PREFIXES)


def check_schema(path: Path, footer: dict, rep: Report,
                 extra: tuple[str, ...]) -> dict:
    """The column contract: which columns, in which order, carrying nothing else."""
    section(f"Schema  —  {path.name}")
    columns = footer["columns"]
    emb_cols = [c for c in columns if c.startswith(EMB_PREFIX)]
    score_cols = [c for c in columns if c.startswith(SCORE_PREFIXES)]

    print(f"  path            : {path}")
    print(f"  size on disk    : {fmt_size(path)}")
    print(f"  rows            : {footer['n_rows']}")
    print(f"  columns         : {len(columns)}  "
          f"({len(emb_cols)} embedding, {len(score_cols)} score)")

    missing = [c for c in STANDARD_COLS if c not in columns]
    if missing:
        rep.error(
            "schema", f"missing standard column(s): {listing(missing)}",
            "A standardised corpus is what combine_domains.py writes: "
            f"{list(STANDARD_COLS)}. A file without uid/domain is a per-domain "
            "parquet from before that stage, not a corpus.",
            rows=missing,
        )
    print(f"  standard cols   : {'all present' if not missing else 'MISSING ' + listing(missing)}")

    # --- text_raw: the marker that the standardise stage ran ---------------
    has_raw = RAW_TEXT_COL in columns
    print(f"  {RAW_TEXT_COL:<16}: {'present' if has_raw else 'ABSENT'}")
    if not has_raw:
        rep.warn(
            "schema", f"no {RAW_TEXT_COL!r} column",
            f"standardize_text.py writes {RAW_TEXT_COL!r} unless --no-raw was "
            "passed, so its absence means either the stage never ran or the "
            "transform is not reversible from this file. The text checks below "
            "answer the first question either way; nothing answers the second.",
        )

    # --- columns nobody downstream expects ---------------------------------
    unexpected = [c for c in columns if not known_column(c, extra)]
    if unexpected:
        rep.warn(
            "schema", f"{len(unexpected)} column(s) outside the standard schema: "
            f"{listing(unexpected)}",
            "combine_domains.py drops everything that is not a standard column, "
            "a score, an emb_* or an explicit --extra-cols, so these will not "
            "survive the next combine. Pass --extra-cols to declare the ones "
            "that are carried on purpose (the science corpora's paper metadata).",
            rows=unexpected,
        )

    # --- order -------------------------------------------------------------
    # text_raw is passed as an extra so it gets a deterministic place. It is not
    # in order_columns' own vocabulary -- combine_domains.py predates the
    # standardise stage -- and left out it would land among the score columns,
    # which is a position no file actually uses.
    expected_order = order_columns(list(columns), extra + (RAW_TEXT_COL,))
    if list(columns) != expected_order:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(columns, expected_order)) if a != b),
            0,
        )
        rep.warn(
            "schema", "column order is not the canonical one "
            f"(first difference at position {first_diff}: "
            f"{columns[first_diff]!r}, expected {expected_order[first_diff]!r})",
            "order_columns() puts uid, id, domain, text, generated_by first, "
            "scores next and emb_* last so any of these files can be eyeballed "
            "with df.iloc[:, :8]. Re-running combine_domains.py restores it.",
        )
    print(f"  column order    : "
          f"{'canonical' if list(columns) == expected_order else 'non-canonical'}")
    print(f"  score columns   : {listing(score_cols, limit=6)}")
    print(f"  emb columns     : {listing(emb_cols, limit=20)}")

    return {"emb_cols": emb_cols, "score_cols": score_cols,
            "unexpected": unexpected, "has_raw": has_raw}


# ---------------------------------------------------------------------------
# Section 2 -- identity and labels
# ---------------------------------------------------------------------------


def check_identity(df: pd.DataFrame, rep: Report, limit: int) -> dict:
    """uid spelling and uniqueness, and the domain vocabulary."""
    section("Identity")
    n = len(df)
    out: dict = {}

    if UID_COL in df.columns and ID_COL in df.columns and TARGET_COL in df.columns:
        expected = (df[ID_COL].astype(str) + LABEL_SEP + df[TARGET_COL].astype(str))
        wrong = df[UID_COL].astype(str) != expected
        n_wrong = int(wrong.sum())
        print(f"  uid == '{{id}}{LABEL_SEP}{{generated_by}}' : "
              f"{n - n_wrong}/{n} rows")
        if n_wrong:
            # WARN, not ERROR, and the same severity audit_corpus.py gives it.
            # Nothing in this codebase PARSES a uid -- it is built (here and in
            # generate_ai.py) and then used as an opaque row identifier, and
            # every split groups on `id`. So a uid that no longer matches the
            # formula is stale provenance, not a broken key: the usual cause is
            # rename_column_values.py rewriting `generated_by` on a finished
            # corpus, which is a deliberate and correct operation that simply
            # has nothing to rebuild the uid with.
            #
            # What it costs is one inference: while the uid is stale, its
            # UNIQUENESS stops implying that (id, generated_by) is unique, since
            # the uid may still be separating rows by a label the corpus no
            # longer uses. That is the finding worth escalating, and
            # audit_corpus.py already makes it an error -- so this points there
            # rather than growing a second copy of the check that could drift.
            rep.warn(
                "identity", f"{pct(n_wrong, n)} rows whose uid is not "
                f"'{{id}}{LABEL_SEP}{{generated_by}}'",
                "Expected after a deliberate relabel — rename_column_values.py "
                "rewrites generated_by and nothing rebuilds the uid. Nothing "
                "parses a uid, so this blocks nothing; rebuild it with "
                "epai/scripts/rebuild_uid.py when convenient. One consequence "
                "worth acting on: a stale uid can be unique while "
                "(id, generated_by) is not, so run audit_corpus.py, which "
                "checks that pair directly.",
                rows=df.loc[wrong, UID_COL].astype(str).head(limit).tolist(),
            )
        out["uid_wrong"] = n_wrong

        dupes = df[UID_COL].duplicated(keep=False)
        n_dupes = int(dupes.sum())
        print(f"  uid unique                      : "
              f"{'yes' if not n_dupes else f'NO — {n_dupes} rows share a uid'}")
        if n_dupes:
            rep.error(
                "identity", f"{pct(n_dupes, n)} rows share a uid with another row",
                "The same document twice inflates every score without ever "
                "looking wrong in a row count.",
                rows=df.loc[dupes, UID_COL].astype(str).head(limit).tolist(),
            )
        out["uid_duplicated"] = n_dupes

    if DOMAIN_COL not in df.columns:
        return out

    # --- domain vocabulary --------------------------------------------------
    domains = df[DOMAIN_COL]
    n_null = int(domains.isna().sum())
    values = sorted(set(domains.dropna().astype(str)))
    print(f"  domains ({len(values)})".ljust(34) + f": {listing(values, limit=12)}")
    if n_null:
        rep.error("identity", f"{pct(n_null, n)} rows have no domain",
                  "Every cross-domain split is a filter on this column; a null "
                  "domain is a row no split can place.")

    unaliased = sorted(v for v in values if v in DOMAIN_ALIASES)
    if unaliased:
        rep.error(
            "identity", f"un-aliased domain name(s): {listing(unaliased)}",
            "combine_domains.py renames these on the way in "
            f"({', '.join(f'{k} -> {v}' for k, v in DOMAIN_ALIASES.items())}); "
            "a file still holding the short form will not match a config that "
            "names the long one, and will silently select no rows.",
            rows=unaliased,
        )

    odd = sorted(v for v in values if v != v.strip() or v != v.lower())
    if odd:
        rep.warn(
            "identity", f"domain name(s) that are not lower-case and trimmed: "
            f"{listing(odd)}",
            "A `where: {domain: ...}` in a config is an exact string match.",
            rows=odd,
        )

    # --- domain vs the id prefix -------------------------------------------
    # Only meaningful where the prefix IS the domain, which is the ghostbuster
    # shape (essay:1). A science id is science:<domain>:<paperId>, so its prefix
    # is the corpus; comparing it here would report every science row as a
    # mismatch. Detected rather than configured: if no row's prefix matches its
    # domain, the file uses the other convention and the check does not apply.
    #
    # Through domains_of, which ALIASES the prefix the way combine_domains.py
    # does. Comparing the raw prefix instead reports every reuter row in every
    # ghostbuster corpus as a mismatch -- `reut:AaronPressman:1` in domain
    # `reuter` is the rename working, not a row that lost its domain.
    prefixes = domains_of(df[ID_COL].astype(str), DOMAIN_ALIASES)
    matches = int((prefixes == domains.astype(str)).sum())
    if matches:
        mismatched = prefixes != domains.astype(str)
        n_mm = int(mismatched.sum())
        print(f"  id prefix == domain             : {n - n_mm}/{n} rows")
        if n_mm:
            rep.warn(
                "identity", f"{pct(n_mm, n)} rows whose id prefix disagrees with "
                "their domain column",
                "Most rows in this file use the id prefix as the domain, so the "
                "ones that do not were either relabelled after the id was "
                "built or came from a differently-shaped corpus.",
                rows=df.loc[mismatched, ID_COL].astype(str).head(limit).tolist(),
            )
        out["domain_prefix_mismatch"] = n_mm
    else:
        print("  id prefix == domain             : n/a "
              "(this corpus's ids are not domain-prefixed)")

    out["domains"] = values
    return out


def check_labels(df: pd.DataFrame, rep: Report) -> dict:
    """The ``generated_by`` vocabulary, against the one convention there is."""
    section("Labels")
    counts = Counter(df[TARGET_COL].astype(str))
    n = len(df)

    width = max((len(k) for k in counts), default=12)
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        if label == HUMAN_LABEL:
            shape = "human"
        elif GENERATOR_LABEL_RE.match(label):
            shape = "generator"
        elif label in LEGACY_LABELS:
            shape = "legacy"
        else:
            shape = "UNRECOGNISED"
        print(f"  {label:<{width}}  {count:>7}  {pct(count, n):>16}  {shape}")

    if HUMAN_LABEL not in counts:
        rep.error(
            "labels", f"no {HUMAN_LABEL!r} rows",
            "Every config binarises on data.human_label: 'human'. A corpus "
            "without that label trains on one class.",
        )

    unrecognised = sorted(
        lbl for lbl in counts
        if lbl != HUMAN_LABEL
        and not GENERATOR_LABEL_RE.match(lbl)
        and lbl not in LEGACY_LABELS
    )
    if unrecognised:
        rep.error(
            "labels", f"generated_by value(s) of no recognised shape: "
            f"{listing(unrecognised)}",
            "A generator is <model_slug>:<prompt_id> — lowercase letters and "
            "digits, ':', then <MMDD><letter>, e.g. 'gpt56luna:0807A'. "
            "An underscore here means label_from_dir_name never ran, and the "
            "corpus is spelling the generator the way the DIRECTORY spells it.",
            rows=unrecognised,
        )

    legacy = sorted(lbl for lbl in counts if lbl in LEGACY_LABELS)
    if legacy:
        stale = sorted(set(legacy) & set(DEFAULT_DROP_LABELS))
        rep.warn(
            "labels", f"original-Ghostbuster generator label(s) present: "
            f"{listing(legacy)}"
            + (f" — {listing(stale)} are on combine_domains.py's default drop "
               "list" if stale else ""),
            "These are prompt variants of one model, not separate generators. "
            "Keeping them turns human-vs-machine into a mix of generators with "
            "different class balances per domain, which is not the corpus the "
            "configs expect.",
            rows=legacy,
        )

    return {"labels": {k: int(v) for k, v in counts.items()}}


# ---------------------------------------------------------------------------
# Section 3 -- text standardisation
# ---------------------------------------------------------------------------


def conforming_mode(texts: pd.Series, pinned: str | None) -> tuple[str | None, dict]:
    """The tightest whitespace mode ``texts`` is already a fixpoint of.

    THE CENTRAL CHECK. ``standardize_text`` is idempotent, so a text it would
    still change is a text the stage never saw -- which makes "does the stage
    change this text" a complete test for "was this text standardised", with no
    inventory of individual artifacts needed to reach the verdict. The inventory
    below exists to say WHICH artifact, not WHETHER.

    Returns ``(mode or None, {mode: n_rows_that_differ})``. ``None`` means the
    file conforms to neither convention.
    """
    modes = (pinned,) if pinned else TIGHTEST_FIRST
    changed: dict[str, int] = {}
    for mode in modes:
        n = int(sum(1 for t in texts if standardize_text(t, whitespace=mode) != t))
        changed[mode] = n
        if n == 0:
            return mode, changed
    return None, changed


def check_text(df: pd.DataFrame, rep: Report, limit: int,
               pinned: str | None,
               blank_human_frac: float = BLANK_HUMAN_ERROR_FRAC) -> dict:
    """Is ``text`` standardised, and is any residue symmetric across the classes?"""
    section("Text standardisation")
    n = len(df)

    blank = df[TEXT_COL].apply(is_blank)
    n_blank = int(blank.sum())
    if n_blank:
        # Split by class, because the two cases have different causes and
        # different severities and lumping them together gets both wrong.
        #
        # A blank HUMAN row is upstream data. The Ghostbuster essay release
        # ships six empty documents; nothing in either pipeline can refill them
        # (fix_empty_generations.py regenerates from a prompt, and a human
        # document has none), and they are dropped at training time anyway --
        # empty text means a Null score and a Null embedding, which the loader's
        # missing-row drop removes. Expected, not actionable here.
        #
        # A blank MACHINE row is a failed generation that got carried in under
        # --allow-empty, and its label IS a lie: it is filed as machine-written
        # and holds nothing.
        keys = (df.loc[blank, UID_COL].astype(str).head(limit).tolist()
                if UID_COL in df.columns else None)
        blank_labels = Counter(df.loc[blank, TARGET_COL].astype(str))
        n_human = blank_labels.get(HUMAN_LABEL, 0)
        n_machine = n_blank - n_human

        if n_machine:
            rep.error(
                "text", f"{pct(n_machine, n)} MACHINE rows have no text",
                "A generation that failed and was carried in anyway. The row is "
                "filed as machine-written and holds nothing, so its label is a "
                "lie. Refill with the ghostbuster pipeline's "
                "'--only fix-empty', or drop the rows.",
                rows=keys,
            )
        if n_human:
            n_human_rows = int((df[TARGET_COL].astype(str) == HUMAN_LABEL).sum())
            frac = n_human / n_human_rows if n_human_rows else 0.0
            message = (
                f"{pct(n_human, n_human_rows)} of the human class has no text"
            )
            expected = (
                "The upstream Ghostbuster essay release ships six empty "
                "documents and nothing downstream can refill them (a human "
                "document has no prompt to regenerate from). They carry a Null "
                "score and a Null embedding and are dropped at training time. "
                "combine_domains.py --drop-incomplete removes them from the "
                "file itself — which is why a corpus built with it has 5994 "
                "rows where one built without has 6000."
            )
            if frac < blank_human_frac:
                rep.warn("text", message,
                         f"Under the {blank_human_frac:.2%} threshold, so read "
                         f"as upstream noise. " + expected, rows=keys)
            else:
                rep.error(
                    "text", message + f" — over the {blank_human_frac:.2%} "
                    "threshold",
                    "Too much of the human side to be the documented upstream "
                    "blanks. Something dropped human text: check the seeded "
                    "human/ tree the ghostbuster pipeline reads, or the filter "
                    "stage for the science one. "
                    "--max-blank-human-frac moves the line.",
                    rows=keys,
                )
    texts = df.loc[~blank, TEXT_COL].astype(str)

    # --- the fixpoint test --------------------------------------------------
    mode, changed = conforming_mode(texts, pinned)
    for name in (pinned,) if pinned else TIGHTEST_FIRST:
        n_changed = changed.get(name)
        if n_changed is None:
            continue
        mark = "conforms" if n_changed == 0 else f"{pct(n_changed, len(texts))} would change"
        print(f"  whitespace '{name}'".ljust(34) + f": {mark}")

    if mode is None:
        detail = ", ".join(f"{k}: {v} rows" for k, v in changed.items())
        rep.error(
            "text", "text is not a fixpoint of standardize_text "
            f"({detail})",
            "standardize_text is idempotent, so a text it would still change "
            "was never passed through it. Run "
            "epai/ai_detection/prepare/standardize_text.py over this parquet — "
            "with --dry-run FIRST, since the per-class inventory below is the "
            "evidence that an asymmetry existed and standardising destroys it.",
        )
    else:
        also = " (and therefore 'lines')" if mode == "flat" and not pinned else ""
        print(f"  {GREEN}standardised under mode '{mode}'{also}{RESET}")

    # --- which artifact, per domain and per class ---------------------------
    kept = df.loc[~blank]
    totals = group_totals(kept)
    residue = residual_inventory(kept)
    print_residue(residue, totals)
    if residue:
        asymmetric = asymmetric_groups(residue, totals)
        rep.error(
            "text", f"residual typographic characters in {len(residue)} group(s): "
            f"{listing(sorted(residue))}",
            "These are exactly the substitutions standardize_text performs, so "
            "their presence dates the file to before that stage.",
            rows=sorted(residue),
        )
        if asymmetric:
            rep.error(
                "text", "the residue is ASYMMETRIC across classes for "
                f"{listing(sorted(asymmetric))}",
                "This is the failure mode the module was written for: a curly "
                "apostrophe that means human in one domain and machine in "
                "another is not a weak feature, it is a strong feature pointing "
                "the wrong way, and it drives transfer BELOW chance rather than "
                "to it. See info/ and analyse_surface_artifacts.py.",
                rows=sorted(asymmetric),
            )

    # --- text_raw actually differs -----------------------------------------
    if RAW_TEXT_COL in df.columns:
        raw = df.loc[~blank, RAW_TEXT_COL].astype(str)
        n_diff = int((raw != texts).sum())
        print(f"  {RAW_TEXT_COL} differs from {TEXT_COL}".ljust(34)
              + f": {pct(n_diff, len(texts))} rows")
        if n_diff == 0:
            rep.warn(
                "text", f"{RAW_TEXT_COL} is identical to {TEXT_COL} in every row",
                "Either the corpus was already clean when the stage ran, or "
                f"{RAW_TEXT_COL} was written from an already-standardised copy. "
                "The stage guards against the second by writing the column only "
                "when it is absent, but a hand-edited file can still get there.",
            )

    length = check_lengths(kept, rep)
    return {"whitespace_mode": mode, "would_change": changed,
            "blank_rows": n_blank, "residue": residue, "lengths": length}


def residual_inventory(df: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
    """``{group: {domain: {label: n_docs_containing}}}`` for what standardising
    would still remove.

    Counted in DOCUMENTS rather than characters, and split by domain and class,
    because the number that matters is the presence RATE per class -- one
    document with two hundred curly quotes and two hundred documents with one
    each are the same character count and completely different features.
    """
    out: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(Counter))
    domains = (df[DOMAIN_COL].astype(str) if DOMAIN_COL in df.columns
               else pd.Series(["(whole file)"] * len(df), index=df.index))
    labels = df[TARGET_COL].astype(str)
    for text, domain, label in zip(df[TEXT_COL].astype(str), domains, labels):
        for group in substitution_counts(text):
            out[group][domain][label] += 1
    return {g: {d: dict(c) for d, c in doms.items()} for g, doms in out.items()}


def group_totals(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Rows per (domain, label), the denominator for a presence rate."""
    domains = (df[DOMAIN_COL].astype(str) if DOMAIN_COL in df.columns
               else pd.Series(["(whole file)"] * len(df), index=df.index))
    out: dict[str, Counter] = defaultdict(Counter)
    for domain, label in zip(domains, df[TARGET_COL].astype(str)):
        out[domain][label] += 1
    return {d: dict(c) for d, c in out.items()}


def print_residue(residue: dict, totals: dict[str, dict[str, int]]) -> None:
    """The inventory, as a PRESENCE RATE per class.

    Rates rather than counts, and against the per-(domain, class) denominator,
    because the classes are not the same size and the comparison being made is
    between them. "essay: human 902/1000, gpt56luna:0811A 3/1000" is the finding;
    "902 and 3" is the same finding with the reader left to divide.
    """
    if not residue:
        print(f"  residual typographic characters : {GREEN}none{RESET}")
        return
    print(f"  residual typographic characters : "
          f"{RED}{len(residue)} group(s){RESET}  "
          f"{DIM}(documents containing at least one, as a share of the class){RESET}")
    for group in sorted(residue):
        print(f"    {group}")
        for domain in sorted(residue[group]):
            per_label = residue[group][domain]
            cells = []
            for label in sorted(totals.get(domain, per_label)):
                n = per_label.get(label, 0)
                total = totals.get(domain, {}).get(label, 0)
                cells.append(f"{label}={pct(n, total)}" if total else f"{label}={n}")
            print(f"      {domain:<24} {'  '.join(cells)}")


def asymmetric_groups(residue: dict,
                      totals: dict[str, dict[str, int]]) -> set[str]:
    """Groups whose residue lands on one class of a domain and not the other.

    Symmetric residue is a nuisance -- both classes carry it, so it cannot carry
    the label. Residue present in one class of a domain and absent from the
    other IS the label, which is a different severity of the same finding.

    ``totals`` is what makes the distinction real: a Counter only holds the
    labels that HAD residue, so "one label present" and "one label exists in
    this domain at all" are the same shape and only the denominator tells them
    apart. A single-class domain is not an asymmetry, it is a single-class
    domain, and audit_corpus.py is where that gets reported.
    """
    out = set()
    for group, domains in residue.items():
        for domain, per_label in domains.items():
            classes = set(totals.get(domain, {})) or set(per_label)
            if len(classes) < 2:
                continue
            with_residue = {lbl for lbl, n in per_label.items() if n}
            if with_residue and with_residue != classes:
                out.add(group)
                break
    return out


def check_lengths(df: pd.DataFrame, rep: Report) -> dict:
    """Mean document length per class, per domain.

    Length is a standardisation characteristic and not only a corpus property:
    ``truncate_text.py`` and the ``_200w`` / ``_10w`` variants exist because a
    class whose documents are systematically longer hands a classifier the label
    without a single word being read.
    """
    print()
    print("  document length (words, zero-width-blind)")
    domains = (df[DOMAIN_COL].astype(str) if DOMAIN_COL in df.columns
               else pd.Series(["(whole file)"] * len(df), index=df.index))
    words = df[TEXT_COL].astype(str).map(count_words)
    labels = df[TARGET_COL].astype(str)

    out: dict[str, dict[str, float]] = {}
    for domain in sorted(set(domains)):
        rows = domains == domain
        means = {lbl: float(words[rows & (labels == lbl)].mean())
                 for lbl in sorted(set(labels[rows]))}
        out[domain] = means
        cells = "  ".join(f"{lbl}={m:.0f}" for lbl, m in means.items())
        print(f"    {domain:<24} {cells}")

        human = means.get(HUMAN_LABEL)
        machine = [m for lbl, m in means.items() if lbl != HUMAN_LABEL]
        if human and machine:
            worst = max(machine + [human]) / max(min(machine + [human]), 1e-9)
            if worst > LENGTH_RATIO_WARN:
                rep.warn(
                    "text", f"{domain}: mean length differs by {worst:.2f}x "
                    f"between classes ({cells})",
                    "A length gap is a label a classifier can read without "
                    "looking at a word. The truncated corpus variants "
                    "(_200w, _10w) exist to remove exactly this.",
                )
    return out


# ---------------------------------------------------------------------------
# Section 4 -- feature columns
# ---------------------------------------------------------------------------


def check_features(footer: dict, schema: dict, rep: Report) -> dict:
    """Which scores and embeddings the file carries, and how wide they are.

    Form only. Whether the vectors are healthy -- nulls, all-zero, NaN, width
    disagreement WITHIN a column -- is audit_corpus.py's job and is not repeated
    here; what matters for standardisation is that the SET is the set the other
    corpora carry, because a suite comparison silently reduces to the
    intersection.
    """
    section("Features")
    emb_cols, score_cols = schema["emb_cols"], schema["score_cols"]

    print(f"  score columns ({len(score_cols)})".ljust(34)
          + f": {listing(score_cols, limit=6)}")
    if not score_cols:
        rep.warn("features", "no score column (binoculars / llm judge)",
                 "Expected if the score stage has not run. A suite comparison "
                 "that includes a score feature block will drop this corpus.")

    widths = embedding_widths(footer, emb_cols) if emb_cols else {}
    print(f"  emb columns ({len(emb_cols)})".ljust(34) + ":")
    for col in emb_cols:
        width = widths.get(col)
        print(f"    {col:<24} {width if width is not None else '(no vector in row group 0)'}")
    if not emb_cols:
        rep.warn("features", "no emb_* column",
                 "Nothing here distinguishes a scored corpus from an embedded "
                 "one; run the embed stage before training on it.")

    return {"emb_widths": {k: v for k, v in widths.items()},
            "score_cols": score_cols}


# ---------------------------------------------------------------------------
# Section 5 -- across files
# ---------------------------------------------------------------------------


def compare_files(results: list[dict], rep: Report) -> None:
    """Do these files agree well enough to be compared with each other?

    The reason for naming more than one file. Every disagreement here is a
    difference that survives into a results table looking like a finding about
    detectors: a corpus with three embeddings compared against one with fifteen
    is not a fair comparison of embedders, it is a comparison of what happened
    to be computed.
    """
    section("Comparability across files")
    names = [Path(r["file"]).name for r in results]

    def report_axis(axis: str, per_file: dict[str, set], hint: str,
                    level: str = "error") -> None:
        shared = set.intersection(*per_file.values()) if per_file else set()
        union = set().union(*per_file.values()) if per_file else set()
        missing = {name: sorted(union - values)
                   for name, values in per_file.items() if union - values}
        print(f"\n  {axis}")
        print(f"    shared by all : {listing(sorted(shared), limit=20)}")
        if not missing:
            print(f"    {GREEN}every file carries the same set{RESET}")
            return
        for name, absent in missing.items():
            print(f"    {name:<48} missing: {listing(absent, limit=12)}")
        getattr(rep, level)(
            "comparability",
            f"{axis}: the files do not carry the same set "
            f"({len(shared)} shared, {len(union)} in the union)",
            hint,
            rows=[f"{name}: missing {', '.join(absent)}"
                  for name, absent in missing.items()],
        )

    report_axis(
        "emb_* columns",
        {Path(r["file"]).name: set(r["features"]["emb_widths"]) for r in results},
        "A cross-suite run over a feature axis these files do not share "
        "silently reduces to the intersection, and the ranking it produces is "
        "then over a different set of embedders than the one that was asked "
        "for. Run add_embeddings.py for the missing ones, or cut the axis down "
        "to the intersection deliberately.",
    )
    report_axis(
        "score columns",
        {Path(r["file"]).name: set(r["features"]["score_cols"]) for r in results},
        "A baseline block that names a score column absent from one corpus "
        "drops that corpus from the comparison rather than failing.",
    )
    report_axis(
        "identity and metadata columns",
        {Path(r["file"]).name:
         {c for c in r["types"]
          if not c.startswith(EMB_PREFIX) and not c.startswith(SCORE_PREFIXES)}
         for r in results},
        f"The one to read first: {RAW_TEXT_COL!r} in some files and not others "
        "means 'was this text standardised' is answerable for some of this "
        "suite and not the rest, and a metadata column present in one corpus "
        "only is a column no shared config can name.",
        level="warn",
    )
    report_axis(
        "generated_by labels",
        {Path(r["file"]).name: set(r["labels"]["labels"]) for r in results},
        "Expected across generators — that IS the variable in a generator "
        "comparison. Worth reading anyway: an unexpected extra label here is a "
        "stale generator directory that got collected into one corpus and not "
        "the others.",
        level="warn",
    )

    # --- embedding widths ---------------------------------------------------
    print("\n  emb_* widths")
    widths: dict[str, dict[str, int | None]] = defaultdict(dict)
    for result in results:
        for col, width in result["features"]["emb_widths"].items():
            widths[col][Path(result["file"]).name] = width
    disagreed = {col: per_file for col, per_file in widths.items()
                 if len({w for w in per_file.values() if w is not None}) > 1}
    if disagreed:
        for col, per_file in sorted(disagreed.items()):
            cells = ", ".join(f"{name}={w}" for name, w in per_file.items())
            print(f"    {RED}{col:<24}{RESET} {cells}")
        rep.error(
            "comparability",
            f"{len(disagreed)} emb_* column(s) have different widths in "
            f"different files: {listing(sorted(disagreed))}",
            "The same column name holding vectors of two widths means two "
            "different models, or one model at two truncation lengths, under "
            "one name. Nothing downstream can tell them apart.",
            rows=sorted(disagreed),
        )
    else:
        print(f"    {GREEN}every shared emb_* column has one width{RESET}")

    # --- arrow types on shared columns -------------------------------------
    print("\n  arrow types on shared columns")
    types: dict[str, dict[str, str]] = defaultdict(dict)
    for result in results:
        for col, arrow_type in result["types"].items():
            types[col][Path(result["file"]).name] = arrow_type
    conflicts = {col: per_file for col, per_file in types.items()
                 if len(set(per_file.values())) > 1}
    if conflicts:
        for col, per_file in sorted(conflicts.items()):
            cells = ", ".join(f"{name}={t}" for name, t in per_file.items())
            print(f"    {RED}{col:<24}{RESET} {cells}")
        rep.error(
            "comparability",
            f"{len(conflicts)} column(s) have different arrow types across "
            f"files: {listing(sorted(conflicts))}",
            "This is the one that fails loudly and late: reading two such files "
            "TOGETHER raises ArrowNotImplementedError, and a column that is "
            "all-null in one file is written with arrow type 'null' rather than "
            "the type it would have had. parquet_utils.coerce_dtypes exists for "
            "exactly this and should be applied at write time.",
            rows=sorted(conflicts),
        )
    else:
        print(f"    {GREEN}every shared column has one arrow type{RESET}")

    # --- whitespace convention ---------------------------------------------
    print("\n  whitespace convention")
    modes = {Path(r["file"]).name: r["text"]["whitespace_mode"] for r in results}
    for name, mode in modes.items():
        print(f"    {name:<48} {mode or RED + 'NEITHER' + RESET}")
    distinct = {m for m in modes.values() if m}
    if len(distinct) > 1:
        rep.warn(
            "comparability",
            f"the files use {len(distinct)} different whitespace conventions: "
            f"{listing(sorted(distinct))}",
            "Legitimate between corpus families — 'flat' for science abstracts, "
            "'lines' for the ghostbuster corpora — and NOT legitimate between "
            "two files that are meant to be the same corpus at two generators. "
            "Newline rate is a feature, so a convention that differs by file "
            "differs by whatever the file varies.",
        )

    print(f"\n  {len(names)} file(s) compared.")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def print_verdict(title: str, rep: Report, limit: int,
                  pass_word: str, fail_word: str) -> None:
    section(f"Verdict  —  {title}")
    n_err, n_warn = rep.counts()
    if not rep.findings:
        print(f"  {GREEN}{BOLD}{pass_word}{RESET}  no findings.")
        return

    for finding in rep.findings:
        colour = RED if finding["level"] == "ERROR" else YELLOW
        print(f"  {colour}{finding['level']:<5}{RESET} "
              f"[{finding['section']}] {finding['message']}")
        if finding["hint"]:
            for line in wrap_hint(finding["hint"]):
                print(f"          {DIM}{line}{RESET}")
        if finding["rows"]:
            print(f"          {DIM}-> {listing(finding['rows'], limit)}{RESET}")

    verdict = f"{GREEN}{BOLD}{pass_word}" if not n_err else f"{RED}{BOLD}{fail_word}"
    print(f"\n  {verdict}{RESET}  —  {n_err} error(s), {n_warn} warning(s).")


def wrap_hint(hint: str, width: int = RULE - 12) -> list[str]:
    words, lines, current = hint.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# One file
# ---------------------------------------------------------------------------


def check_file(path: Path, args: argparse.Namespace) -> tuple[Report, dict]:
    rep = Report()
    footer = read_footer(path)

    schema = check_schema(path, footer, rep, tuple(args.extra_cols))

    missing = [c for c in (ID_COL, TEXT_COL, TARGET_COL) if c not in footer["columns"]]
    if missing:
        # The row-wise sections all read these three; without them there is
        # nothing to check and guessing would report noise.
        rep.error("schema", f"missing required column(s) {listing(missing)} — the "
                  "identity, label and text sections cannot run",
                  "This is not a corpus parquet at all. summary_parquet.py will "
                  "say what it is.")
        print_verdict(path.name, rep, args.list, "STANDARDISED", "NOT STANDARDISED")
        return rep, {"file": str(path), "n_rows": footer["n_rows"],
                     "types": footer["types"], "schema": schema,
                     "identity": {}, "labels": {"labels": {}},
                     "text": {"whitespace_mode": None},
                     "features": {"emb_widths": {}, "score_cols": []},
                     "findings": rep.findings}

    row_cols = [c for c in ROW_COLS if c in footer["columns"]]
    df = read_rows(path, row_cols)

    identity = check_identity(df, rep, args.list)
    labels = check_labels(df, rep)
    text = check_text(df, rep, args.list, args.whitespace,
                      args.max_blank_human_frac)
    features = check_features(footer, schema, rep)

    print_verdict(path.name, rep, args.list, "STANDARDISED", "NOT STANDARDISED")

    return rep, {
        "file": str(path),
        "n_rows": footer["n_rows"],
        "types": footer["types"],
        "schema": {k: v for k, v in schema.items() if k != "handle"},
        "identity": identity,
        "labels": labels,
        "text": text,
        "features": features,
        "findings": rep.findings,
    }


# ---------------------------------------------------------------------------
# Saving the run
# ---------------------------------------------------------------------------


class Tee:
    """A stdout that also keeps what went through it.

    The report is written for a person to read, and the JSON is not: it holds
    every measurement but none of the tables, the per-domain residue rates or
    the ordering that makes them mean something. Re-deriving the printed report
    from the JSON would be a second renderer to keep in step with the first, so
    the printed report IS the saved report -- captured as it is produced, with
    the terminal colours taken out on the way to disk.
    """

    def __init__(self, stream) -> None:
        self.stream = stream
        self.buffer = io.StringIO()

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.buffer.write(data)
        return len(data)

    def flush(self) -> None:
        self.stream.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    @property
    def text(self) -> str:
        return _ANSI_RE.sub("", self.buffer.getvalue())


def run_name(paths: list[Path]) -> str:
    """A directory name for this set of inputs, stable across runs.

    Stable is the requirement: the same command must land in the same directory
    so that re-running after a fix overwrites the old verdict rather than
    leaving two reports and no way to tell which is current. Sorted for the same
    reason -- the order the files were typed in is not part of what was checked.
    """
    stems = sorted(path.stem for path in paths)
    if len(stems) <= MAX_STEMS_IN_NAME:
        return "__".join(stems)
    return f"{stems[0]}__and_{len(stems) - 1}_more"


def save_run(out_dir: Path, transcript: str, payload: dict,
             json_path: Path | None) -> list[Path]:
    """Write ``report.txt`` and ``report.json``; return what was written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "report.txt"]
    written[0].write_text(transcript)

    target = json_path or (out_dir / "report.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str))
    written.append(target)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("parquets", nargs="+", type=Path,
                    help="Corpus parquet(s) to check. Two or more also get a "
                         "comparability section, which is the point of naming "
                         "several at once.")
    ap.add_argument("--whitespace", choices=sorted(WHITESPACE_MODES), default=None,
                    help="Hold every file to this whitespace convention instead "
                         "of reporting the tightest one it already meets. "
                         f"Modes: {', '.join(sorted(WHITESPACE_MODES))}.")
    ap.add_argument("--extra-cols", nargs="*", default=[], metavar="COL",
                    help="Columns carried on purpose beyond the standard schema "
                         "(the science corpora's doi, title, prompt, ...). "
                         "Without this they are reported as unexpected.")
    ap.add_argument("--max-blank-human-frac", type=float,
                    default=BLANK_HUMAN_ERROR_FRAC, metavar="FRAC",
                    help="Share of the human class that may be blank before it "
                         f"is an error rather than a warning (default: "
                         f"{BLANK_HUMAN_ERROR_FRAC:.3f}). The upstream "
                         "Ghostbuster essay release ships six empty human "
                         "documents that nothing can refill, so a small "
                         "fraction is expected.")
    ap.add_argument("--list", type=int, default=MAX_LISTED, metavar="N",
                    help=f"Row keys / names to print per finding "
                         f"(default: {MAX_LISTED}).")
    ap.add_argument("--out-dir", type=Path, default=None, metavar="DIR",
                    help=f"Directory for report.txt and report.json (default: "
                         f"{DEFAULT_OUT_DIR}/<inputs>/). The same inputs always "
                         "produce the same directory, so re-running after a fix "
                         "replaces the old verdict instead of accumulating.")
    ap.add_argument("--json", type=Path, default=None, metavar="PATH",
                    help="Write the machine-readable half somewhere else than "
                         "the run directory. report.txt is still written.")
    ap.add_argument("--no-save", action="store_true",
                    help="Print only. The default is to save, since the report "
                         "is the input to deciding what to standardise.")
    ap.add_argument("--fail-on", choices=("error", "warn", "never"),
                    default="error",
                    help="Exit non-zero on findings at this level or above "
                         "(default: error).")
    return ap.parse_args()


def check_all(args: argparse.Namespace) -> tuple[dict, int, int]:
    """Every section of the run, printed. Returns (payload, errors, warnings).

    Split from :func:`main` so that everything it prints can be captured in one
    place; main's job is the plumbing around it.
    """
    print(f"check_standardization.py  —  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"inputs: {len(args.parquets)} file(s)")
    if args.whitespace:
        print(f"whitespace mode pinned to {args.whitespace!r}: "
              f"{WHITESPACE_MODES[args.whitespace]}")

    results, reports = [], []
    for path in args.parquets:
        rep, result = check_file(path, args)
        reports.append(rep)
        results.append(result)

    suite_rep = Report()
    if len(results) > 1:
        compare_files(results, suite_rep)
        print_verdict("all files", suite_rep, args.list,
                      "COMPARABLE", "NOT COMPARABLE")

    n_err = sum(r.counts()[0] for r in reports) + suite_rep.counts()[0]
    n_warn = sum(r.counts()[1] for r in reports) + suite_rep.counts()[1]

    section("Summary")
    for rep, result in zip(reports, results):
        errs, warns = rep.counts()
        state = (f"{GREEN}STANDARDISED{RESET}" if not errs
                 else f"{RED}NOT STANDARDISED{RESET}")
        print(f"  {Path(result['file']).name:<48} {state:<28} "
              f"{errs} error(s), {warns} warning(s)")
    if len(results) > 1:
        errs, warns = suite_rep.counts()
        state = (f"{GREEN}COMPARABLE{RESET}" if not errs
                 else f"{RED}NOT COMPARABLE{RESET}")
        print(f"  {'across files':<48} {state:<28} "
              f"{errs} error(s), {warns} warning(s)")
    print(f"\n  {n_err} error(s), {n_warn} warning(s) in total.")

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "whitespace_pinned": args.whitespace,
        "inputs": [str(path) for path in args.parquets],
        "n_errors": n_err,
        "n_warnings": n_warn,
        "files": results,
        "comparability": suite_rep.findings,
    }
    return payload, n_err, n_warn


def main() -> None:
    args = parse_args()

    # Before anything is printed: a missing input should fail on the command
    # line, not eight sections into a report that then gets saved as if it were
    # a verdict on the whole set.
    missing = [path for path in args.parquets if not path.is_file()]
    if missing:
        raise SystemExit("Not found: " + ", ".join(str(m) for m in missing))

    tee = Tee(sys.stdout)
    with contextlib.redirect_stdout(tee):
        payload, n_err, n_warn = check_all(args)

    if not args.no_save:
        out_dir = args.out_dir or (DEFAULT_OUT_DIR / run_name(args.parquets))
        written = save_run(out_dir, tee.text, payload, args.json)
        print()
        for path in written:
            print(f"Wrote {path}")

    if args.fail_on == "error" and n_err:
        raise SystemExit(1)
    if args.fail_on == "warn" and (n_err or n_warn):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

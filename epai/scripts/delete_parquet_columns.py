"""Delete columns from Parquet files, after showing exactly what would go.

A general parquet tool, next to ``rename_column_values.py`` — but DEFAULTED for
``epai/ai_detection/llm_judge/add_llm_judge_scores.py``: with no column
names given it targets that script's default judge column
(``llm_judge_score_gpt54_mini``), which is the column to drop when a file was
scored under the *old* convention where 1.0 meant AI. The judge now writes
P(human) -- 1.0 ⇒ human, 0.0 ⇒ machine -- into a column of the same name, so
stale values cannot be told apart from fresh ones and re-judging means clearing
the column first.

Nothing is written until the plan is printed and confirmed. The plan lists, per
targeted column, its dtype, how many values it holds, its range, and -- when the
file has a ``generated_by`` column -- its mean per generator. That last table is
what identifies the convention a judge column was written under: under P(human)
the ``human`` rows have the HIGHER mean, under the old P(AI) the LOWER one. Read
it before typing 'yes'. With several files named, the whole plan for all of them
is printed first and there is ONE confirmation for the lot -- a prompt per file
trains you to type 'yes' without reading, which is the opposite of the point.

THREE WAYS TO NAME THE COLUMNS
------------------------------
``--columns`` takes literal names, glob patterns, or both::

    --columns llm_judge_score_gpt54_mini      # this column
    --columns 'emb_openai_*' 'emb_qwen3'      # a pattern and a name
    --columns 'llm_judge_score*'              # every judge column

Quote the patterns: an unquoted ``emb_*`` is expanded by the shell against the
files in your working directory, and it will match nothing there.

``--intersect`` names them for you, and is the reason this script takes more
than one file. Across every named parquet it finds the feature columns that are
NOT present in all of them, and deletes exactly those -- which is what makes a
set of corpora comparable, since a cross-suite run over a feature axis the files
do not share silently reduces to the intersection anyway. The set it deletes is
the set ``check_standardization.py`` reports as not shared; run that first.

WHAT ``--intersect`` WILL NOT TOUCH, and why
--------------------------------------------
Only ``emb_*`` and the score columns, never the identity or metadata ones. A
column that is missing from one file is not always a column to delete from the
others: ``text_raw`` is the ONLY record of what a document looked like before
``standardize_text.py`` rewrote it, and deleting it to match a corpus that never
had one destroys the evidence rather than aligning the schema. The science
corpora's ``doi``/``title``/``prompt`` block is the same case. Widen the rule
with ``--intersect-prefixes`` if you mean it; the default will not do it by
accident.

Requested columns that a file does not have are reported and skipped, so the
same command can be pointed at a whole set of parquets where only some carry the
column; ``--strict`` turns a missing column into an error instead. A pattern
that matches nothing in ANY file is always reported -- a misspelled pattern and
a successful no-op are otherwise indistinguishable. Every other column, and the
row order, is left exactly as it was.

The write is in place unless ``--out`` is given (which needs a single input).

THERE IS NO UNDO. ``.gitignore`` holds ``/data/``, so these parquets are NOT in
git and ``git checkout`` will not bring a column back -- the only way to rebuild
one is to re-run the stage that computed it, which for an ``emb_*`` column means
the GPU. Use ``--dry-run`` first, and ``-o`` when you are not certain.

Run:
    # the default judge column, in place, with a confirmation prompt
    uv run python epai/scripts/delete_parquet_columns.py data/parquet/wp_binox0.parquet

    # look, decide nothing
    uv run python epai/scripts/delete_parquet_columns.py data/parquet/wp_binox0.parquet --dry-run

    # specific columns, and patterns
    uv run python epai/scripts/delete_parquet_columns.py data/parquet/wp_binox0.parquet \
        -c llm_judge_score_gpt54_mini 'emb_openai_*'

    # make a suite comparable: drop every emb_*/score column they do not all share
    uv run python epai/scripts/delete_parquet_columns.py \
        data/parquet/ghostbuster_gpt35t.parquet \
        data/parquet/ghostbuster_gpt54mini.parquet \
        data/parquet/ghostbuster_gpt56luna.parquet \
        --intersect --dry-run

    # into a copy, leaving the input untouched
    uv run python epai/scripts/delete_parquet_columns.py data/parquet/wp_binox0.parquet \
        -c llm_judge_score_gpt54_mini -o data/parquet/wp_binox0_nojudge.parquet
"""

from __future__ import annotations

import argparse
import fnmatch
import textwrap
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from epai.ai_detection.llm_judge.add_llm_judge_scores import (
    COL_PREFIX,
    DEFAULT_MODEL,
    GENERATED_BY_COL,
    score_column,
)
from epai.ai_detection.prepare.combine_domains import EMB_PREFIX, SCORE_PREFIXES
from epai.utils.parquet_utils import load_parquet, save_parquet

# With no columns on the command line, the target is the column
# add_llm_judge_scores.py writes by default -- the one that has to be cleared
# before a file scored under the old P(AI) convention can be re-judged.
DEFAULT_COLUMNS = [score_column(DEFAULT_MODEL)]

# What --intersect considers a FEATURE column, and therefore safe to align by
# deleting. Deliberately not "every column": see the module docstring. Imported
# from combine_domains.py rather than spelled again, so the definition of a
# feature column cannot drift between the script that writes these files and the
# script that trims them.
INTERSECT_PREFIXES = (EMB_PREFIX,) + SCORE_PREFIXES

# Non-numeric columns are previewed by example rather than by range.
EXAMPLE_VALUES = 3
EXAMPLE_CHARS = 60

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

RULE = 78


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "parquets",
        nargs="+",
        type=Path,
        metavar="PARQUET",
        help="Parquet file(s) to delete columns from.",
    )
    ap.add_argument(
        "-c",
        "--columns",
        nargs="*",
        default=None,
        metavar="COLUMN",
        help="Columns to delete: literal names, glob patterns, or both "
        "(quote the patterns). Default: %s." % " ".join(DEFAULT_COLUMNS),
    )
    ap.add_argument(
        "--intersect",
        action="store_true",
        help="Ignore --columns and instead delete, from every file, the "
        "feature columns the files do not all share. This is the "
        "'make these corpora comparable' mode.",
    )
    ap.add_argument(
        "--intersect-prefixes",
        nargs="*",
        default=list(INTERSECT_PREFIXES),
        metavar="PREFIX",
        help="Which columns --intersect is allowed to consider "
        "(default: %s). Widening this to identity or metadata columns will "
        "delete text_raw and the science paper metadata; read the module "
        "docstring before you do." % " ".join(INTERSECT_PREFIXES),
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output path (default: overwrite each input in place). Only "
        "valid with a single input file.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Error out if a requested column is not in a file. Default: "
        "report it and delete the ones that are there.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and stop. Nothing is asked and nothing is written.",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt (the delete is destructive; the "
        "prompt is the default).",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Choosing the columns
# ---------------------------------------------------------------------------


def resolve_patterns(requested: list[str] | None) -> list[str]:
    """The patterns to delete by, de-duplicated, order preserved."""
    values = list(DEFAULT_COLUMNS) if not requested else list(requested)
    seen: set[str] = set()
    return [c for c in values if not (c in seen or seen.add(c))]


def match_columns(patterns: list[str], present: list[str]) -> list[str]:
    """Columns of ``present`` matched by any of ``patterns``, in file order.

    File order, not pattern order: the result is printed as "what leaves this
    file", and a reader compares it against the file's own column list.
    ``fnmatch`` rather than a regex because the patterns are typed on a command
    line beside file paths, where ``*`` already means what a user expects it to.
    A literal name is simply a pattern with no wildcard in it, so there is one
    code path and not two.
    """
    return [col for col in present
            if any(fnmatch.fnmatchcase(col, pat) for pat in patterns)]


def read_columns(path: Path) -> list[str]:
    """A file's column names from its footer. No row is read.

    Planning reads schemas only. The frame is loaded once, later, and only for
    the files that actually have something to drop -- these corpora run to
    hundreds of megabytes and a plan that loaded all of them to decide it had
    nothing to do would cost more than the delete.
    """
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema_arrow.names)


def intersect_targets(
    schemas: dict[Path, list[str]], prefixes: tuple[str, ...]
) -> dict[Path, list[str]]:
    """Per file, the feature columns not every file has.

    The set difference is computed over the union rather than pairwise, so the
    result is symmetric: whatever survives is a column every input carries, and
    running the command twice changes nothing the second time.
    """
    def features(columns: list[str]) -> set[str]:
        return {c for c in columns if c.startswith(prefixes)}

    per_file = {path: features(cols) for path, cols in schemas.items()}
    shared = set.intersection(*per_file.values()) if per_file else set()

    # An empty intersection would delete EVERY feature column from every file,
    # which is never what "make these comparable" means. It happens the moment
    # one input carries no features at all -- a *_human.parquet from before the
    # embed stage, say -- and the deletion it would authorise is unrecoverable
    # except from git. Refuse and name the file, rather than ask the user to
    # notice it in a plan of two hundred columns.
    if not shared and any(per_file.values()):
        empty = sorted(str(path) for path, cols in per_file.items() if not cols)
        raise SystemExit(
            "--intersect: these files share NO feature column, so aligning "
            "them by deletion would remove every embedding and score from all "
            f"of them. Refusing.\n"
            + (f"  file(s) carrying no feature column at all: "
               f"{', '.join(empty)}\n" if empty else "")
            + "  per file: "
            + "; ".join(f"{path.name}={len(cols)}"
                        for path, cols in per_file.items())
            + "\n  Drop the odd file out of the command, or widen "
              "--intersect-prefixes if these really are the columns you mean."
        )

    return {
        path: [c for c in schemas[path] if c in cols and c not in shared]
        for path, cols in per_file.items()
    }


# ---------------------------------------------------------------------------
# Describing what would go
# ---------------------------------------------------------------------------


def _example_values(series: pd.Series) -> str:
    """A few real values from a non-numeric column, truncated for printing."""
    present = series.dropna().astype(str).head(EXAMPLE_VALUES).tolist()
    if not present:
        return "(no values)"
    return ", ".join(
        repr(v[:EXAMPLE_CHARS] + "…" if len(v) > EXAMPLE_CHARS else v[:EXAMPLE_CHARS])
        for v in present
    )


def _embedding_width(series: pd.Series) -> int | None:
    for value in series:
        if value is None:
            continue
        try:
            return len(value)
        except TypeError:
            continue
    return None


def describe_column(df: pd.DataFrame, col: str) -> list[str]:
    """What the file currently holds in ``col``, as printable lines.

    Enough detail to recognise the column before it is thrown away: how much of
    it is filled in, and either its numeric range, its width if it is an
    embedding, or a few of its values.
    """
    series = df[col]
    n_null = int(series.isna().sum())
    lines = [
        f"    dtype       : {series.dtype}",
        f"    values      : {len(series) - n_null} of {len(series)} rows "
        f"({n_null} null)",
    ]

    values = series.dropna()
    if col.startswith(EMB_PREFIX):
        # An embedding's own values say nothing a reader can use -- printing
        # three of 768 floats is noise. Its width is what identifies it.
        width = _embedding_width(values)
        lines.append(f"    width       : {width if width is not None else '(no vector)'}")
    elif is_numeric_dtype(series) and not values.empty:
        lines.append(
            f"    range       : min {values.min():.4f}   "
            f"mean {values.mean():.4f}   max {values.max():.4f}"
        )
        lines.extend(_means_per_generator(df, col))
    else:
        lines.append(f"    examples    : {_example_values(series)}")
    return lines


def _means_per_generator(df: pd.DataFrame, col: str) -> list[str]:
    """Mean of ``col`` per ``generated_by`` value, which reveals its direction.

    For a judge column this is the check that matters: P(human) puts the
    ``human`` rows at the high end, the old P(AI) convention at the low end.
    """
    if GENERATED_BY_COL not in df.columns:
        return []

    lines = [f"    mean per {GENERATED_BY_COL}:"]
    groups = df.groupby(df[GENERATED_BY_COL].astype(str), sort=True)[col]
    for name, group in groups:
        present = group.dropna()
        if present.empty:
            lines.append(f"      {name:<14}: (no values, n=0)")
        else:
            lines.append(
                f"      {name:<14}: {present.mean():.4f}  (n={len(present)})"
            )

    if col.startswith(COL_PREFIX):
        lines.append(
            "      ^ judge column: 'human' highest ⇒ P(human), the current "
            "convention;"
        )
        lines.append(
            "        'human' lowest ⇒ the old P(AI) scores, which need this "
            "delete."
        )
    return lines


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def print_file_plan(
    path: Path,
    out_path: Path,
    columns: list[str],
    targets: list[str],
    df: pd.DataFrame | None,
) -> None:
    """What would happen to one file.

    ``df`` is None when the file loses nothing: there is then no column to
    describe, and loading a 400 MB frame to print "nothing to delete" is the
    cost this avoids.
    """
    remaining = [c for c in columns if c not in set(targets)]

    print("-" * RULE)
    print(f"{BOLD}{path}{RESET}")
    print(f"  columns       : {len(columns)}")
    if not targets:
        print(f"  {GREEN}nothing to delete — left untouched{RESET}")
        return

    print(f"  rows          : {len(df)}")
    print()
    print(f"  {RED}DELETE ({len(targets)}){RESET}")
    for col in targets:
        print(f"  - {col}")
        for line in describe_column(df, col):
            print(line)
    print()
    print(f"  KEEP ({len(remaining)}), unchanged")
    for line in textwrap.wrap(", ".join(remaining), width=RULE - 4) or ["(none)"]:
        print(f"    {line}")
    print()
    print(f"  write         : {out_path}")
    print(
        "  mode          : "
        + (
            f"{YELLOW}IN PLACE — the input file is overwritten{RESET}"
            if out_path == path
            else "new file — the input is left untouched"
        )
    )


def confirm(total_columns: int, n_files: int) -> bool:
    """Ask once, for the whole plan. Anything but 'yes' aborts."""
    print()
    print(
        f"{YELLOW}This deletes {total_columns} column(s) and their values "
        f"across {n_files} file(s), above.{RESET}"
    )
    answer = input("Proceed? Type 'yes': ").strip().lower()
    return answer == "yes"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Duplicates would print the plan twice and write the file twice; keep the
    # order the arguments were given in, which is the order the plan is read in.
    seen: set[Path] = set()
    args.parquets = [p for p in args.parquets
                     if not (p in seen or seen.add(p))]

    missing_files = [p for p in args.parquets if not p.is_file()]
    if missing_files:
        # This script used to take `<parquet> <column> <column>`, so a stale
        # command lands here with column names where files should be. Say so:
        # "Not found: llm_judge_score_gpt54_mini" is a true statement and a
        # useless one.
        looks_like_columns = [str(p) for p in missing_files
                              if p.suffix != ".parquet" and len(p.parts) == 1]
        hint = ""
        if looks_like_columns:
            hint = (
                f"\n  {', '.join(looks_like_columns)} look(s) like column "
                "name(s). Columns now go after -c/--columns, because the "
                "positional arguments are the files:\n"
                f"    ... {args.parquets[0] if args.parquets else 'FILE.parquet'} "
                f"-c {' '.join(looks_like_columns)}"
            )
        raise SystemExit(
            "Not found: " + ", ".join(str(p) for p in missing_files) + hint
        )
    if args.out and len(args.parquets) > 1:
        raise SystemExit(
            "--out names one output path but "
            f"{len(args.parquets)} input files were given. Delete in place "
            "(the parquets are in git), or run one file at a time."
        )

    schemas = {path: read_columns(path) for path in args.parquets}

    # --- what to delete, per file ------------------------------------------
    if args.intersect:
        prefixes = tuple(args.intersect_prefixes)
        plan = intersect_targets(schemas, prefixes)
        print(f"--intersect over prefixes {list(prefixes)}: deleting the "
              f"feature columns the {len(schemas)} file(s) do not all share.")
        patterns: list[str] = []
    else:
        patterns = resolve_patterns(args.columns)
        plan = {path: match_columns(patterns, cols)
                for path, cols in schemas.items()}

    # A pattern that matched nothing anywhere is a typo, and looks exactly like
    # a successful no-op unless it is said out loud.
    if patterns:
        matched_any = {
            pat for pat in patterns
            for cols in schemas.values()
            if any(fnmatch.fnmatchcase(c, pat) for c in cols)
        }
        unmatched = [pat for pat in patterns if pat not in matched_any]
        if unmatched:
            message = (
                f"pattern(s) matching no column in any input: "
                f"{', '.join(unmatched)}"
            )
            if args.strict:
                raise SystemExit(f"{RED}{message}{RESET} (--strict)")
            print(f"{YELLOW}NOTE: {message}{RESET}")

    if args.strict:
        for path, cols in schemas.items():
            absent = [p for p in patterns
                      if not any(fnmatch.fnmatchcase(c, p) for c in cols)]
            if absent:
                raise SystemExit(
                    f"{path}: no column matches {absent} (--strict). "
                    f"It has: {cols}"
                )

    for path, targets in plan.items():
        if targets and len(targets) == len(schemas[path]):
            raise SystemExit(
                f"{path}: that would delete every column and leave an empty "
                "file. Refusing."
            )

    total = sum(len(t) for t in plan.values())
    if not total:
        print(f"\n{GREEN}Nothing to delete in any of "
              f"{len(args.parquets)} file(s).{RESET}")
        if not args.intersect:
            print(f"Looked for: {', '.join(patterns)}")
        return

    # --- print the whole plan, then ask once --------------------------------
    print("=" * RULE)
    frames: dict[Path, pd.DataFrame] = {}
    for path in args.parquets:
        targets = plan[path]
        if targets:
            frames[path] = load_parquet(path)
        print_file_plan(path, args.out or path, schemas[path], targets,
                        frames.get(path))
    print("=" * RULE)
    print(f"Total: {total} column deletion(s) across "
          f"{sum(1 for t in plan.values() if t)} of {len(plan)} file(s).")

    if args.dry_run:
        print("\n--dry-run: nothing was written.")
        return
    if not args.yes and not confirm(total, sum(1 for t in plan.values() if t)):
        print("Aborted — no changes made.")
        return

    # --- write --------------------------------------------------------------
    print()
    for path, targets in plan.items():
        if not targets:
            continue
        save_parquet(frames[path].drop(columns=targets), args.out or path)
        print(f"  {path.name}: deleted {len(targets)} column(s): "
              f"{', '.join(targets)}")
    print(f"\n{GREEN}Done — {total} column(s) deleted.{RESET}")


if __name__ == "__main__":
    main()

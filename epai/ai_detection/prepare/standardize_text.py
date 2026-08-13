r"""Stage: standardise a corpus parquet's text, both classes, in place.

Runs between ``combine`` and ``score`` in the Ghostbuster pipeline and between
``promote`` and ``score`` in the science one. It rewrites the ``text`` column
with :func:`epai.utils.text_standardize.standardize_text` -- typographic
characters folded to ASCII, layout put on one convention -- and touches nothing
else.

WHY IT IS A STAGE AND NOT A FLAG ON prepare
-------------------------------------------
Because it must see BOTH classes, and because it must run before anything reads
the text. A per-domain prepare script sees one domain at a time and a generate
script sees one class at a time; either is a place to introduce an asymmetry,
not to remove one. Here the transform is applied to a whole column with no
reference to ``generated_by``, so "did this apply to human or AI text" is not a
question the code can get wrong.

Before ``score`` and ``embed`` because those read ``text``: a Binoculars score
or an embedding computed on the pre-standardised text is a number about a
document that is no longer in the corpus.

OUTPUT
------
The parquet is rewritten in place (``-o`` writes elsewhere) with two columns
touched:

  text        the standardised text
  text_raw    the text exactly as it was before this stage ran

``text_raw`` is written only when it is absent, so re-running the stage never
replaces the true original with an already-standardised copy. ``--no-raw``
skips it, at the cost of making the transform irreversible from this file
alone. Note the suffix: ``normalize_text.py`` uses the opposite order,
``raw_text``, for the text before *its* experiment — the two are deliberately
distinct so a corpus can carry both without either being ambiguous.

Nothing else is renamed or dropped. Scores and embeddings are not recomputed,
which is why the stage refuses to run when they are already populated.

WHAT IT WILL NOT DO
-------------------
Remove text by keyword. No preambles, no section labels, no titles, no
reference lists. Those are content edits and they hide in a diff; this stage is
character substitution and whitespace only, so the word count of every document
survives it — counted with ``count_words``, which ignores tokens made of
nothing but zero-width characters, since deleting those is the job.
Word-level cleanup that a corpus genuinely needs belongs upstream,
in the generator, applied to both classes by construction -- which is what
``science_data.rules.strip_structured_labels`` does.

USAGE

    uv run python epai/ai_detection/prepare/standardize_text.py <parquet>
    uv run python epai/ai_detection/prepare/standardize_text.py <parquet> --whitespace flat
    uv run python epai/ai_detection/prepare/standardize_text.py <parquet> -o out.parquet
    uv run python epai/ai_detection/prepare/standardize_text.py <parquet> --dry-run

``--dry-run`` reports the per-class inventory and writes nothing, which is the
form worth running first: the inventory is the evidence that an asymmetry
existed, and standardising destroys it.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from epai.utils.parquet_utils import load_parquet, save_parquet  # noqa: E402
from epai.utils.text_standardize import (  # noqa: E402
    WHITESPACE_MODES,
    count_words,
    standardize_text,
    substitution_counts,
)

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

DEFAULT_TEXT_COL = "text"
DEFAULT_CLASS_COL = "generated_by"

# Where the pre-standardisation text is kept, so the transform is reversible.
#
# ``<col>_raw``, not normalize_text.py's ``raw_<col>``: that script is a
# separate, later, opt-in experiment, and the two prefixes staying distinct is
# what lets a corpus carry both -- `text_raw` is the text as the pipeline built
# it, `raw_text` is the text as it stood before a normalisation experiment.
RAW_SUFFIX = "_raw"

# Columns whose values describe the OLD text. Standardising with any of these
# already populated leaves numbers that silently refer to a document that is no
# longer in the file.
STALE_PREFIXES = ("emb_",)
STALE_EXACT = ("binoculars_score",)


def stale_columns(df: pd.DataFrame) -> list[str]:
    """Populated score/embedding columns, which standardising would invalidate."""
    candidates = [
        c for c in df.columns
        if c in STALE_EXACT or any(c.startswith(p) for p in STALE_PREFIXES)
    ]
    return [c for c in candidates if df[c].notna().any()]


def class_label(value) -> str:
    """``human`` vs the generator label, for the per-class report."""
    if value is None or value != value:
        return "(unlabelled)"
    return str(value)


def report_inventory(df: pd.DataFrame, text_col: str, class_col: str) -> None:
    """Per-class count of documents containing each substitutable group.

    The point of the split is the asymmetry: one class carrying curly quotes
    and the other not is the artifact, and a single corpus-wide total would
    hide exactly that.
    """
    if class_col not in df.columns:
        print(f"{YELLOW}  no {class_col!r} column; reporting the corpus as one group{RESET}")
        groups = [("(all rows)", df)]
    else:
        groups = [(class_label(k), g) for k, g in df.groupby(class_col, dropna=False)]

    print(f"\n{BOLD}  Documents containing each substitutable group, before:{RESET}")
    all_groups: set[str] = set()
    per_class: dict[str, Counter] = {}
    for label, g in groups:
        counter: Counter = Counter()
        for text in g[text_col].fillna(""):
            for group in substitution_counts(str(text)):
                counter[group] += 1
        per_class[label] = counter
        all_groups.update(counter)

    if not all_groups:
        print(f"    {GREEN}nothing to substitute — the corpus is already ASCII-clean{RESET}")
        return

    width = max(len(label) for label, _ in groups) + 2
    cols = sorted(all_groups)
    header = "".join(f"{c[:13]:>15}" for c in cols)
    print(f"    {'class':<{width}}{'rows':>7}{header}")
    for label, g in groups:
        n = len(g)
        cells = "".join(
            f"{(per_class[label].get(c, 0) / n if n else 0):>14.1%} " for c in cols
        )
        print(f"    {label:<{width}}{n:>7}{cells}")
    print(f"\n{DIM}    A group that differs sharply between classes is a tell the "
          f"classifier can read.\n    Sign can flip between domains, which is what "
          f"sends cross-domain transfer below chance.{RESET}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Standardise a corpus parquet's text column, both classes, "
                    "without removing any text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="whitespace modes:\n" + "\n".join(
            f"  {name:<7} {desc}" for name, desc in WHITESPACE_MODES.items()
        ),
    )
    ap.add_argument("parquet", type=Path, help="corpus to standardise")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="write here instead of rewriting the input in place")
    ap.add_argument("--whitespace", choices=sorted(WHITESPACE_MODES), default="lines",
                    help="layout convention to impose (default: lines)")
    ap.add_argument("--text-column", default=DEFAULT_TEXT_COL,
                    help=f"column to standardise (default: {DEFAULT_TEXT_COL})")
    ap.add_argument("--class-column", default=DEFAULT_CLASS_COL,
                    help=f"column naming the class, for the report only "
                         f"(default: {DEFAULT_CLASS_COL})")
    ap.add_argument("--raw-column", default=None,
                    help=f"where to keep the pre-standardisation text (default: "
                         f"<text-column>{RAW_SUFFIX}). Written only if absent, so "
                         f"re-running never overwrites the true original with an "
                         f"already-standardised copy.")
    ap.add_argument("--no-raw", action="store_true",
                    help="do not keep the original text. The transform is then "
                         "irreversible from this file alone.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the inventory and the number of rows that would "
                         "change; write nothing")
    ap.add_argument("--force", action="store_true",
                    help="standardise even when scores or embeddings are already "
                         "populated. They describe the old text and are NOT "
                         "recomputed — only pass this if you will re-run score "
                         "and embed afterwards.")
    args = ap.parse_args()

    print(f"{BOLD}standardize_text.py{RESET}  —  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  input       {args.parquet}")
    print(f"  whitespace  {args.whitespace}")

    if not args.parquet.is_file():
        print(f"{RED}Not found: {args.parquet}{RESET}")
        return 1

    df = load_parquet(args.parquet)
    if args.text_column not in df.columns:
        print(f"{RED}No {args.text_column!r} column. Found: {list(df.columns)[:12]}{RESET}")
        return 1

    print(f"  rows        {len(df)}")
    report_inventory(df, args.text_column, args.class_column)

    original = df[args.text_column].fillna("").astype(str)
    standardised = original.map(lambda t: standardize_text(t, whitespace=args.whitespace))

    changed = (original != standardised).sum()
    # Word count is preserved by construction -- this stage substitutes
    # characters, it does not delete text. A mismatch means a transform is
    # doing more than it claims, which is the failure worth catching loudly.
    #
    # count_words, not len(t.split()): a run of zero-width characters standing
    # between two spaces is a token to str.split and a word to nobody. Deleting
    # it is this stage's job and it moves the naive count without any text
    # having been lost, so the invariant is stated over words a reader can see.
    words_before = original.map(count_words)
    words_after = standardised.map(count_words)
    total_before, total_after = words_before.sum(), words_after.sum()

    print(f"\n  rows changed        {changed} of {len(df)}")
    print(f"  words before/after  {total_before} / {total_after}")
    if total_before != total_after:
        delta = words_after - words_before
        offenders = delta[delta != 0]
        print(f"{RED}  Word count changed by {total_after - total_before}. This stage "
              f"must never add or remove words —\n  a transform is doing more than "
              f"substitution. Not writing.{RESET}")
        print(f"  {len(offenders)} row(s) changed length; first few:")
        for idx, d in offenders.head(5).items():
            print(f"    row {idx}: {words_before[idx]} → {words_after[idx]} ({d:+d})")
            print(f"      {DIM}{original[idx][:120]!r}{RESET}")
        return 1
    print(f"  {GREEN}word count preserved{RESET}")

    # Not silent: these tokens really are gone from the text, and the line above
    # deliberately does not count them. Say so rather than let the two numbers
    # agree for a reason nobody can see.
    zero_width = original.map(lambda t: len(t.split())).sum() - total_before
    if zero_width:
        print(f"{DIM}  ({zero_width} zero-width-only token(s) dropped — no readable "
              f"word removed){RESET}")

    if args.dry_run:
        print(f"\n{YELLOW}--dry-run: nothing written.{RESET}")
        return 0

    stale = stale_columns(df)
    if stale and changed and not args.force:
        print(f"\n{RED}Refusing to write: {len(stale)} column(s) already hold values "
              f"computed from the old text:{RESET}")
        for c in stale[:8]:
            print(f"    {c}")
        if len(stale) > 8:
            print(f"    ... and {len(stale) - 8} more")
        print("  Standardising would leave them describing text that is no longer "
              "here.\n  Run this stage before score and embed, or pass --force and "
              "re-run both after.")
        return 1
    if stale and changed:
        print(f"\n{YELLOW}--force: {len(stale)} score/embedding column(s) now describe "
              f"the old text. Re-run score and embed.{RESET}")

    raw_col = args.raw_column or f"{args.text_column}{RAW_SUFFIX}"
    keep_raw = not args.no_raw and raw_col not in df.columns

    if args.no_raw:
        print(f"\n{YELLOW}--no-raw: the original text is not kept, so this "
              f"transform is irreversible from this file alone.{RESET}")
    elif not keep_raw:
        # A second run must not clobber it. The column already holds the text
        # as it was BEFORE the first standardisation; overwriting it with the
        # current text would replace the true original with an
        # already-standardised copy and quietly make the transform
        # irreversible -- the exact thing keeping it is for.
        print(f"\n{DIM}  {raw_col} already present — left as it is; it holds the "
              f"text from before the first standardisation.{RESET}")
    else:
        df[raw_col] = original
        print(f"\n{GREEN}Kept{RESET} the original text in {raw_col!r}")

    df[args.text_column] = standardised

    if keep_raw:
        # Beside the column it describes rather than appended at the end, so
        # `df.iloc[:, :6]` shows the pair. Done after both assignments: a
        # reindexed frame is a new object, and assigning into it afterwards is
        # what raises SettingWithCopyWarning.
        cols = list(df.columns)
        cols.insert(cols.index(args.text_column) + 1, cols.pop(cols.index(raw_col)))
        df = df[cols]

    dest = args.output or args.parquet
    save_parquet(df, dest)
    print(f"{GREEN}Wrote{RESET} {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

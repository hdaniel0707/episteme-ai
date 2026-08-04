r"""Cut every document to a fixed length, so accuracy can be plotted against it.

The length-ablation companion to ``normalize_text.py``. In-domain detection sits
at ~98% on the full documents; the question this script exists to answer is how
much of that survives when the classifier is shown only the first N characters
(or words) of each text. Run it once per length, re-embed, re-run the same
matrix, and the result is a curve rather than a single number:

    PREDICTION  the diagonal falls monotonically as N shrinks and meets the
                transfer cells somewhere. Where it meets them is the interesting
                number. If the diagonal stays at 98% down to a couple of hundred
                characters, whatever it is keyed on lives in the opening of the
                document -- a header, a title convention, a punctuation habit --
                and not in the writing.

Truncation is a strictly destructive transform: nothing is added, no text is
rewritten, and the first N units of the output are byte-identical to the first N
units of the input. That is the point -- any drop in accuracy is attributable to
missing text and to nothing else.

Output policy: a NEW parquet is written (never in place), named
``<stem>_red_<length>.parquet``, and every column that was computed from the
full text is DROPPED rather than renamed:

  text            the truncated text
  emb_*           DROPPED -- vectors of the full text
  binoculars*     DROPPED -- a score of the full text
  llm_judge*      DROPPED -- a score of the full text
  raw_*           DROPPED -- the same, under normalize_text.py's prefix
  raw_text        DROPPED unless --keep-raw-text (it is the untruncated text)
  truncated       ADDED -- bool, whether this row actually lost any text

Dropped, not prefixed, because unlike normalisation there is no reading under
which a full-text score belongs in a short-text experiment: a Binoculars number
computed over 800 words says nothing about the 200 words that remain, so keeping
it under any name only invites a run that silently mixes the two regimes. The
consequence is deliberate and breaking -- a config with ``binoculars: "binoculars_score"``
(the ``binox_mlp`` entries) or ``binoculars_baseline``/``llm_judge_baseline``
enabled will fail loudly on a truncated parquet until those scores are
recomputed on the short text or the entries are removed.

``--drop-short`` matters more than it looks. Without it, a corpus whose
documents are mostly shorter than N comes out barely changed, and the curve
flattens for a reason that has nothing to do with detection: most rows were
never truncated. With it, every surviving row is exactly N long, so the x-axis
means what it says -- at the cost of dropping rows, unevenly across classes if
the classes differ in length (they do: wp_gpt54mini machine texts average 771
words against 532 for its humans). The report prints the class balance before
and after so that cost is visible before it is paid.

Usage:
    # one length, the six normalised corpora
    uv run python epai/ai_detection/prepare/truncate_text.py \
        data/parquet/essay_gpt54mini_binox0.parquet \
        data/parquet/reuter_gpt54mini_binox0.parquet \
        data/parquet/wp_gpt54mini_binox0.parquet \
        --length 200 --unit words

    # one length, the six normalised corpora
    uv run python epai/ai_detection/prepare/truncate_text.py \
        data/parquet/essay_binox0_norm_binox0.parquet \
        data/parquet/reuter_binox0_norm_binox0.parquet \
        data/parquet/wp_binox0_norm_binox0.parquet \
        data/parquet/essay_gpt54mini_binox0_norm_binox0.parquet \
        data/parquet/reuter_gpt54mini_binox0_norm_binox0.parquet \
        data/parquet/wp_gpt54mini_binox0_norm_binox0.parquet \
        --length 1000

    # the whole sweep in one go: one output parquet per input per length
    uv run python epai/ai_detection/prepare/truncate_text.py \
        data/parquet/essay_gpt54mini_binox0_norm_binox0.parquet \
        --length 250 500 1000 2000 4000

    # words instead of characters -> *_red_200w.parquet
    uv run python epai/ai_detection/prepare/truncate_text.py \
        data/parquet/wp_binox0_norm_binox0.parquet --length 200 --unit words

    # every surviving row exactly 1000 characters long
    uv run python epai/ai_detection/prepare/truncate_text.py \
        data/parquet/wp_binox0_norm_binox0.parquet --length 1000 --drop-short

    # look first, write nothing
    uv run python epai/ai_detection/prepare/truncate_text.py \
        data/parquet/essay_binox0_norm_binox0.parquet --length 500 --dry-run

Then, for every output, re-embed before training:
    uv run python epai/ai_detection/embed/add_embeddings.py \
        data/parquet/essay_binox0_norm_binox0_red_1000.parquet --cpu-threads 16 --gpu 0

and point a copy of the training config's ``datasets[].parquet_path`` at the new
files -- one config per length, so each run yields one point of the curve.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.parquet_utils import load_parquet, save_parquet

BASE_DIR = Path(__file__).parent.parent.parent.parent

TEXT_COL = "text"
RAW_TEXT_COL = "raw_text"
TRUNCATED_COL = "truncated"
TARGET_COL = "generated_by"
HUMAN_LABEL = "human"

UNITS = ("chars", "words")
# Suffix tag per unit, appended to the length in the output name. Characters get
# no tag, so the default run produces exactly ``_red_<length>``.
UNIT_TAG = {"chars": "", "words": "w"}

# Columns computed FROM the full text, and therefore meaningless once the text
# is cut. Matched as a case-insensitive substring, which also catches the
# ``raw_`` prefixed copies normalize_text.py leaves behind.
STALE_PREFIXES = ("emb_",)
STALE_SUBSTRINGS = ("binoculars", "binox", "llm_judge", "perplexity", "logprob")

WORD_RE = re.compile(r"\S+")
# The half word a character cut leaves behind: the last run of non-whitespace,
# together with the whitespace in front of it. Any whitespace counts, so a cut
# landing inside the first word of a line does not back off to the previous
# line's last space and keep a fragment of that line.
# \Z, not $: $ also matches before a final newline, which would make a cut
# landing right after one back off past the whole preceding word.
TRAILING_PARTIAL_RE = re.compile(r"\s\S*\Z")


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def truncate_chars(text: str, length: int, whole_words: bool) -> str:
    """First ``length`` characters, optionally backed off to a word boundary.

    Backing off avoids ending every document on half a word, which is a token
    the tokenizer has never seen in that position and a tell of its own -- one
    that would be present in every corpus at every length, but not equally, since
    corpora differ in word length. The back-off is skipped when it would leave
    nothing (a single token longer than the cut), where a hard cut is the lesser
    evil.
    """
    if len(text) <= length:
        return text
    cut = text[:length]
    if whole_words and not text[length].isspace():
        match = TRAILING_PARTIAL_RE.search(cut)
        if match and cut[: match.start()].strip():
            cut = cut[: match.start()]
    return cut.rstrip()


def truncate_words(text: str, length: int) -> str:
    """First ``length`` whitespace-delimited tokens, with the original spacing.

    Sliced out of the source string rather than rebuilt by ``" ".join``, so line
    breaks and paragraph structure inside the kept region survive exactly as
    they were. Layout is one of the things being measured here -- rebuilding it
    would flatten every corpus onto one convention and silently do part of
    normalize_text.py's job.
    """
    ends = [m.end() for m in WORD_RE.finditer(text)]
    if len(ends) <= length:
        return text
    return text[: ends[length - 1]]


def truncate(text: str, length: int, unit: str, whole_words: bool) -> str:
    if unit == "words":
        return truncate_words(text, length)
    return truncate_chars(text, length, whole_words)


def size(text: str, unit: str) -> int:
    """Length of ``text`` in the unit being truncated on."""
    return len(WORD_RE.findall(text)) if unit == "words" else len(text)


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def is_stale(col: str) -> bool:
    """True for a column whose value was computed from the untruncated text."""
    if any(col.startswith(p) for p in STALE_PREFIXES):
        return True
    return any(s in col.lower() for s in STALE_SUBSTRINGS)


def stale_columns(df: pd.DataFrame, keep_raw_text: bool) -> list[str]:
    cols = [c for c in df.columns if is_stale(c)]
    if RAW_TEXT_COL in df.columns and not keep_raw_text and RAW_TEXT_COL not in cols:
        cols.append(RAW_TEXT_COL)
    return cols


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def length_report(
    before: list[str], after: list[str], y: np.ndarray | None
) -> pd.DataFrame:
    """Document length in both units, before and after, split by class.

    Both units are reported whichever one was truncated on, because the cut is
    only exact in one of them: cutting at 1000 characters leaves a word count
    that still varies by corpus, and vice versa. If a length effect is to be
    read off the curve, the residual variation in the other unit is the first
    thing to check it against.
    """
    rows = []
    for unit in UNITS:
        pre = np.array([size(t, unit) for t in before], dtype=float)
        post = np.array([size(t, unit) for t in after], dtype=float)
        row: dict[str, object] = {"unit": unit}
        for name, arr in (("before", pre), ("after", post)):
            row[f"all_{name}"] = float(arr.mean()) if arr.size else float("nan")
            if y is not None and len(set(y.tolist())) == 2:
                h = float(arr[y == 0].mean())
                m = float(arr[y == 1].mean())
                row[f"human_{name}"] = h
                row[f"machine_{name}"] = m
                row[f"gap_{name}"] = m - h
        rows.append(row)
    return pd.DataFrame(rows)


def print_length_report(report: pd.DataFrame) -> None:
    if "gap_before" not in report.columns:
        print(report.to_string(index=False))
        return
    header = f"  {'unit':<10}{'human':>20}{'machine':>20}{'gap':>20}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in report.iterrows():
        print(
            f"  {r['unit']:<10}"
            f"{r['human_before']:>9.0f}->{r['human_after']:<10.0f}"
            f"{r['machine_before']:>9.0f}->{r['machine_after']:<10.0f}"
            f"{r['gap_before']:>+9.0f}->{r['gap_after']:<+10.0f}"
        )
    print(
        "\n  gap = machine - human. A gap that survives truncation is a length\n"
        "  difference the classifier can still use; --drop-short removes it in\n"
        "  the truncated unit by making every kept row the same length."
    )


def class_counts(df: pd.DataFrame) -> str:
    if TARGET_COL not in df.columns:
        return f"{len(df)} rows"
    counts = df[TARGET_COL].value_counts()
    return f"{len(df)} rows  (" + ", ".join(f"{k}: {v}" for k, v in counts.items()) + ")"


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def output_path(path: Path, length: int, unit: str, args: argparse.Namespace) -> Path:
    if args.out:
        return Path(args.out)
    suffix = args.suffix or f"red_{length}{UNIT_TAG[unit]}"
    return path.with_name(f"{path.stem}_{suffix}.parquet")


def process(path: Path, length: int, args: argparse.Namespace) -> None:
    unit = args.unit
    out_path = output_path(path, length, unit, args)

    print("=" * 88)
    print(f"Input      : {path}")
    print(f"Output     : {out_path}{'   (DRY RUN — not written)' if args.dry_run else ''}")
    print(f"Cut to     : {length} {unit}"
          f"{'' if unit == 'words' or not args.whole_words else ', backed off to a word boundary'}")
    print(f"Short rows : {'DROPPED' if args.drop_short else 'kept as they are'}")
    print("=" * 88)

    df = load_parquet(path)
    if TEXT_COL not in df.columns:
        raise SystemExit(f"{path.name}: missing column {TEXT_COL!r}.")
    if TRUNCATED_COL in df.columns and not args.force:
        raise SystemExit(
            f"{path.name} already has a {TRUNCATED_COL!r} column — it looks like "
            "the output of this script. Truncating twice gives a file whose name "
            "states one length and whose text is another; pass --force if that "
            "is really what you want."
        )

    before = df[TEXT_COL].fillna("").astype(str).tolist()
    after = [truncate(t, length, unit, args.whole_words) for t in before]

    y = None
    if TARGET_COL in df.columns:
        y = (df[TARGET_COL] != HUMAN_LABEL).astype(int).to_numpy()

    # --- what changed -------------------------------------------------------
    was_truncated = np.array([a != b for a, b in zip(before, after)], dtype=bool)
    chars_before = sum(len(t) for t in before)
    chars_after = sum(len(t) for t in after)
    emptied = [i for i, (a, b) in enumerate(zip(before, after)) if a.strip() and not b.strip()]

    n = max(len(df), 1)
    print(f"\nrows                 : {len(df)}")
    print(f"rows truncated       : {int(was_truncated.sum())}  "
          f"({100.0 * was_truncated.sum() / n:.1f}%)")
    print(f"characters           : {chars_before:,} -> {chars_after:,} "
          f"({chars_after - chars_before:+,})")
    if emptied:
        print(f"WARNING  {len(emptied)} row(s) became empty; row indices: {emptied[:20]}")

    if y is not None:
        for cls, name in ((0, "human"), (1, "machine")):
            sel = np.flatnonzero(y == cls)
            if sel.size == 0:
                continue
            print(f"  {name:<8} rows truncated: {int(was_truncated[sel].sum())}/{sel.size}")

    if was_truncated.mean() < args.min_truncated_frac:
        print(
            f"\nWARNING  only {100.0 * was_truncated.mean():.1f}% of rows are longer "
            f"than {length} {unit}, so this file is nearly unchanged. A point on "
            "the curve\n         drawn from it measures almost the same corpus as "
            "the full-length run — cut shorter, or pass --drop-short so the kept "
            "rows really are\n         the stated length."
        )

    report = length_report(before, after, y)
    print("\nDocument length, before -> after")
    print_length_report(report)

    stale = stale_columns(df, args.keep_raw_text)
    if args.dry_run:
        print("\nColumn plan")
        print(f"  {TEXT_COL!r} <- truncated text")
        print(f"  {TRUNCATED_COL!r} <- added (bool)")
        if stale:
            print(f"  drop   ({len(stale)}): {stale}")
        if args.drop_short:
            keep = np.array([size(t, unit) >= length for t in before], dtype=bool)
            print(f"\n  --drop-short would keep {int(keep.sum())}/{len(df)} rows.")
        print("\nDry run — nothing written.\n")
        return

    # --- build the output ---------------------------------------------------
    out = df.copy()
    out[TEXT_COL] = after
    out[TRUNCATED_COL] = was_truncated

    if stale:
        out = out.drop(columns=stale)
        print(f"\nDropped {len(stale)} column(s) describing the full text: {stale}")
        print("  Re-run epai/ai_detection/embed/add_embeddings.py on the output "
              "before training, and remove or recompute any Binoculars / LLM-judge\n"
              "  entry in the config — it names a column this file no longer has.")

    if args.drop_short:
        keep = np.array([size(t, unit) >= length for t in before], dtype=bool)
        print(f"\n--drop-short: {class_counts(out)}")
        out = out.loc[keep].reset_index(drop=True)
        print(f"          ->  {class_counts(out)}")
        if len(out) < args.min_rows:
            print(f"WARNING  only {len(out)} rows left, below --min-rows "
                  f"({args.min_rows}). The training preflight will refuse this file.")

    print()
    save_parquet(out, out_path)

    if args.report_dir:
        report_dir = Path(args.report_dir) / out_path.stem
        report_dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_dir / "length_report.csv", index=False)
        print(f"Report -> {report_dir}/")
    print()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquets", nargs="+", type=Path, help="Parquet file(s) to truncate.")
    ap.add_argument(
        "-n", "--length", nargs="+", type=int, required=True,
        help="Keep the first N units of each text. Several values run the whole "
        "sweep: one output parquet per input per length.",
    )
    ap.add_argument(
        "--unit", choices=UNITS, default="chars",
        help="Unit the length is counted in (default: chars). 'words' counts "
        "whitespace-delimited tokens and tags the output name with 'w', so the "
        "two units cannot overwrite each other.",
    )
    ap.add_argument(
        "-o", "--out", default=None,
        help="Explicit output path. Only valid with a single input file and a "
        "single length.",
    )
    ap.add_argument(
        "--suffix", default=None,
        help="Suffix for the output stem (default: 'red_<length>'). Only valid "
        "with a single length, since it is the length that distinguishes the "
        "outputs of a sweep.",
    )
    ap.add_argument(
        "--drop-short", action="store_true",
        help="Drop rows shorter than the target, so every kept row is exactly "
        "that length. Removes the 'most rows were never truncated' confound at "
        "the cost of rows, unevenly across classes if they differ in length.",
    )
    ap.add_argument(
        "--no-whole-words", dest="whole_words", action="store_false",
        help="Cut at exactly --length characters even mid-word (--unit chars "
        "only; the default backs off to the last word boundary).",
    )
    ap.add_argument(
        "--keep-raw-text", action="store_true",
        help=f"Keep an existing {RAW_TEXT_COL!r} column. It holds full-length "
        "text, so anything trained on it is not a short-text result.",
    )
    ap.add_argument(
        "--min-truncated-frac", type=float, default=0.5,
        help="Warn when fewer than this fraction of rows were actually longer "
        "than the target (default: 0.5).",
    )
    ap.add_argument(
        "--min-rows", type=int, default=50,
        help="Warn when --drop-short leaves fewer than this many rows (default: "
        "50, the training preflight's min_rows).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change; write nothing.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help=f"Truncate even if a {TRUNCATED_COL!r} column is already present.",
    )
    ap.add_argument(
        "--report-dir", default=str(BASE_DIR / "results" / "truncation"),
        help="Where to write the per-file length_report.csv.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.out and (len(args.parquets) > 1 or len(args.length) > 1):
        raise SystemExit("-o/--out only makes sense with a single input file and length.")
    if args.suffix and len(args.length) > 1:
        raise SystemExit("--suffix only makes sense with a single length.")
    if any(n <= 0 for n in args.length):
        raise SystemExit("--length must be positive.")

    print(f"truncate_text.py  —  {datetime.now():%Y-%m-%d %H:%M:%S}")
    for path in args.parquets:
        if not path.is_file():
            print(f"SKIPPED — not found: {path}")
            continue
        for length in args.length:
            process(path, length, args)


if __name__ == "__main__":
    main()

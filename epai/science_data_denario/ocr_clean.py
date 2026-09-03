r"""Stage: repair PDF-extraction / OCR artifacts in a detector corpus's text.

WHY THIS EXISTS
---------------
The Denario machine abstracts were pulled out of PDFs
(``epai/ai_detection/fetch/fetch_denario.py`` :: ``extract_abstract``). PDF text
extraction reliably produces three artifacts that clean, tokenised text (the
OpenAlex human class) never has:

  * runs of words with the inter-word spaces dropped, almost always in the
    opening sentence — e.g. ``Understandingtheevolutionofdarkmatterhalo...``
  * lost ligatures (``efﬁciency`` -> ``eﬃciency`` or worse)
  * a space before ``.,;:`` and after ``(``

info/iclr27_plan.html finding 5 shows the judge's ``errors`` cue is already
partly reading exactly this kind of damage — so left in place it is a class
label, not a fact about who wrote the text.

Like ``standardize_text.py`` this runs over the WHOLE ``text`` column, both
classes. On the clean side every rule is a no-op; applying it to one side only
would just be a new artifact pointing the other way.

WHAT IT DOES NOT DO
-------------------
It never removes text by keyword — no section labels, no boilerplate, no
sentence deletion. Run-together spans are SPLIT (word count goes up), hyphenated
line-wraps are JOINED (word count goes down); nothing else changes the words.
A span it cannot segment into known words is left exactly as it was and counted
in the per-class report, so an incomplete repair is visible rather than silent.

WORD SEGMENTATION
-----------------
A run-together span is split only when it can be covered *entirely* by words the
vocabulary already knows, via a Viterbi search over unigram log-probabilities.
The vocabulary is built from the corpus itself (both classes) plus
``--extra-text-parquet`` plus a small function-word seed — which is what makes it
reliable here: ``Plasmodiumfalciparum`` splits because ``Plasmodium`` and
``falciparum`` occur, spaced, elsewhere in the same corpus. A span with any
fragment the vocabulary does not know is left alone.

OUTPUT
------
``text`` rewritten; ``text_preclean`` added (the text before this stage), unless
already present. Refuses to run once embeddings or judge/detector scores exist —
they describe the pre-repair text. ``-o`` writes elsewhere.

Run:
    python -m epai.science_data_denario.ocr_clean data/parquet/denario_hc_raw.parquet \
        --extra-text-parquet data/science_denario/human_candidates.parquet \
        -o data/parquet/denario_hc_clean.parquet
    python -m epai.science_data_denario.ocr_clean <parquet> --dry-run
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from epai.utils.parquet_utils import load_parquet, save_parquet

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
RESET = "\033[0m"

TEXT_COL = "text"
PRECLEAN_COL = "text_preclean"
CLASS_COL = "generated_by"

# Columns whose values describe the pre-repair text. Same guard as
# standardize_text.py, same reason.
STALE_PREFIXES = ("emb_", "binoculars", "fastdetect", "detector_score",
                  "detector_ntok", "llm_judge")

# Ligature glyphs that survive extraction as a single codepoint. standardize_text
# folds these too; doing it here first only means the segmenter sees ASCII.
LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st",
}

_TOKEN_RE = re.compile(r"[a-z]+")
_SEED_WORDS = (
    "the of and to in a is that for we this with are as was by on it be from an "
    "at which has have not or their its these can our were also been but more "
    "than when using used such other into between both during each all one two "
    "no if then so may our study results show shown present propose approach "
    "method model data based paper here new"
).split()

# The segmenter's PIECE vocabulary is deliberately short-biased: a corpus word
# longer than this is kept as a candidate piece only if the corpus shows it at
# least twice, so a one-off run-together span ("convectiondriven") cannot become
# a piece it splits into elsewhere. Long words still get PROTECTED from being
# split (Vocab.real_words), just not used as glue.
_LONG_WORD = 12
_MIN_COUNT_LONG = 2
_REAL_WORD_MIN_COUNT = 3

# Short pieces allowed in a split; every other piece must be >= 4 chars, which
# together with _BAD_PIECE rejects fragments ("tr"/"cy") and peeled suffixes
# ("ally"/"ion"). This list is the closed set of real words short enough that
# the length rule would otherwise drop them.
_SHORT_OK = frozenset(
    "a i of to in is it we on an at or by be as my no so up he us "
    "the and for was are has its not but can our new two one all may who how".split()
)

# A run of <prefix><real word> is a single word ("underestimation",
# "counterintuitively", "nonlinear", "multidimensional") — never split it, even
# though both halves are words. "high"/"coarse"/"population" are NOT prefixes,
# so "highdimensional" still splits.
_PREFIXES = frozenset(
    "under over counter non anti multi inter intra trans semi pseudo hyper hypo "
    "micro macro nano meta proto poly mono bio geo neuro psycho socio electro "
    "thermo photo hydro auto self co re un pre post sub super supra ultra "
    "spatio spatiotemporal ethno equi para quasi".split()
)
# Morphological fragments the segmenter peels off real words ("computationally"
# -> "computation ally", "transcriptomic" -> "transcript omic"). A split with
# any of these as a piece is rejected.
_BAD_PIECE = frozenset(
    "ally ion ions ism isms ist ists ation ations ator ators ical ically ology "
    "ologies ness ment ments tion tions tional sional ational sions sion isa "
    "isation ization ized ised izing ising ing ings able ible omic omics ome "
    "seq ese ian eous ious ity ties ives ely".split()
)


class Vocab:
    """Corpus unigram counts, for the Viterbi word-segmenter.

    Two vocabularies, doing opposite jobs:

      ``counts``      candidate PIECES — short and well-attested words only, so
                      the Viterbi search glues a run together out of things that
                      are really words.
      ``real_words``  words a run is not allowed to be split INTO existence
                      against — the full system dictionary plus anything the
                      corpus shows >= 3 times. ``interpretability`` is here, so
                      it is never split to ``interpret ability``.
    """

    def __init__(self, counts: Counter, real_words: set[str]):
        self.counts = counts
        self.real_words = real_words
        self.total = sum(counts.values()) or 1
        self.maxlen = min(max((len(w) for w in counts), default=1), 24)

    def known(self, word: str) -> bool:
        return word.lower() in self.real_words

    @classmethod
    def build(cls, texts, system_dict: "Path | None" = None,
              min_len: int = 2, max_len: int = 24) -> "Vocab":
        raw: Counter = Counter()
        for text in texts:
            for word in _TOKEN_RE.findall(str(text).lower()):
                if min_len <= len(word) <= max_len:
                    raw[word] += 1
        counts = Counter({
            w: c for w, c in raw.items()
            if len(w) <= _LONG_WORD or c >= _MIN_COUNT_LONG
        })
        real_words = {w for w, c in raw.items() if c >= _REAL_WORD_MIN_COUNT}
        if system_dict and Path(system_dict).is_file():
            for line in Path(system_dict).read_text(errors="ignore").splitlines():
                w = line.strip().lower()
                if not w.isalpha():
                    continue
                real_words.add(w)
                # As a piece, trust the system list only for short glue words —
                # it carries no frequency, and a long entry there is as likely
                # to enable a wrong split as a right one.
                if min_len <= len(w) <= _LONG_WORD and w not in counts:
                    counts[w] = 1
        for word in _SEED_WORDS:
            counts[word] += 1
        return cls(counts, real_words)


def _piece_ok(word: str, vocab: Vocab) -> bool:
    """Is ``word`` allowed to be one piece of a split?

    A real word (>= 4 chars, attested or in the dictionary, not a bare
    morphological fragment) or one of the closed set of short glue words.
    """
    if word in _SHORT_OK:
        return True
    if word in _BAD_PIECE or len(word) < 4:
        return False
    return vocab.known(word) or vocab.counts.get(word, 0) >= _MIN_COUNT_LONG


def segment(run: str, vocab: Vocab) -> list[str]:
    """Highest-probability split of ``run`` into ALLOWED pieces, or ``[run]``.

    Constrained Viterbi: a transition is taken only through a word that is both
    in the piece vocabulary (for its probability) and passes :func:`_piece_ok`,
    so the path returned is the best one whose every piece is acceptable — not
    the globally best path, which might peel a suffix off a real word and then
    be thrown away. ``[run]`` means no all-allowed cover exists.
    """
    n = len(run)
    lower = run.lower()
    best = [-math.inf] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(0, i - vocab.maxlen), i):
            word = lower[j:i]
            count = vocab.counts.get(word)
            if not count or not _piece_ok(word, vocab):
                continue
            score = best[j] + math.log(count / vocab.total)
            if score > best[i]:
                best[i] = score
                back[i] = j
    if best[n] == -math.inf:
        return [run]
    pieces: list[str] = []
    i = n
    while i > 0:
        pieces.append(run[back[i]:i])
        i = back[i]
    return pieces[::-1]


def clean_ocr_text(
    text: str, vocab: Vocab, min_run: int = 14
) -> tuple[str, Counter, list[str], list[tuple[str, str]]]:
    """Return ``(repaired_text, fix_counts, unresolved_runs, applied_splits)``."""
    fixes: Counter = Counter()
    unresolved: list[str] = []
    splits: list[tuple[str, str]] = []

    out = text.replace("�", "")
    if out != text:
        fixes["replacement_char"] += 1

    for glyph, repl in LIGATURES.items():
        if glyph in out:
            fixes["ligature"] += out.count(glyph)
            out = out.replace(glyph, repl)

    # Markdown emphasis carried over from the Denario source ("*in vivo*",
    # "*Plasmodium falciparum*", sometimes glued: "of*Plasmodium"). The OpenAlex
    # human class has none, so a stray ``*`` is a class label. Every ``*`` run
    # becomes a space; ``_`` is left alone (it carries subscripts — ``Omega_m``).
    unstarred = re.sub(r"\*+", " ", out)
    if unstarred != out:
        fixes["markdown_emphasis"] += 1
        out = unstarred

    # hyphenated line-wrap: "pro-\ncess" -> "process". extract_abstract already
    # does the common case; this catches what is left in other corpora.
    joined = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", out)
    if joined != out:
        fixes["dehyphenate"] += 1
        out = joined

    # space before closing punctuation, space after opening bracket
    stripped = re.sub(r"[ \t]+([,;:.!?%])", r"\1", out)
    stripped = re.sub(r"([(\[])[ \t]+", r"\1", stripped)
    if stripped != out:
        fixes["space_around_punct"] += 1
        out = stripped

    # missing space after mid-sentence punctuation: "success.We" -> "success. We".
    # Both sides constrained so abbreviations ("e.g.", "et al.") and initials
    # are not touched.
    def _add_space(m: re.Match) -> str:
        fixes["missing_space_after_punct"] += 1
        return f"{m.group(1)}{m.group(2)} {m.group(3)}"

    out = re.sub(r"([a-z]{3})([.;:,!?])([A-Z][a-z]+)", _add_space, out)

    # run-together words. ``interpretability`` stays whole (it is a real word);
    # ``<prefix><word>`` stays whole (one derivation, not a compound); everything
    # else is split only if segment() finds an all-allowed cover.
    def _split(m: re.Match) -> str:
        run = m.group(0)
        low = run.lower()
        if vocab.known(run):
            return run
        # <prefix><real word> is one derived word ("underestimation",
        # "nonlinear"), not a compound — the remainder must itself be a word,
        # or every "re..."/"co..." run would be frozen.
        if any(low.startswith(p) and vocab.known(low[len(p):]) for p in _PREFIXES):
            return run
        pieces = segment(run, vocab)
        if len(pieces) < 2:
            unresolved.append(run)
            return run
        fixes["word_segmentation"] += 1
        joined = " ".join(pieces)
        joined = joined[0].upper() + joined[1:] if run[:1].isupper() else joined
        splits.append((run, joined))
        return joined

    out = re.sub(rf"[A-Za-z]{{{min_run},}}", _split, out)

    out = re.sub(r"[ \t]{2,}", " ", out)
    return out, fixes, unresolved, splits


def stale_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if any(c.startswith(p) for p in STALE_PREFIXES) and df[c].notna().any()
    ]


def per_class_report(
    df: pd.DataFrame,
    fixes: list[Counter],
    unresolved: list[list[str]],
    splits: list[list[tuple[str, str]]],
) -> None:
    groups = (
        df.groupby(CLASS_COL).groups if CLASS_COL in df.columns
        else {"(all rows)": df.index}
    )
    kinds = sorted({k for f in fixes for k in f})
    width = max((len(str(g)) for g in groups), default=8) + 2
    print(f"\n  {'class':<{width}}{'rows':>6}" + "".join(f"{k[:16]:>18}" for k in kinds)
          + f"{'unresolved':>12}")
    for name, idx in groups.items():
        pos = [df.index.get_loc(i) for i in idx]
        row = f"  {str(name):<{width}}{len(pos):>6}"
        for k in kinds:
            row += f"{sum(fixes[p][k] for p in pos):>18}"
        row += f"{sum(len(unresolved[p]) for p in pos):>12}"
        print(row)

    every_split = sorted({s for row in splits for s in row})
    if every_split:
        print(f"\n{DIM}  word splits applied ({len(every_split)} distinct) — "
              f"eyeball these:{RESET}")
        for run, joined in every_split[:40]:
            print(f"{DIM}    {run:<40} -> {joined}{RESET}")
        if len(every_split) > 40:
            print(f"{DIM}    ... {len(every_split) - 40} more{RESET}")
    leftover = sorted({r for u in unresolved for r in u})
    if leftover:
        print(f"\n{DIM}  spans left unsegmented (no full known-word cover): "
              f"{leftover}{RESET}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("parquet", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--extra-text-parquet", type=Path, action="append", default=[],
                    help="another parquet whose 'abstract'/'text' column feeds the "
                         "segmenter vocabulary (repeatable)")
    ap.add_argument("--min-run", type=int, default=14,
                    help="only alphabetic runs at least this long are considered "
                         "for splitting (default 14)")
    ap.add_argument("--system-dict", type=Path,
                    default=Path("/usr/share/dict/words"),
                    help="plain word list, one per line, added to the segmenter "
                         "vocabulary as short glue words only (default "
                         "/usr/share/dict/words if present; pass a missing path "
                         "to disable)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="run even though score/embedding columns are present")
    args = ap.parse_args()

    if not args.parquet.is_file():
        print(f"{RED}not found: {args.parquet}{RESET}")
        return 1
    df = load_parquet(args.parquet)
    if TEXT_COL not in df.columns:
        print(f"{RED}no {TEXT_COL!r} column{RESET}")
        return 1

    vocab_texts = list(df[TEXT_COL].fillna(""))
    for extra in args.extra_text_parquet:
        edf = load_parquet(extra)
        col = "abstract" if "abstract" in edf.columns else TEXT_COL
        vocab_texts += list(edf[col].fillna(""))
    vocab = Vocab.build(vocab_texts, system_dict=args.system_dict)
    print(f"segmenter vocabulary: {len(vocab.counts):,} words from "
          f"{len(vocab_texts):,} texts"
          + (f" + {args.system_dict}" if args.system_dict.is_file() else ""))

    original = df[TEXT_COL].fillna("").astype(str)
    results = [clean_ocr_text(t, vocab, args.min_run) for t in original]
    repaired = [r[0] for r in results]
    fixes = [r[1] for r in results]
    unresolved = [r[2] for r in results]
    splits = [r[3] for r in results]

    changed = sum(a != b for a, b in zip(original, repaired))
    w_before = sum(len(t.split()) for t in original)
    w_after = sum(len(t.split()) for t in repaired)
    print(f"rows changed: {changed}/{len(df)}   words: {w_before} -> {w_after} "
          f"({w_after - w_before:+d})")
    per_class_report(df, fixes, unresolved, splits)

    if args.dry_run:
        print(f"\n{YELLOW}--dry-run: nothing written.{RESET}")
        return 0

    stale = stale_columns(df)
    if stale and changed and not args.force:
        print(f"\n{RED}refusing to write: {stale[:6]} already describe the "
              f"pre-repair text. Run this before embed/score, or pass --force "
              f"and recompute them.{RESET}")
        return 1

    if PRECLEAN_COL not in df.columns:
        df[PRECLEAN_COL] = original
        cols = list(df.columns)
        cols.insert(cols.index(TEXT_COL) + 1, cols.pop(cols.index(PRECLEAN_COL)))
        df = df[cols]
    else:
        print(f"{DIM}  {PRECLEAN_COL} already present — kept as it was{RESET}")
    df[TEXT_COL] = repaired

    dest = args.output or args.parquet
    save_parquet(df, dest)
    print(f"{GREEN}wrote{RESET} {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

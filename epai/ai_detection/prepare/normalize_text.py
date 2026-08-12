r"""Normalise corpus text so the surface artifacts are gone, then re-embed.

Step 2 of the "why is in-domain 98% and transfer below chance" investigation,
and the decisive experiment: strip the typographic and layout tells that
``analyse_surface_artifacts.py`` inventories, re-embed, and re-run the identical
cross-domain matrix.

    PREDICTION  the below-chance cells (macro-F1 0.33-0.37) move up to ~0.50 and
                the diagonal drops. If the diagonal stays at 98% there is a
                second artifact and you go back to step 1.

The point of normalising rather than deleting features is that a curly
apostrophe and a straight one are different token ids to every tokenizer in the
pipeline, so the distinction survives into the embedding whether or not it means
anything. Mapping them onto one character is the smallest change that removes
the tell while leaving the writing intact.

Transforms, each independently switchable so the effect can be attributed:

  unicode      typographic punctuation -> ASCII (curly quotes, dashes, ellipsis,
               exotic spaces, ligatures, soft hyphens). The main event.
  headers      leading "Introduction:" / "Title:" labels, markdown heading
               marks and ** emphasis. Boilerplate that lives in the opening
               characters, which is exactly what a 512-token encoder sees.
  whitespace   CRLF -> LF, leading and trailing whitespace per line, runs of
               spaces, runs of blank lines, whitespace at either end of the
               document.
  ascii_only   drop whatever non-ASCII is left. OFF by default: it also destroys
               accented names, which is real content, not an artifact.
  strip_title  drop a short leading line with no terminal punctuation. OFF by
               default: on some corpora that line is a real first sentence.
  reflow       unwrap hard-wrapped lines: join the lines inside each blank-line
               separated block into one line, keep the blank line between
               blocks. Kills the line-wrapping convention (72 columns vs one
               line per paragraph) while keeping paragraph structure. OFF by
               default -- it rewrites layout everywhere, not just where a tell
               exists, and a document that separates paragraphs with a bare
               newline instead of a blank line collapses to a single paragraph.
  flatten      the nuclear version: the whole document becomes one line. Line
               length and newline rate then carry exactly zero information in
               every corpus, by construction, rather than merely less. OFF by
               default. Applied after reflow, so asking for both gives flatten.

Controls (applied last, one at a time — they destroy the text on purpose):

  shuffle_words  shuffle the whitespace tokens. Destroys syntax, keeps the
                 vocabulary and the punctuation. If accuracy survives this, the
                 classifier was never reading sentences.
  punct_only     delete every letter and digit, keep punctuation and layout. If
                 this alone reaches ~90% in-domain, that is the headline result.
  letters_only   the complement: delete punctuation, keep the words.

Output policy: a NEW parquet is written (never in place by default), and every
column that describes the OLD text is made obvious in its name or removed.

  text                        the normalised text
  raw_text                    the original, kept verbatim
  raw_binoculars_score        was binoculars_score
  raw_llm_judge_score_<model> was llm_judge_score_<model>
  emb_*                       DROPPED

Scores keep their values and gain a ``raw_`` prefix, because a Binoculars or
judge score computed on the original text is still a valid number -- just not a
number about the text now sitting in the ``text`` column. Embeddings are dropped
rather than renamed because their whole purpose is to be re-computed; leaving
them would let a later run silently train on stale vectors. Re-run
``epai/ai_detection/embed/add_embeddings.py`` on the output before training.

The rename is deliberately breaking: anything configured against
``binoculars_score`` (``DEFAULT_BINOX_COL`` in the training scripts) will fail
loudly on a normalised parquet until the config names ``raw_binoculars_score``,
which is the intended prompt to decide whether that score still belongs in the
experiment.

Usage:
    # the default set (unicode + headers + whitespace), all six corpora
    uv run python epai/ai_detection/prepare/normalize_text.py \
        data/parquet/essay_binox0.parquet \
        data/parquet/reuter_binox0.parquet \
        data/parquet/wp_binox0.parquet \
        data/parquet/essay_gpt54mini_binox0.parquet \
        data/parquet/reuter_gpt54mini_binox0.parquet \
        data/parquet/wp_gpt54mini_binox0.parquet
    
    # THIS
    uv run python epai/ai_detection/prepare/normalize_text.py \
        data/parquet/essay_binox0.parquet \
        data/parquet/reuter_binox0.parquet \
        data/parquet/wp_binox0.parquet \
        data/parquet/essay_gpt54mini_binox0.parquet \
        data/parquet/reuter_gpt54mini_binox0.parquet \
        data/parquet/wp_gpt54mini_binox0.parquet \
        --transforms unicode headers whitespace reflow ascii_only strip_title \
        --suffix norm 

    # EVERYTHING ON, all six corpora -- copy-paste. Drop --dry-run to write.
    # --suffix norm keeps the output names predictable: point a copy of
    # cfg/ai_detection/train/clf_cd/clf_cd_example.yaml at the *_norm parquets.
    uv run python epai/ai_detection/prepare/normalize_text.py \
        data/parquet/essay_binox0.parquet \
        data/parquet/reuter_binox0.parquet \
        data/parquet/wp_binox0.parquet \
        data/parquet/essay_gpt54mini_binox0.parquet \
        data/parquet/reuter_gpt54mini_binox0.parquet \
        data/parquet/wp_gpt54mini_binox0.parquet \
        --transforms unicode headers whitespace reflow ascii_only strip_title \
        --length-match \
        --suffix norm \
        --dry-run

    # Same, but with flatten instead of reflow: line length and newline rate
    # are then identical in every corpus and provably carry no signal. Written
    # to *_flat.parquet so both versions can be compared side by side.
    uv run python epai/ai_detection/prepare/normalize_text.py \
        data/parquet/essay_binox0.parquet \
        data/parquet/reuter_binox0.parquet \
        data/parquet/wp_binox0.parquet \
        data/parquet/essay_gpt54mini_binox0.parquet \
        data/parquet/reuter_gpt54mini_binox0.parquet \
        data/parquet/wp_gpt54mini_binox0.parquet \
        --transforms unicode headers whitespace flatten ascii_only strip_title \
        --length-match \
        --suffix flat \
        --dry-run

    # Backing off, if the maximal run removes too much: `ascii_only` also
    # destroys accented names and `strip_title` can eat a real first sentence,
    # so the conservative version of the above is
    #     --transforms unicode headers whitespace reflow
    # and the minimal one is the default set, --transforms unicode headers whitespace.

    # punctuation only, as the artifact control
    uv run python epai/ai_detection/prepare/normalize_text.py \
        data/parquet/essay_binox0.parquet \
        --transforms unicode whitespace --control punct_only --suffix punct

    # dry run on one file: report what would change, write nothing
    uv run python epai/ai_detection/prepare/normalize_text.py \
        data/parquet/wp_gpt54mini_binox0.parquet --dry-run

Then, for each of the six outputs:
    uv run python epai/ai_detection/embed/add_embeddings.py \
        data/parquet/essay_binox0_norm.parquet --cpu-threads 16 --gpu 0
    uv run python epai/ai_detection/embed/add_embeddings.py \
        data/parquet/reuter_binox0_norm.parquet --cpu-threads 16 --gpu 0
    uv run python epai/ai_detection/embed/add_embeddings.py \
        data/parquet/wp_binox0_norm.parquet --cpu-threads 16 --gpu 0
    uv run python epai/ai_detection/embed/add_embeddings.py \
        data/parquet/essay_gpt54mini_binox0_norm.parquet --cpu-threads 16 --gpu 0
    uv run python epai/ai_detection/embed/add_embeddings.py \
        data/parquet/reuter_gpt54mini_binox0_norm.parquet --cpu-threads 16 --gpu 0
    uv run python epai/ai_detection/embed/add_embeddings.py \
        data/parquet/wp_gpt54mini_binox0_norm.parquet --cpu-threads 16 --gpu 0

Then re-run the matrix on both arms -- two copies of
cfg/ai_detection/train/clf_cd/clf_cd_example.yaml differing ONLY in the parquet
paths (raw vs *_norm) and in `alg_id`, so the seed, split and algorithms are held
fixed and the difference is the normalisation -- and re-run the step-1 audit on
the normalised parquets to confirm the artifacts are gone:
    uv run python epai/ai_detection/train/run_clf_cross_domain.py --cfg RAW.yaml NORM.yaml
    uv run python epai/ai_detection/analyse/analyse_surface_artifacts.py \
        --config NORM.yaml --tag norm
"""

from __future__ import annotations

import argparse
import random
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from epai.utils.parquet_utils import load_parquet, save_parquet

BASE_DIR = Path(__file__).parent.parent.parent.parent

TEXT_COL = "text"
RAW_TEXT_COL = "raw_text"
TARGET_COL = "generated_by"
HUMAN_LABEL = "human"

DEFAULT_TRANSFORMS = ("unicode", "headers", "whitespace")
ALL_TRANSFORMS = (
    "unicode",
    "headers",
    "whitespace",
    "ascii_only",
    "strip_title",
    "reflow",
    "flatten",
)
CONTROLS = ("shuffle_words", "punct_only", "letters_only")

# Columns computed FROM the text, which therefore describe the OLD text once it
# is normalised. emb_* is dropped (it exists to be recomputed); everything
# matching TEXT_DERIVED keeps its value and gains the RAW_PREFIX, so no column
# in the output can be mistaken for a measurement of the normalised text.
STALE_PREFIXES = ("emb_",)
RAW_PREFIX = "raw_"
TEXT_DERIVED = ("binoculars", "binox", "llm_judge", "perplexity", "logprob")


# ---------------------------------------------------------------------------
# Character maps
# ---------------------------------------------------------------------------

# Imported, not defined here. The pipelines standardise every corpus with this
# same table before scoring (epai/ai_detection/prepare/standardize_text.py), and
# a research tool that folded a slightly different set of characters than the
# corpora were built with would attribute the difference to its own transform.
# UNICODE_GROUPS stays grouped so the report below can name which class of
# substitution fired rather than just "n characters changed".
from epai.utils.text_standardize import (  # noqa: E402
    UNICODE_GROUPS,
    UNICODE_MAP,
    UNICODE_TABLE,
)

# Leading labels that Ghostbuster / gpt rewrites emit as a header line. The
# content after the colon is kept; only the label goes.
LEADING_LABEL_RE = re.compile(
    r"^\s*(?:\*{0,2})"
    r"(Title|Introduction|Abstract|Summary|Essay|Story|Article|Response|Answer)"
    r"(?:\*{0,2})\s*:\s*",
    re.IGNORECASE,
)
MARKDOWN_HEADING_RE = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*")
MARKDOWN_EMPHASIS_RE = re.compile(r"\*{2,3}")
TITLE_LINE_RE = re.compile(r"^([^\n]{1,80})\n")

BLOCK_SPLIT_RE = re.compile(r"\n[ \t]*\n")

MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
TRAILING_WS_RE = re.compile(r"(?m)[ \t]+$")
LEADING_WS_RE = re.compile(r"(?m)^[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

LETTER_DIGIT_RE = re.compile(r"[^\W_]", re.UNICODE)  # letters and digits
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def t_unicode(text: str) -> str:
    """Typographic punctuation -> ASCII, via the explicit table above.

    An explicit table rather than ``unicodedata.normalize('NFKC')`` because NFKC
    leaves curly quotes exactly as they are — it is not a punctuation folder —
    while also rewriting things nobody asked it to touch.
    """
    return text.translate(UNICODE_TABLE)


def t_ascii_only(text: str) -> str:
    """Drop whatever non-ASCII survives, after decomposing accents."""
    decomposed = unicodedata.normalize("NFKD", text)
    return decomposed.encode("ascii", "ignore").decode("ascii")


def t_headers(text: str) -> str:
    """Remove leading section labels and markdown heading / emphasis marks."""
    text = LEADING_LABEL_RE.sub("", text, count=1)
    text = MARKDOWN_HEADING_RE.sub("", text)
    text = MARKDOWN_EMPHASIS_RE.sub("", text)
    return text


def t_strip_title(text: str) -> str:
    """Drop a short opening line that is not a sentence.

    Off by default: in some corpora that line is a genuine first sentence, and
    removing it costs real content rather than boilerplate.
    """
    match = TITLE_LINE_RE.match(text)
    if not match:
        return text
    first = match.group(1).strip()
    if not first or first.endswith((".", "!", "?", ",", ";", ":")):
        return text
    if len(first.split()) > 12:
        return text
    return text[match.end() :]


def t_whitespace(text: str) -> str:
    """Line endings, per-line leading/trailing whitespace, runs of spaces and blank lines.

    Indentation goes too, not just trailing whitespace: whether a corpus indents
    its paragraphs is exactly the kind of layout convention that separates the
    Word-scraped humans from the newswire ones, and it has nothing to do with
    who wrote the text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = TRAILING_WS_RE.sub("", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = LEADING_WS_RE.sub("", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def t_reflow(text: str) -> str:
    """Unwrap hard-wrapped lines, keeping blank-line paragraph breaks.

    A blank line separates paragraphs; every other newline is treated as a wrap
    and joined away. That maps "wrapped at 72 columns" and "one long line per
    paragraph" onto the same normal form, which is the point: whether a corpus
    was stored wrapped says nothing about who wrote it.

    The cost, and the reason this is off by default: a document that separates
    its paragraphs with a bare newline rather than a blank line has no
    paragraph breaks by this definition and collapses to one line. That is a
    real loss of structure, not just of convention -- use ``flatten`` instead if
    you would rather every corpus lose it equally than have some keep it.
    """
    blocks = []
    for block in BLOCK_SPLIT_RE.split(text):
        joined = " ".join(line.strip() for line in block.split("\n") if line.strip())
        if joined:
            blocks.append(joined)
    return "\n\n".join(blocks)


def t_flatten(text: str) -> str:
    """Collapse the document to a single line of single-spaced words.

    After this, line count, line length and newline rate are identical in every
    corpus, so they cannot carry a signal at all. Blunter than ``reflow`` and
    more honest about it: paragraph structure is a real stylistic property, and
    this removes it from every corpus rather than from some of them.
    """
    return " ".join(text.split())


def c_shuffle_words(text: str, rng: random.Random) -> str:
    """Shuffle whitespace tokens: syntax destroyed, vocabulary and punctuation kept."""
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def c_punct_only(text: str, rng: random.Random) -> str:
    """Delete letters and digits; punctuation and layout survive."""
    return LETTER_DIGIT_RE.sub("", text)


def c_letters_only(text: str, rng: random.Random) -> str:
    """Delete punctuation; words and whitespace survive."""
    return PUNCT_RE.sub("", text)


TRANSFORM_FNS = {
    "unicode": t_unicode,
    "headers": t_headers,
    "whitespace": t_whitespace,
    "ascii_only": t_ascii_only,
    "strip_title": t_strip_title,
    "reflow": t_reflow,
    "flatten": t_flatten,
}
CONTROL_FNS = {
    "shuffle_words": c_shuffle_words,
    "punct_only": c_punct_only,
    "letters_only": c_letters_only,
}

# Order matters: fold the characters first, then strip the labels those
# characters may have been wrapped in, then rewrite the line layout, then tidy
# the whitespace every earlier step left behind. reflow runs before flatten so
# that asking for both is well defined (flatten wins), and both run after
# headers so a markdown heading is removed rather than absorbed into the
# paragraph next to it.
TRANSFORM_ORDER = (
    "unicode",
    "ascii_only",
    "headers",
    "strip_title",
    "reflow",
    "flatten",
    "whitespace",
)


def apply_transforms(
    text: str, transforms: set[str], control: str | None, seed: int, row_index: int
) -> str:
    """All requested transforms, in the fixed order, then the control."""
    for name in TRANSFORM_ORDER:
        if name in transforms:
            text = TRANSFORM_FNS[name](text)
    if control is not None:
        # Per-row seed: the shuffle must be reproducible and must not be the
        # same permutation for every document.
        rng = random.Random(seed * 1_000_003 + row_index)
        text = CONTROL_FNS[control](text, rng)
    return text


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def single_bit_accuracy(present: np.ndarray, y: np.ndarray) -> float:
    """Accuracy of the best rule using only 'does this document contain it'.

    The number the grep evidence was quoted in: "that single bit alone is ~93%
    accurate". After normalisation it should be at the majority baseline.
    """
    n = len(y)
    if n == 0:
        return float("nan")
    correct_pos = int(((present == 1) & (y == 1)).sum() + ((present == 0) & (y == 0)).sum())
    correct_neg = n - correct_pos
    return max(correct_pos, correct_neg) / n


def artifact_report(
    before: list[str], after: list[str], y: np.ndarray | None
) -> pd.DataFrame:
    """Per character group: prevalence by class, before and after.

    This is the check that the normalisation actually removed the tell, rather
    than the check that it ran.
    """
    rows = []
    for group, mapping in UNICODE_GROUPS.items():
        chars = "".join(mapping)
        rx = re.compile("[" + re.escape(chars) + "]")

        pre = np.array([bool(rx.search(t)) for t in before], dtype=int)
        post = np.array([bool(rx.search(t)) for t in after], dtype=int)

        row = {
            "group": group,
            "docs_before": int(pre.sum()),
            "docs_after": int(post.sum()),
            "chars_before": int(sum(len(rx.findall(t)) for t in before)),
            "chars_after": int(sum(len(rx.findall(t)) for t in after)),
        }
        if y is not None and len(set(y.tolist())) == 2:
            row |= {
                "presence_human_before": float(pre[y == 0].mean()),
                "presence_machine_before": float(pre[y == 1].mean()),
                "presence_human_after": float(post[y == 0].mean()),
                "presence_machine_after": float(post[y == 1].mean()),
                "single_bit_acc_before": single_bit_accuracy(pre, y),
                "single_bit_acc_after": single_bit_accuracy(post, y),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _line_metrics(text: str) -> dict[str, float]:
    """The layout quantities that showed up as the strongest sign flips."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return {
        "n_lines": float(len(lines)),
        "mean_line_length": float(np.mean([len(ln) for ln in lines])) if lines else 0.0,
        "newlines_per_1k": 1000.0 * text.count("\n") / max(len(text), 1),
        "blank_line_paras": float(len(BLOCK_SPLIT_RE.findall(text))),
    }


def line_structure_report(
    before: list[str], after: list[str], y: np.ndarray | None
) -> pd.DataFrame:
    """Line-layout stats by class, before and after — the reflow / flatten check.

    ``gap`` is machine minus human. ``mean_line_length`` and ``newlines_per_1k``
    were the two strongest sign-flipping features in the step-1 audit, so the
    thing to look for is their gap collapsing towards zero. Under ``flatten`` it
    must reach exactly zero; under ``reflow`` it only shrinks, because paragraph
    lengths still differ and that difference is real.
    """
    pre = [_line_metrics(t) for t in before]
    post = [_line_metrics(t) for t in after]

    rows = []
    for metric in ("n_lines", "mean_line_length", "newlines_per_1k", "blank_line_paras"):
        row: dict[str, object] = {"metric": metric}
        for name, values in (("before", pre), ("after", post)):
            arr = np.array([m[metric] for m in values], dtype=float)
            row[f"all_{name}"] = float(arr.mean())
            if y is not None and len(set(y.tolist())) == 2:
                h = float(arr[y == 0].mean())
                m = float(arr[y == 1].mean())
                row[f"human_{name}"] = h
                row[f"machine_{name}"] = m
                row[f"gap_{name}"] = m - h
        rows.append(row)
    return pd.DataFrame(rows)


def residual_non_ascii(texts: list[str], top_k: int = 15) -> list[tuple[str, int, str]]:
    """The non-ASCII characters still present, most common first."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(ch for ch in text if ord(ch) > 127)
    out = []
    for ch, n in counter.most_common(top_k):
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = "<unnamed>"
        out.append((f"U+{ord(ch):04X}", n, name))
    return out


# ---------------------------------------------------------------------------
# Length matching (optional)
# ---------------------------------------------------------------------------


def length_match_mask(texts: list[str], y: np.ndarray, caliper: float, seed: int) -> np.ndarray:
    """Boolean mask selecting a length-balanced human/machine subset.

    Greedy 1:1 nearest-neighbour matching on log word count, within ``caliper``
    log units. Written as a column rather than applied, so the full corpus stays
    intact and downstream code opts in.

    wp_gpt54mini is the corpus this exists for: its machine texts average 771
    words against 532 for its humans, so a length-only rule already separates
    them and any conclusion drawn there is confounded until length is matched.
    """
    counts = np.array([max(len(t.split()), 1) for t in texts], dtype=float)
    logs = np.log(counts)

    human_idx = list(np.flatnonzero(y == 0))
    machine_idx = list(np.flatnonzero(y == 1))

    rng = np.random.default_rng(seed)
    rng.shuffle(human_idx)

    available = sorted(machine_idx, key=lambda i: logs[i])
    avail_logs = np.array([logs[i] for i in available])
    used = np.zeros(len(available), dtype=bool)

    keep = np.zeros(len(texts), dtype=bool)
    for h in human_idx:
        distances = np.abs(avail_logs - logs[h])
        distances[used] = np.inf
        j = int(np.argmin(distances))
        if not np.isfinite(distances[j]) or distances[j] > caliper:
            continue
        used[j] = True
        keep[h] = True
        keep[available[j]] = True
    return keep


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def is_text_derived(col: str) -> bool:
    """True for a column whose value was computed from the text.

    Excludes anything already carrying the prefix, so re-running on an
    already-normalised file cannot produce ``raw_raw_binoculars_score``.
    """
    if col.startswith(RAW_PREFIX):
        return False
    return any(s in col.lower() for s in TEXT_DERIVED)


def rename_text_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix every text-derived score column with ``raw_``, and say so.

    Values are untouched. A score computed on the original text stays a valid
    number; the rename only stops it from being read as a measurement of the
    normalised text sitting in ``text``.
    """
    mapping = {c: RAW_PREFIX + c for c in df.columns if is_text_derived(c)}
    if not mapping:
        return df

    clashes = {old: new for old, new in mapping.items() if new in df.columns}
    if clashes:
        raise SystemExit(
            "Cannot rename — the target name already exists: "
            + ", ".join(f"{old} -> {new}" for old, new in clashes.items())
            + ".\nThat parquet has been through this script before. Sort the "
            "columns out first, or pass --keep-derived-names."
        )

    print(f"\nRenamed {len(mapping)} text-derived column(s) — the values are "
          "unchanged, they just describe the raw text:")
    for old, new in mapping.items():
        print(f"  {old}  ->  {new}")
    print("  Anything configured against the old name (e.g. "
          "DEFAULT_BINOX_COL = 'binoculars_score')\n  must be pointed at the new "
          "one before it will run on this file.")
    return df.rename(columns=mapping)


def default_suffix(transforms: set[str], control: str | None) -> str:
    if control is not None:
        return control
    if transforms == set(DEFAULT_TRANSFORMS):
        return "norm"
    return "_".join(sorted(transforms)) or "raw"


def process(path: Path, args: argparse.Namespace) -> None:
    transforms = set(args.transforms)
    control = args.control
    suffix = args.suffix or default_suffix(transforms, control)
    out_path = Path(args.out) if args.out else path.with_name(f"{path.stem}_{suffix}.parquet")

    print("=" * 88)
    print(f"Input      : {path}")
    print(f"Output     : {out_path}{'   (DRY RUN — not written)' if args.dry_run else ''}")
    print(f"Transforms : {[t for t in TRANSFORM_ORDER if t in transforms] or ['(none)']}")
    print(f"Control    : {control or '(none)'}")
    print("=" * 88)

    df = load_parquet(path)
    if TEXT_COL not in df.columns:
        raise SystemExit(f"{path.name}: missing column {TEXT_COL!r}.")
    if RAW_TEXT_COL in df.columns and not args.force:
        raise SystemExit(
            f"{path.name} already has a {RAW_TEXT_COL!r} column — it looks like a "
            "normalised file. Normalising twice is almost never what you want; "
            "pass --force to do it anyway."
        )

    before = df[TEXT_COL].fillna("").astype(str).tolist()
    after = [
        apply_transforms(t, transforms, control, args.seed, i)
        for i, t in enumerate(before)
    ]

    y = None
    if TARGET_COL in df.columns:
        y = (df[TARGET_COL] != HUMAN_LABEL).astype(int).to_numpy()

    # --- what changed -------------------------------------------------------
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    chars_before = sum(len(t) for t in before)
    chars_after = sum(len(t) for t in after)
    emptied = [i for i in changed if before[i].strip() and not after[i].strip()]

    print(f"\nrows                 : {len(df)}")
    print(f"rows changed         : {len(changed)}  ({100.0 * len(changed) / max(len(df), 1):.1f}%)")
    print(f"characters           : {chars_before:,} -> {chars_after:,} "
          f"({chars_after - chars_before:+,})")
    if emptied:
        print(f"WARNING  {len(emptied)} row(s) became empty; row indices: {emptied[:20]}")

    if y is not None:
        for cls, name in ((0, "human"), (1, "machine")):
            sel = np.flatnonzero(y == cls)
            if sel.size == 0:
                continue
            n_ch = sum(1 for i in sel if before[i] != after[i])
            print(f"  {name:<8} rows changed: {n_ch}/{sel.size}")

    # --- did it remove the artifact ----------------------------------------
    report = artifact_report(before, after, y)
    print("\nUnicode punctuation, before -> after")
    if "single_bit_acc_before" in report.columns:
        header = (
            f"  {'group':<15}{'docs':>14}{'chars':>16}"
            f"{'pres_h':>16}{'pres_m':>16}{'1-bit acc':>16}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, r in report.iterrows():
            print(
                f"  {r['group']:<15}"
                f"{int(r['docs_before']):>6}->{int(r['docs_after']):<7}"
                f"{int(r['chars_before']):>7}->{int(r['chars_after']):<8}"
                f"{r['presence_human_before']:>7.2f}->{r['presence_human_after']:<8.2f}"
                f"{r['presence_machine_before']:>7.2f}->{r['presence_machine_after']:<8.2f}"
                f"{r['single_bit_acc_before']:>7.3f}->{r['single_bit_acc_after']:<8.3f}"
            )
        print(
            "\n  '1-bit acc' is the accuracy of 'does this document contain the "
            "character class'.\n  It should fall to the majority baseline "
            "(~0.50 on a balanced corpus) after normalisation."
        )
    else:
        print(report.to_string(index=False))

    lines_report = line_structure_report(before, after, y)
    print("\nLine layout, before -> after"
          + ("   (gap = machine - human; watch it collapse)" if y is not None else ""))
    if "gap_before" in lines_report.columns:
        header = f"  {'metric':<20}{'human':>18}{'machine':>18}{'gap':>18}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, r in lines_report.iterrows():
            print(
                f"  {r['metric']:<20}"
                f"{r['human_before']:>8.1f}->{r['human_after']:<9.1f}"
                f"{r['machine_before']:>8.1f}->{r['machine_after']:<9.1f}"
                f"{r['gap_before']:>+8.1f}->{r['gap_after']:<+9.1f}"
            )
    else:
        print(lines_report.to_string(index=False))

    residual = residual_non_ascii(after)
    if residual:
        print("\nNon-ASCII characters remaining (top 15):")
        for code, n, name in residual:
            print(f"  {code}  {n:>7}  {name}")
    else:
        print("\nNo non-ASCII characters remain.")

    # --- idempotency --------------------------------------------------------
    sample = after[: min(200, len(after))]
    twice = [
        apply_transforms(t, transforms, None, args.seed, i) for i, t in enumerate(sample)
    ]
    if control is None and twice != sample:
        n_diff = sum(1 for a, b in zip(sample, twice) if a != b)
        print(
            f"\nWARNING  the transform is not idempotent: re-applying it changes "
            f"{n_diff}/{len(sample)} sampled rows. Check the transform order."
        )

    if args.dry_run:
        would_drop = [
            c for c in df.columns if any(c.startswith(p) for p in STALE_PREFIXES)
        ]
        would_rename = [c for c in df.columns if is_text_derived(c)]
        print("\nColumn plan")
        print(f"  {TEXT_COL!r} <- normalised text, original kept as {RAW_TEXT_COL!r}")
        if would_drop:
            print(f"  drop   ({len(would_drop)}): {would_drop}"
                  if not args.keep_embeddings else f"  keep   ({len(would_drop)}) "
                  f"embeddings at your request: {would_drop}")
        if would_rename and not args.keep_derived_names:
            for c in would_rename:
                print(f"  rename : {c}  ->  {RAW_PREFIX + c}")
        elif would_rename:
            print(f"  keep names on: {would_rename}")
        print("\nDry run — nothing written.\n")
        return

    # --- build the output ---------------------------------------------------
    out = df.copy()
    out[RAW_TEXT_COL] = before
    out[TEXT_COL] = after

    dropped = [c for c in out.columns if any(c.startswith(p) for p in STALE_PREFIXES)]
    if dropped and not args.keep_embeddings:
        out = out.drop(columns=dropped)
        print(f"\nDropped {len(dropped)} stale embedding column(s): {dropped}")
        print("  Re-run epai/ai_detection/embed/add_embeddings.py on the output "
              "before training.")
    elif dropped:
        print(f"\nKEPT {len(dropped)} embedding column(s) at your request: {dropped}")
        print("  WARNING  these vectors describe the OLD text. Training on them "
              "measures nothing.")

    if args.keep_derived_names:
        kept = [c for c in out.columns if is_text_derived(c)]
        if kept:
            print(f"\nKEPT the original names on {len(kept)} text-derived "
                  f"column(s) at your request: {kept}")
            print("  WARNING  nothing in the output now distinguishes a score "
                  "computed on the raw text\n           from one computed on "
                  "the normalised text.")
    else:
        out = rename_text_derived(out)

    if args.length_match:
        if y is None:
            print("\nSkipping --length-match: no "
                  f"{TARGET_COL!r} column to define the classes.")
        else:
            keep = length_match_mask(after, y, args.length_caliper, args.seed)
            out["length_matched"] = keep
            kept_h = int((keep & (y == 0)).sum())
            kept_m = int((keep & (y == 1)).sum())
            print(f"\nlength_matched column added: {kept_h} human + {kept_m} machine "
                  f"rows kept (caliper {args.length_caliper} log-words).")

    print()
    save_parquet(out, out_path)

    if args.report_dir:
        report_dir = Path(args.report_dir) / out_path.stem
        report_dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_dir / "normalization_report.csv", index=False)
        lines_report.to_csv(report_dir / "line_structure_report.csv", index=False)
        print(f"Report -> {report_dir}/")
    print()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquets", nargs="+", type=Path, help="Parquet file(s) to normalise.")
    ap.add_argument(
        "-o", "--out", default=None,
        help="Explicit output path. Only valid with a single input file.",
    )
    ap.add_argument(
        "--suffix", default=None,
        help="Suffix for the output stem (default: 'norm' for the default "
        "transform set, or the control name).",
    )
    ap.add_argument(
        "--transforms", nargs="*", choices=ALL_TRANSFORMS, default=list(DEFAULT_TRANSFORMS),
        help=f"Transforms to apply (default: {' '.join(DEFAULT_TRANSFORMS)}). "
        "Pass with no values to apply none, e.g. for a control-only run.",
    )
    ap.add_argument(
        "--control", choices=CONTROLS, default=None,
        help="Destructive control applied after the transforms.",
    )
    ap.add_argument(
        "--keep-embeddings", action="store_true",
        help="Do not drop emb_* columns. They will not match the new text.",
    )
    ap.add_argument(
        "--keep-derived-names", action="store_true",
        help=f"Do not prefix text-derived score columns (binoculars / llm_judge "
        f"/ ...) with {RAW_PREFIX!r}. Their values describe the raw text either "
        "way; without the prefix nothing in the file says so.",
    )
    ap.add_argument(
        "--length-match", action="store_true",
        help="Add a boolean 'length_matched' column selecting a human/machine "
        "subset matched on log word count (does not drop any row).",
    )
    ap.add_argument(
        "--length-caliper", type=float, default=0.10,
        help="Maximum log-word-count distance for a length match (default: 0.10, "
        "about 10%% in word count).",
    )
    ap.add_argument("--seed", type=int, default=42, help="Seed for the controls and matching.")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change; write nothing.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help=f"Normalise even if a {RAW_TEXT_COL!r} column is already present.",
    )
    ap.add_argument(
        "--report-dir", default=str(BASE_DIR / "results" / "normalization"),
        help="Where to write the per-file normalization_report.csv.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.out and len(args.parquets) > 1:
        raise SystemExit("-o/--out only makes sense with a single input file.")

    print(f"normalize_text.py  —  {datetime.now():%Y-%m-%d %H:%M:%S}")
    for path in args.parquets:
        if not path.is_file():
            print(f"SKIPPED — not found: {path}")
            continue
        process(path, args)


if __name__ == "__main__":
    main()

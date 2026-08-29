r"""Score a corpus with pretrained supervised AI-text detectors.

    uv run python epai/ai_detection/score/add_detector_scores.py \
        data/parquet/ghostbuster_gpt56luna.parquet --gpu 0

The third kind of baseline in this repo, and the one nothing else covers.
Binoculars and Fast-DetectGPT are zero-shot statistics over general LMs; the
LLM judges are prompted; every supervised number so far is *our* classifier on
*our* embeddings, fitted on this corpus. These detectors are supervised
classifiers trained by other people on other data, which never saw this corpus
at all -- so they measure something neither of the others does: whether a strong
off-the-shelf detector, of the kind actually deployed, transfers here.

RAID (arXiv:2405.07940) is the reason to expect it might not: its top detectors
hold up on the domain they were tuned for and then hit 95%+ error rates when the
generator or the decoding strategy changes. That is the same shape as this
project's own in-domain/cross-domain gap, arrived at from the other side.

Runs in THIS repo's venv, not a submodule's -- both checkpoints are plain
transformers sequence classifiers and need nothing the repo does not already
pin. The EditLens fork is checked out as a read-only reference: its licence is
CC BY-NC-SA (share-alike), so this file reimplements the head arithmetic from
its inference.py rather than importing it, keeping episteme-ai unencumbered.

--------------------------------------------------------------------------------
THE TWO DETECTORS ARE NOT ANSWERING THE SAME QUESTION
--------------------------------------------------------------------------------
``radar``            binary AI-vs-human. P(AI) straight out of a softmax.

``pangram_roberta``  EditLens: how MUCH of the text is AI, not whether it is.
                     The head is an n-bucket ordinal classifier and the score is
                     the expected bucket, normalised to [0, 1] -- 0 = untouched
                     human, 1 = fully machine. On a corpus of wholly-human and
                     wholly-generated documents those are exactly the end
                     buckets, so it is directly comparable here; on part-edited
                     text it would be answering a different and better question.

Both are "higher = more AI", the same direction as fastdetect_crit_* and the
OPPOSITE of binoculars_score.

--------------------------------------------------------------------------------
TWO THINGS EDITLENS DOES TO THE TEXT, BOTH REPORTED PER RUN
--------------------------------------------------------------------------------
Upstream applies `preprocess.clean_text` before tokenizing, and reproducing it
is not optional -- the model was trained on text put through it. It does two
things this corpus should care about:

1. **It lowercases and collapses all whitespace.** Which means EditLens is
   structurally blind to the surface artifact that drives this project's
   in-domain accuracy: the curly-quote / punctuation / layout signal whose sign
   flips between domains (4.3% curly quotes on human wp, 90.2% on human essay).
   That is a *feature* of this baseline, not a defect. It cannot be winning on
   the artifact, so whatever it scores is coming from somewhere else, and the
   gap between it and an embedding classifier on the same rows is informative.

2. **It can delete the first line.** `remove_ai_header` drops the opening
   paragraph when it starts with any of "Sure", "Here", "Abstract", "Title",
   "Certainly", "I'm happy to help" -- assistant boilerplate. On the science
   corpus "Abstract" is not boilerplate, it is the document, and a human abstract
   beginning with that word loses its first line while a generated one that does
   not begin with it keeps everything. That is a class-correlated edit to the
   input, i.e. exactly the kind of thing this project exists to catch.

   So this script counts the rows it strips, reports them, and puts the count in
   the summary JSON as ``header_stripped``. If that number is not near zero, or
   is lopsided by class, the column is measuring the preprocessing as much as
   the text. Check it on the science corpus before using the numbers.

--------------------------------------------------------------------------------
CONTEXT LIMITS ARE SHORTER THAN EVERYTHING ELSE HERE
--------------------------------------------------------------------------------
Both RoBERTa checkpoints cap at 512 tokens, against Binoculars' and
Fast-DetectGPT's 512 (comparable) but the corpus's own documents run far longer
on the science side. Every detector therefore also writes
``detector_ntok_<name>``, the tokens actually seen, and the summary reports how
many rows hit the cap. Without it there is no way to tell "the detector
disagrees" from "the detector read the first page".

--------------------------------------------------------------------------------
LICENCES
--------------------------------------------------------------------------------
Both checkpoints are non-commercial. RADAR inherits Vicuna's licence; EditLens
is CC BY-NC-SA 4.0. Research use is fine, which is what this is. The scores are
derived data and this file is not a derivative of either model's code.
"""

from __future__ import annotations

# Standard library only until --gpu / --cpu-threads have been applied: both are
# read when the CUDA driver and the OpenMP runtime initialise, inside torch.
import argparse
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

COLUMN_PREFIX = "detector_"
DEFAULT_TEXT_COL = "text"

# Texts per forward pass. Both default detectors are RoBERTa-large (355M) at 512
# tokens, which is small; 64 leaves plenty of headroom on a 40GB A100 and the
# automatic halving below handles anything tighter.
BATCH_SIZE = 64
MIN_BATCH_SIZE = 1


DETECTORS: dict[str, dict] = {
    "radar": {
        "checkpoint": "TrustSafeAI/RADAR-Vicuna-7B",
        # The tokenizer ships with the checkpoint here, unlike EditLens.
        "tokenizer": "TrustSafeAI/RADAR-Vicuna-7B",
        "head": "binary",
        # THE INDEX IS 0, AND IT IS NOT DISCOVERABLE FROM THE MODEL.
        # config.json carries no id2label, so transformers labels the outputs
        # LABEL_0/LABEL_1 and nothing in the artifact says which is which.
        # IBM's own inference snippet (github.com/IBM/RADAR) reads
        #   F.log_softmax(logits, -1)[:, 0].exp()
        # and calls that "the probability of AI-generated texts". Getting this
        # backwards does not raise -- it silently produces a below-chance
        # detector, which is the failure this corpus has already been burned by
        # once with an inverted Binoculars pair.
        "ai_index": 0,
        "max_length": 512,
        "clean_text": False,
        "score_column": "detector_score_radar",
        "ntok_column": "detector_ntok_radar",
        "bucket_column": None,
        "requires": (),
        "note": "RoBERTa-large despite the name -- the 'Vicuna-7B' is the "
                "generator it was adversarially trained against, not the "
                "detector. Non-commercial licence, inherited from Vicuna.",
    },
    "pangram_roberta": {
        "checkpoint": "pangram/editlens_roberta-large",
        # Upstream's inference.py takes the tokenizer from the BASE model, not
        # from the checkpoint. Followed exactly: a different tokenizer would
        # change the token ids the head was trained on.
        "tokenizer": "FacebookAI/roberta-large",
        "head": "editlens",
        "ai_index": None,
        "max_length": 512,
        "clean_text": True,
        "score_column": "detector_score_pangram_roberta",
        "ntok_column": "detector_ntok_pangram_roberta",
        "bucket_column": "detector_bucket_pangram_roberta",
        "requires": ("emoji",),
        "note": "Open Pangram / EditLens (ICLR 2026). Ordinal bucket head; the "
                "score is the expected bucket in [0,1]. CC BY-NC-SA 4.0.",
    },
    "pangram_llama3b": {
        "checkpoint": "pangram/editlens_Llama-3.2-3B",
        "tokenizer": "meta-llama/Llama-3.2-3B",
        "head": "editlens_qlora",
        "ai_index": None,
        "max_length": 1024,
        "clean_text": True,
        "score_column": "detector_score_pangram_llama3b",
        "ntok_column": "detector_ntok_pangram_llama3b",
        "bucket_column": "detector_bucket_pangram_llama3b",
        # A 4-bit QLoRA adapter over a gated base, with a custom NormedLinear
        # score head that has to be swapped in before the adapter loads. None of
        # peft, bitsandbytes or that head is in this repo's venv, so the
        # detector is registered and blocked rather than silently absent -- the
        # findings table names it and says what it needs.
        "requires": ("peft", "bitsandbytes", "emoji"),
        "note": "Stronger of the two Open Pangram models and a 1024-token "
                "context, but needs peft + bitsandbytes and a gated Llama base. "
                "Not in the default set.",
    },
}

DEFAULT_DETECTORS = ("radar", "pangram_roberta")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquet", type=Path, help="Parquet to add detector scores to.")
    ap.add_argument("--detectors", nargs="+", default=list(DEFAULT_DETECTORS),
                    metavar="NAME", choices=sorted(DETECTORS),
                    help="Which detectors to run (default: %(default)s)")
    ap.add_argument("--text-column", default=DEFAULT_TEXT_COL,
                    help="Column holding the text to score (default: %(default)s)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help=f"Texts per forward pass (default: {BATCH_SIZE}, halved "
                         f"automatically on CUDA OOM)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Smoke test: score only the first N rows. Rows past the limit "
                         "keep what they have; they are not dropped. Point it at a "
                         "throwaway copy -- a part-filled column shrinks every later "
                         "comparison that reads it.")
    ap.add_argument("--max-length", type=int, default=None,
                    help="Override the truncation cap for every detector. Default: each "
                         "detector's own (512 for both RoBERTa checkpoints). Raising it "
                         "past a model's position embeddings will fail, not help.")
    ap.add_argument("--no-emoji-normalization", action="store_true",
                    help="Run EditLens preprocessing without the `emoji` package. This "
                         "DEVIATES from what the model was trained on and is recorded "
                         "in the summary; without it, an EditLens detector is blocked "
                         "when `emoji` is missing rather than scored differently.")
    ap.add_argument("--gpu", default=None,
                    help="value for CUDA_VISIBLE_DEVICES (e.g. '0')")
    ap.add_argument("--cpu-threads", type=int, default=None,
                    help="limit CPU threads (sets OMP_NUM_THREADS / MKL_NUM_THREADS)")
    ap.add_argument("--recompute", action="store_true",
                    help="Rescore rows that already have a score, without asking.")
    ap.add_argument("--yes", action="store_true", help="Skip the confirmation prompts.")
    ap.add_argument("--no-summary-file", action="store_true",
                    help="Print the summary but write no *_detectors_summary.json")
    return ap.parse_args()


args = parse_args()

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.cpu_threads is not None:
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)

import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from epai.utils.parquet_utils import load_parquet, save_parquet  # noqa: E402

load_dotenv(BASE_DIR / ".env")

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# EditLens preprocessing, reimplemented from external/EditLens/scripts/preprocess.py
# ---------------------------------------------------------------------------
# Reimplemented rather than imported: that repo is CC BY-NC-SA (share-alike), so
# importing it into episteme-ai would propagate the licence into this tree. The
# behaviour has to match exactly all the same -- the checkpoint was trained on
# text put through it -- so this is written against the same spec and verified
# against the same fixtures. If you pull EditLens and preprocess.py has changed,
# this is the function to re-check.
BOILERPLATE_STARTS = (
    "Sure", "Here", "Abstract", "Title", "I'm happy to help", "Certainly",
)


def _demojize(text: str, enabled: bool) -> str:
    if not enabled:
        return text
    import emoji
    return emoji.demojize(text)


def _strip_emoji(text: str, enabled: bool) -> str:
    if not enabled:
        return text
    import emoji
    return emoji.replace_emoji(text, "")


def remove_think_tag(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[1].strip()
    return text


def remove_ai_header(text: str, emoji_enabled: bool) -> tuple[str, bool]:
    """Drop the opening line when it reads as assistant boilerplate.

    Returns (text, stripped). The flag is the point: on a science corpus
    "Abstract" is a section heading, not boilerplate, and dropping it is a
    class-correlated edit to the input that the caller has to be able to count.
    """
    paragraphs = [p for p in text.split("\n") if p.strip()]
    if len(paragraphs) == 0:
        return text, False
    first = paragraphs[0]
    first = re.sub(r"^[^a-zA-Z0-9]*", "", first)
    first = _strip_emoji(first, emoji_enabled)
    if any(first.startswith(phrase) for phrase in BOILERPLATE_STARTS):
        if len(paragraphs) > 1:
            return "\n".join(paragraphs[1:]), True
    return text, False


def clean_text(text: str, emoji_enabled: bool) -> tuple[str, bool]:
    """EditLens's clean_text. Returns (cleaned, header_was_stripped)."""
    text = _demojize(text, emoji_enabled)
    text = remove_think_tag(text)
    text, stripped = remove_ai_header(text, emoji_enabled)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text, stripped


# ---------------------------------------------------------------------------
# Parquet inspection
# ---------------------------------------------------------------------------
@dataclass
class ColumnStatus:
    """What one detector currently looks like in the Parquet."""

    name: str
    col: str
    exists: bool
    n_filled: int
    n_missing: int
    blocked: str | None = None

    @property
    def state(self) -> str:
        if self.blocked:
            return "blocked"
        if not self.exists:
            return "absent"
        if self.n_filled == 0:
            return "empty"
        if self.n_missing > 0:
            return "partial"
        return "complete"


def _is_empty_text(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def _as_score(value) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(score) else score


def blocking_reason(name: str) -> str | None:
    """Why this detector cannot run here, or None if it can.

    Checked up front so a two-model run does not die on the second, and so the
    findings table can name a detector it is not going to run instead of quietly
    omitting it.
    """
    spec = DETECTORS[name]
    for package in spec["requires"]:
        if package == "emoji" and args.no_emoji_normalization:
            continue  # explicitly waived, and recorded in the summary
        if importlib.util.find_spec(package) is None:
            if package == "emoji":
                return ("needs emoji (or --no-emoji-normalization, which changes "
                        "the numbers)")
            return f"needs {package}"
    return None


def batch_size_for(cli: int | None) -> int:
    return cli if cli is not None else BATCH_SIZE


def max_length_for(name: str) -> int:
    return args.max_length if args.max_length is not None else DETECTORS[name]["max_length"]


def inspect(df: pd.DataFrame, names: list[str], text_col: str,
            limit: int | None) -> tuple[np.ndarray, list[ColumnStatus]]:
    """The scoreable mask and one status per detector."""
    if len(df) == 0:
        raise SystemExit(f"{RED}{args.parquet} has no rows.{RESET}")
    if text_col not in df.columns:
        raise SystemExit(
            f"{RED}Column {text_col!r} not found. Available: {list(df.columns)}{RESET}"
        )

    nonblank = np.array([not _is_empty_text(v) for v in df[text_col]])
    window = np.ones(len(df), dtype=bool)
    if limit is not None:
        if limit < 1:
            raise SystemExit(f"{RED}--limit must be >= 1, got {limit}{RESET}")
        window[limit:] = False
    scoreable = nonblank & window

    statuses = []
    for name in names:
        col = DETECTORS[name]["score_column"]
        if col not in df.columns:
            filled = np.zeros(len(df), dtype=bool)
            exists = False
        else:
            filled = np.array([_as_score(v) is not None for v in df[col]])
            exists = True
        statuses.append(ColumnStatus(
            name=name,
            col=col,
            exists=exists,
            n_filled=int((scoreable & filled).sum()),
            n_missing=int((scoreable & ~filled).sum()),
            blocked=blocking_reason(name),
        ))
    return scoreable, statuses


def positions_to_score(df: pd.DataFrame, st: ColumnStatus,
                       scoreable: np.ndarray, recompute: bool) -> np.ndarray:
    if recompute or not st.exists:
        return np.flatnonzero(scoreable)
    filled = np.array([_as_score(v) is not None for v in df[st.col]])
    return np.flatnonzero(scoreable & ~filled)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
class Detector:
    """One loaded checkpoint and the head arithmetic that turns it into a score."""

    def __init__(self, name: str) -> None:
        spec = DETECTORS[name]
        self.name = name
        self.spec = spec
        self.max_length = max_length_for(name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.emoji_enabled = not args.no_emoji_normalization

        print(f"Loading {name}: {spec['checkpoint']} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(spec["checkpoint"])
        self.model.to(self.device)
        self.model.eval()

        if spec["head"] == "editlens_qlora":
            # Reachable only if someone installs peft and bitsandbytes, which
            # unblocks the detector without making it loadable: the checkpoint is
            # a 4-bit adapter over a gated base, and its score head is EditLens's
            # custom NormedLinear, which has to be swapped in BEFORE the adapter
            # loads. AutoModelForSequenceClassification alone gets none of that,
            # and would fail somewhere less legible than here.
            raise SystemExit(
                f"{RED}{self.name}: QLoRA adapters are not supported by this "
                f"script.{RESET}\n"
                f"  {spec['checkpoint']} needs peft + bitsandbytes AND EditLens's "
                f"NormedLinear score head.\n"
                f"  Run it through external/EditLens/scripts/inference.py in its own "
                f"environment, or\n  use --detectors pangram_roberta, which is a plain "
                f"fine-tune and loads here."
            )

        self.n_buckets = int(AutoConfig.from_pretrained(spec["checkpoint"]).num_labels)
        if spec["head"] == "binary" and self.n_buckets != 2:
            raise SystemExit(
                f"{RED}{name}: expected a 2-class head, found {self.n_buckets}. "
                f"The ai_index in the registry is only meaningful for a binary "
                f"head, so this is not safe to score.{RESET}"
            )
        # Upstream infers n_buckets the same way for a full fine-tune, and the
        # score formula divides by (n_buckets - 1), so a 1-class head would be a
        # division by zero rather than a wrong number.
        if spec["head"] == "editlens" and self.n_buckets < 2:
            raise SystemExit(f"{RED}{name}: bucket head has {self.n_buckets} label(s).{RESET}")
        if spec["head"] == "editlens":
            print(f"  {DIM}{self.n_buckets} buckets; score = expected bucket / "
                  f"{self.n_buckets - 1}{RESET}")

    def prepare(self, text: str) -> tuple[str, bool]:
        if self.spec["clean_text"]:
            return clean_text(text, self.emoji_enabled)
        return text, False

    @torch.no_grad()
    def score_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """(scores, buckets|None, tokens seen) for a batch of already-cleaned texts."""
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits.float()
        probs = torch.softmax(logits, dim=-1)
        ntok = enc["attention_mask"].sum(dim=1).cpu().numpy().astype(int)

        if self.spec["head"] == "binary":
            scores = probs[:, self.spec["ai_index"]].cpu().numpy()
            return scores, None, ntok

        # EditLens: expected bucket, normalised to [0, 1].
        #   score = (probs @ [0..n-1]) / (n - 1)
        # Same arithmetic as EditLens's inference.py, written out rather than
        # imported for the licence reason at the top of this file.
        ladder = torch.arange(self.n_buckets, device=probs.device, dtype=probs.dtype)
        scores = ((probs * ladder).sum(dim=-1) / (self.n_buckets - 1)).cpu().numpy()
        buckets = probs.argmax(dim=-1).cpu().numpy().astype(int)
        return scores, buckets, ntok


def score_with_backoff(det: Detector, texts: list[str], batch_size: int):
    """Score a chunk, halving the batch on CUDA OOM down to a single row.

    A detector that OOMs at batch 64 on a shared card is not a detector that
    cannot run; it is one that needs a smaller batch. Same policy as
    add_embeddings.py, and it keeps a long run from dying at row 4000 because
    something else landed on the GPU.
    """
    while True:
        try:
            return det.score_batch(texts[:batch_size]), batch_size
        except torch.cuda.OutOfMemoryError:
            if batch_size <= MIN_BATCH_SIZE:
                raise
            torch.cuda.empty_cache()
            batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
            tqdm.write(f"  {YELLOW}CUDA OOM — retrying at batch {batch_size}{RESET}")


def score_detector(det: Detector, df: pd.DataFrame, text_col: str,
                   positions: np.ndarray, batch_size: int) -> tuple[pd.DataFrame, dict]:
    """Fill one detector's columns for the given row positions."""
    spec = det.spec
    n = len(df)
    score_col, ntok_col, bucket_col = spec["score_column"], spec["ntok_column"], spec["bucket_column"]

    scores: list = list(df[score_col]) if score_col in df.columns else [float("nan")] * n
    ntoks: list = list(df[ntok_col]) if ntok_col in df.columns else [None] * n
    buckets: list = (list(df[bucket_col]) if (bucket_col and bucket_col in df.columns)
                     else [None] * n)

    raw = df[text_col].astype(str).to_numpy()
    counters = {"header_stripped": 0, "truncated": 0, "empty_after_clean": 0}
    new_scores: list[float] = []

    i = 0
    pending = list(positions)
    with tqdm(total=len(pending), desc=det.name, unit="doc", leave=False) as bar:
        while i < len(pending):
            chunk = pending[i:i + batch_size]
            prepared, stripped_flags = [], []
            for pos in chunk:
                cleaned, stripped = det.prepare(raw[pos])
                if not cleaned.strip():
                    # Cleaning can empty a row that was not blank to begin with
                    # (a text that was nothing but an assistant header). Feeding
                    # "" to the model produces a confident score for no content.
                    cleaned = raw[pos]
                    counters["empty_after_clean"] += 1
                    stripped = False
                prepared.append(cleaned)
                stripped_flags.append(stripped)

            (batch_scores, batch_buckets, batch_ntok), batch_size = score_with_backoff(
                det, prepared, batch_size
            )
            used = chunk[:len(batch_scores)]
            for k, pos in enumerate(used):
                scores[pos] = float(batch_scores[k])
                ntoks[pos] = int(batch_ntok[k])
                new_scores.append(float(batch_scores[k]))
                if batch_buckets is not None and bucket_col:
                    buckets[pos] = int(batch_buckets[k])
                if stripped_flags[k]:
                    counters["header_stripped"] += 1
                if int(batch_ntok[k]) >= det.max_length:
                    counters["truncated"] += 1
            i += len(used)
            bar.update(len(used))

    df[score_col] = pd.Series(scores, index=df.index, dtype="float64")
    df[ntok_col] = pd.Series(ntoks, index=df.index, dtype="Int64")
    if bucket_col:
        df[bucket_col] = pd.Series(buckets, index=df.index, dtype="Int64")
    counters["scores"] = new_scores
    return df, counters


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def describe_device() -> str:
    if not torch.cuda.is_available():
        return "CPU (no CUDA available)"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    suffix = f", CUDA_VISIBLE_DEVICES={visible}" if visible else ""
    return f"cuda:0 — {name}, {total:.0f} GiB{suffix}"


def print_findings(statuses: list[ColumnStatus], n_rows: int, n_blank: int) -> None:
    print()
    print("=" * 92)
    print("Pretrained detectors")
    print(f"  parquet     : {args.parquet}")
    print(f"  rows        : {n_rows}   ({n_blank} blank, never scored)")
    print(f"  text column : {args.text_column!r}")
    print(f"  direction   : {YELLOW}HIGH score = more AI{RESET} "
          f"{DIM}(same as fastdetect_crit_*, opposite to binoculars_score){RESET}")
    print(f"  device      : {describe_device()}")
    print(f"  batch size  : {batch_size_for(args.batch_size)}")
    if args.no_emoji_normalization:
        print(f"  {YELLOW}emoji normalisation DISABLED — EditLens scores will differ "
              f"from a faithful run{RESET}")
    print("=" * 92)
    print(f"{'detector':<18}{'score column':<38}{'state':<10}{'filled':>8}{'missing':>9}")
    print("-" * 92)
    for st in statuses:
        print(f"{st.name:<18}{st.col:<38}{st.state:<10}{st.n_filled:>8}{st.n_missing:>9}")
        if st.blocked:
            print(f"  {YELLOW}↳ {st.blocked}{RESET}")
    print("-" * 92)
    for st in statuses:
        note = DETECTORS[st.name].get("note")
        if note:
            print(f"  {DIM}{st.name}: {note}{RESET}")


def ask_yes_no(question: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    try:
        answer = input(question + suffix).strip().lower()
    except EOFError:
        print(f"\n{YELLOW}No input available; assuming no.{RESET}")
        return False
    if not answer:
        return default
    return answer in ("y", "yes")


def _distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size), "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)), "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)), "max": float(arr.max()),
        "mean": float(arr.mean()), "sd": float(arr.std()),
    }


def summary_for(name: str, n_scored: int, seconds: float, counters: dict) -> dict:
    spec = DETECTORS[name]
    return {
        "detector": name,
        "checkpoint": spec["checkpoint"],
        "tokenizer": spec["tokenizer"],
        "head": spec["head"],
        "ai_index": spec["ai_index"],
        "score_column": spec["score_column"],
        "bucket_column": spec["bucket_column"],
        "tokens_column": spec["ntok_column"],
        "direction": "high score = more AI",
        "max_length": max_length_for(name),
        "editlens_cleaning": spec["clean_text"],
        "emoji_normalization": (not args.no_emoji_normalization) if spec["clean_text"] else None,
        "rows_scored": n_scored,
        "rows_truncated_at_cap": counters["truncated"],
        "header_stripped": counters["header_stripped"],
        "empty_after_clean": counters["empty_after_clean"],
        "seconds": round(seconds, 1),
        "docs_per_second": round(n_scored / seconds, 2) if seconds > 0 else None,
        "score_distribution": _distribution(counters["scores"]),
    }


def print_detector_stats(payload: dict, indent: str = "      ") -> None:
    dist = payload["score_distribution"]
    if dist:
        print(f"{indent}score    min {dist['min']:.3f}  median {dist['median']:.3f}  "
              f"max {dist['max']:.3f}  (mean {dist['mean']:.3f}, sd {dist['sd']:.3f})")
    if payload["rows_truncated_at_cap"]:
        pct = 100 * payload["rows_truncated_at_cap"] / max(payload["rows_scored"], 1)
        print(f"{indent}⚠️  {payload['rows_truncated_at_cap']} row(s) ({pct:.0f}%) hit the "
              f"{payload['max_length']}-token cap — scored on a prefix.")
    if payload["header_stripped"]:
        pct = 100 * payload["header_stripped"] / max(payload["rows_scored"], 1)
        print(f"{indent}❗ {payload['header_stripped']} row(s) ({pct:.0f}%) had their first "
              f"line removed by EditLens's boilerplate rule.")
        print(f"{indent}   Check how that splits by class before using the column — on a "
              f"corpus where\n{indent}   'Abstract' opens the human documents it is a "
              f"class-correlated edit, not cleaning.")
    if payload["empty_after_clean"]:
        print(f"{indent}⚠️  {payload['empty_after_clean']} row(s) were empty after "
              f"cleaning; the raw text was scored instead.")


def main() -> int:
    if not args.parquet.is_file():
        raise SystemExit(f"{RED}Not found: {args.parquet}{RESET}")

    df = load_parquet(args.parquet)
    scoreable, statuses = inspect(df, args.detectors, args.text_column, args.limit)
    n_blank = int(len(df) - np.array([not _is_empty_text(v) for v in df[args.text_column]]).sum())
    print_findings(statuses, len(df), n_blank)

    runnable = [st for st in statuses if not st.blocked]
    if not runnable:
        raise SystemExit(f"{RED}Every selected detector is blocked. Nothing to do.{RESET}")

    n_present = sum(1 for st in runnable if st.n_filled > 0)
    recompute = args.recompute
    if n_present and not recompute and not args.yes:
        recompute = ask_yes_no(
            f"{n_present} detector(s) already have scores. Rescore those rows?",
            default=False,
        )

    todo = [st for st in runnable
            if st.n_missing > 0 or (recompute and st.n_filled > 0)]
    if not todo:
        print("\nEverything is already scored — nothing to do.")
        return 0

    print()
    print(f"Will run {len(todo)} detector(s), writing IN PLACE into {args.parquet.name}:")
    for st in todo:
        positions = positions_to_score(df, st, scoreable, recompute)
        print(f"  {st.name:<18} {len(positions):>7} rows  ({st.state})")
    if not args.yes:
        answer = input("Proceed? Type 'yes': ").strip().lower()
        if answer != "yes":
            raise SystemExit("Aborted — no changes made.")

    payloads: list[dict] = []
    failed: list[tuple[str, str]] = []
    for st in todo:
        positions = positions_to_score(df, st, scoreable, recompute)
        if len(positions) == 0:
            continue
        print(f"\n→ {st.name}  ({len(positions)} rows)")
        started = time.perf_counter()
        det = None
        try:
            det = Detector(st.name)
            df, counters = score_detector(
                det, df, args.text_column, positions, batch_size_for(args.batch_size)
            )
        except Exception as exc:  # one bad detector must not lose the other
            failed.append((st.name, f"{type(exc).__name__}: {exc}"))
            print(f"  {RED}failed: {type(exc).__name__}: {exc}{RESET}")
            continue
        finally:
            # Free the card before the next checkpoint loads, or two 355M models
            # and their activations sit on it at once for no reason. Bound to
            # None ahead of the try so this is safe when the constructor itself
            # raised, which is the common failure (gated repo, missing weights).
            det = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        payload = summary_for(st.name, len(positions), time.perf_counter() - started, counters)
        payloads.append(payload)
        # Save after every detector so an interrupt does not lose finished work.
        save_parquet(df, args.parquet)
        print_detector_stats(payload)

    if payloads and not args.no_summary_file:
        out = args.parquet.with_name(f"{args.parquet.stem}__detectors_summary.json")
        out.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "parquet": str(args.parquet),
            "limit": args.limit,
            "recomputed_existing": recompute,
            "detectors": payloads,
        }, indent=2) + "\n")
        print(f"\nSummary: {out}")

    print()
    for payload in payloads:
        print(f"  {GREEN}✅ {payload['detector']}{RESET}  {payload['rows_scored']} rows in "
              f"{payload['seconds']}s ({payload['docs_per_second']} doc/s)")
    for name, reason in failed:
        print(f"  {RED}❌ {name}{RESET}  {reason}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

r"""Surface-artifact audit: what a classifier could learn from the raw text alone.

Step 1 of the "why is in-domain 98% and transfer below chance" investigation.
No model, no embedding, no LLM — just deterministic string features, ranked by
how well each one separates human from machine *inside* each corpus, and, the
part that matters, **with the sign reported per corpus**.

The sign is the deliverable. A feature that says "curly apostrophe -> human" in
one corpus and "curly apostrophe -> machine" in another is not a weak feature
that fails to generalise; it is a strong feature pointing the wrong way, which
is what drives transfer *below* chance (macro-F1 0.33-0.37) rather than to it.

For each corpus and each feature the script reports:

  human_mean / machine_mean  the raw value per class
  presence_h / presence_m    fraction of docs with a non-zero value
  acc                        accuracy of the best single-threshold rule on this
                             feature alone (the "one bit is 93% accurate" number)
  dir                        +1 if a higher value means machine, -1 if human
  mi_bits                    mutual information of that thresholded bit
                             with the label, in bits (max 1.0)
  strength                   dir * max(0, acc - majority_acc), the signed
                             number plotted; zero for a feature that beats
                             nothing but the majority baseline

From those, three cross-corpus products:

  sign_flips.csv         features that are strong in two corpora with opposite
                         signs — the artifacts that actively invert a transfer
  artifact_alignment.csv cosine between corpora of their signed strength
                         vectors: +1 = the artifacts agree, -1 = they oppose
  char n-gram log-odds   the same treatment for character 3-grams, over the
                         whole document and over its first / last 100 chars
                         (where per-corpus boilerplate lives, and where a
                         512-token embedder is guaranteed to look)

With ``--compare-matrix`` pointing at a ``cross_domain.json``, the script also
correlates the predicted artifact alignment against the observed transfer
accuracy. That turns the audit into a falsifiable claim rather than a list of
suspicious strings: if alignment does not predict transfer, this diagnosis is
wrong and the collapse is caused by something else.

Nothing is written back to the corpora; the parquets are only read.

Usage:
    # all six corpora, using the same dataset list and drop_classes as training
    uv run python epai/ai_detection/analyse/analyse_surface_artifacts.py \
        --config cfg/ai_detection/train/clf_cd/hd2/ALL_logistic.yaml \
        --compare-matrix results/clf_cd/clf_cd_20260721_121101/ALL_logistic/cross_domain.json

    # ad-hoc, straight from parquet paths
    uv run python epai/ai_detection/analyse/analyse_surface_artifacts.py \
        data/parquet/essay_binox0.parquet data/parquet/wp_gpt54mini_binox0.parquet

    # after normalising, to confirm the artifacts are actually gone
    uv run python epai/ai_detection/analyse/analyse_surface_artifacts.py \
        --config cfg/ai_detection/train/clf_cd/norm/NORM_luar_crud.yaml --tag norm
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from epai.utils.config_utils import load_cd_config

BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "surface_artifacts"

ID_COL = "id"
TEXT_COL = "text"
# Same column the training code binarises on (epai/utils/train_utils.CLF_TARGET_COL),
# repeated here so this script does not import the training stack (torch, sklearn).
TARGET_COL = "generated_by"
HUMAN_LABEL = "human"

# Generators the cross-domain configs drop, so an ad-hoc run without --config
# sees the same binary problem the classifier was trained on.
DEFAULT_DROP_CLASSES = (
    "claude",
    "gpt_prompt1",
    "gpt_prompt2",
    "gpt_semantic",
    "gpt_writing",
)

# A feature counts as "strong enough to matter" at this single-feature accuracy.
# Two strong features with opposite signs are what a sign flip means.
FLIP_MIN_ACC = 0.60

# Signed strength below this is treated as no signal when deciding whether a
# row of the sign table spans both signs.
FLIP_EPS = 0.05

# Width, in heatmap columns, of the gutter reserved on the right for FLIP tags.
FLIP_GUTTER = 0.9

# Character n-gram settings.
NGRAM_N = 3
NGRAM_MIN_DOC_FRAC = 0.05  # keep n-grams seen in >= 5% of docs of either class
NGRAM_TOP_K = 40
NGRAM_PRIOR = 0.5  # Dirichlet smoothing on the document-frequency log-odds
HEAD_TAIL_CHARS = 100

# Diverging blue<->red with a neutral gray midpoint (blue = human, red = machine).
# One hue per pole and gray in the middle, so "no signal" reads as nothing.
CMAP_SIGN = LinearSegmentedColormap.from_list(
    "artifact_sign",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2b8b6", "#e34948", "#8f2321"],
)
# Single-hue sequential for magnitudes that have no polarity.
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "artifact_mag", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"


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
# Feature definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """One deterministic surface feature."""

    name: str
    group: str  # unicode | whitespace | structure | lexical | length
    fn: Callable[[str], float]
    unit: str  # per1k | frac | binary | raw
    note: str = ""


# Non-ASCII characters that get their own feature, so `other_non_ascii` means
# "non-ASCII that is not one of the usual typographic suspects".
NAMED_NON_ASCII = frozenset(
    "‘’“”–—… •·"
    "‐‑‒―­​′″﻿"
)


def _rate(count: int, text: str) -> float:
    """Occurrences per 1000 characters — length-independent, unlike a raw count."""
    return 1000.0 * count / max(len(text), 1)


def char_rate(chars: str) -> Callable[[str], float]:
    def fn(text: str) -> float:
        return _rate(sum(text.count(c) for c in chars), text)

    return fn


def regex_rate(pattern: str, flags: int = 0) -> Callable[[str], float]:
    rx = re.compile(pattern, flags)

    def fn(text: str) -> float:
        return _rate(len(rx.findall(text)), text)

    return fn


def regex_presence(pattern: str, flags: int = 0) -> Callable[[str], float]:
    rx = re.compile(pattern, flags)
    return lambda text: float(bool(rx.search(text)))


def other_non_ascii_rate(text: str) -> float:
    n = sum(1 for ch in text if ord(ch) > 127 and ch not in NAMED_NON_ASCII)
    return _rate(n, text)


def trailing_ws_line_frac(text: str) -> float:
    lines = text.split("\n")
    if not lines:
        return 0.0
    return sum(1 for ln in lines if ln != ln.rstrip()) / len(lines)


def mean_line_length(text: str) -> float:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return float(np.mean([len(ln) for ln in lines])) if lines else 0.0

def mean_word_length(text: str) -> float:
    words = text.split()
    return float(np.mean([len(w) for w in words])) if words else 0.0


def mean_sentence_words(text: str) -> float:
    parts = [p for p in re.split(r"[.!?]+\s", text) if p.strip()]
    return float(np.mean([len(p.split()) for p in parts])) if parts else 0.0


def type_token_ratio(text: str) -> float:
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    return len(set(words)) / len(words) if words else 0.0


def first_line_is_header(text: str) -> float:
    """A short opening line with no sentence punctuation — a title or section head.

    Catches the literal ``Introduction:`` that opens every Ghostbuster essay/gpt
    file as well as bare title lines, without hard-coding either.
    """
    first = text.strip().split("\n", 1)[0].strip()
    if not first or len(first) > 80:
        return 0.0
    if first.endswith((".", "!", "?", ",", ";")):
        return 0.0
    return float(len(first.split()) <= 12)


FEATURES: tuple[FeatureSpec, ...] = (
    # --- unicode typography: the prime suspect --------------------------------
    FeatureSpec("curly_apostrophe", "unicode", char_rate("’"), "per1k", "U+2019"),
    FeatureSpec("curly_single_open", "unicode", char_rate("‘"), "per1k", "U+2018"),
    FeatureSpec("straight_apostrophe", "unicode", char_rate("'"), "per1k", "ASCII '"),
    FeatureSpec(
        "curly_double_quote", "unicode", char_rate("“”"), "per1k", "U+201C/D"
    ),
    FeatureSpec("straight_double_quote", "unicode", char_rate('"'), "per1k", 'ASCII "'),
    FeatureSpec("em_dash", "unicode", char_rate("—"), "per1k", "U+2014"),
    FeatureSpec("en_dash", "unicode", char_rate("–"), "per1k", "U+2013"),
    FeatureSpec("ascii_double_hyphen", "unicode", regex_rate(r"--"), "per1k", ""),
    FeatureSpec("ellipsis_char", "unicode", char_rate("…"), "per1k", "U+2026"),
    FeatureSpec("ellipsis_ascii", "unicode", regex_rate(r"\.\.\."), "per1k", ""),
    FeatureSpec("nbsp", "unicode", char_rate(" "), "per1k", "U+00A0"),
    FeatureSpec("bullet_char", "unicode", char_rate("•·"), "per1k", ""),
    FeatureSpec("other_non_ascii", "unicode", other_non_ascii_rate, "per1k", ""),
    FeatureSpec(
        "any_curly_quote",
        "unicode",
        regex_presence(r"[‘’“”]"),
        "binary",
        "the single bit from the grep",
    ),
    # --- whitespace / layout --------------------------------------------------
    FeatureSpec("newline", "whitespace", char_rate("\n"), "per1k", ""),
    FeatureSpec("blank_line_para", "whitespace", regex_rate(r"\n[ \t]*\n"), "per1k", ""),
    FeatureSpec(
        "single_line_doc", "whitespace", lambda t: float("\n" not in t.strip()), "binary", ""
    ),
    FeatureSpec("tab_char", "whitespace", char_rate("\t"), "per1k", ""),
    FeatureSpec("crlf", "whitespace", regex_rate(r"\r\n"), "per1k", ""),
    FeatureSpec("double_space", "whitespace", regex_rate(r"(?<=\S)  +(?=\S)"), "per1k", ""),
    FeatureSpec("trailing_ws_lines", "whitespace", trailing_ws_line_frac, "frac", ""),
    FeatureSpec(
        "leading_whitespace",
        "whitespace",
        lambda t: float(bool(t) and t[0].isspace()),
        "binary",
        "",
    ),
    FeatureSpec(
        "trailing_newline", "whitespace", lambda t: float(t.endswith("\n")), "binary", ""
    ),
    FeatureSpec("mean_line_length", "whitespace", mean_line_length, "raw", ""),
    # --- document structure / boilerplate -------------------------------------
    FeatureSpec("first_line_is_header", "structure", first_line_is_header, "binary", ""),
    FeatureSpec(
        "leading_label_colon",
        "structure",
        regex_presence(r"^\s*(Title|Introduction|Abstract|Summary|Essay)\s*:", re.I),
        "binary",
        "",
    ),
    FeatureSpec(
        "markdown_heading", "structure", regex_rate(r"(?m)^#{1,6}\s"), "per1k", ""
    ),
    FeatureSpec("markdown_bold", "structure", regex_rate(r"\*\*"), "per1k", ""),
    FeatureSpec(
        "numbered_section", "structure", regex_rate(r"(?m)^\s*\d+[.)]\s+[A-Z]"), "per1k", ""
    ),
    FeatureSpec("bullet_list_line", "structure", regex_rate(r"(?m)^\s*[-*•]\s"), "per1k", ""),
    FeatureSpec(
        "all_caps_line", "structure", regex_rate(r"(?m)^[A-Z0-9 ,.'\"-]{8,}$"), "per1k", ""
    ),
    FeatureSpec(
        "newswire_dateline",
        "structure",
        regex_presence(r"^[A-Z][A-Z .]{2,30},?\s+(\w+\s+\d{1,2}\s*)?[-–—]"),
        "binary",
        "Reuters-style opener",
    ),
    # --- lexical tells --------------------------------------------------------
    FeatureSpec(
        "in_conclusion", "lexical", regex_presence(r"\bin conclusion\b", re.I), "binary", ""
    ),
    FeatureSpec(
        "overall_opener", "lexical", regex_rate(r"(?m)^\s*(Overall|In summary|Ultimately)\b"), "per1k", ""
    ),
    FeatureSpec(
        "llmism_words",
        "lexical",
        regex_rate(
            r"\b(delve|tapestry|underscore[sd]?|moreover|furthermore|nuanced|"
            r"multifaceted|it'?s worth noting|it is important to note)\b",
            re.I,
        ),
        "per1k",
        "",
    ),
    FeatureSpec(
        "inline_citation",
        "lexical",
        regex_rate(r"\([A-Z][a-z]+,? [0-9]{4}\)"),
        "per1k",
        "academic (Author, 1999)",
    ),
    FeatureSpec(
        "works_cited", "lexical", regex_presence(r"\b(Works Cited|References)\b"), "binary", ""
    ),
    FeatureSpec("contraction", "lexical", regex_rate(r"\b\w+['’](s|t|re|ve|ll|d|m)\b"), "per1k", ""),
    FeatureSpec("digit", "lexical", regex_rate(r"\d"), "per1k", ""),
    FeatureSpec("exclamation", "lexical", char_rate("!"), "per1k", ""),
    FeatureSpec("question_mark", "lexical", char_rate("?"), "per1k", ""),
    FeatureSpec("semicolon", "lexical", char_rate(";"), "per1k", ""),
    FeatureSpec("colon", "lexical", char_rate(":"), "per1k", ""),
    FeatureSpec("comma", "lexical", char_rate(","), "per1k", ""),
    FeatureSpec("parenthesis", "lexical", char_rate("()"), "per1k", ""),
    # --- length ---------------------------------------------------------------
    FeatureSpec("n_chars", "length", lambda t: float(len(t)), "raw", ""),
    FeatureSpec("n_words", "length", lambda t: float(len(t.split())), "raw", ""),
    FeatureSpec("mean_word_length", "length", mean_word_length, "raw", ""),
    FeatureSpec("mean_sentence_words", "length", mean_sentence_words, "raw", ""),
    FeatureSpec("type_token_ratio", "length", type_token_ratio, "raw", ""),
)


# ---------------------------------------------------------------------------
# Single-feature discrimination
# ---------------------------------------------------------------------------


def best_threshold_rule(values: np.ndarray, y: np.ndarray) -> tuple[float, int, float]:
    """Accuracy of the best single-threshold rule on one feature.

    Returns ``(accuracy, direction, threshold)`` where ``direction`` is +1 when
    "value > threshold => machine" is the winning rule and -1 when
    "value <= threshold => machine" is. Both directions and the two degenerate
    all-one-class rules are considered, so the accuracy never falls below the
    majority-class baseline — read it against ``majority_acc``, not against 0.5,
    on an unbalanced corpus.

    Ties in ``values`` are handled by only allowing a split where the value
    actually changes; otherwise a threshold inside a tied block would claim an
    accuracy no rule can achieve.
    """
    n = len(y)
    if n == 0:
        return 0.5, 1, 0.0

    order = np.argsort(values, kind="mergesort")
    v = values[order]
    yy = y[order].astype(np.int64)

    n1 = int(yy.sum())  # machine
    n0 = n - n1

    c1 = np.cumsum(yy)  # machine count at or below position i
    c0 = np.cumsum(1 - yy)  # human count at or below position i

    # Split points: between i and i+1, only where the value changes.
    can_split = np.flatnonzero(v[:-1] < v[1:])

    # Degenerate rules, both expressed with dir=+1 so the caller's
    # "value > threshold => machine" reconstruction stays valid:
    #   thr=-inf -> everything is machine   thr=+inf -> everything is human
    best_correct = max(n1, n0)
    best_dir = 1
    best_thr = float("-inf") if n1 >= n0 else float("inf")

    if can_split.size:
        # dir=+1: human at or below the split, machine above.
        pos = c0[can_split] + (n1 - c1[can_split])
        # dir=-1: machine at or below the split, human above.
        neg = c1[can_split] + (n0 - c0[can_split])

        i_pos = int(np.argmax(pos))
        i_neg = int(np.argmax(neg))
        if pos[i_pos] > best_correct:
            best_correct = int(pos[i_pos])
            best_dir = 1
            best_thr = float(v[can_split[i_pos]])
        if neg[i_neg] > best_correct:
            best_correct = int(neg[i_neg])
            best_dir = -1
            best_thr = float(v[can_split[i_neg]])

    return best_correct / n, best_dir, best_thr


def mi_bits(pred: np.ndarray, y: np.ndarray) -> float:
    """Mutual information between a binary prediction and the label, in bits.

    Computed analytically on the 2x2 table, so it is exactly reproducible —
    unlike the k-NN estimator in ``sklearn.feature_selection.mutual_info_classif``.
    """
    n = len(y)
    if n == 0:
        return 0.0
    total = 0.0
    for a in (0, 1):
        pa = float(np.mean(pred == a))
        if pa == 0.0:
            continue
        for b in (0, 1):
            pb = float(np.mean(y == b))
            pab = float(np.mean((pred == a) & (y == b)))
            if pab == 0.0 or pb == 0.0:
                continue
            total += pab * np.log2(pab / (pa * pb))
    return float(total)


def analyse_corpus(name: str, texts: list[str], y: np.ndarray) -> pd.DataFrame:
    """Every feature scored on one corpus, with its sign."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    majority_acc = max(n0, n1) / max(len(y), 1)

    rows = []
    for spec in FEATURES:
        values = np.asarray([spec.fn(t) for t in texts], dtype=float)
        acc, direction, thr = best_threshold_rule(values, y)
        pred = (values > thr).astype(int) if direction > 0 else (values <= thr).astype(int)

        rows.append(
            {
                "corpus": name,
                "feature": spec.name,
                "group": spec.group,
                "unit": spec.unit,
                "note": spec.note,
                "human_mean": float(values[y == 0].mean()) if n0 else float("nan"),
                "machine_mean": float(values[y == 1].mean()) if n1 else float("nan"),
                "presence_h": float((values[y == 0] > 0).mean()) if n0 else float("nan"),
                "presence_m": float((values[y == 1] > 0).mean()) if n1 else float("nan"),
                "threshold": thr,
                "acc": acc,
                "majority_acc": majority_acc,
                "lift": acc - majority_acc,
                "dir": direction,
                "mi_bits": mi_bits(pred, y),
                # Lift over the majority baseline, not over 0.5: a constant
                # feature on an unbalanced corpus scores majority_acc with an
                # arbitrary direction, and must contribute nothing to the sign
                # table or to the alignment.
                "strength": direction * max(0.0, acc - majority_acc),
                "n": len(y),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-corpus products
# ---------------------------------------------------------------------------


def row_flips(row: np.ndarray, eps: float = FLIP_EPS) -> bool:
    """True when one row of the sign table carries both signs.

    The row-level view of a flip: the feature is used by the classifier in
    opposite directions in different corpora. ``find_sign_flips`` is the
    stricter, pairwise version that also demands a minimum accuracy.
    """
    finite = np.asarray(row, dtype=float)
    finite = finite[~np.isnan(finite)]
    return bool(finite.size and finite.min() < -eps and finite.max() > eps)


def sign_table(per_corpus: pd.DataFrame, corpora: list[str]) -> pd.DataFrame:
    """Feature x corpus matrix of signed strength, ordered by peak strength."""
    table = per_corpus.pivot(index="feature", columns="corpus", values="strength")
    table = table.reindex(columns=corpora)
    order = table.abs().max(axis=1).sort_values(ascending=False).index
    return table.loc[order]


def find_sign_flips(
    per_corpus: pd.DataFrame, corpora: list[str], min_acc: float
) -> pd.DataFrame:
    """Features that are strong in two corpora and point opposite ways.

    ``flip_severity`` is the smaller of the two accuracies minus 0.5: a flip only
    hurts as much as the *weaker* of the two conflicting signals, since that is
    the one that has to be wrong for the transfer to invert.
    """
    idx = per_corpus.set_index(["feature", "corpus"])
    rows = []
    for feature in per_corpus["feature"].unique():
        strong = {}
        for corpus in corpora:
            try:
                rec = idx.loc[(feature, corpus)]
            except KeyError:
                continue
            # An accuracy that only matches the majority baseline is not signal,
            # however high it looks on an unbalanced corpus.
            if rec["acc"] >= min_acc and rec["acc"] > rec["majority_acc"]:
                strong[corpus] = (int(rec["dir"]), float(rec["acc"]))

        pos = {c: a for c, (d, a) in strong.items() if d > 0}
        neg = {c: a for c, (d, a) in strong.items() if d < 0}
        if not pos or not neg:
            continue

        for cp, ap in pos.items():
            for cn, an in neg.items():
                rows.append(
                    {
                        "feature": feature,
                        "machine_side_corpus": cp,
                        "machine_side_acc": ap,
                        "human_side_corpus": cn,
                        "human_side_acc": an,
                        "flip_severity": min(ap, an) - 0.5,
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("flip_severity", ascending=False).reset_index(drop=True)
    return out


def artifact_alignment(table: pd.DataFrame, corpora: list[str]) -> pd.DataFrame:
    """Cosine similarity between corpora of their signed strength vectors.

    +1  the two corpora's surface artifacts point the same way — a classifier
        that leans on them transfers *well* (for the wrong reason).
     0  unrelated artifacts — transfer should land near chance.
    -1  the artifacts oppose — a confident classifier transfers below chance.

    Symmetric by construction, while the observed transfer matrix is not, so
    compare it against the symmetrised observation.
    """
    mat = table.reindex(columns=corpora).fillna(0.0).to_numpy()
    norms = np.linalg.norm(mat, axis=0)
    norms[norms == 0] = 1.0
    unit = mat / norms
    return pd.DataFrame(unit.T @ unit, index=corpora, columns=corpora)


# ---------------------------------------------------------------------------
# Character n-grams
# ---------------------------------------------------------------------------


def doc_ngrams(text: str, n: int) -> set[str]:
    """The distinct character n-grams of one document."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def ngram_log_odds(
    texts: list[str], y: np.ndarray, n: int, min_doc_frac: float, prior: float
) -> pd.DataFrame:
    """Document-frequency log-odds per character n-gram, machine vs human.

    Document frequency rather than raw count, for the same reason the original
    evidence was gathered with ``grep -l``: one document that repeats a string a
    hundred times should not outvote a hundred documents that use it once.
    """
    df_h: Counter[str] = Counter()
    df_m: Counter[str] = Counter()
    for text, label in zip(texts, y):
        grams = doc_ngrams(text, n)
        (df_m if label == 1 else df_h).update(grams)

    n_m = int(y.sum())
    n_h = len(y) - n_m
    floor_h = min_doc_frac * max(n_h, 1)
    floor_m = min_doc_frac * max(n_m, 1)

    rows = []
    for gram in set(df_h) | set(df_m):
        h = df_h.get(gram, 0)
        m = df_m.get(gram, 0)
        if h < floor_h and m < floor_m:
            continue
        lo = np.log((m + prior) / (n_m - m + prior)) - np.log(
            (h + prior) / (n_h - h + prior)
        )
        rows.append(
            {
                "ngram": gram,
                "df_human": h,
                "df_machine": m,
                "frac_human": h / max(n_h, 1),
                "frac_machine": m / max(n_m, 1),
                "log_odds": float(lo),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_log_odds"] = out["log_odds"].abs()
        out = out.sort_values("abs_log_odds", ascending=False).reset_index(drop=True)
    return out


def ngram_flip_table(
    per_corpus_ngrams: dict[str, pd.DataFrame], corpora: list[str], top_k: int
) -> pd.DataFrame:
    """Log-odds of each corpus's top n-grams, in every corpus, flips flagged."""
    keep: set[str] = set()
    for corpus in corpora:
        frame = per_corpus_ngrams.get(corpus)
        if frame is not None and not frame.empty:
            keep.update(frame["ngram"].head(top_k))

    rows = []
    for gram in sorted(keep):
        row: dict[str, object] = {"ngram": gram}
        signs = []
        for corpus in corpora:
            frame = per_corpus_ngrams.get(corpus)
            value = float("nan")
            if frame is not None and not frame.empty:
                hit = frame.loc[frame["ngram"] == gram, "log_odds"]
                if len(hit):
                    value = float(hit.iloc[0])
            row[corpus] = value
            if not np.isnan(value) and abs(value) >= 1.0:
                signs.append(np.sign(value))
        row["flips"] = bool(signs and len(set(signs)) > 1)
        row["max_abs"] = float(
            np.nanmax([abs(row[c]) for c in corpora]) if corpora else float("nan")
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["flips", "max_abs"], ascending=[False, False])
        out = out.reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Comparison against an observed transfer matrix
# ---------------------------------------------------------------------------


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared — the ranking Spearman needs."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # Average the ranks inside each tied block.
    sorted_values = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or sorted_values[i] != sorted_values[start]:
            ranks[order[start:i]] = (start + i - 1) / 2.0
            start = i
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, computed with numpy only."""
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / denom) if denom else float("nan")


def load_transfer_matrix(path: Path, corpora: list[str]) -> pd.DataFrame | None:
    """Off-diagonal accuracy and macro-F1 from a ``cross_domain.json``."""
    with open(path) as f:
        payload = json.load(f)
    summary = payload.get("summary", {})

    rows = []
    for src in corpora:
        for tgt in corpora:
            if src == tgt:
                continue
            cell = summary.get(src, {}).get(tgt)
            if cell is None:
                continue
            rows.append(
                {
                    "source": src,
                    "target": tgt,
                    "accuracy": float(cell.get("accuracy_mean", float("nan"))),
                    "macro_f1": float(cell.get("macro_f1_mean", float("nan"))),
                }
            )
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _finish(fig: plt.Figure, fig_dir: Path, fname: str) -> None:
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / fname, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  [OK] {fname}")


def plot_sign_heatmap(
    table: pd.DataFrame, corpora: list[str], fig_dir: Path, top_n: int
) -> None:
    """Signed strength per feature per corpus — the sign table, as a picture."""
    shown = table.head(top_n)
    if shown.empty:
        return

    data = shown.to_numpy(dtype=float)
    n_rows, n_cols = data.shape
    flipped = [row_flips(data[i]) for i in range(n_rows)]

    # The FLIP tags go in a reserved gutter to the RIGHT of the last column.
    # Putting them left of the y axis (where they were) drew them straight on
    # top of the feature names.
    gutter = FLIP_GUTTER if any(flipped) else 0.0

    height = max(4.0, 0.30 * n_rows + 1.8)
    width = 1.6 * len(corpora) + 5.5 + (0.8 if gutter else 0.0)
    fig, ax = plt.subplots(figsize=(width, height))

    limit = float(np.nanmax(np.abs(data))) or 0.5
    im = ax.imshow(data, cmap=CMAP_SIGN, vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(shown.columns, rotation=30, ha="right", fontsize=9,
                       color=INK_SECONDARY)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(shown.index, fontsize=8, color=INK_SECONDARY)

    # Direct labels: the value is the point, and there are few enough cells.
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.6 * limit else INK_PRIMARY,
            )

    # Widen the x range rather than drawing outside it, so the tags are inside
    # the axes and tight_layout accounts for them.
    ax.set_xlim(-0.5, n_cols - 0.5 + gutter)
    for i, is_flip in enumerate(flipped):
        if is_flip:
            ax.text(n_cols - 0.40, i, "FLIP", ha="left", va="center", fontsize=7,
                    color="#8f2321", fontweight="bold")

    ax.set_title(
        "Signed single-feature strength   dir x (accuracy - majority baseline)\n"
        "red = higher in machine   blue = higher in human   "
        "both signs in a row = the artifact inverts across corpora",
        fontsize=11, color=INK_PRIMARY,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("signed strength", fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(np.arange(-0.5, len(shown.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(shown), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)  # 2px surface gap
    ax.tick_params(which="minor", length=0)

    _finish(fig, fig_dir, "fig_sign_heatmap.png")


def plot_alignment(alignment: pd.DataFrame, fig_dir: Path) -> None:
    """Corpus x corpus artifact alignment — the predicted transfer damage."""
    fig, ax = plt.subplots(figsize=(1.1 * len(alignment) + 4.0, 1.0 * len(alignment) + 3.0))
    data = alignment.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=CMAP_SIGN, vmin=-1.0, vmax=1.0)

    ax.set_xticks(range(len(alignment)))
    ax.set_xticklabels(alignment.columns, rotation=30, ha="right", fontsize=9,
                       color=INK_SECONDARY)
    ax.set_yticks(range(len(alignment)))
    ax.set_yticklabels(alignment.index, fontsize=9, color=INK_SECONDARY)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else INK_PRIMARY)

    ax.set_title(
        "Artifact alignment between corpora\n"
        "cosine of the signed surface-feature profiles; "
        "negative predicts below-chance transfer",
        fontsize=11, color=INK_PRIMARY,
    )
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).set_label(
        "alignment", fontsize=9, color=INK_SECONDARY
    )
    ax.set_xticks(np.arange(-0.5, len(alignment), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(alignment), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    _finish(fig, fig_dir, "fig_alignment_heatmap.png")


def plot_alignment_vs_transfer(
    merged: pd.DataFrame, rho_acc: float, rho_f1: float, fig_dir: Path
) -> None:
    """The falsification plot, one figure per metric.

    Accuracy and macro-F1 tell nearly the same story on a class-balanced test
    set, so they get a full canvas each rather than sharing one and colliding.

    Read a panel in this order.

    1. The TREND. Spearman rho over all points is the actual statistic. Strongly
       positive means the surface artifacts explain the transfer pattern.
    2. The QUADRANTS. Green where prediction and observation agree (artifacts
       oppose and transfer fails / artifacts agree and transfer works), red
       where they contradict. A crude dichotomisation of a continuous
       relationship — an aid to reading the trend, not a replacement for it.
    3. The CONTRADICTIONS, ringed and named. Each is a pair the artifact story
       does not cover, and needs its own explanation before the story is done.

    Two reference lines on the macro-F1 figure, because on a class-balanced
    test set they mean different things:

      0.500  a coin flip
      0.333  a CONSTANT predictor — everything called one class

    A cell at F1 ~0.33 with accuracy ~0.50 has not inverted; it has collapsed
    onto one label, which is a shifted decision boundary rather than a reversed
    one. Only a cell with accuracy clearly below 0.50 is genuinely inverted.
    That distinction changes which fix applies, so the lines are drawn.
    """
    _plot_alignment_panel(
        merged, "accuracy", rho_acc, "Transfer accuracy",
        fig_dir, "fig_alignment_vs_accuracy.png",
    )
    _plot_alignment_panel(
        merged, "macro_f1", rho_f1, "Transfer macro-F1",
        fig_dir, "fig_alignment_vs_macro_f1.png",
    )


def _plot_alignment_panel(
    merged: pd.DataFrame,
    col: str,
    rho: float,
    label: str,
    fig_dir: Path,
    fname: str,
) -> None:
    """One metric, one figure.

    Deliberately carries no legend and no statistics box. The ring around a
    point says exactly what the red quadrant it sits in already says, and the
    numbers read better in the subtitle than in a box floating over the data —
    both were overlays competing with the points for the same space.
    """
    support = "#e7f3ec"  # prediction and observation agree
    contradict = "#fdecea"  # they do not
    ring = "#e34948"
    chance = 0.5

    x = merged["alignment"].to_numpy(dtype=float)
    y = merged[col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))

    pad_x = 0.10 * (np.ptp(x) or 1.0)
    pad_y = 0.12 * (np.ptp(y) or 1.0)
    xlim = (x.min() - pad_x, x.max() + pad_x)
    ylim = (min(y.min() - pad_y, chance - pad_y), max(y.max() + pad_y, chance + pad_y))

    # --- quadrants ----------------------------------------------------------
    for x0, x1, y0, y1, colour in (
        (xlim[0], 0.0, ylim[0], chance, support),      # oppose -> fails   OK
        (0.0, xlim[1], chance, ylim[1], support),      # agree  -> works   OK
        (xlim[0], 0.0, chance, ylim[1], contradict),   # oppose -> works   ??
        (0.0, xlim[1], ylim[0], chance, contradict),   # agree  -> fails   ??
    ):
        ax.add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=colour,
                          edgecolor="none", zorder=0)
        )

    agrees = ((x < 0) & (y < chance)) | ((x > 0) & (y > chance))
    n_ok = int(agrees.sum())
    n_bad = int((~agrees).sum())

    for xf, yf, ha, va, count, text in (
        (0.012, 0.012, "left", "bottom", int(((x < 0) & (y < chance)).sum()),
         "artifacts OPPOSE → transfer fails\nas predicted"),
        (0.988, 0.988, "right", "top", int(((x > 0) & (y > chance)).sum()),
         "artifacts AGREE → transfer works\nas predicted"),
        (0.012, 0.988, "left", "top", int(((x < 0) & (y > chance)).sum()),
         "artifacts oppose but transfer WORKS\ncontradicts"),
        (0.988, 0.012, "right", "bottom", int(((x > 0) & (y < chance)).sum()),
         "artifacts agree but transfer FAILS\ncontradicts"),
    ):
        ax.text(xf, yf, f"{text}: {count}", transform=ax.transAxes, ha=ha, va=va,
                fontsize=8, color=INK_SECONDARY, linespacing=1.6, zorder=1)

    # --- reference lines ----------------------------------------------------
    ax.axvline(0.0, color=INK_SECONDARY, linestyle=":", linewidth=1, zorder=2)
    ax.axhline(chance, color=INK_SECONDARY, linestyle=":", linewidth=1, zorder=2)
    ax.text(xlim[0], chance, " chance 0.500", ha="left", va="bottom",
            fontsize=7.5, color=INK_SECONDARY, zorder=2)

    if col == "macro_f1":
        constant_f1 = 1.0 / 3.0
        if ylim[0] < constant_f1 < ylim[1]:
            ax.axhline(constant_f1, color="#8f2321", linestyle="--",
                       linewidth=1, alpha=0.7, zorder=2)
            ax.text(xlim[0], constant_f1, " constant predictor 0.333",
                    ha="left", va="bottom", fontsize=7.5, color="#8f2321", zorder=2)

    # --- trend, direct-labelled instead of put in a legend -------------------
    if len(x) >= 3 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.array(xlim)
        ax.plot(xs, slope * xs + intercept, color="#1c5cab", linewidth=2,
                alpha=0.85, zorder=2)
        y_end = float(np.clip(slope * xlim[1] + intercept, *ylim))
        ax.text(xlim[1], y_end, "trend ", ha="right", va="bottom", fontsize=8,
                color="#1c5cab", fontweight="bold", zorder=3)

    # --- points -------------------------------------------------------------
    ax.scatter(x[agrees], y[agrees], s=80, color="#2a78d6", edgecolor="white",
               linewidth=2, zorder=4)
    ax.scatter(x[~agrees], y[~agrees], s=80, color="#2a78d6", edgecolor=ring,
               linewidth=2.2, zorder=5)

    # Name every contradiction plus the single worst cell; nothing else, or the
    # labels become the chart. Offsets point away from the vertical midline so
    # the two side clusters lean outwards instead of into each other.
    to_label = pd.concat([merged[~agrees], merged.nsmallest(1, col)])
    to_label = to_label.drop_duplicates(subset=["source", "target"])
    x_mid = 0.5 * (xlim[0] + xlim[1])
    for _, r in to_label.iterrows():
        left = r["alignment"] < x_mid
        ax.annotate(
            f"{r['source']}→{r['target']}",
            (r["alignment"], r[col]),
            textcoords="offset points",
            xytext=(8 if left else -8, 6),
            ha="left" if left else "right",
            fontsize=7.5, color=INK_PRIMARY, zorder=6,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(
        "Predicted artifact alignment (cosine)\n"
        "← artifacts oppose          artifacts agree →",
        fontsize=9.5, color=INK_SECONDARY,
    )
    ax.set_ylabel(label, fontsize=10.5, color=INK_SECONDARY)

    strength = "strong" if abs(rho) >= 0.7 else "moderate" if abs(rho) >= 0.4 else "weak"
    ax.set_title(
        f"{label} vs predicted artifact alignment\n"
        f"Spearman ρ = {rho:+.3f} ({strength})   ·   n = {len(x)} pairs   ·   "
        f"consistent {n_ok} / contradicting {n_bad} (ringed)\n"
        "CONFIRM: points slope up, most sit in the green quadrants.   "
        "REJECT: flat cloud, or failures on both sides of zero.",
        fontsize=10, color=INK_PRIMARY, linespacing=1.7, pad=14,
    )
    ax.grid(True, alpha=0.2, zorder=1)
    ax.set_axisbelow(True)

    _finish(fig, fig_dir, fname)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_corpus(
    name: str, parquet_path: Path, drop_classes: list[str], out: Tee
) -> tuple[list[str], np.ndarray]:
    """One corpus as (texts, binary labels), binarised human=0 / machine=1."""
    df = pd.read_parquet(parquet_path)
    for col in (TEXT_COL, TARGET_COL):
        if col not in df.columns:
            raise SystemExit(f"{parquet_path.name}: missing column {col!r}.")

    if drop_classes:
        df = df[~df[TARGET_COL].isin(drop_classes)]

    df = df[df[TEXT_COL].notna()]
    df = df[df[TEXT_COL].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)

    texts = df[TEXT_COL].astype(str).tolist()
    y = (df[TARGET_COL] != HUMAN_LABEL).astype(int).to_numpy()

    counts = df[TARGET_COL].value_counts().to_dict()
    out.print(f"  {name:<20} {len(df):>5} rows   human={int((y == 0).sum())}   "
              f"machine={int((y == 1).sum())}   {counts}")
    return texts, y


def datasets_from_args(args: argparse.Namespace, out: Tee) -> list[tuple[str, Path, list[str]]]:
    """(name, parquet path, drop_classes) per corpus, from --config or positionals."""
    if args.config:
        cfg = load_cd_config(Path(args.config))
        specs = cfg["data"]["datasets"]
        out.print(f"Config: {args.config}   ({len(specs)} datasets)")
        return [
            (s["name"], BASE_DIR / s["parquet_path"], list(s.get("drop_classes", [])))
            for s in specs
        ]

    drop = list(DEFAULT_DROP_CLASSES) if args.drop_classes is None else args.drop_classes
    out.print(f"Parquets given on the command line; drop_classes={drop}")
    return [(Path(p).stem, Path(p), drop) for p in args.parquets]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report_corpus(frame: pd.DataFrame, name: str, top_n: int, out: Tee) -> None:
    top = frame.sort_values("acc", ascending=False).head(top_n)
    out.print(f"\n[{name}]  top {len(top)} features by single-feature accuracy")
    header = (
        f"  {'feature':<24}{'grp':<11}{'acc':>7}{'lift':>7}{'higher in':>10}{'MI':>7}"
        f"{'human':>11}{'machine':>11}{'pres_h':>8}{'pres_m':>8}"
    )
    out.print(header)
    out.print("  " + "-" * (len(header) - 2))
    for _, r in top.iterrows():
        arrow = "machine" if r["dir"] > 0 else "human"
        out.print(
            f"  {r['feature']:<24}{r['group']:<11}{r['acc']:>7.3f}{r['lift']:>7.3f}"
            f"{arrow:>10}{r['mi_bits']:>7.3f}"
            f"{r['human_mean']:>11.3f}{r['machine_mean']:>11.3f}"
            f"{r['presence_h']:>8.2f}{r['presence_m']:>8.2f}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "parquets", nargs="*", help="Parquet files to audit (ignored when --config is given)."
    )
    ap.add_argument(
        "--config",
        default=None,
        help="Cross-domain YAML config; takes the dataset list and drop_classes "
        "from it so the audit sees exactly what the classifier was trained on.",
    )
    ap.add_argument(
        "--drop-classes",
        nargs="*",
        default=None,
        help="Generators to drop in positional mode "
        f"(default: {' '.join(DEFAULT_DROP_CLASSES)}).",
    )
    ap.add_argument(
        "--compare-matrix",
        default=None,
        help="Path to a cross_domain.json; correlates predicted artifact "
        "alignment against the observed transfer matrix.",
    )
    ap.add_argument(
        "--flip-min-acc",
        type=float,
        default=FLIP_MIN_ACC,
        help=f"Single-feature accuracy a feature needs in both corpora to count "
        f"as a sign flip (default: {FLIP_MIN_ACC}).",
    )
    ap.add_argument(
        "--top-n", type=int, default=25, help="Features listed per corpus in the log."
    )
    ap.add_argument(
        "--heatmap-features",
        type=int,
        default=35,
        help="Features shown in the sign heatmap, strongest first.",
    )
    ap.add_argument(
        "--no-ngrams", action="store_true", help="Skip the character n-gram section."
    )
    ap.add_argument(
        "--tag", default=None, help="Suffix for the results directory name."
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config and not args.parquets:
        raise SystemExit("Give --config or at least one parquet path.")

    run_id = "artifacts_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.tag:
        run_id += f"_{args.tag}"
    fig_dir = RESULTS_DIR / run_id
    fig_dir.mkdir(parents=True, exist_ok=True)

    out = Tee()
    out.print("=" * 96)
    out.print("Surface-artifact audit")
    out.print(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    out.print(f"Output   : {fig_dir}")
    out.print("=" * 96)

    specs = datasets_from_args(args, out)
    out.print("")

    corpora: list[str] = []
    texts_by_corpus: dict[str, list[str]] = {}
    labels_by_corpus: dict[str, np.ndarray] = {}
    frames: list[pd.DataFrame] = []

    for name, path, drop in specs:
        if not path.is_file():
            out.print(f"  {name:<20} SKIPPED — not found: {path}")
            continue
        texts, y = load_corpus(name, path, drop, out)
        if len(set(y.tolist())) < 2:
            out.print(f"  {name:<20} SKIPPED — only one class present.")
            continue
        corpora.append(name)
        texts_by_corpus[name] = texts
        labels_by_corpus[name] = y
        frames.append(analyse_corpus(name, texts, y))

    if not frames:
        raise SystemExit("No usable corpora.")

    per_corpus = pd.concat(frames, ignore_index=True)
    per_corpus.to_csv(fig_dir / "features_per_corpus.csv", index=False)

    for name in corpora:
        report_corpus(per_corpus[per_corpus["corpus"] == name], name, args.top_n, out)

    # --- the sign table -----------------------------------------------------
    table = sign_table(per_corpus, corpora)
    table.to_csv(fig_dir / "sign_table.csv")

    out.print("\n" + "=" * 96)
    out.print("SIGN TABLE  —  dir x (accuracy - majority baseline) per corpus")
    out.print("  positive = the feature is higher in MACHINE text")
    out.print("  negative = the feature is higher in HUMAN text")
    out.print("=" * 96)
    header = f"  {'feature':<26}" + "".join(f"{c[:14]:>16}" for c in corpora)
    out.print(header)
    out.print("  " + "-" * (len(header) - 2))
    for feature, row in table.iterrows():
        values = row.to_numpy(dtype=float)
        flag = "  FLIP" if row_flips(values) else ""
        out.print(
            f"  {feature:<26}" + "".join(f"{v:>+16.3f}" for v in values) + flag
        )

    # --- flips --------------------------------------------------------------
    flips = find_sign_flips(per_corpus, corpora, args.flip_min_acc)
    flips.to_csv(fig_dir / "sign_flips.csv", index=False)

    out.print("\n" + "=" * 96)
    out.print(
        f"SIGN FLIPS  —  features with accuracy >= {args.flip_min_acc:.2f} in two "
        "corpora and opposite signs"
    )
    out.print("=" * 96)
    if flips.empty:
        out.print("  none — no feature is strong in two corpora with opposite signs.")
    else:
        out.print(
            f"  {'feature':<26}{'says MACHINE in':<20}{'acc':>7}   "
            f"{'says HUMAN in':<20}{'acc':>7}{'severity':>10}"
        )
        out.print("  " + "-" * 92)
        for _, r in flips.iterrows():
            out.print(
                f"  {r['feature']:<26}{r['machine_side_corpus']:<20}"
                f"{r['machine_side_acc']:>7.3f}   {r['human_side_corpus']:<20}"
                f"{r['human_side_acc']:>7.3f}{r['flip_severity']:>10.3f}"
            )

    # --- alignment ----------------------------------------------------------
    alignment = artifact_alignment(table, corpora)
    alignment.to_csv(fig_dir / "artifact_alignment.csv")

    out.print("\n" + "=" * 96)
    out.print("ARTIFACT ALIGNMENT  —  cosine between corpora of the signed profiles")
    out.print("  +1 artifacts agree (transfer flatters the model)")
    out.print("  -1 artifacts oppose (transfer inverts: below chance)")
    out.print("=" * 96)
    out.print(f"  {'':<20}" + "".join(f"{c[:14]:>16}" for c in corpora))
    for src in corpora:
        out.print(
            f"  {src:<20}" + "".join(f"{alignment.loc[src, t]:>+16.3f}" for t in corpora)
        )

    # --- character n-grams --------------------------------------------------
    if not args.no_ngrams:
        scopes = {
            "full": lambda t: t,
            f"head{HEAD_TAIL_CHARS}": lambda t: t[:HEAD_TAIL_CHARS],
            f"tail{HEAD_TAIL_CHARS}": lambda t: t[-HEAD_TAIL_CHARS:],
        }
        for scope, slicer in scopes.items():
            per_ngram: dict[str, pd.DataFrame] = {}
            for name in corpora:
                sliced = [slicer(t) for t in texts_by_corpus[name]]
                frame = ngram_log_odds(
                    sliced, labels_by_corpus[name], NGRAM_N, NGRAM_MIN_DOC_FRAC, NGRAM_PRIOR
                )
                frame.insert(0, "corpus", name)
                per_ngram[name] = frame

            joined = pd.concat(per_ngram.values(), ignore_index=True)
            joined.to_csv(fig_dir / f"char_ngrams_{scope}.csv", index=False)

            flip_frame = ngram_flip_table(per_ngram, corpora, NGRAM_TOP_K)
            flip_frame.to_csv(fig_dir / f"char_ngram_flips_{scope}.csv", index=False)

            out.print("\n" + "=" * 96)
            out.print(
                f"CHARACTER {NGRAM_N}-GRAM LOG-ODDS  —  scope: {scope}  "
                f"(positive = machine, negative = human)"
            )
            out.print("=" * 96)
            for name in corpora:
                frame = per_ngram[name]
                if frame.empty:
                    out.print(f"\n  [{name}] no n-gram passed the min document frequency.")
                    continue
                top = frame.head(12)
                shown = "  ".join(
                    f"{gram!r}:{lo:+.2f}"
                    for gram, lo in zip(top["ngram"], top["log_odds"])
                )
                out.print(f"\n  [{name}]  {shown}")

            n_flips = int(flip_frame["flips"].sum()) if not flip_frame.empty else 0
            out.print(
                f"\n  n-grams flipping sign across corpora (|log-odds| >= 1 on both "
                f"sides): {n_flips} of {len(flip_frame)}"
            )
            if n_flips:
                worst = flip_frame[flip_frame["flips"]].head(10)
                for _, r in worst.iterrows():
                    cells = "  ".join(
                        f"{c[:10]}:{r[c]:+.2f}" if not np.isnan(r[c]) else f"{c[:10]}:  -  "
                        for c in corpora
                    )
                    out.print(f"    {r['ngram']!r:<8} {cells}")

    # --- comparison against the observed transfer matrix --------------------
    merged = None
    if args.compare_matrix:
        observed = load_transfer_matrix(Path(args.compare_matrix), corpora)
        if observed is None or observed.empty:
            out.print(f"\nNo comparable cells found in {args.compare_matrix}.")
        else:
            observed["alignment"] = [
                float(alignment.loc[r["source"], r["target"]])
                for _, r in observed.iterrows()
            ]
            merged = observed
            merged.to_csv(fig_dir / "alignment_vs_transfer.csv", index=False)

            rho_acc = spearman(
                merged["alignment"].to_numpy(), merged["accuracy"].to_numpy()
            )
            rho_f1 = spearman(
                merged["alignment"].to_numpy(), merged["macro_f1"].to_numpy()
            )

            out.print("\n" + "=" * 96)
            out.print("PREDICTED ALIGNMENT vs OBSERVED TRANSFER")
            out.print(f"  matrix: {args.compare_matrix}")
            out.print(f"  off-diagonal cells compared: {len(merged)}")
            out.print(f"  Spearman rho (alignment, transfer accuracy): {rho_acc:+.3f}")
            out.print(f"  Spearman rho (alignment, transfer macro-F1): {rho_f1:+.3f}")
            out.print("=" * 96)
            out.print(
                f"  {'source':<20}{'target':<20}{'align':>9}{'acc':>9}{'macroF1':>9}"
            )
            out.print("  " + "-" * 65)
            for _, r in merged.sort_values("macro_f1").iterrows():
                out.print(
                    f"  {r['source']:<20}{r['target']:<20}{r['alignment']:>+9.3f}"
                    f"{r['accuracy']:>9.3f}{r['macro_f1']:>9.3f}"
                )
            out.print(
                "\n  Reading: a strong positive rho means the surface artifacts "
                "explain the transfer pattern.\n"
                "  A rho near zero means they do not, and the collapse has another "
                "cause — in which case\n  this whole diagnosis is wrong and step 2 "
                "will not fix it."
            )

    # --- figures ------------------------------------------------------------
    print()
    plot_sign_heatmap(table, corpora, fig_dir, args.heatmap_features)
    plot_alignment(alignment, fig_dir)
    if merged is not None:
        rho_acc = spearman(merged["alignment"].to_numpy(), merged["accuracy"].to_numpy())
        rho_f1 = spearman(merged["alignment"].to_numpy(), merged["macro_f1"].to_numpy())
        plot_alignment_vs_transfer(merged, rho_acc, rho_f1, fig_dir)

    out.save(fig_dir / "surface_artifacts.log")
    print(f"\nAll outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()

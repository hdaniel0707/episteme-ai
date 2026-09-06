#!/usr/bin/env python3
"""Is this parquet fully and sanely populated? One report, any schema, an exit code.

Column by column: nulls, empty strings / arrays, non-finite numbers, and — the
check a null count misses — **degenerate columns**: a score or embedding column
that came out constant, near-constant, all-zero, or ragged. That is the shape a
flaky scoring endpoint leaves behind (every row "filled" with 0.5), and nothing
downstream notices until a figure comes out at AUROC 0.5.

The other thing a null count misses is a **failed call written as content**: an
API error body, a refused completion, or leaked model scaffolding (``<think>``,
``<|channel|>``) saved into a ``text`` / ``abstract`` column. That row is not
null and not empty, so only its wording gives it away. Every ``TEXTISH`` column
is scanned for those signatures, for byte-identical duplicate documents (one
error string repeated across N rows is the classic tell), for text that is
**not an abstract at all** — a scrape of the publisher's page, a book's table of
contents, a reference list (``rules.abstract_is_boilerplate``, the same test the
fetch/filter stages reject on) — and for text that is **not English** (a
wrong-language abstract, or a generator that answered in a non-English title's
language, leaving an English human row paired with a non-English machine one).

When both ``domain`` and ``generated_by`` are present it also prints the
domain × generated_by count matrix and the rows-per-``id`` count, and warns if a
training corpus (more than one ``generated_by`` class) is uneven across those —
a generator that dropped papers, a domain that came up short.

Not tied to the training schema (that is ``epai/ai_detection/analyse/audit_corpus.py``,
which checks pairing, balance and text quality in full). This one answers the
narrower question: did every stage that was supposed to write a column actually
write a real value into every row.

Runs on the repo interpreter — ``pyarrow`` + ``numpy``, one column in memory at
a time (~0.8 GB, ~20 s on a 320 MB / 6000-row corpus).

    .venv/bin/python epai/scripts/check_parquet_completeness.py data/parquet/denario_hc.parquet

    .venv/bin/python epai/scripts/check_parquet_completeness.py data/parquet/science_v3f_multi_0811A.parquet

    .venv/bin/python epai/scripts/check_parquet_completeness.py data/parquet/ghostbuster_multi.parquet


    # does this file have every column science_v3 has?
    .venv/bin/python epai/scripts/check_parquet_completeness.py data/parquet/denario_hc.parquet \
        --reference data/parquet/science_v3_gpt56luna_0811A.parquet

    # gate a pipeline: exit 1 on any ERROR (or on WARN too, with --strict)
    .venv/bin/python epai/scripts/check_parquet_completeness.py data/parquet/denario_hc.parquet \
        --fail-on error  --json results/denario_hc/completeness.json

Exit code: 0 if clean (or only WARN, unless --strict / --fail-on warn), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from epai.science_data.rules import (
        FilterConfig,
        _function_word_ratio,
        abstract_is_boilerplate,
        title_looks_english,
    )
    _TITLE_CFG = FilterConfig()
except Exception:  # keep the checker runnable outside the package
    abstract_is_boilerplate = _function_word_ratio = title_looks_english = None

# Below this share of English function words a document is not English prose —
# a wrong-language abstract, or a generator that answered in the title's
# language. Real English abstracts here sit at 0.13+, the non-English ones at
# 0.06 and below, so the threshold has a wide margin either side.
NON_ENGLISH_FW_RATIO = 0.12

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Bounded score columns — P(...) that must sit in [0, 1]. Anything outside is an
# ERROR, not a WARN.
BOUNDED_SCORE_PREFIXES = ("llm_judge_score_", "detector_score_")
# Every column a scorer/embedder fills with a meaningful value: a constant or
# near-constant one of these is a scorer that fell back to a default.
SCORE_PREFIXES = BOUNDED_SCORE_PREFIXES + (
    "binoculars_score", "fastdetect_crit_",
)

# Fixed-width vector columns: ragged / all-zero / NaN-inside are errors here.
# Other list columns (`*_cues_*`, `author_ids`, `queries`) are legitimately
# variable length.
VECTOR_PREFIXES = ("emb_",)
VECTOR_SUFFIXES = ("_cuevec_gpt56luna_v3",)
VECTOR_CONTAINS = ("_cuevec_",)

# Allowed to be entirely null — upstream does not always fill them and says so
# where they are written.
OPTIONAL_NULL_PREFIXES = ("fastdetect_prob_", "fastdetect_ntok_")

# Share of a score column's rows sitting on one value: at/above CONSTANT it is
# an ERROR (the scorer wrote a default), at/above SUSPICIOUS a WARN (could be a
# genuinely clumpy judge — worth a look, not necessarily broken).
CONSTANT_FRAC = 0.99
SUSPICIOUS_FRAC = 0.80

# Text columns whose emptiness is a real problem, and whose length is worth a
# distribution.
TEXTISH = ("text", "text_raw", "text_preclean", "abstract", "abstract_raw")
TITLEISH = ("title",)

# A failed API call is sometimes written into a text column as content — an
# error body, a refusal, or leaked model scaffolding. Not null, not empty; only
# the wording gives it away. Three tiers, all matched case-insensitively:
#
#   STRUCTURAL   things a real document never contains, anywhere       -> ERROR
#   LEADING      a refusal / error phrase in the first ~80 chars       -> ERROR
#                (a document that *starts* "I'm sorry, but I can't" is
#                 a refused call, not prose that mentions one)
#   the same refusal / error phrases anywhere else in the document     -> WARN
TEXT_ERROR_SIGNATURES_STRUCTURAL = (
    '"error":', "'error':", "<!doctype html", "<html", "<think>", "</think>",
    "<|channel|>", "<|start|>", "<|message|>", "<|end|>", "assistantfinal",
)
TEXT_ERROR_SIGNATURES_PHRASE = (
    "as an ai language model", "i cannot assist", "i can't help with that",
    "i'm sorry, but i can", "i am sorry, but i can", "i am unable to",
    "i apologize, but i cannot", "i'm not able to", "sorry, but i can",
    "internal server error", "bad gateway", "service unavailable",
    "gateway time-out", "gateway timeout", "upstream connect error",
    "too many requests", "rate limit exceeded", "rate_limit_exceeded",
    "insufficient_quota", "context_length_exceeded", "model_not_found",
    "an error occurred", "request timed out", "please try again later",
)
LEADING_WINDOW = 80

# Columns where a partial null is a real gap. Everywhere else a partial null is
# an expected consequence of the two classes carrying different metadata (only
# machine rows have a `prompt`, only some works have a `doi`), so it is a WARN.
REQUIRED_COLS = ("uid", "id", "text", "generated_by", "domain", "y", "group")

# Runaway-generation detector. Both corpus families ask the generator for text
# about as long as the human original (ghostbuster: round_to_50(human words);
# science: "approximately the same length"), so a machine row many times longer
# is the shape of a decoding loop -- a model re-writing its own conclusion until
# it hits the context window (seen on llama-3.3-70b via llama.cpp). Flagged only
# when BOTH gates trip: the ratio to the human partner (>= ERROR / >= WARN), and
# an absolute bar of max(LEN_ABS_MIN_WORDS, LEN_ABS_MULT x median human words) --
# a normal-length reply to a freakishly long human original is not a
# malfunction. Human partners under LEN_MIN_REF words don't anchor a ratio.
# Under-generation is left to the blank / <20-word / refusal scans.
LEN_OUTLIER_ERROR_RATIO = 4.0
LEN_OUTLIER_WARN_RATIO = 2.0
LEN_MIN_REF_WORDS = 20
LEN_ABS_MIN_WORDS = 900
LEN_ABS_MULT = 3.0


def _partial_null_is_error(name: str) -> bool:
    return name in REQUIRED_COLS or name.startswith(SCORE_PREFIXES) or _is_vector(name)


def _null_line(name: str, nulls: int, n: int) -> str:
    pct = 100 * nulls / n
    shown = f"{pct:.1f}%" if pct >= 0.1 else f"{pct:.2g}%"
    return f"{name}: {nulls}/{n} null ({shown}) — partial fill"


def _is_vector(name: str) -> bool:
    return (name.startswith(VECTOR_PREFIXES)
            or name.endswith(VECTOR_SUFFIXES)
            or any(s in name for s in VECTOR_CONTAINS))


class Report:
    def __init__(self, verbose: bool = False) -> None:
        self.errors: list[str] = []
        self.warns: list[str] = []
        self.verbose = verbose
        self.ids: dict[str, list] = {}  # identifier columns, for --verbose row refs
        self.gen: list | None = None    # generated_by per row, when the column exists
        self.domain: list | None = None  # domain per row, when the column exists
        self.rid: list | None = None    # id per row, when the column exists
        self.len_error_ratio: float = LEN_OUTLIER_ERROR_RATIO
        self.len_warn_ratio: float = LEN_OUTLIER_WARN_RATIO
        self.allow_incomplete_groups: bool = False
        self.allow_unstandardized: bool = False

    def rowref(self, i: int) -> str:
        """One offending row, tagged with whatever identifier columns exist."""
        if not self.ids:
            return f"row {i}"
        extra = "  ".join(f"{c}={self.ids[c][i]}" for c in self.ids)
        return f"row {i}  {extra}"

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        print(f"    {RED}ERROR{RESET} {msg}")

    def warn(self, msg: str) -> None:
        self.warns.append(msg)
        print(f"    {YELLOW}WARN {RESET} {msg}")

    def ok(self, msg: str) -> None:
        print(f"    {GREEN}ok   {RESET}{DIM} {msg}{RESET}")

    def detail(self, msg: str) -> None:
        print(f"    {DIM}       {msg}{RESET}")


# ---------------------------------------------------------------------------


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "NA"
    return f"{x:.4g}"


def _is_optional_null(name: str) -> bool:
    return any(name.startswith(p) for p in OPTIONAL_NULL_PREFIXES)


def check_scalar(name: str, arr: pa.ChunkedArray, n: int, rep: Report,
                  max_essay_human_blanks: int | None = None) -> dict:
    """Nulls, distinctness and — for numbers — range and non-finite values."""
    nulls = arr.null_count
    info: dict = {"nulls": nulls}
    is_score = name.startswith(SCORE_PREFIXES)

    if nulls == n:
        if _is_optional_null(name):
            rep.warn(f"{name}: entirely null ({n} rows) — expected for this column")
        elif is_score:
            rep.error(f"{name}: entirely null ({n} rows) — the scorer wrote nothing")
        else:
            rep.warn(f"{name}: entirely null ({n} rows) — unused column?")
        return info
    if nulls:
        (rep.error if _partial_null_is_error(name) else rep.warn)(_null_line(name, nulls, n))

    non_null = arr.drop_null()
    n_distinct = len(pa.compute.unique(non_null))
    info["distinct"] = n_distinct

    if pa.types.is_floating(arr.type) or pa.types.is_integer(arr.type):
        v = non_null.to_numpy(zero_copy_only=False).astype(float)
        n_nan = int(np.isnan(v).sum())
        n_inf = int(np.isinf(v).sum())
        finite = v[np.isfinite(v)]
        info.update(
            nan=n_nan, inf=n_inf,
            min=float(finite.min()) if finite.size else None,
            max=float(finite.max()) if finite.size else None,
            mean=float(finite.mean()) if finite.size else None,
            std=float(finite.std()) if finite.size else None,
            zeros=int((finite == 0).sum()),
        )
        if n_nan:
            rep.error(f"{name}: {n_nan} NaN value(s) in a non-null column")
        if n_inf:
            rep.error(f"{name}: {n_inf} infinite value(s)")

        bounded = name.startswith(BOUNDED_SCORE_PREFIXES)
        if bounded and finite.size and (finite.min() < 0 or finite.max() > 1):
            rep.error(
                f"{name}: values outside [0, 1] "
                f"(min {_fmt(info['min'])}, max {_fmt(info['max'])}) — "
                f"a P(...) column should not"
            )

        if is_score:
            top_frac = _top_value_fraction(non_null)
            info["top_value_frac"] = top_frac
            if n_distinct == 1:
                rep.error(f"{name}: CONSTANT — every row is {_fmt(finite[0])}. "
                          f"The scorer wrote one value into all {len(finite)} rows.")
            elif (info["std"] is not None and info["std"] < 1e-9) or top_frac >= CONSTANT_FRAC:
                rep.error(f"{name}: effectively constant — {top_frac:.0%} of rows "
                          f"share one value (std {_fmt(info['std'])})")
            elif top_frac >= SUSPICIOUS_FRAC:
                rep.warn(f"{name}: {top_frac:.0%} of rows share one value "
                         f"({n_distinct} distinct total) — check this scorer ran "
                         f"properly (or it is a very clumpy judge)")
        rep.detail(f"{name}: {n_distinct} distinct, "
                   f"[{_fmt(info['min'])}, {_fmt(info['max'])}] "
                   f"mean {_fmt(info['mean'])} std {_fmt(info['std'])}"
                   + (f", {info['nan']} NaN" if info["nan"] else ""))
    else:  # string / bool / etc.
        if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
            # Identity mapping (None) when nothing was dropped: non_null's
            # positions then line up 1:1 with the full column's row indices,
            # which is what _check_string needs to look up domain/generated_by
            # for a blank row without re-reading the column with nulls kept in.
            valid_mask = arr.is_valid().to_numpy(zero_copy_only=False) if nulls else None
            _check_string(name, non_null, n, n_distinct, rep, info,
                          max_essay_human_blanks, valid_mask)
        else:
            rep.ok(f"{name}: {n_distinct} distinct")

    return info


def _top_value_fraction(non_null: pa.ChunkedArray) -> float:
    vc = non_null.value_counts()
    counts = vc.field("counts").to_numpy()
    return float(counts.max() / counts.sum()) if counts.size else 0.0


def _looks_like_error(doc: str) -> bool:
    low = doc.lower()
    lead = low[:LEADING_WINDOW]
    return (
        any(s in low for s in TEXT_ERROR_SIGNATURES_STRUCTURAL)
        or any(s in lead for s in TEXT_ERROR_SIGNATURES_PHRASE)
        or doc.lstrip()[:1] in ("{", "<")
    )


def _scan_text_errors(name: str, docs: list[str], rep: Report, info: dict) -> None:
    """Flag documents that look like a failed API call rather than real text:
    an error / refusal signature, or one document repeated byte-for-byte across
    many rows (one error string written into every failed row is the tell)."""
    low = [d.lower() for d in docs]

    def _hits(test) -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        for i, d in enumerate(low):
            if (sig := test(d)) is not None:
                out.append((i, sig))
        return out

    def _first(sigs, hay):
        return next((s for s in sigs if s in hay), None)

    structural = _hits(lambda d: _first(TEXT_ERROR_SIGNATURES_STRUCTURAL, d))
    leading = _hits(lambda d: _first(TEXT_ERROR_SIGNATURES_PHRASE, d[:LEADING_WINDOW]))
    mid = _hits(lambda d: _first(TEXT_ERROR_SIGNATURES_PHRASE, d))
    mid = [h for h in mid if h[0] not in {i for i, _ in leading}]

    tiers = (
        ("structural", structural, rep.error),
        ("leading refusal / error", leading, rep.error),
        ("mid-document refusal / error phrase", mid, rep.warn),
    )
    for label, hits, emit in tiers:
        info.setdefault("error_signature_hits", {})[label] = len(hits)
        if not hits:
            continue
        by_sig = Counter(sig for _, sig in hits)
        emit(f"{name}: {len(hits)} document(s) — {label} — likely a failed call "
             f"saved as content: "
             + ", ".join(f"{s!r}x{c}" for s, c in by_sig.most_common(4)))
        for i, sig in (hits if rep.verbose else hits[:3]):
            rep.detail(f"[{sig!r}] {rep.rowref(i)}  {docs[i][:140]!r}")
    if not any(hits for _, hits, _ in tiers):
        rep.ok(f"{name}: {len(docs)} doc(s) scanned, no error / refusal signatures")

    counts = Counter(docs)
    dups = {d: c for d, c in counts.items() if c > 1}
    if not dups:
        return
    n_dup_rows = sum(dups.values())
    top_doc, top_c = max(dups.items(), key=lambda x: x[1])
    err_groups = {d: c for d, c in dups.items() if _looks_like_error(d)}
    info["duplicate_docs"] = {
        "groups": len(dups), "rows": n_dup_rows, "largest_group": top_c,
        "error_like_groups": len(err_groups),
    }
    threshold = max(5, round(0.005 * len(docs)))
    if err_groups:
        d, c = max(err_groups.items(), key=lambda x: x[1])
        rep.error(f"{name}: an error-like document is repeated across {c} row(s) "
                  f"({len(err_groups)} such group(s)) — a failed call written "
                  f"into every row it hit")
        rep.detail(f"{c}x: {d[:140]!r}")
    elif top_c >= threshold:
        rep.warn(f"{name}: one document appears in {top_c} rows "
                 f"(threshold {threshold}) — check it is real text")
        rep.detail(f"{top_c}x: {top_doc[:140]!r}")
    else:
        rep.detail(f"{name}: {n_dup_rows} row(s) in {len(dups)} duplicate group(s), "
                   f"largest {top_c} — expected in a paired / multi-generator corpus")

    if rep.verbose:
        pos: dict[str, list[int]] = {}
        for i, d in enumerate(docs):
            if d in dups:
                pos.setdefault(d, []).append(i)
        rep.detail(f"{name}: {len(dups)} duplicate group(s) —")
        for d, idxs in sorted(pos.items(), key=lambda kv: -len(kv[1])):
            rep.detail(f"  x{len(idxs)}: " + "  |  ".join(rep.rowref(i) for i in idxs))
            rep.detail(f"       {d[:200]!r}")


def _low_fw(doc: str) -> bool:
    return (_function_word_ratio is not None and len(doc.split()) >= 30
            and _function_word_ratio(doc) < NON_ENGLISH_FW_RATIO)


def _scan_language(name: str, docs: list[str], rep: Report, info: dict) -> None:
    """Documents that are not English prose — a wrong-language abstract, or (the
    common one here) a generator that answered in a non-English title's
    language, leaving the human row English and its machine row not. Detected by
    a very low English function-word ratio; see ``NON_ENGLISH_FW_RATIO``."""
    if _function_word_ratio is None:
        return
    hits = [i for i, d in enumerate(docs) if _low_fw(d)]
    info["non_english_docs"] = len(hits)
    if not hits:
        return
    by_kind = (Counter(rep.gen[i] for i in hits)
               if rep.gen is not None and len(rep.gen) == len(docs) else None)
    tail = f" — {dict(by_kind)}" if by_kind else ""
    rep.warn(f"{name}: {len(hits)} document(s) are not English prose{tail}")
    for i in (hits if rep.verbose else hits[:3]):
        rep.detail(f"{rep.rowref(i)}  {' '.join(docs[i].split())[:140]!r}")


def _scan_non_prose(name: str, docs: list[str], rep: Report, info: dict) -> None:
    """Flag documents that are not an abstract at all — a scrape of the
    publisher's page, a book's contents / front matter, a reference list. Same
    test the fetch/filter stages reject on (``rules.abstract_is_boilerplate``);
    non-English text (also caught by that test) is reported by _scan_language."""
    if abstract_is_boilerplate is None:
        return
    hits = [i for i, d in enumerate(docs)
            if abstract_is_boilerplate(d) and not _low_fw(d)]
    info["non_prose_docs"] = len(hits)
    if not hits:
        return
    rep.warn(f"{name}: {len(hits)} document(s) are a page scrape / contents "
             f"page / reference list, not prose")
    for i in (hits if rep.verbose else hits[:3]):
        rep.detail(f"{rep.rowref(i)}  {' '.join(docs[i].split())[:140]!r}")


def _scan_length_outliers(name: str, docs: list[str], orig_idx, rep: Report,
                          info: dict) -> None:
    """Machine documents that are a runaway generation — many times longer than
    both their human partner AND anything else in the corpus.

    Two gates, because either alone is noisy: the *ratio* to the human partner
    (a corpus with an atypically short human original would trip a pure ratio
    check), and an *absolute* bar tied to the corpus's own median human length
    (a model writing a normal-length essay in reply to a freakishly long human
    one is not malfunctioning). Only the long side — under-generation is already
    covered by the blank / <20-word / refusal scans above.

    Needs ``id`` and ``generated_by`` on the report and a ``human`` class.
    """
    if rep.rid is None or rep.gen is None:
        return
    rid, gen = rep.rid, rep.gen
    if not any(str(g) == "human" for g in gen):
        return

    doc_words = [len(d.split()) for d in docs]
    human_words: dict[str, int] = {}
    for i, oi in enumerate(orig_idx):
        oi = int(oi)
        if str(gen[oi]) == "human":
            human_words[str(rid[oi])] = doc_words[i]
    if not human_words:
        return

    median_human = float(np.median(list(human_words.values())))
    # "too long for this corpus, whatever the human partner was."
    abs_bar = max(LEN_ABS_MIN_WORDS, LEN_ABS_MULT * median_human)
    e, w = rep.len_error_ratio, rep.len_warn_ratio

    err: list[tuple] = []
    warn: list[tuple] = []
    ratios: list[float] = []
    for i, oi in enumerate(orig_idx):
        oi = int(oi)
        g = str(gen[oi])
        if g == "human":
            continue
        h = human_words.get(str(rid[oi]))
        if not h or h < LEN_MIN_REF_WORDS:
            continue
        gw = doc_words[i]
        r = gw / h
        ratios.append(r)
        if gw < abs_bar:
            continue
        hit = (oi, h, gw, r, g)
        if r >= e:
            err.append(hit)
        elif r >= w:
            warn.append(hit)

    if not ratios:
        return
    arr = np.array(ratios)
    info["length_vs_human"] = {
        "checked": len(ratios), "ratio_median": float(np.median(arr)),
        "ratio_p95": float(np.percentile(arr, 95)), "ratio_max": float(arr.max()),
        "abs_bar_words": abs_bar, "error_runaway": len(err), "warn_long": len(warn),
    }

    def _report(hits: list[tuple], verb: str, emit) -> None:
        by_gen = Counter(g for *_, g in hits)
        emit(f"{name}: {len(hits)} machine document(s) are a runaway generation "
             f"({verb}, and >= {abs_bar:.0f} words) ({dict(by_gen)}) — "
             f"regenerate those rows")
        for oi, h, gw, r, g in sorted(hits, key=lambda x: -x[2])[
                : None if rep.verbose else 5]:
            rep.detail(f"row {oi}  id={rid[oi]}  {g}  {gw}w = {r:.1f}x its {h}w human partner")

    if err:
        _report(err, f">= {e:g}x the human partner", rep.error)
    if warn:
        _report(warn, f">= {w:g}x the human partner", rep.warn)
    if not err and not warn:
        rep.ok(f"{name}: {len(ratios)} machine doc(s), none a runaway "
               f"(median {np.median(arr):.2f}x human, bar {abs_bar:.0f}w)")


def _scan_standardization(name: str, docs: list[str], orig_idx, rep: Report,
                          info: dict) -> None:
    """Is this ``text`` column standardised?

    ``standardize_text`` folds typographic characters to ASCII (curly quotes,
    em-dashes, ellipsis, non-breaking spaces, …) — a corpus that skipped that
    stage, or an arm appended without it, carries a feature that is about the
    text's *route* to the parquet, not its author, and every score / embedding
    on it describes text a standardised corpus no longer holds. Only the
    character fold is checked (whitespace mode is per-corpus); ``text_raw``
    legitimately keeps the raw form and is not checked.
    """
    try:
        from epai.utils.text_standardize import substitution_counts
    except Exception:
        return

    groups: Counter = Counter()
    hits: list[int] = []
    for i, d in enumerate(docs):
        c = substitution_counts(d)
        cr = d.count("\r")
        if c or cr:
            hits.append(int(orig_idx[i]))
            groups.update(c)
            if cr:
                groups["carriage_return"] += cr
    info["unstandardized"] = {"rows": len(hits), "groups": dict(groups)}
    if not hits:
        rep.ok(f"{name}: standardised — no foldable typographic characters")
        return

    by_gen = (Counter(str(rep.gen[oi]) for oi in hits)
              if rep.gen is not None else None)
    emit = rep.warn if rep.allow_unstandardized else rep.error
    emit(f"{name}: {len(hits)} row(s) still hold characters standardize_text "
         f"folds ({dict(groups.most_common())}"
         + (f"; {dict(by_gen)}" if by_gen else "")
         + ") — this column is not standardised. Re-run the standardize stage "
         f"(or standardize_text.py) before scoring / embedding.")
    for oi in (hits if rep.verbose else hits[:5]):
        rep.detail(rep.rowref(oi))


def _essay_human_blank_count(blank_idx: list[int], rep: Report) -> int:
    """How many of these row indices are a known essay/human seed blank.

    The upstream Ghostbuster essay release ships six blank human documents —
    seeded input, not model output, so nothing downstream can regenerate them.
    Needs both ``domain`` and ``generated_by`` on the report to tell them apart
    from a genuinely empty generated row.
    """
    if rep.domain is None or rep.gen is None:
        return 0
    return sum(
        1 for i in blank_idx
        if i < len(rep.domain) and i < len(rep.gen)
        and rep.domain[i] == "essay" and rep.gen[i] == "human"
    )


def _check_string(name: str, non_null, n: int, n_distinct: int,
                  rep: Report, info: dict,
                  max_essay_human_blanks: int | None = None,
                  valid_mask: np.ndarray | None = None) -> None:
    s = non_null.to_pylist()
    empty = sum(1 for x in s if x == "" or (isinstance(x, str) and not x.strip()))
    info.update(empty_str=empty, distinct=n_distinct)

    if name in TEXTISH:
        if empty:
            exempt = 0
            if max_essay_human_blanks:
                idx = np.arange(len(s)) if valid_mask is None else np.nonzero(valid_mask)[0]
                blank_orig = [
                    int(idx[i]) for i, x in enumerate(s)
                    if x == "" or (isinstance(x, str) and not x.strip())
                ]
                eligible = _essay_human_blank_count(blank_orig, rep)
                exempt = min(eligible, max_essay_human_blanks)
            remaining = empty - exempt
            if remaining:
                rep.error(f"{name}: {remaining} empty / whitespace-only document(s)"
                          + (f" (+{exempt} more exempted: known essay human seed "
                             f"blanks, cap {max_essay_human_blanks})" if exempt else ""))
            elif exempt:
                rep.warn(f"{name}: {exempt} empty / whitespace-only document(s) — "
                         f"known essay human seed blanks (exempted, cap "
                         f"{max_essay_human_blanks})")
        lengths = np.array([len(x) for x in s], dtype=float)
        words = np.array([len(x.split()) for x in s], dtype=float)
        short = int((words < 20).sum())
        truncated = sum(1 for x in s if x.rstrip().endswith(("…", "...")))
        info.update(len_min=int(lengths.min()), len_med=int(np.median(lengths)),
                    len_max=int(lengths.max()), words_med=int(np.median(words)))
        if short:
            rep.warn(f"{name}: {short} document(s) under 20 words")
        if truncated:
            rep.warn(f"{name}: {truncated} document(s) end in an ellipsis (truncated?)")
        _scan_text_errors(name, s, rep, info)
        _scan_non_prose(name, s, rep, info)
        _scan_language(name, s, rep, info)
        if name in ("text", "abstract"):
            orig_idx = (np.arange(len(s)) if valid_mask is None
                        else np.nonzero(valid_mask)[0])
            _scan_length_outliers(name, s, orig_idx, rep, info)
            _scan_standardization(name, s, orig_idx, rep, info)
        rep.detail(f"{name}: {info['words_med']} words median, "
                   f"chars [{info['len_min']}, {info['len_max']}], {n_distinct} distinct")
    elif name in TITLEISH:
        if empty:
            rep.warn(f"{name}: {empty} empty title(s)")
        if title_looks_english is not None:
            seen_t: set[str] = set()
            ne = []
            for i, x in enumerate(s):
                if x and x not in seen_t and not title_looks_english(x, _TITLE_CFG):
                    seen_t.add(x)
                    ne.append(i)
            if ne:
                rep.warn(f"{name}: {len(ne)} distinct title(s) are not English "
                         f"(a title-only prompt then generates non-English text)")
                for i in (ne if rep.verbose else ne[:3]):
                    rep.detail(f"{rep.rowref(i)}  {s[i][:110]!r}")
        rep.detail(f"{name}: {n_distinct} distinct")
    elif name == "uid":
        dupes = len(s) - n_distinct
        if dupes:
            rep.error(f"{name}: {dupes} duplicate value(s) — not unique per row")
        else:
            rep.ok(f"{name}: {n_distinct} unique")
    elif name == "generated_by":
        vc = non_null.value_counts()
        pairs = {vc.field("values")[i].as_py(): vc.field("counts")[i].as_py()
                 for i in range(len(vc))}
        rep.ok(f"{name}: " + ", ".join(f"{k}={v}" for k, v in pairs.items()))
        if len(pairs) < 2:
            rep.warn(f"{name}: single class — fine for an eval-only holdout, "
                     f"not for a training corpus")
    else:
        if empty and n_distinct == 1:
            rep.warn(f"{name}: every non-null value is empty")
        rep.detail(f"{name}: {n_distinct} distinct"
                   + (f", {empty} empty" if empty else ""))


def check_list(name: str, arr: pa.ChunkedArray, n: int, rep: Report) -> dict:
    """List columns. Fixed-width vectors (emb_*, *_cuevec_*) get the full
    embedding sanity check; variable-length lists just get null / empty."""
    nulls = arr.null_count
    info: dict = {"nulls": nulls}
    vector = _is_vector(name)
    if nulls == n:
        (rep.error if vector else rep.warn)(f"{name}: entirely null")
        return info
    if nulls:
        (rep.error if vector else rep.warn)(_null_line(name, nulls, n))

    flat = arr.drop_null().to_pylist()
    lengths = np.array([len(x) if x is not None else 0 for x in flat])
    empty = int((lengths == 0).sum())
    widths = sorted(set(lengths[lengths > 0].tolist()))
    info.update(empty=empty, widths=widths)

    if not vector:
        if empty:
            rep.detail(f"{name}: {empty} empty list(s) (variable-length column)")
        rep.detail(f"{name}: list<{arr.type.value_type}>, "
                   f"lengths {lengths.min()}..{lengths.max()}")
        return info

    if empty:
        rep.error(f"{name}: {empty} empty vector(s)")
    if len(widths) > 1:
        rep.error(f"{name}: ragged — widths {widths} (a vector column must be one width)")
        return info
    if not widths:
        return info

    dim = widths[0]
    M = np.vstack([np.asarray(x, dtype=float) for x in flat if x])
    n_nan = int(np.isnan(M).any(axis=1).sum())
    finite_rows = ~np.isnan(M).any(axis=1)
    n_zero = int((finite_rows & (np.abs(M).sum(axis=1) == 0)).sum())
    norms = np.linalg.norm(np.nan_to_num(M), axis=1)
    info.update(dim=dim, nan_vectors=n_nan, zero_vectors=n_zero,
                norm_min=float(norms.min()), norm_max=float(norms.max()))
    if n_nan:
        rep.error(f"{name}: {n_nan} vector(s) contain NaN")
    if n_zero:
        rep.error(f"{name}: {n_zero} all-zero vector(s) — the embedder produced "
                  f"nothing for those rows")
    # Identical high-dim vectors are a real red flag (two docs, one embedding);
    # for a low-dim / binary vector (the 11-dim cue vector) collisions are
    # expected and mean nothing.
    if dim > 32:
        n_identical = M.shape[0] - len({m.tobytes() for m in M})
        if n_identical:
            rep.warn(f"{name}: {n_identical} vector(s) byte-identical to another row")
            if rep.verbose:
                groups: dict[bytes, list[int]] = {}
                for j, m in enumerate(M):
                    groups.setdefault(m.tobytes(), []).append(j)
                ref = rep.rowref if (nulls == 0 and empty == 0) else (
                    lambda j: f"non-null #{j}")
                for idxs in sorted((v for v in groups.values() if len(v) > 1),
                                   key=lambda v: -len(v)):
                    rep.detail("  identical: " + "  |  ".join(ref(j) for j in idxs))
    rep.detail(f"{name}: dim {dim}, norm [{_fmt(info['norm_min'])}, "
               f"{_fmt(info['norm_max'])}]")
    return info


def check_class_balance(path: Path, schema, rep: Report) -> dict:
    """The domain × generated_by count matrix, printed, plus a balance check.

    A paired multi-generator corpus should carry the same row count in every
    (domain, generated_by) cell — one human abstract and one row per generator
    for each source paper — and a constant number of rows per source ``id``.
    An uneven column is a generator that dropped papers or a domain that came
    up short. Full pairing/balance analysis is ``audit_corpus.py``'s job; this
    is the quick read on the way past.
    """
    names = set(schema.names)
    if "generated_by" not in names or "domain" not in names:
        return {}

    key = next((c for c in ("id", "group") if c in names), None)
    cols = ["domain", "generated_by"] + ([key] if key else [])
    t = pq.read_table(path, columns=cols)
    dom = [str(x) for x in t.column("domain").to_pylist()]
    gen = [str(x) for x in t.column("generated_by").to_pylist()]

    domains = sorted(set(dom))
    gens = sorted(set(gen))
    cell = Counter(zip(dom, gen))
    w = max(8, max(len(g) for g in gens)) + 2
    lab = max(10, max(len(d) for d in domains)) + 2

    print(f"\n  {BOLD}class balance{RESET}{DIM}  domain × generated_by{RESET}")
    print("    " + " " * lab + "".join(f"{g:>{w}}" for g in gens) + f"{'total':>{w}}")
    for d in domains:
        row = [cell[(d, g)] for g in gens]
        print(f"    {d:<{lab}}" + "".join(f"{c:>{w}}" for c in row)
              + f"{sum(row):>{w}}")
    tot = [sum(cell[(d, g)] for d in domains) for g in gens]
    print(f"    {'total':<{lab}}" + "".join(f"{c:>{w}}" for c in tot)
          + f"{sum(tot):>{w}}")

    dom_tot = {d: sum(cell[(d, g)] for g in gens) for d in domains}
    info: dict = {"matrix": {d: {g: cell[(d, g)] for g in gens} for d in domains}}
    # Balance only means something for a training corpus (>1 generated_by class);
    # an eval-only holdout is one class and legitimately uneven across domains.
    if len(gens) > 1:
        if len(domains) > 1 and len(set(dom_tot.values())) > 1:
            rep.warn(f"domain totals are uneven: {dom_tot}")
        for d in domains:
            counts = {g: cell[(d, g)] for g in gens}
            if len(set(counts.values())) > 1:
                rep.warn(f"{d}: generated_by classes uneven — {counts}")

    if key:
        key_vals = [str(x) for x in t.column(key).to_pylist()]
        per = Counter(key_vals)
        sizes = Counter(per.values())
        info["groups"] = {"count": len(per), "rows_per_group": dict(sizes)}
        if len(sizes) > 1:
            top = sizes.most_common(1)[0][0]
            odd = sorted(g for g, c in per.items() if c != top)
            rep.warn(f"{key}: rows per group vary — {dict(sizes)}; "
                     f"{len(odd)} group(s) not {top}"
                     + (f" (e.g. {odd[:5]})" if rep.verbose else ""))
        else:
            rep.ok(f"{key}: {next(iter(sizes))} rows/group, {len(per)} groups")

        # Every group should carry the full generator set its domain has — a
        # missing label is an unpaired example (the human and machine answers to
        # one source prompt no longer both exist), which usually means those
        # rows need regenerating. Per domain, since domains may use different
        # generators.
        if len(gens) > 1:
            dom_labels: dict[str, set] = {}
            grp_labels: dict[tuple, set] = {}
            for d, g, k in zip(dom, gen, key_vals):
                dom_labels.setdefault(d, set()).add(g)
                grp_labels.setdefault((d, k), set()).add(g)
            missing_by_label: Counter = Counter()
            bad: list[tuple] = []
            for (d, k), present in grp_labels.items():
                gap = dom_labels[d] - present
                if gap:
                    bad.append((k, sorted(gap)))
                    missing_by_label.update(gap)
            info["incomplete_groups"] = {
                "count": len(bad), "by_missing_label": dict(missing_by_label)}
            if bad:
                emit = rep.warn if rep.allow_incomplete_groups else rep.error
                emit(f"{key}: {len(bad)} group(s) missing a generator their domain "
                     f"has — {dict(missing_by_label.most_common())}; regenerate "
                     f"those, or drop the group")
                for k, gap in (bad if rep.verbose else bad[:5]):
                    rep.detail(f"{key}={k}  missing {gap}")
            else:
                rep.ok(f"{key}: every group has its domain's full generator set")
    return info


def check_file(path: Path, reference_cols: set[str] | None,
               args: argparse.Namespace) -> Report:
    rep = Report(args.verbose)
    print(f"\n{BOLD}{'=' * 78}{RESET}\n{BOLD}{path}{RESET}\n{'=' * 78}")

    if not path.is_file():
        rep.error("file does not exist")
        return rep
    size_mb = path.stat().st_size / 1e6

    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    schema = pf.schema_arrow
    print(f"  {n} rows, {len(schema)} columns, {size_mb:.1f} MB")
    if n == 0:
        rep.error("file has 0 rows")
        return rep

    if reference_cols is not None:
        have = set(schema.names)
        missing = sorted(reference_cols - have)
        extra = sorted(have - reference_cols)
        if missing:
            rep.error(f"missing {len(missing)} column(s) the reference has: {missing}")
        if extra:
            rep.warn(f"has {len(extra)} column(s) the reference does not: {extra}")

    if args.verbose:
        idcols = [c for c in ("uid", "id", "generated_by") if c in schema.names]
        if idcols:
            t = pq.read_table(path, columns=idcols)
            rep.ids = {c: t.column(c).to_pylist() for c in idcols}
    if "generated_by" in schema.names:
        rep.gen = pq.read_table(path, columns=["generated_by"]).column(0).to_pylist()
    if "domain" in schema.names:
        rep.domain = pq.read_table(path, columns=["domain"]).column(0).to_pylist()
    if "id" in schema.names:
        rep.rid = pq.read_table(path, columns=["id"]).column(0).to_pylist()
    rep.len_error_ratio = args.len_outlier_error
    rep.len_warn_ratio = args.len_outlier_warn
    rep.allow_incomplete_groups = args.allow_incomplete_groups
    rep.allow_unstandardized = args.allow_unstandardized

    col_info: dict[str, dict] = {}
    print(f"\n  {BOLD}columns{RESET}")
    for field in schema:
        arr = pq.read_table(path, columns=[field.name]).column(0)
        if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
            col_info[field.name] = check_list(field.name, arr, n, rep)
        else:
            col_info[field.name] = check_scalar(
                field.name, arr, n, rep, args.max_essay_human_blanks
            )

    balance = check_class_balance(path, schema, rep)
    if balance:
        col_info["_class_balance"] = balance

    # verdict
    ne, nw = len(rep.errors), len(rep.warns)
    print(f"\n  {BOLD}verdict{RESET}: ", end="")
    if ne:
        print(f"{RED}{ne} error(s){RESET}, {nw} warning(s)")
    elif nw:
        print(f"{YELLOW}{nw} warning(s){RESET}, no errors")
    else:
        print(f"{GREEN}clean{RESET}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "path": str(path), "rows": n, "columns": len(schema),
            "errors": rep.errors, "warnings": rep.warns,
            "column_info": col_info,
        }, indent=2, default=str))
        print(f"  {DIM}json → {args.json}{RESET}")
    return rep


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("parquets", nargs="+", type=Path)
    ap.add_argument("--reference", type=Path, default=None,
                    help="another parquet whose column set every input must contain")
    ap.add_argument("--fail-on", choices=("error", "warn", "never"), default="error",
                    help="which findings set exit code 1 (default: error)")
    ap.add_argument("--strict", action="store_true", help="alias for --fail-on warn")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the offending rows — duplicate documents, "
                         "byte-identical vectors, error signatures — each tagged "
                         "with its uid / id / generated_by")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the machine-readable report here (last file wins)")
    ap.add_argument("--len-outlier-error", type=float, default=LEN_OUTLIER_ERROR_RATIO,
                    metavar="R",
                    help="a machine document >= R x its human partner's word "
                         "count AND absolutely long for the corpus is an ERROR "
                         f"(default {LEN_OUTLIER_ERROR_RATIO}) — the shape of a looped generation")
    ap.add_argument("--len-outlier-warn", type=float, default=LEN_OUTLIER_WARN_RATIO,
                    metavar="R",
                    help=f"the same, as a WARN (default {LEN_OUTLIER_WARN_RATIO})")
    ap.add_argument("--allow-incomplete-groups", action="store_true",
                    help="downgrade the 'group missing a generator' finding from "
                         "ERROR to WARN — for a deliberately unpaired corpus")
    ap.add_argument("--allow-unstandardized", action="store_true",
                    help="downgrade the 'text not standardised' finding from "
                         "ERROR to WARN — for a deliberately raw corpus "
                         "(e.g. ghostbuster_gpt35t)")
    ap.add_argument("--max-essay-human-blanks", type=int, default=None, nargs="?",
                    const=6,
                    help="exempt up to this many empty/whitespace documents from "
                         "the blank-text ERROR when they are domain=='essay', "
                         "generated_by=='human' rows — the upstream Ghostbuster "
                         "essay release ships six such seeds and nothing "
                         "downstream can regenerate them. Off by default (every "
                         "blank document is an ERROR); pass with no value for the "
                         "known cap of 6, or a number to override. Ghostbuster-"
                         "specific — leave unset for science or other corpora, "
                         "where a blank text row is a real problem.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.strict:
        args.fail_on = "warn"

    reference_cols = None
    if args.reference:
        reference_cols = set(pq.ParquetFile(args.reference).schema_arrow.names)
        print(f"{DIM}reference: {args.reference} ({len(reference_cols)} columns){RESET}")

    worst_error = worst_warn = False
    for path in args.parquets:
        rep = check_file(path, reference_cols, args)
        worst_error |= bool(rep.errors)
        worst_warn |= bool(rep.warns)

    if args.fail_on == "error" and worst_error:
        sys.exit(1)
    if args.fail_on == "warn" and (worst_error or worst_warn):
        sys.exit(1)


if __name__ == "__main__":
    main()

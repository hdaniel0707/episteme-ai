r"""Is this corpus fit to train on? One report, for ghostbuster and science alike.

``summary_parquet.py`` answers "what is in this file" column by column. This
script answers the question that comes after it — **what is wrong with the
corpus** — and does it in the corpus's own vocabulary: documents, generators,
domains and pairs, rather than columns and dtypes.

The two pipelines write the same core schema, which is what makes one script
enough::

    ghostbuster : uid  id  domain  text  generated_by  binoculars_score  emb_*
    science_v1  : uid  id  domain  text  generated_by  binoculars_score  emb_*
                  + doi title year venue fwci prompt ...  (carried, not checked)

Only ``id``, ``text`` and ``generated_by`` are required. ``uid`` and ``domain``
are added by ``combine_domains.py``, so a per-domain parquet from before that
stage is audited whole, under a single synthetic domain.

WHAT IT CHECKS, and why each one is here rather than left to the eye:

  identity     duplicate ``uid``, ids that lost their ``{id}:{generated_by}``
               spelling. A duplicated uid is the same document twice, which
               inflates a score without ever looking wrong in a row count.
  documents    blank text, near-blank text, exact duplicate texts. **This is
               the check the ghostbuster pipeline's collect stage does on the
               text tree, applied one stage later to the corpus** — an empty
               generation that got carried in with --allow-empty is a row whose
               label is a lie: it is filed as machine-written and holds nothing.
  scores       nulls in ``binoculars_score`` / ``llm_judge_score*``, split by
               whether the row HAS a text. A null beside a real text is a
               scoring gap to re-run; a null beside an empty text is the
               expected consequence of a blank document, not a scoring failure.
  embeddings   the same split for every ``emb_*`` column, plus width agreement,
               all-zero and NaN/Inf vectors. Delegated to
               ``summary_parquet.describe_embedding``, which already owns this.
  balance      rows per domain x generated_by, and the machine share per
               domain. A domain whose balance differs from the others' turns
               "which domain is this" into a partial answer to "is this
               machine", and every cross-domain number then has a shortcut
               available to it.
  pairing      ids missing a generator that the rest of their domain has.
               Dropping a broken machine row leaves its human partner behind,
               unmatched — the thing a row-wise drop does silently and a
               pair-wise one does not.
  verdict      how many rows and how many whole pairs survive a
               ``--drop-incomplete``-style filter, and an exit code, so this can
               gate a pipeline instead of being read.

Every finding is an ERROR (blocks training, or silently biases it) or a WARN
(expected, but worth knowing). ``--fail-on`` decides which of them sets the exit
code; nothing here modifies the file.

Usage:
    # the two corpora side by side
    uv run python epai/ai_detection/analyse/audit_corpus.py \
        data/parquet/ghostbuster_gpt56luna_binox0.parquet \
        data/parquet/science_v1_gpt56luna_0807A_binox0.parquet

    # name the offending rows, not just count them
    uv run python epai/ai_detection/analyse/audit_corpus.py \
        data/parquet/ghostbuster_gpt56luna_binox0.parquet --list 40

    # one domain, and a machine-readable copy for a pipeline step
    uv run python epai/ai_detection/analyse/audit_corpus.py \
        data/parquet/ghostbuster_gpt56luna_binox0.parquet \
        --domain essay --json results/audit_essay.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# The schema's names come from the script that writes it, so an audit cannot
# disagree with the file it is auditing about what a column is called.
from epai.ai_detection.prepare.combine_domains import (
    DOMAIN_COL,
    ID_COL,
    REQUIRED_COLS,
    SCORE_PREFIXES,
    TARGET_COL,
    TEXT_COL,
    UID_COL,
)
from epai.ai_detection.embed.summary_parquet import describe_embedding, embedding_columns
from epai.utils.parquet_utils import load_parquet

BASE_DIR = Path(__file__).parent.parent.parent.parent

# Stand-in domain for a per-domain parquet, which has no domain column. Bracketed
# so it cannot collide with a real domain name.
NO_DOMAIN = "(whole file)"

# A text this short is not a document. The number is not arbitrary: the
# ghostbuster generators size their request against the human partner's word
# count, so a very short human original asks for a very short machine one and
# the pair is uninformative either way.
#
# It used to be worse. The budget was round_to_100(human words), and round() is
# HALF TO EVEN -- so a human original of 50 words or fewer asked the model for a
# ZERO-word document and got an empty answer back, with no API error to explain
# it. generate.py now uses round_to_50() with a floor of 50, so new corpora
# cannot contain that failure; corpora built before it (ghostbuster_gpt35t,
# ghostbuster_gpt54mini) still can, which is why this check stays.
SHORT_WORDS = 25

# Runaway-generation detector. The generator is asked for text about as long as
# the human original, so a machine row many times longer is a decoding loop
# (seen on llama-3.3-70b via llama.cpp: it re-wrote its own conclusion until the
# context window). Flagged only when BOTH gates trip: the ratio to the human
# partner, and an absolute bar of max(LEN_ABS_MIN_WORDS, LEN_ABS_MULT x median
# human words) -- a normal-length reply to a freakishly long human original is
# not a malfunction. Under-generation is left to the blank / short-text checks.
LEN_ERROR_RATIO = 4.0
LEN_WARN_RATIO = 2.0
LEN_MIN_REF_WORDS = 20
LEN_ABS_MIN_WORDS = 900
LEN_ABS_MULT = 3.0

RULE = 92
MAX_LISTED = 8


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


class Report:
    """Collected findings, in the order they were raised.

    A section prints its own table as it goes; findings are what survives to the
    verdict at the end, so a long report still ends with a short answer.
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


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def is_blank(value) -> bool:
    """True for a cell that holds no text: None, NaN, or whitespace only."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):  # arrays are never a blank text
        pass
    return str(value).strip() == ""


def blank_mask(series: pd.Series) -> np.ndarray:
    return series.apply(is_blank).to_numpy()


def word_counts(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.split().str.len()


def score_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(SCORE_PREFIXES)]


def pct(n: int, total: int) -> str:
    return f"{n} ({100.0 * n / total:.2f}%)" if total else f"{n}"


def row_keys(df: pd.DataFrame, mask: np.ndarray, limit: int) -> list[str]:
    """Identifiers for the flagged rows — uid where there is one, else id.

    Positions would be shorter, but they change with every filter and cannot be
    grepped for in a text tree; a uid can.
    """
    key = UID_COL if UID_COL in df.columns else ID_COL
    values = df.loc[mask, key].astype(str).tolist()
    return values[:limit]


def listing(values: list[str], total: int, limit: int) -> str:
    if not values:
        return ""
    shown = ", ".join(values[:limit])
    more = total - min(len(values), limit)
    return f"{shown}{f', ... (+{more} more)' if more > 0 else ''}"


def fmt_size(path: Path) -> str:
    mib = path.stat().st_size / 1024**2
    return f"{mib:.1f} MiB" if mib < 1024 else f"{mib / 1024:.2f} GiB"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print()
    print("=" * RULE)
    print(title)
    print("=" * RULE)


def check_schema(path: Path, df: pd.DataFrame, rep: Report) -> None:
    emb_cols = embedding_columns(df)
    scores = score_columns(df)
    section(f"File  —  {path.name}")
    print(f"  path            : {path}")
    print(f"  size on disk    : {fmt_size(path)}")
    print(f"  rows            : {len(df)}")
    print(f"  columns         : {len(df.columns)}  "
          f"({len(emb_cols)} embedding, {len(scores)} score)")
    print(f"  score columns   : {', '.join(scores) or '- (none)'}")
    print(f"  emb columns     : {', '.join(emb_cols) or '- (none)'}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        # Fatal on its own: every section below reads these.
        raise SystemExit(
            f"{path.name}: not a corpus this script can audit — missing required "
            f"column(s) {missing}. Expected at least {list(REQUIRED_COLS)}."
        )

    for col, note in ((UID_COL, "row key"), (DOMAIN_COL, "domain")):
        if col not in df.columns:
            print(f"  NOTE  no {col!r} column ({note}) — this file predates "
                  f"combine_domains.py, or is one domain's own parquet.")

    if not emb_cols:
        rep.warn("schema", "no emb_* column in this file",
                 "Nothing here can tell a scored corpus from an embedded one; "
                 "run the embed stage before training on it.")
    if not scores:
        rep.warn("schema", "no binoculars/llm-judge score column in this file",
                 "Expected if the score stage has not run yet.")


def check_identity(df: pd.DataFrame, rep: Report, limit: int) -> None:
    section("Identity")
    n = len(df)

    for col in (ID_COL, UID_COL):
        if col not in df.columns:
            continue
        blanks = blank_mask(df[col])
        print(f"  {col + ' blank':<28}: {pct(int(blanks.sum()), n)}")
        if blanks.any():
            rep.error("identity", f"{int(blanks.sum())} row(s) with an empty {col}",
                      "A row with no key cannot be grouped, joined or split on.",
                      rows=row_keys(df, blanks, limit))

    if UID_COL in df.columns:
        dup = df[UID_COL].duplicated(keep=False).to_numpy()
        print(f"  {'duplicate uid':<28}: {pct(int(dup.sum()), n)}")
        if dup.any():
            examples = df.loc[dup, UID_COL].value_counts().head(limit)
            rep.error("identity",
                      f"{int(dup.sum())} row(s) share a uid with another row",
                      "uid is the unique row key. A repeat is the same document "
                      "twice: it inflates whichever fold it lands in, and "
                      "straddles train/test if the copies separate.",
                      rows=[f"{k} x{v}" for k, v in examples.items()])

        # combine_domains builds uid as f"{id}:{generated_by}". A row that does
        # not match was written by something else, and grouping on id no longer
        # implies anything about uid.
        expected = df[ID_COL].astype(str) + ":" + df[TARGET_COL].astype(str)
        odd = (df[UID_COL].astype(str) != expected).to_numpy()
        print(f"  {'uid != id:generated_by':<28}: {pct(int(odd.sum()), n)}")
        if odd.any():
            rep.warn("identity",
                     f"{int(odd.sum())} uid(s) are not '{{id}}:{{generated_by}}'",
                     "Harmless if something renamed the labels after combine; a "
                     "sign of a hand-edited file otherwise.",
                     rows=row_keys(df, odd, limit))

    # id is the PAIR key and is expected to repeat once per generator. What must
    # not repeat is (id, generated_by) — that is the same document twice under
    # two row keys, which the uid check above only catches if the uids match too.
    pair_dup = df.duplicated(subset=[ID_COL, TARGET_COL], keep=False).to_numpy()
    print(f"  {'duplicate (id, ' + TARGET_COL + ')':<28}: "
          f"{pct(int(pair_dup.sum()), n)}")
    if pair_dup.any():
        rep.error("identity",
                  f"{int(pair_dup.sum())} row(s) repeat an (id, {TARGET_COL}) pair",
                  "One source prompt has two answers from one generator. Keep "
                  "one: combine_domains.py --drop-duplicate-rows.",
                  rows=row_keys(df, pair_dup, limit))

    print(f"  {'distinct ids (pairs)':<28}: {df[ID_COL].nunique()}")
    print(f"  {'distinct ' + TARGET_COL:<28}: {df[TARGET_COL].nunique()} "
          f"({sorted(df[TARGET_COL].astype(str).unique())})")


def _check_length_vs_human(df: pd.DataFrame, words: pd.Series, blank: np.ndarray,
                           rep: Report, limit: int,
                           err_ratio: float, warn_ratio: float) -> None:
    """Machine texts that are a runaway generation — many times longer than both
    their human partner AND the corpus's own median human length.

    Two gates so neither noise source (an atypically short human original; a
    model writing a normal-length reply to a freakishly long one) trips it
    alone. Long side only; under-generation is the blank / short-text checks'
    job. See ``LEN_ERROR_RATIO``.
    """
    if "human" not in set(df[TARGET_COL]):
        return
    human = (df[TARGET_COL] == "human").to_numpy()
    hw = pd.Series(words.to_numpy()[human], index=df[ID_COL].to_numpy()[human])
    hw = hw[~hw.index.duplicated()]
    if hw.empty:
        return
    abs_bar = max(LEN_ABS_MIN_WORDS, LEN_ABS_MULT * float(hw.median()))

    machine = (~human) & (~blank)
    mrows = df.index[machine]
    ref = df.loc[mrows, ID_COL].map(hw)
    ok = ref.notna().to_numpy() & (ref.to_numpy() >= LEN_MIN_REF_WORDS)
    if not ok.any():
        return

    mrows = mrows[ok]
    mw = words.loc[mrows].to_numpy()
    ratio = mw / ref.to_numpy()[ok]
    gen = df.loc[mrows, TARGET_COL].to_numpy()
    print(f"  length vs human partner    : median {np.median(ratio):.2f}x, "
          f"p95 {np.percentile(ratio, 95):.2f}x, max {ratio.max():.2f}x "
          f"({len(ratio)} machine rows, runaway bar {abs_bar:.0f}w)")

    key = UID_COL if UID_COL in df.columns else ID_COL
    long_enough = mw >= abs_bar

    def _flag(sel: np.ndarray, verb: str, emit) -> None:
        if not sel.any():
            return
        by_gen = pd.Series(gen[sel]).value_counts().to_dict()
        keys = df.loc[mrows[sel][np.argsort(-mw[sel])], key].astype(str).tolist()
        emit("documents",
             f"{int(sel.sum())} machine text(s) are a runaway generation "
             f"({verb}, and >= {abs_bar:.0f} words) ({by_gen})",
             "The generator is asked for text about as long as the human "
             "original; a model many times over that, looping on its own "
             "conclusion, is the llama.cpp failure mode. Drop these rows and "
             "re-run generate for their ids.",
             rows=keys[:limit])

    _flag(long_enough & (ratio >= err_ratio), f">= {err_ratio:g}x", rep.error)
    _flag(long_enough & (ratio >= warn_ratio) & (ratio < err_ratio),
          f">= {warn_ratio:g}x", rep.warn)


def _check_standardization(df: pd.DataFrame, blank: np.ndarray, rep: Report,
                           limit: int, allow_unstandardized: bool) -> None:
    """Is TEXT_COL standardised? A row still carrying a character
    ``standardize_text`` folds (curly quote, em-dash, nbsp, CR) means the
    standardize stage was skipped or an arm was appended without it — the text a
    score/embedding no longer describes. Character fold only; whitespace mode is
    per-corpus. ERROR unless ``--allow-incomplete-groups``' sibling flag."""
    try:
        from epai.utils.text_standardize import substitution_counts
    except Exception:
        return
    groups: Counter = Counter()
    hit = np.zeros(len(df), dtype=bool)
    text = df[TEXT_COL].fillna("").astype(str)
    for i, d in enumerate(text):
        if blank[i]:
            continue
        c = substitution_counts(d)
        cr = d.count("\r")
        if c or cr:
            hit[i] = True
            groups.update(c)
            if cr:
                groups["carriage_return"] += cr
    if not hit.any():
        return
    by_gen = df.loc[hit, TARGET_COL].value_counts().to_dict()
    emit = rep.warn if allow_unstandardized else rep.error
    emit("documents",
         f"{int(hit.sum())} row(s) still hold characters standardize_text folds "
         f"({dict(groups.most_common())}; {by_gen})",
         "This column is not standardised — the standardize stage was skipped, "
         "or an arm was appended without it. Re-run standardize_text.py (or the "
         "pipeline's standardize stage) before scoring. --allow-unstandardized "
         "for a deliberately raw corpus.",
         rows=row_keys(df, hit, limit))


def check_documents(df: pd.DataFrame, rep: Report, limit: int,
                    short_words: int, len_error_ratio: float = LEN_ERROR_RATIO,
                    len_warn_ratio: float = LEN_WARN_RATIO,
                    allow_unstandardized: bool = False) -> None:
    """Blank, near-blank and duplicated texts, per domain x generator."""
    section("Documents")
    n = len(df)
    words = word_counts(df[TEXT_COL])
    blank = blank_mask(df[TEXT_COL])
    short = (~blank) & (words < short_words).to_numpy()

    table = pd.DataFrame({
        DOMAIN_COL: df[DOMAIN_COL],
        TARGET_COL: df[TARGET_COL],
        "blank": blank,
        "short": short,
        "words": words,
    })
    grouped = table.groupby([DOMAIN_COL, TARGET_COL], dropna=False)
    stats = grouped.agg(rows=("words", "size"), blank=("blank", "sum"),
                        short=("short", "sum"), mean_words=("words", "mean"),
                        min_words=("words", "min"), max_words=("words", "max"))

    width = max([len(f"{d}/{t}") for d, t in stats.index] + [20])
    print(f"  {'domain/' + TARGET_COL:<{width}}{'rows':>8}{'blank':>8}"
          f"{'<' + str(short_words) + 'w':>8}{'mean w':>9}{'min':>7}{'max':>8}")
    print("  " + "-" * (width + 48))
    for (domain, target), row in stats.iterrows():
        flag = "  <-- blank" if row["blank"] else ""
        print(f"  {f'{domain}/{target}':<{width}}{int(row['rows']):>8}"
              f"{int(row['blank']):>8}{int(row['short']):>8}"
              f"{row['mean_words']:>9.0f}{int(row['min_words']):>7}"
              f"{int(row['max_words']):>8}{flag}")

    if blank.any():
        per_group = stats["blank"][stats["blank"] > 0]
        where = ", ".join(f"{d}/{t} ({int(v)})" for (d, t), v in per_group.items())
        rep.error("documents", f"{int(blank.sum())} row(s) hold no text  [{where}]",
                  "A blank row is labelled as if it were a document. Its score "
                  "and embeddings are Null, so it trains on nothing and skews "
                  "the class it is counted in. Drop it (combine's "
                  "--drop-incomplete emb) or refill it upstream.\n"
                  "        Blank MACHINE rows whose human partner is also blank "
                  "are the known ghostbuster seeds — the generator was asked for "
                  "a zero-word document. Those cannot be refilled; drop the pair.",
                  rows=row_keys(df, blank, limit))

    if short.any():
        rep.warn("documents",
                 f"{int(short.sum())} row(s) have a text under {short_words} words",
                 "Near-blank rather than blank, so nothing downstream drops "
                 "them. In the ghostbuster corpora these are the rows whose "
                 "human original was short enough to round the generator's word "
                 "budget down to zero.",
                 rows=row_keys(df, short, limit))

    _check_length_vs_human(df, words, blank, rep, limit,
                           len_error_ratio, len_warn_ratio)
    _check_standardization(df, blank, rep, limit, allow_unstandardized)

    # Exact repeats across DIFFERENT ids: one refusal string or one boilerplate
    # answer reused for many prompts. Within a pair it would be caught above.
    filled = df.loc[~blank]
    dup_text = filled.duplicated(subset=[TEXT_COL], keep=False)
    if dup_text.any():
        top = filled.loc[dup_text, TEXT_COL].value_counts().head(3)
        print(f"\n  duplicate texts             : "
              f"{pct(int(dup_text.sum()), n)} of the non-blank rows")
        for text, count in top.items():
            print(f"    x{count:<5} {str(text)[:70]!r}")
        rep.warn("documents",
                 f"{int(dup_text.sum())} non-blank row(s) share their text with "
                 f"another row",
                 "Often one generator returning the same refusal or template for "
                 "many prompts. Those rows teach a detector that one string is "
                 "machine-written and nothing else.")


def check_balance(df: pd.DataFrame, rep: Report) -> None:
    """Rows per domain x generator, and the machine share per domain."""
    section("Class balance")
    table = (df.groupby([DOMAIN_COL, TARGET_COL], dropna=False).size()
             .unstack(fill_value=0).sort_index())
    labels = list(table.columns)
    width = max([len(str(d)) for d in table.index] + [10])

    head = f"  {'domain':<{width}}" + "".join(f"{str(l)[:14]:>15}" for l in labels)
    print(head + f"{'total':>10}{'machine %':>11}")
    print("  " + "-" * (len(head) + 19))

    shares = {}
    for domain, row in table.iterrows():
        total = int(row.sum())
        machine = total - int(row.get("human", 0))
        shares[domain] = 100.0 * machine / max(total, 1)
        print(f"  {str(domain):<{width}}"
              + "".join(f"{int(row[l]):>15}" for l in labels)
              + f"{total:>10}{shares[domain]:>10.1f}%")

    if "human" not in table.columns:
        rep.warn("balance", "no 'human' rows in this corpus",
                 "Every detector here is trained on one class unless the human "
                 "side lives in another file.")
        return

    spread = max(shares.values()) - min(shares.values())
    if len(shares) > 1 and spread > 1.0:
        worst = ", ".join(f"{d} {s:.1f}%" for d, s in
                          sorted(shares.items(), key=lambda kv: -kv[1]))
        rep.warn("balance",
                 f"machine share differs by {spread:.1f} points across domains "
                 f"[{worst}]",
                 "A cross-domain result can then be read off the domain alone. "
                 "Usually the trace of rows dropped from one domain only — see "
                 "the pairing section.")


def check_scores(df: pd.DataFrame, rep: Report, limit: int) -> None:
    """Score nulls, split by whether the row has a text to score."""
    cols = score_columns(df)
    if not cols:
        return
    section("Scores")
    has_text = ~blank_mask(df[TEXT_COL])

    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce")
        null = values.isna().to_numpy()
        nonfinite = (~null) & ~np.isfinite(values.to_numpy(dtype=float))
        gap = null & has_text
        expected = null & ~has_text

        print(f"\n  {col}")
        print(f"    null                      : {pct(int(null.sum()), len(df))}")
        print(f"      with a text (real gap)  : {int(gap.sum())}")
        print(f"      text is blank (expected): {int(expected.sum())}")
        print(f"    non-finite (inf)          : {int(nonfinite.sum())}")

        if (~null).any():
            stats = (df.assign(_v=values).groupby(DOMAIN_COL, dropna=False)["_v"]
                     .agg(["mean", "std", "min", "max"]))
            for domain, row in stats.iterrows():
                print(f"    {str(domain):<12} mean {row['mean']:>8.3f}   "
                      f"sd {row['std']:>7.3f}   "
                      f"[{row['min']:.3f}, {row['max']:.3f}]")

        if gap.any():
            rep.error("scores", f"{int(gap.sum())} row(s) have a text but no {col}",
                      "The scorer skipped or failed on them. Re-run the score "
                      "stage; these rows are dropped by any model that uses the "
                      "score as a feature.",
                      rows=row_keys(df, gap, limit))
        if expected.any():
            rep.warn("scores",
                     f"{int(expected.sum())} row(s) have no {col} because their "
                     f"text is blank",
                     "Not a scoring failure — see the documents section, which "
                     "is where these rows have to be dealt with.")
        if nonfinite.any():
            rep.error("scores", f"{int(nonfinite.sum())} non-finite {col} value(s)",
                      "An inf survives isna() and reaches the model.",
                      rows=row_keys(df, nonfinite, limit))


def check_embeddings(df: pd.DataFrame, rep: Report, limit: int) -> dict[str, dict]:
    """Per emb_* column, delegating the vector arithmetic to summary_parquet."""
    cols = embedding_columns(df)
    if not cols:
        return {}
    section("Embeddings")
    has_text = ~blank_mask(df[TEXT_COL])
    stats = {c: describe_embedding(df[c], has_text) for c in cols}

    width = max([len(c) for c in cols] + [18])
    print(f"  {'column':<{width}}{'dim':>10}{'filled':>9}{'gap':>7}{'blank':>8}"
          f"{'zero':>7}{'NaN/Inf':>9}  note")
    print("  " + "-" * (width + 52))
    for col, st in stats.items():
        dims = st["dims"]
        dim = str(dims[0]) if len(dims) == 1 else ("MIXED " + "/".join(map(str, dims)))
        n_gap = int(st["missing_with_text"].sum())
        n_blank = int(st["missing_no_text"].sum())
        notes = []
        if len(dims) > 1:
            notes.append("inconsistent width")
        if st["n_filled"] == 0:
            notes.append("never computed")
        print(f"  {col:<{width}}{dim:>10}{st['n_filled']:>9}{n_gap:>7}"
              f"{n_blank:>8}{st['n_zero']:>7}{st['n_nonfinite']:>9}  "
              f"{', '.join(notes)}")

        if n_gap:
            rep.error("embeddings", f"{n_gap} row(s) have a text but no {col}",
                      "Re-run the embed stage for this column, or drop the rows.",
                      rows=row_keys(df, st["missing_with_text"], limit))
        if len(dims) > 1:
            rep.error("embeddings", f"{col} holds vectors of {dims} widths",
                      "Two models, or a partly re-run embed stage, in one "
                      "column. Nothing can featurize it as it stands.")
        if st["n_nonfinite"]:
            rep.error("embeddings", f"{st['n_nonfinite']} {col} vector(s) hold "
                      f"NaN/Inf", "They propagate through any distance or scaler.")
        if st["n_zero"]:
            rep.warn("embeddings", f"{st['n_zero']} {col} vector(s) are all zero",
                     "Present but carrying no signal, and a cosine against them "
                     "is undefined. Usually an empty string that reached the "
                     "model anyway.")
    print("  gap = missing beside a real text;  blank = missing beside an empty "
          "text (expected).")
    return stats


def check_pairing(df: pd.DataFrame, rep: Report, limit: int,
                  allow_incomplete_groups: bool = False) -> None:
    """ids missing a generator that the rest of their domain has.

    Per domain, not corpus-wide: domains legitimately use different generators,
    and comparing every id against the union would report the whole corpus as
    broken the moment one domain gained a second generator.

    An orphan is an ERROR by default -- the human and machine answers to one
    source prompt no longer both exist, so those rows want regenerating (or the
    whole group dropped). ``--allow-incomplete-groups`` keeps it a WARN for a
    corpus that is unpaired on purpose.
    """
    section("Pairing  (id = pair key, uid = row key)")
    total_orphans = 0
    for domain, part in df.groupby(DOMAIN_COL, dropna=False):
        labels = sorted(part[TARGET_COL].astype(str).unique())
        # The label set of each id, built once: the orphan report below asks
        # "which generator is this id missing" for every orphan, and a lookup
        # per id would rescan the domain each time.
        present = part.groupby(ID_COL)[TARGET_COL].agg(lambda s: set(s.astype(str)))
        per_id = present.map(len)
        sizes = Counter(per_id.values.tolist())
        complete = sizes.get(len(labels), 0)
        orphans = per_id[per_id < len(labels)]
        total_orphans += len(orphans)

        print(f"\n  {domain}")
        print(f"    generators              : {len(labels)}  ({labels})")
        print(f"    distinct ids            : {len(per_id)}")
        print(f"    complete ids            : {complete}")
        for size in sorted(sizes):
            if size != len(labels):
                print(f"    ids with only {size} label(s) : {sizes[size]}")
        if len(orphans):
            missing_side = Counter()
            for id_ in orphans.index:
                for label in labels:
                    if label not in present.loc[id_]:
                        missing_side[label] += 1
            for label, count in missing_side.most_common():
                print(f"    missing {label!r:<16}: {count} id(s)")
            print(f"    e.g. {listing([str(i) for i in orphans.index], len(orphans), MAX_LISTED)}")

    if total_orphans:
        emit = rep.warn if allow_incomplete_groups else rep.error
        emit("pairing", f"{total_orphans} id(s) are missing a generator their "
             f"domain otherwise has",
             "Each is an unmatched example: the human and machine answers to "
             "one source prompt no longer both exist. Regenerate the missing "
             "rows, or drop by id to keep the pairs whole "
             "(--allow-incomplete-groups to accept it).")
    print("\n  Split on 'id', not 'uid' — the two halves of a pair share a source "
          "prompt,\n  and straddling a fold leaks it across train/test.")


def check_usable(df: pd.DataFrame, emb_stats: dict[str, dict], rep: Report) -> dict:
    """What is left after dropping broken rows, per row and per whole pair.

    The row count is what ``--drop-incomplete`` gives; the pair count is what a
    grouped split actually gets to use, and the two differ by exactly the
    orphans the row-wise drop created.
    """
    section("Usable rows")
    n = len(df)
    broken = blank_mask(df[TEXT_COL])
    reasons = Counter()
    reasons["blank text"] = int(broken.sum())

    for col in score_columns(df):
        null = pd.to_numeric(df[col], errors="coerce").isna().to_numpy()
        reasons[f"no {col}"] = int((null & ~broken).sum())
        broken = broken | null
    for col, st in emb_stats.items():
        miss = st["missing"]
        reasons[f"no {col}"] = int((miss & ~broken).sum())
        broken = broken | miss

    keep = ~broken
    kept_ids = set(df.loc[keep, ID_COL])
    broken_ids = set(df.loc[broken, ID_COL])
    whole_pairs = kept_ids - broken_ids

    print(f"  rows in file              : {n}")
    for reason, count in reasons.items():
        if count:
            print(f"    - {reason:<22}: {count}")
    print(f"  rows with everything      : {pct(int(keep.sum()), n)}")
    print(f"  ids fully intact          : {len(whole_pairs)} of "
          f"{df[ID_COL].nunique()}")
    print(f"  ids touched by a drop     : {len(broken_ids)}")
    print("\n  'rows with everything' is what --drop-incomplete keeps; 'ids fully "
          "intact'\n  is what survives if the drop is applied by pair instead.")

    if broken.any():
        frac = broken.sum() / max(n, 1)
        level = rep.error if frac > 0.05 else rep.warn
        level("usable", f"{pct(int(broken.sum()), n)} of rows are incomplete",
              "Over 5% usually means a column was never computed for part of "
              "the corpus, not that a few documents failed.")
    return {
        "rows": n,
        "rows_complete": int(keep.sum()),
        "ids_total": int(df[ID_COL].nunique()),
        "ids_intact": len(whole_pairs),
        "reasons": dict(reasons),
    }


def print_verdict(path: Path, rep: Report, limit: int) -> None:
    section(f"Verdict  —  {path.name}")
    n_err, n_warn = rep.counts()
    if not rep.findings:
        print("  OK — nothing to report. Every row has a text, a score, an "
              "embedding and a partner.")
        return

    for finding in rep.findings:
        print(f"\n  {finding['level']}  [{finding['section']}] {finding['message']}")
        if finding["hint"]:
            print(f"        {finding['hint']}")
        if finding["rows"]:
            print(f"        rows: "
                  f"{listing(finding['rows'], len(finding['rows']), limit)}")
    print()
    print("-" * RULE)
    print(f"  {n_err} error(s), {n_warn} warning(s).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def audit(path: Path, args: argparse.Namespace) -> tuple[Report, dict]:
    df = load_parquet(path)

    if DOMAIN_COL not in df.columns:
        df = df.assign(**{DOMAIN_COL: NO_DOMAIN})
    df[DOMAIN_COL] = df[DOMAIN_COL].fillna(NO_DOMAIN).astype(str)
    df[TARGET_COL] = df[TARGET_COL].astype(str)

    if args.domain:
        available = sorted(df[DOMAIN_COL].unique())
        unknown = [d for d in args.domain if d not in available]
        if unknown:
            raise SystemExit(f"{path.name}: no such domain {unknown} — this file "
                             f"has {available}.")
        df = df[df[DOMAIN_COL].isin(args.domain)].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"{path.name}: no rows left to audit.")

    rep = Report()
    check_schema(path, df, rep)
    check_identity(df, rep, args.list)
    check_documents(df, rep, args.list, args.short_words,
                    args.len_outlier_error, args.len_outlier_warn,
                    args.allow_unstandardized)
    check_balance(df, rep)
    check_scores(df, rep, args.list)
    emb_stats = check_embeddings(df, rep, args.list)
    check_pairing(df, rep, args.list, args.allow_incomplete_groups)
    usable = check_usable(df, emb_stats, rep)
    print_verdict(path, rep, args.list)

    return rep, {"file": str(path), "usable": usable, "findings": rep.findings}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquets", nargs="+", type=Path,
                    help="Corpus parquet(s) to audit — ghostbuster, science, or "
                         "a single domain's file.")
    ap.add_argument("--domain", nargs="*", default=[], metavar="NAME",
                    help="Audit only these domains (default: all of them).")
    ap.add_argument("--list", type=int, default=MAX_LISTED, metavar="N",
                    help=f"Row keys to name per finding (default: {MAX_LISTED}).")
    ap.add_argument("--short-words", type=int, default=SHORT_WORDS, metavar="N",
                    help=f"Flag non-blank texts under N words (default: "
                         f"{SHORT_WORDS}).")
    ap.add_argument("--len-outlier-error", type=float, default=LEN_ERROR_RATIO,
                    metavar="R",
                    help=f"A machine text >= R x (or <= 1/R of) its human "
                         f"partner's word count is an ERROR (default {LEN_ERROR_RATIO}) "
                         f"— a looped or mostly-failed generation.")
    ap.add_argument("--len-outlier-warn", type=float, default=LEN_WARN_RATIO,
                    metavar="R",
                    help=f"The same, on the long side only, as a WARN (default "
                         f"{LEN_WARN_RATIO}).")
    ap.add_argument("--allow-incomplete-groups", action="store_true",
                    help="Downgrade 'id missing a generator' from ERROR to WARN "
                         "— for a corpus that is unpaired on purpose.")
    ap.add_argument("--allow-unstandardized", action="store_true",
                    help="Downgrade 'text not standardised' from ERROR to WARN "
                         "— for a deliberately raw corpus (ghostbuster_gpt35t).")
    ap.add_argument("--json", type=Path, default=None, metavar="PATH",
                    help="Also write the findings and the usable-row counts "
                         "there, for a pipeline step to read.")
    ap.add_argument("--fail-on", choices=("error", "warn", "never"),
                    default="error",
                    help="Exit non-zero on findings at this level or above "
                         "(default: error).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print(f"audit_corpus.py  —  {datetime.now():%Y-%m-%d %H:%M:%S}")

    results, worst_error, worst_warn = [], 0, 0
    for path in args.parquets:
        if not path.is_file():
            raise SystemExit(f"Not found: {path}")
        rep, result = audit(path, args)
        results.append(result)
        n_err, n_warn = rep.counts()
        worst_error += n_err
        worst_warn += n_warn

    if len(args.parquets) > 1:
        section("All files")
        for result in results:
            usable = result["usable"]
            print(f"  {Path(result['file']).name:<48}"
                  f"{usable['rows_complete']}/{usable['rows']} rows usable, "
                  f"{usable['ids_intact']}/{usable['ids_total']} ids intact")
        print(f"\n  {worst_error} error(s), {worst_warn} warning(s) in total.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nWrote {args.json}")

    if args.fail_on == "error" and worst_error:
        raise SystemExit(1)
    if args.fail_on == "warn" and (worst_error or worst_warn):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

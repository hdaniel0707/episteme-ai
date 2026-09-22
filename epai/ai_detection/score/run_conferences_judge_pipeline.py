r"""Judge pipeline for the conferences corpus: gpt56luna, prompt v3, ord2 + fusion.

Runs the paper's exact gpt56luna / v3 judge pipeline against
``data/parquet/conferences.parquet`` (real human abstracts scraped from
OpenAlex -- see ``build_conferences_parquet.py`` -- with no generated
counterpart), restricted to ``--conference`` if given. Five stages, appended
as new columns and never overwritten unless ``--recompute``, so re-running
after an interruption, or against a different ``--conference`` selection,
only fills in what is missing:

  0. standardize    -> ``abst_standardized``                    (typography/whitespace, see below)
  1. normal judge   -> ``llm_judge_score_gpt56luna_v3``        (parsed P(human))
  2. raw logprobs   -> ``llm_judge_logprobs_gpt56luna_v3_words`` (top-20 answer window)
  3. ord2           -> ``logjudge_ord2_gpt56luna_v3``          (closed-form P(human) from #2)
  4. fusionjudge    -> ``fusionjudge_score``                    (mean of #1 and #3)

Model and prompt are fixed (gpt56luna, v3) -- not flags -- since the point is
to match the main experiments exactly, not to offer a choice.

STANDARDIZATION, SAME AS EVERY OTHER CORPUS
--------------------------------------------
Stage 0 reproduces the SAME two-step pipeline the science corpora's raw
OpenAlex abstracts go through before anything judges them (see
``epai/science_data/rules.py``'s module docstring, "Cleaning happens before
judging", and ``epai/science_data/run_full_pipeline.py:step_standardize``) --
not just the character-folding step alone, which on its own would leave this
corpus dirtier than the ones it is meant to match:

  1. ``epai.science_data.rules.clean_abstract`` -- unescapes HTML entities,
     drops HTML tags, cuts publisher boilerplate (``(c) 2020 ... Published by
     Elsevier ...``), strips a leading "Abstract:" heading, collapses
     whitespace. This is NOT a no-op here: 1.3% of this corpus's abstracts
     carry an HTML entity and 3.6% carry an HTML tag (checked directly against
     ``data/conferences/oalex_abstracts_2020-2025.csv``), unlike the science
     pipeline's own raw abstracts, which arrive already run through this exact
     function at fetch time -- so there this step is upstream of the corpus
     file entirely, and ``standardize_text`` (below) is the only part that
     still runs as its own stage. ``strip_labels`` is left at its default
     (``False``), same as every call site of this function.
  2. ``epai.utils.text_standardize.standardize_text`` -- the exact function
     every other pipeline's ``standardize_text.py`` stage uses -- with
     ``whitespace="flat"``: the same mode ``run_full_pipeline.py`` passes for
     the science corpora, because an abstract is one paragraph already (see
     ``WHITESPACE_MODES["flat"]``'s own note that this is a backstop there,
     not a change -- for THIS corpus it can still do real work, since nothing
     upstream of it already collapsed the whitespace the way
     ``rules.clean_abstract``'s own step 5 does for the science pipeline).

The result of both steps goes into a NEW column, ``abst_standardized``, rather
than overwriting ``abstract`` in place -- ``abstract`` is this corpus's
original scraped text, matching the role ``abstract_raw`` plays for the
science corpora ("The publisher's own text survives beside it in
``abstract_raw``, so nothing here is unrecoverable" -- same rules.py
docstring), so there is no need to shuffle it sideways here. Every downstream
stage (judge, logprob capture) reads ``abst_standardized`` and never
``abstract`` directly. Word-count-preserved is checked the same way
``standardize_text.py`` checks it -- over the ALREADY-cleaned text, since
``clean_abstract`` legitimately removes boilerplate words by design and is not
held to that invariant, exactly as ``rules.py`` documents it -- and the run is
refused (nothing written) if ``standardize_text``'s own step changes the count.

STANDALONE BY DESIGN
---------------------
``capture_judge_logprobs.py`` / ``score_logjudge_ord2.py`` /
``score_fusionjudge_gpt56luna_v3.py`` are hardcoded to a fixed registry of
corpus "stems" (``data/parquet/<stem>.parquet``, ``uid`` primary key, a
literal ``text`` column) that ``conferences.parquet`` does not belong to and
does not carry (its key is ``id``, its column is ``abstract``). Rather than
register a new stem there, stages 2-4 below are reimplemented directly
against ``conferences.parquet``'s own schema -- importing the exact building
blocks those scripts already share, so the prompt, the ord2 formula and the
fusion mean are the same code, not a re-derivation of it:

    add_llm_judge_scores.{MODELS, MODE_PARSED, USER_TEMPLATE,
        LOGPROB_ANSWER_BLOCK, evidence_head, ApiJudge, judge_row,
        RETRIES, BACKOFF_BASE, SAVE_EVERY}
    score_logjudge_ord2.score_ord2
    score_fusionjudge_gpt56luna_v3.mean_combine
    science_data.rules.clean_abstract
    text_standardize.{standardize_text, count_words}

Stage 1 needs no reimplementation at all -- ``ApiJudge`` is the exact class
``add_llm_judge_scores.py``'s own ``main()`` drives for an API judge; it is
just called in-process here (rather than by subprocess) so all 5 stages can
share one row filter and one dataframe.

The judge sees ONLY the standardized abstract, never the title -- prompt v3 is
explicit that it is shown "no title, no author, no source, no metadata, no
context".

BILLED. Stages 1 and 2 each call the OpenAI API (gpt-5.6-luna) once per row.
Stage 0 is local and free. Run under tmux/screen/nohup for anything beyond a
smoke test -- an SSH drop kills a plain foreground process along with it:

    tmux new -s conferences-judge

    # dry run first: prints the plan, calls nothing
    .venv/bin/python epai/ai_detection/score/run_conferences_judge_pipeline.py \
        --conference ICLR --dry-run

    # smoke test
    .venv/bin/python epai/ai_detection/score/run_conferences_judge_pipeline.py \
        --conference ICLR --limit 20 --yes

    # the real run
    .venv/bin/python epai/ai_detection/score/run_conferences_judge_pipeline.py \
        --conference ICLR --yes

    # every conference in the file
    .venv/bin/python epai/ai_detection/score/run_conferences_judge_pipeline.py --yes
"""

from __future__ import annotations

import argparse
import math
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from epai.ai_detection.score.add_llm_judge_scores import (
    BACKOFF_BASE,
    LOGPROB_ANSWER_BLOCK,
    MODE_PARSED,
    MODELS,
    RETRIES,
    SAVE_EVERY,
    USER_TEMPLATE,
    ApiJudge,
    evidence_head,
    judge_row,
)
from epai.ai_detection.score.score_fusionjudge_gpt56luna_v3 import mean_combine
from epai.ai_detection.score.score_logjudge_ord2 import score_ord2
from epai.science_data.rules import clean_abstract
from epai.utils.text_standardize import count_words, standardize_text

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PARQUET = BASE_DIR / "data" / "parquet" / "conferences.parquet"

MODEL_KEY = "gpt56luna"
PROMPT_ID = "v3"
ABSTRACT_COL = "abstract"  # raw text, exactly as scraped -- never sent to a judge
TEXT_COL = "abst_standardized"  # what every stage below actually reads
WHITESPACE_MODE = "flat"  # same convention run_full_pipeline.py uses for science abstracts
TOP_N = 20  # answer-position window width, same as capture_judge_logprobs.py

SCORE_COL = f"llm_judge_score_{MODEL_KEY}_{PROMPT_ID}"
LP_COL = f"llm_judge_logprobs_{MODEL_KEY}_{PROMPT_ID}_words"
ORD2_COL = f"logjudge_ord2_{MODEL_KEY}_{PROMPT_ID}"
FUSION_COL = "fusionjudge_score"

GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def banner(title: str) -> None:
    print(f"\n{BOLD}{'-' * 74}{RESET}\n{BOLD}  {title}{RESET}\n{BOLD}{'-' * 74}{RESET}")


def atomic_save(df: pd.DataFrame, path: Path) -> None:
    """Same atomic temp-file + rename pattern used across this repo's other
    parquet-writing scripts -- data/ is gitignored, so there is no git history
    to recover a partial write from."""
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET, help="input parquet (default: %(default)s)")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output path (default: overwrite --parquet in place)")
    ap.add_argument("--conference", nargs="+", default=None, metavar="CONFERENCE_ID",
                     help="restrict to these conference_id values (default: every conference in the file)")
    ap.add_argument("--limit", type=int, default=None,
                     help="score only the first N eligible rows this run (default: every eligible row)")
    ap.add_argument("--workers", type=int, default=4, help="rows in flight at once (default: %(default)s)")
    ap.add_argument("--retries", type=int, default=RETRIES, help="retries per row (default: %(default)s)")
    ap.add_argument("--save-every", type=int, default=SAVE_EVERY,
                     help="checkpoint the parquet every N scored rows (default: %(default)s)")
    ap.add_argument("--recompute", action="store_true",
                     help="re-score/re-derive rows that already have a value, instead of skipping them")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop; no API calls, nothing written")
    ap.add_argument("--yes", action="store_true", help="skip the billed-call confirmation prompt")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Stage 0 -- clean (science_data.rules.clean_abstract) + standardize
# (text_standardize.standardize_text), both unmodified
# ---------------------------------------------------------------------------


def stage_standardize(df: pd.DataFrame, positions: list[int], args) -> None:
    banner(f"1/5  standardize  ->  {TEXT_COL}")
    if not positions:
        print("  nothing to do")
        return
    if args.dry_run:
        print(f"  {len(positions)} row(s) would be standardized (--dry-run)")
        return
    if TEXT_COL not in df.columns:
        df[TEXT_COL] = None

    raw = df.loc[positions, ABSTRACT_COL].astype(str)
    fix_counts: Counter = Counter()
    cleaned_values = []
    for t in raw:
        c, fixes = clean_abstract(t)
        cleaned_values.append(c)
        fix_counts.update(fixes)
    cleaned = pd.Series(cleaned_values, index=raw.index)

    standardized = cleaned.map(lambda t: standardize_text(t, whitespace=WHITESPACE_MODE))
    words_before = int(cleaned.map(count_words).sum())
    words_after = int(standardized.map(count_words).sum())
    if words_before != words_after:
        raise SystemExit(
            f"standardize_text changed the word count ({words_before} -> {words_after}) "
            "on the already-cleaned conferences abstracts -- refusing to write; "
            "this should never happen."
        )
    changed = int((cleaned != standardized).sum())
    n_empty = int((standardized.str.strip() == "").sum())

    df.loc[positions, TEXT_COL] = standardized
    atomic_save(df, args.parquet_out)
    print(f"  {len(positions)} row(s) standardized (whitespace={WHITESPACE_MODE!r})")
    if fix_counts:
        top = "  ".join(f"{k}={v}" for k, v in fix_counts.most_common())
        print(f"    clean_abstract fixes  {top}")
    print(f"    {changed} row(s) had a further ASCII/whitespace substitution  |  "
          f"words preserved ({words_before})")
    if n_empty:
        print(f"{YELLOW}    {n_empty} row(s) are EMPTY after cleaning -- likely all "
              f"publisher boilerplate; they will still be sent to the judge as an "
              f"empty document unless excluded by hand{RESET}")


# ---------------------------------------------------------------------------
# Stage 1 -- normal judge (add_llm_judge_scores.ApiJudge, unmodified)
# ---------------------------------------------------------------------------


def stage_judge(df: pd.DataFrame, positions: list[int], args) -> None:
    banner(f"2/5  normal judge  ->  {SCORE_COL}")
    if not positions:
        print("  nothing to do")
        return
    if args.dry_run:
        print(f"  {len(positions)} row(s) would be judged (--dry-run)")
        return

    judge = ApiJudge(MODELS[MODEL_KEY], prompt_id=PROMPT_ID, mode=MODE_PARSED)
    texts = df[TEXT_COL].astype(str)
    values = df[SCORE_COL].tolist() if SCORE_COL in df.columns else [None] * len(df)
    fails: list[tuple[int, str]] = []
    lock = threading.Lock()
    done = {"n": 0}

    def _save() -> None:
        df[SCORE_COL] = values
        atomic_save(df, args.parquet_out)

    def _one(pos: int):
        r = judge_row(judge, pos, texts.iloc[pos], args.retries)
        return pos, r.score, r.error

    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    try:
        futs = [pool.submit(_one, p) for p in positions]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="judge", unit="row"):
            pos, score, err = fut.result()
            values[pos] = score
            if err is not None:
                fails.append((pos, err))
            with lock:
                done["n"] += 1
                if done["n"] % args.save_every == 0:
                    _save()
    except KeyboardInterrupt:
        print("\n  interrupted -- writing what was scored")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    _save()

    if fails:
        for msg, c in Counter(m for _, m in fails).most_common(5):
            print(f"    {c:>5}x FAIL  {msg[:100]}")
        print(f"    ({len(fails)} row(s) left Null -- re-run to retry)")
    print(f"  {_distribution(values)}")


# ---------------------------------------------------------------------------
# Stage 2 -- raw answer-token logprob window (gpt56luna, OpenAI Responses API)
# ---------------------------------------------------------------------------


def _window_openai(client, model_id: str, system: str, text: str):
    resp = client.responses.create(
        model=model_id, instructions=system, input=USER_TEMPLATE.format(text=text),
        include=["message.output_text.logprobs"], top_logprobs=TOP_N,
        reasoning={"effort": "none"}, temperature=0, max_output_tokens=16,
    )
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for part in getattr(item, "content", []) or []:
                if getattr(part, "type", None) == "output_text":
                    lp = getattr(part, "logprobs", None)
                    if not lp:
                        raise ValueError("Responses API returned no logprobs")
                    return [{"token": a.token, "logprob": float(a.logprob)} for a in (lp[0].top_logprobs or [])]
    raise ValueError("no output_text part in the Responses object")


def _with_retries(fn, retries: int):
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries:
                time.sleep(BACKOFF_BASE**attempt * (1.0 + random.random()))
    assert last is not None
    raise last


def stage_logprobs(df: pd.DataFrame, positions: list[int], args) -> None:
    banner(f"3/5  raw logprobs  ->  {LP_COL}")
    if not positions:
        print("  nothing to do")
        return
    if args.dry_run:
        print(f"  {len(positions)} row(s) would be captured (--dry-run)")
        return

    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set -- see .env")
    client = OpenAI()
    model_id = MODELS[MODEL_KEY].model
    system = evidence_head(PROMPT_ID) + "\n\n" + LOGPROB_ANSWER_BLOCK
    texts = df[TEXT_COL].astype(str)
    values = df[LP_COL].tolist() if LP_COL in df.columns else [None] * len(df)
    fails: list[tuple[int, str]] = []
    lock = threading.Lock()
    done = {"n": 0}

    def _save() -> None:
        df[LP_COL] = pd.Series(values, index=df.index, dtype=object)
        atomic_save(df, args.parquet_out)

    def _one(pos: int):
        try:
            w = _with_retries(lambda: _window_openai(client, model_id, system, texts.iloc[pos]), args.retries)
            return pos, w, None
        except Exception as exc:  # noqa: BLE001
            return pos, None, f"{type(exc).__name__}: {exc}"

    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    try:
        futs = [pool.submit(_one, p) for p in positions]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="logprobs", unit="row"):
            pos, w, err = fut.result()
            values[pos] = w
            if err is not None:
                fails.append((pos, err))
            with lock:
                done["n"] += 1
                if done["n"] % args.save_every == 0:
                    _save()
    except KeyboardInterrupt:
        print("\n  interrupted -- writing what was captured")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    _save()

    if fails:
        for msg, c in Counter(m for _, m in fails).most_common(5):
            print(f"    {c:>5}x FAIL  {msg[:100]}")
        print(f"    ({len(fails)} row(s) left empty -- re-run to retry)")
    got = [w for w in (values[p] for p in positions) if w]
    if got:
        argmax = Counter(_norm_token(w[0]["token"]) for w in got)
        top = "  ".join(f"{k or '?'}={v}" for k, v in argmax.most_common(5))
        print(f"    captured {len(got)}/{len(positions)}  |  argmax: {top}")


def _norm_token(token: str | None) -> str:
    return "".join(ch for ch in (token or "") if ch.isalpha()).lower()


# ---------------------------------------------------------------------------
# Stage 3 -- ord2 (score_logjudge_ord2.score_ord2, unmodified)
# ---------------------------------------------------------------------------


def stage_ord2(df: pd.DataFrame, positions: list[int], args) -> None:
    banner(f"4/5  ord2  ->  {ORD2_COL}")
    if not positions:
        print("  nothing to do")
        return
    if args.dry_run:
        print(f"  {len(positions)} row(s) would be derived (--dry-run)")
        return
    if ORD2_COL not in df.columns:
        df[ORD2_COL] = np.nan
    # score_ord2 does `if not window`, which raises on a multi-element numpy
    # array (ambiguous truth value) -- a list<struct> column comes back from
    # pd.read_parquet as a numpy array per row, not a plain list, so it has to
    # be converted before it reaches score_ord2 unmodified.
    scores = [score_ord2(list(w) if w is not None else None) for w in (df.at[p, LP_COL] for p in positions)]
    for p, s in zip(positions, scores):
        df.at[p, ORD2_COL] = s if s is not None else np.nan
    atomic_save(df, args.parquet_out)
    print(f"  {_distribution(scores)}")


# ---------------------------------------------------------------------------
# Stage 4 -- fusionjudge (score_fusionjudge_gpt56luna_v3.mean_combine, unmodified)
# ---------------------------------------------------------------------------


def stage_fusion(df: pd.DataFrame, positions: list[int], args) -> None:
    banner(f"5/5  fusionjudge  ->  {FUSION_COL}")
    if not positions:
        print("  nothing to do")
        return
    if args.dry_run:
        print(f"  {len(positions)} row(s) would be fused (--dry-run)")
        return
    if FUSION_COL not in df.columns:
        df[FUSION_COL] = np.nan
    members = np.vstack([
        df.loc[positions, SCORE_COL].to_numpy(dtype=float),
        df.loc[positions, ORD2_COL].to_numpy(dtype=float),
    ])
    fused = mean_combine(members)
    df.loc[positions, FUSION_COL] = fused
    atomic_save(df, args.parquet_out)
    print(f"  {_distribution(list(fused))}")


def _distribution(scores: list) -> str:
    vals = np.array([s for s in scores if s is not None and not (isinstance(s, float) and math.isnan(s))], dtype=float)
    if not len(vals):
        return "all Null"
    return (f"n={len(vals)}  range [{vals.min():.3f}, {vals.max():.3f}]  "
            f"mean {vals.mean():.3f}  median {np.median(vals):.3f}")


# ---------------------------------------------------------------------------


def _has_value(v) -> bool:
    """True for a real (non-null, non-empty) value in either a scalar float
    column or a list<struct> column -- an empty captured window counts as NOT
    having a value, same as ``capture_judge_logprobs.py``'s ``_needs_scoring``."""
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    try:
        return len(v) > 0
    except TypeError:
        return True  # a real scalar (e.g. a judge score), not list-like


def eligible_positions(df: pd.DataFrame, col: str | None, args) -> list[int]:
    """Row positions selected by --conference, restricted to a non-empty
    abstract, minus rows already holding a value in ``col`` (unless
    --recompute or col is None), capped by --limit."""
    mask = np.ones(len(df), dtype=bool)
    if args.conference:
        mask &= df["conference_id"].isin(args.conference).to_numpy()
    # Always against the RAW abstract, not TEXT_COL -- this must give the same
    # answer whether or not stage 0 has populated TEXT_COL for a row yet.
    # notna() first: astype(str) turns a null abstract into the literal string
    # "nan"/"None", which would otherwise look non-empty.
    abstract = df[ABSTRACT_COL]
    mask &= (abstract.notna() & (abstract.astype(str).str.strip() != "")).to_numpy()
    if col is not None and col in df.columns and not args.recompute:
        has_value = df[col].apply(_has_value).to_numpy()
        mask &= ~has_value
    positions = list(np.flatnonzero(mask))
    if args.limit is not None:
        positions = positions[: args.limit]
    return positions


def main() -> None:
    args = parse_args()
    args.parquet_out = args.out or args.parquet

    from epai.utils.parquet_utils import load_parquet

    df = load_parquet(args.parquet)

    if args.conference:
        requested = {c.upper() for c in args.conference}
        available = set(df["conference_id"].unique())
        unknown = requested - available
        if unknown:
            raise SystemExit(f"Unknown conference_id(s): {sorted(unknown)}\nAvailable: {sorted(available)}")
        args.conference = sorted(requested)

    tag = "+".join(args.conference) if args.conference else "all"
    print(f"{BOLD}conferences judge pipeline{RESET}  model={MODEL_KEY}  prompt={PROMPT_ID}  "
          f"conference={tag}  file={args.parquet}")

    std_pos = eligible_positions(df, TEXT_COL, args)
    judge_pos = eligible_positions(df, SCORE_COL, args)
    lp_pos = eligible_positions(df, LP_COL, args)
    print(f"  stage 0 (standardize)  {len(std_pos)} row(s) to derive")
    print(f"  stage 1 (judge)        {len(judge_pos)} row(s) to score")
    print(f"  stage 2 (logprobs)     {len(lp_pos)} row(s) to capture")

    if not args.dry_run and not args.yes and (judge_pos or lp_pos):
        print(f"\n{YELLOW}NOTE  billed API calls -- one short completion per row, "
              f"across 2 stages.{RESET}")
        if input("Proceed? type 'yes': ").strip().lower() != "yes":
            raise SystemExit("aborted -- no calls made")

    # Must run before stage 1/2: they read TEXT_COL, which this stage fills.
    stage_standardize(df, std_pos, args)
    stage_judge(df, judge_pos, args)
    stage_logprobs(df, lp_pos, args)

    if args.dry_run:
        print(f"\n{YELLOW}--dry-run: stages 4 and 5 depend on stage 1/2 output, so their "
              f"row counts are not previewed here.{RESET}")
        return

    # Stages 4 and 5 are cheap and deterministic: re-derive them over the same
    # --conference selection whenever there is a captured window / a judge
    # score to derive from, so a second run over a different --conference
    # selection still fills in its share without needing --recompute.
    ord2_pos = []
    if LP_COL in df.columns:
        ord2_pos = [p for p in eligible_positions(df, ORD2_COL, args) if _has_value(df.at[p, LP_COL])]
    stage_ord2(df, ord2_pos, args)

    fusion_pos = eligible_positions(df, FUSION_COL, args)
    fusion_pos = [
        p for p in fusion_pos
        if SCORE_COL in df.columns and ORD2_COL in df.columns
        and pd.notna(df.at[p, SCORE_COL]) and pd.notna(df.at[p, ORD2_COL])
    ]
    stage_fusion(df, fusion_pos, args)

    print(f"\n{GREEN}Done.{RESET} {args.parquet_out}")


if __name__ == "__main__":
    main()

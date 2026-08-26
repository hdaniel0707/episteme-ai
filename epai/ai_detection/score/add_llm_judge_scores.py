"""LLM-as-a-judge baseline: ask an LLM whether each text is human-written.

Reads a Parquet, walks its rows, and for each one sends *only* the ``text``
column to a judge model that returns P(human-written) in [0, 1]. The score
lands in a new ``llm_judge_score_{model}_{prompt}`` column; nothing else in the
file is touched, so this sits next to ``binoculars_score`` as a second
zero-training baseline and can be compared with it on the same threshold
sweeps.

**1.0 means human, 0.0 means machine.** That is the same direction as
``binoculars_score`` (low => machine), so both baselines read the same way and a
threshold rule is ``score <= thr => machine`` for either.

The judge sees the document and nothing else -- no label, no ``generated_by``,
no domain, no neighbouring rows -- which is the point: the score has to be
earned from the text alone, exactly like the Binoculars score it is compared
against. It is a probability rather than a verdict, so ROC-AUC is available and
the operating point stays tunable after the fact (a hard verdict is just
``score <= 0.5``).

MODEL AND PROMPT BOTH NAME THE COLUMN
-------------------------------------
A judge score only means something within one (model, prompt) pair: change
either and the numbers are no longer on the same scale, but a column called
``llm_judge_score`` would happily hold both and nothing downstream could tell
them apart. So both go in the name -- ``--model gpt56luna --prompt v2`` writes
``llm_judge_score_gpt56luna_v2`` -- and re-running with a different model or
prompt adds a column instead of overwriting one. That is what makes a prompt
A/B (below) safe to run against the real corpus.

``--model`` picks the judge from ``MODELS`` (default ``gpt56luna``); the alias,
not the raw API id, names the column, because ``gpt-5.6-luna`` would not survive
as an identifier and the alias is the name the rest of the pipeline already uses
for that model. ``--prompt`` picks the system prompt from ``PROMPTS`` (default
``v2``); see the note above ``PROMPTS`` for what changed between the versions
and why.

WHICH ROWS ARE JUDGED
---------------------
Every row with a non-empty text, by default. ``--generated-by`` narrows that to
named generators, which in these corpora are ``human`` and a
``<model_slug>:<prompt_id>`` label such as ``gpt56luna:0701A``. The judge never
sees the column; it only decides which rows are worth paying for.

Rows whose text is empty (None/NaN/blank) are never sent to a model and keep a
Null score. Rows that already hold a score are skipped: the run asks whether to
re-judge them and defaults to no (``--recompute`` answers yes up front,
``--yes`` keeps the default without asking), so an interrupted run resumes where
it stopped instead of paying for the whole file twice. The Parquet is written
every ``--save-every`` scored rows for the same reason.

An API error on one row costs that row (Null score, counted in the summary) and
nothing else; the run keeps going. Each call is retried ``--retries`` times with
exponential backoff first, so a transient rate limit is not a lost row.

Run -- the full corpus set:
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt35t.parquet
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt35ts.parquet
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt54mini.parquet
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt56luna.parquet
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/science_v1_gpt56luna_0807A.parquet
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/science_v3_gpt56luna_0811A.parquet
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/x_science_v1_gpt56luna_0811A.parquet

Unlike run_binox_pairs.py these are API calls, not GPU work, so several corpora
can run at once -- but never two processes on the SAME file: adding a column is
a read-modify-write of the whole Parquet and the second writer would silently
drop the first one's column.

IS THE PROMPT ANY GOOD? -- A/B IT BEFORE THE FULL RUN
-----------------------------------------------------
``--limit N`` judges N eligible rows drawn round-robin across the generators, so
a corpus stored human-block-then-machine-block still yields both classes and the
summary's mean-per-group / accuracy / ROC-AUC lines are meaningful. Two cheap
runs on the same file put the prompts side by side in two columns:

    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt56luna.parquet --prompt v1 --limit 200 --yes
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt56luna.parquet --prompt v2 --limit 200 --yes

Compare the two ROC-AUCs, and the per-generator means: a prompt that separates
the classes well but puts BOTH means on one side of 0.5 is mis-calibrated rather
than blind, and is fixed by moving the threshold, not the prompt.

    # smoke test into a scratch copy, leaving the corpus untouched
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt56luna.parquet --limit 20 -o /tmp/judge_smoke.parquet
"""

from __future__ import annotations

import argparse
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

from epai.utils.parquet_utils import load_parquet, save_parquet

load_dotenv()

# Judge models, keyed by the short alias used everywhere else in the pipeline.
# The alias also names the output column, so it has to stay identifier-safe --
# hence `gpt56luna` rather than the API id `gpt-5.6-luna`, matching the slug the
# generation side already writes into filenames and `generated_by` labels.
MODELS: dict[str, dict[str, str]] = {
    "gpt56luna": {"provider": "openai", "model": "gpt-5.6-luna"},
    "gpt54mini": {"provider": "openai", "model": "gpt-5.4-mini"},
    "haiku45": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
}

DEFAULT_MODEL = "gpt56luna"
DEFAULT_TEXT_COL = "text"
COL_PREFIX = "llm_judge_score_"

# Which rows are judged. Default: every row with a non-empty text, human and
# machine alike -- the machine rows are labelled `<model_slug>:<prompt_id>`
# (`gpt56luna:0701A`), so there is no fixed list of generator names that would
# hold across corpora, and a stale one silently judges nothing. `--generated-by`
# names them explicitly when only part of a file is wanted. The judge never sees
# the column; it only decides which rows are worth paying for.
GENERATED_BY_COL = "generated_by"
DEFAULT_GENERATED_BY: list[str] | None = None
ALL_GENERATORS = "all"

# Environment variable each provider needs, checked up front so a long run does
# not die on the first call.
_REQUIRED_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

# Rows scored between two writes of the Parquet.
SAVE_EVERY = 200

# Retries per row before it is given up on, and the base of the exponential
# backoff between them (seconds, jittered).
RETRIES = 3
BACKOFF_BASE = 2.0

# Anthropic wants an explicit cap; the answer is a single number, so this is
# only there to bound a pathological response.
MAX_TOKENS = 256

# System prompts, keyed by version. The version names the output column together
# with the model alias, so both are kept here rather than edited in place: a
# score is only comparable to another score from the same (model, prompt) pair,
# and editing a prompt under a fixed name is exactly how two incomparable sets of
# numbers end up in one column.
#
# v1 -> v2, and why:
#
#   * v1 told the judge that typos, ragged formatting and inconsistent
#     punctuation indicate a human, and that their absence indicates a machine --
#     then, three lines later, to ignore formatting artefacts. On this corpus the
#     first instruction is not just contradictory but wrong: `prepare/` normalises
#     and standardises every document before it is stored, so surface noise has
#     been rewritten out of the human rows and the machine rows alike. A judge
#     following v1 reads "no typos" off a normalised human essay and marks it
#     machine. v2 states the pipeline explicitly and rules the whole surface layer
#     out of evidence.
#
#   * v1's human cues -- digressions, personal voice, lived detail -- describe an
#     essay or a forum post. Half these corpora are wire reports and scientific
#     abstracts, where a human author has none of those by convention, so the cues
#     drag exactly the human rows of those domains toward `machine`. v2 makes the
#     comparison genre-relative: unusual *for this kind of document*, not unusual
#     in general.
#
#   * v1 asked for a probability but nothing discouraged round numbers, and judges
#     pile answers onto 0.2/0.5/0.8. Ties are free AUC lost -- the metric only
#     reads the ranking. v2 asks for two decimals and says why.
PROMPTS: dict[str, str] = {}

PROMPTS["v1"] = """You are a forensic linguist specialising in the detection of machine-generated text.

You will be shown one document and nothing else: no title, no author, no source, no metadata, no context. Judge the text on its own evidence.

Your task is to estimate the probability that the document was written by a human rather than by a large language model.

Evidence that tends to indicate human authorship:
- Uneven pacing, digressions, sentences that change direction mid-way.
- Concrete, checkable specifics; idiom, slang, humour, irony, strong or unbalanced opinion.
- Genuine errors: typos, grammatical slips, inconsistent tense, ragged formatting.
- Domain habits and shorthand a general-purpose model would not reach for.
- Personal voice or lived detail that is not reconstructable from general knowledge.

Evidence that tends to indicate machine authorship:
- Even, predictable rhythm; uniform sentence and paragraph lengths.
- Fluent but generic vocabulary; hedged, balanced, non-committal framing.
- Tidy scaffolding: explicit enumeration, symmetrical structure, a summarising closing sentence that adds nothing new.
- Broad, safe claims in place of specific detail; no verifiable particulars, names, numbers, or first-hand observation.
- Absence of typos, false starts, inconsistent punctuation, or idiosyncratic formatting.

Calibration rules:
- Return a probability, not a verdict. 1.0 means certainly human, 0.0 means certainly machine, 0.5 means the text carries no usable evidence either way.
- Use the whole range, and reserve values below 0.1 or above 0.9 for documents where the evidence is genuinely strong.
- Do not assume any particular mix of human and machine documents; judge each one on its own.
- Polished, competent writing is not by itself evidence of machine authorship, and a short or dull document is not either.
- Formatting artefacts (truncation, stray whitespace, encoding damage) come from the corpus, not from the author. Ignore them.

Reply with the probability only."""

PROMPTS["v2"] = """You are a forensic linguist estimating whether a document was written by a human or generated by a large language model.

You will be shown one document and nothing else: no title, no author, no source, no metadata, no context. Judge it on its own evidence.

JUDGE RELATIVE TO THE GENRE
The document may be a student essay, a news report, a piece of short fiction, a scientific abstract, or something else entirely. Work out what kind of document it is trying to be, then ask whether it reads like a competent human working in that genre or like a model imitating one. A wire report carries no personal voice and an abstract carries no digressions; in those genres that is the convention, not evidence of a machine. The question is always "unusual for this kind of document?", never "unusual in general?".

EVIDENCE THAT TENDS TO INDICATE HUMAN AUTHORSHIP
- Specifics a model could not reconstruct from general knowledge: particular names, numbers, dates, places, results, quoted words -- and load-bearing rather than decorative, so that removing one would damage the text.
- Commitment: a definite claim, an unbalanced opinion, an argument that goes somewhere instead of hedging back to neutral.
- Asymmetry: uneven emphasis, a paragraph that runs long because the writer cared about it, a point raised and dropped, a sentence that changes direction mid-way.
- Idiom, slang, humour, irony, in-group shorthand; the habits of a field used the way a practitioner uses them rather than the way they would be explained.

EVIDENCE THAT TENDS TO INDICATE MACHINE AUTHORSHIP
- Fluent but generic vocabulary; an abstraction where a specific would have been natural.
- Even, predictable rhythm: uniform sentence and paragraph lengths, repetitive clause structure.
- Tidy scaffolding: symmetrical structure, explicit enumeration, a closing sentence that restates what came before and adds nothing.
- Hedged, balanced framing that declines to commit: every side acknowledged, none pressed.
- Coverage in place of depth: each obvious aspect of the topic given a proportionate share, nothing pursued further than the rest.

EVIDENCE THAT IS NOT AVAILABLE HERE -- DO NOT USE IT
These documents were normalised and truncated by a preprocessing pipeline before you saw them. Therefore:
- Punctuation and typography (quotation marks, dashes, apostrophes, ellipses, spacing, capitalisation) were rewritten by that pipeline. They tell you about the pipeline, not about the author.
- The absence of typos and misspellings is NOT evidence of machine authorship here, and their presence is at most weak evidence of human authorship.
- A document may stop mid-sentence, or lack an ending, because it was cut to a length budget. Never read truncation as a choice the author made.
- Any other formatting damage -- stray whitespace, broken encoding, lost markup, a stranded fragment -- comes from the corpus.
- Length is not evidence. A short document is harder to judge, not more likely to be machine-written.
- Polish is not evidence. Careful, edited human writing is fluent and correct.

CALIBRATION
- Return a probability that the document is HUMAN-written. 1.0 means certainly human, 0.0 means certainly machine, 0.5 means the text carries no usable evidence either way.
- Use the whole range, and give two decimal places. Do not settle on round values such as 0.20, 0.50 or 0.80: if your belief falls between two of them, say so (0.37, 0.62). These scores are ranked against each other, so every tie is information thrown away.
- Reserve values below 0.10 and above 0.90 for documents where the evidence is genuinely strong.
- Do not assume any particular mix of human and machine documents, and do not try to balance your answers across documents. Judge each one on its own.

Reply with the probability only."""

DEFAULT_PROMPT = "v2"

USER_TEMPLATE = "<document>\n{text}\n</document>"


class JudgeVerdict(BaseModel):
    """The judge's answer: one calibrated probability, no free text.

    Deliberately a single field -- a rationale would double the output tokens on
    every row of a 2000-row corpus and is never stored.
    """

    # The [0, 1] range is stated in the description rather than as ge/le: the
    # OpenAI strict-schema subset rejects `minimum`/`maximum`, and a model that
    # answered 1.2 anyway would raise here instead of being clamped in judge_row.
    human_probability: float = Field(
        description=(
            "Probability, between 0.0 and 1.0, that the document was written by "
            "a human. 1.0 = certainly human, 0.0 = certainly machine (written "
            "by a large language model)."
        ),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquet", type=Path, help="Parquet to add judge scores to.")
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output path (default: overwrite the input in place).",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=sorted(MODELS),
        help=f"Judge model alias (default: {DEFAULT_MODEL}).",
    )
    ap.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        choices=sorted(PROMPTS),
        help=f"System prompt version (default: {DEFAULT_PROMPT}). It names the "
        "output column together with the model, so two versions can be scored "
        "side by side on the same file and compared.",
    )
    ap.add_argument(
        "--text-col",
        default=DEFAULT_TEXT_COL,
        help=f"Column holding the text to judge (default: {DEFAULT_TEXT_COL!r}). "
        "It is the only column the model ever sees.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Rows judged in parallel (default: 4). Lower it if the API "
        "rate-limits; 1 makes the run strictly sequential.",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Truncate each text to this many characters before sending it. "
        "Default: send the whole text.",
    )
    ap.add_argument(
        "--generated-by",
        nargs="+",
        default=None,
        metavar="VALUE",
        help="Only judge rows whose "
        f"{GENERATED_BY_COL!r} is one of these values, e.g. --generated-by "
        "human gpt56luna:0701A. Default: judge every generator. "
        f"Pass {ALL_GENERATORS!r} for the default explicitly.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only judge N eligible rows, drawn round-robin across the "
        f"{GENERATED_BY_COL!r} values so both classes are represented (see "
        "apply_limit). For smoke tests and prompt comparisons. The rest keep "
        "whatever they already have, they are not dropped from the output.",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=RETRIES,
        help=f"Retries per row before it is given up on (default: {RETRIES}).",
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=SAVE_EVERY,
        help=f"Write the Parquet every N scored rows (default: {SAVE_EVERY}).",
    )
    ap.add_argument(
        "--recompute",
        action="store_true",
        help="Re-judge rows that already have a score for this model, without "
        "asking. Default: keep them.",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Skip both prompts (the run is a billed API call): proceed, and "
        "keep existing scores unless --recompute is also given.",
    )
    return ap.parse_args()


def score_column(model_alias: str, prompt_id: str) -> str:
    """Output column for one judge, e.g. ``llm_judge_score_gpt56luna_v2``.

    Both parts are in the name deliberately: scores from two models, or from two
    prompts, are not on the same scale, and a shared column would mix them with
    nothing downstream able to tell which row came from which.
    """
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", f"{model_alias}_{prompt_id}").strip("_").lower()
    return f"{COL_PREFIX}{safe}"


def _is_empty_text(value) -> bool:
    """True for None/NaN or a string that is empty or only whitespace."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def _as_score(value) -> float | None:
    """Coerce a stored cell to a score in [0, 1], or None if it holds none."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 0.0 <= score <= 1.0 else None


class Judge:
    """One judge model under one system prompt, callable per row.

    Holds the provider client (thread-safe in all three SDKs, so the same
    instance serves every worker) and turns whatever the provider returns into a
    plain float.
    """

    def __init__(self, alias: str, prompt_id: str = DEFAULT_PROMPT) -> None:
        spec = MODELS[alias]
        self.alias = alias
        self.provider = spec["provider"]
        self.model = spec["model"]
        self.prompt_id = prompt_id
        self.system_prompt = PROMPTS[prompt_id]
        self._client = self._build_client()

    def _build_client(self):
        # Imported here rather than at module level so a missing optional SDK
        # only blocks the provider that needs it.
        if self.provider == "openai":
            from openai import OpenAI

            return OpenAI()
        if self.provider == "anthropic":
            import anthropic

            return anthropic.Anthropic()
        if self.provider == "gemini":
            from google import genai

            return genai.Client()
        raise ValueError(f"Unsupported provider: {self.provider}")

    def __call__(self, text: str) -> float:
        """P(human-written) for one document. Raises on a failed call."""
        if self.provider == "openai":
            return self._score_openai(text)
        if self.provider == "anthropic":
            return self._score_anthropic(text)
        return self._score_gemini(text)

    def _score_openai(self, text: str) -> float:
        # No temperature: the reasoning-capable models reject anything but the
        # default, and the schema already pins the output shape.
        response = self._client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": USER_TEMPLATE.format(text=text)},
            ],
            response_format=JudgeVerdict,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("model returned no parsable verdict")
        return float(parsed.human_probability)

    def _score_anthropic(self, text: str) -> float:
        # Same forced-tool trick as epai/utils/genai_utils.py: it is the way to
        # get a schema-checked answer out of the Messages API.
        response = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=self.system_prompt,
            tools=[
                {
                    "name": "report_probability",
                    "description": (
                        "Report the probability that the document was written "
                        "by a human rather than by a large language model."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "human_probability": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            }
                        },
                        "required": ["human_probability"],
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "report_probability"},
            messages=[{"role": "user", "content": USER_TEMPLATE.format(text=text)}],
        )
        block = next((b for b in response.content if b.type == "tool_use"), None)
        if block is None or "human_probability" not in block.input:
            raise ValueError("model returned no parsable verdict")
        return float(block.input["human_probability"])

    def _score_gemini(self, text: str) -> float:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=USER_TEMPLATE.format(text=text),
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                response_mime_type="application/json",
                response_schema=JudgeVerdict,
            ),
        )
        parsed = response.parsed
        if parsed is None:
            raise ValueError("model returned no parsable verdict")
        return float(parsed.human_probability)


@dataclass
class RowResult:
    """Outcome for one row: a score, or the error that stopped it."""

    position: int
    score: float | None
    error: str | None = None


def judge_row(
    judge: Judge, position: int, text: str, retries: int
) -> RowResult:
    """Score one row, retrying transient API failures with backoff.

    A row that fails every attempt returns a None score and its last error; the
    caller records it and moves on rather than losing the rest of the run.
    """
    last_error = "unknown error"
    for attempt in range(retries + 1):
        try:
            score = judge(text)
            # Providers do occasionally step outside a schema's bounds, and a
            # score above 1 would silently poison any threshold sweep.
            return RowResult(position=position, score=float(np.clip(score, 0.0, 1.0)))
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == retries:
                break
            # Jittered so parallel workers do not retry in lockstep after a
            # shared rate limit.
            time.sleep(BACKOFF_BASE**attempt * (1.0 + random.random()))
    return RowResult(position=position, score=None, error=last_error)


def check_parquet(df: pd.DataFrame, text_col: str, path: Path) -> np.ndarray:
    """Validate the Parquet and return the boolean mask of judgeable rows."""
    if len(df) == 0:
        raise ValueError(f"{path} has no rows.")
    if text_col not in df.columns:
        raise KeyError(
            f"{path} is missing the text column {text_col!r}. "
            f"Available columns: {list(df.columns)}"
        )

    judgeable = ~df[text_col].apply(_is_empty_text).to_numpy()
    if not judgeable.any():
        raise ValueError(f"{path}: every value in {text_col!r} is empty — nothing to do.")
    return judgeable


def resolve_generated_by(requested: list[str] | None) -> list[str] | None:
    """The effective generator filter; None means 'judge every generator'."""
    if requested is None:
        return list(DEFAULT_GENERATED_BY) if DEFAULT_GENERATED_BY else None
    values = list(requested)
    if any(v.strip().lower() == ALL_GENERATORS for v in values):
        return None
    return values


def apply_generated_by(
    df: pd.DataFrame,
    judgeable: np.ndarray,
    allowed: list[str] | None,
    explicit: bool,
    path: Path,
) -> np.ndarray:
    """Restrict the judgeable mask to rows whose generator is in ``allowed``.

    A file without the column is only an error when the filter was asked for on
    the command line; the default list otherwise steps aside rather than making
    every un-annotated Parquet unrunnable.
    """
    if allowed is None:
        return judgeable
    if GENERATED_BY_COL not in df.columns:
        if explicit:
            raise KeyError(
                f"{path} is missing the {GENERATED_BY_COL!r} column, so "
                f"--generated-by cannot be applied. Available columns: "
                f"{list(df.columns)}"
            )
        print(
            f"NOTE  {path} has no {GENERATED_BY_COL!r} column — the default "
            "generator filter is skipped and every row is eligible."
        )
        return judgeable

    values = df[GENERATED_BY_COL].astype(str).to_numpy()
    filtered = judgeable & np.isin(values, allowed)
    missing = [v for v in allowed if v not in set(values)]
    if missing:
        print(
            f"NOTE  no rows for {GENERATED_BY_COL} "
            + ", ".join(repr(m) for m in missing)
            + "; the file has: "
            + ", ".join(sorted(set(values)))
        )
    if not filtered.any():
        raise ValueError(
            f"{path}: no row with a non-empty text has {GENERATED_BY_COL} in "
            f"{allowed} — the file has {sorted(set(values))}."
        )
    return filtered


def apply_limit(
    df: pd.DataFrame, judgeable: np.ndarray, limit: int | None
) -> np.ndarray:
    """Restrict the judgeable mask to ``limit`` eligible rows, spread evenly.

    Counted over eligible rows rather than file positions, and drawn round-robin
    across the ``generated_by`` values rather than in file order. These corpora
    are stored one generator after another -- every human row, then every machine
    row -- so taking the first N would hand back one class only, and the summary's
    accuracy and ROC-AUC lines (which is the whole point of a small run) would
    have nothing to separate.

    Within a generator the rows are still taken in file order, so the same
    ``--limit N`` on the same file always selects the same rows: two prompt
    versions are then compared on identical documents rather than on two samples.
    """
    if limit is None:
        return judgeable
    if limit < 1:
        raise ValueError(f"--limit must be >= 1, got {limit}")

    eligible = np.flatnonzero(judgeable)
    if GENERATED_BY_COL in df.columns:
        generators = df[GENERATED_BY_COL].astype(str).to_numpy()
        buckets: dict[str, list[int]] = {}
        for pos in eligible:
            buckets.setdefault(generators[pos], []).append(int(pos))
        chosen: list[int] = []
        depth = 0
        while len(chosen) < limit and any(len(b) > depth for b in buckets.values()):
            for name in sorted(buckets):
                if len(buckets[name]) > depth:
                    chosen.append(buckets[name][depth])
                    if len(chosen) == limit:
                        break
            depth += 1
        picked = np.array(chosen, dtype=int)
    else:
        picked = eligible[:limit]

    limited = np.zeros_like(judgeable)
    limited[picked] = True
    return limited


def blocking_reason(provider: str) -> str | None:
    """Why this provider cannot run here, or None if it can."""
    env_var = _REQUIRED_ENV[provider]
    if not os.environ.get(env_var):
        return f"needs {env_var}"
    return None


def positions_to_judge(
    df: pd.DataFrame, col: str, judgeable: np.ndarray, recompute: bool
) -> np.ndarray:
    """Integer row positions to send to the judge."""
    if recompute or col not in df.columns:
        return np.flatnonzero(judgeable)
    scored = np.array([_as_score(v) is not None for v in df[col]])
    return np.flatnonzero(judgeable & ~scored)


def count_scored(df: pd.DataFrame, col: str, judgeable: np.ndarray) -> int:
    """Judgeable rows that already hold a usable score for this judge."""
    if col not in df.columns:
        return 0
    scored = np.array([_as_score(v) is not None for v in df[col]])
    return int((judgeable & scored).sum())


def confirm_recompute(n_scored: int) -> bool:
    """Ask whether to re-judge rows that already have a score. Default: no.

    Keeping them is nearly always right -- the column is named after this exact
    (model, prompt) pair, so whatever is in it was produced by this same judge --
    and re-judging means paying for those rows a second time.
    """
    answer = input(
        f"{n_scored} judgeable row(s) already have a score for this judge. "
        "Re-judge them? [y/N]: "
    ).strip().lower()
    return answer in {"y", "yes"}


def print_findings(
    path: Path,
    df: pd.DataFrame,
    text_col: str,
    col: str,
    alias: str,
    judge_model: str,
    prompt_id: str,
    judgeable: np.ndarray,
    positions: np.ndarray,
    allowed: list[str] | None,
) -> None:
    n_empty = int(df[text_col].apply(_is_empty_text).sum())
    n_scored = count_scored(df, col, judgeable)

    print("=" * 78)
    print("Parquet")
    print(f"  file          : {path}")
    print(f"  rows          : {len(df)}")
    print(f"  text column   : {text_col!r}  (the only column sent to the model)")
    print(f"  empty text    : {n_empty}  (never judged, score stays Null)")
    if GENERATED_BY_COL not in df.columns:
        print("  generated_by  : column absent — every row is eligible")
    else:
        print(
            "  generated_by  : "
            + ("all generators" if allowed is None else ", ".join(allowed))
        )
        # Per generator, always: with a filter it shows the filter worked, and
        # without one it is the check that a --limit run has both classes in it.
        generators = df[GENERATED_BY_COL].astype(str).to_numpy()
        for name in sorted(set(generators)):
            n = int((judgeable & (generators == name)).sum())
            print(f"    {name:<18}: {n}")
    print(f"  eligible      : {int(judgeable.sum())}")
    print("Judge")
    print(f"  alias         : {alias}")
    print(f"  api model     : {judge_model}")
    print(f"  prompt        : {prompt_id}")
    print("  score         : P(human)  —  1.0 ⇒ human, 0.0 ⇒ machine")
    print(f"  column        : {col}  ({'exists' if col in df.columns else 'new'})")
    print(f"  already scored: {n_scored}")
    print(f"  to judge now  : {len(positions)}")
    print("=" * 78)


def evaluate(df: pd.DataFrame, col: str) -> None:
    """Quick read on the baseline, if the file carries a ground-truth column.

    ``generated_by`` is 'human' for the human rows and a generator name for the
    rest, which is all that is needed to check whether the judge separates them
    at all. Reported here only as a sanity check -- the real comparison against
    Binoculars happens in the train/ scripts, off the stored column.
    """
    if "generated_by" not in df.columns:
        return

    scores = np.array([_as_score(v) for v in df[col]], dtype=object)
    has_score = np.array([s is not None for s in scores])
    truth = df["generated_by"].astype(str).to_numpy()
    known = has_score & (truth != "unknown")
    if not known.any():
        return

    # The score is P(human), so 'human' is the positive class here and the
    # verdict is `score >= 0.5 ⇒ human`. A judge that separates the two groups at
    # all puts the human rows at the HIGH end.
    y_true = (truth[known] == "human").astype(int)
    y_score = np.array([s for s in scores[known]], dtype=float)

    print()
    print("Sanity check against 'generated_by' (not a substitute for the train scripts)")
    print("  score is P(human): 1.0 ⇒ human, 0.0 ⇒ machine")
    for name in sorted(set(truth[known])):
        group = y_score[truth[known] == name]
        print(f"  mean score, {name:<18}: {group.mean():.3f}  (n={len(group)})")
    accuracy = ((y_score >= 0.5).astype(int) == y_true).mean()
    print(f"  accuracy @ 0.5{'':<16}: {accuracy:.3f}")

    # How much of the ranking the judge threw away by answering in round numbers.
    # ROC-AUC only reads the order, so tied scores are the one failure mode a
    # good mean separation can hide -- and the thing the v2 prompt asks against.
    n_distinct = len(set(y_score.tolist()))
    print(
        f"  distinct values{'':<15}: {n_distinct} over {len(y_score)} scored rows"
    )

    if len(set(y_true)) == 2:
        from sklearn.metrics import roc_auc_score

        print(f"  ROC-AUC{'':<23}: {roc_auc_score(y_true, y_score):.3f}")


def print_summary(
    out_path: Path,
    col: str,
    n_done: int,
    failures: list[tuple[int, str]],
    seconds: float,
    df: pd.DataFrame,
) -> None:
    print()
    print("=" * 78)
    print("Judge scores written")
    print(f"  output  : {out_path}")
    print(f"  column  : {col}")
    rate = n_done / seconds if seconds else float("inf")
    print(f"  scored  : {n_done} rows in {seconds:.1f}s  ({rate:.2f} rows/s)")
    print(f"  failed  : {len(failures)}")
    print("=" * 78)

    if failures:
        # One line per distinct error, so a rate limit does not print 400 times.
        by_message: dict[str, int] = {}
        for _, message in failures:
            by_message[message] = by_message.get(message, 0) + 1
        print("  failures (score left Null, re-run to retry them):")
        for message, count in sorted(by_message.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>5}x  {message[:100]}")

    evaluate(df, col)
    print()


def main() -> None:
    args = parse_args()
    out_path = args.out or args.parquet
    alias = args.model
    prompt_id = args.prompt
    spec = MODELS[alias]
    col = score_column(alias, prompt_id)

    blocked = blocking_reason(spec["provider"])
    if blocked:
        raise SystemExit(f"Cannot run judge {alias!r}: {blocked}.")

    allowed = resolve_generated_by(args.generated_by)

    df = load_parquet(args.parquet)
    judgeable = check_parquet(df, args.text_col, args.parquet)
    judgeable = apply_generated_by(
        df, judgeable, allowed, args.generated_by is not None, args.parquet
    )
    n_eligible = int(judgeable.sum())
    judgeable = apply_limit(df, judgeable, args.limit)

    recompute = args.recompute
    positions = positions_to_judge(df, col, judgeable, recompute)
    print_findings(
        args.parquet,
        df,
        args.text_col,
        col,
        alias,
        spec["model"],
        prompt_id,
        judgeable,
        positions,
        allowed,
    )
    if args.limit is not None:
        print(
            f"NOTE  --limit {args.limit}: {int(judgeable.sum())} of "
            f"{n_eligible} eligible row(s) are considered, drawn round-robin "
            "across the generators; the counts above are within that window."
        )

    # Existing scores are kept unless the run is told otherwise. --recompute
    # says yes up front, --yes takes the default (keep) without stopping a
    # non-interactive run on a prompt it cannot answer.
    n_scored = count_scored(df, col, judgeable)
    if n_scored and not recompute and not args.yes:
        print()
        if confirm_recompute(n_scored):
            recompute = True
            positions = positions_to_judge(df, col, judgeable, recompute)
            print(f"Re-judging all {len(positions)} judgeable row(s).")
        else:
            print("Keeping the existing scores; only unscored rows are judged.")

    if len(positions) == 0:
        print("Every judgeable row already has a score — nothing to do.")
        print("(Pass --recompute to judge them again.)")
        return

    print()
    print(f"NOTE  this is a billed API call: {len(positions)} row(s) will be sent.")
    if not args.yes:
        answer = input(f"Proceed and write to {out_path}? Type 'yes': ").strip().lower()
        if answer != "yes":
            print("Aborted — no changes made.")
            return

    judge = Judge(alias, prompt_id)

    # Existing values are carried over untouched; only `positions` are replaced.
    values: list[float | None] = (
        [_as_score(v) for v in df[col]] if col in df.columns else [None] * len(df)
    )

    # The judge sees this and only this: the text of one row, optionally
    # truncated. Nothing else from the Parquet is ever put in a prompt.
    texts = {
        pos: (
            str(df.iloc[pos][args.text_col])[: args.max_chars]
            if args.max_chars
            else str(df.iloc[pos][args.text_col])
        )
        for pos in positions
    }

    failures: list[tuple[int, str]] = []
    n_done = 0
    save_lock = threading.Lock()
    started = time.perf_counter()

    def write_out() -> None:
        """Persist what has been scored so far, without clobbering other columns."""
        df[col] = pd.Series(values, index=df.index, dtype="float64")
        save_parquet(df, out_path)

    # The executor is managed by hand rather than with a `with` block: on Ctrl-C
    # the block's exit would wait for every queued row to be judged, which on a
    # 2000-row file is exactly the wait the interrupt was meant to avoid.
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    try:
        futures = {
            pool.submit(judge_row, judge, pos, texts[pos], args.retries): pos
            for pos in positions
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Judging ({alias})",
            unit="row",
        ):
            result = future.result()
            values[result.position] = result.score
            if result.error is not None:
                failures.append((result.position, result.error))
                tqdm.write(f"  row {result.position} failed: {result.error}")
            n_done += 1

            # Save periodically so an interrupt or a crash does not throw away
            # calls that have already been paid for.
            with save_lock:
                if n_done % args.save_every == 0:
                    write_out()
    except KeyboardInterrupt:
        print("\nInterrupted — writing the scores obtained so far.")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    write_out()
    print_summary(
        out_path, col, n_done - len(failures), failures, time.perf_counter() - started, df
    )


if __name__ == "__main__":
    main()

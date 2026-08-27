"""Ask the judge what it saw: rationales and countable cues, no scores touched.

``add_llm_judge_scores.py`` produces one number per document and nothing else --
deliberately, because a rationale would double the output tokens on every row of
a 6000-row corpus. This script is the other half: it goes back over a *sample* of
rows and asks the same model, under the same prompt version, what evidence the
document actually carries. It never writes a score column and never reads one
except to choose which rows are worth asking about, so the numbers already in the
corpora cannot be disturbed by running it.

WHAT COMES BACK, AND WHY IT IS NOT PROSE ALONE
----------------------------------------------
Three things per row:

``llm_judge_cues_human_*`` / ``llm_judge_cues_machine_*``
    Lists drawn from a closed vocabulary (HUMAN_CUES / MACHINE_CUES below).
    That vocabulary is not invented here: it is one entry per bullet of the
    prompt version's own evidence lists, which is what makes a count
    interpretable -- ``even_rhythm`` cited on 78% of machine rows and 12% of
    human rows is a statement about the same cue the score was formed from.
    6000 free-text rationales is something you read twenty of; 6000 cue labels
    is a table.

``llm_judge_reason_*``
    One or two sentences naming the strongest single piece of evidence. This is
    the part a person reads while looking at the document beside it.

``llm_judge_other_*``
    The escape hatch, and the only field that can surprise you. The cue lists
    are closed -- on the API paths they are enum-constrained by the response
    schema, so the model *cannot* return a label that is not in them. That is
    what makes the counts trustworthy and it is also a blindfold: if the judge
    is really separating these corpora on something no bullet of the prompt
    mentions, a table built only from the vocabulary will come back tidy,
    complete-looking, and silent about it. This project has already been bitten
    by exactly that shape of thing once -- an unnormalised-punctuation artefact
    whose label polarity flipped between corpora, which no cue name here
    covers. So the model is asked, separately and in free text, for evidence it
    found that none of the cue names fits. Usually empty. Read the ones that
    are not: they are dumped verbatim to ``other_evidence.csv`` beside the cue
    table, because a recurring phrase there is a candidate cue for the next
    prompt version.

THE MODEL IS NOT SHOWN ITS OWN SCORE
------------------------------------
The obvious design -- "you rated this 0.08, why?" -- measures compliance, not
evidence: a model handed a number will justify that number. So the call here is
blind. It sees the document and the evidence half of the prompt version, and is
asked what is present; it is told explicitly not to give a verdict or a
probability. The stored score is then something to compare the cues *against*
rather than something the cues were derived from.

Which also means: a rationale is a plausible account produced about a document,
not a trace of the computation that produced the score. Use it to form
hypotheses ("it leans on hedging"), and test those hypotheses with the counts.

WHICH ROWS -- THE BANDS
-----------------------
Explaining all 6000 rows would cost more than the scoring run did and answer
nothing that 300 rows do not. ``--select`` picks bands off the existing score
column, deduplicated in the order below and capped per band by ``--per-band``:

``wrong``
    The judge's verdict disagrees with ``generated_by``, most confidently wrong
    first -- where the failure modes live. They are not spread evenly: on
    ghostbuster_gpt56luna the judge calls ~49% of human ``essay`` rows machine,
    ~5% of human ``wp`` rows, and 0% of human ``reuter`` rows, so this band is
    mostly essays unless it is drawn per domain, which it is (below).
``uncertain``
    Score within ``--uncertain-band`` of 0.5, closest first: the documents that
    carried no usable evidence either way.
``confident``
    Verdict correct and at least ``--confident-band`` from 0.5, most confident
    first. The control -- without it you read only failures and conclude the
    judge is bad.
``disagree``
    Needs ``--vs-cols``. Rows where this judge's verdict and another judge's
    verdict differ, widest score gap first. ``wrong`` conflates two things a
    corpus of three judges can tell apart: a row every judge gets wrong is a
    hard document, and says something about the corpus; a row this judge gets
    wrong while the others get it right is this judge's own quirk, and says
    something about the judge. Only the second is a lead. A row needs a score
    in every column named -- primary and ``--vs-cols`` alike -- to be eligible,
    so a partially-scored file yields a smaller band rather than a band whose
    rows are not comparable.
``all``
    Every eligible row, file order. For a small corpus, or with --per-band.

Within a band, rows are drawn round-robin across ``domain`` x ``generated_by``
-- so every domain and both classes are represented, in the proportions the file
allows -- and in a deterministic order, so the same command twice selects the
same documents. ``--per-band 400`` on a three-domain corpus is therefore about
66 rows from each (domain, class) cell of that band, and the three default bands
come to ~1200 rows explained.

Run:
    uv run python epai/ai_detection/score/add_llm_judge_reasons.py \
        data/parquet/ghostbuster_gpt56luna.parquet

    # the same, explained by the local model instead of the API model that
    # scored. --gpu and --cpu-threads mean what they mean in
    # add_llm_judge_scores.py and add_embeddings.py: pin this run to one card
    # and cap the CPU pools.
    uv run python epai/ai_detection/score/add_llm_judge_reasons.py \
        data/parquet/ghostbuster_gpt56luna.parquet \
        --model qwen3-14b --score-col llm_judge_score_gpt56luna_v3 \
        --gpu 0 --cpu-threads 16

    # only the failures, more of them
    uv run python epai/ai_detection/score/add_llm_judge_reasons.py \
        data/parquet/ghostbuster_gpt56luna.parquet --select wrong --per-band 200

    # the rows this judge reads differently from the other two: what gpt56luna
    # sees that sonnet5 and qwen3-14b do not, on the same document.
    uv run python epai/ai_detection/score/add_llm_judge_reasons.py \
        data/parquet/ghostbuster_gpt56luna.parquet \
        --select disagree --per-band 150 \
        --vs-cols llm_judge_score_sonnet5_v3 llm_judge_score_qwen3_14b_v3

The cue frequency table is printed and saved under
``results/llm_judge_reasons/<parquet>+<judge>/``. Rows already carrying a
rationale are skipped unless ``--recompute``, and the Parquet is checkpointed
every ``--save-every`` rows, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

from epai.ai_detection.score.add_llm_judge_scores import (
    ALL_GENERATORS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_MAX_DOC_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_TEXT_COL,
    DOMAIN_COL,
    GENERATED_BY_COL,
    MODE_PARSED,
    RETRIES,
    SAVE_EVERY,
    USER_TEMPLATE,
    ApiJudge,
    JudgeModel,
    LocalJudge,
    _as_score,
    _identifier,
    apply_generated_by,
    apply_runtime_limits,
    blocking_reason,
    check_parquet,
    dtype_for,
    evidence_head,
    print_models,
    resolve_generated_by,
    resolve_model,
    score_column,
    strata,
    take_round_robin,
)
from epai.utils.parquet_utils import load_parquet, save_parquet

load_dotenv()

# The cue vocabulary: one entry per bullet of a prompt version's evidence lists,
# named so the name alone is readable in a table. Keep these in step with
# PROMPTS[...] in add_llm_judge_scores.py -- if a prompt version grows a new
# bullet, a cue belongs here too, or the model will have evidence it cannot
# report and the counts will quietly under-report it.
HUMAN_CUES: dict[str, str] = {
    "specifics": "particular names, numbers, dates, places, results or quoted "
                 "words, load-bearing rather than decorative",
    "commitment": "a definite claim, an unbalanced opinion, an argument that "
                  "goes somewhere instead of hedging back to neutral",
    "asymmetry": "uneven emphasis, a paragraph that runs long, a point raised "
                 "and dropped, a sentence that changes direction mid-way",
    "idiom": "slang, humour, irony, in-group shorthand, the habits of a field "
             "used the way a practitioner uses them",
    "errors": "a misspelling, a wrong-word substitution, an agreement or tense "
              "slip, a missing word, a run-on or fragmentary sentence",
}

MACHINE_CUES: dict[str, str] = {
    "generic_vocabulary": "fluent but generic wording; an abstraction where a "
                          "specific would have been natural",
    "even_rhythm": "uniform sentence and paragraph lengths, repetitive clause "
                   "structure, predictable pacing",
    "scaffolding": "symmetrical structure, explicit enumeration, a closing "
                   "sentence that restates what came before",
    "hedging": "balanced framing that declines to commit: every side "
               "acknowledged, none pressed",
    "coverage": "each obvious aspect given a proportionate share, nothing "
                "pursued further than the rest",
    "assistant_register": "the connectives, intensifiers and stock transitions "
                          "of assistant prose",
}

# The schema's version of the same two lists. Literal rather than str: it is
# what makes the provider enforce the vocabulary instead of the parser having to
# repair it afterwards.
HumanCue = Literal[
    "specifics", "commitment", "asymmetry", "idiom", "errors"
]
MachineCue = Literal[
    "generic_vocabulary", "even_rhythm", "scaffolding", "hedging", "coverage",
    "assistant_register",
]

# Two spellings of one vocabulary is one spelling too many; this is the check
# that they have not drifted apart.
assert set(get_args(HumanCue)) == set(HUMAN_CUES), "HumanCue != HUMAN_CUES"
assert set(get_args(MachineCue)) == set(MACHINE_CUES), "MachineCue != MACHINE_CUES"

# Columns. Deliberately NOT under `llm_judge_score_`, which is the prefix
# analyse_score_human_vs_ai.py collects as scores: a rationale is not a score and
# must not turn up in a threshold sweep. `llm_judge` is still in the name, so
# truncate_text.py drops these along with the scores when a corpus is re-cut --
# which is right, they are judgements of the untruncated text.
REASON_PREFIX = "llm_judge_reason_"
HUMAN_CUES_PREFIX = "llm_judge_cues_human_"
MACHINE_CUES_PREFIX = "llm_judge_cues_machine_"
OTHER_PREFIX = "llm_judge_other_"

# Bands, in the order they are filled. A row taken by an earlier band is not
# offered to a later one.
BAND_WRONG = "wrong"
BAND_UNCERTAIN = "uncertain"
BAND_CONFIDENT = "confident"
BAND_DISAGREE = "disagree"
BAND_ALL = "all"
BANDS = (BAND_WRONG, BAND_UNCERTAIN, BAND_CONFIDENT, BAND_DISAGREE, BAND_ALL)
DEFAULT_BANDS = (BAND_WRONG, BAND_UNCERTAIN, BAND_CONFIDENT)

DEFAULT_PER_BAND = 100
DEFAULT_THRESHOLD = 0.5
DEFAULT_UNCERTAIN_BAND = 0.10
DEFAULT_CONFIDENT_BAND = 0.40

# An explanation is short by construction; these bound a model that decides
# otherwise. The API cap is generous because a truncated JSON object is a lost
# row, while the local generator stops at the end-of-turn token anyway.
MAX_TOKENS = 700
LOCAL_MAX_NEW_TOKENS = 400

# Rationale length asked for and enforced. Long enough for the evidence, short
# enough to read a hundred of them.
MAX_RATIONALE_CHARS = 400

# The free-text escape hatch is bounded harder. It exists to name one thing the
# vocabulary has no word for, and a model given room to write a paragraph there
# writes a second rationale instead -- which is not what the column is read for.
MAX_OTHER_CHARS = 300

REPORT_DIR = Path("results") / "llm_judge_reasons"


def _cue_listing() -> str:
    """The vocabulary as the model sees it, generated from the same dicts."""
    lines = ["human_cues -- evidence of a human author:"]
    lines += [f"  {name}: {desc}" for name, desc in HUMAN_CUES.items()]
    lines.append("")
    lines.append("machine_cues -- evidence of a language model:")
    lines += [f"  {name}: {desc}" for name, desc in MACHINE_CUES.items()]
    return "\n".join(lines)


EXPLAIN_BLOCK = """WHAT TO REPORT

You are NOT scoring this document. Do not give a verdict, a probability, or any
number. Do not say which way the document leans overall. Report only what is
present in it, using the vocabulary below.

{cues}

Report, for this document:
- human_cues: every cue from the human list that is genuinely present. Cite a
  cue only if you could point at the words that carry it. An empty list is a
  correct answer for a document that carries none.
- machine_cues: the same, from the machine list.
- other_evidence: anything you noticed that is genuinely evidence about who
  wrote this document but that NONE of the cue names above covers -- a habit of
  punctuation or spacing, a formatting or encoding trace, a typographic
  convention, a metadata remnant, a quirk of how the text is laid out, anything
  else. Name it in a short phrase and quote the few characters or words that
  carry it. Leave it empty -- an empty string -- if the cue names above covered
  everything you found, which is the ordinary case. Do not restate a cue you
  already listed in different words, and do not put a verdict here.
- rationale: one or two sentences naming the strongest single piece of evidence
  you found, and where in the document it is. Quote a few words of it if that is
  the clearest way to say it. No verdict, no probability.

A document may carry cues from both lists. Report both; it is not a
contradiction, and resolving it is not your job here."""


def explain_prompt(prompt_id: str) -> str:
    """The system prompt: a prompt version's evidence half, then what to report.

    The evidence half is shared verbatim with the scoring run (see
    ``evidence_head``) -- same genre framing, same two lists, same rules about
    what is never evidence. Only the closing instruction differs, because this
    call asks what is there rather than for a number. That sharing is the whole
    basis for reading the cue counts as being about the score.
    """
    return evidence_head(prompt_id) + "\n\n" + EXPLAIN_BLOCK.format(cues=_cue_listing())


class Explanation(BaseModel):
    """One blind read of a document: what is present, and the strongest of it.

    Cues before rationale on purpose: the countable fields are generated first,
    so they are not written to agree with a sentence of prose the model has
    already committed to. ``other_evidence`` sits between them for the same
    reason in reverse -- it is asked only after the model has tried and failed
    to say the thing in the closed vocabulary, so it holds what the vocabulary
    could not express rather than a paraphrase of what it could.
    """

    human_cues: list[HumanCue] = Field(
        description="Cues of human authorship that are genuinely present. May be empty."
    )
    machine_cues: list[MachineCue] = Field(
        description="Cues of machine authorship that are genuinely present. May be empty."
    )
    # Required, with no default, deliberately. This schema is sent verbatim to
    # the providers, and OpenAI's strict structured-output mode wants every
    # property in `required` and is unforgiving about extra keywords such as
    # `default`. The model is told in EXPLAIN_BLOCK to send "" when it has
    # nothing to add, and the two non-schema paths coerce a missing key to ""
    # in explanation_from_payload, so nothing here relies on a default.
    other_evidence: str = Field(
        description="Evidence about authorship that none of the cue names "
                    "covers -- punctuation or spacing habits, formatting or "
                    "encoding traces, typographic conventions, layout quirks. "
                    "A short phrase plus the few characters that carry it. "
                    "Empty string if the cue names covered everything.",
    )
    rationale: str = Field(
        description="One or two sentences naming the strongest single piece of "
                    "evidence in the document and where it is. No verdict, no "
                    "probability."
    )


def clean_explanation(raw: Explanation) -> Explanation:
    """Deduplicate the cue lists and bound the rationale.

    Providers do repeat a list entry, and a duplicate would count twice in the
    frequency table -- a silent, plausible-looking inflation of exactly the
    number this script exists to produce.
    """
    return Explanation(
        human_cues=list(dict.fromkeys(raw.human_cues)),
        machine_cues=list(dict.fromkeys(raw.machine_cues)),
        other_evidence=" ".join((raw.other_evidence or "").split())[:MAX_OTHER_CHARS],
        rationale=" ".join(raw.rationale.split())[:MAX_RATIONALE_CHARS],
    )


def coerce_cues(values, vocabulary: dict[str, str], unknown: dict[str, int]) -> list[str]:
    """Keep the cues that are in the vocabulary; count the ones that are not.

    A model that answers ``"Even rhythm"`` or invents ``"no_typos"`` has not
    failed the row -- the rationale is still worth having -- but the label
    cannot go in a frequency table under a name nothing else uses. Repaired
    silently would be worse: the count would look complete while quietly
    dropping evidence, so the summary reports what was discarded.
    """
    kept: list[str] = []
    for value in values or []:
        key = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
        if key in vocabulary:
            kept.append(key)
        else:
            label = str(value)[:40]
            unknown[label] = unknown.get(label, 0) + 1
    return kept


def explanation_from_payload(payload: dict, unknown: dict[str, int]) -> Explanation:
    """Build an Explanation from a raw dict (a tool call, or parsed JSON)."""
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object, got {type(payload).__name__}")
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("no rationale in the answer")
    other = payload.get("other_evidence")
    return clean_explanation(
        Explanation(
            human_cues=coerce_cues(payload.get("human_cues"), HUMAN_CUES, unknown),
            machine_cues=coerce_cues(payload.get("machine_cues"), MACHINE_CUES, unknown),
            # A model that has nothing to add here answers "", None, "none" or
            # omits the key; all four mean the same thing and none of them is
            # worth a row in other_evidence.csv.
            other_evidence=other if isinstance(other, str) else "",
            rationale=rationale,
        )
    )


def parse_json_object(text: str) -> dict:
    """The first JSON object in a free-text answer.

    Local models fence their JSON, prefix it, or add a closing remark. The
    braces are found rather than the string trusted, and a fenced block is
    unwrapped first so a `````json`` line cannot be mistaken for content.
    """
    body = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE)
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in answer: {body.strip()[:120]!r}")
    return json.loads(body[start : end + 1])


class ApiExplainer(ApiJudge):
    """The explaining call over an API: one request per row, many in flight.

    Subclasses the judge only to inherit its client construction and its
    provider dispatch; the system prompt and the response schema are this
    script's, and no method here can write a score.
    """

    def __init__(self, spec: JudgeModel, prompt_id: str = DEFAULT_PROMPT) -> None:
        super().__init__(
            spec, prompt_id, MODE_PARSED, system_prompt=explain_prompt(prompt_id)
        )
        self.unknown_cues: dict[str, int] = {}

    def __call__(self, text: str) -> Explanation:  # type: ignore[override]
        if self.provider == "openai":
            return self._explain_openai(text)
        if self.provider == "anthropic":
            return self._explain_anthropic(text)
        return self._explain_gemini(text)

    def _explain_openai(self, text: str) -> Explanation:
        response = self._client.chat.completions.parse(
            model=self.model_id,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": USER_TEMPLATE.format(text=text)},
            ],
            response_format=Explanation,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("model returned no parsable explanation")
        return clean_explanation(parsed)

    def _explain_anthropic(self, text: str) -> Explanation:
        # Same forced-tool trick as the scoring path; the schema is generated
        # from the pydantic model so the enums cannot drift from HUMAN_CUES.
        schema = Explanation.model_json_schema()
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=MAX_TOKENS,
            system=self.system_prompt,
            tools=[
                {
                    "name": "report_evidence",
                    "description": "Report the cues present in the document.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": "report_evidence"},
            messages=[{"role": "user", "content": USER_TEMPLATE.format(text=text)}],
        )
        block = next((b for b in response.content if b.type == "tool_use"), None)
        if block is None:
            raise ValueError("model returned no parsable explanation")
        return explanation_from_payload(dict(block.input), self.unknown_cues)

    def _explain_gemini(self, text: str) -> Explanation:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=USER_TEMPLATE.format(text=text),
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                response_mime_type="application/json",
                response_schema=Explanation,
            ),
        )
        parsed = response.parsed
        if parsed is None:
            raise ValueError("model returned no parsable explanation")
        return clean_explanation(parsed)


# Local models have no schema to be held to, so the shape is asked for in words.
# Appended to the system prompt rather than folded into EXPLAIN_BLOCK: the API
# providers enforce the schema themselves and telling them about JSON as well
# invites them to describe it instead of filling it in.
LOCAL_JSON_INSTRUCTION = """OUTPUT FORMAT
Reply with one JSON object and nothing else -- no markdown fence, no commentary
before or after it:
{"human_cues": ["..."], "machine_cues": ["..."], "other_evidence": "", "rationale": "..."}
Use only the cue names listed above. Either list may be empty, and
"other_evidence" is an empty string unless you found something the cue names do
not cover."""


class LocalExplainer(LocalJudge):
    """The explaining call on the GPU, a batch at a time.

    Inherits the loading, the chat templating, the document truncation and the
    left-padded batching from LocalJudge, and replaces only what is asked and
    how the answer is read. Generation is required here -- there is no single
    token to read a distribution off -- so this is the one local path that
    decodes, and it is why the run is a sample rather than a whole corpus.
    """

    def __init__(
        self,
        spec: JudgeModel,
        prompt_id: str = DEFAULT_PROMPT,
        device: str = DEFAULT_DEVICE,
        dtype: str = DEFAULT_DTYPE,
        max_doc_tokens: int = DEFAULT_MAX_DOC_TOKENS,
        cpu_threads: int | None = None,
    ) -> None:
        super().__init__(
            spec,
            prompt_id=prompt_id,
            mode=MODE_PARSED,
            device=device,
            dtype=dtype,
            max_doc_tokens=max_doc_tokens,
            system_prompt=explain_prompt(prompt_id) + "\n\n" + LOCAL_JSON_INSTRUCTION,
            cpu_threads=cpu_threads,
        )
        self.unknown_cues: dict[str, int] = {}

    def __call__(self, text: str) -> Explanation:  # type: ignore[override]
        return self.explain_batch([text])[0]

    def explain_batch(self, texts: list[str]) -> list[Explanation]:
        torch = self._torch
        encoded = self._encode(texts)
        with torch.inference_mode():
            out = self.model.generate(
                **encoded,
                max_new_tokens=LOCAL_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        answers = self.tokenizer.batch_decode(
            out[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return [
            explanation_from_payload(parse_json_object(answer), self.unknown_cues)
            for answer in answers
        ]


def build_explainer(spec: JudgeModel, prompt_id: str, args):
    """The explainer for this run: local models load here, APIs just connect."""
    if spec.is_local:
        return LocalExplainer(
            spec,
            prompt_id=prompt_id,
            device=args.device,
            dtype=dtype_for(spec, args.dtype),
            max_doc_tokens=args.max_doc_tokens,
            cpu_threads=args.cpu_threads,
        )
    return ApiExplainer(spec, prompt_id=prompt_id)


@dataclass
class RowExplanation:
    """Outcome for one row: an explanation, or the error that stopped it."""

    position: int
    explanation: Explanation | None
    error: str | None = None


def explain_row(explainer, position: int, text: str, retries: int) -> RowExplanation:
    """Explain one row, retrying transient failures with backoff."""
    last_error = "unknown error"
    for attempt in range(retries + 1):
        try:
            return RowExplanation(position=position, explanation=explainer(text))
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == retries:
                break
            delay = explainer.backoff(attempt)
            if delay:
                time.sleep(delay)
    return RowExplanation(position=position, explanation=None, error=last_error)


def explain_batch(
    explainer, positions: list[int], texts: dict[int, str], retries: int
) -> list[RowExplanation]:
    """Explain a batch in one generate call, falling back to row-at-a-time.

    More likely to be needed than in the scoring run: one row whose JSON does
    not parse takes the batch with it, and the fallback is what isolates it.
    """
    try:
        results = explainer.explain_batch([texts[pos] for pos in positions])
        return [
            RowExplanation(position=pos, explanation=item)
            for pos, item in zip(positions, results)
        ]
    except Exception as exc:
        tqdm.write(
            f"  batch of {len(positions)} failed ({type(exc).__name__}: {exc}); "
            "retrying those rows one at a time"
        )
        return [explain_row(explainer, pos, texts[pos], retries) for pos in positions]


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
# Which rows get explained
# ---------------------------------------------------------------------------


@dataclass
class Selection:
    """The chosen rows, and which band each one came from."""

    positions: np.ndarray
    band_of: dict[int, str]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for band in self.band_of.values():
            out[band] = out.get(band, 0) + 1
        return out


def select_rows(
    df: pd.DataFrame,
    eligible: np.ndarray,
    scores: np.ndarray,
    bands: list[str],
    per_band: int,
    threshold: float,
    uncertain_band: float,
    confident_band: float,
    vs_scores: dict[str, np.ndarray] | None = None,
) -> Selection:
    """Pick the rows worth asking about, band by band.

    Bands are filled in the order given and a row already taken is not offered
    again, so `wrong` keeps its most interesting rows rather than losing them to
    `confident`. Within a band the order is by how extreme the row is -- most
    confidently wrong, closest to the fence, most confidently right -- which is
    what makes a small --per-band the *interesting* rows rather than the first
    ones in the file.

    That order is then drawn round-robin across `domain` x `generated_by` (the
    same strata a limited scoring run uses), so a band is not spent on one
    domain. It matters more here than it does there: the judge's failures are
    not spread evenly -- nearly every `wrong` row of a Ghostbuster corpus is a
    human `essay` -- so an unstratified `wrong` band would be essays only and
    the cue table would have nothing to say about why `reuter` never fails.

    `vs_scores` maps a column name to that judge's scores and is what the
    `disagree` band is formed from; it is ignored by every other band.
    """
    has_score = ~np.isnan(scores)
    has_truth = GENERATED_BY_COL in df.columns
    generators = (
        df[GENERATED_BY_COL].astype(str).to_numpy()
        if has_truth
        else np.array(["all"] * len(df))
    )
    is_human = generators == "human"
    # NaN compares False, so an unscored row is never in a score-based band.
    predicted_human = scores >= threshold
    distance = np.abs(scores - 0.5)

    # The disagree band. Every judge named must have scored the row, or the
    # comparison is between a verdict and an absence: `all_scored` is the guard,
    # and it is why a half-scored file gives a short band rather than a
    # misleading one. `spread` orders it -- the widest gap between any two
    # judges on the same document first, which is where the reading is most
    # likely to differ for a reason rather than by rounding.
    panel = np.vstack([scores] + [vs_scores[col] for col in sorted(vs_scores or {})])
    all_scored = ~np.isnan(panel).any(axis=0)
    verdicts = panel >= threshold
    split = verdicts.any(axis=0) & ~verdicts.all(axis=0)
    # +/-inf rather than nanmax/nanmin: an all-NaN column would warn, and the
    # result is masked by all_scored anyway.
    spread = np.where(
        all_scored,
        np.where(np.isnan(panel), -np.inf, panel).max(axis=0)
        - np.where(np.isnan(panel), np.inf, panel).min(axis=0),
        0.0,
    )

    definitions = {
        BAND_WRONG: (
            eligible & has_score & (predicted_human != is_human),
            lambda pos: -abs(scores[pos] - threshold),
        ),
        BAND_UNCERTAIN: (
            eligible & has_score & (distance <= uncertain_band),
            lambda pos: distance[pos],
        ),
        BAND_CONFIDENT: (
            eligible & has_score & (predicted_human == is_human)
            & (distance >= confident_band),
            lambda pos: -distance[pos],
        ),
        BAND_DISAGREE: (
            eligible & all_scored & split,
            lambda pos: -spread[pos],
        ),
        BAND_ALL: (eligible, lambda pos: pos),
    }

    if BAND_DISAGREE in bands and not vs_scores:
        raise KeyError(
            f"--select {BAND_DISAGREE} needs at least one other judge to "
            "disagree with. Pass --vs-cols with the score columns of the other "
            "judges, e.g. --vs-cols llm_judge_score_sonnet5_v3 "
            "llm_judge_score_qwen3_14b_v3."
        )
    if not has_truth and (BAND_WRONG in bands or BAND_CONFIDENT in bands):
        raise KeyError(
            f"The file has no {GENERATED_BY_COL!r} column, so 'wrong' and "
            "'confident' have nothing to be right or wrong about. Use --select "
            f"{BAND_UNCERTAIN} or --select {BAND_ALL}."
        )

    labels = strata(df)
    taken: set[int] = set()
    band_of: dict[int, str] = {}
    for band in bands:
        mask, key = definitions[band]
        candidates = [int(p) for p in np.flatnonzero(mask) if int(p) not in taken]
        candidates.sort(key=key)
        for pos in take_round_robin(candidates, labels, per_band):
            taken.add(pos)
            band_of[pos] = band

    positions = np.array(sorted(taken), dtype=int)
    return Selection(positions=positions, band_of=band_of)


# ---------------------------------------------------------------------------
# What the cues add up to
# ---------------------------------------------------------------------------


def _as_cue_list(value) -> list[str] | None:
    """A stored cell back as a list of cue names, or None if it holds none."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v) for v in value]
    return None


def _as_reason(value) -> str | None:
    """A stored cell back as a rationale, or None if it holds none."""
    if value is None or not isinstance(value, str) or not value.strip():
        return None
    return value


# What a model writes when it means "nothing to add here". The schema asks for
# an empty string and most of them comply, but enough answer in words that not
# filtering these would fill other_evidence.csv with rows saying "none".
_EMPTY_OTHER = {
    "", "-", "--", "n/a", "na", "none", "none.", "nothing", "nothing.",
    "no other evidence", "no other evidence.", "not applicable", "null",
}


def _as_other(value) -> str | None:
    """A stored cell back as free-text evidence, or None if it holds none."""
    if value is None or not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text or text.lower().strip(".") in {e.strip(".") for e in _EMPTY_OTHER}:
        return None
    return text


def cue_report(
    df: pd.DataFrame,
    columns: tuple[str, str, str, str],
    band_of: dict[int, str],
    out: Tee,
    examples: int = 2,
) -> list[dict]:
    """Print, and return, how often each cue was cited for each class.

    This is the point of the closed vocabulary: one line per cue, human rows
    against machine rows. A cue whose two percentages are close is one the judge
    sees everywhere and cannot be separating on, however much of the prompt is
    devoted to it -- which is as informative as a cue that splits cleanly.
    """
    reason_col, human_col, machine_col, _other_col = columns
    reasons = [_as_reason(v) for v in df[reason_col]]
    explained = np.array([r is not None for r in reasons])
    rows: list[dict] = []

    if not explained.any():
        out.print("No explanations in the file yet — nothing to summarise.")
        return rows

    generators = (
        df[GENERATED_BY_COL].astype(str).to_numpy()
        if GENERATED_BY_COL in df.columns
        else np.array(["all"] * len(df))
    )
    is_human = explained & (generators == "human")
    is_machine = explained & (generators != "human")
    n_human, n_machine = int(is_human.sum()), int(is_machine.sum())

    out.print("=" * 78)
    out.print("Cue frequencies")
    out.print(f"  explained rows : {int(explained.sum())}  "
              f"(human {n_human}, machine {n_machine})")
    if band_of:
        tally: dict[str, int] = {}
        for band in band_of.values():
            tally[band] = tally.get(band, 0) + 1
        out.print("  bands          : "
                  + ", ".join(f"{b}={n}" for b, n in sorted(tally.items())))
    else:
        # Re-summarising a file explained by an earlier run: the rows are there,
        # the record of which band each came from is not.
        out.print("  bands          : (not recorded — summarising a previous run)")
    out.print("-" * 78)
    out.print(f"{'cue':<22}{'side':<9}{'human':>14}{'machine':>14}{'lift':>10}")
    out.print("-" * 78)

    for side, vocabulary, column in (
        ("human", HUMAN_CUES, human_col),
        ("machine", MACHINE_CUES, machine_col),
    ):
        cited = [_as_cue_list(v) or [] for v in df[column]]
        for cue in vocabulary:
            flags = np.array([cue in c for c in cited])
            h = int((flags & is_human).sum())
            m = int((flags & is_machine).sum())
            h_pct = 100.0 * h / n_human if n_human else float("nan")
            m_pct = 100.0 * m / n_machine if n_machine else float("nan")
            # Machine minus human: positive means the cue marks machine rows,
            # negative that it marks human rows. Sign, not magnitude, is what
            # says whether the judge is using the cue the way the prompt says.
            lift = m_pct - h_pct
            out.print(
                f"{cue:<22}{side:<9}{h:>6} ({h_pct:>4.0f}%){m:>6} ({m_pct:>4.0f}%)"
                f"{lift:>+9.0f}%"
            )
            rows.append({
                "cue": cue, "side": side,
                "human_n": h, "human_pct": round(h_pct, 2),
                "machine_n": m, "machine_pct": round(m_pct, 2),
                "lift_pct": round(lift, 2),
            })

    # A row with no cues at all is the judge declining to name evidence, which
    # is worth seeing separately from a row it had plenty to say about.
    for label, mask, total in (("human", is_human, n_human), ("machine", is_machine, n_machine)):
        if not total:
            continue
        empty = sum(
            1 for i in np.flatnonzero(mask)
            if not (_as_cue_list(df[human_col].iloc[i]) or [])
            and not (_as_cue_list(df[machine_col].iloc[i]) or [])
        )
        out.print(f"  no cues at all, {label:<8}: {empty} of {total}")

    if examples:
        out.print()
        out.print("Example rationales")
        shown: dict[str, int] = {}
        for pos in sorted(band_of, key=lambda p: (band_of[p], p)):
            band = band_of[pos]
            if shown.get(band, 0) >= examples or reasons[pos] is None:
                continue
            shown[band] = shown.get(band, 0) + 1
            cues = (_as_cue_list(df[human_col].iloc[pos]) or []) + \
                   (_as_cue_list(df[machine_col].iloc[pos]) or [])
            out.print(f"  [{band}] row {pos}  {generators[pos]}  cues: "
                      f"{', '.join(cues) if cues else '(none)'}")
            out.print(f"      {reasons[pos]}")
    out.print("=" * 78)
    return rows


def other_report(
    df: pd.DataFrame,
    columns: tuple[str, str, str, str],
    band_of: dict[int, str],
    out: Tee,
    examples: int = 8,
) -> list[dict]:
    """Print how often the vocabulary fell short, and return every instance.

    The cue table can only ever report the eleven things the prompt already
    thought of. This is the other half of the answer: how often the judge said
    it saw something the vocabulary has no name for, split by class, and what
    it said. A rate near zero means the vocabulary covers the evidence. A rate
    that is high, or high on one class only, means the counts above are
    describing a fraction of what the judge is actually reading -- and the rows
    returned here, dumped to other_evidence.csv, are where the missing cue is.

    Nothing is aggregated into percentages beyond the two rates: these are free
    text, so the reading is done by eye, and the CSV is the artefact.
    """
    reason_col, _human_col, _machine_col, other_col = columns
    if other_col not in df.columns:
        out.print()
        out.print("Other evidence: column not in the file (explained by an "
                  "earlier run, before it existed).")
        return []

    others = [_as_other(v) for v in df[other_col]]
    explained = np.array([_as_reason(v) is not None for v in df[reason_col]])
    generators = (
        df[GENERATED_BY_COL].astype(str).to_numpy()
        if GENERATED_BY_COL in df.columns
        else np.array(["all"] * len(df))
    )
    domains = (
        df[DOMAIN_COL].astype(str).to_numpy()
        if DOMAIN_COL in df.columns
        else np.array(["all"] * len(df))
    )
    has_other = np.array([o is not None for o in others]) & explained

    out.print()
    out.print("=" * 78)
    out.print("Other evidence — what the cue vocabulary had no name for")
    for label, mask in (
        ("human", explained & (generators == "human")),
        ("machine", explained & (generators != "human")),
    ):
        total = int(mask.sum())
        if not total:
            continue
        hit = int((mask & has_other).sum())
        out.print(f"  {label:<8}: {hit} of {total} rows "
                  f"({100.0 * hit / total:.0f}%)")

    rows = [
        {
            "row": int(pos),
            "band": band_of.get(int(pos), ""),
            "domain": domains[pos],
            "generated_by": generators[pos],
            "other_evidence": others[pos],
            "rationale": _as_reason(df[reason_col].iloc[pos]) or "",
        }
        for pos in np.flatnonzero(has_other)
    ]

    if rows and examples:
        out.print("-" * 78)
        for row in rows[:examples]:
            out.print(f"  row {row['row']}  {row['domain']} / "
                      f"{row['generated_by']}  [{row['band'] or '?'}]")
            out.print(f"      {row['other_evidence']}")
        if len(rows) > examples:
            out.print(f"  ... and {len(rows) - examples} more, in "
                      "other_evidence.csv")
    elif not rows:
        out.print("  nothing reported — the vocabulary covered every explained row.")
    out.print("=" * 78)
    return rows


def save_report(
    rows: list[dict],
    out: Tee,
    report_dir: Path,
    stem: str,
    other_rows: list[dict] | None = None,
) -> Path:
    """Write the log and the cue table where the analyse scripts put theirs."""
    directory = report_dir / stem
    directory.mkdir(parents=True, exist_ok=True)
    out.save(directory / "cue_summary.log")
    if rows:
        pd.DataFrame(rows).to_csv(directory / "cue_frequencies.csv", index=False)
    if other_rows:
        pd.DataFrame(other_rows).to_csv(directory / "other_evidence.csv", index=False)
    return directory


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def reason_columns(spec: JudgeModel, prompt_id: str) -> tuple[str, str, str, str]:
    """The four columns one explaining judge writes.

    In the order they are unpacked everywhere below: the rationale, the two cue
    lists, and the free-text evidence the vocabulary had no name for.
    """
    stem = _identifier(f"{spec.column}_{prompt_id}")
    return (
        f"{REASON_PREFIX}{stem}",
        f"{HUMAN_CUES_PREFIX}{stem}",
        f"{MACHINE_CUES_PREFIX}{stem}",
        f"{OTHER_PREFIX}{stem}",
    )


def parse_args() -> argparse.Namespace:
    if "--list-models" in sys.argv:
        print_models()
        raise SystemExit(0)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquet", type=Path, help="Parquet to add rationales to.")
    ap.add_argument("--list-models", action="store_true",
                    help="Print the judge registry and exit.")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output path (default: overwrite the input in place).")
    ap.add_argument("--model", default=DEFAULT_MODEL, metavar="ALIAS|REPO_ID",
                    help=f"Model that does the explaining (default: {DEFAULT_MODEL}). "
                         "Same registry and repo-id support as the scoring script.")
    ap.add_argument("--column-name", default=None,
                    help="Override the column stem the rationales are written under.")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help=f"Prompt version whose evidence half is used (default: "
                         f"{DEFAULT_PROMPT}). Use the one the scores came from: "
                         "the cue vocabulary is that version's own evidence list.")
    ap.add_argument("--score-col", default=None,
                    help="Score column the bands are read off. Default: the column "
                         "this model and prompt would have written. Set it when the "
                         "explaining model is not the model that scored, e.g. "
                         "--model qwen3-14b --score-col llm_judge_score_gpt56luna_v3.")
    ap.add_argument("--select", nargs="+", default=list(DEFAULT_BANDS), choices=BANDS,
                    metavar="BAND",
                    help=f"Bands to draw from, in order (default: "
                         f"{' '.join(DEFAULT_BANDS)}). Choices: {', '.join(BANDS)}. "
                         f"'{BAND_DISAGREE}' needs --vs-cols.")
    ap.add_argument("--vs-cols", nargs="+", default=None, metavar="COLUMN",
                    help="Score columns of the OTHER judges, for the "
                         f"'{BAND_DISAGREE}' band: a row is in that band when "
                         "this judge's verdict and one of theirs differ, widest "
                         "score gap first. e.g. --vs-cols "
                         "llm_judge_score_sonnet5_v3 llm_judge_score_qwen3_14b_v3. "
                         "Only rows scored in every column named are eligible.")
    ap.add_argument("--per-band", type=int, default=DEFAULT_PER_BAND,
                    help=f"Rows per band (default: {DEFAULT_PER_BAND}), drawn "
                         "round-robin across domain x generated_by so every "
                         "domain and both classes are represented.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"score >= this counts as a 'human' verdict when deciding "
                         f"which rows are wrong (default: {DEFAULT_THRESHOLD}). Use the "
                         "operating point the analyse script reported, if you have one.")
    ap.add_argument("--uncertain-band", type=float, default=DEFAULT_UNCERTAIN_BAND,
                    help=f"Half-width around 0.5 for the 'uncertain' band "
                         f"(default: {DEFAULT_UNCERTAIN_BAND}).")
    ap.add_argument("--confident-band", type=float, default=DEFAULT_CONFIDENT_BAND,
                    help=f"Minimum distance from 0.5 for the 'confident' band "
                         f"(default: {DEFAULT_CONFIDENT_BAND}).")
    ap.add_argument("--text-col", default=DEFAULT_TEXT_COL,
                    help=f"Column holding the text (default: {DEFAULT_TEXT_COL!r}). "
                         "It is the only column the model ever sees.")
    ap.add_argument("--generated-by", nargs="+", default=None, metavar="VALUE",
                    help="Restrict to these generators before the bands are formed. "
                         f"Pass {ALL_GENERATORS!r} for the default (all of them).")
    ap.add_argument("--max-chars", type=int, default=None,
                    help="Truncate each text to this many characters before sending it.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Rows in flight for an API model (default: 4).")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help=f"Rows per generate call for a local model (default: "
                         f"{DEFAULT_BATCH_SIZE}).")
    ap.add_argument("--device", default=DEFAULT_DEVICE,
                    help=f"Device for a local model (default: {DEFAULT_DEVICE}). "
                         "Say 'cpu' here; to choose a card use --gpu.")
    ap.add_argument("--gpu", default=None,
                    help="Value for CUDA_VISIBLE_DEVICES (which GPU(s) to use, "
                         "e.g. '0' or '0,1'). Default: leave unset so all visible "
                         "GPUs are used.")
    ap.add_argument("--cpu-threads", type=int, default=None,
                    help="Limit CPU threads used for inference (sets "
                         "OMP_NUM_THREADS / MKL_NUM_THREADS and torch's intra-op "
                         "pool). Default: leave unset.")
    ap.add_argument("--dtype", default=DEFAULT_DTYPE,
                    help=f"Weight dtype for a local model (default: {DEFAULT_DTYPE}).")
    ap.add_argument("--max-doc-tokens", type=int, default=DEFAULT_MAX_DOC_TOKENS,
                    help=f"Document token budget for a local model (default: "
                         f"{DEFAULT_MAX_DOC_TOKENS}); 0 sends it whole.")
    ap.add_argument("--retries", type=int, default=RETRIES,
                    help=f"Retries per row before it is given up on (default: {RETRIES}).")
    ap.add_argument("--save-every", type=int, default=SAVE_EVERY,
                    help=f"Write the Parquet every N explained rows (default: {SAVE_EVERY}).")
    ap.add_argument("--report-dir", type=Path, default=REPORT_DIR,
                    help=f"Where the cue summary is written (default: {REPORT_DIR}).")
    ap.add_argument("--recompute", action="store_true",
                    help="Re-explain rows that already have a rationale.")
    ap.add_argument("--summary-only", action="store_true",
                    help="Do not call any model: just re-print and re-save the cue "
                         "table from the rationales already in the file.")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the confirmation prompt.")
    return ap.parse_args()


def print_findings(
    path: Path, df: pd.DataFrame, spec: JudgeModel, prompt_id: str, score_col: str,
    columns: tuple[str, str, str, str], selection: Selection, positions: np.ndarray,
    args, vs_cols: list[str] | None = None,
) -> None:
    reason_col = columns[0]
    scored = int(sum(_as_score(v) is not None for v in df[score_col])) \
        if score_col in df.columns else 0
    print("=" * 78)
    print("Parquet")
    print(f"  file           : {path}")
    print(f"  rows           : {len(df)}")
    print(f"  score column   : {score_col}  ({scored} scored)")
    for column in vs_cols or []:
        n = int(sum(_as_score(v) is not None for v in df[column]))
        print(f"  compared with  : {column}  ({n} scored)")
    print("Explainer")
    print(f"  model          : {spec.key}  ({spec.model})")
    print(f"  prompt         : {prompt_id}  (evidence half; no score is asked for)")
    if spec.is_local:
        print(f"  device         : {args.device} ({dtype_for(spec, args.dtype)})"
              f", batch {args.batch_size}")
        print(f"  gpu            : "
              f"{args.gpu if args.gpu is not None else 'all visible (unset)'}"
              f"   cpu threads: "
              f"{args.cpu_threads if args.cpu_threads is not None else 'unset'}")
    print(f"  columns        : {', '.join(columns)}")
    print("Selection")
    for band, count in sorted(selection.counts().items()):
        print(f"    {band:<12}: {count}")
    print(f"  selected       : {len(selection.positions)}")
    print(f"  already done   : {len(selection.positions) - len(positions)}")
    print(f"  to explain now : {len(positions)}")
    print("=" * 78)


def main() -> None:
    args = parse_args()
    # Before blocking_reason, which imports torch to ask about CUDA.
    apply_runtime_limits(args)
    out_path = args.out or args.parquet
    spec = resolve_model(args.model, args.column_name)
    prompt_id = args.prompt
    columns = reason_columns(spec, prompt_id)
    reason_col, human_col, machine_col, other_col = columns
    score_col = args.score_col or score_column(spec, prompt_id, MODE_PARSED)
    stem = f"{args.parquet.stem}+{_identifier(f'{spec.column}_{prompt_id}')}"

    df = load_parquet(args.parquet)

    if args.summary_only:
        # Nothing is called and nothing is written to the Parquet; this only
        # re-reads what previous runs stored.
        out = Tee()
        rows = cue_report(df, columns, {}, out)
        other_rows = other_report(df, columns, {}, out)
        directory = save_report(rows, out, args.report_dir, stem, other_rows)
        print(f"\nReport → {directory}")
        return

    blocked = blocking_reason(spec, args.device)
    if blocked:
        raise SystemExit(f"Cannot run explainer {spec.key!r}: {blocked}.")

    eligible = check_parquet(df, args.text_col, args.parquet)
    allowed = resolve_generated_by(args.generated_by)
    eligible = apply_generated_by(
        df, eligible, allowed, args.generated_by is not None, args.parquet
    )

    if score_col not in df.columns and set(args.select) - {BAND_ALL}:
        raise KeyError(
            f"{args.parquet} has no score column {score_col!r}, so the bands "
            "have nothing to select on. Score the file first, pass --score-col, "
            f"or use --select {BAND_ALL}."
        )
    # NaN for "no score here", which is what keeps an unscored row out of every
    # band that is defined by a score without needing a second mask.
    stored = (
        [_as_score(v) for v in df[score_col]] if score_col in df.columns
        else [None] * len(df)
    )
    scores = np.array([np.nan if s is None else float(s) for s in stored], dtype=float)

    # The other judges' columns, read the same way, for the `disagree` band. A
    # named column that is not in the file is a typo or an unscored corpus, and
    # either way silently comparing against nothing would produce an empty band
    # that looks like agreement.
    vs_cols = list(args.vs_cols or [])
    missing = [c for c in vs_cols if c not in df.columns]
    if missing:
        available = sorted(c for c in df.columns if c.startswith("llm_judge_score_"))
        raise KeyError(
            f"{args.parquet} has no score column(s) {', '.join(missing)}. "
            f"Score columns present: {', '.join(available) or '(none)'}."
        )
    vs_scores = {
        column: np.array(
            [np.nan if _as_score(v) is None else float(_as_score(v))
             for v in df[column]],
            dtype=float,
        )
        for column in vs_cols
    }

    selection = select_rows(
        df, eligible, scores, args.select, args.per_band, args.threshold,
        args.uncertain_band, args.confident_band, vs_scores,
    )
    if len(selection.positions) == 0:
        print("No rows matched the selected bands — nothing to do.")
        return

    done = (
        np.array([_as_reason(v) is not None for v in df[reason_col]])
        if reason_col in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    positions = (
        selection.positions if args.recompute
        else np.array([p for p in selection.positions if not done[p]], dtype=int)
    )

    print_findings(args.parquet, df, spec, prompt_id, score_col, columns,
                   selection, positions, args, vs_cols)

    if len(positions) == 0:
        print("Every selected row already has a rationale — nothing to do.")
        print("(Pass --recompute to ask again, or --summary-only for the table.)")
        return

    print()
    if spec.is_local:
        print(f"NOTE  {len(positions)} row(s) will be explained on {args.device}.")
    else:
        print(f"NOTE  this is a billed API call: {len(positions)} row(s) will be sent.")
    if not args.yes:
        answer = input(f"Proceed and write to {out_path}? Type 'yes': ").strip().lower()
        if answer != "yes":
            print("Aborted — no changes made.")
            return

    explainer = build_explainer(spec, prompt_id, args)

    reasons: list = list(df[reason_col]) if reason_col in df.columns else [None] * len(df)
    human_cues: list = list(df[human_col]) if human_col in df.columns else [None] * len(df)
    machine_cues: list = list(df[machine_col]) if machine_col in df.columns else [None] * len(df)
    others: list = list(df[other_col]) if other_col in df.columns else [None] * len(df)

    texts = {
        int(pos): (
            str(df.iloc[pos][args.text_col])[: args.max_chars]
            if args.max_chars else str(df.iloc[pos][args.text_col])
        )
        for pos in positions
    }

    failures: list[tuple[int, str]] = []
    counters = {"done": 0}
    save_lock = threading.Lock()
    started = time.perf_counter()

    def write_out() -> None:
        df[reason_col] = pd.Series(reasons, index=df.index, dtype="object")
        df[human_col] = pd.Series(human_cues, index=df.index, dtype="object")
        df[machine_col] = pd.Series(machine_cues, index=df.index, dtype="object")
        df[other_col] = pd.Series(others, index=df.index, dtype="object")
        save_parquet(df, out_path)

    def record(result: RowExplanation) -> None:
        if result.explanation is not None:
            reasons[result.position] = result.explanation.rationale
            human_cues[result.position] = list(result.explanation.human_cues)
            machine_cues[result.position] = list(result.explanation.machine_cues)
            others[result.position] = result.explanation.other_evidence
        if result.error is not None:
            failures.append((result.position, result.error))
            tqdm.write(f"  row {result.position} failed: {result.error}")
        counters["done"] += 1
        with save_lock:
            if counters["done"] % args.save_every == 0:
                write_out()

    order = [int(p) for p in positions]
    try:
        if spec.is_local:
            batches = [
                order[i : i + max(1, args.batch_size)]
                for i in range(0, len(order), max(1, args.batch_size))
            ]
            with tqdm(total=len(order), desc=f"Explaining ({spec.key})", unit="row") as bar:
                for batch in batches:
                    for result in explain_batch(explainer, batch, texts, args.retries):
                        record(result)
                    bar.update(len(batch))
        else:
            pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
            try:
                futures = [
                    pool.submit(explain_row, explainer, pos, texts[pos], args.retries)
                    for pos in order
                ]
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"Explaining ({spec.key})", unit="row"):
                    record(future.result())
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
    except KeyboardInterrupt:
        print("\nInterrupted — writing the rationales obtained so far.")

    write_out()

    seconds = time.perf_counter() - started
    print()
    print("=" * 78)
    print("Rationales written")
    print(f"  output  : {out_path}")
    print(f"  columns : {', '.join(columns)}")
    print(f"  done    : {counters['done'] - len(failures)} rows in {seconds:.1f}s")
    print(f"  failed  : {len(failures)}")
    if failures:
        by_message: dict[str, int] = {}
        for _, message in failures:
            by_message[message] = by_message.get(message, 0) + 1
        for message, count in sorted(by_message.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>5}x  {message[:100]}")
    unknown = getattr(explainer, "unknown_cues", {})
    if unknown:
        print("  cue labels outside the vocabulary, dropped from the counts:")
        for label, count in sorted(unknown.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {count:>5}x  {label!r}")
    print("=" * 78)
    print()

    out = Tee()
    rows = cue_report(df, columns, selection.band_of, out)
    other_rows = other_report(df, columns, selection.band_of, out)
    directory = save_report(rows, out, args.report_dir, stem, other_rows)
    print(f"\nReport → {directory}")


if __name__ == "__main__":
    main()

"""LLM-as-a-judge baseline: ask an LLM whether each text is human-written.

Reads a Parquet, walks its rows, and for each one sends *only* the ``text``
column to a judge model that returns P(human-written) in [0, 1]. The score
lands in a new ``llm_judge_score_{model}_{prompt}`` column; nothing else in the
file is touched, so this sits next to ``binoculars_score`` as a second
zero-training baseline and can be compared with it on the same threshold
sweeps.

The judge is whatever ``--model`` names: an API model (the default), or a
Hugging Face model held on this machine's GPU. Everything else about the run --
which rows, which prompt, the resume behaviour, the summary -- is the same
either way.

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
for that model. It also takes a Hugging Face repo id directly -- ``--model
Qwen/Qwen3-14B`` -- for a model that has no registry entry; ``--column-name``
overrides what that is called on disk. ``--list-models`` prints the table.
``--prompt`` picks the system prompt from ``PROMPTS`` (default ``v3``); see the
note above ``PROMPTS`` for what changed between the versions and why.

A LOCAL JUDGE, AND WHY IT SCORES DIFFERENTLY
--------------------------------------------
``--model qwen3-14b`` (or any other local entry) loads the model with
transformers and scores on the GPU. Nothing else changes: same corpus, same
prompt, same column convention, same resume-and-checkpoint behaviour. Rows are
batched (``--batch-size``) instead of being sent concurrently (``--workers``),
and a batch that runs out of memory is retried one row at a time rather than
lost.

``--gpu`` and ``--cpu-threads`` mean what they mean in add_embeddings.py and
run_binox_pairs.py: which card this run is pinned to, and a cap on the CPU
pools, so a judge run and an embedding run can share a multi-GPU box without
fighting over it. Both are applied before torch is imported. ``--gpu`` hides
the other cards, so the one it names is ``cuda:0`` inside the process and
``--device`` stays at its default; ``--device`` is only there to say ``cpu``.

``--mode`` picks how the score is obtained:

``parsed`` (the default, local and API alike)
    The model writes the probability and it is parsed out of the answer. The
    only thing an API can do, and what every ``llm_judge_score_*`` column
    already in the corpora holds -- which is why it is also the default
    locally. A local judge is run first to answer "does a different model see
    the same thing", and that question is only answered by a score on the same
    scale as the one it is being compared against.

``logprob`` (local only, ``--mode logprob``)
    The prompt asks for one word, ``human`` or ``machine``, and the score is
    read off that position's distribution:
    ``P(human) / (P(human) + P(machine))``. One forward pass, nothing sampled,
    deterministic. It is not a cheaper version of ``parsed`` but a better one:
    a written probability piles onto 0.2/0.5/0.8 however firmly the prompt asks
    it not to, and ROC-AUC reads only the ranking, so every tie is discarded
    information. A distribution has no ties.

The mode is part of the column name (``_lp``), because a logprob score and a
parsed score are not on one scale. So is the reason the two prompts differ: in
``logprob`` mode the evidence half of the prompt version is used verbatim and
its answer-format half is replaced, since there is no written answer to format
(see LOGPROB_ANSWER_BLOCK).

WHAT A LOCAL JUDGE IS EXPECTED TO DO, AND WHAT TO WATCH
-------------------------------------------------------
The Binoculars work already found the shape of the risk: a scorer that is much
weaker than the generator stops separating the classes and can invert outright
(``external/Binoculars/model_pairs.py``). A 14-32B judge reading text from a
frontier model is the same axis, so expect it to hold up on
``ghostbuster_gpt35t`` and to drift toward chance on ``ghostbuster_gpt56luna``
and the science corpora. That is a result, not a bug -- but it is only readable
if the API columns stay in the file, so a local run adds a column and never
replaces one.

Two things in the summary say whether the judge is doing what it was asked:
``distinct values`` (a logprob run should have essentially one per row) and the
``answer mass`` line (how much of the distribution sat on the two answer words;
low means the model answered some third thing and the ratio is noise).

    # smallest useful check: does the wiring work at all
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt56luna.parquet \
        --model qwen3-8b --limit 20 --gpu 0 --cpu-threads 16 \
        -o /tmp/judge_smoke.parquet --yes

    # the comparison worth running before any full pass: one easy corpus, one
    # hard one, against the API column already in the file
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt35t.parquet --model qwen3-14b --limit 400 \
        --gpu 0 --cpu-threads 16 --yes
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt56luna.parquet --model qwen3-14b --limit 400 \
        --gpu 0 --cpu-threads 16 --yes

    # (there is no 32B line: qwen3-32b-w4a16 cannot run under transformers --
    # see the note on its registry entry. qwen3-14b is the largest local judge
    # that fits a 40GB card.)

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

With an API judge these are calls rather than GPU work, so several corpora can
run at once. With a local judge they cannot: one process owns the card, and the
corpora queue up behind each other exactly as in run_binox_pairs.py. Either way,
never two processes on the SAME file -- adding a column is a read-modify-write
of the whole Parquet and the second writer would silently drop the first one's
column.

IS THE PROMPT ANY GOOD? -- A/B IT BEFORE THE FULL RUN
-----------------------------------------------------
Without ``--limit`` every eligible row in the file is judged -- all domains, both
classes. ``--limit N`` cuts that to N rows drawn round-robin across ``domain`` x
``generated_by``, so a corpus stored human-block-then-machine-block still yields
both classes, a three-domain corpus yields a third of the sample from each
domain, and the summary's mean-per-group / accuracy / ROC-AUC lines describe the
file rather than whichever domain it happens to start with. Two cheap runs on the
same file put the prompts side by side in two columns:

    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt56luna.parquet --prompt v2 --limit 200 --yes
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt56luna.parquet --prompt v3 --limit 200 --yes

Compare the two ROC-AUCs, and the per-generator means: a prompt that separates
the classes well but puts BOTH means on one side of 0.5 is mis-calibrated rather
than blind, and is fixed by moving the threshold, not the prompt.

The prompt says nothing about how a corpus was preprocessed, which is what
makes the surface artifact measurable rather than assumed: the same ``v3`` run
against ``ghostbuster_gpt35t.parquet`` and ``ghostbuster_gpt35ts.parquet`` is
one judge on one set of documents, once before standardisation and once after,
with nothing but the text differing between the two. Whatever ROC-AUC separates
them was riding on punctuation.

    # smoke test into a scratch copy, leaving the corpus untouched
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt56luna.parquet --limit 20 -o /tmp/judge_smoke.parquet

    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
      data/parquet/ghostbuster_gpt35t.parquet \
      --model qwen3-14b --cpu-threads 16 --yes --gpu 1

    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
      data/parquet/ghostbuster_gpt54mini.parquet \
      --model qwen3-14b --cpu-threads 16 --yes --gpu 2

    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
      data/parquet/ghostbuster_gpt56luna.parquet \
      --model qwen3-14b --cpu-threads 16 --yes --gpu 3
    
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/science_v3_gpt56luna_0811A.parquet\
      --model qwen3-14b --cpu-threads 16 --yes --gpu 4

    ------------------------------------------------------------------------

    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
      data/parquet/ghostbuster_gpt35t.parquet \
      --model sonnet5 --yes
  
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt35ts.parquet \
        --model sonnet5--yes
    
    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt54mini.parquet \
        --model sonnet5 --yes

    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/ghostbuster_gpt56luna.parquet \
        --model sonnet5 --yes

    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/science_v3_gpt56luna_0811A.parquet \
        --model sonnet5 --yes
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

from epai.utils.parquet_utils import load_parquet, save_parquet

load_dotenv()

LOCAL_PROVIDER = "local"

_GIB = 1024**3


@dataclass(frozen=True)
class JudgeModel:
    """One judge model, and the column stem its scores belong in.

    Modelled on ``external/Binoculars/model_pairs.py``: the registry entry, not
    the call site, decides what a run is called and where its numbers land, so
    two judges can never write into one column. ``column`` is the identifier-safe
    stem -- ``gpt56luna``, ``qwen3_14b`` -- that :func:`score_column` combines
    with the prompt version and the scoring mode.

    ``params`` and ``quant`` exist only to print a weights estimate before a
    multi-GB download starts; they are ignored for the API providers, which have
    no local footprint.
    """

    key: str  # --model value
    provider: str  # "openai" | "anthropic" | "gemini" | "local"
    model: str  # API model id, or Hugging Face repo id when local
    column: str  # score column stem, so two judges never overwrite each other
    params: float | None = None  # parameters in billions (local only)
    quant: str | None = None  # None => bf16 weights, else the checkpoint's format
    resident_gib: float | None = None  # measured size, when it is known
    note: str = ""

    @property
    def is_local(self) -> bool:
        return self.provider == LOCAL_PROVIDER

    @property
    def weights_gib(self) -> float | None:
        """Resident weights, or None for a model that never lands on this box.

        `params x bytes-per-param` is only an estimate, and for a quantised
        checkpoint a bad one: the scales, the zero points and whatever layers
        the quantiser skipped (lm_head, usually) are all still full precision,
        which on a 32B model is another few GiB. `resident_gib` carries the
        real figure where it has been checked, because the number's whole job
        is answering "does this fit the card".
        """
        if self.resident_gib is not None:
            return self.resident_gib
        if self.params is None:
            return None
        bytes_per_param = 0.5 if self.quant else 2.0
        return self.params * 1e9 * bytes_per_param / _GIB


# Judge models, keyed by the short alias used everywhere else in the pipeline.
# The alias also names the output column, so it has to stay identifier-safe --
# hence `gpt56luna` rather than the API id `gpt-5.6-luna`, matching the slug the
# generation side already writes into filenames and `generated_by` labels.
#
# The local entries are sized for one 40GB A100 with room left for activations.
# `--model` also takes a raw Hugging Face repo id for anything not listed, so
# this is a shortlist of the ones worth reporting, not a limit on what can run:
# see resolve_model.
MODELS: dict[str, JudgeModel] = {
    "gpt56luna": JudgeModel(
        key="gpt56luna",
        provider="openai",
        model="gpt-5.6-luna",
        column="gpt56luna",
        note="The API baseline every existing llm_judge_score_* column came "
             "from. Keep it as the control: it is what a local judge is "
             "measured against.",
    ),
    "gpt54mini": JudgeModel(
        key="gpt54mini",
        provider="openai",
        model="gpt-5.4-mini",
        column="gpt54mini",
    ),
    "sonnet5": JudgeModel(
        key="sonnet5",
        provider="anthropic",
        model="claude-sonnet-5",
        column="sonnet5",
        note="The cross-vendor control. Frontier-tier and from a third vendor, "
             "so the only thing that changes against the gpt56luna column is "
             "who is reading the document. Deliberately not haiku45 for this "
             "job: a smaller, cheaper judge confounds vendor with capability, "
             "and capability is the axis the Binoculars work already showed "
             "moves these scores (external/Binoculars/model_pairs.py).",
    ),
    "haiku45": JudgeModel(
        key="haiku45",
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        column="haiku45",
        note="Not the vendor control -- see sonnet5. Useful as a point on the "
             "capability axis instead: the cheap API judge to set beside the "
             "local 8B/14B/32B ladder.",
    ),
    "gptoss-120b": JudgeModel(
        key="gptoss-120b",
        provider="genai4science",
        model="gpt-oss:120b",
        column="gptoss_120b",
        note="GenAI4Science (HUN-REN SZTAKI) open-weight endpoint. Needs "
             "GENAI4SCIENCE_PERFORMANCE_API_KEY (or GENAI4SCIENCE_API_KEY for "
             "the shared endpoint). The largest open-weight judge reachable "
             "there and the top open-weight rung of the capability ladder; the "
             "same models are Track-1/2 generators, so a judge and a generator "
             "here can be the same weights.",
    ),
    "llama33-70b": JudgeModel(
        key="llama33-70b",
        provider="genai4science",
        model="llama3.3:70b",
        column="llama33_70b",
        note="GenAI4Science open-weight endpoint. A second open-weight family "
             "beside gpt-oss:120b, one rung below it.",
    ),
    "qwen38-27b": JudgeModel(
        key="qwen38-27b",
        provider="genai4science",
        model="qwen3.8:27b",
        column="qwen38_27b",
        note="GenAI4Science open-weight endpoint. A third open-weight family "
             "beside gpt-oss:120b and llama3.3:70b, a rung below llama3.3:70b. "
             "Unlike llama3.3:70b (downgraded, off dedicated GPUs) it holds a "
             "dedicated GPU and actually serves the perf endpoint's 4 "
             "concurrent calls in parallel.",
    ),
    "qwen3-8b": JudgeModel(
        key="qwen3-8b",
        provider=LOCAL_PROVIDER,
        model="Qwen/Qwen3-8B",
        column="qwen3_8b",
        params=8.2,
        note="Smoke test: small enough to load in a couple of minutes, so the "
             "wiring can be checked before a 32B download is started.",
    ),
    "qwen3-14b": JudgeModel(
        key="qwen3-14b",
        provider=LOCAL_PROVIDER,
        model="Qwen/Qwen3-14B",
        column="qwen3_14b",
        params=14.8,
        note="Default local judge: bf16 weights fit a 40GB card with room to "
             "spare and need no quantisation kernels, so plain transformers "
             "runs it.",
    ),
    "qwen3-32b-w4a16": JudgeModel(
        key="qwen3-32b-w4a16",
        provider=LOCAL_PROVIDER,
        model="RedHatAI/Qwen3-32B-quantized.w4a16",
        column="qwen3_32b_w4a16",
        params=32.8,
        quant="compressed-tensors",
        resident_gib=19.2,
        note="DOES NOT RUN under transformers -- kept as the record of why, and "
             "as the entry to point a vLLM server at. ~19 GiB of int4 on disk "
             "(group 128, symmetric, lm_head left in fp16), but transformers "
             "has no int4 kernel for this format: compressed-tensors 0.18 "
             "dropped CompressedLinear, made `run_compressed` a no-op, and now "
             "registers a forward pre-hook that decompresses the whole model to "
             "fp16 on the first forward pass (model_compressor.add_decompress_hook). "
             "For 32.8B that is ~65 GiB on a 40 GiB card, so the format buys "
             "download size and nothing else here. Observed 2026-08-27 on "
             "transformers 4.57.6 / compressed-tensors 0.18.0: the hook itself "
             "raised KeyError('weight_packed') on every row. Use qwen3-14b for "
             "a local judge, or serve this checkpoint with vLLM, which has the "
             "marlin/machete kernels the format was built for. Not the Qwen AWQ "
             "build of the same model either: AutoAWQ was deprecated in 2025 "
             "and its last tested stack (torch 2.6, transformers 4.51) is far "
             "behind this project's.",
    ),
    "gemma3-12b": JudgeModel(
        key="gemma3-12b",
        provider=LOCAL_PROVIDER,
        model="google/gemma-3-12b-it",
        column="gemma3_12b",
        params=12.2,
        note="A second family, for when a result needs to be shown not to be "
             "an artefact of one model lineage. Gated: needs HF_TOKEN with the "
             "licence accepted.",
    ),
}

DEFAULT_MODEL = "gpt56luna"
DEFAULT_TEXT_COL = "text"
COL_PREFIX = "llm_judge_score_"

# How the score is obtained, which is a third axis of comparability beside the
# model and the prompt -- hence part of the column name (see score_column).
#
#   parsed   the model writes the probability and it is parsed out of the
#            answer. The only option through an API, and what every existing
#            column holds.
#   logprob  the model answers with one word and the score is read off the token
#            distribution at that position: P(human) / (P(human) + P(machine)).
#            Local only, because it needs the logits. One forward pass per row,
#            no generation, and the score is continuous -- which matters more
#            than it sounds: ROC-AUC reads only the ranking, and a written
#            probability piles onto 0.2/0.5/0.8 no matter how firmly the prompt
#            asks it not to. Every tie is AUC thrown away.
MODE_PARSED = "parsed"
MODE_LOGPROB = "logprob"
MODES = (MODE_PARSED, MODE_LOGPROB)

# Appended to the column name, so two modes on one model never collide. Parsed
# gets no suffix: it is what the API columns already written are, and renaming
# them would orphan every score in the corpora.
MODE_SUFFIX = {MODE_PARSED: "", MODE_LOGPROB: "_lp"}

# `parsed` everywhere, local judges included. A logprob score is the better
# score, but it is a different score: it has no suffix-free column, it cannot be
# put beside the API columns already in the corpora, and it answers "human" or
# "machine" rather than the two decimals every existing number came from. The
# first thing a local judge is for is the like-for-like comparison against
# `llm_judge_score_gpt56luna_v3`, so that is what it does unless asked otherwise
# (`--mode logprob`, local only).
DEFAULT_MODE = MODE_PARSED

# Local run defaults. The batch is small because the documents are long and the
# weights already own most of the card; --batch-size raises it.
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_DOC_TOKENS = 4096

# Which rows are judged. Default: every row with a non-empty text, human and
# machine alike -- the machine rows are labelled `<model_slug>:<prompt_id>`
# (`gpt56luna:0701A`), so there is no fixed list of generator names that would
# hold across corpora, and a stale one silently judges nothing. `--generated-by`
# names them explicitly when only part of a file is wanted. The judge never sees
# the column; it only decides which rows are worth paying for.
GENERATED_BY_COL = "generated_by"
DEFAULT_GENERATED_BY: list[str] | None = None
ALL_GENERATORS = "all"

# The other axis a sample has to be balanced on. A combined corpus holds several
# domains in one file (`essay`/`reuter`/`wp`, or the science fields) and the
# judge does not behave alike across them, so a limited run that drew from one
# domain would report a number about that domain under the file's name. Not
# every corpus has the column; see `strata`.
DOMAIN_COL = "domain"

# Environment variable each provider needs, checked up front so a long run does
# not die on the first call. GenAI4Science is resolved separately
# (:func:`genai4science_endpoint` — the var name depends on whether a
# performance key is set), so it is not listed here.
_REQUIRED_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

# A served reasoning model (gpt-oss) can wrap its chain of thought in <think>…
# </think> before the answer; strip it so parse_probability reads the verdict,
# not a number from the reasoning.
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)

# Rows scored between two writes of the Parquet.
SAVE_EVERY = 200

# Retries per row before it is given up on, and the base of the exponential
# backoff between them (seconds, jittered).
RETRIES = 3
BACKOFF_BASE = 2.0

# Anthropic wants an explicit cap; the answer is a single number, so this is
# only there to bound a pathological response.
MAX_TOKENS = 256

# Tokens generated by a local judge in `parsed` mode. Enough for "0.37" in any
# tokenisation, plus a short prefix ("Probability: 0.37") that parse_probability
# will still read. Nothing longer is wanted: a model that starts explaining
# itself should be cut off rather than waited for, and the generation stops at
# the end-of-turn token anyway.
LOCAL_MAX_NEW_TOKENS = 16

# The two answers a logprob-mode judge chooses between, with the casings and
# leading-space variants a tokeniser might produce. Only the FIRST token of each
# string is used -- that is the position whose distribution is read -- and the
# two sets must stay disjoint, which LocalJudge asserts at load time.
HUMAN_WORDS = ("human", " human", "Human", " Human", "HUMAN")
MACHINE_WORDS = ("machine", " machine", "Machine", " Machine", "MACHINE")

# Below this, the two answer words hold so little of the distribution that the
# model is answering some third thing and the ratio is noise. Not an error --
# the score is still written -- but counted and reported, because a judge that
# ignores the answer format is the one failure mode a plausible-looking AUC can
# hide.
MIN_ANSWER_MASS = 0.10

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

PROMPTS["v3"] = """You are a forensic linguist estimating whether a document was written by a human or generated by a large language model.

You will be shown one document and nothing else: no title, no author, no source, no metadata, no context. Judge it on its own evidence.

JUDGE RELATIVE TO THE GENRE
Work out what kind of document this is trying to be, then ask whether it reads like a competent human working in that genre or like a model imitating one. A wire report carries no personal voice and a scientific abstract carries no digressions; in those genres that is the convention, not evidence of a machine. The question is always "unusual for this kind of document?", never "unusual in general?".

EVIDENCE THAT TENDS TO INDICATE HUMAN AUTHORSHIP
- Specifics a model could not reconstruct from general knowledge: particular names, numbers, dates, places, results, quoted words -- load-bearing rather than decorative, so that removing one would damage the text.
- Commitment: a definite claim, an unbalanced opinion, an argument that goes somewhere instead of hedging back to neutral.
- Asymmetry: uneven emphasis, a paragraph that runs long because the writer cared about it, a point raised and dropped, a sentence that changes direction mid-way.
- Idiom, slang, humour, irony, in-group shorthand; the habits of a field used the way a practitioner uses them rather than the way they would be explained.
- Errors of language: a misspelling, a wrong-word substitution, an agreement or tense slip, a missing word, a run-on or fragmentary sentence, a construction that does not resolve.

EVIDENCE THAT TENDS TO INDICATE MACHINE AUTHORSHIP
- Fluent but generic vocabulary; an abstraction where a specific would have been natural.
- Even, predictable rhythm: uniform sentence and paragraph lengths, repetitive clause structure.
- Tidy scaffolding: symmetrical structure, explicit enumeration, a closing sentence that restates what came before and adds nothing.
- Hedged, balanced framing that declines to commit: every side acknowledged, none pressed.
- Coverage in place of depth: each obvious aspect of the topic given a proportionate share, nothing pursued further than the rest.
- The connectives, intensifiers and stock transitions of assistant prose, and the reach for a summarising abstraction where the writing could have stopped.

WHAT IS NEVER EVIDENCE
- Length. A short document is harder to judge, not more likely to be machine-written.
- Polish. Careful, edited human writing is fluent and correct.
- Subject matter. Any topic can be written about by either.
- Damage that is not the author's doing: a stranded fragment, broken encoding, lost markup, a document that begins or ends abruptly or stops mid-sentence. Judge the writing, not the state the document reached you in.

CALIBRATION
- Return a probability that the document is HUMAN-written. 1.0 means certainly human, 0.0 means certainly machine, 0.5 means the text carries no usable evidence either way.
- Use the whole range, and give two decimal places. Do not settle on round values such as 0.20, 0.50 or 0.80: if your belief falls between two of them, say so (0.37, 0.62). These scores are ranked against each other, so every tie is information thrown away.
- Reserve values below 0.10 and above 0.90 for documents where the evidence is genuinely strong.
- Do not assume any particular mix of human and machine documents, and do not try to balance your answers across documents. Judge each one on its own.

Reply with the probability only."""

DEFAULT_PROMPT = "v3"

# Everything above the calibration heading is the judge's evidence: what counts
# as a human cue, what counts as a machine cue, what is not evidence at all.
# Below it is only how to phrase the answer. A logprob run does not phrase an
# answer -- it is read off the distribution over one token -- so the evidence
# half of the prompt is used verbatim and this replaces the half that would
# otherwise be asking for two decimal places that nothing will ever parse.
#
# That substitution is exactly why the mode is in the column name: `v3` and
# `v3_lp` share their evidence and differ in their answer, which makes them
# comparable as judges but not interchangeable as scores.
LOGPROB_ANSWER_BLOCK = """ANSWER
Answer with exactly one word, and nothing else: `human` if the document was more likely written by a human, `machine` if it was more likely generated by a large language model. No probability, no explanation, no punctuation.
Judge each document on its own. Do not assume any particular mix of human and machine documents, and do not try to balance your answers across documents."""

# The heading each prompt version uses for its answer-format section. v1 wrote
# it in sentence case; v2 and v3 shout it.
_CALIBRATION_HEADINGS = ("CALIBRATION", "Calibration rules:")


def evidence_head(prompt_id: str) -> str:
    """A prompt version with its answer-format section cut off.

    What remains is what makes the version a judge -- the genre framing, both
    evidence lists, what is ruled out as evidence -- and none of how to phrase
    an answer. Two callers want exactly that: the logprob mode below, and
    add_llm_judge_reasons.py, which asks the same model what it sees in a
    document without asking it for a score. Sharing this is what keeps the cues
    a rationale reports and the cues a score was formed from the same list.
    """
    body = PROMPTS[prompt_id]
    for heading in _CALIBRATION_HEADINGS:
        marker = f"\n{heading}"
        if marker in body:
            return body[: body.index(marker)].rstrip()
    raise ValueError(
        f"Prompt {prompt_id!r} has no answer-format section "
        f"({' / '.join(_CALIBRATION_HEADINGS)}), so its evidence half cannot be "
        "separated from its answer half. Add one of those headings."
    )


def system_prompt_for(prompt_id: str, mode: str) -> str:
    """The system prompt for one (prompt version, scoring mode) pair.

    In `parsed` mode this is the stored prompt, unchanged. In `logprob` mode the
    answer-format section is swapped for LOGPROB_ANSWER_BLOCK; see the note
    there for why that is a substitution rather than an edit to the prompt.
    """
    if mode == MODE_PARSED:
        return PROMPTS[prompt_id]
    try:
        return evidence_head(prompt_id) + "\n\n" + LOGPROB_ANSWER_BLOCK
    except ValueError as exc:
        raise ValueError(
            f"{exc} The logprob variant cannot be derived from it; run this "
            f"prompt with --mode {MODE_PARSED}."
        ) from exc


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
    # Answered before parsing: --list-models is a question about the registry,
    # not about a file, and requiring a Parquet to ask it would be silly.
    if "--list-models" in sys.argv:
        print_models()
        raise SystemExit(0)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquet", type=Path, help="Parquet to add judge scores to.")
    ap.add_argument(
        "--list-models",
        action="store_true",
        help="Print the judge registry (aliases, providers, weight sizes) and exit.",
    )
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
        metavar="ALIAS|REPO_ID",
        help=f"Judge model (default: {DEFAULT_MODEL}). Either a registry alias "
        f"({', '.join(sorted(MODELS))}) or a Hugging Face repo id such as "
        "Qwen/Qwen3-14B, which is loaded onto the GPU and run locally. "
        "--list-models prints the table.",
    )
    ap.add_argument(
        "--column-name",
        default=None,
        help="Override the column stem this judge writes under. Only needed "
        "for a repo id whose derived name collides with another judge's, or to "
        "keep two configurations of one model apart.",
    )
    ap.add_argument(
        "--mode",
        default=None,
        choices=MODES,
        help=f"How the score is obtained. {MODE_PARSED!r}: the model writes the "
        f"probability and it is parsed (the only option through an API, and "
        f"what the existing columns hold). {MODE_LOGPROB!r}: the model answers "
        "with one word and the score is read off that token's distribution — "
        f"local only, one forward pass per row, no ties. Default: {DEFAULT_MODE} "
        "everywhere, so a local run is comparable with the API columns already "
        "in the file.",
    )
    ap.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Device for a local judge (default: {DEFAULT_DEVICE}). "
        "'cpu' works but is not worth the wait. To choose a card use --gpu, "
        "which is what the other scripts in this repo take.",
    )
    ap.add_argument(
        "--gpu",
        default=None,
        help="Value for CUDA_VISIBLE_DEVICES (which GPU(s) to use, e.g. '0' or "
        "'0,1'). Default: leave unset so all visible GPUs are used.",
    )
    ap.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Limit CPU threads used for inference (sets OMP_NUM_THREADS / "
        "MKL_NUM_THREADS and torch's intra-op pool). Default: leave unset.",
    )
    ap.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        help=f"torch dtype for a local judge's weights (default: {DEFAULT_DTYPE}). "
        "An AWQ checkpoint is loaded in float16 instead, which its kernels "
        "require, unless this says otherwise.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows per forward pass for a local judge (default: "
        f"{DEFAULT_BATCH_SIZE}). Raise it until the card complains; a failed "
        "batch is retried row by row, so an over-large value costs time rather "
        "than scores. Ignored by the API judges, which use --workers.",
    )
    ap.add_argument(
        "--max-doc-tokens",
        type=int,
        default=DEFAULT_MAX_DOC_TOKENS,
        help=f"Cut each document to this many of the local model's tokens "
        f"(default: {DEFAULT_MAX_DOC_TOKENS}); 0 sends it whole. The prompt is "
        "never cut, only the document. Applies to local judges; --max-chars is "
        "the provider-agnostic equivalent.",
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
        "rate-limits; 1 makes the run strictly sequential. A local judge "
        "ignores this and batches instead — see --batch-size.",
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
        help="Only judge N eligible rows, drawn round-robin across "
        f"{DOMAIN_COL!r} x {GENERATED_BY_COL!r} so every domain and both "
        "classes are represented equally (see apply_limit). Default: every "
        "eligible row in the file. For smoke tests and prompt comparisons. The "
        "rest keep whatever they already have, they are not dropped from the "
        "output.",
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


def _identifier(text: str) -> str:
    """Squash anything into a lowercase identifier fragment."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()


def score_column(spec: JudgeModel, prompt_id: str, mode: str) -> str:
    """Output column for one judge, e.g. ``llm_judge_score_qwen3_14b_v3_lp``.

    All three parts are in the name deliberately: scores from two models, from
    two prompts, or from two scoring modes are not on the same scale, and a
    shared column would mix them with nothing downstream able to tell which row
    came from which. The prefix is what ``analyse_score_human_vs_ai.py --metric
    llm_judge`` collects, so a new judge shows up there without being registered
    anywhere else.

    `parsed` contributes no suffix, which keeps the columns already in the
    corpora (``llm_judge_score_gpt56luna_v3``) spelled the way they are.
    """
    stem = _identifier(f"{spec.column}_{prompt_id}")
    return f"{COL_PREFIX}{stem}{MODE_SUFFIX[mode]}"


def resolve_model(requested: str, column_name: str | None) -> JudgeModel:
    """Turn a ``--model`` value into a JudgeModel.

    A registry key is looked up. Anything containing a slash is taken as a
    Hugging Face repo id and run locally, so a judge that is not worth a
    permanent entry can still be scored without editing this file -- its column
    stem is derived from the repo name (``Qwen/Qwen3-14B`` -> ``qwen3_14b``)
    unless ``--column-name`` overrides it.

    A repo id whose derived stem collides with a registry entry is refused
    rather than silently appended to that entry's column: the two are not the
    same judge, and mixing them is unrecoverable once written.
    """
    if requested in MODELS:
        if column_name:
            return replace(MODELS[requested], column=_identifier(column_name))
        return MODELS[requested]

    # A repo id that a registry entry already covers is that entry, not a new
    # judge: same weights, same column, and the entry's note and size estimate
    # are worth keeping.
    listed = next((m for m in MODELS.values() if m.model == requested), None)
    if listed is not None:
        return replace(listed, column=_identifier(column_name)) if column_name else listed

    if "/" not in requested:
        raise SystemExit(
            f"Unknown judge {requested!r}. Known aliases: "
            f"{', '.join(sorted(MODELS))}. A Hugging Face repo id (with a '/') "
            "is also accepted and runs locally; --list-models prints the table."
        )

    stem = _identifier(column_name or requested.split("/")[-1])
    clash = next((m for m in MODELS.values() if m.column == stem), None)
    if clash is not None:
        raise SystemExit(
            f"{requested} would write the column stem {stem!r}, which already "
            f"belongs to the judge {clash.key!r} ({clash.model}). Pass "
            "--column-name to give this one its own column."
        )
    return JudgeModel(
        key=requested, provider=LOCAL_PROVIDER, model=requested, column=stem,
        note="Not a registry entry: run from a Hugging Face repo id.",
    )


def resolve_mode(spec: JudgeModel, requested: str | None) -> str:
    """The scoring mode for this judge: what was asked for, or `parsed`."""
    mode = requested or DEFAULT_MODE
    if mode == MODE_LOGPROB and not spec.is_local:
        raise SystemExit(
            f"--mode {MODE_LOGPROB} needs the token distribution, which the "
            f"{spec.provider} API does not return here. Run {spec.key!r} with "
            f"--mode {MODE_PARSED}, or pick a local judge (--list-models)."
        )
    return mode


def print_models() -> None:
    """The registry, with what each local judge costs to hold in memory."""
    print(f"{'alias':<16}{'provider':<14}{'model':<38}{'column stem':<18}{'weights':>9}")
    print("-" * 95)
    for spec in MODELS.values():
        weights = spec.weights_gib
        size = f"{weights:.1f} GiB" if weights else "-"
        print(
            f"{spec.key:<16}{spec.provider:<14}{spec.model:<38}"
            f"{spec.column:<18}{size:>9}"
        )
    print()
    for spec in MODELS.values():
        if spec.note:
            print(f"  {spec.key}: {spec.note}")
    print()
    print("Any Hugging Face repo id (containing '/') also works and runs locally.")
    print("Weights are the resident model only: leave a few GiB for activations.")


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


def parse_probability(text: str) -> float:
    """The first probability in a free-text answer, or ValueError.

    Only used by a local judge in `parsed` mode -- the APIs return a
    schema-checked number and never come through here. Small models do prefix
    the answer ("Probability: 0.37"), which is worth accepting; anything without
    a number in [0, 1] is a failed row, not a 0.
    """
    for match in re.finditer(r"(?<![\w.])(?:[01](?:\.\d+)?|\.\d+)", text):
        value = float(match.group())
        if 0.0 <= value <= 1.0:
            return value
    raise ValueError(f"no probability in answer: {text.strip()[:80]!r}")


_PROB_NUM = r"(\d{1,3}\s*%|(?:0?\.\d+)|[01](?:\.\d+)?)"
_MARKED_PROB_RE = re.compile(
    r"(?:probabilit(?:y|ies)|p\s*\(\s*human\s*\)|human[_ ]?prob\w*|answer|verdict|"
    r"score)\D{0,25}" + _PROB_NUM,
    re.IGNORECASE,
)
_BARE_PROB_RE = re.compile(r"(?<![\w.])" + _PROB_NUM)


def parse_judge_probability(text: str) -> float:
    """P(human) from a free-text answer — for a chatty open-weight judge.

    Unlike :func:`parse_probability` (first number, fine for a terse small
    model), this prefers the number right after a ``probability`` / ``answer`` /
    ``P(human)`` marker, and otherwise takes the LAST number in [0, 1]: a model
    that reasons out loud states its verdict at the end, and the first number in
    "the text is ~0.3 formal, so P(human) = 0.8" is the wrong one. ``<think>``
    blocks are stripped first.
    """
    text = _THINK_RE.sub("", text)

    def _to_unit(raw: str) -> float | None:
        raw = raw.strip()
        v = float(raw.rstrip(" %")) / (100.0 if raw.endswith("%") else 1.0)
        return v if 0.0 <= v <= 1.0 else None

    m = _MARKED_PROB_RE.search(text)
    if m and (v := _to_unit(m.group(1))) is not None:
        return v
    hits = [v for raw in _BARE_PROB_RE.findall(text) if (v := _to_unit(raw)) is not None]
    if hits:
        return hits[-1]
    raise ValueError(f"no probability in answer: {text.strip()[:100]!r}")


class Judge:
    """One judge model under one system prompt and one scoring mode.

    Subclasses differ in where the model runs, not in what a score means: both
    return P(human-written) in [0, 1] for a single document, and both are called
    the same way by the runner.
    """

    # True when the judge wants whole batches (a GPU) rather than many
    # concurrent single calls (an API). It is what main() dispatches on.
    batched = False

    def __init__(self, spec: JudgeModel, prompt_id: str = DEFAULT_PROMPT,
                 mode: str = MODE_PARSED, system_prompt: str | None = None) -> None:
        self.spec = spec
        self.alias = spec.key
        self.provider = spec.provider
        self.model_id = spec.model
        self.prompt_id = prompt_id
        self.mode = mode
        # Overridable so a subclass can ask this model something other than for
        # a score (add_llm_judge_reasons.py) without reimplementing the client
        # setup, the chat templating and the batching underneath.
        self.system_prompt = system_prompt or system_prompt_for(prompt_id, mode)

    def __call__(self, text: str) -> float:
        """P(human-written) for one document. Raises on a failed call."""
        raise NotImplementedError

    def score_batch(self, texts: list[str]) -> list[float]:
        """P(human-written) for several documents. Raises if any one fails."""
        return [self(text) for text in texts]

    def diagnostics(self) -> list[str]:
        """Anything the run learned about itself, printed in the summary."""
        return []

    def backoff(self, attempt: int) -> float:
        """Seconds to wait before retry `attempt`. Jittered so that parallel
        workers do not all come back at once after a shared rate limit."""
        return BACKOFF_BASE**attempt * (1.0 + random.random())


class ApiJudge(Judge):
    """A judge reached over an API: one call per row, many rows in flight.

    Holds the provider client (thread-safe in all three SDKs, so the same
    instance serves every worker) and turns whatever the provider returns into a
    plain float.
    """

    def __init__(self, spec: JudgeModel, prompt_id: str = DEFAULT_PROMPT,
                 mode: str = MODE_PARSED, system_prompt: str | None = None) -> None:
        super().__init__(spec, prompt_id, mode, system_prompt)
        self._client = self._build_client()
        # genai4science only: how each score was actually obtained, so a run on
        # the fragile free-text path is visible in the summary. Guarded by a
        # lock because one instance serves every worker thread.
        self._g4s_lock = threading.Lock()
        self._g4s_structured = 0
        self._g4s_fallback = 0
        self._g4s_parsefail = 0

    def _build_client(self):
        # Imported here rather than at module level so a missing optional SDK
        # only blocks the provider that needs it.
        if self.provider == "openai":
            from openai import OpenAI

            return OpenAI()
        if self.provider == "genai4science":
            from openai import OpenAI

            from epai.utils.env_utils import genai4science_endpoint

            base_url, api_key_env = genai4science_endpoint()
            return OpenAI(base_url=base_url, api_key=os.environ[api_key_env])
        if self.provider == "anthropic":
            import anthropic

            return anthropic.Anthropic()
        if self.provider == "gemini":
            from google import genai

            return genai.Client()
        raise ValueError(f"Unsupported provider: {self.provider}")

    def __call__(self, text: str) -> float:
        if self.provider == "openai":
            return self._score_openai(text)
        if self.provider == "genai4science":
            return self._score_genai4science(text)
        if self.provider == "anthropic":
            return self._score_anthropic(text)
        return self._score_gemini(text)

    def _score_genai4science(self, text: str) -> float:
        """OpenAI-compatible endpoint, open-weight model. Structured output
        first; plain completion + :func:`parse_judge_probability` if the server
        does not honour ``response_format`` (many vLLM/Ollama deployments do not
        for every model). Which path each row took is counted for the summary."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": USER_TEMPLATE.format(text=text)},
        ]
        try:
            response = self._client.chat.completions.parse(
                model=self.model_id, messages=messages, response_format=JudgeVerdict,
            )
            parsed = response.choices[0].message.parsed
            if parsed is not None:
                with self._g4s_lock:
                    self._g4s_structured += 1
                return float(parsed.human_probability)
        except Exception:
            pass
        with self._g4s_lock:
            self._g4s_fallback += 1
        answer = self._client.chat.completions.create(
            model=self.model_id, messages=messages, max_tokens=2048,
        ).choices[0].message.content or ""
        try:
            return parse_judge_probability(answer)
        except ValueError:
            with self._g4s_lock:
                self._g4s_parsefail += 1
            raise

    def diagnostics(self) -> list[str]:
        if self.provider != "genai4science":
            return []
        n = self._g4s_structured + self._g4s_fallback
        if not n:
            return []
        lines = [
            f"    genai4science: {self._g4s_structured} row(s) via structured "
            f"output, {self._g4s_fallback} via free-text fallback"
        ]
        if self._g4s_fallback and not self._g4s_structured:
            lines.append(
                "    the endpoint never honoured response_format — every score "
                "came from parsing free text; read the AUROC and distinct-value "
                "count in the checks below before trusting this column"
            )
        if self._g4s_parsefail:
            lines.append(
                f"    {self._g4s_parsefail} free-text parse failure(s) (retried) "
                f"— the model is not answering with a probability"
            )
        return lines

    def _score_openai(self, text: str) -> float:
        # No temperature: the reasoning-capable models reject anything but the
        # default, and the schema already pins the output shape.
        response = self._client.chat.completions.parse(
            model=self.model_id,
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
            model=self.model_id,
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
            model=self.model_id,
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


class LocalJudge(Judge):
    """A judge held on this machine's GPU, scored a batch at a time.

    In `logprob` mode -- the reason to run locally at all -- a row costs one
    forward pass and no generation: the prompt asks for a single word, and the
    score is that position's distribution, ``P(human) / (P(human) + P(machine))``,
    over the token ids in HUMAN_WORDS and MACHINE_WORDS. Nothing is sampled, so
    the run is deterministic and the score is continuous rather than piled onto
    whichever round numbers the model likes to write.

    In `parsed` mode it generates the probability and parses it, which is the
    only way to compare a local model against the API columns on equal terms:
    same prompt, same answer format, only the model changed.

    Loading is deferred to __init__ rather than done at import, and torch is
    imported here rather than at module level, so the API providers still run on
    a machine with no CUDA and no torch build worth speaking of.
    """

    batched = True

    def __init__(
        self,
        spec: JudgeModel,
        prompt_id: str = DEFAULT_PROMPT,
        mode: str = MODE_LOGPROB,
        device: str = DEFAULT_DEVICE,
        dtype: str = DEFAULT_DTYPE,
        max_doc_tokens: int = DEFAULT_MAX_DOC_TOKENS,
        system_prompt: str | None = None,
        cpu_threads: int | None = None,
    ) -> None:
        super().__init__(spec, prompt_id, mode, system_prompt)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if cpu_threads is not None:
            # The env vars apply_runtime_limits set cover the OpenMP/MKL pools;
            # this covers torch's own intra-op pool, which does not always
            # follow them. It has to come after torch is imported, which is why
            # it is here and not beside them.
            torch.set_num_threads(cpu_threads)

        self._torch = torch
        self.device = device
        self.dtype = dtype
        self.max_doc_tokens = max_doc_tokens
        self.n_truncated = 0
        self.answer_mass: list[float] = []

        print(f"Loading {spec.model} on {device} ({dtype}) …")
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model)
        # Left padding, because these are decoder-only models scored at their
        # LAST position: right padding would put the answer slot behind a run of
        # pad tokens and read the distribution off padding.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {"dtype": getattr(torch, dtype)}
        has_accelerate = importlib.util.find_spec("accelerate") is not None
        if spec.quant and not has_accelerate:
            # For a quantised checkpoint accelerate is not an optimisation:
            # transformers rebuilds the linear layers through
            # `accelerate.init_empty_weights`, so the CPU-RAM path below does
            # not exist and the load dies inside the quantizer. Raised before
            # the weights are fetched rather than after several GiB of them.
            raise RuntimeError(
                f"{spec.key} is a {spec.quant} checkpoint and transformers "
                "cannot build a quantised model without accelerate. Either "
                "`pip install accelerate`, or run an unquantised judge "
                "(--model qwen3-14b) instead."
            )
        if has_accelerate:
            # accelerate streams the shards straight onto the card.
            load_kwargs["device_map"] = device
        elif device != "cpu":
            # Without it an unquantised model is built in CPU RAM and moved
            # afterwards, which works but needs that much free RAM and one
            # extra copy of a multi-GB model. Said out loud because the symptom
            # is a load that sits there for minutes, which reads like a hang.
            size = f"~{spec.weights_gib:.0f} GiB " if spec.weights_gib else ""
            print(
                f"NOTE  accelerate is not installed, so the {size}weights are "
                f"staged through CPU RAM before moving to {device}. Loading is "
                "slower; nothing else differs."
            )
        self.model = AutoModelForCausalLM.from_pretrained(spec.model, **load_kwargs)
        if "device_map" not in load_kwargs:
            self.model.to(device)
        self.model.eval()

        self._supports_system = self._probe_system_role()
        self._human_ids, self._machine_ids = self._answer_token_ids()

    # -- setup helpers ----------------------------------------------------

    def _probe_system_role(self) -> bool:
        """Whether this chat template accepts a system turn (Gemma does not)."""
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return True
        except Exception:
            return False

    def _answer_token_ids(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """First-token ids for the two answers, checked to be usable.

        Only meaningful in logprob mode, and only correct if the two sets are
        disjoint: a tokeniser that opened both words with one shared token would
        make the ratio read the same number twice.
        """
        if self.mode != MODE_LOGPROB:
            return (), ()

        def first_ids(words: tuple[str, ...]) -> tuple[int, ...]:
            ids = []
            for word in words:
                encoded = self.tokenizer(word, add_special_tokens=False)["input_ids"]
                if encoded:
                    ids.append(int(encoded[0]))
            return tuple(dict.fromkeys(ids))

        human, machine = first_ids(HUMAN_WORDS), first_ids(MACHINE_WORDS)
        if not human or not machine:
            raise RuntimeError(
                f"{self.model_id}: could not tokenise the answer words, so "
                f"--mode {MODE_LOGPROB} has nothing to read. Use --mode "
                f"{MODE_PARSED}."
            )
        shared = set(human) & set(machine)
        if shared:
            raise RuntimeError(
                f"{self.model_id}: 'human' and 'machine' share the opening "
                f"token(s) {sorted(shared)}, so their probabilities cannot be "
                f"told apart. Use --mode {MODE_PARSED}."
            )
        return human, machine

    # -- prompt assembly --------------------------------------------------

    def _truncate(self, text: str) -> str:
        """Cut the document to the token budget, counting the model's tokens."""
        if not self.max_doc_tokens:
            return text
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.max_doc_tokens:
            return text
        self.n_truncated += 1
        return self.tokenizer.decode(ids[: self.max_doc_tokens], skip_special_tokens=True)

    def _render(self, text: str) -> str:
        """The full chat string for one document, generation prompt included."""
        user = USER_TEMPLATE.format(text=self._truncate(text))
        if self._supports_system:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ]
        else:
            # No system turn: the instructions ride in front of the document,
            # which is what the models without one expect anyway.
            messages = [{"role": "user", "content": f"{self.system_prompt}\n\n{user}"}]

        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            # Qwen3 and friends: keep the reasoning block empty. A judge that
            # thinks first costs 10-20x the tokens for one scalar, and in
            # logprob mode the answer slot would hold `<think>` instead of the
            # answer.
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _encode(self, texts: list[str]):
        prompts = [self._render(text) for text in texts]
        # add_special_tokens=False: the chat template already wrote whatever BOS
        # the model wants, and a second one shifts every position.
        encoded = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        )
        # Only the two fields every causal LM takes: a tokeniser that also
        # returns token_type_ids would make forward() raise on a model whose
        # signature has no such argument, and the batch would be written off as
        # an error rather than a mismatch.
        return {
            key: encoded[key].to(self.model.device)
            for key in ("input_ids", "attention_mask")
            if key in encoded
        }

    # -- scoring ----------------------------------------------------------

    def __call__(self, text: str) -> float:
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str]) -> list[float]:
        encoded = self._encode(texts)
        if self.mode == MODE_LOGPROB:
            return self._score_logprob(encoded)
        return self._score_generated(encoded)

    def _score_logprob(self, encoded: dict) -> list[float]:
        torch = self._torch
        with torch.inference_mode():
            try:
                # Only the last position is ever read, and materialising logits
                # for every position of a 4k-token batch is gigabytes of vocab-
                # wide tensor for nothing.
                out = self.model(**encoded, use_cache=False, logits_to_keep=1)
            except TypeError:
                out = self.model(**encoded, use_cache=False)

        # float() before the softmax: the ratio of two small probabilities is
        # exactly where bf16's 8-bit mantissa shows up.
        logits = out.logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        human = probs[:, list(self._human_ids)].sum(dim=-1)
        machine = probs[:, list(self._machine_ids)].sum(dim=-1)
        mass = human + machine
        scores = (human / mass.clamp_min(1e-12)).tolist()
        self.answer_mass.extend(mass.tolist())
        return [float(s) for s in scores]

    def _score_generated(self, encoded: dict) -> list[float]:
        torch = self._torch
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
        return [parse_probability(answer) for answer in answers]

    # -- reporting --------------------------------------------------------

    def backoff(self, attempt: int) -> float:
        """No wait. A forward pass is deterministic: a row that failed to parse
        will fail again identically, and an out-of-memory error is answered by
        the smaller batch the caller already fell back to, not by waiting."""
        return 0.0

    def diagnostics(self) -> list[str]:
        out = []
        if self.n_truncated:
            out.append(
                f"  {self.n_truncated} document(s) cut to {self.max_doc_tokens} "
                "tokens before judging"
            )
        if self.mode == MODE_LOGPROB and self.answer_mass:
            mass = np.array(self.answer_mass, dtype=float)
            weak = int((mass < MIN_ANSWER_MASS).sum())
            out.append(
                f"  answer mass: mean {mass.mean():.3f}, min {mass.min():.3f} "
                f"(share of the distribution on 'human'/'machine')"
            )
            if weak:
                out.append(
                    f"  WARNING  {weak} row(s) below {MIN_ANSWER_MASS:.2f} — the "
                    "model answered something other than the two words there, "
                    "and those scores are a ratio of two near-zero numbers."
                )
        return out


def dtype_for(spec: JudgeModel, requested: str) -> str:
    """The weight dtype to load with, corrected for what a checkpoint wants.

    The default here is bf16 because everything unquantised prefers it, but the
    int4 kernels are written around fp16: AWQ refuses anything else, and
    transformers' compressed-tensors path says outright that fp16 is the
    efficient one. So a quantised entry flips the default. An explicit --dtype
    is left alone -- overriding what was actually typed would be worse than
    whatever it causes.

    Called by the findings block as well as by the loader, so it stays silent
    and the substitution is shown on the device line instead.
    """
    if spec.quant and requested == DEFAULT_DTYPE:
        return "float16"
    return requested


def build_judge(spec: JudgeModel, prompt_id: str, mode: str, args) -> Judge:
    """The judge for this run: local models load here, APIs just connect."""
    if spec.is_local:
        return LocalJudge(
            spec,
            prompt_id=prompt_id,
            mode=mode,
            device=args.device,
            dtype=dtype_for(spec, args.dtype),
            max_doc_tokens=args.max_doc_tokens,
            cpu_threads=args.cpu_threads,
        )
    return ApiJudge(spec, prompt_id=prompt_id, mode=mode)


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
            delay = judge.backoff(attempt)
            if delay:
                time.sleep(delay)
    return RowResult(position=position, score=None, error=last_error)


def judge_batch(
    judge: Judge, positions: list[int], texts: dict[int, str], retries: int
) -> list[RowResult]:
    """Score a batch of rows in one forward pass.

    A batch is all-or-nothing -- an out-of-memory error or a single unparsable
    answer takes the whole thing down -- so a failure is retried one row at a
    time. That costs a pass over the batch but isolates the row actually
    responsible instead of writing off eight good ones with it.
    """
    try:
        scores = judge.score_batch([texts[pos] for pos in positions])
        return [
            RowResult(position=pos, score=float(np.clip(score, 0.0, 1.0)))
            for pos, score in zip(positions, scores)
        ]
    except Exception as exc:
        tqdm.write(
            f"  batch of {len(positions)} failed ({type(exc).__name__}: {exc}); "
            "retrying those rows one at a time"
        )
        release_memory(judge)
        return [judge_row(judge, pos, texts[pos], retries) for pos in positions]


def release_memory(judge: Judge) -> None:
    """Hand the allocator's cached blocks back, after an out-of-memory error."""
    torch = getattr(judge, "_torch", None)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def strata(df: pd.DataFrame) -> np.ndarray:
    """One stratum label per row: ``domain | generated_by``, as available.

    What a limited run has to keep balanced. Both axes matter and for different
    reasons:

    ``generated_by``
        These corpora are stored one generator after another -- every human row,
        then every machine row -- so a sample taken in file order is one class,
        and the summary's accuracy and ROC-AUC lines (the whole point of a small
        run) have nothing to separate.

    ``domain``
        The judge does not score the domains alike -- on ghostbuster_gpt56luna
        it is perfect on ``reuter`` and calls half the human ``essay`` rows
        machine -- so a sample drawn from one domain says nothing about the
        file, and a number from it cannot be compared with a number from a
        differently-drawn sample of the same file.

    A file missing either column contributes nothing to the label, and one
    missing both leaves a single stratum, which is the old file-order behaviour.
    """
    parts = [
        df[col].astype(str).to_numpy()
        for col in (DOMAIN_COL, GENERATED_BY_COL)
        if col in df.columns
    ]
    if not parts:
        return np.full(len(df), "all", dtype=object)
    return np.array([" | ".join(vals) for vals in zip(*parts)], dtype=object)


def take_round_robin(positions: list[int], labels: np.ndarray, cap: int) -> list[int]:
    """Up to ``cap`` positions, one per stratum per pass, strata in name order.

    Order within a stratum is the order it was given in, so the same call on the
    same file always returns the same rows: two prompt versions, or two judges,
    are then compared on identical documents rather than on two samples. A
    stratum that runs out drops out of the rotation and the rest keep going, so
    ``cap`` is always filled if there are that many rows to fill it with -- an
    exhausted domain costs the sample its balance, not its size.
    """
    buckets: dict[str, list[int]] = {}
    for pos in positions:
        buckets.setdefault(str(labels[pos]), []).append(int(pos))
    chosen: list[int] = []
    depth = 0
    while len(chosen) < cap and any(len(b) > depth for b in buckets.values()):
        for name in sorted(buckets):
            if len(buckets[name]) > depth:
                chosen.append(buckets[name][depth])
                if len(chosen) == cap:
                    break
        depth += 1
    return chosen


def apply_limit(
    df: pd.DataFrame, judgeable: np.ndarray, limit: int | None
) -> np.ndarray:
    """Restrict the judgeable mask to ``limit`` eligible rows, spread evenly.

    ``limit is None`` -- the default -- judges every eligible row in the file.
    A number counts eligible rows rather than file positions, and draws them
    round-robin across ``domain`` x ``generated_by`` (see :func:`strata`), so
    ``--limit 600`` on a three-domain Ghostbuster corpus is 100 human and 100
    machine rows from each of essay, reuter and wp rather than 600 rows of
    whatever the file happens to start with.
    """
    if limit is None:
        return judgeable
    if limit < 1:
        raise ValueError(f"--limit must be >= 1, got {limit}")

    eligible = [int(pos) for pos in np.flatnonzero(judgeable)]
    picked = np.array(
        take_round_robin(eligible, strata(df), limit), dtype=int
    )

    limited = np.zeros_like(judgeable)
    limited[picked] = True
    return limited


# Extra packages a quantised checkpoint needs on top of transformers, and the
# pip name to suggest. Checked up front: the alternative is a multi-GB download
# that fails on the last line of from_pretrained.
_QUANT_PACKAGES = {
    "compressed-tensors": (("compressed_tensors",), "compressed-tensors"),
    "gptq": (("gptqmodel", "auto_gptq"), "gptqmodel"),
    # Kept so an AWQ checkpoint still fails with a sentence rather than a
    # traceback, but there is no registry entry for one on purpose: AutoAWQ is
    # deprecated and untested against this project's torch and transformers.
    "awq": (("awq",), "autoawq (deprecated — prefer a compressed-tensors build)"),
}


def blocking_reason(spec: JudgeModel, device: str = DEFAULT_DEVICE) -> str | None:
    """Why this judge cannot run here, or None if it can."""
    if not spec.is_local:
        if spec.provider == "genai4science":
            from epai.utils.env_utils import genai4science_endpoint

            _, env_var = genai4science_endpoint()
        else:
            env_var = _REQUIRED_ENV[spec.provider]
        return None if os.environ.get(env_var) else f"needs {env_var}"

    for package in ("torch", "transformers"):
        if importlib.util.find_spec(package) is None:
            return f"needs {package}, which is not importable here"

    if spec.quant:
        if spec.quant == "compressed-tensors":
            # Not a missing package: there is no int4 inference path to install.
            # See the note on the qwen3-32b-w4a16 entry -- transformers expands
            # these weights to fp16 on the first forward pass, so the checkpoint
            # is smaller than the model it becomes.
            return (
                "compressed-tensors weights have no int4 kernel in transformers "
                "-- the first forward pass decompresses them to fp16, which for "
                "a 32B model is ~65 GiB. Use --model qwen3-14b, or serve this "
                "checkpoint with vLLM"
            )
        modules, pip_name = _QUANT_PACKAGES[spec.quant]
        if all(importlib.util.find_spec(m) is None for m in modules):
            return (
                f"{spec.quant.upper()} weights need kernels that are not "
                f"installed (pip install {pip_name})"
            )
        if importlib.util.find_spec("accelerate") is None:
            # Unquantised weights load without accelerate, just slowly, through
            # CPU RAM. Quantised ones do not load at all: transformers swaps
            # the linear layers under `accelerate.init_empty_weights`.
            return (
                "quantised weights need accelerate, which is not installed "
                "(pip install accelerate)"
            )

    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            return "no CUDA device visible (pass --device cpu to run anyway)"
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


def plan_details(spec: JudgeModel, args) -> list[tuple[str, str]]:
    """The judge-specific lines of the findings block.

    Derived from the spec and the arguments rather than from a live judge: the
    plan is printed before anything is loaded, which is the whole point of
    printing it -- a 30GB download should not start ahead of the question of
    whether this is even the right file.
    """
    if not spec.is_local:
        return [("workers", f"{args.workers} row(s) in flight")]

    weights = spec.weights_gib
    lines = [
        ("device", args.device + f" ({dtype_for(spec, args.dtype)}"
         + (f", {spec.quant}" if spec.quant else "") + ")"),
        ("gpu", args.gpu if args.gpu is not None else "all visible (unset)"),
        ("cpu threads",
         str(args.cpu_threads) if args.cpu_threads is not None else "unset"),
        ("batch size", str(args.batch_size)),
        ("doc budget", f"{args.max_doc_tokens} tokens"
         if args.max_doc_tokens else "whole document"),
    ]
    if weights:
        lines.append(("weights", f"~{weights:.1f} GiB resident"))
    return lines


def print_findings(
    path: Path,
    df: pd.DataFrame,
    text_col: str,
    col: str,
    spec: JudgeModel,
    prompt_id: str,
    mode: str,
    judgeable: np.ndarray,
    positions: np.ndarray,
    allowed: list[str] | None,
    details: list[tuple[str, str]],
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
    print(f"  alias         : {spec.key}")
    print(f"  provider      : {spec.provider}")
    print(f"  model         : {spec.model}")
    print(f"  prompt        : {prompt_id}")
    print(f"  mode          : {mode}"
          + ("  (score read off the answer token's distribution)"
             if mode == MODE_LOGPROB else "  (probability written by the model)"))
    for label, value in details:
        print(f"  {label:<14}: {value}")
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
    judge: Judge | None = None,
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

    diagnostics = judge.diagnostics() if judge is not None else []
    if diagnostics:
        print("  judge:")
        for line in diagnostics:
            print(line)

    evaluate(df, col)
    print()


def run_threaded(
    judge: Judge,
    positions: np.ndarray,
    texts: dict[int, str],
    workers: int,
    retries: int,
    record,
) -> None:
    """Score an API judge: many single-row calls in flight at once."""
    # The executor is managed by hand rather than with a `with` block: on Ctrl-C
    # the block's exit would wait for every queued row to be judged, which on a
    # 2000-row file is exactly the wait the interrupt was meant to avoid.
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futures = {
            pool.submit(judge_row, judge, pos, texts[pos], retries): pos
            for pos in positions
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Judging ({judge.alias})",
            unit="row",
        ):
            record(future.result())
    except KeyboardInterrupt:
        print("\nInterrupted — writing the scores obtained so far.")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def run_batched(
    judge: Judge,
    positions: np.ndarray,
    texts: dict[int, str],
    batch_size: int,
    retries: int,
    record,
) -> None:
    """Score a local judge: one forward pass per batch, batches in order.

    Sequential on purpose. The GPU is the bottleneck and threads would only
    interleave batches onto one device, turning a predictable memory ceiling
    into a race for it.
    """
    order = list(positions)
    batches = [
        order[i : i + max(1, batch_size)] for i in range(0, len(order), max(1, batch_size))
    ]
    try:
        with tqdm(total=len(order), desc=f"Judging ({judge.alias})", unit="row") as bar:
            for batch in batches:
                for result in judge_batch(judge, batch, texts, retries):
                    record(result)
                bar.update(len(batch))
    except KeyboardInterrupt:
        print("\nInterrupted — writing the scores obtained so far.")


def apply_runtime_limits(args) -> None:
    """Pin the run to one GPU and cap the CPU pools, as the other scripts do.

    CUDA_VISIBLE_DEVICES is read when the CUDA driver initialises and the thread
    limits when the OpenMP runtime loads, both of which happen inside torch. This
    module imports torch lazily -- inside LocalJudge and inside blocking_reason --
    precisely so this can run first; call it before either of them.

    ``--gpu`` renumbers the cards it leaves visible, so the first one named is
    always ``cuda:0``. That makes ``--device cuda:N`` alongside it a way to point
    at a card that is no longer there, and it is refused rather than half-honoured.
    """
    if args.gpu is not None:
        if not args.device.startswith("cuda"):
            raise SystemExit(
                f"--gpu {args.gpu} with --device {args.device}: pick one. --gpu "
                "chooses which card to run on, --device cpu says not to use one."
            )
        if ":" in args.device:
            raise SystemExit(
                f"--gpu {args.gpu} with --device {args.device}: --gpu hides the "
                "other cards and renumbers what is left, so the card it selects "
                f"is cuda:0 inside this process. Drop --device, or drop --gpu and "
                f"use --device cuda:{args.gpu.split(',')[0]} alone."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.cpu_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)


def main() -> None:
    args = parse_args()
    # Before blocking_reason, which imports torch to ask about CUDA.
    apply_runtime_limits(args)
    out_path = args.out or args.parquet
    spec = resolve_model(args.model, args.column_name)
    prompt_id = args.prompt
    mode = resolve_mode(spec, args.mode)
    col = score_column(spec, prompt_id, mode)

    blocked = blocking_reason(spec, args.device)
    if blocked:
        raise SystemExit(f"Cannot run judge {spec.key!r}: {blocked}.")

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
        spec,
        prompt_id,
        mode,
        judgeable,
        positions,
        allowed,
        plan_details(spec, args),
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
    if spec.is_local:
        print(
            f"NOTE  {len(positions)} row(s) will be scored on {args.device}. No "
            "API cost, but the weights are downloaded on first use and the run "
            "holds the card until it finishes."
        )
    else:
        print(f"NOTE  this is a billed API call: {len(positions)} row(s) will be sent.")
    if not args.yes:
        answer = input(f"Proceed and write to {out_path}? Type 'yes': ").strip().lower()
        if answer != "yes":
            print("Aborted — no changes made.")
            return

    # Built after the confirmation, not before: for a local judge this is where
    # tens of GB are downloaded and moved onto the card.
    judge = build_judge(spec, prompt_id, mode, args)

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
    counters = {"done": 0}
    save_lock = threading.Lock()
    started = time.perf_counter()

    def write_out() -> None:
        """Persist what has been scored so far, without clobbering other columns."""
        df[col] = pd.Series(values, index=df.index, dtype="float64")
        save_parquet(df, out_path)

    def record(result: RowResult) -> None:
        """Take one row's outcome, and checkpoint the file every --save-every."""
        values[result.position] = result.score
        if result.error is not None:
            failures.append((result.position, result.error))
            tqdm.write(f"  row {result.position} failed: {result.error}")
        counters["done"] += 1

        # Save periodically so an interrupt or a crash does not throw away work
        # that has already been paid for -- in money or in GPU time.
        with save_lock:
            if counters["done"] % args.save_every == 0:
                write_out()

    if judge.batched:
        run_batched(judge, positions, texts, args.batch_size, args.retries, record)
    else:
        run_threaded(judge, positions, texts, args.workers, args.retries, record)

    write_out()
    print_summary(
        out_path,
        col,
        counters["done"] - len(failures),
        failures,
        time.perf_counter() - started,
        df,
        judge,
    )


if __name__ == "__main__":
    main()

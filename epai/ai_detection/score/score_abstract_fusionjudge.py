r"""Score one abstract with the gpt56luna FusionJudge.

Standalone inference: given a ``.txt`` file holding a single abstract, this
1) standardises the text the same way the corpus pipeline does
   (``epai.utils.text_standardize.standardize_text``), 2) makes two API calls
   to gpt56luna (OpenAI) under the ``v3`` judge prompt -- one asking directly
   for a calibrated probability (``parsed`` mode, schema-checked), one asking
   for a single answer word and reading back the answer-position logprob
   window (``logprob`` mode) -- and 3) fuses the two P(human) readouts by
   plain arithmetic mean, per the paper's FusionJudge (an unweighted,
   sum-rule combination of a judge with its own logprob-derived readout; see
   ``score_fusionjudge_gpt56luna_v3.py``). The logprob member is scored with
   the paper-exact ``ord2`` closed form -- no floor, no branch, no clamp
   (``score_logjudge_ord2.py``).

Prints the (standardised) abstract, both member scores, the FusionJudge
score, and a verdict -- all in P(human) in [0, 1] (1.0 = human, 0.0 =
machine).

Needs OPENAI_API_KEY (see .env) -- this makes two billed API calls.

    .venv/bin/python epai/ai_detection/score/score_abstract_fusionjudge.py abstract.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dotenv import load_dotenv  # noqa: E402

from epai.ai_detection.score.add_llm_judge_scores import (  # noqa: E402
    MODE_LOGPROB,
    MODE_PARSED,
    MODELS,
    OPENAI_MAX_TOKENS,
    USER_TEMPLATE,
    JudgeVerdict,
    system_prompt_for,
)
from epai.ai_detection.score.score_fusionjudge_gpt56luna_v3 import mean_combine  # noqa: E402
from epai.ai_detection.score.score_logjudge_ord2 import score_ord2  # noqa: E402
from epai.utils.text_standardize import standardize_text  # noqa: E402

load_dotenv()

MODEL_KEY = "gpt56luna"
PROMPT_ID = "v3"
TOP_N = 20


def _judge_parsed(client, model_id: str, system: str, text: str) -> float:
    """P(human) read directly off the model's own calibrated answer --
    same call shape as ``add_llm_judge_scores.ApiJudge._score_openai``."""
    response = client.chat.completions.parse(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": USER_TEMPLATE.format(text=text)},
        ],
        response_format=JudgeVerdict,
        max_completion_tokens=OPENAI_MAX_TOKENS,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("model returned no parsable verdict")
    return float(parsed.human_probability)


def _logjudge_window(client, model_id: str, system: str, text: str) -> list[dict]:
    """The ranked top-k window at the answer position (temperature 0)."""
    resp = client.responses.create(
        model=model_id, instructions=system,
        input=USER_TEMPLATE.format(text=text),
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
                    return [{"token": a.token, "logprob": float(a.logprob)}
                            for a in (lp[0].top_logprobs or [])]
    raise ValueError("no output_text part in the Responses object")


# Confidence bands on P(human), matching the judge prompt's own calibration
# language ("reserve values below 0.10 and above 0.90 for genuinely strong
# evidence"). Not a new metric -- just labels for the same score.
def verdict(score: float) -> str:
    if score >= 0.90:
        return "HUMAN-written (high confidence)"
    if score >= 0.50:
        return "Likely human-written"
    if score >= 0.10:
        return "Likely machine-generated"
    return "MACHINE-generated (high confidence)"


def _print_report(abstract: str, judge_score: float, logjudge_score: float,
                   fusion_score: float) -> None:
    width = 78
    rule = "=" * width
    thin = "-" * width
    print(f"\n{rule}")
    print("ABSTRACT")
    print(rule)
    print(textwrap.fill(abstract, width=width))
    print(f"\n{rule}")
    print("FUSIONJUDGE RESULT  (gpt56luna, v3)")
    print(rule)
    print(f"  {'Judge (parsed)':<24} P(human) = {judge_score:.4f}")
    print(f"  {'LogJudge (ord2)':<24} P(human) = {logjudge_score:.4f}")
    print(thin)
    print(f"  {'FusionJudge':<24} P(human) = {fusion_score:.4f}")
    print(f"  {'Verdict':<24} {verdict(fusion_score)}")
    print(rule)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("txt", type=Path, help="path to a .txt file holding the abstract")
    ap.add_argument("--whitespace", choices=("lines", "flat"), default="lines",
                     help="standardisation layout convention (default: lines, "
                          "matching the corpus prepare stage)")
    args = ap.parse_args()

    if not args.txt.is_file():
        print(f"Not found: {args.txt}", file=sys.stderr)
        return 1
    raw = args.txt.read_text(encoding="utf-8")
    if not raw.strip():
        print("Empty abstract.", file=sys.stderr)
        return 1

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set -- see .env", file=sys.stderr)
        return 1

    text = standardize_text(raw, whitespace=args.whitespace)

    from openai import OpenAI
    client = OpenAI()

    spec = MODELS[MODEL_KEY]
    judge_score = _judge_parsed(
        client, spec.model, system_prompt_for(PROMPT_ID, MODE_PARSED), text,
    )
    window = _logjudge_window(
        client, spec.model, system_prompt_for(PROMPT_ID, MODE_LOGPROB), text,
    )
    logjudge_score = score_ord2(window)
    if logjudge_score is None:
        raise ValueError("empty logprob window -- capture failed")
    fusion_score = float(mean_combine([judge_score, logjudge_score]))

    _print_report(text, judge_score, logjudge_score, fusion_score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

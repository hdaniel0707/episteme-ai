r"""Score one abstract with the gpt56luna LogJudge ``wlp`` method.

Standalone inference: given a ``.txt`` file holding a single abstract, this
1) standardises the text the same way the corpus pipeline does
   (``epai.utils.text_standardize.standardize_text``), 2) sends it to
   gpt56luna (OpenAI Responses API) with the ``v3`` judge prompt asking for one
   word (``human``/``machine``) and reads back the answer-position logprob
   window, and 3) scores that window with the ``wlp`` contrast formula --
   the same prompt, model and formula as
   ``capture_judge_logprobs.py``/``score_logprob_windows.py`` run over a whole
   corpus, applied here to one document. See ``score_logprob_windows.md`` for
   the derivation.

Prints the (standardised) abstract, the P(human) score in [0, 1]
(1.0 = human, 0.0 = machine), and a verdict.

Needs OPENAI_API_KEY (see .env) -- this makes one billed API call.

    .venv/bin/python epai/ai_detection/score/score_abstract_wlp.py abstract.txt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dotenv import load_dotenv  # noqa: E402

from epai.ai_detection.score.add_llm_judge_scores import (  # noqa: E402
    LOGPROB_ANSWER_BLOCK,
    MODELS,
    USER_TEMPLATE,
    evidence_head,
)
from epai.utils.text_standardize import standardize_text  # noqa: E402

load_dotenv()

MODEL_KEY = "gpt56luna"
PROMPT_ID = "v3"
TOP_N = 20

# Below this, neither answer word carries usable mass: the model answered
# off-format. Same threshold as score_logprob_windows.py's _contrast.
MIN_ANSWER_MASS = 0.05


def _system_prompt() -> str:
    return evidence_head(PROMPT_ID) + "\n\n" + LOGPROB_ANSWER_BLOCK


def _window_openai(client, model_id: str, system: str, text: str) -> list[dict]:
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


def _norm(token: str | None) -> str:
    return "".join(ch for ch in (token or "") if ch.isalpha()).lower()


def _answer_probs(window: list[dict]) -> dict[str, float]:
    p = {"human": 0.0, "machine": 0.0}
    for s in window:
        lp = s.get("logprob")
        if lp is None or not math.isfinite(lp):
            continue
        tok = _norm(s.get("token"))
        if tok in p:
            p[tok] += math.exp(lp)
    return p


def score_wlp(window: list[dict]) -> float:
    """P(human) from the answer window: ``(1 + p_h - p_m) / 2``, leftover
    (non-answer) mass counted as 0.5 -- uninformative."""
    if not window:
        raise ValueError("empty logprob window -- capture failed")
    p = _answer_probs(window)
    p_h, p_m = p["human"], p["machine"]
    if p_h + p_m < MIN_ANSWER_MASS:
        return 0.5
    return float(min(1.0, max(0.0, (1.0 + p_h - p_m) / 2.0)))


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


def _print_report(abstract: str, score: float) -> None:
    width = 78
    rule = "=" * width
    print(f"\n{rule}")
    print("ABSTRACT")
    print(rule)
    print(textwrap.fill(abstract, width=width))
    print(f"\n{rule}")
    print("LOGJUDGE RESULT  (gpt56luna, wlp)")
    print(rule)
    print(f"  P(human) = {score:.4f}")
    print(f"  Verdict  = {verdict(score)}")
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
    window = _window_openai(client, spec.model, _system_prompt(), text)
    score = score_wlp(window)

    _print_report(text, score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

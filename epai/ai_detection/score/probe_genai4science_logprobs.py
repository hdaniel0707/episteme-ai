r"""One-shot capability probe: does the GenAI4Science endpoint return logprobs?

    uv run python epai/ai_detection/score/probe_genai4science_logprobs.py

Sends ONE short chat-completion request per model (``gpt-oss:120b``,
``llama3.3:70b`` by default) with ``logprobs=True, top_logprobs=20``, and
prints whatever the response actually carries for the first generated token.
Nothing here reads or writes a parquet -- it exists to answer one question
before ``add_llm_judge_scores.py``'s ``--mode logprob`` is extended to these
judges: ``resolve_mode()`` currently hard-blocks logprob mode for every
non-local provider, asserting the API "does not return" a token
distribution. That assumption has never been tested against GenAI4Science
specifically -- this is the test, at the cost of two 1-token completions.

WHAT THE READING MEANS
    populated    ``response.choices[0].logprobs.content[0].top_logprobs`` is a
                 non-empty list of (token, logprob) pairs -- the endpoint DOES
                 return logprobs on a chat completion, and a real
                 genai4science logprob-mode judge (mirroring the local
                 ``_score_logprob`` path) is worth building.
    empty / None the field came back null or empty -- ``resolve_mode()``'s
                 assumption holds for this endpoint too, and logprob mode
                 stays local-only.

Needs ``GENAI4SCIENCE_PERFORMANCE_API_KEY`` or ``GENAI4SCIENCE_API_KEY`` in
the environment (this repo's ``.env``), same as any other GenAI4Science call.

Run:
    uv run python epai/ai_detection/score/probe_genai4science_logprobs.py
    uv run python epai/ai_detection/score/probe_genai4science_logprobs.py --models gptoss-120b
"""

from __future__ import annotations

import argparse
import os

# One short doc, picked arbitrarily -- the point is the wiring, not the verdict.
_SAMPLE_TEXT = (
    "The committee reviewed the proposal and, after some deliberation, "
    "decided to postpone the vote until the next quarterly meeting."
)

_SYSTEM = (
    'You are a strict classifier. Read the text and answer with EXACTLY one '
    'word: either "human" or "machine". No punctuation, no explanation.'
)

MODEL_IDS = {
    "gptoss-120b": "gpt-oss:120b",
    "llama33-70b": "llama3.3:70b",
    "gemma4-31b": "gemma4:31b",
}


def probe(client, model_id: str, top_n: int) -> None:
    print(f"\n{'=' * 70}\n{model_id}\n{'=' * 70}")
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _SAMPLE_TEXT},
        ],
        max_tokens=1,
        logprobs=True,
        top_logprobs=top_n,
    )
    choice = response.choices[0]
    print(f"answer token   : {choice.message.content!r}")
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    if not content:
        print("logprobs       : NONE / empty -- endpoint did not return them")
        return
    first = content[0]
    alts = first.top_logprobs or []
    print(f"logprobs       : populated -- {len(alts)} alternative(s) for the first token")
    for alt in alts:
        print(f"    {alt.token!r:<16} logprob={alt.logprob:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models", nargs="+", default=list(MODEL_IDS), choices=list(MODEL_IDS),
                     help="which judges to probe (default: both)")
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    from openai import OpenAI

    from epai.utils.env_utils import genai4science_endpoint

    base_url, api_key_env = genai4science_endpoint()
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"{api_key_env} is not set -- see .env_sample")
    client = OpenAI(base_url=base_url, api_key=api_key)
    print(f"endpoint: {base_url}  (key: {api_key_env})")

    for alias in args.models:
        try:
            probe(client, MODEL_IDS[alias], args.top_n)
        except Exception as exc:
            print(f"\n{alias}: request failed -- {exc}")


if __name__ == "__main__":
    main()

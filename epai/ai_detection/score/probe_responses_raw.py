r"""Minimal Responses-API logprobs probe -- dump the raw object.

Isolates WHY the earlier probes saw a window of one token for ``gpt-5.6-luna``.
Every earlier call set ``include=["message.output_text.logprobs"]`` +
``reasoning={"effort":"none"}`` + ``max_output_tokens``; this one strips it back
to the bare call the user wants to see, then adds the parameters one at a time,
across prompts of increasing next-token entropy.

    from openai import OpenAI
    client = OpenAI()
    r = client.responses.create(model="gpt-5.6-luna",
                                input="The capital of France is", top_logprobs=20)
    print(r)

For each (prompt x config) it prints the full ``response.model_dump()`` for the
first config, then a compact line (window size + top tokens) for the rest, so a
one-token window on a genuinely uncertain prompt is obvious.

Needs ``OPENAI_API_KEY`` (this repo's ``.env``). Reads/writes nothing else.

    uv run python epai/ai_detection/score/probe_responses_raw.py
    uv run python epai/ai_detection/score/probe_responses_raw.py --model gpt54mini
    uv run python epai/ai_detection/score/probe_responses_raw.py --full   # full dump for every call
"""

from __future__ import annotations

import argparse
import json
import math
import os

# Prompts from near-deterministic to near-uniform next token.
_PROMPTS = [
    ("certain",   "The capital of France is"),
    ("certain-2", "Two plus two equals"),
    ("some",      "My favourite colour is"),
    ("coinflip",  "Flip a fair coin. Answer with exactly one word, heads or tails:"),
    ("uniform",   "Pick a single digit from 1 to 9 uniformly at random. Answer with just the digit:"),
]

# Parameter shapes, layered.
_CONFIGS = [
    ("bare: top_logprobs only",
     dict(top_logprobs=20)),
    ("+ include logprobs",
     dict(top_logprobs=20, include=["message.output_text.logprobs"])),
    ("+ reasoning effort none",
     dict(top_logprobs=20, include=["message.output_text.logprobs"],
          reasoning={"effort": "none"})),
    ("+ max_output_tokens=16",
     dict(top_logprobs=20, include=["message.output_text.logprobs"],
          reasoning={"effort": "none"}, max_output_tokens=16)),
    ("+ temperature=2",
     dict(top_logprobs=20, include=["message.output_text.logprobs"],
          reasoning={"effort": "none"}, max_output_tokens=16, temperature=2.0)),
]


def _first_logprobs(resp):
    """The answer-position logprob list from a Responses object, or None."""
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for part in getattr(item, "content", []) or []:
                if getattr(part, "type", None) == "output_text":
                    lp = getattr(part, "logprobs", None)
                    return lp[0] if lp else None
    return None


def _compact(resp) -> str:
    items = [getattr(i, "type", "?") for i in (getattr(resp, "output", []) or [])]
    txt = getattr(resp, "output_text", None)
    slot = _first_logprobs(resp)
    if slot is None:
        return f"output={items} text={txt!r}  logprobs: NONE"
    alts = slot.top_logprobs or []
    top = "  ".join(f"{a.token!r}:{math.exp(a.logprob):.3g}" for a in alts[:6])
    return (f"output={items} text={txt!r}  window={len(alts)}  "
            f"[{top}{'  ...' if len(alts) > 6 else ''}]")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--full", action="store_true",
                    help="print the full response.model_dump() for every call, "
                         "not just the first")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set -- see .env")
    from openai import OpenAI
    client = OpenAI()

    import openai as _openai
    print(f"openai sdk : {_openai.__version__}")
    print(f"model      : {args.model}")

    first = True
    for pname, prompt in _PROMPTS:
        print(f"\n{'=' * 74}\nPROMPT [{pname}]  {prompt!r}\n{'=' * 74}")
        for cname, kw in _CONFIGS:
            print(f"\n  -- {cname}")
            try:
                resp = client.responses.create(model=args.model, input=prompt, **kw)
            except Exception as exc:  # noqa: BLE001 -- the error IS the result
                print(f"     ERROR: {type(exc).__name__}: {exc}")
                continue
            if args.full or first:
                first = False
                try:
                    print(json.dumps(resp.model_dump(), indent=2, default=str))
                except Exception:
                    print(repr(resp))
            else:
                print(f"     {_compact(resp)}")

    print(f"\n{'=' * 74}\nWhat to look for\n{'=' * 74}")
    print("- window stays 1 even on [coinflip] / [uniform]  -> the API is "
          "returning only the sampled token; the model's entropy is not the "
          "cause. Check the search notes next.")
    print("- window > 1 on the uncertain prompts, 1 only on [certain]  -> the "
          "model really is that peaked on the judge task; a rating readout "
          "cannot help and parsed mode is the answer.")
    print("- 'bare' returns no logprobs but '+ include' does  -> include is "
          "required (known); not the bug.")


if __name__ == "__main__":
    main()

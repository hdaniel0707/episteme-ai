r"""Capability probe: can an OpenAI judge model do the logprob-mode judge?

The sibling of ``probe_genai4science_logprobs.py``, for the ``provider="openai"``
judges in ``add_llm_judge_scores.MODELS`` (``gpt56luna``, ``gpt54mini``). Nothing
here reads or writes a parquet -- it answers one question before
``add_llm_judge_scores.py``'s ``--mode logprob`` is extended to an OpenAI model:

    does the API return a usable token distribution for the answer, the way
    ``_score_genai4science_logprob`` reads one?

``resolve_mode()`` currently hard-blocks logprob mode for every non-local
provider except the ``LOGPROB_API_MODELS`` set (``gemma4:31b`` only), and
``__call__`` has no ``_score_openai_logprob`` path -- so the code is NOT ready for
``gpt56luna --mode logprob`` yet.

Two API surfaces are tried, because they behave differently on the GPT-5.x line:

  * **Chat Completions** ``logprobs=True`` -- the path ``_score_genai4science``
    uses. GPT-5.x reasoning models reject it outright
    (``'logprobs' is not supported with this model``).
  * **Responses API** ``include=["message.output_text.logprobs"]`` +
    ``reasoning={"effort": "none"}`` -- the documented way to get logprobs from a
    reasoning-capable model. Reportedly works with effort ``none`` and returns
    ``[]`` (or errors) with any real effort; structured output also suppresses
    it, but logprob mode does not use structured output, so that does not apply
    here. ``top_logprobs`` up to 20.

For every call it prints the exact outcome: the error verbatim, or the answer
text, the output-item types (is there a reasoning item?), the usage (reasoning
tokens included), and the returned ``top_logprobs`` list with linear
probabilities -- plus whether token 0 is an answer word and whether BOTH
``human`` and ``machine`` are inside the window.

VERDICT
    VIABLE      some call returned a first-token distribution whose top token is
                (a spelling of) ``human`` / ``machine`` and both answer words are
                reachable -- a ``_score_openai_logprob`` is worth building
                (mirror ``_score_genai4science_logprob``; note the Responses API
                vs Chat Completions shape).
    PARTIAL     logprobs come back but the first token is a reasoning / format
                token, or only one answer word appears -- usable only with more
                work (forced answer prefix, logit_bias, bigger window).
    NOT VIABLE  every attempt errors or returns no logprobs -- ``resolve_mode()``'s
                block is correct for this model; stay on ``--mode parsed``.

Needs ``OPENAI_API_KEY`` in the environment (this repo's ``.env``).

    uv run python epai/ai_detection/score/probe_openai_logprobs.py
    uv run python epai/ai_detection/score/probe_openai_logprobs.py --model gpt54mini
    uv run python epai/ai_detection/score/probe_openai_logprobs.py --toy --top-n 20
"""

from __future__ import annotations

import argparse
import math
import os

from epai.ai_detection.score.add_llm_judge_scores import (
    LOGPROB_API_TOP_N,
    MODELS,
    USER_TEMPLATE,
    _LP_HUMAN_SET,
    _LP_MACHINE_SET,
    _norm_lp_token,
    system_prompt_for,
)

_OPENAI_ALIASES = [k for k, m in MODELS.items() if m.provider == "openai"]

# One short document per class, so a returned distribution shows shape, not just
# whether the field populates. Deliberately tiny -- the probe is about wiring.
_DOCS = {
    "human-ish": (
        "honestly not sure why this took me three tries. the wiring diagram in "
        "the manual is wrong -- pin 2 and pin 5 are swapped -- and i only worked "
        "it out after the second fuse went."
    ),
    "machine-ish": (
        "In conclusion, the proposed framework offers a robust and scalable "
        "solution. By leveraging these key advantages, organizations can "
        "effectively navigate the evolving landscape and unlock significant "
        "value across multiple dimensions."
    ),
    "neutral": (
        "The committee reviewed the proposal and, after some deliberation, "
        "decided to postpone the vote until the next quarterly meeting."
    ),
}

_TOY_SYSTEM = (
    'You are a strict classifier. Read the text and answer with EXACTLY one '
    'word: either "human" or "machine". No punctuation, no explanation.'
)


def _analyse_logprobs(lp_list) -> bool:
    """Print the distribution for the first generated token. ``lp_list`` items
    need ``.token`` and ``.top_logprobs`` (list of ``.token`` / ``.logprob``);
    both the Chat Completions ``logprobs.content`` and the Responses API
    ``output_text.logprobs`` have that shape. Returns True if token 0 is an
    answer word and at least one answer word is in the window."""
    if not lp_list:
        print("      logprobs: NONE / empty -- not returned for this call")
        return False
    slot = lp_list[0]
    alts = slot.top_logprobs or []
    tok0 = _norm_lp_token(slot.token)
    is_answer = tok0 in _LP_HUMAN_SET or tok0 in _LP_MACHINE_SET
    print(f"      token 0 : {slot.token!r}  (normalised {tok0!r}) -- "
          f"{'an answer word' if is_answer else 'NOT an answer word'}")
    print(f"      window  : {len(alts)} alternative(s)")
    human = machine = 0.0
    for a in alts:
        p = math.exp(a.logprob)
        n = _norm_lp_token(a.token)
        tag = "  <-human" if n in _LP_HUMAN_SET else ("  <-machine" if n in _LP_MACHINE_SET else "")
        print(f"        {a.token!r:<18} logprob={a.logprob:8.4f}  p={p:.4g}{tag}")
        if n in _LP_HUMAN_SET:
            human += p
        elif n in _LP_MACHINE_SET:
            machine += p
    mass = human + machine
    both = human > 0 and machine > 0
    print(f"      answer words in window: human={'yes' if human else 'no'}, "
          f"machine={'yes' if machine else 'no'}  (mass {mass:.3f})")
    if mass > 0:
        print(f"      -> P(human) ratio = {human / mass:.4f}"
              + ("" if both else "   [SATURATED -- losing word outside the window]"))
    got_h = any(_norm_lp_token(a.token) in _LP_HUMAN_SET for a in alts)
    got_m = any(_norm_lp_token(a.token) in _LP_MACHINE_SET for a in alts)
    return is_answer and (got_h or got_m)


def _print_usage(resp) -> None:
    u = getattr(resp, "usage", None)
    if u is None:
        return
    try:
        print(f"      usage   : {u.model_dump()}")
    except Exception:
        print(f"      usage   : {u}")


# --------------------------------------------------------------------------
# Chat Completions
# --------------------------------------------------------------------------


def _call_chat(client, model_id, system, doc, *, label, **kw) -> bool:
    print(f"\n  -- [chat] {label}")
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": USER_TEMPLATE.format(text=doc)}],
            **kw,
        )
    except Exception as exc:  # noqa: BLE001 -- the error text IS the result
        print(f"      ERROR: {type(exc).__name__}: {exc}")
        return False
    ch = resp.choices[0]
    print(f"      content : {ch.message.content!r}")
    print(f"      finish  : {ch.finish_reason}")
    _print_usage(resp)
    lp = getattr(ch, "logprobs", None)
    return _analyse_logprobs(getattr(lp, "content", None) if lp is not None else None)


# --------------------------------------------------------------------------
# Responses API
# --------------------------------------------------------------------------


def _responses_output_text_part(resp):
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for part in getattr(item, "content", []) or []:
                if getattr(part, "type", None) == "output_text":
                    return part
    return None


def _call_responses(client, model_id, system, doc, *, label, effort, top_n) -> bool:
    print(f"\n  -- [responses] {label}")
    try:
        resp = client.responses.create(
            model=model_id,
            instructions=system,
            input=USER_TEMPLATE.format(text=doc),
            include=["message.output_text.logprobs"],
            top_logprobs=top_n,
            reasoning={"effort": effort},
            max_output_tokens=16,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"      ERROR: {type(exc).__name__}: {exc}")
        return False
    items = [getattr(i, "type", "?") for i in (getattr(resp, "output", []) or [])]
    print(f"      output items : {items}")
    print(f"      output_text  : {getattr(resp, 'output_text', None)!r}")
    _print_usage(resp)
    part = _responses_output_text_part(resp)
    if part is None:
        print("      no output_text part in the response")
        return False
    return _analyse_logprobs(getattr(part, "logprobs", None))


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="gpt56luna", choices=_OPENAI_ALIASES,
                    help="OpenAI judge alias to probe (default: gpt56luna)")
    ap.add_argument("--prompt", default="v3",
                    help="prompt id for the real system prompt (default: v3)")
    ap.add_argument("--toy", action="store_true",
                    help="use a one-line generic classifier prompt instead of "
                         "system_prompt_for(prompt, 'logprob')")
    ap.add_argument("--top-n", type=int, default=LOGPROB_API_TOP_N,
                    help=f"top_logprobs window (default {LOGPROB_API_TOP_N}, max 20)")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set -- see .env / .env_sample")
    from openai import OpenAI

    spec = MODELS[args.model]
    model_id = spec.model
    system = _TOY_SYSTEM if args.toy else system_prompt_for(args.prompt, "logprob")
    client = OpenAI()

    print(f"model    : {args.model}  ->  {model_id}")
    print(f"prompt   : {'toy one-liner' if args.toy else f'system_prompt_for({args.prompt!r}, logprob)'}")
    print(f"top_n    : {args.top_n}")
    print("=" * 72)

    any_ok = False

    print("\n[control] Responses API, no logprobs -- does the key + model work?")
    for name, doc in _DOCS.items():
        _call_responses(client, model_id, system, doc, label=f"control / {name}",
                        effort="none", top_n=args.top_n)  # include still set; fine

    print(f"\n{'=' * 72}\n[A] Chat Completions -- logprobs=True (the _score_genai4science path)\n{'=' * 72}")
    for name, doc in _DOCS.items():
        any_ok |= _call_chat(client, model_id, system, doc, label=name,
                             max_completion_tokens=16, logprobs=True,
                             top_logprobs=args.top_n)

    print(f"\n{'=' * 72}\n[B] Chat Completions -- logprobs=True + reasoning_effort='none'\n{'=' * 72}")
    for name, doc in _DOCS.items():
        any_ok |= _call_chat(client, model_id, system, doc, label=name,
                             max_completion_tokens=16, logprobs=True,
                             top_logprobs=args.top_n, reasoning_effort="none")

    print(f"\n{'=' * 72}\n[C] Responses API -- include logprobs, reasoning effort 'none'\n{'=' * 72}")
    for name, doc in _DOCS.items():
        any_ok |= _call_responses(client, model_id, system, doc, label=name,
                                  effort="none", top_n=args.top_n)

    print(f"\n{'=' * 72}\n[D] Responses API -- same, reasoning effort 'low' (control: should suppress)\n{'=' * 72}")
    for name, doc in _DOCS.items():
        _call_responses(client, model_id, system, doc, label=name,
                        effort="low", top_n=args.top_n)

    print(f"\n{'=' * 72}\nVERDICT\n{'=' * 72}")
    if any_ok:
        print("VIABLE -- a call returned a first-token distribution whose top "
              "token is an answer word. Build _score_openai_logprob; note WHICH "
              "path worked (Chat Completions vs Responses API) and mirror its "
              "response shape. Add the model id to LOGPROB_API_MODELS (or a new "
              "LOGPROB_OPENAI_MODELS) and smoke test like run_gemma_logprob_judge.py.")
    else:
        print("NOT VIABLE (or PARTIAL) -- no call gave a usable answer-word "
              "distribution. Read the errors: an explicit 'logprobs not "
              "supported' / empty logprobs on every path means resolve_mode()'s "
              "block is correct for this model -- stay on --mode parsed. If a "
              "distribution came back but token 0 was a reasoning / format "
              "token, a forced answer prefix or logit_bias would be needed.")


if __name__ == "__main__":
    main()

r"""Which request switches gemma-4-31b's optional thinking mode on, on the
GenAI4Science endpoint (LiteLLM -> Ollama)? No sidecar is written.

Sends the paper's BaseJudge call (JudgeVerdict structured answer, endpoint
defaults) for a few texts, once per variant:

    off            the paper's call unchanged (baseline: no thinking expected)
    think_param    extra_body {"think": true}                (Ollama's own switch)
    effort_high    reasoning_effort="high"                   (LiteLLM -> Ollama think)
    template_kw    extra_body chat_template_kwargs.enable_thinking=true  (vLLM style)
    think_token    "<|think|>" at the start of the system prompt (Gemma 4's template token)

and prints, per call: seconds, completion / reasoning tokens, whether a
reasoning field or thought text came back, and the parsed answer. A variant
switches thinking on if it shows reasoning tokens, a reasoning_content field, or
thought text, and takes clearly longer than ``off``.

    .venv/bin/python -m epai.ai_detection.rebuttal.probe_gemma_thinking [--n 3]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from epai.ai_detection.rebuttal.capture import _build_client
from epai.ai_detection.rebuttal.conditions import CONDITIONS, HUMAN, build_subset, load_core
from epai.ai_detection.score.add_llm_judge_scores import (
    GENAI4SCIENCE_MAX_TOKENS,
    MODELS,
    USER_TEMPLATE,
    JudgeVerdict,
)

VARIANTS = {
    "off": {},
    "think_param": {"extra_body": {"think": True}},
    "effort_high": {"reasoning_effort": "high"},
    "template_kw": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    "think_token": {"_system_prefix": "<|think|>\n"},
}


def _call(client, model_id, system, text, kw):
    kw = dict(kw)
    prefix = kw.pop("_system_prefix", "")
    t0 = time.time()
    resp = client.chat.completions.parse(
        model=model_id,
        messages=[{"role": "system", "content": prefix + system},
                  {"role": "user", "content": USER_TEMPLATE.format(text=text)}],
        response_format=JudgeVerdict, max_tokens=GENAI4SCIENCE_MAX_TOKENS, **kw)
    dt = time.time() - t0
    msg = resp.choices[0].message
    u = resp.usage
    det = getattr(u, "completion_tokens_details", None)
    extra = msg.model_dump()
    reasoning_field = {k: v for k, v in extra.items()
                       if k in ("reasoning_content", "reasoning", "thinking") and v}
    content = msg.content or ""
    return {
        "s": round(dt, 1),
        "completion_tokens": getattr(u, "completion_tokens", None),
        "reasoning_tokens": getattr(det, "reasoning_tokens", None),
        "reasoning_field": {k: str(v)[:120] for k, v in reasoning_field.items()},
        "thought_in_content": any(t in content for t in ("<|channel>", "thought", "<think")),
        "content": content[:120],
        "parsed": None if msg.parsed is None else msg.parsed.human_probability,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3, help="texts (alternating human / machine)")
    ap.add_argument("--only", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    args = ap.parse_args()

    spec = MODELS["gemma4-31b"]
    cond = CONDITIONS["base_default"]          # the paper's BaseJudge prompt and call
    rows = build_subset(load_core(), "humans_plus_third")
    rng = np.random.default_rng(1)
    h = rows[rows["generated_by"].eq(HUMAN)]
    m = rows[~rows["generated_by"].eq(HUMAN)]
    pick = [h.iloc[rng.integers(len(h))] if i % 2 == 0 else m.iloc[rng.integers(len(m))]
            for i in range(args.n)]
    client = _build_client(spec.provider, spec.model)
    for name in args.only:
        print(f"\n=== {name}  {VARIANTS[name]}")
        for r in pick:
            try:
                out = _call(client, spec.model, cond.system, r["text"], VARIANTS[name])
                print(f"  {r['uid'][:40]:40} [{r['generated_by'][:10]}] "
                      + json.dumps(out, ensure_ascii=False))
            except Exception as exc:  # noqa: BLE001
                print(f"  {r['uid'][:40]:40} FAIL {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    main()

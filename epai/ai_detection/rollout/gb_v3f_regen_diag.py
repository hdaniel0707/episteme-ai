r"""Throwaway: find out why generate.py's OpenAI call for gpt-5.6-luna 400s.

Run in the fork's environment (it has ``openai`` and the key handling):

    uv run --project external/ghostbuster python \
        epai/ai_detection/rollout/gb_v3f_regen_diag.py

Tries the chat call four ways — the fork's current ``max_tokens=16384``, a
smaller ``max_tokens``, ``max_completion_tokens`` instead, and no cap — and
prints which succeed. The pattern says what the model wants; an identical
failure on all four points at the model id or the key/endpoint instead.
Delete this file once the real regen call is fixed.
"""

from __future__ import annotations

import os

MODEL = "gpt-5.6-luna"
MSGS = [{"role": "user", "content": "Write one sentence about the sea."}]

ATTEMPTS = [
    ("max_tokens=16384", {"max_tokens": 16384}),
    ("max_tokens=4096", {"max_tokens": 4096}),
    ("max_completion_tokens=16384", {"max_completion_tokens": 16384}),
    ("no token cap", {}),
]


def main() -> None:
    try:
        from dotenv import load_dotenv

        for env in (".env", "external/ghostbuster/.env"):
            load_dotenv(env)
    except Exception:
        pass

    import openai

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not in env (checked .env and "
                         "external/ghostbuster/.env)")
    print(f"key: ...{key[-4:]}   base_url: {os.environ.get('OPENAI_BASE_URL') or 'default (api.openai.com)'}")
    print(f"model: {MODEL}\n")

    client = openai.OpenAI(api_key=key)
    for label, kw in ATTEMPTS:
        try:
            r = client.chat.completions.create(model=MODEL, messages=MSGS, **kw)
            print(f"  {label:30} OK   -> {r.choices[0].message.content!r}")
        except Exception as e:
            print(f"  {label:30} {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

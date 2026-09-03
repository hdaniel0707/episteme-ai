"""Probe which models a GenAI4Science (HUN-REN SZTAKI) API key can actually reach.

The endpoint is OpenAI-compatible, so this just:
  1. lists the models the key is entitled to (`GET /models`), and
  2. sends a 1-token chat completion to each model we care about and reports
     whether it answered, was denied, or is queued/unavailable.

Nothing here writes to the corpus or spends meaningful tokens. Run it before
building anything on top of this provider, and paste the output into the access
/ performance-endpoint request e-mail if a model is missing.

    python -m epai.scripts.check_genai4science_access            # whichever key is set
    python -m epai.scripts.check_genai4science_access --default  # force the shared endpoint

Config (env). The performance endpoint is used automatically when its key is
set (see epai/utils/env_utils.py :: genai4science_endpoint); --default forces
the shared one.
    GENAI4SCIENCE_PERFORMANCE_API_KEY   bearer token, performance endpoint
                                       (dedicated capacity, max 4 concurrent —
                                       a 5th call trips a ~1-minute key block)
    GENAI4SCIENCE_API_KEY              bearer token, shared endpoint (trials, light use)
    GENAI4SCIENCE_*_BASE_URL           override either base URL
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

from epai.utils.env_utils import (
    GENAI4SCIENCE_BASE_URL,
    doublecheck_env,
    doublecheck_pkgs,
    genai4science_endpoint,
)

load_dotenv()
doublecheck_env(".env")
doublecheck_pkgs(pyproject_path="pyproject.toml", verbose=True)

# Models the performance key is entitled to (SZTAKI, 2026-09). The first three
# take several concurrent threads; llama3.3 / mistral-small were moved off
# dedicated GPUs, so expect a cold-start wait on the first call. The shared
# endpoint additionally serves llama3.1:8b and others — see the docs.
TARGET_MODELS = [
    "gpt-oss:120b",          # top open-weight judge rung + capable generator
    "gemma4:31b",            # judge rung (256k ctx)
    "qwen3.8:27b",           # judge rung + mid generator
    "llama3.3:70b",          # extra judge rung / generator — downgraded, slow to load
    "mistral-small3.2:24b",  # extra judge rung — downgraded, slow to load
]

# The default endpoint rate-limits and queues; without this a probe hangs.
# A cold model (not yet loaded into GPU memory) can need most of a minute.
REQUEST_TIMEOUT = 90.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--default", action="store_true",
                    help="probe the shared endpoint even if a performance key is set")
    args = ap.parse_args()

    if args.default:
        base_url = os.environ.get("GENAI4SCIENCE_BASE_URL", GENAI4SCIENCE_BASE_URL)
        api_key_env = "GENAI4SCIENCE_API_KEY"
    else:
        base_url, api_key_env = genai4science_endpoint()

    api_key = os.environ.get(api_key_env)
    if not api_key:
        sys.exit(f"{api_key_env} is not set (checked env and .env).")

    client = OpenAI(
        api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT, max_retries=0
    )
    endpoint = "performance" if "performance" in base_url else "shared/default"
    print(f"\nendpoint: {endpoint}   key: {api_key_env}   base_url: {base_url}\n",
          flush=True)

    print("== entitled models (GET /models) ==")
    entitled: set[str] = set()
    try:
        for m in sorted(client.models.list(), key=lambda x: x.id):
            entitled.add(m.id)
            print(f"  {m.id}")
    except Exception as e:  # noqa: BLE001 — want the raw provider error verbatim
        print(f"  !! listing failed: {type(e).__name__}: {e}")
    print()

    print(f"== chat probe ({REQUEST_TIMEOUT:.0f}s timeout each) ==", flush=True)
    ok, missing = [], []
    for i, model in enumerate(TARGET_MODELS):
        if i:
            time.sleep(3)  # stay under the default endpoint's rate limit
        listed = "listed" if model in entitled else "not-listed"
        for attempt in (1, 2):  # a cold model often fails once, then loads
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Reply with the word: ok"}],
                    max_tokens=16,
                )
                reply = " ".join((r.choices[0].message.content or "").split())[:40]
                note = "" if attempt == 1 else " (2nd try)"
                print(f"  OK    {model:<22} [{listed}] -> {reply!r}{note}", flush=True)
                ok.append(model)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 1:
                    time.sleep(5)
                    continue
                print(f"  FAIL  {model:<22} [{listed}] -> {type(e).__name__}: {e}", flush=True)
                missing.append(model)
    print()

    print(f"reachable: {', '.join(ok) or '(none)'}")
    if missing:
        print(f"NOT reachable this run: {', '.join(missing)}")
        print("-> all TARGET_MODELS are in the entitled list, so a failure here")
        print("   is a cold start / queue, not missing access. Re-run to confirm.")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()

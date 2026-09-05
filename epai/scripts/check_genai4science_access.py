"""Probe which models a GenAI4Science (HUN-REN SZTAKI) API key can actually reach.

The endpoint is OpenAI-compatible, so this just:
  1. lists the models the key is entitled to (`GET /models`), and
  2. sends a 1-token chat completion to each model we care about and reports
     whether it answered, was denied, or is queued/unavailable.
  3. with --perf, times a realistic generation on each model so a slow one can
     be swapped for a faster peer.

Nothing here writes to the corpus or spends meaningful tokens. Run it before
building anything on top of this provider, and paste the output into the access
/ performance-endpoint request e-mail if a model is missing.

    python -m epai.scripts.check_genai4science_access            # whichever key is set
    python -m epai.scripts.check_genai4science_access --default  # force the shared endpoint
    python -m epai.scripts.check_genai4science_access --perf     # + inference-time benchmark
    python -m epai.scripts.check_genai4science_access --perf --skip llama  # llama is offline
    python -m epai.scripts.check_genai4science_access --perf --perf-model qwen3.8:27b gemma4:31b

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
import statistics
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

from epai.utils.env_utils import (
    GENAI4SCIENCE_BASE_URL,
    doublecheck_env,
    doublecheck_pkgs,
    genai4science_endpoint,
    qwen3_thinking_off,
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

# --perf: a request shaped like the real generate stage — one science title,
# asked for an abstract of a bounded length. The point is a like-for-like
# wall-time / tokens-per-second comparison across models, so the prompt and the
# cap are the same for every one of them.
PERF_SYSTEM = "You are a researcher writing the abstract of a paper."
PERF_USER = (
    "Write the abstract (about 200 words, one paragraph, no title, no headings) "
    "of a research paper titled: 'Contrastive Representation Learning for "
    "Low-Resource Neural Machine Translation'."
)
# High enough that a non-thinking abstract (~260 tok) finishes well inside it,
# and a Qwen3 <think> pass usually completes too — so both numbers are real
# rather than "hit the cap". The real generate call is uncapped, so a [thinking]
# row here is still a lower bound on how slow that mode is in the pipeline.
PERF_MAX_TOKENS = 1000
PERF_TIMEOUT = 300.0

_DIM = "\033[2m"
_RESET = "\033[0m"


def _one_perf_call(client: OpenAI, model: str, *, think: bool,
                   max_tokens: int) -> tuple[float, int | None]:
    """One realistic generation. Returns (wall_seconds, completion_tokens|None)."""
    messages = [
        {"role": "system", "content": PERF_SYSTEM},
        {"role": "user", "content": PERF_USER},
    ]
    if not think:
        messages = qwen3_thinking_off(model, messages)
    t0 = time.monotonic()
    r = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens,
    )
    wall = time.monotonic() - t0
    usage = getattr(r, "usage", None)
    toks = getattr(usage, "completion_tokens", None) if usage else None
    return wall, toks


def run_perf(client: OpenAI, models: list[str], runs: int,
             max_tokens: int = PERF_MAX_TOKENS) -> None:
    """Time `runs` realistic generations per model and print a speed table.

    Run 1 of each model includes cold model-load on this endpoint (llama3.3 /
    mistral-small are off dedicated GPUs), so it is timed but excluded from the
    median. For a Qwen3 model both modes are measured — with `/no_think` (how
    the pipeline calls it) and with its default `<think>` pass — because that
    gap is usually the whole story.
    """
    print(f"\n== inference performance ({runs} run(s), ~200-word abstract, "
          f"max_tokens={max_tokens}) ==", flush=True)
    print(f"  {'model':<26} {'run':>4} {'wall':>9} {'out_tok':>8} {'tok/s':>8}")

    summary: list[tuple[str, float, float]] = []
    variants = []
    for model in models:
        variants.append((model, model, False))
        if "qwen3" in model.lower():
            variants.append((f"{model} [thinking]", model, True))

    for i, (label, model, think) in enumerate(variants):
        if i:
            time.sleep(3)
        walls: list[float] = []
        for run in range(1, runs + 1):
            try:
                wall, toks = _one_perf_call(client, model, think=think,
                                            max_tokens=max_tokens)
            except Exception as e:  # noqa: BLE001
                print(f"  {label:<26} {run:>4}   FAIL  {type(e).__name__}: "
                      f"{str(e).splitlines()[0][:60]}", flush=True)
                walls = []
                break
            walls.append(wall)
            tps = (toks / wall) if toks and wall else 0.0
            tag = "  cold" if run == 1 else ""
            print(f"  {label:<26} {run:>4} {wall:>8.1f}s {str(toks or '?'):>8} "
                  f"{tps:>8.1f}{tag}", flush=True)
        timed = walls[1:] or walls  # drop the cold run when there is more than one
        if timed:
            med = statistics.median(timed)
            print(f"  {'':<26} {'':>4} {_DIM}median {med:.1f}s over "
                  f"{len(timed)} warm run(s){_RESET}")
            summary.append((label, med, min(timed)))

    if len(summary) > 1:
        print("\n  fastest first (median warm wall):")
        for label, med, best in sorted(summary, key=lambda x: x[1]):
            print(f"    {label:<26} {med:>7.1f}s   (best {best:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--default", action="store_true",
                    help="probe the shared endpoint even if a performance key is set")
    ap.add_argument("--perf", action="store_true",
                    help="also time a realistic generation on each reachable model")
    ap.add_argument("--perf-model", nargs="+", metavar="MODEL", default=None,
                    help="models to benchmark (default: the reachable TARGET_MODELS)")
    ap.add_argument("--perf-runs", type=int, default=3, metavar="N",
                    help="generations to time per model; run 1 is the cold one (default 3)")
    ap.add_argument("--perf-max-tokens", type=int, default=PERF_MAX_TOKENS, metavar="N",
                    help=f"completion cap for the perf calls (default {PERF_MAX_TOKENS})")
    ap.add_argument("--skip", nargs="+", default=[], metavar="MODEL",
                    help="skip models whose id contains one of these (substring), "
                         "in both the chat probe and --perf — e.g. --skip llama "
                         "while that endpoint is offline")
    args = ap.parse_args()

    def _skipped(model: str) -> bool:
        return any(s.lower() in model.lower() for s in args.skip)

    targets = [m for m in TARGET_MODELS if not _skipped(m)]
    if args.skip:
        dropped = [m for m in TARGET_MODELS if _skipped(m)]
        print(f"\n--skip: not probing {', '.join(dropped) or '(nothing matched)'}")

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
    for i, model in enumerate(targets):
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
        print("-> a probed model in the entitled list that fails here is a cold")
        print("   start / queue, not missing access. Re-run to confirm.")

    if args.perf:
        if args.perf_model:
            perf_models = [m for m in args.perf_model if not _skipped(m)]
        else:
            perf_models = [m for m in targets if m in ok]
        if not perf_models:
            print("\n-- perf: no reachable model to benchmark --")
        else:
            perf_client = OpenAI(api_key=api_key, base_url=base_url,
                                 timeout=PERF_TIMEOUT, max_retries=0)
            run_perf(perf_client, perf_models, max(1, args.perf_runs),
                     max_tokens=args.perf_max_tokens)

    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()

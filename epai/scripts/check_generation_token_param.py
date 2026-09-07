r"""Which output-length parameter each generation endpoint accepts.

The Ghostbuster / science generators call ``chat.completions.create`` with an
output cap. OpenAI's GPT-5 family (``gpt-5.6-luna``) rejects the legacy
``max_tokens`` and requires ``max_completion_tokens``; the GenAI4Science-hosted
open-weight models still take ``max_tokens``. This probes both so a caller knows
which name to send — and whether the cap value itself (16384 by default) is
within a model's limit.

Each model is hit five ways with a one-line prompt:
  max_tokens=64 · max_tokens=<cap> · max_completion_tokens=64 ·
  max_completion_tokens=<cap> · (no cap)
and a verdict is printed per model. Tokens spent are negligible; some calls are
billed (OpenAI).

    python -m epai.scripts.check_generation_token_param
    python -m epai.scripts.check_generation_token_param --openai-only
    python -m epai.scripts.check_generation_token_param --skip llama --cap 8192
    python -m epai.scripts.check_generation_token_param --default   # shared G4S endpoint

Env (see epai/utils/env_utils.py :: genai4science_endpoint):
    OPENAI_API_KEY
    GENAI4SCIENCE_PERFORMANCE_API_KEY / GENAI4SCIENCE_API_KEY
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from epai.utils.env_utils import genai4science_endpoint

BASE_DIR = Path(__file__).resolve().parents[2]

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

OPENAI_MODEL = "gpt-5.6-luna"
G4S_MODELS = ["gpt-oss:120b", "gemma4:31b", "llama3.3:70b"]


def attempts(cap: int) -> list[tuple[str, dict]]:
    return [
        ("max_tokens=64", {"max_tokens": 64}),
        (f"max_tokens={cap}", {"max_tokens": cap}),
        ("max_completion_tokens=64", {"max_completion_tokens": 64}),
        (f"max_completion_tokens={cap}", {"max_completion_tokens": cap}),
        ("no cap", {}),
    ]


def probe(client, model: str, cap: int) -> dict[str, str]:
    """label -> 'ok' or 'ERR: <first line>' for each attempt."""
    msgs = [{"role": "user", "content": "Write one short sentence about the sea."}]
    out: dict[str, str] = {}
    for label, kw in attempts(cap):
        try:
            r = client.chat.completions.create(model=model, messages=msgs, **kw)
            txt = (r.choices[0].message.content or "").strip()
            out[label] = "ok" if txt else "ok (empty)"
        except Exception as e:  # noqa: BLE001
            first = (str(e).splitlines() or [""])[0]
            out[label] = f"ERR: {first[:160]}"
    return out


def verdict(res: dict[str, str], cap: int, needs: str) -> tuple[bool, str]:
    """(ok, message). ``needs`` is the parameter this endpoint's code path sends."""
    def okc(label: str) -> bool:
        return res[label].startswith("ok")

    ours = f"{needs}={cap}"
    other = "max_completion_tokens" if needs == "max_tokens" else "max_tokens"
    other_ok = okc(f"{other}={cap}")

    if okc(ours):
        tail = (f"; {other} also works" if other_ok
                else f"; {other} rejected (path is correct as-is)")
        return True, f"{GREEN}{needs} works at {cap}{RESET}{DIM}{tail}{RESET}"

    if okc(f"{needs}=64"):
        return False, (f"{RED}{needs} rejected at {cap}{RESET} but ok at 64 — "
                       f"lower the cap for this model, or use "
                       + (f"{other} (works)" if other_ok else "another model"))
    if other_ok:
        return False, (f"{RED}{needs} rejected{RESET}; only {other} works — the "
                       f"code path for this endpoint sends the wrong name")
    return False, f"{RED}no call succeeded{RESET} — model id, key, or endpoint problem"


def run_endpoint(name: str, client, models: list[str], cap: int, needs: str) -> bool:
    print(f"\n{BOLD}{name}{RESET}  {DIM}code path sends {needs}{RESET}")
    clean = True
    for model in models:
        res = probe(client, model, cap)
        print(f"  {BOLD}{model}{RESET}")
        for label, _ in attempts(cap):
            r = res[label]
            colour = GREEN if r.startswith("ok") else RED
            print(f"    {label:32} {colour}{r}{RESET}")
        ok, msg = verdict(res, cap, needs)
        print(f"    {DIM}->{RESET} {msg}")
        clean &= ok
    return clean


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cap", type=int, default=16384,
                    help="the 'large' cap value to test (default 16384, the "
                         "generators' MAX_OUTPUT_TOKENS)")
    ap.add_argument("--openai-model", default=OPENAI_MODEL)
    ap.add_argument("--g4s-models", nargs="+", default=G4S_MODELS)
    ap.add_argument("--skip", nargs="+", default=[], metavar="SUBSTR",
                    help="skip any model whose name contains one of these")
    ap.add_argument("--openai-only", action="store_true")
    ap.add_argument("--g4s-only", action="store_true")
    ap.add_argument("--default", action="store_true",
                    help="force the shared GenAI4Science endpoint")
    args = ap.parse_args()

    for env in (BASE_DIR / ".env", BASE_DIR / "external/ghostbuster/.env"):
        load_dotenv(env)

    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit(f"{RED}openai package not importable{RESET}")

    def keep(m: str) -> bool:
        return not any(s.lower() in m.lower() for s in args.skip)

    ok = True
    ran = 0

    if not args.g4s_only:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print(f"{YELLOW}OPENAI_API_KEY not set — skipping OpenAI{RESET}")
        elif keep(args.openai_model):
            ran += 1
            ok &= run_endpoint(
                f"OpenAI direct  (…{key[-4:]})",
                OpenAI(api_key=key, max_retries=0),
                [args.openai_model], args.cap, needs="max_completion_tokens")

    if not args.openai_only:
        if args.default:
            os.environ.pop("GENAI4SCIENCE_PERFORMANCE_API_KEY", None)
        base_url, key_env = genai4science_endpoint()
        key = os.environ.get(key_env)
        models = [m for m in args.g4s_models if keep(m)]
        if not key:
            print(f"\n{YELLOW}{key_env} not set — skipping GenAI4Science{RESET}")
        elif models:
            ran += 1
            ok &= run_endpoint(
                f"GenAI4Science  ({key_env}, {base_url})",
                OpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=60),
                models, args.cap, needs="max_tokens")

    print()
    if not ran:
        raise SystemExit(f"{RED}nothing tested — no keys, or everything "
                         f"skipped.{RESET}")
    if ok:
        print(f"{GREEN}OK — every endpoint's generator sends a parameter it "
              f"accepts at cap {args.cap}.{RESET}\n"
              f"{DIM}  OpenAI path: max_completion_tokens (gb_v3f_regen.py).  "
              f"GenAI4Science path: max_tokens (the ghostbuster fork).{RESET}")
    else:
        print(f"{YELLOW}At least one model needs attention — see its verdict "
              f"above.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()

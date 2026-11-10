r"""Run one rebuttal condition (``conditions.py``) on one row subset, and store
every answer with the top-20 log-probabilities of every output token.

One sidecar per (model, condition, subset), keyed by ``(stem, uid, rep)`` (the
2992 human texts shared by ghostbuster_orig and manugram_gb have the same uid in
both, with slightly different text, so uid alone is not a row key):

    data/parquet/tmp/rebuttal/<model>/<condition>__<subset>.parquet

with columns uid, stem, rep, answer (raw text), score (quick score: the parsed
number for ``prob`` conditions, (1 + p_h - p_m)/2 at position 0 for ``label``
ones), tokens (list<struct{token, logprob, top: list<struct{token, logprob}>}>),
error. The master parquets are only read. Re-running the same command resumes:
rows with an answer are skipped, and failed rows are retried.

Call settings, identical for every condition (so that only the prompt differs):
  gpt56luna   OpenAI Responses API, reasoning effort none, temperature from the
              condition, top_logprobs=20, include message.output_text.logprobs
  gemma4-31b  GenAI4Science chat completions, logprobs=True, top_logprobs=20

Order of use (see RUNBOOK.md):
    .venv/bin/python -m epai.ai_detection.rebuttal.capture --list
    .venv/bin/python -m epai.ai_detection.rebuttal.capture --model gpt56luna \
        --condition base_hp4 --dry-run
    .venv/bin/python -m epai.ai_detection.rebuttal.capture --model gpt56luna \
        --condition base_hp4 --probe 12          # prints answers + token windows, writes nothing
    .venv/bin/python -m epai.ai_detection.rebuttal.capture --model gpt56luna \
        --condition base_hp4 --yes
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from epai.ai_detection.rebuttal.conditions import (
    CONDITIONS,
    HUMAN,
    SUBSETS,
    WORKSPACE,
    Condition,
    build_subset,
    load_core,
)
from epai.ai_detection.score.add_llm_judge_scores import (
    BACKOFF_BASE,
    MODELS,
    RETRIES,
    USER_TEMPLATE,
)

TOP_N = 20
JUDGES = ("gpt56luna", "gemma4-31b")
CHARS_PER_TOKEN = 4.2  # measured on these corpora; for the cost estimate only


# ---------------------------------------------------------------------------
# One call -> (answer text, per-token windows)
# ---------------------------------------------------------------------------


def _call_openai(client, model_id: str, cond: Condition, text: str):
    resp = client.responses.create(
        model=model_id, instructions=cond.system,
        input=USER_TEMPLATE.format(text=text),
        include=["message.output_text.logprobs"], top_logprobs=TOP_N,
        reasoning={"effort": "none"}, temperature=cond.temperature,
        max_output_tokens=cond.max_output_tokens,
    )
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", []) or []:
            if getattr(part, "type", None) != "output_text":
                continue
            lps = getattr(part, "logprobs", None)
            if not lps:
                raise ValueError("Responses API returned no logprobs")
            toks = [{"token": t.token, "logprob": float(t.logprob),
                     "top": [{"token": a.token, "logprob": float(a.logprob)}
                             for a in (t.top_logprobs or [])]} for t in lps]
            return part.text, toks, getattr(resp, "model", None)
    raise ValueError("no output_text part in the Responses object")


def _call_genai4science(client, model_id: str, cond: Condition, text: str):
    from epai.utils.env_utils import qwen3_thinking_off
    messages = qwen3_thinking_off(model_id, [
        {"role": "system", "content": cond.system},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ])
    resp = client.chat.completions.create(
        model=model_id, messages=messages, temperature=cond.temperature,
        max_tokens=1 if cond.kind == "label" else cond.max_output_tokens,
        logprobs=True, top_logprobs=TOP_N,
    )
    ch = resp.choices[0]
    content = getattr(getattr(ch, "logprobs", None), "content", None)
    if not content:
        raise ValueError("endpoint returned no logprobs")
    toks = [{"token": t.token, "logprob": float(t.logprob),
             "top": [{"token": a.token, "logprob": float(a.logprob)}
                     for a in (t.top_logprobs or [])]} for t in content]
    return ch.message.content or "", toks, getattr(resp, "model", None)


def _build_client(provider: str, model_id: str):
    from openai import OpenAI
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set -- see .env")
        return OpenAI()
    from epai.utils.env_utils import genai4science_endpoint, report_genai4science_endpoint
    report_genai4science_endpoint(probe_model=model_id)
    base_url, key_env = genai4science_endpoint()
    if not os.environ.get(key_env):
        raise SystemExit(f"{key_env} is not set -- see .env")
    return OpenAI(base_url=base_url, api_key=os.environ[key_env])


# ---------------------------------------------------------------------------
# Quick scores (the offline scorer recomputes everything from `tokens`)
# ---------------------------------------------------------------------------

_NUM = re.compile(r"(?<![\w.])(?:[01](?:\.\d+)?|\.\d+)")


def quick_score(cond: Condition, answer: str, tokens: list[dict]) -> float | None:
    if cond.kind == "prob":
        for m in _NUM.finditer(answer or ""):
            v = float(m.group())
            if 0.0 <= v <= 1.0:
                return v
        return None
    if not tokens:
        return None
    p = {HUMAN: 0.0, "machine": 0.0}
    for a in tokens[0]["top"]:
        lab = cond.labels.get(re.sub(r"[^a-z]", "", a["token"].lower()))
        if lab:
            p[lab] += math.exp(a["logprob"])
    return (1.0 + p[HUMAN] - p["machine"]) / 2.0


# ---------------------------------------------------------------------------


def _sidecar(model: str, cond: str, subset: str):
    return WORKSPACE / model / f"{cond}__{subset}.parquet"


def _load_done(path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["uid", "stem", "rep", "answer", "score", "tokens",
                                 "model_version", "error"])


def _print_probe(cond: Condition, uid: str, label: str, answer: str, toks: list[dict],
                 version: str | None):
    print(f"\n  {uid}  [{label}]  served by {version}  answer={answer!r}  "
          f"quick score={quick_score(cond, answer, toks)}")
    for i, t in enumerate(toks):
        top = "  ".join(f"{a['token']!r}:{math.exp(a['logprob']):.3f}" for a in t["top"][:6])
        print(f"    pos {i}: {t['token']!r}  |  {top}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list conditions and subsets, then exit")
    ap.add_argument("--model", choices=JUDGES)
    ap.add_argument("--condition", choices=sorted(CONDITIONS))
    ap.add_argument("--subset", choices=SUBSETS, default=None,
                    help="default: the condition's default_subset")
    ap.add_argument("--reps", type=int, default=None,
                    help="repeats per row (default: the condition's default_reps)")
    ap.add_argument("--only", nargs="+", default=None, metavar="STEM",
                    help="restrict to these core splits")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=RETRIES)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--probe", type=int, default=0,
                    help="call N rows (half human), print answers and token windows, write nothing")
    ap.add_argument("--dry-run", action="store_true", help="counts and token estimate, no calls")
    ap.add_argument("--yes", action="store_true", help="skip the billed-call prompt")
    args = ap.parse_args()

    if args.list:
        for c in CONDITIONS.values():
            print(f"{c.name:12} {c.kind:5} T={c.temperature:<3} [{c.todo}]  "
                  f"default subset={c.default_subset} reps={c.default_reps}")
        print("subsets:", ", ".join(SUBSETS))
        return
    if not (args.model and args.condition):
        ap.error("--model and --condition are required (or --list)")

    cond = CONDITIONS[args.condition]
    subset = args.subset or cond.default_subset
    reps = args.reps or cond.default_reps
    spec = MODELS[args.model]
    path = _sidecar(args.model, cond.name, subset)

    core = load_core()
    rows = build_subset(core, subset)
    if args.only:
        rows = rows[rows["stem"].isin(args.only)]
    rows = rows[rows["text"].astype(str).str.strip().str.len() > 0]

    print(f"model {args.model} ({spec.model})  condition {cond.name} [{cond.todo}]  "
          f"subset {subset}  reps {reps}  temperature {cond.temperature}")
    by = rows.groupby("stem", sort=False)["generated_by"]
    for stem, g in by:
        print(f"  {stem:22} {len(g):6} rows  ({int(g.eq(HUMAN).sum())} human)")

    # --- probe: a handful of calls, printed, nothing written -----------------
    if args.probe:
        rng = np.random.default_rng(1)
        h = rows[rows["generated_by"].eq(HUMAN)]
        m = rows[~rows["generated_by"].eq(HUMAN)]
        k = args.probe // 2
        sample = pd.concat([h.iloc[rng.permutation(len(h))[:k]],
                            m.iloc[rng.permutation(len(m))[: args.probe - k]]])
        client = _build_client(spec.provider, spec.model)
        call = _call_openai if spec.provider == "openai" else _call_genai4science
        print(f"\nPROBE  {len(sample)} calls\n--- system prompt tail ---\n"
              f"{cond.system[-500:]}\n---")
        for _, r in sample.iterrows():
            try:
                ans, toks, version = call(client, spec.model, cond, r["text"])
                _print_probe(cond, r["uid"], r["generated_by"], ans, toks, version)
            except Exception as exc:  # noqa: BLE001
                print(f"\n  {r['uid']}  FAIL {type(exc).__name__}: {exc}")
        return

    # --- what still needs sending ---------------------------------------------
    done = _load_done(path)
    good = done[done["answer"].notna()]
    ok = set(zip(good["stem"], good["uid"], good["rep"]))
    todo = [(r.uid, r.stem, rep, r.text) for r in rows.itertuples() for rep in range(reps)
            if (r.stem, r.uid, rep) not in ok]
    n_in = sum(len(t[3]) + len(cond.system) for t in todo) / CHARS_PER_TOKEN
    print(f"\n  calls: {len(rows) * reps} total, {len(rows) * reps - len(todo)} done, "
          f"{len(todo)} to send")
    print(f"  estimated input tokens to send: {n_in / 1e6:.1f} M  "
          f"(output: <= {cond.max_output_tokens} tokens/call)")
    print(f"  sidecar: {path}")
    if args.dry_run or not todo:
        return
    if spec.provider == "openai" and not args.yes:
        if input("Billed API calls. Proceed? type 'yes': ").strip().lower() != "yes":
            raise SystemExit("aborted -- no calls made")

    client = _build_client(spec.provider, spec.model)
    call = _call_openai if spec.provider == "openai" else _call_genai4science
    keep = done[done["answer"].notna()].to_dict("records")
    new: dict[tuple[str, str, int], dict] = {}
    lock = threading.Lock()

    def _save():
        with lock:
            out = pd.DataFrame(keep + list(new.values()))
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)

    def _one(item):
        uid, stem, rep, text = item
        last = None
        for attempt in range(args.retries + 1):
            try:
                ans, toks, version = call(client, spec.model, cond, text)
                score = quick_score(cond, ans, toks)
                if score is None:
                    # a "prob" answer with no number in [0, 1]: a failed row, not a
                    # result -- retried like an API error
                    raise ValueError(f"no probability in answer {ans!r:.60}")
                return {"uid": uid, "stem": stem, "rep": rep, "answer": ans,
                        "score": score, "tokens": toks,
                        "model_version": version, "error": None}
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    time.sleep(BACKOFF_BASE ** attempt * (1.0 + random.random()))
        return {"uid": uid, "stem": stem, "rep": rep, "answer": None, "score": None,
                "tokens": None, "model_version": None, "error": last[:300]}

    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    n = n_fail = 0
    interrupted = False
    n_before = len(rows) * reps - len(todo)
    print(f"  started {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"(bar: calls this session; done/total incl. earlier sessions on the right)")
    try:
        futs = [pool.submit(_one, t) for t in todo]
        bar = tqdm(as_completed(futs), total=len(futs), desc=cond.name, unit="call",
                   mininterval=5, dynamic_ncols=True)
        for fut in bar:
            res = fut.result()
            with lock:
                new[(res["stem"], res["uid"], res["rep"])] = res
            n += 1
            n_fail += res["error"] is not None
            bar.set_postfix_str(f"done {n_before + n}/{len(rows) * reps}  failed {n_fail}",
                                refresh=False)
            if n % args.save_every == 0:
                _save()
    except KeyboardInterrupt:
        print("\ninterrupted -- writing what was captured")
        interrupted = True
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    _save()

    res = list(new.values())
    fails = Counter(r["error"] for r in res if r["error"])
    scores = [r["score"] for r in res if r["score"] is not None]
    versions = Counter(r["model_version"] for r in res if r["model_version"])
    print(f"\n  captured {len(res) - sum(fails.values())}/{len(res)}  "
          f"distinct quick scores: {len(set(scores))}")
    print(f"  served model version(s): {dict(versions)}")
    if len(versions) > 1:
        print("  WARNING  more than one model version served this run")
    if interrupted:
        # non-zero so that the run_*.sh scripts stop instead of starting the next step
        raise SystemExit(130)
    for msg, c in fails.most_common(5):
        print(f"  {c:>5}x FAIL {msg[:120]}")
    if fails:
        print("  re-run the same command to retry the failed rows")
        raise SystemExit(3)  # the run_*.sh scripts retry on this exit code


if __name__ == "__main__":
    main()

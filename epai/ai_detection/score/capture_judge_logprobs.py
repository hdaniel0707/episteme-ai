r"""Capture ONE LLM judge's answer-token logprobs for the four core benchmark
splits, into a small per-model sidecar. ``merge_captured_logprobs.py`` folds the
sidecars back into the real parquets.

One ``--model`` per run. Two runs (``gpt56luna`` on the OpenAI API,
``gemma4-31b`` on GenAI4Science) can go **at the same time** -- each writes its
own sidecar, neither reads or rewrites an 800 MB parquet, and the one expensive
read+write happens once, at merge.

Two answer formats per model (``--modes``):

  ``words``   the existing ``LOGPROB_ANSWER_BLOCK`` -- answer ``human`` / ``machine``.
  ``scale``   a 5-level rating, **A = human end**: A almost-certainly-human,
              B probably-human, C uncertain, D probably-machine,
              E almost-certainly-machine. A first so any answer-letter bias
              leans toward ``human`` -- the safe side for a low-FPR operating point.

For every eligible row it stores the **ranked top-k window at the answer
position** -- ``list<struct{token, logprob}>``. No score is computed: run the
paid calls once, try scoring functions later off the windows. ``window[0]`` is
the sampled token (temperature 0).

Sidecar: ``<workspace>/<stem>.parquet`` = ``uid`` + one column per (mode):

    llm_judge_logprobs_<model-column>_v3_words
    llm_judge_logprobs_<model-column>_v3_scale

Default workspace ``data/parquet/tmp/<model>/`` (created if missing;
``data/`` is gitignored) -- deterministic, so re-running the same command
resumes. Pass ``--workspace`` for an isolated run. **Running more than one
``--prompt`` into the same workspace overwrites the sidecar's other prompt's
columns** (the sidecar write keeps only the current run's columns) -- give
each prompt its own ``--workspace`` when capturing more than one.

``--prompt`` picks the evidence prompt from ``add_llm_judge_scores.PROMPTS``
(default ``v3``) -- any registered id, including a candidate loaded from
``cfg/ai_detection/judge_prompts/candidates/`` (see
``select_diverse_judge_prompts.py``). The prompt id is part of the column
name, so two prompts captured for the same model never collide once merged.

Paths: gemma4-31b -> GenAI4Science chat completions ``logprobs=True``;
gpt56luna -> OpenAI Responses API ``reasoning={"effort":"none"}`` +
``include=["message.output_text.logprobs"]``. Needs the relevant key in ``.env``.

    uv run python epai/ai_detection/score/capture_judge_logprobs.py --model gemma4-31b --dry-run
    uv run python epai/ai_detection/score/capture_judge_logprobs.py --model gemma4-31b --yes
    uv run python epai/ai_detection/score/capture_judge_logprobs.py --model gpt56luna --only manugram_aiscientist --limit 60 --yes
    uv run python epai/ai_detection/score/capture_judge_logprobs.py --model gpt56luna --prompt cand03 \
        --workspace data/parquet/tmp/gpt56luna-cand03 --only manugram_aiscientist --yes
    # rephrase corpora (opt-in, REPHRASE_STEMS) -- words mode only, score_logjudge_ord2.py
    # doesn't use scale:
    uv run python epai/ai_detection/score/capture_judge_logprobs.py --model gpt56luna \
        --only manugram_reph_gb manugram_reph_sci --modes words --dry-run
    # then, once both models are done:
    uv run python epai/ai_detection/score/merge_captured_logprobs.py
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
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from epai.ai_detection.score.add_llm_judge_scores import (
    BACKOFF_BASE,
    LOGPROB_ANSWER_BLOCK,
    MODELS,
    PROMPTS,
    RETRIES,
    USER_TEMPLATE,
    apply_limit,
    evidence_head,
)

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
CAPTURE_DIR = PARQUET_DIR / "tmp"  # sidecars live here; data/ is gitignored

CORE_STEMS = ["ghostbuster_orig", "manugram_gb", "manugram_sci", "manugram_aiscientist"]
#: opt-in only -- pass via --only, never swept by a bare (no --only) run, same
#: convention as score_logjudge_ord2.py's REPHRASE_STEMS. Only the two
#: genuinely-new rephrase classes need a window (human/full-ai in these files
#: are unused duplicates of the core-split rows -- see REPH_GEN_FILTER).
REPHRASE_STEMS = ["manugram_reph_gb", "manugram_reph_sci"]
ALL_STEMS = CORE_STEMS + REPHRASE_STEMS
#: stem -> the only generated_by values worth capturing there. manugram_reph_*
#: also carries a duplicate 'human' row and a duplicate full-ai
#: ('...:0701A'/'...:0811A') row per document -- both already scored via the
#: core-split parquet (plot_score_histograms.build() sources human/ai from
#: there, never from the reph file), so capturing them here would just be a
#: billed no-op.
REPH_GEN_FILTER: dict[str, set[str]] = {
    "manugram_reph_gb": {"gpt56luna:0908R", "gpt56luna:0916R"},
    "manugram_reph_sci": {"gpt56luna:0908R", "gpt56luna:0916R"},
}
MODES = ("words", "scale")
DEFAULT_PROMPT = "v3"
TOP_N = 20
TEXT_COL = "text"
KEY_COL = "uid"

# A = human, E = machine -- see module docstring for why A leads.
SCALE_ANSWER_BLOCK = """ANSWER
Answer with exactly one letter, and nothing else, rating whether the document was written by a human or generated by a large language model:
A  almost certainly written by a human
B  probably written by a human
C  genuinely uncertain
D  probably generated by a large language model
E  almost certainly generated by a large language model
No explanation, no punctuation. Judge each document on its own; do not assume any particular mix of human and machine documents."""

_ANSWER_BLOCK = {"words": LOGPROB_ANSWER_BLOCK, "scale": SCALE_ANSWER_BLOCK}
_VALID = {"words": {"human", "machine"}, "scale": set("ABCDE")}


def column(model_key: str, mode: str, prompt_id: str) -> str:
    return f"llm_judge_logprobs_{MODELS[model_key].column}_{prompt_id}_{mode}"


def _system_prompt(mode: str, prompt_id: str) -> str:
    return evidence_head(prompt_id) + "\n\n" + _ANSWER_BLOCK[mode]


def _norm(token: str, mode: str) -> str:
    s = re.sub(r"[^A-Za-z]", "", token)
    return s.lower() if mode == "words" else s.upper()


def _needs_scoring(x) -> bool:
    if x is None:
        return True
    try:
        return len(x) == 0
    except TypeError:  # scalar NaN
        return True


# ---------------------------------------------------------------------------
# One call -> the ranked answer-position window, or an exception
# ---------------------------------------------------------------------------


def _window_openai(client, model_id, system, text):
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


def _window_genai4science(client, model_id, system, text):
    from epai.utils.env_utils import qwen3_thinking_off
    messages = qwen3_thinking_off(model_id, [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ])
    resp = client.chat.completions.create(
        model=model_id, messages=messages, max_tokens=1, temperature=0,
        logprobs=True, top_logprobs=min(TOP_N, 20),
    )
    ch = resp.choices[0]
    lp = getattr(ch, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    if not content or not content[0].top_logprobs:
        raise ValueError("chat endpoint returned no logprobs for the answer token")
    return [{"token": a.token, "logprob": float(a.logprob)}
            for a in content[0].top_logprobs]


def _with_retries(fn, retries: int):
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries:
                time.sleep(BACKOFF_BASE ** attempt * (1.0 + random.random()))
    assert last is not None
    raise last


def _build_client(provider: str, model_id: str):
    from openai import OpenAI
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set -- see .env")
        return OpenAI()
    if provider == "genai4science":
        from epai.utils.env_utils import (
            genai4science_endpoint,
            report_genai4science_endpoint,
        )
        report_genai4science_endpoint(probe_model=model_id)
        base_url, key_env = genai4science_endpoint()
        key = os.environ.get(key_env)
        if not key:
            raise SystemExit(f"{key_env} is not set -- see .env")
        return OpenAI(base_url=base_url, api_key=key)
    raise SystemExit(f"no logprob path for provider {provider!r}")


def _summarise(windows, mode: str) -> str:
    got = [w for w in windows if w is not None and len(w)]
    if not got:
        return "      (no windows captured)"
    sizes = [len(w) for w in got]
    argmax = Counter(_norm(w[0]["token"], mode) for w in got)
    valid = sum(1 for w in got if _norm(w[0]["token"], mode) in _VALID[mode])
    sat = sum(1 for w in got if math.exp(w[0]["logprob"]) > 0.98)
    top = "  ".join(f"{k or '?'}={v}" for k, v in argmax.most_common(7))
    return (f"      captured {len(got)}  |  window mean {np.mean(sizes):.1f} "
            f"(size-1: {sizes.count(1)})  |  argmax in-format {valid}/{len(got)}  |"
            f"  saturated>0.98: {sat}/{len(got)}\n      argmax: {top}")


# ---------------------------------------------------------------------------


def _run_mode(light: pd.DataFrame, col: str, mode: str, eligible: np.ndarray,
              spec, sidecar: Path, cols: list[str], args) -> None:
    need_raw = eligible & np.array([_needs_scoring(v) for v in light[col]])
    already = int((eligible & ~need_raw).sum())
    need = apply_limit(light, need_raw, args.limit)
    positions = np.flatnonzero(need)
    print(f"    {mode:5}  {int(eligible.sum())} eligible | {already} captured | "
          f"{len(positions)} to send" + (f" (--limit {args.limit})" if args.limit else ""))
    if len(positions) == 0 or args.dry_run:
        return

    client = _build_client(spec.provider, spec.model)
    system = _system_prompt(mode, args.prompt)
    call = _window_openai if spec.provider == "openai" else _window_genai4science
    text = light[TEXT_COL].astype(str)
    texts = {int(p): (text.iloc[int(p)][: args.max_chars] if args.max_chars
                      else text.iloc[int(p)]) for p in positions}
    values = list(light[col])

    done, fails, lock = {"n": 0}, [], threading.Lock()

    def _save() -> None:
        light[col] = pd.Series(values, index=light.index, dtype=object)
        _write_sidecar(light, sidecar, cols)

    def _one(pos: int):
        try:
            w = _with_retries(
                lambda: call(client, spec.model, system, texts[pos]), args.retries)
            return pos, w, None
        except Exception as exc:  # noqa: BLE001
            return pos, None, f"{type(exc).__name__}: {exc}"

    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    try:
        futs = [pool.submit(_one, int(p)) for p in positions]
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=f"{sidecar.parent.name} {mode}", unit="row"):
            pos, w, err = fut.result()
            values[pos] = w
            if err is not None:
                fails.append((pos, err))
            with lock:
                done["n"] += 1
                if done["n"] % args.save_every == 0:
                    _save()
    except KeyboardInterrupt:
        print("\n    interrupted -- writing what was captured")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    _save()
    if fails:
        for m, c in Counter(m for _, m in fails).most_common(5):
            print(f"      {c:>5}x FAIL  {m[:100]}")
        print(f"      ({len(fails)} row(s) left empty -- re-run to retry)")
    print(_summarise(values, mode))


def _write_sidecar(light: pd.DataFrame, path: Path, cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    light[[KEY_COL, *cols]].to_parquet(path, index=False)


def capture_stem(stem: str, model_key: str, modes: list[str], args) -> None:
    spec = MODELS[model_key]
    src = PARQUET_DIR / f"{stem}.parquet"
    sidecar = args.workspace / f"{stem}.parquet"
    cols = [column(model_key, md, args.prompt) for md in modes]
    print(f"\n{'-' * 74}\n{stem}   ->  {sidecar}")
    if not src.is_file():
        print("  source parquet not on disk -- skipped")
        return

    have = set(pq.read_schema(src).names)
    if KEY_COL not in have:
        print(f"  no {KEY_COL!r} column -- cannot merge back, skipped")
        return
    light_cols = [c for c in (KEY_COL, TEXT_COL, "domain", "generated_by") if c in have]
    light = pd.read_parquet(src, columns=light_cols)

    if sidecar.is_file():
        prev = pd.read_parquet(sidecar)
        keep = [KEY_COL] + [c for c in cols if c in prev.columns]
        light = light.merge(prev[keep], on=KEY_COL, how="left")
    for c in cols:
        if c not in light.columns:
            light[c] = None

    eligible = (light[TEXT_COL].astype(str).str.strip().str.len() > 0).to_numpy()
    if stem in REPH_GEN_FILTER:
        eligible = eligible & light["generated_by"].isin(REPH_GEN_FILTER[stem]).to_numpy()
    for mode in modes:
        _run_mode(light, column(model_key, mode, args.prompt), mode, eligible, spec,
                  sidecar, cols, args)
    if not args.dry_run:
        _write_sidecar(light, sidecar, cols)  # consolidating flush


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=list(MODELS),
                    help="one judge alias from add_llm_judge_scores.MODELS")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, choices=sorted(PROMPTS),
                    help=f"evidence prompt id from add_llm_judge_scores.PROMPTS "
                         f"(default: {DEFAULT_PROMPT}); includes any candidate "
                         "loaded from cfg/ai_detection/judge_prompts/candidates/")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--only", nargs="+", choices=ALL_STEMS, metavar="STEM")
    ap.add_argument("--workspace", type=Path, default=None,
                    help="sidecar dir (default data/parquet/tmp/<model>/, created if missing)")
    ap.add_argument("--limit", type=int, default=None,
                    help="judge only N eligible rows per file, round-robin over "
                         "domain x generated_by")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=RETRIES)
    ap.add_argument("--save-every", type=int, default=200,
                    help="rows between sidecar checkpoints (default 200; the "
                         "sidecar is small, so this is cheap)")
    ap.add_argument("--max-chars", type=int, default=None,
                    help="truncate each document to N chars (default: whole document)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the billed-call prompt")
    args = ap.parse_args()

    args.workspace = args.workspace or (CAPTURE_DIR / args.model)
    if not args.dry_run:
        args.workspace.mkdir(parents=True, exist_ok=True)
    stems = args.only or CORE_STEMS
    spec = MODELS[args.model]

    print(f"model {args.model} ({spec.provider} -> {spec.model})   "
          f"prompt {args.prompt}   modes {args.modes}")
    print(f"workspace {args.workspace}")
    print("columns:  " + "  ".join(column(args.model, md, args.prompt) for md in args.modes))

    if spec.provider == "openai" and not args.dry_run and not args.yes:
        print("\nNOTE  billed API call -- one short completion per row. "
              "Run --dry-run first for the counts.")
        if input("Proceed? type 'yes': ").strip().lower() != "yes":
            raise SystemExit("aborted -- no calls made")

    for stem in stems:
        capture_stem(stem, args.model, args.modes, args)

    print(f"\n{'=' * 74}\ndone -- sidecars in {args.workspace}\n"
          f"next: uv run python epai/ai_detection/score/merge_captured_logprobs.py")


if __name__ == "__main__":
    main()

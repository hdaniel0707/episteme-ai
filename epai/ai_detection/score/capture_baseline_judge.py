r"""Capture ONE LLM judge's PARSED probability under a chosen prompt, into a
small per-model sidecar. ``merge_captured_judge_scores.py`` folds the sidecars
back into the real parquets.

The parsed-mode twin of ``capture_judge_logprobs.py``: same
capture -> sidecar -> merge shape, so an expensive judge run never rewrites an
800 MB parquet until the one merge at the end. Unlike that script this stores a
single number per row (the probability the model writes), not a token window.

Built for the ``base`` prompt -- the control that names the task and the genre
but gives no cue guidance (see ``PROMPTS["base"]`` in ``add_llm_judge_scores``)
-- so it can be compared against the cue-loaded ``v3`` judge already in the
corpora. ``--prompt`` takes any registered prompt, though.

Sidecar: ``<workspace>/<stem>.parquet`` = ``uid`` + one column:

    llm_judge_score_<model-column>_<prompt>        (e.g. llm_judge_score_gemma4_31b_base)

Default workspace ``data/parquet/tmp/judge_scores/<model>/`` (created if
missing; ``data/`` is gitignored) -- deterministic, so re-running the same
command resumes and only unscored rows are sent. An interrupt (``Ctrl-C``)
writes what was captured.

    uv run python epai/ai_detection/score/capture_baseline_judge.py --model gemma4-31b --dry-run
    uv run python epai/ai_detection/score/capture_baseline_judge.py --model gemma4-31b --yes
    uv run python epai/ai_detection/score/capture_baseline_judge.py --model gpt56luna --only manugram_aiscientist --limit 60 --yes
    # then, once both models are done:
    uv run python epai/ai_detection/score/merge_captured_judge_scores.py
"""

from __future__ import annotations

import argparse
import math
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from epai.ai_detection.score.add_llm_judge_scores import (
    MODE_PARSED,
    MODELS,
    RETRIES,
    ApiJudge,
    apply_limit,
    judge_row,
    score_column,
)

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
CAPTURE_DIR = PARQUET_DIR / "tmp" / "judge_scores"  # data/ is gitignored

CORE_STEMS = ["ghostbuster_orig", "manugram_gb", "manugram_sci", "manugram_aiscientist"]
TEXT_COL = "text"
KEY_COL = "uid"
DEFAULT_PROMPT = "base"


def _needs_scoring(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _summarise(values: list) -> str:
    got = np.array([v for v in values if not _needs_scoring(v)], dtype=float)
    if not got.size:
        return "      (nothing scored)"
    uniq, counts = np.unique(np.round(got, 4), return_counts=True)
    clump = counts.max() / counts.sum()
    return (f"      scored {got.size}  |  mean {got.mean():.3f}  median "
            f"{np.median(got):.3f}  |  distinct {len(uniq)}  largest tie "
            f"{clump:.0%}  |  range [{got.min():.2f}, {got.max():.2f}]")


def _write_sidecar(light: pd.DataFrame, path: Path, col: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    light[[KEY_COL, col]].to_parquet(path, index=False)


def _run(light: pd.DataFrame, col: str, eligible: np.ndarray, judge,
         sidecar: Path, args) -> None:
    need_raw = eligible & np.array([_needs_scoring(v) for v in light[col]])
    already = int((eligible & ~need_raw).sum())
    need = apply_limit(light, need_raw, args.limit)
    positions = [int(p) for p in np.flatnonzero(need)]
    print(f"    {int(eligible.sum())} eligible | {already} scored | "
          f"{len(positions)} to send" + (f" (--limit {args.limit})" if args.limit else ""))
    if not positions or args.dry_run:
        return

    text = light[TEXT_COL].astype(str)
    texts = {p: (text.iloc[p][: args.max_chars] if args.max_chars else text.iloc[p])
             for p in positions}
    values = list(light[col])
    done, fails, lock = {"n": 0}, [], threading.Lock()

    def _save() -> None:
        light[col] = pd.Series(values, index=light.index, dtype=object)
        _write_sidecar(light, sidecar, col)

    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    try:
        futs = {pool.submit(judge_row, judge, p, texts[p], args.retries): p
                for p in positions}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=sidecar.parent.name, unit="row"):
            rr = fut.result()
            values[rr.position] = rr.score
            if rr.error is not None:
                fails.append((rr.position, rr.error))
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
        for msg, c in Counter(m for _, m in fails).most_common(5):
            print(f"      {c:>5}x FAIL  {msg[:100]}")
        print(f"      ({len(fails)} row(s) left Null -- re-run to retry)")
    print(_summarise(values))


def capture_stem(stem: str, judge, col: str, args) -> None:
    src = PARQUET_DIR / f"{stem}.parquet"
    sidecar = args.workspace / f"{stem}.parquet"
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
        if col in prev.columns:
            light = light.merge(prev[[KEY_COL, col]], on=KEY_COL, how="left")
    if col not in light.columns:
        light[col] = None

    eligible = (light[TEXT_COL].astype(str).str.strip().str.len() > 0).to_numpy()
    _run(light, col, eligible, judge, sidecar, args)
    if not args.dry_run:
        _write_sidecar(light, sidecar, col)  # consolidating flush


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=list(MODELS),
                    help="one judge alias from add_llm_judge_scores.MODELS")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help=f"system prompt id (default {DEFAULT_PROMPT!r}); names the column")
    ap.add_argument("--only", nargs="+", choices=CORE_STEMS, metavar="STEM")
    ap.add_argument("--workspace", type=Path, default=None,
                    help="sidecar dir (default data/parquet/tmp/judge_scores/<model>/)")
    ap.add_argument("--limit", type=int, default=None,
                    help="judge only N eligible rows per file, round-robin over "
                         "domain x generated_by")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent requests (default 4). GenAI4Science throttles "
                         "-- 4 is smooth there; OpenAI takes 16+ comfortably.")
    ap.add_argument("--retries", type=int, default=RETRIES)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=None,
                    help="truncate each document to N chars (default: whole document)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the billed-call prompt")
    args = ap.parse_args()

    spec = MODELS[args.model]
    args.workspace = args.workspace or (CAPTURE_DIR / args.model)
    if not args.dry_run:
        args.workspace.mkdir(parents=True, exist_ok=True)
    col = score_column(spec, args.prompt, MODE_PARSED)

    print(f"model {args.model} ({spec.provider} -> {spec.model})   prompt {args.prompt!r}")
    print(f"workspace {args.workspace}")
    print(f"column:   {col}")

    if spec.provider == "openai" and not args.dry_run and not args.yes:
        print("\nNOTE  billed API call -- one short completion per row. "
              "Run --dry-run first for the counts.")
        if input("Proceed? type 'yes': ").strip().lower() != "yes":
            raise SystemExit("aborted -- no calls made")

    # One judge (one client, one connection pool) for the whole run -- fewer
    # cold starts than a client per file.
    judge = None if args.dry_run else ApiJudge(
        spec, prompt_id=args.prompt, mode=MODE_PARSED)

    for stem in (args.only or CORE_STEMS):
        capture_stem(stem, judge, col, args)

    if judge is not None:
        for line in judge.diagnostics():
            print(line)
    print(f"\n{'=' * 74}\ndone -- sidecars in {args.workspace}\n"
          f"next: uv run python epai/ai_detection/score/merge_captured_judge_scores.py")


if __name__ == "__main__":
    main()

r"""Fill the gemma4:31b rows that ``gemma_science.py`` could not reach.

``gemma_science.py`` generates against ``science_v3``'s 3000 paper prompts and
appends the ~2946 whose ``id`` also lives in ``science_v3f`` — the two paper
sets differ by 54, because ``science_v3f`` is ``science_v3`` rebuilt three times
(v3→v3d→v3e→v3f) with bad human rows dropped and replacements backfilled. Those
54 backfilled papers have human + gpt56luna + gpt-oss + llama rows but no gemma
one.

This closes the gap: it reads those 54 papers' prompts straight out of
``science_v3f_multi_0811A.parquet`` (an existing generator row carries the
prompt), generates gemma abstracts for them, cleans and standardises the text
the same way, and appends the rows with every detector-suite column Null.

    python epai/ai_detection/rollout/gemma_science_backfill.py --gpu 4 --yes \
        --skip-judge llama33-70b

Then it re-runs ``run_holdout_pipeline.py --from embed`` so the 54 new rows get
scored (skip it with ``--no-score`` and run the printed command yourself).

Resumable: generations are checkpointed to
``data/science/science_v3_gemma_backfill.jsonl``; a re-run only re-issues the
calls that failed. Appending is a no-op once the label already covers a paper,
so running it twice is safe.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from epai.ai_detection.rollout._common import (
    BASE_DIR, DIM, GREEN, RED, RESET, YELLOW, check_completeness, holdout_score,
    rel, resolve_judges,
)
from epai.science_data.config import (
    generator_endpoint,
    load_pipeline_config,
    load_prompt_config,
    strip_structured_labels_enabled,
)
from epai.science_data.generate.generate_ai import _key, clean_generated
from epai.science_data.llm import Shard, openai_text_call, run_resumable
from epai.utils.env_utils import report_genai4science_endpoint
from epai.utils.text_standardize import standardize_text

CFG = BASE_DIR / "cfg/science_data/science_v3.yaml"
TARGET = BASE_DIR / "data/parquet/science_v3f_multi_0811A.parquet"
SHARD = BASE_DIR / "data/science/science_v3_gemma_backfill.jsonl"
LABEL = "gemma431b:0811A"
MODEL = "gemma4:31b"
TEMPLATE_LABEL = "gpt56luna:0811A"  # an existing generator row to copy prompt + metadata from

CORPUS_COLS = [
    "uid", "id", "domain", "generated_by", "doi", "title", "author_ids",
    "author_count", "year", "venue", "citation_count", "fwci", "language",
    "fields_of_study", "query", "also_matched_domains", "prompt",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target", type=Path, default=TARGET)
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--cfg", type=Path, default=CFG,
                    help="science pipeline config (for the system prompt + clean opts)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-issue calls previously recorded as failed")
    ap.add_argument("--no-score", action="store_true",
                    help="append only; print the holdout command instead of running it")
    ap.add_argument("--skip-judge", nargs="+", default=[], metavar="JUDGE")
    ap.add_argument("--gpu", default=None, metavar="ID")
    ap.add_argument("--cpu-threads", type=int, default=None, metavar="N")
    ap.add_argument("--per-band", type=int, default=20000)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.target.is_file():
        raise SystemExit(f"{RED}not found: {args.target}{RESET}")

    tgt = pq.read_table(args.target)
    schema = tgt.schema
    gb = tgt.column("generated_by").to_pylist()
    ids = tgt.column("id").to_pylist()

    have = {ids[i] for i, g in enumerate(gb) if g == args.label}
    template = {ids[i]: i for i, g in enumerate(gb) if g == TEMPLATE_LABEL}
    missing = sorted(pid for pid in template if pid not in have)

    print(f"\ntarget   {rel(args.target)}  ({tgt.num_rows} rows)")
    print(f"label    {args.label}   ({len(have)} present)")
    print(f"missing  {len(missing)} paper(s) with a {TEMPLATE_LABEL} row but no {args.label} row")
    if not missing:
        print(f"\n{GREEN}Nothing to backfill.{RESET}")
        return
    print(f"{DIM}    e.g. {missing[:3]}{RESET}")

    cfg = load_pipeline_config(args.cfg)
    prompts_cfg = load_prompt_config(cfg)
    system = prompts_cfg["generation_system_prompt"]
    strip_labels = strip_structured_labels_enabled(cfg)

    # user prompt for each missing paper: the prompt its template row carries.
    prompt_col = tgt.column("prompt").to_pylist()
    user_of = {pid: prompt_col[template[pid]] for pid in missing}

    if args.dry_run:
        print(f"\n{DIM}dry-run — would generate {len(missing)} {args.model} "
              f"abstract(s), append, and {'stop' if args.no_score else 'score'}."
              f"{RESET}")
        return

    import os

    from openai import OpenAI

    base_url, api_key_env = generator_endpoint(
        {"model": args.model, "provider": "genai4science"}
    )
    report_genai4science_endpoint(probe_model=args.model)
    client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url, timeout=300)

    shard = Shard(SHARD)
    tasks = [
        (_key(pid, args.label),
         openai_text_call(client, args.model, system, user_of[pid],
                          base_url=base_url))
        for pid in missing
    ]
    run_resumable(
        tasks, shard, max_workers=args.workers, desc=f"{args.label} backfill",
        retry_failed=args.retry_failed,
        payload_of=lambda _k, text: {"abstract": clean_generated(text, strip_labels=strip_labels)},
    )

    ok = shard.ok_records()
    rows_for = [pid for pid in missing if _key(pid, args.label) in ok]
    failed = [pid for pid in missing if pid not in rows_for]
    if failed:
        print(f"{YELLOW}  {len(failed)} still failing — re-run with --retry-failed. "
              f"Appending the {len(rows_for)} that succeeded.{RESET}")
    if not rows_for:
        raise SystemExit(f"{RED}no successful generations to append.{RESET}")

    # Build the new rows on the target's schema.
    raw = [ok[_key(pid, args.label)]["abstract"] for pid in rows_for]
    std = [standardize_text(t, whitespace="flat") for t in raw]
    tmpl_idx = [template[pid] for pid in rows_for]

    def col_for(field: pa.Field) -> pa.Array:
        c = field.name
        if c == "text":
            return pa.array(std, type=field.type)
        if c == "text_raw":
            return pa.array(raw, type=field.type)
        if c == "generated_by":
            return pa.array([args.label] * len(rows_for), type=field.type)
        if c == "uid":
            return pa.array([f"{pid}:{args.label}" for pid in rows_for], type=field.type)
        if c in CORPUS_COLS:
            return tgt.column(c).take(tmpl_idx).cast(field.type).combine_chunks()
        return pa.nulls(len(rows_for), type=field.type)

    appended = pa.table([col_for(f) for f in schema], schema=schema)

    bak = args.target.with_suffix(args.target.suffix + ".bak")
    shutil.copy2(args.target, bak)
    print(f"{DIM}  backup → {rel(bak)}{RESET}")

    combined = pa.concat_tables([tgt, appended])
    order = pc.sort_indices(
        combined, sort_keys=[("id", "ascending"), ("generated_by", "ascending")]
    )
    pq.write_table(combined.take(order), args.target)
    print(f"\n{GREEN}appended {len(rows_for)} row(s){RESET} → {rel(args.target)} "
          f"({combined.num_rows} rows)")

    judges = resolve_judges(args.skip_judge)
    if args.no_score:
        print(f"\n  next: run_holdout_pipeline.py --from embed -o {rel(args.target)} "
              f"--judges {' '.join(judges)} --per-band {args.per_band} "
              f"--skip-existing --gpu <id> --yes")
        return

    # Score first -- the rows just appended are all-Null, so a completeness check
    # here would trivially fail. Then check that the scoring actually filled them.
    code = holdout_score(args.target, judges, gpu=args.gpu,
                         cpu_threads=args.cpu_threads, per_band=args.per_band,
                         yes=args.yes, dry_run=False)
    if code != 0:
        raise SystemExit(code)
    raise SystemExit(check_completeness(args.target))


if __name__ == "__main__":
    main()

r"""Build the RAW ``ghostbuster_orig.parquet`` end to end, unattended.

The Ghostbuster-raw sibling of ``run_holdout_pipeline.py``. It splits the
3-class source into the raw variant (``split_ghostbuster_orig.py``: human ``text``
<- ``text_raw``, which is byte-identical to the ``external/ghostbuster`` human
``.txt`` files; every human feature column nulled), then re-scores the human
rows through the whole suite while the ``gpt35t`` machine rows -- already
raw-scored -- are skipped from the inside by every stage.

DIFFERENCES FROM run_holdout_pipeline.py
---------------------------------------
* No ``restructure`` and NO ``standardize`` stage. Raw text is the point of this
  variant; ``standardize_text.py`` must never touch it.
* Adds the stages run_holdout lacks: logprob-window capture + merge + scoring,
  the ``base`` control judge capture + merge, and the soft-min ensemble.
* Does NOT run the gemma ``_lp`` judge (``run_gemma_logprob_judge.py``).
* The final ``verify`` uses ``--allow-unstandardized``: the standardisation
  finding is a WARN, not an ERROR, so the pipeline does not stop on it.

Every stage is a separate process with its own progress bars and exit code; a
stage that exits 0 having filled nothing is caught by the column check after it,
and re-run (the scorers skip filled rows) up to ``--fill-retries`` times. Set
``--yes`` for an unattended run. ``NTFY_PIPELINE_TOPIC`` in ``.env`` -> a phone
push at every stage boundary.

    # first, from the 3-class source (once):
    uv run python epai/ai_detection/prepare/split_ghostbuster_orig.py \
        --drop-cols llm_judge_score_sonnet5_v3
    # then leave this running overnight:
    uv run python epai/ai_detection/prepare/run_ghostbuster_orig_pipeline.py \
        --yes --gpu 0

    uv run python epai/ai_detection/prepare/run_ghostbuster_orig_pipeline.py --dry-run
    uv run python epai/ai_detection/prepare/run_ghostbuster_orig_pipeline.py \
        --from logprob-capture-gpt56luna --yes --gpu 0    # resume
    uv run python epai/ai_detection/prepare/run_ghostbuster_orig_pipeline.py \
        --split --yes --gpu 0                             # run the split too

``--split`` prepends the split stage AND wipes this pipeline's capture
workspaces (a fresh split invalidates the human captures). Without it the split
is assumed done and the workspaces persist for a resume.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from epai.ai_detection.prepare.run_holdout_pipeline import (
    BASE_DIR,
    BINOX_EXTENDED_TOKENIZER,
    BOLD,
    DEFAULT_BINOX_PAIRS,
    DEFAULT_DETECTORS,
    DEFAULT_EMBEDDERS,
    DEFAULT_FASTDETECT_PAIRS,
    DEFAULT_JUDGES,
    DEFAULT_PER_BAND,
    DIM,
    EMBED_DIR,
    GREEN,
    INTERRUPTED,
    PREPARE_DIR,
    PROMPT_VERSION,
    RED,
    RESET,
    SCORE_DIR,
    Stage,
    YELLOW,
    _binox_column,
    _suite_judge_kinds,
    blank_text_count,
    judge_stem,
    rel,
    run,
    verify,
)
from epai.utils import notify_utils

STEM = "ghostbuster_orig"
OUT = BASE_DIR / "data" / "parquet" / f"{STEM}.parquet"
SPLIT_SCRIPT = f"{PREPARE_DIR}/split_ghostbuster_orig.py"

# Isolated from the CORE_STEMS capture dirs (data/parquet/tmp/<model>/,
# data/parquet/tmp/judge_scores/<model>/) so a stale sibling workspace is never
# folded in by a default-workspace merge.
WS_ROOT = BASE_DIR / "data" / "parquet" / "tmp" / "gb_orig_pipe"

CUE_JUDGE = "gpt56luna"          # rationales -> cue vector; column carries it
LOGPROB_MODELS = ("gpt56luna", "gemma4-31b")
BASELINE_MODELS = ("gpt56luna", "gemma4-31b")
ENSEMBLE_TAUS = ("0.05", "0.1", "0.15", "0.2")


def _stem(model: str) -> str:
    return judge_stem(model)


def build_stages(args: argparse.Namespace) -> list[Stage]:
    out = str(OUT)
    gpu = ["--gpu", str(args.gpu)] if args.gpu is not None else []
    cpu = ["--cpu-threads", str(args.cpu_threads)] if args.cpu_threads else []
    yes = ["--yes"] if args.yes else []
    api_w = ["--workers", str(args.api_workers)]
    g4s_w = ["--workers", str(min(args.api_workers, args.g4s_workers))]
    g4s = _suite_judge_kinds().get("genai4science", set())

    per_band = max(args.per_band, _row_count(out))
    lp_ws = {m: WS_ROOT / "logprob" / m for m in LOGPROB_MODELS}
    base_ws = {m: WS_ROOT / "baseline" / m for m in BASELINE_MODELS}

    stages: list[Stage] = []

    if args.split:
        stages.append(Stage(
            "split", SPLIT_SCRIPT,
            ["--variant", "orig", "--force", "--drop-cols",
             "llm_judge_score_sonnet5_v3",       # retired, absent on gpt35ts
             "llm_judge_score_gemma4_31b_v3_lp"],  # this pipeline does not refill _lp
            produces=[], reads_text=False,
        ))

    stages.append(Stage(
        "embed", f"{EMBED_DIR}/add_embeddings.py",
        [out, "--embedders", *args.embedders, *gpu, *cpu, *yes],
        produces=[f"emb_{e}" for e in args.embedders],
    ))

    strict = [p for p in args.binox_pairs if p not in BINOX_EXTENDED_TOKENIZER]
    extended = [p for p in args.binox_pairs if p in BINOX_EXTENDED_TOKENIZER]
    if strict:
        stages.append(Stage(
            "binox", f"{SCORE_DIR}/run_binox_pairs.py",
            [out, "--pairs", *strict, *gpu, *cpu, *yes],
            produces=[_binox_column(p) for p in strict],
        ))
    if extended:
        stages.append(Stage(
            "binox-extended", f"{SCORE_DIR}/run_binox_pairs.py",
            [out, "--pairs", *extended, "--allow-token-mismatch", *gpu, *cpu, *yes],
            produces=[_binox_column(p) for p in extended],
        ))

    stages += [
        Stage(
            "fastdetect", f"{SCORE_DIR}/run_fastdetect_pairs.py",
            [out, "--pairs", *args.fastdetect_pairs, *gpu, *cpu, *yes],
            produces=[f"fastdetect_crit_{p.replace('-', '_')}"
                      for p in args.fastdetect_pairs],
        ),
        Stage(
            "detectors", f"{SCORE_DIR}/add_detector_scores.py",
            [out, "--detectors", *args.detectors, *gpu, *cpu, *yes],
            produces=[f"detector_score_{d}" for d in args.detectors],
        ),
    ]

    for j in args.judges:
        w = g4s_w if j in g4s else api_w
        stages.append(Stage(
            f"judge-{j}", f"{SCORE_DIR}/add_llm_judge_scores.py",
            [out, "--model", j, "--prompt", PROMPT_VERSION, *w, *gpu, *cpu, *yes],
            produces=[f"llm_judge_score_{_stem(j)}_{PROMPT_VERSION}"],
        ))

    cue_stem = f"{_stem(CUE_JUDGE)}_{PROMPT_VERSION}"
    stages += [
        Stage(
            "cue-reasons", f"{SCORE_DIR}/add_llm_judge_reasons.py",
            [out, "--model", CUE_JUDGE, "--prompt", PROMPT_VERSION,
             "--select", "all", "--per-band", str(per_band),
             *(g4s_w if CUE_JUDGE in g4s else api_w), *gpu, *cpu, *yes],
            produces=[f"llm_judge_reason_{cue_stem}"],
        ),
        Stage(
            "cue-features", f"{SCORE_DIR}/add_cue_features.py",
            [out, "--judge", cue_stem],
            produces=[f"llm_judge_cuevec_{cue_stem}"],
        ),
    ]

    # logprob windows: one capture per model (own workspace, resumable), one
    # merge (uid left-join -> whole column replaced, so --overwrite + the
    # sidecar must cover every uid), then the P(human) readouts.
    for m in LOGPROB_MODELS:
        w = g4s_w if m in g4s else api_w
        stages.append(Stage(
            f"logprob-capture-{m}", f"{SCORE_DIR}/capture_judge_logprobs.py",
            ["--model", m, "--only", STEM, "--workspace", str(lp_ws[m]), *w, *yes],
            produces=[], reads_text=False,
        ))
    stages.append(Stage(
        "logprob-merge", f"{SCORE_DIR}/merge_captured_logprobs.py",
        ["--only", STEM, "--overwrite",
         "--workspaces", *[str(lp_ws[m]) for m in LOGPROB_MODELS]],
        produces=[f"llm_judge_logprobs_{_stem(m)}_v3_{mode}"
                  for m in LOGPROB_MODELS for mode in ("words", "scale")],
        reads_text=False,
    ))
    stages.append(Stage(
        "logprob-windows", f"{SCORE_DIR}/score_logprob_windows.py",
        ["--only", STEM, "--overwrite"],
        produces=[f"llm_judge_score_{_stem(m)}_v3_{suf}"
                  for m in LOGPROB_MODELS for suf in ("wlp", "slp", "sbc")],
        reads_text=False,
    ))

    # baseline ('base' control prompt) — same capture -> merge shape.
    for m in BASELINE_MODELS:
        w = g4s_w if m in g4s else api_w
        stages.append(Stage(
            f"baseline-capture-{m}", f"{SCORE_DIR}/capture_baseline_judge.py",
            ["--model", m, "--prompt", "base", "--only", STEM,
             "--workspace", str(base_ws[m]), *w, *yes],
            produces=[], reads_text=False,
        ))
    stages.append(Stage(
        "baseline-merge", f"{SCORE_DIR}/merge_captured_judge_scores.py",
        ["--only", STEM, "--overwrite",
         "--workspaces", *[str(base_ws[m]) for m in BASELINE_MODELS]],
        produces=[f"llm_judge_score_{_stem(m)}_base" for m in BASELINE_MODELS],
        reads_text=False,
    ))

    stages.append(Stage(
        "ensemble", f"{SCORE_DIR}/ensemble_judge_scores.py",
        ["--only", STEM, "--name", "ens", "--tau", *ENSEMBLE_TAUS, "--overwrite"],
        produces=[f"llm_judge_score_ens_v3_smin_{t.replace('.', 'p')}"
                  for t in ENSEMBLE_TAUS],
        reads_text=False,
    ))

    stages.append(Stage(
        "verify", "epai/scripts/check_parquet_completeness.py",
        [out, "--allow-unstandardized", "--fail-on", "error"],
        produces=[], reads_text=False,
    ))
    return stages


def _row_count(path: str) -> int:
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return 0


def _check_split_done() -> None:
    """When the split stage is NOT running, the output must already be the
    2-class raw variant — otherwise every scoring stage is a silent no-op on the
    fully-scored 3-class source."""
    import pyarrow.parquet as pq

    gens = set(pq.read_table(OUT, columns=["generated_by"]).column(0).to_pylist())
    if gens != {"human", "gpt35t:0000A"}:
        raise SystemExit(
            f"{RED}{rel(OUT)} has classes {sorted(gens)} — this is not the raw "
            f"2-class variant.{RESET}\nRun  split_ghostbuster_orig.py --variant "
            f"orig --drop-cols llm_judge_score_sonnet5_v3 "
            f"llm_judge_score_gemma4_31b_v3_lp  first, or add --split."
        )


def select(stages: list[Stage], args: argparse.Namespace) -> list[Stage]:
    keys = [s.key for s in stages]
    if args.only:
        bad = [k for k in args.only if k not in keys]
        if bad:
            raise SystemExit(f"unknown stage(s) {bad}. choose from: {keys}")
        return [s for s in stages if s.key in args.only]
    if args.start_from:
        if args.start_from not in keys:
            raise SystemExit(f"unknown stage {args.start_from!r}. choose from: {keys}")
        return stages[keys.index(args.start_from):]
    return stages


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--split", action="store_true",
                   help="prepend the split stage and wipe the capture "
                        "workspaces (a fresh split invalidates the human "
                        "captures). Off: the split is assumed already done.")
    p.add_argument("--gpu", default=None, metavar="ID")
    p.add_argument("--cpu-threads", type=int, default=None, metavar="N")
    p.add_argument("--embedders", nargs="+", default=list(DEFAULT_EMBEDDERS))
    p.add_argument("--binox-pairs", nargs="+", default=list(DEFAULT_BINOX_PAIRS))
    p.add_argument("--fastdetect-pairs", nargs="+",
                   default=list(DEFAULT_FASTDETECT_PAIRS))
    p.add_argument("--detectors", nargs="+", default=list(DEFAULT_DETECTORS))
    p.add_argument("--judges", nargs="+", default=list(DEFAULT_JUDGES))
    p.add_argument("--per-band", type=int, default=DEFAULT_PER_BAND)
    p.add_argument("--api-workers", type=int, default=8, metavar="N")
    p.add_argument("--g4s-workers", type=int, default=4, metavar="N")
    p.add_argument("--from", dest="start_from", metavar="STAGE")
    p.add_argument("--only", nargs="+", metavar="STAGE")
    p.add_argument("--yes", action="store_true",
                   help="answer yes to every stage's confirmation prompt "
                        "(required for an unattended run)")
    p.add_argument("--fill-retries", type=int, default=2, metavar="N")
    p.add_argument("--fill-cooldown", type=float, default=60.0, metavar="SECONDS")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not OUT.is_file() and not (args.split or (args.only and "split" in args.only)):
        raise SystemExit(
            f"{RED}{rel(OUT)} not found — run split_ghostbuster_orig.py first, "
            f"or pass --split.{RESET}"
        )

    stages = select(build_stages(args), args)
    if not stages:
        raise SystemExit("no stages selected")

    running_split = any(s.key == "split" for s in stages)
    if not running_split and not args.dry_run:
        _check_split_done()
    if running_split and not args.dry_run and WS_ROOT.is_dir():
        print(f"{DIM}wiping capture workspaces under {rel(WS_ROOT)} "
              f"(fresh split){RESET}")
        shutil.rmtree(WS_ROOT)

    print(f"{BOLD}ghostbuster_orig (raw) pipeline{RESET}")
    print(f"  output   {rel(OUT)}")
    print(f"  gpu      {args.gpu if args.gpu is not None else 'not set'}")
    print(f"  stages   {' -> '.join(s.key for s in stages)}")
    cols = [c for s in stages for c in s.produces]
    print(f"  columns  {len(cols)}")
    if args.dry_run:
        print(f"\n{YELLOW}--dry-run: nothing runs.{RESET}")
    elif not args.yes:
        print(f"\n{YELLOW}NOTE: --yes not given — the scoring stages will stop "
              f"to ask before writing.{RESET}")

    notify = notify_utils.configure(
        "gb_orig", STEM, [s.key for s in stages], enabled=not args.dry_run
    )
    notify.started(f"{len(stages)} stages")

    results: list[tuple[Stage, int, float, str]] = []
    failed: Stage | None = None
    n_blank: int | None = None

    for stage in stages:
        code, note, secs = 0, "", 0.0
        for attempt in range(args.fill_retries + 1):
            try:
                code, s = run(stage, args.dry_run)
            except KeyboardInterrupt:
                print(f"\n{YELLOW}Interrupted during {stage.key}.{RESET}")
                code, s = INTERRUPTED, 0.0
            secs += s

            if args.dry_run or code == INTERRUPTED:
                break
            if code != 0:
                if attempt >= args.fill_retries:
                    note = f"exit {code}"
                    break
                print(f"\n{YELLOW}{stage.key} exited {code} — retry "
                      f"{attempt + 1}/{args.fill_retries} in "
                      f"{args.fill_cooldown:.0f}s{RESET}", flush=True)
                notify.stage(f"{stage.key} retry {attempt + 1}", code, s, critical=False)
                time.sleep(args.fill_cooldown)
                continue

            if n_blank is None and OUT.is_file():
                n_blank = blank_text_count(OUT)
            complete, msgs, gap = verify(stage, OUT, n_blank or 0)
            if complete:
                note = "; ".join(msgs)
                break
            if gap == 0 or attempt >= args.fill_retries:
                code = 1
                note = "; ".join(msgs) or f"{gap} row(s) unfilled"
                print(f"\n{RED}{stage.key}: not usable:{RESET}\n    "
                      + "\n    ".join(msgs))
                break
            print(f"\n{YELLOW}{stage.key}: {gap} row(s) still Null — retry "
                  f"{attempt + 1}/{args.fill_retries} in {args.fill_cooldown:.0f}s"
                  f"{RESET}", flush=True)
            notify.stage(f"{stage.key} retry {attempt + 1}", 0, s, critical=False)
            time.sleep(args.fill_cooldown)

        results.append((stage, code, secs, note))
        notify.stage(stage.key, code, secs, critical=True)
        if code != 0:
            failed = stage
            break

    print(f"\n{BOLD}{'-' * 60}\n  summary\n{'-' * 60}{RESET}")
    for stage, code, secs, note in results:
        mark = (GREEN + "ok" if code == 0 else
                YELLOW + "interrupted" if code == INTERRUPTED else
                RED + f"FAILED ({code})")
        print(f"  {stage.key:<27}{mark:<22}{RESET}{secs:>8.1f}s"
              + (f"   {DIM}{note}{RESET}" if note else ""))
    total = sum(s for _, _, s, _ in results)
    print(f"\n  total {total:.1f}s")

    if failed is not None:
        resume = " ".join([
            "uv run python epai/ai_detection/prepare/run_ghostbuster_orig_pipeline.py",
            f"--from {failed.key}",
            *([f"--gpu {args.gpu}"] if args.gpu is not None else []),
            *(["--yes"] if args.yes else []),
        ])
        notify.failed(failed.key, results[-1][1], results[-1][2], resume=resume)
        print(f"\n{RED}Stopped at '{failed.key}'.{RESET}  Fix the cause, then:\n"
              f"    {resume}")
        sys.exit(1)

    if args.dry_run:
        return
    notify.finished(f"{rel(OUT)}", total)
    print(f"\n{GREEN}Done.{RESET} {rel(OUT)} — every column filled "
          f"(text intentionally NOT standardised).\n"
          f"  Register gb_orig_stand in splits.py, update the README, then "
          f"rebuild_all.py — see {PREPARE_DIR}/GHOSTBUSTER_ORIG_SPLIT.md")


if __name__ == "__main__":
    main()

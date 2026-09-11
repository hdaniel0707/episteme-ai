r"""Build ``ai_scientist_hc.parquet`` end to end, unattended.

The ai_scientist sibling of ``run_ghostbuster_orig_pipeline.py`` -- same reason
to exist: ``run_holdout_pipeline.py``'s own stage list (embed -> binox ->
fastdetect -> detector -> judges -> cue-reasons/features) is what
``denario_detect.parquet`` / ``denario_hc.parquet`` actually have today, but it
lacks everything added since (the logprob-window judge readouts, the 'base'
control-prompt baseline, the soft-min ensemble). This builds the FULLER set --
the same one ``ghostbuster_orig.parquet`` has -- so a later merge with Denario
does not have to explain away a thinner column set on the new side too.

INPUT: ``data/parquet/ai_scientist_hc_raw.parquet`` -- 20 machine + 20 matched
human rows, written by ``build_denario_hc.py`` against
``ai_scientist_detect.parquet`` (see ``--denario`` / ``--id-prefix aiscihc`` in
that script). NOT standardised, NOT scored. This pipeline:

    1. ocr-clean    repair PDF-extraction artifacts in BOTH classes (same
                     rule denario_hc applies -- an asymmetry here would let a
                     detector key on provenance instead of authorship). Only
                     3 of the 20 machine rows (sakanav2, no LaTeX source) and
                     6 (cycleresearcher, no LaTeX source) went through raw PDF
                     extraction; the other 10 (sakana v1, LaTeX) and 1 (zochi,
                     arXiv API) are already clean prose -- ocr_clean.py is a
                     no-op on text that has nothing to repair.
    2. standardize  ``--whitespace flat``, same stage/setting denario_hc used
                     (matches science_v3). MUST precede every scoring stage --
                     see run_holdout_pipeline.py's own note on this.
    3..N  the full detector suite, same shape as ghostbuster_orig's pipeline:
          embed -> binox(+extended) -> fastdetect -> radar -> the 5 suite
          judges (v3) -> cue-reasons/features -> logprob capture
          (gpt56luna, gemma4-31b) / merge / windows -> 'base' baseline capture
          / merge -> soft-min ensemble -> verify.

CAUTION -- TWO THINGS THAT BIT THE ghostbuster_orig RUN, HANDLED HERE
----------------------------------------------------------------------
* GPU FLAG: commit 39025ab fixed run_ghostbuster_orig_pipeline.py because two
  stages (a judge, and cue-reasons) were built WITHOUT ``*gpu`` in their arg
  list -- ``add_llm_judge_scores.py`` / ``add_llm_judge_reasons.py`` both take
  ``--gpu`` (the local judge, qwen3-14b, runs on it) and silently fell back to
  "all visible GPUs" instead of the one asked for. Every stage below that
  wraps a script accepting ``--gpu`` gets it -- checked against each script's
  own argparse, not assumed by analogy. ocr-clean, standardize, cue-features,
  the logprob/baseline capture-and-merge scripts, and ensemble take no
  ``--gpu`` at all (verified: none of their argparsers define it) and do not
  get the flag -- passing it would be an unrecognised argument, not a no-op.

* MAX TOKENS ON THE REAL OPENAI API: ``add_llm_judge_scores.py``'s
  ``_score_openai`` (used for the ``gpt56luna`` judge) and
  ``add_llm_judge_reasons.py``'s ``_explain_openai`` called the real OpenAI
  API's ``.parse()`` with NO token cap at all -- unlike the Anthropic path
  (``max_tokens=256``) and the GenAI4Science path (``max_tokens=16384``,
  because gpt-oss:120b can spend thousands of tokens on its reasoning channel
  before writing the answer). The real API has the SAME reasoning-token risk
  but names the parameter differently (``max_completion_tokens``, not
  ``max_tokens`` -- a reasoning model rejects the old name outright). Fixed in
  both files (``OPENAI_MAX_TOKENS = 16384``) as part of this change, so this
  pipeline is not the first thing to hit it.

WORKERS -- run this AS FAST AS THE PROVIDERS ALLOW
----------------------------------------------------
20 rows total (well under any real API's rate limit), so ``--api-workers``
defaults to 20 here (gb_orig's 6000-row corpus used 8) -- one row in flight
per worker, the whole judge finishes in roughly one request's latency instead
of 20 sequential ones. ``--g4s-workers`` stays capped at 4 regardless: that is
a hard ceiling GenAI4Science's performance endpoint enforces (a 5th concurrent
call trips a ~1-minute key block), not a preference to push against.

Every stage is a separate process with its own progress bars and exit code; a
stage that exits 0 having filled nothing is caught by the column check after
it, and re-run (the scorers skip filled rows) up to ``--fill-retries`` times.
Set ``--yes`` for an unattended run. ``NTFY_PIPELINE_TOPIC`` in ``.env`` -> a
phone push at every stage boundary.

Run:
    # first, once (already done -- see epai/science_data_denario/README.md /
    # the ai_scientist_detect.parquet + build_denario_hc.py conversation):
    #   data/parquet/ai_scientist_hc_raw.parquet must exist.
    uv run python epai/ai_detection/prepare/run_ai_scientist_hc_pipeline.py --dry-run
    uv run python epai/ai_detection/prepare/run_ai_scientist_hc_pipeline.py --yes --gpu 0

    uv run python epai/ai_detection/prepare/run_ai_scientist_hc_pipeline.py \
        --from logprob-capture-gpt56luna --yes --gpu 0    # resume
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import argparse

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

STEM = "ai_scientist_hc"
RAW_IN = BASE_DIR / "data" / "parquet" / "ai_scientist_hc_raw.parquet"
CANDIDATES = BASE_DIR / "data" / "science_denario" / "ai_scientist_hc_candidates.parquet"
CLEAN = BASE_DIR / "data" / "parquet" / "ai_scientist_hc_clean.parquet"
OUT = BASE_DIR / "data" / "parquet" / f"{STEM}.parquet"

WS_ROOT = BASE_DIR / "data" / "parquet" / "tmp" / "ai_scientist_hc_pipe"

CUE_JUDGE = "gpt56luna"
LOGPROB_MODELS = ("gpt56luna", "gemma4-31b")
BASELINE_MODELS = ("gpt56luna", "gemma4-31b")
ENSEMBLE_TAUS = ("0.05", "0.1", "0.15", "0.2")


def _stem(model: str) -> str:
    return judge_stem(model)


def _row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return 0


def build_stages(args: argparse.Namespace) -> list[Stage]:
    out = str(OUT)
    gpu = ["--gpu", str(args.gpu)] if args.gpu is not None else []
    cpu = ["--cpu-threads", str(args.cpu_threads)] if args.cpu_threads else []
    yes = ["--yes"] if args.yes else []
    api_w = ["--workers", str(args.api_workers)]
    g4s_w = ["--workers", str(min(args.api_workers, args.g4s_workers))]
    g4s = _suite_judge_kinds().get("genai4science", set())

    per_band = max(args.per_band, _row_count(OUT if OUT.is_file() else RAW_IN))
    lp_ws = {m: WS_ROOT / "logprob" / m for m in LOGPROB_MODELS}
    base_ws = {m: WS_ROOT / "baseline" / m for m in BASELINE_MODELS}

    stages: list[Stage] = []

    if args.clean:
        # No --gpu / --cpu-threads: neither script's argparser defines them
        # (checked, not assumed -- see the module docstring's CAUTION section).
        ocr_args = [str(RAW_IN), "-o", str(CLEAN)]
        if CANDIDATES.is_file():
            ocr_args += ["--extra-text-parquet", str(CANDIDATES)]
        for w in args.ocr_keep_word:
            ocr_args += ["--keep-word", w]
        stages.append(Stage(
            "ocr-clean", "epai/science_data_denario/ocr_clean.py",
            ocr_args, produces=[], reads_text=False,
        ))
        stages.append(Stage(
            "standardize", f"{PREPARE_DIR}/standardize_text.py",
            [str(CLEAN), "-o", out, "--whitespace", "flat"],
            produces=["text", "text_raw"], reads_text=False,
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
    # merge (uid left-join), then the P(human) readouts. Neither capture
    # script takes --gpu or --cpu-threads (checked).
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

    # baseline ('base' control judge prompt) -- same capture -> merge shape.
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
        # --allow-incomplete-groups: this corpus pairs ONE human to ONE
        # specific generator per group (sakana xor sakanav2 xor
        # cycleresearcher xor zochi -- never more than one), unlike mg_gb/
        # mg_sci where every human has a version from every generator. The
        # checker's "every group should carry its domain's full generator
        # set" rule assumes the latter shape and would otherwise ERROR on
        # every group for correctly NOT having the other three generators --
        # not a real gap, nothing to regenerate. mg_den never hits this
        # because it only ever has one generator label (denario) per domain;
        # this is the first corpus with >1 generator inside a 1:1-matched
        # structure.
        [out, "--fail-on", "error", "--allow-incomplete-groups"],
        produces=[], reads_text=False,
    ))
    return stages


def _check_raw_exists() -> None:
    if not RAW_IN.is_file():
        raise SystemExit(
            f"{RED}{rel(RAW_IN)} not found.{RESET}\nBuild it first:\n"
            f"  uv run python -m epai.science_data_denario.fetch_human_candidates "
            f"--denario data/parquet/ai_scientist_detect.parquet "
            f"-o {rel(CANDIDATES)} --yes\n"
            f"  uv run python -m epai.science_data_denario.build_denario_hc "
            f"--denario data/parquet/ai_scientist_detect.parquet "
            f"--candidates {rel(CANDIDATES)} --id-prefix aiscihc "
            f"--report-dir results/ai_scientist_hc -o {rel(RAW_IN)} "
            f"--gpu 0 --strip-structured-labels"
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
    p.add_argument("--clean", action="store_true", default=True,
                   help="run ocr-clean + standardize first (default: on -- pass "
                        "--no-clean only if OUT already exists, standardised, "
                        "from a previous run of this pipeline)")
    p.add_argument("--no-clean", dest="clean", action="store_false")
    p.add_argument("--ocr-keep-word", action="append", default=[], metavar="WORD",
                   help="passthrough to ocr_clean.py --keep-word (repeatable); "
                        "run ocr_clean.py --dry-run on the raw file by hand "
                        "first if you want to check before committing")
    p.add_argument("--gpu", default=None, metavar="ID")
    p.add_argument("--cpu-threads", type=int, default=None, metavar="N")
    p.add_argument("--embedders", nargs="+", default=list(DEFAULT_EMBEDDERS))
    p.add_argument("--binox-pairs", nargs="+", default=list(DEFAULT_BINOX_PAIRS))
    p.add_argument("--fastdetect-pairs", nargs="+",
                   default=list(DEFAULT_FASTDETECT_PAIRS))
    p.add_argument("--detectors", nargs="+", default=list(DEFAULT_DETECTORS))
    p.add_argument("--judges", nargs="+", default=list(DEFAULT_JUDGES))
    p.add_argument("--per-band", type=int, default=DEFAULT_PER_BAND)
    p.add_argument("--api-workers", type=int, default=20, metavar="N",
                   help="rows in flight per API judge (default 20 -- this "
                        "corpus is 20 machine + 20 human rows, so this is "
                        "already maximum useful parallelism)")
    p.add_argument("--g4s-workers", type=int, default=4, metavar="N",
                   help="HARD CAP from GenAI4Science's performance endpoint -- "
                        "a 5th concurrent call trips a ~1-minute key block. Do "
                        "not raise this past 4.")
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

    running_clean = args.clean and not (args.only and "ocr-clean" not in args.only
                                         and "standardize" not in args.only)
    if not running_clean and not OUT.is_file():
        reason = (
            "--no-clean given" if not args.clean
            else f"--only {args.only} does not include 'ocr-clean' or 'standardize'"
        )
        raise SystemExit(
            f"{RED}{rel(OUT)} not found and {reason}.{RESET} Either run the "
            f"clean/standardize stages first, or point --from/--only past "
            f"'standardize' only once {rel(OUT)} already exists."
        )
    if running_clean:
        _check_raw_exists()

    stages = select(build_stages(args), args)
    if not stages:
        raise SystemExit("no stages selected")

    if any(s.key == "ocr-clean" for s in stages) and not args.dry_run and WS_ROOT.is_dir():
        print(f"{DIM}wiping capture workspaces under {rel(WS_ROOT)} "
              f"(fresh clean/standardize invalidates them){RESET}")
        shutil.rmtree(WS_ROOT)

    print(f"{BOLD}ai_scientist_hc pipeline{RESET}")
    print(f"  output       {rel(OUT)}")
    print(f"  gpu          {args.gpu if args.gpu is not None else 'not set'}")
    print(f"  api-workers  {args.api_workers}   g4s-workers  {args.g4s_workers}")
    print(f"  stages       {' -> '.join(s.key for s in stages)}")
    cols = [c for s in stages for c in s.produces]
    print(f"  columns      {len(cols)}")
    if args.dry_run:
        print(f"\n{YELLOW}--dry-run: nothing runs.{RESET}")
    elif not args.yes:
        print(f"\n{YELLOW}NOTE: --yes not given — the scoring stages will stop "
              f"to ask before writing.{RESET}")

    notify = notify_utils.configure(
        "ai_scientist_hc", STEM, [s.key for s in stages], enabled=not args.dry_run
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
            "uv run python epai/ai_detection/prepare/run_ai_scientist_hc_pipeline.py",
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
    print(f"\n{GREEN}Done.{RESET} {rel(OUT)} — every column filled.\n"
          f"  Next: the merge script (ai_scientist_hc.parquet + denario_hc.parquet).")


if __name__ == "__main__":
    main()

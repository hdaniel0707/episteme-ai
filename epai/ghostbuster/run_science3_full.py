r"""Run the whole 3-generator science corpus end to end, unattended.

Chains everything from the two new-generator pipelines through to a verified,
combined corpus, so it can be started once and left running:

    nohup uv run python epai/ghostbuster/run_science3_full.py --yes --gpu 0 \
        > results/ghostbuster_science3.log 2>&1 &

Six steps, in order:

  1. **gptoss120b**       — the whole ``run_full_pipeline.py`` chain (generate
     through embed) for ``cfg/ghostbuster/gb_gptoss120b.yaml``.
  2. **llama3370b**       — same, for ``cfg/ghostbuster/gb_llama33_70b.yaml``.
  3. **verify_gpt56luna** — ``check_parquet_completeness.py`` +
     ``audit_corpus.py`` on the EXISTING ``ghostbuster_gpt56luna.parquet``
     (reused as-is, never regenerated here — see step 3's own docstring).
  4. **combine**          — ``build_ghostbuster_multi.py`` unions the three
     into ``data/parquet/ghostbuster_multi_sci3.parquet``.
  5. **verify_combined**  — the same two checks on the combined file.
  6. **judge**            — ``add_llm_judge_scores.py`` on the combined file,
     with the same (model, prompt) as ``--judge-reference`` already carries.
     OFF by default (``--with-judge`` to opt in) — it is a second LLM bill,
     separate from generation, and worth running once the corpus above is
     confirmed good rather than blind.

**Stops at the first failure.** Nothing after it runs, and nothing before it
is lost: every step here either wrote a file the next step depends on, or
verified one that a previous run already wrote — restarting a step from
scratch costs nothing it hasn't already paid for, because every tool this
script calls is itself resumable (``run_full_pipeline.py``'s own generate
shard, ``score_parquets.py`` and ``add_embeddings.py``'s "fill only the
holes" default). So resuming after a failure is just re-running from the step
that failed:

    uv run python epai/ghostbuster/run_science3_full.py --from combine --yes

**Notifications.** If ``NTFY_PIPELINE_TOPIC`` is set (see ``.env_sample`` and
``epai/utils/notify_utils.py``), a push goes to your phone when this starts,
after every step, and — the one that matters while you are away — the moment
anything fails, with the exact ``--from`` command to resume. Test the wiring
before a long unattended run:

    uv run python -m epai.utils.notify_utils

**Nothing here blocks on stdin.** Every subprocess is invoked with its own
``--yes``, so a run left overnight cannot stall at a confirmation prompt it
has no one to answer. Run it under ``tmux``/``screen`` or ``nohup`` regardless
— an SSH disconnect kills a plain foreground process along with it.

Run:
    uv run python epai/ghostbuster/run_science3_full.py --yes --gpu 0
    uv run python epai/ghostbuster/run_science3_full.py --from combine --yes
    uv run python epai/ghostbuster/run_science3_full.py --with-judge --yes --gpu 0
    uv run python epai/ghostbuster/run_science3_full.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from epai.ghostbuster.config import BASE_DIR
from epai.utils import notify_utils

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

INTERRUPTED = 130

STEP_KEYS = (
    "gptoss120b", "llama3370b", "verify_gpt56luna", "combine",
    "verify_combined", "judge",
)

CFG_DIR = BASE_DIR / "cfg" / "ghostbuster"
PARQUET_DIR = BASE_DIR / "data" / "parquet"
RUN_FULL_PIPELINE = "epai/ghostbuster/run_full_pipeline.py"
COMPLETENESS_SCRIPT = "epai/scripts/check_parquet_completeness.py"
AUDIT_SCRIPT = "epai/ai_detection/analyse/audit_corpus.py"
COMBINE_SCRIPT = "epai/ai_detection/prepare/build_ghostbuster_multi.py"
JUDGE_SCRIPT = "epai/ai_detection/score/add_llm_judge_scores.py"

GPT56LUNA_CORPUS = PARQUET_DIR / "ghostbuster_gpt56luna.parquet"
GPTOSS120B_CORPUS = PARQUET_DIR / "ghostbuster_gptoss120b.parquet"
LLAMA3370B_CORPUS = PARQUET_DIR / "ghostbuster_llama3370b.parquet"
COMBINED_CORPUS = PARQUET_DIR / "ghostbuster_multi_sci3.parquet"
DEFAULT_JUDGE_REFERENCE = PARQUET_DIR / "science_v3_multi_0811A.parquet"


@dataclass
class Result:
    key: str
    code: int
    seconds: float


def banner(key: str, detail: str) -> None:
    print(f"\n{BOLD}{'━' * 74}{RESET}")
    print(f"{BOLD}  {key}{RESET}{DIM}   {detail}{RESET}")
    print(f"{BOLD}{'━' * 74}{RESET}")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def run_cmd(cmd: list[str], dry_run: bool, env: dict[str, str] | None = None) -> int:
    print(f"{DIM}$ {' '.join(cmd)}{RESET}\n", flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=BASE_DIR, env=env).returncode


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_full_pipeline(args: argparse.Namespace, cfg_name: str, step_key: str) -> int:
    """One full ``run_full_pipeline.py`` run (generate through embed)."""
    cfg = CFG_DIR / cfg_name
    banner(step_key, f"{RUN_FULL_PIPELINE} --cfg {rel(cfg)}")

    cmd = [sys.executable, str(BASE_DIR / RUN_FULL_PIPELINE), "--cfg", rel(cfg), "--yes"]
    if args.gpu is not None:
        cmd += ["--gpu", args.gpu]
    if args.cpu_threads is not None:
        cmd += ["--cpu-threads", str(args.cpu_threads)]
    if args.gen_limit is not None:
        cmd += ["--gen-limit", str(args.gen_limit)]
    if args.score_limit is not None:
        cmd += ["--score-limit", str(args.score_limit)]
    if args.embed_limit is not None:
        cmd += ["--embed-limit", str(args.embed_limit)]
    if args.dry_run:
        cmd.append("--dry-run")

    # Folds this sub-pipeline's own per-stage pushes into THIS run's counter
    # instead of a second, independent one — see notify_utils.nested_env.
    return run_cmd(cmd, args.dry_run, env=notify_utils.nested_env(step_key))


def _verify_one(args: argparse.Namespace, path: Path, reference: Path | None = None) -> int:
    """``check_parquet_completeness.py`` + ``audit_corpus.py`` on one file.

    Both gate on ERROR (their own default), so either finding a refusal saved
    as content, a broken pairing, or a missing score/embedding stops the run
    here rather than downstream. ``--max-essay-human-blanks`` exempts (up to a
    cap of 6) the known blank human seeds the upstream essay release ships,
    which every one of these files carries and nothing here can regenerate.
    """
    if not args.dry_run and not path.is_file():
        print(f"{RED}Not found: {rel(path)}{RESET}")
        return 1

    cmd = [sys.executable, str(BASE_DIR / COMPLETENESS_SCRIPT), rel(path),
           "--max-essay-human-blanks"]
    if reference is not None:
        cmd += ["--reference", rel(reference)]
    code = run_cmd(cmd, args.dry_run)
    if code != 0:
        return code

    return run_cmd(
        [sys.executable, str(BASE_DIR / AUDIT_SCRIPT), rel(path)], args.dry_run
    )


def step_verify_gpt56luna(args: argparse.Namespace) -> int:
    """Confirm the REUSED gpt56luna corpus is clean before folding it in.

    Never regenerated here — per the plan, this leg is existing text, and this
    step is exactly the safety net for reusing it unseen: a refusal or a
    pairing gap in it is caught now, not after the combine step has already
    unioned it into the corpus you are about to score and embed the other two
    generators against.
    """
    banner("verify_gpt56luna", rel(GPT56LUNA_CORPUS))
    return _verify_one(args, GPT56LUNA_CORPUS)


def step_combine(args: argparse.Namespace) -> int:
    """Union human + the three generators into one corpus.

    A separate file (``ghostbuster_multi_sci3.parquet``), not the existing
    ``ghostbuster_multi.parquet`` — this is the human/gpt56luna/gpt-oss:120b/
    llama3.3:70b corpus, kept apart from the gpt35t/gpt54mini/gpt56luna one so
    neither run touches the other's file.
    """
    banner("combine", COMBINE_SCRIPT)
    if not args.dry_run and COMBINED_CORPUS.exists() and not args.overwrite_combined:
        print(
            f"{YELLOW}{rel(COMBINED_CORPUS)} already exists.{RESET}\n"
            "  Pass --overwrite-combined to rebuild it (e.g. after a re-run of "
            "an earlier step), or remove it by hand first."
        )
        return 1

    cmd = [
        sys.executable, str(BASE_DIR / COMBINE_SCRIPT),
        "-o", rel(COMBINED_CORPUS),
        "--source", "ghostbuster_gpt56luna", "human",
        "--source", "ghostbuster_gpt56luna", "ai",
        "--source", "ghostbuster_gptoss120b", "ai",
        "--source", "ghostbuster_llama3370b", "ai",
    ]
    return run_cmd(cmd, args.dry_run)


def step_verify_combined(args: argparse.Namespace) -> int:
    """The same two checks, on the combined corpus, against the science shape."""
    banner("verify_combined", rel(COMBINED_CORPUS))
    reference = args.judge_reference if args.judge_reference.is_file() else None
    return _verify_one(args, COMBINED_CORPUS, reference)


def _detect_judge_args(reference: Path) -> tuple[str, str]:
    """``(model, prompt)`` read off the first ``llm_judge_score_<model>_<prompt>``
    column in ``reference``, so the ghostbuster run judges with the exact same
    (model, prompt) pair science_v3_multi already carries, rather than a
    guessed one. Raises if the reference has no such column."""
    prefix = "llm_judge_score_"
    names = [c for c in pq.ParquetFile(reference).schema_arrow.names
             if c.startswith(prefix) and not c.endswith("_lp")]
    if not names:
        raise SystemExit(
            f"{RED}No {prefix}<model>_<prompt> column in {rel(reference)}.{RESET}\n"
            "  Pass --judge-model/--judge-prompt directly, or --judge-reference "
            "another parquet that already carries a judge score column."
        )
    model, _, prompt = names[0][len(prefix):].rpartition("_")
    return model, prompt


def step_judge(args: argparse.Namespace) -> int:
    """LLM-as-judge on the combined corpus, same (model, prompt) as science."""
    banner("judge", JUDGE_SCRIPT)

    if args.judge_model and args.judge_prompt:
        model, prompt = args.judge_model, args.judge_prompt
    elif args.dry_run and not args.judge_reference.is_file():
        model, prompt = "<detected>", "<detected>"
    else:
        model, prompt = _detect_judge_args(args.judge_reference)
    print(f"{DIM}  judging with --model {model} --prompt {prompt} "
          f"(from {rel(args.judge_reference)}){RESET}")

    cmd = [
        sys.executable, str(BASE_DIR / JUDGE_SCRIPT), rel(COMBINED_CORPUS),
        "--model", model, "--prompt", prompt, "--yes",
    ]
    if args.gpu is not None:
        cmd += ["--gpu", args.gpu]
    if args.cpu_threads is not None:
        cmd += ["--cpu-threads", str(args.cpu_threads)]
    return run_cmd(cmd, args.dry_run)


# ---------------------------------------------------------------------------


def select(args: argparse.Namespace) -> list[str]:
    steps = list(STEP_KEYS)
    if not args.with_judge:
        steps.remove("judge")
    if args.start_from:
        if args.start_from not in steps:
            raise SystemExit(
                f"unknown or excluded step {args.start_from!r}. Choose from: {steps}"
                + ("" if args.with_judge else " (add --with-judge for 'judge')")
            )
        steps = steps[steps.index(args.start_from):]
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--from", dest="start_from", metavar="STEP",
                        help=f"start here instead of at the beginning: {', '.join(STEP_KEYS)}")
    parser.add_argument("--with-judge", action="store_true",
                        help="also run the LLM-judge step (a separate API bill) "
                             "after the corpus is combined and verified")
    parser.add_argument("--judge-reference", type=Path, default=DEFAULT_JUDGE_REFERENCE,
                        help="parquet to read the judge (model, prompt) off of, "
                             f"and to diff --reference against in verify_combined "
                             f"(default: {rel(DEFAULT_JUDGE_REFERENCE)})")
    parser.add_argument("--judge-model", default=None,
                        help="override the model detected from --judge-reference")
    parser.add_argument("--judge-prompt", default=None,
                        help="override the prompt detected from --judge-reference")
    parser.add_argument("--overwrite-combined", action="store_true",
                        help="rebuild ghostbuster_multi_sci3.parquet even if it "
                             "already exists (e.g. re-running combine by hand)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands without running anything")

    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES, passed through")
    parser.add_argument("--cpu-threads", type=int, default=None, help="passed through")
    parser.add_argument("--gen-limit", type=int, default=None,
                        help="passed through to run_full_pipeline.py --gen-limit")
    parser.add_argument("--score-limit", type=int, default=None,
                        help="passed through to run_full_pipeline.py --score-limit")
    parser.add_argument("--embed-limit", type=int, default=None,
                        help="passed through to run_full_pipeline.py --embed-limit")
    parser.add_argument("--yes", action="store_true",
                        help="accepted for symmetry with the other runners; "
                             "every subprocess this script calls already runs "
                             "unattended (--yes) regardless")

    args = parser.parse_args()
    steps = select(args)

    print(f"{BOLD}ghostbuster science-3 — full run{RESET}")
    print(f"  steps      {' → '.join(steps)}")
    print(f"  combined   {rel(COMBINED_CORPUS)}")
    if args.dry_run:
        print(f"\n{YELLOW}--dry-run: nothing will be executed.{RESET}")

    notify = notify_utils.configure(
        "ghostbuster science3", "gpt56luna + gptoss120b + llama3370b", steps,
        enabled=not args.dry_run,
    )
    notify.started(f"{len(steps)} steps")

    runners = {
        "gptoss120b": lambda: step_full_pipeline(args, "gb_gptoss120b.yaml", "gptoss120b"),
        "llama3370b": lambda: step_full_pipeline(args, "gb_llama33_70b.yaml", "llama3370b"),
        "verify_gpt56luna": lambda: step_verify_gpt56luna(args),
        "combine": lambda: step_combine(args),
        "verify_combined": lambda: step_verify_combined(args),
        "judge": lambda: step_judge(args),
    }

    results: list[Result] = []
    failed = None
    for key in steps:
        start = time.monotonic()
        try:
            code = runners[key]()
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Interrupted during {key}.{RESET}")
            code = INTERRUPTED
        results.append(Result(key, code, time.monotonic() - start))
        notify.stage(key, code, results[-1].seconds)
        if code != 0:
            failed = key
            break

    print(f"\n{BOLD}{'━' * 74}{RESET}")
    print(f"{BOLD}  summary{RESET}")
    print(f"{BOLD}{'━' * 74}{RESET}")
    for r in results:
        mark, colour = ("ok", GREEN) if r.code == 0 else (f"FAILED ({r.code})", RED)
        print(f"  {r.key:<18} {colour}{mark:<14}{RESET} {r.seconds:>8.1f}s")
    ran = {r.key for r in results}
    for key in steps:
        if key not in ran:
            print(f"  {key:<18} {DIM}{'not reached':<14}{RESET}")
    print(f"\n  total {sum(r.seconds for r in results):.1f}s")

    if failed is not None:
        resume = f"uv run python epai/ghostbuster/run_science3_full.py --from {failed}"
        notify.failed(failed, results[-1].code, results[-1].seconds, resume=resume)
        print(f"\n{RED}Stopped at '{failed}'.{RESET}\n  Fix the cause, then resume with:\n    {resume}")
        sys.exit(1)

    if args.dry_run:
        print(f"\n{YELLOW}Dry run — nothing was executed.{RESET}")
        return

    print(f"\n{GREEN}Done.{RESET} {rel(COMBINED_CORPUS)} is built, verified"
          f"{' and judged' if 'judge' in ran else ''}.")
    notify.finished(rel(COMBINED_CORPUS), sum(r.seconds for r in results))


if __name__ == "__main__":
    main()

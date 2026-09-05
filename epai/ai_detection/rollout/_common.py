r"""Shared helpers for the gemma roll-out scripts (``gemma_*.py`` in this dir).

Each ``gemma_*.py`` drives ONE corpus family through the stages it needs to carry
``cfg/detector_suite.yaml``'s new judge ``gemma4-31b`` — and, for the two that
generate, the new generator ``gemma4:31b``. They are separate files on purpose:
a science corpus and a Ghostbuster corpus reach the suite by different routes.
What they share is only this module: how a subprocess is run, and which judges
the suite currently lists.

``--skip-gen`` / ``--skip-judge`` on every script take substrings and match a
model / judge alias loosely (``llama`` hits ``llama3.3:70b`` and ``llama33-70b``),
so one ``--skip-judge llama33-70b`` covers a dead endpoint everywhere.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[3]
SUITE_PATH = BASE_DIR / "cfg" / "detector_suite.yaml"

JUDGE_SCRIPT = "epai/ai_detection/score/add_llm_judge_scores.py"
HOLDOUT_SCRIPT = "epai/ai_detection/prepare/run_holdout_pipeline.py"
COMPLETENESS_SCRIPT = "epai/scripts/check_parquet_completeness.py"
AUDIT_SCRIPT = "epai/ai_detection/analyse/audit_corpus.py"

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def banner(title: str, detail: str = "") -> None:
    print(f"\n{BOLD}{'━' * 74}{RESET}")
    print(f"{BOLD}  {title}{RESET}{DIM}   {detail}{RESET}")
    print(f"{BOLD}{'━' * 74}{RESET}")


def run(cmd: list, dry_run: bool, env: dict | None = None) -> int:
    """Run one subprocess from the repo root; return its exit code."""
    print(f"{DIM}$ {' '.join(str(c) for c in cmd)}{RESET}\n", flush=True)
    if dry_run:
        return 0
    return subprocess.run([str(c) for c in cmd], cwd=BASE_DIR, env=env).returncode


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path)


# --- the detector suite ----------------------------------------------------


def _suite() -> dict:
    try:
        return yaml.safe_load(SUITE_PATH.read_text()) or {}
    except FileNotFoundError:
        return {}


def suite_judges() -> list[str]:
    """Every judge alias listed in ``cfg/detector_suite.yaml``, in suite order."""
    return [j["key"] if isinstance(j, dict) else str(j)
            for j in _suite().get("judges", [])]


def prompt_version() -> str:
    return _suite().get("judge_prompt_version", "v3")


def resolve_judges(skip: list[str] | None) -> list[str]:
    """The suite's judges minus any whose alias contains a ``--skip-judge`` string."""
    judges = suite_judges()
    if skip:
        needles = [s.lower() for s in skip]
        judges = [j for j in judges if not any(n in j.lower() for n in needles)]
    if not judges:
        raise SystemExit(f"{RED}--skip-judge left no judges of {suite_judges()}{RESET}")
    return judges


def match_skip(name: str, skip: list[str] | None) -> bool:
    """True if ``name`` contains one of the ``--skip-*`` substrings (case-insensitive)."""
    return bool(skip) and any(s.lower() in name.lower() for s in skip)


# --- stages every family shares -----------------------------------------------


def add_judges(parquet: Path, judges: list[str], *, workers: int = 4,
               cpu_threads: int | None = None, yes: bool = False,
               dry_run: bool = False) -> int:
    """One ``add_llm_judge_scores.py`` run per judge (it skips filled rows itself)."""
    pv = prompt_version()
    for judge in judges:
        cmd = [sys.executable, BASE_DIR / JUDGE_SCRIPT, parquet,
               "--model", judge, "--prompt", pv, "--workers", workers]
        if cpu_threads:
            cmd += ["--cpu-threads", cpu_threads]
        if yes:
            cmd.append("--yes")
        code = run(cmd, dry_run)
        if code != 0:
            print(f"{RED}judge {judge} failed on {rel(parquet)} (exit {code}).{RESET}")
            return code
    return 0


def holdout_score(parquet: Path, judges: list[str], *, gpu=None,
                  cpu_threads: int | None = None, per_band: int = 20000,
                  yes: bool = False, dry_run: bool = False) -> int:
    """``run_holdout_pipeline.py --from embed`` over an already-standardised corpus.

    Fills every Null in the suite's embed / Binoculars / Fast-DetectGPT /
    detector / judge columns, plus any judge column the corpus never had, over
    ``parquet`` in place. ``--skip-existing`` skips a stage whose columns are
    all full; the scorers still skip filled rows from the inside, so only new
    rows cost anything.
    """
    cmd = [sys.executable, BASE_DIR / HOLDOUT_SCRIPT, "--from", "embed",
           "-o", parquet, "--judges", *judges, "--per-band", per_band,
           "--skip-existing"]
    if gpu is not None:
        cmd += ["--gpu", gpu]
    if cpu_threads:
        cmd += ["--cpu-threads", cpu_threads]
    if yes:
        cmd.append("--yes")
    return run(cmd, dry_run)


def verify_corpus(parquet: Path, *, reference: Path | None = None,
                  dry_run: bool = False) -> int:
    """``check_parquet_completeness.py`` + ``audit_corpus.py`` on one file."""
    if not dry_run and not Path(parquet).is_file():
        print(f"{RED}not found: {rel(parquet)}{RESET}")
        return 1
    cmd = [sys.executable, BASE_DIR / COMPLETENESS_SCRIPT, parquet,
           "--max-essay-human-blanks"]
    if reference is not None:
        cmd += ["--reference", reference]
    code = run(cmd, dry_run)
    if code != 0:
        return code
    return run([sys.executable, BASE_DIR / AUDIT_SCRIPT, parquet], dry_run)


def run_steps(steps: list[str], runners: dict, script_rel: str) -> None:
    """Run ``steps`` in order, stop at the first failure, print a summary."""
    import time

    done: list[tuple[str, int, float]] = []
    failed = None
    for key in steps:
        banner(key)
        start = time.monotonic()
        try:
            code = runners[key]()
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Interrupted during {key}.{RESET}")
            code = 130
        done.append((key, code, time.monotonic() - start))
        if code != 0:
            failed = key
            break

    print(f"\n{BOLD}  summary{RESET}")
    for key, code, secs in done:
        mark, colour = ("ok", GREEN) if code == 0 else (f"FAILED ({code})", RED)
        print(f"  {key:<16} {colour}{mark:<14}{RESET} {secs:>8.1f}s")
    for key in steps:
        if key not in {k for k, _, _ in done}:
            print(f"  {key:<16} {DIM}not reached{RESET}")

    if failed is not None:
        print(f"\n{RED}Stopped at '{failed}'.{RESET} Fix the cause, then resume:\n"
              f"    python {script_rel} --from {failed}")
        raise SystemExit(1)
    print(f"\n{GREEN}Done.{RESET}")

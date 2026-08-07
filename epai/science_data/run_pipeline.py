"""Run every science_data stage in order, once.

    uv run python epai/science_data/run_pipeline.py --yes

That is the whole corpus build: fetch → filter → prompts → generate, with each
stage's analysis report in between. The last thing it prints is the ``mv``
command that promotes the result to ``data/parquet/``; it does not run that for
you, deliberately.

**Stages are separate processes**, not imported functions. Each one is a CLI
with its own argument parsing and its own ``sys.exit`` codes, so running them
as subprocesses means the runner does not have to fake ``sys.argv`` or catch
``SystemExit``, exit codes propagate on their own, and progress bars render as
they normally do. It also means a stage that dies takes nothing with it.

**Analysis stages cannot fail the run.** They are diagnostics — a missing font
or a matplotlib backend problem should not throw away a completed fetch. They
are reported as failures in the summary and the pipeline carries on. The four
real stages are fatal: each one consumes the previous one's output, so
continuing past a failure only produces a confusing second error.

**Analyses are interleaved, not saved for the end**, so a warning arrives while
it is still actionable. `analyse_fetch` reporting a domain below target tells
you the filter stage is about to fail, before stage 3 spends money on prompts.

Resuming after a failure:

    --from prompts           skip everything before the prompt stage
    --only filter generate   run exactly these

Nothing is skipped automatically on the grounds that its output already exists.
The stages that cost money already checkpoint per call and skip completed work
from the inside, and the ones that do not are cheap to redo — whereas guessing
that a parquet on disk is current is how a corpus ends up half-built from two
different configs.

Run:
    uv run python epai/science_data/run_pipeline.py --yes
    uv run python epai/science_data/run_pipeline.py --from prompts
    uv run python epai/science_data/run_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from epai.science_data.config import (
    BASE_DIR,
    DEFAULT_CFG_PATH,
    DEFAULT_QUERIES_PATH,
    final_path,
    generated_by_label,
    load_pipeline_config,
)

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

STAGE_DIR = "epai/science_data"

# oa_fetch exits 130 after a Ctrl-C, having saved a partial corpus. That is not
# a stage failure to report and move past — the corpus is incomplete, so every
# later stage would run on a truncated fetch.
INTERRUPTED = 130


@dataclass
class Stage:
    key: str
    script: str
    args: list[str] = field(default_factory=list)
    # False for the analysis stages: diagnostics, not dependencies.
    critical: bool = True

    @property
    def path(self) -> Path:
        return BASE_DIR / self.script


def build_stages(args: argparse.Namespace, variant: str) -> list[Stage]:
    """The pipeline, in order, with each stage's arguments already resolved."""
    cfg_args = ["--cfg", str(args.cfg)]

    fetch_args = [*cfg_args, "--queries", str(args.queries)]
    if args.yes:
        fetch_args.append("--yes")
    if args.allow_shortfall:
        fetch_args.append("--allow-shortfall")

    retry = ["--retry-failed"] if args.retry_failed else []
    # Resolved once and passed explicitly to both stages that take it, so they
    # cannot disagree about which file they are working on — each derives the
    # same default independently, and an override given to the runner has to
    # reach both.
    var_args = ["--variant", variant]

    return [
        Stage("fetch", f"{STAGE_DIR}/fetch/oa_fetch.py", fetch_args),
        Stage("analyse-fetch", f"{STAGE_DIR}/analyse/analyse_fetch.py",
              cfg_args, critical=False),
        Stage("filter", f"{STAGE_DIR}/filter/filter.py", cfg_args),
        Stage("analyse-filter", f"{STAGE_DIR}/analyse/analyse_filter.py",
              cfg_args, critical=False),
        Stage("prompts", f"{STAGE_DIR}/prompt/make_prompts.py", [*cfg_args, *retry]),
        Stage("analyse-prompts", f"{STAGE_DIR}/analyse/analyse_prompts.py",
              cfg_args, critical=False),
        Stage("generate", f"{STAGE_DIR}/generate/generate_ai.py",
              [*cfg_args, *retry, *var_args]),
        Stage("analyse-generation", f"{STAGE_DIR}/analyse/analyse_generation.py",
              [*cfg_args, *var_args], critical=False),
    ]


def select(stages: list[Stage], args: argparse.Namespace) -> list[Stage]:
    """Apply --from / --only / --skip-analyse."""
    keys = [s.key for s in stages]

    if args.only:
        unknown = [k for k in args.only if k not in keys]
        if unknown:
            raise SystemExit(f"unknown stage(s) {unknown}. Choose from: {keys}")
        # Config order, not the order they were typed: the stages form a chain,
        # and running them out of order feeds a stage its predecessor's stale
        # output rather than the one this run just produced.
        stages = [s for s in stages if s.key in args.only]
    elif args.start_from:
        if args.start_from not in keys:
            raise SystemExit(f"unknown stage {args.start_from!r}. Choose from: {keys}")
        stages = stages[keys.index(args.start_from):]

    if args.skip_analyse:
        stages = [s for s in stages if s.critical]

    return stages


def run(stage: Stage, dry_run: bool) -> tuple[int, float]:
    """Run one stage; return ``(returncode, seconds)``."""
    cmd = [sys.executable, str(stage.path), *stage.args]

    print(f"\n{BOLD}{'━' * 74}{RESET}")
    print(f"{BOLD}  {stage.key}{RESET}{DIM}   {stage.script}{RESET}")
    print(f"{BOLD}{'━' * 74}{RESET}")
    print(f"{DIM}$ {' '.join(cmd[1:])}{RESET}\n", flush=True)

    if dry_run:
        return 0, 0.0

    start = time.monotonic()
    # No capture: stdout and stderr are inherited, so progress bars and prompts
    # behave exactly as they do when the stage is run by hand.
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    return proc.returncode, time.monotonic() - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cfg",
        type=Path,
        default=DEFAULT_CFG_PATH,
        help=f"pipeline config YAML (default: {DEFAULT_CFG_PATH.relative_to(BASE_DIR)})",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help=f"queries_domains YAML (default: {DEFAULT_QUERIES_PATH.relative_to(BASE_DIR)})",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="filename suffix for the generate output. Defaults to the single "
        "generator's label with ':' replaced, or 'multi'.",
    )
    parser.add_argument(
        "--from",
        dest="start_from",
        metavar="STAGE",
        help="start here instead of at the beginning, e.g. --from prompts",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STAGE",
        help="run only these stages (always in pipeline order)",
    )
    parser.add_argument(
        "--skip-analyse",
        action="store_true",
        help="skip the four analysis stages",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="answer yes to the fetch stage's prompts. Needed for an "
        "unattended run — without it the fetch stops to ask before "
        "overwriting an existing raw corpus.",
    )
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="pass --allow-shortfall to the fetch stage",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="pass --retry-failed to the two LLM stages",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands without running anything",
    )
    args = parser.parse_args()

    # Load the config here as well as in every stage: it validates the whole
    # thing including the prompt config, so a typo fails now rather than after
    # the fetch has run.
    cfg = load_pipeline_config(args.cfg)
    generators = cfg["generate"]["generators"]
    variant = args.variant or (
        generated_by_label(generators[0]["model"], cfg).replace(":", "_")
        if len(generators) == 1
        else "multi"
    )

    stages = select(build_stages(args, variant), args)
    if not stages:
        raise SystemExit("no stages selected")

    print(f"{BOLD}science_data pipeline{RESET}")
    print(f"  config       {args.cfg}")
    print(f"  dataset_id   {cfg['dataset_id']}")
    print(f"  prompts      {cfg['prompt_config']}")
    print(f"  labels       {', '.join(generated_by_label(g['model'], cfg) for g in generators)}")
    print(f"  variant      {variant}")
    print(f"  stages       {' → '.join(s.key for s in stages)}")
    if args.dry_run:
        print(f"\n{YELLOW}--dry-run: nothing will be executed.{RESET}")
    elif not args.yes and any(s.key == "fetch" for s in stages):
        print(
            f"\n{YELLOW}NOTE: --yes not given, so the fetch stage will stop to "
            f"ask before overwriting an existing raw corpus.{RESET}"
        )

    results: list[tuple[Stage, int, float]] = []
    failed_critical = None

    for stage in stages:
        try:
            code, seconds = run(stage, args.dry_run)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Interrupted during {stage.key}.{RESET}")
            code, seconds = INTERRUPTED, 0.0

        results.append((stage, code, seconds))

        if code == 0:
            continue

        if code == INTERRUPTED:
            # Whatever the stage saved is partial by definition, so no later
            # stage should consume it.
            print(
                f"\n{YELLOW}{stage.key} was interrupted. Stopping — its output "
                f"is incomplete, so the stages after it would run on a partial "
                f"corpus.{RESET}"
            )
            failed_critical = stage
            break

        if stage.critical:
            failed_critical = stage
            break

        print(
            f"\n{YELLOW}{stage.key} failed (exit {code}). It is a diagnostic "
            f"stage, so the pipeline continues.{RESET}"
        )

    # ------------------------------------------------------------- summary --
    print(f"\n{BOLD}{'━' * 74}{RESET}")
    print(f"{BOLD}  summary{RESET}")
    print(f"{BOLD}{'━' * 74}{RESET}")
    for stage, code, seconds in results:
        if code == 0:
            mark, colour = "ok", GREEN
        elif not stage.critical:
            mark, colour = f"failed ({code}), non-fatal", YELLOW
        else:
            mark, colour = f"FAILED ({code})", RED
        print(f"  {stage.key:<20} {colour}{mark:<26}{RESET} {seconds:>7.1f}s")

    ran = {s.key for s, _, _ in results}
    for stage in stages:
        if stage.key not in ran:
            print(f"  {stage.key:<20} {DIM}{'not reached':<26}{RESET}")

    total = sum(s for _, _, s in results)
    print(f"\n  total {total:.1f}s")

    if failed_critical is not None:
        print(
            f"\n{RED}Pipeline stopped at '{failed_critical.key}'.{RESET}\n"
            f"  Fix the cause, then resume with:\n"
            f"    uv run python epai/science_data/run_pipeline.py "
            f"--from {failed_critical.key}"
        )
        sys.exit(1)

    if args.dry_run:
        print(f"\n{YELLOW}Dry run — nothing was executed.{RESET}")
        return

    if any(s.key == "generate" for s, code, _ in results if code == 0):
        out = final_path(cfg, variant)
        print(
            f"\n{GREEN}Done.{RESET} Promote the result when you are happy with it:\n"
            f"  mv {out.relative_to(BASE_DIR)} data/parquet/{out.name}"
        )


if __name__ == "__main__":
    main()

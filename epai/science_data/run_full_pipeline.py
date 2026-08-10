r"""Build a science corpus and score it with Binoculars, end to end.

    uv run python epai/science_data/run_full_pipeline.py --yes --gpu 0

Four steps, in order:

  1. **science** — the whole ``run_pipeline.py`` chain (fetch → filter →
     prompts → generate, with the analyses interleaved), producing
     ``data/science/<dataset_id>_<variant>.parquet``.
  2. **promote** — *copy* that file to ``data/parquet/``. A copy, not a move:
     ``data/science/`` stays the record of what the run actually produced, so a
     later step that mangles the promoted file cannot destroy the only copy.
  3. **score** — run the Binoculars fork over the promoted file, writing
     ``<name>_binox0.parquet`` beside it. The ``-o`` is passed explicitly:
     ``score_parquets.py`` overwrites its input by default, and the unscored
     promoted copy is worth keeping — it is what gets re-scored when the
     detector or its model pair changes.
  4. **embed** — ``embed/add_embeddings.py`` over the scored file, filling every
     ``emb_*`` column in place. The corpus is then ready for training.

**Step 3 runs in a different environment.** This repo pins
``transformers>=4.57`` and the fork pins ``>=5.12``, so every step except the
scoring one runs under this interpreter and step 3 goes through
``uv run --project external/Binoculars``.
That is the only reason this script shells out for that step instead of
importing ``Binoculars``: the import would resolve against the wrong venv.
``VIRTUAL_ENV`` and ``UV_PROJECT_ENVIRONMENT`` are stripped from that
subprocess's environment — inherited from an outer ``uv run`` they point at this
repo's ``.venv``, which is precisely the env the fork must not use.

**Everything is checked before the fetch runs.** The config, the submodule
checkout, and ``uv`` on PATH are all validated up front, because step 1 costs
API calls and an hour of wall clock and it would be absurd to discover a missing
submodule afterwards.

Resuming, since step 1 is the expensive one:

    --from promote           the corpus is built; copy it, score it, embed it
    --only score             it is already in data/parquet/; just score it
    --science-from prompts   resume *inside* the science pipeline

Stopping early, since steps 3–4 are the GPU-hours:

    --until promote          build the corpus and stop before score and embed
    --from score             later, on a GPU box, finish where that left off

``--from`` and ``--until`` are both inclusive and compose into any window;
``--only`` names steps explicitly and cannot be combined with either.

**A ten-row debug run** of the two GPU steps, over a corpus that already exists.
This is the cheap way to check that scoring and embedding work end to end — ten
rows each, minutes rather than hours, and the output is still a complete copy in
which only those ten rows were touched:

    uv run python epai/science_data/run_full_pipeline.py --only score embed \
        --limit 10 --embed-limit 10 --yes --gpu 0

Unlike pipeline 4, the *corpus* half has no ``--limit``: how many papers get
fetched and generated lives in the config (``fetch.target_per_domain: 10``), since
the fetch stage's stop condition and the filter stage's admission test have to
agree. So a ten-paper end-to-end debug run is that key plus the flags above:

    uv run python epai/science_data/run_full_pipeline.py --yes --gpu 0 \
        --limit 10 --embed-limit 10        # with target_per_domain: 10 in the cfg

Run:
    uv run python epai/science_data/run_full_pipeline.py --yes --cpu-threads 16 --gpu 5
    uv run python epai/science_data/run_full_pipeline.py --from promote --gpu 0
    uv run python epai/science_data/run_full_pipeline.py --until promote --yes
    uv run python epai/science_data/run_full_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from epai.science_data.config import (
    BASE_DIR,
    DEFAULT_CFG_PATH,
    DEFAULT_QUERIES_PATH,
    embed_config,
    final_path,
    generated_by_label,
    load_pipeline_config,
)
from epai.science_data.run_pipeline import (
    BOLD,
    DIM,
    GREEN,
    INTERRUPTED,
    RED,
    RESET,
    YELLOW,
)
from epai.utils.embed_select import check_embedder_names

SCIENCE_RUNNER = "epai/science_data/run_pipeline.py"
BINOCULARS_PROJECT = BASE_DIR / "external" / "Binoculars"
BINOCULARS_SCRIPT = BINOCULARS_PROJECT / "score_parquets.py"
EMBED_SCRIPT = "epai/ai_detection/embed/add_embeddings.py"
PARQUET_DIR = BASE_DIR / "data" / "parquet"

# The suffix this pipeline gives the scored corpus, passed to score_parquets.py
# as -o. It writes in place without one, so this is the only place the name is
# decided.
SCORED_SUFFIX = "_binox0"

STEP_KEYS = ("science", "promote", "score", "embed")


@dataclass
class Result:
    key: str
    code: int
    seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def banner(key: str, detail: str) -> None:
    print(f"\n{BOLD}{'━' * 74}{RESET}")
    print(f"{BOLD}  {key}{RESET}{DIM}   {detail}{RESET}")
    print(f"{BOLD}{'━' * 74}{RESET}")


def run_cmd(cmd: list[str], dry_run: bool, env: dict[str, str] | None = None) -> int:
    """Run one subprocess with inherited stdio; return its exit code."""
    print(f"{DIM}$ {' '.join(cmd)}{RESET}\n", flush=True)
    if dry_run:
        return 0
    # No capture: progress bars and prompts behave as they do by hand.
    return subprocess.run(cmd, cwd=BASE_DIR, env=env).returncode


def confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        return input(f"{question} [y/N]: ").strip().lower() in ("y", "yes")
    except EOFError:
        # Unattended run without --yes. Refusing is the safe reading: the
        # question is only ever asked before overwriting something.
        print(f"\n{YELLOW}No input available; assuming no. Pass --yes to skip "
              f"this prompt.{RESET}")
        return False


def binoculars_env() -> dict[str, str]:
    """The environment for the fork's subprocess.

    Both variables are set by an outer ``uv run`` and point at *this* repo's
    venv; leaving either in place makes ``uv run --project external/Binoculars``
    either warn and ignore it or, worse, honour it and import the wrong
    transformers.
    """
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    return env


def preflight(args: argparse.Namespace, cfg: dict, steps: list[str]) -> None:
    """Fail before step 1 for anything knowable now."""
    if "embed" in steps:
        if not (BASE_DIR / EMBED_SCRIPT).is_file():
            raise SystemExit(f"{RED}{EMBED_SCRIPT} not found.{RESET}")
        # A misspelled embedder would otherwise surface at the very last step,
        # after the fetch, the generation and the scoring have been paid for.
        names = args.embedders or embed_config(cfg).get("embedders")
        try:
            check_embedder_names(names)
        except ValueError as exc:
            raise SystemExit(f"{RED}embedders: {exc}{RESET}")
        except ImportError as exc:
            # torch or transformers missing here means the embed step cannot run
            # at all, but the earlier steps still can — so this is a warning.
            print(f"{YELLOW}NOTE: could not verify the embedder names "
                  f"({str(exc).splitlines()[0]}). They will be checked when the "
                  f"embed step runs.{RESET}")

    if "score" not in steps:
        return

    if not BINOCULARS_SCRIPT.is_file():
        raise SystemExit(
            f"{RED}{BINOCULARS_SCRIPT.relative_to(BASE_DIR)} not found.{RESET}\n"
            "  The Binoculars submodule is not checked out. Run:\n"
            "    git submodule update --init --recursive\n"
            "    uv sync --project external/Binoculars"
        )

    if not (BINOCULARS_PROJECT / ".venv").exists() and not args.dry_run:
        print(
            f"{YELLOW}NOTE: {BINOCULARS_PROJECT.relative_to(BASE_DIR)}/.venv does "
            f"not exist yet. 'uv run --project' will create and sync it on the "
            f"score step, which can take a while (torch).{RESET}"
        )

    if shutil.which("uv") is None:
        raise SystemExit(
            f"{RED}'uv' not found on PATH.{RESET}\n"
            "  The score step needs it: the fork runs in its own environment "
            "and cannot be imported from this one."
        )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_science(args: argparse.Namespace, variant: str) -> int:
    cmd = [
        sys.executable,
        str(BASE_DIR / SCIENCE_RUNNER),
        "--cfg", str(args.cfg),
        "--queries", str(args.queries),
        "--variant", variant,
        # The inner runner's closing advice is to promote the file by hand.
        # This script is about to do exactly that, and printing a stale 'mv'
        # instruction mid-run invites someone to run it and lose the copy in
        # data/science/.
        "--no-hint",
    ]
    if args.science_from:
        cmd += ["--from", args.science_from]
    if args.yes:
        cmd.append("--yes")
    if args.allow_shortfall:
        cmd.append("--allow-shortfall")
    if args.retry_failed:
        cmd.append("--retry-failed")
    if args.skip_analyse:
        cmd.append("--skip-analyse")

    banner("science", SCIENCE_RUNNER)
    return run_cmd(cmd, args.dry_run)


def step_promote(args: argparse.Namespace, src: Path, dest: Path) -> int:
    banner("promote", f"{src.relative_to(BASE_DIR)} → {dest.relative_to(BASE_DIR)}")
    print(f"{DIM}$ cp {src.relative_to(BASE_DIR)} {dest.relative_to(BASE_DIR)}{RESET}\n")

    if args.dry_run:
        return 0

    if not src.is_file():
        print(
            f"{RED}Not found: {src.relative_to(BASE_DIR)}{RESET}\n"
            f"  The generate step writes it. Run the science step first, or "
            f"pass --variant if this run used a different one."
        )
        return 1

    if dest.exists() and not confirm(
        f"{YELLOW}{dest.relative_to(BASE_DIR)} already exists. Overwrite?{RESET}",
        args.yes,
    ):
        print("Aborted.")
        return 1

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    # copy2 rather than copy: the mtime is how a promoted corpus is matched
    # back to the run in data/science/ that produced it.
    shutil.copy2(src, dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"{GREEN}Copied{RESET} {size_mb:.1f} MB to {dest.relative_to(BASE_DIR)}")
    return 0


def step_score(args: argparse.Namespace, dest: Path, scored: Path) -> int:
    cmd = [
        "uv", "run", "--project", str(BINOCULARS_PROJECT.relative_to(BASE_DIR)),
        "python", str(BINOCULARS_SCRIPT.relative_to(BASE_DIR)),
        str(dest.relative_to(BASE_DIR)),
        # Explicit, because score_parquets.py writes IN PLACE by default. The
        # promoted corpus is worth keeping unscored: it is what gets re-scored
        # when the detector or its model pair changes.
        "-o", str(scored.relative_to(BASE_DIR)),
        "--text-column", args.text_column,
        "--batch-size", str(args.batch_size),
    ]
    if args.gpu is not None:
        cmd += ["--gpu", args.gpu]
    if args.cpu_threads is not None:
        cmd += ["--cpu-threads", str(args.cpu_threads)]
    if args.observer:
        cmd += ["--observer", args.observer]
    if args.performer:
        cmd += ["--performer", args.performer]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.yes:
        cmd.append("--yes")

    banner("score", "external/Binoculars/score_parquets.py")

    if not args.dry_run:
        if not dest.is_file():
            print(
                f"{RED}Not found: {dest.relative_to(BASE_DIR)}{RESET}\n"
                f"  Nothing to score. Run the promote step first."
            )
            return 1
        if scored.exists() and not confirm(
            f"{YELLOW}{scored.relative_to(BASE_DIR)} already exists and will be "
            f"overwritten. Continue?{RESET}",
            args.yes,
        ):
            print("Aborted.")
            return 1

    return run_cmd(cmd, args.dry_run, env=binoculars_env())


def step_embed(args: argparse.Namespace, cfg: dict, scored: Path) -> int:
    """Fill every ``emb_*`` column on the scored corpus.

    In place, which is add_embeddings.py's own default: an embedding is derived
    from the text alone, so a separate file would be the same corpus twice.

    Runs under *this* interpreter, unlike the score step — the embedders live in
    ``epai/utils/embedding_utils.py``.

    Which models run comes from the config's optional ``embed.embedders``, or
    ``--embedders`` when given; neither means the script's own ``EMBEDDERS``
    default, which is all thirteen including two billed OpenAI endpoints.
    """
    emb = embed_config(cfg)
    embedders = args.embedders or emb.get("embedders")
    batch_size = (
        args.embed_batch_size if args.embed_batch_size is not None
        else emb.get("batch_size")
    )
    limit = args.embed_limit or emb.get("limit")

    cmd = [
        sys.executable,
        str(BASE_DIR / EMBED_SCRIPT),
        str(scored.relative_to(BASE_DIR)),
        "--text-col", args.text_column,
    ]
    if embedders:
        cmd += ["--embedders", *embedders]
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]
    if args.gpu is not None:
        cmd += ["--gpu", args.gpu]
    if args.cpu_threads is not None:
        cmd += ["--cpu-threads", str(args.cpu_threads)]
    if limit:
        cmd += ["--limit", str(limit)]
    if args.embed_recompute or emb.get("recompute", False):
        cmd.append("--recompute")
    if args.yes:
        cmd.append("--yes")

    banner("embed", EMBED_SCRIPT)

    if not args.dry_run and not scored.is_file():
        print(
            f"{RED}Not found: {scored.relative_to(BASE_DIR)}{RESET}\n"
            f"  Nothing to embed. Run the score step first."
        )
        return 1

    return run_cmd(cmd, args.dry_run)


# ---------------------------------------------------------------------------


def select(args: argparse.Namespace) -> list[str]:
    if args.only:
        if args.start_from or args.stop_after:
            raise SystemExit(
                "--only names the steps explicitly; it cannot be combined with "
                "--from or --until."
            )
        unknown = [k for k in args.only if k not in STEP_KEYS]
        if unknown:
            raise SystemExit(f"unknown step(s) {unknown}. Choose from: {list(STEP_KEYS)}")
        # Pipeline order, not the order they were typed: each step consumes the
        # previous one's output.
        return [k for k in STEP_KEYS if k in args.only]

    # Both bounds inclusive, so --from/--until name steps that run rather than
    # steps that don't — the same reading in both directions.
    start, stop = 0, len(STEP_KEYS)
    if args.start_from:
        if args.start_from not in STEP_KEYS:
            raise SystemExit(
                f"unknown step {args.start_from!r}. Choose from: {list(STEP_KEYS)}"
            )
        start = STEP_KEYS.index(args.start_from)
    if args.stop_after:
        if args.stop_after not in STEP_KEYS:
            raise SystemExit(
                f"unknown step {args.stop_after!r}. Choose from: {list(STEP_KEYS)}"
            )
        stop = STEP_KEYS.index(args.stop_after) + 1

    steps = list(STEP_KEYS[start:stop])
    if not steps:
        raise SystemExit(
            f"--from {args.start_from!r} comes after --until {args.stop_after!r}, "
            f"so no steps are selected. Pipeline order: {list(STEP_KEYS)}"
        )
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    g = parser.add_argument_group("corpus")
    g.add_argument("--cfg", type=Path, default=DEFAULT_CFG_PATH,
                   help=f"pipeline config YAML (default: "
                        f"{DEFAULT_CFG_PATH.relative_to(BASE_DIR)})")
    g.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH,
                   help=f"queries_domains YAML (default: "
                        f"{DEFAULT_QUERIES_PATH.relative_to(BASE_DIR)})")
    g.add_argument("--variant", default=None,
                   help="filename suffix for the generate output. Defaults to "
                        "the single generator's label with ':' replaced, or 'multi'.")

    g = parser.add_argument_group("which steps")
    g.add_argument("--from", dest="start_from", metavar="STEP",
                   help=f"start here instead of at the beginning: {', '.join(STEP_KEYS)}")
    g.add_argument("--until", dest="stop_after", metavar="STEP",
                   help="stop after this step, inclusive. --until promote builds "
                        "and promotes the corpus and leaves the GPU steps (score, "
                        "embed) for a later --from score.")
    g.add_argument("--only", nargs="+", metavar="STEP",
                   help="run only these steps (always in pipeline order). Cannot "
                        "be combined with --from or --until.")
    g.add_argument("--science-from", metavar="STAGE",
                   help="resume inside the science pipeline, e.g. --science-from "
                        "prompts. Passed through as its --from.")
    g.add_argument("--skip-analyse", action="store_true",
                   help="skip the science pipeline's analysis stages")
    g.add_argument("--dry-run", action="store_true",
                   help="print the commands without running anything")

    g = parser.add_argument_group("science pipeline")
    g.add_argument("--yes", action="store_true",
                   help="unattended run: answers yes to the fetch stage's prompts "
                        "and to this script's overwrite prompts")
    g.add_argument("--allow-shortfall", action="store_true",
                   help="pass --allow-shortfall to the fetch stage")
    g.add_argument("--retry-failed", action="store_true",
                   help="pass --retry-failed to the two LLM stages")

    g = parser.add_argument_group("binoculars")
    g.add_argument("--gpu", default=None,
                   help="CUDA_VISIBLE_DEVICES for the score and embed steps, "
                        "e.g. '0' or '0,1'")
    g.add_argument("--cpu-threads", type=int, default=None,
                   help="cap CPU threads used for inference, in both GPU steps")
    g.add_argument("--batch-size", type=int, default=1,
                   help="texts scored per batch (default: 1). The embed step has "
                        "its own per-model batch sizes; see --embed-batch-size.")
    g.add_argument("--text-column", default="text",
                   help="column to score and embed (default: text)")
    g.add_argument("--observer", default=None, help="observer model (default: the fork's)")
    g.add_argument("--performer", default=None, help="performer model (default: the fork's)")
    g.add_argument("--limit", type=int, default=0,
                   help="score only the first N rows, for debugging (0 = all)")

    g = parser.add_argument_group("embed")
    g.add_argument("--embedders", nargs="+", default=None, metavar="NAME",
                   help="which embedders to run, overriding the config's "
                        "embed.embedders. Unset in both means the default list "
                        "in add_embeddings.py.")
    g.add_argument("--embed-batch-size", type=int, default=None,
                   help="texts per forward pass, applied to every embedder. "
                        "Default: the script's own per-model values.")
    g.add_argument("--embed-limit", type=int, default=0,
                   help="embed only the first N rows, for debugging (0 = all)")
    g.add_argument("--embed-recompute", action="store_true",
                   help="recompute embedders that already have values instead of "
                        "filling only the holes")

    args = parser.parse_args()

    # Validated here as well as inside every stage, so a typo fails now rather
    # than after the fetch — and because the variant, and therefore every path
    # below, is derived from it.
    cfg = load_pipeline_config(args.cfg)
    generators = cfg["generate"]["generators"]
    variant = args.variant or (
        generated_by_label(generators[0]["model"], cfg).replace(":", "_")
        if len(generators) == 1
        else "multi"
    )

    src = final_path(cfg, variant)
    dest = PARQUET_DIR / src.name
    scored = dest.with_name(f"{dest.stem}{SCORED_SUFFIX}{dest.suffix}")

    steps = select(args)
    preflight(args, cfg, steps)

    print(f"{BOLD}science_data → binoculars → embeddings{RESET}")
    print(f"  config       {args.cfg}")
    print(f"  dataset_id   {cfg['dataset_id']}")
    print(f"  variant      {variant}")
    print(f"  generated    {src.relative_to(BASE_DIR)}")
    print(f"  promoted     {dest.relative_to(BASE_DIR)}")
    print(f"  scored       {scored.relative_to(BASE_DIR)}")
    print(f"  steps        {' → '.join(steps)}")
    if args.dry_run:
        print(f"\n{YELLOW}--dry-run: nothing will be executed.{RESET}")
    elif not args.yes:
        print(f"\n{YELLOW}NOTE: --yes not given, so this run will stop to ask "
              f"before overwriting anything.{RESET}")

    runners = {
        "science": lambda: step_science(args, variant),
        "promote": lambda: step_promote(args, src, dest),
        "score": lambda: step_score(args, dest, scored),
        "embed": lambda: step_embed(args, cfg, scored),
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

        if code != 0:
            # Every step consumes the one before it, so there is no useful work
            # left to do after a failure — only a second, more confusing error.
            failed = key
            break

    # ------------------------------------------------------------- summary --
    print(f"\n{BOLD}{'━' * 74}{RESET}")
    print(f"{BOLD}  summary{RESET}")
    print(f"{BOLD}{'━' * 74}{RESET}")
    for r in results:
        mark, colour = ("ok", GREEN) if r.code == 0 else (f"FAILED ({r.code})", RED)
        print(f"  {r.key:<12} {colour}{mark:<20}{RESET} {r.seconds:>8.1f}s")
    ran = {r.key for r in results}
    for key in steps:
        if key not in ran:
            print(f"  {key:<12} {DIM}{'not reached':<20}{RESET}")
    print(f"\n  total {sum(r.seconds for r in results):.1f}s")

    if failed is not None:
        print(
            f"\n{RED}Stopped at '{failed}'.{RESET}\n"
            f"  Fix the cause, then resume with:\n"
            f"    uv run python epai/science_data/run_full_pipeline.py --from {failed}"
        )
        sys.exit(1)

    if args.dry_run:
        print(f"\n{YELLOW}Dry run — nothing was executed.{RESET}")
        return

    print(f"\n{GREEN}Done.{RESET}")
    if "embed" in ran:
        print(f"  Corpus, scored and embedded: {scored.relative_to(BASE_DIR)}")
        print("  It is ready for epai/ai_detection/train/ and analyse/.")
    elif "score" in ran:
        print(f"  Scored corpus: {scored.relative_to(BASE_DIR)}")
    elif "promote" in ran:
        print(f"  Promoted corpus: {dest.relative_to(BASE_DIR)}")

    # The steps that were selected away, and how to pick them up. Only worth
    # saying when the run stopped short on purpose — a completed pipeline has
    # nothing left to resume.
    remaining = STEP_KEYS[STEP_KEYS.index(steps[-1]) + 1:] if steps else ()
    if remaining:
        print(f"  Remaining: {' → '.join(remaining)}\n"
              f"  Resume with: uv run python epai/science_data/run_full_pipeline.py "
              f"--from {remaining[0]}")


if __name__ == "__main__":
    main()

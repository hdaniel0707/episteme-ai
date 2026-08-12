r"""Build a Ghostbuster corpus with a newer generator and score it, end to end.

    uv run python epai/ghostbuster/run_full_pipeline.py --yes --gpu 0

Seven stages, in order:

  1. **generate**  — ``external/ghostbuster/generate.py``, once per (domain,
     type), in the fork's own environment. Writes
     ``external/ghostbuster/data/<domain>/<model_slug>_<prompt_id>/``. Runs
     ``generate.max_workers`` calls at a time (``--workers`` overrides); the
     domains themselves are still sequential subprocesses, so that setting is
     the only concurrency the stage has.
  2. **fix-empty** — ``fix_empty_generations.py``, once per (domain, type),
     refilling files that came back blank. Skipped when
     ``generate.fix_empty`` is false.
  3. **collect**   — audit the generated text tree, then *copy* this run's part
     of it to ``data/ghostbuster-data-<variant>/``. A copy, not a move: the
     fork's tree is what makes a re-run of stage 1 free, since every loop there
     skips an output file that already exists. Only the seeded inputs and this
     run's own generator directories are copied — the fork's tree accumulates
     every variant ever generated, and a corpus holds one.
  4. **prepare**   — ``prepare_{essay,wp,reuter}.py`` over the collected tree,
     one parquet per domain in ``data/parquet/``. Told which generator
     directories to read (``--generators human <this run's>``): a collected tree
     is copied into, never emptied, so a directory left by an earlier run into
     the same variant would otherwise add a second set of machine rows under its
     own old label.
  5. **combine**   — ``combine_domains.py`` unions them into
     ``data/parquet/<dataset_id>_<variant>.parquet`` with a ``domain`` column,
     then deletes the per-domain files: every row of them is in the combined
     corpus, nothing downstream reads them, and a leftover
     ``essay_<variant>.parquet`` is a corpus-shaped file waiting to be trained
     on by mistake. ``combine.remove_domain_parquets: false`` keeps them.
  6. **score**     — ``external/Binoculars/score_parquets.py`` over that file,
     writing ``..._binox0.parquet``.
  7. **embed**     — ``embed/add_embeddings.py`` over the scored file, filling
     every ``emb_*`` column in place. The corpus is then ready for training.

**Three environments, and stages 1–2 and 6 run outside this one.** This repo
pins ``transformers>=4.57``; the ghostbuster fork and the Binoculars fork both
pin ``>=5.12``. So those stages go through ``uv run --project external/<fork>``
with ``VIRTUAL_ENV`` and ``UV_PROJECT_ENVIRONMENT`` stripped — inherited from an
outer ``uv run`` they point at this repo's ``.venv``, which is precisely the env
the forks must not use. Stages 1–2 additionally run with ``cwd`` set to the
fork, because ``generate.py`` does ``import utils`` and reads and writes
``data/<domain>/...`` relative to the working directory; neither is
configurable there.

**A generator is named after its model and its prompts, everywhere.** The fork's
own name for a generator (``gpt_plain``) says which prompt was used and nothing
about which model used it, so every model would generate into one directory —
and since the fork skips output files that already exist, the second model to
run there writes nothing and its "corpus" is the first model's text under a new
filename. This pipeline passes ``--out_name <model_slug>_<prompt_id>`` instead
(``gpt56luna_0701A``), and ``prepare_*.py`` turn that back into the corpus label
``gpt56luna:0701A`` — the same ``<model>:<prompts>`` convention the science
pipeline uses. The model half is derived from ``generate.gpt_model``; the prompt
half is ``generate.prompt_ids``, which has to be stated because the prompts
themselves live in the fork.

**The seeded inputs are never written by this pipeline.** ``human/``,
``prompts/`` and reuter's ``gpt/<author>/headlines/`` are placed under
``external/ghostbuster/data/`` by hand (see the README), once, and reused by
every model variant — that is what makes two variants comparable instead of
merely similar. The preflight fails if they are missing, before the first API
call, and a config that asks the fork to regenerate them is rejected outright.

**The collect stage is the gate on whether generation actually finished.**
``generate.py`` records a failed API call and carries on, exiting 0 with the
output file simply absent, so a corpus can lose rows with nothing in the exit
code to show it. The audit counts documents and blank files per (domain, type)
and stops the run on a shortfall (``--allow-shortfall`` to accept one) or on a
blank *generated* file (``--allow-empty``). Blank *human* files are reported and
never fatal: they are seeded input, and the upstream essay corpus ships six.

**``--clean`` starts a run from a clean slate.** These directories are written
to by more than one run: the fork's tree is shared by every model variant, and
the collected tree is copied *into*, never emptied. So a previous run's
generator directory can sit beside this one's, and until the prepare stage was
told which generators to read, that is exactly how a corpus ended up holding its
machine rows twice — once under ``gpt_plain`` and once under
``gpt56luna:0701A``. ``--clean`` removes those leftovers before the first step,
at all three levels:

  1. ``external/ghostbuster/data/<domain>/`` — generator directories that are
     neither seeded inputs nor this run's;
  2. ``data/ghostbuster-data-<variant>/`` — the same, plus whole domains this
     config no longer builds and any stray ``logprobs*`` cache;
  3. ``data/parquet/<domain>_<variant>.parquet`` — but only when ``prepare`` is
     among the selected steps, so it never deletes what a later step needs.

The rule at every level is one rule: *inside the places this config writes,
remove what this config would not produce*. The seeded inputs and this run's own
directories are what a leftover is defined against, so they cannot be selected —
which is what makes the flag safe to pass on every run — and nothing outside
these directories is even looked at. It prints the full list with document
counts and asks before deleting (``--yes`` answers it, ``--dry-run`` shows the
list and deletes nothing). Read the list: the fork-tree entries are usually
another model's generated text, which cost API calls and is kept nowhere else.

Resuming, since stage 1 is the expensive one:

    --from collect            the text is generated; audit, copy, prepare, score
    --only prepare combine    rebuild the parquets from an existing text tree
    --from score              the corpus is combined; just score it

Stopping early, since stages 6–7 are the GPU-hours:

    --until combine           build the corpus and stop before score and embed
    --from score              later, on a GPU box, finish where that left off

``--from`` and ``--until`` are both inclusive and compose into any window;
``--only`` names steps explicitly and cannot be combined with either.

**A ten-item debug run**, all seven stages, nothing billed — the fork mocks every
LLM call under ``--debug``, and only ten rows get scored and embedded, so the
whole thing is minutes rather than a day:

    uv run python epai/ghostbuster/run_full_pipeline.py --yes --debug \
        --gen-limit 10 --score-limit 10 --embed-limit 10 --gpu 0 --until combine --clean

Drop ``--debug`` to spend real API calls on ten items per domain, which is the
run to do before committing to a full one. Read its output knowing three things:

  - the generated text is the literal string ``[DEBUG]`` while ``--debug`` is on,
    so the parquet is structurally real and semantically worthless;
  - the human side is **never** limited, so the corpus holds every human document
    (~1000 per domain) against ten generated ones. The collect stage says so;
  - ``--gen-limit`` caps reuter by position in the flattened (author, index)
    list, so ten items means ten articles by one author.

Run:
    uv run python epai/ghostbuster/run_full_pipeline.py --yes --cpu-threads 16 --gpu 5
    uv run python epai/ghostbuster/run_full_pipeline.py --clean --dry-run
    uv run python epai/ghostbuster/run_full_pipeline.py --clean --yes --gpu 0
    uv run python epai/ghostbuster/run_full_pipeline.py --gen-limit 3 --debug --dry-run
    uv run python epai/ghostbuster/run_full_pipeline.py --until combine --yes
    uv run python epai/ghostbuster/run_full_pipeline.py --from score --gpu 0
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

from epai.ghostbuster.config import (
    BASE_DIR,
    BINOCULARS_PROJECT,
    BINOCULARS_SCRIPT,
    COMBINE_SCRIPT,
    DEFAULT_CFG_PATH,
    FIX_EMPTY_SCRIPT,
    GB_DATA_DIR,
    GENERATE_SCRIPT,
    GHOSTBUSTER_PROJECT,
    GPT_TYPES,
    PARQUET_DIR,
    PREPARE_SCRIPTS,
    SEED_SUBDIRS,
    collected_dir,
    combined_path,
    count_txt,
    domain_parquet,
    domain_source_dir,
    domains,
    expected_documents,
    generate_flag,
    generated_by_label,
    load_pipeline_config,
    missing_seeds,
    output_dir_name,
    output_dir_names,
    scored_path,
    types,
    variant,
)
from epai.utils.embed_select import check_embedder_names

# Defined here rather than imported from another pipeline's runner: these are
# six terminal escapes, and importing a sibling entry point to get them would
# tie this pipeline to that one for nothing.
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# A stage that exits 130 was Ctrl-C'd. Whatever it wrote is partial, so nothing
# after it should consume the result.
INTERRUPTED = 130

STEP_KEYS = (
    "generate", "fix-empty", "collect", "prepare", "combine", "score", "embed"
)

EMBED_SCRIPT = "epai/ai_detection/embed/add_embeddings.py"

# Copied into the collected tree, or not: logprobs are the fork's token
# log-probability caches, written only by its --logprob* stages, which this
# pipeline never runs. prepare_*.py ignore them anyway, so copying them would
# duplicate gigabytes to be skipped later.
COPY_IGNORE = shutil.ignore_patterns("logprobs*")


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


def rel(path: Path) -> str:
    """A repo-relative path for printing, absolute if it lies outside the repo."""
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def run_cmd(
    cmd: list[str],
    dry_run: bool,
    cwd: Path = BASE_DIR,
    env: dict[str, str] | None = None,
    stdin_text: str | None = None,
) -> int:
    """Run one subprocess with inherited stdio; return its exit code."""
    where = "" if cwd == BASE_DIR else f"{DIM}  (cwd {rel(cwd)}){RESET}"
    print(f"{DIM}$ {' '.join(cmd)}{RESET}{where}\n", flush=True)
    if dry_run:
        return 0
    # No capture: progress bars and prompts behave as they do by hand. stdin is
    # inherited too, unless a canned answer is supplied for an unattended run.
    if stdin_text is None:
        return subprocess.run(cmd, cwd=cwd, env=env).returncode
    return subprocess.run(cmd, cwd=cwd, env=env, input=stdin_text, text=True).returncode


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


def submodule_env(cuda: bool = True) -> dict[str, str]:
    """The environment for a subprocess that runs in one of the forks.

    ``VIRTUAL_ENV`` and ``UV_PROJECT_ENVIRONMENT`` are set by an outer ``uv
    run`` and point at *this* repo's venv; leaving either in place makes ``uv
    run --project external/<fork>`` either warn and ignore it or, worse, honour
    it and import the wrong transformers.

    ``cuda=False`` hides every GPU from the subprocess. The generate and
    fix-empty stages are pure API calls, but they ``import utils`` to reach
    ``utils/prompt_utils.py``, and the fork's ``utils/__init__.py`` does
    ``from .write_logprobs import *``, which imports torch — so torch loads,
    probes the driver, and prints a CUDA warning in a stage that will never use
    a GPU. Hiding the devices silences it and, more to the point, keeps an
    hours-long generate run from holding a context on a card another job wants.
    """
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    if not cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def fork_cmd(project: Path, script: str, extra: list[str]) -> list[str]:
    """``uv run --project <fork> python <script> ...``, with an absolute project.

    Absolute because stages 1–2 run with ``cwd`` inside the fork, where a
    repo-relative ``--project external/ghostbuster`` would not resolve.
    """
    return ["uv", "run", "--project", str(project), "python", script, *extra]


def model_args(cfg: dict, gen_type: str | None = None) -> list[str]:
    """``--gpt_model`` / ``--claude_model``, only for the models actually used.

    Passing a model that no selected type will call would put a generator in the
    printed command line that produced none of the text. ``gen_type`` narrows it
    further, to the one model that invocation will call — the fork's stages run
    one type at a time, so that is all any of them needs.
    """
    gen = cfg["generate"]
    selected = [gen_type] if gen_type else types(cfg)
    out: list[str] = []
    if any(t in GPT_TYPES for t in selected):
        out += ["--gpt_model", gen["gpt_model"]]
    if "claude" in selected:
        out += ["--claude_model", gen["claude_model"]]
    return out


def preflight(cfg: dict, args: argparse.Namespace, steps: list[str]) -> None:
    """Fail before stage 1 for anything knowable now."""
    needs_fork = any(k in steps for k in ("generate", "fix-empty"))
    needs_uv = needs_fork or "score" in steps
    # collect needs them too, though it writes nothing there: the seeded inputs
    # are what its audit measures the generated documents against, and an
    # unseeded tree would otherwise report "0 of 0" and pass.
    needs_seeds = needs_fork or "collect" in steps

    if needs_uv and shutil.which("uv") is None:
        raise SystemExit(
            f"{RED}'uv' not found on PATH.{RESET}\n"
            "  The generate and score stages need it: both forks run in their "
            "own environments and cannot be imported from this one."
        )

    if needs_fork:
        script = GHOSTBUSTER_PROJECT / GENERATE_SCRIPT
        if not script.is_file():
            raise SystemExit(
                f"{RED}{rel(script)} not found.{RESET}\n"
                "  The ghostbuster submodule is not checked out. Run:\n"
                "    git submodule update --init\n"
                "    uv sync --project external/ghostbuster"
            )
        if not (GHOSTBUSTER_PROJECT / ".env").is_file() and not args.debug:
            raise SystemExit(
                f"{RED}{rel(GHOSTBUSTER_PROJECT / '.env')} not found.{RESET}\n"
                "  The fork calls load_dotenv() against its own directory, so "
                "the key has to live there:\n"
                "    cp external/ghostbuster/.env_sample external/ghostbuster/.env\n"
                "  then set OPENAI_API_KEY (and ANTHROPIC_API_KEY for claude "
                "rows). Pass --debug to run without keys."
            )
        if not (GHOSTBUSTER_PROJECT / ".venv").exists() and not args.dry_run:
            print(
                f"{YELLOW}NOTE: {rel(GHOSTBUSTER_PROJECT)}/.venv does not exist "
                f"yet. 'uv run --project' will create and sync it on the "
                f"generate stage, which can take a while (torch).{RESET}"
            )

    if needs_seeds:
        gaps = missing_seeds(cfg)
        if gaps:
            listed = "\n".join(f"    {rel(p)}" for p in gaps)
            raise SystemExit(
                f"{RED}The seeded inputs are missing or empty:{RESET}\n{listed}\n"
                "  They are placed by hand and shared by every model variant — "
                "see the README section 'Pipeline 4 — Ghostbuster with a newer "
                "generator' → 'Seed the shared inputs first'. Generating them "
                "here would change the human text and the prompts this corpus "
                "holds fixed."
            )

    if "prepare" in steps:
        for domain in domains(cfg):
            script = BASE_DIR / PREPARE_SCRIPTS[domain]
            if not script.is_file():
                raise SystemExit(f"{RED}{rel(script)} not found.{RESET}")

    if "embed" in steps:
        if not (BASE_DIR / EMBED_SCRIPT).is_file():
            raise SystemExit(f"{RED}{EMBED_SCRIPT} not found.{RESET}")
        # A misspelled embedder would otherwise surface at the very last stage,
        # after the generation and the scoring have been paid for.
        try:
            check_embedder_names(cfg["embed"].get("embedders"))
        except ValueError as exc:
            raise SystemExit(f"{RED}embed.embedders: {exc}{RESET}")
        except ImportError as exc:
            # torch or transformers missing here means the embed stage cannot run
            # at all, but the earlier stages still can — so this is a warning.
            print(f"{YELLOW}NOTE: could not verify embed.embedders "
                  f"({str(exc).splitlines()[0]}). The names will be checked when "
                  f"the embed stage runs.{RESET}")

    if "score" in steps:
        if not BINOCULARS_SCRIPT.is_file():
            raise SystemExit(
                f"{RED}{rel(BINOCULARS_SCRIPT)} not found.{RESET}\n"
                "  The Binoculars submodule is not checked out. Run:\n"
                "    git submodule update --init\n"
                "    uv sync --project external/Binoculars"
            )
        if not (BINOCULARS_PROJECT / ".venv").exists() and not args.dry_run:
            print(
                f"{YELLOW}NOTE: {rel(BINOCULARS_PROJECT)}/.venv does not exist "
                f"yet. 'uv run --project' will create and sync it on the score "
                f"stage, which can take a while (torch).{RESET}"
            )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def step_generate(cfg: dict, args: argparse.Namespace) -> int:
    """One subprocess per (domain, type), in the fork's environment and directory.

    Per (domain, type) rather than one call with every flag, for two reasons: a
    failure names the pair it happened in and a re-run can be pointed at that
    pair alone, and the fork's ``--out_name`` names one directory, so one
    invocation can only describe one type. The fork's own skip-if-exists logic
    makes the repeat free.
    """
    gen = cfg["generate"]
    limit = args.gen_limit if args.gen_limit is not None else gen.get("limit")
    workers = args.workers if args.workers is not None else gen.get("max_workers", 8)

    banner("generate", f"{rel(GHOSTBUSTER_PROJECT)}/{GENERATE_SCRIPT}")
    if args.debug:
        print(f"{YELLOW}--debug: the fork mocks every LLM call. The files it "
              f"writes are placeholders, not a corpus.{RESET}\n")

    for domain in domains(cfg):
        for gen_type in types(cfg):
            out_name = output_dir_name(cfg, gen_type)
            extra = [
                generate_flag(domain, gen_type),
                # What makes this run's documents its own. Without it the fork
                # writes to data/<domain>/<type>/, which is the same directory
                # every model would use, and the second model to run there
                # skips every file and generates nothing.
                "--out_name", out_name,
                *model_args(cfg, gen_type),
                "--seed", str(gen.get("seed", 0)),
                # The fork calls one document at a time per worker, and the
                # domains here run as separate subprocesses, so this is the only
                # concurrency in the stage. It is what separates an overnight
                # run from a coffee-length one.
                "--workers", str(workers),
            ]
            if limit is not None:
                extra += ["--limit", str(limit)]
            if args.debug:
                extra.append("--debug")

            print(f"{BOLD}  {domain}/{gen_type}{RESET} {DIM}→ data/{domain}/"
                  f"{out_name}/{RESET}")
            code = run_cmd(
                fork_cmd(GHOSTBUSTER_PROJECT, GENERATE_SCRIPT, extra),
                args.dry_run,
                cwd=GHOSTBUSTER_PROJECT,
                env=submodule_env(cuda=False),
            )
            if code != 0:
                print(f"{RED}generate failed for {domain}/{gen_type} "
                      f"(exit {code}).{RESET}")
                return code
    return 0


def step_fix_empty(cfg: dict, args: argparse.Namespace) -> int:
    """Refill blank generations, one (domain, type) at a time.

    The fork's script takes exactly one flag per invocation and asks before
    writing anything, so an unattended run passes ``--yes``. It exits 0 when
    there is nothing to do, which is the common case, and also when files are
    left empty — only ``--strict`` there makes that an error, and it must not
    be passed from here: the six blank essay seeds are unfixable and present in
    every run, so a strict fix-empty would abort every pipeline.
    """
    banner("fix-empty", f"{rel(GHOSTBUSTER_PROJECT)}/{FIX_EMPTY_SCRIPT}")

    if not cfg["generate"].get("fix_empty", True):
        print(f"{DIM}generate.fix_empty is false — skipped.{RESET}")
        return 0

    for domain in domains(cfg):
        for gen_type in types(cfg):
            extra = [
                generate_flag(domain, gen_type),
                # Same directory the generate stage wrote to; the flag above
                # still picks the prompt, so a refilled file gets the prompt its
                # neighbours got.
                "--out_name", output_dir_name(cfg, gen_type),
                *model_args(cfg, gen_type),
            ]
            if args.debug:
                extra.append("--debug")
            if args.yes:
                extra.append("--yes")

            print(f"{BOLD}  {domain}/{gen_type}{RESET}")
            code = run_cmd(
                fork_cmd(GHOSTBUSTER_PROJECT, FIX_EMPTY_SCRIPT, extra),
                args.dry_run,
                cwd=GHOSTBUSTER_PROJECT,
                env=submodule_env(cuda=False),
                # Belt and braces behind --yes: the fork's prompt is a bare
                # input(), so an unattended run without the flag would block on
                # an inherited stdin or raise EOFError on a closed one.
                stdin_text="y\n" if args.yes else None,
            )
            if code != 0:
                print(f"{RED}fix-empty failed for {domain}/{gen_type} "
                      f"(exit {code}).{RESET}")
                return code
    return 0


def audit(cfg: dict, args: argparse.Namespace) -> int:
    """Count documents and blank files per (domain, type); 0 if the tree is sound.

    This is the only check that generation finished. ``generate.py`` catches a
    failed API call, records it and carries on, exiting 0 with the output file
    simply absent — so a missing document shows up nowhere except in a count.
    The baseline is the fork's own loop bound for the domain (see
    :func:`expected_documents`), which is what a complete run writes.
    """
    gen_limit = args.gen_limit if args.gen_limit is not None else cfg["generate"].get("limit")

    print(f"  {'domain/directory':<28} {'files':>7} {'expected':>9} {'missing':>8} {'blank':>7}")
    shortfall: list[str] = []
    blanks: list[str] = []
    # Reported but never fatal, unlike the generated blanks. A blank human
    # document is a property of the upstream corpus, not of this run — the
    # Ghostbuster essay release ships six of them — so there is nothing to fix
    # here and no reason to block on it. fix_empty_generations.py cannot help
    # either: it regenerates from a prompt, and a human document has none.
    seed_blanks: list[str] = []

    for domain in domains(cfg):
        human, human_blank = count_txt(GB_DATA_DIR / domain / "human")
        expected = expected_documents(cfg, domain, gen_limit)
        # Printed for context, not as the baseline: the human tree can hold more
        # files than there are prompts to generate from.
        print(f"  {domain + '/human':<28} {human:>7} {'—':>9} {'—':>8} {human_blank:>7}")
        if human_blank:
            seed_blanks.append(f"{domain}/human ({human_blank})")

        # Counted under this run's own directories, not under the type names:
        # another model's rows may be sitting in data/<domain>/<its dir>/, and
        # counting those would report a run as complete that never happened.
        for gen_type in types(cfg):
            out_name = output_dir_name(cfg, gen_type)
            files, blank = count_txt(GB_DATA_DIR / domain / out_name)
            missing = max(expected - files, 0)
            colour = RED if (missing or blank) else ""
            reset = RESET if colour else ""
            print(f"  {colour}{domain + '/' + out_name:<28} {files:>7} {expected:>9} "
                  f"{missing:>8} {blank:>7}{reset}")
            if missing:
                shortfall.append(f"{domain}/{out_name} (-{missing})")
            if blank:
                blanks.append(f"{domain}/{out_name} ({blank})")

    ok = True
    if blanks and not args.allow_empty:
        print(
            f"\n{RED}Blank generated documents: {', '.join(blanks)}.{RESET}\n"
            "  A blank file is a failed API call that left an empty file behind. "
            "Refill them, then collect again:\n"
            "    uv run python epai/ghostbuster/run_full_pipeline.py "
            "--only fix-empty\n"
            "  Or pass --allow-empty to carry them into the corpus, where they "
            "will hold empty text and a Null Binoculars score."
        )
        ok = False
    if seed_blanks:
        print(
            f"\n{YELLOW}Blank human documents: {', '.join(seed_blanks)}.{RESET}\n"
            "  Not an error: these are seeded inputs, and the Ghostbuster essay "
            "release ships six empty files. Nothing here can refill them.\n"
            "  What it means downstream: each one is a row with empty text, so it "
            "gets a Null Binoculars score and a Null embedding, and its generated "
            "partner was written against a word count of zero. Drop them at "
            "training time, or with combine's --drop-incomplete once the "
            "embeddings are in."
        )
    if shortfall and not args.allow_shortfall:
        print(
            f"\n{RED}Fewer documents than the run should have produced: "
            f"{', '.join(shortfall)}.{RESET}\n"
            "  generate.py reports a failed call and carries on, so this is what "
            "an incomplete run looks like. Re-run it — it skips what already "
            "exists, so only the gaps cost anything:\n"
            "    uv run python epai/ghostbuster/run_full_pipeline.py "
            "--only generate\n"
            "  Or pass --allow-shortfall to build an unpaired corpus."
        )
        ok = False

    # Blank files that survive an --allow-empty are worth naming again here:
    # they are rows, and any pairing check downstream will count them.
    # A limited run is a plumbing test, not a small corpus, and the parquet it
    # produces says so: collect copies the whole human tree, because the human
    # documents are the seeded input and nothing here knows which of them the
    # limit happened to cover.
    if ok and gen_limit is not None:
        print(
            f"\n{YELLOW}NOTE: generating {gen_limit} item(s) per domain, but the "
            f"whole human tree is collected. The corpus will hold every human "
            f"document against {gen_limit} generated one(s) per domain — fine "
            f"for testing the pipeline, not a balanced corpus.{RESET}"
        )

    if blanks and args.allow_empty:
        print(f"\n{YELLOW}--allow-empty: keeping {', '.join(blanks)}.{RESET}")
    if shortfall and args.allow_shortfall:
        print(f"{YELLOW}--allow-shortfall: accepting {', '.join(shortfall)}.{RESET}")

    return 0 if ok else 1


def collect_subdirs(cfg: dict, domain: str) -> list[str]:
    """What collect copies out of ``data/<domain>/``, and nothing else.

    The seeded inputs plus this run's own generator directories. Selective
    rather than "copy the domain", because the fork's tree is shared by every
    model variant: once a second variant has been generated there,
    ``data/essay/`` holds ``gpt56luna_0701A`` *and* ``gpt57x_0701A``, and
    copying the lot would put both models in one corpus under one filename.
    """
    return [*SEED_SUBDIRS[domain], *output_dir_names(cfg)]


def clean_plan(cfg: dict, var: str, steps: list[str]) -> list[tuple[str, Path, str]]:
    """``(level, path, detail)`` for everything ``--clean`` would remove.

    One rule at every level: **inside the places this config writes, remove what
    this config would not produce.** That is what a leftover is — the seeded
    inputs and this run's own directories are what it is defined against, so
    neither can ever be selected, and nothing outside these two trees is looked
    at at all. Another variant's collected tree and another variant's parquets
    belong to another corpus and are not this run's business.

    The three levels, in the order a run touches them:

      1. ``external/ghostbuster/data/<domain>/`` — generator directories that
         are neither seeded nor this run's. **This is generated text that cost
         API calls**, usually another model's, and it is the one level worth
         reading the list before confirming.
      2. ``data/ghostbuster-data-<variant>/`` — the same, plus whole domains
         this config no longer builds, plus the fork's ``logprobs*`` caches if
         an older collect ever copied them (this one does not).
      3. ``data/parquet/<domain>_<variant>.parquet`` — only when the run will
         rebuild them, i.e. when ``prepare`` is among the selected steps.
         Otherwise they are somebody's input, not a leftover.
    """
    plan: list[tuple[str, Path, str]] = []

    def documents(path: Path) -> str:
        files, _ = count_txt(path)
        return f"{files} document(s)"

    for domain in domains(cfg):
        root = GB_DATA_DIR / domain
        expected = set(collect_subdirs(cfg, domain))
        if root.is_dir():
            for path in sorted(root.iterdir()):
                if path.is_dir() and path.name not in expected:
                    plan.append(("fork tree", path, documents(path)))

    dest_root = collected_dir(cfg, var)
    if dest_root.is_dir():
        configured = set(domains(cfg))
        for path in sorted(dest_root.iterdir()):
            if path.is_dir() and path.name not in configured:
                plan.append(("collected", path,
                             f"domain not in this config, {documents(path)}"))
        for domain in domains(cfg):
            root = dest_root / domain
            expected = set(collect_subdirs(cfg, domain))
            if not root.is_dir():
                continue
            for path in sorted(root.iterdir()):
                if not path.is_dir():
                    continue
                if path.name not in expected:
                    plan.append(("collected", path, documents(path)))
                    continue
                # Kept directory, but the fork's token log-probability caches
                # inside it are not part of any corpus and are never copied by
                # this pipeline — so any that are here are from an older one.
                for cache in sorted(p for p in path.rglob("logprobs*") if p.is_dir()):
                    plan.append(("collected", cache, "logprob cache, unused here"))

    if "prepare" in steps:
        for domain in domains(cfg):
            path = domain_parquet(cfg, var, domain)
            if path.is_file():
                plan.append(("parquet", path,
                             f"{path.stat().st_size / 1e6:.1f} MB, rebuilt by prepare"))

    return plan


def step_clean(cfg: dict, args: argparse.Namespace, var: str, steps: list[str]) -> int:
    """Remove what :func:`clean_plan` found, after showing it. 0 unless aborted."""
    banner("clean", "leftovers from previous runs")
    plan = clean_plan(cfg, var, steps)

    if not plan:
        print(f"{GREEN}Nothing to remove.{RESET}\n")
        return 0

    for level, path, detail in plan:
        print(f"  {level:<10} {rel(path):<52} {DIM}{detail}{RESET}")

    if any(level == "fork tree" for level, _, _ in plan):
        print(f"\n{YELLOW}  The 'fork tree' entries are generated text — another "
              f"run paid for them, and nothing else keeps a copy.{RESET}")

    if args.dry_run:
        print()
        for _, path, _ in plan:
            print(f"{DIM}$ rm -rf {rel(path)}{RESET}")
        print()
        return 0

    if not confirm(f"\n{YELLOW}Remove these {len(plan)} path(s)?{RESET}", args.yes):
        print("Aborted.")
        return 1

    roots = (GB_DATA_DIR.resolve(), collected_dir(cfg, var).resolve(), PARQUET_DIR.resolve())
    for _, path, _ in plan:
        # Nothing is deleted from outside the three directories this pipeline
        # writes to, whatever a config says.
        if not any(r in path.resolve().parents for r in roots):
            print(f"{RED}Refusing to remove {rel(path)}: outside this "
                  f"pipeline's directories.{RESET}")
            return 1
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    print(f"{GREEN}Removed{RESET} {len(plan)} path(s).\n")
    return 0


def step_collect(cfg: dict, args: argparse.Namespace, var: str) -> int:
    """Audit the fork's text tree, then copy this run's part of it into the repo."""
    dest_root = collected_dir(cfg, var)
    banner("collect", f"{rel(GB_DATA_DIR)} → {rel(dest_root)}")

    if args.dry_run:
        for domain in domains(cfg):
            for sub in collect_subdirs(cfg, domain):
                print(f"{DIM}$ cp -a {rel(GB_DATA_DIR / domain / sub)} "
                      f"{rel(dest_root / domain / sub)}{RESET}")
        return 0

    code = audit(cfg, args)
    if code != 0:
        return code

    print()
    for domain in domains(cfg):
        dest = dest_root / domain
        if dest.exists() and not confirm(
            f"{YELLOW}{rel(dest)} already exists and will be updated in place. "
            f"Continue?{RESET}",
            args.yes,
        ):
            print("Aborted.")
            return 1

        for sub in collect_subdirs(cfg, domain):
            src = GB_DATA_DIR / domain / sub
            if not src.is_dir():
                print(f"{RED}Not found: {rel(src)}{RESET}\n"
                      f"  The generate stage writes it. Run that first, or drop "
                      f"{domain!r} from generate.domains.")
                return 1

            (dest / sub).mkdir(parents=True, exist_ok=True)
            # dirs_exist_ok so a second collect refreshes the tree rather than
            # failing; the fork's tree is authoritative, and files removed there
            # are left behind here deliberately — nothing should silently delete
            # a document from a corpus directory.
            shutil.copytree(src, dest / sub, dirs_exist_ok=True, ignore=COPY_IGNORE)

        files, _ = count_txt(dest)
        print(f"{GREEN}Copied{RESET} {files} document(s) to {rel(dest)}"
              f" ({', '.join(collect_subdirs(cfg, domain))})")

        # Nothing here deletes, so a directory from an earlier run into the same
        # variant stays behind. It costs only disk — the prepare stage reads the
        # directories this config names and no others — but it is worth saying
        # out loud, because it is text that looks like part of this corpus and
        # is not in it.
        expected = set(collect_subdirs(cfg, domain))
        leftover = sorted(
            p.name for p in dest.iterdir() if p.is_dir() and p.name not in expected
        )
        if leftover:
            print(f"{YELLOW}  NOTE: {rel(dest)} also holds {', '.join(leftover)}, "
                  f"from an earlier run.{RESET}\n"
                  f"    Not included in the corpus — the prepare stage reads "
                  f"human/ and {', '.join(output_dir_names(cfg))} only. Remove "
                  f"with:  rm -rf {' '.join(rel(dest / n) for n in leftover)}")

    return 0


def step_prepare(cfg: dict, args: argparse.Namespace, var: str) -> int:
    """One prepare_*.py per domain: text tree → parquet."""
    banner("prepare", ", ".join(PREPARE_SCRIPTS[d] for d in domains(cfg)))
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    for domain in domains(cfg):
        source = domain_source_dir(cfg, var, domain)
        if not args.dry_run and not source.is_dir():
            print(f"{RED}Not found: {rel(source)}{RESET}\n"
                  f"  The collect stage writes it. Run that first, or pass "
                  f"--variant if the tree was collected under a different one.")
            return 1

        cmd = [
            sys.executable,
            str(BASE_DIR / PREPARE_SCRIPTS[domain]),
            "--source-dir", rel(source),
            "-o", rel(domain_parquet(cfg, var, domain)),
            # Named, not discovered. A collected tree is added to rather than
            # replaced — collect copies into it without deleting — so a
            # generator directory from an earlier run survives there and would
            # otherwise contribute a second set of machine rows under its own
            # old label. This makes the config decide what the corpus holds.
            "--generators", "human", *output_dir_names(cfg),
        ]
        if not cfg["prepare"].get("embeddings", False):
            cmd.append("--no-embedding")

        print(f"{BOLD}  {domain}{RESET}")
        code = run_cmd(cmd, args.dry_run)
        if code != 0:
            print(f"{RED}prepare failed for domain {domain!r} (exit {code}).{RESET}")
            return code
    return 0


def remove_domain_parquets(inputs: list[Path], out: Path, dry_run: bool) -> None:
    """Delete the per-domain parquets the combined corpus was built from.

    They are an intermediate, not an artefact: every row of every one of them is
    in ``out``, with a ``domain`` column saying where it came from, and nothing
    downstream — score, embed, train, analyse — reads them again. Keeping them
    doubles the corpus on disk and leaves three files that look like corpora,
    which is how a run ends up trained on ``essay_gpt56luna.parquet`` from two
    generations ago.

    Rebuilding one costs a prepare run over the collected text tree: local,
    minutes, no API calls. That is what makes this safe to do automatically —
    and why it happens only after combine has exited 0 and written ``out``.
    """
    if dry_run:
        for path in inputs:
            print(f"{DIM}$ rm {rel(path)}{RESET}")
        return

    removed = []
    for path in inputs:
        # Guard against deleting the thing that was just written, however the
        # names are configured.
        if path.resolve() == out.resolve() or not path.is_file():
            continue
        path.unlink()
        removed.append(path)
    if removed:
        print(f"{DIM}Removed {len(removed)} per-domain parquet(s): "
              f"{', '.join(rel(p) for p in removed)}{RESET}")


def step_combine(cfg: dict, args: argparse.Namespace, var: str) -> int:
    """Union the per-domain parquets into one corpus, then drop them."""
    comb = cfg["combine"]
    out = combined_path(cfg, var)
    inputs = [domain_parquet(cfg, var, d) for d in domains(cfg)]

    banner("combine", COMBINE_SCRIPT)

    cmd = [
        sys.executable,
        str(BASE_DIR / COMBINE_SCRIPT),
        *(rel(p) for p in inputs),
        "-o", rel(out),
        "--drop-incomplete", comb.get("drop_incomplete", "emb"),
        "--max-drop-frac", str(comb.get("max_drop_frac", 0.05)),
        # Always passed, with no values when the list is empty: the script's own
        # default drops the prompt-variant labels, which this corpus does not
        # have, and it reports every label that matched nothing. The labels
        # here are the corpus ones (gpt56luna:0701A), not the fork's types.
        "--drop-generated-by", *comb.get("drop_generated_by", []),
    ]

    if not args.dry_run:
        absent = [p for p in inputs if not p.is_file()]
        if absent:
            print(f"{RED}Not found: {', '.join(rel(p) for p in absent)}{RESET}\n"
                  f"  The prepare stage writes them. Run that first.")
            return 1
        if out.exists():
            if not confirm(
                f"{YELLOW}{rel(out)} already exists. Overwrite?{RESET}", args.yes
            ):
                print("Aborted.")
                return 1
            cmd.append("--overwrite")

    code = run_cmd(cmd, args.dry_run)
    if code != 0:
        return code

    if comb.get("remove_domain_parquets", True):
        # Only ever after a successful combine, and only once the file it wrote
        # is on disk: the inputs are the one copy of this text in parquet form
        # until that is true.
        if args.dry_run or out.is_file():
            remove_domain_parquets(inputs, out, args.dry_run)
        else:
            print(f"{YELLOW}combine reported success but {rel(out)} is not "
                  f"there; keeping the per-domain parquets.{RESET}")
    return 0


def step_score(cfg: dict, args: argparse.Namespace, var: str) -> int:
    """Score the combined corpus with the Binoculars fork."""
    score = cfg["score"]
    src = combined_path(cfg, var)
    out = scored_path(cfg, var)

    gpu = args.gpu if args.gpu is not None else score.get("gpu")
    cpu_threads = (
        args.cpu_threads if args.cpu_threads is not None else score.get("cpu_threads")
    )
    batch_size = (
        args.batch_size if args.batch_size is not None else score.get("batch_size", 1)
    )
    limit = args.score_limit if args.score_limit is not None else score.get("limit")

    extra = [
        rel(src),
        # Explicit, because score_parquets.py writes IN PLACE by default. The
        # combined corpus is worth keeping unscored: it is what gets re-scored
        # when the detector or its model pair changes.
        "-o", rel(out),
        "--text-column", score.get("text_column", "text"),
        "--batch-size", str(batch_size),
    ]
    if gpu is not None:
        extra += ["--gpu", str(gpu)]
    if cpu_threads is not None:
        extra += ["--cpu-threads", str(cpu_threads)]
    if score.get("observer"):
        extra += ["--observer", score["observer"]]
    if score.get("performer"):
        extra += ["--performer", score["performer"]]
    if limit is not None:
        extra += ["--limit", str(limit)]
    if args.yes:
        extra.append("--yes")

    banner("score", f"{rel(BINOCULARS_SCRIPT)}")

    if not args.dry_run:
        if not src.is_file():
            print(f"{RED}Not found: {rel(src)}{RESET}\n"
                  f"  Nothing to score. Run the combine stage first.")
            return 1
        if out.exists() and not confirm(
            f"{YELLOW}{rel(out)} already exists and will be overwritten. "
            f"Continue?{RESET}",
            args.yes,
        ):
            print("Aborted.")
            return 1

    return run_cmd(
        fork_cmd(BINOCULARS_PROJECT, rel(BINOCULARS_SCRIPT), extra),
        args.dry_run,
        env=submodule_env(),
    )


def step_embed(cfg: dict, args: argparse.Namespace, var: str) -> int:
    """Fill every ``emb_*`` column on the scored corpus.

    In place, which is add_embeddings.py's own default and how the existing
    ``*_binox0.parquet`` files were built: an embedding is derived from the text
    alone, so a separate file would be the same corpus twice.

    Runs in *this* environment, unlike the two forks — the embedders live in
    ``epai/utils/embedding_utils.py``.

    Which models run comes from ``embed.embedders``; unset means the script's own
    ``EMBEDDERS`` default, which is all thirteen including two billed OpenAI
    endpoints.
    """
    emb = cfg["embed"]
    target = scored_path(cfg, var)

    gpu = args.gpu if args.gpu is not None else emb.get("gpu")
    cpu_threads = (
        args.cpu_threads if args.cpu_threads is not None else emb.get("cpu_threads")
    )

    cmd = [
        sys.executable,
        str(BASE_DIR / EMBED_SCRIPT),
        rel(target),
        "--text-col", emb.get("text_col", "text"),
    ]
    if emb.get("embedders"):
        cmd += ["--embedders", *emb["embedders"]]
    if emb.get("batch_size") is not None:
        cmd += ["--batch-size", str(emb["batch_size"])]
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    if cpu_threads is not None:
        cmd += ["--cpu-threads", str(cpu_threads)]
    limit = args.embed_limit if args.embed_limit is not None else emb.get("limit")
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if emb.get("recompute", False):
        cmd.append("--recompute")
    if args.yes:
        cmd.append("--yes")

    banner("embed", EMBED_SCRIPT)

    if not args.dry_run and not target.is_file():
        print(f"{RED}Not found: {rel(target)}{RESET}\n"
              f"  Nothing to embed. Run the score stage first.")
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
            raise SystemExit(
                f"unknown step(s) {unknown}. Choose from: {list(STEP_KEYS)}"
            )
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
                   help=f"pipeline config YAML (default: {rel(DEFAULT_CFG_PATH)})")
    g.add_argument("--variant", default=None,
                   help="override the filename token. Defaults to the generator "
                        "model's slug, e.g. gpt56luna.")

    g = parser.add_argument_group("which steps")
    g.add_argument("--from", dest="start_from", metavar="STEP",
                   help=f"start here instead of at the beginning: {', '.join(STEP_KEYS)}")
    g.add_argument("--until", dest="stop_after", metavar="STEP",
                   help="stop after this step, inclusive. --until combine builds "
                        "the corpus and leaves the GPU stages (score, embed) for "
                        "a later --from score.")
    g.add_argument("--only", nargs="+", metavar="STEP",
                   help="run only these steps (always in pipeline order). Cannot "
                        "be combined with --from or --until.")
    g.add_argument("--dry-run", action="store_true",
                   help="print the commands without running anything")
    g.add_argument("--clean", action="store_true",
                   help="before the first step, remove what previous runs left "
                        "in the places this config writes: foreign generator "
                        "directories in the fork's tree and in the collected "
                        "tree, domains this config no longer builds, stray "
                        "logprob caches, and (when prepare runs) the per-domain "
                        "parquets. Never touches the seeded inputs or this "
                        "run's own directories. Lists everything and asks first; "
                        "--dry-run shows the list and deletes nothing.")

    g = parser.add_argument_group("generation")
    g.add_argument("--yes", action="store_true",
                   help="unattended run: answers yes to this script's overwrite "
                        "prompts and to the fork's refill prompt")
    g.add_argument("--gen-limit", type=int, default=None, metavar="N",
                   help="override generate.limit — items per domain. Note the "
                        "fork's --limit caps reuter's (author, index) list, so a "
                        "small value gives one author's articles, not a sample.")
    g.add_argument("--workers", type=int, default=None, metavar="N",
                   help="concurrent API calls in the generate stage. Overrides "
                        "generate.max_workers (default 8). 1 is the old "
                        "strictly-sequential behaviour; lower it if the account's "
                        "rate limit is being hit.")
    g.add_argument("--debug", action="store_true",
                   help="pass --debug to the fork: every LLM call is mocked, no "
                        "keys needed. For testing the plumbing.")
    g.add_argument("--allow-empty", action="store_true",
                   help="carry blank generated documents into the corpus instead "
                        "of stopping at the collect stage")
    g.add_argument("--allow-shortfall", action="store_true",
                   help="accept fewer generated documents than human originals "
                        "instead of stopping at the collect stage")

    g = parser.add_argument_group("gpu stages (score, embed)")
    g.add_argument("--gpu", default=None,
                   help="CUDA_VISIBLE_DEVICES for both GPU stages, e.g. '0' or "
                        "'0,1'. Overrides score.gpu and embed.gpu.")
    g.add_argument("--cpu-threads", type=int, default=None,
                   help="cap CPU threads used for inference, in both GPU stages")
    g.add_argument("--batch-size", type=int, default=None,
                   help="texts scored per batch (default: the config's). The "
                        "embed stage keeps its own per-model batch sizes; set "
                        "embed.batch_size to override those.")
    g.add_argument("--score-limit", type=int, default=None, metavar="N",
                   help="score only the first N rows, for timing a run. "
                        "Overrides score.limit.")
    g.add_argument("--embed-limit", type=int, default=None, metavar="N",
                   help="embed only the first N rows, for timing a run. "
                        "Overrides embed.limit.")

    args = parser.parse_args()

    # Validated here rather than inside each stage: the stages are other
    # people's scripts, and this config is the only thing that knows the whole
    # chain agrees with itself.
    cfg = load_pipeline_config(args.cfg)
    var = args.variant or variant(cfg)

    steps = select(args)
    preflight(cfg, args, steps)

    combined = combined_path(cfg, var)
    print(f"{BOLD}ghostbuster → binoculars{RESET}")
    print(f"  config       {args.cfg}")
    print(f"  dataset_id   {cfg['dataset_id']}")
    print(f"  variant      {var}")
    print(f"  domains      {', '.join(domains(cfg))}")
    print(f"  types        {', '.join(types(cfg))}")
    # The label is what a row will actually be called, so it is worth reading
    # before a run rather than discovering in the parquet afterwards.
    print(f"  generated_by {', '.join(generated_by_label(cfg, t) for t in types(cfg))}")
    print(f"  generated    {rel(GB_DATA_DIR)}/<domain>/"
          f"{{{','.join(output_dir_names(cfg))}}}/")
    print(f"  collected    {rel(collected_dir(cfg, var))}/")
    removing = cfg["combine"].get("remove_domain_parquets", True)
    print(f"  per-domain   "
          f"{', '.join(rel(domain_parquet(cfg, var, d)) for d in domains(cfg))}"
          f"{DIM + '  (removed after combine)' + RESET if removing else ''}")
    print(f"  combined     {rel(combined)}")
    print(f"  scored       {rel(scored_path(cfg, var))}")
    print(f"  steps        {' → '.join(steps)}")
    if args.dry_run:
        print(f"\n{YELLOW}--dry-run: nothing will be executed.{RESET}")
    elif not args.yes:
        print(f"\n{YELLOW}NOTE: --yes not given, so this run will stop to ask "
              f"before overwriting anything.{RESET}")

    runners = {
        "generate": lambda: step_generate(cfg, args),
        "fix-empty": lambda: step_fix_empty(cfg, args),
        "collect": lambda: step_collect(cfg, args, var),
        "prepare": lambda: step_prepare(cfg, args, var),
        "combine": lambda: step_combine(cfg, args, var),
        "score": lambda: step_score(cfg, args, var),
        "embed": lambda: step_embed(cfg, args, var),
    }

    # Before the first step, not as one of them: it is about the state a run
    # starts from, so it has to happen whichever step that run begins at.
    if args.clean and step_clean(cfg, args, var, steps) != 0:
        sys.exit(1)

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
            f"    uv run python epai/ghostbuster/run_full_pipeline.py --from {failed}"
        )
        sys.exit(1)

    if args.dry_run:
        print(f"\n{YELLOW}Dry run — nothing was executed.{RESET}")
        return

    print(f"\n{GREEN}Done.{RESET}")
    if "embed" in ran:
        print(f"  Corpus, scored and embedded: {rel(scored_path(cfg, var))}")
        print(f"  It is ready for epai/ai_detection/train/ and analyse/.")
    elif "score" in ran:
        print(f"  Scored corpus: {rel(scored_path(cfg, var))}")
    elif "combine" in ran:
        print(f"  Combined corpus: {rel(combined)}")

    # The steps that were selected away, and how to pick them up. Only worth
    # saying when the run stopped short on purpose — a completed pipeline has
    # nothing left to resume.
    remaining = STEP_KEYS[STEP_KEYS.index(steps[-1]) + 1:] if steps else ()
    if remaining:
        print(f"  Remaining: {' → '.join(remaining)}\n"
              f"  Resume with: uv run python epai/ghostbuster/run_full_pipeline.py "
              f"--from {remaining[0]}")


if __name__ == "__main__":
    main()

r"""Score ONE corpus with several Binoculars model pairs, one pair after another.

    uv run python -m epai.ai_detection.score.run_binox_pairs \
        data/parquet/science_v3_gpt56luna_0811A.parquet --gpu 0 --cpu-threads 16

Each pair writes its own column -- ``binoculars_score_qwen25_7b``,
``binoculars_score_falcon3_7b``, ... -- alongside the Falcon-7B
``binoculars_score`` that is already there, so afterwards one parquet holds every
detector side by side and the classifier configs and the human-vs-AI analysis can
read them all. Nothing is overwritten: a new pair means a new column, which is
why no ``--recompute`` appears anywhere below.

WHY THIS IS A LOOP AND NOT A PARALLEL RUN
-----------------------------------------
Adding a column is a read-modify-write of the whole Parquet: ``score_parquets.py``
reads the frame at startup and writes it back when the file finishes. Two pairs
scoring the SAME file at once would both start from the version without either
column, and whichever finished last would silently drop the other's -- with a
zero exit status and a summary claiming success. So the pairs are strictly
sequential *per file*.

Different files are independent, so the way to use the GPUs is one process per
corpus, each pinned to its own card:

    uv run python -m epai.ai_detection.score.run_binox_pairs \
        data/parquet/science_v3_gpt56luna_0811A.parquet --gpu 0 --cpu-threads 16 &
    uv run python -m epai.ai_detection.score.run_binox_pairs \
        data/parquet/ghostbuster_gpt56luna.parquet --gpu 1 --cpu-threads 16 &

WHAT IT DOES PER PAIR
---------------------
1. Runs ``external/Binoculars/score_parquets.py --pair <key>`` as a subprocess,
   with ``--yes``: this script asks once, up front, for the whole plan, and an
   hours-long run must not then block on a prompt nobody is watching.
2. Checks the column it was supposed to fill actually got filled, on the file
   itself. The exit code alone is not enough -- score_parquets.py deliberately
   survives a per-file failure so one bad file cannot lose a multi-file run, so
   "the process exited 0" and "the scores are there" are different claims.
3. Times it, and projects the remaining pairs from the rate the finished ones
   actually managed.

A pair that fails does not stop the others (``--stop-on-error`` if you want it
to): the failures are usually pair-specific -- a gated repo, a tokenizer
mismatch, an out-of-memory at this batch size -- and the pairs that do work are
worth having. The run still exits non-zero if anything failed.

Re-running is safe. A pair whose column is already complete makes
score_parquets.py report "nothing to do" and exit 0, so an interrupted run is
resumed by issuing the same command again.

OPTIONS WORTH KNOWING
---------------------
``--fp32-metrics`` (on by default) computes the two perplexity terms in fp32
while the models still run in bfloat16. Binoculars' cross-perplexity is a
soft-target cross-entropy summed over the whole vocabulary; the replacement
pairs have 128k-152k-wide vocabularies against Falcon's 65k, so that sum runs
over four times as many bf16 terms, and bf16 carries only ~3 decimal digits. It
costs one transient copy of the logits and removes numerical noise as an
explanation for anything the comparison turns up. Turn it off with
``--no-fp32-metrics`` if memory is tight -- that is the first thing to drop
before lowering ``--batch-size``.

``--batch-size`` defaults to 1, which fits any pair on any card.
``external/Binoculars/check_pairs.py --gpu N`` prints the largest batch size that
fits the card actually present; pass that here.

``--limit N`` scores only the first N rows -- a smoke test. Point it at a
throwaway copy of the corpus, never at the real one: a half-filled column drops
every unscored row from the human-vs-AI comparison later, because a row missing
ANY selected score column is skipped there.

Run ``check_pairs.py`` and ``download_models.py`` first. This script does not
download anything itself; a pair whose weights are missing spends the first half
hour of its slot fetching them instead of scoring.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from epai.utils import notify_utils
from epai.utils.parquet_utils import check_column_filled, column_fill

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

INTERRUPTED = 130

BASE_DIR = Path(__file__).resolve().parents[3]
BINOCULARS_PROJECT = BASE_DIR / "external" / "Binoculars"
BINOCULARS_SCRIPT = BINOCULARS_PROJECT / "score_parquets.py"
PAIRS_MODULE = BINOCULARS_PROJECT / "model_pairs.py"

# The three pairs the comparison is for: the strongest modern proxy, the same
# family as the paper's pair one generation on (so the only thing that changed is
# the age of the scoring models), and a third family as a check that the result
# is not a quirk of one tokenizer or data mix.
DEFAULT_PAIRS = ("qwen25-7b", "falcon3-7b", "llama31-8b")


@dataclass
class Result:
    """What one pair's slot produced."""

    key: str
    column: str
    code: int  # subprocess exit status
    seconds: float
    n_scored: int  # rows holding a score afterwards
    n_rows: int
    verified: bool  # the column is present and not all-Null
    message: str  # the postcondition check's own wording

    @property
    def ok(self) -> bool:
        return self.code == 0 and self.verified


def load_registry():
    """``external/Binoculars/model_pairs.py``, imported by path.

    That module is deliberately free of torch -- it is the pair registry and the
    memory arithmetic, nothing else -- so it imports cleanly into *this* repo's
    venv even though the scoring itself only runs in the submodule's. Reading it
    here is what lets the plan below name each pair's score column, and the
    postcondition check look for it, without either side hardcoding a second copy
    of the naming rule.
    """
    if not PAIRS_MODULE.is_file():
        raise SystemExit(
            f"{RED}Not found: {PAIRS_MODULE.relative_to(BASE_DIR)}{RESET}\n"
            f"  The Binoculars submodule is not checked out. Run:\n"
            f"    git submodule update --init --recursive"
        )
    spec = importlib.util.spec_from_file_location("binox_model_pairs", PAIRS_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def submodule_env() -> dict[str, str]:
    """The environment for the fork's subprocess.

    ``VIRTUAL_ENV`` and ``UV_PROJECT_ENVIRONMENT`` are set by the outer ``uv
    run`` and point at *this* repo's venv; leaving either in place makes ``uv run
    --project external/Binoculars`` either warn and ignore it or, worse, honour
    it and import the wrong transformers. Same reasoning as the two pipeline
    runners, which strip them for the same call.
    """
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    return env


def score_cmd(args: argparse.Namespace, pair) -> list[str]:
    """The ``score_parquets.py`` command line for one pair.

    ``--yes`` is always passed: this script has already asked about the whole
    plan, and a multi-hour unattended run that stops on a prompt at pair two is
    the failure mode the single up-front confirmation exists to prevent. No
    ``--recompute`` either -- a new pair writes a new column, so every row is
    missing and the default fills them all.
    """
    cmd = [
        "uv", "run", "--project", str(BINOCULARS_PROJECT.relative_to(BASE_DIR)),
        "python", str(BINOCULARS_SCRIPT.relative_to(BASE_DIR)),
        corpus_arg(args.parquet),
        "--pair", pair.key,
        "--text-column", args.text_column,
        "--batch-size", str(args.batch_size),
        "--yes",
    ]
    if args.gpu is not None:
        cmd += ["--gpu", args.gpu]
    if args.cpu_threads is not None:
        cmd += ["--cpu-threads", str(args.cpu_threads)]
    if args.max_tokens is not None:
        cmd += ["--max-tokens", str(args.max_tokens)]
    if args.fp32_metrics:
        cmd.append("--fp32-metrics")
    if args.allow_token_mismatch:
        cmd.append("--allow-token-mismatch")
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    return cmd


def corpus_arg(path: Path) -> str:
    """The corpus path as the subprocess should see it.

    The subprocess runs with ``cwd=BASE_DIR``, so a path relative to wherever
    this script was invoked from would resolve somewhere else -- or nowhere.
    Repo-relative when it can be (it reads better in the echoed command line),
    absolute when the file lives outside the repo.
    """
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def banner(index: int, total: int, pair) -> None:
    print(f"\n{BOLD}{'━' * 78}{RESET}")
    print(f"{BOLD}  pair {index}/{total}  {pair.key}{RESET}"
          f"{DIM}   {pair.observer} + {pair.performer}  →  {pair.column}{RESET}")
    print(f"{BOLD}{'━' * 78}{RESET}")


def human_time(seconds: float) -> str:
    """``1h 12m`` / ``6m 30s`` / ``42s``."""
    seconds = int(round(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


def print_plan(args: argparse.Namespace, pairs: list) -> None:
    """Everything this run will do, before it does any of it."""
    print()
    print("=" * 78)
    print(f"{BOLD}Binoculars pair comparison{RESET}   {args.parquet}")
    print("=" * 78)
    print(f"  gpu           : {args.gpu if args.gpu is not None else 'all visible (unset)'}")
    print(f"  cpu threads   : {args.cpu_threads if args.cpu_threads is not None else 'unset'}")
    print(f"  batch size    : {args.batch_size}")
    print(f"  metrics dtype : {'fp32' if args.fp32_metrics else 'bf16 (model dtype)'}")
    print(f"  text column   : {args.text_column!r}")
    if args.max_tokens is not None:
        print(f"  max tokens    : {args.max_tokens}")
    if args.limit is not None:
        print(f"  {YELLOW}limit         : {args.limit} row(s) -- SMOKE TEST, leaves the "
              f"column part-filled{RESET}")
    print()
    print(f"{'pair':<14}{'score column':<34}{'present':>9}{'filled':>9}{'rows':>8}")
    print("-" * 78)
    for pair in pairs:
        present, filled, rows = column_fill(args.parquet, pair.column)
        print(f"{pair.key:<14}{pair.column:<34}{str(present):>9}{filled:>9}{rows:>8}")
    print("-" * 78)
    print(f"  {len(pairs)} pair(s), one after another, writing IN PLACE into "
          f"{args.parquet.name}.")
    print(f"  {DIM}Every pair adds a column; none of them touches another's. Do not run a "
          f"second\n  process against this same file -- the last writer would drop the "
          f"other's column.{RESET}")


def confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    # Pushed before blocking, not after: an unattended run that reaches a prompt
    # does nothing at all until someone answers, and not discovering that hours
    # later is the whole point of the notifications.
    notify_utils.active().blocked(question)
    try:
        return input(f"{question} [y/N]: ").strip().lower() in ("y", "yes")
    except EOFError:
        print(f"\n{YELLOW}No input available; assuming no. Pass --yes to skip this "
              f"prompt.{RESET}")
        return False


def run_pair(args: argparse.Namespace, pair, index: int, total: int) -> Result:
    """One pair's slot: score, verify, time."""
    banner(index, total, pair)
    cmd = score_cmd(args, pair)
    print(f"{DIM}$ {' '.join(cmd)}{RESET}\n", flush=True)

    started = time.perf_counter()
    if args.dry_run:
        code = 0
    else:
        # No capture: the progress bars and the findings table render as they do
        # when the fork is run by hand.
        code = subprocess.run(cmd, cwd=BASE_DIR, env=submodule_env()).returncode
    seconds = time.perf_counter() - started

    if args.dry_run:
        return Result(pair.key, pair.column, 0, 0.0, 0, 0, True, "dry run")

    verified, message = check_column_filled(args.parquet, pair.column, args.limit)
    _, filled, rows = column_fill(args.parquet, pair.column)
    print(message)
    return Result(pair.key, pair.column, code, seconds, filled, rows, verified, message)


def print_summary(results: list[Result], total_seconds: float) -> None:
    print()
    print("=" * 78)
    print(f"{BOLD}Summary{RESET}   {human_time(total_seconds)} total")
    print("=" * 78)
    print(f"{'pair':<14}{'score column':<34}{'scored':>9}{'time':>10}  result")
    print("-" * 78)
    for res in results:
        mark = f"{GREEN}ok{RESET}" if res.ok else f"{RED}FAILED{RESET}"
        detail = "" if res.ok else f" (exit {res.code}, {'verified' if res.verified else 'column not filled'})"
        rate = f"{res.n_scored / res.seconds:.1f}/s" if res.seconds and res.n_scored else ""
        print(f"{res.key:<14}{res.column:<34}{res.n_scored:>9}{human_time(res.seconds):>10}"
              f"  {mark}{detail}  {DIM}{rate}{RESET}")
    print("-" * 78)

    failed = [r for r in results if not r.ok]
    if failed:
        print(f"{RED}{len(failed)} pair(s) failed:{RESET} "
              f"{', '.join(r.key for r in failed)}")
        print(f"  {DIM}The columns that did land are complete and usable; re-run this "
              f"command to retry\n  only the missing ones -- a finished column makes "
              f"score_parquets.py exit 0 with nothing to do.{RESET}")
    else:
        print(f"{GREEN}Every pair scored and verified.{RESET}")
        print(f"  {DIM}Next: one analysis run per column, e.g.\n"
              f"    uv run python -m epai.ai_detection.analyse.analyse_score_human_vs_ai \\\n"
              f"        <parquet>:<generator> --score-cols {results[0].column}{RESET}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("parquet", type=Path,
                    help="The ONE corpus to score. Run a second process for a second "
                         "corpus; do not point two processes at this one.")
    ap.add_argument("--gpu", default=None,
                    help="value for CUDA_VISIBLE_DEVICES in the scoring subprocess, "
                         "e.g. '0'. Default: leave unset, so all visible GPUs are used")
    ap.add_argument("--cpu-threads", type=int, default=None,
                    help="cap on the subprocess's CPU pools (OMP/MKL), so corpora "
                         "scored side by side do not fight over threads")
    ap.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS),
                    metavar="KEY",
                    help="Pair keys from external/Binoculars/model_pairs.py "
                         "(default: %(default)s)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="rows per batch (default: %(default)s, which fits any pair on "
                         "any card). check_pairs.py suggests the largest that fits")
    ap.add_argument("--fp32-metrics", action=argparse.BooleanOptionalAction, default=True,
                    help="compute the perplexity terms in fp32 while the models stay in "
                         "bf16 (default: on). --no-fp32-metrics is the first thing to "
                         "drop when memory is tight")
    ap.add_argument("--allow-token-mismatch", action="store_true",
                    help="pass through to score_parquets.py, for a pair check_pairs.py "
                         "reports as 'extended' (added control tokens only)")
    ap.add_argument("--text-column", default="text",
                    help="column holding the text to score (default: %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="truncation cap in tokens (default: the detector's own, 512)")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: score only the first N rows. Use a throwaway copy "
                         "of the corpus -- a part-filled column shrinks every later "
                         "comparison that reads it")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="abort at the first failing pair instead of carrying on")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the commands, run nothing")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    args.parquet = args.parquet.expanduser().resolve()
    if not args.parquet.is_file():
        raise SystemExit(f"{RED}Not found: {args.parquet}{RESET}")
    if not BINOCULARS_SCRIPT.is_file():
        raise SystemExit(
            f"{RED}Not found: {BINOCULARS_SCRIPT.relative_to(BASE_DIR)}{RESET}\n"
            f"  Run: git submodule update --init --recursive"
        )

    registry = load_registry()
    pairs = [registry.resolve(key) for key in args.pairs]

    print_plan(args, pairs)
    if not confirm(f"Score {args.parquet.name} with {len(pairs)} pair(s)?",
                   args.yes or args.dry_run):
        raise SystemExit("Aborted — no changes made.")

    # Silent unless a topic is configured; see epai/utils/notify_utils.py.
    notify = notify_utils.configure("binox-pairs", args.parquet.stem,
                                    [p.key for p in pairs])
    notify.started(f"{len(pairs)} pairs on {args.parquet.name}")

    results: list[Result] = []
    started = time.perf_counter()
    for i, pair in enumerate(pairs, 1):
        try:
            res = run_pair(args, pair, i, len(pairs))
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Interrupted during {pair.key}. Columns already written "
                  f"are intact; re-run to continue.{RESET}")
            notify.failed(pair.key, INTERRUPTED, time.perf_counter() - started)
            return INTERRUPTED

        results.append(res)
        notify.stage(pair.key, 0 if res.ok else (res.code or 1), res.seconds,
                     critical=args.stop_on_error)

        # Project the rest from what the finished pairs actually managed, rather
        # than from an estimate: the pairs are within a factor of two of each
        # other in size, and the first one has just measured this corpus, this
        # card and this batch size.
        done = [r for r in results if r.ok and r.seconds]
        remaining = len(pairs) - i
        if done and remaining:
            mean = sum(r.seconds for r in done) / len(done)
            print(f"{DIM}  {remaining} pair(s) left, ~{human_time(mean * remaining)} "
                  f"at this rate.{RESET}")

        if not res.ok and args.stop_on_error:
            print(f"{RED}Stopping: {pair.key} failed and --stop-on-error is set.{RESET}")
            break

    total = time.perf_counter() - started
    print_summary(results, total)

    failed = [r for r in results if not r.ok]
    if failed:
        notify.failed(failed[0].key, failed[0].code or 1, total)
        return 1
    notify.finished(f"{len(results)} pairs on {args.parquet.name}", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())

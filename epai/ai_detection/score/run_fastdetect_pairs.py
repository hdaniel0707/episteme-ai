r"""Score ONE corpus with several Fast-DetectGPT model pairs, one after another.

    uv run python epai/ai_detection/score/run_fastdetect_pairs.py \
        data/parquet/ghostbuster_gpt56luna.parquet --cpu-threads 16 --gpu 0

    uv run python epai/ai_detection/score/run_fastdetect_pairs.py \
        data/parquet/science_v3_gpt56luna_0811A.parquet --cpu-threads 16 --gpu 1

The sibling of ``run_binox_pairs.py``, and deliberately the same shape: one
process per corpus, pairs strictly sequential within it, a postcondition check
on the file itself after each pair, and a projection of the remaining time from
the rate the finished pairs actually managed.

WHY THIS DETECTOR IS WORTH A COLUMN NEXT TO BINOCULARS
------------------------------------------------------
It is a different statistic over the *same* models, not another instance of the
same one. Binoculars is a ratio of two perplexities; Fast-DetectGPT is the
conditional probability curvature -- the scoring model's log-likelihood of the
text standardised against the mean and variance it would expect under the
sampling model's own distribution. So the default pair set here mirrors the
Binoculars registry exactly:

    falcon-7b     -> fastdetect_crit_falcon_7b    beside binoculars_score
    qwen25-7b     -> fastdetect_crit_qwen25_7b    beside binoculars_score_qwen25_7b
    falcon3-7b    -> fastdetect_crit_falcon3_7b   beside binoculars_score_falcon3_7b
    mistral-v03   -> fastdetect_crit_mistral_v03  beside binoculars_score_mistral_v03

Every one of those pairs is already in the Hub cache from the Binoculars runs,
so the default set downloads nothing. Each column is then a paired observation:
same corpus, same two models, same 512-token cap, one statistic swapped. If
Fast-DetectGPT transfers across domains where Binoculars does not (or the other
way round), that is a fact about the statistic, which is the only way to get
such a fact cheaply.

``llama3-8b`` is NOT in the default set. It is the pair the authors currently
recommend and one of only two the paper calibrated, but it is Llama-3, not
Llama-3.1 -- the Llama-3.1 weights cached for ``binoculars_score_llama31_8b`` do
not satisfy it -- so selecting it means a fresh ~32GB gated download. Ask for it
explicitly when you want it:

    --pairs falcon-7b llama3-8b

MIND THE SIGN
-------------
Fast-DetectGPT reads HIGH for machine text. ``binoculars_score`` reads LOW. Any
config or plot that consumes both has to carry the direction per column; the
`score_baselines:` entries in cfg/ai_detection/train/ take a threshold, so add
the direction there when you wire these in.

THE PROBABILITY COLUMNS ARE MOSTLY NULL, ON PURPOSE
---------------------------------------------------
Upstream's criterion-to-probability mapping is a pair of fitted Normals, and it
published constants for four pairs only. Of this script's defaults, one
(falcon-7b) has them. The rest write ``fastdetect_prob_*`` as Null rather than
borrow another pair's constants. Even where the mapping exists it is calibrated
on the paper's dev set and generators, and it is non-monotone in the left tail
-- score_parquets.py counts the rows that land there and says so. Treat the
criterion as the number and the probability as a convenience.

WHY THIS IS A LOOP AND NOT A PARALLEL RUN
-----------------------------------------
Identical to the Binoculars case: adding a column is a read-modify-write of the
whole Parquet. Two pairs scoring the SAME file at once would both start from the
version without either column, and whichever finished last would silently drop
the other's -- with a zero exit status. So the pairs are sequential per file,
and different corpora get one process each, pinned to their own card.

Re-running is safe. A pair whose column is already complete makes
score_parquets.py report "nothing to do" and exit 0, so an interrupted run is
resumed by issuing the same command again.
"""

from __future__ import annotations

import argparse
import ast
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
FDG_PROJECT = BASE_DIR / "external" / "fast-detect-gpt"
FDG_SCRIPT = FDG_PROJECT / "score_parquets.py"
PAIRS_MODULE = FDG_PROJECT / "detector_pairs.py"
UPSTREAM_CRITERION = FDG_PROJECT / "scripts" / "fast_detect_gpt.py"

CRITERION_FN = "get_sampling_discrepancy_analytic"

# The four pairs that mirror the Binoculars registry, so every existing
# binoculars_score* column gets a Fast-DetectGPT twin over the same two models.
# All four are already in the Hub cache from those runs.
DEFAULT_PAIRS = ("falcon-7b", "qwen25-7b", "falcon3-7b", "mistral-v03")


@dataclass
class Result:
    """What one pair's slot produced."""

    key: str
    column: str
    code: int
    seconds: float
    n_scored: int
    n_rows: int
    verified: bool
    message: str

    @property
    def ok(self) -> bool:
        return self.code == 0 and self.verified


def load_registry():
    """``external/fast-detect-gpt/detector_pairs.py``, imported by path.

    Free of torch on purpose -- it is the registry, the memory arithmetic and
    the calibration constants, nothing else -- so it imports cleanly into THIS
    repo's venv even though the scoring only runs in the submodule's. Reading it
    here is what lets the plan name each pair's columns, and the postcondition
    look for them, without a second copy of the naming rule on this side.
    """
    if not PAIRS_MODULE.is_file():
        raise SystemExit(
            f"{RED}Not found: {PAIRS_MODULE.relative_to(BASE_DIR)}{RESET}\n"
            f"  The fast-detect-gpt submodule is not checked out. Run:\n"
            f"    git submodule update --init --recursive"
        )
    name = "fdg_detector_pairs"
    spec = importlib.util.spec_from_file_location(name, PAIRS_MODULE)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec_module, not after. @dataclass resolves string
    # annotations (anything under `from __future__ import annotations`) through
    # sys.modules[cls.__module__]; a module loaded by path and left unregistered
    # makes that lookup return None, and the dataclass decorator dies with an
    # AttributeError inside dataclasses.py that names nothing in this repo.
    # detector_pairs.py avoids the future import for the same reason, but the
    # registration is what makes the loader safe for whatever it is pointed at.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _extract_function(path: Path, name: str) -> str | None:
    """The source of one top-level function, or None if it is not there."""
    try:
        source = path.read_text()
    except OSError:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1:node.end_lineno])
    return None


def check_vendored_criterion() -> str | None:
    """Has the copied criterion drifted from upstream's? Returns a warning or None.

    ``score_parquets.py`` carries a verbatim copy of upstream's
    ``get_sampling_discrepancy_analytic`` rather than importing it, because the
    module it lives in pulls datasets, sklearn and matplotlib in at import time
    for the sake of twenty lines of pure torch. A copy is the right trade there
    and the wrong one to leave unchecked: if a later ``git pull`` in the
    submodule changes the upstream function, the copy keeps computing the old
    statistic and nothing says so.

    Cheap enough (two ast parses of files already on disk) to run on every
    invocation, and only ever a warning -- a drifted copy still computes *a*
    well-defined statistic, so stopping a multi-hour run over it would be worse
    than saying so loudly.
    """
    ours = _extract_function(FDG_SCRIPT, CRITERION_FN)
    theirs = _extract_function(UPSTREAM_CRITERION, CRITERION_FN)
    if ours is None:
        return f"{CRITERION_FN} not found in {FDG_SCRIPT.name} — cannot verify it."
    if theirs is None:
        return (f"{CRITERION_FN} not found in "
                f"{UPSTREAM_CRITERION.relative_to(BASE_DIR)} — cannot verify the "
                f"vendored copy against upstream.")
    if ours != theirs:
        return (f"The vendored {CRITERION_FN} in {FDG_SCRIPT.name} NO LONGER MATCHES "
                f"upstream's.\n  Upstream changed, or the copy was edited. Diff them "
                f"before trusting these numbers:\n"
                f"    diff <(sed -n '/def {CRITERION_FN}/,/^$/p' "
                f"{UPSTREAM_CRITERION.relative_to(BASE_DIR)}) \\\n"
                f"         <(sed -n '/def {CRITERION_FN}/,/^$/p' "
                f"{FDG_SCRIPT.relative_to(BASE_DIR)})")
    return None


def submodule_env() -> dict[str, str]:
    """The environment for the fork's subprocess.

    ``VIRTUAL_ENV`` and ``UV_PROJECT_ENVIRONMENT`` are set by the outer ``uv
    run`` and point at THIS repo's venv; leaving either in place makes
    ``uv run --project external/fast-detect-gpt`` either warn and ignore it or,
    worse, honour it and import the wrong transformers.
    """
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    return env


def corpus_arg(path: Path) -> str:
    """The corpus path as the subprocess should see it (cwd is BASE_DIR)."""
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def score_cmd(args: argparse.Namespace, pair) -> list[str]:
    """The ``score_parquets.py`` command line for one pair.

    ``--yes`` always: this script asks once, up front, about the whole plan, and
    an unattended multi-hour run that stops on a prompt at pair three is exactly
    what the single confirmation exists to prevent. No ``--recompute`` either --
    a new pair writes a new column, so every row is missing and the default
    fills them all.
    """
    cmd = [
        "uv", "run", "--project", str(FDG_PROJECT.relative_to(BASE_DIR)),
        "python", str(FDG_SCRIPT.relative_to(BASE_DIR)),
        corpus_arg(args.parquet),
        "--pair", pair.key,
        "--text-column", args.text_column,
        "--dtype", args.dtype,
        "--yes",
    ]
    if args.gpu is not None:
        cmd += ["--gpu", args.gpu]
    if args.cpu_threads is not None:
        cmd += ["--cpu-threads", str(args.cpu_threads)]
    if args.max_tokens is not None:
        cmd += ["--max-tokens", str(args.max_tokens)]
    cmd.append("--fp32-metrics" if args.fp32_metrics else "--no-fp32-metrics")
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    return cmd


def banner(index: int, total: int, pair) -> None:
    calib = "calibrated" if pair.calib is not None else f"{DIM}criterion only{RESET}"
    print(f"\n{BOLD}{'━' * 78}{RESET}")
    print(f"{BOLD}  pair {index}/{total}  {pair.key}{RESET}"
          f"{DIM}   {pair.sampling} + {pair.scoring}{RESET}")
    print(f"{DIM}  → {pair.crit_column}{RESET}  ({calib})")
    print(f"{BOLD}{'━' * 78}{RESET}")


def human_time(seconds: float) -> str:
    """``1h 12m`` / ``6m 30s`` / ``42s``."""
    seconds = int(round(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


def print_plan(args: argparse.Namespace, pairs: list, drift: str | None) -> None:
    """Everything this run will do, before it does any of it."""
    print()
    print("=" * 78)
    print(f"{BOLD}Fast-DetectGPT pair comparison{RESET}   {args.parquet}")
    print("=" * 78)
    print(f"  gpu           : {args.gpu if args.gpu is not None else 'all visible (unset)'}")
    print(f"  cpu threads   : {args.cpu_threads if args.cpu_threads is not None else 'unset'}")
    print(f"  model dtype   : {args.dtype}")
    print(f"  metrics dtype : {'fp32' if args.fp32_metrics else 'model dtype'}")
    print(f"  text column   : {args.text_column!r}")
    print(f"  max tokens    : {args.max_tokens if args.max_tokens is not None else '512 (detector default)'}")
    print(f"  direction     : {YELLOW}HIGH criterion = machine{RESET} "
          f"{DIM}(binoculars_score is the other way round){RESET}")
    if args.limit is not None:
        print(f"  {YELLOW}limit         : {args.limit} row(s) -- SMOKE TEST, leaves the "
              f"columns part-filled{RESET}")
    if drift:
        print()
        print(f"  {RED}⚠️  {drift}{RESET}")
    print()
    print(f"{'pair':<14}{'criterion column':<32}{'prob':>6}{'present':>9}{'filled':>9}{'rows':>7}")
    print("-" * 78)
    for pair in pairs:
        present, filled, rows = column_fill(args.parquet, pair.crit_column)
        calib = "yes" if pair.calib is not None else f"-"
        print(f"{pair.key:<14}{pair.crit_column:<32}{calib:>6}"
              f"{str(present):>9}{filled:>9}{rows:>7}")
    print("-" * 78)
    gated = [p.key for p in pairs if p.gated]
    if gated:
        print(f"  {YELLOW}gated on the Hub (needs HF_TOKEN + accepted licence): "
              f"{', '.join(gated)}{RESET}")
    uncal = [p.key for p in pairs if p.calib is None]
    if uncal:
        print(f"  {DIM}criterion only, no probability column: {', '.join(uncal)}{RESET}")
    print(f"  {len(pairs)} pair(s), one after another, writing IN PLACE into "
          f"{args.parquet.name}.")
    print(f"  {DIM}Every pair adds its own columns; none touches another's. Do not run a "
          f"second\n  process against this same file -- the last writer would drop the "
          f"other's columns.{RESET}")


def confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    # Pushed before blocking, not after: an unattended run that reaches a prompt
    # does nothing at all until someone answers.
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
        return Result(pair.key, pair.crit_column, 0, 0.0, 0, 0, True, "dry run")
    # No capture: the progress bar and findings table render as they do by hand.
    code = subprocess.run(cmd, cwd=BASE_DIR, env=submodule_env()).returncode
    seconds = time.perf_counter() - started

    # The exit code says the subprocess ran, not that it did the work:
    # score_parquets.py survives a per-file failure on purpose. Check the file.
    verified, message = check_column_filled(args.parquet, pair.crit_column, args.limit)
    _, filled, rows = column_fill(args.parquet, pair.crit_column)
    print(message)
    return Result(pair.key, pair.crit_column, code, seconds, filled, rows, verified, message)


def print_summary(results: list[Result], total_seconds: float) -> None:
    print()
    print("=" * 78)
    print(f"{BOLD}Summary{RESET}   {human_time(total_seconds)} total")
    print("=" * 78)
    print(f"{'pair':<14}{'criterion column':<32}{'scored':>9}{'time':>10}  result")
    print("-" * 78)
    for res in results:
        mark = f"{GREEN}ok{RESET}" if res.ok else f"{RED}FAILED{RESET}"
        detail = ("" if res.ok else
                  f" (exit {res.code}, {'verified' if res.verified else 'column not filled'})")
        rate = f"{res.n_scored / res.seconds:.1f}/s" if res.seconds and res.n_scored else ""
        print(f"{res.key:<14}{res.column:<32}{res.n_scored:>9}"
              f"{human_time(res.seconds):>10}  {mark}{detail}  {DIM}{rate}{RESET}")
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
        print(f"  {DIM}Read the per-file summary JSON before using the numbers: it "
              f"reports how many\n  rows hit the token cap and how many fall in the "
              f"non-monotone region of the\n  probability mapping.{RESET}")


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
                         "e.g. '0'. Default: leave unset")
    ap.add_argument("--cpu-threads", type=int, default=None,
                    help="cap on the subprocess's CPU pools (OMP/MKL), so corpora "
                         "scored side by side do not fight over threads")
    ap.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS), metavar="KEY",
                    help="Pair keys from external/fast-detect-gpt/detector_pairs.py "
                         "(default: %(default)s -- the four that mirror the Binoculars "
                         "columns and need no new downloads)")
    ap.add_argument("--dtype", default="float16",
                    choices=("float16", "bfloat16", "float32"),
                    help="model dtype (default: %(default)s, upstream's choice for "
                         "these models and what its calibration was measured under)")
    ap.add_argument("--fp32-metrics", action=argparse.BooleanOptionalAction, default=True,
                    help="compute the criterion in fp32 while the models stay in "
                         "--dtype (default: on)")
    ap.add_argument("--text-column", default="text",
                    help="column holding the text to score (default: %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="truncation cap in tokens (default: the detector's own, 512, "
                         "which matches the Binoculars runs)")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: score only the first N rows. Use a throwaway copy "
                         "of the corpus -- a part-filled column shrinks every later "
                         "comparison that reads it")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="abort at the first failing pair instead of carrying on")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the commands, run nothing")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    args.parquet = args.parquet.expanduser().resolve()
    if not args.parquet.is_file():
        raise SystemExit(f"{RED}Not found: {args.parquet}{RESET}")
    if not FDG_SCRIPT.is_file():
        raise SystemExit(
            f"{RED}Not found: {FDG_SCRIPT.relative_to(BASE_DIR)}{RESET}\n"
            f"  Run: git submodule update --init --recursive"
        )

    registry = load_registry()
    pairs = [registry.resolve(key) for key in args.pairs]
    drift = check_vendored_criterion()

    print_plan(args, pairs, drift)
    if not confirm(f"Score {args.parquet.name} with {len(pairs)} pair(s)?",
                   args.yes or args.dry_run):
        raise SystemExit("Aborted — no changes made.")

    # Silent unless a topic is configured; see epai/utils/notify_utils.py.
    notify = notify_utils.configure("fastdetect-pairs", args.parquet.stem,
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

        # Project the rest from what the finished pairs actually managed: the
        # pairs are within a factor of two of each other in size, and the first
        # has just measured this corpus on this card.
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

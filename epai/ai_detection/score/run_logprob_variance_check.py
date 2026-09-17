r"""Run the whole logprob-variance check in one call: capture each model
twice into isolated workspaces, then diff the two captures.

Chains, in order, per model in ``--models``:

    capture_judge_logprobs.py --model <m> --only <stem> --limit <n> \
        --workspace data/parquet/tmp/variance_a_<m> --yes
    capture_judge_logprobs.py --model <m> --only <stem> --limit <n> \
        --workspace data/parquet/tmp/variance_b_<m> --yes
    check_logprob_variance.py --a .../variance_a_<m> --b .../variance_b_<m>

Same rows both times: ``capture_judge_logprobs.py``'s ``--limit`` picks rows
by a deterministic round-robin over file position
(``add_llm_judge_scores.apply_limit``), not at random, so replicate A and B
always judge the identical ``uid`` set.

Two billed API calls per model (gpt56luna via OpenAI, gemma4-31b via
GenAI4Science) -- this asks for confirmation ONCE, up front, covering every
sub-call, rather than once per capture (each capture is invoked with its own
``--yes`` after that single confirmation).

Run:
    .venv/bin/python epai/ai_detection/score/run_logprob_variance_check.py --dry-run
    .venv/bin/python epai/ai_detection/score/run_logprob_variance_check.py --yes
    .venv/bin/python epai/ai_detection/score/run_logprob_variance_check.py \
        --only manugram_sci --limit 300 --models gpt56luna --yes

Each capture writes its own isolated workspace under
``data/parquet/tmp/variance_{a,b}_<model>/`` -- this never touches the real
parquets, ``logjudge_params.csv``, or any prior capture. Re-running resumes a
partial capture step (``capture_judge_logprobs.py``'s own default behaviour
for its workspace), but the checker step always re-reads and re-diffs
everything currently in both workspaces.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
SCORE_DIR = Path(__file__).resolve().parent
PARQUET_DIR = BASE_DIR / "data" / "parquet"

MODELS = ("gpt56luna", "gemma4-31b")
DEFAULT_STEM = "ghostbuster_orig"
DEFAULT_LIMIT = 150


def workspace(model: str, replicate: str) -> Path:
    return PARQUET_DIR / "tmp" / f"variance_{replicate}_{model}"


def run(cmd: list[str], dry_run: bool) -> None:
    print(f"\n$ {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=BASE_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", default=DEFAULT_STEM, metavar="STEM",
                     help=f"which core split to sample from (default {DEFAULT_STEM})")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                     help=f"rows to capture per replicate (default {DEFAULT_LIMIT})")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--dry-run", action="store_true",
                     help="print every command without running any of them")
    ap.add_argument("--yes", action="store_true", help="skip the billed-call prompt")
    args = ap.parse_args()

    if not args.dry_run and not args.yes:
        print(f"NOTE  billed API calls -- {len(args.models)} model(s) x 2 replicates x "
              f"{args.limit} rows x up to 2 modes (words, scale) each.")
        if input("Proceed? type 'yes': ").strip().lower() != "yes":
            raise SystemExit("aborted -- no calls made")

    py = sys.executable
    capture_script = str(SCORE_DIR / "capture_judge_logprobs.py")
    checker_script = str(SCORE_DIR / "check_logprob_variance.py")

    for model in args.models:
        for replicate in ("a", "b"):
            ws = workspace(model, replicate)
            run([py, capture_script, "--model", model, "--only", args.only,
                 "--limit", str(args.limit), "--workspace", str(ws), "--yes"],
                args.dry_run)

    for model in args.models:
        ws_a, ws_b = workspace(model, "a"), workspace(model, "b")
        run([py, checker_script, "--a", str(ws_a), "--b", str(ws_b)], args.dry_run)

    print("\n[dry run] -- nothing executed" if args.dry_run else f"\n{'=' * 74}\ndone")


if __name__ == "__main__":
    main()

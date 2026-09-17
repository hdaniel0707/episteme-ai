r"""Run the prompt-diversity pilot's parsed-mode sanity check and words-mode
logprob capture -- steps 2 and 3 of the pilot -- as ONE script instead of 8
separate CLI invocations.

For each (model, prompt) pair, this is exactly:

    uv run python epai/ai_detection/score/add_llm_judge_scores.py \
        data/parquet/<stem>.parquet --model <model> --prompt <prompt> --yes
    uv run python epai/ai_detection/score/capture_judge_logprobs.py \
        --model <model> --prompt <prompt> --modes words --only <stem> \
        --workspace data/parquet/tmp/<model>-<prompt> --yes

run by setting ``sys.argv`` and calling each script's own ``main()`` in turn --
nothing about how a row is judged or captured is duplicated here, only the
per-combination CLI-argument plumbing. Every capture gets its own
``--workspace`` (see the "running more than one --prompt into the same
workspace" note in ``capture_judge_logprobs.py``) so the 4 sidecars never
collide.

One confirmation for the whole batch (skip with ``--yes``): both add up to
``2 prompts x 2 models x 2 steps = 8`` billed/rate-limited calls, but nobody
wants 8 separate "type yes" prompts. A ``SystemExit`` from any one combination
(a blocked judge, an aborted sub-confirmation, an already-fully-scored file)
is caught and reported without stopping the rest of the batch.

    uv run python epai/ai_detection/score/run_prompt_pilot_capture.py --prompts cand03 cand10 --dry-run
    uv run python epai/ai_detection/score/run_prompt_pilot_capture.py --prompts cand03 cand10
    uv run python epai/ai_detection/score/run_prompt_pilot_capture.py --prompts cand03 cand10 --yes

Next, once this is done: merge_captured_logprobs.py, then score_logprob_windows.py,
then ensemble_judge_scores.py --combine mean (see the pilot plan).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from epai.ai_detection.score import add_llm_judge_scores, capture_judge_logprobs

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
CAPTURE_DIR = PARQUET_DIR / "tmp"

DEFAULT_MODELS = ["gpt56luna", "gemma4-31b"]
DEFAULT_STEM = "manugram_aiscientist"


def _run(label: str, module, argv: list[str], dry_run: bool) -> None:
    print(f"\n{'-' * 74}\n{label}")
    print("  " + " ".join(argv))
    if dry_run:
        return
    old_argv = sys.argv
    sys.argv = [module.__file__, *argv]
    try:
        module.main()
    except SystemExit as exc:
        print(f"  [stopped] {exc}")
    finally:
        sys.argv = old_argv


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prompts", nargs=2, required=True, metavar=("A", "B"),
                     help="the 2 candidate (or any) prompt ids to run, e.g. cand03 cand10")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                     help=f"judge models to run (default: {DEFAULT_MODELS})")
    ap.add_argument("--stem", default=DEFAULT_STEM,
                     help=f"benchmark parquet stem (default: {DEFAULT_STEM})")
    ap.add_argument("--dry-run", action="store_true",
                     help="print the 8 equivalent invocations, run nothing")
    ap.add_argument("--yes", action="store_true",
                     help="skip the one whole-batch confirmation")
    args = ap.parse_args()

    unknown = [p for p in args.prompts if p not in add_llm_judge_scores.PROMPTS]
    if unknown:
        raise SystemExit(f"unknown prompt id(s) {unknown}; choose from "
                          f"{sorted(add_llm_judge_scores.PROMPTS)}")

    parquet = PARQUET_DIR / f"{args.stem}.parquet"
    print(f"stem: {args.stem}  ({parquet})")
    print(f"models: {args.models}")
    print(f"prompts: {args.prompts}")
    print(f"\nthis will run {len(args.models) * len(args.prompts)} parsed-mode "
          f"scoring call(s) and {len(args.models) * len(args.prompts)} "
          "words-mode capture call(s), each --yes'd through its own billed-call prompt.")

    if not args.dry_run and not args.yes:
        if input("Proceed? type 'yes': ").strip().lower() != "yes":
            raise SystemExit("aborted -- no calls made")

    print(f"\n{'=' * 74}\nSTEP 2 -- parsed-mode sanity check")
    for prompt in args.prompts:
        for model in args.models:
            _run(f"parsed  model={model}  prompt={prompt}", add_llm_judge_scores, [
                str(parquet), "--model", model, "--prompt", prompt, "--yes",
            ], args.dry_run)

    print(f"\n{'=' * 74}\nSTEP 3 -- words-mode logprob capture")
    for prompt in args.prompts:
        for model in args.models:
            workspace = CAPTURE_DIR / f"{model}-{prompt}"
            _run(f"capture  model={model}  prompt={prompt}", capture_judge_logprobs, [
                "--model", model, "--prompt", prompt, "--modes", "words",
                "--only", args.stem, "--workspace", str(workspace), "--yes",
            ], args.dry_run)

    print(f"\n{'=' * 74}\ndone" + (" (dry run)" if args.dry_run else ""))
    if not args.dry_run:
        workspaces = " ".join(str(CAPTURE_DIR / f"{m}-{p}")
                               for p in args.prompts for m in args.models)
        print(f"\nnext: uv run python epai/ai_detection/score/merge_captured_logprobs.py "
              f"--only {args.stem} --workspaces {workspaces}")


if __name__ == "__main__":
    main()

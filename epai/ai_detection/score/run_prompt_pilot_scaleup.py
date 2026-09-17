r"""Scale the prompt-diversity LogJudge pilot (validated on MG-AI-Sci, see
``run_prompt_pilot_capture.py`` and the pilot's bar charts) up to the 3
remaining core splits -- ``gpt56luna`` only, ``cand03`` + ``cand10`` only.

One script, run end-to-end on the GPU server, covering the whole remaining
pipeline (steps 3-6 of the pilot plan, at full scale):

  1. (optional, ``--include-parsed``, off by default) parsed-mode sanity
     scoring for each (stem, prompt) -- doubles the API calls for a check
     already done at pilot scale; skip it to keep this to what the wlp
     mean-of-3 ("the new algorithm") actually needs.
  2. words-mode logprob capture (``capture_judge_logprobs.py``) for
     ``cand03``/``cand10``, one workspace per prompt, all 3 stems in one call
     each (workspaces already keep sidecars per-stem internally).
  3. merge the 2 workspaces' sidecars into the real parquets
     (``merge_captured_logprobs.py``).
  4. compute ord2/``wlp`` for the new prompts (``score_logprob_windows.py`` --
     already prompt-id-generic, so this also leaves ``v3``'s columns alone).
  5. the wlp mean-of-3 ensemble (``ensemble_judge_scores.py --combine mean``),
     writing ``llm_judge_score_gpt56luna_3p_promptmean`` -- the SAME column
     name the MG-AI-Sci pilot already has, so all 4 splits line up under one
     name once this is done. With ``--include-parsed``, also the parsed
     3-way mean, ``llm_judge_score_gpt56luna_v3_c03_c10_parsed_promptmean``.

Internally this just sets ``sys.argv`` and calls each script's own ``main()``
in turn (the same trick ``run_prompt_pilot_capture.py`` uses for steps 2+3) --
nothing about how a row is judged, captured, scored or combined is duplicated
here, only the per-stage CLI-argument plumbing. A ``SystemExit`` from any one
stage is caught and reported without stopping the rest of the batch.

Row counts as of this write-up (``ghostbuster_orig`` 5988, ``manugram_gb``
14960, ``manugram_sci`` 15000 -- ~35,950 rows): with ``--include-parsed`` off
(the default) this is ``2 prompts x ~35,950 rows x 1 mode (words, 1 token)``
call(s) to gpt-5.6-luna, plus one merge/score/ensemble pass. Turning on
``--include-parsed`` roughly doubles the OpenAI calls. One confirmation gates
the whole batch (``--yes`` to skip it, e.g. for an unattended/nohup run) --
already-scored rows are skipped on any re-run, so this resumes safely if
interrupted.

    uv run python epai/ai_detection/score/run_prompt_pilot_scaleup.py --dry-run
    uv run python epai/ai_detection/score/run_prompt_pilot_scaleup.py
    uv run python epai/ai_detection/score/run_prompt_pilot_scaleup.py --yes --workers 8
    uv run python epai/ai_detection/score/run_prompt_pilot_scaleup.py --include-parsed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from epai.ai_detection.score import (
    add_llm_judge_scores,
    capture_judge_logprobs,
    ensemble_judge_scores,
    merge_captured_logprobs,
    score_logprob_windows,
)
from epai.ai_detection.score.run_prompt_pilot_capture import _run

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
CAPTURE_DIR = PARQUET_DIR / "tmp"

DEFAULT_STEMS = ["ghostbuster_orig", "manugram_gb", "manugram_sci"]
DEFAULT_MODELS = ["gpt56luna"]
DEFAULT_PROMPTS = ["cand03", "cand10"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stems", nargs="+", default=DEFAULT_STEMS,
                     help=f"benchmark parquet stems (default: {DEFAULT_STEMS})")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                     help=f"judge models (default: {DEFAULT_MODELS})")
    ap.add_argument("--prompts", nargs=2, default=DEFAULT_PROMPTS, metavar=("A", "B"),
                     help=f"the 2 prompt ids to run (default: {DEFAULT_PROMPTS})")
    ap.add_argument("--include-parsed", action="store_true",
                     help="also run the parsed-mode sanity check + parsed "
                          "3-way mean at full scale (roughly doubles the "
                          "OpenAI calls; off by default)")
    ap.add_argument("--workers", type=int, default=4,
                     help="capture_judge_logprobs.py --workers (default: 4)")
    ap.add_argument("--dry-run", action="store_true",
                     help="print every equivalent invocation, run nothing")
    ap.add_argument("--yes", action="store_true",
                     help="skip the one whole-batch confirmation")
    args = ap.parse_args()

    unknown = [p for p in args.prompts if p not in add_llm_judge_scores.PROMPTS]
    if unknown:
        raise SystemExit(f"unknown prompt id(s) {unknown}; choose from "
                          f"{sorted(add_llm_judge_scores.PROMPTS)}")

    print(f"stems: {args.stems}")
    print(f"models: {args.models}")
    print(f"prompts: {args.prompts}")
    print(f"include parsed-mode pass: {args.include_parsed}")
    n_parsed = len(args.stems) * len(args.models) * len(args.prompts) if args.include_parsed else 0
    n_capture = len(args.models) * len(args.prompts)
    print(f"\nthis will run {n_parsed} parsed-mode scoring call(s) (one per "
          f"stem x model x prompt) and {n_capture} words-mode capture call(s) "
          "(one per model x prompt, each covering all --stems), followed by "
          "one merge, one score, and one ensemble pass per model.")

    if not args.dry_run and not args.yes:
        if input("Proceed? type 'yes': ").strip().lower() != "yes":
            raise SystemExit("aborted -- no calls made")

    if args.include_parsed:
        print(f"\n{'=' * 74}\nSTEP -- parsed-mode sanity check (full scale)")
        for stem in args.stems:
            parquet = PARQUET_DIR / f"{stem}.parquet"
            for prompt in args.prompts:
                for model in args.models:
                    _run(f"parsed  stem={stem}  model={model}  prompt={prompt}",
                         add_llm_judge_scores, [
                             str(parquet), "--model", model, "--prompt", prompt, "--yes",
                         ], args.dry_run)

    print(f"\n{'=' * 74}\nSTEP -- words-mode logprob capture")
    workspaces: dict[tuple[str, str], Path] = {}
    for prompt in args.prompts:
        for model in args.models:
            workspace = CAPTURE_DIR / f"{model}-{prompt}"
            workspaces[(model, prompt)] = workspace
            _run(f"capture  model={model}  prompt={prompt}  stems={args.stems}",
                 capture_judge_logprobs, [
                     "--model", model, "--prompt", prompt, "--modes", "words",
                     "--only", *args.stems, "--workspace", str(workspace),
                     "--workers", str(args.workers), "--yes",
                 ], args.dry_run)

    print(f"\n{'=' * 74}\nSTEP -- merge sidecars into the real parquets")
    _run(f"merge  stems={args.stems}", merge_captured_logprobs, [
        "--only", *args.stems, "--workspaces", *[str(w) for w in workspaces.values()],
    ], args.dry_run)

    print(f"\n{'=' * 74}\nSTEP -- score ord2 (wlp) for the new prompts")
    _run(f"score  stems={args.stems}", score_logprob_windows, [
        "--only", *args.stems,
    ], args.dry_run)

    print(f"\n{'=' * 74}\nSTEP -- mean-of-3 ensemble(s)")
    for model in args.models:
        a, b = args.prompts
        _run(f"ensemble wlp  model={model}  stems={args.stems}", ensemble_judge_scores, [
            "--combine", "mean",
            "--members", f"llm_judge_score_{model}_v3_wlp",
                         f"llm_judge_score_{model}_{a}_wlp",
                         f"llm_judge_score_{model}_{b}_wlp",
            "--name", f"{model}_3p", "--only", *args.stems,
        ], args.dry_run)
        if args.include_parsed:
            _run(f"ensemble parsed  model={model}  stems={args.stems}", ensemble_judge_scores, [
                "--combine", "mean",
                "--members", f"llm_judge_score_{model}_v3",
                             f"llm_judge_score_{model}_{a}",
                             f"llm_judge_score_{model}_{b}",
                "--name", f"{model}_v3_{a}_{b}_parsed", "--only", *args.stems,
            ], args.dry_run)

    print(f"\n{'=' * 74}\ndone" + (" (dry run)" if args.dry_run else ""))
    for model in args.models:
        print(f"  llm_judge_score_{model}_3p_promptmean  now on: "
              f"{args.stems} (and manugram_aiscientist, from the pilot)")


if __name__ == "__main__":
    main()

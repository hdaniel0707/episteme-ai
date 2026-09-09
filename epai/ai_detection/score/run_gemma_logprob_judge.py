r"""Add the gemma4-31b *logprob-mode* judge column to the four core benchmark
splits.

Background. In ``results/trainv2/judge_by_generator`` the ``gemma4_31b`` judge
is the closest open-weight judge to ``gpt56luna`` on mean AUROC (0.91 vs 0.98)
but well behind on TPR@1%FPR (0.43 vs 0.57). ``analyse_judge_operating_point``
showed why for the weaker open-weight judges: a written probability clumps onto
a few round values, and a tied clump cannot be split by any threshold, which is
exactly what a strict low-FPR operating point needs. ``--mode logprob`` avoids
that -- it reads ``P(human) / (P(human) + P(machine))`` off the answer token's
distribution, which has no ties -- and ``add_llm_judge_scores.py`` now supports
it for ``gemma4-31b`` over the GenAI4Science endpoint (see
``LOGPROB_API_MODELS`` and ``_score_genai4science_logprob`` there, and the
``probe_genai4science_logprobs.py`` capability check behind it).

This is a thin loop over ``add_llm_judge_scores.py`` -- one run per file, with
``--model gemma4-31b --mode logprob``. It writes a NEW column,
``llm_judge_score_gemma4_31b_v3_lp``, beside the existing parsed
``llm_judge_score_gemma4_31b_v3``; nothing else in the files is touched, and a
re-run only pays for rows that are still Null (the underlying script skips
filled ones). Every call is verified inside that script before its score is
kept: the endpoint returned a distribution, the generated token really is
``human`` / ``machine``, and the two words hold enough of the window for the
ratio to mean something. A row that keeps failing those checks is left Null and
counted in the per-file summary.

    # look before leaping
    python epai/ai_detection/score/run_gemma_logprob_judge.py --dry-run

    # one small file first, into a scratch copy, to eyeball the summary
    python epai/ai_detection/score/run_gemma_logprob_judge.py \
        --only manugram_den --out-dir /tmp/lp_smoke --limit 60 --yes

    # the real thing, in place, all four files
    python epai/ai_detection/score/run_gemma_logprob_judge.py --yes
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
JUDGE_SCRIPT = BASE_DIR / "epai/ai_detection/score/add_llm_judge_scores.py"

# The four core splits the benchmark reports on (gb_orig / mg_gb / mg_sci /
# mg_den). The rephrase splits are a separate pipeline and not in scope here.
STEMS = ["ghostbuster_orig", "manugram_gb", "manugram_sci", "manugram_den"]

MODEL = "gemma4-31b"
PROMPT = "v3"

_BOLD, _RED, _GREEN, _YELLOW, _DIM, _RESET = (
    "\033[1m", "\033[91m", "\033[92m", "\033[93m", "\033[2m", "\033[0m"
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="+", default=None, metavar="STEM", choices=STEMS,
                    help="run only these of %(choices)s")
    ap.add_argument("--out-dir", type=Path, default=None, metavar="DIR",
                    help="write <stem>.parquet into DIR instead of editing the "
                         "file in place -- for a safe smoke test")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="pass through to add_llm_judge_scores.py: judge only N "
                         "rows, drawn round-robin across domain x generator")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent API calls; 4 is the GenAI4Science "
                         "performance-endpoint cap (default)")
    ap.add_argument("--save-every", type=int, default=None, metavar="N",
                    help="pass through: rows scored between two writes of the "
                         "Parquet (default 200). Raise it for the ~800 MB "
                         "manugram_gb / ghostbuster_orig files so the run is not "
                         "dominated by re-serialising them")
    ap.add_argument("--yes", action="store_true",
                    help="skip add_llm_judge_scores.py's confirm prompts")
    ap.add_argument("--recompute", action="store_true",
                    help="re-judge rows that already have a logprob score")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    args = ap.parse_args()

    if args.out_dir is not None and not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    stems = args.only or STEMS
    print(f"{_BOLD}gemma4-31b logprob judge -> "
          f"llm_judge_score_gemma4_31b_{PROMPT}_lp{_RESET}")
    print(f"  files   {', '.join(stems)}")
    print(f"  mode    logprob (P(human) off the answer token; verified per call)\n")

    failures: list[str] = []
    for stem in stems:
        parquet = BASE_DIR / "data/parquet" / f"{stem}.parquet"
        print(f"{_BOLD}{'━' * 74}\n  {stem}{_RESET}  {_DIM}{parquet}{_RESET}")
        if not args.dry_run and not parquet.is_file():
            print(f"{_YELLOW}  not on disk — skipped{_RESET}")
            failures.append(stem)
            continue

        cmd = [sys.executable, str(JUDGE_SCRIPT), str(parquet),
               "--model", MODEL, "--mode", "logprob", "--prompt", PROMPT,
               "--workers", str(args.workers)]
        if args.out_dir is not None:
            cmd += ["-o", str(args.out_dir / f"{stem}.parquet")]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.save_every is not None:
            cmd += ["--save-every", str(args.save_every)]
        if args.recompute:
            cmd.append("--recompute")
        if args.yes:
            cmd.append("--yes")

        print(f"{_DIM}$ {' '.join(cmd)}{_RESET}\n", flush=True)
        if args.dry_run:
            continue
        if subprocess.run(cmd, cwd=BASE_DIR).returncode != 0:
            failures.append(stem)

    if failures:
        print(f"\n{_RED}failed: {', '.join(failures)}{_RESET}\n"
              "  Re-run — filled rows are skipped, so it only retries the gaps.")
        raise SystemExit(1)
    if not args.dry_run:
        print(f"\n{_GREEN}Done — all four files carry "
              f"llm_judge_score_gemma4_31b_{PROMPT}_lp.{_RESET}")


if __name__ == "__main__":
    main()

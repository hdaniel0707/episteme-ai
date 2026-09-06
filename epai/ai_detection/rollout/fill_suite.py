r"""Bring a corpus up to the full detector suite — every column
``cfg/detector_suite.yaml`` names.

Runs ``run_holdout_pipeline.py --from embed --skip-existing`` over each parquet:
the suite's embedders, Binoculars pairs, Fast-DetectGPT pairs, detectors,
judges, and the cue vector. Every scoring script skips rows that already have a
value, and ``--skip-existing`` skips a whole stage whose column is 100% full, so
this only computes what is genuinely missing — a judge just added to the suite
(``gemma4-31b``), an arm just appended to a corpus, a stage that failed before.

It does NOT generate, restructure or standardise — the text must already be in
the shape the training corpora are (a finished corpus, or the output of
``append_generator_rows.py``). After scoring it runs
``check_parquet_completeness.py --fail-on error`` unless ``--no-verify``.

    python epai/ai_detection/rollout/fill_suite.py data/parquet/denario_hc.parquet --gpu 2 --yes

    python epai/ai_detection/rollout/fill_suite.py \
        data/parquet/ghostbuster_gpt56luna.parquet \
        data/parquet/ghostbuster_gpt35t.parquet \
        data/parquet/ghostbuster_gpt35ts.parquet --gpu 2 --yes

    python epai/ai_detection/rollout/fill_suite.py data/parquet/denario_hc.parquet \
        --skip-judge llama33-70b --dry-run     # if a judge endpoint is down
"""

from __future__ import annotations

import argparse
from pathlib import Path

from epai.ai_detection.rollout._common import (
    GREEN, RED, RESET, YELLOW, banner, check_completeness, holdout_score, rel,
    resolve_judges,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("parquets", nargs="+", type=Path,
                    help="corpus parquet(s) to bring up to the full suite")
    ap.add_argument("--skip-judge", nargs="+", default=[], metavar="JUDGE",
                    help="suite judges to leave out (substring), e.g. llama33-70b "
                         "if that endpoint is down")
    ap.add_argument("--gpu", default=None, metavar="ID")
    ap.add_argument("--cpu-threads", type=int, default=None, metavar="N")
    ap.add_argument("--per-band", type=int, default=20000,
                    help="cue-vector coverage cap; must exceed the largest "
                         "domain's row count (default 20000)")
    ap.add_argument("--allow-incomplete-groups", action="store_true",
                    help="pass through to the post-scoring completeness check")
    ap.add_argument("--allow-unstandardized", action="store_true",
                    help="pass through to the post-scoring completeness check "
                         "(for a deliberately raw corpus, e.g. ghostbuster_gpt35t)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the completeness check after scoring")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    judges = resolve_judges(args.skip_judge)
    print(f"  suite judges: {', '.join(judges)}")

    failures: list[str] = []
    for parquet in args.parquets:
        banner(parquet.name, rel(parquet))
        if not args.dry_run and not parquet.is_file():
            print(f"{YELLOW}not on disk — skipped{RESET}")
            failures.append(str(parquet))
            continue

        code = holdout_score(parquet, judges, gpu=args.gpu,
                             cpu_threads=args.cpu_threads, per_band=args.per_band,
                             yes=args.yes, dry_run=args.dry_run)
        if code != 0:
            print(f"{RED}scoring failed on {rel(parquet)} (exit {code}).{RESET}")
            failures.append(str(parquet))
            continue

        if not args.no_verify:
            code = check_completeness(
                parquet, allow_incomplete_groups=args.allow_incomplete_groups,
                allow_unstandardized=args.allow_unstandardized,
                dry_run=args.dry_run,
            )
            if code != 0:
                failures.append(str(parquet))

    if failures:
        print(f"\n{RED}{len(failures)} corpus/corpora failed: "
              f"{', '.join(failures)}{RESET}\n  Re-run — filled rows / stages are "
              f"skipped, so it only retries the gaps.")
        raise SystemExit(1)
    if not args.dry_run:
        print(f"\n{GREEN}Done — {len(args.parquets)} corpus/corpora carry the "
              f"full detector suite.{RESET}")


if __name__ == "__main__":
    main()

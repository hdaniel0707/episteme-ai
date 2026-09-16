r"""Fill the embed / zero-shot / detector / judge columns on MG-Reph-GB and
MG-Reph-Sci.

Thin wrapper around ``rollout._common.holdout_score`` -- the same
``run_holdout_pipeline.py --from embed --skip-existing`` that
``rephrase_build.py``'s own ``score`` step uses -- run over both halves, so a
pass after ``reph_add_rewrite.py`` / ``reph_rephrase_rewrite.py`` fills
whatever the merge or the new gpt56luna-rephrase-of-rewrite rows are still
missing. Each stage only touches Null cells, so a rerun costs nothing for rows
that already have a score.

Default judge is ``gpt56luna`` only, matching MG-Reph's existing pattern
(``split_manugram_reph.py``'s docstring: only the gpt56luna judge is scored on
both classes there, by design -- MG-Reph is opt-in and excluded from the
pooled / multi-judge runs). Pass ``--judges`` to widen that.

    python epai/ai_detection/rollout/reph_fill_scores.py --gpu 0 --yes
    python epai/ai_detection/rollout/reph_fill_scores.py --dry-run
    python epai/ai_detection/rollout/reph_fill_scores.py --only gb --yes
    python epai/ai_detection/rollout/reph_fill_scores.py --judges gpt56luna sonnet5 --yes
"""

from __future__ import annotations

import argparse

from epai.ai_detection.rollout._common import GREEN, RED, RESET, banner, holdout_score, rel
from epai.ai_detection.rollout.reph_add_rewrite import FAMILIES


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="+", choices=sorted(FAMILIES), default=None,
                    help="restrict to one or both families (default: both)")
    ap.add_argument("--judges", nargs="+", default=["gpt56luna"], metavar="JUDGE")
    ap.add_argument("--gpu", default=None, metavar="ID")
    ap.add_argument("--cpu-threads", type=int, default=None, metavar="N")
    ap.add_argument("--per-band", type=int, default=20000,
                    help="cue-vector coverage cap; must exceed the row count")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fams = args.only or sorted(FAMILIES)
    for fam in fams:
        target = FAMILIES[fam]["target"]
        banner("score", rel(target))
        code = holdout_score(
            target, args.judges, gpu=args.gpu, cpu_threads=args.cpu_threads,
            per_band=args.per_band, yes=args.yes, dry_run=args.dry_run,
        )
        if code != 0:
            print(f"{RED}scoring failed on {rel(target)} (exit {code}).{RESET}")
            raise SystemExit(code)

    print(f"\n{GREEN}Done.{RESET}")


if __name__ == "__main__":
    main()

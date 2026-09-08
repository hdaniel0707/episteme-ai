r"""Top up the ``qwen3-14b`` judge column on the Ghostbuster v3f corpus.

The suite judge ``qwen3-14b`` is a local bf16 model; a run of it left ~100 rows
(0.7%, almost all gpt-oss-120b) with a null ``llm_judge_score_qwen3_14b_v3``.
This re-runs ``add_llm_judge_scores.py`` for that one judge: it skips every row
that already has a score, so only the nulls are re-judged. Nothing else in the
file is touched.

Needs a GPU. ``--prompt`` follows ``cfg/detector_suite.yaml``'s
``judge_prompt_version`` (v3), so the refilled rows land in the same column as
the rest.

    python epai/ai_detection/rollout/gb_v3f_refill_qwen.py --gpu 2 --dry-run
    python epai/ai_detection/rollout/gb_v3f_refill_qwen.py --gpu 2 --yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

from epai.ai_detection.rollout._common import (
    BASE_DIR, COMPLETENESS_SCRIPT, GREEN, JUDGE_SCRIPT, RED, RESET, YELLOW,
    banner, prompt_version, rel, run,
)

CORPUS = BASE_DIR / "data" / "parquet" / "ghostbuster_v3f_multi_clean.parquet"
JUDGE = "qwen3-14b"


def null_count(parquet: Path, column: str) -> tuple[int, int]:
    """(null rows, total rows) for ``column``; (-1, total) if the column is absent."""
    if column not in pq.read_schema(parquet).names:
        return -1, pq.read_metadata(parquet).num_rows
    col = pq.read_table(parquet, columns=[column]).column(column)
    return col.null_count, len(col)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("parquet", type=Path, nargs="?", default=CORPUS,
                    help=f"corpus to patch in place (default {rel(CORPUS)})")
    ap.add_argument("--gpu", default=None, metavar="ID",
                    help="CUDA_VISIBLE_DEVICES for the local judge (required "
                         "in practice)")
    ap.add_argument("--cpu-threads", type=int, default=16, metavar="N")
    ap.add_argument("--batch-size", type=int, default=None, metavar="N",
                    help="rows per forward pass (passthrough to the judge)")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the completeness report afterwards")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src: Path = args.parquet
    if not src.is_file():
        raise SystemExit(f"{RED}not a file: {rel(src)}{RESET}")

    pv = prompt_version()
    column = f"llm_judge_score_{JUDGE.replace('-', '_')}_{pv}"
    nulls, total = null_count(src, column)

    banner(f"{JUDGE} refill", rel(src))
    if nulls == 0:
        print(f"{GREEN}{column} is already full ({total} rows). Nothing to do.{RESET}")
        return
    if nulls < 0:
        print(f"{YELLOW}{column} does not exist yet — this will score every "
              f"row ({total}).{RESET}")
    else:
        print(f"{column}: {nulls}/{total} null -> will be judged "
              f"(the other {total - nulls} are skipped)")
    if args.gpu is None:
        print(f"{YELLOW}no --gpu given; the judge will try every visible card.{RESET}")

    cmd = [sys.executable, BASE_DIR / JUDGE_SCRIPT, src,
           "--model", JUDGE, "--prompt", pv, "--cpu-threads", args.cpu_threads]
    if args.gpu is not None:
        cmd += ["--gpu", args.gpu]
    if args.batch_size is not None:
        cmd += ["--batch-size", args.batch_size]
    if args.yes:
        cmd.append("--yes")

    code = run(cmd, args.dry_run)
    if args.dry_run:
        print(f"\n{YELLOW}--dry-run: nothing scored.{RESET}")
        return
    if code != 0:
        raise SystemExit(f"{RED}judge run failed (exit {code}).{RESET}")

    left, _ = null_count(src, column)
    print(f"\n{column}: {left} null remaining")
    if left:
        print(f"{YELLOW}some rows still null — re-run to retry them.{RESET}")

    if not args.no_check:
        banner("completeness report", rel(src))
        run([sys.executable, BASE_DIR / COMPLETENESS_SCRIPT, src,
             "--max-essay-human-blanks"], False)


if __name__ == "__main__":
    main()

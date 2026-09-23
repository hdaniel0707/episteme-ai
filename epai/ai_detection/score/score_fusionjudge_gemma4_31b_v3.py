r"""FusionJudge for the gemma-4-31b judge: arithmetic mean of its two v3-prompt
P(human) readouts -- the parsed judge score and the ord2 LogJudge score --
into one column:

    fusionjudge_gemma4_31b_v3_score
        = mean(llm_judge_score_gemma4_31b_v3, logjudge_ord2_gemma4_31b_v3)

The gemma twin of ``score_fusionjudge_gpt56luna_v3.py`` (``fusionjudge_score``):
same members per judge, same unweighted ``np.nanmean`` rule, same refusal to
replace an existing column without ``--overwrite``.

Run:
    uv run python epai/ai_detection/score/score_fusionjudge_gemma4_31b_v3.py --dry-run
    uv run python epai/ai_detection/score/score_fusionjudge_gemma4_31b_v3.py
    uv run python epai/ai_detection/score/score_fusionjudge_gemma4_31b_v3.py --only manugram_gb

Deterministic, no API, one read + write per file (``--out-dir`` to divert).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from epai.ai_detection.score.score_fusionjudge_gpt56luna_v3 import (
    ALL_STEMS,
    CORE_STEMS,
    PARQUET_DIR,
    _distribution,
    mean_combine,
)
from epai.utils.parquet_utils import load_parquet, save_parquet

MEMBERS = ["llm_judge_score_gemma4_31b_v3", "logjudge_ord2_gemma4_31b_v3"]
OUT_COL = "fusionjudge_gemma4_31b_v3_score"


def process_stem(stem: str, args: argparse.Namespace) -> None:
    src = PARQUET_DIR / f"{stem}.parquet"
    out = (Path(args.out_dir) / f"{stem}.parquet") if args.out_dir else src
    print(f"\n{'-' * 74}\n{stem}")
    if not src.is_file():
        print("  not on disk -- skipped")
        return
    names = set(pq.read_schema(src).names)
    missing = [c for c in MEMBERS if c not in names]
    if missing:
        print(f"  missing member column(s), skipped: {missing}")
        return
    if OUT_COL in names and not args.overwrite:
        print(f"  {OUT_COL} already present -- pass --overwrite; skipped")
        return

    M = np.vstack([
        pq.read_table(src, columns=[c]).column(0).to_numpy(zero_copy_only=False).astype(float)
        for c in MEMBERS
    ])
    s = mean_combine(M)
    print(f"  mean({MEMBERS})  ->  {OUT_COL}")
    print(_distribution(s))

    if args.dry_run:
        print("  [dry run] -- nothing written")
        return

    df = load_parquet(src)
    df[OUT_COL] = s
    save_parquet(df, out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="+", choices=ALL_STEMS, metavar="STEM")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write scored <stem>.parquet here instead of in place")
    ap.add_argument("--overwrite", action="store_true",
                    help=f"replace a {OUT_COL} column that already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print the score distribution, write nothing")
    args = ap.parse_args()

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    for stem in (args.only or CORE_STEMS):
        process_stem(stem, args)
    print(f"\n{'=' * 74}\ndone")


if __name__ == "__main__":
    main()

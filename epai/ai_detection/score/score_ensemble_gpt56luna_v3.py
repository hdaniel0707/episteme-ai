r"""Arithmetic mean of gpt-5.6-luna's two v3-prompt P(human) readouts --
the directly-parsed judge score and the logprob-derived ord2 score -- into
one column:

    ensemble_score

    ensemble_score = mean(llm_judge_score_gpt56luna_v3, logjudge_ord2_gpt56luna_v3)

Both members are already P(human) in [0, 1] on the same orientation
(high = human), so this is a plain, unweighted average -- no sign-flip or
rescaling needed, unlike combining across score families with genuinely
different scales/directions (e.g. binoculars_score / fastdetect_crit, whose
ranges and orientation don't match either of these).

Row-wise arithmetic mean over whichever member is non-null for that row
(``np.nanmean`` -- same policy as every other ``*mean`` script in this
directory); a row null in both members stays null.

Run:
    uv run python epai/ai_detection/score/score_ensemble_gpt56luna_v3.py --dry-run
    uv run python epai/ai_detection/score/score_ensemble_gpt56luna_v3.py
    uv run python epai/ai_detection/score/score_ensemble_gpt56luna_v3.py --only manugram_gb

Deterministic, no API, one read + write per file (``--out-dir`` to divert).
Refuses to replace an existing ``ensemble_score`` column without
``--overwrite``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from epai.utils.parquet_utils import load_parquet, save_parquet

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
CORE_STEMS = ["ghostbuster_orig", "manugram_gb", "manugram_sci", "manugram_aiscientist"]

MEMBERS = ["llm_judge_score_gpt56luna_v3", "logjudge_ord2_gpt56luna_v3"]
OUT_COL = "ensemble_score"


def mean_combine(scores: np.ndarray) -> np.ndarray:
    """``scores``: ``(M, N)``, NaN for a missing member. Row-wise arithmetic
    mean over the members present. An all-NaN row stays NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN row -> NaN
        return np.clip(np.nanmean(scores, axis=0), 0.0, 1.0)


def _distribution(v: np.ndarray) -> str:
    ok = v[np.isfinite(v)]
    if not len(ok):
        return "    all Null"
    uniq, cnt = np.unique(np.round(ok, 6), return_counts=True)
    return (f"    n={len(ok)}  null={len(v) - len(ok)}  distinct={len(uniq)}  "
            f"clump={cnt.max() / cnt.sum():.0%}\n"
            f"    range [{ok.min():.3f}, {ok.max():.3f}]  mean {ok.mean():.3f}  "
            f"median {np.median(ok):.3f}")


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
    ap.add_argument("--only", nargs="+", choices=CORE_STEMS, metavar="STEM")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write scored <stem>.parquet here instead of in place")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an ensemble_score column that already exists")
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

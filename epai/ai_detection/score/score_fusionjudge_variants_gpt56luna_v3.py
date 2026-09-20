r"""Two more cross-readout fusions of gpt-5.6-luna's two v3-prompt P(human)
readouts -- the directly-parsed judge score and the logprob-derived ord2
(LogJudge) score -- alongside the plain arithmetic mean already computed by
``score_fusionjudge_gpt56luna_v3.py`` (``fusionjudge_score``). Same two
members, same [0, 1] / high=human scale, both self-dual
(``F(1-a,1-b) = 1-F(a,b)``) like the mean, from the fusion-methods
discussion (2026-09-20 conversation):

    fusionjudge_logodds_score = sigmoid( mean( logit(a), logit(b) ) )
        Log-odds ("logarithmic opinion pool") average -- instead of
        diluting a confident readout with an uncertain one, it extremizes
        toward whichever member is more decisive: a=0.99, b=0.5 lands near
        0.99, not 0.745 (the plain-mean result). Members are clipped to
        [EPS, 1-EPS] first since either can be exactly 0 or 1 -- LogJudge's
        score is reported near-bimodal (most AI-class mass at 0).

    fusionjudge_bayes_score = a*b / (a*b + (1-a)*(1-b))
        Naive-Bayes / normalized product rule: the closed-form Bayes-optimal
        fusion of two conditionally-independent estimates of the same
        posterior. Rewards agreement more sharply than the log-odds average.
        Null at an exact (1,0)/(0,1) full contradiction -- the limit is
        path-dependent there (no single continuous value), not merely a 0/0
        we can round away.

(A wider sweep of this family -- a tempered gamma generalization, an
agreement-gated blend with the mean, and two disagreement-penalizing
alternatives -- was tried and dropped: none beat the plain mean at the
paper's 1%-FPR operating point, and a diagnostic traced why to LogJudge
being the more reliable member 92.6% of the time overall, so anything that
discounts its confidence gives up more than it protects. These two are kept
because they are the well-known, citable members of the family and stay
self-dual/commutative; the paper's headline fusion remains the plain mean.)

Both are commutative (order of the two members doesn't matter) and computed
row-wise with the same missing-member policy as
``score_fusionjudge_gpt56luna_v3.py``: whichever member is present when the
other is Null is passed through unchanged; a row Null in both stays Null.

Run:
    uv run python epai/ai_detection/score/score_fusionjudge_variants_gpt56luna_v3.py --dry-run
    uv run python epai/ai_detection/score/score_fusionjudge_variants_gpt56luna_v3.py
    uv run python epai/ai_detection/score/score_fusionjudge_variants_gpt56luna_v3.py --only manugram_gb
    uv run python epai/ai_detection/score/score_fusionjudge_variants_gpt56luna_v3.py --only manugram_reph_gb manugram_reph_sci

Deterministic, no API, one read + write per file (``--out-dir`` to divert).
Refuses to replace either column without ``--overwrite``.
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
#: opt-in only -- pass via --only, never swept by a bare (no --only) run, so
#: the rephrase task's scores get computed on request, not by default.
REPHRASE_STEMS = ["manugram_reph_gb", "manugram_reph_sci"]
ALL_STEMS = CORE_STEMS + REPHRASE_STEMS

MEMBERS = ["llm_judge_score_gpt56luna_v3", "logjudge_ord2_gpt56luna_v3"]
EPS = 1e-6


def _rowwise(a: np.ndarray, b: np.ndarray, both: np.ndarray) -> np.ndarray:
    """``both`` where both members are finite; whichever member is finite is
    passed through unchanged when the other is Null; Null when both are
    Null -- same policy as score_fusionjudge_gpt56luna_v3.py's nanmean."""
    out = np.where(np.isnan(a) & np.isnan(b), np.nan, both)
    out = np.where(np.isnan(b) & ~np.isnan(a), a, out)
    out = np.where(np.isnan(a) & ~np.isnan(b), b, out)
    return np.clip(out, 0.0, 1.0)


def logodds_combine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # NaN members -> NaN
        ac, bc = np.clip(a, EPS, 1 - EPS), np.clip(b, EPS, 1 - EPS)
        logit_sum = np.log(ac / (1 - ac)) + np.log(bc / (1 - bc))
        both = 1.0 / (1.0 + np.exp(-logit_sum / 2.0))
    return _rowwise(a, b, both)


def bayes_combine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        num = a * b
        den = num + (1 - a) * (1 - b)
        both = np.where(den == 0, np.nan, num / np.where(den == 0, 1.0, den))
    return _rowwise(a, b, both)


OUT_COLS = {
    "fusionjudge_logodds_score": logodds_combine,
    "fusionjudge_bayes_score": bayes_combine,
}


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

    todo = {c: fn for c, fn in OUT_COLS.items() if c not in names or args.overwrite}
    skipped = [c for c in OUT_COLS if c not in todo]
    if skipped:
        print(f"  already present -- pass --overwrite; skipped: {skipped}")
    if not todo:
        return

    a, b = (pq.read_table(src, columns=[c]).column(0).to_numpy(zero_copy_only=False).astype(float)
            for c in MEMBERS)
    computed = {}
    for col, fn in todo.items():
        computed[col] = fn(a, b)
        print(f"  {fn.__name__}({MEMBERS})  ->  {col}")
        print(_distribution(computed[col]))

    if args.dry_run:
        print("  [dry run] -- nothing written")
        return

    df = load_parquet(src)
    for col, s in computed.items():
        df[col] = s
    save_parquet(df, out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="+", choices=ALL_STEMS, metavar="STEM")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write scored <stem>.parquet here instead of in place")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace either fusion column that already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print the score distributions, write nothing")
    args = ap.parse_args()

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    for stem in (args.only or CORE_STEMS):
        process_stem(stem, args)
    print(f"\n{'=' * 74}\ndone")


if __name__ == "__main__":
    main()

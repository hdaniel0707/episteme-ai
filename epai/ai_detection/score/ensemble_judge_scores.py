r"""Soft-min or mean ensemble of judge P(human) score columns.

``--combine softmin`` (default) -- combining across judge *models*:

    s = −T · log( (1/M) Σ_m exp( −s_m / T ) )

over the member columns, per row. Symmetric in the members, extends to any
number of them, and robust to adding a weak member (a near-chance score is
rarely the minimum, so it self-neutralises -- like a chance model in a
product-of-experts). T → 0 recovers the hard min; T → ∞ the mean; T ≈ 0.1 is
the knee. See ``score_logprob_windows.md`` § "Ensembling across models".

``--combine mean`` -- combining across judge *prompts*, same model: a plain
per-row arithmetic mean over the members. Intended for the prompt-diversity
LogJudge pilot -- averaging a model's ord2 (``_wlp``) score under several
prompts (e.g. ``v3`` + 2 diverse candidates from
``select_diverse_judge_prompts.py``) into one column, rather than averaging
across models the way ``softmin`` does. No shrinkage, no temperature: unlike
``softmin`` this has no reason to favour the lowest (most machine-leaning)
member, since the members here disagree about the same document under
different framings rather than being judges of differing reliability.

Inputs and output are all **P(human)** (1.0 human, 0.0 machine,
``machine_is_high=False``), so trainv2's ``llm_judge_score_`` prefix picks the
new column(s) up with no config change (``run_zeroshot_performance.py``,
``calibrate_judge_baseline.py``, …).

Output column:
  ``softmin``  ``llm_judge_score_<name>_v3_smin_<tau>``, one per ``--tau``
               value (``.`` → ``p``: T=0.1 → ``smin_0p1``)
  ``mean``     ``llm_judge_score_<name>_promptmean``, ``--tau`` ignored

``<name>``  ``--name`` (default ``ens``)

    .venv/bin/python epai/ai_detection/score/ensemble_judge_scores.py --dry-run
    .venv/bin/python epai/ai_detection/score/ensemble_judge_scores.py --tau 0.1
    .venv/bin/python epai/ai_detection/score/ensemble_judge_scores.py --tau 0.05 0.1 0.15 0.2
    .venv/bin/python epai/ai_detection/score/ensemble_judge_scores.py \
        --members llm_judge_score_gemma4_31b_v3_wlp llm_judge_score_gpt56luna_v3_wlp \
                  llm_judge_score_gpt56luna_v3 --name g3 --tau 0.1
    .venv/bin/python epai/ai_detection/score/ensemble_judge_scores.py --combine mean \
        --members llm_judge_score_gpt56luna_v3_wlp llm_judge_score_gpt56luna_cand03_wlp \
                  llm_judge_score_gpt56luna_cand07_wlp \
        --name gpt56luna_3prompt --only manugram_aiscientist

Deterministic, no API, one read + write per file (``--out-dir`` to divert).
Refuses to replace an existing column without ``--overwrite``.
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

DEFAULT_MEMBERS = [
    "llm_judge_score_gemma4_31b_v3_wlp",
    "llm_judge_score_gpt56luna_v3_wlp",
]


def soft_min(scores: np.ndarray, tau: float) -> np.ndarray:
    """``scores``: ``(M, N)``, NaN for a missing member. Row-wise soft-min over
    the members present. An all-NaN row stays NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN row -> NaN
        mean_e = np.nanmean(np.exp(-scores / tau), axis=0)
    return np.clip(-tau * np.log(mean_e), 0.0, 1.0)


def mean_combine(scores: np.ndarray) -> np.ndarray:
    """``scores``: ``(M, N)``, NaN for a missing member. Row-wise arithmetic
    mean over the members present. An all-NaN row stays NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN row -> NaN
        return np.clip(np.nanmean(scores, axis=0), 0.0, 1.0)


def _tau_tag(tau: float) -> str:
    return f"{tau:g}".replace(".", "p").replace("-", "m")


def _dist(v: np.ndarray) -> str:
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
    missing = [c for c in args.members if c not in names]
    if missing:
        print(f"  missing member column(s), skipped: {missing}")
        return

    M = np.vstack([
        pq.read_table(src, columns=[c]).column(0).to_numpy(zero_copy_only=False).astype(float)
        for c in args.members
    ])
    print(f"  members ({len(args.members)}): {args.members}")

    computed: dict[str, np.ndarray] = {}
    if args.combine == "mean":
        col = f"llm_judge_score_{args.name}_promptmean"
        if col in names and not args.overwrite:
            print(f"  {col} already present -- pass --overwrite; skipped")
        else:
            s = mean_combine(M)
            computed[col] = s
            print(f"  mean  ->  {col}")
            print(_dist(s))
    else:
        for tau in args.tau:
            col = f"llm_judge_score_{args.name}_v3_smin_{_tau_tag(tau)}"
            if col in names and not args.overwrite:
                print(f"  {col} already present -- pass --overwrite; skipped")
                continue
            s = soft_min(M, tau)
            computed[col] = s
            print(f"  T={tau:g}  ->  {col}")
            print(_dist(s))

    if args.dry_run:
        print("  [dry run] -- nothing written")
        return
    if not computed:
        return
    df = load_parquet(src)
    for col, s in computed.items():
        df[col] = s
    save_parquet(df, out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS,
                    help="judge P(human) score columns to combine "
                         f"(default: {DEFAULT_MEMBERS})")
    ap.add_argument("--combine", choices=["softmin", "mean"], default="softmin",
                    help="'softmin' (default): shrinkage combiner across judge "
                         "models, one column per --tau. 'mean': plain average "
                         "across judge prompts (same model), --tau ignored, "
                         "one column.")
    ap.add_argument("--tau", nargs="+", type=float, default=[0.1],
                    help="soft-min temperature(s); one output column per value "
                         "(default: 0.1). Ignored for --combine mean.")
    ap.add_argument("--name", default="ens",
                    help="stem in the output column name (default: 'ens')")
    ap.add_argument("--only", nargs="+", choices=CORE_STEMS, metavar="STEM")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write scored <stem>.parquet here instead of in place")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an ..._smin_<tau> column that already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and print the distributions, write nothing")
    args = ap.parse_args()

    if args.combine == "softmin" and any(t <= 0 for t in args.tau):
        ap.error("--tau values must be positive")
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    for stem in (args.only or CORE_STEMS):
        process_stem(stem, args)
    print(f"\n{'=' * 74}\ndone")


if __name__ == "__main__":
    main()

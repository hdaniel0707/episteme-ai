r"""Score the captured ``llm_judge_logprobs_*_words`` windows with the
paper's exact ``ord2`` LogJudge mechanism (Section ``method_logjudge``,
Eq. ``logjudge_simple`` / ``logjudge_contrast`` in
``latex/_ICLR_2027__AI_Detect/iclr2027_conference.tex``):

    L = {human, machine},  v(human) = 1,  v(machine) = 0,  lambda(x) = P(x)

which collapses to the closed form

    s(x) = (1 + p_h(x) - p_m(x)) / 2

-- p_h(x), p_m(x) the probability mass the judge placed on the ``human`` /
``machine`` answer tokens (summed over casing / leading-space spellings) at
the answer position, read off the captured top-k logprob window. s(x) is
P(human) directly, per the paper's orientation convention ("Edge cases and
orientation": s(x) is Pr(y=0|x)).

This is a LITERAL implementation of the paper's formula -- no floor, no
branch, no renormalisation, no clamp. The paper is explicit that the
contrast readout "needs no floor, branching, or clamp": as
P(x) = p_h(x) + p_m(x) -> 0, s(x) -> 0.5 on its own (the shrinkage is baked
into the formula), so a low-mass answer already self-shrinks toward the
uninformative prior without any special-casing. That's why this is a
separate script from ``score_logprob_windows.py``'s ``wlp`` /
``logjudge_t0000`` columns, which add an extra MIN_ANSWER_MASS floor
(< 0.05 combined answer mass -> hard 0.5) that is not in the paper's
equation.

Only edge case: an empty window (the judge call itself failed,
W(x) = emptyset) scores Null, not 0.5 -- a failed measurement is not the
same as an answered-but-uninformative one (paper's "Edge cases and
orientation").

Output columns, one per (model, prompt id) with a captured ``_words``
window -- any prompt id present (``v3``, a candidate id such as ``cand03``,
...), read straight off the input column name, so this needs no ``--prompt``
of its own:

    logjudge_ord2_<model>_<id>

Run:
    uv run python epai/ai_detection/score/score_logjudge_ord2.py --dry-run
    uv run python epai/ai_detection/score/score_logjudge_ord2.py
    uv run python epai/ai_detection/score/score_logjudge_ord2.py --only manugram_aiscientist
    uv run python epai/ai_detection/score/score_logjudge_ord2.py --out-dir data/parquet/tmp/scored
    uv run python epai/ai_detection/score/score_logjudge_ord2.py --only manugram_reph_gb manugram_reph_sci

Deterministic, no API, one read + write per file (``--out-dir`` to divert).
Refuses to replace an existing ``logjudge_ord2_*`` column without
``--overwrite``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from epai.ai_detection.score.add_llm_judge_scores import MODELS
from epai.utils.parquet_utils import load_parquet, save_parquet

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
CORE_STEMS = ["ghostbuster_orig", "manugram_gb", "manugram_sci", "manugram_aiscientist"]
#: opt-in only -- pass via --only, never swept by a bare (no --only) run, so
#: the rephrase task's scores get computed on request, not by default.
REPHRASE_STEMS = ["manugram_reph_gb", "manugram_reph_sci"]
ALL_STEMS = CORE_STEMS + REPHRASE_STEMS

IN_PREFIX = "llm_judge_logprobs_"
IN_SUFFIX = "_words"
OUT_PREFIX = "logjudge_ord2_"

# Longest first: a column is <prefix><model_column>_<prompt_id><suffix>, so
# the model column has to be matched greedily against the known aliases in
# add_llm_judge_scores.MODELS -- the prompt id in between (v3, or a
# candidate id like cand03) is open-ended and can't otherwise be told apart
# from a model column that also contains underscores.
_MODEL_COLUMNS = sorted({m.column for m in MODELS.values()}, key=len, reverse=True)


def _parse_column(name: str) -> tuple[str, str] | None:
    """``llm_judge_logprobs_gpt56luna_cand03_words`` -> ``('gpt56luna', 'cand03')``."""
    if not (name.startswith(IN_PREFIX) and name.endswith(IN_SUFFIX)):
        return None
    body = name[len(IN_PREFIX): -len(IN_SUFFIX)]
    for model_col in _MODEL_COLUMNS:
        prefix = model_col + "_"
        if body.startswith(prefix) and len(body) > len(prefix):
            return model_col, body[len(prefix):]
    return None


def _norm(token: str | None) -> str:
    """The bare letters of a returned token, lowercased: ``' Human'`` -> ``'human'``
    -- the same fold ``capture_judge_logprobs.py`` applied when it picked the
    answer."""
    return "".join(ch for ch in (token or "") if ch.isalpha()).lower()


def answer_mass(window) -> tuple[float, float] | None:
    """(p_h(x), p_m(x)) -- summed probability mass the judge placed on the
    ``human`` / ``machine`` answer tokens in the window. ``None`` for an
    empty window (failed capture). The building block score_ord2() reduces
    to a scalar."""
    if not window:
        return None
    p_h = p_m = 0.0
    for s in window:
        lp = s.get("logprob")
        if lp is None or not math.isfinite(lp):
            continue
        tok = _norm(s.get("token"))
        if tok == "human":
            p_h += math.exp(lp)
        elif tok == "machine":
            p_m += math.exp(lp)
    return p_h, p_m


def score_ord2(window) -> float | None:
    """s(x) = (1 + p_h(x) - p_m(x)) / 2 -- the paper's ``ord2`` closed form,
    exactly (Eq. logjudge_simple). ``None`` for an empty window (failed
    capture), per the paper's Null-not-0.5 edge case; no other branch or
    floor -- the paper is explicit this readout "needs no floor, branching,
    or clamp" (a low-mass answer already shrinks toward 0.5 through the
    formula itself). The one ``np.clip`` below is float-precision safety,
    not a semantic floor: summing many ``exp(logprob)`` terms can land a
    hair outside [0, 1] (e.g. -1e-16), which this only cleans up, never
    overrides a real answer the way MIN_ANSWER_MASS does in
    score_logprob_windows.py."""
    mass = answer_mass(window)
    if mass is None:
        return None
    p_h, p_m = mass
    return float(np.clip((1.0 + p_h - p_m) / 2.0, 0.0, 1.0))


def _distribution(scores: list[float | None]) -> str:
    vals = np.array([s for s in scores if s is not None], dtype=float)
    n_null = len(scores) - len(vals)
    if not len(vals):
        return f"    all {n_null} row(s) Null"
    half = int((vals == 0.5).sum())
    uniq, counts = np.unique(np.round(vals, 6), return_counts=True)
    clump = counts.max() / counts.sum()
    return (f"    n={len(vals)}  null={n_null}  =0.5: {half}  "
            f"distinct={len(uniq)}  clump={clump:.0%}\n"
            f"    range [{vals.min():.3f}, {vals.max():.3f}]  "
            f"mean {vals.mean():.3f}  std {vals.std():.3f}  "
            f"median {np.median(vals):.3f}")


def process_stem(stem: str, args: argparse.Namespace) -> None:
    src = PARQUET_DIR / f"{stem}.parquet"
    out = (Path(args.out_dir) / f"{stem}.parquet") if args.out_dir else src
    print(f"\n{'-' * 74}\n{stem}")
    if not src.is_file():
        print("  not on disk -- skipped")
        return

    schema = pq.read_schema(src)
    todo: list[tuple[str, str]] = []  # (in_col, out_col)
    for name in schema.names:
        parsed = _parse_column(name)
        if parsed is None:
            continue
        model, prompt_id = parsed
        out_col = f"{OUT_PREFIX}{model}_{prompt_id}"
        if out_col in schema.names and not args.overwrite:
            print(f"  {out_col} already present -- pass --overwrite; skipped")
            continue
        todo.append((name, out_col))

    if not todo:
        print("  nothing to do")
        return

    computed: dict[str, list[float | None]] = {}
    for in_col, out_col in todo:
        window = pq.read_table(src, columns=[in_col]).column(0).to_pylist()
        scores = [score_ord2(w) for w in window]
        computed[out_col] = scores
        print(f"  {in_col}  ->  {out_col}")
        print(_distribution(scores))

    if args.dry_run:
        print("  [dry run] -- nothing written")
        return

    df = load_parquet(src)
    for out_col, scores in computed.items():
        df[out_col] = np.array(scores, dtype=float)  # None -> NaN
    save_parquet(df, out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", nargs="+", choices=ALL_STEMS, metavar="STEM")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write scored <stem>.parquet here instead of in place")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace a logjudge_ord2_* column that already exists")
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

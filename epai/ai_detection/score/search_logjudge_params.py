r"""Random hyperparameter search over the LogJudge scoring family.

Purely a post-hoc re-scoring of the already-captured judge logprob windows
(``llm_judge_logprobs_<model>_v3_{words,scale}``, see ``score_logprob_windows.py``)
under many choices of the paper's general form (Section 3.3 "LogJudge" /
Appendix "LogJudge: General Formulation" in
``latex/ICLR_AIDetect/iclr2027_conference.tex``)::

    s(x) = lambda(x) * sum_l [p_l(x)/P(x)] * v(l)  +  (1 - lambda(x)) * 0.5

No API calls, no logprob re-capture -- every trial just re-reads the same
top-k windows already on disk.

What's searched, per the paper's own axes (not a free-form sweep):

  prompt    ``words`` (|L|=2) or ``scale`` (|L|=5) -- the two prompts already
            captured; not interpolated, not sub-set. Looped over, not sampled.

  lambda    one of three families, sampled uniformly per trial:
            ``contrast`` (lambda(x)=P(x), the paper's adopted readout -- no
            renormalisation, no floor:
                s = sum_l p_l(x)*v(l) + (1 - P(x))*0.5
            ``threshold`` (lambda(x)=1[P(x)>=eps], the paper's named
            generalisation-axis alternative -- renormalises over the labels
            present, discards the residual below a floor:
                s = sum_l [p_l(x)/P(x)]*v(l)   if P(x) >= eps,  else 0.5
            ``eps`` sampled log-uniform in ``[EPS_LO, EPS_HI]`` per trial; or
            ``naive`` (lambda(x)=1 always -- the degenerate "just pick
            whichever label has more probability" baseline, no shrinkage at
            all, not even a floor -- only guarded against the P(x)=0
            divide-by-zero):
                s = sum_l [p_l(x)/P(x)]*v(l)   if P(x) > 0,  else 0.5
            ``naive`` is ``threshold`` with eps -> 0, kept as its own family
            (not folded into the eps sweep) so it always gets sampled rather
            than only approximated when a small eps happens to be drawn.

  v         only free under the ``scale`` prompt. The endpoints are fixed by
            the paper's own definition (v(A)=1 "certainly human", v(E)=0
            "certainly machine"); v(B), v(C), v(D) are drawn uniform on
            [0, 1] and sorted descending so the ladder stays monotonic
            (v(A) >= v(B) >= v(C) >= v(D) >= v(E)), matching the ordering the
            prompt itself states. Under the ``words`` prompt there is no free
            v (always v(human)=1, v(machine)=0) -- so a ``words`` trial only
            varies the lambda axis.

Objective: pooled AT-HM (paper's "Composite Metric" section) --
``epai.ai_detection.trainv2.core.goodness(auroc, tpr)`` applied to the POOLED
(mean-over-splits) AUROC and TPR@1%FPR, *not* a mean of per-split AT-HM
values -- this matches how ``plots.goodness_bars`` folds the existing
``performance.csv``'s ``split == 'mean'`` row (see ``run_zeroshot_performance.py``).
AUROC / TPR@1%FPR themselves come from ``epai.utils.cd_core``
(``auroc_score``, ``oriented_scores``, ``roc_point_at_fpr``) -- the exact
functions ``run_zeroshot_performance.py`` uses -- so a trial's numbers sit on
the same scale as the ``wlp``/``slp``/``sbc``/``logjudge_t0000``/
``logjudge_t0001`` rows already in
``results/trainv2/zeroshot_performance/performance.csv``.

Splits: ``epai.ai_detection.trainv2.splits.DEFAULT`` (the paper's four core
splits: gb_orig, mg_gb, mg_sci, mg_aisci), resolved the same way
``run_zeroshot_performance.py`` does. ``manugram_aiscientist.parquet`` already
carries the raw ``llm_judge_logprobs_*`` windows and the ``wlp``/``slp``/``sbc``
scores (written by a separate pipeline, not by these ``score/`` scripts) --
confirmed directly against the file, not assumed. The sibling ``score/``
scripts' own ``CORE_STEMS`` lists previously omitted this split (they listed
the superseded ``manugram_den`` and a nonexistent ``ai_scientist_hc`` instead)
but have since been corrected to match ``splits.py``; this script was never
affected since it reads splits from ``splits.py`` directly rather than from
any ``CORE_STEMS`` list. It still resolves splits from ``splits.py`` and
SKIPS, with a printed warning, any split whose parquet happens to lack the
needed column for a given model/prompt, as a defensive check -- not because
that is expected to trigger today.

Edge cases (same convention as ``score_logprob_windows.py``):

  * empty window (capture failed)               -> row excluded from that
    trial's metrics (like a Null score)
  * a split with < 2 classes, < 10 scored rows, or no reachable 1%-FPR point
    -> excluded from that trial's pooled mean (matches
       ``run_zeroshot_performance.py``'s "reachable" rule)

Run:
    .venv/bin/python epai/ai_detection/score/search_logjudge_params.py --dry-run
    .venv/bin/python epai/ai_detection/score/search_logjudge_params.py
    .venv/bin/python epai/ai_detection/score/search_logjudge_params.py \
        --n-trials 1000 --seed 7
    .venv/bin/python epai/ai_detection/score/search_logjudge_params.py \
        --models gpt56luna --prompts scale --top-k 20

Writes one CSV (default ``results/logjudge_param_search/trials.csv``), one
row per (trial, split) plus a pooled ``split == 'mean'`` row per trial; prints
the top-K trials by pooled AT-HM per (model, prompt) group. Read-only on the
parquets -- writes nothing back into them, and never re-scores ``wlp`` /
``slp`` / ``sbc`` / ``logjudge_t0000`` / ``logjudge_t0001`` (those stay
exactly as ``score_logprob_windows.py`` left them). A trial explored here is
NOT the same as a registered paramid: this script's trials are ephemeral
(fully described by their own CSV row), and only get a permanent
``logjudge_<paramid>`` column + a ``logjudge_params.csv`` registry entry if
someone later decides to promote one.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from epai.ai_detection.score.score_logprob_windows import _answer_probs
from epai.ai_detection.trainv2.core import goodness
from epai.ai_detection.trainv2.splits import DEFAULT as CORE_SPLITS
from epai.ai_detection.trainv2.splits import resolve as resolve_splits
from epai.utils.cd_core import auroc_score, oriented_scores, roc_point_at_fpr

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT = BASE_DIR / "results" / "logjudge_param_search" / "trials.csv"

MODELS = ("gemma4_31b", "gpt56luna")
PROMPTS = ("words", "scale")
LABELS = {"words": ("human", "machine"), "scale": ("A", "B", "C", "D", "E")}
IN_COL_SUFFIX = {"words": "_v3_words", "scale": "_v3_scale"}
GEN_COL = "generated_by"
FPR = 0.01

#: log-uniform range for the threshold readout's eps, sampled per trial.
EPS_LO, EPS_HI = 0.01, 0.30

CSV_FIELDS = [
    "model", "prompt", "trial", "lambda", "eps", "v_B", "v_C", "v_D",
    "split", "n", "n_human", "n_machine", "auroc", "tpr_at_fpr", "reachable",
    "at_hm",
]


@dataclass
class Stratum:
    """One (model, prompt, split)'s precomputed, trial-independent arrays.

    ``p_total`` and ``probs`` don't depend on lambda or v, so they are built
    once per stratum and every trial just re-weights them -- the expensive
    per-row token parsing (``_answer_probs``) happens once, not once per
    trial."""

    split: str
    y: np.ndarray               # 1 = machine, 0 = human
    valid: np.ndarray           # False where the window was empty (Null)
    p_total: np.ndarray         # P(x)
    probs: dict[str, np.ndarray]  # label -> p_l(x), same length as y


def build_stratum(path: Path, split_key: str, model: str, prompt: str,
                   human_label: str) -> Stratum | None:
    """``None`` if this parquet doesn't carry the raw window column for
    ``model``/``prompt`` at all (see the module docstring's KNOWN GAP)."""
    win_col = f"llm_judge_logprobs_{model}{IN_COL_SUFFIX[prompt]}"
    names = pq.read_schema(path).names
    if win_col not in names or GEN_COL not in names:
        return None

    # Read the window column via pyarrow's own to_pylist(), not
    # pd.read_parquet -- pandas materialises a list<struct> column as numpy
    # arrays per cell, and `if not window` in _answer_probs() is ambiguous on
    # an array. to_pylist() gives genuine Python lists of dicts, matching
    # what score_logprob_windows.py's own process_stem() already relies on.
    table = pq.read_table(path, columns=[win_col, GEN_COL])
    windows = table.column(win_col).to_pylist()
    generated_by = table.column(GEN_COL).to_pylist()

    keys = LABELS[prompt]
    n = len(windows)
    probs = {k: np.full(n, np.nan) for k in keys}
    for i, window in enumerate(windows):
        p = _answer_probs(window, prompt)
        if p is None:
            continue  # empty window -- Null, stays NaN
        for k in keys:
            probs[k][i] = p[k]

    p_total = np.sum([probs[k] for k in keys], axis=0)
    valid = np.isfinite(p_total)
    y = np.array([str(g) != human_label for g in generated_by], dtype=int)
    return Stratum(split=split_key, y=y, valid=valid, p_total=p_total, probs=probs)


def sample_trial(rng: np.random.Generator, prompt: str) -> dict:
    """One trial's params: always a lambda choice; a monotonic v ladder only
    under ``scale`` (fixed v=(1,0) under ``words``, so nothing to sample)."""
    lam_family = rng.choice(["contrast", "threshold", "naive"])
    eps = float(10 ** rng.uniform(np.log10(EPS_LO), np.log10(EPS_HI))) \
        if lam_family == "threshold" else None
    if prompt == "words":
        v_map = {"human": 1.0, "machine": 0.0}
        v_bcd = (None, None, None)
    else:
        v_bcd = tuple(float(x) for x in np.sort(rng.uniform(0.0, 1.0, 3))[::-1])
        v_map = {"A": 1.0, "B": v_bcd[0], "C": v_bcd[1], "D": v_bcd[2], "E": 0.0}
    return {"lambda": lam_family, "eps": eps, "v_map": v_map, "v_bcd": v_bcd}


def score_array(stratum: Stratum, v_map: dict[str, float], lam_family: str,
                 eps: float | None) -> np.ndarray:
    """Eq. logjudge_general, vectorised over every row of ``stratum`` at once."""
    p_total = stratum.p_total
    weighted = np.zeros_like(p_total)
    for k, arr in stratum.probs.items():
        weighted = weighted + arr * v_map[k]
    if lam_family == "contrast":
        s = weighted + (1.0 - p_total) * 0.5
    elif lam_family == "threshold":
        with np.errstate(divide="ignore", invalid="ignore"):
            base = weighted / p_total
        s = np.where(p_total >= eps, base, 0.5)
    else:  # naive: lambda(x) = 1 always -- renormalise unconditionally, no
        # shrinkage toward 0.5 at all; the only guard is the P(x) = 0
        # divide-by-zero (a measurement failure, not something to shrink
        # toward). This is "just pick whichever label has more probability."
        with np.errstate(divide="ignore", invalid="ignore"):
            base = weighted / p_total
        s = np.where(p_total > 0.0, base, 0.5)
    return np.clip(s, 0.0, 1.0)


def eval_cell(stratum: Stratum, scores: np.ndarray, fpr: float = FPR) -> dict | None:
    """AUROC + oracle TPR@``fpr`` for one (trial, split); ``None`` if this
    split can't support a reading (too few rows, or a single class)."""
    mask = stratum.valid & np.isfinite(scores)
    y, s = stratum.y[mask], scores[mask]
    if len(y) < 10 or np.unique(y).size < 2:
        return None
    oriented = oriented_scores(s, machine_is_high=False)  # LogJudge is P(human)
    a = auroc_score(y, oriented)
    tpr, thr = roc_point_at_fpr(y, oriented, fpr)
    n_h = int((y == 0).sum())
    return {
        "n": int(len(y)), "n_human": n_h, "n_machine": int((y == 1).sum()),
        "auroc": float(a), "tpr_at_fpr": float(tpr),
        "reachable": bool(n_h * fpr >= 1.0 and np.isfinite(thr)),
    }


def pool_cells(cells: dict[str, dict]) -> dict:
    """The paper's pooled ('mean') row: AUROC meaned over every scored split,
    TPR meaned over only the reachable ones, AT-HM folded from those pooled
    numbers (not averaged per-split) -- mirrors ``run_zeroshot_performance.py``
    / ``plots.goodness_bars``."""
    aurocs = [c["auroc"] for c in cells.values()]
    tprs = [c["tpr_at_fpr"] for c in cells.values() if c["reachable"]]
    pooled_auroc = float(np.mean(aurocs)) if aurocs else float("nan")
    pooled_tpr = float(np.mean(tprs)) if tprs else float("nan")
    at_hm = (goodness(pooled_auroc, pooled_tpr)
             if np.isfinite(pooled_auroc) and np.isfinite(pooled_tpr) else float("nan"))
    return {"auroc": pooled_auroc, "tpr_at_fpr": pooled_tpr, "at_hm": at_hm}


def run_group(model: str, prompt: str, splits: list[tuple[str, Path]],
              human_label: str, n_trials: int, rng: np.random.Generator,
              top_k: int) -> list[dict]:
    strata: dict[str, Stratum] = {}
    for split_key, path in splits:
        if not path.is_file():
            continue
        st = build_stratum(path, split_key, model, prompt, human_label)
        if st is None:
            print(f"  {model} {prompt}: {split_key} has no matching "
                  f"llm_judge_logprobs_{model}{IN_COL_SUFFIX[prompt]} column -- skipped")
            continue
        strata[split_key] = st

    if not strata:
        print(f"  {model} {prompt}: no split has this column -- group skipped")
        return []

    rows: list[dict] = []
    ranked: list[tuple[float, int, dict]] = []
    for t in range(n_trials):
        trial = sample_trial(rng, prompt)
        cells: dict[str, dict] = {}
        for split_key, st in strata.items():
            scores = score_array(st, trial["v_map"], trial["lambda"], trial["eps"])
            cell = eval_cell(st, scores)
            if cell is None:
                continue
            cells[split_key] = cell
            rows.append({
                "model": model, "prompt": prompt, "trial": t,
                "lambda": trial["lambda"], "eps": trial["eps"],
                "v_B": trial["v_bcd"][0], "v_C": trial["v_bcd"][1], "v_D": trial["v_bcd"][2],
                "split": split_key, **cell,
                "at_hm": (goodness(cell["auroc"], cell["tpr_at_fpr"])
                          if cell["reachable"] else float("nan")),
            })
        pooled = pool_cells(cells)
        rows.append({
            "model": model, "prompt": prompt, "trial": t,
            "lambda": trial["lambda"], "eps": trial["eps"],
            "v_B": trial["v_bcd"][0], "v_C": trial["v_bcd"][1], "v_D": trial["v_bcd"][2],
            "split": "mean", "n": "", "n_human": "", "n_machine": "",
            "reachable": "", **pooled,
        })
        ranked.append((pooled["at_hm"], t, trial))
    _print_top(model, prompt, ranked, top_k)
    return rows


def _print_top(model: str, prompt: str, ranked: list[tuple[float, int, dict]], k: int) -> None:
    finite = [r for r in ranked if np.isfinite(r[0])]
    finite.sort(key=lambda r: r[0], reverse=True)
    print(f"\n  top {min(k, len(finite))} of {len(ranked)} trials -- {model} / {prompt}")
    for at_hm, t, trial in finite[:k]:
        v = trial["v_bcd"]
        v_str = "" if v[0] is None else f"  v=(1,{v[0]:.2f},{v[1]:.2f},{v[2]:.2f},0)"
        eps_str = f"  eps={trial['eps']:.3f}" if trial["eps"] is not None else ""
        print(f"    #{t:<4d} AT-HM={at_hm:.3f}  lambda={trial['lambda']}{eps_str}{v_str}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--prompts", nargs="+", choices=PROMPTS, default=list(PROMPTS))
    ap.add_argument("--splits", nargs="+", default=None, metavar="KEY",
                     help=f"default: the paper's four core splits {CORE_SPLITS}")
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true",
                     help="compute and print the top-K trials, write nothing")
    args = ap.parse_args()

    splits = resolve_splits(args.splits or [], PARQUET_DIR)
    print(f"models  : {list(args.models)}")
    print(f"prompts : {list(args.prompts)}")
    print(f"splits  : {[s for s, _ in splits]}")
    print(f"trials  : {args.n_trials} per (model, prompt), seed {args.seed}")

    rng = np.random.default_rng(args.seed)
    all_rows: list[dict] = []
    for model in args.models:
        for prompt in args.prompts:
            all_rows.extend(
                run_group(model, prompt, splits, args.human_label, args.n_trials, rng,
                          args.top_k)
            )

    if not all_rows:
        raise SystemExit("nothing evaluated")

    if args.dry_run:
        print("\n[dry run] -- nothing written")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows, columns=CSV_FIELDS).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

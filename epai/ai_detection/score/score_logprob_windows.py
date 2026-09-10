r"""Turn the captured LLM-judge logprob windows into P(human) score columns.

``capture_judge_logprobs.py`` + ``merge_captured_logprobs.py`` left four
``list<struct{token, logprob}>`` columns on each core parquet -- the ranked
answer-position window, ``window[0]`` the sampled token, ``[]`` a failed row::

    llm_judge_logprobs_<model>_v3_words     answer: human / machine
    llm_judge_logprobs_<model>_v3_scale     answer: A (human) .. E (machine)

This reads a detection score off each window and writes it as a plain scalar
column, so the trainv2 judge comparison picks it up by the ``llm_judge_score_``
prefix like every other judge::

    llm_judge_score_<model>_v3_wlp    from _words   (w-ords l-og p-rob)
    llm_judge_score_<model>_v3_slp    from _scale   (s-cale l-og p-rob)

All scores are **P(human)**: 1.0 human, 0.0 machine, ``machine_is_high=False``.

Formulation -- see ``score_logprob_windows.md`` for the derivation:

  words   p_h, p_m = summed probability of the ``human`` / ``machine`` tokens in
          the window (over casing / leading-space spellings). Score is
              s = (1 + p_h - p_m) / 2
          i.e. the answer distribution's expected human-ness with the leftover
          (non-answer) mass counted as 0.5 -- uninformative. Continuous, always
          in [0, 1], and equal to the renormalised ratio p_h / (p_h + p_m)
          whenever the two answer words hold all the mass (they nearly always
          do here). No window-edge floor, no branching on which words are
          present.

  scale   p_A .. p_E = summed probability of each letter. Renormalise over the
          letters present (q_k = p_k / Σ p_j) and take the expected value
              s = Σ q_k · v(k),   v = (A 1.0, B 0.75, C 0.5, D 0.25, E 0.0)
          The ordered rating spreads a confident judge across neighbouring
          letters, which is the graded signal ``_words`` cannot give over a
          top-k window.

Edge cases, both modes:

  * empty window (capture failed)                         -> Null  (not 0.5)
  * window present but answer mass < MIN_ANSWER_MASS      -> 0.5
    (answered off-format; the sampled token is not an answer token and no
    answer token is anywhere in the window)

    .venv/bin/python epai/ai_detection/score/score_logprob_windows.py --dry-run
    .venv/bin/python epai/ai_detection/score/score_logprob_windows.py
    .venv/bin/python epai/ai_detection/score/score_logprob_windows.py --only manugram_den
    .venv/bin/python epai/ai_detection/score/score_logprob_windows.py --out-dir data/parquet/tmp/scored

Deterministic, no API, one read + write per file (``--out-dir`` to divert).
Refuses to replace an existing score column without ``--overwrite``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from epai.utils.parquet_utils import load_parquet, save_parquet

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
CORE_STEMS = ["ghostbuster_orig", "manugram_gb", "manugram_sci", "manugram_den"]

IN_PREFIX = "llm_judge_logprobs_"
MODE_SUFFIX = {"_v3_words": "words", "_v3_scale": "scale"}
OUT_SUFFIX = {"words": "_v3_wlp", "scale": "_v3_slp"}

# A = human end, E = machine end -- P(human) direction, so A scores 1.0.
SCALE_VALUES = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25, "E": 0.0}
_LETTERS = frozenset(SCALE_VALUES)

# Below this, no answer token carries usable mass: the judge answered
# off-format and there is nothing to read. Score 0.5 (an answered row with no
# evidence), distinct from Null (the window was never captured).
MIN_ANSWER_MASS = 0.05


def _norm(token: str | None, mode: str) -> str:
    """The bare letters of a returned token: ``' Human'`` -> ``'human'``,
    ``'A.'`` -> ``'A'``. Lowercase for words, uppercase for scale -- the same
    fold capture_judge_logprobs.py applied when it picked the answer."""
    s = "".join(ch for ch in (token or "") if ch.isalpha())
    return s.lower() if mode == "words" else s.upper()


def score_window(window, mode: str) -> float | None:
    """P(human) for one window, or None if the window is empty (failed row)."""
    if not window:
        return None

    if mode == "words":
        p_h = p_m = 0.0
        for s in window:
            lp = s.get("logprob")
            if lp is None or not math.isfinite(lp):
                continue
            tok = _norm(s.get("token"), mode)
            if tok == "human":
                p_h += math.exp(lp)
            elif tok == "machine":
                p_m += math.exp(lp)
        if p_h + p_m < MIN_ANSWER_MASS:
            return 0.5
        return float(np.clip((1.0 + p_h - p_m) / 2.0, 0.0, 1.0))

    # scale
    p = dict.fromkeys(_LETTERS, 0.0)
    for s in window:
        lp = s.get("logprob")
        if lp is None or not math.isfinite(lp):
            continue
        tok = _norm(s.get("token"), mode)
        if tok in _LETTERS:
            p[tok] += math.exp(lp)
    total = sum(p.values())
    if total < MIN_ANSWER_MASS:
        return 0.5
    return float(np.clip(
        sum(p[k] / total * SCALE_VALUES[k] for k in _LETTERS), 0.0, 1.0))


def _parse_column(name: str) -> tuple[str, str] | None:
    """``llm_judge_logprobs_gemma4_31b_v3_words`` -> ``('gemma4_31b', 'words')``."""
    if not name.startswith(IN_PREFIX):
        return None
    rest = name[len(IN_PREFIX):]
    for suf, mode in MODE_SUFFIX.items():
        if rest.endswith(suf):
            return rest[: -len(suf)], mode
    return None


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
    todo: list[tuple[str, str, str]] = []  # (in_col, out_col, mode)
    for name in schema.names:
        parsed = _parse_column(name)
        if parsed is None:
            continue
        model, mode = parsed
        out_col = f"llm_judge_score_{model}{OUT_SUFFIX[mode]}"
        if out_col in schema.names and not args.overwrite:
            print(f"  {out_col} already present -- pass --overwrite; skipped")
            continue
        todo.append((name, out_col, mode))

    if not todo:
        print("  nothing to do")
        return

    computed: dict[str, list[float | None]] = {}
    for in_col, out_col, mode in todo:
        windows = pq.read_table(src, columns=[in_col]).column(0).to_pylist()
        scores = [score_window(w, mode) for w in windows]
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
    ap.add_argument("--only", nargs="+", choices=CORE_STEMS, metavar="STEM")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write scored <stem>.parquet here instead of in place")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an llm_judge_score_*_wlp / _slp column that exists")
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

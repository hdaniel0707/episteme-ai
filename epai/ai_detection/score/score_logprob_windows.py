r"""Turn the captured LLM-judge logprob windows into P(human) score columns.

``capture_judge_logprobs.py`` + ``merge_captured_logprobs.py`` left four
``list<struct{token, logprob}>`` columns on each core parquet -- the ranked
answer-position window, ``window[0]`` the sampled token, ``[]`` a failed row::

    llm_judge_logprobs_<model>_v3_words     answer: human / machine
    llm_judge_logprobs_<model>_v3_scale     answer: A (human) .. E (machine)

This reads a detection score off each window and writes it as a plain scalar
column, so the trainv2 judge comparison picks it up by the ``llm_judge_score_``
prefix like every other judge::

    llm_judge_score_<model>_v3_wlp             from _words   words, contrast
    llm_judge_score_<model>_v3_slp             from _scale   scale, expected value
    llm_judge_score_<model>_v3_sbc             from _scale   scale, contrast
    llm_judge_score_<model>_v3_logjudge_t0000  from _words   paper's ord2
    llm_judge_score_<model>_v3_logjudge_t0001  from _scale   paper's ord5

All scores are **P(human)**: 1.0 human, 0.0 machine, ``machine_is_high=False``.

The ``logjudge_<paramid>`` suffix is the naming convention for every column
that instantiates the paper's general LogJudge formula (Section
``method_logjudge`` / Appendix ``logjudge_general``): a sequential id
(``t0000``, ``t0001``, ...) assigned once, permanently, the moment a
particular (prompt, lambda, v) configuration becomes a real column -- never
reused, never renumbered. ``epai/ai_detection/score/logjudge_params.csv`` is
the registry: one row per paramid, decoding it back to its exact prompt,
lambda family, floor (if any), and value function. ``t0000``/``t0001`` are the
paper's two reported variants, ``ord2`` and ``ord5`` (see the registry's
``note`` column) -- both instances of the general contrast readout
s(x) = sum_l p_l(x)*v(l) + (1-P(x))*0.5, with lambda(x)=P(x), differing only
in the answer scheme's label set L. A search over further (lambda, v)
choices (``search_logjudge_params.py``) only assigns a paramid -- and only
writes a real column -- for a trial someone decides to promote; exploring a
trial does not by itself spend an id. ``wlp`` / ``slp`` / ``sbc`` predate this
convention and are kept unchanged for continuity with earlier results.

Formulation -- see ``score_logprob_windows.md`` for the derivation:

  wlp   p_h, p_m = summed probability of the ``human`` / ``machine`` tokens in
        the window (over casing / leading-space spellings). Score is
            s = (1 + p_h - p_m) / 2
        i.e. the answer distribution's expected human-ness with the leftover
        (non-answer) mass counted as 0.5 -- uninformative. Continuous, always
        in [0, 1], and equal to the renormalised ratio p_h / (p_h + p_m)
        whenever the two answer words hold all the mass (they nearly always
        do here). No window-edge floor, no branching on which words are present.

  slp   p_A .. p_E = summed probability of each letter. Renormalise over the
        letters present (q_k = p_k / Σ p_j) and take the expected value
            s = Σ q_k · v(k),   v = (A 1.0, B 0.75, C 0.5, D 0.25, E 0.0)
        The ordered rating was meant to spread a confident judge across
        neighbouring letters; in practice the fixed ruler + first-letter bias
        cost ranking vs ``wlp`` (see the .md), so ``sbc`` is the alternative.

  sbc   score the scale prompt with the ``wlp`` formula: {A, B} vs {D, E},
            s = (1 + (p_A + p_B) - (p_D + p_E)) / 2
        C and non-letter mass -> 0.5. Drops the ordinal ruler entirely. Same
        functional form as ``wlp``, so ``sbc`` vs ``wlp`` isolates the prompt
        (5-level rating vs binary) from the scoring.

  t0000 (ord2)  identical to ``wlp`` -- the paper's binary instance
        (L={human, machine}, v=(1,0)) of the same general contrast formula.
        A separate column only so the paper's variant names resolve
        directly to a parquet column.

  t0001 (ord5)  the paper's ordinal instance (L={A,B,C,D,E},
        v=(1,.75,.5,.25,0)) of the *same* contrast formula as
        ``wlp``/``t0000``, applied to all five letters without
        renormalising or collapsing any of them:
            s = p_A*1 + p_B*.75 + p_C*.5 + p_D*.25 + p_E*0 + (1 - P)*0.5,
            P = p_A + p_B + p_C + p_D + p_E
        Unlike ``slp`` this never divides by P (no renormalisation, no
        clumping at exact letter values); unlike ``sbc`` it keeps all five
        letters and their ordinal weights instead of collapsing to a binary
        split. Together, ``t0000`` vs ``t0001`` isolates the answer scheme's
        resolution while holding the readout fixed -- see the paper.

Edge cases, both modes:

  * empty window (capture failed)                         -> Null  (not 0.5)
  * window present but answer mass < MIN_ANSWER_MASS      -> 0.5
    (answered off-format; the sampled token is not an answer token and no
    answer token is anywhere in the window)

    .venv/bin/python epai/ai_detection/score/score_logprob_windows.py --dry-run
    .venv/bin/python epai/ai_detection/score/score_logprob_windows.py
    .venv/bin/python epai/ai_detection/score/score_logprob_windows.py --only manugram_aiscientist
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
CORE_STEMS = ["ghostbuster_orig", "manugram_gb", "manugram_sci", "manugram_aiscientist"]

IN_PREFIX = "llm_judge_logprobs_"
MODE_SUFFIX = {"_v3_words": "words", "_v3_scale": "scale"}

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


def _answer_probs(window, mode: str) -> dict[str, float] | None:
    """Summed probability per answer token in the window. ``words`` ->
    ``{'human': p, 'machine': p}``; ``scale`` -> ``{'A': p, ..., 'E': p}``.
    Non-finite logprobs skipped. ``None`` for an empty window (failed row)."""
    if not window:
        return None
    keys = ("human", "machine") if mode == "words" else _LETTERS
    p = dict.fromkeys(keys, 0.0)
    for s in window:
        lp = s.get("logprob")
        if lp is None or not math.isfinite(lp):
            continue
        tok = _norm(s.get("token"), mode)
        if tok in p:
            p[tok] += math.exp(lp)
    return p


def _contrast(p_h: float, p_m: float) -> float:
    """P(human) from a two-outcome probability split, leftover mass = 0.5
    (uninformative). Continuous, bounded, no renormalisation floor."""
    if p_h + p_m < MIN_ANSWER_MASS:
        return 0.5
    return float(np.clip((1.0 + p_h - p_m) / 2.0, 0.0, 1.0))


def score_words(window) -> float | None:
    """``wlp``: P(human) from the ``human`` / ``machine`` answer tokens."""
    p = _answer_probs(window, "words")
    return None if p is None else _contrast(p["human"], p["machine"])


def score_scale_ev(window) -> float | None:
    """``slp``: expected value of the A..E rating, renormalised over the
    letters present, against the fixed scale ``SCALE_VALUES``."""
    p = _answer_probs(window, "scale")
    if p is None:
        return None
    total = sum(p.values())
    if total < MIN_ANSWER_MASS:
        return 0.5
    return float(np.clip(
        sum(p[k] / total * SCALE_VALUES[k] for k in _LETTERS), 0.0, 1.0))


def score_scale_contrast(window) -> float | None:
    """``sbc``: score the scale prompt like the words prompt -- {A, B} vs
    {D, E}, dropping the ordinal ruler. C and any non-letter mass count as 0.5
    (uncertain). Same functional form as ``wlp``, so a ``sbc`` vs ``wlp`` gap
    is purely the prompt (5-level rating vs binary), not the scoring."""
    p = _answer_probs(window, "scale")
    if p is None:
        return None
    return _contrast(p["A"] + p["B"], p["D"] + p["E"])


def score_scale_contrast_full(window) -> float | None:
    """``t0001`` (paper's ``ord5``): the general contrast readout
    (Eq. logjudge_contrast, lambda(x)=P(x)) applied to the full A..E ladder --
    the un-renormalised weighted sum of all five letters' raw mass plus a
    residual*0.5 term, same functional form as ``wlp``/``t0000`` and ``sbc``,
    just without collapsing the ladder to a binary split first. Paired with
    ``t0000`` (=``wlp``) under the same readout, it isolates the answer
    scheme's resolution alone. See ``logjudge_params.csv`` for the registry
    entry."""
    p = _answer_probs(window, "scale")
    if p is None:
        return None
    total = sum(p.values())
    if total < MIN_ANSWER_MASS:
        return 0.5
    s = sum(p[k] * SCALE_VALUES[k] for k in _LETTERS) + (1.0 - total) * 0.5
    return float(np.clip(s, 0.0, 1.0))


#: (input mode, output column suffix, scorer). A ``_scale`` window feeds three
#: output columns; a ``_words`` window feeds two. ``_v3_logjudge_<paramid>``
#: suffixes are registered in ``logjudge_params.csv`` -- add a row there
#: before adding a new one here.
SCORERS: tuple[tuple[str, str, "callable"], ...] = (
    ("words", "_v3_wlp", score_words),
    ("words", "_v3_logjudge_t0000", score_words),
    ("scale", "_v3_slp", score_scale_ev),
    ("scale", "_v3_sbc", score_scale_contrast),
    ("scale", "_v3_logjudge_t0001", score_scale_contrast_full),
)


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
    todo: list[tuple[str, str, "callable"]] = []  # (in_col, out_col, scorer)
    for name in schema.names:
        parsed = _parse_column(name)
        if parsed is None:
            continue
        model, mode = parsed
        for smode, suffix, fn in SCORERS:
            if smode != mode:
                continue
            out_col = f"llm_judge_score_{model}{suffix}"
            if out_col in schema.names and not args.overwrite:
                print(f"  {out_col} already present -- pass --overwrite; skipped")
                continue
            todo.append((name, out_col, fn))

    if not todo:
        print("  nothing to do")
        return

    windows: dict[str, list] = {}
    computed: dict[str, list[float | None]] = {}
    for in_col, out_col, fn in todo:
        if in_col not in windows:
            windows[in_col] = pq.read_table(src, columns=[in_col]).column(0).to_pylist()
        scores = [fn(w) for w in windows[in_col]]
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
                    help="replace an llm_judge_score_*_{wlp,slp,sbc,"
                         "logjudge_t0000,logjudge_t0001} column that exists")
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

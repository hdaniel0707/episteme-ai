r"""Readouts and metrics shared by ``check.py`` and ``evaluate.py``: one score per
sidecar row, recomputed from the stored answer and token windows (never from the
quick ``score`` column), plus the paper's metrics.

Readouts (all P(human), high = human):

  parsed   ``prob`` conditions: the first number in [0, 1] in the answer (BaseJudge).
  ev       ``prob`` conditions, A2: the expected value of the written number under
           the token distribution of the same call. At every token position that
           writes fractional digits, the digit-only alternatives in the top window
           are renormalised and their digit strings averaged, each conditioned on
           the greedy prefix:  ev = int part + sum_i E[digits_i | greedy prefix_i].
           luna writes ``0`` ``.`` + one 2-3-digit chunk (+ one digit for hp4), so
           this is the chunk average; gemma writes one digit per token, so it is
           sum_k 10^-k E[d_k | greedy prefix].
  mass     the probability mass of those digit alternatives at the FIRST
           fractional position (how much of the distribution ``ev`` sees).
  lj       ``label`` conditions: the paper's LogJudge contrast at position 0,
           (1 + p_human - p_machine) / 2, leftover mass = 1/2, < 5% label mass -> 1/2
           (``score_logprob_windows._contrast``), with the condition's label fold.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from epai.ai_detection.rebuttal.conditions import CONDITIONS, HUMAN
from epai.ai_detection.score.score_logprob_windows import MIN_ANSWER_MASS, _contrast, _norm
from epai.ai_detection.trainv2.core import goodness
from epai.utils.cd_core import roc_point_at_fpr

FPR = 0.01
SPLIT_NAMES = {"ghostbuster_orig": "GB-orig", "manugram_gb": "MG-GB",
               "manugram_sci": "MG-Sci", "manugram_aiscientist": "MG-AI-Sci"}

# a written probability: "0.18", ".18", "1", "0", "1.0"
_NUM = re.compile(r"(?<![\w.])(?:[01](?:\.\d+)?|\.\d+)")
_BARE = re.compile(r"^\s*(?:[01](?:\.\d+)?|\.\d+)\s*$")


# ---------------------------------------------------------------------------
# per-row readouts
# ---------------------------------------------------------------------------


def parse_prob(answer) -> float | None:
    """The first number in [0, 1] in the answer (same rule as capture.quick_score)."""
    for m in _NUM.finditer(answer or ""):
        v = float(m.group())
        if 0.0 <= v <= 1.0:
            return v
    return None


def is_bare_number(answer) -> bool:
    return bool(_BARE.match(answer or ""))


def n_decimals(answer) -> int | None:
    m = _NUM.search(answer or "")
    if not m:
        return None
    s = m.group()
    return len(s.split(".", 1)[1]) if "." in s else 0


def expected_value(tokens) -> tuple[float | None, float | None]:
    """``(ev, mass)`` of the written number (see the module docstring). ``ev`` is
    the parsed value when the number has no fractional part; ``(None, None)`` when
    the greedy text holds no number."""
    if tokens is None or len(tokens) == 0:
        return None, None
    greedy = [t["token"] for t in tokens]
    text = "".join(greedy)
    m = _NUM.search(text)
    if not m:
        return None, None
    a, b = m.span()
    d = text.find(".", a, b)
    if d < 0:
        return float(m.group()), None
    ev = float(text[a:d]) if d > a else 0.0
    first_mass = None
    s = 0
    for tok, g in zip(tokens, greedy):
        e = s + len(g)
        if e > d + 1 and s < b:                   # this token writes fractional digits
            off = max(0, s - (d + 1))             # fractional digits before it
            head = text[s:d + 1] if s <= d else ""  # e.g. "0." when the token is "0.18"
            num = den = 0.0
            for alt in tok["top"]:
                t = alt["token"]
                if not t.startswith(head):
                    continue
                digits = t[len(head):]
                if not digits.isdigit():
                    continue
                p = math.exp(alt["logprob"])
                num += p * int(digits) / 10 ** (off + len(digits))
                den += p
            if first_mass is None:
                first_mass = den
            if den > 0:
                ev += num / den
            else:                                 # nothing usable: keep the greedy digits
                frac = text[max(s, d + 1):min(e, b)]
                ev += int(frac) / 10 ** (off + len(frac)) if frac.isdigit() else 0.0
        s = e
        if s >= b:
            break
    return float(np.clip(ev, 0.0, 1.0)), first_mass


def label_probs(cond_name: str, tokens) -> tuple[float, float] | None:
    """``(p_human, p_machine)`` at answer position 0 under the condition's fold."""
    if tokens is None or len(tokens) == 0:
        return None
    labels = CONDITIONS[cond_name].labels
    p = {HUMAN: 0.0, "machine": 0.0}
    for alt in tokens[0]["top"]:
        lp = alt["logprob"]
        if lp is None or not math.isfinite(lp):
            continue
        lab = labels.get(_norm(alt["token"], "words"))
        if lab:
            p[lab] += math.exp(lp)
    return p[HUMAN], p["machine"]


def label_score(cond_name: str, tokens) -> float | None:
    pr = label_probs(cond_name, tokens)
    return None if pr is None else _contrast(*pr)


def first_token_is_label(cond_name: str, tokens) -> bool:
    if tokens is None or len(tokens) == 0:
        return False
    return _norm(tokens[0]["token"], "words") in CONDITIONS[cond_name].labels


def add_readouts(df: pd.DataFrame, cond_name: str) -> pd.DataFrame:
    """Sidecar rows -> the same rows with ``parsed``/``ev``/``mass`` (prob) or
    ``lj``/``label_mass`` (label) columns."""
    df = df.copy()
    if CONDITIONS[cond_name].kind == "prob":
        df["parsed"] = [parse_prob(a) for a in df["answer"]]
        evm = [expected_value(t) for t in df["tokens"]]
        df["ev"] = [x[0] for x in evm]
        df["mass"] = [x[1] for x in evm]
    else:
        pr = [label_probs(cond_name, t) for t in df["tokens"]]
        df["lj"] = [None if x is None else _contrast(*x) for x in pr]
        df["label_mass"] = [None if x is None else x[0] + x[1] for x in pr]
    for c in ("parsed", "ev", "mass", "lj", "label_mass"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# metrics (the paper's: oracle 1%-FPR point, AT-HM, mean over the four splits)
# ---------------------------------------------------------------------------


def tie_stats(y: np.ndarray, s: np.ndarray, level: float = FPR) -> dict:
    """Ties at the operating point of oriented score ``s`` (high = machine); the
    same definition as the release repo's ``compute/tie_table.py``."""
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), s.size - 1]
    tp, fp = np.cumsum(y)[last], (last + 1) - np.cumsum(y)[last]
    n_h, n_m = int((y == 0).sum()), int((y == 1).sum())
    fpr, tpr = np.r_[0.0, fp / n_h], np.r_[0.0, tp / n_m]
    i = int(np.flatnonzero(fpr <= level + 1e-12)[-1])
    j = min(i + 1, fpr.size - 1)
    grp_h = np.diff(np.r_[0, fp])
    grp_m = np.diff(np.r_[0, tp])
    _, counts = np.unique(s, return_counts=True)
    return {
        "n_distinct": int(np.unique(s).size),
        "n_distinct_human": int(np.unique(s[y == 0]).size),
        "share_tied_top": float(counts.max() / s.size),
        "fpr_below": float(fpr[i]), "tpr_below": float(tpr[i]),
        "fpr_above": float(fpr[j]), "tpr_above": float(tpr[j]),
        "tie_humans": int(grp_h[j - 1]) if j > i else 0,
        "tie_machines": int(grp_m[j - 1]) if j > i else 0,
    }


def split_metrics(y: np.ndarray, p_human: np.ndarray) -> dict:
    """AUROC, oracle TPR@1%FPR (largest achievable FPR <= 1%), AT-HM and ties."""
    from sklearn.metrics import roc_auc_score
    s = -np.asarray(p_human, dtype=float)          # oriented: high = machine
    y = np.asarray(y, dtype=int)
    auroc = float(roc_auc_score(y, s))
    tpr, thr = roc_point_at_fpr(y, s, FPR)
    return {"n": int(y.size), "n_human": int((y == 0).sum()), "n_machine": int((y == 1).sum()),
            "auroc": auroc, "tpr": tpr, "athm": goodness(auroc, tpr), "threshold": thr,
            **tie_stats(y, s)}


def mean_row(rows: list[dict]) -> dict:
    """The paper's four-split mean: AUROC and TPR averaged, AT-HM of the means."""
    au = float(np.mean([r["auroc"] for r in rows]))
    tp = float(np.mean([r["tpr"] for r in rows]))
    return {"auroc": au, "tpr": tp, "athm": goodness(au, tp)}

"""The primitives every trainv2 experiment shares: the split, the two metrics,
the variant grid, and seed aggregation.

Deliberately tiny and dependency-free (numpy + sklearn only). The metric
conventions match the old ``epai/utils/cd_core`` so a trainv2 number can be laid
beside a legacy one:

  * AUROC with machine (``y == 1``) positive, NOT orientation-corrected — a
    score that runs backwards reads below 0.5 and should.
  * TPR@FPR is the *achievable* point at the largest FPR <= the level (the ROC
    grid only steps at an observed human score), i.e. a threshold you could
    actually set, not an interpolated one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

from epai.algs import build_algorithm

POOLED = "all"  #: domain-slice name for "every domain at once"


# ---------------------------------------------------------------------------
# The variant grid: embedding x algorithm
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One cell of the grid: one embedding column, one algorithm.

    ``key`` names the results-directory / figure-panel slot; ``label`` is what a
    reader sees ("luar_crud · logistic").
    """

    embedding: str
    alg_name: str
    alg_label: str = ""
    alg_params: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.embedding}__{self.alg_name}"

    @property
    def label(self) -> str:
        emb = self.embedding[4:] if self.embedding.startswith("emb_") else self.embedding
        return f"{emb} · {self.alg_label or self.alg_name}"


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def grouped_split(
    y: np.ndarray, groups: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """One grouped train/test split. A group never straddles the cut.

    ``groups`` is the group id per row -- here always the document id, so a
    human text and every rewrite of it (which share that id) move together.
    Returns ``(train_idx, test_idx)`` into the arrays passed in. The splitter
    only reads the row count, so a dummy stands in for the design matrix. The
    assert makes the group-disjointness explicit (and catches a caller that
    passed row indices instead of group ids).
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros((len(y), 1)), y, groups))
    assert set(groups[train_idx]).isdisjoint(groups[test_idx]), \
        "grouped_split: a group ended up on both sides of the cut"
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# The two metrics
# ---------------------------------------------------------------------------


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """AUROC, machine positive. NaN on a single-class set. Not flipped."""
    y = np.asarray(y)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def tpr_at_fpr(y: np.ndarray, scores: np.ndarray, level: float) -> float:
    """Oracle TPR at the largest achievable FPR <= ``level``.

    The threshold is fit on these very rows — an upper bound on what the
    operating point delivers, and the only reading that needs no calibration
    slice. NaN on a single-class set.
    """
    y = np.asarray(y)
    if np.unique(y).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, np.asarray(scores, dtype=float))
    idx = int(np.flatnonzero(fpr <= level)[-1])  # fpr starts at 0, never empty
    return float(tpr[idx])


#: A split whose 1%-FPR point is unreachable (< 100 human rows) doesn't get a
#: 1%-FPR reading -- the ROC grid steps by 1/n_human. `eff_fpr_level` loosens
#: the target to the smallest FPR the grid CAN hit (one human flagged), so
#: every split yields a real operating-point TPR; capped so a tiny split
#: doesn't loosen without limit.
_EFF_FPR_CAP = 0.05


def eff_fpr_level(n_human: int, target: float) -> float:
    """``target`` where the FPR grid can reach it, else ``1/n_human`` (one human
    flagged), capped at ``_EFF_FPR_CAP``. Use this instead of ``target`` when a
    reading is wanted on every split (e.g. the goodness-score figure); the
    caller should record which level was actually used."""
    if n_human <= 0:
        return target
    return min(_EFF_FPR_CAP, max(target, 1.0 / n_human + 1e-9))


def fpr_tag(fpr: float) -> str:
    """``0.01 -> '1pct'``, ``0.05 -> '5pct'`` -- the filename suffix that keeps
    a run's per-FPR figure sets apart."""
    return f"{fpr * 100:g}pct"


def rows_at_fpr(rows: list[dict], fpr: float) -> list[dict]:
    """The CSV records for one FPR level. Raises a clear error (rather than a
    ``KeyError`` deep in a plot) on a ``performance.csv`` written before ``--fpr``
    became a list -- those have no ``fpr`` column and must be regenerated."""
    if rows and "fpr" not in rows[0]:
        raise SystemExit(
            "this performance.csv predates the multi-FPR change (no 'fpr' "
            "column) — re-run the script without --replot to regenerate it")
    return [r for r in rows if float(r["fpr"]) == fpr]


def goodness(auroc_val: float, tpr: float, how: str = "hmean") -> float:
    """Fold ranking + operating point into ``[0, 1]``. ``G = 2*AUROC - 1``
    (Gini, clipped at 0), ``T = TPR`` (clipped at 0), combined by ``how``
    (``hmean`` weakest-link default, ``gmean``, or ``amean``).

    Shared by the goodness-bar figure (:func:`epai.ai_detection.trainv2.plots.
    goodness_bars`) and the trained runners, which compute it per seed / fold so
    the bar can carry a real SD."""
    g = max(0.0, 2.0 * float(auroc_val) - 1.0)
    t = max(0.0, float(tpr))
    if how == "amean":
        return (g + t) / 2.0
    if how == "gmean":
        return (g * t) ** 0.5
    return 2.0 * g * t / (g + t) if (g + t) > 0 else 0.0


# ---------------------------------------------------------------------------
# Fit + evaluate one variant, over every seed — shared by every runner
# ---------------------------------------------------------------------------

#: The two metrics every experiment reports, in the order they're computed.
METRIC_KEYS = ("auroc", "tpr_at_fpr")


def score_alg(alg, X: np.ndarray) -> np.ndarray | None:
    """The continuous machine-ness score, or None for a label-only algorithm."""
    try:
        return alg.decision_score(X)
    except NotImplementedError:
        return None


def train_eval(
    variant: Variant,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seeds: list[int],
    test_size: float,
    fpr: float,
) -> list[dict]:
    """Fit ``variant`` and score it on the held-out split, once per seed.

    One grouped train/test split per seed (never a calibration slice — both
    metrics are oracle, fit on the evaluation rows themselves). Returns one row
    per seed with the split sizes and the two metrics.
    """
    rows: list[dict] = []
    for seed in seeds:
        train_idx, test_idx = grouped_split(y, groups, test_size, seed)
        alg = build_algorithm(variant.alg_name, seed, **variant.alg_params)
        alg.set_fit_groups(groups[train_idx])
        alg.fit(X[train_idx], y[train_idx])

        scores = score_alg(alg, X[test_idx])
        y_test = y[test_idx]
        rows.append(
            {
                "seed": seed,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_test_human": int((y_test == 0).sum()),
                "n_groups": int(len(set(groups))),
                "auroc": auroc(y_test, scores) if scores is not None else float("nan"),
                "tpr_at_fpr": (
                    tpr_at_fpr(y_test, scores, fpr) if scores is not None else float("nan")
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Seed aggregation
# ---------------------------------------------------------------------------


def aggregate_seeds(per_seed: list[dict], keys: tuple[str, ...]) -> dict:
    """``{key}_mean`` / ``{key}_std`` over seeds, NaN-tolerant.

    A single degenerate seed (single-class test half) is averaged out with
    ``nanmean`` rather than erasing the others. Constant fields (row counts) are
    taken from the first seed.
    """
    out: dict = {"n_seeds": len(per_seed)}
    for key in keys:
        vals = np.asarray([r[key] for r in per_seed], dtype=float)
        if np.isfinite(vals).any():
            out[f"{key}_mean"] = float(np.nanmean(vals))
            out[f"{key}_std"] = float(np.nanstd(vals))
        else:
            out[f"{key}_mean"] = float("nan")
            out[f"{key}_std"] = float("nan")
    for const in ("n_train", "n_test", "n_test_human", "n_groups"):
        if const in per_seed[0]:
            out[const] = per_seed[0][const]
    return out

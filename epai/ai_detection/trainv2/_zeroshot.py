"""Zero-shot score columns evaluated on the SAME rows as the trained detectors.

``run_zeroshot_performance.py`` (experiment D) scores each column on a full
MANUGRAM split. The trained-detector runners
(``run_in_domain_performance.py`` / ``run_lodo_performance.py`` and their
rephrase siblings) instead overlay the ``best_zeroshot.yaml`` picks *on the
trained models' own evaluation rows* -- the per-seed 70/30 grouped test
partition (in-domain) or the held-out domain fold (LODO) -- so the two detector
kinds are compared on identical data and the zero-shot points pick up the same
seed / fold SD.

The eval-only OOD corpus (MG-Den) is the exception: there the trained models
score the full corpus, so the zero-shot picks do too -- a single deterministic
reading, no error bar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from epai.utils.cd_core import oriented_scores

from epai.ai_detection.trainv2.core import auroc, eff_fpr_level, goodness, tpr_at_fpr
from epai.ai_detection.trainv2.run_best_zeroshot_performance import _detector, _load_columns
from epai.ai_detection.trainv2.run_zeroshot_performance import FAMILIES


def resolve_specs(available, config_path: Path, *,
                  columns: list[str] | None = None) -> list[tuple[str, str, bool]]:
    """``[(column, detector_label, machine_is_high)]`` for every requested
    column present in ``available`` (a set/iterable of parquet column names), in
    order. Requested columns are ``columns`` if given, else the
    ``best_zeroshot.yaml`` list at ``config_path``. A column absent from this
    parquet is silently skipped -- some judges are not scored on every split."""
    have = set(available)
    out: list[tuple[str, str, bool]] = []
    for col in (columns if columns is not None else _load_columns(config_path)):
        for fam, spec in FAMILIES.items():
            if col.startswith(spec["prefix"]):
                if col in have:
                    out.append((col, _detector(fam, spec["label"](col)),
                                spec["machine_is_high"]))
                break
    return out


def orient(raw: np.ndarray, machine_is_high: bool) -> np.ndarray:
    """A raw score column oriented high = machine, NaNs preserved."""
    return oriented_scores(np.asarray(raw, dtype=float), machine_is_high)


def eval_column(oriented: np.ndarray, y: np.ndarray,
                fpr: float) -> tuple[float, float, float]:
    """``(auroc, tpr@fpr, tpr@eff_fpr)`` for one oriented column on the rows
    passed in. Rows with a non-finite score are dropped (per-column, like
    ``run_zeroshot_performance``); returns all-NaN if < 2 classes survive."""
    oriented = np.asarray(oriented, dtype=float)
    y = np.asarray(y)
    m = np.isfinite(oriented)
    oriented, y = oriented[m], y[m]
    if np.unique(y).size < 2:
        return float("nan"), float("nan"), float("nan")
    n_h = int((y == 0).sum())
    return (auroc(y, oriented),
            tpr_at_fpr(y, oriented, fpr),
            tpr_at_fpr(y, oriented, eff_fpr_level(n_h, fpr)))


def _nanmean(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def _nanstd(v) -> float:
    v = [x for x in np.asarray(v, dtype=float) if np.isfinite(x)]
    return float(np.std(v)) if len(v) > 1 else 0.0


def agg_cell(aurocs, tprs, tpr_effs, combine: str) -> dict:
    """Per-replicate (seed or fold) metric lists -> one cell dict with the mean
    and SD of each metric, plus the goodness computed **per replicate** then
    averaged, so the goodness bar can carry a real SD."""
    goods = [goodness(a, te, combine) for a, te in zip(aurocs, tpr_effs)
             if np.isfinite(a) and np.isfinite(te)]
    return {
        "auroc": _nanmean(aurocs), "auroc_std": _nanstd(aurocs),
        "tpr_at_fpr": _nanmean(tprs), "tpr_at_fpr_std": _nanstd(tprs),
        "tpr_eff": _nanmean(tpr_effs), "tpr_eff_std": _nanstd(tpr_effs),
        "goodness": float(np.mean(goods)) if goods else float("nan"),
        "goodness_std": float(np.std(goods)) if len(goods) > 1 else 0.0,
    }


def overall_cell(cells: list[dict], fpr: float) -> dict:
    """Pool per-corpus zero-shot cells into the 'mean' row: each metric is the
    mean over corpora, and its SD is the mean of the finite per-corpus SDs (the
    'typical per-corpus SD' -- same convention as the trained overall row). The
    1%-FPR TPR mean skips a corpus whose point is unreachable."""
    ok = [c for c in cells if c.get("reachable", True)]

    def m(k: str, src=cells) -> float:
        return _nanmean([c[k] for c in src])

    def msd(k: str, src=cells) -> float:
        vals = [c[k] for c in src if np.isfinite(c.get(k, float("nan")))]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "auroc": m("auroc"), "auroc_std": msd("auroc_std"),
        "tpr_at_fpr": m("tpr_at_fpr", ok or cells),
        "tpr_at_fpr_std": msd("tpr_at_fpr_std", ok or cells),
        "tpr_eff": m("tpr_eff"), "tpr_eff_std": msd("tpr_eff_std"),
        "goodness": m("goodness"), "goodness_std": msd("goodness_std"),
        "eff_fpr": fpr,
    }

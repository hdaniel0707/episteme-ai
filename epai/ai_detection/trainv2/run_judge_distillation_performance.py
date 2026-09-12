r"""Pilot: distilling the ``gpt56luna`` judge's continuous score into an
``emb_roberta`` linear probe, instead of training on the GT label -- does a
soft judge target generalize to MG-AI-Sci better than GT does?

Motivating numbers (``results/trainv2/in_domain_performance/performance.csv``,
2026-09-12): on MG-AI-Sci, ``trained/roberta·logistic`` (fit on GT) reaches
AUROC 0.903 / oracle TPR@1%FPR 0.315, while the zero-shot
``llm_judge_score_gpt56luna_v3_wlp`` column itself reaches 0.958 / 0.56 -- no
trained detector, including the multi-embedding architectures in
``embedding_ideas.md``, closes that gap. This script asks whether regressing
``emb_roberta`` onto the judge's own continuous score (instead of / alongside
the binary GT label) buys any of it back, or whether the gap is a
representation problem the training target can't fix.

**First result (in-domain 70/30, 2026-09-13): the hybrid arm is the best of
the new arms**, so this run also (a) sweeps ``judge_distill_hybrid``'s own
hyperparameters and (b) adds the LODO regime, to see whether the improvement
holds under domain shift and not just a random split.

An **opt-in add-on**, run by hand, same treatment as ``run_dr_transfer.py`` /
``run_robust_performance.py``: absent from ``rebuild_all.TASKS``; ``_trained.py``
/ ``core.py`` / ``splits.py`` / the two main runners / ``rebuild_all.py`` are
only ever imported from.

Three "core" new arms. Two (:mod:`epai.algs.judge_distill`) are
auxiliary-column specs reading ``llm_judge_score_gpt56luna_v3_wlp`` as ``Z``
instead of the usual judge cue vector (``_trained.load_features`` /
``run_lodo_performance._load``'s ``aux_col`` parameter):

    judge_distill_regress   pure distillation -- Ridge on emb_roberta -> judge
                             score, GT label never seen
    judge_distill_hybrid    hybrid -- linear probe, CE(y) + lambda*MSE(judge),
                             now early-stopped on validation AUROC (``val_frac``
                             / ``patience`` -- see ``epai/algs/judge_distill.py``)

The third, ``roberta_logistic_balanced``, answers a separate question that
came up alongside this one -- class imbalance (MG-GB/MG-Sci are 4:1
machine:human). Defined once, in ``run_class_weight_performance.py`` (its own
standalone pilot); imported here, not redefined.

``HYBRID_SWEEP_SPECS`` -- since the hybrid arm won, a small sweep of its own
hyperparameters, each a named arm so every point lands in the same CSV/figure
as everything else:

    judge_distill_hybrid_lam{0.1,0.25,1,2}   lambda_distill away from the
                                              default 0.5 (how much weight the
                                              soft judge target gets vs. GT)
    judge_distill_hybrid_balanced            default lambda + class_weight
                                              balanced together -- the two
                                              levers combined

``--no-sweep`` drops these 5 extra arms (cheaper run, core comparison only).

Two regimes, like ``run_in_domain_performance.py`` / ``run_lodo_performance.py``
(and their ``run_robust_performance.py`` combination): in-domain 70/30 (own
CSV/figures) and leave-one-domain-out (own CSV/figures), both over the same
``ALL_SPECS``. ``--regimes`` picks one or both (default both).

compute -> CSV -> plot from the CSV, one regime = one CSV = one figure set:

    results/trainv2/judge_distillation_performance/
        in_domain_performance.csv, in_domain_{1pct,5pct}.png (+ _goodness_, _denario_)
        lodo_performance.csv, lodo_{1pct,5pct}.png (+ _goodness_, _denario_)

Run (sandbox has no ``uv`` -- use ``.venv/bin/python`` directly):

    .venv/bin/python epai/ai_detection/trainv2/run_judge_distillation_performance.py
    .venv/bin/python epai/ai_detection/trainv2/run_judge_distillation_performance.py --regimes lodo --seeds 0 1
    .venv/bin/python epai/ai_detection/trainv2/run_judge_distillation_performance.py --no-sweep
    .venv/bin/python epai/ai_detection/trainv2/run_judge_distillation_performance.py \
        --replot results/trainv2/judge_distillation_performance
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from epai.ai_detection.trainv2._trained import TRAINED_SPECS, eval_on_fprs, fit, load_features
from epai.ai_detection.trainv2.core import eff_fpr_level, fpr_tag, goodness, grouped_split, rows_at_fpr
from epai.ai_detection.trainv2.run_best_zeroshot_performance import DEFAULT_CONFIG as ZEROSHOT_CONFIG
from epai.ai_detection.trainv2.run_class_weight_performance import BALANCED_SPECS
from epai.ai_detection.trainv2.run_in_domain_performance import (
    CSV_FIELDS as CSV_FIELDS_IN_DOMAIN,
)
from epai.ai_detection.trainv2.run_in_domain_performance import (
    _overall_cell,
    _print_table as _print_table_in_domain,
    _row,
)
from epai.ai_detection.trainv2.run_in_domain_performance import (
    _std,
    compute_zeroshot as compute_zeroshot_in_domain,
)
from epai.ai_detection.trainv2.run_lodo_performance import CSV_FIELDS as CSV_FIELDS_LODO
from epai.ai_detection.trainv2.run_lodo_performance import (
    _fold,
    _load,
    _load_ood,
    _print_table as _print_table_lodo,
)
from epai.ai_detection.trainv2.run_lodo_performance import _std as _std_lodo
from epai.ai_detection.trainv2.run_lodo_performance import compute_zeroshot as compute_zeroshot_lodo
from epai.ai_detection.trainv2.run_zeroshot_performance import DEFAULT_FPRS
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "judge_distillation_performance"

DEFAULT_SPLITS = ["gb_orig", "mg_gb", "mg_sci"]
DEFAULT_OOD = ["mg_aisci"]
REGIMES = ("in_domain", "lodo")

#: the judge column distilled into the judge_distill_* arms -- the strongest
#: zero-shot detector on MG-AI-Sci (0.958 AUROC / 0.56 oracle TPR@1%FPR), a
#: continuous P(human) score in [0, 1]. See ``epai/ai_detection/score/
#: score_logprob_windows.py`` for how it's computed.
JUDGE_COL = "llm_judge_score_gpt56luna_v3_wlp"

#: hybrid's own defaults (epai/algs/judge_distill.py): lambda_distill 0.5,
#: val_frac 0.15 + patience 10 (early-stopped), class_weight "none".
_HYBRID_DEFAULT_PARAMS = {"lambda_distill": 0.5, "epochs": 100, "lr": 1e-3}

DISTILL_SPECS: dict[str, dict] = {
    "judge_distill_regress": {
        "label": "roberta·judge_distill(regress)", "alg": "judge_distill_regress",
        "aux": True, "params": {"alpha": 1.0},
    },
    "judge_distill_hybrid": {
        "label": "roberta·judge_distill(hybrid)", "alg": "judge_distill_hybrid",
        "aux": True, "params": dict(_HYBRID_DEFAULT_PARAMS),
    },
}

#: named hyperparameter-sweep arms for judge_distill_hybrid, the current best
#: new arm. lambda_distill controls how much the soft judge target counts
#: against GT; judge_distill_hybrid_balanced combines it with the class-
#: weighting idea from run_class_weight_performance.py.
HYBRID_SWEEP_SPECS: dict[str, dict] = {
    f"judge_distill_hybrid_lam{str(lam).replace('.', 'p')}": {
        "label": f"roberta·judge_distill(hybrid,λ={lam})", "alg": "judge_distill_hybrid",
        "aux": True, "params": {**_HYBRID_DEFAULT_PARAMS, "lambda_distill": lam},
    }
    for lam in (0.1, 0.25, 1, 2)
} | {
    "judge_distill_hybrid_balanced": {
        "label": "roberta·judge_distill(hybrid,balanced)", "alg": "judge_distill_hybrid",
        "aux": True, "params": {**_HYBRID_DEFAULT_PARAMS, "class_weight": "balanced"},
    },
}

#: existing baseline + the distillation arms + the class-weighted arm
#: (defined once in run_class_weight_performance.py, imported here so the
#: two pilots don't carry two copies of the same spec) -- the sweep is added
#: separately in main() unless --no-sweep.
CORE_SPECS: dict[str, dict] = {**TRAINED_SPECS, **DISTILL_SPECS, **BALANCED_SPECS}


# ---------------------------------------------------------------------------
# Regime: in-domain 70/30 -- same row shape as
# run_in_domain_performance.compute_trained, over ALL_SPECS
# ---------------------------------------------------------------------------


def _task_in_domain(split_key, path, spec_key, spec, ood_splits, seeds, test_size,
                    fprs, group_col, human_label, combine, cpu_threads):
    """One (split, detector) unit of in-domain work, self-contained -- reloads
    its own data and OOD sets rather than receiving them as arguments, so it
    is safe to run in its own process: an algorithm's ``fit`` calls
    ``torch.manual_seed``/``np.random.seed``, both PROCESS-GLOBAL, so two
    specs fitting inside the same process at once would stomp on each
    other's random state (see ``run_in_domain_performance._multi_emb_task``,
    the template this mirrors). Reloading a Parquet is a fraction of a
    second; that is cheaper and simpler than shipping the arrays across the
    process boundary."""
    if cpu_threads:
        torch.set_num_threads(cpu_threads)
    X, y, Z, groups = load_features(path, group_col, human_label, aux_col=JUDGE_COL)

    ood_sets: dict[str, tuple] = {}
    for name, opath in ood_splits:
        oX, oy, oZ, _ = load_features(opath, group_col, human_label, aux_col=JUDGE_COL)
        ood_sets[name] = (oX, oy, oZ, int((oy == 0).sum()))

    a: list[float] = []
    t = {f: [] for f in fprs}
    te_ = {f: [] for f in fprs}
    ood_reads: dict[str, dict[float, list]] = {n: {f: [] for f in fprs} for n in ood_sets}
    n_tr = n_te = n_te_h = 0
    for seed in seeds:
        tr, te = grouped_split(y, groups, test_size, seed)
        clf = fit(spec, X, y, Z, groups, tr, seed)
        au, per = eval_on_fprs(clf, spec, X[te], y[te], Z[te], fprs)
        a.append(au)
        for f in fprs:
            t[f].append(per[f][0])
            te_[f].append(per[f][1])
        n_tr, n_te, n_te_h = len(tr), len(te), int((y[te] == 0).sum())
        for name, (oX, oy, oZ, _) in ood_sets.items():
            oau, oper = eval_on_fprs(clf, spec, oX, oy, oZ, fprs)
            for f in fprs:
                ood_reads[name][f].append((oau, oper[f][0], oper[f][1]))

    cells: dict[float, dict] = {}
    for f in fprs:
        g = [goodness(ai, ei, combine) for ai, ei in zip(a, te_[f])]
        cells[f] = {
            "split": split_key, "fpr": f, "n_train": n_tr, "n_test": n_te,
            "n_test_human": n_te_h, "n_seeds": len(seeds),
            "auroc": float(np.mean(a)), "auroc_std": _std(a),
            "tpr_at_fpr": float(np.mean(t[f])), "tpr_at_fpr_std": _std(t[f]),
            "tpr_eff": float(np.mean(te_[f])), "tpr_eff_std": _std(te_[f]),
            "goodness": float(np.mean(g)), "goodness_std": _std(g),
            "eff_fpr": eff_fpr_level(n_te_h, f),
        }
    ood_n_h = {name: n_h for name, (_, _, _, n_h) in ood_sets.items()}
    return spec_key, cells, ood_reads, ood_n_h


def compute_in_domain(all_specs, train_splits, ood_splits, seeds, test_size, fprs,
                      group_col, human_label, combine, jobs=1,
                      cpu_threads=None) -> list[dict]:
    """Same procedure as ``run_in_domain_performance.compute_trained``, over
    ``all_specs``, but every load reads ``JUDGE_COL`` as the auxiliary column
    instead of the judge cue vector. ``jobs > 1`` fits every (split, detector)
    pair in its own OS process (see :func:`_task_in_domain`); ``jobs <= 1``
    runs the exact same calls in a plain loop, so the two modes compute
    identical numbers and only differ in wall-clock time."""
    tasks = [(split_key, path, spec_key, spec)
             for split_key, path in train_splits
             for spec_key, spec in all_specs.items()]

    results = []
    if jobs > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            futs = [
                ex.submit(_task_in_domain, split_key, path, spec_key, spec,
                         ood_splits, seeds, test_size, fprs, group_col,
                         human_label, combine, cpu_threads)
                for split_key, path, spec_key, spec in tasks
            ]
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc="corpus x detector (parallel)", unit="task"):
                results.append(fut.result())
    else:
        for split_key, path, spec_key, spec in tqdm(tasks, desc="corpus x detector", unit="task"):
            results.append(_task_in_domain(split_key, path, spec_key, spec, ood_splits,
                                           seeds, test_size, fprs, group_col,
                                           human_label, combine, cpu_threads))

    cells: dict[str, dict[float, list[dict]]] = {k: {f: [] for f in fprs} for k in all_specs}
    ood_reads: dict[str, dict[str, dict[float, list]]] = {
        k: {n: {f: [] for f in fprs} for n, _ in ood_splits} for k in all_specs}
    ood_n_h: dict[str, int] = {}
    for spec_key, task_cells, task_ood_reads, task_ood_n_h in results:
        for f in fprs:
            cells[spec_key][f].append(task_cells[f])
        for name, by_fpr in task_ood_reads.items():
            for f in fprs:
                ood_reads[spec_key][name][f].extend(by_fpr[f])
        ood_n_h.update(task_ood_n_h)

    rows: list[dict] = []
    for spec_key, spec in all_specs.items():
        detector = f"trained/{spec['label']}"
        for f in fprs:
            spec_cells = list(cells[spec_key][f])
            for c in spec_cells:
                rows.append(_row(detector, c["split"], "in_domain", c))
            for name, n_h in ood_n_h.items():
                reads = ood_reads[spec_key][name][f]
                oa = [r[0] for r in reads]
                ot = [r[1] for r in reads]
                ote = [r[2] for r in reads]
                og = [goodness(ai, ei, combine) for ai, ei in zip(oa, ote)]
                c = {"split": name, "fpr": f, "n_test": "", "n_test_human": n_h,
                     "n_seeds": len(seeds),
                     "auroc": float(np.mean(oa)), "auroc_std": _std(oa),
                     "tpr_at_fpr": float(np.mean(ot)), "tpr_at_fpr_std": _std(ot),
                     "tpr_eff": float(np.mean(ote)), "tpr_eff_std": _std(ote),
                     "goodness": float(np.mean(og)), "goodness_std": _std(og),
                     "eff_fpr": eff_fpr_level(n_h, f)}
                spec_cells.append(c)
                rows.append(_row(detector, name, "ood", c))
            rows.append(_row(detector, "mean", "overall",
                             _overall_cell(spec_cells, len(seeds), f)))
    return rows


# ---------------------------------------------------------------------------
# Regime: leave-one-domain-out -- same row shape as
# run_lodo_performance.compute_trained, over ALL_SPECS
# ---------------------------------------------------------------------------


def _task_lodo(split_key, path, spec_key, spec, ood_splits, seeds, fprs,
              human_label, min_domain_rows, cpu_threads):
    """One (split, detector) unit of LODO work -- every fold for ``spec`` on
    ``split_key``, all seeds -- self-contained like :func:`_task_in_domain`
    (reloads its own data + OOD sets), so it is safe to run in its own
    process. Domain eligibility (``domains``) is a pure function of the
    split's own ``y``/``dom`` columns, not of ``spec``, so recomputing it once
    per task is redundant but never inconsistent across specs of the same
    split. Returns ``(spec_key, split_key, folds, ood_reads, ood_n_h)`` --
    ``folds`` is ``[]`` when the split has fewer than 2 usable domains (the
    caller drops it, exactly like the serial ``compute_lodo`` used to)."""
    if cpu_threads:
        torch.set_num_threads(cpu_threads)
    X, y, Z, dom, ids = _load(path, human_label, aux_col=JUDGE_COL)
    ood_sets = {name: _load_ood(opath, human_label, aux_col=JUDGE_COL)
               for name, opath in ood_splits}

    counts = {d: int((dom == d).sum()) for d in np.unique(dom)}
    domains = sorted(d for d, c in counts.items()
                     if c >= min_domain_rows and np.unique(y[dom == d]).size == 2)
    if len(domains) < 2:
        return spec_key, split_key, [], {}, {}

    folds = []
    ood_reads: dict[str, dict[float, list]] = {n: {f: [] for f in fprs} for n in ood_sets}
    for d in domains:
        tr, te = dom != d, dom == d
        assert d not in set(dom[tr]) and set(dom[te]) == {d}
        assert tr.sum() + te.sum() == len(dom) and not (tr & te).any()
        assert set(ids[tr]).isdisjoint(ids[te]), "a group straddled the fold"
        fd = _fold(spec, X, y, Z, ids, tr, te, seeds, fprs, ood_sets)
        fd["held_out_domain"] = d
        for name, by_fpr in fd.pop("ood").items():
            for f in fprs:
                ood_reads[name][f].extend(by_fpr[f])
        folds.append(fd)
    ood_n_h = {name: n_h for name, (_, _, _, n_h) in ood_sets.items()}
    return spec_key, split_key, folds, ood_reads, ood_n_h


def compute_lodo(all_specs, splits, ood_splits, seeds, fprs, human_label,
                 min_domain_rows, combine, jobs=1, cpu_threads=None) -> list[dict]:
    """Mirror of ``run_lodo_performance.compute_trained``, over ``all_specs``,
    loading ``JUDGE_COL`` as the auxiliary column via ``_load``/``_load_ood``'s
    ``aux_col`` parameter. ``jobs > 1`` fits every (split, detector) pair's
    full set of domain folds in its own OS process (see :func:`_task_lodo`);
    ``jobs <= 1`` runs the exact same calls in a plain loop."""
    tasks = [(split_key, path, spec_key, spec)
             for split_key, path in splits
             for spec_key, spec in all_specs.items()]

    results = []
    if jobs > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            futs = [
                ex.submit(_task_lodo, split_key, path, spec_key, spec, ood_splits,
                         seeds, fprs, human_label, min_domain_rows, cpu_threads)
                for split_key, path, spec_key, spec in tasks
            ]
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc="LODO (parallel)", unit="task"):
                results.append(fut.result())
    else:
        for split_key, path, spec_key, spec in tqdm(tasks, desc="LODO", unit="task"):
            results.append(_task_lodo(split_key, path, spec_key, spec, ood_splits,
                                      seeds, fprs, human_label, min_domain_rows,
                                      cpu_threads))

    all_folds: dict[str, dict[str, list[dict]]] = {}
    ood_reads: dict[str, dict[str, dict[float, list]]] = {
        k: {n: {f: [] for f in fprs} for n, _ in ood_splits} for k in all_specs
    }
    ood_n_h: dict[str, int] = {}
    for spec_key, split_key, folds, task_ood_reads, task_ood_n_h in results:
        if not folds:
            print(f"  {split_key}: fewer than 2 usable domains — skipped")
            continue
        all_folds.setdefault(split_key, {})[spec_key] = folds
        for name, by_fpr in task_ood_reads.items():
            for f in fprs:
                ood_reads[spec_key][name][f].extend(by_fpr[f])
        ood_n_h.update(task_ood_n_h)

    if not all_folds:
        return []

    rows: list[dict] = []
    for spec_key, spec in all_specs.items():
        detector = f"trained/{spec['label']}"
        for fpr in fprs:
            split_rows: list[dict] = []
            for split_key, spec_folds in all_folds.items():
                folds = spec_folds[spec_key]
                gf = [goodness(fd["auroc"], fd["by_fpr"][fpr]["tpr_eff"], combine)
                      for fd in folds]
                for fd, gfi in zip(folds, gf):
                    bf = fd["by_fpr"][fpr]
                    rows.append({
                        "level": "fold", "kind": "trained", "detector": detector,
                        "split": split_key, "held_out_domain": fd["held_out_domain"],
                        "fpr": fpr, "n_test": fd["n_test"], "n_folds": "",
                        "n_seeds": len(seeds),
                        "auroc": fd["auroc"], "auroc_std": "",
                        "tpr_at_fpr": bf["tpr_at_fpr"], "tpr_at_fpr_std": "",
                        "tpr_eff": bf["tpr_eff"], "tpr_eff_std": "",
                        "eff_fpr": bf["eff_fpr"], "goodness": gfi, "goodness_std": "",
                    })
                ok = [fd for fd in folds if fd["n_test_human"] >= 1 / fpr]
                use = ok or folds
                sr = {
                    "level": "split", "kind": "trained", "detector": detector,
                    "split": split_key, "held_out_domain": "", "fpr": fpr,
                    "n_test": "", "n_folds": len(folds), "n_seeds": len(seeds),
                    "reachable": bool(ok),
                    "auroc": float(np.mean([fd["auroc"] for fd in folds])),
                    "auroc_std": _std_lodo([fd["auroc"] for fd in folds]),
                    "tpr_at_fpr": float(np.mean([fd["by_fpr"][fpr]["tpr_at_fpr"] for fd in use])),
                    "tpr_at_fpr_std": _std_lodo([fd["by_fpr"][fpr]["tpr_at_fpr"] for fd in use]),
                    "tpr_eff": float(np.mean([fd["by_fpr"][fpr]["tpr_eff"] for fd in folds])),
                    "tpr_eff_std": _std_lodo([fd["by_fpr"][fpr]["tpr_eff"] for fd in folds]),
                    "eff_fpr": float(np.mean([fd["by_fpr"][fpr]["eff_fpr"] for fd in folds])),
                    "goodness": float(np.mean(gf)), "goodness_std": _std_lodo(gf),
                }
                rows.append(sr)
                split_rows.append(sr)

            for name, n_h in ood_n_h.items():
                reads = ood_reads[spec_key][name][fpr]
                oa, ot = [r[0] for r in reads], [r[1] for r in reads]
                ote = [r[2] for r in reads]
                og = [goodness(a_, te, combine) for a_, te in zip(oa, ote)]
                sr = {
                    "level": "split", "kind": "trained", "detector": detector,
                    "split": name, "held_out_domain": "", "fpr": fpr, "n_test": "",
                    "n_folds": len(reads), "n_seeds": len(seeds),
                    "reachable": bool(n_h >= 1 / fpr),
                    "auroc": float(np.mean(oa)), "auroc_std": _std_lodo(oa),
                    "tpr_at_fpr": float(np.mean(ot)), "tpr_at_fpr_std": _std_lodo(ot),
                    "tpr_eff": float(np.mean(ote)), "tpr_eff_std": _std_lodo(ote),
                    "eff_fpr": eff_fpr_level(n_h, fpr),
                    "goodness": float(np.mean(og)), "goodness_std": _std_lodo(og),
                }
                rows.append(sr)
                split_rows.append(sr)

            tpr_ok = [r for r in split_rows if r.get("reachable")]
            rows.append({
                "level": "overall", "kind": "trained", "detector": detector,
                "split": "mean", "held_out_domain": "", "fpr": fpr, "n_test": "",
                "n_folds": sum(len(all_folds[s][spec_key]) for s in all_folds),
                "n_seeds": len(seeds),
                "auroc": float(np.mean([r["auroc"] for r in split_rows])),
                "auroc_std": float(np.mean([r["auroc_std"] for r in split_rows])),
                "tpr_at_fpr": float(np.mean([r["tpr_at_fpr"] for r in tpr_ok])) if tpr_ok else float("nan"),
                "tpr_at_fpr_std": float(np.mean([r["tpr_at_fpr_std"] for r in tpr_ok])) if tpr_ok else float("nan"),
                "tpr_eff": float(np.mean([r["tpr_eff"] for r in split_rows])),
                "tpr_eff_std": float(np.mean([r["tpr_eff_std"] for r in split_rows])),
                "eff_fpr": fpr,
                "goodness": float(np.mean([r["goodness"] for r in split_rows])),
                "goodness_std": float(np.mean([r["goodness_std"] for r in split_rows])),
            })
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def plots_in_domain(csv_path: Path, meta: dict, out_dir: Path) -> None:
    rows = pd.read_csv(csv_path).to_dict("records")
    from epai.ai_detection.trainv2.plots import denario_scatter, goodness_bars, zeroshot_scatter

    out_dir.mkdir(parents=True, exist_ok=True)
    ood = "/".join(meta["ood"])
    n_seeds = meta["n_seeds"]
    for fpr in meta["fprs"]:
        fmeta = {**meta, "fpr": fpr}
        tag = fpr_tag(fpr)
        frows = rows_at_fpr(rows, fpr)
        zeroshot_scatter(
            frows, fmeta, out_dir / f"in_domain_{tag}.png", key="detector",
            title=f"Judge distillation pilot (70/30, {n_seeds} seeds) -- "
                  f"roberta·logistic (GT) vs. judge-distilled / class-weighted "
                  f"arms vs. zero-shot picks, plus {ood} as eval-only OOD\n"
                  f"AUROC vs. oracle TPR@{fpr * 100:g}%FPR",
        )
        goodness_bars(
            frows, fmeta, out_dir / f"in_domain_goodness_{tag}.png", key="detector",
            combine=meta.get("combine", "hmean"),
            title=f"Judge distillation pilot (70/30, {n_seeds} seeds) -- "
                  f"goodness (TPR@{fpr * 100:g}%FPR) across the MANUGRAM splits",
        )
        denario_scatter(
            frows, fmeta, out_dir / f"in_domain_denario_{tag}.png",
            title=f"Judge distillation pilot (70/30, {n_seeds} seeds) on {ood} "
                  f"(eval-only OOD)\nAUROC vs. oracle TPR@{fpr * 100:g}%FPR",
        )


def plots_lodo(csv_path: Path, meta: dict, out_dir: Path) -> None:
    all_rows = [r for r in pd.read_csv(csv_path).to_dict("records")
               if r["level"] in ("split", "overall")]
    from epai.ai_detection.trainv2.plots import denario_scatter, goodness_bars, zeroshot_scatter

    out_dir.mkdir(parents=True, exist_ok=True)
    ood = "/".join(meta["ood"])
    for fpr in meta["fprs"]:
        fmeta = {**meta, "fpr": fpr}
        tag = fpr_tag(fpr)
        rows = rows_at_fpr(all_rows, fpr)
        zeroshot_scatter(
            rows, fmeta, out_dir / f"lodo_{tag}.png", key="detector",
            title=f"Judge distillation pilot, leave-one-domain-out -- "
                  f"roberta·logistic (GT) vs. judge-distilled / class-weighted "
                  f"arms vs. zero-shot picks, plus {ood} as eval-only OOD\n"
                  f"AUROC vs. oracle TPR@{fpr * 100:g}%FPR",
        )
        goodness_bars(
            rows, fmeta, out_dir / f"lodo_goodness_{tag}.png", key="detector",
            combine=meta.get("combine", "hmean"),
            title=f"Judge distillation pilot, leave-one-domain-out -- "
                  f"goodness (TPR@{fpr * 100:g}%FPR) across the MANUGRAM splits",
        )
        denario_scatter(
            rows, fmeta, out_dir / f"lodo_denario_{tag}.png",
            title=f"Judge distillation pilot, leave-one-domain-out, on {ood} "
                  f"(eval-only OOD)\nAUROC vs. oracle TPR@{fpr * 100:g}%FPR",
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    ap.add_argument("--splits", nargs="+", default=None, metavar="KEY",
                    help=f"training splits (default: {DEFAULT_SPLITS})")
    ap.add_argument("--ood-splits", nargs="+", default=None, metavar="KEY",
                    help=f"eval-only corpora, never trained on (default: {DEFAULT_OOD})")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS, metavar="LEVEL",
                    help=f"oracle FPR level(s); one figure set per level "
                         f"(default: {DEFAULT_FPRS})")
    ap.add_argument("--combine", choices=("hmean", "gmean", "amean"), default="hmean")
    ap.add_argument("--min-domain-rows", type=int, default=100)
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--zeroshot-config", type=Path, default=ZEROSHOT_CONFIG)
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--no-sweep", action="store_true",
                    help="drop the 5 judge_distill_hybrid hyperparameter-sweep "
                         "arms (HYBRID_SWEEP_SPECS); core comparison only")
    ap.add_argument("--jobs", type=int, default=1, metavar="N",
                    help="fit N (split, detector) pairs at once, each in its own "
                         "process (default: 1, i.e. serial). Applies to both "
                         "regimes -- for LODO, one task is a detector's full set "
                         "of domain folds on one split")
    ap.add_argument("--gpu", default=None, metavar="IDS",
                    help="value for CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1'). "
                         "Default: leave unset so all visible GPUs are used. Applied "
                         "before any model is fit; every --jobs worker inherits it")
    ap.add_argument("--cpu-threads", type=int, default=None, metavar="N",
                    help="cap CPU threads used per fit (sets OMP_NUM_THREADS / "
                         "MKL_NUM_THREADS and torch's intra-op pool -- in every "
                         "--jobs worker too). Default: leave unset. With --jobs > 1, "
                         "consider capping this so N workers don't each grab every "
                         "core")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="OUT_DIR",
                    help="re-plot from an existing out-dir's CSV(s), no compute")
    args = ap.parse_args()

    # Applied before any `fit()` call (the only place a torch algorithm ever
    # touches CUDA): CUDA device enumeration and the BLAS/OpenMP thread pools
    # are triggered lazily, on first use, so setting these here is still
    # early enough. A --jobs worker is a freshly spawned process that
    # inherits this environment from scratch, so it applies there too.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.cpu_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
        torch.set_num_threads(args.cpu_threads)

    all_specs = dict(CORE_SPECS) if args.no_sweep else {**CORE_SPECS, **HYBRID_SWEEP_SPECS}

    train_splits = resolve_splits(args.splits or DEFAULT_SPLITS, PARQUET_DIR)
    ood_splits = resolve_splits(args.ood_splits or DEFAULT_OOD, PARQUET_DIR)
    meta = {"fprs": args.fpr, "n_seeds": len(args.seeds), "combine": args.combine,
            "ood": [s for s, _ in ood_splits],
            "splits": [s for s, _ in train_splits] + [s for s, _ in ood_splits]}

    if args.replot is not None:
        id_csv = args.replot / "in_domain_performance.csv"
        lodo_csv = args.replot / "lodo_performance.csv"
        if id_csv.exists():
            plots_in_domain(id_csv, meta, args.replot)
        if lodo_csv.exists():
            plots_lodo(lodo_csv, meta, args.replot)
        if not id_csv.exists() and not lodo_csv.exists():
            raise SystemExit(f"no performance.csv under {args.replot}")
        print(f"replotted from {args.replot}")
        return

    missing = [str(p) for _, p in [*train_splits, *ood_splits] if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")

    print(f"detectors: {[s['label'] for s in TRAINED_SPECS.values()]} (existing) "
          f"+ {[s['label'] for s in DISTILL_SPECS.values()]} (distilled from {JUDGE_COL!r}) "
          f"+ {[s['label'] for s in BALANCED_SPECS.values()]} (class-weighted)"
          + ("" if args.no_sweep else
             f" + {[s['label'] for s in HYBRID_SWEEP_SPECS.values()]} (sweep)"))
    print(f"regimes  : {args.regimes}")
    print(f"train on : {[s for s, _ in train_splits]}   OOD eval: {[s for s, _ in ood_splits]}")
    print(f"seeds    : {args.seeds}   --jobs {args.jobs}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if "in_domain" in args.regimes:
        rows = compute_in_domain(all_specs, train_splits, ood_splits, args.seeds,
                                 args.test_size, args.fpr, args.group_col,
                                 args.human_label, args.combine, args.jobs,
                                 args.cpu_threads)
        if not args.no_zeroshot:
            rows += compute_zeroshot_in_domain(
                train_splits, ood_splits, args.zeroshot_config, args.seeds,
                args.test_size, args.fpr, args.group_col, args.human_label, args.combine)
        csv_path = args.out_dir / "in_domain_performance.csv"
        pd.DataFrame(rows, columns=CSV_FIELDS_IN_DOMAIN).to_csv(csv_path, index=False)
        _print_table_in_domain(rows, args.fpr)
        print(f"\nwrote {csv_path}")
        if not args.no_plots:
            plots_in_domain(csv_path, meta, args.out_dir)

    if "lodo" in args.regimes:
        rows = compute_lodo(all_specs, train_splits, ood_splits, args.seeds, args.fpr,
                            args.human_label, args.min_domain_rows, args.combine,
                            args.jobs, args.cpu_threads)
        if not args.no_zeroshot:
            rows += compute_zeroshot_lodo(
                train_splits, ood_splits, args.zeroshot_config, args.seeds,
                args.fpr, args.human_label, args.min_domain_rows, args.combine)
        csv_path = args.out_dir / "lodo_performance.csv"
        pd.DataFrame(rows, columns=CSV_FIELDS_LODO).to_csv(csv_path, index=False)
        _print_table_lodo(rows, args.fpr)
        print(f"\nwrote {csv_path}")
        if not args.no_plots:
            plots_lodo(csv_path, meta, args.out_dir)


if __name__ == "__main__":
    main()

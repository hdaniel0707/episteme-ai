r"""Pilot: does class-weighting fix (part of) the MG-AI-Sci gap?

MG-GB/MG-Sci are 4:1 machine:human (4 generators pooled per human doc), and
nothing in the training pipeline corrected for it -- sklearn's
``LogisticRegression`` defaults to ``class_weight=None`` and the shared torch
trainer (``epai.algs._torch_common.train_torch_model``) used a plain
``CrossEntropyLoss`` with no ``weight=`` tensor (both gained an opt-in
``class_weight: "balanced"`` hyperparameter -- sklearn's inverse-frequency
convention, computed from the training rows only).

One new arm: ``roberta_logistic_balanced`` -- the existing GT baseline's own
architecture (``logistic`` on ``emb_roberta``), class-weighting turned on,
nothing else different. Reported alongside ``trained/roberta·logistic``
(unweighted) and the ``best_zeroshot.yaml`` picks, so the comparison is
exactly: best zero-shot vs. roberta·logistic vs. roberta·logistic(balanced).

``BALANCED_SPECS`` (this module) is also imported by
``run_judge_distillation_performance.py``, so the balanced arm shows up
there too, next to the judge-distillation arms -- defined once, here.

An **opt-in add-on**, run by hand, same treatment as ``run_dr_transfer.py`` /
``run_robust_performance.py`` / ``run_judge_distillation_performance.py``:
absent from ``rebuild_all.TASKS``; ``_trained.py`` / ``core.py`` /
``splits.py`` / the two main runners / ``rebuild_all.py`` are only ever
imported from. In-domain 70/30 only (no LODO sibling yet).

compute -> CSV -> plot from the CSV:

    results/trainv2/class_weight_performance/
        performance.csv   same shape as run_in_domain_performance.py's
        in_domain_{1pct,5pct}.png (+ _goodness_, _denario_)

Run (sandbox has no ``uv`` -- use ``.venv/bin/python`` directly):

    .venv/bin/python epai/ai_detection/trainv2/run_class_weight_performance.py
    .venv/bin/python epai/ai_detection/trainv2/run_class_weight_performance.py --seeds 0 1
    .venv/bin/python epai/ai_detection/trainv2/run_class_weight_performance.py \
        --replot results/trainv2/class_weight_performance/performance.csv
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
from epai.ai_detection.trainv2.run_in_domain_performance import (
    CSV_FIELDS,
    _overall_cell,
    _print_table,
    _row,
    _std,
    compute_zeroshot,
)
from epai.ai_detection.trainv2.run_zeroshot_performance import DEFAULT_FPRS
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "class_weight_performance"

DEFAULT_SPLITS = ["gb_orig", "mg_gb", "mg_sci"]
DEFAULT_OOD = ["mg_aisci"]

#: GT-trained, same architecture as trained/roberta·logistic, class-weighted.
#: Shared with run_judge_distillation_performance.py -- defined once, here.
BALANCED_SPECS: dict[str, dict] = {
    "roberta_logistic_balanced": {
        "label": "roberta·logistic(balanced)", "alg": "logistic", "aux": False,
        "params": {"class_weight": "balanced"},
    },
}

#: existing baseline + the new arm, one dict every compute loop iterates.
ALL_SPECS: dict[str, dict] = {**TRAINED_SPECS, **BALANCED_SPECS}


def _task(split_key, path, spec_key, spec, ood_splits, seeds, test_size, fprs,
         group_col, human_label, combine, cpu_threads):
    """One (split, detector) unit of work, self-contained -- reloads its own
    data and OOD sets rather than receiving them as arguments, so it is safe
    to run in its own process: an algorithm's ``fit`` calls
    ``torch.manual_seed``/``np.random.seed``, both PROCESS-GLOBAL, so two
    specs fitting inside the same process at once would stomp on each
    other's random state (see ``run_in_domain_performance._multi_emb_task``,
    the template this mirrors). Reloading a Parquet is a fraction of a
    second; that is cheaper and simpler than shipping the arrays across the
    process boundary."""
    if cpu_threads:
        torch.set_num_threads(cpu_threads)
    X, y, Z, groups = load_features(path, group_col, human_label)

    ood_sets: dict[str, tuple] = {}
    for name, opath in ood_splits:
        oX, oy, oZ, _ = load_features(opath, group_col, human_label)
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


def compute_in_domain(train_splits, ood_splits, seeds, test_size, fprs,
                      group_col, human_label, combine, jobs=1,
                      cpu_threads=None) -> list[dict]:
    """Identical procedure to ``run_in_domain_performance.compute_trained``,
    over ``ALL_SPECS`` instead of just ``TRAINED_SPECS``. ``jobs > 1`` fits
    every (split, detector) pair in its own OS process (see :func:`_task`);
    ``jobs <= 1`` runs the exact same calls in a plain loop, so the two modes
    compute identical numbers and only differ in wall-clock time."""
    tasks = [(split_key, path, spec_key, spec)
             for split_key, path in train_splits
             for spec_key, spec in ALL_SPECS.items()]

    results = []
    if jobs > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            futs = [
                ex.submit(_task, split_key, path, spec_key, spec, ood_splits,
                         seeds, test_size, fprs, group_col, human_label,
                         combine, cpu_threads)
                for split_key, path, spec_key, spec in tasks
            ]
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc="corpus x detector (parallel)", unit="task"):
                results.append(fut.result())
    else:
        for split_key, path, spec_key, spec in tqdm(tasks, desc="corpus x detector", unit="task"):
            results.append(_task(split_key, path, spec_key, spec, ood_splits, seeds,
                                 test_size, fprs, group_col, human_label, combine,
                                 cpu_threads))

    cells: dict[str, dict[float, list[dict]]] = {k: {f: [] for f in fprs} for k in ALL_SPECS}
    ood_reads: dict[str, dict[str, dict[float, list]]] = {
        k: {n: {f: [] for f in fprs} for n, _ in ood_splits} for k in ALL_SPECS}
    ood_n_h: dict[str, int] = {}
    for spec_key, task_cells, task_ood_reads, task_ood_n_h in results:
        for f in fprs:
            cells[spec_key][f].append(task_cells[f])
        for name, by_fpr in task_ood_reads.items():
            for f in fprs:
                ood_reads[spec_key][name][f].extend(by_fpr[f])
        ood_n_h.update(task_ood_n_h)

    rows: list[dict] = []
    for spec_key, spec in ALL_SPECS.items():
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


def plots_from_csv(csv_path: Path, meta: dict, out_dir: Path) -> None:
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
            title=f"Class-weighting pilot (70/30, {n_seeds} seeds) -- "
                  f"roberta·logistic vs. its class-weighted twin vs. zero-shot "
                  f"picks, plus {ood} as eval-only OOD\nAUROC vs. oracle "
                  f"TPR@{fpr * 100:g}%FPR",
        )
        goodness_bars(
            frows, fmeta, out_dir / f"in_domain_goodness_{tag}.png", key="detector",
            combine=meta.get("combine", "hmean"),
            title=f"Class-weighting pilot (70/30, {n_seeds} seeds) -- "
                  f"goodness (TPR@{fpr * 100:g}%FPR) across the MANUGRAM splits",
        )
        denario_scatter(
            frows, fmeta, out_dir / f"in_domain_denario_{tag}.png",
            title=f"Class-weighting pilot (70/30, {n_seeds} seeds) on {ood} "
                  f"(eval-only OOD)\nAUROC vs. oracle TPR@{fpr * 100:g}%FPR",
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--zeroshot-config", type=Path, default=ZEROSHOT_CONFIG)
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, metavar="N",
                    help="fit N (split, detector) pairs at once, each in its own "
                         "process (default: 1, i.e. serial)")
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
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV")
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

    train_splits = resolve_splits(args.splits or DEFAULT_SPLITS, PARQUET_DIR)
    ood_splits = resolve_splits(args.ood_splits or DEFAULT_OOD, PARQUET_DIR)
    meta = {"fprs": args.fpr, "n_seeds": len(args.seeds), "combine": args.combine,
            "ood": [s for s, _ in ood_splits],
            "splits": [s for s, _ in train_splits] + [s for s, _ in ood_splits]}

    if args.replot is not None:
        plots_from_csv(args.replot, meta, args.replot.parent)
        print(f"replotted from {args.replot}")
        return

    missing = [str(p) for _, p in [*train_splits, *ood_splits] if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")

    print(f"detectors: {[s['label'] for s in TRAINED_SPECS.values()]} (existing) "
          f"+ {[s['label'] for s in BALANCED_SPECS.values()]} (new)")
    print(f"train on : {[s for s, _ in train_splits]}   OOD eval: {[s for s, _ in ood_splits]}")
    print(f"seeds    : {args.seeds}   split: {int(round((1 - args.test_size) * 100))}/"
          f"{int(round(args.test_size * 100))} grouped on {args.group_col!r}"
          f"   --jobs {args.jobs}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = compute_in_domain(train_splits, ood_splits, args.seeds, args.test_size,
                             args.fpr, args.group_col, args.human_label, args.combine,
                             args.jobs, args.cpu_threads)
    if not args.no_zeroshot:
        print(f"\nzero-shot: {args.zeroshot_config} "
              f"(scored on the trained models' own eval rows)")
        rows += compute_zeroshot(train_splits, ood_splits, args.zeroshot_config,
                                 args.seeds, args.test_size, args.fpr,
                                 args.group_col, args.human_label, args.combine)

    csv_path = args.out_dir / "performance.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    _print_table(rows, args.fpr)
    print(f"\nwrote {csv_path}")

    if not args.no_plots:
        plots_from_csv(csv_path, meta, args.out_dir)


if __name__ == "__main__":
    main()

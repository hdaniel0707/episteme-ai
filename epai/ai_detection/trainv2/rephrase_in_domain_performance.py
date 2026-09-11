r"""In-domain trained detection on the rephrase task, with rephrased text
counted as HUMAN ground truth -- otherwise identical to
``run_in_domain_performance.py``.

Three trained detectors (:mod:`._trained`: ``emb_roberta`` -> logistic, and the
two ``emb_roberta`` + gpt56luna-judge-cuevec hybrids -- no gemma anywhere).
Within each rephrase corpus (MG-GB = MG-GB + MG-Reph-GB, MG-Sci = MG-Sci +
MG-Reph-Sci): a grouped 70/30 split -- a document's human / rephrase /
full-generation rows share one ``id`` and never straddle the cut -- repeated
over N seeds. The label:

    y = 1   the full gpt56luna generation ('...:0701A' GB, '...:0811A' Sci)
    y = 0   human AND rephrase (gpt56luna:0908R)

Every fitted model is ALSO scored, unchanged, on each ``--ood-splits`` corpus
(default ``mg_aisci`` -- human vs. denario, never trained on). The rephrase-GT
best zero-shot picks (``best_zeroshot.yaml`` minus any gemma judge) are scored
on the SAME rows as the trained models -- each seed's 70/30 test partition per
rephrase corpus (a seed mean +- SD, like a trained point), the full corpus per
OOD split (normal GT, no error bar) -- not on the whole dataset.

compute -> CSV -> plot from the CSV:

    results/trainv2/rephrase_in_domain_performance/
        performance.csv
        in_domain.png

Run:
    uv run python epai/ai_detection/trainv2/rephrase_in_domain_performance.py
    uv run python epai/ai_detection/trainv2/rephrase_in_domain_performance.py --seeds 0 1 2
    uv run python epai/ai_detection/trainv2/rephrase_in_domain_performance.py \
        --replot results/trainv2/rephrase_in_domain_performance/performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import pyarrow.parquet as pq

from epai.ai_detection.trainv2._reph_trained import REPH_SPLITS, load_reph_features
from epai.ai_detection.trainv2._trained import (
    TRAINED_SPECS,
    eval_on_fprs,
    fit,
    load_features,
)
from epai.ai_detection.trainv2._zeroshot import (
    agg_cell,
    eval_column_fprs,
    orient,
    overall_cell,
    resolve_specs,
)
from epai.ai_detection.trainv2.core import fpr_tag, grouped_split, rows_at_fpr
from epai.ai_detection.trainv2.plot_score_histograms import CORPUS_SOURCES
from epai.ai_detection.trainv2.plot_score_histograms import PARQUET_DIR as REPH_PARQUET_DIR
from epai.ai_detection.trainv2.rephrase_best_zeroshot_performance import _pick_columns
from epai.ai_detection.trainv2.run_best_zeroshot_performance import DEFAULT_CONFIG, _detector
from epai.ai_detection.trainv2.run_in_domain_performance import (
    CSV_FIELDS,
    _overall_cell,
    _print_table,
    _row,
    _std,
)
from epai.ai_detection.trainv2.run_zeroshot_performance import DEFAULT_FPRS
from epai.ai_detection.trainv2.run_zeroshot_performance import FAMILIES as ZS_FAMILIES
from epai.ai_detection.trainv2.run_zeroshot_performance import compute as zeroshot_compute
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE / "data" / "parquet"
DEFAULT_OUT_DIR = BASE / "results" / "trainv2" / "rephrase_in_domain_performance"
DEFAULT_OOD = ["mg_aisci"]


def compute_trained(reph_splits, ood_splits, seeds, test_size, fprs) -> list[dict]:
    ood_sets: dict[str, tuple] = {}
    for name, path in ood_splits:
        oX, oy, oZ, _ = load_features(path, "id", "human")
        ood_sets[name] = (oX, oy, oZ, int((oy == 0).sum()))
        print(f"  OOD eval corpus {name!r}: {len(oy)} rows, {int((oy == 0).sum())} "
              f"human (never trained on)")

    cells: dict[str, dict[float, list[dict]]] = {
        k: {f: [] for f in fprs} for k in TRAINED_SPECS}
    ood_reads: dict[str, dict[str, dict[float, list]]] = {
        k: {n: {f: [] for f in fprs} for n in ood_sets} for k in TRAINED_SPECS}

    for split_key in tqdm(reph_splits, desc="corpus", unit="split"):
        X, y, Z, _dom, groups = load_reph_features(split_key)
        g0, g1 = grouped_split(y, groups, test_size, seeds[0])
        assert set(groups[g0]).isdisjoint(groups[g1])
        print(f"  {split_key}: grouped {int(round((1 - test_size) * 100))}/"
              f"{int(round(test_size * 100))} on 'id' — {len(set(groups))} groups "
              f"-> {len(set(groups[g0]))} train / {len(set(groups[g1]))} test, disjoint "
              f"({int((y == 1).sum())} full-gen, {int((y == 0).sum())} human+rephrase)")

        for spec_key, spec in TRAINED_SPECS.items():
            a: list[float] = []
            t = {f: [] for f in fprs}
            n_tr = n_te = n_te_h = 0
            for seed in seeds:
                tr, te = grouped_split(y, groups, test_size, seed)
                clf = fit(spec, X, y, Z, groups, tr, seed)
                au, per = eval_on_fprs(clf, spec, X[te], y[te], Z[te], fprs)
                a.append(au)
                for f in fprs:
                    t[f].append(per[f][0])
                n_tr, n_te, n_te_h = len(tr), len(te), int((y[te] == 0).sum())
                for name, (oX, oy, oZ, _) in ood_sets.items():
                    oau, oper = eval_on_fprs(clf, spec, oX, oy, oZ, fprs)
                    for f in fprs:
                        ood_reads[spec_key][name][f].append((oau, oper[f][0]))
            for f in fprs:
                cells[spec_key][f].append({
                    "split": split_key, "fpr": f, "n_train": n_tr, "n_test": n_te,
                    "n_test_human": n_te_h, "n_seeds": len(seeds),
                    "auroc": float(np.mean(a)), "auroc_std": _std(a),
                    "tpr_at_fpr": float(np.mean(t[f])), "tpr_at_fpr_std": _std(t[f])})

    rows: list[dict] = []
    for spec_key, spec in TRAINED_SPECS.items():
        detector = f"trained/{spec['label']}"
        for f in fprs:
            spec_cells = list(cells[spec_key][f])
            for c in spec_cells:
                rows.append(_row(detector, c["split"], "in_domain", c))
            for name, (_, _, _, n_h) in ood_sets.items():
                reads = ood_reads[spec_key][name][f]
                oa = [r[0] for r in reads]
                ot = [r[1] for r in reads]
                c = {"split": name, "fpr": f, "n_test": "", "n_test_human": n_h,
                     "n_seeds": len(seeds),
                     "auroc": float(np.mean(oa)), "auroc_std": _std(oa),
                     "tpr_at_fpr": float(np.mean(ot)), "tpr_at_fpr_std": _std(ot)}
                spec_cells.append(c)
                rows.append(_row(detector, name, "ood", c))
            rows.append(_row(detector, "mean", "overall",
                             _overall_cell(spec_cells, len(seeds), f)))
    return rows


def compute_zeroshot(config, reph_splits, ood_splits, seeds, test_size, fprs) -> list[dict]:
    """The rephrase-GT best zero-shot picks, scored on the trained models' own
    eval rows: each seed's 70/30 test partition per rephrase corpus (mean +- SD
    over seeds, rephrase-as-human label), and the full corpus per OOD split
    (normal GT, one deterministic reading). One cell per (detector, corpus, fpr)."""
    picked = _pick_columns(config)
    # detector -> fpr -> list of per-corpus cells
    per_det: dict[str, dict[float, list[dict]]] = {}

    def _add(det: str, f: float, cell: dict) -> None:
        per_det.setdefault(det, {}).setdefault(f, []).append(cell)

    for split_key in reph_splits:
        c = CORPUS_SOURCES[split_key]
        avail: set[str] | None = None
        for pf in (c["full"], c["reph"]):
            s = set(pq.read_schema(REPH_PARQUET_DIR / pf).names)
            avail = s if avail is None else (avail & s)
        specs = resolve_specs(avail, config, columns=picked)
        if not specs:
            continue
        _, y, _, _, groups, raw = load_reph_features(
            split_key, score_cols=tuple(col for col, _, _ in specs))
        te_by_seed = [grouped_split(y, groups, test_size, s)[1] for s in seeds]
        n_te = int(len(te_by_seed[0]))
        for col, det, machine_is_high in specs:
            o = orient(raw[col], machine_is_high)
            reads = [eval_column_fprs(o[te], y[te], fprs) for te in te_by_seed]
            for f in fprs:
                cell = agg_cell([r[0] for r in reads],
                                [r[1][f][0] for r in reads],
                                [r[1][f][1] for r in reads], "hmean")
                cell.update(split=split_key, role="eval", fpr=f, n_test=n_te,
                            n_seeds=len(seeds))
                _add(det, f, cell)

    for name, path in ood_splits:
        for r in zeroshot_compute([(name, path)], ZS_FAMILIES, picked, fprs, "human"):
            if r["split"] == "mean":
                continue
            _add(_detector(r["family"], r["version"]), r["fpr"], {
                "split": r["split"], "role": "ood", "fpr": r["fpr"],
                "n_test": r["n"], "n_seeds": len(seeds),
                "auroc": r["auroc"], "auroc_std": float("nan"),
                "tpr_at_fpr": r["tpr_at_fpr"], "tpr_at_fpr_std": float("nan"),
                "tpr_eff": r.get("tpr_eff", ""), "tpr_eff_std": float("nan"),
                "goodness": float("nan"), "goodness_std": float("nan"),
            })

    out: list[dict] = []
    for det, by_fpr in per_det.items():
        for f, cells in by_fpr.items():
            for cc in cells:
                out.append(_row(det, cc["split"], cc["role"], cc, kind="zeroshot"))
            mean = overall_cell(cells, f)
            mean.update(n_seeds=len(seeds), fpr=f)
            out.append(_row(det, "mean", "overall", mean, kind="zeroshot"))
    return out


def plots_from_csv(csv_path, meta, out_dir) -> None:
    from epai.ai_detection.trainv2.plots import zeroshot_scatter

    rows = pd.read_csv(csv_path).to_dict("records")
    out_dir.mkdir(parents=True, exist_ok=True)
    ood = "/".join(meta["ood"])
    for fpr in meta["fprs"]:
        frows = rows_at_fpr(rows, fpr)
        zeroshot_scatter(
            frows, {**meta, "fpr": fpr},
            out_dir / f"in_domain_{fpr_tag(fpr)}.png", key="detector",
            title=f"Rephrase in-domain trained (70/30, {meta['n_seeds']} seeds) — "
                  f"rephrase counted as human — held-out test per corpus, plus "
                  f"{ood} OOD, vs. zero-shot picks\nAUROC vs. oracle "
                  f"TPR@{fpr * 100:g}%FPR",
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--splits", nargs="+", default=None, metavar="KEY",
                    help=f"rephrase training corpora (default: {REPH_SPLITS})")
    ap.add_argument("--ood-splits", nargs="+", default=None, metavar="KEY",
                    help=f"eval-only corpora, never trained on (default: {DEFAULT_OOD})")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS,
                    metavar="LEVEL",
                    help=f"oracle FPR level(s); one figure set per level "
                         f"(default: {DEFAULT_FPRS})")
    ap.add_argument("--zeroshot-config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV")
    args = ap.parse_args()

    reph_splits = args.splits or REPH_SPLITS
    bad = [s for s in reph_splits if s not in REPH_SPLITS]
    if bad:
        raise SystemExit(f"unknown rephrase corpus {bad}; choose from {REPH_SPLITS}")
    ood_splits = resolve_splits(args.ood_splits or DEFAULT_OOD, PARQUET_DIR)
    meta = {"fprs": args.fpr, "n_seeds": len(args.seeds),
            "ood": [s for s, _ in ood_splits],
            "splits": [*reph_splits] + [s for s, _ in ood_splits]}

    if args.replot is not None:
        plots_from_csv(args.replot, meta, args.replot.parent)
        print(f"replotted from {args.replot}")
        return 0

    missing = [str(p) for _, p in ood_splits if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")

    print(f"trained : {[s['label'] for s in TRAINED_SPECS.values()]}")
    print(f"train on: {reph_splits}   OOD eval: {[s for s, _ in ood_splits]}")
    print("label   : y=1 the full gpt56luna generation; human AND rephrase -> y=0")
    print(f"seeds   : {args.seeds}   split: {int(round((1 - args.test_size) * 100))}/"
          f"{int(round(args.test_size * 100))} grouped on 'id'\n")

    rows = compute_trained(reph_splits, ood_splits, args.seeds, args.test_size, args.fpr)
    if not args.no_zeroshot:
        rows += compute_zeroshot(args.zeroshot_config, reph_splits, ood_splits,
                                 args.seeds, args.test_size, args.fpr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "performance.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    _print_table(rows, args.fpr)
    print(f"\nwrote {csv_path}")

    if not args.no_plots:
        plots_from_csv(csv_path, meta, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

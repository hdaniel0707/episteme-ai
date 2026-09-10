r"""Leave-one-domain-out (LODO) trained detection on the rephrase task, with
rephrased text counted as HUMAN ground truth -- otherwise identical to
``run_lodo_performance.py``.

Same three trained detectors as ``rephrase_in_domain_performance.py``
(``emb_roberta`` -> logistic, and the two ``emb_roberta`` + gpt56luna-judge-cue
hybrids -- no gemma). Within each rephrase corpus (MG-GB = MG-GB + MG-Reph-GB,
MG-Sci = MG-Sci + MG-Reph-Sci), each of its domains is held out in turn:

    for each domain d:
        train on every OTHER domain's rows (label below)
        score the held-out domain d's rows in full -- the only rows this fold's
        model never saw

    y = 1   the full gpt56luna generation
    y = 0   human AND rephrase (gpt56luna:0908R)

The split is group-wise: a document's human / rephrase / full-generation rows
share one ``id`` and one domain, so a fold never straddles them. Each fold is
averaged over N seeds; the per-corpus number is the mean over the domain folds,
its SD the spread ACROSS those folds. Every fold's model is also scored on each
``--ood-splits`` corpus (default ``mg_den``). The rephrase-GT best zero-shot
picks (``best_zeroshot.yaml`` minus any gemma judge) are scored on the SAME rows
as the trained models -- each held-out domain fold per rephrase corpus (a
per-corpus mean +- across-fold SD), the full corpus per OOD split (normal GT,
no error bar) -- not on the whole dataset.

compute -> CSV -> plot from the CSV:

    results/trainv2/rephrase_lodo_performance/
        performance.csv
        lodo.png

Run:
    uv run python epai/ai_detection/trainv2/rephrase_lodo_performance.py
    uv run python epai/ai_detection/trainv2/rephrase_lodo_performance.py --seeds 0 1 2
    uv run python epai/ai_detection/trainv2/rephrase_lodo_performance.py \
        --replot results/trainv2/rephrase_lodo_performance/performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import pyarrow.parquet as pq

from epai.ai_detection.trainv2._reph_trained import REPH_SPLITS, load_reph_features
from epai.ai_detection.trainv2._trained import TRAINED_SPECS
from epai.ai_detection.trainv2._zeroshot import (
    agg_cell,
    eval_column,
    orient,
    overall_cell,
    resolve_specs,
)
from epai.ai_detection.trainv2.core import eff_fpr_level, goodness
from epai.ai_detection.trainv2.plot_score_histograms import CORPUS_SOURCES
from epai.ai_detection.trainv2.plot_score_histograms import PARQUET_DIR as REPH_PARQUET_DIR
from epai.ai_detection.trainv2.rephrase_best_zeroshot_performance import _pick_columns
from epai.ai_detection.trainv2.run_best_zeroshot_performance import DEFAULT_CONFIG, _detector
from epai.ai_detection.trainv2.run_lodo_performance import (
    CSV_FIELDS,
    _fold,
    _load_ood,
    _print_table,
    _std,
    _zs_row,
)
from epai.ai_detection.trainv2.run_zeroshot_performance import FAMILIES as ZS_FAMILIES
from epai.ai_detection.trainv2.run_zeroshot_performance import compute as zeroshot_compute
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE / "data" / "parquet"
DEFAULT_OUT_DIR = BASE / "results" / "trainv2" / "rephrase_lodo_performance"
DEFAULT_OOD = ["mg_den"]


def compute_trained(reph_splits, ood_splits, seeds, fpr, min_domain_rows) -> list[dict]:
    ood_sets = {name: _load_ood(path, "human") for name, path in ood_splits}
    for name, (_, oy, _, n_h) in ood_sets.items():
        print(f"  OOD eval corpus {name!r}: {len(oy)} rows, {n_h} human (never trained on)")

    all_folds: dict[str, dict[str, list[dict]]] = {}
    ood_reads: dict[str, dict[str, list]] = {
        k: {n: [] for n in ood_sets} for k in TRAINED_SPECS
    }

    for split_key in tqdm(reph_splits, desc="LODO", unit="split"):
        X, y, Z, dom, ids = load_reph_features(split_key)
        counts = {d: int((dom == d).sum()) for d in np.unique(dom)}
        domains = sorted(d for d, n in counts.items()
                         if n >= min_domain_rows and np.unique(y[dom == d]).size == 2)
        if len(domains) < 2:
            print(f"  {split_key}: only {len(domains)} usable domain(s) ({counts}) — skipped")
            continue
        print(f"\n  {split_key}: {len(domains)} domains {domains}")

        spec_folds = {k: [] for k in TRAINED_SPECS}
        for d in domains:
            tr, te = dom != d, dom == d
            assert d not in set(dom[tr]) and set(dom[te]) == {d}
            assert tr.sum() + te.sum() == len(dom) and not (tr & te).any()
            assert set(ids[tr]).isdisjoint(ids[te]), "a group straddled the fold"
            print(f"    hold out {d!r:<22} train {sorted(set(dom[tr]))} "
                  f"(n={tr.sum()}, {len(set(ids[tr]))} groups, y0 {int((y[tr] == 0).sum())})"
                  f"  eval {d!r} (n={te.sum()}, {len(set(ids[te]))} groups, "
                  f"y0 {int((y[te] == 0).sum())})")
            for spec_key, spec in TRAINED_SPECS.items():
                f = _fold(spec, X, y, Z, ids, tr, te, seeds, fpr, ood_sets)
                f["held_out_domain"] = d
                for name, reads in f.pop("ood").items():
                    ood_reads[spec_key][name].extend(reads)
                spec_folds[spec_key].append(f)
        all_folds[split_key] = spec_folds

    if not all_folds:
        return []

    rows: list[dict] = []
    for spec_key, spec in TRAINED_SPECS.items():
        detector = f"trained/{spec['label']}"
        split_rows: list[dict] = []
        for split_key, spec_folds in all_folds.items():
            folds = spec_folds[spec_key]
            for f in folds:
                rows.append({
                    "level": "fold", "kind": "trained", "detector": detector,
                    "split": split_key, "held_out_domain": f["held_out_domain"],
                    "n_test": f["n_test"], "n_folds": "", "n_seeds": len(seeds),
                    "auroc": f["auroc"], "auroc_std": "",
                    "tpr_at_fpr": f["tpr_at_fpr"], "tpr_at_fpr_std": "",
                })
            ok = [f for f in folds if f["n_test_human"] >= 1 / fpr]
            sr = {
                "level": "split", "kind": "trained", "detector": detector,
                "split": split_key, "held_out_domain": "", "n_test": "",
                "n_folds": len(folds), "n_seeds": len(seeds), "reachable": bool(ok),
                "auroc": float(np.mean([f["auroc"] for f in folds])),
                "auroc_std": _std([f["auroc"] for f in folds]),
                "tpr_at_fpr": float(np.mean([f["tpr_at_fpr"] for f in (ok or folds)])),
                "tpr_at_fpr_std": _std([f["tpr_at_fpr"] for f in (ok or folds)]),
            }
            rows.append(sr)
            split_rows.append(sr)

        for name, (_, _, _, n_h) in ood_sets.items():
            reads = ood_reads[spec_key][name]
            oa, ot = [r[0] for r in reads], [r[1] for r in reads]
            sr = {
                "level": "split", "kind": "trained", "detector": detector,
                "split": name, "held_out_domain": "", "n_test": "",
                "n_folds": len(reads), "n_seeds": len(seeds),
                "reachable": bool(n_h >= 1 / fpr),
                "auroc": float(np.mean(oa)), "auroc_std": _std(oa),
                "tpr_at_fpr": float(np.mean(ot)), "tpr_at_fpr_std": _std(ot),
            }
            rows.append(sr)
            split_rows.append(sr)

        tpr_ok = [r for r in split_rows if r.get("reachable")]
        rows.append({
            "level": "overall", "kind": "trained", "detector": detector,
            "split": "mean", "held_out_domain": "", "n_test": "",
            "n_folds": sum(len(all_folds[s][spec_key]) for s in all_folds),
            "n_seeds": len(seeds),
            "auroc": float(np.mean([r["auroc"] for r in split_rows])),
            "auroc_std": float(np.mean([r["auroc_std"] for r in split_rows])),
            "tpr_at_fpr": float(np.mean([r["tpr_at_fpr"] for r in tpr_ok])) if tpr_ok else float("nan"),
            "tpr_at_fpr_std": float(np.mean([r["tpr_at_fpr_std"] for r in tpr_ok])) if tpr_ok else float("nan"),
        })
    return rows


def compute_zeroshot(config, reph_splits, ood_splits, seeds, fpr,
                     min_domain_rows) -> list[dict]:
    """The rephrase-GT best zero-shot picks, scored on the trained models' own
    eval rows: each held-out domain fold per rephrase corpus (per-corpus reading
    = mean +- SD across folds), and the full corpus per OOD split (normal GT,
    one deterministic reading). ``fold`` / ``split`` / ``overall`` rows."""
    picked = _pick_columns(config)
    n_seeds = len(seeds)
    per_det: dict[str, list[dict]] = {}
    rows: list[dict] = []

    for split_key in reph_splits:
        c = CORPUS_SOURCES[split_key]
        avail: set[str] | None = None
        for pf in (c["full"], c["reph"]):
            s = set(pq.read_schema(REPH_PARQUET_DIR / pf).names)
            avail = s if avail is None else (avail & s)
        specs = resolve_specs(avail, config, columns=picked)
        if not specs:
            continue
        _, y, _, dom, _, raw = load_reph_features(
            split_key, score_cols=tuple(col for col, _, _ in specs))
        counts = {d: int((dom == d).sum()) for d in np.unique(dom)}
        domains = sorted(d for d, n in counts.items()
                         if n >= min_domain_rows
                         and np.unique(y[dom == d]).size == 2)
        if len(domains) < 2:
            continue
        for col, det, machine_is_high in specs:
            o = orient(raw[col], machine_is_high)
            fold_reads = []
            for d in domains:
                te = dom == d
                a, t, te_ = eval_column(o[te], y[te], fpr)
                fold_reads.append((a, t, te_))
                gd = (goodness(a, te_, "hmean")
                      if np.isfinite(a) and np.isfinite(te_) else float("nan"))
                rows.append(_zs_row(det, "fold", split_key, d, {
                    "auroc": a, "tpr_at_fpr": t, "tpr_eff": te_,
                    "eff_fpr": eff_fpr_level(int((y[te] == 0).sum()), fpr),
                    "goodness": gd, "n_test": int(te.sum())}, "", n_seeds))
            cell = agg_cell([r[0] for r in fold_reads], [r[1] for r in fold_reads],
                            [r[2] for r in fold_reads], "hmean")
            cell["reachable"] = any(
                int((y[dom == d] == 0).sum()) >= 1 / fpr for d in domains)
            cell["eff_fpr"] = float(np.mean(
                [eff_fpr_level(int((y[dom == d] == 0).sum()), fpr) for d in domains]))
            rows.append(_zs_row(det, "split", split_key, "", cell,
                                len(domains), n_seeds))
            per_det.setdefault(det, []).append({**cell, "split": split_key})

    for name, path in ood_splits:
        for r in zeroshot_compute([(name, path)], ZS_FAMILIES, picked, fpr, "human"):
            if r["split"] == "mean":
                continue
            det = _detector(r["family"], r["version"])
            cell = {
                "auroc": r["auroc"], "auroc_std": float("nan"),
                "tpr_at_fpr": r["tpr_at_fpr"], "tpr_at_fpr_std": float("nan"),
                "tpr_eff": r.get("tpr_eff", ""), "tpr_eff_std": float("nan"),
                "goodness": float("nan"), "goodness_std": float("nan"),
                "eff_fpr": r.get("eff_fpr", ""), "n_test": r["n"],
                "reachable": bool(r.get("reachable")),
            }
            rows.append(_zs_row(det, "split", r["split"], "", cell, "", n_seeds))
            per_det.setdefault(det, []).append({**cell, "split": r["split"]})

    for det, cells in per_det.items():
        rows.append(_zs_row(det, "overall", "mean", "", overall_cell(cells, fpr),
                            "", n_seeds))
    return rows


def plots_from_csv(csv_path: Path, meta: dict, out_dir: Path) -> None:
    from epai.ai_detection.trainv2.plots import zeroshot_scatter

    rows = [r for r in pd.read_csv(csv_path).to_dict("records")
            if r["level"] in ("split", "overall")]
    out_dir.mkdir(parents=True, exist_ok=True)
    ood = "/".join(meta["ood"])
    zeroshot_scatter(
        rows, meta, out_dir / "lodo.png", key="detector",
        title=f"Rephrase leave-one-domain-out trained — rephrase counted as "
              f"human — plus {ood} OOD, vs. zero-shot picks\nAUROC vs. oracle "
              f"TPR@{meta['fpr'] * 100:g}%FPR — trained mean error bar = mean per-corpus SD",
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--splits", nargs="+", default=None, metavar="KEY",
                    help=f"rephrase corpora to LODO over (default: {REPH_SPLITS})")
    ap.add_argument("--ood-splits", nargs="+", default=None, metavar="KEY",
                    help=f"eval-only corpora, never trained on (default: {DEFAULT_OOD})")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--min-domain-rows", type=int, default=100)
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
    meta = {"fpr": args.fpr, "n_seeds": len(args.seeds),
            "ood": [s for s, _ in ood_splits],
            "splits": [*reph_splits] + [s for s, _ in ood_splits]}

    if args.replot is not None:
        plots_from_csv(args.replot, meta, args.replot.parent)
        print(f"replotted from {args.replot}")
        return 0

    missing = [str(p) for _, p in ood_splits if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")

    print(f"trained : {[s['label'] for s in TRAINED_SPECS.values()]}, leave-one-domain-out")
    print(f"train on: {reph_splits}   OOD eval: {[s for s, _ in ood_splits]}   seeds: {args.seeds}")
    print("label   : y=1 the full gpt56luna generation; human AND rephrase -> y=0\n")

    rows = compute_trained(reph_splits, ood_splits, args.seeds, args.fpr,
                           args.min_domain_rows)
    if not any(r["level"] == "overall" for r in rows):
        raise SystemExit("no corpus had >= 2 usable domains")
    if not args.no_zeroshot:
        rows += compute_zeroshot(args.zeroshot_config, reph_splits, ood_splits,
                                 args.seeds, args.fpr, args.min_domain_rows)

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

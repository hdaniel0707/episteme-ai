r"""In-domain trained detection across the MANUGRAM splits, with MG-Den
as an eval-only OOD corpus and the zero-shot picks for reference — AUROC vs.
oracle TPR@1%FPR on one figure.

Three trained detectors (:mod:`epai.ai_detection.trainv2._trained`): a plain
logistic probe on ``emb_roberta``, and the two ``emb_roberta`` + judge
cue-vector hybrids -- concatenation fusion and a learned gate. Within each
TRAINING split (default GB-orig / MG-GB / MG-Sci): a grouped 70/30
train/test split (a document and every rewrite of it share an ``id`` and never
straddle the cut), repeated over N seeds, human vs. every AI row pooled.
AUROC / TPR@1%FPR are the seed mean.

Every fitted model is ALSO scored, unchanged, on each ``--ood-splits`` corpus
in full (default MG-Den) — never trained on, an additional held-out
test corpus. Its reading is the mean over every (training split, seed) model.

The zero-shot picks from ``cfg/ai_detection/trainv2/best_zeroshot.yaml`` are
scored on the SAME rows as the trained models -- each seed's 70/30 test
partition per training split, the full corpus per OOD split -- not on the whole
dataset. So a per-training-split zero-shot point is a seed mean +- SD, exactly
like a trained one, and the two detector kinds are compared on identical data.
The OOD zero-shot point is a single deterministic full-corpus reading (no
resampling -> no error bar).

Every per-corpus point now carries its seed SD (zero-shot on the OOD split
excepted). The 'overall' row is the mean over the in-domain readings AND the
OOD readings; ITS SD is the typical per-corpus SD (mean of them).

The goodness bar folds AUROC + oracle TPR into one score PER SEED, so its bar
also carries a seed SD (``goodness`` / ``goodness_std`` in the CSV).

``--fpr`` takes one or more oracle FPR levels (default 1% and 5%). The models
are trained once; each level is just a different ROC readout. The CSV carries an
``fpr`` column (one row per detector/corpus/level) and each level gets its own
figure set, suffixed ``_1pct`` / ``_5pct``.

MG-Den has < 100 humans, so its 1%-FPR point is a floor: kept in the AUROC mean,
left out of the TPR mean. At 5% FPR it clears the floor (needs only 20 humans).

compute -> CSV -> plot from the CSV:

    results/trainv2/in_domain_performance/
        performance.csv   kind, detector, split, role, fpr, n_train, n_test,
                          n_seeds, auroc(_std), tpr_at_fpr(_std), tpr_eff(_std),
                          eff_fpr, goodness(_std)
        in_domain_1pct.png / in_domain_5pct.png   (+ _goodness_, _denario_)

Run:
    uv run python epai/ai_detection/trainv2/run_in_domain_performance.py
    uv run python epai/ai_detection/trainv2/run_in_domain_performance.py \
        --splits mg_gb mg_sci --ood-splits mg_den --seeds 0 1 2 3 4
    uv run python epai/ai_detection/trainv2/run_in_domain_performance.py \
        --replot results/trainv2/in_domain_performance/performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

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
from epai.ai_detection.trainv2.core import (
    eff_fpr_level,
    fpr_tag,
    goodness,
    grouped_split,
    rows_at_fpr,
)
from epai.ai_detection.trainv2.run_best_zeroshot_performance import DEFAULT_CONFIG
from epai.ai_detection.trainv2.run_zeroshot_performance import DEFAULT_FPRS
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "in_domain_performance"

DEFAULT_SPLITS = ["gb_orig", "mg_gb", "mg_sci"]
DEFAULT_OOD = ["mg_den"]

CSV_FIELDS = ["kind", "detector", "split", "role", "fpr", "n_train", "n_test",
              "n_seeds", "auroc", "auroc_std", "tpr_at_fpr", "tpr_at_fpr_std",
              "tpr_eff", "tpr_eff_std", "eff_fpr", "goodness", "goodness_std"]


def _std(v: list[float]) -> float:
    v = [x for x in v if np.isfinite(x)]
    return float(np.std(v)) if len(v) > 1 else 0.0


def _row(detector, split, role, cell, kind: str = "trained") -> dict:
    return {
        "kind": kind, "detector": detector, "split": split, "role": role,
        "fpr": cell.get("fpr", ""),
        "n_train": cell.get("n_train", ""), "n_test": cell.get("n_test", ""),
        "n_seeds": cell["n_seeds"],
        "auroc": cell["auroc"], "auroc_std": cell["auroc_std"],
        "tpr_at_fpr": cell["tpr_at_fpr"], "tpr_at_fpr_std": cell["tpr_at_fpr_std"],
        "tpr_eff": cell.get("tpr_eff", ""), "tpr_eff_std": cell.get("tpr_eff_std", ""),
        "eff_fpr": cell.get("eff_fpr", ""),
        "goodness": cell.get("goodness", ""), "goodness_std": cell.get("goodness_std", ""),
    }


def _overall_cell(cells: list[dict], n_seeds: int, fpr: float) -> dict:
    """Mean over every per-corpus cell; the TPR@fpr mean skips a corpus whose
    point is unreachable (< 1/fpr humans), the eff-FPR TPR mean uses them all.
    SD is the mean per-corpus SD."""
    ok = [c for c in cells if c["n_test_human"] >= 1 / fpr]
    eff = [c for c in cells if "tpr_eff" in c]
    good = [c for c in cells if "goodness" in c]
    return {
        "n_seeds": n_seeds, "fpr": fpr,
        "auroc": float(np.mean([c["auroc"] for c in cells])),
        "auroc_std": float(np.mean([c["auroc_std"] for c in cells])),
        "tpr_at_fpr": float(np.mean([c["tpr_at_fpr"] for c in ok])) if ok else float("nan"),
        "tpr_at_fpr_std": float(np.mean([c["tpr_at_fpr_std"] for c in ok])) if ok else float("nan"),
        "tpr_eff": float(np.mean([c["tpr_eff"] for c in eff])) if eff else float("nan"),
        "tpr_eff_std": float(np.mean([c["tpr_eff_std"] for c in eff])) if eff else float("nan"),
        "goodness": float(np.mean([c["goodness"] for c in good])) if good else float("nan"),
        "goodness_std": float(np.mean([c["goodness_std"] for c in good])) if good else float("nan"),
        "eff_fpr": fpr,
    }


def compute_trained(train_splits, ood_splits, seeds, test_size, fprs,
                    group_col, human_label, combine) -> list[dict]:
    ood_sets: dict[str, tuple] = {}
    for name, path in ood_splits:
        oX, oy, oZ, _ = load_features(path, group_col, human_label)
        ood_sets[name] = (oX, oy, oZ, int((oy == 0).sum()))
        print(f"  OOD eval corpus {name!r}: {len(oy)} rows, {int((oy == 0).sum())} "
              f"human (never trained on)")

    # per spec -> per fpr: in-domain cells (one per training corpus) + OOD reads
    cells: dict[str, dict[float, list[dict]]] = {
        k: {f: [] for f in fprs} for k in TRAINED_SPECS}
    ood_reads: dict[str, dict[str, dict[float, list]]] = {
        k: {n: {f: [] for f in fprs} for n in ood_sets} for k in TRAINED_SPECS}

    for split_key, path in tqdm(train_splits, desc="corpus", unit="split"):
        X, y, Z, groups = load_features(path, group_col, human_label)
        g0, g1 = grouped_split(y, groups, test_size, seeds[0])
        assert set(groups[g0]).isdisjoint(groups[g1])
        print(f"  {split_key}: grouped {int(round((1 - test_size) * 100))}/"
              f"{int(round(test_size * 100))} on {group_col!r} — {len(set(groups))} "
              f"groups -> {len(set(groups[g0]))} train / {len(set(groups[g1]))} test, disjoint")

        for spec_key, spec in TRAINED_SPECS.items():
            a: list[float] = []
            t = {f: [] for f in fprs}
            te_ = {f: [] for f in fprs}
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
                        ood_reads[spec_key][name][f].append(
                            (oau, oper[f][0], oper[f][1]))
            for f in fprs:
                g = [goodness(ai, ei, combine) for ai, ei in zip(a, te_[f])]
                cells[spec_key][f].append({
                    "split": split_key, "fpr": f, "n_train": n_tr, "n_test": n_te,
                    "n_test_human": n_te_h, "n_seeds": len(seeds),
                    "auroc": float(np.mean(a)), "auroc_std": _std(a),
                    "tpr_at_fpr": float(np.mean(t[f])), "tpr_at_fpr_std": _std(t[f]),
                    "tpr_eff": float(np.mean(te_[f])), "tpr_eff_std": _std(te_[f]),
                    "goodness": float(np.mean(g)), "goodness_std": _std(g),
                    "eff_fpr": eff_fpr_level(n_te_h, f)})

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


def compute_zeroshot(train_splits, ood_splits, config, seeds, test_size, fprs,
                     group_col, human_label, combine) -> list[dict]:
    """The ``best_zeroshot.yaml`` picks, scored on the trained models' own eval
    rows: each seed's 70/30 grouped TEST partition per training split (mean +-
    SD over seeds), and the full corpus per OOD split (one deterministic
    reading). One cell per (detector, corpus, fpr); same per-corpus / 'mean'
    row shape as ``compute_trained``."""
    import pyarrow.parquet as pq

    # detector -> fpr -> list of per-corpus cells
    per_det: dict[str, dict[float, list[dict]]] = {}

    def _add(det: str, f: float, cell: dict) -> None:
        per_det.setdefault(det, {}).setdefault(f, []).append(cell)

    for split_key, path in train_splits:
        specs = resolve_specs(pq.read_schema(path).names, config)
        if not specs:
            continue
        _, y, _, groups, raw = load_features(
            path, group_col, human_label,
            score_cols=tuple(c for c, _, _ in specs))
        te_by_seed = [grouped_split(y, groups, test_size, s)[1] for s in seeds]
        n_te = int(len(te_by_seed[0]))
        n_te_h = int((y[te_by_seed[0]] == 0).sum())
        for col, det, machine_is_high in specs:
            o = orient(raw[col], machine_is_high)
            reads = [eval_column_fprs(o[te], y[te], fprs) for te in te_by_seed]
            for f in fprs:
                cell = agg_cell([r[0] for r in reads],
                                [r[1][f][0] for r in reads],
                                [r[1][f][1] for r in reads], combine)
                cell.update(split=split_key, role="eval", fpr=f, n_test=n_te,
                            n_seeds=len(seeds), eff_fpr=eff_fpr_level(n_te_h, f),
                            reachable=n_te_h >= 1 / f)
                _add(det, f, cell)

    for split_key, path in ood_splits:
        specs = resolve_specs(pq.read_schema(path).names, config)
        if not specs:
            continue
        _, y, _, _, raw = load_features(
            path, group_col, human_label,
            score_cols=tuple(c for c, _, _ in specs))
        n_h = int((y == 0).sum())
        for col, det, machine_is_high in specs:
            au, per = eval_column_fprs(orient(raw[col], machine_is_high), y, fprs)
            for f in fprs:
                t, te = per[f]
                gd = (goodness(au, te, combine)
                      if np.isfinite(au) and np.isfinite(te) else float("nan"))
                _add(det, f, {
                    "split": split_key, "role": "ood", "fpr": f,
                    "n_test": int(len(y)), "n_seeds": len(seeds),
                    "auroc": au, "auroc_std": float("nan"),
                    "tpr_at_fpr": t, "tpr_at_fpr_std": float("nan"),
                    "tpr_eff": te, "tpr_eff_std": float("nan"),
                    "goodness": gd, "goodness_std": float("nan"),
                    "eff_fpr": eff_fpr_level(n_h, f), "reachable": n_h >= 1 / f,
                })

    out: list[dict] = []
    for det, by_fpr in per_det.items():
        for f, cells in by_fpr.items():
            for c in cells:
                out.append(_row(det, c["split"], c["role"], c, kind="zeroshot"))
            mean = overall_cell(cells, f)
            mean.update(n_seeds=len(seeds), fpr=f)
            out.append(_row(det, "mean", "overall", mean, kind="zeroshot"))
    return out


def plots_from_csv(csv_path: Path, meta: dict, out_dir: Path) -> None:
    rows = pd.read_csv(csv_path).to_dict("records")
    from epai.ai_detection.trainv2.plots import (
        denario_scatter,
        goodness_bars,
        zeroshot_scatter,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    ood = "/".join(meta["ood"])
    n_seeds = meta["n_seeds"]
    for fpr in meta["fprs"]:
        fmeta = {**meta, "fpr": fpr}
        tag = fpr_tag(fpr)
        frows = rows_at_fpr(rows, fpr)
        zeroshot_scatter(
            frows, fmeta, out_dir / f"in_domain_{tag}.png", key="detector",
            title=f"In-domain (70/30, {n_seeds} seeds) — trained vs. zero-shot "
                  f"picks on the same held-out test rows, plus {ood} as eval-only "
                  f"OOD\nAUROC vs. oracle TPR@{fpr * 100:g}%FPR",
        )
        goodness_bars(
            frows, fmeta, out_dir / f"in_domain_goodness_{tag}.png", key="detector",
            combine=meta.get("combine", "hmean"),
            title=f"In-domain trained (70/30, {n_seeds} seeds) + zero-shot picks — "
                  f"goodness (TPR@{fpr * 100:g}%FPR) across the MANUGRAM splits",
        )
        denario_scatter(
            frows, fmeta, out_dir / f"in_domain_denario_{tag}.png",
            title=f"In-domain trained (70/30, {n_seeds} seeds) on {ood} "
                  f"(eval-only OOD)\nsame detectors — AUROC vs. oracle "
                  f"TPR@{fpr * 100:g}%FPR",
        )


def _print_table(rows: list[dict], fprs: list[float]) -> None:
    print(f"\n{'kind':<9}{'detector':<24}{'split':<9}{'role':<10}{'FPR':>6}"
          f"{'auroc':>15}{'TPR':>15}")
    for r in rows:
        def fmt(v, sd):
            if not np.isfinite(v):
                return "--"
            bare = (sd in ("", None) or
                    (isinstance(sd, float) and not np.isfinite(sd)))
            return f"{v:.3f}" if bare else f"{v:.3f} ± {sd:.3f}"
        mark = "  <<" if r["split"] == "mean" else ""
        print(f"{r['kind']:<9}{r['detector']:<24}{r['split']:<9}{r['role']:<10}"
              f"{float(r['fpr']) * 100:>5g}%"
              f"{fmt(r['auroc'], r['auroc_std']):>15}"
              f"{fmt(r['tpr_at_fpr'], r['tpr_at_fpr_std']):>15}{mark}")


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
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS,
                    metavar="LEVEL",
                    help=f"oracle FPR level(s); one figure set per level "
                         f"(default: {DEFAULT_FPRS})")
    ap.add_argument("--combine", choices=("hmean", "gmean", "amean"), default="hmean",
                    help="goodness-bar score = this mean of Gini and oracle TPR")
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--zeroshot-config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV")
    args = ap.parse_args()

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

    print(f"trained : {[s['label'] for s in TRAINED_SPECS.values()]}")
    print(f"train on: {[s for s, _ in train_splits]}   OOD eval: {[s for s, _ in ood_splits]}")
    print(f"seeds   : {args.seeds}   split: {int(round((1 - args.test_size) * 100))}/"
          f"{int(round(args.test_size * 100))} grouped on {args.group_col!r}\n")

    rows = compute_trained(train_splits, ood_splits, args.seeds, args.test_size,
                           args.fpr, args.group_col, args.human_label, args.combine)
    if not args.no_zeroshot:
        print(f"\nzero-shot: {args.zeroshot_config} "
              f"(scored on the trained models' own eval rows)")
        rows += compute_zeroshot(train_splits, ood_splits, args.zeroshot_config,
                                 args.seeds, args.test_size, args.fpr,
                                 args.group_col, args.human_label, args.combine)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "performance.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    _print_table(rows, args.fpr)
    print(f"\nwrote {csv_path}")

    if not args.no_plots:
        plots_from_csv(csv_path, meta, args.out_dir)


if __name__ == "__main__":
    main()

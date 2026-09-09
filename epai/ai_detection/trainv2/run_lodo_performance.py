r"""Leave-one-domain-out (LODO) trained detection across the MANUGRAM splits,
with the zero-shot picks for reference — AUROC vs. oracle TPR@1%FPR.

The LODO sibling of ``run_in_domain_performance.py``. Same three trained
detectors (:mod:`epai.ai_detection.trainv2._trained`: ``emb_roberta`` ->
logistic, and the two ``emb_roberta`` + judge-cue hybrids). But instead of a
random 70/30 split, each of a split's domains is held out in turn:

    for each domain d in the split:
        train on every OTHER domain's rows (human vs. all AI, the full union)
        score the held-out domain d's rows — the only rows this fold's model
        never saw — for AUROC and oracle TPR@1%FPR

The split is group-wise: a document and every rewrite of it share one ``id``
and one domain (checked on load), so holding out a domain holds out those
whole groups — a (human, gen1..genN) set is never split across the fold.

Every fold's model is ALSO scored, unchanged, on each ``--ood-splits`` corpus
in full (default MG-Den) — never trained on, an additional held-out
test corpus. Its reading is the mean over every (training split, held-out
domain, seed) model.

Each fold is averaged over N seeds (only the logistic ``random_state`` moves,
so this just steadies the fold reading). The reported per-split number is the
mean over the domain folds; **its SD is the spread ACROSS those folds** — how
much the reading depends on which domain is unseen, i.e. the domain-transfer
robustness. The 'overall' row is the mean across splits; its SD is the typical
per-split fold SD (mean of them), never the spread across splits.

Only splits with >= 2 domains of at least ``--min-domain-rows`` rows are used
(MG-Den's domains are ~15 rows each — no LODO there), so the default
is GB-orig / MG-GB / MG-Sci, 3 domains each.

The zero-shot picks from ``cfg/ai_detection/trainv2/best_zeroshot.yaml`` are
scored on each full split (no folding — nothing is trained) and overlaid for
reference, with no error bars.

compute -> CSV -> plot from the CSV:

    results/trainv2/lodo_performance/
        performance.csv   level (fold/split/overall), kind, detector, split,
                          held_out_domain, n_test, n_folds, n_seeds,
                          auroc, auroc_std, tpr_at_fpr, tpr_at_fpr_std
        lodo.png

Run:
    uv run python epai/ai_detection/trainv2/run_lodo_performance.py
    uv run python epai/ai_detection/trainv2/run_lodo_performance.py --seeds 0 1 2
    uv run python epai/ai_detection/trainv2/run_lodo_performance.py \
        --replot results/trainv2/lodo_performance/performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from epai.ai_detection.trainv2._trained import (
    AUX_COL,
    EMB_COL,
    TRAINED_SPECS,
    _stack,
    eval_on,
    fit,
)
from epai.ai_detection.trainv2.data import GEN_COL, _is_missing
from epai.ai_detection.trainv2.run_best_zeroshot_performance import (
    DEFAULT_CONFIG,
    _detector,
    _load_columns,
)
from epai.ai_detection.trainv2.run_zeroshot_performance import FAMILIES
from epai.ai_detection.trainv2.run_zeroshot_performance import compute as zeroshot_compute
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "lodo_performance"
DEFAULT_SPLITS = ["gb_orig", "mg_gb", "mg_sci"]
DEFAULT_OOD = ["mg_den"]

DOMAIN_COL = "domain"
GROUP_COL = "id"

CSV_FIELDS = ["level", "kind", "detector", "split", "held_out_domain", "n_test",
              "n_folds", "n_seeds", "auroc", "auroc_std", "tpr_at_fpr",
              "tpr_at_fpr_std"]


def _load(path: Path, human_label: str):
    """``(X_emb, y, Z_cue, dom, ids)``. Group dropped whole on ANY missing
    feature; a group must sit in one domain (checked)."""
    df = pd.read_parquet(path, columns=[GROUP_COL, DOMAIN_COL, GEN_COL,
                                        EMB_COL, AUX_COL])
    bad = df[EMB_COL].map(_is_missing) | df[AUX_COL].map(_is_missing)
    if bad.any():
        df = df[~df[GROUP_COL].isin(df.loc[bad, GROUP_COL].unique())]
    df = df.reset_index(drop=True)
    span = df.groupby(GROUP_COL)[DOMAIN_COL].nunique()
    if (span > 1).any():
        raise SystemExit(f"{path.name}: {int((span > 1).sum())} group(s) span "
                         f"more than one domain — LODO would split them")
    return (_stack(df[EMB_COL]),
            (df[GEN_COL].to_numpy() != human_label).astype(int),
            _stack(df[AUX_COL]), df[DOMAIN_COL].astype(str).to_numpy(),
            df[GROUP_COL].to_numpy())


def _load_ood(path: Path, human_label: str):
    """Full-corpus eval set (never trained on): (X, y, Z, n_human)."""
    df = pd.read_parquet(path, columns=[GEN_COL, EMB_COL, AUX_COL])
    bad = df[EMB_COL].map(_is_missing) | df[AUX_COL].map(_is_missing)
    df = df[~bad].reset_index(drop=True)
    y = (df[GEN_COL].to_numpy() != human_label).astype(int)
    return _stack(df[EMB_COL]), y, _stack(df[AUX_COL]), int((y == 0).sum())


def _fold(spec, X, y, Z, groups, tr, te, seeds, fpr, ood_sets) -> dict:
    """One LODO fold for one spec, averaged over seeds. Each fold's model also
    scores every OOD eval set."""
    a, t = [], []
    ood: dict[str, list[tuple[float, float]]] = {n: [] for n in ood_sets}
    for seed in seeds:
        clf = fit(spec, X, y, Z, groups, tr, seed)
        ia, it = eval_on(clf, spec, X[te], y[te], Z[te], fpr)
        a.append(ia)
        t.append(it)
        for name, (oX, oy, oZ, _) in ood_sets.items():
            ood[name].append(eval_on(clf, spec, oX, oy, oZ, fpr))
    return {"auroc": float(np.mean(a)), "tpr_at_fpr": float(np.mean(t)),
            "n_test": int(te.sum()), "n_test_human": int((y[te] == 0).sum()),
            "ood": ood}


def _std(vals: list[float]) -> float:
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.std(vals)) if len(vals) > 1 else 0.0


def compute_trained(splits, ood_splits, seeds, fpr, human_label,
                    min_domain_rows) -> list[dict]:
    ood_sets = {name: _load_ood(path, human_label) for name, path in ood_splits}
    for name, (_, oy, _, n_h) in ood_sets.items():
        print(f"  OOD eval corpus {name!r}: {len(oy)} rows, {n_h} human (never trained on)")

    # per split -> per spec -> list of fold dicts
    all_folds: dict[str, dict[str, list[dict]]] = {}
    ood_reads: dict[str, dict[str, list]] = {
        k: {n: [] for n in ood_sets} for k in TRAINED_SPECS
    }

    for split_key, path in tqdm(splits, desc="LODO", unit="split"):
        X, y, Z, dom, ids = _load(path, human_label)
        counts = {d: int((dom == d).sum()) for d in np.unique(dom)}
        domains = sorted(d for d, n in counts.items()
                         if n >= min_domain_rows
                         and np.unique(y[dom == d]).size == 2)
        if len(domains) < 2:
            print(f"  {split_key}: only {len(domains)} usable domain(s) "
                  f"({counts}) — skipped")
            continue
        print(f"\n  {split_key}: {len(domains)} domains {domains}")

        spec_folds = {k: [] for k in TRAINED_SPECS}
        for d in domains:
            tr, te = dom != d, dom == d
            # LODO invariants: held-out domain scored in FULL, never in
            # training; train + test partition; no group (id) straddles.
            assert d not in set(dom[tr]) and set(dom[te]) == {d}
            assert tr.sum() + te.sum() == len(dom) and not (tr & te).any()
            assert set(ids[tr]).isdisjoint(ids[te]), "a group straddled the fold"
            print(f"    hold out {d!r:<22} train {sorted(set(dom[tr]))} "
                  f"(n={tr.sum()}, {len(set(ids[tr]))} groups, human {int((y[tr] == 0).sum())})"
                  f"  eval {d!r} (n={te.sum()}, {len(set(ids[te]))} groups, "
                  f"human {int((y[te] == 0).sum())})")
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


def compute_zeroshot(splits, cols, fpr, human_label) -> list[dict]:
    out: list[dict] = []
    for r in zeroshot_compute(splits, FAMILIES, cols, fpr, human_label):
        out.append({
            "level": "overall" if r["split"] == "mean" else "split",
            "kind": "zeroshot", "detector": _detector(r["family"], r["version"]),
            "split": r["split"], "held_out_domain": "",
            "n_test": r["n"] if r["split"] != "mean" else "",
            "n_folds": "", "n_seeds": "",
            "auroc": r["auroc"], "auroc_std": "",
            "tpr_at_fpr": r["tpr_at_fpr"], "tpr_at_fpr_std": "",
        })
    return out


def plots_from_csv(csv_path: Path, meta: dict, out_dir: Path) -> None:
    rows = [r for r in pd.read_csv(csv_path).to_dict("records")
            if r["level"] in ("split", "overall")]
    from epai.ai_detection.trainv2.plots import zeroshot_scatter

    out_dir.mkdir(parents=True, exist_ok=True)
    ood = "/".join(meta["ood"])
    zeroshot_scatter(
        rows, meta, out_dir / "lodo.png", key="detector",
        title=f"Leave-one-domain-out trained, plus {ood} as eval-only OOD, "
              f"vs. zero-shot picks\nAUROC vs. oracle TPR@{meta['fpr'] * 100:g}%FPR "
              f"— trained mean error bar = mean per-corpus SD",
    )


def _print_table(rows: list[dict], fpr: float) -> None:
    pct = f"{fpr * 100:g}"
    print(f"\n{'kind':<9}{'detector':<26}{'split':<10}{'domain':<16}"
          f"{'auroc':>16}{f'TPR@{pct}%':>16}")
    for r in rows:
        if r["level"] == "fold":
            continue
        a = (f"{r['auroc']:.3f}" if r["auroc_std"] in ("", None) or
             (isinstance(r["auroc_std"], float) and not np.isfinite(r["auroc_std"]))
             else f"{r['auroc']:.3f} ± {r['auroc_std']:.3f}")
        t = ("--" if not np.isfinite(r["tpr_at_fpr"]) else
             f"{r['tpr_at_fpr']:.3f}" if r["tpr_at_fpr_std"] in ("", None) or
             (isinstance(r["tpr_at_fpr_std"], float) and not np.isfinite(r["tpr_at_fpr_std"]))
             else f"{r['tpr_at_fpr']:.3f} ± {r['tpr_at_fpr_std']:.3f}")
        dom = r["held_out_domain"] or ("(all)" if r["level"] == "split" else "")
        mark = "  <<" if r["level"] == "overall" else ""
        print(f"{r['kind']:<9}{r['detector']:<26}{r['split']:<10}{dom:<16}"
              f"{a:>16}{t:>16}{mark}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--splits", nargs="+", default=None, metavar="KEY",
                    help=f"MANUGRAM splits to LODO over (default: {DEFAULT_SPLITS})")
    ap.add_argument("--ood-splits", nargs="+", default=None, metavar="KEY",
                    help=f"eval-only corpora, never trained on (default: {DEFAULT_OOD})")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--min-domain-rows", type=int, default=100)
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--zeroshot-config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV")
    args = ap.parse_args()

    splits = resolve_splits(args.splits or DEFAULT_SPLITS, PARQUET_DIR)
    ood_splits = resolve_splits(args.ood_splits or DEFAULT_OOD, PARQUET_DIR)
    meta = {"fpr": args.fpr, "n_seeds": len(args.seeds),
            "ood": [s for s, _ in ood_splits],
            "splits": [s for s, _ in splits] + [s for s, _ in ood_splits]}

    if args.replot is not None:
        plots_from_csv(args.replot, meta, args.replot.parent)
        print(f"replotted from {args.replot}")
        return

    missing = [str(p) for _, p in [*splits, *ood_splits] if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")

    print(f"trained : {[sp['label'] for sp in TRAINED_SPECS.values()]}, leave-one-domain-out")
    print(f"train on: {[s for s, _ in splits]}   OOD eval: {[s for s, _ in ood_splits]}"
          f"   seeds: {args.seeds}\n")

    rows = compute_trained(splits, ood_splits, args.seeds, args.fpr,
                           args.human_label, args.min_domain_rows)
    if not any(r["level"] == "overall" for r in rows):
        raise SystemExit("no split had >= 2 usable domains")

    if not args.no_zeroshot:
        cols = _load_columns(args.zeroshot_config)
        print(f"zero-shot: {cols}")
        rows += compute_zeroshot(splits + ood_splits, cols, args.fpr, args.human_label)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "performance.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    _print_table(rows, args.fpr)
    print(f"\nwrote {csv_path}")

    if not args.no_plots:
        plots_from_csv(csv_path, meta, args.out_dir)


if __name__ == "__main__":
    main()

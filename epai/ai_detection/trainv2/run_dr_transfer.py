r"""Domain randomization on the training embeddings -> MG-Den transfer.

An **opt-in add-on**, run by hand. It re-runs the two transfer measurements the
main runners already do -- in-domain 70/30 and leave-one-domain-out, each scored
on the MG-Den (Denario) OOD corpus -- but for every trained detector it also
trains a **domain-randomized twin**: the same algorithm, fit on training
embeddings that have been perturbed with Gaussian noise (config:
``cfg/ai_detection/trainv2/domain_randomization.yaml``). The question is whether
the twin's MG-Den AUROC / oracle TPR@1%FPR beats the base detector's.

What it does NOT do: touch ``_trained.py``, ``core.py``, the ``run_*_performance``
runners, ``epai/algs/*``, ``splits.py`` or ``rebuild_all.py``. When this script is
not invoked the pipeline runs byte-identical. It is deliberately absent from
``rebuild_all.TASKS`` -- same treatment as the MG-Reph family.

Noise touches TRAINING rows only. Every eval set (in-domain held-out test, LODO
held-out domain, MG-Den) is scored on clean embeddings. Only the RoBERTa
embedding ``X`` is noised; the 11-dim cue vector ``Z`` is tiled verbatim (Gaussian
noise on multi-hot bits is meaningless). One caveat for the torch specs
(``fusion_mlp`` / ``gated_mlp``): their internal early-stopping validation split
is carved out AFTER augmentation, so it watches a noised-val AUROC -- consistent
with "train under domain randomization", but worth knowing.

MG-Den has 80 human rows (< 1/fpr), so its oracle TPR@1%FPR point floors at
FPR=0: read the **base -> twin delta**, not the absolute.

compute -> CSV -> plot from the CSV:

    results/trainv2/dr_transfer/
        performance.csv   regime, variant (base|dr|ref), kind, detector, split,
                          role (source|ood), n_source_models, n_seeds,
                          auroc, auroc_std, tpr_at_fpr, tpr_at_fpr_std
        dr_transfer.png

Run (sandbox has no ``uv`` -- use ``.venv/bin/python`` directly):

    .venv/bin/python epai/ai_detection/trainv2/run_dr_transfer.py
    .venv/bin/python epai/ai_detection/trainv2/run_dr_transfer.py --regimes in_domain --seeds 0 1
    .venv/bin/python epai/ai_detection/trainv2/run_dr_transfer.py \
        --replot results/trainv2/dr_transfer/performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from epai.ai_detection.trainv2._trained import TRAINED_SPECS, eval_on, fit, load_features
from epai.ai_detection.trainv2.core import grouped_split
from epai.ai_detection.trainv2.run_best_zeroshot_performance import (
    DEFAULT_CONFIG as ZEROSHOT_CONFIG,
)
from epai.ai_detection.trainv2.run_best_zeroshot_performance import _load_columns
from epai.ai_detection.trainv2.run_in_domain_performance import compute_zeroshot
from epai.ai_detection.trainv2.run_lodo_performance import _load, _load_ood, _std
from epai.ai_detection.trainv2.splits import resolve as resolve_splits
from epai.algs import build_algorithm

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DR_CONFIG = BASE_DIR / "cfg" / "ai_detection" / "trainv2" / "domain_randomization.yaml"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "dr_transfer"

DEFAULT_SPLITS = ["gb_orig", "mg_gb", "mg_sci"]
DEFAULT_OOD = ["mg_den"]
REGIMES = ("in_domain", "lodo")

CSV_FIELDS = ["regime", "variant", "kind", "detector", "split", "role",
              "n_source_models", "n_seeds", "auroc", "auroc_std",
              "tpr_at_fpr", "tpr_at_fpr_std"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_dr_config(path: Path) -> dict:
    """Parse + validate the noise config. ``raise SystemExit`` on a bad shape --
    the idiom of ``run_best_zeroshot_performance._load_columns``."""
    cfg = yaml.safe_load(path.read_text()) or {}
    n = cfg.get("noise")
    try:
        assert isinstance(cfg.get("enabled"), bool)
        assert isinstance(n, dict)
        assert n.get("kind") in ("gaussian", "none")
        assert isinstance(n.get("sigma"), (int, float)) and n["sigma"] >= 0
        assert isinstance(n.get("per_feature_std"), bool)
        assert isinstance(n.get("n_copies"), int) and n["n_copies"] >= 0
        assert isinstance(cfg.get("seed_offset"), int)
    except AssertionError:
        raise SystemExit(
            f"{path}: expected\n"
            "  enabled: bool\n"
            "  noise: {kind: gaussian|none, sigma: >=0, per_feature_std: bool, "
            "n_copies: int >=0}\n"
            "  seed_offset: int"
        )
    return cfg


def _dr_summary(cfg: dict) -> str:
    """One-line noise description for the figure legend title."""
    n = cfg["noise"]
    if not cfg["enabled"] or n["kind"] == "none":
        return "domain randomization OFF (twin == base)"
    unit = "x featstd" if n["per_feature_std"] else " abs"
    mode = f"+{n['n_copies']} noised copies" if n["n_copies"] else "in-place jitter"
    return f"gaussian sigma={n['sigma']:g}{unit}, {mode}"


# ---------------------------------------------------------------------------
# The augmentation + the base-or-DR fit
# ---------------------------------------------------------------------------


def _augment(Xtr, ytr, Ztr, gtr, seed: int, cfg: dict):
    """Perturb an already-sliced TRAINING set. Returns ``(Xa, ya, Za, ga)``.

    ``enabled: false`` / ``kind: none`` -> inputs returned unchanged (the twin
    then coincides with the base -- a visible null control). ``n_copies == 0``
    -> one in-place jittered copy, row count unchanged. ``n_copies == k >= 1``
    -> the clean rows plus ``k`` independently-noised replicas (``y`` / ``Z`` /
    ``groups`` tiled to match; a group and its replicas keep one id).
    """
    noise = cfg["noise"]
    if not cfg["enabled"] or noise["kind"] == "none":
        return Xtr, ytr, Ztr, gtr

    rng = np.random.default_rng(int(seed) + int(cfg["seed_offset"]))
    if noise["per_feature_std"]:
        scale = (float(noise["sigma"]) * Xtr.std(axis=0, keepdims=True)).astype(np.float32)
    else:
        scale = np.float32(noise["sigma"])

    def _jit():
        return (Xtr + rng.standard_normal(Xtr.shape, dtype=np.float32) * scale).astype(np.float32)

    k = int(noise["n_copies"])
    if k == 0:
        return _jit(), ytr, Ztr, gtr

    reps = k + 1
    Xa = np.vstack([Xtr, *(_jit() for _ in range(k))])
    return Xa, np.tile(ytr, reps), np.tile(Ztr, (reps, 1)), np.tile(gtr, reps)


def fit_dr(spec: dict, X, y, Z, groups, idx, seed: int, cfg: dict):
    """Mirror of ``_trained.fit`` with the training slice perturbed first.
    ``_trained.fit`` is untouched -- ``_fit_variant`` calls it for the base."""
    Xa, ya, Za, ga = _augment(X[idx], y[idx], Z[idx], groups[idx], seed, cfg)
    clf = build_algorithm(spec["alg"], seed, **spec["params"])
    clf.set_fit_groups(ga)
    if spec["aux"]:
        clf.fit(Xa, ya, Za)
    else:
        clf.fit(Xa, ya)
    return clf


def _fit_variant(spec, X, y, Z, groups, idx, seed, cfg, variant: str):
    if variant == "base":
        return fit(spec, X, y, Z, groups, idx, seed)
    return fit_dr(spec, X, y, Z, groups, idx, seed, cfg)


def _seed_loop(spec, X, y, Z, groups, splitter, seeds, fpr, ood_sets, cfg, variant):
    """One (split | LODO fold): fit base-or-DR per seed on ``splitter(seed) ->
    (tr, te)``, eval on the clean held-out rows and on every clean OOD set.
    Returns ``(src_auroc_mean, src_tpr_mean, {ood_name: [(auroc, tpr), ...]})``.
    """
    a, t = [], []
    ood: dict[str, list[tuple[float, float]]] = {n: [] for n in ood_sets}
    for seed in seeds:
        tr, te = splitter(seed)
        clf = _fit_variant(spec, X, y, Z, groups, tr, seed, cfg, variant)
        ia, it, _ = eval_on(clf, spec, X[te], y[te], Z[te], fpr)
        a.append(ia)
        t.append(it)
        for name, (oX, oy, oZ, _) in ood_sets.items():
            ood[name].append(eval_on(clf, spec, oX, oy, oZ, fpr))
    return float(np.mean(a)), float(np.mean(t)), ood


# ---------------------------------------------------------------------------
# Row assembly, shared by both regimes
# ---------------------------------------------------------------------------


def _rows_for_variant(regime, variant, seeds, src_of, ood_reads):
    """``src_of[spec_key]`` = list of ``(auroc, tpr)`` source seed-means (one per
    split or fold). ``ood_reads[spec_key][name]`` = every source-model read.
    Emits one ``role=source`` + one ``role=ood`` row per (spec, ood corpus)."""
    rows = []
    for spec_key, spec in TRAINED_SPECS.items():
        detector = f"trained/{spec['label']}"
        sa = [v[0] for v in src_of[spec_key]]
        st = [v[1] for v in src_of[spec_key]]
        rows.append({
            "regime": regime, "variant": variant, "kind": "trained",
            "detector": detector, "split": "pooled", "role": "source",
            "n_source_models": len(src_of[spec_key]), "n_seeds": len(seeds),
            "auroc": float(np.mean(sa)), "auroc_std": _std(sa),
            "tpr_at_fpr": float(np.mean(st)), "tpr_at_fpr_std": _std(st),
        })
        for name, reads in ood_reads[spec_key].items():
            oa = [r[0] for r in reads]
            ot = [r[1] for r in reads]
            rows.append({
                "regime": regime, "variant": variant, "kind": "trained",
                "detector": detector, "split": name, "role": "ood",
                "n_source_models": len(reads), "n_seeds": len(seeds),
                "auroc": float(np.mean(oa)), "auroc_std": _std(oa),
                "tpr_at_fpr": float(np.mean(ot)), "tpr_at_fpr_std": _std(ot),
            })
    return rows


# ---------------------------------------------------------------------------
# Regime: in-domain 70/30
# ---------------------------------------------------------------------------


def compute_in_domain(train_splits, ood_splits, seeds, test_size, fpr,
                      group_col, human_label, cfg) -> list[dict]:
    ood_sets: dict[str, tuple] = {}
    for name, path in ood_splits:
        oX, oy, oZ, _ = load_features(path, group_col, human_label)
        ood_sets[name] = (oX, oy, oZ, int((oy == 0).sum()))
        print(f"  OOD eval corpus {name!r}: {len(oy)} rows, "
              f"{int((oy == 0).sum())} human (never trained on)")

    loaded = [(k, load_features(p, group_col, human_label)) for k, p in train_splits]

    rows: list[dict] = []
    for variant in ("base", "dr"):
        src = {k: [] for k in TRAINED_SPECS}
        ood_reads = {k: {n: [] for n in ood_sets} for k in TRAINED_SPECS}
        for split_key, (X, y, Z, groups) in tqdm(
                loaded, desc=f"in_domain/{variant}", unit="split"):
            def splitter(seed, y=y, groups=groups):
                return grouped_split(y, groups, test_size, seed)
            for spec_key, spec in TRAINED_SPECS.items():
                sa, st, ood = _seed_loop(spec, X, y, Z, groups, splitter,
                                         seeds, fpr, ood_sets, cfg, variant)
                src[spec_key].append((sa, st))
                for name, reads in ood.items():
                    ood_reads[spec_key][name].extend(reads)
        rows += _rows_for_variant("in_domain", variant, seeds, src, ood_reads)
    return rows


# ---------------------------------------------------------------------------
# Regime: leave-one-domain-out
# ---------------------------------------------------------------------------


def compute_lodo(splits, ood_splits, seeds, fpr, human_label,
                 min_domain_rows, cfg) -> list[dict]:
    ood_sets = {name: _load_ood(path, human_label) for name, path in ood_splits}
    for name, (_, oy, _, n_h) in ood_sets.items():
        print(f"  OOD eval corpus {name!r}: {len(oy)} rows, {n_h} human "
              "(never trained on)")

    loaded = []
    for split_key, path in splits:
        X, y, Z, dom, ids = _load(path, human_label)
        counts = {d: int((dom == d).sum()) for d in np.unique(dom)}
        domains = sorted(d for d, c in counts.items()
                         if c >= min_domain_rows
                         and np.unique(y[dom == d]).size == 2)
        if len(domains) < 2:
            print(f"  {split_key}: only {len(domains)} usable domain(s) "
                  f"({counts}) — skipped")
            continue
        print(f"  {split_key}: {len(domains)} domains {domains}")
        loaded.append((split_key, X, y, Z, dom, ids, domains))

    if not loaded:
        return []

    rows: list[dict] = []
    for variant in ("base", "dr"):
        src = {k: [] for k in TRAINED_SPECS}
        ood_reads = {k: {n: [] for n in ood_sets} for k in TRAINED_SPECS}
        for split_key, X, y, Z, dom, ids, domains in tqdm(
                loaded, desc=f"lodo/{variant}", unit="split"):
            for d in domains:
                tr, te = dom != d, dom == d
                assert set(ids[tr]).isdisjoint(ids[te]), "a group straddled the fold"
                splitter = lambda seed, tr=tr, te=te: (tr, te)  # noqa: E731
                for spec_key, spec in TRAINED_SPECS.items():
                    sa, st, ood = _seed_loop(spec, X, y, Z, ids, splitter,
                                             seeds, fpr, ood_sets, cfg, variant)
                    src[spec_key].append((sa, st))
                    for name, reads in ood.items():
                        ood_reads[spec_key][name].extend(reads)
        rows += _rows_for_variant("lodo", variant, seeds, src, ood_reads)
    return rows


# ---------------------------------------------------------------------------
# Zero-shot reference on the OOD corpus
# ---------------------------------------------------------------------------


def compute_zeroshot_ood(ood_splits, cols, fpr, human_label, regimes) -> list[dict]:
    ood_keys = {k for k, _ in ood_splits}
    base = [r for r in compute_zeroshot(ood_splits, cols, fpr, human_label)
            if r["split"] in ood_keys]
    out = []
    for regime in regimes:
        for r in base:
            out.append({
                "regime": regime, "variant": "ref", "kind": "zeroshot",
                "detector": r["detector"], "split": r["split"], "role": "ood",
                "n_source_models": "", "n_seeds": "",
                "auroc": r["auroc"], "auroc_std": "",
                "tpr_at_fpr": r["tpr_at_fpr"], "tpr_at_fpr_std": "",
            })
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt(v, sd) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(v):
        return "--"
    try:
        sd = float(sd)
        bare = not np.isfinite(sd)
    except (TypeError, ValueError):
        bare = True
    return f"{v:.3f}" if bare else f"{v:.3f} ± {sd:.3f}"


def _print_table(rows: list[dict], fpr: float) -> None:
    pct = f"{fpr * 100:g}"
    print(f"\n{'regime':<11}{'variant':<8}{'detector':<26}{'role':<8}"
          f"{'auroc':>16}{f'TPR@{pct}%':>16}")
    order = {("in_domain", "source"): 0, ("in_domain", "ood"): 1,
             ("lodo", "source"): 2, ("lodo", "ood"): 3}
    for r in sorted(rows, key=lambda r: (order.get((r["regime"], r["role"]), 9),
                                         r["detector"], r["variant"])):
        print(f"{r['regime']:<11}{r['variant']:<8}{r['detector']:<26}{r['role']:<8}"
              f"{_fmt(r['auroc'], r['auroc_std']):>16}"
              f"{_fmt(r['tpr_at_fpr'], r['tpr_at_fpr_std']):>16}")


def plots_from_csv(csv_path: Path, meta: dict, out_dir: Path) -> None:
    rows = pd.read_csv(csv_path).to_dict("records")
    from epai.ai_detection.trainv2.plots import dr_scatter

    out_dir.mkdir(parents=True, exist_ok=True)
    dr_scatter(
        rows, meta, out_dir / "dr_transfer.png",
        title="Domain randomization on the training embeddings -> MG-Den transfer\n"
              f"AUROC vs. oracle TPR@{meta['fpr'] * 100:g}%FPR, {meta['n_seeds']} seeds",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    ap.add_argument("--splits", nargs="+", default=None, metavar="KEY",
                    help=f"training splits (default: {DEFAULT_SPLITS})")
    ap.add_argument("--ood-splits", nargs="+", default=None, metavar="KEY",
                    help=f"eval-only OOD corpora (default: {DEFAULT_OOD})")
    ap.add_argument("--config", type=Path, default=DR_CONFIG)
    ap.add_argument("--zeroshot-config", type=Path, default=ZEROSHOT_CONFIG)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--min-domain-rows", type=int, default=100)
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV")
    args = ap.parse_args()

    dr_cfg = _load_dr_config(args.config)
    ood_splits = resolve_splits(args.ood_splits or DEFAULT_OOD, PARQUET_DIR)
    meta = {"fpr": args.fpr, "n_seeds": len(args.seeds), "regimes": args.regimes,
            "ood": [s for s, _ in ood_splits], "dr": _dr_summary(dr_cfg)}

    if args.replot is not None:
        plots_from_csv(args.replot, meta, args.replot.parent)
        print(f"replotted from {args.replot}")
        return

    if not dr_cfg["enabled"] or dr_cfg["noise"]["kind"] == "none":
        print("!! domain randomization is OFF in the config — the 'dr' twin will "
              "coincide with 'base' (null control).")

    train_splits = resolve_splits(args.splits or DEFAULT_SPLITS, PARQUET_DIR)
    missing = [str(p) for _, p in [*train_splits, *ood_splits] if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")

    print(f"detectors : {[s['label'] for s in TRAINED_SPECS.values()]}  x {{base, dr}}")
    print(f"regimes   : {args.regimes}   noise: {meta['dr']}")
    print(f"train on  : {[s for s, _ in train_splits]}   OOD: {[s for s, _ in ood_splits]}"
          f"   seeds: {args.seeds}\n")

    rows: list[dict] = []
    if "in_domain" in args.regimes:
        rows += compute_in_domain(train_splits, ood_splits, args.seeds,
                                  args.test_size, args.fpr, args.group_col,
                                  args.human_label, dr_cfg)
    if "lodo" in args.regimes:
        rows += compute_lodo(train_splits, ood_splits, args.seeds, args.fpr,
                             args.human_label, args.min_domain_rows, dr_cfg)

    if not args.no_zeroshot:
        cols = _load_columns(args.zeroshot_config)
        print(f"zero-shot : {cols}")
        rows += compute_zeroshot_ood(ood_splits, cols, args.fpr,
                                     args.human_label, args.regimes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "performance.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    _print_table(rows, args.fpr)
    print(f"\nwrote {csv_path}")

    if not args.no_plots:
        plots_from_csv(csv_path, meta, args.out_dir)


if __name__ == "__main__":
    main()

r"""Domain randomization on the training embeddings -> MG-AI-Sci transfer.

An **opt-in add-on**, run by hand. It re-runs the two transfer measurements the
main runners already do -- in-domain 70/30 and leave-one-domain-out, each scored
on the MG-AI-Sci OOD corpus (Denario + Sakana AI-Scientist / CycleResearcher /
Zochi, merged) -- but for every trained detector it also
trains a **domain-randomized twin**: the same algorithm, fit on training
embeddings that have been perturbed with Gaussian noise (config:
``cfg/ai_detection/trainv2/domain_randomization.yaml``). The question is whether
the twin's MG-AI-Sci AUROC / oracle TPR@1%FPR beats the base detector's.

Output shape and plots now match ``run_in_domain_performance.py`` /
``run_lodo_performance.py`` exactly -- same CSV schema, same figures
(``zeroshot_scatter`` / ``goodness_bars`` / ``denario_scatter``), not a bespoke
figure. The twin is folded in as a same-hue, one-shade-darker variant of its
base (``trained/<label>`` base, ``trained/<label>_dr`` twin), the same
colour convention ``zeroshot_scatter`` already uses for a judge's ``_wlp``
readout variant. Every row-building / zero-shot / table-printing helper below
is imported, unmodified, from the two main runners -- this script only adds
the domain-randomized fit and the variant-suffixed detector label; ``_trained.py``,
``core.py``, ``splits.py``, the two main runners and ``rebuild_all.py`` are
never edited or imported for anything but their pure functions. Deliberately
absent from ``rebuild_all.TASKS`` -- same treatment as the MG-Reph family.
When this script is not invoked the pipeline runs byte-identical.

Noise touches TRAINING rows only. Every eval set (in-domain held-out test, LODO
held-out domain, MG-AI-Sci) is scored on clean embeddings. Only the RoBERTa
embedding ``X`` is noised; the 11-dim cue vector ``Z`` is tiled verbatim (Gaussian
noise on multi-hot bits is meaningless). One caveat for the torch specs
(``fusion_mlp`` / ``gated_mlp``): their internal early-stopping validation split
is carved out AFTER augmentation, so it watches a noised-val AUROC -- consistent
with "train under domain randomization", but worth knowing.

MG-AI-Sci has 100 human rows now (was 80, pre-merge with the ai_scientist
corpus) -- right at 1/fpr for the 1% level rather than clearly under it, so its
oracle TPR@1%FPR point is no longer a guaranteed floor at FPR=0, just a
still-noisy single-threshold reading (the achievable FPR grid is 0%, 1%, 2%,
... at n=100; it was 0%, 1.25%, 2.5%, ... at n=80). Read the **base -> twin
delta** either way, not the absolute.

compute -> CSV -> plot from the CSV, one regime = one CSV = one figure set,
same layout as the main runners:

    results/trainv2/dr_transfer/
        in_domain_performance.csv, in_domain_{1pct,5pct}.png (+ _goodness_, _denario_)
        lodo_performance.csv, lodo_{1pct,5pct}.png (+ _goodness_, _denario_)

Run (sandbox has no ``uv`` -- use ``.venv/bin/python`` directly):

    .venv/bin/python epai/ai_detection/trainv2/run_dr_transfer.py
    .venv/bin/python epai/ai_detection/trainv2/run_dr_transfer.py --regimes in_domain --seeds 0 1
    .venv/bin/python epai/ai_detection/trainv2/run_dr_transfer.py \
        --replot results/trainv2/dr_transfer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from epai.ai_detection.trainv2._trained import TRAINED_SPECS, eval_on_fprs, fit, load_features
from epai.ai_detection.trainv2.core import (
    eff_fpr_level,
    fpr_tag,
    goodness,
    grouped_split,
    rows_at_fpr,
)
from epai.ai_detection.trainv2.run_best_zeroshot_performance import (
    DEFAULT_CONFIG as ZEROSHOT_CONFIG,
)
from epai.ai_detection.trainv2.run_in_domain_performance import (
    CSV_FIELDS as CSV_FIELDS_IN_DOMAIN,
)
from epai.ai_detection.trainv2.run_in_domain_performance import (
    _overall_cell,
    _print_table as _print_table_in_domain,
    _row,
)
from epai.ai_detection.trainv2.run_in_domain_performance import (
    compute_zeroshot as compute_zeroshot_in_domain,
)
from epai.ai_detection.trainv2.run_lodo_performance import CSV_FIELDS as CSV_FIELDS_LODO
from epai.ai_detection.trainv2.run_lodo_performance import (
    _load,
    _load_ood,
    _print_table as _print_table_lodo,
    _std,
)
from epai.ai_detection.trainv2.run_lodo_performance import compute_zeroshot as compute_zeroshot_lodo
from epai.ai_detection.trainv2.run_zeroshot_performance import DEFAULT_FPRS
from epai.ai_detection.trainv2.splits import resolve as resolve_splits
from epai.algs import build_algorithm

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DR_CONFIG = BASE_DIR / "cfg" / "ai_detection" / "trainv2" / "domain_randomization.yaml"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "dr_transfer"

DEFAULT_SPLITS = ["gb_orig", "mg_gb", "mg_sci"]
DEFAULT_OOD = ["mg_aisci"]
REGIMES = ("in_domain", "lodo")

#: the twin's detector label is its base's + this suffix -- the exact pattern
#: plots.py's ``_variant_palette`` already reads for a judge's ``_wlp`` readout
#: variant, so passing ``variant_suffix="_dr"`` makes the DR twin draw one
#: shade darker's *base* and the plain hue's *twin*, same colour, same figure
#: types as every other trained-detector run.
VARIANT_SUFFIX = "_dr"


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


def _dr_label(spec: dict, variant: str) -> str:
    return spec["label"] + (VARIANT_SUFFIX if variant == "dr" else "")


# ---------------------------------------------------------------------------
# Regime: in-domain 70/30 -- same row shape as
# run_in_domain_performance.compute_trained, variant-suffixed detector label
# ---------------------------------------------------------------------------


def compute_in_domain(train_splits, ood_splits, seeds, test_size, fprs,
                      group_col, human_label, combine, dr_cfg) -> list[dict]:
    ood_sets: dict[str, tuple] = {}
    for name, path in ood_splits:
        oX, oy, oZ, _ = load_features(path, group_col, human_label)
        ood_sets[name] = (oX, oy, oZ, int((oy == 0).sum()))
        print(f"  OOD eval corpus {name!r}: {len(oy)} rows, "
              f"{int((oy == 0).sum())} human (never trained on)")

    loaded = [(k, load_features(p, group_col, human_label)) for k, p in train_splits]

    rows: list[dict] = []
    for variant in ("base", "dr"):
        cells: dict[str, dict[float, list[dict]]] = {
            k: {f: [] for f in fprs} for k in TRAINED_SPECS}
        ood_reads: dict[str, dict[str, dict[float, list]]] = {
            k: {n: {f: [] for f in fprs} for n in ood_sets} for k in TRAINED_SPECS}

        for split_key, (X, y, Z, groups) in tqdm(
                loaded, desc=f"in_domain/{variant}", unit="split"):
            for spec_key, spec in TRAINED_SPECS.items():
                a: list[float] = []
                t = {f: [] for f in fprs}
                te_ = {f: [] for f in fprs}
                n_tr = n_te = n_te_h = 0
                for seed in seeds:
                    tr, te = grouped_split(y, groups, test_size, seed)
                    clf = _fit_variant(spec, X, y, Z, groups, tr, seed, dr_cfg, variant)
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

        for spec_key, spec in TRAINED_SPECS.items():
            detector = f"trained/{_dr_label(spec, variant)}"
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


# ---------------------------------------------------------------------------
# Regime: leave-one-domain-out -- same row shape as
# run_lodo_performance.compute_trained, variant-suffixed detector label
# ---------------------------------------------------------------------------


def _fold_variant(spec, X, y, Z, groups, tr, te, seeds, fprs, ood_sets,
                  cfg, variant: str) -> dict:
    """Mirror of ``run_lodo_performance._fold``, fit through ``_fit_variant``
    instead of ``_trained.fit`` so the LODO training slice can be the
    domain-randomized twin."""
    a: list[float] = []
    t = {f: [] for f in fprs}
    te_ = {f: [] for f in fprs}
    ood: dict[str, dict[float, list]] = {n: {f: [] for f in fprs} for n in ood_sets}
    for seed in seeds:
        clf = _fit_variant(spec, X, y, Z, groups, tr, seed, cfg, variant)
        au, per = eval_on_fprs(clf, spec, X[te], y[te], Z[te], fprs)
        a.append(au)
        for f in fprs:
            t[f].append(per[f][0])
            te_[f].append(per[f][1])
        for name, (oX, oy, oZ, _) in ood_sets.items():
            oau, oper = eval_on_fprs(clf, spec, oX, oy, oZ, fprs)
            for f in fprs:
                ood[name][f].append((oau, oper[f][0], oper[f][1]))
    n_h = int((y[te] == 0).sum())
    return {
        "auroc": float(np.mean(a)),
        "by_fpr": {f: {"tpr_at_fpr": float(np.mean(t[f])),
                       "tpr_eff": float(np.mean(te_[f])),
                       "eff_fpr": eff_fpr_level(n_h, f)} for f in fprs},
        "n_test": int(te.sum()), "n_test_human": n_h, "ood": ood}


def compute_lodo(splits, ood_splits, seeds, fprs, human_label,
                 min_domain_rows, combine, dr_cfg) -> list[dict]:
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
        all_folds: dict[str, dict[str, list[dict]]] = {}
        ood_reads: dict[str, dict[str, dict[float, list]]] = {
            k: {n: {f: [] for f in fprs} for n in ood_sets} for k in TRAINED_SPECS
        }

        for split_key, X, y, Z, dom, ids, domains in tqdm(
                loaded, desc=f"lodo/{variant}", unit="split"):
            spec_folds = {k: [] for k in TRAINED_SPECS}
            for d in domains:
                tr, te = dom != d, dom == d
                # LODO invariants: held-out domain scored in FULL, never in
                # training; train + test partition; no group (id) straddles.
                assert d not in set(dom[tr]) and set(dom[te]) == {d}
                assert tr.sum() + te.sum() == len(dom) and not (tr & te).any()
                assert set(ids[tr]).isdisjoint(ids[te]), "a group straddled the fold"
                for spec_key, spec in TRAINED_SPECS.items():
                    fd = _fold_variant(spec, X, y, Z, ids, tr, te, seeds, fprs,
                                       ood_sets, dr_cfg, variant)
                    fd["held_out_domain"] = d
                    for name, by_fpr in fd.pop("ood").items():
                        for f in fprs:
                            ood_reads[spec_key][name][f].extend(by_fpr[f])
                    spec_folds[spec_key].append(fd)
            all_folds[split_key] = spec_folds

        for spec_key, spec in TRAINED_SPECS.items():
            detector = f"trained/{_dr_label(spec, variant)}"
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
                        "auroc_std": _std([fd["auroc"] for fd in folds]),
                        "tpr_at_fpr": float(np.mean([fd["by_fpr"][fpr]["tpr_at_fpr"] for fd in use])),
                        "tpr_at_fpr_std": _std([fd["by_fpr"][fpr]["tpr_at_fpr"] for fd in use]),
                        "tpr_eff": float(np.mean([fd["by_fpr"][fpr]["tpr_eff"] for fd in folds])),
                        "tpr_eff_std": _std([fd["by_fpr"][fpr]["tpr_eff"] for fd in folds]),
                        "eff_fpr": float(np.mean([fd["by_fpr"][fpr]["eff_fpr"] for fd in folds])),
                        "goodness": float(np.mean(gf)), "goodness_std": _std(gf),
                    }
                    rows.append(sr)
                    split_rows.append(sr)

                for name, (_, _, _, n_h) in ood_sets.items():
                    reads = ood_reads[spec_key][name][fpr]
                    oa, ot = [r[0] for r in reads], [r[1] for r in reads]
                    ote = [r[2] for r in reads]
                    og = [goodness(a_, te, combine) for a_, te in zip(oa, ote)]
                    sr = {
                        "level": "split", "kind": "trained", "detector": detector,
                        "split": name, "held_out_domain": "", "fpr": fpr, "n_test": "",
                        "n_folds": len(reads), "n_seeds": len(seeds),
                        "reachable": bool(n_h >= 1 / fpr),
                        "auroc": float(np.mean(oa)), "auroc_std": _std(oa),
                        "tpr_at_fpr": float(np.mean(ot)), "tpr_at_fpr_std": _std(ot),
                        "tpr_eff": float(np.mean(ote)), "tpr_eff_std": _std(ote),
                        "eff_fpr": eff_fpr_level(n_h, fpr),
                        "goodness": float(np.mean(og)), "goodness_std": _std(og),
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
    dr = meta.get("dr", "")
    for fpr in meta["fprs"]:
        fmeta = {**meta, "fpr": fpr}
        tag = fpr_tag(fpr)
        frows = rows_at_fpr(rows, fpr)
        zeroshot_scatter(
            frows, fmeta, out_dir / f"in_domain_{tag}.png", key="detector",
            variant_suffix=VARIANT_SUFFIX,
            title=f"In-domain (70/30, {n_seeds} seeds) — base vs. domain-"
                  f"randomized twin, plus {ood} as eval-only OOD\nAUROC vs. "
                  f"oracle TPR@{fpr * 100:g}%FPR — {dr}",
        )
        goodness_bars(
            frows, fmeta, out_dir / f"in_domain_goodness_{tag}.png", key="detector",
            combine=meta.get("combine", "hmean"), variant_suffix=VARIANT_SUFFIX,
            title=f"In-domain (70/30, {n_seeds} seeds) — base vs. twin — "
                  f"goodness (TPR@{fpr * 100:g}%FPR) across the MANUGRAM splits",
        )
        denario_scatter(
            frows, fmeta, out_dir / f"in_domain_denario_{tag}.png",
            variant_suffix=VARIANT_SUFFIX,
            title=f"In-domain (70/30, {n_seeds} seeds) on {ood} (eval-only OOD)"
                  f"\nbase vs. domain-randomized twin — AUROC vs. oracle "
                  f"TPR@{fpr * 100:g}%FPR",
        )


def plots_lodo(csv_path: Path, meta: dict, out_dir: Path) -> None:
    all_rows = [r for r in pd.read_csv(csv_path).to_dict("records")
               if r["level"] in ("split", "overall")]
    from epai.ai_detection.trainv2.plots import denario_scatter, goodness_bars, zeroshot_scatter

    out_dir.mkdir(parents=True, exist_ok=True)
    ood = "/".join(meta["ood"])
    dr = meta.get("dr", "")
    for fpr in meta["fprs"]:
        fmeta = {**meta, "fpr": fpr}
        tag = fpr_tag(fpr)
        rows = rows_at_fpr(all_rows, fpr)
        zeroshot_scatter(
            rows, fmeta, out_dir / f"lodo_{tag}.png", key="detector",
            variant_suffix=VARIANT_SUFFIX,
            title=f"Leave-one-domain-out — base vs. domain-randomized twin, "
                  f"plus {ood} as eval-only OOD\nAUROC vs. oracle "
                  f"TPR@{fpr * 100:g}%FPR — {dr}",
        )
        goodness_bars(
            rows, fmeta, out_dir / f"lodo_goodness_{tag}.png", key="detector",
            combine=meta.get("combine", "hmean"), variant_suffix=VARIANT_SUFFIX,
            title=f"Leave-one-domain-out — base vs. twin — goodness "
                  f"(TPR@{fpr * 100:g}%FPR) across the MANUGRAM splits",
        )
        denario_scatter(
            rows, fmeta, out_dir / f"lodo_denario_{tag}.png",
            variant_suffix=VARIANT_SUFFIX,
            title=f"Leave-one-domain-out on {ood} (eval-only OOD)\nbase vs. "
                  f"domain-randomized twin — AUROC vs. oracle TPR@{fpr * 100:g}%FPR",
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
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS, metavar="LEVEL",
                    help=f"oracle FPR level(s); one figure set per level "
                         f"(default: {DEFAULT_FPRS})")
    ap.add_argument("--combine", choices=("hmean", "gmean", "amean"), default="hmean")
    ap.add_argument("--min-domain-rows", type=int, default=100)
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="OUT_DIR",
                    help="re-plot from an existing out-dir's CSV(s), no compute")
    args = ap.parse_args()

    dr_cfg = _load_dr_config(args.config)
    train_splits = resolve_splits(args.splits or DEFAULT_SPLITS, PARQUET_DIR)
    ood_splits = resolve_splits(args.ood_splits or DEFAULT_OOD, PARQUET_DIR)
    meta = {"fprs": args.fpr, "n_seeds": len(args.seeds), "combine": args.combine,
            "regimes": args.regimes, "ood": [s for s, _ in ood_splits],
            "splits": [s for s, _ in train_splits] + [s for s, _ in ood_splits],
            "dr": _dr_summary(dr_cfg)}

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

    if not dr_cfg["enabled"] or dr_cfg["noise"]["kind"] == "none":
        print("!! domain randomization is OFF in the config — the 'dr' twin will "
              "coincide with 'base' (null control).")

    missing = [str(p) for _, p in [*train_splits, *ood_splits] if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")

    print(f"detectors : {[s['label'] for s in TRAINED_SPECS.values()]}  x {{base, dr}}")
    print(f"regimes   : {args.regimes}   noise: {meta['dr']}")
    print(f"train on  : {[s for s, _ in train_splits]}   OOD: {[s for s, _ in ood_splits]}"
          f"   seeds: {args.seeds}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if "in_domain" in args.regimes:
        rows = compute_in_domain(train_splits, ood_splits, args.seeds,
                                 args.test_size, args.fpr, args.group_col,
                                 args.human_label, args.combine, dr_cfg)
        if not args.no_zeroshot:
            rows += compute_zeroshot_in_domain(
                train_splits, ood_splits, args.zeroshot_config, args.seeds,
                args.test_size, args.fpr, args.group_col, args.human_label,
                args.combine)
        csv_path = args.out_dir / "in_domain_performance.csv"
        pd.DataFrame(rows, columns=CSV_FIELDS_IN_DOMAIN).to_csv(csv_path, index=False)
        _print_table_in_domain(rows, args.fpr)
        print(f"\nwrote {csv_path}")
        if not args.no_plots:
            plots_in_domain(csv_path, meta, args.out_dir)

    if "lodo" in args.regimes:
        rows = compute_lodo(train_splits, ood_splits, args.seeds, args.fpr,
                            args.human_label, args.min_domain_rows,
                            args.combine, dr_cfg)
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

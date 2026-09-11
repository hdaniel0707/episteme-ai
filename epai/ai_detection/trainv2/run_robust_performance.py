r"""Tail-aware loss (TopPush) + domain-invariance regularizers (CORAL / V-REx)
vs. the existing trained detectors -- in-domain 70/30 and leave-one-domain-out,
each scored on the MG-AI-Sci OOD corpus, exactly like
``run_in_domain_performance.py`` / ``run_lodo_performance.py``.

An **opt-in add-on**, run by hand -- same treatment as ``run_dr_transfer.py``:
absent from ``rebuild_all.TASKS``, and ``_trained.py`` / ``core.py`` /
``splits.py`` / the two main runners / ``rebuild_all.py`` are only ever
imported from, never edited. See
``epai/ai_detection/trainv2/improving_sota_ideas.md`` (#1, #2) for the method
and citations.

Nine new ``robust_mlp`` arms (:mod:`epai.algs.robust_mlp`), all a **linear
probe on ``emb_roberta``** -- same architecture as `trained/roberta·logistic`
and each other, so only the loss / regularizer (and, for CORAL/V-REx, its
weight) varies:

    erm_torch              CE, no domain penalty          (architecture-matched control)
    tail_toppush           CE + TopPush                   (#1)
    domain_coral_w{1,100,10000}   CE + CORAL, weight swept (#2)
    domain_vrex_w{1,10,100}       CE + V-REx, weight swept (#2)
    tail_toppush_coral     CE + TopPush + CORAL(w=1)     (#1 + #2)

The weight sweep exists because a first run at weight=1 showed both
regularizers contributing ~nothing (`domain_coral_w1` tied `erm_torch` to 3
decimals everywhere) -- CORAL's textbook 1/(4d^2) normalization (d=512) makes
its natural scale tiny next to the CE loss, so weight=1 likely under-powers
it rather than the method being ineffective. See `improving_sota_ideas.md`'s
results notes for the reasoning.

Reported alongside the existing three ``_trained.py`` detectors
(``roberta_logistic``, ``roberta_fusion_cuevec``, ``roberta_gated_cuevec``) and
the ``best_zeroshot.yaml`` picks, on the SAME figures
(``zeroshot_scatter`` / ``goodness_bars`` / ``denario_scatter``) the two main
runners already produce -- not a bespoke figure. CORAL / V-REx read each
training split's *internal* domain column (the same one
``run_lodo_performance.py`` LODO-folds over) as the "environment" label: in
the in-domain 70/30 regime that's whatever domain mix falls out of the random
cut (usually all of them); in a LODO fold it's exactly the domains still in
the training union after holding one out.

compute -> CSV -> plot from the CSV, one regime = one CSV = one figure set:

    results/trainv2/robust_performance/
        in_domain_performance.csv, in_domain_{1pct,5pct}.png (+ _goodness_, _denario_)
        lodo_performance.csv, lodo_{1pct,5pct}.png (+ _goodness_, _denario_)

Run (sandbox has no ``uv`` -- use ``.venv/bin/python`` directly):

    .venv/bin/python epai/ai_detection/trainv2/run_robust_performance.py
    .venv/bin/python epai/ai_detection/trainv2/run_robust_performance.py --regimes lodo --seeds 0 1
    .venv/bin/python epai/ai_detection/trainv2/run_robust_performance.py \
        --replot results/trainv2/robust_performance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from epai.ai_detection.trainv2._trained import TRAINED_SPECS, eval_on_fprs
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
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "robust_performance"

DEFAULT_SPLITS = ["gb_orig", "mg_gb", "mg_sci"]
DEFAULT_OOD = ["mg_aisci"]
REGIMES = ("in_domain", "lodo")

#: Shared shape for every new arm: a linear probe (no hidden layer) on
#: ``emb_roberta`` alone (``aux: False``), with an actual held-out validation
#: split (AUROC-scored) since generalizing under domain shift is the whole
#: point. Only ``loss_kind`` / ``domain_penalty`` differ between arms.
_BASE_PARAMS = {"hidden_dims": [], "val_frac": 0.15, "val_metric": "auroc",
                "epochs": 100, "patience": 10}

#: ``domain_penalty_weight`` here is a first-cut starting point, not a tuned
#: value -- worth a sweep before it goes in the paper. See
#: ``improving_sota_ideas.md`` for the method.
ROBUST_SPECS: dict[str, dict] = {
    "erm_torch": {
        "label": "roberta·erm_torch", "alg": "robust_mlp", "aux": False,
        "params": {**_BASE_PARAMS},
    },
    "tail_toppush": {
        "label": "roberta·toppush", "alg": "robust_mlp", "aux": False,
        "params": {**_BASE_PARAMS, "loss_kind": "toppush", "push_frac": 0.05,
                  "push_weight": 1.0, "push_margin": 1.0},
    },
    # CORAL's textbook normalization (Sun & Saenko 2016) divides the squared
    # Frobenius distance between two d x d covariance matrices by 4*d^2 -- with
    # d=512 that's ~1.05M, which crushes the penalty to ~1e-4..1e-3 against a
    # ~O(0.1-1) cross-entropy loss at weight=1.0: numerically negligible, not a
    # verdict on the method. A first run at weight=1 showed `coral` tied
    # `erm_torch` to 3 decimals on every split -- so this sweeps the weight
    # over several orders of magnitude instead of asserting one value. Kept as
    # separate spec keys, not a CLI flag, so one run reports the whole sweep.
    "domain_coral_w1": {
        "label": "roberta·coral_w1", "alg": "robust_mlp", "aux": False,
        "params": {**_BASE_PARAMS, "domain_penalty": "coral",
                  "domain_penalty_weight": 1.0},
    },
    "domain_coral_w100": {
        "label": "roberta·coral_w100", "alg": "robust_mlp", "aux": False,
        "params": {**_BASE_PARAMS, "domain_penalty": "coral",
                  "domain_penalty_weight": 100.0},
    },
    "domain_coral_w10000": {
        "label": "roberta·coral_w10000", "alg": "robust_mlp", "aux": False,
        "params": {**_BASE_PARAMS, "domain_penalty": "coral",
                  "domain_penalty_weight": 10000.0},
    },
    # V-REx's natural scale (variance of per-domain mean CE) is bigger than
    # CORAL's un-scaled penalty but still small next to the CE term itself --
    # weight=1 was already slightly *worse* than erm_torch once, consistent
    # with "present but small", so this sweeps a smaller range than CORAL's.
    "domain_vrex_w1": {
        "label": "roberta·vrex_w1", "alg": "robust_mlp", "aux": False,
        "params": {**_BASE_PARAMS, "domain_penalty": "vrex",
                  "domain_penalty_weight": 1.0},
    },
    "domain_vrex_w10": {
        "label": "roberta·vrex_w10", "alg": "robust_mlp", "aux": False,
        "params": {**_BASE_PARAMS, "domain_penalty": "vrex",
                  "domain_penalty_weight": 10.0},
    },
    "domain_vrex_w100": {
        "label": "roberta·vrex_w100", "alg": "robust_mlp", "aux": False,
        "params": {**_BASE_PARAMS, "domain_penalty": "vrex",
                  "domain_penalty_weight": 100.0},
    },
    "tail_toppush_coral": {
        "label": "roberta·toppush+coral_w1", "alg": "robust_mlp", "aux": False,
        "params": {**_BASE_PARAMS, "loss_kind": "toppush", "push_frac": 0.05,
                  "push_weight": 1.0, "push_margin": 1.0,
                  "domain_penalty": "coral", "domain_penalty_weight": 1.0},
    },
}

#: existing detectors + the new arms, one dict every compute function iterates.
ALL_SPECS: dict[str, dict] = {**TRAINED_SPECS, **ROBUST_SPECS}


def fit_robust(spec, X, y, Z, groups, dom, idx, seed: int):
    """Like ``_trained.fit``, but also tells the algorithm which domain each
    training row belongs to (harmless no-op for every spec that doesn't read
    it -- every existing ``_trained.py`` spec included)."""
    clf = build_algorithm(spec["alg"], seed, **spec["params"])
    clf.set_fit_groups(groups[idx])
    clf.set_fit_domains(dom[idx])
    if spec["aux"]:
        clf.fit(X[idx], y[idx], Z[idx])
    else:
        clf.fit(X[idx], y[idx])
    return clf


# ---------------------------------------------------------------------------
# Regime: in-domain 70/30 -- same row shape as
# run_in_domain_performance.compute_trained, over ALL_SPECS
# ---------------------------------------------------------------------------


def compute_in_domain(train_splits, ood_splits, seeds, test_size, fprs,
                      human_label, combine) -> list[dict]:
    ood_sets: dict[str, tuple] = {}
    for name, path in ood_splits:
        oX, oy, oZ, n_h = _load_ood(path, human_label)
        ood_sets[name] = (oX, oy, oZ, n_h)
        print(f"  OOD eval corpus {name!r}: {len(oy)} rows, {n_h} human "
              f"(never trained on)")

    loaded = [(k, _load(p, human_label)) for k, p in train_splits]

    cells: dict[str, dict[float, list[dict]]] = {k: {f: [] for f in fprs} for k in ALL_SPECS}
    ood_reads: dict[str, dict[str, dict[float, list]]] = {
        k: {n: {f: [] for f in fprs} for n in ood_sets} for k in ALL_SPECS}

    for split_key, (X, y, Z, dom, ids) in tqdm(loaded, desc="corpus", unit="split"):
        for spec_key, spec in ALL_SPECS.items():
            a: list[float] = []
            t = {f: [] for f in fprs}
            te_ = {f: [] for f in fprs}
            n_tr = n_te = n_te_h = 0
            for seed in seeds:
                tr, te = grouped_split(y, ids, test_size, seed)
                clf = fit_robust(spec, X, y, Z, ids, dom, tr, seed)
                au, per = eval_on_fprs(clf, spec, X[te], y[te], Z[te], fprs)
                a.append(au)
                for f in fprs:
                    t[f].append(per[f][0])
                    te_[f].append(per[f][1])
                n_tr, n_te, n_te_h = len(tr), len(te), int((y[te] == 0).sum())
                for name, (oX, oy, oZ, _) in ood_sets.items():
                    oau, oper = eval_on_fprs(clf, spec, oX, oy, oZ, fprs)
                    for f in fprs:
                        ood_reads[spec_key][name][f].append((oau, oper[f][0], oper[f][1]))
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
    for spec_key, spec in ALL_SPECS.items():
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


# ---------------------------------------------------------------------------
# Regime: leave-one-domain-out -- same row shape as
# run_lodo_performance.compute_trained, over ALL_SPECS
# ---------------------------------------------------------------------------


def _fold_robust(spec, X, y, Z, groups, dom, tr, te, seeds, fprs, ood_sets) -> dict:
    """Mirror of ``run_lodo_performance._fold``, fit through :func:`fit_robust`
    so the fold's training union can also be told its own internal domains."""
    a: list[float] = []
    t = {f: [] for f in fprs}
    te_ = {f: [] for f in fprs}
    ood: dict[str, dict[float, list]] = {n: {f: [] for f in fprs} for n in ood_sets}
    for seed in seeds:
        clf = fit_robust(spec, X, y, Z, groups, dom, tr, seed)
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
                 min_domain_rows, combine) -> list[dict]:
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

    all_folds: dict[str, dict[str, list[dict]]] = {}
    ood_reads: dict[str, dict[str, dict[float, list]]] = {
        k: {n: {f: [] for f in fprs} for n in ood_sets} for k in ALL_SPECS
    }

    for split_key, X, y, Z, dom, ids, domains in tqdm(loaded, desc="LODO", unit="split"):
        spec_folds = {k: [] for k in ALL_SPECS}
        for d in domains:
            tr, te = dom != d, dom == d
            assert d not in set(dom[tr]) and set(dom[te]) == {d}
            assert tr.sum() + te.sum() == len(dom) and not (tr & te).any()
            assert set(ids[tr]).isdisjoint(ids[te]), "a group straddled the fold"
            for spec_key, spec in ALL_SPECS.items():
                fd = _fold_robust(spec, X, y, Z, ids, dom, tr, te, seeds, fprs, ood_sets)
                fd["held_out_domain"] = d
                for name, by_fpr in fd.pop("ood").items():
                    for f in fprs:
                        ood_reads[spec_key][name][f].extend(by_fpr[f])
                spec_folds[spec_key].append(fd)
        all_folds[split_key] = spec_folds

    rows: list[dict] = []
    for spec_key, spec in ALL_SPECS.items():
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
    for fpr in meta["fprs"]:
        fmeta = {**meta, "fpr": fpr}
        tag = fpr_tag(fpr)
        frows = rows_at_fpr(rows, fpr)
        zeroshot_scatter(
            frows, fmeta, out_dir / f"in_domain_{tag}.png", key="detector",
            title=f"In-domain (70/30, {n_seeds} seeds) — tail-loss / domain-"
                  f"invariance arms vs. existing trained + zero-shot picks, "
                  f"plus {ood} as eval-only OOD\nAUROC vs. oracle "
                  f"TPR@{fpr * 100:g}%FPR",
        )
        goodness_bars(
            frows, fmeta, out_dir / f"in_domain_goodness_{tag}.png", key="detector",
            combine=meta.get("combine", "hmean"),
            title=f"In-domain (70/30, {n_seeds} seeds) — goodness "
                  f"(TPR@{fpr * 100:g}%FPR) across the MANUGRAM splits",
        )
        denario_scatter(
            frows, fmeta, out_dir / f"in_domain_denario_{tag}.png",
            title=f"In-domain (70/30, {n_seeds} seeds) on {ood} (eval-only OOD)"
                  f"\nAUROC vs. oracle TPR@{fpr * 100:g}%FPR",
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
            title=f"Leave-one-domain-out — tail-loss / domain-invariance arms "
                  f"vs. existing trained + zero-shot picks, plus {ood} as "
                  f"eval-only OOD\nAUROC vs. oracle TPR@{fpr * 100:g}%FPR",
        )
        goodness_bars(
            rows, fmeta, out_dir / f"lodo_goodness_{tag}.png", key="detector",
            combine=meta.get("combine", "hmean"),
            title=f"Leave-one-domain-out — goodness (TPR@{fpr * 100:g}%FPR) "
                  f"across the MANUGRAM splits",
        )
        denario_scatter(
            rows, fmeta, out_dir / f"lodo_denario_{tag}.png",
            title=f"Leave-one-domain-out on {ood} (eval-only OOD)\nAUROC vs. "
                  f"oracle TPR@{fpr * 100:g}%FPR",
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
    ap.add_argument("--zeroshot-config", type=Path, default=ZEROSHOT_CONFIG)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS, metavar="LEVEL",
                    help=f"oracle FPR level(s); one figure set per level "
                         f"(default: {DEFAULT_FPRS})")
    ap.add_argument("--combine", choices=("hmean", "gmean", "amean"), default="hmean")
    ap.add_argument("--min-domain-rows", type=int, default=100)
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="OUT_DIR",
                    help="re-plot from an existing out-dir's CSV(s), no compute")
    args = ap.parse_args()

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

    print(f"detectors : {[s['label'] for s in TRAINED_SPECS.values()]} "
          f"(existing) + {[s['label'] for s in ROBUST_SPECS.values()]} (new)")
    print(f"regimes   : {args.regimes}")
    print(f"train on  : {[s for s, _ in train_splits]}   OOD: {[s for s, _ in ood_splits]}"
          f"   seeds: {args.seeds}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if "in_domain" in args.regimes:
        rows = compute_in_domain(train_splits, ood_splits, args.seeds,
                                 args.test_size, args.fpr, args.human_label,
                                 args.combine)
        if not args.no_zeroshot:
            rows += compute_zeroshot_in_domain(
                train_splits, ood_splits, args.zeroshot_config, args.seeds,
                args.test_size, args.fpr, "id", args.human_label, args.combine)
        csv_path = args.out_dir / "in_domain_performance.csv"
        pd.DataFrame(rows, columns=CSV_FIELDS_IN_DOMAIN).to_csv(csv_path, index=False)
        _print_table_in_domain(rows, args.fpr)
        print(f"\nwrote {csv_path}")
        if not args.no_plots:
            plots_in_domain(csv_path, meta, args.out_dir)

    if "lodo" in args.regimes:
        rows = compute_lodo(train_splits, ood_splits, args.seeds, args.fpr,
                            args.human_label, args.min_domain_rows, args.combine)
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

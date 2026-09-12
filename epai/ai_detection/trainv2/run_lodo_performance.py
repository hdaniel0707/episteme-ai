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
in full (default MG-AI-Sci) — never trained on, an additional held-out
test corpus. Its reading is the mean over every (training split, held-out
domain, seed) model.

Each fold is averaged over N seeds (only the logistic ``random_state`` moves,
so this just steadies the fold reading). The reported per-split number is the
mean over the domain folds; **its SD is the spread ACROSS those folds** — how
much the reading depends on which domain is unseen, i.e. the domain-transfer
robustness. The 'overall' row is the mean across splits; its SD is the typical
per-split fold SD (mean of them), never the spread across splits.

Only splits with >= 2 domains of at least ``--min-domain-rows`` rows are used
(MG-AI-Sci's domains run ~8-40 rows each, post-merge with the ai_scientist
corpus -- still no LODO there), so the default is GB-orig / MG-GB / MG-Sci,
3 domains each.

The zero-shot picks from ``cfg/ai_detection/trainv2/best_zeroshot.yaml`` are
scored on the SAME rows as the trained models: each held-out domain fold per
LODO split (per-split reading = mean +- SD across the domain folds, exactly
like a trained one), and the full corpus per OOD split (one deterministic
reading, no error bar). Not on the whole dataset — so the two detector kinds
are compared on identical data.

The goodness bar folds AUROC + oracle TPR into one score PER FOLD, so its bar
carries the across-fold SD too (``goodness`` / ``goodness_std`` in the CSV).

``--fpr`` takes one or more oracle FPR levels (default 1% and 5%). The models
are trained once; each level is a different ROC readout. The CSV carries an
``fpr`` column (one fold/split/overall row set per level) and each level gets
its own figure set, suffixed ``_1pct`` / ``_5pct``.

compute -> CSV -> plot from the CSV:

    results/trainv2/lodo_performance/
        performance.csv   level (fold/split/overall), kind, detector, split,
                          held_out_domain, fpr, n_test, n_folds, n_seeds,
                          auroc(_std), tpr_at_fpr(_std), tpr_eff(_std), eff_fpr,
                          goodness(_std)
        lodo_1pct.png / lodo_5pct.png   (+ _goodness_, _denario_)

Run:
    uv run python epai/ai_detection/trainv2/run_lodo_performance.py
    uv run python epai/ai_detection/trainv2/run_lodo_performance.py --seeds 0 1 2
    uv run python epai/ai_detection/trainv2/run_lodo_performance.py \
        --replot results/trainv2/lodo_performance/performance.csv
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from epai.ai_detection.trainv2._trained import (
    AUX_COL,
    EMB_COL,
    MULTI_EMB_COLS,
    MULTI_EMB_SPECS,
    TRAINED_SPECS,
    _stack,
    eval_on_fprs,
    fit,
    load_multi_features,
    load_multi_ood,
    with_emb_dims,
)
from epai.ai_detection.trainv2._zeroshot import (
    agg_cell,
    eval_column_fprs,
    orient,
    overall_cell,
    resolve_specs,
)
from epai.ai_detection.trainv2.core import eff_fpr_level, fpr_tag, goodness, rows_at_fpr
from epai.ai_detection.trainv2.data import GEN_COL, _is_missing
from epai.ai_detection.trainv2.run_best_zeroshot_performance import DEFAULT_CONFIG
from epai.ai_detection.trainv2.run_zeroshot_performance import DEFAULT_FPRS
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "lodo_performance"
DEFAULT_SPLITS = ["gb_orig", "mg_gb", "mg_sci"]
DEFAULT_OOD = ["mg_aisci"]

DOMAIN_COL = "domain"
GROUP_COL = "id"

CSV_FIELDS = ["level", "kind", "detector", "split", "held_out_domain", "fpr",
              "n_test", "n_folds", "n_seeds", "auroc", "auroc_std", "tpr_at_fpr",
              "tpr_at_fpr_std", "tpr_eff", "tpr_eff_std", "eff_fpr",
              "goodness", "goodness_std"]


def _load(path: Path, human_label: str, *, score_cols: tuple[str, ...] = ()):
    """``(X_emb, y, Z_cue, dom, ids)``. Group dropped whole on ANY missing
    feature; a group must sit in one domain (checked).

    With ``score_cols`` (raw zero-shot columns, all present in the parquet), a
    6th element ``{col: float_array}`` aligned to the surviving rows is
    returned -- for scoring a zero-shot pick on each held-out domain fold."""
    cols = list(dict.fromkeys([GROUP_COL, DOMAIN_COL, GEN_COL, EMB_COL, AUX_COL,
                               *score_cols]))
    df = pd.read_parquet(path, columns=cols)
    bad = df[EMB_COL].map(_is_missing) | df[AUX_COL].map(_is_missing)
    if bad.any():
        df = df[~df[GROUP_COL].isin(df.loc[bad, GROUP_COL].unique())]
    df = df.reset_index(drop=True)
    span = df.groupby(GROUP_COL)[DOMAIN_COL].nunique()
    if (span > 1).any():
        raise SystemExit(f"{path.name}: {int((span > 1).sum())} group(s) span "
                         f"more than one domain — LODO would split them")
    out = (_stack(df[EMB_COL]),
           (df[GEN_COL].to_numpy() != human_label).astype(int),
           _stack(df[AUX_COL]), df[DOMAIN_COL].astype(str).to_numpy(),
           df[GROUP_COL].to_numpy())
    if score_cols:
        return (*out, {c: df[c].to_numpy(dtype=float) for c in score_cols})
    return out


def _load_ood(path: Path, human_label: str, *, score_cols: tuple[str, ...] = ()):
    """Full-corpus eval set (never trained on): (X, y, Z, n_human). With
    ``score_cols``, a 5th element ``{col: float_array}`` aligned to the same
    rows (feature-missing rows dropped) is returned."""
    cols = list(dict.fromkeys([GEN_COL, EMB_COL, AUX_COL, *score_cols]))
    df = pd.read_parquet(path, columns=cols)
    bad = df[EMB_COL].map(_is_missing) | df[AUX_COL].map(_is_missing)
    df = df[~bad].reset_index(drop=True)
    y = (df[GEN_COL].to_numpy() != human_label).astype(int)
    out = _stack(df[EMB_COL]), y, _stack(df[AUX_COL]), int((y == 0).sum())
    if score_cols:
        return (*out, {c: df[c].to_numpy(dtype=float) for c in score_cols})
    return out


def _fold(spec, X, y, Z, groups, tr, te, seeds, fprs, ood_sets) -> dict:
    """One LODO fold for one spec, averaged over seeds, at every FPR level. Each
    fold's model also scores every OOD eval set (once, all levels)."""
    a: list[float] = []
    t = {f: [] for f in fprs}
    te_ = {f: [] for f in fprs}
    ood: dict[str, dict[float, list]] = {n: {f: [] for f in fprs} for n in ood_sets}
    for seed in seeds:
        clf = fit(spec, X, y, Z, groups, tr, seed)
        au, per = eval_on_fprs(clf, spec, X[te], y[te], Z[te] if Z is not None else None, fprs)
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


def _std(vals: list[float]) -> float:
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.std(vals)) if len(vals) > 1 else 0.0


def compute_trained(splits, ood_splits, seeds, fprs, human_label,
                    min_domain_rows, combine) -> list[dict]:
    ood_sets = {name: _load_ood(path, human_label) for name, path in ood_splits}
    for name, (_, oy, _, n_h) in ood_sets.items():
        print(f"  OOD eval corpus {name!r}: {len(oy)} rows, {n_h} human (never trained on)")

    # per split -> per spec -> list of fold dicts (each fold carries `by_fpr`)
    all_folds: dict[str, dict[str, list[dict]]] = {}
    # per spec -> per ood name -> per fpr -> list of (auroc, tpr, tpr_eff)
    ood_reads: dict[str, dict[str, dict[float, list]]] = {
        k: {n: {f: [] for f in fprs} for n in ood_sets} for k in TRAINED_SPECS
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
                fd = _fold(spec, X, y, Z, ids, tr, te, seeds, fprs, ood_sets)
                fd["held_out_domain"] = d
                for name, by_fpr in fd.pop("ood").items():
                    for f in fprs:
                        ood_reads[spec_key][name][f].extend(by_fpr[f])
                spec_folds[spec_key].append(fd)
        all_folds[split_key] = spec_folds

    if not all_folds:
        return []

    rows: list[dict] = []
    for spec_key, spec in TRAINED_SPECS.items():
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
                og = [goodness(a, te, combine) for a, te in zip(oa, ote)]
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


def _multi_emb_lodo_task(split_key, path, spec_key, spec, embeddings, ood_splits,
                         seeds, fprs, human_label, min_domain_rows, cpu_threads):
    """One (split, architecture) unit of work for :func:`compute_multi_emb`,
    self-contained -- see :func:`run_in_domain_performance._multi_emb_task`
    for why (process-global RNG state under ``fit``, and avoiding pickling a
    multi-embedding matrix across the process boundary once per task).

    Returns ``(spec_key, split_key, fold_dicts, ood_reads, ood_n_human)`` --
    ``fold_dicts`` (each already stripped of ``_fold``'s ``'ood'`` key) and
    ``ood_reads`` (``{name: {fpr: [(auroc, tpr, tpr_eff), ...]}}``, one triple
    per fold, matching every other spec's shape) are empty when the split has
    fewer than 2 usable domains.
    """
    if cpu_threads:
        torch.set_num_threads(cpu_threads)
    X, y, groups, dom, dims = load_multi_features(
        path, GROUP_COL, human_label, embeddings, domain_col=DOMAIN_COL)
    run_spec = with_emb_dims(spec, dims, embeddings)
    counts = {d: int((dom == d).sum()) for d in np.unique(dom)}
    domains = sorted(d for d, n in counts.items()
                     if n >= min_domain_rows and np.unique(y[dom == d]).size == 2)
    if len(domains) < 2:
        return spec_key, split_key, [], {}, {}

    ood_sets = {}
    for name, opath in ood_splits:
        oX, oy, on_h, _odims = load_multi_ood(opath, human_label, embeddings)
        ood_sets[name] = (oX, oy, None, on_h)

    folds = []
    ood_reads: dict[str, dict[float, list]] = {n: {f: [] for f in fprs} for n in ood_sets}
    for d in domains:
        tr, te = dom != d, dom == d
        assert d not in set(dom[tr]) and set(dom[te]) == {d}
        assert tr.sum() + te.sum() == len(dom) and not (tr & te).any()
        assert set(groups[tr]).isdisjoint(groups[te]), "a group straddled the fold"
        fd = _fold(run_spec, X, y, None, groups, tr, te, seeds, fprs, ood_sets)
        fd["held_out_domain"] = d
        for name, by_fpr in fd.pop("ood").items():
            for f in fprs:
                ood_reads[name][f].extend(by_fpr[f])
        folds.append(fd)
    ood_n_h = {name: n_h for name, (_, _, _, n_h) in ood_sets.items()}
    return spec_key, split_key, folds, ood_reads, ood_n_h


def compute_multi_emb(splits, ood_splits, seeds, fprs, human_label,
                      min_domain_rows, combine, embeddings=MULTI_EMB_COLS,
                      jobs=1, cpu_threads=None) -> list[dict]:
    """LODO for the multi-embedding architectures (``MULTI_EMB_SPECS``), on the
    concatenation of ``embeddings``. Identical fold procedure to
    :func:`compute_trained` (reuses :func:`_fold` unchanged) -- kept as a
    separate function rather than folded into ``compute_trained`` because
    ``compute_trained`` is imported directly by ``run_cascade_performance.py``
    / ``run_dr_transfer.py`` / ``run_robust_performance.py`` for their own,
    unrelated comparisons -- see ``_trained.py``'s module docstring.

    ``jobs > 1`` fits every (split, architecture) pair -- all of that split's
    domain folds -- in its own OS process (``ProcessPoolExecutor``, ``spawn``
    context; see :func:`_multi_emb_lodo_task` for why not a thread pool).
    ``jobs <= 1`` runs the exact same task calls in a plain loop in this
    process, so the two modes compute identical numbers and only differ in
    wall-clock time.
    """
    tasks = [(split_key, path, spec_key, spec)
             for split_key, path in splits
             for spec_key, spec in MULTI_EMB_SPECS.items()]

    results = []
    if jobs > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            futs = [
                ex.submit(_multi_emb_lodo_task, split_key, path, spec_key, spec,
                         embeddings, ood_splits, seeds, fprs, human_label,
                         min_domain_rows, cpu_threads)
                for split_key, path, spec_key, spec in tasks
            ]
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc="LODO multi-emb (parallel)", unit="task"):
                results.append(fut.result())
    else:
        for split_key, path, spec_key, spec in tqdm(tasks, desc="LODO multi-emb", unit="task"):
            results.append(_multi_emb_lodo_task(split_key, path, spec_key, spec, embeddings,
                                                ood_splits, seeds, fprs, human_label,
                                                min_domain_rows, cpu_threads))

    all_folds: dict[str, dict[str, list[dict]]] = {}
    ood_reads: dict[str, dict[str, dict[float, list]]] = {
        k: {name: {f: [] for f in fprs} for name, _ in ood_splits} for k in MULTI_EMB_SPECS
    }
    ood_n_h: dict[str, int] = {}
    for spec_key, split_key, folds, task_ood_reads, task_ood_n_h in results:
        if not folds:
            continue
        all_folds.setdefault(split_key, {k: [] for k in MULTI_EMB_SPECS})[spec_key] = folds
        for name, by_fpr in task_ood_reads.items():
            for f in fprs:
                ood_reads[spec_key][name][f].extend(by_fpr[f])
        ood_n_h.update(task_ood_n_h)

    if not all_folds:
        return []

    rows: list[dict] = []
    for spec_key, spec in MULTI_EMB_SPECS.items():
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

            for name, n_h in ood_n_h.items():
                reads = ood_reads[spec_key][name][fpr]
                oa, ot = [r[0] for r in reads], [r[1] for r in reads]
                ote = [r[2] for r in reads]
                og = [goodness(a, te, combine) for a, te in zip(oa, ote)]
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


def _zs_row(det: str, level: str, split: str, domain: str, cell: dict,
            n_folds, n_seeds) -> dict:
    return {
        "level": level, "kind": "zeroshot", "detector": det, "split": split,
        "held_out_domain": domain, "fpr": cell.get("fpr", ""),
        "n_test": cell.get("n_test", ""), "n_folds": n_folds, "n_seeds": n_seeds,
        "auroc": cell["auroc"], "auroc_std": cell.get("auroc_std", ""),
        "tpr_at_fpr": cell["tpr_at_fpr"], "tpr_at_fpr_std": cell.get("tpr_at_fpr_std", ""),
        "tpr_eff": cell.get("tpr_eff", ""), "tpr_eff_std": cell.get("tpr_eff_std", ""),
        "eff_fpr": cell.get("eff_fpr", ""),
        "goodness": cell.get("goodness", ""), "goodness_std": cell.get("goodness_std", ""),
    }


def compute_zeroshot(splits, ood_splits, config, seeds, fprs, human_label,
                     min_domain_rows, combine) -> list[dict]:
    """The ``best_zeroshot.yaml`` picks, scored on the trained models' own eval
    rows: each held-out domain fold per LODO split (per-split reading = mean +-
    SD across the domain folds, matching the trained per-split SD), and the full
    corpus per OOD split (one deterministic reading). ``fold`` / ``split`` /
    ``overall`` rows per FPR level, same as ``compute_trained``."""
    import pyarrow.parquet as pq

    n_seeds = len(seeds)
    # detector -> fpr -> list of per-corpus cells
    per_det: dict[str, dict[float, list[dict]]] = {}
    rows: list[dict] = []

    def _hum(y, mask):
        return int((y[mask] == 0).sum())

    for split_key, path in splits:
        specs = resolve_specs(pq.read_schema(path).names, config)
        if not specs:
            continue
        _, y, _, dom, _, raw = _load(
            path, human_label, score_cols=tuple(c for c, _, _ in specs))
        counts = {d: int((dom == d).sum()) for d in np.unique(dom)}
        domains = sorted(d for d, n in counts.items()
                         if n >= min_domain_rows
                         and np.unique(y[dom == d]).size == 2)
        if len(domains) < 2:
            continue
        for col, det, machine_is_high in specs:
            o = orient(raw[col], machine_is_high)
            dom_reads = {d: eval_column_fprs(o[dom == d], y[dom == d], fprs)
                         for d in domains}
            for f in fprs:
                fold_reads = []
                for d in domains:
                    au, per = dom_reads[d]
                    tpr, teff = per[f]
                    fold_reads.append((au, tpr, teff))
                    gd = (goodness(au, teff, combine)
                          if np.isfinite(au) and np.isfinite(teff) else float("nan"))
                    rows.append(_zs_row(det, "fold", split_key, d, {
                        "auroc": au, "tpr_at_fpr": tpr, "tpr_eff": teff, "fpr": f,
                        "eff_fpr": eff_fpr_level(_hum(y, dom == d), f),
                        "goodness": gd, "n_test": int((dom == d).sum())}, "", n_seeds))
                cell = agg_cell([r[0] for r in fold_reads],
                                [r[1] for r in fold_reads],
                                [r[2] for r in fold_reads], combine)
                cell["fpr"] = f
                cell["reachable"] = any(_hum(y, dom == d) >= 1 / f for d in domains)
                cell["eff_fpr"] = float(np.mean(
                    [eff_fpr_level(_hum(y, dom == d), f) for d in domains]))
                rows.append(_zs_row(det, "split", split_key, "", cell,
                                    len(domains), n_seeds))
                per_det.setdefault(det, {}).setdefault(f, []).append(
                    {**cell, "split": split_key})

    for split_key, path in ood_splits:
        specs = resolve_specs(pq.read_schema(path).names, config)
        if not specs:
            continue
        # same rows as the trained OOD reading (_load_ood: feature-missing rows
        # dropped), plus the score columns aligned.
        _, y2, _, _, raw = _load_ood(
            path, human_label, score_cols=tuple(c for c, _, _ in specs))
        n_h = int((y2 == 0).sum())
        for col, det, machine_is_high in specs:
            au, per = eval_column_fprs(orient(raw[col], machine_is_high), y2, fprs)
            for f in fprs:
                tpr, teff = per[f]
                gd = (goodness(au, teff, combine)
                      if np.isfinite(au) and np.isfinite(teff) else float("nan"))
                cell = {
                    "fpr": f, "auroc": au, "auroc_std": float("nan"),
                    "tpr_at_fpr": tpr, "tpr_at_fpr_std": float("nan"),
                    "tpr_eff": teff, "tpr_eff_std": float("nan"),
                    "goodness": gd, "goodness_std": float("nan"),
                    "eff_fpr": eff_fpr_level(n_h, f), "n_test": int(len(y2)),
                    "reachable": n_h >= 1 / f,
                }
                rows.append(_zs_row(det, "split", split_key, "", cell, "", n_seeds))
                per_det.setdefault(det, {}).setdefault(f, []).append(
                    {**cell, "split": split_key})

    for det, by_fpr in per_det.items():
        for f, cells in by_fpr.items():
            mean = overall_cell(cells, f)
            mean["fpr"] = f
            rows.append(_zs_row(det, "overall", "mean", "", mean, "", n_seeds))
    return rows


def plots_from_csv(csv_path: Path, meta: dict, out_dir: Path) -> None:
    all_rows = [r for r in pd.read_csv(csv_path).to_dict("records")
                if r["level"] in ("split", "overall")]
    from epai.ai_detection.trainv2.plots import (
        denario_scatter,
        goodness_bars,
        zeroshot_scatter,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    ood = "/".join(meta["ood"])
    for fpr in meta["fprs"]:
        fmeta = {**meta, "fpr": fpr}
        tag = fpr_tag(fpr)
        rows = rows_at_fpr(all_rows, fpr)
        zeroshot_scatter(
            rows, fmeta, out_dir / f"lodo_{tag}.png", key="detector",
            title=f"Leave-one-domain-out, plus {ood} as eval-only OOD — trained "
                  f"vs. zero-shot picks on the same eval rows\nAUROC vs. oracle "
                  f"TPR@{fpr * 100:g}%FPR — per-split bar = across-fold SD, "
                  f"mean marker = mean per-split SD",
        )
        goodness_bars(
            rows, fmeta, out_dir / f"lodo_goodness_{tag}.png", key="detector",
            combine=meta.get("combine", "hmean"),
            title=f"Leave-one-domain-out trained + zero-shot picks — goodness "
                  f"(TPR@{fpr * 100:g}%FPR) across the MANUGRAM splits",
        )
        denario_scatter(
            rows, fmeta, out_dir / f"lodo_denario_{tag}.png",
            title=f"Leave-one-domain-out trained on {ood} (eval-only OOD)\n"
                  f"same detectors — AUROC vs. oracle TPR@{fpr * 100:g}%FPR",
        )


def _print_table(rows: list[dict], fprs: list[float]) -> None:
    print(f"\n{'kind':<9}{'detector':<24}{'split':<9}{'domain':<15}{'FPR':>6}"
          f"{'auroc':>15}{'TPR':>15}")
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
        print(f"{r['kind']:<9}{r['detector']:<24}{r['split']:<9}{dom:<15}"
              f"{float(r['fpr']) * 100:>5g}%{a:>15}{t:>15}{mark}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--splits", nargs="+", default=None, metavar="KEY",
                    help=f"MANUGRAM splits to LODO over (default: {DEFAULT_SPLITS})")
    ap.add_argument("--ood-splits", nargs="+", default=None, metavar="KEY",
                    help=f"eval-only corpora, never trained on (default: {DEFAULT_OOD})")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS,
                    metavar="LEVEL",
                    help=f"oracle FPR level(s); one figure set per level "
                         f"(default: {DEFAULT_FPRS})")
    ap.add_argument("--combine", choices=("hmean", "gmean", "amean"), default="hmean",
                    help="goodness-bar score = this mean of Gini and oracle TPR")
    ap.add_argument("--min-domain-rows", type=int, default=100)
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--zeroshot-config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--multi-embeddings", nargs="+", default=None, metavar="EMB_COL",
                    help=f"embeddings the 4 multi-embedding architectures "
                         f"concatenate (default: {list(MULTI_EMB_COLS)})")
    ap.add_argument("--no-multi-emb", action="store_true",
                    help="skip the 4 multi-embedding architectures (emb_stacking, "
                         "emb_early_fusion, emb_gated_n, emb_attn_fusion)")
    ap.add_argument("--jobs", type=int, default=1, metavar="N",
                    help="fit N (split, multi-embedding architecture) pairs at once "
                         "-- each running all of that split's domain folds -- each "
                         "in its own process (default: 1, i.e. serial). Does not "
                         "affect the roberta_logistic baseline, which stays serial "
                         "-- see compute_multi_emb's docstring for why a process "
                         "pool, not a thread pool")
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
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"default: a fresh {DEFAULT_OUT_DIR.name}_<YYYYMMDD_HHMM> "
                         f"directory under {DEFAULT_OUT_DIR.parent}/, so past runs "
                         f"are never overwritten")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV")
    args = ap.parse_args()
    out_dir = args.out_dir or (
        DEFAULT_OUT_DIR.parent / f"{DEFAULT_OUT_DIR.name}_{datetime.now():%Y%m%d_%H%M}"
    )

    # Applied before any `fit()` call (the only place a torch algorithm ever
    # touches CUDA -- see epai/algs/*.py): CUDA device enumeration and the
    # BLAS/OpenMP thread pools are triggered lazily, on first use, not merely
    # by `import torch` above, so setting these here -- rather than needing
    # add_embeddings.py's "parse argv before importing torch" trick -- is
    # still early enough. A --jobs worker is a freshly spawned process that
    # inherits this environment from scratch, so it applies there too.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.cpu_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
        torch.set_num_threads(args.cpu_threads)

    splits = resolve_splits(args.splits or DEFAULT_SPLITS, PARQUET_DIR)
    ood_splits = resolve_splits(args.ood_splits or DEFAULT_OOD, PARQUET_DIR)
    meta = {"fprs": args.fpr, "n_seeds": len(args.seeds), "combine": args.combine,
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
                           args.human_label, args.min_domain_rows, args.combine)
    if not any(r["level"] == "overall" for r in rows):
        raise SystemExit("no split had >= 2 usable domains")

    if not args.no_multi_emb:
        multi_emb = tuple(args.multi_embeddings) if args.multi_embeddings else MULTI_EMB_COLS
        print(f"\nmulti-embedding architectures: {list(MULTI_EMB_SPECS)} on {multi_emb}"
              f", leave-one-domain-out  (--jobs {args.jobs})")
        rows += compute_multi_emb(splits, ood_splits, args.seeds, args.fpr,
                                  args.human_label, args.min_domain_rows,
                                  args.combine, multi_emb, args.jobs, args.cpu_threads)

    if not args.no_zeroshot:
        print(f"zero-shot: {args.zeroshot_config} "
              f"(scored on the held-out domain folds)")
        rows += compute_zeroshot(splits, ood_splits, args.zeroshot_config,
                                 args.seeds, args.fpr, args.human_label,
                                 args.min_domain_rows, args.combine)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "performance.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    _print_table(rows, args.fpr)
    print(f"\nwrote {csv_path}")

    if not args.no_plots:
        plots_from_csv(csv_path, meta, out_dir)


if __name__ == "__main__":
    main()

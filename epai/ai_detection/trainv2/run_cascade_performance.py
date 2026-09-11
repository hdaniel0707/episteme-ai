r"""Judge-anchored cascade (selective escalation) -- leave-one-domain-out,
scored on MG-AI-Sci. See ``epai/ai_detection/trainv2/improving_sota_ideas.md``
(#3) for the method and citations.

An **opt-in add-on**, run by hand -- same treatment as ``run_dr_transfer.py`` /
``run_robust_performance.py``: absent from ``rebuild_all.TASKS``, and
``_trained.py`` / ``core.py`` / ``splits.py`` / the main runners /
``rebuild_all.py`` are only ever imported from, never edited.

The cheap detector (``trained/roberta·logistic``, unmodified) is fit as usual
on a LODO fold's training union, but a **calibration slice** is first carved
out of that union (grouped, disjoint from the fit rows -- the same discipline
as the outer split). On the calibration slice: find the score threshold at
each ``--fpr`` level, then rank calibration rows by their margin
(``|score - threshold|``) -- a split-conformal-style nonconformity measure.
For an eval row (the held-out domain, or the OOD corpus), **escalate** it to
the zero-shot judge (``llm_judge_score_gpt56luna_v3_wlp``, the strongest
zero-shot signal on this corpus) when its margin falls in the calibration
set's bottom ``alpha`` quantile -- the least-confident, most likely
near-threshold rows.

The trained probe's raw score and the judge's raw score live on different,
unrelated numeric scales, so an escalated row can't just be given the judge's
raw value in place of the trained one -- AUROC reads the *joint* ranking of
the blended array, and a value from the wrong scale lands in it by
coincidence, not by relative confidence (an earlier version of this script did
exactly that, and made performance *worse* the more it escalated -- a
tell-tale sign, not a negative result about the method). Both scores are
first transformed to their own percentile rank within the SAME calibration
slice (:func:`_ecdf_percentile`); escalated rows plug in the judge's
percentile, every other row keeps the trained probe's -- now directly
comparable, in [0, 1], relative to one shared reference population. The
escalation DECISION itself is unaffected (it's a monotonic transform of the
same raw margin). Reported as one pseudo-detector per escalation budget
(``cascade@5pct`` / ``cascade@10pct`` / ``cascade@20pct`` by default).

This is a practical approximation of split-conformal calibration (the
calibration and eval margin distributions are assumed similar, not proven
exchangeable), good enough to read the coverage-vs-TPR@1%FPR trade-off, not a
publication-grade coverage guarantee -- flag that plainly in the write-up.
The fixed-margin-band baseline routing rule named in ``improving_sota_ideas.md``
is not computed here yet; this first cut is the conformal-budget version only.

Reported alongside the existing three ``_trained.py`` LODO detectors and the
``best_zeroshot.yaml`` picks (both **unmodified imports** from
``run_lodo_performance.py`` -- this script adds no new fits for them), on the
SAME figures (``zeroshot_scatter`` / ``goodness_bars`` / ``denario_scatter``)
the main LODO runner already produces.

compute -> CSV -> plot from the CSV:

    results/trainv2/cascade_performance/
        lodo_performance.csv
        lodo_{1pct,5pct}.png (+ _goodness_, _denario_)

Run (sandbox has no ``uv`` -- use ``.venv/bin/python`` directly):

    .venv/bin/python epai/ai_detection/trainv2/run_cascade_performance.py
    .venv/bin/python epai/ai_detection/trainv2/run_cascade_performance.py --alphas 0.05 0.1 0.2 0.3
    .venv/bin/python epai/ai_detection/trainv2/run_cascade_performance.py \
        --replot results/trainv2/cascade_performance/lodo_performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from tqdm import tqdm

from epai.ai_detection.trainv2._trained import TRAINED_SPECS, fit
from epai.ai_detection.trainv2._zeroshot import orient
from epai.ai_detection.trainv2.core import auroc, eff_fpr_level, fpr_tag, goodness, grouped_split, rows_at_fpr, tpr_at_fpr
from epai.ai_detection.trainv2.run_best_zeroshot_performance import (
    DEFAULT_CONFIG as ZEROSHOT_CONFIG,
)
from epai.ai_detection.trainv2.run_lodo_performance import CSV_FIELDS as CSV_FIELDS_LODO
from epai.ai_detection.trainv2.run_lodo_performance import (
    _load,
    _load_ood,
    _print_table as _print_table_lodo,
    _std,
)
from epai.ai_detection.trainv2.run_lodo_performance import compute_trained as compute_trained_lodo
from epai.ai_detection.trainv2.run_lodo_performance import compute_zeroshot as compute_zeroshot_lodo
from epai.ai_detection.trainv2.run_zeroshot_performance import DEFAULT_FPRS
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "cascade_performance"

DEFAULT_SPLITS = ["gb_orig", "mg_gb", "mg_sci"]
DEFAULT_OOD = ["mg_aisci"]
DEFAULT_ALPHAS = [0.05, 0.10, 0.20]

#: the cheap detector every cascade arm escalates FROM. Any ``_trained.py``
#: spec key works; the plain linear probe is the natural "fast filter".
BASE_SPEC_KEY = "roberta_logistic"

#: the specialist every cascade arm escalates TO -- the strongest zero-shot
#: signal on this corpus (see improving_sota_ideas.md's diagnosis table).
JUDGE_COL = "llm_judge_score_gpt56luna_v3_wlp"
JUDGE_MACHINE_IS_HIGH = False  # FAMILIES["judge"]["machine_is_high"] in run_zeroshot_performance.py


def _threshold_at_fpr(y: np.ndarray, scores: np.ndarray, level: float) -> float:
    """The score threshold at the largest achievable FPR <= ``level`` -- the
    threshold twin of ``core.tpr_at_fpr``, which returns the TPR there instead."""
    y = np.asarray(y)
    fpr, _, thr = roc_curve(y, np.asarray(scores, dtype=float))
    idx = int(np.flatnonzero(fpr <= level)[-1])
    return float(thr[idx])


def _escalate_mask(s_calib: np.ndarray, threshold: float, s_eval: np.ndarray,
                   alpha: float) -> np.ndarray:
    """True for the eval rows whose margin to ``threshold`` falls in the
    calibration set's bottom ``alpha`` quantile of margins -- the
    least-confident, most likely near-threshold rows."""
    m_calib = np.abs(np.asarray(s_calib) - threshold)
    cutoff = float(np.quantile(m_calib, alpha)) if m_calib.size else 0.0
    m_eval = np.abs(np.asarray(s_eval) - threshold)
    return m_eval <= cutoff


def _label(alpha: float) -> str:
    return f"cascade@{alpha * 100:g}pct"


def _ecdf_percentile(calib_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Each of ``values``' percentile rank within ``calib_values``' empirical
    distribution, in [0, 1].

    The trained probe's raw score (an unbounded logit margin) and the judge's
    raw score live on different, unrelated numeric scales. Splicing one in for
    the other at escalated rows -- as a first cut of this script did -- does
    NOT approximate "use the better signal on hard rows": AUROC over the
    spliced array reads the *joint* ranking, and a judge value dropped into
    the middle of the trained-score range (or vice versa) lands there by
    scale coincidence, not by relative confidence, so the merged ranking is
    partly scrambled rather than improved -- and more escalation means more of
    it corrupted this way. Transforming BOTH scores onto their own
    calibration-relative percentile before blending fixes that: an escalated
    row's judge-percentile is now directly comparable in magnitude to the
    un-escalated trained-percentiles around it.
    """
    sorted_calib = np.sort(np.asarray(calib_values, dtype=float))
    n = len(sorted_calib)
    ranks = np.searchsorted(sorted_calib, np.asarray(values, dtype=float), side="right")
    return ranks / n if n else np.zeros_like(ranks, dtype=float)


def _score(clf, spec: dict, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """``clf.decision_score``, aux-aware like ``_trained.eval_on_fprs`` -- so
    ``BASE_SPEC_KEY`` can be swapped for an aux hybrid (``roberta_fusion_cuevec``
    / ``roberta_gated_cuevec``) without silently dropping the cue vector."""
    return clf.decision_score(X, Z) if spec["aux"] else clf.decision_score(X)


def compute_cascade(splits, ood_splits, seeds, fprs, human_label, min_domain_rows,
                    calib_frac, alphas, combine) -> list[dict]:
    spec = TRAINED_SPECS[BASE_SPEC_KEY]

    ood_sets: dict[str, tuple] = {}
    for name, path in ood_splits:
        oX, oy, oZ, n_h, oraw = _load_ood(path, human_label, score_cols=(JUDGE_COL,))
        ood_sets[name] = (oX, oy, oZ, orient(oraw[JUDGE_COL], JUDGE_MACHINE_IS_HIGH), n_h)
        print(f"  OOD eval corpus {name!r}: {len(oy)} rows, {n_h} human (never trained on)")

    loaded = []
    for split_key, path in splits:
        X, y, Z, dom, ids, raw = _load(path, human_label, score_cols=(JUDGE_COL,))
        judge = orient(raw[JUDGE_COL], JUDGE_MACHINE_IS_HIGH)
        counts = {d: int((dom == d).sum()) for d in np.unique(dom)}
        domains = sorted(d for d, c in counts.items()
                         if c >= min_domain_rows and np.unique(y[dom == d]).size == 2)
        if len(domains) < 2:
            print(f"  {split_key}: only {len(domains)} usable domain(s) "
                  f"({counts}) — skipped")
            continue
        print(f"  {split_key}: {len(domains)} domains {domains}")
        loaded.append((split_key, X, y, Z, dom, ids, judge, domains))
    if not loaded:
        return []

    # alpha -> split -> list of fold dicts, each {"by_fpr": {f: {...}}, ...}
    all_folds: dict[float, dict[str, list[dict]]] = {a: {} for a in alphas}
    # alpha -> ood name -> fpr -> list of (auroc, tpr, tpr_eff) reads
    ood_reads: dict[float, dict[str, dict[float, list]]] = {
        a: {n: {f: [] for f in fprs} for n in ood_sets} for a in alphas}

    for split_key, X, y, Z, dom, ids, judge, domains in tqdm(
            loaded, desc="cascade/LODO", unit="split"):
        fold_by_alpha: dict[float, list[dict]] = {a: [] for a in alphas}
        for d in domains:
            tr_idx = np.flatnonzero(dom != d)
            te_idx = np.flatnonzero(dom == d)
            assert set(ids[tr_idx]).isdisjoint(ids[te_idx]), "a group straddled the fold"
            n_te_h = int((y[te_idx] == 0).sum())

            acc = {a: {f: {"a": [], "t": [], "te": []} for f in fprs} for a in alphas}
            for seed in seeds:
                fit_rel, calib_rel = grouped_split(y[tr_idx], ids[tr_idx], calib_frac, seed)
                fit_idx, calib_idx = tr_idx[fit_rel], tr_idx[calib_rel]
                clf = fit(spec, X, y, Z, ids, fit_idx, seed)
                s_calib = _score(clf, spec, X[calib_idx], Z[calib_idx])
                s_eval = _score(clf, spec, X[te_idx], Z[te_idx])
                s_ood = {name: _score(clf, spec, oX, oZ)
                        for name, (oX, _, oZ, _, _) in ood_sets.items()}

                # Common [0, 1] scale for blending: each score's percentile
                # within ITS OWN calibration-set distribution. The escalation
                # DECISION (thr / mask below) stays on the raw trained score --
                # a monotonic transform wouldn't change which rows it picks --
                # only the VALUE plugged in for an escalated row is rescaled.
                judge_calib = judge[calib_idx]
                r_eval = _ecdf_percentile(s_calib, s_eval)
                rj_eval = _ecdf_percentile(judge_calib, judge[te_idx])
                r_ood = {name: _ecdf_percentile(s_calib, s_ood[name]) for name in ood_sets}
                rj_ood = {name: _ecdf_percentile(judge_calib, ojudge)
                         for name, (_, _, _, ojudge, _) in ood_sets.items()}

                for f in fprs:
                    thr = _threshold_at_fpr(y[calib_idx], s_calib, f)
                    for a in alphas:
                        mask = _escalate_mask(s_calib, thr, s_eval, a)
                        blended = np.where(mask, rj_eval, r_eval)
                        acc[a][f]["a"].append(auroc(y[te_idx], blended))
                        acc[a][f]["t"].append(tpr_at_fpr(y[te_idx], blended, f))
                        acc[a][f]["te"].append(
                            tpr_at_fpr(y[te_idx], blended, eff_fpr_level(n_te_h, f)))
                        for name, (_, oy, _, _, on_h) in ood_sets.items():
                            mask_o = _escalate_mask(s_calib, thr, s_ood[name], a)
                            blended_o = np.where(mask_o, rj_ood[name], r_ood[name])
                            ood_reads[a][name][f].append((
                                auroc(oy, blended_o),
                                tpr_at_fpr(oy, blended_o, f),
                                tpr_at_fpr(oy, blended_o, eff_fpr_level(on_h, f))))
            for a in alphas:
                fold_by_alpha[a].append({
                    "held_out_domain": d, "n_test": int(te_idx.size), "n_test_human": n_te_h,
                    "by_fpr": {f: {"auroc": float(np.mean(acc[a][f]["a"])),
                                  "tpr_at_fpr": float(np.mean(acc[a][f]["t"])),
                                  "tpr_eff": float(np.mean(acc[a][f]["te"])),
                                  "eff_fpr": eff_fpr_level(n_te_h, f)} for f in fprs},
                })
        for a in alphas:
            all_folds[a][split_key] = fold_by_alpha[a]

    rows: list[dict] = []
    for a in alphas:
        detector = f"trained/{_label(a)}"
        for fpr in fprs:
            split_rows: list[dict] = []
            for split_key, folds in all_folds[a].items():
                gf = [goodness(fd["by_fpr"][fpr]["auroc"], fd["by_fpr"][fpr]["tpr_eff"], combine)
                      for fd in folds]
                for fd, gfi in zip(folds, gf):
                    bf = fd["by_fpr"][fpr]
                    rows.append({
                        "level": "fold", "kind": "trained", "detector": detector,
                        "split": split_key, "held_out_domain": fd["held_out_domain"],
                        "fpr": fpr, "n_test": fd["n_test"], "n_folds": "",
                        "n_seeds": len(seeds),
                        "auroc": bf["auroc"], "auroc_std": "",
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
                    "auroc": float(np.mean([fd["by_fpr"][fpr]["auroc"] for fd in folds])),
                    "auroc_std": _std([fd["by_fpr"][fpr]["auroc"] for fd in folds]),
                    "tpr_at_fpr": float(np.mean([fd["by_fpr"][fpr]["tpr_at_fpr"] for fd in use])),
                    "tpr_at_fpr_std": _std([fd["by_fpr"][fpr]["tpr_at_fpr"] for fd in use]),
                    "tpr_eff": float(np.mean([fd["by_fpr"][fpr]["tpr_eff"] for fd in folds])),
                    "tpr_eff_std": _std([fd["by_fpr"][fpr]["tpr_eff"] for fd in folds]),
                    "eff_fpr": float(np.mean([fd["by_fpr"][fpr]["eff_fpr"] for fd in folds])),
                    "goodness": float(np.mean(gf)), "goodness_std": _std(gf),
                }
                rows.append(sr)
                split_rows.append(sr)

            for name, (_, _, _, _, n_h) in ood_sets.items():
                reads = ood_reads[a][name][fpr]
                oa, ot, ote = [r[0] for r in reads], [r[1] for r in reads], [r[2] for r in reads]
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
                "n_folds": sum(len(all_folds[a][s]) for s in all_folds[a]),
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
            title=f"Leave-one-domain-out — {BASE_SPEC_KEY} cascaded to the "
                  f"judge at {[f'{a * 100:g}%' for a in meta['alphas']]} escalation, "
                  f"vs. existing trained + zero-shot, plus {ood} as eval-only OOD"
                  f"\nAUROC vs. oracle TPR@{fpr * 100:g}%FPR",
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--splits", nargs="+", default=None, metavar="KEY",
                    help=f"training splits (default: {DEFAULT_SPLITS})")
    ap.add_argument("--ood-splits", nargs="+", default=None, metavar="KEY",
                    help=f"eval-only OOD corpora (default: {DEFAULT_OOD})")
    ap.add_argument("--zeroshot-config", type=Path, default=ZEROSHOT_CONFIG)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS, metavar="LEVEL",
                    help=f"oracle FPR level(s); one figure set per level "
                         f"(default: {DEFAULT_FPRS})")
    ap.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS,
                    metavar="FRACTION",
                    help=f"escalation budgets (default: {DEFAULT_ALPHAS})")
    ap.add_argument("--calib-frac", type=float, default=0.2,
                    help="fraction of each fold's training union held out for "
                         "conformal calibration (default: 0.2)")
    ap.add_argument("--combine", choices=("hmean", "gmean", "amean"), default="hmean")
    ap.add_argument("--min-domain-rows", type=int, default=100)
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--no-trained-baselines", action="store_true",
                    help="skip the existing trained/{logistic,fusion,gated} rows")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="PERFORMANCE_CSV")
    args = ap.parse_args()

    train_splits = resolve_splits(args.splits or DEFAULT_SPLITS, PARQUET_DIR)
    ood_splits = resolve_splits(args.ood_splits or DEFAULT_OOD, PARQUET_DIR)
    meta = {"fprs": args.fpr, "n_seeds": len(args.seeds), "combine": args.combine,
            "alphas": args.alphas, "ood": [s for s, _ in ood_splits],
            "splits": [s for s, _ in train_splits] + [s for s, _ in ood_splits]}

    if args.replot is not None:
        plots_lodo(args.replot, meta, args.replot.parent)
        print(f"replotted from {args.replot}")
        return

    missing = [str(p) for _, p in [*train_splits, *ood_splits] if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")

    print(f"cascade base: trained/{TRAINED_SPECS[BASE_SPEC_KEY]['label']}  ->  "
          f"judge column {JUDGE_COL!r}")
    print(f"escalation budgets: {args.alphas}   calibration slice: "
          f"{args.calib_frac:.0%} of each fold's training union")
    print(f"train on: {[s for s, _ in train_splits]}   OOD: {[s for s, _ in ood_splits]}"
          f"   seeds: {args.seeds}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = compute_cascade(train_splits, ood_splits, args.seeds, args.fpr,
                           args.human_label, args.min_domain_rows,
                           args.calib_frac, args.alphas, args.combine)
    if not args.no_trained_baselines:
        rows += compute_trained_lodo(train_splits, ood_splits, args.seeds, args.fpr,
                                     args.human_label, args.min_domain_rows, args.combine)
    if not args.no_zeroshot:
        rows += compute_zeroshot_lodo(train_splits, ood_splits, args.zeroshot_config,
                                      args.seeds, args.fpr, args.human_label,
                                      args.min_domain_rows, args.combine)

    csv_path = args.out_dir / "lodo_performance.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS_LODO).to_csv(csv_path, index=False)
    _print_table_lodo(rows, args.fpr)
    print(f"\nwrote {csv_path}")

    if not args.no_plots:
        plots_lodo(csv_path, meta, args.out_dir)


if __name__ == "__main__":
    main()

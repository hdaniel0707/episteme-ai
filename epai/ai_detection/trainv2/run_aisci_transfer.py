r"""Transfer to Manugram-AI-Scientist by training corpus (a sibling comparison
to ``run_sentinel_comparison.py`` -- see that script's module docstring for
the fixed-eval-set design this one borrows).

Where ``run_sentinel_comparison.py`` asks "how does a detector degrade as its
OWN training/eval corpus drifts" (In-Domain -> Near-OOD -> Far-OOD within a
MG-GB/MG-Sci pair), this one asks a narrower, simpler question: "which
training corpus transfers best to Manugram-AI-Scientist" -- the paper's
hardest, most out-of-distribution eval target (Section 3, Figure~
\ref{fig:binox-difficulty}) -- holding the eval set fixed throughout.

Design
------
THREE scenarios, one per training corpus: MG-Sci, MG-GB, and the legacy
GB-orig (``gpt-3.5-turbo``-only, Section~\ref{sec:method_benchmark}). For
each, a trained detector fits on the FULL corpus (no held-out split needed --
unlike ``run_sentinel_comparison.py``, train and eval are always different
corpora here, so nothing needs to be reserved for a fair held-out read) and
is scored on the full MG-AI-Sci corpus, N model-fit seeds for error bars. A
zero-shot pick has no training corpus and MG-AI-Sci is a single fixed target,
so it is scored exactly ONCE and that one number is reused across all three
scenario columns -- not three separate reads that happen to agree, the same
principle as ``run_sentinel_comparison.py``'s fixed zero-shot reads.

Reuses ``run_sentinel_comparison``'s ``SPECS`` (the paper-matched RoBERTa-
logistic / RoBERTa-Sentinel hyperparameters), ``_load_features``/
``_load_ood`` loading wrappers, and ``resolve_zs_specs`` unchanged, rather
than redefining them -- this script owns only what's actually different
(no split, three training corpora instead of two, one eval target instead of
a primary/other pair).

compute -> CSV -> printed table (see ``plot_aisci_transfer.py`` for the
figure and ``build_aisci_transfer_table.py`` for the matching LaTeX table):

    results/trainv2/aisci_transfer_<timestamp>/
        performance.csv   kind, detector, train_split, eval_split, fpr,
                          n_train, n_test, n_test_human, n_seeds,
                          auroc(_std), tpr_at_fpr(_std), tpr_eff(_std),
                          eff_fpr, goodness(_std)

Run:
    uv run python epai/ai_detection/trainv2/run_aisci_transfer.py
    uv run python epai/ai_detection/trainv2/run_aisci_transfer.py --seeds 0 1 2
    uv run python epai/ai_detection/trainv2/run_aisci_transfer.py --no-zeroshot
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from epai.ai_detection.trainv2._trained import eval_on_fprs, fit
from epai.ai_detection.trainv2._zeroshot import eval_column_fprs, orient
from epai.ai_detection.trainv2.core import eff_fpr_level, goodness
from epai.ai_detection.trainv2.run_sentinel_comparison import (
    SPECS,
    ZS_COLUMNS,
    _load_features,
    _load_ood,
    _std,
    resolve_zs_specs,
)
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "aisci_transfer"

#: the three training-corpus scenarios, in the order every figure/table uses.
TRAIN_SPLITS = ["mg_sci", "mg_gb", "gb_orig"]
EVAL_SPLIT = "mg_aisci"

DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_FPRS = [0.01]

CSV_FIELDS = [
    "kind", "detector", "train_split", "eval_split", "fpr",
    "n_train", "n_test", "n_test_human", "n_seeds",
    "auroc", "auroc_std", "tpr_at_fpr", "tpr_at_fpr_std",
    "tpr_eff", "tpr_eff_std", "eff_fpr", "goodness", "goodness_std",
]


def _row(detector: str, train_split: str, cell: dict, kind: str = "trained",
        eval_split: str = EVAL_SPLIT) -> dict:
    """``eval_split`` defaults to this module's own fixed target (MG-AI-Sci)
    but is a parameter -- not hardcoded -- so sibling single-eval-target
    experiments (e.g. ``run_math_holdout_transfer.py``) can import this same
    row-formatting helper for their own fixed eval set instead of
    duplicating it."""
    return {
        "kind": kind, "detector": detector, "train_split": train_split,
        "eval_split": eval_split, "fpr": cell["fpr"],
        "n_train": cell.get("n_train", ""), "n_test": cell.get("n_test", ""),
        "n_test_human": cell.get("n_test_human", ""), "n_seeds": cell["n_seeds"],
        "auroc": cell["auroc"], "auroc_std": cell["auroc_std"],
        "tpr_at_fpr": cell["tpr_at_fpr"], "tpr_at_fpr_std": cell["tpr_at_fpr_std"],
        "tpr_eff": cell["tpr_eff"], "tpr_eff_std": cell["tpr_eff_std"],
        "eff_fpr": cell["eff_fpr"],
        "goodness": cell["goodness"], "goodness_std": cell["goodness_std"],
    }


def compute_trained(train_splits, eval_path, seeds, fprs, group_col, human_label,
                    combine) -> list[dict]:
    eX, ey, en_h, _, _ = _load_ood(eval_path, human_label)
    print(f"  eval corpus {EVAL_SPLIT!r}: {len(ey)} rows, {en_h} human (never trained on)")

    rows: list[dict] = []
    for train_key, path in tqdm(train_splits, desc="aisci-transfer", unit="corpus"):
        X, y, groups, _, _ = _load_features(path, group_col, human_label)
        idx_all = np.arange(len(y))
        for spec_key, spec in SPECS.items():
            detector = f"trained/{spec['label']}"
            a: list[float] = []
            t = {f: [] for f in fprs}
            te_ = {f: [] for f in fprs}
            for seed in seeds:
                clf = fit(spec, X, y, None, groups, idx_all, seed)
                au, per = eval_on_fprs(clf, spec, eX, ey, None, fprs)
                a.append(au)
                for f in fprs:
                    t[f].append(per[f][0])
                    te_[f].append(per[f][1])
            for f in fprs:
                g = [goodness(ai, ei, combine, alpha=f) for ai, ei in zip(a, te_[f])]
                cell = {
                    "fpr": f, "n_train": len(y), "n_test": len(ey), "n_test_human": en_h,
                    "n_seeds": len(seeds),
                    "auroc": float(np.mean(a)), "auroc_std": _std(a),
                    "tpr_at_fpr": float(np.mean(t[f])), "tpr_at_fpr_std": _std(t[f]),
                    "tpr_eff": float(np.mean(te_[f])), "tpr_eff_std": _std(te_[f]),
                    "eff_fpr": eff_fpr_level(en_h, f),
                    "goodness": float(np.mean(g)), "goodness_std": _std(g),
                }
                rows.append(_row(detector, train_key, cell))
    return rows


def compute_zeroshot(eval_path, zs_specs, fprs, human_label, combine) -> list[dict]:
    """Each zero-shot pick scored exactly ONCE on the full MG-AI-Sci corpus --
    the one and only eval target in this experiment -- then the same reading
    is emitted under all three training-corpus columns (a zero-shot pick has
    no training corpus; see the module docstring)."""
    if not zs_specs:
        return []
    zs_cols = tuple(c for c, _, _ in zs_specs)
    eX, ey, en_h, _, e_raw = _load_ood(eval_path, human_label, zs_cols)

    rows: list[dict] = []
    for col, det, mih in zs_specs:
        au, per = eval_column_fprs(orient(e_raw[col], mih), ey, fprs)
        for f in fprs:
            t_, te_ = per[f]
            gd = (goodness(au, te_, combine, alpha=f)
                 if np.isfinite(au) and np.isfinite(te_) else float("nan"))
            cell = {
                "fpr": f, "n_train": "", "n_test": len(ey), "n_test_human": en_h,
                "n_seeds": 1,
                "auroc": au, "auroc_std": float("nan"),
                "tpr_at_fpr": t_, "tpr_at_fpr_std": float("nan"),
                "tpr_eff": te_, "tpr_eff_std": float("nan"),
                "eff_fpr": eff_fpr_level(en_h, f),
                "goodness": gd, "goodness_std": float("nan"),
            }
            for train_key in TRAIN_SPLITS:
                rows.append(_row(det, train_key, cell, kind="zeroshot"))
    return rows


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'kind':<9}{'detector':<26}{'train':<10}{'eval':<10}{'FPR':>6}"
          f"{'auroc':>15}{'TPR':>15}{'AT-HM':>10}")
    for r in rows:
        def fmt(v, sd):
            if v == "" or (isinstance(v, float) and not np.isfinite(v)):
                return "--"
            bare = (sd in ("", None) or (isinstance(sd, float) and not np.isfinite(sd)))
            return f"{v:.3f}" if bare else f"{v:.3f}±{sd:.3f}"
        print(f"{r['kind']:<9}{r['detector']:<26}{r['train_split']:<10}{r['eval_split']:<10}"
              f"{float(r['fpr']) * 100:>5g}%"
              f"{fmt(r['auroc'], r['auroc_std']):>15}"
              f"{fmt(r['tpr_at_fpr'], r['tpr_at_fpr_std']):>15}"
              f"{fmt(r['goodness'], r['goodness_std']):>10}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs=3, default=TRAIN_SPLITS, metavar="KEY",
                    help=f"exactly 3 training corpora (default: {TRAIN_SPLITS})")
    ap.add_argument("--eval-split", default=EVAL_SPLIT, metavar="KEY",
                    help=f"the one fixed eval-only corpus (default: {EVAL_SPLIT!r})")
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                    help="model-fit seeds (no data split here, so these are the only "
                         "source of variance for the trained detectors)")
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS, metavar="LEVEL",
                    help=f"oracle FPR level(s) (default: {DEFAULT_FPRS})")
    ap.add_argument("--combine", choices=("hmean", "gmean", "amean"), default="hmean",
                    help="AT-HM = this mean of Gini(AUROC) and oracle TPR (default: hmean)")
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--zs-columns", nargs="*", default=ZS_COLUMNS, metavar="COLUMN",
                    help=f"zero-shot score column(s), each scored once on the full eval "
                         f"corpus and reused across all three scenarios (default: "
                         f"{ZS_COLUMNS})")
    ap.add_argument("--no-zeroshot", action="store_true",
                    help="skip the zero-shot picks entirely (kind='trained' rows only)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"default: a fresh {DEFAULT_OUT_DIR.name}_<YYYYMMDD_HHMM> "
                         f"directory under {DEFAULT_OUT_DIR.parent}/")
    args = ap.parse_args()
    out_dir = args.out_dir or (
        DEFAULT_OUT_DIR.parent / f"{DEFAULT_OUT_DIR.name}_{datetime.now():%Y%m%d_%H%M}")

    train_splits = resolve_splits(args.splits, PARQUET_DIR)
    [(_, eval_path)] = resolve_splits([args.eval_split], PARQUET_DIR)
    missing = [str(p) for _, p in [*train_splits, (args.eval_split, eval_path)]
              if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")

    zs_columns = [] if args.no_zeroshot else args.zs_columns
    zs_specs = resolve_zs_specs([eval_path], zs_columns)

    print(f"detectors  : {[s['label'] for s in SPECS.values()]}")
    if zs_specs:
        print(f"zero-shot  : {[det for _, det, _ in zs_specs]}")
    elif zs_columns:
        print(f"zero-shot  : none of {zs_columns} found on {args.eval_split!r} — skipped")
    print(f"train on   : {[s for s, _ in train_splits]}   eval on: {args.eval_split!r}")
    print(f"seeds      : {args.seeds}   fpr: {args.fpr}\n")

    rows = compute_trained(train_splits, eval_path, args.seeds, args.fpr,
                           args.group_col, args.human_label, args.combine)
    rows += compute_zeroshot(eval_path, zs_specs, args.fpr, args.human_label, args.combine)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "performance.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    _print_table(rows)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()

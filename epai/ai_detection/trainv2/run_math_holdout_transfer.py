r"""Transfer to a held-out MG-Sci domain by training corpus -- a second
sibling of ``run_sentinel_comparison.py``, alongside ``run_aisci_
transfer.py`` (see that script's module docstring for the shared design
philosophy: one FIXED eval set, reused unchanged across every scenario).

Where ``run_aisci_transfer.py`` asks "which training corpus transfers best
to Manugram-AI-Scientist", this one asks the same question against a nearer,
narrower target: one of MG-Sci's three domains (``--holdout-domain``,
default ``mathematics``), held out of training entirely and never touched
by any of the three scenarios below.

Design
------
The eval set is fixed once: the chosen MG-Sci domain's rows, set aside up
front. THREE training scenarios, each scored against that same fixed set:

  1. ``mg_sci_ex_<code>``  fit on the REST of MG-Sci (the other two domains
                            only, the held-out one entirely excluded) --
                            the "near" scenario: same corpus, adjacent
                            domains, held-out target genre.
  2. ``mg_gb``              fit on the full MG-GB corpus -- a different genre
                            entirely (news/essay/fiction, not scientific
                            abstracts).
  3. ``gb_orig``            fit on the full legacy GB-orig corpus -- different
                            genre AND a much older, single-generator
                            (``gpt-3.5-turbo``) source.

No held-out split is needed for scenarios 2/3: their training corpus is
already disjoint from the held-out domain's rows, so nothing needs to be
reserved for a fair read, same as ``run_aisci_transfer.py``. Scenario 1 is
the one exception -- its training rows come from the SAME corpus as the
eval rows, so the held-out domain must be excluded from training by domain,
not merely left unsampled; the group-id disjointness between the excluded
domain's train rows and the held-out eval rows is asserted at load time,
mirroring ``run_sentinel_comparison.py``'s Near-OOD fold construction.

A zero-shot pick has no training corpus and the eval set is fixed, so it is
scored exactly ONCE on the held-out domain's rows and that one number is
reused across all three scenario columns.

Reuses ``run_sentinel_comparison``'s ``SPECS`` / ``DOMAIN_COL`` / ``GROUP_COL``
and ``run_aisci_transfer``'s ``_row`` / ``_std`` / ``CSV_FIELDS`` /
``_print_table`` unchanged -- this script owns only what's actually
different (the domain-exclusion load for scenario 1).

``plot_math_holdout_transfer.py`` (figure) and ``build_math_holdout_
transfer_table.py`` (LaTeX table) both read the held-out domain's short code
(``mg_sci_ex_<code>``) straight off the CSV's own ``train_split`` column --
no ``--holdout-domain`` flag to keep in sync on their end, one run's CSV is
fully self-describing.

compute -> CSV -> printed table:

    results/trainv2/math_holdout_transfer_<code>_<timestamp>/
        performance.csv   kind, detector, train_split, eval_split, fpr,
                          n_train, n_test, n_test_human, n_seeds,
                          auroc(_std), tpr_at_fpr(_std), tpr_eff(_std),
                          eff_fpr, goodness(_std)

Run:
    uv run python epai/ai_detection/trainv2/run_math_holdout_transfer.py
    uv run python epai/ai_detection/trainv2/run_math_holdout_transfer.py \
        --holdout-domain computer_science
    uv run python epai/ai_detection/trainv2/run_math_holdout_transfer.py \
        --holdout-domain sociology
    uv run python epai/ai_detection/trainv2/run_math_holdout_transfer.py --no-zeroshot
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
from epai.ai_detection.trainv2.run_aisci_transfer import CSV_FIELDS, _print_table, _row, _std
from epai.ai_detection.trainv2.run_sentinel_comparison import (
    DOMAIN_COL,
    GROUP_COL,
    SPECS,
    ZS_COLUMNS,
    _load_features,
    resolve_zs_specs,
)
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "math_holdout_transfer"

#: the corpus the held-out domain is carved out of.
HOLDOUT_SPLIT = "mg_sci"
DEFAULT_HOLDOUT_DOMAIN = "mathematics"
#: MG-Sci's three domains (Section~\ref{sec:method_benchmark}) -> the short
#: code used in the ``mg_sci_ex_<code>`` scenario key and ``mg_sci_<code>``
#: eval-split label -- matches the paper's own cs/math/soc shorthand.
DOMAIN_CODE = {"mathematics": "math", "computer_science": "cs", "sociology": "soc"}

DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_FPRS = [0.01]


def _fit_eval_rows(train_key: str, eval_label: str, X, y, groups, train_idx, eX, ey, seeds,
                   fprs, combine) -> list[dict]:
    n_ev, n_ev_h = len(ey), int((ey == 0).sum())
    rows: list[dict] = []
    for spec_key, spec in SPECS.items():
        detector = f"trained/{spec['label']}"
        a: list[float] = []
        t = {f: [] for f in fprs}
        te_ = {f: [] for f in fprs}
        for seed in seeds:
            clf = fit(spec, X, y, None, groups, train_idx, seed)
            au, per = eval_on_fprs(clf, spec, eX, ey, None, fprs)
            a.append(au)
            for f in fprs:
                t[f].append(per[f][0])
                te_[f].append(per[f][1])
        for f in fprs:
            g = [goodness(ai, ei, combine, alpha=f) for ai, ei in zip(a, te_[f])]
            cell = {
                "fpr": f, "n_train": len(train_idx), "n_test": n_ev, "n_test_human": n_ev_h,
                "n_seeds": len(seeds),
                "auroc": float(np.mean(a)), "auroc_std": _std(a),
                "tpr_at_fpr": float(np.mean(t[f])), "tpr_at_fpr_std": _std(t[f]),
                "tpr_eff": float(np.mean(te_[f])), "tpr_eff_std": _std(te_[f]),
                "eff_fpr": eff_fpr_level(n_ev_h, f),
                "goodness": float(np.mean(g)), "goodness_std": _std(g),
            }
            rows.append(_row(detector, train_key, cell, eval_split=eval_label))
    return rows


def compute_trained(holdout_domain: str, ex_key: str, eval_label: str, other_splits,
                    seeds, fprs, group_col, human_label, combine,
                    zs_cols: tuple[str, ...]) -> tuple[list[dict], dict, object]:
    """Returns ``(rows, eval_raw, ey)`` -- ``eval_raw``/``ey`` carry the
    held-out domain's zero-shot score columns and labels, for
    ``compute_zeroshot``."""
    [(_, holdout_path)] = resolve_splits([HOLDOUT_SPLIT], PARQUET_DIR)
    Xs, ys, groups_s, dom_s, _, raw_s = _load_features(
        holdout_path, group_col, human_label, domain_col=DOMAIN_COL, score_cols=zs_cols)
    train_idx = np.where(dom_s != holdout_domain)[0]
    eval_idx = np.where(dom_s == holdout_domain)[0]
    assert set(groups_s[train_idx]).isdisjoint(groups_s[eval_idx]), \
        f"{holdout_domain!r} rows leaked into the {ex_key!r} training set"
    eX, ey = Xs[eval_idx], ys[eval_idx]
    eval_raw = {col: vals[eval_idx] for col, vals in raw_s.items()}
    print(f"  {HOLDOUT_SPLIT!r}/{holdout_domain!r} held out: {len(eval_idx)} eval rows "
          f"({int((ey == 0).sum())} human), {len(train_idx)} rows left for "
          f"{ex_key!r} training")

    rows: list[dict] = []
    rows += _fit_eval_rows(ex_key, eval_label, Xs, ys, groups_s, train_idx, eX, ey,
                           seeds, fprs, combine)

    for train_key, path in tqdm(other_splits, desc="domain-holdout", unit="corpus"):
        X, y, groups, _, _ = _load_features(path, group_col, human_label)
        idx_all = np.arange(len(y))
        rows += _fit_eval_rows(train_key, eval_label, X, y, groups, idx_all, eX, ey,
                               seeds, fprs, combine)

    return rows, eval_raw, ey


def compute_zeroshot(train_splits, eval_label: str, eval_raw: dict, ey, zs_specs, fprs,
                     combine) -> list[dict]:
    """Each zero-shot pick scored exactly ONCE on the held-out domain's
    rows, then the same reading is emitted under all three training-corpus
    columns -- a zero-shot pick has no training corpus, see the module
    docstring."""
    if not zs_specs:
        return []
    en_h = int((ey == 0).sum())
    rows: list[dict] = []
    for col, det, mih in zs_specs:
        au, per = eval_column_fprs(orient(eval_raw[col], mih), ey, fprs)
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
            for train_key in train_splits:
                rows.append(_row(det, train_key, cell, kind="zeroshot", eval_split=eval_label))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout-domain", choices=sorted(DOMAIN_CODE),
                    default=DEFAULT_HOLDOUT_DOMAIN,
                    help=f"which MG-Sci domain to hold out for eval (default: "
                         f"{DEFAULT_HOLDOUT_DOMAIN!r})")
    ap.add_argument("--other-splits", nargs=2, default=["mg_gb", "gb_orig"], metavar="KEY",
                    help="the two full-corpus training scenarios, alongside "
                         "'mg_sci_ex_<code>' (default: mg_gb gb_orig)")
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                    help="model-fit seeds")
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS, metavar="LEVEL",
                    help=f"oracle FPR level(s) (default: {DEFAULT_FPRS})")
    ap.add_argument("--combine", choices=("hmean", "gmean", "amean"), default="hmean",
                    help="AT-HM = this mean of Gini(AUROC) and oracle TPR (default: hmean)")
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--zs-columns", nargs="*", default=ZS_COLUMNS, metavar="COLUMN",
                    help=f"zero-shot score column(s), each scored once on the held-out "
                         f"domain's rows and reused across all three scenarios "
                         f"(default: {ZS_COLUMNS})")
    ap.add_argument("--no-zeroshot", action="store_true",
                    help="skip the zero-shot picks entirely (kind='trained' rows only)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"default: a fresh {DEFAULT_OUT_DIR.name}_<code>_<YYYYMMDD_HHMM> "
                         f"directory under {DEFAULT_OUT_DIR.parent}/")
    args = ap.parse_args()
    code = DOMAIN_CODE[args.holdout_domain]
    ex_key = f"mg_sci_ex_{code}"
    eval_label = f"mg_sci_{code}"
    train_splits = [ex_key, *args.other_splits]
    out_dir = args.out_dir or (
        DEFAULT_OUT_DIR.parent
        / f"{DEFAULT_OUT_DIR.name}_{code}_{datetime.now():%Y%m%d_%H%M}")

    other_splits = resolve_splits(args.other_splits, PARQUET_DIR)
    [(_, holdout_path)] = resolve_splits([HOLDOUT_SPLIT], PARQUET_DIR)
    missing = [str(p) for _, p in [*other_splits, (HOLDOUT_SPLIT, holdout_path)]
              if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")

    zs_columns = [] if args.no_zeroshot else args.zs_columns
    zs_specs = resolve_zs_specs([holdout_path], zs_columns)
    zs_cols = tuple(c for c, _, _ in zs_specs)

    print(f"detectors  : {[s['label'] for s in SPECS.values()]}")
    if zs_specs:
        print(f"zero-shot  : {[det for _, det, _ in zs_specs]}")
    elif zs_columns:
        print(f"zero-shot  : none of {zs_columns} found on {HOLDOUT_SPLIT!r} — skipped")
    print(f"scenarios  : {train_splits}   eval on: "
          f"{HOLDOUT_SPLIT!r}/{args.holdout_domain!r}")
    print(f"seeds      : {args.seeds}   fpr: {args.fpr}\n")

    rows, eval_raw, ey = compute_trained(args.holdout_domain, ex_key, eval_label, other_splits,
                                         args.seeds, args.fpr, args.group_col,
                                         args.human_label, args.combine, zs_cols)
    rows += compute_zeroshot(train_splits, eval_label, eval_raw, ey, zs_specs, args.fpr,
                             args.combine)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "performance.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    _print_table(rows)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()

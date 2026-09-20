r"""Transfer to a FIXED 30% eval slice of one MG-Sci domain, by training
corpus -- a refinement of ``run_math_holdout_transfer.py`` (full-domain
exclusion) that instead partitions the held-out domain itself 70/30, adding
a fourth, more "in-domain" training scenario. Third sibling of ``run_
sentinel_comparison.py`` alongside ``run_aisci_transfer.py`` / ``run_math_
holdout_transfer.py`` -- same design philosophy: one FIXED eval set, reused
unchanged across every scenario.

Design
------
Pick one MG-Sci domain to hold out (``--holdout-domain``, default
``mathematics``). Within THAT domain only, draw one grouped 70/30 split
(``--split-seed``, fixed, not redrawn per model-fit seed) -- the 30% eval
half is the ONE fixed eval set for the rest of this experiment; the other
two MG-Sci domains are never split, always used in full. FOUR training
scenarios, each scored against that same fixed 30% slice:

  1. ``mg_sci_in_<code>``  fit on the OTHER two domains (full) PLUS the
                            held-out domain's own 70% train rows -- almost
                            all of MG-Sci, missing only the eval slice
                            itself. The "most in-domain" scenario.
  2. ``mg_sci_ex_<code>``  fit on the OTHER two domains ONLY -- the held-out
                            domain entirely excluded, including its 70%
                            train rows. The "near" scenario from ``run_math_
                            holdout_transfer.py``, unchanged in spirit but
                            now scored against the smaller 30% slice instead
                            of the full held-out domain.
  3. ``mg_gb``              fit on the full MG-GB corpus -- a different genre
                            entirely.
  4. ``gb_orig``            fit on the full legacy GB-orig corpus -- different
                            genre AND a much older, single-generator
                            (``gpt-3.5-turbo``) source.

No held-out split is needed for scenarios 3/4: their training corpus is
already disjoint from the eval rows. Scenarios 1/2 draw from the SAME
corpus as the eval rows, so group-id disjointness between each scenario's
training rows and the fixed eval slice is asserted at load time (the 70/30
split and the other-domain rows are disjoint from it by construction: a
document group never spans two domains, and ``core.grouped_split`` never
puts a group on both sides of its own cut).

A zero-shot pick has no training corpus and the eval set is fixed, so it is
scored exactly ONCE on the fixed 30% slice and that one number is reused
across all four scenario columns.

Run this whole script three times, once per MG-Sci domain as the one being
split -- three SEPARATE runs into three separate output directories (one
``--holdout-domain`` per invocation), never combined in one run: in each
run, only ONE domain is 70/30-split; the other two are always used whole.

Reuses ``run_sentinel_comparison``'s ``SPECS`` / ``DOMAIN_COL`` / ``GROUP_COL``,
``run_aisci_transfer``'s ``_row`` / ``_std`` / ``CSV_FIELDS`` / ``_print_
table``, ``run_math_holdout_transfer``'s ``DOMAIN_CODE`` / ``_fit_eval_rows``
/ ``compute_zeroshot`` (both already domain-agnostic, no change needed), and
``core.grouped_split`` unchanged -- this script owns only the new
in-domain/ex-domain split-and-concatenate load.

compute -> CSV -> printed table (see ``plot_domain_split_transfer.py`` for
the figure and ``build_domain_split_transfer_table.py`` for the matching
LaTeX table):

    results/trainv2/domain_split_transfer_<code>_<timestamp>/
        performance.csv   kind, detector, train_split, eval_split, fpr,
                          n_train, n_test, n_test_human, n_seeds,
                          auroc(_std), tpr_at_fpr(_std), tpr_eff(_std),
                          eff_fpr, goodness(_std)

Run:
    uv run python epai/ai_detection/trainv2/run_domain_split_transfer.py
    uv run python epai/ai_detection/trainv2/run_domain_split_transfer.py \
        --holdout-domain computer_science
    uv run python epai/ai_detection/trainv2/run_domain_split_transfer.py \
        --holdout-domain sociology
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from epai.ai_detection.trainv2.core import grouped_split
from epai.ai_detection.trainv2.run_aisci_transfer import CSV_FIELDS, _print_table
from epai.ai_detection.trainv2.run_math_holdout_transfer import (
    DOMAIN_CODE,
    _fit_eval_rows,
)
from epai.ai_detection.trainv2.run_math_holdout_transfer import (
    compute_zeroshot as _compute_zeroshot,
)
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
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "domain_split_transfer"

HOLDOUT_SPLIT = "mg_sci"
DEFAULT_HOLDOUT_DOMAIN = "mathematics"
DEFAULT_SEEDS = [0, 1, 2]
#: the fixed 70/30 split within the held-out domain -- deliberately separate
#: from the model-fit seeds, same principle as run_sentinel_comparison.py's
#: SPLIT_SEED.
DEFAULT_SPLIT_SEED = 0
DEFAULT_TEST_SIZE = 0.30
DEFAULT_FPRS = [0.01]


def compute_trained(holdout_domain: str, code: str, other_splits, seeds, split_seed: int,
                    test_size: float, fprs, group_col, human_label, combine,
                    zs_cols: tuple[str, ...]) -> tuple[list[dict], dict, object, list[str]]:
    """Returns ``(rows, eval_raw, ey, train_splits)``."""
    [(_, holdout_path)] = resolve_splits([HOLDOUT_SPLIT], PARQUET_DIR)
    Xs, ys, groups_s, dom_s, _, raw_s = _load_features(
        holdout_path, group_col, human_label, domain_col=DOMAIN_COL, score_cols=zs_cols)

    hd_idx = np.where(dom_s == holdout_domain)[0]
    other_idx = np.where(dom_s != holdout_domain)[0]
    y_hd, groups_hd = ys[hd_idx], groups_s[hd_idx]
    tr_local, ev_local = grouped_split(y_hd, groups_hd, test_size, split_seed)
    tr_hd, ev_hd = hd_idx[tr_local], hd_idx[ev_local]

    eX, ey = Xs[ev_hd], ys[ev_hd]
    eval_raw = {col: vals[ev_hd] for col, vals in raw_s.items()}
    print(f"  {HOLDOUT_SPLIT!r}/{holdout_domain!r} split (seed={split_seed}): "
          f"{len(tr_hd)} train / {len(ev_hd)} eval rows ({int((ey == 0).sum())} human), "
          f"{len(other_idx)} rows in the other two domains (used whole)")

    in_key, ex_key = f"mg_sci_in_{code}", f"mg_sci_ex_{code}"
    eval_label = f"mg_sci_{code}"
    other_keys = [k for k, _ in other_splits]

    train_idx_in = np.concatenate([other_idx, tr_hd])
    assert set(groups_s[train_idx_in]).isdisjoint(groups_s[ev_hd]), \
        f"{holdout_domain!r} eval rows leaked into the {in_key!r} training set"
    assert set(groups_s[other_idx]).isdisjoint(groups_s[ev_hd]), \
        f"{holdout_domain!r} eval rows leaked into the {ex_key!r} training set"

    rows: list[dict] = []
    rows += _fit_eval_rows(in_key, eval_label, Xs, ys, groups_s, train_idx_in, eX, ey,
                           seeds, fprs, combine)
    rows += _fit_eval_rows(ex_key, eval_label, Xs, ys, groups_s, other_idx, eX, ey,
                           seeds, fprs, combine)

    for train_key, path in tqdm(other_splits, desc="domain-split", unit="corpus"):
        X, y, groups, _, _ = _load_features(path, group_col, human_label)
        idx_all = np.arange(len(y))
        rows += _fit_eval_rows(train_key, eval_label, X, y, groups, idx_all, eX, ey,
                               seeds, fprs, combine)

    return rows, eval_raw, ey, [in_key, ex_key, *other_keys], eval_label


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout-domain", choices=sorted(DOMAIN_CODE),
                    default=DEFAULT_HOLDOUT_DOMAIN,
                    help=f"which MG-Sci domain to 70/30-split for eval (default: "
                         f"{DEFAULT_HOLDOUT_DOMAIN!r})")
    ap.add_argument("--other-splits", nargs=2, default=["mg_gb", "gb_orig"], metavar="KEY",
                    help="the two full-corpus training scenarios, alongside "
                         "'mg_sci_in_<code>' / 'mg_sci_ex_<code>' (default: mg_gb gb_orig)")
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                    help="model-fit seeds")
    ap.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED,
                    help=f"the fixed 70/30 split within the held-out domain (default: "
                         f"{DEFAULT_SPLIT_SEED})")
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                    help=f"eval fraction of the held-out domain (default: {DEFAULT_TEST_SIZE})")
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS, metavar="LEVEL",
                    help=f"oracle FPR level(s) (default: {DEFAULT_FPRS})")
    ap.add_argument("--combine", choices=("hmean", "gmean", "amean"), default="hmean",
                    help="AT-HM = this mean of Gini(AUROC) and oracle TPR (default: hmean)")
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--zs-columns", nargs="*", default=ZS_COLUMNS, metavar="COLUMN",
                    help=f"zero-shot score column(s), each scored once on the fixed 30%% "
                         f"eval slice and reused across all four scenarios (default: "
                         f"{ZS_COLUMNS})")
    ap.add_argument("--no-zeroshot", action="store_true",
                    help="skip the zero-shot picks entirely (kind='trained' rows only)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"default: a fresh {DEFAULT_OUT_DIR.name}_<code>_<YYYYMMDD_HHMM> "
                         f"directory under {DEFAULT_OUT_DIR.parent}/")
    args = ap.parse_args()
    code = DOMAIN_CODE[args.holdout_domain]
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
    print(f"holdout    : {HOLDOUT_SPLIT!r}/{args.holdout_domain!r}   "
          f"other splits: {args.other_splits}")
    print(f"seeds      : {args.seeds}   split-seed: {args.split_seed}   "
          f"test-size: {args.test_size}   fpr: {args.fpr}\n")

    rows, eval_raw, ey, train_splits, eval_label = compute_trained(
        args.holdout_domain, code, other_splits, args.seeds, args.split_seed, args.test_size,
        args.fpr, args.group_col, args.human_label, args.combine, zs_cols)
    rows += _compute_zeroshot(train_splits, eval_label, eval_raw, ey, zs_specs, args.fpr,
                              args.combine)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "performance.csv"
    pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    _print_table(rows)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()

r"""Run ``run_domain_split_transfer.py``'s full pipeline for all THREE
MG-Sci domains (mathematics, computer_science, sociology) as the held-out
one, then average each (algorithm, scenario-role) cell's AUROC/TPR across
the three domains and write ONE averaged ``performance.csv`` -- consumed by
the existing ``plot_domain_split_transfer.py`` / ``build_domain_split_
transfer_table.py`` exactly like a single-domain run, no changes needed to
either.

Why this is a separate script, not a flag on ``run_domain_split_transfer.
py``: that script is deliberately one-domain-per-run (see its module
docstring's "Design" section -- "in each run, only ONE domain is 70/30-
split"). This one calls its ``compute_trained``/``compute_zeroshot``
functions directly, once per domain, and adds only the averaging step;
the per-domain compute logic itself is unchanged and unduplicated.

Averaging
---------
Each domain's own CSV already has 4 scenario rows per (detector, fpr): the
in-domain and ex-domain scenarios carry that domain's short code in their
``train_split`` key (``mg_sci_in_<code>`` / ``mg_sci_ex_<code>``), while
``mg_gb``/``gb_orig`` are literally the same key across all three domains.
To average across domains, each row is first mapped to its scenario ROLE
("in" / "ex" / "mg_gb" / "gb_orig") rather than its literal key; for a given
(detector, role, fpr), the AUROC and TPR@FPR are averaged across the three
domains' readings, and AT-HM is recomputed from that averaged pair -- the
same "goodness from the already-averaged AUROC/TPR" convention every other
pooled table/figure in this codebase uses (e.g.
``run_sentinel_comparison.py``'s two-corpus-orientation mean). A zero-shot
pick is flat within each domain's own run (see ``run_domain_split_transfer.
py``'s docstring), so averaging its three domain-specific flat readings
still yields one flat value -- the star-like "doesn't move" property survives
averaging, it just now means "doesn't move, and is also stable across which
domain was held out".

The averaged CSV's in-/ex-domain scenario keys are ``mg_sci_in_avg`` /
``mg_sci_ex_avg`` (code ``"avg"``) so ``plot_domain_split_transfer.
scenario_layout`` picks them up exactly like any single-domain code, with a
dedicated, clearer label for that one case (see this script's edit to that
function).

compute (x3 domains) -> per-domain CSVs (for transparency) -> averaged CSV
-> printed table:

    results/trainv2/domain_split_transfer_avg_<timestamp>/
        performance.csv                  the averaged CSV -- feed this to
                                          plot_domain_split_transfer.py /
                                          build_domain_split_transfer_table.py
        per_domain/<code>_performance.csv   each domain's own, unaveraged run

Run:
    uv run python epai/ai_detection/trainv2/run_domain_split_transfer_avg.py
    uv run python epai/ai_detection/trainv2/plot_domain_split_transfer.py \
        results/trainv2/domain_split_transfer_avg_<timestamp>/performance.csv
    uv run python epai/ai_detection/trainv2/build_domain_split_transfer_table.py \
        results/trainv2/domain_split_transfer_avg_<timestamp>/performance.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from epai.ai_detection.trainv2.core import goodness
from epai.ai_detection.trainv2.run_aisci_transfer import CSV_FIELDS, _print_table, _row, _std
from epai.ai_detection.trainv2.run_domain_split_transfer import (
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_FPRS,
    DOMAIN_CODE,
    compute_trained,
)
from epai.ai_detection.trainv2.run_math_holdout_transfer import compute_zeroshot
from epai.ai_detection.trainv2.run_sentinel_comparison import (
    PARQUET_DIR,
    ZS_COLUMNS,
    resolve_zs_specs,
)
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "domain_split_transfer_avg"

HOLDOUT_DOMAINS = ["mathematics", "computer_science", "sociology"]


def _role(train_split: str, code: str) -> str:
    """Map a domain-specific ``train_split`` key to its scenario role, the
    axis every domain's run shares -- ``mg_gb``/``gb_orig`` need no mapping,
    they're already identical across domains."""
    if train_split == f"mg_sci_in_{code}":
        return "in"
    if train_split == f"mg_sci_ex_{code}":
        return "ex"
    return train_split


def average_rows(per_domain_rows: dict[str, list[dict]], combine: str) -> list[dict]:
    buckets: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for holdout_domain, rows in per_domain_rows.items():
        code = DOMAIN_CODE[holdout_domain]
        for r in rows:
            role = _role(r["train_split"], code)
            buckets[(r["detector"], role, r["fpr"])].append(r)

    role_train_split = {"in": "mg_sci_in_avg", "ex": "mg_sci_ex_avg"}
    out: list[dict] = []
    for (detector, role, fpr), items in buckets.items():
        aurocs = [it["auroc"] for it in items]
        tprs = [it["tpr_at_fpr"] for it in items]
        tpr_effs = [it["tpr_eff"] for it in items]
        n_test_h = [it["n_test_human"] for it in items if it["n_test_human"] != ""]
        eff_fprs = [it["eff_fpr"] for it in items if it["eff_fpr"] != ""]
        mean_auroc, mean_tpr = float(np.mean(aurocs)), float(np.mean(tprs))
        cell = {
            "fpr": fpr, "n_train": "", "n_test": "",
            "n_test_human": int(round(float(np.mean(n_test_h)))) if n_test_h else "",
            "n_seeds": items[0]["n_seeds"],
            "auroc": mean_auroc, "auroc_std": _std(aurocs),
            "tpr_at_fpr": mean_tpr, "tpr_at_fpr_std": _std(tprs),
            "tpr_eff": float(np.mean(tpr_effs)), "tpr_eff_std": _std(tpr_effs),
            "eff_fpr": float(np.mean(eff_fprs)) if eff_fprs else "",
            "goodness": goodness(mean_auroc, mean_tpr, combine, alpha=fpr),
            "goodness_std": "",
        }
        train_split = role_train_split.get(role, role)
        out.append(_row(detector, train_split, cell, kind=items[0]["kind"],
                        eval_split="mg_sci_avg"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--other-splits", nargs=2, default=["mg_gb", "gb_orig"], metavar="KEY")
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    ap.add_argument("--fpr", nargs="+", type=float, default=DEFAULT_FPRS, metavar="LEVEL")
    ap.add_argument("--combine", choices=("hmean", "gmean", "amean"), default="hmean")
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--zs-columns", nargs="*", default=ZS_COLUMNS, metavar="COLUMN")
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"default: a fresh {DEFAULT_OUT_DIR.name}_<YYYYMMDD_HHMM> "
                         f"directory under {DEFAULT_OUT_DIR.parent}/")
    args = ap.parse_args()
    out_dir = args.out_dir or (
        DEFAULT_OUT_DIR.parent / f"{DEFAULT_OUT_DIR.name}_{datetime.now():%Y%m%d_%H%M}")

    other_splits = resolve_splits(args.other_splits, PARQUET_DIR)
    zs_columns = [] if args.no_zeroshot else args.zs_columns

    per_domain_rows: dict[str, list[dict]] = {}
    for holdout_domain in HOLDOUT_DOMAINS:
        code = DOMAIN_CODE[holdout_domain]
        print(f"\n=== holdout domain: {holdout_domain!r} (code={code!r}) ===")
        [(_, holdout_path)] = resolve_splits(["mg_sci"], PARQUET_DIR)
        zs_specs = resolve_zs_specs([holdout_path], zs_columns)
        zs_cols = tuple(c for c, _, _ in zs_specs)

        rows, eval_raw, ey, train_splits, eval_label = compute_trained(
            holdout_domain, code, other_splits, args.seeds, args.split_seed, args.test_size,
            args.fpr, args.group_col, args.human_label, args.combine, zs_cols)
        rows += compute_zeroshot(train_splits, eval_label, eval_raw, ey, zs_specs, args.fpr,
                                 args.combine)
        per_domain_rows[holdout_domain] = rows

        domain_dir = out_dir / "per_domain"
        domain_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=CSV_FIELDS).to_csv(
            domain_dir / f"{code}_performance.csv", index=False)

    print("\n=== averaged across the 3 domains ===")
    avg_rows = average_rows(per_domain_rows, args.combine)
    _print_table(avg_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "performance.csv"
    pd.DataFrame(avg_rows, columns=CSV_FIELDS).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()

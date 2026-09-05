r"""Leave-one-domain-out: trained classifiers alongside the best zero-shot
detector of each family.

    uv run python epai/ai_detection/trainv2/run_trained_vs_zeroshot_lodo_performance.py

Same three trained series as ``run_trained_vs_zeroshot_performance.py``
(``roberta_logistic``, ``roberta_fusion_cuevec``, ``roberta_gated_cuevec``),
and the same zero-shot best-of-family comparison -- but the trained side is
now leave-one-domain-out over ``--science``'s domains (computer_science,
mathematics, sociology by default, discovered from the file) instead of one
random 70/30 split:

    for each domain d:
        train on every OTHER domain's rows (the full union, not a 70/30 slice
        of it)
        calibrate a 1%-FPR threshold on d's own held-out rows -- the only
        science rows this fold's model never trained on -- and report that
        as this fold's science reading
        apply that SAME threshold, unchanged, to the full --denario file

The REPORTED numbers are the MEAN over the three folds -- one science number
and one denario number per algorithm, not nine. That is what "leave one
out" means as a single comparison point: how the algorithm does on a domain
it has never seen, averaged over which domain that was, rather than a single
lucky (or unlucky) train/test split.

Denario is evaluated in full every fold, with no split: it was never in the
training pool either way, so "from denario there is only test" is true fold
by fold as well as overall.

The zero-shot side is unchanged and un-folded: best-of-family picked once by
AUROC on the FULL --science file, each winner's already-calibrated 1%-FPR
threshold applied to the full science file and to denario -- there is nothing
to leave out for a method that trains on nothing.

Output:
    <csv>   one row per (series, dataset): mean auroc/tpr/fpr/n over folds
            for the trained series, a single reading for the zero-shot ones
    <fig>   the AUROC vs TPR@<fpr-target>%FPR scatter, all seven series

Run:
    uv run python epai/ai_detection/trainv2/run_trained_vs_zeroshot_lodo_performance.py
    uv run python epai/ai_detection/trainv2/run_trained_vs_zeroshot_lodo_performance.py --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .run_best_of_family_performance import (
    FAMILIES,
    GEN_COL,
    _KIND_BY_KEY,
    evaluate as evaluate_zeroshot,
    pick_best,
)
from .run_trained_vs_zeroshot_performance import (
    DEFAULT_DENARIO,
    DEFAULT_SCIENCE,
    HUMAN_LABEL,
    TRAINED_SPECS,
    evaluate_trained,
    fit_and_calibrate,
    load_features,
    plot,
)

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "trained_vs_zeroshot_lodo_performance"
DOMAIN_COL = "domain"


def _mean_row(rows: list[dict]) -> dict:
    """Mean over folds. ``n`` is the mean fold size too, not a total -- these
    are per-fold reads of the SAME held-out corpus (denario) or of disjoint
    slices of one corpus (the held-out science domains), and either way what
    a reader wants next to the mean AUROC is "about how many rows was that
    read on", not a sum that double-counts denario three times over."""
    return {
        "n": int(round(np.mean([r["n"] for r in rows]))),
        "auroc": float(np.mean([r["auroc"] for r in rows])),
        "tpr": float(np.mean([r["tpr"] for r in rows])),
        "fpr": float(np.mean([r["fpr"] for r in rows])),
        "n_folds": len(rows),
    }


def lodo_folds(
    spec: dict, sci_df: pd.DataFrame, domains: list[str], denario_df: pd.DataFrame,
    seed: int, fpr_target: float,
) -> tuple[list[dict], list[dict]]:
    """One fold per domain; returns ``(science_fold_rows, denario_fold_rows)``."""
    domain_col = sci_df[DOMAIN_COL].to_numpy()
    sci_rows: list[dict] = []
    den_rows: list[dict] = []
    for d in domains:
        train_idx = np.flatnonzero(domain_col != d)
        test_idx = np.flatnonzero(domain_col == d)
        fit = fit_and_calibrate(spec, sci_df, train_idx, test_idx, seed, fpr_target)
        sci_rows.append(fit["science_test_row"])
        r = evaluate_trained(spec, fit["alg"], fit["threshold"], denario_df)
        if r is not None:
            den_rows.append(r)
    return sci_rows, den_rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--science", type=Path, default=DEFAULT_SCIENCE)
    ap.add_argument("--denario", type=Path, default=DEFAULT_DENARIO)
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--human-label", default=HUMAN_LABEL)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--fig", type=Path, default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    csv_path = args.csv or DEFAULT_OUT_DIR / "trained_vs_zeroshot_lodo_performance.csv"
    fig_path = args.fig or DEFAULT_OUT_DIR / "trained_vs_zeroshot_lodo_performance.png"
    rows: list[dict] = []

    # ---- zero-shot side: unfolded, exactly as in the non-LODO sibling ----
    calib_df = pd.read_parquet(args.science)
    if GEN_COL not in calib_df.columns:
        raise SystemExit(f"{args.science}: no {GEN_COL!r} column")
    denario_df_full = pd.read_parquet(args.denario)

    print(f"zero-shot: best of each family, by AUROC on {args.science.name} "
          f"({len(calib_df)} rows)\n")
    print(f"{'family':<12}{'winner':<20}{'ref auroc':>10}")
    for family, (kind_key, cfg_path) in FAMILIES.items():
        entry, ref_auroc = pick_best(cfg_path, kind_key, calib_df, args.human_label)
        print(f"{family:<12}{entry['label']:<20}{ref_auroc:>10.3f}")
        kind = _KIND_BY_KEY[kind_key]
        for ds_name, ds_df in ((args.science.stem, calib_df), (args.denario.stem, denario_df_full)):
            r = evaluate_zeroshot(ds_df, entry["column"], entry["threshold"],
                                   kind.machine_is_high, args.human_label)
            if r is None:
                continue
            rows.append({"series": family, "label": entry["label"], "dataset": ds_name, **r})

    # ---- trained side: emb_roberta, leave-one-domain-out on science ----
    need_aux = any(s["aux"] for s in TRAINED_SPECS.values())
    sci_df = load_features(args.science, need_aux=True, extra_cols=(DOMAIN_COL,))
    den_df = load_features(args.denario, need_aux=True)
    domains = sorted(sci_df[DOMAIN_COL].unique().tolist())

    science_mean_name = f"{args.science.stem} (LODO mean, {len(domains)} domains)"
    print(f"\ntrained: leave-one-domain-out over {domains}, seed {args.seed}\n")
    print(f"{'series':<24}{'dataset':<40}{'n':>7}{'auroc':>8}{'tpr':>8}{'fpr':>8}")

    for name, spec in TRAINED_SPECS.items():
        sci_folds, den_folds = lodo_folds(spec, sci_df, domains, den_df, args.seed,
                                           args.fpr_target)
        sci_mean = _mean_row(sci_folds)
        rows.append({"series": name, "label": spec["label"], "dataset": science_mean_name,
                      **{k: v for k, v in sci_mean.items() if k != "n_folds"}})
        print(f"{name:<24}{science_mean_name:<40}{sci_mean['n']:>7}{sci_mean['auroc']:>8.3f}"
              f"{sci_mean['tpr']:>8.3f}{sci_mean['fpr']:>8.3f}")

        if den_folds:
            den_mean = _mean_row(den_folds)
            rows.append({"series": name, "label": spec["label"], "dataset": args.denario.stem,
                          **{k: v for k, v in den_mean.items() if k != "n_folds"}})
            print(f"{name:<24}{args.denario.stem:<40}{den_mean['n']:>7}{den_mean['auroc']:>8.3f}"
                  f"{den_mean['tpr']:>8.3f}{den_mean['fpr']:>8.3f}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")

    if not args.no_plot:
        plot(rows, args.fpr_target, fig_path)


if __name__ == "__main__":
    main()

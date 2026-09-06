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

The REPORTED numbers are the MEAN +/- STD over every (domain, seed) reading --
3 domains x ``--seeds`` (default 3: 42/43/44) = 9 readings pooled into one
science number and one denario number per algorithm, not printed as nine.
That std mixes two sources of variation at once, deliberately not separated:
which domain was held out, and that fold's own model fit (init, internal
validation split, optimizer noise) -- "how much does this reading move if you
redo the whole thing", not a controlled ablation of either source alone.

Denario is evaluated in full every fold, with no split: it was never in the
training pool either way, so "from denario there is only test" is true fold
by fold as well as overall.

Best-of-family is still picked once, by AUROC on the FULL --science file --
there is no reason to fold that choice, since it costs no training either
way. But its SCIENCE reading now gets a real std too: since a zero-shot score
never changes, there is nothing to refit per fold, so instead each of the
three held-out DOMAINS' rows is scored once (no seed dependence -- the domain
slice is all that varies for a fixed score) and the three domain-level
readings are what the mean +/- std is taken over, the zero-shot analogue of
the trained side's per-fold reads. Denario still gets a single deterministic
reading either way: nothing here ever splits it.

Output, three levels in one CSV (a ``level`` column tells them apart) -- see
``run_trained_vs_zeroshot_performance.py``'s own docstring for the full
description; here a ``run`` row also carries ``held_out_domain``:
    run           trained: one row per (series, dataset, seed, held_out_domain).
                  zero-shot: one row per held-out domain for science (seed is
                  always ``None`` -- nothing about the score depends on it);
                  a single row for denario (domain and seed both ``None``).
    dataset_mean  one row per (series, dataset): mean +/- std over every
                  reading feeding it -- (domain, seed) pairs for a trained
                  series, domains alone for a zero-shot one. Absent for
                  denario, which has only the one deterministic run.
    overall_mean  one row per series: mean +/- std OF the (up to two)
                  per-dataset numbers feeding it -- "the mean of the
                  datasets", drawn as the solid circle and its error bar.
    <fig>   the AUROC vs TPR@<fpr-target>%FPR scatter, all seven series

Run:
    uv run python epai/ai_detection/trainv2/run_trained_vs_zeroshot_lodo_performance.py
    uv run python epai/ai_detection/trainv2/run_trained_vs_zeroshot_lodo_performance.py --seeds 7 8 9
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
    mean_row,
    plot,
)

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "trained_vs_zeroshot_lodo_performance"
DOMAIN_COL = "domain"


def lodo_folds(
    spec: dict, sci_df: pd.DataFrame, domains: list[str], denario_df: pd.DataFrame,
    seed: int, fpr_target: float,
) -> tuple[list[tuple[str, dict, float]], list[tuple[str, dict, float]]]:
    """One fold per domain; returns ``(science_folds, denario_folds)``, each a
    list of ``(held_out_domain, row, threshold)``."""
    domain_col = sci_df[DOMAIN_COL].to_numpy()
    sci_folds: list[tuple[str, dict, float]] = []
    den_folds: list[tuple[str, dict, float]] = []
    for d in domains:
        train_idx = np.flatnonzero(domain_col != d)
        test_idx = np.flatnonzero(domain_col == d)
        fit = fit_and_calibrate(spec, sci_df, train_idx, test_idx, seed, fpr_target)
        sci_folds.append((d, fit["science_test_row"], fit["threshold"]))
        r = evaluate_trained(spec, fit["alg"], fit["threshold"], denario_df)
        if r is not None:
            den_folds.append((d, r, fit["threshold"]))
    return sci_folds, den_folds


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--science", type=Path, default=DEFAULT_SCIENCE)
    ap.add_argument("--denario", type=Path, default=DEFAULT_DENARIO)
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                     help="one LODO pass (all domains) per seed; reported numbers "
                          "are mean +/- std over every (domain, seed) reading "
                          "(default: 3 seeds)")
    ap.add_argument("--human-label", default=HUMAN_LABEL)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--fig", type=Path, default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    csv_path = args.csv or DEFAULT_OUT_DIR / "trained_vs_zeroshot_lodo_performance.csv"
    fig_path = args.fig or DEFAULT_OUT_DIR / "trained_vs_zeroshot_lodo_performance.png"
    rows: list[dict] = []

    # ---- ONE science frame for everything, domain column included, so the
    # zero-shot per-domain slices below are the identical rows the trained
    # folds hold out.
    sci_df = load_features(args.science, need_aux=True, extra_cols=(DOMAIN_COL,),
                            all_columns=True)
    if GEN_COL not in sci_df.columns:
        raise SystemExit(f"{args.science}: no {GEN_COL!r} column")
    denario_full = pd.read_parquet(args.denario)  # zero-shot denario: full file, no filter
    den_df = load_features(args.denario, need_aux=True)  # trained denario: emb/aux-filtered
    domains = sorted(sci_df[DOMAIN_COL].unique().tolist())
    science_mean_name = f"{args.science.stem} (LODO mean, {len(domains)} domains)"

    # ---- zero-shot side. Best-of-family still picked once, on the pooled
    # file (no training to fold). Science: one reading per held-out domain --
    # the zero-shot analogue of a trained fold, real domain-driven variance
    # instead of the std-of-two-arbitrary-numbers the first version of this
    # script computed. Denario: one deterministic reading, as always.
    print(f"zero-shot: best of each family, by AUROC on {args.science.name} "
          f"({len(sci_df)} rows)\n")
    print(f"{'family':<12}{'winner':<20}{'ref auroc':>10}")
    domain_col = sci_df[DOMAIN_COL].to_numpy()
    for family, (kind_key, cfg_path) in FAMILIES.items():
        entry, ref_auroc = pick_best(cfg_path, kind_key, sci_df, args.human_label)
        print(f"{family:<12}{entry['label']:<20}{ref_auroc:>10.3f}")
        kind = _KIND_BY_KEY[kind_key]

        sci_reads: list[dict] = []
        for d in domains:
            sub = sci_df[domain_col == d]
            r = evaluate_zeroshot(sub, entry["column"], entry["threshold"],
                                   kind.machine_is_high, args.human_label)
            if r is None:
                continue
            sci_reads.append(r)
            rows.append({"series": family, "label": entry["label"], "dataset": science_mean_name,
                          "level": "run", "seed": None, "held_out_domain": d, **r})
        sci_mean = mean_row(sci_reads) if sci_reads else None
        if sci_mean:
            rows.append({"series": family, "label": entry["label"], "dataset": science_mean_name,
                          "level": "dataset_mean", "seed": None, "held_out_domain": None,
                          **{k: v for k, v in sci_mean.items() if k != "n_reads"},
                          "n_reads": sci_mean["n_reads"]})

        den_r = evaluate_zeroshot(denario_full, entry["column"], entry["threshold"],
                                   kind.machine_is_high, args.human_label)
        if den_r is not None:
            rows.append({"series": family, "label": entry["label"], "dataset": args.denario.stem,
                          "level": "run", "seed": None, "held_out_domain": None, **den_r})

        dataset_means = ([sci_mean] if sci_mean else []) + ([den_r] if den_r is not None else [])
        if dataset_means:
            overall = mean_row(dataset_means)
            rows.append({"series": family, "label": entry["label"], "dataset": "overall",
                          "level": "overall_mean",
                          **{k: v for k, v in overall.items() if k != "n_reads"}})

    print(f"\ntrained: leave-one-domain-out over {domains}, seeds {args.seeds}\n")
    print(f"{'series':<24}{'dataset':<40}{'n':>7}{'auroc':>8}{'+/-':>7}"
          f"{'tpr':>8}{'+/-':>7}{'fpr':>8}")

    for name, spec in TRAINED_SPECS.items():
        sci_reads: list[dict] = []
        den_reads: list[dict] = []
        for seed in args.seeds:
            sci_folds, den_folds = lodo_folds(spec, sci_df, domains, den_df, seed,
                                               args.fpr_target)
            for d, row, thr in sci_folds:
                sci_reads.append(row)
                rows.append({"series": name, "label": spec["label"], "dataset": science_mean_name,
                              "level": "run", "seed": seed, "held_out_domain": d,
                              "threshold": thr, **row})
            for d, row, thr in den_folds:
                den_reads.append(row)
                rows.append({"series": name, "label": spec["label"], "dataset": args.denario.stem,
                              "level": "run", "seed": seed, "held_out_domain": d,
                              "threshold": thr, **row})

        sci_mean = mean_row(sci_reads)
        rows.append({"series": name, "label": spec["label"], "dataset": science_mean_name,
                      "level": "dataset_mean", "seed": None, "held_out_domain": None,
                      **{k: v for k, v in sci_mean.items() if k != "n_reads"},
                      "n_reads": sci_mean["n_reads"]})
        print(f"{name:<24}{science_mean_name:<40}{sci_mean['n']:>7}{sci_mean['auroc']:>8.3f}"
              f"{sci_mean['auroc_std']:>7.3f}{sci_mean['tpr']:>8.3f}{sci_mean['tpr_std']:>7.3f}"
              f"{sci_mean['fpr']:>8.3f}")

        den_mean = None
        if den_reads:
            den_mean = mean_row(den_reads)
            rows.append({"series": name, "label": spec["label"], "dataset": args.denario.stem,
                          "level": "dataset_mean", "seed": None, "held_out_domain": None,
                          **{k: v for k, v in den_mean.items() if k != "n_reads"},
                          "n_reads": den_mean["n_reads"]})
            print(f"{name:<24}{args.denario.stem:<40}{den_mean['n']:>7}{den_mean['auroc']:>8.3f}"
                  f"{den_mean['auroc_std']:>7.3f}{den_mean['tpr']:>8.3f}{den_mean['tpr_std']:>7.3f}"
                  f"{den_mean['fpr']:>8.3f}")

        dataset_means = [m for m in (sci_mean, den_mean) if m]
        overall = mean_row(dataset_means)
        rows.append({"series": name, "label": spec["label"], "dataset": "overall",
                      "level": "overall_mean",
                      **{k: v for k, v in overall.items() if k != "n_reads"}})
        print(f"{name:<24}{'overall (mean of datasets)':<40}{overall['n']:>7}"
              f"{overall['auroc']:>8.3f}{overall['auroc_std']:>7.3f}{overall['tpr']:>8.3f}"
              f"{overall['tpr_std']:>7.3f}{overall['fpr']:>8.3f}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")

    if not args.no_plot:
        plot(rows, args.fpr_target, fig_path)


if __name__ == "__main__":
    main()

"""Cross-domain analysis: per-domain metrics + leave-one-domain-out validation.

v1: Per-domain metrics of existing runs (no retraining).
v2: Leave-one-domain-out (LODO) via domain_holdout config.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from workflow.src.iwf import paths
from workflow.src.iwf.evaluation.baseline_compare import _best_seed_test_preds
from workflow.src.iwf.evaluation import plots

# Per-run aggregate subfolder holding all model-benchmark artefacts (per-domain metrics,
# cross-domain info, and the benchmark-suite outputs). Renamed from the old "cross_domain".
MODEL_BENCHMARKS_DIR = "model_benchmarks"


def _pct_correct_clf(y_true, y_pred) -> float:
    """% of test items whose predicted impact class equals the true class."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    return 100.0 * float(np.mean(yt == yp)) if yt.size else float("nan")


def _pct_correct_reg(y_true, y_pred, thresholds) -> float:
    """% correct after binning the [0,1] regression target + prediction into the same
    impact classes (via the run's class_thresholds) — so reg and clf report on one scale."""
    thr = list(thresholds or [0.5])
    yt = np.digitize(np.asarray(y_true, dtype=float), thr)
    yp = np.digitize(np.asarray(y_pred, dtype=float), thr)
    return 100.0 * float(np.mean(yt == yp)) if yt.size else float("nan")


def per_domain_metrics(wf_dir: Path, settings, task: str) -> pd.DataFrame:
    """Compute per-domain metrics on test split for an existing run.

    ``task`` must be "reg" or "clf" (workflow_results.task_of(wf_type)) — NOT
    settings.target_kind, which names the *impact target* (e.g.
    "field_year_percentile") and is never equal to "reg"/"clf".
    """
    # Get best-seed test predictions
    preds = _best_seed_test_preds(wf_dir, task, getattr(settings, "class_thresholds", None))
    if preds is None:
        return pd.DataFrame()

    y_true, y_pred = preds

    # Load test-split domains
    try:
        prepared_path = paths.prepared_parquet(settings.dataset_id)
        df = pd.read_parquet(prepared_path, columns=["split", "domains", "subfield", "title", "year"])
    except Exception as e:
        print(f"[cross_domain] error loading prepared data: {e}")
        return pd.DataFrame()

    # Filter to test split
    test_mask = df["split"].astype(str) == "test"
    test_df = df[test_mask]

    if len(test_df) != len(y_true):
        print(f"[cross_domain] warning: {len(test_df)} test rows but {len(y_true)} predictions; skipping")
        return pd.DataFrame()

    # Extract domains
    domains = plots._get_domain(test_df)

    # Compute per-domain metrics
    rows = []
    for domain in sorted(domains.unique()):
        mask = domains == domain
        n = mask.sum()
        if n < 20:
            continue  # Skip small domains

        y_true_domain = y_true[mask]
        y_pred_domain = y_pred[mask]

        # One scale for both tasks: % of items whose predicted impact class is correct
        # (reg targets are binned into the same classes as the classifier first).
        if task == "reg":
            value = _pct_correct_reg(y_true_domain, y_pred_domain, settings.class_thresholds)
        else:  # clf
            value = _pct_correct_clf(y_true_domain, y_pred_domain)

        rows.append({
            "domain": domain,
            "n": int(n),
            "metric": "pct_correct",
            "value": float(value),
        })

    # ALL-domain aggregate row (whole test split, unfiltered) — reference line for the plot
    if rows:
        if task == "reg":
            all_value = _pct_correct_reg(y_true, y_pred, settings.class_thresholds)
        else:
            all_value = _pct_correct_clf(y_true, y_pred)
        rows.append({"domain": "ALL", "n": int(len(y_true)),
                     "metric": "pct_correct", "value": float(all_value)})

    return pd.DataFrame(rows)


def generate_cross_domain(wf_type: str, task: str, wf_dir: Path, settings) -> None:
    """Generate per-domain metrics and plot for a workflow run."""
    wf_dir = Path(wf_dir)

    try:
        metrics_df = per_domain_metrics(wf_dir, settings, task)
        if metrics_df.empty:
            return

        # Write CSV
        out_dir = wf_dir / "aggregate" / MODEL_BENCHMARKS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "per_domain_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)

        # Plot
        try:
            plots.plot_per_domain_bars(
                metrics_df,
                run_name=wf_dir.name,
                path=out_dir / "per_domain_bars.png"
            )
        except Exception as e:
            print(f"[cross_domain] plot error: {e}")

    except Exception as e:
        print(f"[cross_domain] error: {e}")


def _collect_per_domain(results_dir: Path) -> pd.DataFrame:
    """Tidy (run, domain, value, n) table from every run's per_domain_metrics.csv,
    reading the new ``model_benchmarks`` folder (or the legacy ``cross_domain`` one)."""
    rows = []
    for run_dir in results_dir.glob("*_[0-9]*"):
        for sub in (MODEL_BENCHMARKS_DIR, "cross_domain"):
            mp = run_dir / "aggregate" / sub / "per_domain_metrics.csv"
            if mp.is_file():
                mdf = pd.read_csv(mp)
                for _, row in mdf.iterrows():
                    rows.append({"run": run_dir.name, "domain": row["domain"],
                                 "value": row["value"], "n": int(row.get("n", 0))})
                break
    return pd.DataFrame(rows)


def cross_domain_matrix(results_dir: Path) -> None:
    """Cross-domain validation summary (results/summary/):

      * cross_domain_matrix.png (blue)  — run × domain per-domain metric heatmap + CSV
      * cross_domain_gap.png (olive)    — per-domain metric minus each run's ALL-domain
        value: which domains a model generalises to better / worse than its own average
      * cross_domain_difficulty.png (red) — mean per-domain metric across all runs, sorted:
        which domains are intrinsically hard/easy for every model

    Each figure ships its own CSV so it can be re-plotted.
    """
    results_dir = Path(results_dir)
    combined = _collect_per_domain(results_dir)
    if combined.empty:
        return

    summary_dir = results_dir / "summary" / "cross_domain"
    summary_dir.mkdir(parents=True, exist_ok=True)

    per_domain = combined[combined["domain"] != "ALL"]
    all_rows = combined[combined["domain"] == "ALL"].set_index("run")["value"]

    # 1) run × domain matrix (blue)
    matrix = per_domain.pivot_table(index="run", columns="domain", values="value")
    matrix.to_csv(summary_dir / "cross_domain_matrix.csv")
    try:
        plots.plot_cross_domain_matrix(matrix, path=summary_dir / "cross_domain_matrix.png")
    except Exception as e:
        print(f"[cross_domain] matrix plot error: {e}")

    # 2) generalization gap = per-domain value − that run's ALL value (olive)
    if not all_rows.empty:
        gap = per_domain.copy()
        gap["all_value"] = gap["run"].map(all_rows)
        gap["gap"] = gap["value"] - gap["all_value"]
        gap_matrix = gap.pivot_table(index="run", columns="domain", values="gap")
        gap_matrix.to_csv(summary_dir / "cross_domain_gap.csv")
        try:
            plots.plot_cross_domain_gap(gap_matrix, path=summary_dir / "cross_domain_gap.png")
        except Exception as e:
            print(f"[cross_domain] gap plot error: {e}")

    # 3) per-domain difficulty across runs (red)
    difficulty = (per_domain.groupby("domain")
                  .agg(mean_value=("value", "mean"), std_value=("value", "std"),
                       n_runs=("value", "size"))
                  .reset_index().sort_values("mean_value"))
    difficulty["std_value"] = difficulty["std_value"].fillna(0.0)
    difficulty.to_csv(summary_dir / "cross_domain_difficulty.csv", index=False)
    try:
        plots.plot_cross_domain_difficulty(difficulty,
                                           path=summary_dir / "cross_domain_difficulty.png")
    except Exception as e:
        print(f"[cross_domain] difficulty plot error: {e}")

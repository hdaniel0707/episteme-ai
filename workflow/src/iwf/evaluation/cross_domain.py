"""Cross-domain analysis: per-domain metrics + leave-one-domain-out validation.

v1: Per-domain metrics of existing runs (no retraining).
v2: Leave-one-domain-out (LODO) via domain_holdout config.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from workflow.src.iwf import paths
from workflow.src.iwf.evaluation.baseline_compare import _best_seed_test_preds
from workflow.src.iwf.evaluation.metrics import regression_metrics, classification_metrics
from workflow.src.iwf.evaluation import plots


def per_domain_metrics(wf_dir: Path, settings) -> pd.DataFrame:
    """Compute per-domain metrics on test split for an existing run."""
    task = settings.target_kind

    # Get best-seed test predictions
    preds = _best_seed_test_preds(wf_dir)
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

        if task == "reg":
            metrics_dict = regression_metrics(y_true_domain, y_pred_domain)
            metric_name = "spearman"
            value = metrics_dict.get(metric_name, np.nan)
        else:  # clf
            class_names = [str(i) for i in range(len(settings.class_names))]
            metrics_dict = classification_metrics(y_true_domain, y_pred_domain, class_names)
            metric_name = "macro_f1"
            value = metrics_dict.get(metric_name, np.nan)

        rows.append({
            "domain": domain,
            "n": int(n),
            "metric": metric_name,
            "value": float(value),
        })

    return pd.DataFrame(rows)


def generate_cross_domain(wf_type: str, task: str, wf_dir: Path, settings) -> None:
    """Generate per-domain metrics and plot for a workflow run."""
    wf_dir = Path(wf_dir)

    try:
        metrics_df = per_domain_metrics(wf_dir, settings)
        if metrics_df.empty:
            return

        # Write CSV
        out_dir = wf_dir / "aggregate" / "cross_domain"
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


def cross_domain_matrix(results_dir: Path) -> None:
    """Generate LODO matrix: runs with domain_holdout vs test domain."""
    results_dir = Path(results_dir)

    # Collect runs with domain_holdout set
    lodo_runs = {}
    for run_dir in results_dir.glob("*_[0-9]*"):
        config_path = run_dir / "run_config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        domain_holdout = config.get("domain_holdout", "")
        if domain_holdout:
            lodo_runs[run_dir.name] = domain_holdout

    if not lodo_runs:
        return

    # Collect per-domain metrics
    rows = []
    for run_dir in results_dir.glob("*_[0-9]*"):
        metrics_path = run_dir / "aggregate" / "cross_domain" / "per_domain_metrics.csv"
        if metrics_path.is_file():
            metrics_df = pd.read_csv(metrics_path)
            for _, row in metrics_df.iterrows():
                rows.append({
                    "run": run_dir.name,
                    "domain": row["domain"],
                    "metric_value": row["value"],
                })

    if not rows:
        return

    combined = pd.DataFrame(rows)

    # Pivot to matrix
    matrix = combined.pivot_table(
        index="run",
        columns="domain",
        values="metric_value"
    )

    # Plot matrix
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    try:
        plots.plot_cross_domain_matrix(
            matrix,
            path=summary_dir / "cross_domain_matrix.png"
        )
    except Exception as e:
        print(f"[cross_domain] matrix plot error: {e}")

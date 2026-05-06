"""
Sequentially run one or more training experiments defined by YAML config files.

Usage:
    python run_experiments.py cfg/exp1.yaml cfg/exp2.yaml ...
    python run_experiments.py --cfg-dir cfg/          # run every *.yaml in a folder

For each config the script:
  1. Loads and validates the YAML.
  2. Runs N_RUNS training seeds.
  3. Saves metrics (runs.json, summary.json) + a verbatim copy of the config
     to results/<run_id>/ for full reproducibility.
  4. Saves the final model + scaler from the last seed.
"""

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from train import build_features, train_run
from utils.parquet_utils import load_parquet
from utils.visualization_utils import plot_results
from compare_runs import compare_runs

BASE_DIR = Path(__file__).parent


CFG_DIR = BASE_DIR / "cfg" / "hd" / "alma"
ENSAMBLE_RUN_ID = "hd_" + datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

REQUIRED_KEYS = {"data", "features", "model"}
REQUIRED_DATA_KEYS = {"parquet_path"}
REQUIRED_MODEL_KEYS = {"seed", "n_runs", "ml_alg"}
GB_MODEL_KEYS = {"n_estimators", "learning_rate", "max_depth", "subsample"}


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    missing = REQUIRED_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"{path}: missing top-level keys: {missing}")

    missing_data = REQUIRED_DATA_KEYS - cfg["data"].keys()
    if missing_data:
        raise ValueError(f"{path}: missing data keys: {missing_data}")

    missing_data = REQUIRED_MODEL_KEYS - cfg["model"].keys()
    if missing_data:
        raise ValueError(f"{path}: missing data keys: {missing_data}")

    # Auto-generate run_id from filename + timestamp if not provided
    if not cfg.get("run_id"):
        cfg["run_id"] = f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return cfg


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------


def run_experiment(cfg: dict, cfg_source_path: Path) -> dict:
    import joblib

    run_id = cfg["run_id"]
    print(f"\n{'='*60}")
    print(f"  Experiment: {run_id}")
    print(f"  Config:     {cfg_source_path}")
    print(f"{'='*60}")

    # Resolve paths
    parquet_path = BASE_DIR / cfg["data"]["parquet_path"]
    results_dir = BASE_DIR / "results" / ENSAMBLE_RUN_ID / run_id
    metrics_dir = results_dir / "metrics"
    models_dir = results_dir / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save a verbatim copy of the config alongside the results
    shutil.copy(cfg_source_path, results_dir / "config.yaml")
    # Also embed the config in a JSON for easy programmatic access
    with open(results_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Load data
    df = load_parquet(parquet_path)
    features = cfg["features"]
    X, y, feature_labels = build_features(df, features)
    print(f"  Features:   {features}")
    print(f"  X shape:    {X.shape}   y shape: {y.shape}")

    # Unpack model hyperparams
    m = cfg["model"]
    n_runs: int = m["n_runs"]
    base_seed: int = m["seed"]
    ml_alg = m["ml_alg"]

    all_metrics = []
    model = scaler = None
    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        (
            model,
            scaler,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            train_losses,
            val_losses,
            metrics,
        ) = train_run(
            X,
            y,
            seed,
            ml_alg=ml_alg,
        )

        metrics["seed"] = seed
        metrics["feature_labels"] = feature_labels
        all_metrics.append(metrics)

        print(
            f"  Run {run_idx+1:2d}/{n_runs}  seed={seed}"
            f"  train RMSE={metrics['train_rmse']:.4f}  R²={metrics['train_r2']:.4f}"
            f"  val RMSE={metrics['val_rmse']:.4f}  R²={metrics['val_r2']:.4f}"
            f"  test RMSE={metrics['test_rmse']:.4f}  R²={metrics['test_r2']:.4f}"
        )

    # Aggregate summary
    metric_keys = [
        "train_rmse",
        "train_r2",
        "val_rmse",
        "val_r2",
        "test_rmse",
        "test_r2",
    ]
    summary = {}
    print(f"\n  --- Summary over {n_runs} runs ---")
    for key in metric_keys:
        vals = np.array([m[key] for m in all_metrics])
        q1, q3 = np.percentile(vals, [25, 75])
        iqm_mask = (vals >= q1) & (vals <= q3)
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "iqm": float(np.mean(vals[iqm_mask])),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
        print(
            f"  {key:12s}  mean={summary[key]['mean']:.4f}  std={summary[key]['std']:.4f}"
            f"  min={summary[key]['min']:.4f}  max={summary[key]['max']:.4f}"
        )

    # Persist metrics
    with open(metrics_dir / "runs.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(metrics_dir / "summary.json", "w") as f:
        json.dump({"run_id": run_id, "n_runs": n_runs, "metrics": summary}, f, indent=2)
    print(f"\n  Metrics -> {metrics_dir}")

    # Persist model + scaler from the last seed
    if model is not None:
        joblib.dump(model, models_dir / "model.joblib")
        joblib.dump(scaler, models_dir / "scaler.joblib")
        print(f"  Model   -> {models_dir}")

    return {
        "run_id": run_id,
        "run_label": cfg.get("run_labels", run_id),
        "summary": summary,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def collect_configs(cfg_dir: Path) -> list[Path]:
    paths: list[Path] = []

    if not cfg_dir.is_dir():
        print(f"ERROR: --cfg-dir '{cfg_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    paths = sorted(cfg_dir.glob("*.yaml")) + sorted(cfg_dir.glob("*.yml"))
    if not paths:
        print(f"ERROR: No YAML files found in '{cfg_dir}'.", file=sys.stderr)
        sys.exit(1)

    seen, unique = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def main():
    config_paths = collect_configs(CFG_DIR)

    print(f"Found {len(config_paths)} experiment(s) to run:")
    for p in config_paths:
        print(f"  {p}")

    results = []
    for cfg_path in config_paths:
        try:
            cfg = load_config(cfg_path)
        except Exception as e:
            print(f"\nERROR loading config '{cfg_path}': {e}", file=sys.stderr)
            results.append(
                {"config": str(cfg_path), "status": "config_error", "error": str(e)}
            )
            continue

        try:
            result = run_experiment(cfg, cfg_path)
        except Exception:
            tb = traceback.format_exc()
            print(f"\nERROR during experiment '{cfg_path}':\n{tb}", file=sys.stderr)
            results.append(
                {"config": str(cfg_path), "status": "runtime_error", "error": tb}
            )
            continue

        results.append({"config": str(cfg_path), **result})

        try:
            results_dir = BASE_DIR / "results" / ENSAMBLE_RUN_ID / result["run_id"]
            plot_results(results_dir)
        except Exception:
            print(
                f"{YELLOW}\nWARN: plotting failed for '{result['run_id']}':\n{traceback.format_exc()}{RESET}",
                file=sys.stderr,
            )
            assert False

    # Final summary across all experiments
    print(f"\n{'='*60}")
    print("  All experiments complete")
    print(f"{'='*60}")
    for r in results:
        status = r["status"]
        label = r.get("run_id", r["config"])
        if status == "ok":
            test_rmse = r["summary"]["test_rmse"]["mean"]
            test_r2 = r["summary"]["test_r2"]["mean"]
            print(f"  [OK]    {label:30s}  test RMSE={test_rmse:.4f}  R²={test_r2:.4f}")
        else:
            print(f"  [FAIL]  {label}  ({status})")

    # Run comparison across all successful experiments
    successful = [r for r in results if r.get("status") == "ok"]
    if len(successful) >= 2:
        run_ids = [r["run_id"] for r in successful]
        run_labels = [r.get("run_label", r["run_id"]) for r in successful]
        compare_id = (
            CFG_DIR.name + "_" + "_".join(rid.split("_")[-1] for rid in run_ids)
        )
        compare_runs(ENSAMBLE_RUN_ID, run_ids, run_labels)
    else:
        print("\n  Skipping comparison: need at least 2 successful experiments.")


if __name__ == "__main__":
    main()

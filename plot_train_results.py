import json
from pathlib import Path

import numpy as np

from utils.visualization_utils import (
    plot_loss_curves_ci,
    plot_actual_vs_predicted,
    plot_feature_importances,
)

BASE_DIR = Path(__file__).parent
RUN_ID = "0505_A4"
RESULTS_DIR = BASE_DIR / "results" / RUN_ID / "metrics"
FIG_DIR = BASE_DIR / "results" / RUN_ID / "figs" / "train"

TOP_K_FEATURES = 20

if __name__ == "__main__":
    with open(RESULTS_DIR / "runs.json") as f:
        runs = json.load(f)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Loss curves with 95% CI ---
    all_train = [r["train_losses"] for r in runs]
    all_val = [r["val_losses"] for r in runs]
    plot_loss_curves_ci(all_train, all_val, FIG_DIR / "loss_curves_ci.png", ci=0.95)

    # --- Per-seed actual vs predicted and feature importance ---
    for run in runs:
        seed = run["seed"]

        if "y_test" not in run or "y_pred" not in run:
            print(
                f"Seed {seed}: no predictions in runs.json — re-run train.py to generate them"
            )
            continue

        y_test = np.array(run["y_test"])
        y_pred = np.array(run["y_pred"])
        plot_actual_vs_predicted(
            y_test,
            y_pred,
            f"Test set  (seed={seed})",
            FIG_DIR / f"actual_vs_predicted_seed{seed}.png",
        )

        if "feature_importances" not in run or "feature_labels" not in run:
            print(
                f"Seed {seed}: no feature importances in runs.json — re-run train.py to generate them"
            )
            continue

        importances = np.array(run["feature_importances"])
        labels = run["feature_labels"]
        plot_feature_importances(
            importances,
            labels,
            TOP_K_FEATURES,
            FIG_DIR / f"feature_importances_seed{seed}.png",
        )

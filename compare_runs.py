import json
from pathlib import Path

import numpy as np
from scipy import stats

from utils.visualization_utils import plot_loss_curves_comparison, plot_test_metrics_ci

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

CI = 0.95
METRICS = ["test_rmse", "test_r2", "val_rmse", "val_r2"]

#################################
RUN_IDS = ["0505_A2", "0505_A3", "0505_A4"]
RUN_LABELS = ["all", "only embeddings", "only h-index"]
COMPARE_ID = "0505A2_A3_A4"
OUT_DIR = RESULTS_DIR / "compared" / COMPARE_ID


def load_run(run_id: str) -> list[dict]:
    path = RESULTS_DIR / run_id / "metrics" / "runs.json"
    if not path.exists():
        raise FileNotFoundError(f"No runs.json found for run '{run_id}' at {path}")
    with open(path) as f:
        return json.load(f)


def _ci_stats(values: np.ndarray, ci: float) -> dict:
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf((1 + ci) / 2, df=max(n - 1, 1))
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - t_crit * se,
        "ci_high": mean + t_crit * se,
        "n": n,
    }


def build_comparison(
    run_ids: list[str], runs_data: dict[str, list[dict]], ci: float
) -> dict:
    result: dict = {"run_ids": run_ids, "ci": ci, "runs": {}}
    for run_id in run_ids:
        run_stats: dict = {}
        for metric in METRICS:
            values = np.array([r[metric] for r in runs_data[run_id] if metric in r])
            if len(values):
                run_stats[metric] = _ci_stats(values, ci)
        result["runs"][run_id] = run_stats
    return result


def main() -> None:

    fig_dir = OUT_DIR / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Comparing runs: {RUN_IDS}")
    print(f"Output: {OUT_DIR}")

    runs_data: dict[str, list[dict]] = {}
    for run_id in RUN_IDS:
        runs_data[run_id] = load_run(run_id)
        print(f"  Loaded '{run_id}': {len(runs_data[run_id])} seeds")

    comparison = build_comparison(RUN_IDS, runs_data, CI)

    json_out = OUT_DIR / "comparison.json"
    with open(json_out, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved {json_out}")

    plot_test_metrics_ci(comparison, fig_dir, labels=RUN_LABELS)
    plot_loss_curves_comparison(RUN_IDS, runs_data, fig_dir, CI, labels=RUN_LABELS)

    print("Done.")


if __name__ == "__main__":
    main()

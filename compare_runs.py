import json
from pathlib import Path

import numpy as np
from scipy import stats

from utils.visualization_utils import plot_loss_curves_comparison, plot_metrics_ci

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

CI = 0.95
METRICS = ["train_rmse", "train_r2", "val_rmse", "val_r2", "test_rmse", "test_r2"]


def load_run(result_dir: Path) -> list[dict]:
    path = result_dir / "metrics" / "runs.json"
    if not path.exists():
        raise FileNotFoundError(f"No runs.json found for run '{result_dir}' at {path}")
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


def compare_runs(
    ensemble_run_ids: Path,
    run_ids: list[str],
    run_labels: list[str],
) -> None:
    out_dir = RESULTS_DIR / ensemble_run_ids / "compared"
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Comparing runs: {run_ids}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    runs_data: dict[str, list[dict]] = {}
    for run_id in run_ids:
        result_dir = RESULTS_DIR / ensemble_run_ids / run_id
        runs_data[run_id] = load_run(result_dir)
        print(f"  Loaded '{run_id}': {len(runs_data[run_id])} seeds")

    comparison = build_comparison(run_ids, runs_data, CI)

    json_out = out_dir / "comparison.json"
    with open(json_out, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Saved {json_out}")

    plot_metrics_ci(comparison, fig_dir, labels=run_labels)
    plot_loss_curves_comparison(run_ids, runs_data, fig_dir, CI, labels=run_labels)

    print("Done.")


def main():
    ensemble_run_ids = RESULTS_DIR / "a"
    run_ids = ["0505_C_1", "0505_C_2", "0505_C_3"]
    run_labels = ["c1", "c2", "c3"]
    compare_runs(run_ids, run_labels)


if __name__ == "__main__":
    main()

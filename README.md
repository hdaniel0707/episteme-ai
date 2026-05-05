# Article Impact Dataset

> **Note**: This repository is currently under construction.

A pipeline for fetching academic paper data from the Semantic Scholar API, analysing it, and training regression models to predict article citation impact.

## Package Manager

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

```bash
uv sync
```

---

## Project Structure

```
article-impact-dataset/
├── fetch.py              # Fetch papers + author metadata from Semantic Scholar API
├── analyse.py            # Dimensionality reduction, EDA plots, exports to Parquet
├── train.py              # Train a GradientBoosting model on a single feature set
├── run_experiments.py    # Run multiple experiments from YAML config files
├── compare_runs.py       # Compare metrics across experiment runs with confidence intervals
├── plot_train_results.py # Plot loss curves, predictions, and feature importances for one run
│
├── cfg/                  # Experiment config files (YAML)
│   └── example.yaml      # Example config showing all available options
│
├── data/
│   ├── sql/              # Raw SQLite databases produced by fetch.py
│   └── paquet/           # Processed Parquet files produced by analyse.py
│
├── results/              # One sub-directory per run_id
│   └── <run_id>/
│       ├── metrics/      # runs.json (per-seed metrics), summary.json
│       ├── models/       # Saved model + scaler (joblib)
│       └── figs/         # Plots for this run
│
└── utils/                # Shared helper modules
    ├── s2_utils.py        # Semantic Scholar API client
    ├── sqlite_utils.py    # DB init, save, and load helpers
    ├── parquet_utils.py   # Parquet read/write helpers
    ├── dimred_utils.py    # PCA / dimensionality reduction
    ├── visualization_utils.py  # All plotting functions
    └── env_utils.py       # .env and dependency checks
```

---

## Pipeline

```
fetch.py  →  data/sql/*.db
                ↓
          analyse.py  →  data/paquet/
                              ↓
                    run_experiments.py  ←  cfg/*.yaml
                              ↓
                         results/<run_id>/
                              ↓
                    compare_runs.py / plot_train_results.py
```

### Step-by-step

1. **Fetch** — queries Semantic Scholar for papers and enriches each record with author h-index, citation counts, and paper embeddings.
2. **Analyse** — loads the SQLite DB, reduces embeddings via PCA, generates EDA plots, and exports a Parquet dataset.
3. **Train** — fits a `GradientBoostingRegressor` to predict `citationCount`. Supports configurable feature sets (`embedding`, `year`, `best_author_h_index`, …).
4. **Experiment runner** — wraps `train.py` to run multiple YAML-defined experiments sequentially, saving metrics and a config snapshot for full reproducibility.
5. **Compare / Plot** — visualise loss curves, confidence intervals, and feature importances across runs.

---

## Configuration

Experiments are defined as YAML files in `cfg/`. Example:

```yaml
run_id: "my_experiment"

data:
  parquet_path: "data/paquet/0505_A"

features:
  - embedding
  - year
  - best_author_h_index

model:
  n_estimators: 300
  learning_rate: 0.05
  max_depth: 4
  subsample: 0.8
  seed: 42
  n_runs: 3
```

Run one or more configs:

```bash
python run_experiments.py cfg/example.yaml
python run_experiments.py --cfg-dir cfg/   # run all *.yaml in the folder
```

---

## Environment

Copy `.env_sample` to `.env` and fill in your Semantic Scholar API key before running `fetch.py`.

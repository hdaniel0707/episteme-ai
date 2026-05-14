# Article Impact Dataset

> **Note**: This repository is currently under construction.

A pipeline for fetching academic paper data from the Semantic Scholar API, analysing it, and training regression models to predict article citation impact.

## Claude Code

For AI-assisted development, copy the [Andrej Karpathy CLAUDE.md](https://github.com/forrestchang/andrej-karpathy-skills/blob/main/CLAUDE.md) into your global Claude config folder:

```bash
~/.claude/CLAUDE.md
```

## Package Manager

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

```bash
uv sync
```

For installing the aip package run:
```bash
uv pip install -e .
```

## Environment

Copy `.env_sample` to `.env` and fill in your API keys:

```bash
cp .env_sample .env
```

The key variable used by the pipeline is `SEMANTIC_SCHOLAR_API_KEY`.

---

## Pipeline Overview

The pipeline has four sequential stages:

```
Fetch → Analyse Data → Run Experiments → Analyse Experiments
```

### 1. Fetch (`aip/fetch.py`)

Queries the Semantic Scholar API and stores results in a local SQLite database.

```bash
python -m aip.fetch
```

Fields collected per paper: title, abstract, SPECTER embedding, citation counts, year, venue, and author metadata (h-index, citation count, affiliations).

Output: `data/sql/<name>.db`

---

### 2. Analyse Data (`aip/analyse_data.py`)

Loads the SQLite database, converts it to Parquet, runs dimensionality reduction, and generates interactive EDA plots.

```bash
python -m aip.analyse_data
```

Plots saved to `results/analyse_data/<name>/`:
- `years_histogram.html` — publication year distribution
- `citation_year_correlation.html` — citations vs. year
- `hindex_citation_correlation.html` — best author h-index vs. citations
- `dimred_2d.html` — 2D PCA of SPECTER embeddings

---

### 3. Run Experiments (`aip/run_exp.py`)

Reads YAML config files from `cfg/`, trains regression models over multiple seeds, and saves metrics and model artefacts.

```bash
python -m aip.run_exp
```

Each run:
1. Loads features from a Parquet dataset
2. Splits data 70 / 15 / 15 (train / val / test)
3. Applies `StandardScaler`
4. Trains the configured model and evaluates RMSE and R²
5. Saves results to `results/<experiment_id>/<alg_id>/<seed>/`

After all runs complete, `run_exp` automatically triggers both analysis scripts.

---

### 4. Analyse Experiments

Both analysis scripts are called automatically at the end of `run_exp.py`. They can also be run standalone by editing the `experiment_id` at the bottom of each file.

#### Per-run analysis (`aip/analyse_exp.py`)

Produces figures for each run, aggregated across seeds. Saved to `results/<experiment_id>/<alg_id>/figs/`:
- `loss_curves.png` — mean ± std MSE over boosting stages (gradient boosting only)
- `feature_importance.png` — top-k features with error bars
- `metrics_summary.png` — RMSE and R² across train/val/test splits

#### Cross-run comparison (`aip/analyse_exp_compare.py`)

Overlays all runs in an experiment to compare configurations. Saved to `results/<experiment_id>/figs/`:
- `metrics_comparison.png` — grouped bar chart of RMSE and R² per run
- `loss_curves_comparison.png` — overlaid CI loss curves (gradient boosting only)
- `feature_importance_comparison.png` — grouped or per-run feature importances

---

## Configuration

Experiment runs are controlled by YAML files in `cfg/`. See `cfg/example.yaml`:

```yaml
alg_id: "my_run"
alg_label: "all features"
seed: 42

data:
  parquet_path: "data/paquet/0506_B"   # relative to project root

features:
  - embedding        # SPECTER vector (768-dim)
  - year
  - best_author_h_index

model:
  ml_alg: "gradient_boosting"  # gradient_boosting | linear | svm | mlp
  n_runs: 3
```

**Available features:** `embedding`, `year`, `best_author_h_index`, `h_index`, `citation_count`, `author_count`, `page_count`

**Target variable:** `log1p(citation_count)`

---

## Project Structure

```
article-impact-dataset/
├── aip/
│   ├── fetch.py                  # Fetch papers from Semantic Scholar API → SQLite
│   ├── analyse_data.py           # EDA: dimensionality reduction and correlation plots
│   ├── run_exp.py                # Train models from YAML configs, save results
│   ├── analyse_exp.py            # Per-run figures: loss curves, feature importance, metrics
│   ├── analyse_exp_compare.py    # Cross-run comparison figures
│   └── utils/
│       ├── config_utils.py       # Load and validate YAML experiment configs
│       ├── dimred_utils.py       # PCA, KernelPCA, UMAP wrappers
│       ├── env_utils.py          # .env variable checking
│       ├── parquet_utils.py      # Parquet save/load helpers
│       ├── s2_utils.py           # Semantic Scholar API client
│       ├── sqlite_utils.py       # SQLite schema, save, and load helpers
│       ├── train_utils.py        # Feature assembly and model factory
│       └── visualization_utils.py # Plotly and matplotlib plot functions
├── cfg/
│   └── example.yaml              # Example experiment config
├── data/
│   ├── sql/                      # SQLite databases (one per fetch run)
│   └── paquet/                   # Parquet datasets (one per DB)
├── results/                      # Experiment outputs (gitignored)
├── plot_train_results.py         # Legacy standalone plotting script
├── pyproject.toml
└── .env_sample                   # Environment variable template
```


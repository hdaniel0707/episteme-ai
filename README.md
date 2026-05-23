# Episteme AI

A research pipeline for studying academic papers at scale. It fetches papers from the [Semantic Scholar](https://www.semanticscholar.org/) API, enriches them with LLM-generated synthetic variants and scientific embeddings, optionally scores them with the [NAIP](https://huggingface.co/spaces/ssocean/Newborn_Article_Impact_Predict) impact predictor, and trains machine learning models for two tasks:

1. **Citation impact regression** — predict how many citations a paper will receive from its metadata and text embeddings.
2. **AI-text detection (classification)** — classify whether an abstract was written by a human or rephrased by an LLM.

---

## Table of Contents

- [Setup](#setup)
- [Git LFS — Store Folder](#git-lfs--store-folder)
- [Environment Variables](#environment-variables)
- [File Naming Convention](#file-naming-convention)
- [Pipeline Overview](#pipeline-overview)
- [Stage 1 — Fetch](#stage-1--fetch)
- [Stage 2 — Summarise DB](#stage-2--summarise-db)
- [Stage 3 — Prepare Data](#stage-3--prepare-data)
- [Stage 4 — Fetch NAIP Scores](#stage-4--fetch-naip-scores)
- [Stage 5 — Analyse Data](#stage-5--analyse-data)
- [Stage 6 — Analyse NAIP](#stage-6--analyse-naip)
- [Stage 7 — Training](#stage-7--training)
  - [Regression](#regression-run_regpy)
  - [Representation Learning](#representation-learning-run_reprpy)
  - [Classification](#classification-run_clfpy)
- [Stage 8 — Analyse Experiments](#stage-8--analyse-experiments)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [Key Concepts for Students](#key-concepts-for-students)

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install all dependencies
uv sync

# Install the epai package in editable mode
uv pip install -e .
```

You need Python 3.12 or newer.

---

## Git LFS — Store Folder

Selected datasets are stored in `store/` and tracked with [Git LFS](https://git-lfs.github.com/) (Large File Storage). These are the canonical, versioned snapshots of key `.parquet` and `.db` files used in experiments.

After cloning the repository, LFS files are not downloaded automatically. Run the following to get them:

```bash
# 1. Install Git LFS (one-time, system-wide)
sudo apt install git-lfs      # Linux / WSL
# brew install git-lfs        # macOS

# 2. Enable LFS in your local clone (one-time per repo)
git lfs install

# 3. Download all LFS-tracked files
git lfs pull

# Or download only the store/ folder
git lfs pull --include="store/**"
```

The `store/` folder mirrors the layout of `data/`:

```
store/
└── data/
    ├── parquet/    # selected Parquet datasets
    └── sql/        # selected SQLite databases
```

The `data/` folder (your local working data) is gitignored and is not synced. Copy files from `store/` into `data/` to use them in the pipeline.

---

## Environment Variables

Copy the sample file and fill in your API keys:

```bash
cp .env_sample .env
```

| Variable | Required by | Purpose |
|---|---|---|
| `SEMANTIC_SCHOLAR_API_KEY` | `fetch.py` | Increases the Semantic Scholar API rate limit |
| `OPENAI_API_KEY` | `prepare_data.py`, `embedding_utils.py` | GPT-4o-mini rephrasing and OpenAI embeddings |
| `ANTHROPIC_API_KEY` | `prepare_data.py` | Claude Haiku abstract rephrasing |
| `GOOGLE_API_KEY` | `prepare_data.py` | Gemini abstract rephrasing |
| `HF_TOKEN` | `fetch_naip.py` | HuggingFace token for the NAIP Gradio API (optional but avoids rate limits) |

The Semantic Scholar key is optional but strongly recommended — without it you will be heavily rate-limited. The LLM keys are only needed if you enable the `ADD_GEN_AI` flag in `prepare_data.py`.

---

## Pipeline Overview

The stages run sequentially. Stages 2, 4, 5, and 6 are optional analysis or enrichment steps.

```
[fetch.py]
   ↓  data/sql/<id>.db
[summarise_db.py]              ← optional: EDA on raw database
   ↓
[prepare_data.py]
   ↓  data/parquet/<id>.parquet  or  <id>_AI.parquet
[fetch_naip.py]                ← optional: add NAIP impact scores
   ↓  data/parquet/<id>_naip.parquet
[analyse_data.py]              ← optional: EDA on processed Parquet
[analyse_naip.py]              ← optional: EDA on NAIP scores
   ↓
┌──────────────────────────────────────┐
│  Training (pick one or more)         │
│  run_reg.py   → results/reg/         │
│  run_repr.py  → results/repr/        │
│  run_clf.py   → results/clf/         │
└──────────────────────────────────────┘
   ↓
[analyse_exp.py + analyse_exp_compare.py]  ← called automatically after training
```

---

## Stage 1 — Fetch

**Script:** `epai/fetch.py`
**Output:** `data/sql/<id>.db`

```bash
python -m epai.fetch
```

Queries the Semantic Scholar bulk-search API for each (domain, query) pair defined in a fetch config YAML. For every paper found it stores:

- Title, abstract, publication year, venue
- SPECTER embedding (768-dimensional vector pre-computed by Semantic Scholar)
- Citation count and influential citation count
- Per-author: name, h-index, citation count, paper count, affiliations

Papers are stored in a local SQLite database with three tables: `papers`, `authors`, and `paper_authors`.

**Configuration** — edit the constants at the top of `fetch.py`:

```python
DB_PATH = DB_DIR / "sql" / "my_dataset.db"          # output file
CFG_PATH = BASE_DIR / "cfg" / "fetch" / "hd" / "fetch_cfg_reduced.yaml"  # query config
```

The fetch config maps domains to a list of search queries (see `cfg/fetch/fetch_cfg_example.yaml`). `TOPIC_LIMIT` controls how many papers to fetch per domain, distributed evenly across queries in that domain.

---

## Stage 2 — Summarise DB

**Script:** `epai/summarise_db.py`
**Output:** `results/analyse_sql/<id>/*.html`

```bash
python -m epai.summarise_db
```

An optional exploratory step. Loads the raw SQLite database, prints descriptive statistics to the terminal, and saves a set of interactive Plotly HTML charts:

- Citation count distributions (raw and log-transformed)
- Papers per year, top venues
- Author h-index and citation distributions
- Authors per paper, best-author h-index vs paper citations
- NULL percentage per column (useful for data quality checks)
- A single-page overview dashboard

Edit `DB_NAME` at the top of the script to point at your database.

---

## Stage 3 — Prepare Data

**Script:** `epai/prepare_data.py`
**Output:** `data/parquet/<id>.parquet` or `data/parquet/<id>_AI.parquet`

```bash
python -m epai.prepare_data
```

Converts the SQLite database to a Parquet file (a columnar format that loads much faster for ML) and optionally enriches the dataset in two ways:

### 3a — AI-generated synthetic abstracts (`ADD_GEN_AI = True`)

For each human-written paper, several LLMs each produce a rephrased version of the title and abstract as structured JSON output. Two rephrasing styles are available:

- **normal** — preserves meaning, rewrites sentence structure and vocabulary
- **aggressive** — same meaning but maximally different phrasing and paragraph structure

The following model/style combinations are used by default:
- `gemini-2.5-flash-lite` / normal
- `claude-haiku-4-5-20251001` / normal
- `gpt-4o-mini` / normal
- `gpt-4o-mini` / aggressive

Only papers for which **all** models succeed are kept. Each successful paper produces N+1 rows in the output: one original (human) row and one row per model/style, tagged with a `generated_by` column (e.g. `"gpt-4o-mini:aggressive"`). This labelled dataset is what the classification task uses.

### 3b — Local scientific embeddings (`ADD_EMB_SPECTER2 = True`)

Computes fresh embeddings locally using [SPECTER2](https://huggingface.co/allenai/specter2_base). The result is stored as a new column `emb_specter2`. This is important for the classification task because Semantic Scholar only provides embeddings for real papers — the AI-generated abstracts need their own embeddings computed locally.

Other supported embedding models (see `embedding_utils.py`): `specter2`, `scibert`, `openai`.

---

## Stage 4 — Fetch NAIP Scores

**Script:** `epai/fetch_naip.py`
**Input:** `data/parquet/<id>_AI.parquet`
**Output:** `data/parquet/<id>_AI_naip.parquet`

```bash
python -m epai.fetch_naip
```

An optional enrichment step. Queries the [NAIP](https://huggingface.co/spaces/ssocean/Newborn_Article_Impact_Predict) (Newborn Article Impact Predictor) Gradio API on HuggingFace for each row in the dataset and adds two columns:

| Column | Description |
|---|---|
| `naip_raw_score` | Raw model output |
| `naip_final_score` | Normalised TNCSI_sp score (0–1) |

The script is resume-safe: if the output file already exists, previously scored rows are loaded from it and skipped. This means you can interrupt and re-run without re-querying the API.

**Configuration** — edit the constants at the top of `fetch_naip.py`:

```python
PARQUET_IN  = ...   # input Parquet (AI-augmented)
PARQUET_OUT = ...   # output path
MODEL_VERSION = "v2.1"      # "v1" or "v2.1"
SLEEP_BETWEEN = 0.2         # seconds between requests
```

A `HF_TOKEN` in `.env` is optional but recommended to avoid rate limits on the public Gradio space.

---

## Stage 5 — Analyse Data

**Script:** `epai/analyse_data.py`
**Output:** `results/analyse_data/<id>/`

```bash
python -m epai.analyse_data
```

Exploratory data analysis on the processed Parquet file. Edit `DB_ID` and `MODE_AI` at the top.

**Standard mode (`MODE_AI = False`)** produces:
- Publication year histogram and citation count histograms (raw and log-transformed)
- Citations vs year correlation (scatter + mean trend line)
- Author h-index vs citations correlation
- 2D PCA and UMAP projections of SPECTER embeddings, colored by year and by domain

**AI-augmented mode (`MODE_AI = True`)** produces:
- 2D PCA of SPECTER2 embeddings colored by `generated_by` label (human vs each LLM)
- Human vs LLM abstract semantic similarity — pairwise cosine distances between human and AI abstract embeddings, showing how far each LLM's rephrasing drifts in embedding space

All plots are saved as interactive `.html` files (Plotly).

---

## Stage 6 — Analyse NAIP

**Script:** `epai/analyse_naip.py`
**Input:** `data/parquet/<id>_naip.parquet`
**Output:** `results/naip/<parquet_stem>/`

```bash
python -m epai.analyse_naip
```

An optional analysis step for datasets that have been enriched with NAIP scores (Stage 4). Produces:

- `overall_distribution.png` — histograms of raw and final score across all rows
- `grouped_raw.png` / `grouped_final.png` — score distributions per `generated_by` group
- `grouped_stats_{raw,final}.png` — mean ± std bar chart per group
- `corr_raw_citations.png` / `corr_final_citations.png` — NAIP score vs `log(1 + citation_count)`
- `corr_raw_yearly_citations.png` / `corr_final_yearly_citations.png` — NAIP score vs yearly citations

Edit the `PARQUET_PATH` constant at the top of the script.

---

## Stage 7 — Training

All three training scripts follow the same pattern:

1. Load all YAML configs from a directory (`cfg/train/<task>/...`)
2. For each config, run `n_runs` independent seeds
3. Each seed: load data → build features → split 70/15/15 → `StandardScaler` → train model → save results as JSON + model as `.joblib`
4. After all configs finish, automatically run `analyse_exp.py` and `analyse_exp_compare.py`

### Regression (`run_reg.py`)

**Output:** `results/reg/<experiment_id>/`

```bash
python -m epai.run_reg
```

Predicts `log1p(citation_count)` from paper features. Metrics saved per seed: RMSE and R² on train, val, and test splits.

**Available models** (`ml_alg` in config):

| Value | Model |
|---|---|
| `gradient_boosting` | Gradient Boosting Regressor (sklearn) |
| `linear` | Linear Regression |
| `svm` | Support Vector Regression |
| `mlp` | MLP Regressor (sklearn) |
| `ae` | Autoencoder + Ridge head (see Representation Learning) |

**Available features** (`features` list in config):

| Name | Description |
|---|---|
| `embedding` | 768-dim SPECTER vector pre-computed by Semantic Scholar |
| `emb_specter2` | Locally computed SPECTER2 vector |
| `year` | Publication year (scalar) |
| `best_author_h_index` | H-index of the most cited author on the paper (scalar) |
| `author_count` | Number of authors (scalar) |
| `page_count` | Number of pages (scalar) |

Multiple features can be combined — they are concatenated column-wise before training.

### Representation Learning (`run_repr.py`)

**Output:** `results/repr/<experiment_id>/`

```bash
python -m epai.run_repr
```

Trains the **Autoencoder Regressor** defined in `epai/autoencoder.py`. This is a two-stage model:

1. A deep autoencoder is trained to reconstruct the input embedding (768 → 512 → 256 → **128** → 256 → 512 → 768). The loss is a weighted sum of MSE and cosine similarity.
2. A Ridge regression head is fitted on the compressed 128-dimensional **latent space** to predict `log1p(citation_count)`.

This lets you study what structure emerges in a compressed embedding space and whether that compressed structure is predictive of citations. Use `ml_alg: ae` in the repr config.

### Classification (`run_clf.py`)

**Output:** `results/clf/<experiment_id>/`

```bash
python -m epai.run_clf
```

Classifies whether an abstract is human-written or generated by a specific model/style, based on the `generated_by` label column. The feature is the `emb_specter2` embedding. Metrics: accuracy and a per-class classification report (precision, recall, F1).

Train/val/test splits are **stratified** to ensure balanced class representation across splits. The label encoder mapping (integer → class name) is saved as `label_mapping.json` next to the model.

**Available classifiers** (`ml_alg` in config):

| Value | Model |
|---|---|
| `gradient_boosting` | Gradient Boosting Classifier |
| `logistic` | Logistic Regression |
| `svm` | Support Vector Classifier |
| `mlp` | MLP Classifier |

You can exclude specific classes from training using `drop_classes` in the config — for example, to run a binary human-vs-AI experiment instead of multi-class.

---

## Stage 8 — Analyse Experiments

These two scripts are called automatically at the end of each training script. They can also be run standalone by editing the `experiment_id` variable at the bottom of each file.

### Per-run analysis (`epai/analyse_exp.py`)

For each algorithm in an experiment, produces figures aggregated across all seeds. Saved to `results/<task>/<experiment_id>/<alg_id>/figs/`:

- `loss_curves.png` — mean ± std loss over boosting stages or autoencoder epochs
- `feature_importance.png` — top-20 features with error bars across seeds (gradient boosting only)
- `metrics_summary.png` — RMSE and R² across train/val/test splits with per-seed dots
- `pred_vs_gt.png` — hexbin scatter of predicted vs. ground truth citations per seed

### Cross-run comparison (`epai/analyse_exp_compare.py`)

Overlays all algorithm configs within one experiment. Saved to `results/<task>/<experiment_id>/figs/`:

- `results_comparison.png` — grouped bar chart of metrics per algorithm
- `loss_curves_comparison.png` — overlaid CI bands for all algorithms
- `feature_importance_comparison.png` — grouped or per-algorithm feature importances

---

## Configuration Reference

### Fetch config (`cfg/fetch/fetch_cfg_example.yaml`)

```yaml
queries_domains:
  Physics:
    - "quantum mechanics particle physics"
    - "condensed matter solid state physics"
  Computer Science:
    - "machine learning deep learning"
    - "natural language processing large language models"
```

Each domain maps to a list of free-text search queries sent to the Semantic Scholar API.

### Training configs (`cfg/train/<task>/`)

```yaml
alg_id: "A1"           # unique name → determines the output subdirectory
alg_label: "SPECTER + year + h-index"  # human-readable label used in plots

seed: 42               # base random seed; run i uses seed + i

data:
  parquet_path: "data/parquet/my_dataset.parquet"  # relative to project root
  drop_classes: []     # classification only — list classes to exclude

features:
  - embedding          # one or more feature columns (regression/repr)
  - year
  - best_author_h_index

model:
  ml_alg: "gradient_boosting"   # see model tables above
  n_runs: 3                     # number of independent seeds to average over
```

Place multiple YAML files in a directory and point `CFG_DIR` in the training script at that directory — all configs will be run as separate algorithms within the same timestamped experiment, enabling direct comparison.

---

## Project Structure

```
episteme-ai/
├── epai/
│   ├── fetch.py               # Stage 1: fetch papers from Semantic Scholar → SQLite
│   ├── summarise_db.py        # Stage 2: EDA on raw SQLite database
│   ├── prepare_data.py        # Stage 3: SQLite → Parquet, LLM rephrases & embeddings
│   ├── fetch_naip.py          # Stage 4: add NAIP impact scores to a Parquet file
│   ├── analyse_data.py        # Stage 5: EDA on processed Parquet dataset
│   ├── analyse_naip.py        # Stage 6: EDA on NAIP impact scores
│   ├── run_reg.py             # Stage 7a: regression — predict citation count
│   ├── run_repr.py            # Stage 7b: representation learning via autoencoder
│   ├── run_clf.py             # Stage 7c: classification — human vs. AI-generated abstract
│   ├── analyse_exp.py         # Stage 8a: per-algorithm plots aggregated across seeds
│   ├── analyse_exp_compare.py # Stage 8b: cross-algorithm comparison plots
│   ├── autoencoder.py         # AERegressor: PyTorch autoencoder + Ridge regression head
│   └── utils/
│       ├── config_utils.py       # Load and collect YAML experiment configs
│       ├── cuda_util.py          # Check CUDA availability
│       ├── dimred_utils.py       # PCA, KernelPCA, UMAP wrappers
│       ├── embedding_utils.py    # SPECTER2 / SciBERT / OpenAI embedding computation
│       ├── env_utils.py          # .env variable and installed-package checks
│       ├── genai_utils.py        # Multi-LLM abstract rephrasing (OpenAI, Anthropic, Gemini)
│       ├── parquet_utils.py      # Parquet save/load helpers
│       ├── s2_utils.py           # Semantic Scholar API client
│       ├── sqlite_utils.py       # SQLite schema, save, and load helpers
│       ├── train_utils.py        # Feature assembly and model factory
│       └── visualization_utils.py # Plotly and matplotlib plot functions
├── cfg/
│   ├── fetch/
│   │   ├── fetch_cfg_example.yaml   # Template: domain → query mapping
│   │   └── hd/                      # Fetch configs used in experiments
│   └── train/
│       ├── reg/
│       │   └── hd/                  # Regression experiment configs
│       ├── repr/
│       │   └── hd/                  # Representation learning configs
│       └── clf/
│           └── hd/                  # Classification experiment configs
├── store/                     # Versioned datasets tracked with Git LFS
│   └── data/
│       ├── parquet/           # Selected Parquet snapshots
│       └── sql/               # Selected SQLite snapshots
├── data/                      # Local working data (gitignored)
│   ├── sql/                   # SQLite databases (one per fetch run)
│   └── parquet/               # Parquet datasets
├── results/                   # All experiment outputs (gitignored)
│   ├── analyse_sql/           # summarise_db.py HTML charts
│   ├── analyse_data/          # analyse_data.py HTML charts
│   ├── naip/                  # analyse_naip.py figures
│   ├── reg/                   # Regression experiment outputs
│   ├── repr/                  # Representation learning outputs
│   └── clf/                   # Classification experiment outputs
├── .gitattributes             # Git LFS tracking rules (*.parquet, *.db)
├── pyproject.toml
└── .env_sample                # Template for API keys
```

---

## Key Concepts for Students

### Why log1p(citation_count)?

Citation counts follow a heavy-tailed distribution — a handful of papers accumulate tens of thousands of citations while most receive just a few. Training a regression model on raw counts causes the loss function to be dominated by extreme outliers. Applying `log1p(x) = log(1 + x)` compresses the scale and makes the target distribution more symmetric, which is much easier to model. `log1p` is used instead of plain `log` because it handles papers with zero citations gracefully (`log(0)` is undefined, `log1p(0) = 0`).

### What is a SPECTER embedding?

SPECTER and its successor SPECTER2 are transformer models trained specifically on the citation graph of academic papers. Given a paper's title and abstract as input, the model outputs a fixed-length 768-dimensional vector (an embedding) such that papers that frequently cite each other end up close together in this vector space. Using this vector as a feature lets a downstream ML model reason about a paper's topic and scientific community without manually engineering keyword or topic features.

### What does the autoencoder do?

The `AERegressor` in `epai/autoencoder.py` learns a **compressed representation** of the 768-dim input embeddings. The encoder network progressively reduces dimensionality (768 → 512 → 256 → 128); the decoder reconstructs the original vector from this 128-dim bottleneck. After training on reconstruction alone, the bottleneck layer (the latent space) is used as input to a simple Ridge regression that predicts `log1p(citation_count)`.

The motivation is that forcing the model to compress information may discard noise and retain only the structure that is predictive of citation impact. The reconstruction loss combines MSE (to match magnitudes) and cosine similarity (to match directions) so the latent space preserves both the magnitude and orientation of the original embeddings.

### What is the classification task testing?

When LLMs rephrase abstracts, they inevitably shift the text's semantic embedding even if the scientific meaning is preserved — different word choices produce slightly different SPECTER2 vectors. The classification task asks: **can a model distinguish human-written from LLM-rephrased abstracts using only their embeddings?** If yes, it suggests that LLM-generated scientific text carries a detectable stylistic signature at the embedding level — even when the content is identical to the human original.

### What is NAIP?

NAIP (Newborn Article Impact Predictor) is an external model hosted on HuggingFace that predicts the likely long-term impact of a paper from its title and abstract alone, without needing citation data. The `naip_final_score` (0–1) is the normalised TNCSI_sp score. Adding NAIP scores to the dataset lets you study whether LLM-rephrased abstracts are rated as more or less impactful than the human originals, and how NAIP scores correlate with actual citation counts.

### Train / Val / Test split

All training scripts use a 70 / 15 / 15 split (train / validation / test). The `StandardScaler` is always fitted on the **training set only** and then applied to validation and test without refitting — this is essential to prevent data leakage. Fitting the scaler on the full dataset would let the model peek at test-set statistics, producing overly optimistic results that would not generalise. Results are averaged across `n_runs` independent random seeds to give a more reliable estimate of model performance and its variance.

### Experiment IDs and reproducibility

Each training run generates a timestamped experiment ID (e.g. `reg_20260519_143022`). All config files, per-seed JSON result files, trained model `.joblib` files, and figures are saved under that ID. This makes it easy to go back and reproduce any individual run or compare results across different experiments.

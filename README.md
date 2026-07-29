# Episteme AI

A research platform for studying AI-generated text in scientific literature. The repository contains two independent but related pipelines that share a common library:

| Pipeline | Short name | Goal |
|---|---|---|
| **Citation Impact** | `epai/citation_impact/` | Fetch academic papers from Semantic Scholar, enrich them with LLM-generated synthetic variants, and train models to predict citation impact. |
| **AI Detection** | `epai/ai_detection/` | Prepare AI-vs-human text corpora (Ghostbuster / Denario), compute embeddings, and train classifiers that distinguish human from machine-generated text. |

Both pipelines draw on the shared library under `epai/utils/`, `epai/train/`, and `epai/analyse/`.

---

## Table of Contents

- [Repository Layout](#repository-layout)
- [Setup](#setup)
- [Git LFS — Store Folder](#git-lfs--store-folder)
- [Environment Variables](#environment-variables)
- [File Naming Convention](#file-naming-convention)
- [Pipeline 1 — Citation Impact](#pipeline-1--citation-impact)
  - [Stage 1 — Fetch](#stage-1--fetch)
  - [Stage 2 — Summarise DB](#stage-2--summarise-db)
  - [Stage 3 — Prepare Data](#stage-3--prepare-data)
  - [Stage 4 — Fetch NAIP Scores](#stage-4--fetch-naip-scores)
  - [Stage 5 — Analyse Data](#stage-5--analyse-data)
  - [Stage 6 — Analyse NAIP](#stage-6--analyse-naip)
  - [Stage 7 — Training (Regression / Repr Learning)](#stage-7--training-regression--repr-learning)
  - [Stage 8 — Analyse Experiments](#stage-8--analyse-experiments)
- [Pipeline 2 — AI Detection](#pipeline-2--ai-detection)
  - [Stage A — Prepare Corpora](#stage-a--prepare-corpora)
  - [Stage B — Embed](#stage-b--embed)
  - [Stage C — Analyse Embeddings / Binoculars](#stage-c--analyse-embeddings--binoculars)
  - [Stage D — Training (Classification)](#stage-d--training-classification)
- [Shared Library](#shared-library)
- [Configuration Reference](#configuration-reference)
- [Key Concepts for Students](#key-concepts-for-students)

---

## Repository Layout

```
episteme-ai/
├── epai/
│   ├── citation_impact/           # Pipeline 1 — Semantic Scholar / citation prediction
│   │   ├── fetch/fetch.py         #   Stage 1: Semantic Scholar → SQLite
│   │   ├── db/summarise_db.py     #   Stage 2: EDA on raw SQLite
│   │   ├── prepare/prepare_data.py#   Stage 3: SQLite → Parquet, LLM rephrasing
│   │   ├── score/add_naip_scores.py#  Stage 4: add NAIP impact scores
│   │   ├── analyse/               #   Stages 5–6: EDA on Parquet & NAIP scores
│   │   └── train/                 #   Stage 7: run_reg.py, run_repr.py
│   │
│   ├── ai_detection/              # Pipeline 2 — AI vs human text detection
│   │   ├── fetch/fetch_denario.py #   Download Denario papers
│   │   ├── prepare/               #   prepare_{reuter,wp,essay,denario}.py,
│   │   │                          #   recompute_embedding.py, merge_parquets.py
│   │   ├── embed/add_embeddings.py#   Add multiple embedding columns to a Parquet
│   │   ├── analyse/               #   Binoculars analysis, 2D embedding vis, corpus QA
│   │   └── train/                 #   run_clf.py, run_clf_cross_domain*.py,
│   │                              #   replot_cross_domain.py
│   │
│   ├── train/                     # Shared: algs/ (classifiers/regressors), autoencoder.py
│   ├── analyse/                   # Shared: analyse_exp.py, analyse_exp_compare.py
│   └── utils/                     # Shared: all utility modules
│
├── cfg/
│   ├── citation_impact/
│   │   ├── fetch/                 # Semantic Scholar query configs
│   │   └── train/reg/, repr/      # Regression / repr-learning experiment configs
│   └── ai_detection/
│       └── train/clf/, clf_cd/,   # Classification experiment configs
│                clf_cd_algs/, clf_cd_emb/
│
├── store/                         # Versioned datasets tracked with Git LFS
│   └── data/parquet/, sql/
├── data/                          # Local working data (gitignored)
│   ├── sql/                       # SQLite databases (citation impact pipeline)
│   ├── parquet/                   # Parquet datasets (both pipelines)
│   ├── ghostbuster-data*/         # Raw Ghostbuster text corpora
│   └── papers/                    # Downloaded Denario PDFs
└── results/                       # All experiment outputs (gitignored)
```

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install all dependencies
uv sync

# Install the epai package in editable mode
uv pip install -e .
```

For the compute server, add this at the end of your `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
torchaudio = { index = "pytorch-cu128" }
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
| `SEMANTIC_SCHOLAR_API_KEY` | `citation_impact/fetch/fetch.py` | Increases the Semantic Scholar API rate limit |
| `OPENAI_API_KEY` | `citation_impact/prepare/prepare_data.py`, `utils/embedding_utils.py` | GPT-4o-mini rephrasing and OpenAI embeddings |
| `ANTHROPIC_API_KEY` | `citation_impact/prepare/prepare_data.py` | Claude Haiku abstract rephrasing |
| `GOOGLE_API_KEY` | `citation_impact/prepare/prepare_data.py` | Gemini abstract rephrasing |
| `HF_TOKEN` | `citation_impact/score/add_naip_scores.py` | HuggingFace token for the NAIP Gradio API (optional but avoids rate limits) |

The Semantic Scholar key is optional but strongly recommended — without it you will be heavily rate-limited. The LLM keys are only needed if you enable the `ADD_GEN_AI` flag in `prepare_data.py`.

---

## File Naming Convention

Parquet files produced by the citation-impact pipeline follow a `<id>[_AI][_naip].parquet` pattern:

- `<id>` — a short experiment name, e.g. `0523_A_100`
- `_AI` suffix — the file contains both human and LLM-rephrased rows
- `_naip` suffix — NAIP scores have been added

Parquet files produced by the AI-detection pipeline follow a `<corpus>[_<variant>][_binox<N>].parquet` pattern:

- `<corpus>` — one of `reuter`, `wp`, `essay`, `denario_papers`
- `_<variant>` — e.g. `_gpt54mini` for the GPT-4o-mini variant
- `_binox<N>` — Binoculars scores have been added (N = run index)

---

## Pipeline 1 — Citation Impact

```
[fetch.py]
   ↓  data/sql/<id>.db
[summarise_db.py]              ← optional: EDA on raw database
   ↓
[prepare_data.py]
   ↓  data/parquet/<id>.parquet  or  <id>_AI.parquet
[add_naip_scores.py]           ← optional: add NAIP impact scores
   ↓  data/parquet/<id>_AI_naip.parquet
[analyse_data.py]              ← optional: EDA on processed Parquet
[analyse_naip.py]              ← optional: EDA on NAIP scores
   ↓
┌──────────────────────────────────────┐
│  Training (pick one or more)         │
│  run_reg.py   → results/reg/         │
│  run_repr.py  → results/repr/        │
└──────────────────────────────────────┘
   ↓
[analyse_exp.py + analyse_exp_compare.py]  ← called automatically after training
```

### Stage 1 — Fetch

**Script:** `epai/citation_impact/fetch/fetch.py`
**Config:** `cfg/citation_impact/fetch/hd/fetch_cfg_reduced.yaml`
**Output:** `data/sql/<id>.db`

```bash
python -m epai.citation_impact.fetch.fetch
```

Queries the Semantic Scholar bulk-search API for each (domain, query) pair defined in a fetch config YAML. For every paper found it stores title, abstract, publication year, venue, SPECTER embedding, citation count, influential citation count, and author metadata (name, h-index, citation count, affiliations).

Papers are stored in a local SQLite database with three tables: `papers`, `authors`, and `paper_authors`.

**Configuration** — edit the constants at the top of `fetch.py`:

```python
DB_PATH = DB_DIR / "sql" / "my_dataset.db"
CFG_PATH = BASE_DIR / "cfg" / "citation_impact" / "fetch" / "hd" / "fetch_cfg_reduced.yaml"
```

The fetch config maps domains to a list of search queries (see `cfg/citation_impact/fetch/fetch_cfg_example.yaml`). `TOPIC_LIMIT` controls how many papers to fetch per domain.

---

### Stage 2 — Summarise DB

**Script:** `epai/citation_impact/db/summarise_db.py`
**Output:** `results/analyse_sql/<id>/*.html`

```bash
python -m epai.citation_impact.db.summarise_db
```

An optional exploratory step. Loads the raw SQLite database, prints descriptive statistics, and saves interactive Plotly HTML charts (citation distributions, papers per year, author statistics, NULL percentages, overview dashboard).

Edit `DB_NAME` at the top of the script to point at your database.

---

### Stage 3 — Prepare Data

**Script:** `epai/citation_impact/prepare/prepare_data.py`
**Output:** `data/parquet/<id>.parquet` or `data/parquet/<id>_AI.parquet`

```bash
python -m epai.citation_impact.prepare.prepare_data
```

Converts the SQLite database to a Parquet file and optionally enriches it:

**3a — AI-generated synthetic abstracts (`ADD_GEN_AI = True`):** For each human-written paper, several LLMs each produce a rephrased version. Model/style combinations used by default:
- `gemini-2.5-flash-lite` / normal
- `claude-haiku-4-5-20251001` / normal
- `gpt-4o-mini` / normal
- `gpt-4o-mini` / aggressive

Each successful paper produces N+1 rows: one original (human) and one per model/style tagged with a `generated_by` column. This labelled dataset is what the citation-impact classification task uses.

**3b — Local scientific embeddings (`ADD_EMB_SPECTER2 = True`):** Computes fresh embeddings locally using [SPECTER2](https://huggingface.co/allenai/specter2_base). Required for the classification task because Semantic Scholar only provides embeddings for real papers.

---

### Stage 4 — Fetch NAIP Scores

**Script:** `epai/citation_impact/score/add_naip_scores.py`
**Output:** `data/parquet/<id>_AI_naip.parquet`

```bash
python -m epai.citation_impact.score.add_naip_scores
```

Queries the [NAIP](https://huggingface.co/spaces/ssocean/Newborn_Article_Impact_Predict) Gradio API on HuggingFace and adds `naip_raw_score` and `naip_final_score` (0–1) columns. Resume-safe: previously scored rows are skipped on re-run.

---

### Stage 5 — Analyse Data

**Script:** `epai/citation_impact/analyse/analyse_data.py`
**Output:** `results/analyse_data/<id>/`

```bash
python -m epai.citation_impact.analyse.analyse_data
```

EDA on the processed Parquet. Standard mode produces citation histograms, correlation plots, and 2D PCA/UMAP projections of SPECTER embeddings. AI-augmented mode (`MODE_AI = True`) produces 2D PCA colored by `generated_by` and pairwise cosine distances between human and AI abstract embeddings.

---

### Stage 6 — Analyse NAIP

**Script:** `epai/citation_impact/analyse/analyse_naip.py`
**Output:** `results/naip/<parquet_stem>/`

```bash
python -m epai.citation_impact.analyse.analyse_naip
```

EDA on NAIP-enriched datasets. Produces score distributions per `generated_by` group, mean ± std bar charts, and correlations between NAIP scores and citation counts.

---

### Stage 7 — Training (Regression / Repr Learning)

All training scripts follow the same pattern: load YAML configs → run N seeds → save results as JSON + model `.joblib` → auto-run analysis.

**Configs live in:** `cfg/citation_impact/train/`

#### Regression (`run_reg.py`)

**Output:** `results/reg/<experiment_id>/`

```bash
python -m epai.citation_impact.train.run_reg
```

Predicts `log1p(citation_count)` from paper features. Metrics per seed: RMSE and R² on train/val/test.

Available models (`ml_alg` in config): `gradient_boosting`, `linear`, `svm`, `mlp`, `ae`

Available features: `embedding`, `emb_specter2`, `year`, `best_author_h_index`, `author_count`, `page_count`

#### Representation Learning (`run_repr.py`)

**Output:** `results/repr/<experiment_id>/`

```bash
python -m epai.citation_impact.train.run_repr
```

Trains the `AERegressor` in `epai/train/autoencoder.py`: a deep autoencoder (768→512→256→128→256→512→768) whose 128-dim latent space is used as input to a Ridge regression head predicting `log1p(citation_count)`.

---

### Stage 8 — Analyse Experiments

These two scripts from `epai/analyse/` are called automatically at the end of each training script. They can also be run standalone.

- `analyse_exp.py` — per-algorithm plots (loss curves, feature importance, metrics summary, pred vs. gt scatter)
- `analyse_exp_compare.py` — cross-algorithm comparison (grouped bar charts, overlaid CI bands, feature importances)

---

## Pipeline 2 — AI Detection

```
[prepare_{reuter,wp,essay,denario}.py]   ← Stage A: raw text → Parquet
   ↓  data/parquet/<corpus>.parquet
[merge_parquets.py]                      ← optional: combine corpora
[recompute_embedding.py]                 ← optional: verify embedding stability
   ↓
[add_embeddings.py]                      ← Stage B: add embedding columns
   ↓  data/parquet/<corpus>_binox*.parquet
[analyse_binox_*.py]                     ← Stage C: EDA — Binoculars scores & embeddings
[analyse_emb_2d.py]
[analyse_parquet_*.py]
   ↓
┌────────────────────────────────────────────────────────────┐
│  Classification (pick one or more)                         │
│  run_clf.py                → results/clf/                  │
│  run_clf_cross_domain.py   → results/clf_cd/               │
│  run_clf_cross_domain_algs.py  → results/clf_cd_algs/      │
│  run_clf_cross_domain_embeddings.py → results/clf_cd_emb/  │
└────────────────────────────────────────────────────────────┘
```

### Stage A — Prepare Corpora

All prepare scripts read raw text files from `data/ghostbuster-data*/` (or `data/papers/denario/`) and produce a structured Parquet with columns: `id`, `text`, `generated_by`, embedding columns.

```bash
python -m epai.ai_detection.prepare.prepare_reuter
python -m epai.ai_detection.prepare.prepare_wp
python -m epai.ai_detection.prepare.prepare_essay
python -m epai.ai_detection.prepare.prepare_denario
```

To fetch the Denario papers first:
```bash
python epai/ai_detection/fetch/fetch_denario.py
```

To combine multiple Parquets into one:
```bash
python -m epai.ai_detection.prepare.merge_parquets data/parquet/a.parquet data/parquet/b.parquet
```

**Configs live in:** `cfg/ai_detection/train/`

---

### Stage B — Embed

**Script:** `epai/ai_detection/embed/add_embeddings.py`
**Output:** overwrites / extends the input Parquet with new `emb_*` columns

```bash
uv run python epai/ai_detection/embed/add_embeddings.py data/parquet/reuter_binox0.parquet --cpu-threads 16 --gpu 0
```

Adds multiple embedding columns in one pass. Supported embedders (subset, see `EMBEDDERS` list in the script):

| Key | Model |
|---|---|
| `specter2` | SPECTER2 (scientific) |
| `scibert` | SciBERT |
| `mpnet` | all-mpnet-base-v2 |
| `minilm` | all-MiniLM-L6-v2 |
| `e5` | E5 |
| `bge` | BGE |
| `qwen3` | Qwen3-Embedding-0.6B |
| `luar` / `luar_crud` | LUAR (authorship) |
| `styledistance` | StyleDistance |
| `openai_small` / `openai_large` | OpenAI text-embedding-3-* |

The Parquet is written after every embedder finishes — a crash does not throw away completed embeddings.

---

### Stage C — Analyse Embeddings / Binoculars

```bash
# 2D UMAP/PCA of embeddings coloured by generated_by
python -m epai.ai_detection.analyse.analyse_emb_2d data/parquet/reuter_binox0.parquet --by-generator

# Binoculars human vs. AI per-id score differences (violins, histograms, ROC/PR, confusion)
uv run python epai/ai_detection/analyse/analyse_binox_human_vs_ai.py data/parquet/wp_binox0.parquet

# ...the same analysis on the LLM-judge scores instead
uv run python epai/ai_detection/analyse/analyse_binox_human_vs_ai.py \
    data/parquet/wp_binox0.parquet:gpt_plain --metric llm_judge

# Pairwise comparison between two Binoculars runs
uv run python epai/ai_detection/analyse/analyse_binox_pairwise.py

# Corpus QA (row/column counts, missing values)
uv run python epai/ai_detection/analyse/analyse_parquet_compare.py data/parquet/a.parquet data/parquet/b.parquet
uv run python epai/ai_detection/analyse/analyse_parquet_summary.py data/parquet/reuter_binox0.parquet

# Text length distributions
uv run python epai/ai_detection/analyse/analyse_text_lengths.py

# Count and flag empty .txt files in a raw-data folder
uv run python epai/ai_detection/analyse/analyse_txt_folder.py data/ghostbuster-data/essay
```

---

### Stage D — Training (Classification)

#### In-domain classification (`run_clf.py`)

**Output:** `results/clf/<experiment_id>/`

```bash
python -m epai.ai_detection.train.run_clf
```

Classifies whether a text is human-written or generated by a specific model/style based on the `generated_by` label column and `emb_specter2` embeddings. Metrics: accuracy and per-class F1. Splits are stratified.

#### Cross-domain transfer control (`run_clf_cross_domain.py`)

**Output:** `results/clf_cd/<experiment_id>/`

```bash
python -m epai.ai_detection.train.run_clf_cross_domain
```

Trains on each corpus in turn and evaluates on all of them, producing a source × target accuracy matrix. The off-diagonal shows how well the classifier transfers to unseen corpora — distinguishing "learned machine-generated-ness" from "learned corpus artifacts".

#### Cross-domain × algorithm sweep (`run_clf_cross_domain_algs.py`)

**Output:** `results/clf_cd_algs/<experiment_id>/`

```bash
python -m epai.ai_detection.train.run_clf_cross_domain_algs
```

Same as the cross-domain control but sweeps across ML algorithms (logistic, SVM, gradient boosting, MLP, Binoculars-MLP) on one fixed feature set. Each algorithm gets its own matrix and a comparison ranking.

#### Cross-domain × embedding sweep (`run_clf_cross_domain_embeddings.py`)

**Output:** `results/clf_cd_emb/<experiment_id>/`

```bash
python -m epai.ai_detection.train.run_clf_cross_domain_embeddings
```

Same as the cross-domain control but sweeps across all embedding columns in the config. Each embedder gets its own matrix and a final comparison figure. Includes a zero-shot Binoculars baseline.

To regenerate figures from saved results without retraining:
```bash
uv run python epai/ai_detection/train/replot_cross_domain.py results/clf_cd_algs
uv run python epai/ai_detection/train/replot_cross_domain.py results/clf_cd_emb
```

---

## Shared Library

The `epai/utils/`, `epai/train/`, and `epai/analyse/` modules are used by both pipelines and should not be modified for pipeline-specific purposes.

| Module | Purpose |
|---|---|
| `epai/utils/config_utils.py` | Load and collect YAML experiment configs |
| `epai/utils/cuda_util.py` | Check CUDA availability |
| `epai/utils/dimred_utils.py` | PCA, KernelPCA, UMAP wrappers |
| `epai/utils/embedding_utils.py` | SPECTER2 / SciBERT / OpenAI embedding computation |
| `epai/utils/env_utils.py` | `.env` variable and installed-package checks |
| `epai/utils/genai_utils.py` | Multi-LLM abstract rephrasing (OpenAI, Anthropic, Gemini) |
| `epai/utils/parquet_utils.py` | Parquet save/load helpers |
| `epai/utils/s2_utils.py` | Semantic Scholar API client |
| `epai/utils/sqlite_utils.py` | SQLite schema, save, and load helpers |
| `epai/utils/train_utils.py` | Feature assembly and model factory |
| `epai/utils/visualization_utils.py` | Plotly and matplotlib plot functions |
| `epai/train/algs/` | Pluggable classifier/regressor subclasses |
| `epai/train/autoencoder.py` | `AERegressor`: PyTorch autoencoder + Ridge regression head |
| `epai/analyse/analyse_exp.py` | Per-algorithm plots (lib, called by training scripts) |
| `epai/analyse/analyse_exp_compare.py` | Cross-algorithm comparison plots (lib) |

---

## Configuration Reference

### Fetch config (`cfg/citation_impact/fetch/fetch_cfg_example.yaml`)

```yaml
queries_domains:
  Physics:
    - "quantum mechanics particle physics"
    - "condensed matter solid state physics"
  Computer Science:
    - "machine learning deep learning"
    - "natural language processing large language models"
```

### Regression / repr-learning configs (`cfg/citation_impact/train/`)

```yaml
alg_id: "A1"
alg_label: "SPECTER + year + h-index"
seed: 42
data:
  parquet_path: "data/parquet/my_dataset.parquet"
features:
  - embedding
  - year
  - best_author_h_index
model:
  ml_alg: "gradient_boosting"
  n_runs: 3
```

### Classification configs (`cfg/ai_detection/train/clf/`, `clf_cd/`, etc.)

```yaml
alg_id: "A1"
alg_label: "SPECTER2"
seed: 42
data:
  parquet_path: "data/parquet/reuter_binox0.parquet"
  drop_classes: []
features:
  - emb_specter2
model:
  ml_alg: "logistic"
  n_runs: 3
```

---

## Key Concepts for Students

### Why log1p(citation_count)?

Citation counts follow a heavy-tailed distribution — a handful of papers accumulate tens of thousands of citations while most receive just a few. Training a regression model on raw counts causes the loss to be dominated by outliers. `log1p(x) = log(1 + x)` compresses the scale and handles zero-citation papers gracefully (`log(0)` is undefined, `log1p(0) = 0`).

### What is a SPECTER embedding?

SPECTER and SPECTER2 are transformer models trained on the citation graph of academic papers. Given a paper's title and abstract, the model outputs a 768-dimensional vector such that papers that frequently cite each other end up close together. Using this as a feature lets downstream ML models reason about a paper's topic and scientific community without manual feature engineering.

### What does the autoencoder do?

The `AERegressor` in `epai/train/autoencoder.py` learns a **compressed representation** of 768-dim input embeddings. The encoder progressively reduces dimensionality (768→512→256→128); the decoder reconstructs the original from this 128-dim bottleneck. After training on reconstruction alone, the bottleneck layer is used as input to a Ridge regression head that predicts `log1p(citation_count)`.

### What is the citation-impact classification task testing?

When LLMs rephrase abstracts, they shift the text's semantic embedding even if the scientific meaning is preserved. The classification task asks: **can a model distinguish human-written from LLM-rephrased abstracts using only their embeddings?** If yes, LLM-generated scientific text carries a detectable stylistic signature at the embedding level.

### What is the AI-detection cross-domain task testing?

In-domain accuracy for the classification task is very high (~97–99%). The open question is what signal the classifier has actually learned: **machine-generated-ness** (which should transfer to unseen corpora) vs. **corpus artifacts** (which would collapse to ~50% under transfer). The cross-domain scripts produce a source × target matrix to answer this.

### What is NAIP?

NAIP (Newborn Article Impact Predictor) is an external model on HuggingFace that predicts the likely long-term impact of a paper from its title and abstract alone. The `naip_final_score` (0–1) is the normalised TNCSI_sp score. Adding NAIP scores lets you study whether LLM-rephrased abstracts are rated as more or less impactful than the human originals.

### Train / Val / Test split

All training scripts use a 70 / 15 / 15 split. The `StandardScaler` is always fitted on the **training set only** and then applied to validation and test — this prevents data leakage. Results are averaged across `n_runs` independent random seeds.

### Experiment IDs and reproducibility

Each training run generates a timestamped experiment ID (e.g. `reg_20260519_143022`). All config files, per-seed JSON result files, trained model `.joblib` files, and figures are saved under that ID.

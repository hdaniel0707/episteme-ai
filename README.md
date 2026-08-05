# Episteme AI

A research platform for studying AI-generated text in scientific literature. The repository contains two independent but related pipelines that share a common library:

| Pipeline | Short name | Goal |
|---|---|---|
| **Citation Impact** | `epai/citation_impact/` | Fetch academic papers from Semantic Scholar and enrich them with LLM-generated synthetic variants and NAIP impact scores. Data pipeline only — its training stage has been removed. |
| **AI Detection** | `epai/ai_detection/` | Prepare AI-vs-human text corpora (Ghostbuster / Denario), compute embeddings and detector scores, and train classifiers that distinguish human from machine-generated text. The active pipeline. |
| **Science Data** | `epai/science_data/` | Build a *new* AI-vs-human corpus from Semantic Scholar abstracts under the Ghostbuster protocol: fetch → filter → prompt → generate. The generator sees only the prompt, never the human text. |

All three draw on the shared library under `epai/utils/` and `epai/algs/`.

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
- [Pipeline 2 — AI Detection](#pipeline-2--ai-detection)
  - [Stage A — Prepare Corpora](#stage-a--prepare-corpora)
  - [Stage B — Embed and Score](#stage-b--embed-and-score)
  - [Stage C — Combine the corpora into one file](#stage-c--combine-the-corpora-into-one-file)
  - [Stage D — Analyse](#stage-d--analyse)
  - [Stage E — Training (Cross-domain classification)](#stage-e--training-cross-domain-classification)
- [Pipeline 3 — Science Data](#pipeline-3--science-data)
- [Shared Library](#shared-library)
- [Configuration Reference](#configuration-reference)

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
│   │   └── analyse/               #   Stages 5–6: EDA on Parquet & NAIP scores
│   │
│   ├── ai_detection/              # Pipeline 2 — AI vs human text detection
│   │   ├── fetch/fetch_denario.py #   Download Denario papers
│   │   ├── prepare/               #   prepare_{reuter,wp,essay,denario}.py,
│   │   │                          #   combine_domains.py, normalize_text.py,
│   │   │                          #   truncate_text.py
│   │   ├── embed/add_embeddings.py#   Add multiple embedding columns to a Parquet
│   │   ├── llm_judge/             #   LLM-as-judge scores
│   │   ├── analyse/               #   PCA/geometry, surface artifacts, score EDA
│   │   └── train/                 #   run_clf_cross_domain*.py, sweep_*.py,
│   │                              #   replot_cross_domain.py
│   │
│   ├── science_data/              # Pipeline 3 — build a new corpus, Ghostbuster protocol
│   │   ├── fetch/fetch_s2.py      #   Stage 1: Semantic Scholar → raw parquet
│   │   ├── filter/                #   Stage 2: admission rules → N per domain
│   │   ├── prompt/make_prompts.py #   Stage 3: one generation prompt per paper
│   │   ├── generate/generate_ai.py#   Stage 4: prompt → AI abstract
│   │   └── analyse/               #   per-stage reports
│   │
│   ├── algs/                      # Shared: pluggable classifier algorithms
│   └── utils/                     # Shared: all utility modules
│
├── cfg/
│   ├── citation_impact/
│   │   └── fetch/                 # Semantic Scholar query configs
│   ├── ai_detection/
│   │   └── train/clf_cd/,         # Cross-domain experiment configs
│   │       clf_cd_algs/,          #   (clf_cd/ghostbuster/ = the combined-parquet set)
│   │       clf_cd_emb/, sweep/
│   └── science_data/              # Corpus-building config + query lists
│
├── store/                         # Versioned datasets tracked with Git LFS
│   └── data/parquet/, sql/
├── data/                          # Local working data (gitignored)
│   ├── sql/                       # SQLite databases (citation impact pipeline)
│   ├── parquet/                   # Parquet datasets (citation impact + AI detection)
│   ├── science/                   # Science-data corpus + per-domain shards
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

Parquet files produced by the AI-detection pipeline come in two shapes.

Per-corpus files, `<corpus>[_<variant>][_binox<N>][_norm|_red_<len>].parquet`:

- `<corpus>` — one of `reuter`, `wp`, `essay`, `denario_papers`
- `_<variant>` — e.g. `_gpt54mini` for the GPT-4o-mini regeneration
- `_binox<N>` — Binoculars scores have been added (N = run index)
- `_norm` — text normalised by `prepare/normalize_text.py`
- `_red_<len>[w]` — text truncated by `prepare/truncate_text.py`

Combined files, `ghostbuster_<generator>.parquet`, written by
`prepare/combine_domains.py`: several corpora in one file, one per `domain`.
This is what the training and analysis scripts read now — a *corpus* is a
`(file, domain)` pair rather than a file.

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
```

> **Note:** the training stage of this pipeline (`run_reg.py`, `run_repr.py`) and its
> plotting library (`epai/analyse/`) have been removed. What remains is the data
> pipeline: fetch → summarise → prepare → score → analyse.

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

## Pipeline 2 — AI Detection

```
[prepare_{reuter,wp,essay,denario}.py]   ← Stage A: raw text → Parquet (one per corpus)
   ↓  data/parquet/<corpus>.parquet
[add_embeddings.py]                      ← Stage B: add emb_* columns
[add_llm_judge_scores.py]                ←          add llm_judge_score_* columns
   ↓  data/parquet/<corpus>_binox*.parquet
[combine_domains.py]                     ← Stage C: the per-corpus files → ONE file
   ↓  data/parquet/ghostbuster_<generator>.parquet   (uid, id, domain, text, …)
[analyse_domain_pca.py]                  ← Stage D: EDA on the combined file
[analyse_emb_class_stats.py] [analyse_surface_artifacts.py]
[analyse_text_lengths.py]    [analyse_score_human_vs_ai.py]
   ↓
┌────────────────────────────────────────────────────────────┐
│  Cross-domain classification (pick one)                    │
│  run_clf_cross_domain.py            → results/clf_cd/      │
│  run_clf_cross_domain_algs.py       → results/clf_cd_algs/ │
│  run_clf_cross_domain_embeddings.py → results/clf_cd_emb/  │
│  sweep_clf_cross_domain.py          → results/sweep_clf_cd/│
└────────────────────────────────────────────────────────────┘
```

**One parquet, one dataloader, one preflight.** A *corpus* is no longer a file: it
is a `CorpusSpec` — a parquet plus an optional `domain` — defined in
`epai/utils/corpus_utils.py` and read by `read_corpus()`. Every training and
analysis script goes through it, so essay/reuter/wp are three slices of one
combined file rather than three paths repeated in every config. A dataset entry
with no `domain` key means the whole file, so configs written against the old
per-corpus parquets still run unchanged.

Every script is a normal CLI with defaults — no module-level constants to edit
between runs:

```bash
uv run python epai/ai_detection/train/run_clf_cross_domain.py                  # defaults
uv run python epai/ai_detection/train/run_clf_cross_domain.py --cfg <one.yaml> # one config
uv run python epai/ai_detection/train/run_clf_cross_domain.py --cfg-dir <dir>  # a folder
```

### Stage A — Prepare Corpora

All prepare scripts read raw text files from `data/ghostbuster-data*/` (or
`data/papers/denario/`) and produce a Parquet with `id`, `text`, `generated_by`
and an `emb_specter2` column.

```bash
uv run python epai/ai_detection/prepare/prepare_reuter.py
uv run python epai/ai_detection/prepare/prepare_wp.py
uv run python epai/ai_detection/prepare/prepare_essay.py
uv run python epai/ai_detection/prepare/prepare_denario.py

# any source directory / output path, and a fast smoke run
uv run python epai/ai_detection/prepare/prepare_essay.py \
    --source-dir data/ghostbuster-data/essay -o data/parquet/essay.parquet \
    --limit 20 --no-embedding
```

To fetch the Denario papers first:
```bash
uv run python epai/ai_detection/fetch/fetch_denario.py
```

---

### Stage B — Embed and Score

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

```bash
# LLM-as-judge P(human) per row
uv run python epai/ai_detection/llm_judge/add_llm_judge_scores.py data/parquet/reuter_binox0.parquet
```

---

### Stage C — Combine the corpora into one file

**Script:** `epai/ai_detection/prepare/combine_domains.py`

```bash
uv run python epai/ai_detection/prepare/combine_domains.py \
    data/parquet/essay_binox0.parquet \
    data/parquet/reuter_binox0.parquet \
    data/parquet/wp_binox0.parquet \
    -o data/parquet/ghostbuster_gpt35t.parquet
```

Output schema: `uid`, `id`, `domain`, `text`, `generated_by`, `binoculars_score`,
`llm_judge_score*`, `emb_*`. Two keys with two jobs — **`uid` identifies a row,
`id` identifies a pair** (a source prompt and every rewrite of it share one), so
every split groups on `id` and the two halves of a pair never straddle a fold.
The domain is read back off the id prefix, `--drop-generated-by` removes the
prompt-variant generations, and `--drop-incomplete` removes rows with no
embedding. The run ends with a statistics block.

Text transforms that produce new corpora:
```bash
uv run python epai/ai_detection/prepare/normalize_text.py <parquet>...   # strip surface artifacts
uv run python epai/ai_detection/prepare/truncate_text.py <parquet> -n 200 --unit words
```

---

### Stage D — Analyse

Every analysis takes parquets positionally and `--domains` to pick slices of a
combined file; with no `--domains` a combined file becomes one corpus per domain
it holds.

```bash
# PCA in 2D and 3D (PNG + interactive HTML) plus the corpus and geometry statistics
uv run python epai/ai_detection/analyse/analyse_domain_pca.py \
    data/parquet/ghostbuster_gpt35t.parquet --feature emb_luar_crud

# class geometry: how much of the in-domain score is one direction?
uv run python epai/ai_detection/analyse/analyse_emb_class_stats.py \
    data/parquet/ghostbuster_gpt54mini.parquet

# what a classifier could learn from the raw text alone, with the sign per corpus
uv run python epai/ai_detection/analyse/analyse_surface_artifacts.py \
    --config cfg/ai_detection/train/clf_cd/ghostbuster/ALL_logistic.yaml

# detector scores, human vs AI: violins, ROC/PR, confusion
uv run python epai/ai_detection/analyse/analyse_score_human_vs_ai.py \
    data/parquet/ghostbuster_gpt35t.parquet:gpt

# text length distributions per domain
uv run python epai/ai_detection/analyse/analyse_text_lengths.py \
    data/parquet/ghostbuster_gpt35t.parquet

# corpus QA
uv run python epai/ai_detection/embed/summary_parquet.py data/parquet/ghostbuster_gpt35t.parquet
uv run python epai/ai_detection/analyse/analyse_txt_folder.py data/ghostbuster-data/essay
```

---

### Stage E — Training (Cross-domain classification)

Every runner shares the same three arguments — `--cfg`, `--cfg-dir`,
`--results-dir` — and the same preflight: every config's shape, every parquet,
every column and every `domain` is checked before the first model is fitted, and
all problems are reported together.

#### Cross-domain transfer control (`run_clf_cross_domain.py`)

**Configs:** `cfg/ai_detection/train/clf_cd/ghostbuster/` · **Output:** `results/clf_cd/<experiment_id>/`

```bash
uv run python epai/ai_detection/train/run_clf_cross_domain.py
uv run python epai/ai_detection/train/run_clf_cross_domain.py \
    --cfg cfg/ai_detection/train/clf_cd/ghostbuster/ALL_logistic.yaml
```

Trains on each corpus in turn and evaluates on all of them, producing a source ×
target accuracy matrix. The off-diagonal shows how well the classifier transfers
to unseen corpora — distinguishing "learned machine-generated-ness" from
"learned corpus artifacts".

#### Cross-domain × algorithm sweep (`run_clf_cross_domain_algs.py`)

**Configs:** `cfg/ai_detection/train/clf_cd_algs/` · **Output:** `results/clf_cd_algs/<experiment_id>/`

```bash
uv run python epai/ai_detection/train/run_clf_cross_domain_algs.py --dry-run   # preflight only
uv run python epai/ai_detection/train/run_clf_cross_domain_algs.py
```

Same matrix, swept across ML algorithms (logistic, SVM, gradient boosting, MLP,
Binoculars-MLP) on one fixed feature set, plus zero-shot Binoculars / LLM-judge
baselines. Each algorithm gets its own matrix and a comparison ranking.

#### Cross-domain × embedding sweep (`run_clf_cross_domain_embeddings.py`)

**Configs:** `cfg/ai_detection/train/clf_cd_emb/` · **Output:** `results/clf_cd_emb/<experiment_id>/`

```bash
uv run python epai/ai_detection/train/run_clf_cross_domain_embeddings.py
```

Same matrix, swept across every embedding column in the config. Each embedder
gets its own matrix and a final comparison figure.

#### Hyper-parameter search (`sweep_clf_cross_domain.py`)

**Configs:** `cfg/ai_detection/train/sweep/` · **Output:** `results/sweep_clf_cd/<sweep_id>/`

```bash
uv run python epai/ai_detection/train/sweep_clf_cross_domain.py --cfg cfg/ai_detection/train/sweep/luar_mlp_reg.yaml
uv run python epai/ai_detection/train/sweep_clf_cross_domain.py --report   # leaderboard
uv run python epai/ai_detection/train/sweep_clf_cross_domain.py --final    # winner on the full matrix
```

To regenerate figures from saved results without retraining:
```bash
uv run python epai/ai_detection/train/replot_cross_domain.py results/clf_cd_algs
uv run python epai/ai_detection/train/replot_cross_domain.py results/clf_cd_emb
```

---

## Pipeline 3 — Science Data

A newer corpus builder: instead of reusing Ghostbuster's essays and news, it
assembles an AI-vs-human corpus of *scientific abstracts*. The protocol is the
part that matters — each generator sees only a prompt derived from the paper,
never the human abstract, so the AI class does not inherit the human text's
content and structure the way a rephrasing setup does.

```
[fetch_s2.py]        ← Stage 1: Semantic Scholar → data/science/<id>_raw.parquet
   ↓                            (stops on the POST-filter count, per domain)
[filter_abstracts.py]← Stage 2: admission rules → exactly N per domain
   ↓
[make_prompts.py]    ← Stage 3: one generation prompt per paper
   ↓
[generate_ai.py]     ← Stage 4: prompt → AI abstract, checkpointed per call
   ↓  data/science/<id>.parquet
[analyse_{fetch,filter,prompts,generation}.py]   ← per-stage reports
```

All five stages read one config (`cfg/science_data/science_cfg_example.yaml`),
with the domain/query lists in a separate file so they can be swapped
independently:

```bash
uv run python -m epai.science_data.fetch.fetch_s2 \
    --cfg cfg/science_data/science_cfg_example.yaml \
    --queries cfg/science_data/queries_example.yaml
uv run python -m epai.science_data.filter.filter_abstracts --cfg cfg/science_data/science_cfg_example.yaml
uv run python -m epai.science_data.prompt.make_prompts    --cfg cfg/science_data/science_cfg_example.yaml
uv run python -m epai.science_data.generate.generate_ai   --cfg cfg/science_data/science_cfg_example.yaml
uv run python -m epai.science_data.analyse.analyse_fetch  --cfg cfg/science_data/science_cfg_example.yaml
```

Once exported, the corpus joins the AI-detection pipeline at Stage B (embed) —
and `prepare/combine_domains.py` can fold it in beside the Ghostbuster domains.

---

## Shared Library

The `epai/utils/` and `epai/algs/` modules are shared and should not be modified for pipeline-specific purposes.

| Module | Purpose |
|---|---|
| `epai/utils/config_utils.py` | Load and collect YAML experiment configs; the shared `--cfg` / `--cfg-dir` / `--results-dir` CLI arguments |
| `epai/utils/corpus_utils.py` | **The one dataloader.** `CorpusSpec` (a parquet + an optional `domain`), `read_corpus`, and the dataset/file/domain preflight checks |
| `epai/utils/preflight_utils.py` | **The one preflight.** `Report`, `print_report`, and the config-shape checks every runner shares |
| `epai/utils/cuda_util.py` | Check CUDA availability |
| `epai/utils/dimred_utils.py` | PCA, KernelPCA, UMAP wrappers |
| `epai/utils/embedding_utils.py` | SPECTER2 / SciBERT / OpenAI embedding computation |
| `epai/utils/env_utils.py` | `.env` variable and installed-package checks |
| `epai/utils/genai_utils.py` | Multi-LLM abstract rephrasing (OpenAI, Anthropic, Gemini) |
| `epai/utils/parquet_utils.py` | Parquet save/load helpers |
| `epai/utils/s2_utils.py` | Semantic Scholar API client |
| `epai/utils/sqlite_utils.py` | SQLite schema, save, and load helpers |
| `epai/utils/train_utils.py` | Missing-value rules (`drop_missing_rows`), feature stacking, and the classifier factory (`build_clf_model`) |
| `epai/utils/visualization_utils.py` | Plotly and matplotlib plot functions |
| `epai/algs/` | Pluggable `ClfAlgorithm` subclasses (logistic, svm, gradient_boosting, simple_mlp, binox_mlp), each owning its own preprocessing and hyper-parameters |

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

### Science-data config (`cfg/science_data/science_cfg_example.yaml`)

One annotated YAML drives all five stages of pipeline 3, with the domain and
query lists in `queries_example.yaml` so they can be swapped independently.

### Classification configs (`cfg/ai_detection/train/clf_cd/`, `clf_cd_algs/`, `clf_cd_emb/`, `sweep/`)

```yaml
alg_id: "A1"                 # names the output directory under results/<family>/
alg_label: "SPECTER2"        # shown on the figures
seed: 42
data:
  human_label: "human"       # everything else binarises to "machine"
  group_col: "id"            # rows sharing this never straddle a split
  parquet_path: "data/parquet/ghostbuster_gpt35t.parquet"   # default for every dataset
  datasets:                  # one entry per corpus; every one is trained on in turn
    - name: "essay"
      label: "essay"
      domain: "essay"        # one domain of the combined file
    - name: "reuter"
      label: "reuter"
      domain: "reuter"
features:                    # `embeddings:` (a list) in the clf_cd_emb configs
  - emb_specter2
model:
  ml_alg: "logistic"         # gradient_boosting | logistic | svm | mlp
  n_runs: 3
  test_size: 0.30
```

`parquet_path` (and `drop_classes`) may be set once under `data`, as above, or
per dataset — an entry that names its own wins. Set it once when every corpus is
a domain of one file; name it per entry when the matrix spans several files, as
`clf_cd/ghostbuster/ALL_logistic.yaml` does with two generators.

Omit `domain` to use the whole file — that is what a per-corpus parquet holds, so
configs written before `combine_domains.py` still run. A `domain` that is not in
the file is a preflight error listing the ones that are.

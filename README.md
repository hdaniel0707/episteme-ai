# Episteme AI

A research platform for studying AI-generated text in scientific literature. The repository contains two independent but related pipelines that share a common library:

| Pipeline | Short name | Goal |
|---|---|---|
| **Citation Impact** | `epai/citation_impact/` | Fetch academic papers from Semantic Scholar and enrich them with LLM-generated synthetic variants and NAIP impact scores. Data pipeline only — its training stage has been removed. |
| **AI Detection** | `epai/ai_detection/` | Prepare AI-vs-human text corpora (Ghostbuster / Denario), compute embeddings and detector scores, and train classifiers that distinguish human from machine-generated text. The active pipeline. |
| **Science Data** | `epai/science_data/` | Build a *new* AI-vs-human corpus from OpenAlex abstracts under the Ghostbuster protocol: fetch → filter → prompt → generate. The generator sees only the prompt, never the human text. |
| **Ghostbuster** | `epai/ghostbuster/` | Rebuild the *original* Ghostbuster corpora (essay / wp / reuter) with a newer generator, then parquet them and score them. Drives the ghostbuster fork as a submodule; the human text and prompts are held fixed so two model variants are comparable. |

All four draw on the shared library under `epai/utils/` and `epai/algs/`.

---

## Table of Contents

- [Repository Layout](#repository-layout)
- [Setup](#setup)
  - [Clean clone](#clean-clone)
  - [CUDA wheels — compute server only](#cuda-wheels--compute-server-only)
  - [Environments](#environments)
  - [Updating a submodule](#updating-a-submodule)
  - [Claude Code notifications — optional](#claude-code-notifications--optional)
  - [Pipeline notifications — optional](#pipeline-notifications--optional)
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
- [Pipeline 4 — Ghostbuster with a newer generator](#pipeline-4--ghostbuster-with-a-newer-generator)
  - [Seed the shared inputs first](#seed-the-shared-inputs-first)
  - [Running it](#running-it)
  - [The collect stage is the gate](#the-collect-stage-is-the-gate)
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
│   │   ├── score/                 #   run_binox_pairs.py (Binoculars pairs),
│   │   │                          #   add_llm_judge_scores.py (LLM-as-judge)
│   │   ├── analyse/               #   PCA/geometry, surface artifacts, score EDA
│   │   └── train/                 #   run_clf_cross_domain.py + replot_cross_domain.py,
│   │                              #   run_clf_in_domain.py    + replot_in_domain.py,
│   │                              #   run_sweep_cross_domain.py + replot_sweep.py
│   │
│   ├── science_data/              # Pipeline 3 — build a new corpus, Ghostbuster protocol
│   │   ├── fetch/oa_fetch.py      #   Stage 1: OpenAlex → raw parquet
│   │   ├── filter/filter.py       #   Stage 2: admission rules → N per domain
│   │   ├── prompt/make_prompts.py #   Stage 3: one generation prompt per paper
│   │   ├── generate/generate_ai.py#   Stage 4: prompt → AI abstract
│   │   ├── analyse/               #   per-stage reports
│   │   ├── run_pipeline.py        #   Runs every stage above, in order
│   │   └── run_full_pipeline.py   #   ...then promote → score → embed
│   │
│   ├── ghostbuster/               # Pipeline 4 — Ghostbuster corpora, newer generator
│   │   ├── config.py              #   The one config: validation, derived paths
│   │   └── run_full_pipeline.py   #   generate → fix-empty → collect → prepare
│   │                              #   → combine → score, in order
│   │
│   ├── scripts/                   # Standalone parquet tools, pipeline-agnostic
│   │   ├── export_records_txt.py  #   Parquet → one readable .txt per record
│   │   └── rename_column_values.py#   Relabel values inside one column
│   │
│   ├── algs/                      # Shared: pluggable classifier algorithms
│   └── utils/                     # Shared: all utility modules
│
├── external/
│   ├── Binoculars/                # Git submodule: the Binoculars fork, own uv env
│   └── ghostbuster/               # Git submodule: the ghostbuster fork, own uv env
│       └── data/                  #   gitignored; seeded by hand, written by generate.py
│
├── cfg/
│   ├── citation_impact/
│   │   └── fetch/                 # Semantic Scholar query configs
│   ├── ai_detection/
│   │   └── train/clf_cd/,         # One config each, annotated sample and working
│   │       train/clf_lodo/,       #   experiment at once — the three training
│   │       train/clf_id/,         #   designs (cross / lodo / in-domain),
│   │       train/sweep/           #   plus sweep_example.yaml
│   ├── science_data/              # Corpus-building config + query lists
│   │   └── prompts/               #   Every prompt sent, one file per prompt_config id
│   └── ghostbuster/               # One config for the whole ghostbuster pipeline
│
├── store/                         # Versioned datasets tracked with Git LFS
│   └── data/parquet/, sql/
├── data/                          # Local working data (gitignored)
│   ├── sql/                       # SQLite databases (citation impact pipeline)
│   ├── parquet/                   # Parquet datasets (citation impact + AI detection)
│   ├── science/                   # Science-data pipeline: ALL stages read/write here
│   ├── inspect/                   # export_records_txt.py output, one dir per corpus
│   ├── ghostbuster-data*/         # Raw Ghostbuster text corpora, one dir per variant
│   └── papers/                    # Downloaded Denario PDFs
└── results/                       # All experiment outputs (gitignored)
```

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and needs Python 3.12 or newer.

### Clean clone

Two forks live here as git submodules — Binoculars (scoring) and ghostbuster (generation) — so
clone with `--recurse-submodules`:

```bash
git clone --recurse-submodules git@github.com:hdaniel0707/episteme-ai.git
cd episteme-ai

# If you cloned without the flag, or a submodule is empty:
git submodule update --init
```

The submodules are cloned over SSH, so the machine needs SSH access to them as well as to this one.
Datasets under `store/` are Git LFS pointers after a fresh clone — see
[Git LFS — Store Folder](#git-lfs--store-folder) to pull the real files.

### CUDA wheels — compute server only

**Only for the compute server. Otherwise, skip this section entirely**

On the compute server, do this **before** the `uv sync` commands in the next section — the index
has to be in place when an environment is first resolved, or uv installs CPU torch and you have to
tear the venv down and sync again.

Append the block below to **both** pyproject files. Each environment installs its own torch, so
each one needs its own index — doing only the first is the usual mistake:

1. `pyproject.toml` — this repo
2. `external/Binoculars/pyproject.toml` — the submodule

**Not** `external/ghostbuster/pyproject.toml`. That fork depends on torch but only its `--logprob*`
and `--gen_perturb*` stages touch it, and pipeline 4 runs neither: generation is API calls. CPU
torch there is correct and keeps the venv small.

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

Both files are tracked by git, so keep this edit local to the server — don't commit it to either
repo, or every laptop clone will start pulling CUDA wheels too.

### Environments

There are **three** uv environments, deliberately kept apart: this repo pins `transformers>=4.57`,
and both forks pin `transformers>=5.12`. They cannot share a venv.

```bash
# 1. This repo -> .venv  (also installs epai editable, via [tool.uv] package = true)
uv sync

# 2. The Binoculars fork -> external/Binoculars/.venv
uv sync --project external/Binoculars

# 3. The ghostbuster fork -> external/ghostbuster/.venv
uv sync --project external/ghostbuster
```

If step 3 fails while building the `ghostbuster` package itself — its `pyproject.toml` says
`version = "0.1.0"` while its `setup.py` says `1.0`, which some setuptools versions reject — add
`--no-install-project`. Nothing is lost: `generate.py` is always run with `cwd` inside the fork, so
`import utils` resolves from there whether or not the project was installed.

Confirm they really are separate — the three prefixes must differ:

```bash
uv run                                python -c "import sys, transformers; print(sys.prefix, transformers.__version__)"
uv run --project external/Binoculars  python -c "import sys, transformers; print(sys.prefix, transformers.__version__)"
uv run --project external/ghostbuster python -c "import sys, transformers; print(sys.prefix, transformers.__version__)"
```

On the server, add `torch.cuda.is_available()` to that check for this repo and Binoculars — it must
be `True` in both; `False` means the CUDA index above was missing when that venv was resolved. The
ghostbuster env is expected to report `False`, and that is fine (see the CUDA section).

Never add `binoculars` or `ghostbuster` to this project's `dependencies` / `[tool.uv.sources]`, and
never declare a `[tool.uv.workspace]` here. Any of those fuses the environments back together.

The ghostbuster fork also needs its own `.env`, because it calls `load_dotenv()` against its own
directory — this repo's `.env` is not visible to it:

```bash
cp external/ghostbuster/.env_sample external/ghostbuster/.env   # then set OPENAI_API_KEY
uv run --project external/ghostbuster python -c \
  "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('brown')"
```

The nltk downloads are one-off: `generate.py` fetches the first two at import time on every run
otherwise.

### Updating a submodule

A submodule points at a specific commit of its fork and never moves on its own — that is what ties a
set of scores, or a corpus, to the code that produced it. To pick up fork changes (note the branch:
`main` for Binoculars, `master` for ghostbuster):

```bash
git -C external/Binoculars pull origin main
git add external/Binoculars
git commit -m "Bump Binoculars to $(git -C external/Binoculars rev-parse --short HEAD)"

git -C external/ghostbuster pull origin master
git add external/ghostbuster
git commit -m "Bump ghostbuster to $(git -C external/ghostbuster rev-parse --short HEAD)"
```

Both halves matter: the pull moves the working copy, the commit records the new SHA so a fresh
clone gets the same code. Re-run `uv sync --project external/<fork>` only when the pull changed
that project's `pyproject.toml` or `uv.lock`; plain code changes need nothing, the install is editable.

When editing a fork, commit and push **inside** `external/<fork>` first, then commit the
moved pointer here — otherwise the pin references a commit nobody else can fetch.

### Claude Code notifications — optional

Stages in this repo run long enough that you will not sit and watch them. If you drive them
through [Claude Code](https://claude.com/claude-code), `.claude/hooks/notify.sh` pushes a
notification to your phone when a turn ends or when Claude is blocked waiting on you — via
[ntfy](https://ntfy.sh), which needs no account.

```bash
cp .claude/hooks/notify.env.example .claude/hooks/notify.env
$EDITOR .claude/hooks/notify.env     # set NTFY_TOPIC to something unguessable
```

Then subscribe to the same topic in the ntfy app, or open `https://ntfy.sh/<topic>` in a browser.
The hook is already registered in the tracked `.claude/settings.json`; Claude Code will ask you to
approve it the first time it sees it.

Pick the topic name carefully and keep it out of version control — `notify.env` is gitignored for
that reason. The public ntfy server has no per-topic authentication, so the topic name is the only
thing gating access, and the pushes quote the tail of what Claude last said. Skip the file entirely
and nothing breaks: the hook still chimes locally, it just sends no push.

Full details, including what each hook event fires on, are in `.claude/hooks/README.md`.

### Pipeline notifications — optional

The same idea for the runs themselves, whether or not Claude Code is involved. Pipeline 3
(`science_data`) and pipeline 4 (`ghostbuster`) push to ntfy at every stage boundary, so a
five-hour corpus build reports itself to your phone from wherever it is running:

| Push | When | Priority |
|---|---|---|
| 🚀 `started` | the run begins, listing its stages | low |
| ✅ `<stage> ok (3/7)` | a stage finished, with its wall clock and what comes next | default |
| ⚠️ `<stage> failed, non-fatal` | an analysis stage failed; the pipeline carries on | default |
| 🙋 `waiting for input` | the run has stopped at a prompt and will sit there until answered | default |
| 🚨 `FAILED at <stage>` | a critical stage failed or was interrupted, with the resume command | **high** |
| 🏁 `done` | every selected stage ran, with the last stage's result, the output path and any steps left | default |

Every push of one run carries the same head — `<pipeline> #<pid>` — because one topic carries every
run you ever start:

```
science full #4821: science 1/5 > fetch ok (2/9)      ← the nested science_data runner
science full #4821: promote ok (2/5)
ghostbuster #4906: score ok (7/8)                     ← a different run, same topic, same evening
```

That matters in two places. Two pipelines running at once interleave their `(n/N)` counters, and
without the head the sequence reads as one run losing its place and starting over. And pipeline 3's
first step *is* the whole of `run_pipeline.py`, a second runner with its own nine stages — it
reports them under the outer run's id and the outer step it belongs to rather than as a run of its
own.

The last stage's ✅ is folded into the 🏁 `done` push rather than sent just before it: two pushes
milliseconds apart can arrive in either order, which is how `embed ok (8/8)` used to turn up *after*
`done`.

Set one variable in `.env` and subscribe to the same topic in the ntfy app:

```bash
echo "epai-pipeline-$(openssl rand -hex 6)"     # pick an unguessable topic
$EDITOR .env                                    # NTFY_PIPELINE_TOPIC=...
```

Use a **different topic from the Claude Code hook above** — same app, two subscriptions. They are
different feeds: one is "a turn ended", the other is "the GPU stage you started before dinner is
done". Both treat the topic name as the credential, for the reason given above.

With `NTFY_PIPELINE_TOPIC` unset nothing is sent and nothing changes. Sending is best-effort
throughout: a DNS failure, a 502 or a rate limit is swallowed, never retried, and cannot fail a
run. `--dry-run` sends nothing either. Details are in the module docstring of
`epai/utils/notify_utils.py`.

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
| `OPENALEX_API_KEY` | `science_data/fetch/oa_fetch.py` | **Optional.** OpenAlex Premium key — raises the rate limit. The free tier returns the same works and fields |
| `OPENALEX_MAILTO` | `science_data/fetch/oa_fetch.py` | **Optional, not a secret.** Contact address selecting OpenAlex's faster "polite pool"; redundant once a key is set |
| `OPENAI_API_KEY` | `citation_impact/prepare/prepare_data.py`, `utils/embedding_utils.py`, `external/ghostbuster/generate.py` | GPT-4o-mini rephrasing, OpenAI embeddings, and the ghostbuster generator |
| `ANTHROPIC_API_KEY` | `citation_impact/prepare/prepare_data.py`, `external/ghostbuster/generate.py` | Claude Haiku abstract rephrasing; the ghostbuster generator's `claude` type |
| `GOOGLE_API_KEY` | `citation_impact/prepare/prepare_data.py` | Gemini abstract rephrasing |
| `HF_TOKEN` | `citation_impact/score/add_naip_scores.py`, `external/Binoculars/score_parquets.py` | HuggingFace token — for the NAIP Gradio API (optional, avoids rate limits) and for pulling the Binoculars observer/performer models |
| `NTFY_PIPELINE_TOPIC` | `utils/notify_utils.py`, via both pipeline runners | **Optional, but treat it as a secret.** ntfy topic for per-stage phone notifications. Unset means no pushes. See [Pipeline notifications](#pipeline-notifications--optional) |
| `NTFY_SERVER` | `utils/notify_utils.py` | **Optional, not a secret.** Self-hosted ntfy base URL; defaults to `https://ntfy.sh` |

The Semantic Scholar key is optional but strongly recommended — without it you will be heavily rate-limited. The LLM keys are only needed if you enable the `ADD_GEN_AI` flag in `prepare_data.py`.

Pipeline 4 is the exception to "copy `.env_sample` and fill it in once": the ghostbuster fork reads
its *own* directory, so its key goes in `external/ghostbuster/.env`. See [Environments](#environments).

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

Combined files, `ghostbuster_<generator>[_binox<N>].parquet`, written by
`prepare/combine_domains.py`: several corpora in one file, one per `domain`.
This is what the training and analysis scripts read now — a *corpus* is a
`(file, domain)` pair rather than a file.

The `_binox<N>` suffix is **historical**: it marks files scored by hand, or by
pipelines 3 and 4 before those wrote their scores in place. Where it sits says
which order such a corpus was built in — `ghostbuster_gpt54mini.parquet` was
scored per domain and then combined, `ghostbuster_gpt56luna_binox0.parquet`
combined first and scored once. Both runners now fill `binoculars_score` into
the corpus they already built, so a run started today produces
`ghostbuster_<generator>.parquet` and `<dataset_id>_<variant>.parquet` with the
scores inside them and no second file. Same columns either way; the existing
`_binox0` files are still valid corpora and nothing has to be renamed.

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
[add_llm_judge_scores.py]                ←          add llm_judge_score_<model>_<prompt> (API or local GPU)
[add_llm_judge_reasons.py]               ←          add llm_judge_reason_/cues_* on a sample of rows
   ↓  data/parquet/<corpus>_binox*.parquet
[combine_domains.py]                     ← Stage C: the per-corpus files → ONE file
   ↓  data/parquet/ghostbuster_<generator>.parquet   (uid, id, domain, text, …)
[analyse_domain_pca.py]                  ← Stage D: EDA on the combined file
[analyse_emb_class_stats.py] [analyse_surface_artifacts.py]
[analyse_text_lengths.py]    [analyse_score_human_vs_ai.py]
   ↓
┌────────────────────────────────────────────────────────────┐
│  Classification                                            │
│  run_clf_cross_domain.py   transfer → results/clf_cd/      │
│  run_clf_in_domain.py      one corpus → results/clf_id/    │
│  run_sweep_cross_domain.py          → results/sweep_clf_cd/│
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

**Binoculars scores** come from the fork in `external/Binoculars`, which runs in its own
environment. Run it from the repo root. It writes each input **in place** unless `-o` is given, so
pass `-o` when you want the unscored file kept. Pipelines 3 and 4 used to, which is where the
`_binox0` suffix on the older files comes from; both now score in place, since a scored file is
what this script re-scores from anyway (it fills the Null rows, and `--recompute` redoes the rest):

```bash
# in place: reuter.parquet gains a binoculars_score column
uv run --project external/Binoculars python external/Binoculars/score_parquets.py \
    data/parquet/reuter.parquet --cpu-threads 16 --gpu 0

# or keep the input and write a scored copy (-o takes a single input file)
uv run --project external/Binoculars python external/Binoculars/score_parquets.py \
    data/parquet/reuter.parquet -o data/parquet/reuter_binox0.parquet --cpu-threads 16 --gpu 0
```

Pass several files or a whole directory to score them in one run, and `--only-missing` to fill the
gaps in an already-scored file. Two details worth knowing:

- Use the **script path**, not `python -m binoculars...`. The fork is a virtual uv project (no
  `[build-system]`), so `binoculars` is importable via the script's own directory rather than being
  installed into the env.
- The script calls `load_dotenv()`, which walks up from the repo root and picks up **this** repo's
  `.env` — so `HF_TOKEN` is read from there. The submodule needs no `.env` of its own.

**Script:** `epai/ai_detection/embed/add_embeddings.py`
**Output:** overwrites / extends the input Parquet with new `emb_*` columns

```bash
# the EMBEDDERS list at the top of the script (all 13, including the billed OpenAI ones)
uv run python epai/ai_detection/embed/add_embeddings.py data/parquet/reuter_binox0.parquet --cpu-threads 16 --gpu 0

# or name the ones you want — this is what pipelines 3 and 4 pass from their configs
uv run python epai/ai_detection/embed/add_embeddings.py data/parquet/reuter_binox0.parquet \
    --embedders specter2 mpnet luar --cpu-threads 16 --gpu 0
```

Adds multiple embedding columns in one pass. `--embedders` selects a subset per run
(names validated before any model loads); without it the script's own `EMBEDDERS`
list applies. Supported embedders:

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
# LLM-as-judge P(human) per row -> llm_judge_score_<model>_<prompt>[_lp]
uv run python epai/ai_detection/score/add_llm_judge_scores.py data/parquet/ghostbuster_gpt56luna.parquet

# the same run with an open-weights judge on the local GPU
uv run python epai/ai_detection/score/add_llm_judge_scores.py \
    data/parquet/ghostbuster_gpt56luna.parquet --model qwen3-14b
```

```bash
# why did it decide that? blind rationales + countable cues on a SAMPLE of rows
# -> llm_judge_reason_*, llm_judge_cues_human_*, llm_judge_cues_machine_*
uv run python epai/ai_detection/score/add_llm_judge_reasons.py \
    data/parquet/ghostbuster_gpt56luna.parquet
```

`add_llm_judge_reasons.py` never writes a score column. It reads one only to
pick rows — `--select wrong uncertain confident` draws the most confidently
wrong rows, the ones nearest the fence, and a correct-and-confident control —
then asks the same model, under the same prompt version's evidence half, which
cues it can point at. The cues come from a closed vocabulary (one entry per
bullet of that prompt's evidence lists), so they aggregate into a frequency
table per class, saved under `results/llm_judge_reasons/`. The model is not
shown its own score: a model handed a number justifies that number.

`--model` takes a registry alias or any Hugging Face repo id; `--list-models`
prints the table. A local judge defaults to `--mode logprob`, which reads the
score off the distribution over a one-word answer instead of parsing a number
the model wrote — one forward pass per row, and no ties for ROC-AUC to throw
away. The mode is part of the column name (`_lp`), because the two are not on
one scale; every judge adds a column and none replaces one, so a local run can
be compared against the API baseline in the same file.

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
    --config cfg/ai_detection/train/clf_cd/clf_cd_example.yaml

# detector scores, human vs AI: violins, ROC/PR, confusion
# PATH:TARGET, where TARGET is the generated_by label <model_slug>:<prompt_id>
uv run python epai/ai_detection/analyse/analyse_score_human_vs_ai.py \
    data/parquet/ghostbuster_gpt35t.parquet:gpt35t:0000A

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

**Config:** `cfg/ai_detection/train/clf_cd/clf_cd_suites_all.yaml` · **Output:** `results/clf_cd/<experiment_id>/`

```bash
uv run python epai/ai_detection/train/run_clf_cross_domain.py             # the clf_cd folder
uv run python epai/ai_detection/train/run_clf_cross_domain.py --dry-run   # preflight only
uv run python epai/ai_detection/train/run_clf_cross_domain.py --cfg a.yaml b.yaml
```

Trains on each corpus in turn and evaluates on all of them, producing a source ×
target accuracy matrix. The off-diagonal shows how well the classifier transfers
to unseen corpora — distinguishing "learned machine-generated-ness" from
"learned corpus artifacts".

Each folder holds exactly one config, which is both a working experiment and the
annotated reference for every key. Copy it to start a new one — each run also
writes its config into its own results directory, so a finished experiment
carries its config with it.

##### The three training designs

How many corpora each model trains on is the run's `mode`. All three run the
**same** feature sets, algorithms, seeds and splits on the **same** corpora, so
their numbers are directly comparable; only which rows go into each model
changes.

| design | `mode` | config | runner |
|---|---|---|---|
| train on 1 domain, test on all | `cross` | `clf_cd/clf_cd_suites_all.yaml` | `run_clf_cross_domain.py` |
| train on 2 domains, test on the 3rd | `lodo` | `clf_lodo/clf_lodo_suites_all.yaml` | `run_clf_cross_domain.py` |
| train + test on 1 domain, and on the whole parquet | `in_domain` | `clf_id/clf_id_domains_all.yaml` | `run_clf_in_domain.py` |

`mode: lodo` — leave-one-domain-out — is the same matrix over the same corpora,
with each model trained on the **union of every corpus but one**. A row is named
after the corpus it held out, so on every row the **diagonal** cell is the
transfer number and the cells beside it are a held-out split of the training
pool: the in-domain mask is the exact inverse of `cross`. Nothing downstream
reads the diagonal to tell them apart — every cell carries its own `in_domain`
flag, and the printed tables, the boxed heatmap cells, the score-distribution
tints and the figure legends all follow it.

It answers the question `cross` cannot: a single-source model has every incentive
to memorise its corpus, so if transfer stays at chance even when the model *does*
see domain diversity, the failure is not "not enough diversity".

The two live in different folders so a zero-argument run never trains both by
accident:

```bash
uv run python epai/ai_detection/train/run_clf_cross_domain.py \
    --cfg cfg/ai_detection/train/clf_lodo/clf_lodo_suites_all.yaml
```

##### Sweeping an axis

The same script sweeps either axis, because a run is a list of **variants** and a
variant is one feature set crossed with one algorithm:

A run is the cross product of `feature_sets` × `algorithms`, and **every
`feature_sets` entry is one design matrix** — whether it names one column or
five. Hold an axis at a single entry to ask the obvious question of the other:

| config | variants |
|---|---|
| one feature set + several `algorithms` | one per algorithm — *which model transfers best* |
| several feature sets + one algorithm | one per set — *which representation does* |
| one of each | a single matrix, the plain control |

Because an entry may stack several columns, one run can also ablate what a set
*contains* — the same embedding with and without a detector score beside it,
scored on identical rows and ranked side by side:

```yaml
feature_sets:
  - emb_luar_crud                                    # one column
  - [emb_luar_crud, emb_specter2]                    # both, stacked into one matrix
  - {columns: [emb_luar_crud, binoculars_score], id: luar_binox}
algorithms: [logistic, simple_mlp]                   # -> 6 variants
```

Both axes may vary at once, giving one matrix per pair. Every variant gets its
own directory and figures plus a row in a comparison that ranks them by transfer
accuracy, and optional zero-shot Binoculars / LLM-judge baselines join that table
as ★ rows. Algorithms are the `epai/algs` registry — logistic, SVM, gradient
boosting, MLP, Binoculars-MLP — each declaring the hyper-parameters it reads, so
a `params` block either takes effect or fails in preflight.

Each feature set is loaded and stacked **once** and reused by every algorithm
trained on it, so an N × M run costs N corpus reads, not N × M.

Output directories follow whichever axis varies: `<alg_id>/emb_luar/` for a
feature sweep, `<alg_id>/logistic/` for an algorithm sweep,
`<alg_id>/emb_luar__logistic/` only when both move.

#### In-domain control (`run_clf_in_domain.py`)

**Config:** `cfg/ai_detection/train/clf_id/clf_id_domains_all.yaml` · **Output:** `results/clf_id/<experiment_id>/`

```bash
uv run python epai/ai_detection/train/run_clf_in_domain.py             # the clf_id folder
uv run python epai/ai_detection/train/run_clf_in_domain.py --dry-run   # preflight only
```

Trains on a corpus and scores the held-out grouped split of **that same corpus**.
It asks "how well does this separate human from machine text *here*", not "does
it transfer", and the ~97–99% it reports is an upper bound on a deployed detector
rather than an estimate of one. That gap is exactly what the cross-domain script
exists to measure, so this is its control, not its replacement.

Same config shape as a `clf_cd` config — the same `feature_sets` × `algorithms`
cross product, the same `model` block, the same ★ baselines. `data` may name one
corpus directly:

```yaml
data:
  parquet_path: "data/parquet/ghostbuster_gpt54mini.parquet"
  human_label: "human"
  group_col: "id"
  # where: {domain: essay}   # optional; omit to train on ALL the data
```

…or carry a `datasets` list, which runs the whole experiment **once per corpus**,
plus once on the whole parquet with `include_pooled` — the N+1 numbers this
control is usually wanted for:

```yaml
data:
  parquet_path: "data/parquet/ghostbuster_gpt56luna.parquet"
  human_label: "human"
  group_col: "id"
  datasets:
    - {name: essay,  where: {domain: essay}}
    - {name: reuter, where: {domain: reuter}}
    - {name: wp,     where: {domain: wp}}
  include_pooled: true          # + one more on the parquet, unfiltered
```

This list is **not** the cross-domain one: nothing is ever trained on one entry
and tested on another, so a one-entry list is legitimate here where the matrix
runner needs two, and none of the output is a transfer claim. The pooled row is
the number a detector trained on everything available actually gets — and it is
not independent of the per-domain rows, because it is the same documents again.

One corpus writes straight into `<alg_id>/`, exactly as before; several write
`<alg_id>/<corpus>/` each holding what a single-corpus run holds, plus
`corpus_comparison.json` and one figure ranking every variant by its mean
in-domain accuracy with the per-corpus columns still visible, then the same rows
under every headline metric (shape only — the numbers below are made up):

```
variant                    essay   reuter       wp      all     mean  ±corpora    worst
  Logistic Regression      98.7%    99.1%    97.9%    98.6%    98.6%       0.4    97.9%
★ binoculars falcon-7b     86.4%    82.0%    79.3%    82.5%    82.6%       2.5    79.3%

variant                    auroc  tprT@1%    tpr@1%      acc        f1
  Logistic Regression      99.9%    97.4%     99.0%    98.6%     98.6%
★ binoculars falcon-7b     76.5%    24.5%     26.2%    82.6%     56.6%
```

##### The headline metric set

`cd_core.HEADLINE_METRICS` is one tuple read by all three designs, so a variant
is reported under the same five numbers whichever produced it — and it drives the
figure set as well (`CD_PLOT_METRICS` is now that same tuple), so one figure per
metric per variant in every mode.

| | what it is |
|---|---|
| `auroc` | threshold-free separability — the only one that owes nothing to a cutoff |
| `tpr_at_srcfpr_0.01` | **(b)** threshold fitted to 1% FPR on the calibration slice, carried over unchanged — the deployable number |
| `tpr_at_fpr_0.01` | **(a)** the same point, threshold read off the tested cell's own ROC — an oracle, here so the (a)−(b) distance is visible |
| `accuracy` | the headline everyone reads first, kept honest by sitting fourth — its threshold is Youden's J, fit on the same calibration slice as (b) |
| `macro_f1` | accuracy's balance-aware twin, same threshold; tracks it closely on this near-balanced task |

The realized FPR behind (b) is deliberately not in the set — it is not a score,
it is the price of one, and it is annotated inside the cells of the (b) matrix
where the pair has to be read together.

##### Where a threshold comes from

Every threshold above — the transported TPR@FPR levels and the Youden's J
behind accuracy/macro F1 — is fit on a **calibration slice**: a grouped
`cal_frac` (default 15%) of the training pool, carved out with
`cd_core.carve_calibration` and never touched by the fit itself. The model
trains on the rest of the pool; the eval rows (the held-out split, or a
transfer target) are never involved in either. This is true in every mode,
including the cells that used to be special cases — the `cross` diagonal and
the `lodo` held-out row, where calibrating on the eval split itself used to
collapse the transported reading into the oracle one — and it is true for the
zero-shot score baselines too, which calibrate on the identical rows (same
corpus, same seed) the trained variant of that row uses. `cross`, `lodo` and
`in_domain` therefore agree on a shared cell exactly, which they did not
before this: the in-domain runner used to calibrate on the training rows
themselves, which biased the transported TPR too strict.

Every variant is scored on the held-out rows **and** on the rows it trained on,
so the ranking carries a `gap` column — train minus test accuracy, i.e. what
failed to generalise even inside the corpus:

```
algorithm                    dim     train      test     gap   test n
  SVM (RBF)                  512    100.0%     99.5%     0.5     1799
  Logistic Regression        512     99.9%     98.7%     1.2     1799
★ binoculars                   1     86.8%     86.4%     0.4     1799
```

`report_by: [domain, generated_by]` breaks the split down by any column and draws
it as a second panel on the comparison figure. Those columns are read and kept
aligned with the design matrix but never stacked into it — the model must not see
`domain`. Worth setting whenever a run pools subsets: `GroupShuffleSplit` groups
but does **not** stratify, so proportions hold only in expectation (~±1.6 points
per domain at this corpus size), and nothing else in the output would say.

The ± over seeds is not a confidence interval — the seeds resample one corpus, so
the test halves overlap and the spread understates the sampling error.

The split, the binarised target, the missing-value rule and the metrics are
shared with the matrix run (`epai/utils/cd_core.py`), so a variant's held-out
score here is the same number as that variant's diagonal cell over there.

#### Hyper-parameter search (`run_sweep_cross_domain.py`)

**Config:** `cfg/ai_detection/train/sweep/sweep_example.yaml` · **Output:** `results/sweep_clf_cd/<sweep_id>/`

```bash
uv run python epai/ai_detection/train/run_sweep_cross_domain.py            # the sample config
uv run python epai/ai_detection/train/run_sweep_cross_domain.py --report   # leaderboard
uv run python epai/ai_detection/train/run_sweep_cross_domain.py --final    # winner on the full matrix
```

Every trial is one variant of the runner above — same split, same target, same
group-exclusion rule — so the winner's numbers drop straight into the same
comparison as a hand-written config.

To regenerate figures from saved results without retraining:
```bash
uv run python epai/ai_detection/train/replot_cross_domain.py results/clf_cd
uv run python epai/ai_detection/train/replot_in_domain.py results/clf_id
uv run python epai/ai_detection/train/replot_sweep.py results/sweep_clf_cd/luar_mlp_reg
```

---

## Pipeline 3 — Science Data

A newer corpus builder: instead of reusing Ghostbuster's essays and news, it
assembles an AI-vs-human corpus of *scientific abstracts*. The protocol is the
part that matters — each generator sees only a prompt derived from the paper,
never the human abstract, so the AI class does not inherit the human text's
content and structure the way a rephrasing setup does.

```
[oa_fetch.py]        ← Stage 1: OpenAlex → <id>_raw.parquet
   ↓                          (stops on the POST-filter count, per domain)
[filter.py]          ← Stage 2: admission rules → <id>_human.parquet
   ↓                          (exactly N records per domain)
[make_prompts.py]    ← Stage 3: one generation prompt per paper
   ↓                          <id>_prompts_<prompt_config>.parquet
[generate_ai.py]     ← Stage 4: prompt → AI abstract, checkpointed per call
   ↓                          <id>_<variant>.parquet   ← promote this
[analyse_{fetch,filter,prompts,generation}.py]   ← per-stage reports
```

**`dataset_id` names the paper set; `prompt_config` names the arm.** Every path
in `data/science/` derives from those two, and they answer different questions.
Different papers — a changed domain list, changed fetch/filter thresholds, or
*any* re-fetch, since OpenAlex's relevance ranking drifts between runs — is a
new `dataset_id`. Same papers under different prompts, a different mode, or a
different generator is the *same* `dataset_id` and a new `prompt_config`; the
prompts parquet, the generate shard key and the stage-4 output all carry the arm
in their names, so the arms cannot collide.

Follow that and the pipeline has no destructive case: a run that changes the
paper set writes to different files by construction. Ignore it and one command
is genuinely dangerous — `run_full_pipeline.py --yes` under an id that already
has generations re-scrapes, auto-answers the fetch stage's own overwrite prompt,
and strands every paid call in `<id>_generate.jsonl` against a corpus whose ids
no longer match. The only symptom is a `N paper(s) dropped` line at export that
reads like a few failed API calls. Start later in the pipeline
(`--science-from prompts`) whenever the corpus is one you want to keep.

**Everything lives in `data/science/`.** None of the four stages writes to
`data/parquet/`. Stage 4's output is the only artifact meant to leave, and
`run_pipeline.py` does not move it for you:

```bash
mv data/science/science_v1_gpt54mini_0807A.parquet data/parquet/
```

That keeps a half-finished or abandoned run from ever being picked up by an
ai_detection config that globs `data/parquet/`.

**Or run the whole thing, scoring and embedding included.**
`run_full_pipeline.py` wraps the four stages above and adds three more:
`promote` (a *copy* into `data/parquet/`, so `data/science/` stays the record of
what the run produced), `score` (Binoculars, in the fork's environment, filling
`binoculars_score` in place), and `embed` (every `emb_*` column, in place). The
corpus that comes out is ready for `train/` and `analyse/` with nothing left to
run by hand:

```bash
# corpus → data/parquet/, scored, embedded
uv run python epai/science_data/run_full_pipeline.py --yes --cpu-threads 16 --gpu 0

# stop before the GPU steps, finish them later or elsewhere
uv run python epai/science_data/run_full_pipeline.py --until promote --yes
uv run python epai/science_data/run_full_pipeline.py --from score --gpu 0

# resume inside the science half, then carry on through score and embed
uv run python epai/science_data/run_full_pipeline.py --science-from prompts --yes

# ten-row debug run of the two GPU steps, over a corpus that already exists
uv run python epai/science_data/run_full_pipeline.py --only score embed \
    --limit 10 --embed-limit 10 --yes --gpu 0
```

The corpus half has no `--limit` of its own: how many papers get fetched and
generated is `fetch.target_per_domain` in the config, because the fetch stage's
stop condition and the filter stage's admission test have to agree on it.

`--from`, `--until` and `--only` work over `science → promote → score → embed`;
`--science-from` reaches into `run_pipeline.py`'s own stage list.

The embed step is configured by an **optional** `embed:` block in the science
config — optional so that a config written before this step existed still works:

```yaml
embed:
  embedders: ["specter2", "mpnet", "luar"]   # null / absent = all 13, incl. billed OpenAI
  batch_size: null
  limit: null
  recompute: false
```

The names are checked in the preflight, before the fetch. `--embedders`,
`--embed-batch-size`, `--embed-limit` and `--embed-recompute` override the block
per run, and `--gpu` / `--cpu-threads` / `--text-column` apply to both GPU steps.
See [pipeline 4](#pipeline-4--ghostbuster-with-a-newer-generator) for the table of
embedder names.

**Every stage writes one parquet holding every domain**, with `domain` as a
column — the same shape as `ghostbuster_gpt54mini.parquet`. There are no
per-domain files and no shard directory: `epai/utils/corpus_utils.py` defines a
corpus as a *slice* of a file (`where: {domain: "biology"}`), so a per-domain
file would only be a second copy of rows already present. Stage 1 rewrites its
one output after each domain completes, which is what a crash in domain 9 needs
— not a set of pieces to reassemble.

The two `.jsonl` files in `data/science/` are not corpus shards: they record
which paid LLM calls have already been made, and deleting one means paying for
those calls again.

### Prompts are config, and the corpus says which ones it used

Every prompt the pipeline sends lives in `cfg/science_data/prompts/<id>.yaml` —
the spec system prompt, the generation prompt template, and the generation
system prompt. The main config points at one with `prompt_config: "0807A"`.

Its **`mode`** lives there too, and that is the point rather than an accident of
layout. `llm_spec` has a model write a content spec of the paper and splices it
into the generation prompt; `title_only` skips the spec stage entirely and
leaves the generator the title, the field and a target length. So the mode
changes what the prompt *says* — and since the id of this file is the corpus's
only record of what the prompt said, a mode set anywhere else would be the one
input to a generated row that the row's label does not name. The pipeline config
may still carry `prompt.mode`, but only agreeing with the prompt config; a
mismatch fails at load, before anything is spent.

That id is stamped into every generated row:

```
id           = "science:<domain>:<paperId>"       e.g.  science:computer_science:W2143535440
generated_by = "<model_slug>:<prompt_config>"     e.g.  gpt54mini:0807A
uid          = "<id>:<generated_by>"              e.g.  science:computer_science:W2143535440:gpt54mini:0807A
```

Domain names are slugged (`Computer Science` → `computer_science`) where the
queries config is loaded, so the `domain` column, the ids and any `where:`
clause all spell a domain the same way, and no id ever carries a space.

The label is **derived, never typed**, so it cannot disagree with what actually
produced the row. Checkpoints are keyed on it too, so pointing `prompt_config`
at a new id re-issues every generation call rather than silently reusing text
written under the old prompts. It still begins with `gpt`, so `cfg/colors.yaml`
resolves it to the GPT family by longest-prefix as before.

> **Never edit a prompt file that has already produced a corpus.** Copy it to a
> new id (`0808A`, …) and point `prompt_config` at the copy. Editing in place
> makes every existing row's label a lie, and nothing can detect that
> afterwards. `load_prompt_config` enforces that a file's `id` matches its
> filename, which catches the copy-but-forgot-to-rename half of this.

Stage 4 writes the corpus in the column order `combine_domains.py` produces —
`uid, id, domain, text, generated_by`, then metadata — so a science corpus and
a ghostbuster one can be eyeballed with the same `df.iloc[:, :5]`. It also
raises on a duplicate `uid`, and `require_all_generators` (on by default) drops
any paper some generator failed on, so every `id` is a complete pair.

Two fetch-bookkeeping columns ride along: `query` (which search found the
paper) and `also_matched_domains` (which other domains' queries also matched
it). `combine_domains.py` drops both when it runs, so they cost nothing
downstream.

The **`prompt`** column carries the verbatim user message that produced the
row, so the corpus can be read without going back to
`<id>_prompts_<prompt_config>.parquet`.
It is **null on every human row** — a human abstract was not written to a
prompt, and filling the column there with the prompt derived *from* it would
invite exactly the reading the protocol exists to prevent. It sits last, after
the metadata, because a 200-word cell beside `text` makes any `df.head()`
unreadable. The system prompt is not stored per row: it is one string for the
whole run, and `generated_by` already names the prompt config it came from.
A corpus built before this column existed can be rebuilt by re-running stage 4
— the generation checkpoint is keyed on `(id, generated_by)`, so it re-exports
from cached calls and costs nothing.

Stage 2 is `filter.py`, not `filter_abstracts.py`: only 5 of its 14 admission
rules look at the abstract text — the other 9 reject on the id, title, year,
citation count, FWCI, language label, or a cross-domain duplicate. It filters
whole records, and also trims each domain to target and reshapes into the
corpus schema.

All stages read one config (`cfg/science_data/science_cfg_example.yaml`),
with the domain/query lists in a separate file so they can be swapped
independently. Every stage defaults `--cfg` to that example.

**One command runs the whole thing**, each analysis report interleaved after
the stage it describes:

```bash
uv run python epai/science_data/run_pipeline.py --yes
```

`--yes` answers the fetch stage's overwrite prompt; without it the run stops to
ask. The runner ends with a per-stage timing summary and the `mv` that promotes
the result.

| Flag | |
|---|---|
| `--from STAGE` | resume after a failure, e.g. `--from prompts` |
| `--only STAGE …` | run just these (always in pipeline order) |
| `--skip-analyse` | drop the four report stages |
| `--dry-run` | print the commands without running them |
| `--retry-failed` | forwarded to the two LLM stages |

Stage keys: `fetch`, `analyse-fetch`, `filter`, `analyse-filter`, `prompts`,
`analyse-prompts`, `generate`, `analyse-generation`.

A failing **analysis** stage is reported and skipped past — a matplotlib
problem should not discard a completed fetch. A failing **core** stage stops
the run and prints the `--from` command to resume with, since every stage
consumes the previous one's output.

Nothing is skipped on the grounds that its output already exists. The stages
that cost money checkpoint per call and skip completed work from the inside;
the rest are cheap to redo. Guessing that a parquet on disk is current is how a
corpus ends up half-built from two different configs.

The stages are also runnable one at a time, which is what the runner does under
the hood:

```bash
uv run python epai/science_data/fetch/oa_fetch.py
uv run python epai/science_data/filter/filter.py
uv run python epai/science_data/prompt/make_prompts.py
uv run python epai/science_data/generate/generate_ai.py
```

Pass `--cfg` (and `--queries`, which stage 1 defaults to
`queries_example_debug.yaml`) to run anything else:

```bash
uv run python epai/science_data/fetch/oa_fetch.py \
    --cfg cfg/science_data/science_cfg_example.yaml \
    --queries cfg/science_data/queries_example.yaml
```

### Stage 1: fetching from OpenAlex

The admission rules read `paper_id`, `title`, `abstract`, `year`,
`citation_count`, **`fwci`** and **`language`** — a record missing FWCI or not
labelled English is rejected, not merely unannotated. Alongside those:

| Column | From | Notes |
|---|---|---|
| `fwci` | `fwci` | **Required.** Field-Weighted Citation Impact: citations against the expectation for work of the same field, type and age, so 1.0 is exactly average and — unlike `citation_count` — it is comparable across domains and years. Null means OpenAlex could not place the work in a field, so there is nothing to normalise against. A genuine `0.0` is kept |
| `language` | `language` | **Required** to equal `filter.require_language` (`"en"`). Stored as well as enforced, so a `wrong_language` drop is distinguishable from a `not_english` one — the latter is the stopword heuristic disagreeing with OpenAlex's label |
| `author_ids` | `authorships[].author.id` | A genuine parquet **list** column of short ids (`A5011953697`), in authorship order. Ids not names: a name is not an identity — authors publish under varying forms and different people share names — while an id is stable across a person's papers and resolves at `/authors/<id>` |
| `author_count` | `len(authorships)` | Counted *before* unresolved authorships are dropped, so this is the honest author count. About 5% of authorships have a name but no author entity, so `len(author_ids)` can be smaller; comparing the two is the only way to spot it |
| `doi` | `doi` | Bare, no `https://doi.org/` prefix. Empty for the ~8% of works carrying none — not required, since a missing DOI says nothing about whether the abstract is usable |
| `venue` | `primary_location.source.display_name` | |
| `fields_of_study` | `topics[].field.display_name` | The *field* level of OpenAlex's domain → field → subfield → topic hierarchy, i.e. broad subject areas |

The exact request field list is `rules.OA_SELECT`, deliberately next to the
function that reads it. `authorships` is the one expensive field in it — it
carries institutions, countries and affiliation strings per author, none of
which are stored — but OpenAlex selects whole top-level fields, so there is no
ids-only projection.

`require_fwci` and `require_language` are mirrored into server-side filters, so
non-matching works are never fetched; the row-local rules still re-test every
record, because correctness here never depends on the API honouring a filter.
OpenAlex has no "is not null" filter, so FWCI is narrowed with the range
`fwci:>-1` — deliberately not `>0`, which would drop the very common zero-FWCI
works and bias the corpus toward cited work. Once the other filters are applied
(`has_abstract`, `type:article`, pre-2021) requiring FWCI costs only ~0.2% of
the candidate pool, so it is not a plausible cause of a domain shortfall.

Neither OpenAlex identifier is required. `OPENALEX_API_KEY` is a Premium key
that raises the rate limit; the free tier returns the same works with the same
fields, so a missing or rejected key degrades the run rather than ending it and
`oa_fetch.py` offers to continue without it. `OPENALEX_MAILTO` is a plain
contact address that selects OpenAlex's faster "polite pool" — worth setting
for a keyless run, redundant once a key is set, since the key already
identifies the caller. Both are sent when present; `--no-api-key` /
`--no-mailto` ignore them for one run without editing `.env`.

One quirk of the source to know about: OpenAlex serves abstracts as an inverted
index, not as prose, so `abstract_raw` is a reconstruction with normalised
whitespace rather than the verbatim publisher string. The `abstract` column is
unaffected — cleaning collapses whitespace anyway.

The raw corpus is keyed only by `dataset_id`, so a re-run replaces the previous
one; give a new run its own `dataset_id` to keep both. `oa_fetch.py` prompts
before overwriting an existing raw parquet.

`oa_hw_fetch.py` is a standalone one-paper dump of the API, useful for checking
what a field actually contains before wiring it into `rules.py`:

```bash
uv run python epai/science_data/fetch/oa_hw_fetch.py 10.1038/s41562-025-02244-z
```

### Reading a corpus by eye

A pair is unreadable in a dataframe: two 200-word strings and a longer prompt,
one per cell, and `df.head()` shows none of it. `export_records_txt.py` writes
one `.txt` per `id` — the original text, the prompt, then each AI text,
separated by labelled rules and wrapped to 80 columns:

```bash
# every record → data/inspect/<parquet stem>/0001_biology_science_biology_W2143….txt
uv run python epai/scripts/export_records_txt.py data/parquet/science_v1_gpt54mini_0807A.parquet

# 25 random biology papers, somewhere else
uv run python epai/scripts/export_records_txt.py corpus.parquet \
    --where domain=biology --sample 25 --seed 0 -o /tmp/look
```

`--limit` cuts from the head of the file, `--ids` takes named papers, `--wrap 0`
leaves the text single-line. The column names are all flags
(`--text-column`, `--prompt-column`, `--label-column`, `--human-label`), so it
reads a ghostbuster parquet or anything older just as well; a file with no
`prompt` column simply exports without it, and one with no `id` column becomes
one file per row. A `_index.tsv` beside the files maps filename → id → domain.

Once exported, the corpus joins the AI-detection pipeline at Stage B (embed) —
and `prepare/combine_domains.py` can fold it in beside the Ghostbuster domains.

---

## Pipeline 4 — Ghostbuster with a newer generator

The original Ghostbuster corpora were written by GPT-3.5-turbo. This pipeline
rebuilds them with a current model and carries the result all the way to a
scored parquet, so "does the detector still work on a newer generator?" is one
command rather than a week of glue:

```
[generate.py]         ← Stage 1: prompts + human word counts → machine text
   ↓                           external/ghostbuster/data/<domain>/<model>_<prompt>/*.txt
[fix_empty_...py]     ← Stage 2: refill the files that came back blank
   ↓
[collect]             ← Stage 3: audit the tree, copy it into this repo
   ↓                           data/ghostbuster-data-<variant>/
[prepare_{essay,wp,reuter}.py]  ← Stage 4: one parquet per domain, named generators only
   ↓                           data/parquet/<domain>_<variant>.parquet
[combine_domains.py]  ← Stage 5: union with a `domain` column, then drop the inputs
   ↓                           data/parquet/ghostbuster_<variant>.parquet
[score_parquets.py]   ← Stage 6: Binoculars over the whole corpus, in place
   ↓                           data/parquet/ghostbuster_<variant>.parquet
[add_embeddings.py]   ← Stage 7: every emb_* column, in place
                              ↳ ready for train/ and analyse/
```

Stages 1–2 run in the **ghostbuster fork's** environment with `cwd` inside the
submodule (`generate.py` does `import utils` and hardcodes `data/<domain>/...`
relative to the working directory — neither is configurable there). Stage 6 runs
in the **Binoculars fork's** environment. Stages 3–5 and 7 run in this one. The
runner strips `VIRTUAL_ENV` and `UV_PROJECT_ENVIRONMENT` before calling either
fork, so an outer `uv run` cannot point them at this repo's venv.

Which embedders stage 7 runs comes from `embed.embedders` in the config, and the
names are checked in the preflight — before generation, not at the last stage.
Leaving it `null` runs `add_embeddings.py`'s own `EMBEDDERS` list, which is all
thirteen models **including `openai_small` and `openai_large`, billed per row**;
naming the ones you want is how that stays deliberate:

```yaml
embed:
  embedders: ["specter2", "mpnet", "luar"]
```

Available names (from `EMB_TYPES` in `epai/utils/embedding_utils.py`):

| Group | Names |
|---|---|
| content | `specter2`, `scincl`, `scibert` |
| general semantic | `mpnet`, `minilm`, `e5`, `bge`, `qwen3` |
| raw LM | `roberta` |
| style / authorship | `styledistance`, `style_wegmann`, `luar`, `luar_crud` |
| API (billed) | `openai_small`, `openai_large` |

`<variant>` is the generator model's slug, derived not configured:
`gpt_model: "gpt-5.6-luna"` gives `gpt56luna`, hence
`data/parquet/ghostbuster_gpt56luna.parquet`. Same convention as the existing
`ghostbuster_gpt54mini.parquet`.

### What a generated row is called

Every machine row is labelled `<model_slug>:<prompt_id>` — `gpt56luna:0701A` —
the same `generated_by` convention pipeline 3 uses, so one reader of that column
learns one thing. The model half comes from `generate.gpt_model`; the prompt half
is `generate.prompt_ids`, which has to be stated because the prompts live inside
the fork (`utils/prompt_utils.py`) and `generate.types` names an *index* into
them rather than a version of them.

On disk the same generator is a directory: `data/<domain>/gpt56luna_0701A/`, an
underscore standing in for the colon. That is not cosmetic. The fork's own name
for a generator (`gpt_plain`) says which prompt was used and nothing about which
model used it, so every model generates into one directory — and since the fork
skips output files that already exist, **the second model to run there writes
nothing at all and its corpus is the first model's text under a new filename.**
Model-scoped directories make that impossible: a different model has nothing to
skip. The runner passes the name to `generate.py --out_name`, and
`prepare_*.py` map it back to the label (`epai/utils/label_utils.py`).

**A collected tree is added to, never emptied.** Collect copies into
`data/ghostbuster-data-<variant>/` without deleting, so a generator directory
from an earlier run into the same variant stays behind — and if the prepare
stage simply read every directory it found, that old text would come back as a
second set of machine rows under its old label, doubling the AI class. So
prepare is *told* what to read (`--generators human gpt56luna_0701A`), the
config decides what the corpus holds, and collect prints a NOTE naming any
leftover directory and the `rm -rf` for it.

Directory names that are not of that shape — `human`, `gpt`, `gpt_prompt1`,
`gpt_writing`, and the rest of the original release — pass through unchanged, so
the older trees in `data/ghostbuster-data/` and `data/ghostbuster-data-gpt54/`
still prepare exactly as they did.

Changing the prompts in the fork means changing `prompt_ids`. `0701A` is the
plain prompt as it stands, unchanged since the gpt54mini corpus — which is why
that corpus is labelled `gpt54mini:0701A`, and the shared id is what makes the
two comparable: same prompts, different model.

### Seed the shared inputs first

**This is a manual step, once per machine, and the pipeline will refuse to run
without it.** Three things are *inputs*, not outputs: the human documents, the
essay/wp prompts, and reuter's headlines. They must be identical across model
variants — otherwise a difference in detector scores could be a difference in
the human class or in the prompts, and the comparison means nothing. So the
pipeline never writes them, and a config that lists `human` or `prompts` under
`generate.types` is rejected.

Copy them from an existing corpus directory into the submodule:

```bash
mkdir -p external/ghostbuster/data

cp -a data/ghostbuster-data-gpt54/essay  external/ghostbuster/data/essay
cp -a data/ghostbuster-data-gpt54/wp     external/ghostbuster/data/wp
cp -a data/ghostbuster-data-gpt54/reuter external/ghostbuster/data/reuter

# drop the previous model's generations, keep human/, prompts/ and reuter's gpt/<author>/headlines/
rm -rf external/ghostbuster/data/*/gpt_plain
```

That `rm` is housekeeping, not a correctness requirement: generations now live
in per-model directories (`gpt56luna_0701A/`), the collect stage copies only the
ones this run produced, and older ones cost nothing but disk. The exception is a
tree seeded from a corpus built *before* that naming — its generations sit in
`gpt_plain/`, which no model-scoped run would ever write to or read from, so
they are pure dead weight.

What has to be present, per domain:

| Domain | Required | Why |
|---|---|---|
| `essay`, `wp` | `human/<n>.txt`, `prompts/<n>.txt` | the prompt is the input; the human word count sets the target length |
| `reuter` | `human/<author>/<n>.txt`, `gpt/<author>/headlines/<n>.txt` | reuter's "prompt" is a generated headline, and it lives under `gpt/` whatever variant is being written |

`external/ghostbuster/data/` is gitignored in the fork, so none of this dirties
the submodule pointer. Building from scratch instead (`--*_human`, `--*_prompts`)
needs raw sources that are not in this repo — `data/wp/raw/train.wp_source`, the
Reuters C50 dump, the `qwedsacf/ivypanda-essays` dataset — and costs 3000 API
calls, so it is not what this pipeline does.

### Running it

One config drives every stage: `cfg/ghostbuster/gb_cfg_example.yaml`. Set
`generate.gpt_model` and run:

```bash
# the whole thing, unattended
uv run python epai/ghostbuster/run_full_pipeline.py --yes --cpu-threads 16 --gpu 0

# see every command, run nothing
uv run python epai/ghostbuster/run_full_pipeline.py --dry-run

# plumbing test: 3 items per domain, every LLM call mocked, no keys needed
uv run python epai/ghostbuster/run_full_pipeline.py --gen-limit 3 --debug --yes

# ten-item debug run of all seven stages: mocked generations, ten rows scored and embedded
uv run python epai/ghostbuster/run_full_pipeline.py --yes --debug \
    --gen-limit 10 --score-limit 10 --embed-limit 10 --gpu 0
```

Resuming, since stage 1 is the expensive one:

```bash
--from collect            # the text is generated; audit, copy, prepare, combine, score, embed
--only generate           # fill the gaps: the fork skips files that already exist
--only prepare combine    # rebuild the parquets from a collected tree
--from score              # the corpus is combined; score it and embed it
```

Re-running `generate` is cheap and safe: every loop in the fork skips an output
file that already exists, so an interrupted run resumes by being repeated.

**Starting from a clean slate.** These directories are written to by more than
one run — the fork's tree is shared by every variant, and the collected tree is
copied *into*, never emptied — so `--clean` removes what previous runs left, at
all three levels, before the first step:

```bash
uv run python epai/ghostbuster/run_full_pipeline.py --clean --dry-run   # just the list
uv run python epai/ghostbuster/run_full_pipeline.py --clean --yes --gpu 0
```

| Level | Removed |
|---|---|
| `external/ghostbuster/data/<domain>/` | generator directories that are neither seeded nor this run's |
| `data/ghostbuster-data-<variant>/` | the same, plus domains this config no longer builds and stray `logprobs*` caches |
| `data/parquet/<domain>_<variant>.parquet` | only when `prepare` is among the selected steps |

One rule at every level: *inside the places this config writes, remove what this
config would not produce*. The seeded inputs and this run's own directories are
what a leftover is defined against, so they can never be selected — which is
what makes the flag safe to pass on every run — and nothing outside those
directories is looked at, so another variant's corpus is never at risk. It
prints the list with document counts and asks first (`--yes` answers,
`--dry-run` deletes nothing). Read the list: the fork-tree entries are usually
another model's generated text, which cost API calls and is kept nowhere else.

**Stopping before the GPU work.** Stages 6–7 are the GPU-hours, and they often
belong on a different machine than the generation. `--until` stops the run after
a named stage, and `--from` picks it up:

```bash
# on a laptop: build the corpus, stop before score and embed
uv run python epai/ghostbuster/run_full_pipeline.py --until combine --yes

# later, on the compute server: finish it
uv run python epai/ghostbuster/run_full_pipeline.py --from score --gpu 0 --cpu-threads 16
```

Both bounds are inclusive, so they compose into any window (`--from prepare
--until combine`). `--only` names stages explicitly and cannot be mixed with
either. A run that stops short prints what is left and the command that resumes
it. The same two flags work on pipeline 3 (`--until promote`, then `--from score`).

**Reading a limited run.** `--gen-limit N` (or `generate.limit`) is for testing
the plumbing, and two things about it will surprise you otherwise:

- the fork's `--limit` caps essay/wp by document count but reuter by position in
  the flattened (author, index) list, so `--gen-limit 10` gives ten articles by
  *one* author, not a sample of ten authors;
- the human side is **not** limited. The whole seeded human tree is collected, so
  the corpus holds ~1000 human rows against N generated ones per domain. The
  collect stage prints a warning saying so.

### The collect stage is the gate

`generate.py` catches a failed API call, records it, and carries on — exiting 0
with the output file simply absent. Nothing in the exit code says a corpus lost
rows. So the collect stage audits the text tree before copying it, per
(domain, type):

```
  domain/directory               files  expected  missing   blank
  essay/human                     1000         —        —       6
  essay/gpt56luna_0701A           1000      1000        0       0
```

The counts are taken under **this run's** directories, not under the fork's type
names, so another model's generations sitting in the same tree can neither be
mistaken for this run's output nor be copied into its corpus.

`expected` is the fork's own loop bound (1000 for essay/wp, 20 × authors for
reuter), capped by what is actually seeded — deliberately not the human file
count, which counts differently (`essay/human` has an extra `logprobs/` tree, and
its documents are 1000 against 1000 prompts).

A shortfall or a blank **generated** file stops the run with the command that
fixes it: blanks are refilled by stage 2 (`--only fix-empty`), gaps by re-running
stage 1. `--allow-shortfall` and `--allow-empty` override each, and are the honest
way to say "I know this corpus is unpaired".

Blank **human** files are reported and never fatal — the upstream Ghostbuster
essay release ships six empty documents, which is where the "six rows per corpus
with neither embeddings nor scores" in `combine_domains.py` comes from. Nothing in
this pipeline can refill them; they become rows with empty text, a Null score and
a Null embedding, to be dropped at training time or by combine's
`--drop-incomplete` once the embeddings are in.

Two coupled settings the config validates rather than silently reconciling:

- `prepare.embeddings: false` (the default) leaves `emb_specter2` Null for
  stage 7 to fill over the whole corpus in one pass, which means
  `combine.drop_incomplete` **must** be `"none"` — every other mode drops rows
  with a hole in an `emb_*` column, i.e. all of them.
- `score.observer` and `score.performer` are one detector; set both or neither.

**The per-domain parquets do not survive the combine stage.** Every row of
`essay_<variant>.parquet`, `wp_…` and `reuter_…` is in
`ghostbuster_<variant>.parquet` with a `domain` column saying which file it came
from, and nothing downstream reads them again — so they are deleted once combine
has exited 0 and written its output. Keeping them would store the corpus twice
and leave three corpus-shaped files that a training config can be pointed at by
mistake, long after they have gone stale. Rebuilding one is a prepare run over
the collected text tree (`--only prepare`): local, minutes, no API calls. Set
`combine.remove_domain_parquets: false` to keep them while iterating on a
prepare script.

The scored corpus is a new file, not an in-place update (`score_parquets.py`
overwrites by default; the runner always passes `-o`), so
`ghostbuster_<variant>.parquet` survives as the thing to re-score when the
detector changes. Stage 7 then writes the `emb_*` columns **into** the scored
file, because an embedding is derived from the text alone and a second file would
be the same corpus twice.

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
| `epai/utils/label_utils.py` | The `generated_by` convention: `<model>:<prompt_id>` in a corpus, `<model>_<prompt_id>` as a directory, and the map between them |
| `epai/utils/notify_utils.py` | Per-stage phone notifications from the pipeline runners via ntfy. Best-effort and silent unless `NTFY_PIPELINE_TOPIC` is set |
| `epai/utils/parquet_utils.py` | Parquet save/load helpers |
| `epai/utils/oa_utils.py` | OpenAlex API client (no key; `OPENALEX_MAILTO` for the polite pool) |
| `epai/utils/s2_utils.py` | Semantic Scholar API client |
| `epai/utils/sqlite_utils.py` | SQLite schema, save, and load helpers |
| `epai/utils/train_utils.py` | Missing-value rules (`drop_missing_rows`), feature stacking, split helpers |
| `epai/utils/cd_core.py` | The experiment engine: corpus loading, the grouped split, metrics, seed aggregation, the zero-shot score baselines, and the preflight data/split checks — shared by both runners and the sweep |
| `epai/utils/variant_utils.py` | The two config axes (`feature_sets` × `algorithms`), the `Variant` they produce, and the config validation both runners share |
| `epai/utils/visualization_utils.py` | Plotly and matplotlib plot functions |
| `epai/algs/` | Pluggable `ClfAlgorithm` subclasses (logistic, svm, gradient_boosting, simple_mlp, binox_mlp, fusion_mlp, gated_mlp), each owning its own preprocessing and hyper-parameters |

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
query lists in `queries_example.yaml` so they can be swapped independently. Its
`embed:` block is optional and read only by `run_full_pipeline.py`'s embed step.

### Ghostbuster config (`cfg/ghostbuster/gb_cfg_example.yaml`)

One annotated YAML drives all six stages of pipeline 4. The generator model is
the only value most runs change — every filename downstream is derived from it:

```yaml
dataset_id: "ghostbuster"
generate:
  gpt_model: "gpt-5.6-luna"     # -> variant gpt56luna -> ghostbuster_gpt56luna.parquet
  types: ["gpt_plain"]          # NOT human/prompts — those are seeded by hand
  prompt_ids:                   # one per type; -> generated_by "gpt56luna:0701A"
    gpt_plain: "0701A"          #   and directory data/<domain>/gpt56luna_0701A/
  domains: ["essay", "wp", "reuter"]
  limit: null                   # null = the full 1000 per domain
  fix_empty: true
prepare:
  embeddings: false             # add_embeddings.py fills emb_specter2 later
combine:
  drop_generated_by: []         # keep every label
  drop_incomplete: "none"       # required while embeddings is false
  remove_domain_parquets: true  # delete <domain>_<variant>.parquet once unioned
score:
  text_column: "text"
  batch_size: 1
embed:
  embedders: ["specter2", "mpnet", "luar"]   # null = all 13, incl. billed OpenAI
  text_col: "text"              # note: this script spells the flag --text-col
  batch_size: null              # null = the script's per-model values
  recompute: false              # false fills only the holes
```

`embed.gpu` / `embed.cpu_threads` / `embed.limit` exist too; `--gpu` and
`--cpu-threads` on the command line override both the score and embed stages.

### Classification configs (`clf_cd/`, `clf_id/`, `sweep/` — one sample each)

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
      where: {domain: "essay"}   # which rows of the file this corpus is
    - name: "reuter"
      label: "reuter"
      where: {domain: "reuter"}
feature_sets:                # ONE design matrix per entry. An entry is a column,
  - emb_specter2             # a list of columns stacked together, or a mapping
                             # {columns: [...], id: ..., label: ...}
algorithms:                  # one classifier per entry; crossed with the above
  - logistic                 # a registry key from epai/algs, or a mapping with
                             # id / label / params / binoculars
model:
  n_runs: 3
  test_size: 0.30
  cal_frac: 0.15              # share of the training pool held out for
                              # threshold calibration; see "Where a
                              # threshold comes from" above
```

`parquet_path` (and `drop_classes`) may be set once under `data`, as above, or
per dataset — an entry that names its own wins. Set it once when every corpus is
a domain of one file; name it per entry when the matrix spans several files (two
generators, or the per-corpus parquets).

`where` is a mapping of column → the value (or list of values) to keep, ANDed
across columns. `domain` is nothing special — just the column
`combine_domains.py` happens to write — so the matrix can run across generators,
sources or years instead:

```yaml
datasets:
  - {name: essay_gpt,    where: {domain: essay, generated_by: [human, gpt]}}
  - {name: essay_claude, where: {domain: essay, generated_by: [human, claude]}}
```

That asks whether a detector transfers to an LLM it never saw, with the domain
held fixed. `domain: essay` is kept as shorthand for `where: {domain: essay}`.

Omit `where` to use the whole file — that is what a per-corpus parquet holds, so
configs written before `combine_domains.py` still run. Values are compared as
strings, and a value that is not in the file is a preflight error listing the
ones that are. `drop_classes` is the complement: what to remove rather than what
to keep.

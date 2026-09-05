# workflow/ — Normalized Scientific Impact Prediction

A **module of** episteme-ai (not a fork). It reuses the original `epai` package read-only
(database, embeddings, autoencoder, parquet IO) and adds everything specific to
field-normalized impact prediction. **Nothing outside `workflow/` is modified** — see
[`CLAUDE.md`](CLAUDE.md) for the folder-scoped editing rules.

This document is paired with [`WORKFLOW.drawio`](WORKFLOW.drawio) (the same architecture as
a diagram). For the statistics/equations behind every metric and target, see
[`mathematical_documentation.md`](mathematical_documentation.md). For the full parameter
reference (every `config.yaml` / `.env.workflow` key), see
[`WORKFLOW_REFERENCE.md`](WORKFLOW_REFERENCE.md).

## Quickstart — the final two-step operation

The system runs in **two separate sessions**: data collection first (network-bound),
then the training queue (GPU-bound, fully offline once data exists).

The canonical launchers now live in `workflow/deploy_and_run.sh` and
`workflow/deploy_and_run.ps1`; use `workflow/stop_deploy_and_run.sh` or
`workflow/stop_deploy_and_run.ps1` to stop the detached job and reclaim resources.

**Step 1 — data collection** (run once, or whenever you want a bigger corpus):

```bash
# 1a. fresh multi-source fetch -> raw.db (KEEPS incomplete records; dedup only)
python -m workflow.src.main_extend_db bootstrap --target-rows 100000

# 1b. raw.db + legacy DBs -> extended.db (huge archive, provenance in `source`)
#     -> gap-fill missing metadata -> training.db (complete records only)
#     -> holdout benchmark parquet (offline benchmark for every later run)
python -m workflow.src.main_extend_db extend

# 1c. point training at the strict DB: set EPAI_DB=training in .env.workflow

# 1d. (optional) pull external impact-prediction systems for comparison
#     (NAIP v2 repo + weights into gitignored workflow/external/; needs git+uv+HF_TOKEN)
python -m workflow.src.setup_external_models

# 1e. (optional) dataset statistics -> workflow/data/analysis/
python -m workflow.src.main_analyse_dataset --db extended
```

**Step 2 — the training queue** (sized for one 16 GB GPU, ~4-6 h + margin):

```bash
python -m workflow.run_all_workflows
```

Every run writes its plots/benchmarks the moment it finishes
(`results/<run>/aggregate/`); cross-run comparisons (leaderboard, benchmark
overview, LODO cross-domain matrix) land in `results/summary/` at queue end.

### Pushing to GitHub — one-time checklist

Everything generated is gitignored (`workflow/data|models|results|external`,
`*.log`, both `.env` files). Two legacy problems must be cleaned from the git
index once before the first push (files >100 MB are rejected by GitHub):

```bash
git rm --cached store/data/sql/0512_A.db store/data/sql/_deferred_big_db/0512_A.db
git rm --cached store/data/parquet/0512_A.parquet
git rm --cached workflow_run.log extend_db_v2.log
git commit -m "untrack large data + logs"
```

(The files stay on disk; `store/data/` and `*.log` are ignored from now on.)
For containers/VMs, the repo-root `Dockerfile` builds a CUDA-ready image; mount
`workflow/data` + `workflow/results` as volumes and pass secrets via `--env-file`.

## The two-file control surface

You configure everything by editing exactly **two files**:

| File | Purpose | Examples |
|---|---|---|
| [`config.yaml`](config.yaml) | **WHAT to train** — workflow type, target, class count, model architecture, hyperparameters, and the `runs:` queue of training setups | `workflow`, `num_classes`, `model_arch`, `hidden`, `lr`, `runs:` |
| [`.env.workflow`](.env.workflow) | **HOW this machine runs it** — parallelism, GPU batch size, cache-force flags, browser behaviour | `PARALLEL_JOBS`, `PARALLEL_LOAD_TARGET`, `EMBED_BATCH_SIZE`, `FORCE_EMBED` |

Secrets (OpenAI / S2 / HF / MIRAI keys) live in the repo-root `.env` — never in either file.
**One deliberate exception:** the LLM-as-judge workflow and the extend-phase LLM auto-fill
read their credentials (`LLM_JUDGE_PROVIDER`/`LLM_JUDGE_MODEL`/`LLM_JUDGE_API_KEY`/
`LOCAL_LLM_URL`) from `.env.workflow`, so this module stays fully configurable from that one
file without touching the repo-root `.env` — see `.env.workflow.example`.
`config.yaml` is ordered most-changed-first: section 1 holds the knobs you touch every
training (workflow, seeds, class count, target); sections 2-6 hold model, encoder, text
transformer, features and data/split; section 7 is the `runs:` queue with a commented
copy-paste **TEMPLATE** entry — duplicate it, rename it, and you have a new experiment.

## The two entry points

```bash
python -m workflow.run_pipeline           # ONE workflow: the top-level keys of config.yaml
python -m workflow.run_all_workflows      # EVERY entry of config.yaml's runs: queue
```

- `run_pipeline` runs the topology named by `workflow:` end-to-end (or `--from N --to N`
  for a phase range, `--force` to ignore reuse shortcuts). `workflow: all` runs all eight
  topologies once, sharing the common phases.
- `run_all_workflows` expands the `runs:` queue: each entry names a `workflow` (or a list
  of workflows sharing the same parameters) plus any per-run overrides of the top-level
  keys. Every run gets its own auto-indexed folder (`emb_clf_1`, `emb_clf_2`, …) and its
  own log file under `results/`. Overrides are passed as plain dicts — nothing mutates the
  environment. `--fresh` wipes `results/` first (default keeps old runs; numbering
  continues). Expensive shared phases are computed once and reused between runs whose
  data/embedding parameters actually match (the label phase is always re-run because it
  depends on the per-run class configuration).

Parameter sweeps are just several `runs:` entries with different overrides — same
mechanism, one file.

## The 10 workflow topologies

`WORKFLOWS` in [`src/iwf/config.py`](src/iwf/config.py) is the single table every module
derives from — task, feature set and phase list per topology:

| workflow | task | model input (features) | phases |
|---|---|---|---|
| `emb_clf` | classification | encoder latent `z` | 0 1 2 3 4 5 6 7 |
| `emb_reg` | regression | `z` | 0 1 2 3 4 6 7 |
| `emb_meta_clf` | classification | `z` + curated metadata/novelty/feasibility | 0 1 2 3 4 5 6 7 |
| `emb_meta_reg` | regression | same | 0 1 2 3 4 6 7 |
| `emb_all_meta_clf` | classification | `z` + **ALL** numeric columns (intentional-leakage upper bound) | 0 1 2 3 4 5 6 7 |
| `emb_all_meta_reg` | regression | same | 0 1 2 3 4 6 7 |
| `tfm_clf` | classification | raw title+abstract into a fine-tuned transformer | 0 1 5 6 7 |
| `tfm_reg` | regression | same | 0 1 6 7 |
| `llm_clf` | classification | LLM-as-judge: RAG store of precedents, one LLM call per record | 0 1 2 5 6 7 |
| `llm_reg` | regression | same | 0 1 2 6 7 |

The `emb_all_meta_*` pair deliberately bypasses the leakage guard (set automatically from
the workflow type) to measure the performance ceiling when target-derived columns are fed
to the model. The `tfm_*` pair skips embed/encode/reduce entirely — it never touches the
encoder artifact. The `llm_*` pair runs the embed phase (it retrieves over the frozen
specter2 embedding) but skips encode/reduce/training: there is no trained network and no
training step at all. Phase 6 ([`src/main_train_judge.py`](src/main_train_judge.py)) first
builds a complete RAG store, deterministically and with no LLM call, from up to
`judge_max_train_papers` TRAIN papers' true impact values (0 = every TRAIN paper); it then
judges every record — TRAIN, VAL, and TEST alike — exactly once against that one store: up
to 5 similar precedents (cosine similarity, excluding a paper's own exact abstract match)
plus a fixed rubric are sent to an OpenAI-compatible LLM
(`LLM_JUDGE_PROVIDER`/`LLM_JUDGE_MODEL`/`LLM_JUDGE_API_KEY`/`LOCAL_LLM_URL`, all in
`.env.workflow` — this workflow module reads its LLM credentials from `.env.workflow` only,
not the repo-root `.env`), which the system prompt explicitly tells to treat the precedents
as loose calibration only, never to copy or average their scores. With no endpoint
configured every LLM call is a no-op and the run falls back to a deterministic kNN-mean
judge instead of erroring.

## Phases (ids used by `--from` / `--to`)

| # | Name | Script | Reads | Writes |
|---|---|---|---|---|
| 0 | fetch | `src/main_fetch.py` | epai DB (primary) + OpenAlex/S2 (gap-fill) | `data/parquet/<id>_raw.parquet` |
| 1 | prepare | `src/main_prepare.py` | `<id>_raw.parquet` | `<id>.parquet` (+ `target`, `meta_*`, `split`) |
| 2 | embed | `src/main_embed.py` | `<id>.parquet` | `<id>_emb.parquet` (`emb_*` cols) |
| 3 | encoder | `src/main_encoder.py` | `<id>_emb.parquet` | `models/<id>_encoder.pt` + `.json` (k\*) |
| 4 | reduce | `src/main_reduce.py` | `<id>_emb.parquet` + encoder | `<id>_z.parquet` (col `z`) |
| 5 | label | `src/main_label.py` | `<id>_z.parquet` | `+ class_label` |
| 6 | train | `src/main_train.py` (emb) / `src/main_train_tfm.py` (tfm) | `<id>_z.parquet` / `<id>.parquet` | `results/<wf>_N/seed_<i>/` |
| 7 | finalize | `iwf/workflow_results.finalize_workflow` | `seed_<i>/result.json` | `aggregate/` (metrics + all plots + description) |

Phase 6 is ONE script for all embedding workflows: `main_train.py` derives task
(clf/reg) and feature set (z / meta / all_meta) from the workflow type and calls the
shared `run_experiment`. Text workflows use `main_train_tfm.py` (pretokenized fp16
fine-tune with warmup + grad accumulation).

One more standalone script is not part of any phase list:
`src/main_predict_generated.py` scores LLM-generated papers (MIRAI/ADP) with the newest
trained z-only regressor and compares their predicted-impact distribution to the human
test split.

## Layout

```
workflow/
├── config.yaml            WHAT to train (single training config + runs: queue)
├── .env.workflow          HOW the machine runs it (parallelism, GPU, force flags)
├── run_pipeline.py        entry point: one workflow
├── run_all_workflows.py   entry point: the runs: queue
├── README.md (this), WORKFLOW.drawio, WORKFLOW_REFERENCE.md,
│   mathematical_documentation.md, CLAUDE.md
├── data/, models/, results/     generated artifacts (safe to delete + regenerate)
└── src/
    ├── main_fetch.py … main_train.py, main_train_tfm.py, main_predict_generated.py
    ├── iwf/                     the shared library
    │   ├── config.py            Settings + config.yaml loading + the WORKFLOWS table
    │   ├── paths.py             every filesystem location
    │   ├── registry.py          embedder/encoder/model/text-arch lookup by name
    │   ├── parallel.py          worker/GPU budget (reads .env.workflow)
    │   ├── progress.py          live multi-run terminal display
    │   ├── targets.py           the 4 impact targets + class binning
    │   ├── workflow_results.py  per-run folders, aggregation, model description, comparison
    │   ├── data/                identity, parquet IO + THE shared split, corpus sources
    │   ├── embedding/           frozen embedders (epai + scincl/e5)
    │   ├── encoder/             AE/PCA/UMAP/identity + latent-dim (k*) selection
    │   ├── features/            metadata, novelty/feasibility, feature_builder (+ leakage guard)
    │   ├── models/              architectures (mlp/resmlp/ft_transformer), heads,
    │   │                        text_transformers (all 4 backbones in one module)
    │   ├── training/            trainer (torch loop), experiment (multi-seed),
    │   │                        text_trainer (transformer fine-tune)
    │   └── evaluation/          metrics.py (the ONE metric implementation),
    │                            plots.py (the ONE plot module, all figures)
    └── tests/smoke.py           offline smoke suite (26 checks, no keys/downloads)
```

Deliberately centralized (one implementation, used everywhere):

- **`evaluation/metrics.py`** — both the MLP heads and the text trainer report through the
  same two functions, so every seed of every workflow has identical metric keys.
- **`evaluation/plots.py`** — every figure in the project lives here; classification and
  regression get the same suite with only the task-appropriate variants.
- **`training/experiment.py`** — the one multi-seed loop for all six embedding workflows;
  seeds run in parallel under the `.env.workflow` machine budget.
- **Uniform results layout** — every workflow (emb and tfm alike) writes
  `results/<wf>_N/seed_<i>/result.json` in the same format; aggregation and plotting never
  special-case a workflow type.

## Install / prerequisites

No new dependencies. Everything used (`torch`, `scikit-learn`, `scipy`, `transformers`,
`umap-learn`, `requests`, `pandas`, `pyarrow`, `matplotlib`, `plotly`, `dotenv`; `peft`
only for the LoRA arches) is in the repo `pyproject.toml`. Run from the repo root (where
`epai` is importable). The expensive stages (fetch, ~1h embed) are skipped when their
output already exists (`FORCE_FETCH` / `FORCE_EMBED` in `.env.workflow` to rebuild).

## Offline test (no API keys, no model downloads)

```bash
uv run python -m workflow.src.tests.smoke      # exit 0 = all 26 checks passed
```

Exercises config loading + per-run overrides + the class-count knobs, identity, the shared
split, all targets, class binning, metadata, feature builder + neutral scaler + leakage
guard (incl. the intentional bypass), intrinsic-dim, encoders, dim selection, the torch
heads for every registered architecture, metrics, novelty, full `run_experiment` runs for
reg/meta/clf (save + reload), the workflow table, the registry, the progress display, a
mocked text-transformer run, and the aggregation pipeline. All outputs go to a throwaway
temp dir.

## Results layout (identical for every workflow)

```
results/<wf_type>_N/
├── run_config.json          exact Settings snapshot of the run
├── experiment.json          feature list / leakage-guard state (emb workflows)
├── seed_<i>/
│   ├── result.json          flat test metrics + per-split y_true/y_pred + history
│   ├── model/               model.joblib + scaler.joblib (emb) — or —
│   └── checkpoint/          HF checkpoint + head.pt (tfm)
└── aggregate/
    ├── metrics_summary.json mean ± std ± min/max of every scalar metric across seeds
    ├── description_<wf>.md  auto-generated model card (most-important params first)
    └── *.png / *.html       the full plot suite (below)
results/cross_workflow_comparison.{json,png}   scan of every aggregate, reg + clf panels
results/<wf_type>_N.log                        full per-run log (run_all_workflows)
```

## Evaluation plots (phase 7, identical suite for every workflow)

All filenames are suffixed with the workflow type, so results never collide:

| File | What it shows | Task |
|---|---|---|
| `correctness_bars_<wf>.png` | % correct per split (train/val/test), mean +- std across ACCEPTED seeds (error bars), a chance-level dashed line, item counts + seed-count note in labels; regression is binned by the same class thresholds as the classifier | both |
| `learning_curve_<wf>.png` | Training loss + validation metric vs epoch (best seed), one merged legend | both |
| `confusion_matrix_<wf>.png` | Predicted vs actual impact class, row-% averaged across ACCEPTED seeds (regression binned identically) — auto-adjusts to any class count | both |
| `class_boxplot_<wf>.png` | Per-class F1 spread across accepted seeds | clf |
| `spearman_distribution.png`, `predicted_vs_actual.png` (best seed) | Seed spread, calibration scatter | reg |
| `mse_bars.png` | Split MSE, mean +- std across accepted seeds | reg |
| `umap3d_correctness_<wf>.{html,png}`, `umap3d_classes_<wf>.{html,png}` | 3-D UMAP of `z`: gold=correct/purple=wrong; colour per class | clf |
| `umap3d_error_<wf>.{html,png}` | 3-D UMAP, continuous gold→purple by per-sample error | reg |
| `umap3d_domain_*_<wf>.{html,png}` | The same, split per research domain (grid PNG + one HTML per domain) | both |
| `nn_weights_<wf>.{html,png}` | MLP connection-strength diagram (metadata inputs red, z inputs blue); only weights above 15% of the layer max are drawn | both (MLP/ResMLP) |
| `description_<wf>.md` | Auto-generated model card: run summary → architecture (incl. actually-trained layer sizes) → encoder → hyperparameters → data/split → features → full snapshot | both |

See `mathematical_documentation.md` Part 4 for the equations behind each figure.

## Key engineering guarantees

- **One shared split** for every stage: computed once in phase 1 (seeded, stratified),
  stored as the `split` column, read by all later phases. Change `split_seed` to
  regenerate a different-but-shared split.
- **Encoder** trains on `train + val` (unsupervised); every head trains on `train`,
  early-stops on `val`, and `test` is only ever used for output/statistics.
- **Identity & joins:** every artifact carries a canonical `paper_id`; all cross-table
  combination is a keyed join on it (never row order). The epai DB is the **primary**
  paper source (read-only); OpenAlex/S2 fill gaps.
- **Class balancing:** the classifier loss supports `class_weight: balanced`
  (inverse-frequency) and `effective` (Cui 2019 class-balanced) — automatic no-ops when
  the configured class distribution is even, plus optional focal loss.
- **Class count is one knob:** `num_classes` / `class_distribution` / `class_thresholds`
  in config.yaml (2..25 classes); labels, training and every plot auto-adjust.
- **Robust missing data:** any absent feature becomes a neutral `0.0` (standardized mean)
  plus a `<feature>_missing` mask bit, so training never breaks on incomplete rows.
- **Leakage guard:** `year`/`citation_count`/`influential_citation_count`/`domains` build
  the target and are rejected as features — except in the `emb_all_meta_*` workflows,
  where the bypass is set automatically as the intentional upper-bound test.
- **No hidden state between runs:** per-run overrides travel as dicts into `Settings`;
  the OS environment is never mutated, so queued runs cannot contaminate each other.

## Extensions (baselines, novelty/feasibility, data extension)

### Frozen baseline scorers (`external_impact:` in config.yaml)
`naip` (NAID v2.1 HF Space), `specter2_knn` (frozen SPECTER2 + kNN percentile transfer
from TRAIN), `llm_judge` (pinned zero-shot LLM 0-100 rating, prompt v1, temperature 0),
`venue_ais` (UEFISCDI/JCR Article Influence Score of the venue), plus the `forecite` /
`cited_potential` REST stubs. Every configured scorer is (a) fused as a metadata feature
in the `emb_meta_*` / `emb_all_meta_*` workflows and (b) dumped to
`results/<run>/baselines.parquet` for phase 7, which renders per run:
`aggregate/baselines/baseline_agreement_<run>.png` (% exact / ±1 class agreement with
the true class + weighted kappa), `baseline_correlation_<run>.png` (pairwise Spearman
heatmap + ours-vs-NAIP scatter) and `baseline_metrics.csv`. After a `run_all_workflows`
queue, `results/summary/baseline_leaderboard.png` compares every trained model against
every baseline in one figure.

### Novelty & feasibility (plan §3)
`novelty_kind: spherical` (default) is the geodesic kNN distance on the unit hypersphere
(Shibayama-style), percentile-ranked to uniform [0,1]; `cosine_max` is the legacy metric.
Feasibility is a precedent/density/maturity composite over the latent space + years
(literature-supported-claim proxy), also uniform [0,1]. Phase 7 writes
`aggregate/nf/nf_correlation_matrix.png` (Spearman + significance stars + partial
correlations controlling for venue/h-index) and `nf_binned_impact.png` (mean impact per
decile, the inverted-U check); a queue additionally gets
`results/summary/nf_ablation_delta.png` comparing runs with/without these features.

### Extended database (`python -m workflow.src.main_extend_db`)
Grows the workflow-owned `workflow/data/sql/extended.db` (professor's exact schema +
nullable impact/provenance columns; `store/` stays read-only) toward
`fetch_target_rows` (default 100k) real papers: copy of the original DB, UEFISCDI
red/yellow-zone journal seeds (drop the yearly AIS xlsx lists into
`workflow/data/uefiscdi/`; run with `--uefiscdi-hints` for the download pages), a
domain-balanced OpenAlex crawl (abstracts reconstructed from the inverted index; fwci +
citation_normalized_percentile arrive as free impact labels), and an S2 gap-fill of
influential citations + SPECTER embeddings by DOI. All phases are cursor-resumable.
Point the pipeline at the result with `EPAI_DB=extended` in `.env.workflow`.

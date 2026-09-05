# CLAUDE.md — `workflow/` is the working space

**This file is the primary guidance for the repo. The repo-root `CLAUDE.md` was removed;
all workflow work is documented here.**

## 0. The one rule that governs everything

<!-- ============================================================================
     SECTION 0 IS FROZEN. NEVER EDIT, REWORD, SHORTEN OR "IMPROVE" ANYTHING
     BETWEEN THIS MARKER AND THE "END OF FROZEN SECTION" MARKER BELOW.
     Every other section of this file is meant to be kept up to date (see §1).
     ============================================================================ -->

> **`workflow/` is the real working space. Do your work here.**
>
> The ONLY thing that may be added to the **repo root** is a Python **dependency**
> requirement (a package the code imports) — e.g. a new entry in the root `pyproject.toml`
> / `uv` lockfile / `requirements`. Adding a dependency there is allowed **because the venv
> is shared repo-wide** and `uv` resolves from the root.
>
> **Everything else — every new module, script, config, data file, model, result, plot,
> note, doc — MUST be created inside `workflow/`.** Never write generated artifacts,
> scratch files, or new code to the repo root or anywhere outside `workflow/`.

> **NEVER commit and NEVER push to GitHub.** Do not run `git commit`, `git push`, or any
> command that writes to git history or a remote — not even when a task looks finished, and
> not on a new branch. Only ever change files in the working tree; the user handles all
> committing and pushing themselves.

Corollaries:
- `epai/` (repo root package) is the **read-only base library** — import and reuse from it,
  never edit it.
- The professor's databases under `store/data/sql/*.db` are **read-only**. The
  workflow-owned, writable databases live under `workflow/data/sql/` (`raw.db`,
  `extended.db`, `training.db`), all built by `main_extend_db` and safe to delete/regenerate.
- Secrets/API keys live in the repo-root `.env` (never modified). The one deliberate
  exception: the LLM + web-search credentials read from `workflow/.env.workflow` (see §4).
  Read keys via `config.api_key("NAME", ...)` — it ignores placeholders (`XXX`, `your-…`) so
  a fetcher falls back to its keyless tier until a real key is set. Wired for `OPENALEX_API_KEY`,
  `SEMANTIC_SCHOLAR_API_KEY`/`S2_API_KEY`, `HF_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.
- Don't install packages or run long scripts unless asked. When a genuinely new dependency
  is needed, add it to the root `uv`/`pyproject` and say so — that is the single sanctioned
  root-level change.

<!-- ================== END OF FROZEN SECTION — EDIT NOTHING ABOVE ================== -->

---

## 1. Keep this file current — it is the working memory of the project

This file is the first thing to read and the last thing to update. Everything below §0 is a
**living description of what the code currently does**, not a historical record.

**Update the relevant section IN THE SAME CHANGE that makes it stale.** Concretely:

| When you… | Update |
|---|---|
| add/remove/rename a module, function, class, or CLI | §5 (source map) |
| add/remove/rename a `config.yaml` key or `.env.workflow` variable | §4 (parameters) |
| change what a phase does, or the phase order | §3 (pipeline) |
| change which files a run writes, or their names | §6 (outputs) |
| change the launcher scripts or how a run is started | §2 (how to run) |
| change the current experiment (data size, sweep step, active queue) | §7 (current state) |
| establish a new convention or invariant | §8 (conventions) |

Rules for these updates:
- **Describe the present, delete the past.** Do not write "changed from X to Y" here — the
  git history holds that. This file says what IS.
- **Never edit §0** (see the frozen markers). If a §0 rule ever looks wrong, say so in chat
  and let the user decide; do not touch it.
- If you notice a section already describes something the code no longer does, fix it in
  passing — a stale line here is a bug, because the next session trusts this file.

---

## 2. How to run the system

### 2.1 The one principle: the launcher is fixed, the config decides what happens

**There is exactly one way to start a run — the launcher script — and it never takes
"skip this step" arguments.** Every step is gated by `ENABLE_*` flags in `.env.workflow`.

> **This is the rule to apply when the user asks for a behaviour change:**
> a request like *"don't collect more OpenAlex data this time"* is a **config change**
> (`ENABLE_OPENALEX=false`), never an instruction to run a different command by hand. Set the
> flags so the standard launcher does the right thing end to end, then tell the user to run
> the standard launcher. The same applies to every other step: skip the DB build
> (`ENABLE_EXTEND_DB=false`), skip statistics (`ENABLE_ANALYSE=false`), skip training
> (`ENABLE_TRAINING=false`), train without touching the network (all per-source flags false).
>
> After changing any flag, re-check that the launcher's resulting step list is what the user
> actually asked for, and say which flags you changed and what the launcher will now do.

### 2.2 Automatic run (preferred)

| Platform | Start | Stop |
|---|---|---|
| Windows | `workflow\deploy_and_run.ps1` | `workflow\stop_deploy_and_run.ps1` |
| Linux/macOS | `workflow/deploy_and_run.sh` | `workflow/stop_deploy_and_run.sh` |

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "<repo>\workflow\deploy_and_run.ps1"
```

- Runs **detached** by default (safe to close the terminal / sleep the machine); pass
  `-Foreground` to keep it attached to the console.
- `-TargetRows <N>` is passed to the **bootstrap** step only (default 100000); it is
  irrelevant when `ENABLE_BOOTSTRAP=false`.
- Logs: `workflow/logs/pipeline_<timestamp>.log` (+ `.err`), PID in `workflow/logs/pipeline.pid`,
  live one-line-per-workflow dashboard in `workflow/logs/progress.status`, each line labelled
  by the FULL `<workflow_type>:<queue_entry_name>` (e.g. `emb_reg:demo_sklearn_linreg`, never
  truncated to the entry's `demo` prefix).
- Both launchers run `uv sync --frozen` first, then the same four steps:

| Step | Gate | Command it runs |
|---|---|---|
| 1/4 bootstrap | `ENABLE_BOOTSTRAP` + any of `ENABLE_OPENALEX/CROSSREF/ARXIV/S2` | `main_extend_db bootstrap --target-rows N --sources …` |
| 2/4 extend | `ENABLE_EXTEND_DB` | `main_extend_db extend --sources …` (sources from `ENABLE_COPY/PDF/UEFISCDI/OPENALEX/S2/FILL/LLM_FILL`; **all off → `none`**, which only re-derives `training.db` + report + local benchmarks, no network) |
| 3/4 analyse | `ENABLE_ANALYSE` | `main_analyse_dataset --db $ANALYSE_DB --train-db $EPAI_DB` |
| 4/4 train | `ENABLE_TRAINING` | `run_all_workflows` (the whole `runs:` queue) |

### 2.3 Manual runs (debugging / partial work)

Two top-level entry points — everything under `src/` is a phase implementation, not a CLI to
call directly for a normal run:

```bash
python -m workflow.run_pipeline        # ONE workflow, from the top-level config.yaml keys
python -m workflow.run_all_workflows   # EVERY entry in the runs: queue. WIPES results/ first.
```

Other user-invoked utilities:

| Command | Purpose |
|---|---|
| `python -m workflow.src.main_extend_db bootstrap\|extend [--sources …] [--target-rows N]` | build/grow `raw.db` / `extended.db` → `training.db` |
| `python -m workflow.src.main_analyse_dataset --db <db> [--train-db <db>]` | dataset statistics → `workflow/data/analysis/` |
| `python -m workflow.run_plot_refresh` | re-render every plot from existing `results/` (no retraining) |
| `python -m workflow.src.main_predict_generated` | score LLM-generated papers with the latest trained regressor |
| `python -m workflow.src.setup_external_models` | install external baseline systems (NAIP weights etc.) |
| `uv run python -m workflow.src.tests.smoke` | the offline suite — no network, keys, or model downloads |

`run_pipeline` can run a phase subset: `--start N --end M` (phase ids in §3).

---

## 3. The pipeline: phases, workflows, and dispatch

### 3.1 Phases (`PHASE_NAMES` in `run_pipeline.py`)

`0 fetch → 1 prepare → 2 embed → 3 encoder → 4 reduce → 5 label → 6 train → 7 finalize`

| # | Name | Script | Does |
|---|---|---|---|
| 0 | fetch | `main_fetch.py` | load the corpus from `EPAI_DB`, cap to `max_papers`, optional OpenAlex enrich → `raw` parquet |
| 1 | prepare | `main_prepare.py` | compute every `target_*` column, pick the active `target`, build `meta_*`, derive `strat_label`, compute **the one shared split** → `prepared` parquet |
| 2 | embed | `main_embed.py` | frozen title+abstract embeddings (`emb_<name>` columns) → `embedded` parquet |
| 3 | encoder | `main_encoder.py` | select latent dim `k*` and fit the encoder → `models/encoder_<dataset_id>.joblib` |
| 4 | reduce | `main_reduce.py` | apply the frozen encoder → `z` column → `reduced` parquet |
| 5 | label | `main_label.py` | derive `class_label` from `target` (classification workflows only) |
| 6 | train | `main_train.py` / `main_train_tfm.py` / `main_train_judge.py` | multi-seed training → `results/<wf>_N/seed_<i>/` |
| 7 | finalize | `workflow_results.finalize_workflow` | aggregate seeds, all plots, baselines, cross-domain, benchmarks, model description |

`SHARED_PHASES = {0,1,2,3,4}` are computed once and reused across queue entries with matching
data/embedding parameters. Phase 5 is deliberately **not** shared (cheap, and depends on the
per-run class configuration).

### 3.2 Workflow topology (`WORKFLOWS` in `config.py` — the single source of truth)

| workflow | task | features | phases | phase-6 script |
|---|---|---|---|---|
| `emb_clf` | clf | z | 0-7 | `main_train` |
| `emb_reg` | reg | z | 0-4,6,7 | `main_train` |
| `emb_meta_clf` / `emb_meta_reg` | clf/reg | z + curated scalars | as above | `main_train` |
| `emb_all_meta_clf` / `emb_all_meta_reg` | clf/reg | z + ALL numerics (deliberate leakage upper bound) | as above | `main_train` |
| `tfm_clf` | clf | raw text | 0,1,5,6,7 | `main_train_tfm` |
| `tfm_reg` | reg | raw text | 0,1,6,7 | `main_train_tfm` |
| `llm_clf` | clf | RAG store | 0,1,2,5,6,7 | `main_train_judge` |
| `llm_reg` | reg | RAG store | 0,1,2,6,7 | `main_train_judge` |

`workflow: all` runs every topology once. Dispatch is by `WORKFLOWS[wf]["features"]`
(`z`/`meta`/`all_meta` → `main_train`, `text` → `main_train_tfm`, `rag` → `main_train_judge`).

### 3.3 Database building (separate from training)

`main_extend_db`:
- **`bootstrap`** — fresh multi-source crawl (openalex, crossref, arxiv, s2) → `raw.db`,
  keeping incomplete rows.
- **`extend`** — `raw.db` + legacy `store/data/sql/*.db` → `extended.db` → `training.db`.
  Selectable steps via `--sources`, each additionally gated by its `ENABLE_*` flag:
  `copy pdf uefiscdi openalex s2 fill llm_fill`; then **always**: report → derive
  `training.db` (strict: title+abstract+year+citations, `TRAIN_YEAR_FROM/TO`,
  `MAX_TRAINING_ROWS`, LLM-abstract exclusion, unknown-domain exclusion) → build local
  benchmarks.
- **`--sources` with nothing enabled** (or `none`) = **no DB is modified**: `extend` only
  re-derives `training.db` from what is already stored. This is the no-network path.
- **OpenAlex daily-limit survival** (`fetchers.openalex.collect_new_papers`): OpenAlex meters
  a daily budget (measured 2026-07-29: 10000 credits, 1 credit/request, resets midnight UTC).
  On exhaustion it idles `FETCH_RATE_LIMIT_WAIT_HOURS` and resumes, up to `fetch_wait_days`
  times, holding minimal RAM. It stops early WITHOUT waiting when the query is exhausted or
  stalls on duplicates (`FETCH_STALL_LIMIT`). Dedup (`base.Deduper`) means already-stored
  works are never re-inserted. While waiting, an `on_wait` hook runs the LLM domain/abstract
  fill so the idle time is useful.

### 3.4 LLM auto-fill + web search (the data-repair step)

`fetchers/llm_fill.py` runs in the extend `llm_fill` step. All calls are **best-effort** — an
unconfigured/down endpoint just skips the row (idempotent WHERE clause, retried on a later
run) and never crashes the pipeline.

- **Domain fill** (`llm_fill_domains`, gate `ENABLE_LLM_FILL`): picks a field from
  `ALLOWED_DOMAINS` for rows with a title but no domain. Harmless — never touches `source`,
  always kept in training. Cap: `llm_fill_domain_max`.
- **Abstract fill** (`llm_fill_abstracts`, gate `ENABLE_LLM_ABSTRACT_FILL`): reconstructs a
  missing abstract/year. **Web-search-first** when `ENABLE_LLM_WEBSEARCH=true` — looks the
  paper up by title in free, no-key scholarly APIs (`fetchers/websearch.py`: Crossref +
  Europe PMC), hands any confident hit to the LLM to **verify + clean** (never trusted
  blindly); only with no hit does the LLM answer from its own knowledge. Every filled
  abstract sets `source='llm'` so it is labelled and excludable. Cap: `llm_fill_abstract_max`.
- **Training inclusion of LLM abstracts**: `use_llm_abstracts_for_training` (config.yaml;
  falls back to `.env.workflow USE_LLM_ABSTRACTS_FOR_TRAINING`), enforced in BOTH
  `derive_training` and the phase-0 loader. Domain fill is unaffected.

---

## 4. The two-file control surface — every parameter

You configure a session by editing **exactly two files**, both in `workflow/`:

| File | Governs | Read by |
|---|---|---|
| `config.yaml` | **WHAT** to do: topology, target, classes, model/hyperparams, data & split, extension knobs, the `runs:` queue | `iwf/config.py` → `Settings` |
| `.env.workflow` | **HOW** this machine runs it: parallelism, GPU batch, cache flags, `ENABLE_*` toggles, DB selection, LLM endpoint + credentials | `iwf/config.get_*` accessors |

Rule of thumb: if a knob changes *the science / what a run produces or costs*, it belongs in
`config.yaml` and may be overridden per-run inside a `runs:` entry. If it only changes *how
this machine executes* (speed, memory, cache, which endpoint, which steps run), it belongs in
`.env.workflow`.

- Per-run overrides travel as plain dicts into `Settings` via `load_settings(..., overrides=)`
  — **never** mutate `os.environ` to pass a parameter.
- Any key in config.yaml sections 1-6b can also be used **inside a `runs:` entry** to override
  it for that run only.
- `config.yaml.full_backup` holds the original 12-model flagship queue; restore it for a full
  training campaign.

### 4.1 `config.yaml` — section by section

**1. What to run**

| Key | Values | Meaning |
|---|---|---|
| `workflow` | any of §3.2, or `all` | which topology `run_pipeline` executes |
| `n_runs` | int ≥ 1 | seeds per run (model seed varies; the data split never does) |
| `model_seed` | int | base seed; seed *i* trains with `model_seed + i` |
| `seed_outlier_enabled` | bool | turn the outlier-seed exclusion on/off |
| `seed_outlier_min_n` | int ≥ 2 | exclusion only activates at ≥ this many seeds |
| `seed_outlier_gap_pct` | float > 0 | drop a seed whose test correctness is this many points below the **median of the others** |
| `class_distribution` \| `num_classes` \| `class_thresholds` | see below | **set exactly ONE**; resolves the class count (2..25) |
| `class_scheme` | `percentile` \| `css` \| `quantile` | how thresholds become cut-point VALUES: fixed absolute values / Glanzel-Schubert clusters / fractions of this run's own data |
| `class_names` | list | optional; auto-derived `Qn..Q1` when omitted |
| `target_kind` | `field_year_percentile` \| `fwci_percentile` \| `normalized_percentile` \| `bayesian_impact` | the impact target |

Class-count knobs: `class_distribution: [15,35,35,15]` (per-class % shares) ·
`num_classes: 4` (equal-sized) · `class_thresholds: [0.25,0.5,0.75]` (manual boundaries,
N → N+1 classes).

**2. Model & training** (the impact head; ignored by `tfm_*`)

| Key | Values | Meaning |
|---|---|---|
| `model_arch` | `mlp` \| `resmlp` \| `ft_transformer` \| `logreg` \| `linreg` | head architecture. `logreg` = sklearn LogisticRegression (**clf only**), `linreg` = sklearn LinearRegression (**reg only**); both ignore `hidden/dropout/lr/max_epochs/patience/batch_size/loss` |
| `hidden` | list[int] | hidden layer sizes |
| `activation` | `relu\|gelu\|tanh\|htanh\|sigmoid\|leaky_relu\|elu\|silu\|mish` | see `activations.py` |
| `dropout`, `lr`, `weight_decay` | float | standard |
| `max_epochs`, `patience`, `batch_size` | int | training length / early stop / batch |
| `class_weight` | `none` \| `balanced` \| `effective` | classifier only. `effective` = Cui 2019; **the sklearn `logreg` baseline maps only `balanced`, and silently treats `none`/`effective` as sklearn's default** |
| `effective_beta` | float | β for the `effective` scheme |
| `loss` | `ce` \| `focal` | classifier loss |
| `focal_gamma`, `label_smoothing` | float | focal focusing / softmax smoothing |

**3. Embedding & encoder** (`emb_*` only)

| Key | Values | Meaning |
|---|---|---|
| `embedder` / `embedders` | `specter2` \| `specter2_base` \| `specter2_clf` \| `specter2_reg` \| `specter2_adhoc` \| `scibert` \| `scincl` \| `e5` \| `openai` | frozen embedding(s). All `specter2*` flavours share `allenai/specter2_base` weights and differ ONLY in the activated AllenAI task adapter — see the SPECTER2 adapter table below |
| `embed_for_encoder` | one embedder name | which embedding column feeds the encoder |
| `encoder_kind` | `identity` \| `autoencoder` \| `pca` \| `umap` | dimensionality reduction; `identity` = no reduction |
| `dim_grid` | list[int] | candidate latent dims (single value = fixed size, no search) |
| `dim_recon_target` | float | AE: smallest k with reconstruction R² ≥ this |
| `dim_tolerance` | float | probe-metric tolerance when no reconstruction is available |
| `encoder_max_epochs`, `encoder_patience` | int | AE training length |

**SPECTER2 adapter flavours** — registry in `embedding/embedders.py::SPECTER2_VARIANTS`
(purpose strings in `SPECTER2_ADAPTER_PURPOSE`, task pairing in `TASK_MATCHED_EMBEDDER`).
Every flavour loads the same `allenai/specter2_base` weights; the adapter is the only
difference, and AllenAI trains one per SciRepEval task format:

| `embedder` | Adapter loaded | Designed for |
|---|---|---|
| `specter2_base` | *(none)* | plain base weights — the base-vs-adapter control |
| `specter2` | `allenai/specter2` | proximity/retrieval: nearest-neighbour, link prediction |
| `specter2_clf` | `allenai/specter2_classification` | features for a **classifier** head → pair with `emb_clf` |
| `specter2_reg` | `allenai/specter2_regression` | features for a **regressor** head → pair with `emb_reg` |
| `specter2_adhoc` | `allenai/specter2_adhoc_query` | encodes SHORT query strings, not whole papers |

Rules that keep this honest:
- The adapter is **verified active** after loading; if activation ever fails the run raises
  instead of silently returning base-weight vectors under the adapter's column name.
  The `adapters` library still logs "There are adapters available but none are activated"
  once per model load — that message is benign, the assertion is the authority.
- `add_embedding` routes every `specter2*` flavour through the adapter-aware batched path
  regardless of `EMBED_BATCH_SIZE` (`specter2` is also an `EPAI_TYPE`, and the epai path
  does not apply adapters).
- Each flavour caches to its own `emb_<name>` column, so flavours never reuse each other's
  vectors — but note `models/<dataset_id>_encoder.json` is keyed by dataset only.
- `Settings.validate()` rejects unknown embedder names before fetch/prepare.
- The generated `description_<wf>.md` states which adapter produced the embeddings and
  flags when it does not match the run's task format.

**4. Text transformer** (`tfm_*` only): `text_transformer_arch`
(`specter2_ft｜scibert_ft｜specter2_lora｜scibert_lora`), `text_transformer_lr`,
`_epochs`, `_warmup_ratio`, `_batch_size` (micro-batch; effective batch stays 64),
`_max_len`, `_freeze_layers`, `_n_runs` (separate seed count), `_lora_r`, `_lora_alpha`,
`_lora_dropout`.

**5. Metadata features** (`emb_meta_*` / `emb_all_meta_*`)

| Key | Meaning |
|---|---|
| `use_metadata`, `metadata_fields` | fuse curated scalars. **Leakage guard**: `year`, `citation_count`, `influential_citation_count`, `domains` are rejected unless `bypass_leakage_guard: true` |
| `use_novelty`, `novelty_kind` (`spherical｜cosine_max`), `novelty_k` | embedding-geometry novelty |
| `use_feasibility` | precedent/density/maturity composite |
| `external_impact` | frozen external scorers fused as features + dumped to `baselines.parquet`: `naip｜naip_local｜specter2_knn｜llm_judge｜venue_ais｜forecite｜cited_potential`. Every scorer degrades to `None` gracefully |
| `benchmarks` | suites evaluated at phase 7: `holdout_extended｜holdout_recent` (local, built by extend) `｜naip12k｜sciimpact_pairs｜dblp_v14_sample` (**manual download only** — `benchmarks.download_benchmark` deliberately raises with the manual pointer; there is no scriptable URL) |
| `judge_max_train_papers` | `llm_*` only: cap TRAIN papers building the RAG store (0 = all) |

**6. Data & split**

| Key | Meaning |
|---|---|
| `dataset_id` | **cache key** for every parquet/encoder artifact. Changing the corpus, its size, or the split REQUIRES a new id, or the cached parquet of the previous corpus is silently reused |
| `split_seed` | seed for the proportional split |
| `split_ratios` | `[train, val, test]` proportions — **ignored when the §6a holdout is active** |
| `target_min_refset` | min (field × year) papers before pooling |
| `year_from` / `year_to` | corpus year filter |
| `training_sources` | which ALREADY-COLLECTED channels may enter training: `[openalex]｜[s2]｜[openalex, s2]｜any` |
| `max_abstract_chars` | drop longer abstracts (0 = keep all) |
| `max_papers` | phase-0 corpus cap |

**6a. Fixed-holdout split** (a data-size sweep — replaces `split_ratios` when active)

| Key | Meaning |
|---|---|
| `holdout_val_n` / `holdout_test_n` | **absolute** val/test row counts. Both > 0 activates this mode; exact counts are guaranteed even with many strata |
| `holdout_seed` | picks WHICH rows are the holdout (0 = reuse `split_seed`) |
| `train_n` | how many of the remaining rows to train on (0 = all remaining) |
| `train_select_seed` | orders the train pool; the same value across steps makes a smaller `train_n` a **strict subset** of a larger one (0 = reuse `split_seed`) |

Rows past `train_n` get `split="unused"` and are excluded from every mask. **Invariant: for
val/test to stay identical across steps, `max_papers`, `holdout_val_n`, `holdout_test_n` and
`holdout_seed` must not change — only `train_n` (and `dataset_id`) may.**

**6b. Data extension** (`main_extend_db`)

| Key | Meaning |
|---|---|
| `fetch_target_rows` | grow the corpus to N TOTAL rows (no-op once already larger) |
| `fetch_new_rows` | collect exactly N **new** papers regardless of current size — **overrides the target logic; a non-zero value here means every `extend` run with `openalex` enabled fetches** |
| `fetch_from_scratch` | `true` = brand-new OpenAlex-only corpus (backs up `extended.db` non-destructively), `false` = grow the existing DB |
| `fetch_domains` | `all` (every one of OpenAlex's 26 fields, resumable per-field cursor) or a list of filter strings |
| `fetch_wait_days` | max daily-reset waits (0 = never wait) |
| `fetch_per_journal_cap` | UEFISCDI seeding cap per journal |
| `llm_fill_domain_max`, `llm_fill_abstract_max` | row caps for the LLM fills |
| `use_llm_abstracts_for_training` | may `source='llm'` rows enter training |
| `require_known_domain_for_training` | unknown-domain rows are dropped in BOTH `derive_training` and the phase-0 loader |

**`runs:` queue** — a list of entries, each `name` + `workflow` (one type or a list) + any
keys from sections 1-6b as overrides. `run_all_workflows` expands each entry × each workflow
type into one run.

### 4.2 `.env.workflow` — machine knobs

| Variable | Meaning |
|---|---|
| `PARALLEL_JOBS` | `auto` or an int — how many seeds/jobs train at once |
| `PARALLEL_LOAD_TARGET` | admit new workers only while CPU%/GPU% below this fraction |
| `PARALLEL_THREADS_PER_WORKER` | torch threads per worker (0 = split the budget) |
| `EMBED_BATCH_SIZE` | embedding batch (memory only, never affects results) |
| `FORCE_FETCH` / `FORCE_EMBED` / `FORCE_BENCHMARK_EMBED` | recompute even when a cached artifact exists |
| **`ENABLE_BOOTSTRAP`/`ENABLE_EXTEND_DB`/`ENABLE_ANALYSE`/`ENABLE_TRAINING`** | **the four launcher steps (§2.2)** |
| **`ENABLE_COPY`/`OPENALEX`/`S2`/`CROSSREF`/`ARXIV`/`UEFISCDI`/`PDF`/`FILL`** | per-source switches; also honoured when running `main_extend_db` directly |
| `ENABLE_LLM_FILL` / `ENABLE_LLM_ABSTRACT_FILL` / `ENABLE_LLM_WEBSEARCH` | extend-phase LLM repair steps |
| `WEBSEARCH_PROVIDERS` / `_TIMEOUT` / `_MIN_SIMILARITY` / `WEBSEARCH_API_PROVIDER` / `_API_KEY` | abstract lookup (`crossref,europepmc` are free/no-key) |
| `TITLE_SIMILARITY_THRESHOLD` | near-duplicate title merge threshold |
| `TRAIN_YEAR_FROM` / `TRAIN_YEAR_TO` | `training.db` year window |
| `MAX_TRAINING_ROWS` | cap `training.db` to a seeded random subset (0 = no cap) |
| `SELECTION_SEED` | seed for that subset — reruns pick the same rows |
| `USE_LLM_ABSTRACTS_FOR_TRAINING` | fallback for the config.yaml key of the same meaning |
| `EPAI_DB` | which DB feeds phase 0: empty = newest `store/data/sql/*.db`, a filename, an absolute path, or the literal `extended` / `training` |
| `ANALYSE_DB` / `ANALYSE_SOURCES` | which DB the statistics step describes, and an optional acquisition-channel filter for it |
| `LLM_JUDGE_PROVIDER` / `_MODEL` / `_API_KEY` / `LOCAL_LLM_URL` | the OpenAI-compatible endpoint shared by the judge workflow, extend-phase auto-fill, and the `llm_judge` baseline. **Any unset → every LLM call is a no-op returning `None`; callers fall back deterministically and the pipeline never crashes** |
| `LLM_JUDGE_MAX_CONTEXT_TOKENS` / `JUDGE_RAG_TOKEN_BUDGET` / `JUDGE_RAG_CHUNK_TOKENS` | judge retrieval budgets |
| `BASELINE_KNN_K` | k for the `specter2_knn` baseline |
| `FETCH_RATE_LIMIT_WAIT_HOURS` / `FETCH_STALL_LIMIT` | OpenAlex daily-limit survival (§3.3) |
| `OPENALEX_MAILTO` | politeness contact |
| `UMAP_OPEN_BROWSER` | open the interactive 3-D UMAP HTML after a run |
| `FOLLOW_PROGRESS` / `FOLLOW_DELAY_SECONDS` | whether the launcher opens the live dashboard after a detached start |
| `NAIP_WEIGHTS_DIR` / `NAIP_BATCH` | `naip_local` scorer weights + batch size |
| `GENERATED_FILE` / `GENERATED_SOURCE` | `main_predict_generated` input |

---

## 5. Source map — what every file is and what it exposes

### 5.1 Top level

| File | Purpose | Key objects |
|---|---|---|
| `run_pipeline.py` | single-workflow orchestrator | `PHASE_NAMES`, `SHARED_PHASES`, `run_workflow(wf_type, settings, start=, end=, force=, progress=)`, `_run_phase`, `main()` |
| `run_all_workflows.py` | runs every `runs:` entry; **wipes `results/` first**; shares expensive phases between matching runs | `expand_runs(cfg, config_path) -> [(wf_type, name, Settings)]`, `_data_key`, `_embed_key`, `_reset_results`, `main()` |
| `run_plot_refresh.py` | re-render all plots from existing results (no retraining); rebuilds `Settings` from each `run_config.json` | `refresh_dataset_analysis`, `refresh_run_plots`, `refresh_summaries`, `_reconstruct_settings`, `main()` |
| `deploy_and_run.ps1` / `.sh` | the 4-step launchers (§2.2) | params `-TargetRows`, `-Foreground` |
| `stop_deploy_and_run.ps1` / `.sh` | kill the detached run + free GPU/RAM | reads `logs/pipeline.pid` |
| `config.yaml`, `.env.workflow` | the control surface (§4) | — |
| `config.yaml.full_backup` | the 12-model flagship queue, kept for full campaigns | — |
| `README.md`, `DATA_QUERIES.md` | human docs: output-file reference, SQL queries | — |

### 5.2 `src/main_*.py` — one phase or job each

| File | Purpose | Entry |
|---|---|---|
| `main_fetch.py` | phase 0 — load corpus, cap, optional OpenAlex enrich | `main(settings=None)` |
| `main_prepare.py` | phase 1 — targets, metadata, `strat_label`, **the shared split** | `main(settings=None)`; consts `DROP_ROWS_WITHOUT_TARGET`, `DROP_ROWS_WITHOUT_TEXT`, `TEXT_FIELDS` |
| `main_embed.py` | phase 2 — frozen embeddings | `main(settings=None)` |
| `main_encoder.py` | phase 3 — select `k*`, fit + save the encoder | `main(settings=None)` |
| `main_reduce.py` | phase 4 — apply the encoder → `z` | `main(settings=None)` |
| `main_label.py` | phase 5 — `class_label` from `target` | `main(settings=None)` |
| `main_train.py` | phase 6 for every embedding/metadata head | `main(settings=None, workflow_type=, wf_dir=)`, `_ensure_class_label`, `_ensure_fused_columns`, `_dump_baselines` |
| `main_train_tfm.py` | phase 6 for `tfm_*` — fine-tune a transformer on raw text, own seed loop | `main(...)`, `_ensure_text_columns` |
| `main_train_judge.py` | phase 6 for `llm_*` — RAG store built once from TRAIN, exactly one LLM call per record | `main(...)`, `_train_seed`, `_build_rag`, `_retrieve`, `_predict_split`, `_predict_impact`, `_class_ids`, `_effective_token_budget` |
| `main_extend_db.py` | the DB CLI (§3.3) | `main()`, `cmd_bootstrap`, `cmd_extend`, `derive_training`, `copy_legacy_dbs`, `report`, `_filter_sources`, `phase_timeout` |
| `main_analyse_dataset.py` | dataset-statistics CLI | `main()`, `resolve_db_path` |
| `main_predict_generated.py` | score generated papers with the latest regressor | `main(settings=None)` |
| `setup_external_models.py` | one-shot installer for external baselines | `main()`, `install_one` |
| `tests/smoke.py` | offline suite (no network/keys/downloads); `make_corpus()` synthesizes fixtures | ~50 `t_*` cases + `_run_all()` |

### 5.3 `src/iwf/` — the package

**Core**

| Module | Purpose | Key objects |
|---|---|---|
| `config.py` | the control surface loader | `WORKFLOWS`, `WORKFLOW_TYPES`, `TEXT_TRANSFORMER_ARCHS`, `RESERVED_METADATA_FIELDS`, `MIN/MAX_CLASSES`, `Settings` (dataclass, `.update()`, `.validate()`, `.to_dict()`, `.copy()`), `load_settings`, `load_config_yaml`, `resolve_class_thresholds`, typed env accessors (`get_str/int/float/bool`, `secret`, `api_key`), `training_sources`, `analyse_sources`, `is_text_workflow`, `is_judge_workflow` |
| `paths.py` | every filesystem location | `raw_parquet`, `prepared_parquet`, `embedded_parquet`, `reduced_parquet`, `generated_parquet`, `encoder_path`, `encoder_meta_path`, `ensure_dirs`, dir constants |
| `registry.py` | two-level `(slot, name)` module registry | `register_module(slot, name)` decorator, `get_module`, `available_modules`; slots: `embedder`, `encoder`, `metadata_fuser`, `impact_predictor`, `text_transformer` |
| `targets.py` | impact targets + class binning | `compute_target(kind, df, min_refset)`, `field_year_percentile`, `fwci_percentile`, `normalized_percentile`, `bayesian_impact`, `to_classes`, `css_boundaries`, `quantile_boundaries`, `resolve_class_boundaries`, `resolve_settings_class_thresholds`, `default_class_names`, `hazen_percentile` |
| `domains.py` | the canonical ~26-field OpenAlex taxonomy | `ALLOWED_DOMAINS`, `all_field_filters()` |
| `workflow_results.py` | result folders + cross-seed aggregation + cross-run figures | `next_workflow_dir`, `latest_model_dir`, `write_run_config`, `seed_results`, `aggregate`, `finalize_workflow`, `write_model_description`, `cross_workflow_comparison`, `_collect_correctness`, `_plot_correctness_bars`, `_plot_algorithm_comparison` |
| `parallel.py` | resource-aware job runner | `run_jobs(worker, payloads, use_cuda=, vram_per_worker_mib=)`, `plan_workers`, `torch_cuda_available`, `parallel_enabled` |
| `progress.py` | live multi-workflow terminal display | `MultiWorkflowProgress(workflow_ids, labels=, stream=)` — one line per id, labelled `<workflow_type>:<queue_entry_name>` in FULL (column width = the longest label passed in, never hardcoded/truncated), `tqdm()` wrapper |
| `activations.py` | activation-name → torch module | `make_activation`, `valid_activations` |

**`data/`**

| Module | Purpose | Key objects |
|---|---|---|
| `dataset.py` | parquet IO + **the one shared split** | `load_parquet`, `save_parquet`, `make_split(n, seed, ratios, stratify)`, `make_fixed_holdout_split(n, val_n, test_n, train_n, holdout_seed, train_select_seed, stratify)`, `add_split_column(...)` (dispatches proportional vs fixed-holdout), `split_masks`, `_force_exact_counts`, `SPLIT_COL`, `SPLIT_NAMES` |
| `sources.py` | phase-0 corpus loading + OpenAlex enrich | `resolve_db_path`, `load_sqlite`, `assemble`, `enrich_with_openalex`, `openalex_fields` |
| `identity.py` | the ONE join key | `ensure_paper_id`, `normalize_doi`, `stable_text_id`, `keyed_merge` |
| `generated_sources.py` | MIRAI/ADP generated-paper loading | `load_from_file`, `load_from_api` |

**`embedding/` · `encoder/`**

| Module | Purpose | Key objects |
|---|---|---|
| `embedding/embedders.py` | frozen title+abstract embeddings | `add_embedding`, `embed_dataframe`, `emb_matrix` |
| `encoder/base.py` | encoder interface + loading | `Encoder` (ABC: `fit`, `transform`, `save`, `load`), `get_encoder(kind, latent_dim, ...)`, `load_saved_encoder` |
| `encoder/autoencoder.py` | AE reduction | `AutoencoderEncoder`, `EpaiAutoencoderEncoder`, `_AE` |
| `encoder/pca_umap.py` | linear/manifold/no-op | `PCAEncoder`, `UMAPEncoder`, `IdentityEncoder` |
| `encoder/dim_selector.py` | latent-dim search | `select_dim`, `pca_variance_anchor` |
| `encoder/intrinsic_dim.py` | theoretical anchors | `twonn`, `mle_levina_bickel`, `estimate_intrinsic_dim` |

**`features/`**

| Module | Purpose | Key objects |
|---|---|---|
| `feature_builder.py` | assemble model inputs; owns the neutral-zero + mask contract and the leakage guard | `build_xy`, `scalar_feature_columns`, `all_scalar_columns`, `drop_reserved`, `NeutralScaler` |
| `metadata.py` | curated scalar encoders | `build_metadata(df, fields)` |
| `novelty.py` | embedding-geometry novelty | `compute_novelty`, `novelty_spherical`, `novelty_cosine_max`, `novelty_feasibility`, `percentile_01` |
| `feasibility.py` | precedent/density/maturity composite | `feasibility(Z, years, k, ...)` |
| `external_impact.py` | frozen external scorers | `ExternalImpactModel` (ABC), `NAIPModel`, `NAIPLocalModel`, `Specter2KnnBaseline`, `LLMJudgeBaseline`, `VenueAISBaseline`, `ForeCiteModel`, `CitedPotentialModel`, `known_external_models`, `get_external_models` |

**`models/` · `training/`**

| Module | Purpose | Key objects |
|---|---|---|
| `models/architectures.py` | head block factory | `available_archs()` → `mlp｜resmlp｜ft_transformer`, `build_block`, `_ResMLP`, `_FTTransformer` |
| `models/head_base.py` | shared torch training loop for the heads | `TorchHeadBase` (`fit`, `predict`, `history_`, `net_`) |
| `models/classifier.py` | classification head | `TorchMLPClassifier` (`task="clf"`, `predict_proba`), `class_weights(y, n_classes, scheme, beta)`, `_FocalLoss` |
| `models/regressor.py` | regression head | `TorchMLPRegressor` (`task="reg"`) |
| `models/sklearn_baseline.py` | non-neural baselines, duck-typing the torch heads (`net_=None`, `history_={}`) | `SklearnLogisticRegressionClassifier(n_classes=, seed=, class_weight=, max_iter=)`, `SklearnLinearRegressionBaseline(seed=)` |
| `models/text_transformers.py` | `tfm_*` backbones, registry-registered | `TextTransformerBase`, `Specter2FT`, `SciBertFT`, `Specter2LoRA`, `SciBertLoRA`, `LoRALinear` |
| `training/trainer.py` | generic torch loop with early stopping | `train_torch_model`, `configure_runtime`, `set_seed` |
| `training/experiment.py` | **the multi-seed loop** for embedding/metadata workflows | `run_experiment(df, task, settings, scalar_cols, out_dir, label=, bypass_guard=)`, `_train_one_seed` |
| `training/text_trainer.py` | `tfm_*` loop (pre-tokenize once, CUDA-OOM auto-retry) | `TextTransformerTrainer.run(df, seed, output_dir, class_names)` |

**`evaluation/`**

| Module | Purpose | Key objects |
|---|---|---|
| `metrics.py` | **the single metric implementation** | `regression_metrics`, `classification_metrics`, `test_correctness(result, task, thresholds, split="test")`, `accepted_seeds(seeds, task, thresholds, enabled=, min_n=, gap_pct=) -> (accepted, excluded)`, `ndcg_at_frac`, `top_frac_auc`, `pairwise_accuracy`, `generated_vs_human` |
| `plots.py` | **ALL plots** | per-run: `generate_workflow_plots`, `correctness_bars`, `mse_bars`, `learning_curve`, `plot_nn_weights`, `_umap_suite`; cross-seed: `plot_seed_spread_figures`, `_render_confusion_matrix`, `_plot_seed_correctness`; helpers: `_load_all_seeds`, `_pick_best_seed`, `_seed_note`; dataset figures: `plot_counts_year/_domain`, `plot_completeness*`, `plot_correlation_heatmap`, `plot_score_distributions`, `plot_target_by_year`, …; comparison figures: `plot_benchmark_*`, `plot_cross_domain_*`, `plot_embedder_performance`, `plot_per_domain_bars` |
| `palette.py` | the one color system | `apply_style`, `field_color_map`, `class_colors`, `ordinal_colors`, `category_of`, `register_*_seq/div` colormaps |
| `data_analysis.py` | corpus-level statistics + `dataset_report.md` | `analyse(db_path, out_dir, train_db_path=)`, `counts_per_year/_domain`, `completeness_table`, `quality_summary`, `numeric_correlations`, `target_distribution`, `write_report_md` |
| `baseline_compare.py` | ours vs frozen baselines | `generate_baseline_plots`, `baseline_metrics`, `plot_baseline_agreement`, `plot_baseline_correlation`, `baseline_leaderboard` |
| `benchmarks.py` | external benchmark suites | `BenchmarkSpec`, `available_benchmarks`, `build_holdout_benchmark`, `build_temporal_holdout`, `score_with_run_model`, `run_benchmarks`, `benchmark_overview`, `download_benchmark` (**always raises with the manual-download pointer — no scriptable URL exists**) |
| `cross_domain.py` | per-domain + LODO analysis | `per_domain_metrics`, `generate_cross_domain`, `cross_domain_matrix` |
| `nf_analysis.py` | novelty/feasibility statistics | `generate_nf_analysis`, `nf_correlation_matrix`, `nf_binned_impact`, `nf_ablation`, `partial_spearman` |
| `embedder_summary.py` | best model per embedder type | `embedder_performance_summary`, `generate_embedder_summary` |

**`fetchers/` · `llm/`**

| Module | Purpose | Key objects |
|---|---|---|
| `base.py` | extended-DB schema + dedup/resume insertion | `BaseFetcher` (ABC), `Deduper`, `insert_paper`, `merge_paper_metadata`, `open_db`, `get_cursor`/`set_cursor`, `QuotaExhausted` |
| `openalex.py` | the workhorse crawler | `OpenAlexFetcher`, `collect_new_papers(conn, fetcher, target, wait_days=, wait_hours=, stall_limit=, on_wait=)`, `fill_author_h_index`, `work_to_row`, `reconstruct_abstract` |
| `crossref.py` / `arxiv.py` / `s2.py` | other channels | `CrossrefFetcher`, `ArxivFetcher`, `fill_from_s2` |
| `uefiscdi.py` / `uefiscdi_pdf.py` | journal AIS/zone lists | `load_ais_table`, `UefiscdiSeedFetcher`, `print_download_hints` (**manual helper — prints where to download the lists**), `import_journal_data`, `extract_from_pdf` |
| `enrich_metadata.py` | fill incomplete rows by title search | `enrich_incomplete_papers`, `search_openalex_by_title`, `search_s2_by_title` |
| `llm_fill.py` | extend-phase LLM repair | `llm_fill_domains`, `llm_fill_abstracts` |
| `websearch.py` | free, no-key abstract lookup | `search_abstract`, `is_enabled` |
| `merge_all_databases.py` | cumulative merge into `extended.db` | `merge_all_databases` |
| `model_weights.py` | frozen external weights (USER-run only) | `download_naip_weights` |
| `llm/client.py` | the shared OpenAI-compatible client | `complete(prompt, system=, as_json=)` — returns `None` when unconfigured |

---

## 6. Outputs — what a run writes

```
workflow/
  data/
    sql/{raw,extended,training}.db      # workflow-owned, safe to delete/regenerate
    parquet/<dataset_id>_{raw,prepared,embedded,reduced}.parquet
    cache/, pdf/
    analysis/                           # dataset statistics + dataset_report.md
  models/encoder_<dataset_id>.joblib
  logs/                                 # launcher logs, pipeline.pid, progress.status
  results/
    <wf_type>_N/
      run_config.json                   # full Settings snapshot for the run
      baselines.parquet
      seed_<i>/result.json              # THE per-seed contract (see §8)
      seed_<i>/model/{model.joblib, scaler.joblib}
      aggregate/
        metrics_summary.json            # mean/std/min/max per metric across seeds
        description_<wf_type>.md         # per-run record: headline result + at-a-glance
                                        # table, then numbered sections (Run Summary,
                                        # Model, Embedding/Encoder incl. WHICH SPECTER2
                                        # adapter, Training, Data/Split, collapsed full
                                        # parameter snapshot). Sections are numbered as
                                        # emitted, so a skipped one leaves no gap; the
                                        # split section reports the fixed holdout
                                        # (train_n/val/test + seeds) when it is active,
                                        # not the inert split_ratios.
        correctness_bars_<wf>.png/.csv  # % correct per split, mean±std over ACCEPTED seeds
        confusion_matrix_<wf>.png/.csv  # row-% averaged over ACCEPTED seeds
        seed_correctness_<wf>.png/.csv  # per-seed bars; excluded seeds hatched
        learning_curve_<wf>.png, class_boxplot_<wf>.png, mse_bars.png,
        spearman_distribution.png, predicted_vs_actual.png,
        umap3d_*.{html,png}, nn_weights*, model_benchmarks/
    summary/
      cross_workflow_comparison.{png,json,csv}
      algorithm_comparison_train_test.png
      correctness_by_category.png       # train+test bars, chance line, per-run mean±std
      correctness_by_embedder.png
      benchmark_*, cross_domain_*, embedder_performance.png
```

`README.md` holds the human-facing table of every figure and what it shows.

---

## 7. Current state — the data-size sweep

The checked-in `config.yaml` + `.env.workflow` are set up for a **training-set-size sweep** on
an OpenAlex-only corpus:

- **Corpus**: `extended.db` holds ~9000 OpenAlex rows (≈8998 training-usable). No further
  collection is wanted, so **`ENABLE_OPENALEX=false`** — with every source flag off, the
  launcher's extend step only re-derives `training.db` and never touches the network.
- **Pool**: `MAX_TRAINING_ROWS=2500` + `SELECTION_SEED=45` → a fixed 2500-row `training.db`;
  `max_papers: 2500`. These stay **fixed for the whole sweep**.
- **Split**: fixed holdout (§4 6a) — `holdout_val_n/test_n: 250` each with `holdout_seed: 45`,
  so every step evaluates on the identical rows; `train_n` is the only knob that moves
  (2000 now, stepping down toward 100), and smaller train sets are strict subsets of larger.
- **Target/classes**: `fwci_percentile`, 2 classes (`class_thresholds: [0.5]`,
  `class_scheme: quantile` → a data-derived median cut, so Q2/Q1 are exactly half each).
- **Queue**: 5 entries × `n_runs: 5` — `demo_specter2_base` (mlp, clf+reg),
  `demo_sklearn_logreg` (clf), `demo_sklearn_linreg` (reg), `demo_specter2_adapter`
  (mlp, clf+reg), `demo_specter2_ae256` (autoencoder→256 + mlp, clf+reg) = 8 sub-runs.
  Text-transformer and LLM-judge are deliberately out of the queue (GPU/API cost).
- **Seed statistics**: `seed_outlier_min_n: 5` matches `n_runs: 5`, so the outlier rule is
  live; charts report mean±std over accepted seeds and mark any excluded seed.

**To advance the sweep**: change `train_n` and bump `dataset_id` in `config.yaml` — nothing
else. `dataset_id` must change even though the pool doesn't, because phase 1's parquet is
cached by that id and would otherwise serve the previous step's split. `training.db` only
needs deriving once for the whole sweep. `run_all_workflows` wipes `results/` on every run,
so the user saves each step's results before starting the next.

---

## 8. Conventions (do not break these)

- **ONE implementation per concern**: metrics in `evaluation/metrics.py`, all plots in
  `evaluation/plots.py`, the multi-seed loop in `training/experiment.py`, the split in
  `data/dataset.py`. Add a task-variant inside the shared module — never a per-workflow copy.
- **`WORKFLOWS` in `config.py` is the single source of truth** for task / feature-set / phase
  list per workflow type. Extend it there; never hard-code workflow names elsewhere.
- **Uniform results contract**: every workflow writes `results/<wf>_N/seed_<i>/result.json`
  with per-split dicts (`train`/`val`/`test`, each carrying `y_true`/`y_pred` + metrics),
  flat root copies of the TEST metrics for aggregation, `train_<metric>` root copies, and
  the all-rows arrays (`y_all`, `y_pred_all`, `split_all`) the UMAP plots align against.
  Keep new outputs inside it so aggregation/plots stay workflow-agnostic.
- **Split labels**: `train`/`val`/`test` plus `unused` (fixed-holdout mode). `split_masks`
  matches only the first three, so any extra label is excluded from training everywhere.
- **Seed aggregation**: every cross-seed figure and every cross-run correctness number goes
  through `metrics.accepted_seeds` — never average raw seeds ad hoc, and never present a
  single best seed as if it were the result.
- **Chance reference**: a %-scale figure gets `100/n_classes` as its dashed baseline, derived
  from the run's own class count. Never hardcode 50.
- New phase script → add to `src/` and wire it into `run_pipeline._run_phase` + the `WORKFLOWS`
  phase lists, not as a new top-level entry point.
- New model architecture → register it (`registry.register_module`) or extend
  `available_archs()`; the sklearn baselines are the one deliberate exception (special-cased
  by name in `training/experiment.py` and `config.Settings.validate`, because they bypass the
  torch training loop entirely).
- Run the offline suite before and after changes: `uv run python -m workflow.src.tests.smoke`
  (exit 0 = all passed; no network, keys, or model downloads).
- **Never commit, never push** (repeated from §0): change files only; the user does all git.

## 9. Comment style (strict)

**Source code (`.py`) — almost no comments.**
- Remove comments from code. Do NOT explain steps inside a function body with comments.
- The ONLY comment allowed in a source file is **one short line at the definition** of each
  class, method, or function saying what it does.
- That line is **max 10 words**, plain, **beginner-friendly** — easy for a new programmer.
- Prefer a one-line docstring for that sentence; do not add extra prose or inline `#` notes.
- If code needs a comment to be understood, rename things or split it up instead.

```python
def split_masks(df):
    """Return train/val/test boolean masks for the rows."""   # <= 10 words, simple
    ...
```

**Config / non-code files — comment well.**
- `config.yaml`, `.env` / `.env.workflow`, other config files, and results MAY have comments.
- Keep them **precise**, and **always list the valid options** next to a setting, e.g.:
  ```
  LLM_JUDGE_PROVIDER=local      # anthropic | openai | local
  encoder_kind: identity        # identity | autoencoder | pca | umap
  ```
- Say what the key does and the effect of each option, briefly.

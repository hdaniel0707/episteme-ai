# CLAUDE.md — `workflow/` is the working space

**This file is the primary guidance for the repo. The repo-root `CLAUDE.md` was removed;
all workflow work is documented here.**

## 0. The one rule that governs everything

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

## 1. The two-file control surface

You configure a session by editing **exactly two files**, both in `workflow/`:

| File | Governs | Read by |
|------|---------|---------|
| `config.yaml` | **WHAT** to do: workflow topology, target, classes, model/hyperparams, data & split, data-extension knobs, and the `runs:` queue. | `iwf/config.py` → `Settings` |
| `.env.workflow` | **HOW** this machine runs it: parallelism, GPU batch size, cache-force flags, per-source ENABLE_* toggles, DB selection, and the LLM/web-search endpoint + credentials. | `iwf/config.get_*` env accessors |

Rule of thumb: if a knob changes *the science / what a run produces or costs*, it belongs in
`config.yaml` and may be overridden per-run inside a `runs:` entry. If it only changes *how
this machine executes* (speed, memory, cache, which endpoint), it belongs in `.env.workflow`.
`WORKFLOW_REFERENCE.md` documents every key.

- Per-run overrides travel as plain dicts into `Settings` via `load_settings(..., overrides=)`
  — **never** mutate `os.environ` to pass a parameter.
- `config.yaml.full_backup` holds the original 8-hour flagship queue; restore it for a real
  training session. The checked-in `config.yaml` is currently a **rapid-test** setup
  (500 papers, small nets, short training — see §5).

## 2. Entry points & layout

Two top-level runnable scripts (everything else importable lives under `src/`):
- `python -m workflow.run_pipeline` — run ONE workflow from the top-level `config.yaml` keys.
- `python -m workflow.run_all_workflows` — run EVERY entry in the `runs:` queue, each
  inheriting the top-level keys + its own overrides. **Wipes `results/` first.** Auto-indexed
  output folders (`emb_clf_1`, `emb_clf_2`, …). Expensive shared phases (fetch/prepare/embed/
  encode) are computed once and reused between runs with matching data/embedding parameters.

```
workflow/
  run_pipeline.py, run_all_workflows.py   # the only two entry points
  config.yaml, .env.workflow(.example)    # the two control files
  run_plot_refresh.py                     # re-render plots from existing results/
  src/
    main_*.py            # one phase / job each (see §3)
    tests/smoke.py       # offline suite — NO network/keys/model downloads; run before changes
    iwf/                 # THE package (all logic). Import as workflow.src.iwf....
      config.py          # Settings + WORKFLOWS topology table + typed env accessors
      paths.py           # every path constant (SQL_DIR, RESULTS_DIR, cache, …)
      registry.py        # phase dispatch / model+workflow registration
      targets.py         # impact targets, class binning, class names
      domains.py         # ALLOWED_DOMAINS — the canonical ~26-field OpenAlex taxonomy
      progress.py, parallel.py, activations.py, workflow_results.py
      data/              # sources.py (phase-0 corpus loader + OpenAlex enrich), identity, split
      fetchers/          # DB building: openalex, crossref, arxiv, s2, uefiscdi(_pdf),
                         #   enrich_metadata, llm_fill, websearch, base (Deduper), merge_all_databases
      embedding/         # embedders.py (specter2 / specter2_base / scibert / scincl / e5 / openai)
      encoder/           # autoencoder.py + PCA/UMAP/identity dim reduction
      features/          # metadata / novelty / feasibility / external_impact baselines
      models/            # architectures (mlp / resmlp / ft_transformer), heads
      training/          # experiment.py (multi-seed loop), trainer
      llm/               # client.py — shared OpenAI-compatible endpoint client
      evaluation/        # metrics, plots, benchmarks, baseline_compare, cross_domain,
                         #   nf_analysis, embedder_summary, data_analysis, palette
  data/                  # generated: sql/, parquet/, cache/, pdf/, analysis/ (safe to delete)
  models/, results/      # generated artifacts (safe to delete/regenerate)
  README.md, WORKFLOW.drawio, WORKFLOW_REFERENCE.md, mathematical_documentation.md, DATA_QUERIES.md
```

## 3. The pipeline: phases and DB building

**Training phases** (run by `run_pipeline`; `PHASE_NAMES` in `run_pipeline.py`, per-workflow
phase lists in `WORKFLOWS` in `config.py`):

`0 fetch → 1 prepare → 2 embed → 3 encoder → 4 reduce → 5 label → 6 train → 7 finalize`

- `emb_*` use all phases (reg skips the classifier `label` phase). `tfm_*` skip embed/encode
  (2–4) and fine-tune a transformer on raw text. `llm_*` (LLM-as-judge) build a RAG store +
  evolving `notes.md` instead of a trained head (`main_train_judge.py`).
- Phase 0 loads the corpus from the DB chosen by `EPAI_DB` (see `iwf/data/sources.py`),
  caps to `max_papers`, and optionally OpenAlex-enriches impact fields.

**Database building** (`python -m workflow.src.main_extend_db`, separate from training):
- `bootstrap` — fresh multi-source crawl (openalex, crossref, arxiv, s2) → `raw.db`, keeping
  incomplete rows.
- `extend` — `raw.db` + legacy `store/data/sql/*.db` → `extended.db` → `training.db`. Steps
  (each selectable via `--sources`, gated by `ENABLE_*`): `copy pdf uefiscdi openalex s2 fill
  llm_fill`, then always: report → derive `training.db` (strict: title+abstract+year+citations,
  `TRAIN_YEAR_FROM/TO`, `MAX_TRAINING_ROWS`, LLM-abstract exclusion) → build local benchmarks.
  - **`fetch_new_rows`** (config.yaml): collect exactly N *new* OpenAlex papers on top of an
    existing `extended.db`, regardless of its size (the target_rows−have logic resolves to 0
    on a big DB). Use it to add a fixed batch, e.g. `--sources openalex,llm_fill` with
    `fetch_new_rows: 500`.
  - **Daily-limit survival** (`fetchers.openalex.collect_new_papers`): OpenAlex meters daily
    credits (100k/day free, resets ~midnight UTC). If collection hits that limit before the
    target, it idles ~`FETCH_RATE_LIMIT_WAIT_HOURS` (25h) for the reset and resumes, up to
    `fetch_wait_days` (config.yaml) times — holding **minimal RAM** (the per-session `Deduper`
    is freed between sessions). It stops early WITHOUT waiting when the query is exhausted or
    stalls on duplicates (`FETCH_STALL_LIMIT`) — a reset yields no new papers there. Dedup
    (`base.Deduper`, `OA:` ids globally unique) means already-stored works are never
    re-inserted; the stall guard stops paging through them so credits aren't wasted. While
    waiting, an `on_wait` hook does the LLM domain/abstract fill (shared cap), so the idle day
    is spent usefully.
  - **`fetch_from_scratch`** (config.yaml, `true|false`): build a brand-new OpenAlex-only
    corpus — backs up any existing `extended.db` (non-destructive `.pre_scratch_*`) and skips
    the legacy `copy` step. `false` grows the current DB.
  - **`fetch_domains: all`** (config.yaml): crawl every one of OpenAlex's 26 scientific fields
    (`domains.all_field_filters()`, ids 11–36) for wide-domain coverage; the per-field cursor
    makes it resumable and balanced.
  - **`require_known_domain_for_training`** (config.yaml, `true|false`): unknown-domain rows
    are dropped from training in BOTH `derive_training` and the phase-0 loader — unknown
    domain must never be trained on. LLM domain fill runs first to recover as many as its cap.

## 4. LLM auto-fill + web search (the data-repair step)

`iwf/fetchers/llm_fill.py` runs in the extend `llm_fill` step and is the model for how LLM
features are wired. All calls are **best-effort** — an unconfigured/down endpoint just skips
the row (idempotent WHERE clause, picked up on a later run) and never crashes the pipeline.

- **Domain fill** (`llm_fill_domains`, gate `ENABLE_LLM_FILL`): picks a field from
  `ALLOWED_DOMAINS` for rows with a title but no domain. **Harmless** — only fills a missing
  classification, never touches `source`, **always kept in training**. Cap:
  `config.yaml llm_fill_domain_max`.
- **Abstract fill** (`llm_fill_abstracts`, gate `ENABLE_LLM_ABSTRACT_FILL`): reconstructs a
  missing abstract/year. **Web-search-first** when `ENABLE_LLM_WEBSEARCH=true` — look the paper
  up by title in FREE, NO-KEY scholarly APIs (`iwf/fetchers/websearch.py`: Crossref + Europe
  PMC; ResearchGate etc. are Cloudflare-walled and unusable programmatically), hand any
  confident hit to the LLM to **verify + clean** (never trusted blindly); only with no hit does
  the LLM answer from its own knowledge. Every filled abstract sets **`source='llm'`** so it is
  labelled and excludable. Cap: `config.yaml llm_fill_abstract_max`. Optional keyed generic
  search (`tavily`/`serper`) exists but is OFF unless `WEBSEARCH_API_PROVIDER` + key are set.
- **Training inclusion of LLM abstracts**: `config.yaml use_llm_abstracts_for_training`
  (default `false`; falls back to `.env.workflow USE_LLM_ABSTRACTS_FOR_TRAINING`). Enforced in
  BOTH `derive_training` (training.db) and the phase-0 loader `sources.load_sqlite` (any
  EPAI_DB) via `config.use_llm_abstracts_for_training()`. Domain fill is unaffected.

**LLM endpoint config** (`.env.workflow` only — the one exception to root-`.env`-secrets;
`iwf/llm/client.py`): `LLM_JUDGE_PROVIDER` / `LLM_JUDGE_MODEL` / `LLM_JUDGE_API_KEY` /
`LOCAL_LLM_URL` (OpenAI-compatible chat-completions). Shared by the LLM-as-judge workflow, the
extend-phase auto-fill, and the `llm_judge` external-impact baseline.

## 5. Current config (checked-in `config.yaml`) — 200k from-scratch campaign

Build a **fresh OpenAlex-only corpus of 200k papers across all 26 fields** over up to a
**3-day** daily-credit campaign (`fetch_from_scratch: true`, `fetch_domains: all`,
`fetch_target_rows: 200000`, `fetch_wait_days: 3`), LLM-fill **up to 5000 unknown domains**
(`llm_fill_domain_max: 5000`; abstract fill off), drop any still-unknown-domain rows
(`require_known_domain_for_training: true`), then train on the full corpus (`max_papers:
200000`, `MAX_TRAINING_ROWS=0`) with the recommended head (§2: `resmlp [384,256,128]`,
`max_epochs 300`). The `runs:` queue is 12 models (each reg + clf): **(1)** specter2+adapter
768-d · **(2)** specter2 base + autoencoder (optimised dim) · **(3)** specter2 adapter +
autoencoder · **(4)** trf short LoRA (r=8, 8 frozen, 15 ep) · **(5)** trf deep LoRA (r=32, 0
frozen, 15 ep) · **(6)** LLM-as-judge (gpt-oss:20b). Runs end-to-end via `deploy_and_run.sh`.
Restore `config.yaml.full_backup` for the original flagship queue.

## 6. Conventions (do not break these)

- **ONE implementation per concern**: metrics in `evaluation/metrics.py`, all plots in
  `evaluation/plots.py`, the multi-seed loop in `training/experiment.py`. Add a task-variant
  inside the shared module — never a per-workflow copy.
- **`WORKFLOWS` in `config.py` is the single source of truth** for task / feature-set / phase
  list per workflow type. Extend it there; never hard-code workflow names elsewhere.
- **Uniform results contract**: every workflow writes `results/<wf>_N/seed_<i>/result.json`;
  keep new outputs inside it so aggregation/plots stay workflow-agnostic.
- New phase script → add to `src/` and wire it into `run_pipeline._run_phase` + the `WORKFLOWS`
  phase lists, not as a new top-level entry point.
- Run the offline suite before and after changes: `uv run python -m workflow.src.tests.smoke`
  (exit 0 = all passed; no network, keys, or model downloads).
- **Never commit, never push** (repeated from §0): change files only; the user does all git.

## 7. Comment style (strict)

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

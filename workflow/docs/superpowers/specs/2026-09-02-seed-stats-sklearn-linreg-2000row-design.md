# Seed-aware plots, sklearn linreg baseline, 2000-row training set — design

Date: 2026-09-02. Follows the sklearn-logreg/FWCI plan from
`docs/superpowers/plans/2026-08-05-fwci-openalex-sklearn-judge-simplify.md`.

## 1. Problem

Two presentation slides (dia 51-52, dia 54) were built from a pipeline that silently
cherry-picks the best-performing seed for the per-split accuracy chart and the confusion
matrix, which is statistically misleading when `n_runs > 1`. Separately, the task list called
for a scikit-learn regression baseline, a generic (not hardcoded-50%) chance-reference line,
a train-correctness view on the cross-run comparison charts, and a real ~2000-row training set
instead of the current 200-row demo.

## 2. Seed aggregation (dia 51-52)

**Outlier rule** — new `config.yaml` knobs, read into `Settings`:
```yaml
seed_outlier_enabled: true
seed_outlier_min_n: 5        # exclusion never triggers below this many seeds per run
seed_outlier_gap_pct: 15.0   # exclude a seed if its test correctness is this many
                              # percentage points below the median of the OTHER seeds
```
New `evaluation/metrics.py` helper `accepted_seeds(seeds, task, class_thresholds, *, enabled,
min_n, gap_pct) -> (accepted, excluded)`, built on the existing `test_correctness`. Below
`min_n` seeds, every seed is accepted (today's `n_runs=3` is a deliberate no-op; raising seeds
later activates it with no code change).

**"Category/Prediction Accuracy by Split"** (`correctness_bars` in `evaluation/plots.py`)
currently reads one best seed via `_load_best_seed_data`. It changes to: load all seeds
(new `_load_all_seeds(wf_dir)`, factored out of `_load_best_seed_data`) -> `accepted_seeds()`
-> per split (train/val/test), mean % correct + std-dev across accepted seeds -> bars with
error bars + a small "n = 5 seeds" / "n = 4/5 seeds (1 excluded)" annotation + a chance-level
dashed line at `100/len(class_names)`. `mse_bars` gets the identical multi-seed treatment for
consistency (same underlying per-seed split data, same cherry-picking flaw).

**Confusion matrix** — every seed in a run shares the exact same physical test rows (only the
model's init seed differs), so "average of row-percentages across accepted seeds" is
mathematically identical to "sum raw counts across accepted seeds, then row-normalize once."
Implemented the simpler way. Applies to BOTH branches inside `plot_seed_spread_figures`: the
regression class-binned CM (digitize each accepted seed's true/pred by the run's thresholds,
sum, normalize) and the classifier CM (sum each accepted seed's stored `confusion` array,
normalize) — both currently pick one best/representative seed via `_pick_best_seed`, both get
replaced by the same new `_render_confusion_matrix_avg(...)` helper, labelled with the
accepted-seed count. `learning_curve`, the UMAP suite, `plot_nn_weights`, and the regression
predicted-vs-actual scatter stay single-best-seed — out of scope, not what was asked.

**Per-seed bar chart** (`_plot_seed_correctness`, already multi-seed) gets: the same generic
chance line (needs `class_names`/`n_classes` threaded in, replacing the hardcoded 50%), and a
visual marker (hatch + "(excluded)" label) on any seed dropped by `accepted_seeds()`.

## 3. Chance-line generalization

Fix the two hardcoded `ax.axhline(50.0, ...)` / `ax.axvline(50.0, ...)` bugs
(`plots._plot_seed_correctness`, `workflow_results._plot_correctness_bars`) to the generic
`100/n_classes` formula already used correctly in `baseline_compare.py`, and add the
(currently missing) line to the split-accuracy chart. NOT added to: MSE bars, learning-curve,
UMAP, confusion matrix, NN-weight diagram — a horizontal %-line has no meaning on those axes.
`workflow_results._collect_correctness` derives `n_classes` from each run's own stored
`class_thresholds` (`len(thresholds) + 1`); the comparison charts assume one uniform class
count across every run being compared, matching how the current config/queue actually works.

## 4. New sklearn regression baseline

Plain `LinearRegression` (`model_arch: "linreg"`), added to `models/sklearn_baseline.py`
mirroring the existing `SklearnLogisticRegressionClassifier` exactly: `task = "reg"`,
`net_ = None`, `history_ = {}`, `fit(X_train, y_train, X_val=None, y_val=None)` (val accepted,
unused — no early stopping), `predict(X)`. Wired into `training/experiment.py`'s
`model_arch` branch, `config.py` validation (rejects `linreg` for `task == "clf"`, the
mirror of the existing `logreg`-rejected-for-`reg` check), and a new `config.yaml` queue entry
`demo_sklearn_linreg` (`emb_reg`, `specter2_base`, `identity`, `n_runs: 5`).

Flagged during design (accepted by the user regardless): at today's ~100 train rows vs.
768-d embeddings this would be heavily underdetermined (p >> n, rank-deficient, sklearn's
least-squares solver falls back to a minimum-norm solution rather than crashing, but
predictions would be unstable). The upcoming data-size bump (Section 6) brings train rows to
2000 — well above the 768 feature count — which resolves that instability as a side effect
of the data-size increase itself; no regularization is added either way, per the user's
explicit choice of plain `LinearRegression` over `Ridge`.
`class_weight="effective"` gap in the existing logreg baseline: left as-is, not fixed.

## 5. Dia 54 — train bars on the comparison charts

`workflow_results._collect_correctness` also computes TRAIN correctness per seed (new
`split` parameter on `metrics.test_correctness`, default `"test"`, unchanged for every
existing caller). `_plot_correctness_bars` draws a second, grouped bar per run for TRAIN
(same look as `_panel_train_vs_test`'s barh style, applied to the unified 0-100% correctness
scale instead of task-specific spearman/macro-F1, so classifier/regressor/text-transformer/
LLM-judge stay comparable in one figure). The same `accepted_seeds()` filtering from Section 2
applies here too, so `correctness_by_category.png` / `correctness_by_embedder.png` and the
dia 51-52 charts report a consistent "which seeds count" story for a given run.

## 6. Training-set size: 200 -> 2000 rows, OpenAlex only

Verified directly against `workflow/data/sql/extended.db`: 9000 rows total, all
`source='openalex'` (matches `training_sources: [openalex]` already in `config.yaml` and
`.env.workflow`'s `ENABLE_OPENALEX=true` / every other `ENABLE_*` source flag already `false`).
8998 of those 9000 already satisfy every training requirement (title + abstract + year<=2024 +
citation_count + known domain + `citation_normalized_percentile` for the `fwci_percentile`
target). **No new OpenAlex fetching is needed** to reach 2000 training rows — it is purely a
re-derive of `training.db` from data already collected.

Config changes:
- `.env.workflow`: `MAX_TRAINING_ROWS=200` -> `2500` (`SELECTION_SEED=45` unchanged — same
  deterministic subset-selection seed, just a bigger cap).
- `config.yaml`: `max_papers: 200` -> `2500`, `split_ratios: [0.50, 0.25, 0.25]` ->
  `[0.80, 0.10, 0.10]` (100/50/50 -> 2000 train / 250 val / 250 test, per explicit request),
  `dataset_id: wf01_demo200v3` -> `wf01_demo2500v1` (cache-invalidation key — MUST change
  whenever corpus size changes, per the file's own convention).
- `n_runs: 3` -> `5` on every active `runs:` queue entry (`demo_specter2_base`,
  `demo_sklearn_logreg`, `demo_specter2_adapter`, `demo_specter2_ae256`, new
  `demo_sklearn_linreg`) so the new outlier-exclusion machinery (Section 2, `min_n: 5`) has
  real data to work with on the very first run after this change.
- Hyperparameters hand-tuned (not automated search, per explicit choice) for ~1000 train rows
  instead of ~100: `hidden: [128, 64]` -> `[256, 128]`, `batch_size: 32` -> `64`,
  `max_epochs: 100` -> `150`, `patience: 15` -> `20` on the three MLP entries; the autoencoder
  entry's own `encoder_max_epochs: 60` -> `80`, `encoder_patience: 10` -> `12`. `dropout: 0.3`
  and `lr: 0.001` left unchanged — no evidence either needs to move, and overfitting-focused
  tuning is explicitly deferred to a later step.
- `fetch_target_rows` / `fetch_new_rows` / `fetch_from_scratch` / `fetch_wait_days` (the
  9000->18000 growth-campaign knobs already present from before the holiday): **untouched** —
  orthogonal to this task, stays ready for a later, separate collection run.

**Running it** (user executes, not Claude, per `CLAUDE.md`): re-deriving `training.db` must
NOT trigger a fresh OpenAlex fetch (today's `fetch_new_rows: 9000` would otherwise try to
collect 9000 MORE papers on top of the 9000 already stored). `ENABLE_COPY=false` already makes
`extend` "only re-derive training.db from what is already stored" per the code's own comment —
the safe invocation additionally omits `openalex`/`s2` from `--sources` so nothing in
`config.yaml`'s fetch knobs can trigger network calls:
```bash
python -m workflow.src.main_extend_db extend --sources copy
python -m workflow.run_all_workflows
```

## 6a. Addendum — fixed-holdout split for a data-size sweep

After the above landed, the user clarified this feeds a data-size sweep: val/test must be
**exactly 250/250 and physically IDENTICAL across every step**, while train size varies
(2000 down to 100, more steps given later). The plain `split_ratios` mechanism can't do this
— it's proportional, so a smaller total pool changes which rows land in val/test even with
the same seed (verified: `make_split`'s stratified permutation depends on the pool size `n`).

Fix: a new fixed-holdout split path in `iwf/data/dataset.py`
(`make_fixed_holdout_split`, wired into `add_split_column` via new optional `val_n`/`test_n`/
`train_n`/`holdout_seed`/`train_select_seed` params — the default proportional behavior is
byte-for-byte unchanged when these are left at 0, so every existing caller, including every
smoke test, is untouched). It reuses `make_split` internally to carve out the val/test
holdout from the full pool (deterministic given a fixed pool + `holdout_seed`), then takes a
seeded, **nested** prefix of the remainder for train (`train_select_seed` reused across
steps means train_n=100 is a strict subset of train_n=500's train rows). Rows past `train_n`
are labelled `"unused"` — invisible to `split_masks` (which only ever matches
train/val/test), so nothing downstream needed to change. Wired through `Settings`
(`config.py`) and `main_prepare.py`; validated (both holdout_val_n/holdout_test_n must be set
together; `train_n` requires holdout mode).

**Caught during implementation**: `make_split`'s output is a fixed-width `<U5` numpy string
array (sized to fit "train"/"val"/"test"); assigning the 6-character `"unused"` into it
silently truncated to `"unuse"`. Fixed by widening to `object` dtype before the assignment.

**Operational consequence**: `max_papers`/`.env.workflow MAX_TRAINING_ROWS` must now stay
**fixed at 2500** (covering the largest planned train_n, 2000, plus the 500-row holdout) for
the entire sweep — shrinking them per step would change which physical rows even reach the
split step, breaking the identical-val/test guarantee. Only `config.yaml`'s new `train_n`
changes per step (plus `dataset_id`, still bumped every step so `main_prepare`'s parquet
cache doesn't silently serve a stale split). `training.db` itself only needs deriving once
for the whole sweep. `config.yaml`'s `holdout_val_n: 250` / `holdout_test_n: 250` /
`holdout_seed: 45` / `train_n: 2000` / `train_select_seed: 46` are the active values as of
this step; `split_ratios` stays as an inert fallback.

## 7. Explicitly out of scope this round

- Automated hyperparameter search (grid/random/Bayesian) — hand-tuned defaults only, per
  explicit choice.
- Overfitting-focused hyperparameter tuning — later step.
- The 100->250->... manual data-scaling plot — user's own manual work; existing
  `algorithm_comparison_train_test.png` / `_collect_correctness` infra already tracks
  per-run train+test correctness for every model (mlp/logreg/linreg alike), nothing to add.
- Final "future development direction" slide — presentation-only, not code.
- `class_weight="effective"` gap in the sklearn logreg baseline — left as-is.
- `fetch_target_rows`/`fetch_new_rows` growth campaign — untouched, later/separate.
- text-transformer (`tfm_*`) and LLM-judge (`llm_*`) workflows — already excluded from the
  active `runs:` queue (confirmed, not a new change), stay excluded this round.

## 8. Testing

Smoke-test gap closed: `tests/smoke.py` gets new offline cases exercising
`model_arch=logreg` and the new `model_arch=linreg` end-to-end through `run_experiment`
(neither currently exercised — verified by grep), plus a focused case for `accepted_seeds()`
covering: below-`min_n` (no exclusion), a clear single outlier excluded (the
47/49/51/52-vs-23 example), and no-outlier-present. Per root `CLAUDE.md`, Claude does not run
the suite or any training script itself — written so the user can run
`uv run python -m workflow.src.tests.smoke` and `python -m workflow.run_all_workflows`
themselves.

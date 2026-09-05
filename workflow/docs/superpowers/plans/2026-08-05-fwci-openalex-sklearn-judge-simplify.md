# FWCI target + OpenAlex-only + sklearn baseline + single-call LLM judge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the training target to OpenAlex FWCI, keep every plot/metric/benchmark
generic over `target_kind` (no more hardcoded target names), keep data collection scoped to
OpenAlex only, add a scikit-learn LogisticRegression baseline that plugs into the existing
multi-seed/plotting infra, add a train-vs-test "all algorithms" comparison figure, and
simplify the LLM-as-judge workflow to exactly one LLM call per record (no notes-rewrite
"training" step).

**Architecture:** All six asks map onto the existing two-file control surface
(`config.yaml` + `.env.workflow`) plus small, additive changes to already-generic modules
(`targets.py`, `evaluation/*`, `training/experiment.py`, `models/`). No new workflow type is
introduced for the sklearn baseline — it is a new `model_arch: logreg` value that special-cases
inside the existing `emb_clf` multi-seed loop, so it automatically gets every plot, benchmark,
and leaderboard the torch heads already get for free.

**Tech Stack:** Python, pandas/numpy, scikit-learn (`LogisticRegression`, already a project
dependency — no new install), matplotlib, PyYAML.

## Global Constraints

- Root `CLAUDE.md`: do NOT install packages, do NOT run scripts unless asked — this plan is
  config + code changes only, no `run_pipeline`/`run_all_workflows`/smoke-suite execution.
- `workflow/CLAUDE.md` §0: every new file lives under `workflow/`; never commit/push.
- `workflow/CLAUDE.md` §7: source `.py` files get at most a one-line (<=10 word) docstring per
  def/class; `config.yaml`/`.env.workflow` comments should be precise and list valid options.
- `workflow/CLAUDE.md` §6: ONE implementation per concern (metrics in `metrics.py`, plots in
  `plots.py`, multi-seed loop in `training/experiment.py`) — extend, never fork.
- `target_kind` is already a first-class Settings field with 4 valid values
  (`field_year_percentile | fwci_percentile | normalized_percentile | bayesian_impact`,
  `workflow/src/iwf/targets.py`) — switching to `fwci_percentile` needs no new corpus fetch,
  since `main_prepare.py` already computes and stores all four target columns per row.

---

## Task 1: Switch the target to FWCI + fix the two hardcoded `target_kind` benchmarks

**Files:**
- Modify: `workflow/config.yaml:55` (`target_kind`)
- Modify: `workflow/src/iwf/evaluation/benchmarks.py:95-181` (`build_holdout_benchmark`,
  `build_temporal_holdout`)
- Modify: `workflow/src/iwf/config.py` (new `target_kind()` / `target_min_refset()` accessors,
  next to `max_abstract_chars()`)
- Modify: `workflow/src/main_extend_db.py:600-609` (pass the two new accessors through)

**Interfaces:**
- Produces: `config.target_kind() -> str`, `config.target_min_refset() -> int` — read
  `config.yaml` directly (same pattern as `config.max_abstract_chars()`), used by
  `main_extend_db.py` which has no full `Settings` object at that call site.
- `build_holdout_benchmark(db_path, out_path=None, per_domain_n=500, seed=42, *,
  target_kind="field_year_percentile", target_min_refset=30)` — new keyword-only params,
  defaults preserve old behavior for any other caller.
- `build_temporal_holdout(db_path, out_path=None, recent_years=5, per_domain_n=500, seed=42, *,
  target_kind="field_year_percentile", target_min_refset=30)` — same.

- [ ] **Step 1: `config.yaml` — set the active target to FWCI**

  Change line 55 from `target_kind: field_year_percentile` to `target_kind: fwci_percentile`.
  Update the comment above it (line 53-54) to note the coverage/stability tradeoff:
  ```yaml
  # Impact target regressed / classified:
  #   field_year_percentile | fwci_percentile | normalized_percentile | bayesian_impact
  # fwci_percentile = OpenAlex's own field-normalized percentile (citation_normalized_percentile,
  # FWCI-rank fallback). Sparser than field_year_percentile (not every paper has it), but it is
  # OpenAlex's own stable, externally-defined score rather than one computed in this repo.
  target_kind: fwci_percentile
  ```

- [ ] **Step 2: `config.py` — add `target_kind()` / `target_min_refset()` accessors**

  Insert after `max_abstract_chars()` (around line 208):
  ```python
  def target_kind() -> str:
      """Active impact-target kind (config.yaml, default field_year_percentile)."""
      try:
          cfg = load_config_yaml()
      except Exception:
          cfg = {}
      return str(cfg.get("target_kind") or "field_year_percentile").lower()


  def target_min_refset() -> int:
      """Min (field x year) cell size before pooling up a level (config.yaml, default 30)."""
      try:
          cfg = load_config_yaml()
      except Exception:
          cfg = {}
      if cfg.get("target_min_refset") is not None:
          return max(1, _coerce("target_min_refset", "int", cfg["target_min_refset"]))
      return 30
  ```

- [ ] **Step 3: `benchmarks.py` — stop hardcoding `"field_year_percentile"`**

  In `build_holdout_benchmark`, change the signature to:
  ```python
  def build_holdout_benchmark(db_path: Path, out_path: Path | None = None,
                              per_domain_n: int = 500, seed: int = 42, *,
                              target_kind: str = "field_year_percentile",
                              target_min_refset: int = 30) -> Path:
      """Build holdout_extended benchmark from extended.db: stratified sample per domain."""
  ```
  and change line 117 from
  `target_series = compute_target("field_year_percentile", sample_df, min_refset=30)` to
  `target_series = compute_target(target_kind, sample_df, min_refset=target_min_refset)`.
  Also add `manifest["target_kind"] = target_kind` to the manifest dict (line ~126-131) so a
  benchmark built under one target is traceable later.

  Apply the identical change to `build_temporal_holdout` (signature + line 167 + manifest).

- [ ] **Step 4: `main_extend_db.py` — pass the run's target through**

  At the call site (lines 600-609), pass the two new config accessors:
  ```python
  from workflow.src.iwf.evaluation.benchmarks import (
      build_holdout_benchmark, build_temporal_holdout)
  from workflow.src.iwf import config as _config
  train_db = fbase.extended_db_path().parent / "training.db"
  tk, tmr = _config.target_kind(), _config.target_min_refset()
  build_holdout_benchmark(train_db, target_kind=tk, target_min_refset=tmr)
  try:
      build_temporal_holdout(train_db, target_kind=tk, target_min_refset=tmr)
  except Exception as e:
      print(f"[extend] temporal-holdout benchmark skipped: {e}")
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add workflow/config.yaml workflow/src/iwf/config.py \
          workflow/src/iwf/evaluation/benchmarks.py workflow/src/main_extend_db.py
  git commit -m "feat: switch training target to FWCI, thread target_kind through local benchmarks"
  ```

---

## Task 2: Make every plot generically print the configured target name

**Files:**
- Modify: `workflow/src/iwf/evaluation/plots.py:1540-1542` (`plot_score_distributions` default),
  `workflow/src/iwf/evaluation/plots.py:413-444` (`plot_seed_spread_figures`)
- Modify: `workflow/src/iwf/workflow_results.py:135-139` (the one caller of
  `plot_seed_spread_figures`)

**Interfaces:**
- `plot_seed_spread_figures(wf_type, task, seeds, agg, class_names, class_thresholds, *,
  target_label="field_year_percentile")` — new keyword-only param, used for the regression
  scatter's axis label.

- [ ] **Step 1: `plot_score_distributions` — make the mislabeling impossible**

  Both real callers (`data_analysis.py:685,824`) already pass `target_label=target_kind`
  explicitly, so the default is a dead trap that would silently mislabel a forgotten future
  caller. Change line 1540-1542 from
  ```python
  def plot_score_distributions(scores: pd.DataFrame, path: Path, *, color: str | None = None,
                               title: str | None = None,
                               target_label: str = "field_year_percentile") -> None:
  ```
  to a required keyword-only argument:
  ```python
  def plot_score_distributions(scores: pd.DataFrame, path: Path, *, target_label: str,
                               color: str | None = None, title: str | None = None) -> None:
  ```
  (no behavior change for existing callers — they already pass it.)

- [ ] **Step 2: `plot_seed_spread_figures` — thread `target_label` into the reg scatter**

  Change the signature (line 413-414) to add `target_label: str = "field_year_percentile"` as a
  keyword-only param, and change line 440 from
  `ax.set_xlabel("true percentile"); ax.set_ylabel("predicted")` to
  `ax.set_xlabel(f"true {target_label}"); ax.set_ylabel(f"predicted {target_label}")`.

- [ ] **Step 3: `workflow_results.py` — pass the run's real target through**

  Change the call at line 136-139 from
  ```python
  plot_seed_spread_figures(wf_type, task, seeds, agg,
                           list(getattr(settings, "class_names", []) or []),
                           list(getattr(settings, "class_thresholds", []) or []))
  ```
  to
  ```python
  plot_seed_spread_figures(wf_type, task, seeds, agg,
                           list(getattr(settings, "class_names", []) or []),
                           list(getattr(settings, "class_thresholds", []) or []),
                           target_label=getattr(settings, "target_kind", "field_year_percentile"))
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add workflow/src/iwf/evaluation/plots.py workflow/src/iwf/workflow_results.py
  git commit -m "fix: plots always print the configured target_kind, never a hardcoded name"
  ```

---

## Task 3: Scope data collection to OpenAlex only

**Files:**
- Modify: `workflow/.env.workflow:38-51` (`ENABLE_*` toggles)

**Interfaces:** none (config-only).

- [ ] **Step 1: Flip the OpenAlex toggle on, keep every other source off**

  `training_sources: [openalex]` in `config.yaml` already restricts *training* to OpenAlex-
  sourced rows; this step makes *collection* (the `extend` step's per-source fetch) match it,
  so a future `main_extend_db extend` run only talks to OpenAlex:
  ```
  ENABLE_COPY=false            # copy raw.db + legacy store DBs INTO extended.db (off = openalex only)
  ENABLE_OPENALEX=true         # OpenAlex crawl — the ONLY collection source for this config
  ENABLE_S2=false              # Semantic Scholar gap-fill (off — OpenAlex only)
  ENABLE_CROSSREF=false        # Crossref (off — OpenAlex only)
  ENABLE_ARXIV=false           # arXiv (off — OpenAlex only)
  ENABLE_UEFISCDI=false        # UEFISCDI/JCR seeding (off — OpenAlex only)
  ENABLE_PDF=false             # UEFISCDI/JCR PDF extraction (off — OpenAlex only)
  ENABLE_FILL=false            # API metadata gap-fill (off — mixes non-OpenAlex sources)
  ```
  Update the block comment above (lines 33-37) to describe the OpenAlex-only intent instead of
  the "no collection this run" note, and note this is a config change only — nothing is fetched
  until the user runs `main_extend_db extend` themselves.

- [ ] **Step 2: Commit**

  ```bash
  git add workflow/.env.workflow
  git commit -m "config: scope corpus collection to OpenAlex only"
  ```

---

## Task 4: scikit-learn LogisticRegression baseline (`model_arch: logreg`)

**Files:**
- Create: `workflow/src/iwf/models/sklearn_baseline.py`
- Modify: `workflow/src/iwf/training/experiment.py:39-105` (`_train_one_seed`)
- Modify: `workflow/src/iwf/config.py:429-436` (`Settings.validate()` model_arch check)
- Modify: `workflow/config.yaml` (`model_arch` comment; new `demo_sklearn_logreg` `runs:` entry)

**Interfaces:**
- Produces: `SklearnLogisticRegressionClassifier(n_classes, seed=42,
  class_weight="effective", max_iter=2000)` — duck-typed like `TorchMLPClassifier`:
  `.task = "clf"`, `.net_ = None`, `.history_: dict`, `.fit(X_train, y_train, X_val=None,
  y_val=None)`, `.predict(X) -> np.ndarray[int]`, `.predict_proba(X) -> np.ndarray`.
- Consumes: nothing project-specific beyond `sklearn.linear_model.LogisticRegression`
  (already an installed dependency — used elsewhere in `dim_selector.py`/`encoder/pca_umap.py`).

- [ ] **Step 1: Write `models/sklearn_baseline.py`**

  ```python
  """scikit-learn LogisticRegression baseline — same interface as the torch heads."""

  from __future__ import annotations

  import numpy as np
  from sklearn.linear_model import LogisticRegression


  class SklearnLogisticRegressionClassifier:
      """Multinomial logistic regression baseline for the ordinal impact classes."""

      task = "clf"
      net_ = None

      def __init__(self, *, n_classes: int, seed: int = 42,
                  class_weight: str = "effective", max_iter: int = 2000):
          self.n_classes = int(n_classes)
          self.seed = int(seed)
          sk_weight = "balanced" if (class_weight or "").lower() == "balanced" else None
          self.model_ = LogisticRegression(max_iter=int(max_iter), random_state=self.seed,
                                           class_weight=sk_weight)
          self.history_: dict = {}

      def fit(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
          """Fit on TRAIN; val is accepted for interface parity but unused (no early stop)."""
          self.model_.fit(X_train, y_train)
          return self

      def predict(self, X: np.ndarray) -> np.ndarray:
          return self.model_.predict(X)

      def predict_proba(self, X: np.ndarray) -> np.ndarray:
          """Full [N, n_classes] proba matrix even if a class was absent from TRAIN."""
          proba = np.zeros((len(X), self.n_classes))
          raw = self.model_.predict_proba(X)
          for j, c in enumerate(self.model_.classes_):
              proba[:, int(c)] = raw[:, j]
          return proba
  ```

- [ ] **Step 2: `training/experiment.py` — branch on `model_arch == "logreg"`**

  In `_train_one_seed` (around line 61-68), change:
  ```python
  if task == "clf":
      model = TorchMLPClassifier(n_classes=len(class_names),
                                 class_weight=settings.class_weight,
                                 effective_beta=settings.effective_beta,
                                 loss=settings.loss, focal_gamma=settings.focal_gamma,
                                 label_smoothing=settings.label_smoothing, **common)
  else:
      model = TorchMLPRegressor(**common)
  ```
  to:
  ```python
  if task == "clf" and (settings.model_arch or "").lower() == "logreg":
      from workflow.src.iwf.models.sklearn_baseline import SklearnLogisticRegressionClassifier
      model = SklearnLogisticRegressionClassifier(n_classes=len(class_names),
                                                   seed=settings.model_seed + i,
                                                   class_weight=settings.class_weight)
  elif task == "clf":
      model = TorchMLPClassifier(n_classes=len(class_names),
                                 class_weight=settings.class_weight,
                                 effective_beta=settings.effective_beta,
                                 loss=settings.loss, focal_gamma=settings.focal_gamma,
                                 label_smoothing=settings.label_smoothing, **common)
  else:
      model = TorchMLPRegressor(**common)
  ```
  Everything below (fit/predict/proba/save) is already duck-typed and needs no further change —
  `getattr(model, "net_", None)` is already `None`-safe for the `.to("cpu")` skip.

- [ ] **Step 3: `config.py` — validate `model_arch: logreg` (clf-only)**

  In `Settings.validate()` (around line 429-436), change:
  ```python
  try:
      from workflow.src.iwf.models.architectures import available_archs
      archs = available_archs()
      if (self.model_arch or "").lower() not in archs:
          errors.append(f"model_arch={self.model_arch!r} is not registered; choose from {archs}.")
  except Exception as e:
      print(f"[config] (note) could not verify model_arch against the registry: {e}")
  ```
  to:
  ```python
  try:
      from workflow.src.iwf.models.architectures import available_archs
      valid_archs = [*available_archs(), "logreg"]
      arch = (self.model_arch or "").lower()
      if arch not in valid_archs:
          errors.append(f"model_arch={self.model_arch!r} is not registered; choose from {valid_archs}.")
      elif arch == "logreg" and WORKFLOWS.get(wt, {}).get("task") == "reg":
          errors.append("model_arch='logreg' (scikit-learn baseline) only supports "
                        "classification workflows, not regression.")
  except Exception as e:
      print(f"[config] (note) could not verify model_arch against the registry: {e}")
  ```
  (`wt` is already computed earlier in `validate()` as `self.workflow_type.lower()`.)

- [ ] **Step 4: `config.yaml` — document `logreg` and add the demo run**

  Update the §2 comment (line 62): `model_arch: mlp        # mlp | resmlp | ft_transformer |
  logreg (scikit-learn baseline, classification only, ignores hidden/dropout/lr/etc.)`.

  Add a new `runs:` entry after `demo_specter2_base` (§7), and bump that entry plus
  `demo_specter2_adapter`/`demo_specter2_ae256`/`demo_trf_tiny`/`demo_llm_judge` to `n_runs: 3`
  (see Task 6 for the judge side) so every algorithm in the comparison plot has the same 3-seed
  footing the user asked for:
  ```yaml
  # (1b) scikit-learn LogisticRegression baseline — the simplest possible classifier, run
  #      alongside the neural baselines so the comparison plot has a non-neural reference point.
  - name: demo_sklearn_logreg
    workflow: emb_clf
    model_arch: logreg
    n_runs: 3
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add workflow/src/iwf/models/sklearn_baseline.py workflow/src/iwf/training/experiment.py \
          workflow/src/iwf/config.py workflow/config.yaml
  git commit -m "feat: add scikit-learn LogisticRegression baseline as model_arch=logreg"
  ```

---

## Task 5: Train+test "all algorithms" comparison figure, 3 seeds

**Files:**
- Modify: `workflow/src/iwf/training/experiment.py` (promote train-split metrics to root,
  prefixed `train_`)
- Modify: `workflow/src/iwf/training/text_trainer.py` (same promotion)
- Modify: `workflow/src/main_train_judge.py` (same promotion — done as part of Task 6's rewrite)
- Modify: `workflow/src/iwf/workflow_results.py` (new comparison figure)

**Interfaces:**
- Every `result.json` gains `train_<metric>` root keys (e.g. `train_macro_f1`,
  `train_spearman`) alongside the existing unprefixed test metrics — purely additive, no
  existing reader breaks.
- Produces: `workflow_results._plot_algorithm_comparison(comparison, summary_dir) -> None`,
  called from `cross_workflow_comparison()`; writes
  `results/summary/algorithm_comparison_train_test.png`.

- [ ] **Step 1: `experiment.py` — also flatten the TRAIN split to root**

  In `_train_one_seed`'s per-split loop (lines 75-89), change:
  ```python
  for split in ("train", "val", "test"):
      m = masks[split]
      y_true, y_pred = y[m], model.predict(Xs[m])
      metrics_d = (classification_metrics(y_true, y_pred, class_names) if task == "clf"
                  else regression_metrics(y_true, y_pred))
      metrics_d["y_true"] = y_true.tolist()
      metrics_d["y_pred"] = y_pred.tolist()
      result[split] = metrics_d
      if split == "test":
          result.update({k: v for k, v in metrics_d.items()
                        if k not in ("y_true", "y_pred")})
          if task == "clf":
              result["proba_test"] = model.predict_proba(Xs[m]).tolist()
  ```
  to (add an `elif split == "train"` branch):
  ```python
  for split in ("train", "val", "test"):
      m = masks[split]
      y_true, y_pred = y[m], model.predict(Xs[m])
      metrics_d = (classification_metrics(y_true, y_pred, class_names) if task == "clf"
                  else regression_metrics(y_true, y_pred))
      metrics_d["y_true"] = y_true.tolist()
      metrics_d["y_pred"] = y_pred.tolist()
      result[split] = metrics_d
      if split == "test":
          result.update({k: v for k, v in metrics_d.items()
                        if k not in ("y_true", "y_pred")})
          if task == "clf":
              result["proba_test"] = model.predict_proba(Xs[m]).tolist()
      elif split == "train":
          result.update({f"train_{k}": v for k, v in metrics_d.items()
                        if k not in ("y_true", "y_pred") and isinstance(v, (int, float))})
  ```

- [ ] **Step 2: `text_trainer.py` — same promotion in both `best_metrics` constructions**

  After building `best_metrics` in the `if score > best_score...` branch (around line 273-286)
  and in the `if best_metrics is None` fallback branch (around line 299-306), add right after
  each dict literal:
  ```python
  best_metrics.update({f"train_{k}": v for k, v in train_m.items()
                       if isinstance(v, (int, float))})
  ```

- [ ] **Step 3: `workflow_results.py` — new train-vs-test comparison figure**

  Add a new `_panel_train_vs_test` next to `_panel` (after line 566) and call it from
  `_plot_cross_workflow` in a **second** figure (keep the existing test-only leaderboard as-is):
  ```python
  def _panel_train_vs_test(ax, wf_names, comparison, metric, title):
      """Grouped horizontal bars: train vs test mean+-std of one metric per workflow."""
      labels, train_m, train_s, test_m, test_s = [], [], [], [], []
      for wf in wf_names:
          c = comparison.get(wf)
          if not c:
              continue
          te = c.get("metrics", {}).get(metric)
          if not te:
              continue
          tr = c.get("metrics", {}).get(f"train_{metric}")
          labels.append(wf)
          train_m.append(tr["mean"] if tr else np.nan)
          train_s.append(tr["std"] if tr else 0.0)
          test_m.append(te["mean"])
          test_s.append(te["std"])
      if not labels:
          ax.text(0.5, 0.5, "(no runs)", ha="center", va="center", transform=ax.transAxes,
                  fontsize=10, color=palette.GRAY)
          ax.set_title(title)
          return
      y = np.arange(len(labels))
      h = 0.35
      ax.barh(y - h / 2, train_m, xerr=train_s, height=h, color=palette.DUSK_BLUE,
             alpha=0.85, label="train", capsize=3)
      ax.barh(y + h / 2, test_m, xerr=test_s, height=h, color=palette.OLIVE,
             alpha=0.85, label="test", capsize=3)
      ax.set_yticks(y)
      ax.set_yticklabels(labels, fontsize=max(6, 10 - len(labels) // 4))
      ax.set_ylim(-0.6, len(labels) - 0.4)
      finite = [v for v in (*train_m, *test_m) if np.isfinite(v)]
      hi = max(finite) if finite else 1.0
      ax.set_xlim(min(0.0, *finite, 0.0) if finite else 0.0, min(1.05, hi * 1.18 + 0.05))
      ax.set_xlabel(metric)
      ax.set_title(title)
      ax.legend(fontsize=8)


  def _plot_algorithm_comparison(comparison: dict, summary_dir: Path) -> None:
      """Train-vs-test figure across every trained algorithm (all model_arch/workflow types),
      mean+-std across seeds — the 'all algorithms, train+test' comparison view."""
      reg_wfs = sorted(k for k, v in comparison.items() if v.get("task") == "reg")
      clf_wfs = sorted(k for k, v in comparison.items() if v.get("task") == "clf")
      n_max = max(len(reg_wfs), len(clf_wfs), 1)
      height = min(11.0, max(2.6, n_max * 0.6 + 1.6))
      palette.apply_style()
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, height))
      _panel_train_vs_test(ax1, reg_wfs, comparison, "spearman", "Regression (Spearman rho)")
      _panel_train_vs_test(ax2, clf_wfs, comparison, "macro_f1", "Classification (macro-F1)")
      fig.suptitle("All-algorithm comparison — train vs test (mean +- std across seeds)")
      fig.tight_layout()
      fig.savefig(summary_dir / "algorithm_comparison_train_test.png", dpi=120)
      plt.close(fig)
      print(f"[results] algorithm comparison -> "
            f"{summary_dir / 'algorithm_comparison_train_test.png'}")
  ```
  Then in `cross_workflow_comparison()` (around line 486-494), call it alongside the existing
  figure:
  ```python
  if comparison:
      try:
          _write_cross_workflow_csv(comparison, summary_dir)
      except Exception as e:
          print(f"[results] cross-workflow CSV skipped ({e})")
      try:
          _plot_cross_workflow(comparison, summary_dir)
      except Exception as e:
          print(f"[results] cross-workflow figure skipped ({e})")
      try:
          _plot_algorithm_comparison(comparison, summary_dir)
      except Exception as e:
          print(f"[results] algorithm comparison figure skipped ({e})")
  ```
  No change is needed to `_write_cross_workflow_csv` — it already iterates every key in
  `metrics` generically, so `train_spearman`/`train_macro_f1` appear in the existing CSV once
  `experiment.py`/`text_trainer.py`/`main_train_judge.py` start writing them.

- [ ] **Step 4: Commit**

  ```bash
  git add workflow/src/iwf/training/experiment.py workflow/src/iwf/training/text_trainer.py \
          workflow/src/iwf/workflow_results.py
  git commit -m "feat: train-vs-test all-algorithm comparison figure across seeds"
  ```

---

## Task 6: LLM-as-judge — one call per record, no notes-rewrite "training"

**Files:**
- Modify (large rewrite): `workflow/src/main_train_judge.py`
- Modify: `workflow/src/iwf/config.py` (drop the now-unused judge-training Settings fields)
- Modify: `workflow/config.yaml` (drop the now-unused §5b keys; simplify `demo_llm_judge`)
- Modify: `workflow/.env.workflow` (drop `JUDGE_NOTE_TOKENS` / `LLM_JUDGE_MAX_CONTEXT_TOKENS` —
  no longer read by anything)
- Modify: `workflow/src/iwf/workflow_results.py:263-285` (`write_model_description` judge block)

**Interfaces:**
- `_train_seed(df, task, settings, seed_idx, seed_dir, embed_col, progress=None) -> dict` —
  signature unchanged (the smoke test calls it directly), behavior simplified: exactly one
  `_predict_impact` LLM call per TRAIN/VAL/TEST row, notes fixed at `SEED_NOTES` for the whole
  run, RAG store built deterministically (no LLM-authored chunk rewrite).
- Removes: `_rewrite_batch`, `_flush_batch`/`_would_exceed_batch_budget`,
  `_extract_notes_and_chunks`, `_predict_subset_cached`, `_checkpoint_val_score`,
  `_checkpoint_metric`, the VAL-checkpoint escalation loop, `judge_passes` multi-pass loop.
- Removes Settings fields: `judge_passes`, `judge_batch_max_papers`,
  `judge_checkpoint_interval_pct`, `judge_val_eval_start_pct`, `judge_val_eval_max_pct`,
  `judge_early_stop_margin`, `judge_error_gate_threshold`, `judge_good_sample_fraction`.
- Keeps Settings/env knobs still in use: nothing from Settings (the whole judge is now
  parameter-free beyond `n_runs`); `.env.workflow`'s `JUDGE_RAG_TOKEN_BUDGET` (retrieval budget)
  and `JUDGE_RAG_CHUNK_TOKENS` (chunk length cap) are still read (retrieval + chunk-building are
  unchanged).

- [ ] **Step 1: Rewrite `main_train_judge.py`**

  Replace the whole file with the simplified version below. Kept unchanged, verbatim:
  `JUDGE_SYSTEM`, `SEED_NOTES`, `_token_estimate`, `_retrieve`, `_IMPACT_KEYS`,
  `_extract_impact_value`, `_predict_impact`, `_make_chunk`, `_embed_col`, `_class_ids`,
  `_predict_split`, `main`. Removed: `_MIN_VAL_EVAL`, `_BATCH_FIXED_OVERHEAD_CHARS`,
  `_extract_notes_and_chunks`, `_rewrite_batch`, `_would_exceed_batch_budget`,
  `_checkpoint_metric`, `_predict_subset_cached`, `_checkpoint_val_score`. Rewritten:
  `_train_seed` (module docstring updated to match).

  ```python
  """Phase 6 (train, LLM-as-judge workflows) — RAG store of precedents as the "model".

  Used by llm_clf / llm_reg, which skip encoder/reduce (phases 3,4 — see
  config.WORKFLOWS): phase 2 (embed) already computes the frozen specter2 embedding this
  module reuses for retrieval, so there is no latent-dim encoder to train.

  There is no training step: the LLM is called EXACTLY ONCE per record (train, val, and
  test alike) to judge that paper's impact from its title/abstract plus retrieved
  precedents, given a fixed rubric (``SEED_NOTES``) that is never rewritten. Per TRAIN
  paper: (1) retrieve the most similar already-seen precedent chunks within a token
  budget, (2) the LLM predicts this paper's impact in [0,1] given the fixed notes +
  retrieved chunks (always blind — it never sees the true value), (3) a deterministic RAG
  chunk (title/abstract/true impact, no LLM call) is added to the store immediately, so
  later TRAIN papers retrieve richer precedents than earlier ones. VAL/TEST reuse the
  final TRAIN-built store and the same fixed notes, predict-only.

  Every LLM call is best-effort (workflow.src.iwf.llm.client.complete returns None on any
  error - missing key, network, bad JSON) and degrades to a deterministic fallback (the
  mean impact of the retrieved neighbours, or 0.5 with no neighbours yet), so a run with no
  LLM endpoint configured still completes and produces a (weaker, kNN-only) result.

      python -m workflow.src.main_train_judge          # workflow type from config.yaml
  """

  from __future__ import annotations

  import argparse
  import json
  from pathlib import Path

  import joblib
  import numpy as np
  import pandas as pd

  from workflow.src.iwf import config, paths, workflow_results
  from workflow.src.iwf.config import WORKFLOWS, Settings, is_judge_workflow, load_settings
  from workflow.src.iwf.data.dataset import load_parquet, split_masks
  from workflow.src.iwf.evaluation.metrics import classification_metrics, regression_metrics
  from workflow.src.iwf.llm import client as llm_client

  JUDGE_SYSTEM = (
      "You are an expert research-impact judge. You estimate how impactful a scientific "
      "paper's future citations will be, purely from its title and abstract, using the "
      "general principles you have learned and any similar past papers you are shown. "
      "Be concise: never include reasoning or commentary outside the exact JSON schema asked "
      "for — extra explanation wastes tokens and is not read."
  )

  SEED_NOTES = (
      "Judging principles:\n"
      "- A clear, specific, novel contribution tends to have higher impact than vague or "
      "purely incremental work.\n"
      "- Broad relevance across a field, or interdisciplinary appeal, often correlates with "
      "higher citation impact.\n"
      "- Strong methodological rigor and reproducibility support are usually favorable.\n"
      "- Timeliness (addressing a current, actively-studied open problem) tends to increase "
      "impact.\n"
      "These are fixed for the whole run — every paper is judged against the same rubric."
  )


  def _token_estimate(text: str) -> int:
      """Rough token count (~4 chars/token), no tokenizer dependency needed."""
      return max(1, len(text) // 4)


  def _retrieve(vec: np.ndarray, chunk_vecs: list[np.ndarray], chunk_texts: list[str],
               chunk_targets: list[float], token_budget: int) -> tuple[str, list[float]]:
      """Most-similar chunks (cosine, specter2) up to ``token_budget`` tokens, best first.

      Returns the joined chunk text (for the LLM prompt) and the raw target values of the
      chunks actually included (for the deterministic mean-fallback when the LLM is down)."""
      if not chunk_vecs:
          return "", []
      C = np.stack(chunk_vecs)
      vn = vec / (np.linalg.norm(vec) + 1e-9)
      Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
      sims = Cn @ vn
      order = np.argsort(-sims)
      parts: list[str] = []
      targets: list[float] = []
      used = 0
      for idx in order:
          t = chunk_texts[idx]
          tok = _token_estimate(t)
          if parts and used + tok > token_budget:
              break
          parts.append(t)
          targets.append(chunk_targets[idx])
          used += tok
          if used >= token_budget:
              break
      return "\n---\n".join(parts), targets


  # Key names accepted for the impact score in the LLM's JSON reply. Real models don't
  # always use the exact requested key ("impact") — observed in testing that some models
  # substitute a near-synonym like "impact_score" — so several plausible keys are tried
  # before giving up, rather than silently discarding a real answer.
  _IMPACT_KEYS = ("impact", "impact_score", "score", "rating", "value")


  def _extract_impact_value(resp: dict) -> float | None:
      for key in _IMPACT_KEYS:
          if key in resp:
              try:
                  return float(resp[key])
              except (TypeError, ValueError):
                  continue
      for v in resp.values():          # last resort: first numeric value anywhere in the reply
          if isinstance(v, (int, float)):
              return float(v)
      return None


  def _predict_impact(notes: str, retrieved_text: str, title: str, abstract: str) -> float | None:
      """One LLM call: judge this paper's impact in [0,1] from notes + precedents. Always
      blind — the true value is never present in this prompt."""
      prompt = (
          f"General principles for judging paper impact:\n{notes}\n\n"
          + (f"Similar past papers (abstract + their actual impact score, 0=low, 1=high):\n"
             f"{retrieved_text}\n\n" if retrieved_text else "")
          + f"Now rate this paper's expected impact on a 0.0-1.0 scale "
            f"(0 = negligible, 1 = field-defining):\n"
            f"Title: {title}\nAbstract: {abstract}\n\n"
            'Respond with ONLY this JSON object, no other keys, no explanation, no markdown '
            'fences: {"impact": <float between 0 and 1>}'
      )
      resp = llm_client.complete(prompt, system=JUDGE_SYSTEM, as_json=True)
      if isinstance(resp, dict):
          val = _extract_impact_value(resp)
          if val is not None:
              return min(1.0, max(0.0, val))
      return None


  def _make_chunk(title: str, abstract: str, true_val: float, max_tokens: int) -> str:
      """Deterministic RAG chunk (no LLM call) added right after a paper is judged."""
      text = f"Title: {title}\nAbstract: {abstract}\nActual impact score (0-1): {true_val:.3f}"
      return text[: max_tokens * 4]


  def _embed_col(df: pd.DataFrame, settings: Settings) -> str:
      """The frozen embedding column to retrieve on — prefers specter2 (what the plan's kNN
      pattern uses); falls back to embed_for_encoder or any emb_ column if specter2 is absent."""
      for cand in (f"emb_{settings.embed_for_encoder}", "emb_specter2"):
          if cand in df.columns:
              return cand
      others = [c for c in df.columns if c.startswith("emb_")]
      if others:
          return others[0]
      raise KeyError("No emb_* column found — run the embed phase (2) before the judge trainer.")


  def _class_ids(vals, thresholds: list[float]) -> np.ndarray:
      """Bin continuous [0,1] impact values into ordinal class ids via the run's thresholds —
      exact for class_scheme='percentile' (the same rule that produced class_label), an
      approximation for class_scheme='css'."""
      thr = sorted(float(t) for t in thresholds)
      return np.digitize(np.asarray(vals, dtype=float), thr, right=False)


  def _predict_split(df_split: pd.DataFrame, embed_col: str, notes: str,
                     chunk_vecs: list[np.ndarray], chunk_texts: list[str],
                     chunk_targets: list[float], token_budget: int) -> tuple[np.ndarray, np.ndarray]:
      """Predict-only pass against the frozen (notes, RAG store) — one LLM call per row."""
      trues, preds = [], []
      for _, row in df_split.iterrows():
          title, abstract = str(row.get("title") or ""), str(row.get("abstract") or "")
          vec = np.asarray(row[embed_col], dtype=np.float32)
          retrieved_text, retrieved_targets = _retrieve(vec, chunk_vecs, chunk_texts,
                                                         chunk_targets, token_budget)
          predicted = _predict_impact(notes, retrieved_text, title, abstract)
          if predicted is None:
              predicted = float(np.mean(retrieved_targets)) if retrieved_targets else 0.5
          trues.append(float(row["target"]))
          preds.append(predicted)
      return np.asarray(trues, dtype=float), np.asarray(preds, dtype=float)


  def _train_seed(df: pd.DataFrame, task: str, settings: Settings, seed_idx: int,
                  seed_dir: Path, embed_col: str, progress=None) -> dict:
      """Build the RAG store from TRAIN (one LLM call per paper, fixed rubric), score
      VAL/TEST, write the standard result.json + model/ artifact contract."""
      seed_dir.mkdir(parents=True, exist_ok=True)
      model_dir = seed_dir / "model"
      model_dir.mkdir(parents=True, exist_ok=True)

      token_budget = config.get_int("JUDGE_RAG_TOKEN_BUDGET", 3000)
      chunk_tokens = config.get_int("JUDGE_RAG_CHUNK_TOKENS", 200)

      train_df = df[df["split"].astype(str) == "train"].reset_index(drop=True)
      n_train = len(train_df)

      rng = np.random.RandomState(settings.model_seed + seed_idx)
      train_order = rng.permutation(n_train)

      notes = SEED_NOTES
      chunk_vecs: list[np.ndarray] = []
      chunk_texts: list[str] = []
      chunk_targets: list[float] = []
      train_true: list[float] = []
      train_pred: list[float] = []

      for pos_i, pos in enumerate(train_order):
          row = train_df.iloc[pos]
          title, abstract = str(row.get("title") or ""), str(row.get("abstract") or "")
          true_val = float(row["target"])
          vec = np.asarray(row[embed_col], dtype=np.float32)

          retrieved_text, retrieved_targets = _retrieve(vec, chunk_vecs, chunk_texts,
                                                         chunk_targets, token_budget)
          predicted = _predict_impact(notes, retrieved_text, title, abstract)
          if predicted is None:
              predicted = float(np.mean(retrieved_targets)) if retrieved_targets else 0.5
          train_true.append(true_val)
          train_pred.append(predicted)

          chunk_texts.append(_make_chunk(title, abstract, true_val, chunk_tokens))
          chunk_vecs.append(vec)
          chunk_targets.append(true_val)

          if progress is not None and (pos_i % 5 == 0 or pos_i == n_train - 1):
              progress({"seed": seed_idx, "n_runs": settings.n_runs,
                        "epoch": pos_i + 1, "epochs": max(1, n_train), "val_metric": 0.0})

      masks = split_masks(df)
      val_true, val_pred = _predict_split(df[masks["val"]], embed_col, notes,
                                          chunk_vecs, chunk_texts, chunk_targets, token_budget)
      test_true, test_pred = _predict_split(df[masks["test"]], embed_col, notes,
                                            chunk_vecs, chunk_texts, chunk_targets, token_budget)

      if task == "clf":
          thr = settings.class_thresholds
          class_names = settings.class_names
          train_m = classification_metrics(_class_ids(train_true, thr),
                                           _class_ids(train_pred, thr), class_names)
          val_m = classification_metrics(_class_ids(val_true, thr),
                                         _class_ids(val_pred, thr), class_names)
          test_m = classification_metrics(_class_ids(test_true, thr),
                                          _class_ids(test_pred, thr), class_names)
          train_y_true, train_y_pred = _class_ids(train_true, thr), _class_ids(train_pred, thr)
          val_y_true, val_y_pred = _class_ids(val_true, thr), _class_ids(val_pred, thr)
          test_y_true, test_y_pred = _class_ids(test_true, thr), _class_ids(test_pred, thr)
      else:
          train_m = regression_metrics(train_true, train_pred)
          val_m = regression_metrics(val_true, val_pred)
          test_m = regression_metrics(test_true, test_pred)
          train_y_true, train_y_pred = np.asarray(train_true), np.asarray(train_pred)
          val_y_true, val_y_pred = np.asarray(val_true), np.asarray(val_pred)
          test_y_true, test_y_pred = np.asarray(test_true), np.asarray(test_pred)

      result: dict = {
          **test_m,
          "seed": int(seed_idx),
          "n_train": n_train,
          "train": {**train_m, "y_true": train_y_true.tolist(), "y_pred": train_y_pred.tolist()},
          "val": {**val_m, "y_true": val_y_true.tolist(), "y_pred": val_y_pred.tolist()},
          "test": {**test_m, "y_true": test_y_true.tolist(), "y_pred": test_y_pred.tolist()},
      }
      result.update({f"train_{k}": v for k, v in train_m.items() if isinstance(v, (int, float))})
      if task == "clf":
          n_classes = len(settings.class_thresholds) + 1
          proba = np.zeros((len(test_y_pred), n_classes))
          proba[np.arange(len(test_y_pred)), np.clip(test_y_pred, 0, n_classes - 1)] = 1.0
          result["proba_test"] = proba.tolist()

      (seed_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
      (model_dir / "notes.md").write_text(notes, encoding="utf-8")
      chunks_df = pd.DataFrame({
          "text": chunk_texts, "target": chunk_targets,
          "embedding": [v.tolist() for v in chunk_vecs],
      })
      chunks_df.to_parquet(model_dir / "rag_chunks.parquet", index=False)
      joblib.dump({
          "chunk_vecs": (np.stack(chunk_vecs) if chunk_vecs
                        else np.zeros((0, 1), dtype=np.float32)),
          "chunk_targets": np.asarray(chunk_targets, dtype=float),
          "notes": notes, "embed_col": embed_col,
      }, model_dir / "rag_index.joblib")

      test_flat = {k: round(v, 4) for k, v in result["test"].items()
                  if isinstance(v, (int, float))}
      print(f"  [{seed_dir.parent.name}] seed {seed_idx}: test={test_flat} "
            f"({n_train} train papers, 1 LLM call each)")
      return test_flat


  def main(settings: Settings | None = None, *, workflow_type: str | None = None,
          progress=None, wf_dir: Path | None = None) -> Path:
      s = settings or load_settings()
      wf = (workflow_type or s.workflow_type or "llm_reg").lower()
      spec = WORKFLOWS.get(wf)
      if spec is None or not is_judge_workflow(wf):
          raise ValueError(f"main_train_judge handles the llm_* workflows, not {wf!r}.")
      task = spec["task"]

      if wf_dir is None:
          wf_dir = workflow_results.next_workflow_dir(wf)
          wf_dir.mkdir(parents=True, exist_ok=True)

      df = load_parquet(paths.embedded_parquet(s.dataset_id))
      embed_col = _embed_col(df, s)
      if "target" not in df.columns:
          raise KeyError("Prepared parquet has no 'target' column - run the prepare phase first.")

      n_runs = max(1, int(s.n_runs))
      print(f"[train_judge] workflow={wf}  -> {wf_dir.name}  task={task}  "
            f"embed_col={embed_col}  seeds={n_runs}  (1 LLM call/record, no notes-rewrite)")

      workflow_results.write_run_config(wf, s, wf_dir)
      for i in range(n_runs):
          def _cb(info, _i=i):
              if progress is not None:
                  progress({**info})

          _train_seed(df, task, s, i, wf_dir / f"seed_{i}", embed_col,
                     progress=_cb if progress is not None else None)

      print(f"[train_judge] done -> {wf_dir}")
      return wf_dir


  if __name__ == "__main__":
      ap = argparse.ArgumentParser()
      ap.add_argument("--workflow", default=None,
                      help="override the workflow type from config.yaml (llm_clf | llm_reg)")
      args = ap.parse_args()
      main(workflow_type=args.workflow)
  ```

- [ ] **Step 2: `config.py` — drop the now-unused judge Settings fields**

  Remove these 8 dataclass fields from `Settings` (lines 362-369):
  `judge_passes`, `judge_batch_max_papers`, `judge_checkpoint_interval_pct`,
  `judge_val_eval_start_pct`, `judge_val_eval_max_pct`, `judge_early_stop_margin`,
  `judge_error_gate_threshold`, `judge_good_sample_fraction`. Replace the block comment above
  them with a one-line note that the judge workflow has no per-run science knobs left (single
  LLM call per record, fixed rubric).

- [ ] **Step 3: `config.yaml` — drop §5b's now-dead keys, simplify `demo_llm_judge`**

  Replace the whole §5b block (`judge_batch_max_papers` through `judge_good_sample_fraction`,
  lines 140-156) with a short note:
  ```yaml
  # ---------------------------------------------------------------------------
  # 5b. LLM-AS-JUDGE  (llm_clf / llm_reg only — see src/main_train_judge.py)
  # ---------------------------------------------------------------------------
  # No science knobs here: the judge calls the LLM exactly once per record (train/val/test)
  # against a fixed rubric — no notes-rewriting, no checkpointing, no multi-pass training.
  # n_runs (section 1) still controls how many seeds (shuffle orders) are trained.
  ```
  Change the `demo_llm_judge` `runs:` entry (§7) from
  ```yaml
  - name: demo_llm_judge
    workflow: [llm_clf]
    embedder: specter2
    judge_passes: 3
    n_runs: 1
  ```
  to
  ```yaml
  - name: demo_llm_judge
    workflow: [llm_clf]
    embedder: specter2             # embedding is used for RAG retrieval only
    n_runs: 3                      # 3 seeds, same footing as the other algorithms compared
  ```
  Update the queue's block comment (lines 276-284) to drop the "(5) LLM-as-judge ... 3 full
  TRAIN passes" line and replace it with "(5) LLM-as-judge: one LLM call per record, fixed
  rubric, no training step" and add a "(1b) scikit-learn LogisticRegression baseline" line
  matching Task 4 Step 4's new entry.

- [ ] **Step 4: `.env.workflow` — drop the two dead knobs**

  Remove `LLM_JUDGE_MAX_CONTEXT_TOKENS` (was only used to size rewrite batches) if present, and
  remove any `JUDGE_NOTE_TOKENS` line if present; keep `JUDGE_RAG_TOKEN_BUDGET` and
  `JUDGE_RAG_CHUNK_TOKENS` (still read). Grep the file first — `.env.workflow` as read earlier
  in this session did not show `JUDGE_NOTE_TOKENS`/`LLM_JUDGE_MAX_CONTEXT_TOKENS` as explicit
  lines (they may only be referenced via `config.get_int` defaults), so this step may be a
  no-op; confirm with a search before editing.

- [ ] **Step 5: `workflow_results.py` — simplify the judge model-description block**

  In `write_model_description` (lines 263-285), replace the judge `rows` list with:
  ```python
  rows = [
      ("Kind", "RAG store of precedents, ONE LLM call per record, fixed rubric "
              "(no notes rewriting, no trained network)"),
      ("Retrieval embedding", s.get("embed_for_encoder") or "specter2"),
      ("RAG token budget (.env.workflow)",
       _config.get_int("JUDGE_RAG_TOKEN_BUDGET", 3000)),
      ("RAG chunk token budget (.env.workflow)",
       _config.get_int("JUDGE_RAG_CHUNK_TOKENS", 200)),
  ]
  ```
  and update the free-text paragraph right below it (lines 313-321) to describe a fixed-rubric,
  one-call-per-record judge instead of "evolving notes.md"/"error-gated learning".

- [ ] **Step 6: Commit**

  ```bash
  git add workflow/src/main_train_judge.py workflow/src/iwf/config.py workflow/config.yaml \
          workflow/.env.workflow workflow/src/iwf/workflow_results.py
  git commit -m "refactor: LLM judge calls the LLM once per record, drops the notes-rewrite training step"
  ```

---

## Task 7: Sanity-check the smoke suite still describes reality (no execution)

**Files:**
- Read-only check: `workflow/src/tests/smoke.py` — `t_judge_workflow` (around line 561),
  `t_config` / `t_targets` (around lines 68-174).

**Interfaces:** none — this task is a read-and-confirm, not a code change, unless a genuine
mismatch is found.

- [ ] **Step 1: Confirm `t_judge_workflow` still matches the new `_train_seed` signature**

  `_train_seed(df, task, s, seed_idx, seed_dir, embed_col)` keeps its exact signature and
  `result.json`'s `test`/`train`/`val` sub-dicts keep their keys (`spearman`/`mae`/`r2` for reg,
  `macro_f1`/`accuracy` for clf, `proba_test` for clf) — the test's assertions
  (`res["test"].keys()`, `notes.md`/`rag_index.joblib`/`rag_chunks.parquet` existence) hold
  unchanged. The mock `fake_complete` returning `{"impact": 0.5, "notes": ..., "chunks": [...]}`
  is harmless — the simplified code only ever reads the `"impact"` key. **No edit expected**;
  only fix it if actually reading the file after Task 6 turns up a real mismatch.

- [ ] **Step 2: Confirm `t_config`/`t_targets` don't assert removed Settings fields**

  Neither test (per the earlier codebase survey) references `judge_passes` or the other 7
  removed fields — they iterate `ALL_TARGET_KINDS` and check `target_kind` generically. **No
  edit expected.**

- [ ] **Step 3: Tell the user how to verify**

  Do not run `uv run python -m workflow.src.tests.smoke` — root `CLAUDE.md` forbids running
  scripts unless the user asks. Report the command to the user so they can run it themselves.

---

## Self-Review Notes (from the plan author, before handing off)

- **Spec coverage**: (1) FWCI target -> Task 1. (2) generic plots/labels -> Task 1 Step 3
  (manifest) + Task 2. (3) OpenAlex-only collection -> Task 3. (4) 100/50/50 OpenAlex+FWCI split
  -> already satisfied by the current `config.yaml` (`max_papers: 200`, `split_ratios:
  [0.50,0.25,0.25]`, `training_sources: [openalex]`) once Task 1 flips `target_kind`; no code
  change needed there beyond documenting that FWCI's sparser coverage may realize fewer than
  exactly 100/50/50 rows if some OpenAlex rows lack `fwci`/`citation_normalized_percentile`.
  (5) classification experiment plumbing -> Task 4 (sklearn baseline) + Task 5 (comparison
  plot) + Task 4 Step 4 / Task 6 Step 3 (3-seed demo queue); the "simplify then increase
  difficulty" sub-bullet is an experimental methodology note for the user, not a code change —
  called out in the config.yaml queue comment. (6) single-call LLM judge -> Task 6.
- **Placeholder scan**: none — every step above has literal code, not "TBD"/"similar to task N".
- **Type consistency**: `SklearnLogisticRegressionClassifier`'s `.fit`/`.predict`/
  `.predict_proba` signatures match `TorchMLPClassifier`'s exactly, since `_train_one_seed`
  calls them polymorphically. `_train_seed`'s return type/signature in `main_train_judge.py` is
  unchanged from the pre-rewrite version (checked against its one external caller, `main()`, and
  the smoke test).

---

## Amendment (mid-execution, same session): judge retrieval/build-order redesign

The user refined Task 6's design further after initial implementation landed. Applied directly
to `workflow/src/main_train_judge.py` (superseding Task 6's `_train_seed`/`_retrieve`/
`_predict_split`):

- **RAG built first, deterministically, from TRAIN's true values** (`_build_rag`, no LLM call
  needed since TRAIN's targets are already known) — then TRAIN, VAL, and TEST are each judged
  exactly once against that ONE complete store (`_predict_split` reused for all three splits),
  so no split is judged against a sparser RAG than another ("all training data has equal chance
  of precision").
- **Retrieval**: top-`RETRIEVAL_TOP_K` (5) most similar chunks by cosine similarity on the
  existing frozen title+abstract embedding — documented as an abstract-dominated proxy for
  "abstract text only" similarity, since a genuinely separate abstract-only embedding pass was
  judged to be out of scope for "keep this simple, one LLM call per record."
- **Exact-abstract-match exclusion**: a candidate chunk whose stored abstract text exactly
  equals the query paper's own abstract is skipped — necessary now that a TRAIN paper's own
  chunk is already in the RAG when it is judged (previously not needed, since retrieval used to
  happen before a paper's own insertion).
- **Context-budget-aware retrieval** (`_effective_token_budget`): the retrieval token budget is
  `min(JUDGE_RAG_TOKEN_BUDGET, 50% of LLM_JUDGE_MAX_CONTEXT_TOKENS - system/rubric/query
  overhead)` whenever the context window is known; restored `LLM_JUDGE_MAX_CONTEXT_TOKENS` to
  `.env.workflow.example` for this purpose (Task 6 had removed it as dead).
- **System prompt redesign** (`JUDGE_SYSTEM`): explicitly instructs the LLM to treat retrieved
  precedents as loose calibration only, never to copy/average their scores — judge the paper on
  its own title/abstract content first.
- **`judge_max_train_papers`** (Settings + config.yaml, default 0 = all of TRAIN): caps how many
  TRAIN papers build the RAG store and get judged for train metrics, for cost control on huge
  corpora (e.g. 4000). Re-added to `Settings` (Task 6 had removed all `judge_*` fields).

`workflow_results.write_model_description`'s judge block and `.env.workflow.example` were
updated accordingly. `result.json` gained `n_train_used` (vs `n_train`) to record how many
TRAIN papers were actually used when `judge_max_train_papers` truncates.

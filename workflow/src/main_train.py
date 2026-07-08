"""Phase 6 (train) — one script for EVERY embedding/metadata head.

Replaces the former main_train_clf / main_train_reg / main_train_meta trio: the task
(clf/reg) and feature set (z | meta | all_meta) are derived from the workflow type via
``config.WORKFLOWS``, and everything funnels into the one shared ``run_experiment``.
Text workflows (features == "text") train in main_train_tfm instead.

Trains on the shared split: fit on train, early-stop on val; test is only used for
output/statistics. Class-imbalance compensation (class_weight) is handled inside the
classifier — a no-op when the configured classes are balanced.

    python -m workflow.src.main_train            # workflow type from config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from workflow.src.iwf import paths, workflow_results
from workflow.src.iwf.config import WORKFLOWS, Settings, load_settings
from workflow.src.iwf.data.dataset import load_parquet, save_parquet
from workflow.src.iwf.targets import to_classes
from workflow.src.iwf.training.experiment import run_experiment

NN_NEIGHBORS = 10   # kNN size for the novelty/feasibility geometry


def _ensure_class_label(df, s: Settings, path) -> None:
    """Derive class_label inline when the label phase was skipped (never a KeyError)."""
    if "class_label" not in df.columns:
        print("[train] no 'class_label' column - deriving it inline (label-phase logic).")
        df["class_label"] = to_classes(df["target"], scheme=s.class_scheme,
                                       thresholds=s.class_thresholds, names=s.class_names)
        save_parquet(df, path)


def _ensure_fused_columns(df, s: Settings):
    """Compute novelty/feasibility and external-impact scores if not already present."""
    from workflow.src.iwf.features.external_impact import get_external_models
    from workflow.src.iwf.features.novelty import novelty_feasibility

    Z = np.array(df["z"].tolist(), dtype=np.float32)
    if (s.use_novelty or s.use_feasibility) and "novelty" not in df.columns:
        nov, feas = novelty_feasibility(Z, k=NN_NEIGHBORS)
        df["novelty"] = nov
        df["feasibility"] = feas
    for model in get_external_models(s.external_impact):
        if model.name not in df.columns:
            cache = paths.CACHE_DIR / f"{model.name}_cache.json"
            df[model.name] = model.score_dataframe(df, cache)
    return df


def main(settings: Settings | None = None, *, workflow_type: str | None = None,
         wf_dir: Path | None = None) -> Path:
    """Train the head for one embedding/metadata workflow into ``wf_dir/seed_<i>/``."""
    s = settings or load_settings()
    wf = (workflow_type or s.workflow_type).lower()
    spec = WORKFLOWS.get(wf)
    if spec is None or spec["features"] == "text":
        raise ValueError(f"main_train handles the embedding workflows, not {wf!r}.")
    task, features = spec["task"], spec["features"]
    bypass = features == "all_meta"     # intentional-leakage upper bound

    parquet = paths.reduced_parquet(s.dataset_id)
    df = load_parquet(parquet)
    if task == "clf":
        _ensure_class_label(df, s, parquet)

    if features == "z":
        scalar_cols: list[str] = []
    else:
        from workflow.src.iwf.features.feature_builder import (
            all_scalar_columns, scalar_feature_columns,
        )
        df = _ensure_fused_columns(df, s)
        save_parquet(df, parquet)
        if bypass:
            scalar_cols = all_scalar_columns(df)
            print(f"[train] {wf}: leakage guard BYPASSED on purpose - using all "
                  f"{len(scalar_cols)} numeric columns (upper-bound test)")
        else:
            scalar_cols = scalar_feature_columns(
                df, s.use_metadata, s.use_novelty, s.use_feasibility, s.external_impact
            )

    if wf_dir is None:
        wf_dir = workflow_results.next_workflow_dir(wf)
        wf_dir.mkdir(parents=True, exist_ok=True)

    label = f"{features} features -> " + ("impact class" if task == "clf"
                                          else "normalized impact")
    print(f"[train] {wf}  task={task}  features={features}  "
          f"scalar features: {len(scalar_cols)}  -> {wf_dir.name}")
    run_experiment(df, task=task, settings=s, scalar_cols=scalar_cols,
                   out_dir=wf_dir, label=label, bypass_guard=bypass)
    print(f"[train] done -> {wf_dir}")
    return wf_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", default=None,
                    help="override the workflow type from config.yaml")
    args = ap.parse_args()
    main(workflow_type=args.workflow)

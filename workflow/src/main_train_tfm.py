"""Phase 6 (train, text workflows) — fine-tune a transformer directly on title+abstract.

Used by the tfm_clf / tfm_reg workflows, which skip embed/encode/reduce entirely. Reads
the prepared parquet (title, abstract, target, split, optional class_label) — raw text
only, no embedding columns — and fine-tunes the chosen backbone (text_transformer_arch)
over ``text_transformer_n_runs`` seeds, writing ``seed_<i>/`` into the workflow folder.

    python -m workflow.src.main_train_tfm            # workflow type from config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow.src.iwf import paths, workflow_results
from workflow.src.iwf.config import WORKFLOWS, Settings, load_settings
from workflow.src.iwf.data.dataset import load_parquet, save_parquet
from workflow.src.iwf.targets import resolve_settings_class_thresholds, to_classes
from workflow.src.iwf.training.text_trainer import TextTransformerTrainer


def _ensure_text_columns(df, s: Settings, task: str):
    """Make sure title/abstract/target (+ class_label for clf) exist on the parquet."""
    for col in ("title", "abstract"):
        if col not in df.columns:
            raise KeyError(f"Text-transformer workflow needs a '{col}' column in the "
                           "prepared parquet.")
    if "target" not in df.columns:
        raise KeyError("Prepared parquet has no 'target' column - run the prepare phase first.")
    if task == "clf" and "class_label" not in df.columns:
        print("[train_tfm] no 'class_label' column - deriving it inline (label-phase logic).")
        df["class_label"] = to_classes(df["target"], scheme=s.class_scheme,
                                       thresholds=s.class_thresholds, names=s.class_names)
    return df


def main(settings: Settings | None = None, *, workflow_type: str | None = None,
         progress=None, wf_dir: Path | None = None) -> Path:
    s = settings or load_settings()
    wf = (workflow_type or s.workflow_type or "tfm_reg").lower()
    spec = WORKFLOWS.get(wf)
    if spec is None or spec["features"] != "text":
        raise ValueError(f"main_train_tfm handles the tfm_* workflows, not {wf!r}.")
    task = spec["task"]
    tfm_runs = max(1, int(s.text_transformer_n_runs))

    if wf_dir is None:
        wf_dir = workflow_results.next_workflow_dir(wf)
        wf_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_tfm] workflow={wf}  -> {wf_dir.name}  task={task}  "
          f"arch={s.text_transformer_arch}  seeds={tfm_runs}  "
          f"freeze_layers={s.text_transformer_freeze_layers}  "
          f"max_len={s.text_transformer_max_len}")

    df = load_parquet(paths.prepared_parquet(s.dataset_id))
    df = _ensure_text_columns(df, s, task)
    if task == "clf":
        save_parquet(df, paths.prepared_parquet(s.dataset_id))
    resolve_settings_class_thresholds(s, df)

    workflow_results.write_run_config(wf, s, wf_dir)
    for i in range(tfm_runs):
        seed_dir = wf_dir / f"seed_{i}"

        def _cb(info, _i=i):
            if progress is not None:
                progress({"seed": _i, "n_runs": tfm_runs, **info})

        trainer = TextTransformerTrainer(s, task, progress=_cb)
        metrics = trainer.run(df, seed=s.model_seed + i, output_dir=seed_dir,
                              class_names=s.class_names)
        primary = workflow_results.PRIMARY_METRIC[task]
        print(f"  [{wf_dir.name}] seed {i}: {primary}={metrics.get(primary):.4f}")

    print(f"[train_tfm] done -> {wf_dir}")
    return wf_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", default=None,
                    help="override the workflow type from config.yaml (tfm_clf | tfm_reg)")
    args = ap.parse_args()
    main(workflow_type=args.workflow)

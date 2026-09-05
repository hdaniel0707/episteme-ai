"""Phase 5 - label: derive the 3-class label for the classifier (plan section 4.4).

Computes ``class_label`` in {Weak, Average, Good} from the impact target using the
configured scheme (percentile thresholds, CSS, or quantile). NOTE: this does NOT change
the split (the split was stratified on a fixed percentile ``strat_label`` in Phase 1) - so
re-running with a different class scheme never reshuffles which paper is in train/val/test.

``css``/``quantile`` compute their cut-point VALUES from the TRAIN split's own target values
only (never val/test) and apply them to every row. This phase does NOT mutate ``settings``
- phase 6 (``targets.resolve_settings_class_thresholds``) does that once, identically, for
every workflow type (this phase only runs for the emb_*_clf / tfm_clf / llm_clf workflows,
not the regression ones, so it must never be the only place the resolved cut points exist).

    python -m workflow.src.main_label
"""

from __future__ import annotations

from workflow.src.iwf import paths
from workflow.src.iwf.config import Settings, load_settings
from workflow.src.iwf.data.dataset import load_parquet, save_parquet
from workflow.src.iwf.targets import resolve_class_boundaries, to_classes


def main(settings: Settings | None = None) -> None:
    s = settings or load_settings()
    # Reduced parquet (z) is the normal target; text-transformer workflows have no reduce step,
    # so fall back to the prepared parquet (raw title/abstract + target) when z does not exist.
    target_path = paths.reduced_parquet(s.dataset_id)
    if not target_path.is_file():
        target_path = paths.prepared_parquet(s.dataset_id)
    df = load_parquet(target_path)

    scheme = (s.class_scheme or "percentile").lower()
    train_target = df.loc[df["split"] == "train", "target"] if "split" in df.columns else df["target"]
    cuts = resolve_class_boundaries(train_target, scheme, s.class_thresholds, s.class_names)
    df["class_label"] = to_classes(df["target"], scheme="percentile", thresholds=cuts,
                                   names=s.class_names)
    print(f"[label] {target_path.name}  scheme={scheme}  cuts(from TRAIN)={cuts}  "
          f"counts={df['class_label'].value_counts().to_dict()}")
    save_parquet(df, target_path)


if __name__ == "__main__":
    main()

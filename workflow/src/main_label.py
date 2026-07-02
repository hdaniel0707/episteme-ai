"""Phase 5 - label: derive the 3-class label for the classifier (plan section 4.4).

Computes ``class_label`` in {Weak, Average, Good} from the impact target using the
configured scheme (percentile thresholds or CSS). NOTE: this does NOT change the split
(the split was stratified on a fixed percentile ``strat_label`` in Phase 1).

    python -m workflow.src.main_label
"""

from __future__ import annotations

from workflow.src.iwf import paths
from workflow.src.iwf.config import Settings, load_settings
from workflow.src.iwf.data.dataset import load_parquet, save_parquet
from workflow.src.iwf.targets import to_classes


def main(settings: Settings | None = None) -> None:
    s = settings or load_settings()
    # Reduced parquet (z) is the normal target; text-transformer workflows have no reduce step,
    # so fall back to the prepared parquet (raw title/abstract + target) when z does not exist.
    target_path = paths.reduced_parquet(s.dataset_id)
    if not target_path.is_file():
        target_path = paths.prepared_parquet(s.dataset_id)
    df = load_parquet(target_path)
    df["class_label"] = to_classes(df["target"], scheme=s.class_scheme,
                                   thresholds=s.class_thresholds, names=s.class_names)
    print(f"[label] {target_path.name}  scheme={s.class_scheme}  "
          f"counts={df['class_label'].value_counts().to_dict()}")
    save_parquet(df, target_path)


if __name__ == "__main__":
    main()

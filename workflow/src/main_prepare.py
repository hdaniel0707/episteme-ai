"""Phase 1 - prepare: target, metadata, and the ONE shared split (plan section 4.6).

Reads the enriched corpus, computes the field-normalized impact ``target`` (Option A/B),
builds numeric ``meta_*`` columns, derives a percentile-based ``strat_label`` used ONLY to
stratify the split, then computes the single shared 70/15/15 split (seeded). Every later
stage reuses this ``split`` column. Drops rows without a target.

    python -m workflow.src.main_prepare
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from workflow.src.iwf import paths
from workflow.src.iwf.config import Settings, load_settings
from workflow.src.iwf.data.dataset import add_split_column, load_parquet, save_parquet
from workflow.src.iwf.features.metadata import build_metadata
from workflow.src.iwf.targets import ALL_TARGET_KINDS, compute_target, to_classes

# ===== CONFIG =====
DROP_ROWS_WITHOUT_TARGET = True
DROP_ROWS_WITHOUT_TEXT = True   # drop papers missing title/abstract (embedders need both)
TEXT_FIELDS = ["title", "abstract"]
# ==================


def main(settings: Settings | None = None) -> None:
    s = settings or load_settings()
    df = load_parquet(paths.raw_parquet(s.dataset_id))

    # 0) drop incomplete papers up front, so corrupted rows never enter the shared
    #    train/val/test split or any downstream stage (the raw parquet is left untouched).
    if DROP_ROWS_WITHOUT_TEXT:
        before = len(df)
        for c in TEXT_FIELDS:
            if c in df.columns:
                df = df[df[c].notna() & (df[c].astype(str).str.strip() != "")]
        df = df.reset_index(drop=True)
        print(f"[prepare] dropped {before - len(df)} rows missing {TEXT_FIELDS}; {len(df)} remain")

    # 1) impact targets: compute & store EVERY kind as its own column (so switching
    #    TARGET_KIND later needs no re-fetch), then copy the selected one into `target`.
    for kind in ALL_TARGET_KINDS:
        col = f"target_{kind}"
        try:
            vals = compute_target(kind, df, min_refset=s.target_min_refset)
            df[col] = pd.to_numeric(vals, errors="coerce").astype("float64")
        except Exception as e:  # a sparse/unavailable target must not abort the others
            print(f"[prepare] target '{kind}' unavailable ({e}); stored as NaN")
            df[col] = np.nan
        print(f"[prepare]   {col:38s} non-null: {df[col].notna().sum():6d}/{len(df)}")

    active_col = f"target_{s.target_kind}"
    if active_col not in df.columns:
        raise KeyError(f"TARGET_KIND={s.target_kind!r} must be one of {ALL_TARGET_KINDS}.")
    if df[active_col].notna().sum() == 0:
        raise ValueError(
            f"Active target {active_col!r} is empty - check inputs or change TARGET_KIND."
        )
    df["target"] = df[active_col]
    print(f"[prepare] active target = {s.target_kind}  "
          f"(non-null: {df['target'].notna().sum()}/{len(df)})")

    if DROP_ROWS_WITHOUT_TARGET:
        df = df[df["target"].notna()].reset_index(drop=True)

    # 2) numeric metadata columns (NaN where missing -> neutral-0 + mask later)
    meta = build_metadata(df, s.metadata_fields)
    for c in meta.columns:
        df[c] = meta[c]

    # 3) stratification label (fixed percentile scheme) -> keeps the split stable
    df["strat_label"] = to_classes(df["target"], scheme="percentile",
                                    thresholds=s.class_thresholds)

    # 4) THE shared split (computed once; seeded; stratified)
    df = add_split_column(df, seed=s.split_seed, ratios=s.split_ratios,
                          stratify_col="strat_label", domain_holdout=s.domain_holdout)
    counts = df["split"].value_counts().to_dict()
    print(f"[prepare] split (seed={s.split_seed}): {counts}")

    save_parquet(df, paths.prepared_parquet(s.dataset_id))


if __name__ == "__main__":
    main()

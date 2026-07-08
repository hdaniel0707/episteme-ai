"""Plain metadata encoders (plan section 4.5a).

Produces numeric ``meta_*`` columns from the DB metadata. Missing values are left as NaN
on purpose - feature_builder turns them into a neutral 0 plus a ``*_missing`` mask bit
(plan section 6.1), so absent metadata never breaks training.

Venue (high-cardinality categorical) is frequency-encoded over the corpus, which is
label-free and therefore leakage-safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

META_COLUMNS = ["meta_year", "meta_h_index", "meta_author_count", "meta_venue_freq"]


def build_metadata(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Return a frame of numeric meta_* columns (NaN where missing)."""
    out = pd.DataFrame(index=df.index)

    if "year" in fields and "year" in df.columns:
        out["meta_year"] = pd.to_numeric(df["year"], errors="coerce")

    if "best_author_h_index" in fields and "best_author_h_index" in df.columns:
        out["meta_h_index"] = pd.to_numeric(df["best_author_h_index"], errors="coerce")

    if "author_count" in fields and "author_count" in df.columns:
        out["meta_author_count"] = pd.to_numeric(df["author_count"], errors="coerce")

    if "venue" in fields and "venue" in df.columns:
        venue = df["venue"].astype("object").where(df["venue"].notna(), None)
        freq = venue.value_counts(dropna=True)
        out["meta_venue_freq"] = venue.map(freq).astype("float64")  # NaN stays NaN

    return out

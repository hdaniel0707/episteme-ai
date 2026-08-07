from pathlib import Path
import pandas as pd
import random
from tqdm import tqdm


def coerce_dtypes(df: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Force *dtypes* onto whichever of those columns are present.

    Guards a specific parquet failure: a column that is entirely null gets
    written with Arrow type ``null`` rather than the type it would have had
    with data in it. Reading one such file alone is fine, but reading it
    *beside* a file where the same column has values raises
    ``ArrowNotImplementedError: Unsupported cast from double to null`` — which
    is exactly what happens with a per-domain shard directory, or when an
    optional column is populated by one data source and not the other.

    Fixing it at write time rather than at read time keeps the files
    self-describing: every shard of a run then carries the same schema whether
    or not that particular slice had any values.

    Missing columns are skipped, so this doubles as a no-op on frames that
    never carried the column.
    """
    present = {c: t for c, t in dtypes.items() if c in df.columns}
    if not present:
        return df
    out = df.copy()
    for col, dtype in present.items():
        # to_numeric first: an object column of None cannot go straight to a
        # numeric dtype, and a genuine non-numeric value should become null
        # rather than abort the write.
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(dtype)
    return out


def save_parquet(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    tqdm.write(f"Saved {len(df)} rows → {p}")


def load_parquet(p: Path) -> pd.DataFrame:
    if not p.is_file():
        raise FileNotFoundError(f"Parquet file not found: {p}")
    df = pd.read_parquet(p)
    tqdm.write(f"  Loaded {len(df)} rows ← {p}")
    return df


def pick_label(q: str, query_mode: str = "random"):
    if q is None or len(q) == 0:
        return "unknown"
    if query_mode == "random":
        return random.choice(q)
    if query_mode == "concat":
        return " + ".join(q)
    return q[0]

from pathlib import Path
import pandas as pd
import random


def save_parquet(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    print(f"Saved {len(df)} rows → {p}")


def load_parquet(p: Path) -> pd.DataFrame:
    if not p.is_file():
        raise FileNotFoundError(f"Parquet file not found: {p}")
    df = pd.read_parquet(p)
    print(f"Loaded {len(df)} rows ← {p}")
    return df


def pick_label(q: str, query_mode: str = "random"):
    if q is None or len(q) == 0:
        return "unknown"
    if query_mode == "random":
        return random.choice(q)
    if query_mode == "concat":
        return " + ".join(q)
    return q[0]

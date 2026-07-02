"""
Fetch NAIP impact scores for all papers in a Parquet dataset via the
HuggingFace Gradio Space API (ssocean/Newborn_Article_Impact_Predict).

Output: a new Parquet file with two extra columns:
    naip_raw_score   — raw model output
    naip_final_score — normalised TNCSI_sp score (0–1)

Usage:
    python -m epai.score.add_naip_scores
"""

import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from gradio_client import Client
from tqdm import tqdm

load_dotenv()

BASE_DIR = Path(__file__).parent.parent.parent

### INPUTS ############################################################
PARQUET_IN = BASE_DIR / "data" / "parquet" / "0523_A_10_AI.parquet"
PARQUET_OUT = BASE_DIR / "data" / "parquet" / "0523_A_10_AI_naip.parquet"

MODEL_VERSION = "v2.1"  # "v1" or "v2.1"
SLEEP_BETWEEN = 0.2  # seconds between requests — increase if rate limited
DEBUG = False  # if True, only process first 5 rows
#######################################################################


def fetch_score(
    client: Client, title: str, abstract: str
) -> tuple[float | None, float | None]:
    try:
        result = client.predict(
            title=title,
            abstract=abstract,
            model_version=MODEL_VERSION,
            api_name="/predict",
        )
        if result is None:
            tqdm.write("  [ERROR] API returned None")
            return None, None
        # result is a dataframe-like dict: {"data": [["Raw Score", val], ["Final Score", val]]}
        data = result.get("data", []) if isinstance(result, dict) else result
        raw = float(data[0][1]) if len(data) > 0 else None
        final = float(data[1][1]) if len(data) > 1 else None
        return raw, final
    except Exception as e:
        tqdm.write(f"  [ERROR] {e}")
        return None, None


def main() -> None:
    df = pd.read_parquet(PARQUET_IN)
    if DEBUG:
        df = df.head(5)
    print(f"Loaded {len(df)} rows from {PARQUET_IN}")

    # Load cached scores keyed by (paper_id, generated_by)
    cache: dict[tuple, tuple[float | None, float | None]] = {}
    if PARQUET_OUT.exists():
        cached = pd.read_parquet(PARQUET_OUT)
        for _, r in cached.iterrows():
            key = (r["paper_id"], r["generated_by"])
            if pd.notna(r.get("naip_final_score")):
                cache[key] = (r.get("naip_raw_score"), r["naip_final_score"])
        print(f"  Resuming: {len(cache)} rows already scored")

    hf_token = os.environ.get("HF_TOKEN")
    client = Client("ssocean/Newborn_Article_Impact_Predict", token=hf_token)

    raw_scores = []
    final_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="NAIP"):
        key = (row["paper_id"], row["generated_by"])
        if key in cache:
            raw_scores.append(cache[key][0])
            final_scores.append(cache[key][1])
            continue

        title = str(row.get("title", "") or "")
        abstract = str(row.get("abstract", "") or "")

        if not title and not abstract:
            raw_scores.append(None)
            final_scores.append(None)
            continue

        raw, final = fetch_score(client, title, abstract)
        raw_scores.append(raw)
        final_scores.append(final)

        time.sleep(SLEEP_BETWEEN)

        # Save after every successful score so progress survives quota errors
        if final is not None:
            df_so_far = df.iloc[: len(raw_scores)].copy()
            df_so_far["naip_raw_score"] = raw_scores
            df_so_far["naip_final_score"] = final_scores
            df_so_far.to_parquet(PARQUET_OUT, index=False)

    df["naip_raw_score"] = raw_scores
    df["naip_final_score"] = final_scores

    df.to_parquet(PARQUET_OUT, index=False)
    print(f"\nDone. Saved {len(df)} rows → {PARQUET_OUT}")

    n_ok = df["naip_final_score"].notna().sum()
    n_fail = df["naip_final_score"].isna().sum()
    print(f"  Scored: {n_ok}  |  Failed/skipped: {n_fail}")


if __name__ == "__main__":
    main()

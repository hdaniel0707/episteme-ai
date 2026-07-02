"""
Binoculars batch scorer — run the zero-shot AI-text detector from
``hw/detector_hw.py`` over every row of a Parquet dataset, once per model pair.

Based on:
  "Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated
  Text", Hans et al., ICML 2024  |  https://arxiv.org/abs/2401.12070

For each (observer, performer) pair in ``MODEL_PAIRS`` a score column is added,
named after the model pair, e.g. ``binoculars_falcon_rw_1b`` (or
``binoculars_<observer>_x_<performer>`` when the two differ):
    score = PPL(observer) / CrossPPL(observer, performer)
            (below ~THRESHOLD → AI-generated, above → human)

Usage:
    python -m epai.score.add_binoculars_scores
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # use GPU 0 only
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from epai.utils.cuda_util import check_cuda

from dotenv import load_dotenv
from epai.utils.env_utils import doublecheck_env, doublecheck_pkgs

load_dotenv()

# Check and print results
doublecheck_env(".env")  # check environmental variables
doublecheck_pkgs(pyproject_path="pyproject.toml", verbose=True)  # check packages

BASE_DIR = Path(__file__).parent.parent

### INPUTS ############################################################
PARQUET_IN = BASE_DIR / "data" / "parquet" / "0515_A_500_AI_260_naip.parquet"
PARQUET_OUT = BASE_DIR / "data" / "parquet" / "0515_A_500_AI_260_naip_binox.parquet"

# (observer, performer) pairs — one score column is produced per pair.
MODEL_PAIRS = [
    # Falcon
    ("tiiuae/Falcon3-1B-Base", "tiiuae/Falcon3-1B-Instruct"),              # falcon 1b
    #("tiiuae/Falcon-H1-7B-Base", "tiiuae/Falcon-H1-7B-Instruct"),         # falcon 7b
    ("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"),                     # falcon 7b (original)


    # Llama
    ("meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"),      # llama 1b
    #("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),      # llama 8b

    # Gemma
    #("google/gemma-3-1b-pt", "google/gemma-3-1b-it"),                      # gemma 1b
    #("google/gemma-3-4b-pt", "google/gemma-3-4b-it"),                      # gemma 4b

    # Qwen
    ("Qwen/Qwen3-1.7B-Base",            "Qwen/Qwen3-1.7B"),                       # qwen ~1b  
    #("Qwen/Qwen3-8B-Base",              "Qwen/Qwen3-8B"),                         # qwen 8b   
]

TEXT_COLUMN = "abstract"  # column whose text gets scored
MAX_TOKENS = 1024         # truncate long texts (falcon-rw-1b ctx = 2048)
DEBUG = False              # if True, only process first 5 rows
#######################################################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def model_slug(model_id: str) -> str:
    """HF id → identifier-safe slug: drop org prefix, non-alnum → '_', lowercase.

    'tiiuae/falcon-rw-1b'              -> 'falcon_rw_1b'
    'meta-llama/Llama-3.2-1B-Instruct' -> 'llama_3_2_1b_instruct'
    """
    name = model_id.rsplit("/", 1)[-1]
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def score_column_name(observer: str, performer: str) -> str:
    """Column name for the binoculars score of an (observer, performer) pair.

    Collapses to a single model name when both are identical, otherwise joins
    them as 'observer_x_performer'.
    """
    obs, perf = model_slug(observer), model_slug(performer)
    suffix = obs if obs == perf else f"{obs}_x_{perf}"
    return f"binoculars_{suffix}"


# ── Core math (from hw/detector_hw.py) ───────────────────────────────────────

def compute_ppl(model, inputs):
    """Standard perplexity: how surprised is `model` by the actual tokens?"""
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, T, vocab)

    shift_logits = logits[:, :-1]              # predict t from 0..t-1
    shift_labels = inputs["input_ids"][:, 1:]  # actual tokens t=1..T

    loss_per_token = F.cross_entropy(
        shift_logits.permute(0, 2, 1),  # (1, vocab, T-1)
        shift_labels,                   # (1, T-1)
        reduction="none",
    )                                   # (1, T-1)
    return loss_per_token.mean().exp().item()


def compute_cross_ppl(observer, performer, inputs):
    """
    Cross-perplexity: use performer's distribution as the 'target',
    measure observer's surprise at that distribution.
    """
    with torch.no_grad():
        performer_logits = performer(**inputs).logits  # (1, T, vocab)
        observer_logits = observer(**inputs).logits    # (1, T, vocab)

    # Shift to align with next-token prediction
    performer_probs = F.softmax(performer_logits[:, :-1], dim=-1)       # (1, T-1, vocab)
    observer_log_probs = F.log_softmax(observer_logits[:, :-1], dim=-1)

    cross_ent = -(performer_probs * observer_log_probs).sum(dim=-1).mean()
    return cross_ent.exp().item()


def binoculars_score(observer, performer, inputs):
    """Binoculars score = PPL(observer) / CrossPPL(observer, performer)."""
    ppl = compute_ppl(observer, inputs)
    cross_ppl = compute_cross_ppl(observer, performer, inputs)
    return ppl / cross_ppl


def score_text(tokenizer, observer, performer, text):
    """Tokenize a single text and return its Binoculars score (or None)."""
    text = str(text or "").strip()
    if not text:
        return None

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
    ).to(DEVICE)

    # Need at least 2 tokens for a next-token prediction to exist.
    if inputs["input_ids"].shape[1] < 2:
        return None

    return binoculars_score(observer, performer, inputs)


# ── Per-pair scoring ─────────────────────────────────────────────────────────

def run_pair(df: pd.DataFrame, observer_model: str, performer_model: str) -> None:
    """Score every row for one (observer, performer) pair, adding a column to df.

    Mutates ``df`` in place and persists to PARQUET_OUT after each new score so
    progress survives crashes. Rows already scored (column present and non-null)
    are reused, which also lets a previous run resume.
    """
    col = score_column_name(observer_model, performer_model)
    print(f"\n{'=' * 65}")
    print(f"Pair    : {observer_model}  |  {performer_model}")
    print(f"Column  : {col}")
    print(f"{'=' * 65}")

    if col not in df.columns:
        df[col] = pd.Series([pd.NA] * len(df), dtype="object", index=df.index)
    elif df[col].notna().all():
        print("  Already complete — skipping.")
        return

    tokenizer = AutoTokenizer.from_pretrained(observer_model)
    observer = (
        AutoModelForCausalLM.from_pretrained(observer_model, dtype=torch.float16)
        .to(DEVICE)
        .eval()
    )
    # Reuse the same weights for both roles when the pair is identical.
    if performer_model == observer_model:
        performer = observer
    else:
        performer = (
            AutoModelForCausalLM.from_pretrained(performer_model, dtype=torch.float16)
            .to(DEVICE)
            .eval()
        )

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=col):
        if pd.notna(df.at[idx, col]):  # already scored (resume)
            continue

        try:
            score = score_text(tokenizer, observer, performer, row.get(TEXT_COLUMN))
        except Exception as e:  # noqa: BLE001 — keep going on per-row failures
            tqdm.write(f"  [ERROR] {row.get('paper_id')}: {e}")
            score = None

        df.at[idx, col] = score
        if score is not None:
            df.to_parquet(PARQUET_OUT, index=False)

    df.to_parquet(PARQUET_OUT, index=False)

    n_ok = df[col].notna().sum()
    print(f"  Scored: {n_ok}  |  Failed/skipped: {len(df) - n_ok}")

    # Free GPU memory before loading the next pair.
    del performer, observer, tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def merge_existing_scores(df: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """Carry score columns from a previous output back onto df (keyed by paper)."""
    key = ["paper_id", "generated_by"]
    score_cols = [
        c for c in prev.columns if c.startswith("binoculars_") and c not in df.columns
    ]
    if not score_cols:
        return df
    df = df.merge(prev[key + score_cols], on=key, how="left")
    print(f"  Resuming: merged {len(score_cols)} existing score column(s): {score_cols}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if check_cuda() == False:
        return

    df = pd.read_parquet(PARQUET_IN)
    if DEBUG:
        df = df.head(5)
    print(f"Loaded {len(df)} rows from {PARQUET_IN}")

    if PARQUET_OUT.exists():
        df = merge_existing_scores(df, pd.read_parquet(PARQUET_OUT))

    print(f"Using device: {DEVICE}")
    print(f"Model pairs : {len(MODEL_PAIRS)}")

    for observer_model, performer_model in MODEL_PAIRS:
        run_pair(df, observer_model, performer_model)

    df.to_parquet(PARQUET_OUT, index=False)
    print(f"\nDone. Saved {len(df)} rows → {PARQUET_OUT}")
    for observer_model, performer_model in MODEL_PAIRS:
        col = score_column_name(observer_model, performer_model)
        n_ok = df[col].notna().sum()
        print(f"  {col}: scored {n_ok}, failed/skipped {len(df) - n_ok}")


if __name__ == "__main__":
    main()

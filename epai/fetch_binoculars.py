"""
Binoculars batch scorer — run the zero-shot AI-text detector from
``hw/detector_hw.py`` over every row of a Parquet dataset.

Based on:
  "Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated
  Text", Hans et al., ICML 2024  |  https://arxiv.org/abs/2401.12070

Output: a copy of the input Parquet with one extra column:
    binoculars_score — PPL(observer) / CrossPPL(observer, performer)
                       (below ~THRESHOLD → AI-generated, above → human)

Usage:
    python -m epai.fetch_binoculars
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # use GPU 0 only
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).parent.parent

### INPUTS ############################################################
PARQUET_IN = BASE_DIR / "data" / "parquet" / "0515_A_500_AI_260.parquet"
PARQUET_OUT = BASE_DIR / "data" / "parquet" / "0515_A_500_AI_260_binoculars.parquet"

OBSERVER_MODEL = "tiiuae/falcon-rw-1b"   # base model
PERFORMER_MODEL = "tiiuae/falcon-rw-1b"  # ideally the instruct variant

TEXT_COLUMN = "abstract"  # column whose text gets scored
MAX_TOKENS = 1024         # truncate long texts (falcon-rw-1b ctx = 2048)
DEBUG = True             # if True, only process first 5 rows
#######################################################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_parquet(PARQUET_IN)
    if DEBUG:
        df = df.head(5)
    print(f"Loaded {len(df)} rows from {PARQUET_IN}")

    # Resume: reuse scores already computed in a previous (possibly partial) run,
    # keyed by (paper_id, generated_by) — mirrors epai/fetch_naip.py.
    cache: dict[tuple, float] = {}
    if PARQUET_OUT.exists():
        cached = pd.read_parquet(PARQUET_OUT)
        if "binoculars_score" in cached.columns:
            for _, r in cached.iterrows():
                if pd.notna(r.get("binoculars_score")):
                    cache[(r["paper_id"], r["generated_by"])] = r["binoculars_score"]
        print(f"  Resuming: {len(cache)} rows already scored")

    print(f"Using device: {DEVICE}")
    print(f"Loading observer  : {OBSERVER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(OBSERVER_MODEL)
    observer = (
        AutoModelForCausalLM.from_pretrained(OBSERVER_MODEL, dtype=torch.float16)
        .to(DEVICE)
        .eval()
    )

    print(f"Loading performer : {PERFORMER_MODEL}")
    performer = (
        AutoModelForCausalLM.from_pretrained(PERFORMER_MODEL, dtype=torch.float16)
        .to(DEVICE)
        .eval()
    )

    scores: list[float | None] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Binoculars"):
        key = (row["paper_id"], row["generated_by"])
        if key in cache:
            scores.append(cache[key])
            continue

        try:
            score = score_text(tokenizer, observer, performer, row.get(TEXT_COLUMN))
        except Exception as e:  # noqa: BLE001 — keep going on per-row failures
            tqdm.write(f"  [ERROR] {row.get('paper_id')}: {e}")
            score = None
        scores.append(score)

        # Persist after every successful score so progress survives crashes.
        if score is not None:
            df_so_far = df.iloc[: len(scores)].copy()
            df_so_far["binoculars_score"] = scores
            df_so_far.to_parquet(PARQUET_OUT, index=False)

    df["binoculars_score"] = scores
    df.to_parquet(PARQUET_OUT, index=False)
    print(f"\nDone. Saved {len(df)} rows → {PARQUET_OUT}")

    n_ok = df["binoculars_score"].notna().sum()
    n_fail = df["binoculars_score"].isna().sum()
    print(f"  Scored: {n_ok}  |  Failed/skipped: {n_fail}")


if __name__ == "__main__":
    main()

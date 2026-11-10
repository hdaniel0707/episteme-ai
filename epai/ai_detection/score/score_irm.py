r"""Score corpora with IRM, the implicit-reward-model zero-shot detector.

IRM (Liu, Huang, Xiao, Wu; NeurIPS 2025, arXiv 2604.21223) reads a text's
implicit DPO reward under an instruct model (policy) and its base model
(reference), with the prompt set to empty (Eq. 5 of the paper):

    r(y) = sum_i [ log pi_instruct(y_i | y_<i) - log pi_base(y_i | y_<i) ]

Higher means machine. No beta, no length normalisation, no second term -- the
paper drops beta*log Z(x) as a constant once x is empty. The sum is length-
dependent; the paper switches to the per-token mean for one benchmark only, so
the token count is stored too and the mean is irm_score / irm_ntok.

Same pair, tokenizer, truncation and dtype as our Binoculars/Fast-DetectGPT
baseline (external/Binoculars, pair ``llama31-8b``): Llama-3.1-8B base +
Llama-3.1-8B-Instruct, the observer's tokenizer with its BOS, truncation at 512
tokens, bf16 weights, log-probabilities in fp32. So IRM sees exactly the tokens
Binoculars saw.

Runs in THIS repo's venv (torch + transformers are there), not the submodule's:

    .venv/bin/python epai/ai_detection/score/score_irm.py --dry-run
    .venv/bin/python epai/ai_detection/score/score_irm.py --limit 16 --out-dir $TMPDIR/irm_smoke
    .venv/bin/python epai/ai_detection/score/score_irm.py --gpu 0 --batch-size 8
    .venv/bin/python epai/ai_detection/score/score_irm.py --only manugram_aiscientist

Gated on the Hub: needs HF_TOKEN and an accepted licence on both Llama repos
(already the case if binoculars_score_llama31_8b was computed on this machine).

OUTPUT: one sidecar per corpus, ``<out-dir>/<stem>.parquet`` = ``uid`` +
``irm_score_llama31_8b`` + ``irm_ntok_llama31_8b``. The master parquets in
data/parquet/ are only read. Re-running resumes: uids already in the sidecar are
skipped, and the sidecar is rewritten every ``--save-every`` batches, so an
interruption loses at most that much work.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT = PARQUET_DIR / "tmp" / "irm_llama31_8b"

# The four core splits and the two rephrase corpora, i.e. every released stem.
STEMS = ("ghostbuster_orig", "manugram_gb", "manugram_sci", "manugram_aiscientist",
         "manugram_reph_gb", "manugram_reph_sci")

BASE_MODEL = "meta-llama/Llama-3.1-8B"
INSTRUCT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SCORE_COL = "irm_score_llama31_8b"
NTOK_COL = "irm_ntok_llama31_8b"
KEY_COL = "uid"


def load_models(gpu: str | None):
    """Base and instruct model, bf16, eval mode; the base model's tokenizer.

    Two visible GPUs put one model on each, like the Binoculars fork; one GPU
    holds both (~32 GB of weights).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    torch.set_grad_enabled(False)
    dev_base = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev_inst = "cuda:1" if torch.cuda.device_count() > 1 else dev_base
    token = os.environ.get("HF_TOKEN")

    def load(name: str, device: str):
        model = AutoModelForCausalLM.from_pretrained(
            name, device_map={"": device}, torch_dtype=torch.bfloat16, token=token)
        return model.eval()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok, load(BASE_MODEL, dev_base), load(INSTRUCT_MODEL, dev_inst)


def token_logprobs(model, enc):
    """log p(y_i | y_<i) for every position after the first, fp32, on CPU."""
    import torch

    enc = {k: v.to(model.device) for k, v in enc.items()}
    logits = model(**enc).logits[:, :-1, :].float()
    labels = enc["input_ids"][:, 1:]
    nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    return (-nll).cpu()


def score_batch(texts: list[str], tok, base, inst, max_tokens: int):
    """IRM sum and token count for each text in the batch."""
    enc = tok(texts, return_tensors="pt", padding="longest", truncation=True,
              max_length=max_tokens, return_token_type_ids=False)
    mask = enc["attention_mask"][:, 1:].float()
    diff = (token_logprobs(inst, enc) - token_logprobs(base, enc)) * mask
    return diff.sum(1).tolist(), mask.sum(1).long().tolist()


def load_done(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_parquet(path)
    return pd.DataFrame({KEY_COL: pd.Series(dtype=str),
                         SCORE_COL: pd.Series(dtype=float),
                         NTOK_COL: pd.Series(dtype="Int64")})


def score_stem(stem: str, args, models) -> None:
    from tqdm import tqdm

    out_path = args.out_dir / f"{stem}.parquet"
    df = pd.read_parquet(PARQUET_DIR / f"{stem}.parquet", columns=[KEY_COL, "text"])
    if args.limit is not None:
        df = df.head(args.limit)
    done = load_done(out_path)
    todo = df[~df[KEY_COL].isin(set(done[KEY_COL]))]
    # Empty or missing text has no tokens to score: it stays out of the sidecar
    # and so reads as Null after a merge, rather than as a score of 0.
    todo = todo[todo["text"].fillna("").str.strip() != ""]
    print(f"{stem}: {len(df)} rows, {len(done)} already scored, {len(todo)} to score"
          f"  ->  {out_path}")
    if todo.empty or models is None:
        return

    tok, base, inst = models
    # Longest first: similar lengths batch together (less padding), and an
    # out-of-memory shows up in the first batch rather than hours in.
    todo = todo.assign(_len=todo["text"].str.len()).sort_values("_len", ascending=False)
    uids, texts = todo[KEY_COL].tolist(), todo["text"].tolist()
    rows: list[dict] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        nonlocal done, rows
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done[NTOK_COL] = done[NTOK_COL].astype("Int64")
            done.to_parquet(out_path, index=False)
            rows = []

    starts = range(0, len(texts), args.batch_size)
    for n, i in enumerate(tqdm(starts, desc=stem, unit="batch"), 1):
        batch = texts[i:i + args.batch_size]
        sums, ntoks = score_batch(batch, tok, base, inst, args.max_tokens)
        rows += [{KEY_COL: u, SCORE_COL: s, NTOK_COL: t}
                 for u, s, t in zip(uids[i:i + args.batch_size], sums, ntoks)]
        if n % args.save_every == 0:
            save()
    save()
    print(f"{stem}: sidecar holds {len(done)} rows")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=list(STEMS), metavar="STEM",
                    help="corpus stems under data/parquet/ (default: all six released)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help="sidecar directory (default: %(default)s)")
    ap.add_argument("--gpu", default=None,
                    help="CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1' (two cards: one model each)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="truncation, as for Binoculars (default: %(default)s)")
    ap.add_argument("--save-every", type=int, default=50,
                    help="rewrite the sidecar every N batches (default: %(default)s)")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: first N rows per corpus only. Use a throwaway "
                         "--out-dir, or the real sidecar will look finished when it is not")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be scored, load no model")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    unknown = [s for s in args.only if not (PARQUET_DIR / f"{s}.parquet").is_file()]
    if unknown:
        raise SystemExit(f"not found under {PARQUET_DIR}: {unknown}")
    total = sum(pq.ParquetFile(PARQUET_DIR / f"{s}.parquet").metadata.num_rows
                for s in args.only)
    print(f"IRM  {INSTRUCT_MODEL} (policy) vs {BASE_MODEL} (reference), "
          f"max {args.max_tokens} tokens, {len(args.only)} corpora, {total} rows")
    models = None if args.dry_run else load_models(args.gpu)
    for stem in args.only:
        score_stem(stem, args, models)
    return 0


if __name__ == "__main__":
    sys.exit(main())

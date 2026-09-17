r"""Pick 2 judge-prompt candidates that, together with v3, sit furthest apart
in embedding space.

Part of the prompt-diversity LogJudge pilot: rather than hand-picking 2 of the
10 candidate evidence prompts in
``cfg/ai_detection/judge_prompts/candidates/*.yaml`` (each registered into
``add_llm_judge_scores.PROMPTS`` at import time), this embeds each candidate
plus the production ``v3`` prompt with ``roberta-base`` -- mean-pooled last
hidden state, the same "raw LM baseline" recipe ``embed_select.py`` uses for
documents, applied here to whole prompt texts instead -- and picks the pair
that maximises the trio's *worst-case* pairwise distance:

    max over {A, B} of  min( d(v3, A), d(v3, B), d(A, B) )

Maximising the minimum (rather than the sum) rules out a pair that is each far
from v3 but close to each other, which a sum could still favour.

Deterministic (no sampling), no API calls; CPU is instant for 11 short texts.

    .venv/bin/python epai/ai_detection/score/select_diverse_judge_prompts.py
    .venv/bin/python epai/ai_detection/score/select_diverse_judge_prompts.py --metric euclidean
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from epai.ai_detection.score.add_llm_judge_scores import PROMPTS

BASE_DIR = Path(__file__).resolve().parents[3]
CANDIDATES_DIR = BASE_DIR / "cfg" / "ai_detection" / "judge_prompts" / "candidates"
ANCHOR_PROMPT_ID = "v3"


def _load_candidates(dir_path: Path) -> dict[str, str]:
    """id -> text, read straight from the YAMLs in ``dir_path``.

    Deliberately bypasses ``add_llm_judge_scores.PROMPTS``: that registry is
    global and collision-checked, which is right for prompts meant to be
    scored by ``--prompt <id>`` elsewhere, but wrong for a candidate pool that
    only needs to be compared against the ``v3`` anchor here. Reading the
    YAMLs directly lets ``--candidates-dir`` point at any pool (e.g. a v2
    re-run in its own directory) without id collisions against another pool.
    """
    out: dict[str, str] = {}
    for path in sorted(dir_path.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text())
        out[spec["id"]] = spec["text"].rstrip()
    return out


def _embed(texts: list[str]) -> np.ndarray:
    """roberta-base, mean-pooled over the last hidden state -- see
    ``_build_hf``/``_mean_pool`` in ``epai/utils/embed_select.py``, the same
    recipe applied there to documents rather than prompt texts."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModel.from_pretrained("roberta-base")
    model.eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512)
    with torch.no_grad():
        out = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled.numpy()


def _distance(u: np.ndarray, v: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        return float(1.0 - (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    return float(np.linalg.norm(u - v))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    ap.add_argument("--candidates-dir", type=Path, default=CANDIDATES_DIR,
                     help="Directory of candidate YAMLs, e.g. a v2 re-run's "
                     "own directory. Default: the original pilot's candidates/.")
    args = ap.parse_args()

    candidates = _load_candidates(args.candidates_dir)
    if len(candidates) < 2:
        raise SystemExit(f"need at least 2 candidate files in {args.candidates_dir}, "
                          f"found {len(candidates)}")

    ids = [ANCHOR_PROMPT_ID, *candidates]
    texts = [PROMPTS[ANCHOR_PROMPT_ID], *candidates.values()]
    print(f"embedding {len(texts)} prompts with roberta-base ({args.metric} distance)...")
    vecs = _embed(texts)

    n = len(ids)
    dist = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        d = _distance(vecs[i], vecs[j], args.metric)
        dist[i, j] = dist[j, i] = d

    print("\npairwise distance matrix:")
    print("           " + "  ".join(f"{pid:>10}" for pid in ids))
    for i, row_id in enumerate(ids):
        print(f"{row_id:>10}  " + "  ".join(f"{dist[i, j]:10.4f}" for j in range(n)))

    anchor = 0  # v3's index
    others = list(range(1, n))
    best: tuple[float, int, int] | None = None
    for a, b in combinations(others, 2):
        worst = min(dist[anchor, a], dist[anchor, b], dist[a, b])
        if best is None or worst > best[0]:
            best = (worst, a, b)

    worst, a, b = best
    print(f"\nwinning pair: {ids[a]!r} + {ids[b]!r}  "
          f"(worst-case pairwise distance {worst:.4f})")
    print(f"  d(v3, {ids[a]}) = {dist[anchor, a]:.4f}")
    print(f"  d(v3, {ids[b]}) = {dist[anchor, b]:.4f}")
    print(f"  d({ids[a]}, {ids[b]}) = {dist[a, b]:.4f}")
    print(f"\nnext: capture logprobs for these 2 prompt ids -- see "
          f"capture_judge_logprobs.py --prompt {ids[a]} / --prompt {ids[b]}")


if __name__ == "__main__":
    main()

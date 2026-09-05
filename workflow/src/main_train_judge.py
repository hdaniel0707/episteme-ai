"""Phase 6 (train, LLM-as-judge workflows) — RAG store of precedents as the "model".

Used by llm_clf / llm_reg, which skip encoder/reduce (phases 3,4 — see
config.WORKFLOWS): phase 2 (embed) already computes the frozen specter2 embedding this
module reuses for retrieval, so there is no latent-dim encoder to train.

There is no training step and no LLM call is ever needed to build the RAG store: for
TRAIN, the true target value (FWCI-normalized percentile, when
target_kind=fwci_percentile) is already known, so phase A builds the COMPLETE RAG store
first, deterministically, from up to ``judge_max_train_papers`` TRAIN papers (0 = all of
TRAIN) — no LLM calls, just title/abstract/true-value chunks. Phase B then judges every
record — TRAIN, VAL, and TEST alike — against that one complete, final RAG store, one LLM
call per record. Building the RAG before judging (rather than growing it paper-by-paper
during TRAIN) means every TRAIN paper is judged with the SAME full precedent pool as
VAL/TEST — no early-TRAIN papers judged against a near-empty RAG.

Per record: (1) retrieve up to ``RETRIEVAL_TOP_K`` most similar chunks (cosine similarity
on the frozen title+abstract embedding), dropping any exact abstract-text match (a paper
never retrieves itself as its own precedent), within a token budget capped to at most 50%
of the LLM's context window (``LLM_JUDGE_MAX_CONTEXT_TOKENS``, .env.workflow) once system
prompt + query are accounted for; (2) the LLM predicts this paper's impact in [0,1] given
a fixed rubric (``SEED_NOTES``, never rewritten) + the retrieved precedents — the system
prompt explicitly tells it to use those precedents only as loose calibration, never to
copy or average their scores (always blind — the true value is never in the prompt).

Similarity ranks on the existing frozen title+abstract embedding (e.g. specter2) — the
abstract dominates that vector since it is far longer than the title; a genuinely
abstract-only embedding would need a separate embedding pass and was left out to keep
this workflow simple (one embedding column, one LLM call per record).

Every LLM call is best-effort (workflow.src.iwf.llm.client.complete returns None on any
error - missing key, network, bad JSON) and degrades to a deterministic fallback (the
mean impact of the retrieved neighbours, or 0.5 with no neighbours yet), so a run with no
LLM endpoint configured still completes and produces a (weaker, kNN-only) result.

    python -m workflow.src.main_train_judge          # workflow type from config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from workflow.src.iwf import config, paths, workflow_results
from workflow.src.iwf.config import WORKFLOWS, Settings, is_judge_workflow, load_settings
from workflow.src.iwf.data.dataset import load_parquet, split_masks
from workflow.src.iwf.evaluation.metrics import classification_metrics, regression_metrics
from workflow.src.iwf.llm import client as llm_client
from workflow.src.iwf.targets import resolve_settings_class_thresholds

# At most this many retrieved precedents per LLM call, regardless of the token budget.
RETRIEVAL_TOP_K = 5

JUDGE_SYSTEM = (
    "You are an expert research-impact judge. You estimate a scientific paper's true "
    "citation-impact percentile (0 = negligible impact, 1 = field-defining) purely from "
    "its own title and abstract. You will sometimes be shown a few similar past papers "
    "together with their ACTUAL impact scores — these are LOOSE REFERENCE POINTS for "
    "calibration ONLY, never the answer. Do not copy, average, or anchor tightly on the "
    "reference scores: two papers can look topically similar yet differ enormously in "
    "impact for reasons the abstract itself reveals (novelty, rigor, scope, ambition, "
    "timeliness). Judge THIS paper on its own content first; use the references only to "
    "sanity-check that your estimate sits in a plausible range for this kind of paper. "
    "Be concise: never include reasoning or commentary outside the exact JSON schema "
    "asked for — extra explanation wastes tokens and is not read."
)

SEED_NOTES = (
    "Judging principles:\n"
    "- A clear, specific, novel contribution tends to have higher impact than vague or "
    "purely incremental work.\n"
    "- Broad relevance across a field, or interdisciplinary appeal, often correlates with "
    "higher citation impact.\n"
    "- Strong methodological rigor and reproducibility support are usually favorable.\n"
    "- Timeliness (addressing a current, actively-studied open problem) tends to increase "
    "impact.\n"
    "These are fixed for the whole run — every paper is judged against the same rubric."
)


def _token_estimate(text: str) -> int:
    """Rough token count (~4 chars/token), no tokenizer dependency needed."""
    return max(1, len(text) // 4)


def _effective_token_budget(system_prompt: str, notes: str, title: str, abstract: str,
                            base_budget: int, max_context_tokens: int) -> int:
    """Retrieval token budget for one call: ``base_budget``, further capped so
    system prompt + notes + query + retrieved text together stay within 50% of the
    LLM's context window (0/unset = unknown -> ``base_budget`` alone governs)."""
    if max_context_tokens <= 0:
        return base_budget
    overhead = (_token_estimate(system_prompt) + _token_estimate(notes)
               + _token_estimate(title) + _token_estimate(abstract) + 120)
    allowed = int(max_context_tokens * 0.5) - overhead
    return max(0, min(base_budget, allowed))


def _retrieve(vec: np.ndarray, chunk_vecs: list[np.ndarray], chunk_texts: list[str],
             chunk_targets: list[float], chunk_abstracts: list[str], *,
             query_abstract: str, token_budget: int,
             max_chunks: int = RETRIEVAL_TOP_K) -> tuple[str, list[float]]:
    """Up to ``max_chunks`` most-similar chunks (cosine, frozen embedding), best first,
    within ``token_budget`` tokens — excluding any chunk whose abstract text exactly
    matches ``query_abstract`` (a paper must never retrieve itself as a precedent).

    Returns the joined chunk text (for the LLM prompt) and the raw target values of the
    chunks actually included (for the deterministic mean-fallback when the LLM is down)."""
    if not chunk_vecs:
        return "", []
    C = np.stack(chunk_vecs)
    vn = vec / (np.linalg.norm(vec) + 1e-9)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    sims = Cn @ vn
    order = np.argsort(-sims)
    query_norm = query_abstract.strip()
    parts: list[str] = []
    targets: list[float] = []
    used = 0
    for idx in order:
        if len(parts) >= max_chunks:
            break
        if chunk_abstracts[idx].strip() == query_norm:
            continue          # exact-match exclusion: never let a paper see itself
        t = chunk_texts[idx]
        tok = _token_estimate(t)
        if parts and used + tok > token_budget:
            break
        parts.append(t)
        targets.append(chunk_targets[idx])
        used += tok
        if used >= token_budget:
            break
    return "\n---\n".join(parts), targets


# Key names accepted for the impact score in the LLM's JSON reply. Real models don't
# always use the exact requested key ("impact") — observed in testing that some models
# substitute a near-synonym like "impact_score" — so several plausible keys are tried
# before giving up, rather than silently discarding a real answer.
_IMPACT_KEYS = ("impact", "impact_score", "score", "rating", "value")


def _extract_impact_value(resp: dict) -> float | None:
    for key in _IMPACT_KEYS:
        if key in resp:
            try:
                return float(resp[key])
            except (TypeError, ValueError):
                continue
    for v in resp.values():          # last resort: first numeric value anywhere in the reply
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _predict_impact(notes: str, retrieved_text: str, title: str, abstract: str) -> float | None:
    """One LLM call: judge this paper's impact in [0,1] from the fixed rubric + precedents
    shown only for loose calibration (see JUDGE_SYSTEM). Always blind — the true value is
    never present in this prompt."""
    prompt = (
        f"Judging rubric:\n{notes}\n\n"
        + (f"Similar past papers, for LOOSE CALIBRATION ONLY — do not copy or average "
           f"their scores, judge THIS paper on its own merits (abstract + their actual "
           f"impact score, 0=low, 1=high):\n{retrieved_text}\n\n" if retrieved_text else "")
        + f"Now judge THIS paper's own expected impact on a 0.0-1.0 scale "
          f"(0 = negligible, 1 = field-defining), from its own title and abstract:\n"
          f"Title: {title}\nAbstract: {abstract}\n\n"
          'Respond with ONLY this JSON object, no other keys, no explanation, no markdown '
          'fences: {"impact": <float between 0 and 1>}'
    )
    resp = llm_client.complete(prompt, system=JUDGE_SYSTEM, as_json=True)
    if isinstance(resp, dict):
        val = _extract_impact_value(resp)
        if val is not None:
            return min(1.0, max(0.0, val))
    return None


def _make_chunk(title: str, abstract: str, true_val: float, max_tokens: int) -> str:
    """Deterministic RAG chunk (no LLM call) — built once, up front, from the true target."""
    text = f"Title: {title}\nAbstract: {abstract}\nActual impact score (0-1): {true_val:.3f}"
    return text[: max_tokens * 4]


def _embed_col(df: pd.DataFrame, settings: Settings) -> str:
    """The frozen embedding column to retrieve on — prefers specter2 (what the plan's kNN
    pattern uses); falls back to embed_for_encoder or any emb_ column if specter2 is absent."""
    for cand in (f"emb_{settings.embed_for_encoder}", "emb_specter2"):
        if cand in df.columns:
            return cand
    others = [c for c in df.columns if c.startswith("emb_")]
    if others:
        return others[0]
    raise KeyError("No emb_* column found — run the embed phase (2) before the judge trainer.")


def _class_ids(vals, thresholds: list[float]) -> np.ndarray:
    """Bin continuous [0,1] impact values into ordinal class ids via the run's thresholds —
    exact for class_scheme='percentile' (the same rule that produced class_label), an
    approximation for class_scheme='css'."""
    thr = sorted(float(t) for t in thresholds)
    return np.digitize(np.asarray(vals, dtype=float), thr, right=False)


def _predict_split(df_split: pd.DataFrame, embed_col: str, notes: str,
                   chunk_vecs: list[np.ndarray], chunk_texts: list[str],
                   chunk_targets: list[float], chunk_abstracts: list[str],
                   base_budget: int, max_context_tokens: int, *,
                   progress=None, split_label: str = "", seed_idx: int = 0, n_runs: int = 1
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Judge every row of ``df_split`` against the complete, already-built RAG store —
    one LLM call per row, same top-k/exact-match/context-budget retrieval rules
    regardless of split, so TRAIN/VAL/TEST are judged on equal footing."""
    trues, preds = [], []
    n = len(df_split)
    for i, (_, row) in enumerate(df_split.iterrows()):
        title, abstract = str(row.get("title") or ""), str(row.get("abstract") or "")
        vec = np.asarray(row[embed_col], dtype=np.float32)
        budget = _effective_token_budget(JUDGE_SYSTEM, notes, title, abstract,
                                         base_budget, max_context_tokens)
        retrieved_text, retrieved_targets = _retrieve(
            vec, chunk_vecs, chunk_texts, chunk_targets, chunk_abstracts,
            query_abstract=abstract, token_budget=budget)
        predicted = _predict_impact(notes, retrieved_text, title, abstract)
        if predicted is None:
            predicted = float(np.mean(retrieved_targets)) if retrieved_targets else 0.5
        trues.append(float(row["target"]))
        preds.append(predicted)
        if progress is not None and (i % 5 == 0 or i == n - 1):
            progress({"seed": seed_idx, "n_runs": n_runs, "epoch": i + 1, "epochs": max(1, n),
                      "val_metric": 0.0, "split": split_label})
    return np.asarray(trues, dtype=float), np.asarray(preds, dtype=float)


def _build_rag(train_df: pd.DataFrame, rag_order: np.ndarray, embed_col: str,
              chunk_tokens: int) -> tuple[list, list, list, list]:
    """Phase A: the complete RAG store, built once, deterministically, from the true
    target values of the selected TRAIN rows — no LLM call needed, since TRAIN's true
    values are already known."""
    chunk_vecs: list[np.ndarray] = []
    chunk_texts: list[str] = []
    chunk_targets: list[float] = []
    chunk_abstracts: list[str] = []
    for pos in rag_order:
        row = train_df.iloc[pos]
        title, abstract = str(row.get("title") or ""), str(row.get("abstract") or "")
        true_val = float(row["target"])
        chunk_texts.append(_make_chunk(title, abstract, true_val, chunk_tokens))
        chunk_vecs.append(np.asarray(row[embed_col], dtype=np.float32))
        chunk_targets.append(true_val)
        chunk_abstracts.append(abstract)
    return chunk_vecs, chunk_texts, chunk_targets, chunk_abstracts


def _train_seed(df: pd.DataFrame, task: str, settings: Settings, seed_idx: int,
                seed_dir: Path, embed_col: str, progress=None) -> dict:
    """Phase A: build the complete RAG store from TRAIN (deterministic, no LLM calls).
    Phase B: judge TRAIN/VAL/TEST against that one store, one LLM call per record. Write
    the standard result.json + model/ artifact contract."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    model_dir = seed_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    base_budget = config.get_int("JUDGE_RAG_TOKEN_BUDGET", 3000)
    chunk_tokens = config.get_int("JUDGE_RAG_CHUNK_TOKENS", 200)
    max_context_tokens = config.get_int("LLM_JUDGE_MAX_CONTEXT_TOKENS", 0)

    train_df = df[df["split"].astype(str) == "train"].reset_index(drop=True)
    n_train = len(train_df)
    max_train = int(getattr(settings, "judge_max_train_papers", 0) or 0)
    n_use = min(n_train, max_train) if max_train > 0 else n_train

    rng = np.random.RandomState(settings.model_seed + seed_idx)
    rag_order = rng.permutation(n_train)[:n_use]   # which TRAIN rows populate the RAG

    notes = SEED_NOTES
    chunk_vecs, chunk_texts, chunk_targets, chunk_abstracts = _build_rag(
        train_df, rag_order, embed_col, chunk_tokens)

    n_runs = settings.n_runs
    train_split_df = train_df.iloc[rag_order].reset_index(drop=True)
    train_true, train_pred = _predict_split(
        train_split_df, embed_col, notes, chunk_vecs, chunk_texts, chunk_targets,
        chunk_abstracts, base_budget, max_context_tokens,
        progress=progress, split_label="train", seed_idx=seed_idx, n_runs=n_runs)

    masks = split_masks(df)
    val_true, val_pred = _predict_split(
        df[masks["val"]], embed_col, notes, chunk_vecs, chunk_texts, chunk_targets,
        chunk_abstracts, base_budget, max_context_tokens,
        progress=progress, split_label="val", seed_idx=seed_idx, n_runs=n_runs)
    test_true, test_pred = _predict_split(
        df[masks["test"]], embed_col, notes, chunk_vecs, chunk_texts, chunk_targets,
        chunk_abstracts, base_budget, max_context_tokens,
        progress=progress, split_label="test", seed_idx=seed_idx, n_runs=n_runs)

    if task == "clf":
        thr = settings.class_thresholds
        class_names = settings.class_names
        train_m = classification_metrics(_class_ids(train_true, thr),
                                         _class_ids(train_pred, thr), class_names)
        val_m = classification_metrics(_class_ids(val_true, thr),
                                       _class_ids(val_pred, thr), class_names)
        test_m = classification_metrics(_class_ids(test_true, thr),
                                        _class_ids(test_pred, thr), class_names)
        train_y_true, train_y_pred = _class_ids(train_true, thr), _class_ids(train_pred, thr)
        val_y_true, val_y_pred = _class_ids(val_true, thr), _class_ids(val_pred, thr)
        test_y_true, test_y_pred = _class_ids(test_true, thr), _class_ids(test_pred, thr)
    else:
        train_m = regression_metrics(train_true, train_pred)
        val_m = regression_metrics(val_true, val_pred)
        test_m = regression_metrics(test_true, test_pred)
        train_y_true, train_y_pred = np.asarray(train_true), np.asarray(train_pred)
        val_y_true, val_y_pred = np.asarray(val_true), np.asarray(val_pred)
        test_y_true, test_y_pred = np.asarray(test_true), np.asarray(test_pred)

    result: dict = {
        **test_m,
        "seed": int(seed_idx),
        "n_train": n_train,
        "n_train_used": n_use,
        "train": {**train_m, "y_true": train_y_true.tolist(), "y_pred": train_y_pred.tolist()},
        "val": {**val_m, "y_true": val_y_true.tolist(), "y_pred": val_y_pred.tolist()},
        "test": {**test_m, "y_true": test_y_true.tolist(), "y_pred": test_y_pred.tolist()},
    }
    result.update({f"train_{k}": v for k, v in train_m.items() if isinstance(v, (int, float))})
    if task == "clf":
        n_classes = len(settings.class_thresholds) + 1
        proba = np.zeros((len(test_y_pred), n_classes))
        proba[np.arange(len(test_y_pred)), np.clip(test_y_pred, 0, n_classes - 1)] = 1.0
        result["proba_test"] = proba.tolist()

    (seed_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (model_dir / "notes.md").write_text(notes, encoding="utf-8")
    chunks_df = pd.DataFrame({
        "text": chunk_texts, "target": chunk_targets, "abstract": chunk_abstracts,
        "embedding": [v.tolist() for v in chunk_vecs],
    })
    chunks_df.to_parquet(model_dir / "rag_chunks.parquet", index=False)
    joblib.dump({
        "chunk_vecs": (np.stack(chunk_vecs) if chunk_vecs
                      else np.zeros((0, 1), dtype=np.float32)),
        "chunk_targets": np.asarray(chunk_targets, dtype=float),
        "chunk_abstracts": chunk_abstracts,
        "notes": notes, "embed_col": embed_col,
    }, model_dir / "rag_index.joblib")

    test_flat = {k: round(v, 4) for k, v in result["test"].items()
                if isinstance(v, (int, float))}
    print(f"  [{seed_dir.parent.name}] seed {seed_idx}: test={test_flat} "
          f"(RAG built from {n_use}/{n_train} train papers, then {n_use} train + "
          f"{len(val_true)} val + {len(test_true)} test judged once each, top-{RETRIEVAL_TOP_K} RAG)")
    return test_flat


def main(settings: Settings | None = None, *, workflow_type: str | None = None,
        progress=None, wf_dir: Path | None = None) -> Path:
    s = settings or load_settings()
    wf = (workflow_type or s.workflow_type or "llm_reg").lower()
    spec = WORKFLOWS.get(wf)
    if spec is None or not is_judge_workflow(wf):
        raise ValueError(f"main_train_judge handles the llm_* workflows, not {wf!r}.")
    task = spec["task"]

    if wf_dir is None:
        wf_dir = workflow_results.next_workflow_dir(wf)
        wf_dir.mkdir(parents=True, exist_ok=True)

    df = load_parquet(paths.embedded_parquet(s.dataset_id))
    embed_col = _embed_col(df, s)
    if "target" not in df.columns:
        raise KeyError("Prepared parquet has no 'target' column - run the prepare phase first.")
    resolve_settings_class_thresholds(s, df)

    n_runs = max(1, int(s.n_runs))
    print(f"[train_judge] workflow={wf}  -> {wf_dir.name}  task={task}  "
          f"embed_col={embed_col}  seeds={n_runs}  (RAG built first from TRAIN, then 1 LLM "
          f"call/record for train+val+test, top-{RETRIEVAL_TOP_K} RAG, no notes-rewrite)")

    workflow_results.write_run_config(wf, s, wf_dir)
    for i in range(n_runs):
        def _cb(info, _i=i):
            if progress is not None:
                progress({**info})

        _train_seed(df, task, s, i, wf_dir / f"seed_{i}", embed_col,
                   progress=_cb if progress is not None else None)

    print(f"[train_judge] done -> {wf_dir}")
    return wf_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", default=None,
                    help="override the workflow type from config.yaml (llm_clf | llm_reg)")
    args = ap.parse_args()
    main(workflow_type=args.workflow)

"""Phase 6 (train, LLM-as-judge workflows) — RAG store + evolving notes as the "model".

Used by llm_clf / llm_reg, which skip encoder/reduce (phases 3,4 — see
config.WORKFLOWS): phase 2 (embed) already computes the frozen specter2 embedding this
module reuses for retrieval, so there is no latent-dim encoder to train.

Instead of a trained network, the "model" this workflow produces is:
  * ``notes.md``          a short, general set of impact-judging principles, periodically
                          rewritten by the LLM from a batch of TRAIN examples (never
                          domain- or paper-specific).
  * a RAG store of TRAIN abstracts + their true impact, retrieved by specter2 cosine
    similarity to give the judge concrete precedents alongside the general notes.

Per TRAIN paper: (1) retrieve the most similar already-seen chunks within a token budget,
(2) the LLM predicts this paper's impact in [0,1] given the notes + retrieved chunks (this
call is always blind — it never sees the true value), (3) a deterministic RAG chunk for
this paper is added to the store immediately, so retrieval coverage never lags behind
training progress. Separately, an error-gated subset of papers (all mispredicted ones, plus
a small random replay sample of well-predicted ones) is queued for a BATCHED rewrite: every
``judge_batch_max_papers`` queued papers (or fewer if the LLM's context window would be
exceeded — see LLM_JUDGE_MAX_CONTEXT_TOKENS in .env.workflow), one LLM call rewrites the
notes AND authors a better per-paper RAG chunk for the whole batch at once, replacing the
deterministic placeholders. This means "learning" happens in occasional batched calls
instead of one call per paper.

Training also checkpoints periodically against a growing sample of VAL (cheap at first,
escalating only when a plateau is suspected) and can stop early once VAL genuinely stops
improving — the best-VAL checkpoint's (notes, RAG store) is what gets used for the final
VAL/TEST report and saved as the model artifact, not necessarily the final live state.

VAL/TEST are always predict-only: the notes and RAG store used for them are frozen at
whichever checkpoint's state is best, so no label ever leaks forward.

Every LLM call is best-effort (workflow.src.iwf.llm.client.complete returns None on any
error — missing key, network, bad JSON) and degrades to a deterministic fallback (the mean
impact of the retrieved neighbours, or 0.5 with no neighbours yet), so a run with no LLM
endpoint configured still completes and produces a (weaker, kNN-only) result.

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

JUDGE_SYSTEM = (
    "You are an expert research-impact judge. You estimate how impactful a scientific "
    "paper's future citations will be, purely from its title and abstract, using the "
    "general principles you have learned and any similar past papers you are shown. "
    "Be concise: never include reasoning or commentary outside the exact JSON schema asked "
    "for — extra explanation wastes tokens and is not read."
)

SEED_NOTES = (
    "Seed principles (to be refined from data):\n"
    "- A clear, specific, novel contribution tends to have higher impact than vague or "
    "purely incremental work.\n"
    "- Broad relevance across a field, or interdisciplinary appeal, often correlates with "
    "higher citation impact.\n"
    "- Strong methodological rigor and reproducibility support are usually favorable.\n"
    "- Timeliness (addressing a current, actively-studied open problem) tends to increase "
    "impact.\n"
    "These are starting hypotheses; revise them as examples are seen."
)

# Minimum VAL papers evaluated at any checkpoint, even at a tiny eval percentage, so the
# early-stopping metric is never computed from a degenerate 0- or 1-paper sample.
_MIN_VAL_EVAL = 3
# Fixed per-call overhead (chars, /4 for tokens) added on top of the per-paper blocks when
# estimating whether a rewrite batch fits the context-window budget: system prompt +
# instructions + JSON-schema boilerplate. A generous flat estimate, not a precise count.
_BATCH_FIXED_OVERHEAD_CHARS = 900


def _token_estimate(text: str) -> int:
    """Rough token count (~4 chars/token), no tokenizer dependency needed."""
    return max(1, len(text) // 4)


def _retrieve(vec: np.ndarray, chunk_vecs: list[np.ndarray], chunk_texts: list[str],
             chunk_targets: list[float], token_budget: int) -> tuple[str, list[float]]:
    """Most-similar chunks (cosine, specter2) up to ``token_budget`` tokens, best first.

    Returns the joined chunk text (for the LLM prompt) and the raw target values of the
    chunks actually included (for the deterministic mean-fallback when the LLM is down)."""
    if not chunk_vecs:
        return "", []
    C = np.stack(chunk_vecs)
    vn = vec / (np.linalg.norm(vec) + 1e-9)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    sims = Cn @ vn
    order = np.argsort(-sims)
    parts: list[str] = []
    targets: list[float] = []
    used = 0
    for idx in order:
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
    """One LLM task: predict this paper's impact in [0,1] given notes + precedents. Always
    blind — the true value is never present in this prompt."""
    prompt = (
        f"General principles learned so far:\n{notes}\n\n"
        + (f"Similar past papers (abstract + their actual impact score, 0=low, 1=high):\n"
           f"{retrieved_text}\n\n" if retrieved_text else "")
        + f"Now rate this paper's expected impact on a 0.0-1.0 scale "
          f"(0 = negligible, 1 = field-defining):\n"
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
    """Deterministic RAG chunk (no LLM call) — the immediate placeholder added for every
    TRAIN paper right after it is predicted, so retrieval coverage never lags. May later be
    replaced by an LLM-authored chunk once this paper's rewrite batch fires (see
    ``_rewrite_batch``); the embedding (the paper's own specter2 vector) never changes."""
    text = f"Title: {title}\nAbstract: {abstract}\nActual impact score (0-1): {true_val:.3f}"
    return text[: max_tokens * 4]


def _extract_notes_and_chunks(resp, expected_n: int) -> tuple[str | None, list[str | None]]:
    """Parse the batched-rewrite JSON reply. Missing/malformed pieces degrade to None per
    field (caller keeps the old notes / deterministic chunk placeholder for those)."""
    if not isinstance(resp, dict):
        return None, [None] * expected_n
    notes = resp.get("notes")
    notes = notes.strip() if isinstance(notes, str) and notes.strip() else None
    raw_chunks = resp.get("chunks")
    raw_chunks = raw_chunks if isinstance(raw_chunks, list) else []
    chunks: list[str | None] = []
    for i in range(expected_n):
        c = raw_chunks[i] if i < len(raw_chunks) else None
        chunks.append(c.strip() if isinstance(c, str) and c.strip() else None)
    return notes, chunks


def _rewrite_batch(notes: str, batch: list[dict], note_tokens: int,
                   chunk_tokens: int) -> tuple[str | None, list[str | None]]:
    """One LLM call covering up to len(batch) TRAIN examples at once: rewrite the general
    notes given all of them together, and author one improved RAG chunk per example.

    ``batch`` entries: {"title", "abstract", "retrieved_text", "predicted", "true_val"} —
    ``retrieved_text`` is exactly what was shown to the model at that paper's own predict
    time (not recomputed), so the rewrite reasons about what the judge actually saw."""
    examples = []
    for i, e in enumerate(batch, 1):
        examples.append(
            f"Example {i}:\nTitle: {e['title']}\nAbstract: {e['abstract']}\n"
            f"Context shown at prediction time: {e['retrieved_text'] or '(none yet)'}\n"
            f"Predicted impact: {e['predicted']:.3f}\nActual impact: {e['true_val']:.3f}"
        )
    prompt = (
        f"Current general principles for judging paper impact:\n{notes}\n\n"
        f"You will see {len(batch)} example papers below with the judge's predicted vs "
        f"actual impact.\n\n" + "\n\n".join(examples) + "\n\n"
        f"Based on ALL these examples together, do two things:\n"
        f"1. Rewrite the general principles (NOT specific to any one paper), incorporating "
        f"any lessons. Keep it under ~{note_tokens} tokens.\n"
        f"2. For EACH example above, in the same order, write one concise precedent chunk "
        f"(abstract summary + its actual impact) for future retrieval. Keep each under "
        f"~{chunk_tokens} tokens.\n\n"
        'Respond with ONLY this JSON object, no other keys, no explanation, no markdown '
        'fences: {"notes": "<rewritten principles>", '
        '"chunks": ["<chunk 1>", "<chunk 2>", ...]}'
    )
    resp = llm_client.complete(prompt, system=JUDGE_SYSTEM, as_json=True)
    return _extract_notes_and_chunks(resp, len(batch))


def _would_exceed_batch_budget(pending: list[dict], candidate: dict, max_context_tokens: int,
                               note_tokens: int, chunk_tokens: int) -> bool:
    """True if adding ``candidate`` to ``pending`` would push the rewrite call's estimated
    input+output past half the LLM's context window. Returns False (never blocks) when
    LLM_JUDGE_MAX_CONTEXT_TOKENS is unset/0 — the hard judge_batch_max_papers cap alone
    governs batch size in that case."""
    if max_context_tokens <= 0:
        return False
    entries = [*pending, candidate]
    total_input = _token_estimate("x" * _BATCH_FIXED_OVERHEAD_CHARS)
    for e in entries:
        total_input += (_token_estimate(e["title"]) + _token_estimate(e["abstract"])
                        + _token_estimate(e["retrieved_text"]) + 20)
    total_output = note_tokens + len(entries) * chunk_tokens
    return (total_input + total_output) > (max_context_tokens * 0.5)


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


def _checkpoint_metric(task: str, settings: Settings, trues: np.ndarray,
                       preds: np.ndarray) -> float:
    """Single scalar used to compare checkpoints — the same primary metric family reported
    everywhere else (spearman for reg, macro_f1 for clf)."""
    if task == "clf":
        thr = settings.class_thresholds
        return classification_metrics(_class_ids(trues, thr), _class_ids(preds, thr),
                                      settings.class_names)["macro_f1"]
    return regression_metrics(trues, preds)["spearman"]


def _predict_split(df_split: pd.DataFrame, embed_col: str, notes: str,
                   chunk_vecs: list[np.ndarray], chunk_texts: list[str],
                   chunk_targets: list[float], token_budget: int) -> tuple[np.ndarray, np.ndarray]:
    """Predict-only pass against a frozen (notes, RAG store) state — used for VAL/TEST and
    for the final full-set report, always blind (no true value ever enters the prompt)."""
    trues, preds = [], []
    for _, row in df_split.iterrows():
        title, abstract = str(row.get("title") or ""), str(row.get("abstract") or "")
        vec = np.asarray(row[embed_col], dtype=np.float32)
        retrieved_text, retrieved_targets = _retrieve(vec, chunk_vecs, chunk_texts,
                                                       chunk_targets, token_budget)
        predicted = _predict_impact(notes, retrieved_text, title, abstract)
        if predicted is None:
            predicted = float(np.mean(retrieved_targets)) if retrieved_targets else 0.5
        trues.append(float(row["target"]))
        preds.append(predicted)
    return np.asarray(trues, dtype=float), np.asarray(preds, dtype=float)


def _predict_subset_cached(state: dict, val_df: pd.DataFrame, idx_list: list[int],
                           embed_col: str, token_budget: int) -> None:
    """Fill state['val_cache'][idx] -> predicted value for every idx in idx_list not
    already cached, using state's own (notes, RAG) snapshot. Lets re-scoring an earlier
    checkpoint at a larger VAL subset only pay for the newly-added indices."""
    cache = state["val_cache"]
    for idx in idx_list:
        if idx in cache:
            continue
        row = val_df.iloc[idx]
        title, abstract = str(row.get("title") or ""), str(row.get("abstract") or "")
        vec = np.asarray(row[embed_col], dtype=np.float32)
        retrieved_text, retrieved_targets = _retrieve(
            vec, state["chunk_vecs"], state["chunk_texts"], state["chunk_targets"], token_budget)
        predicted = _predict_impact(state["notes"], retrieved_text, title, abstract)
        if predicted is None:
            predicted = float(np.mean(retrieved_targets)) if retrieved_targets else 0.5
        cache[idx] = predicted


def _checkpoint_val_score(state: dict, val_df: pd.DataFrame, idx_list: list[int],
                          embed_col: str, token_budget: int, task: str,
                          settings: Settings) -> float:
    _predict_subset_cached(state, val_df, idx_list, embed_col, token_budget)
    preds = np.array([state["val_cache"][i] for i in idx_list])
    trues = np.array([float(val_df.iloc[i]["target"]) for i in idx_list])
    return _checkpoint_metric(task, settings, trues, preds)


def _train_seed(df: pd.DataFrame, task: str, settings: Settings, seed_idx: int,
                seed_dir: Path, embed_col: str, progress=None) -> dict:
    """Build the RAG store + notes from TRAIN (with batched, error-gated learning and
    VAL-based early stopping), score VAL/TEST with the best checkpoint, write the standard
    result.json + model/ artifact contract. Returns the flat test-metrics dict."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    model_dir = seed_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Machine/connection-level knobs (.env.workflow) — how big a context/budget to work with.
    token_budget = config.get_int("JUDGE_RAG_TOKEN_BUDGET", 3000)
    note_tokens = config.get_int("JUDGE_NOTE_TOKENS", 500)
    chunk_tokens = config.get_int("JUDGE_RAG_CHUNK_TOKENS", 200)
    max_context_tokens = config.get_int("LLM_JUDGE_MAX_CONTEXT_TOKENS", 0)

    # Per-run "science" knobs (config.yaml / Settings) — how this specific run learns.
    batch_max_papers = max(1, int(settings.judge_batch_max_papers))
    checkpoint_interval_pct = float(settings.judge_checkpoint_interval_pct)
    val_eval_start_pct = min(1.0, max(0.0, float(settings.judge_val_eval_start_pct)))
    val_eval_max_pct = min(1.0, max(val_eval_start_pct, float(settings.judge_val_eval_max_pct)))
    early_stop_margin = float(settings.judge_early_stop_margin)
    error_gate_threshold = float(settings.judge_error_gate_threshold)
    good_sample_fraction = min(1.0, max(0.0, float(settings.judge_good_sample_fraction)))

    train_df = df[df["split"].astype(str) == "train"].reset_index(drop=True)
    val_df = df[df["split"].astype(str) == "val"].reset_index(drop=True)
    n_train = len(train_df)
    n_val = len(val_df)

    rng = np.random.RandomState(settings.model_seed + seed_idx)
    train_order = rng.permutation(n_train)
    # Fixed VAL evaluation order for this seed: "first P% of this order" always nests
    # inside "first Q%>P%", so escalating the eval percentage only ever ADDS val papers
    # rather than redrawing an unrelated sample.
    val_order = rng.permutation(n_val).tolist()

    notes = SEED_NOTES
    chunk_vecs: list[np.ndarray] = []
    chunk_texts: list[str] = []
    chunk_targets: list[float] = []
    train_true: list[float] = []
    train_pred: list[float] = []
    pending_batch: list[dict] = []   # entries queued for the next batched rewrite call

    def _flush_batch() -> None:
        nonlocal notes, pending_batch
        if not pending_batch:
            return
        new_notes, new_chunks = _rewrite_batch(notes, pending_batch, note_tokens, chunk_tokens)
        if new_notes:
            notes = new_notes
        for entry, new_text in zip(pending_batch, new_chunks):
            if new_text:
                chunk_texts[entry["chunk_idx"]] = new_text
        pending_batch = []

    checkpoint_interval = max(1, round(checkpoint_interval_pct * n_train)) if n_train else 1
    best_state: dict | None = None
    best_metric = float("-inf")
    eval_pct = val_eval_start_pct
    consecutive_saturation = 0
    stopped_early = False
    train_papers_processed = 0

    for pos_i, pos in enumerate(train_order):
        row = train_df.iloc[pos]
        title, abstract = str(row.get("title") or ""), str(row.get("abstract") or "")
        true_val = float(row["target"])
        vec = np.asarray(row[embed_col], dtype=np.float32)

        retrieved_text, retrieved_targets = _retrieve(vec, chunk_vecs, chunk_texts,
                                                       chunk_targets, token_budget)
        predicted = _predict_impact(notes, retrieved_text, title, abstract)
        if predicted is None:
            predicted = float(np.mean(retrieved_targets)) if retrieved_targets else 0.5
        train_true.append(true_val)
        train_pred.append(predicted)

        # Immediate deterministic chunk — retrieval coverage never lags behind training,
        # regardless of whether/when this paper's batch later fires.
        chunk_texts.append(_make_chunk(title, abstract, true_val, chunk_tokens))
        chunk_vecs.append(vec)
        chunk_targets.append(true_val)
        chunk_idx = len(chunk_texts) - 1
        train_papers_processed += 1

        # Error-gated selection for the (batched) learning step: every mispredicted paper,
        # plus a small random replay sample of well-predicted ones (guards against the
        # notes drifting toward only the hard cases and regressing on the easy ones).
        error = abs(predicted - true_val)
        is_bad = error >= error_gate_threshold
        is_replay_sample = (not is_bad) and (rng.random() < good_sample_fraction)
        if is_bad or is_replay_sample:
            candidate = {"title": title, "abstract": abstract, "retrieved_text": retrieved_text,
                        "predicted": predicted, "true_val": true_val, "chunk_idx": chunk_idx}
            if len(pending_batch) >= batch_max_papers or _would_exceed_batch_budget(
                    pending_batch, candidate, max_context_tokens, note_tokens, chunk_tokens):
                _flush_batch()
            pending_batch.append(candidate)

        if progress is not None and (pos_i % 5 == 0 or pos_i == n_train - 1):
            progress({"seed": seed_idx, "n_runs": settings.n_runs,
                      "epoch": pos_i + 1, "epochs": max(1, n_train), "val_metric": 0.0})

        # VAL checkpoint boundary: flush any pending batch first so the state evaluated is
        # always coherent, then compare against the best-so-far checkpoint.
        at_boundary = (train_papers_processed % checkpoint_interval == 0
                      or pos_i == len(train_order) - 1)
        if at_boundary and n_val > 0:
            _flush_batch()
            n_eval = min(n_val, max(_MIN_VAL_EVAL, round(eval_pct * n_val)))
            idx_list = val_order[:n_eval]

            candidate_state = {"notes": notes, "chunk_vecs": list(chunk_vecs),
                              "chunk_texts": list(chunk_texts),
                              "chunk_targets": list(chunk_targets), "val_cache": {}}
            candidate_metric = _checkpoint_val_score(candidate_state, val_df, idx_list,
                                                     embed_col, token_budget, task, settings)

            if best_state is None:
                best_state, best_metric = candidate_state, candidate_metric
                consecutive_saturation = 0
            else:
                best_metric_here = _checkpoint_val_score(best_state, val_df, idx_list,
                                                          embed_col, token_budget, task, settings)
                if candidate_metric > best_metric_here + early_stop_margin:
                    best_state, best_metric = candidate_state, candidate_metric
                    consecutive_saturation = 0
                else:
                    consecutive_saturation += 1
                    if consecutive_saturation >= 2:
                        stopped_early = True
                        break
                    eval_pct = min(eval_pct * 2, val_eval_max_pct)

    _flush_batch()   # flush any leftover queued papers if the loop ended mid-batch

    if best_state is None:   # n_val == 0, or n_train == 0: no checkpoint ever ran
        best_state = {"notes": notes, "chunk_vecs": chunk_vecs, "chunk_texts": chunk_texts,
                     "chunk_targets": chunk_targets}

    final_notes = best_state["notes"]
    final_chunk_vecs = best_state["chunk_vecs"]
    final_chunk_texts = best_state["chunk_texts"]
    final_chunk_targets = best_state["chunk_targets"]

    # Final report always scores the FULL val/test sets (not the escalating subset used
    # internally for the stop/continue decision), so numbers stay comparable across runs.
    masks = split_masks(df)
    val_true, val_pred = _predict_split(df[masks["val"]], embed_col, final_notes,
                                        final_chunk_vecs, final_chunk_texts,
                                        final_chunk_targets, token_budget)
    test_true, test_pred = _predict_split(df[masks["test"]], embed_col, final_notes,
                                          final_chunk_vecs, final_chunk_texts,
                                          final_chunk_targets, token_budget)

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
        "stopped_early": stopped_early,
        "train_papers_processed": train_papers_processed,
        "train_papers_total": n_train,
        "best_checkpoint_val_metric": None if best_metric == float("-inf") else round(best_metric, 4),
        "train": {**train_m, "y_true": train_y_true.tolist(), "y_pred": train_y_pred.tolist()},
        "val": {**val_m, "y_true": val_y_true.tolist(), "y_pred": val_y_pred.tolist()},
        "test": {**test_m, "y_true": test_y_true.tolist(), "y_pred": test_y_pred.tolist()},
    }
    if task == "clf":
        n_classes = len(settings.class_thresholds) + 1
        proba = np.zeros((len(test_y_pred), n_classes))
        proba[np.arange(len(test_y_pred)), np.clip(test_y_pred, 0, n_classes - 1)] = 1.0
        result["proba_test"] = proba.tolist()

    (seed_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (model_dir / "notes.md").write_text(final_notes, encoding="utf-8")
    chunks_df = pd.DataFrame({
        "text": final_chunk_texts, "target": final_chunk_targets,
        "embedding": [v.tolist() for v in final_chunk_vecs],
    })
    chunks_df.to_parquet(model_dir / "rag_chunks.parquet", index=False)
    joblib.dump({
        "chunk_vecs": (np.stack(final_chunk_vecs) if final_chunk_vecs
                      else np.zeros((0, 1), dtype=np.float32)),
        "chunk_targets": np.asarray(final_chunk_targets, dtype=float),
        "notes": final_notes, "embed_col": embed_col,
    }, model_dir / "rag_index.joblib")

    test_flat = {k: round(v, 4) for k, v in result["test"].items()
                if isinstance(v, (int, float))}
    stop_note = (f"early-stopped at {train_papers_processed}/{n_train} train papers"
                if stopped_early else f"ran all {n_train} train papers")
    print(f"  [{seed_dir.parent.name}] seed {seed_idx}: test={test_flat} ({stop_note})")
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

    n_runs = max(1, int(s.n_runs))
    print(f"[train_judge] workflow={wf}  -> {wf_dir.name}  task={task}  "
          f"embed_col={embed_col}  seeds={n_runs}  "
          f"batch<={s.judge_batch_max_papers}  checkpoint~{s.judge_checkpoint_interval_pct:.0%}")

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

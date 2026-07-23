"""Cross-run summary: best model per embedder TYPE.

Groups every finished run by the *embedder* it used (specter2 base, specter2 + citation
adapter, specter2 + trained autoencoder, scibert, tfm, …) rather than by the specific model,
and keeps the best performer per (task, embedder-type) cell. Answers "which embedding gives
the best impact prediction" independent of which head/arch won. Written to
results/summary/embedder_performance.{csv,png} at the end of a run_all_workflows queue.

The performance value is the ALL-domain "% class correct" already computed per run in
``cross_domain.per_domain_metrics`` (reg and clf on one 0-100 scale), so this figure stays
consistent with the cross-domain matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from workflow.src.iwf.evaluation import plots
from workflow.src.iwf.evaluation.cross_domain import MODEL_BENCHMARKS_DIR

# Friendly label for the SPECTER2 flavours (the embed_for_encoder name alone is opaque).
_EMB_LABEL = {
    "specter2": "specter2+adapter",   # base + citation/proximity adapter
    "specter2_base": "specter2",      # base weights only
    "specter2_adhoc": "specter2+adhoc",
}


def _embedder_type(rc: dict) -> str:
    """Derive the embedder-type bucket key from a run's run_config.json."""
    wf = str(rc.get("workflow_type", ""))
    if wf.startswith("tfm"):
        return "tfm:" + str(rc.get("text_transformer_arch", "specter2_ft"))
    if wf.startswith("llm") or rc.get("features") == "rag":
        return "llm_judge"
    emb = rc.get("embed_for_encoder") or (rc.get("embedders") or ["?"])[0]
    label = _EMB_LABEL.get(emb, str(emb))
    enc = str(rc.get("encoder_kind", "identity"))
    if enc in ("autoencoder", "epai_autoencoder"):
        label += "+AE"
    elif enc == "pca":
        label += "+PCA"
    elif enc == "umap":
        label += "+UMAP"
    return label


def _all_domain_pct_correct(run_dir: Path) -> float | None:
    """The run's ALL-domain % class-correct from per_domain_metrics.csv (or None)."""
    for sub in (MODEL_BENCHMARKS_DIR, "cross_domain"):
        mp = run_dir / "aggregate" / sub / "per_domain_metrics.csv"
        if mp.is_file():
            try:
                mdf = pd.read_csv(mp)
            except Exception:
                continue
            allrow = mdf[mdf["domain"] == "ALL"]
            if not allrow.empty:
                return float(allrow.iloc[0]["value"])
    return None


def embedder_performance_summary(results_dir: Path) -> pd.DataFrame:
    """Best (task, embedder_type) → % correct table across every run in results_dir."""
    from workflow.src.iwf.workflow_results import task_of

    rows = []
    for run_dir in sorted(results_dir.glob("*_[0-9]*")):
        rc_path = run_dir / "run_config.json"
        if not rc_path.is_file():
            continue
        try:
            rc = json.loads(rc_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        val = _all_domain_pct_correct(run_dir)
        if val is None:
            continue
        rows.append({
            "run": run_dir.name,
            "task": task_of(rc.get("workflow_type", "")) or "?",
            "embedder_type": _embedder_type(rc),
            "value": val,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    best_idx = df.groupby(["task", "embedder_type"])["value"].idxmax()
    return df.loc[best_idx].reset_index(drop=True)


def generate_embedder_summary(results_dir: Path) -> None:
    """Write results/summary/embedder_performance.{csv,png}. Best-effort."""
    results_dir = Path(results_dir)
    df = embedder_performance_summary(results_dir)
    if df.empty:
        return
    out = results_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "embedder_performance.csv", index=False)
    try:
        plots.plot_embedder_performance(df, out / "embedder_performance.png")
    except Exception as e:
        print(f"[embedder_summary] plot error: {e}")

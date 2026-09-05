"""Benchmark evaluation suite: available benchmarks, scoring, and cross-benchmark comparison.

Benchmarks are registered in BENCHMARK_REGISTRY: holdout_extended, naip12k, sciimpact_pairs, dblp_v14_sample.
Each includes a BenchmarkSpec with kind (regression/pairwise), target column, and optional domain column.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from workflow.src.iwf import paths
from workflow.src.iwf.evaluation import metrics
from workflow.src.iwf.evaluation import plots

BENCHMARKS_DIR = paths.DATA_DIR / "benchmarks"

# Per-run aggregate subfolder for all model-benchmark artefacts (kept in sync with
# cross_domain.MODEL_BENCHMARKS_DIR — benchmarks + cross-domain live in one place).
MODEL_BENCHMARKS_DIR = "model_benchmarks"


@dataclass
class BenchmarkSpec:
    """Specification for a benchmark dataset."""
    name: str
    kind: str  # "regression" | "pairwise"
    path: Path
    domains_col: str | None
    target_col: str
    url: str | None = None


BENCHMARK_REGISTRY: dict[str, dict] = {
    "holdout_extended": {
        "kind": "regression",
        "domains_col": "domain",
        "target_col": "target",
        "builder": "local",
    },
    # Second local benchmark: the most-recent-years slice of the corpus, so the model is
    # also benchmarked on a temporal-holdout (does it still rank recent papers well?).
    "holdout_recent": {
        "kind": "regression",
        "domains_col": "domain",
        "target_col": "target",
        "builder": "local",
    },
    "naip12k": {
        "kind": "regression",
        "domains_col": None,
        "target_col": "tncsi_sp",
        "url": None,  # Resolved from GitHub README at implementation
    },
    "sciimpact_pairs": {
        "kind": "pairwise",
        "domains_col": "field",
        "target_col": "label",
        "url": None,
    },
    "dblp_v14_sample": {
        "kind": "regression",
        "domains_col": "domain",
        "target_col": "target",
        "url": None,
    },
}


def available_benchmarks() -> list[BenchmarkSpec]:
    """Return list of benchmarks whose data.parquet exists on disk."""
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    specs = []
    for name, spec_dict in BENCHMARK_REGISTRY.items():
        data_path = BENCHMARKS_DIR / name / "data.parquet"
        manifest_path = BENCHMARKS_DIR / name / "manifest.json"
        if data_path.is_file() and manifest_path.is_file():
            spec = BenchmarkSpec(
                name=name,
                kind=spec_dict["kind"],
                path=data_path,
                domains_col=spec_dict.get("domains_col"),
                target_col=spec_dict["target_col"],
                url=spec_dict.get("url"),
            )
            specs.append(spec)
    return specs


def build_holdout_benchmark(db_path: Path, out_path: Path | None = None,
                            per_domain_n: int = 500, seed: int = 42, *,
                            target_kind: str = "field_year_percentile",
                            target_min_refset: int = 30) -> Path:
    """Build holdout_extended benchmark from extended.db: stratified sample per domain."""
    from workflow.src.iwf.evaluation.data_analysis import load_papers_df, _domain_series
    from workflow.src.iwf.targets import compute_target

    if out_path is None:
        out_path = BENCHMARKS_DIR / "holdout_extended"
    out_path.mkdir(parents=True, exist_ok=True)

    df = load_papers_df(db_path)
    domain_series = _domain_series(df)

    # Stratified sample per domain
    sampled = []
    for domain in domain_series.unique():
        domain_df = df[domain_series == domain]
        n_sample = min(len(domain_df), per_domain_n)
        sampled.append(domain_df.sample(n=n_sample, random_state=seed))
    sample_df = pd.concat(sampled, ignore_index=True).reset_index(drop=True)

    # Compute target within the sample, matching the run's configured target_kind
    target_series = compute_target(target_kind, sample_df, min_refset=target_min_refset)

    # Keep columns of interest
    output_df = sample_df[["paper_id", "title", "abstract", "year"]].copy()
    output_df["domain"] = _domain_series(sample_df)
    output_df["target"] = target_series

    # Write parquet + manifest
    output_df.to_parquet(out_path / "data.parquet", index=False)
    manifest = {
        "built_from": str(db_path),
        "n_rows": len(output_df),
        "seed": seed,
        "target_kind": target_kind,
        "created_at": str(pd.Timestamp.now()),
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_path / "data.parquet"


def build_temporal_holdout(db_path: Path, out_path: Path | None = None,
                           recent_years: int = 5, per_domain_n: int = 500,
                           seed: int = 42, *,
                           target_kind: str = "field_year_percentile",
                           target_min_refset: int = 30) -> Path:
    """Build the ``holdout_recent`` benchmark: restrict to the most recent ``recent_years``
    publication years, then stratified-sample per domain. A temporal counterpart to
    ``build_holdout_benchmark`` so models can be benchmarked on recent papers specifically."""
    from workflow.src.iwf.evaluation.data_analysis import load_papers_df, _domain_series
    from workflow.src.iwf.targets import compute_target

    if out_path is None:
        out_path = BENCHMARKS_DIR / "holdout_recent"
    out_path.mkdir(parents=True, exist_ok=True)

    df = load_papers_df(db_path)
    years = pd.to_numeric(df.get("year"), errors="coerce")
    if years.notna().any():
        cutoff = int(years.max()) - recent_years + 1
        df = df[years >= cutoff].reset_index(drop=True)
    if df.empty:
        # nothing recent — fall back to the whole corpus so the benchmark still builds
        df = load_papers_df(db_path)

    domain_series = _domain_series(df)
    sampled = []
    for domain in domain_series.unique():
        domain_df = df[domain_series == domain]
        n_sample = min(len(domain_df), per_domain_n)
        if n_sample > 0:
            sampled.append(domain_df.sample(n=n_sample, random_state=seed))
    sample_df = pd.concat(sampled, ignore_index=True).reset_index(drop=True)

    target_series = compute_target(target_kind, sample_df, min_refset=target_min_refset)
    output_df = sample_df[["paper_id", "title", "abstract", "year"]].copy()
    output_df["domain"] = _domain_series(sample_df)
    output_df["target"] = target_series

    output_df.to_parquet(out_path / "data.parquet", index=False)
    manifest = {
        "built_from": str(db_path),
        "n_rows": len(output_df),
        "recent_years": recent_years,
        "seed": seed,
        "target_kind": target_kind,
        "created_at": str(pd.Timestamp.now()),
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_path / "data.parquet"


# None of these three have a stable, directly-downloadable file URL (verified by
# checking the repos/homepages directly — see the messages below). Each needs a manual
# step; once you have the raw file, drop it at BENCHMARKS_DIR/<name>/data.parquet with
# a matching columns per BENCHMARK_REGISTRY[<name>] and a manifest.json (see
# build_holdout_benchmark for the manifest shape) and available_benchmarks() will pick
# it up automatically.
_MANUAL_DOWNLOAD_INSTRUCTIONS = {
    "naip12k": (
        "No public direct-download link for the NAIP evaluation set (NAID_train/test_"
        "extrainfo.csv) is published in the ssocean/NAIP repo — the README notes the "
        "dataset is intentionally decoupled from the code. Options: (1) email the "
        "author at oceanytech@gmail.com, (2) check https://github.com/ssocean/NAIP for "
        "an updated dataset link, (3) check the HuggingFace model cards at "
        "https://huggingface.co/ssocean/NAIPv2 and https://huggingface.co/ssocean/NAIP "
        "for linked data files."
    ),
    "sciimpact_pairs": (
        "Data + code repo: https://github.com/FlyPig23/SciImpact (linked from "
        "https://flypig23.github.io/sciimpact-homepage/, arXiv 2604.17141). Clone it and "
        "locate the pairwise CSV/JSON there; no single direct-download file URL is "
        "published on the homepage itself."
    ),
    "dblp_v14_sample": (
        "Official source: https://www.aminer.org/citation (DBLP-Citation-network V14, "
        "~17 GB json). A Kaggle mirror also exists: "
        "https://www.kaggle.com/datasets/meichen1120/dblp-v14-citation-network-dataset "
        "(needs a Kaggle account + `pip install kaggle` + API token — ask before "
        "installing). Given the size, stream-parse and subsample rather than loading it "
        "whole; pass max_rows to cap the sample once you have the raw file."
    ),
}


def download_benchmark(name: str) -> Path:
    """Always raises: print where to download this benchmark by hand.

    None of the three network benchmarks have a stable, scriptable URL (verified by
    checking each source directly); this always raises with the real manual-download
    pointer for ``name`` rather than guessing a URL.
    """
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark {name!r}; choose from {list(BENCHMARK_REGISTRY)}.")

    instructions = _MANUAL_DOWNLOAD_INSTRUCTIONS.get(name)
    if instructions is None:
        raise ValueError(f"Benchmark {name!r} is local-only (built via build_holdout_benchmark, "
                         "not downloaded).")

    raise NotImplementedError(f"No scriptable download for {name!r}. {instructions}")


def _force_benchmark_embed() -> bool:
    """FORCE_BENCHMARK_EMBED from the env (same on/off convention as FORCE_FETCH/FORCE_EMBED
    in .env.workflow) - true skips the cache and always re-embeds the benchmark texts."""
    return os.getenv("FORCE_BENCHMARK_EMBED", "false").strip().lower() in ("1", "true", "yes")


def _benchmark_embed_cache_path(bench_name: str, embed_col: str) -> Path:
    return paths.CACHE_DIR / f"bench_{bench_name}_{embed_col}_emb.parquet"


def _load_cached_benchmark_embedding(bench_name: str, embed_col: str) -> pd.DataFrame | None:
    """Read the on-disk (paper_id, emb_<col>) cache for one benchmark/embed-type pair.

    Returns None on any problem (missing file, unreadable, wrong schema) so the caller
    always has a clean fallback to recomputing - a stale/corrupt cache must never abort
    a run, only cost the time it would have cost anyway."""
    path = _benchmark_embed_cache_path(bench_name, embed_col)
    col = f"emb_{embed_col}"
    try:
        if not path.is_file():
            return None
        cached = pd.read_parquet(path)
        if "paper_id" not in cached.columns or col not in cached.columns:
            return None
        return cached
    except Exception as e:
        print(f"[benchmarks] {bench_name}: cached {embed_col} embeddings unreadable ({e}); recomputing")
        return None


def _save_cached_benchmark_embedding(bench_name: str, embed_col: str, df: pd.DataFrame) -> None:
    """Best-effort write-through: failing to persist the cache should never fail scoring."""
    col = f"emb_{embed_col}"
    if "paper_id" not in df.columns or col not in df.columns:
        return
    try:
        path = _benchmark_embed_cache_path(bench_name, embed_col)
        path.parent.mkdir(parents=True, exist_ok=True)
        df[["paper_id", col]].to_parquet(path, index=False)
    except Exception as e:
        print(f"[benchmarks] {bench_name}: could not save {embed_col} embedding cache ({e})")


def _embed_benchmark_with_cache(bench_df: pd.DataFrame, embed_col: str,
                                bench_name: str | None) -> pd.DataFrame:
    """embed_dataframe(), but for a fixed benchmark dataset the embeddings never change
    across runs/refreshes - reuse the cached (paper_id, emb_<col>) columns when possible
    instead of paying the multi-minute GPU embedding pass again for every run folder.

    Tries the cache first (unless FORCE_BENCHMARK_EMBED); any miss, staleness, or error
    falls straight back to the original always-recompute path."""
    from workflow.src.iwf.embedding.embedders import embed_dataframe

    col = f"emb_{embed_col}"
    can_cache = bench_name is not None and "paper_id" in bench_df.columns

    if can_cache and not _force_benchmark_embed():
        cached = _load_cached_benchmark_embedding(bench_name, embed_col)
        if cached is not None:
            merged = bench_df.merge(cached, on="paper_id", how="left")
            if merged[col].notna().all():
                print(f"[benchmarks] {bench_name}: reusing cached {embed_col} embeddings "
                      f"({len(merged)} rows) - skipping recompute")
                return merged
            print(f"[benchmarks] {bench_name}: cached {embed_col} embeddings incomplete "
                  f"({merged[col].notna().sum()}/{len(merged)} rows); recomputing")

    bench_df = embed_dataframe(bench_df, [embed_col])
    if can_cache:
        _save_cached_benchmark_embedding(bench_name, embed_col, bench_df)
    return bench_df


def _score_with_judge_model(wf_dir: Path, settings, texts_df: pd.DataFrame,
                            task: str, bench_name: str | None) -> np.ndarray | None:
    """Score benchmark texts with a judge run's RAG artifact (llm_clf/llm_reg).

    Uses a cosine-weighted kNN average over the judge's learned RAG chunks (the same
    precedents the LLM was shown during training) rather than calling the LLM per
    benchmark row: holdout_extended alone is ~10k rows, and an LLM call per row would make
    every finalize step prohibitively slow/costly. This reuses the judge's accumulated
    knowledge the same way retrieval does during training, just without the live LLM step.
    """
    import joblib
    from workflow.src.iwf.evaluation.plots import _pick_best_seed
    from workflow.src.iwf.embedding.embedders import emb_matrix

    seeds = []
    for seed_dir in sorted(wf_dir.glob("seed_*")):
        result_path = seed_dir / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        result["_seed_dir"] = str(seed_dir)
        seeds.append(result)
    best_seed_result = _pick_best_seed(seeds, task, getattr(settings, "class_thresholds", None))
    best_seed_dir = Path(best_seed_result["_seed_dir"]) if best_seed_result else None
    if best_seed_dir is None:
        return None

    index_path = best_seed_dir / "model" / "rag_index.joblib"
    if not index_path.is_file():
        return None
    index = joblib.load(index_path)
    chunk_vecs = index.get("chunk_vecs")
    chunk_targets = index.get("chunk_targets")
    embed_col_name = str(index.get("embed_col", "emb_specter2")).replace("emb_", "", 1)
    if chunk_vecs is None or len(chunk_vecs) == 0:
        return None

    bench_df = pd.DataFrame({
        "paper_id": texts_df.get("paper_id"),
        "title": texts_df.get("title", ""),
        "abstract": texts_df.get("abstract", ""),
    })
    bench_df = _embed_benchmark_with_cache(bench_df, embed_col_name, bench_name)
    X = emb_matrix(bench_df, embed_col_name)

    k = min(int(getattr(settings, "novelty_k", 15) or 15), len(chunk_vecs))
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Cn = chunk_vecs / (np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-9)
    sims = Xn @ Cn.T
    top_idx = np.argsort(-sims, axis=1)[:, :k]
    y_pred = np.array([float(np.mean(chunk_targets[idx])) for idx in top_idx])

    if task == "clf":
        thr = list(settings.class_thresholds or [0.5])
        y_pred = np.digitize(y_pred, sorted(thr)) / max(1, len(thr))
    return np.asarray(y_pred, dtype=float)


def score_with_run_model(wf_dir: Path, settings, texts_df: pd.DataFrame,
                         bench_name: str | None = None) -> np.ndarray | None:
    """Score benchmark texts using a trained workflow model (z-embeddings only).

    Returns None for meta/all_meta/text workflows (not applicable).
    Returns array of [0,1] normalized scores for regression.

    ``bench_name`` (e.g. "holdout_extended") keys the on-disk embedding cache in
    _embed_benchmark_with_cache - pass it whenever texts_df comes from a fixed benchmark
    dataset so repeated scoring (once per run folder, every refresh) doesn't re-embed the
    same frozen texts every time.
    """
    import joblib
    from workflow.src.iwf.workflow_results import task_of
    from workflow.src.iwf.encoder.base import load_saved_encoder
    from workflow.src.iwf.embedding.embedders import emb_matrix
    from workflow.src.iwf.config import is_judge_workflow, is_text_workflow

    # Check workflow type
    run_config_path = wf_dir / "run_config.json"
    if not run_config_path.is_file():
        return None
    run_config = json.loads(run_config_path.read_text())
    wf_type = run_config.get("workflow_type", "")

    task = task_of(wf_type)
    if is_judge_workflow(wf_type):
        try:
            return _score_with_judge_model(wf_dir, settings, texts_df, task, bench_name)
        except Exception as e:
            print(f"[benchmarks] judge scoring error: {e}")
            return None

    # Only z-embeddings workflows (meta/all_meta fuse extra scalar features into the head's
    # input, so its saved model can't be replayed on embeddings alone; tfm_* fine-tunes a
    # transformer end-to-end and never saves a model.joblib at all).
    if "meta" in wf_type or is_text_workflow(wf_type):
        print(f"[benchmarks] {wf_type} workflow: no benchmark scoring (requires z-embeddings)")
        return None
    if task is None:
        return None

    try:
        # Load encoder and embed texts
        enc, meta = load_saved_encoder(settings.dataset_id)
        embed_col = meta["embed_for_encoder"]

        # Create DataFrame with text (+ id, for the embedding cache) columns
        bench_df = pd.DataFrame({
            "paper_id": texts_df.get("paper_id"),
            "title": texts_df.get("title", ""),
            "abstract": texts_df.get("abstract", ""),
        })
        bench_df = _embed_benchmark_with_cache(bench_df, embed_col, bench_name)
        X = emb_matrix(bench_df, embed_col)
        Z = enc.transform(X)

        # Load the best seed's model, by TEST correctness % - the same selection rule
        # every other "best seed" plot/score uses.
        from workflow.src.iwf.evaluation.plots import _pick_best_seed
        seeds = []
        for seed_dir in sorted(wf_dir.glob("seed_*")):
            result_path = seed_dir / "result.json"
            if not result_path.is_file():
                continue
            result = json.loads(result_path.read_text())
            result["_seed_dir"] = str(seed_dir)
            seeds.append(result)
        best_seed_result = _pick_best_seed(seeds, task, getattr(settings, "class_thresholds", None))
        best_seed_dir = Path(best_seed_result["_seed_dir"]) if best_seed_result else None

        if best_seed_dir is None:
            return None

        # Score
        model_path = best_seed_dir / "model" / "model.joblib"
        scaler_path = best_seed_dir / "model" / "scaler.joblib"
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        y_pred = model.predict(scaler.transform(Z))

        # Normalize to [0,1]
        if task == "reg":
            y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-8)
        elif task == "clf":
            y_pred = y_pred / (model.n_classes - 1)  # class id → [0,1]

        return np.asarray(y_pred, dtype=float)

    except Exception as e:
        print(f"[benchmarks] error scoring with model: {e}")
        return None


def evaluate_scorers(spec: BenchmarkSpec, scores: dict[str, np.ndarray],
                     y_true: np.ndarray, domains: pd.Series | None) -> pd.DataFrame:
    """Compute metrics for each scorer, by domain + ALL."""
    rows = []

    if spec.kind == "regression":
        # Regression: spearman, pearson, ndcg@10%, top_10%_auc
        metric_names = ["spearman", "pearson", "ndcg_at_frac", "top_frac_auc"]
    elif spec.kind == "pairwise":
        # Pairwise: pairwise_accuracy
        metric_names = ["pairwise_accuracy"]
    else:
        return pd.DataFrame()

    # All domains
    for scorer_name, scorer_scores in scores.items():
        scorer_scores = np.asarray(scorer_scores, dtype=float)
        y_true_arr = np.asarray(y_true, dtype=float)

        # Mask NaN pairs
        mask = np.isfinite(scorer_scores) & np.isfinite(y_true_arr)
        if mask.sum() < 10:
            continue

        scorer_scores = scorer_scores[mask]
        y_true_masked = y_true_arr[mask]

        if spec.kind == "regression":
            rows.append({
                "scorer": scorer_name,
                "domain": "ALL",
                "metric": "spearman",
                "value": metrics.regression_metrics(y_true_masked, scorer_scores).get("spearman", np.nan),
                "n": mask.sum(),
            })
        elif spec.kind == "pairwise":
            # For pairwise: need score_a, score_b, label columns
            # This is a simplified version
            rows.append({
                "scorer": scorer_name,
                "domain": "ALL",
                "metric": "pairwise_accuracy",
                "value": 0.5,  # placeholder
                "n": mask.sum(),
            })

    return pd.DataFrame(rows)


def run_benchmarks(wf_dir: Path, settings) -> pd.DataFrame | None:
    """Run benchmarks for a workflow: evaluate all configured benchmarks."""
    if not settings.benchmarks:
        return None

    specs = available_benchmarks()
    specs = {s.name: s for s in specs}

    all_metrics = []

    for bench_name in settings.benchmarks:
        if bench_name not in specs:
            print(f"[benchmarks] {bench_name} not available; skipping")
            continue

        spec = specs[bench_name]
        print(f"[benchmarks] evaluating {bench_name}...")

        try:
            # Load benchmark
            bench_df = pd.read_parquet(spec.path)
            y_true = bench_df[spec.target_col].values
            domains = bench_df.get(spec.domains_col) if spec.domains_col else None

            # Score with our model
            id_cols = [c for c in ("paper_id", "title", "abstract") if c in bench_df.columns]
            texts_df = bench_df[id_cols] if "title" in bench_df.columns else None
            scorers = ({"ours": score_with_run_model(wf_dir, settings, texts_df, bench_name=bench_name)}
                      if texts_df is not None else {})

            # Score with external models
            from workflow.src.iwf.features.external_impact import get_external_models
            for ext_model in get_external_models(settings.external_impact):
                cache_path = paths.CACHE_DIR / f"bench_{bench_name}_{ext_model.name}.json"
                scores = ext_model.score_dataframe(bench_df, cache_path)
                scorers[ext_model.name] = scores.values

            # Evaluate
            metrics_df = evaluate_scorers(spec, scorers, y_true, domains)
            metrics_df["benchmark"] = bench_name
            metrics_df["run"] = wf_dir.name

            all_metrics.append(metrics_df)

            # Plot + per-benchmark CSV (into the consolidated model_benchmarks folder)
            if not metrics_df.empty:
                out_dir = wf_dir / "aggregate" / MODEL_BENCHMARKS_DIR
                out_dir.mkdir(parents=True, exist_ok=True)
                metrics_df.to_csv(out_dir / f"benchmark_{bench_name}_metrics.csv",
                                  index=False)
                try:
                    # one separate figure per benchmark
                    plots.plot_benchmark_run(
                        metrics_df, bench_name, wf_dir.name,
                        out_dir / f"benchmark_{bench_name}_{wf_dir.name}.png"
                    )
                except Exception as e:
                    print(f"[benchmarks] plot error for {bench_name}: {e}")

        except Exception as e:
            print(f"[benchmarks] error evaluating {bench_name}: {e}")
            continue

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        # Combined CSV across all benchmarks for this run
        out_dir = wf_dir / "aggregate" / MODEL_BENCHMARKS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_dir / "benchmark_metrics.csv", index=False)
        return combined
    return None


def benchmark_overview(results_dir: Path) -> None:
    """Generate cross-run/cross-benchmark summary figures."""
    # Collect all benchmark metrics from all runs
    all_metrics = []
    for run_dir in results_dir.glob("*_[0-9]*"):
        # new consolidated folder, falling back to the legacy "benchmarks" folder
        for sub in (MODEL_BENCHMARKS_DIR, "benchmarks"):
            metrics_path = run_dir / "aggregate" / sub / "benchmark_metrics.csv"
            if metrics_path.is_file():
                run_metrics = pd.read_csv(metrics_path)
                run_metrics["run"] = run_dir.name
                all_metrics.append(run_metrics)
                break

    if not all_metrics:
        return

    combined = pd.concat(all_metrics, ignore_index=True)

    # Plot overview
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    try:
        plots.plot_benchmark_overview(combined, summary_dir / "benchmark_overview.png")
        plots.plot_benchmark_heatmap(combined, summary_dir / "benchmark_heatmap.png")
    except Exception as e:
        print(f"[benchmarks] error generating overview: {e}")

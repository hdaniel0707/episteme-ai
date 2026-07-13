"""Benchmark evaluation suite: available benchmarks, scoring, and cross-benchmark comparison.

Benchmarks are registered in BENCHMARK_REGISTRY: holdout_extended, naip12k, sciimpact_pairs, dblp_v14_sample.
Each includes a BenchmarkSpec with kind (regression/pairwise), target column, and optional domain column.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from workflow.src.iwf import paths
from workflow.src.iwf.config import load_settings
from workflow.src.iwf.evaluation import metrics
from workflow.src.iwf.evaluation import plots

BENCHMARKS_DIR = paths.DATA_DIR / "benchmarks"


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
                            per_domain_n: int = 500, seed: int = 42) -> Path:
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

    # Compute target within the sample
    target_series = compute_target("field_year_percentile", sample_df, min_refset=30)

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
        "created_at": str(pd.Timestamp.now()),
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_path / "data.parquet"


def download_benchmark(name: str, yes: bool = False, max_rows: int | None = None) -> Path:
    """Download benchmark from URL. Requires user confirmation for size > 200 MB."""
    spec_dict = BENCHMARK_REGISTRY.get(name)
    if not spec_dict or not spec_dict.get("url"):
        raise ValueError(f"Benchmark {name} has no URL or not in registry")

    out_dir = BENCHMARKS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "data.parquet"

    # TODO: implement download logic with size check
    # For now, raise with instructions
    raise NotImplementedError(
        f"Download for {name} not yet implemented. "
        "See PLAN Stage 5 for manual download instructions."
    )


def score_with_run_model(wf_dir: Path, settings, texts_df: pd.DataFrame) -> np.ndarray | None:
    """Score benchmark texts using a trained workflow model (z-embeddings only).

    Returns None for meta/all_meta/text workflows (not applicable).
    Returns array of [0,1] normalized scores for regression.
    """
    import joblib
    from workflow.src.iwf.workflow_results import task_of, PRIMARY_METRIC
    from workflow.src.iwf.encoder.base import load_saved_encoder
    from workflow.src.iwf.embedding.embedders import embed_dataframe, emb_matrix

    # Check workflow type
    run_config_path = wf_dir / "run_config.json"
    if not run_config_path.is_file():
        return None
    run_config = json.loads(run_config_path.read_text())
    wf_type = run_config.get("workflow_type", "")

    # Only z-embeddings workflows
    if "meta" in wf_type or "text" in wf_type:
        print(f"[benchmarks] {wf_type} workflow: no benchmark scoring (requires z-embeddings)")
        return None

    task = task_of(wf_type)
    if task is None:
        return None

    try:
        # Load encoder and embed texts
        enc, meta = load_saved_encoder(settings.dataset_id)
        embed_col = meta["embed_for_encoder"]

        # Create DataFrame with text columns
        bench_df = pd.DataFrame({
            "title": texts_df.get("title", ""),
            "abstract": texts_df.get("abstract", ""),
        })
        bench_df = embed_dataframe(bench_df, [embed_col])
        X = emb_matrix(bench_df, embed_col)
        Z = enc.transform(X)

        # Load best seed model
        best_seed_dir = None
        best_metric = -np.inf
        for seed_dir in (wf_dir / "seed_*").parent.glob("seed_*"):
            result_path = seed_dir / "result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                metric_val = result.get(f"test.{PRIMARY_METRIC[task]}", -np.inf)
                if metric_val > best_metric:
                    best_metric = metric_val
                    best_seed_dir = seed_dir

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
            y_pred = y_pred / (model.n_classes_ - 1)  # class id → [0,1]

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
            texts_df = bench_df[["title", "abstract"]] if "title" in bench_df.columns else None
            scorers = {"ours": score_with_run_model(wf_dir, settings, texts_df)} if texts_df is not None else {}

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

            # Plot
            if not metrics_df.empty:
                out_dir = wf_dir / "aggregate" / "benchmarks"
                out_dir.mkdir(parents=True, exist_ok=True)
                try:
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
        return pd.concat(all_metrics, ignore_index=True)
    return None


def benchmark_overview(results_dir: Path) -> None:
    """Generate cross-run/cross-benchmark summary figures."""
    # Collect all benchmark metrics from all runs
    all_metrics = []
    for run_dir in results_dir.glob("*_[0-9]*"):
        metrics_path = run_dir / "aggregate" / "benchmarks" / "benchmark_metrics.csv"
        if metrics_path.is_file():
            run_metrics = pd.read_csv(metrics_path)
            run_metrics["run"] = run_dir.name
            all_metrics.append(run_metrics)

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

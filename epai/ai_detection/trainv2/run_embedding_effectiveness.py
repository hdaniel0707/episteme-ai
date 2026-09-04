"""Experiment 2 — embedding effectiveness, entirely in-domain.

Which embedding is best for human-vs-machine detection? The algorithm is fixed
(logistic regression is enough to compare representations) and every embedding
in the config is swept instead. Each embedding is trained human-vs-generator
SEPARATELY for every generator in the multi parquet, per domain and pooled —
the same slices ``run_in_domain.py`` takes — and the per-generator results are
then combined into one ranking: the answer to "which embedding wins".

Combining rule: the ranking averages a domain's AUROC / TPR@1%FPR (oracle)
over every (generator, domain) cell, where domain ranges over the REAL domains
only (``computer_science``, ``mathematics``, ``sociology``) and never the
pooled slice — pooled is the same documents again, so folding it into the
average would double-count evidence already in the per-domain cells. Pooled is
still trained, reported per-generator, and plotted for reference.

    uv run python epai/ai_detection/trainv2/run_embedding_effectiveness.py
    uv run python epai/ai_detection/trainv2/run_embedding_effectiveness.py --dry-run
    uv run python epai/ai_detection/trainv2/run_embedding_effectiveness.py \
        --cfg cfg/ai_detection/trainv2/emb_effectiveness_science_v3f.yaml --no-plots

Output (``results/trainv2/<name>_<timestamp>/``):
    config.yaml          the config as run
    results.jsonl        one row per (embedding, generator, domain, seed)
    summary.jsonl        the same, aggregated over seeds (mean +- std)
    ranking.jsonl        one row per embedding, combined across generators and
                          the three real domains — sorted best (AUROC) first
    fig/ranking.png            the answer: embeddings ranked on both metrics
    fig/by_generator.png       the same ranking's AUROC, generator by generator
    fig/ranking.csv            ranking.jsonl as a CSV — what ranking.png plots
    fig/summary.csv            summary.jsonl as a CSV — what by_generator.png plots

To redraw the figures/CSVs of a run you already have, without retraining:
    uv run python epai/ai_detection/trainv2/replot.py results/trainv2/<run_dir>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import load_config
from .core import METRIC_KEYS, POOLED, Variant, aggregate_seeds, train_eval, write_csv
from .data import load_multi

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CFG = (
    BASE_DIR / "cfg" / "ai_detection" / "trainv2"
    / "emb_effectiveness_science_v3f.yaml"
)
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "trainv2"

RANKING_KEYS = ("auroc_mean", "tpr_at_fpr_mean")


def combine_ranking(summary_rows: list[dict], ranking_domains: list[str]) -> list[dict]:
    """One row per embedding: its per-cell metrics averaged over generators and
    the real domains, sorted best AUROC first.

    Each cell fed in is already a seed-aggregated mean, so the std reported
    here is the spread ACROSS (generator, domain) cells — does this embedding
    hold up everywhere, not seed noise — never confuse it with ``_std`` in
    ``summary.jsonl``, which is the seed spread of a single cell.
    """
    by_emb: dict[str, list[dict]] = {}
    for r in summary_rows:
        if r["domain"] in ranking_domains:
            by_emb.setdefault(r["embedding"], []).append(r)

    rows = []
    for emb, cells in by_emb.items():
        row = {"embedding": emb, "n_cells": len(cells)}
        for key in RANKING_KEYS:
            vals = np.asarray([c[key] for c in cells], dtype=float)
            row[key] = float(np.nanmean(vals))
            row[f"{key.removesuffix('_mean')}_spread"] = float(np.nanstd(vals))
        rows.append(row)
    return sorted(rows, key=lambda r: r["auroc_mean"], reverse=True)


def run(cfg_path: Path, results_dir: Path, dry_run: bool, plots: bool) -> Path | None:
    cfg = load_config(cfg_path, BASE_DIR)
    if len(cfg.algorithms) != 1:
        raise ValueError(
            f"{cfg_path}: this experiment fixes ONE algorithm to isolate the "
            f"embedding axis; got {len(cfg.algorithms)}: "
            f"{[a['name'] for a in cfg.algorithms]}. Use run_in_domain.py to "
            f"sweep several algorithms."
        )
    alg = cfg.algorithms[0]

    corpus = load_multi(
        cfg.parquet,
        embeddings=cfg.embeddings,
        group_col=cfg.group_col,
        human_label=cfg.human_label,
        domain_col=cfg.domain_col,
    )
    ranking_domains = [d for d in corpus.domains if d != POOLED]

    n_fits = len(cfg.embeddings) * len(corpus.generators) * len(corpus.domains) * len(cfg.seeds)
    print(f"config      : {cfg_path}")
    print(f"parquet     : {corpus.source}  ({len(corpus.df)} rows)")
    print(f"generators  : {corpus.generators}")
    print(f"domains     : {corpus.domains}  (pooled = {POOLED!r})")
    print(f"embeddings  : {cfg.embeddings}")
    print(f"algorithm   : {alg['name']}  {alg['params']}")
    print(f"seeds       : {cfg.seeds}")
    print(f"metrics     : AUROC, TPR@{cfg.fpr_pct}%FPR (oracle)")
    print(f"ranking uses: {ranking_domains}  (pooled excluded — same docs again)")
    print(f"→ {n_fits} fits")
    if dry_run:
        print("\nDry run — nothing trained.")
        return None

    exp_id = f"{cfg.name}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = results_dir / exp_id
    (out_dir / "fig").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(cfg_path.read_text())

    per_seed_rows: list[dict] = []
    summary_rows: list[dict] = []

    combos = [
        (emb, g, d)
        for emb in cfg.embeddings
        for g in corpus.generators
        for d in corpus.domains
    ]
    for embedding, generator, domain in tqdm(combos, desc="slices", unit="slice"):
        X, y, groups, info = corpus.slice(generator, domain, embedding)
        variant = Variant(
            embedding=embedding, alg_name=alg["name"],
            alg_label=alg.get("label", alg["name"]), alg_params=alg.get("params", {}),
        )
        seed_rows = train_eval(variant, X, y, groups, cfg.seeds, cfg.test_size, cfg.fpr)
        tag = {"embedding": embedding, "generator": generator, "domain": domain}
        for r in seed_rows:
            per_seed_rows.append({**tag, **r})
        summary_rows.append(
            {**tag, "n_missing_rows": info["n_missing_rows"],
             **aggregate_seeds(seed_rows, METRIC_KEYS)}
        )

    ranking = combine_ranking(summary_rows, ranking_domains)

    meta = {
        "experiment": cfg.name, "source": corpus.source, "fpr": cfg.fpr,
        "algorithm": alg["name"], "ranking_domains": ranking_domains,
        "generators": corpus.generators,
    }
    with open(out_dir / "results.jsonl", "w") as f:
        for r in per_seed_rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "summary.jsonl", "w") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for r in summary_rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "ranking.jsonl", "w") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for r in ranking:
            f.write(json.dumps(r) + "\n")
    # Beside the figures, not the JSON: the numbers ranking.png and
    # by_generator.png plot, each in its own table for a spreadsheet.
    write_csv(ranking, out_dir / "fig" / "ranking.csv")
    write_csv(summary_rows, out_dir / "fig" / "summary.csv")

    _print_ranking(ranking, cfg.fpr_pct, ranking_domains, corpus.generators)

    if plots:
        from .plots import embedding_ranking_plot, by_generator_plot

        embedding_ranking_plot(ranking, meta, out_dir / "fig" / "ranking.png")
        by_generator_plot(summary_rows, ranking, meta, out_dir / "fig" / "by_generator.png")

    try:
        shown = out_dir.relative_to(BASE_DIR)
    except ValueError:
        shown = out_dir
    print(f"\nDone → {shown}")
    return out_dir


def _print_ranking(
    ranking: list[dict], fpr_pct: str, domains: list[str], generators: list[str]
) -> None:
    width = max((len(r["embedding"]) for r in ranking), default=10) + 2
    rule = width + 40
    print(f"\n{'=' * rule}")
    print("  EMBEDDING RANKING — combined over "
          f"{len(generators)} generator(s) x {len(domains)} domain(s), logistic regression")
    print(f"{'=' * rule}")
    print(f"{'embedding':<{width}}{'AUROC':>18}{f'TPR@{fpr_pct}%FPR':>20}")
    print("-" * rule)
    for r in ranking:
        print(
            f"{r['embedding']:<{width}}"
            f"{r['auroc_mean'] * 100:>12.1f} ± {r['auroc_spread'] * 100:<3.1f}"
            f"{r['tpr_at_fpr_mean'] * 100:>13.1f} ± {r['tpr_at_fpr_spread'] * 100:<3.1f}"
        )
    print("-" * rule)
    print("  ± is the spread ACROSS generators/domains (does it hold up "
          "everywhere), not seed noise.")
    print(f"  best embedding: {ranking[0]['embedding']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cfg", type=Path, default=DEFAULT_CFG, help="config YAML")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--dry-run", action="store_true",
                   help="validate the config and print the plan, train nothing")
    p.add_argument("--no-plots", action="store_true", help="write JSON, skip figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.cfg, args.results_dir, args.dry_run, plots=not args.no_plots)


if __name__ == "__main__":
    main()

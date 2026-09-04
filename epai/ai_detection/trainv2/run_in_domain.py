"""Experiment 1 — data-generator effectiveness, entirely in-domain.

For every generator in the multi parquet, train ``human`` vs that generator over
the embedding x algorithm grid, once per domain and once pooled, with a grouped
train/test split repeated over seeds. Report AUROC and TPR@1%FPR (oracle) on the
held-out rows, and draw each training as a point in (AUROC, TPR@1%FPR) space.

    uv run python epai/ai_detection/trainv2/run_in_domain.py
    uv run python epai/ai_detection/trainv2/run_in_domain.py --dry-run
    uv run python epai/ai_detection/trainv2/run_in_domain.py \
        --cfg cfg/ai_detection/trainv2/gen_effectiveness_science_v3f.yaml --no-plots

Output (``results/trainv2/<name>_<timestamp>/``):
    config.yaml          the config as run
    results.jsonl        one row per (variant, generator, domain, seed)
    summary.jsonl        the same, aggregated over seeds (mean +- std)
    fig/scatter_<domain>.png   one small-multiples grid per domain slice
    fig/summary.csv            summary.jsonl as a CSV — the numbers every
                                scatter_<domain>.png plots, in one table

To redraw the figures/CSVs of a run you already have, without retraining:
    uv run python epai/ai_detection/trainv2/replot.py results/trainv2/<run_dir>
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from .config import load_config
from .core import (
    METRIC_KEYS,
    POOLED,
    aggregate_seeds,
    build_variants,
    train_eval,
    write_csv,
)
from .data import load_multi

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CFG = (
    BASE_DIR / "cfg" / "ai_detection" / "trainv2"
    / "gen_effectiveness_science_v3f.yaml"
)
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "trainv2"


def run(cfg_path: Path, results_dir: Path, dry_run: bool, plots: bool) -> Path | None:
    cfg = load_config(cfg_path, BASE_DIR)
    corpus = load_multi(
        cfg.parquet,
        embeddings=cfg.embeddings,
        group_col=cfg.group_col,
        human_label=cfg.human_label,
        domain_col=cfg.domain_col,
    )
    variants = build_variants(cfg.embeddings, cfg.algorithms)

    n_fits = len(variants) * len(corpus.generators) * len(corpus.domains) * len(cfg.seeds)
    print(f"config      : {cfg_path}")
    print(f"parquet     : {corpus.source}  ({len(corpus.df)} rows)")
    print(f"generators  : {corpus.generators}")
    print(f"domains     : {corpus.domains}  (pooled = {POOLED!r})")
    print(f"variants    : {[v.key for v in variants]}")
    print(f"seeds       : {cfg.seeds}")
    print(f"metrics     : AUROC, TPR@{cfg.fpr_pct}%FPR (oracle)")
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
        (v, g, d)
        for v in variants
        for g in corpus.generators
        for d in corpus.domains
    ]
    for variant, generator, domain in tqdm(combos, desc="slices", unit="slice"):
        X, y, groups, info = corpus.slice(generator, domain, variant.embedding)
        seed_rows = train_eval(
            variant, X, y, groups, cfg.seeds, cfg.test_size, cfg.fpr
        )
        tag = {
            "embedding": variant.embedding,
            "algorithm": variant.alg_name,
            "variant": variant.key,
            "generator": generator,
            "domain": domain,
        }
        for r in seed_rows:
            per_seed_rows.append({**tag, **r})
        summary_rows.append(
            {**tag, "n_missing_rows": info["n_missing_rows"],
             **aggregate_seeds(seed_rows, METRIC_KEYS)}
        )

    meta = {"experiment": cfg.name, "source": corpus.source, "fpr": cfg.fpr}
    with open(out_dir / "results.jsonl", "w") as f:
        for r in per_seed_rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "summary.jsonl", "w") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for r in summary_rows:
            f.write(json.dumps(r) + "\n")
    # Beside the figures, not the JSON: the numbers scatter_<domain>.png plots,
    # in one table for a spreadsheet rather than a parser.
    write_csv(summary_rows, out_dir / "fig" / "summary.csv")

    _print_table(summary_rows, cfg.fpr_pct)

    if plots:
        from .plots import scatter_grids

        scatter_grids(summary_rows, meta, out_dir / "fig")

    try:
        shown = out_dir.relative_to(BASE_DIR)
    except ValueError:
        shown = out_dir
    print(f"\nDone → {shown}")
    return out_dir


def _print_table(rows: list[dict], fpr_pct: str) -> None:
    """A flat read of every slice, ordered by AUROC within each domain."""
    width = max((len(r["variant"]) for r in rows), default=10) + 2
    gwidth = max((len(r["generator"]) for r in rows), default=8) + 2

    def auroc_key(r: dict) -> float:
        v = r["auroc_mean"]
        return -1.0 if math.isnan(v) else v

    for domain in sorted({r["domain"] for r in rows}):
        sub = sorted(
            (r for r in rows if r["domain"] == domain),
            key=auroc_key, reverse=True,
        )
        print(f"\n{'=' * (width + gwidth + 34)}")
        print(f"  {domain}")
        print(f"{'=' * (width + gwidth + 34)}")
        print(f"{'variant':<{width}}{'generator':<{gwidth}}{'AUROC':>16}"
              f"{f'TPR@{fpr_pct}%FPR':>18}")
        print("-" * (width + gwidth + 34))
        for r in sub:
            print(
                f"{r['variant']:<{width}}{r['generator']:<{gwidth}}"
                f"{r['auroc_mean'] * 100:>10.1f} ± {r['auroc_std'] * 100:<3.1f}"
                f"{r['tpr_at_fpr_mean'] * 100:>11.1f} ± {r['tpr_at_fpr_std'] * 100:<3.1f}"
            )


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

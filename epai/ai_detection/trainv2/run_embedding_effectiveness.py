r"""Which text embedding best separates human from AI text — on each SCRIBE
split, and on average across them.

The algorithm is fixed (L2 logistic regression on standardised features is
enough to compare representations) and every ``emb_*`` column is swept. Within
one split the task is one binary problem: human vs. every AI row pooled (all
generators, all domains). Trained on a grouped 70/30 split (a human text and
its rewrites never straddle the cut), repeated over N seeds, AUROC and
TPR@1%FPR reported as mean +- SD.

compute -> CSV -> plot from CSV, like ``calibrate_*_baseline.py``:

    results/trainv2/embedding_effectiveness/
        per_seed.csv     one row per (embedding, split, seed)
        summary.csv      one row per (embedding, split) AND per (embedding,
                          'mean') -- 'mean' is the average over the real
                          splits, its SD the spread ACROSS splits
        by_split.png     AUROC per embedding, a column per split + 'mean'
        ranking.png      the 'mean' ranking: AUROC + TPR@1%FPR, best first

Both PNGs are drawn from ``summary.csv`` (read back off disk), so a figure can
never disagree with the table.

Run:
    uv run python epai/ai_detection/trainv2/run_embedding_effectiveness.py
    uv run python epai/ai_detection/trainv2/run_embedding_effectiveness.py \
        --splits sci gb --seeds 0 1 2 3 4
    uv run python epai/ai_detection/trainv2/run_embedding_effectiveness.py \
        --replot results/trainv2/embedding_effectiveness/summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from epai.ai_detection.trainv2.core import (
    METRIC_KEYS,
    Variant,
    aggregate_seeds,
    train_eval,
)
from epai.ai_detection.trainv2.data import GEN_COL, _is_missing
from epai.ai_detection.trainv2.splits import resolve as resolve_splits

BASE_DIR = Path(__file__).resolve().parents[3]
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DEFAULT_OUT_DIR = BASE_DIR / "results" / "trainv2" / "embedding_effectiveness"

ALG = "logistic"
ALG_PARAMS = {"C": 1.0, "max_iter": 1000}

PER_SEED_FIELDS = [
    "embedding", "split", "seed", "n_train", "n_test", "n_test_human",
    "n_groups", "auroc", "tpr_at_fpr",
]
SUMMARY_FIELDS = [
    "embedding", "split", "n_seeds", "n_train", "n_test", "n_test_human",
    "auroc_mean", "auroc_std", "tpr_at_fpr_mean", "tpr_at_fpr_std",
]


def _discover_embeddings(paths: list[Path]) -> list[str]:
    """Every ``emb_*`` column present in ALL the given parquets, sorted."""
    import pyarrow.parquet as pq

    common: set[str] | None = None
    for p in paths:
        embs = {c for c in pq.read_schema(p).names if c.startswith("emb_")}
        common = embs if common is None else (common & embs)
    return sorted(common or [])


def _build_xyg(
    df: pd.DataFrame, embedding: str, group_col: str, human_label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """human vs. all-AI, for one embedding. A group is dropped whole if any of
    its rows has a missing embedding, so a human text and its rewrites are
    never separated."""
    sub = df[[group_col, GEN_COL, embedding]]
    bad = sub[embedding].map(_is_missing)
    if bad.any():
        sub = sub[~sub[group_col].isin(sub.loc[bad, group_col].unique())]
    sub = sub.reset_index(drop=True)
    X = np.asarray(np.stack(sub[embedding].to_numpy()), dtype=np.float32)
    y = (sub[GEN_COL].to_numpy() != human_label).astype(int)
    groups = sub[group_col].to_numpy()
    return X, y, groups


def compute(
    splits: list[tuple[str, Path]],
    embeddings: list[str],
    seeds: list[int],
    test_size: float,
    fpr: float,
    group_col: str,
    human_label: str,
) -> tuple[list[dict], list[dict]]:
    """-> (per_seed_rows, summary_rows). ``summary_rows`` includes the
    ``split == 'mean'`` rows (average over the real splits)."""
    per_seed: list[dict] = []
    summary: list[dict] = []

    combos = [(s, p, e) for s, p in splits for e in embeddings]
    cur_split: str | None = None
    df: pd.DataFrame | None = None
    for split, path, emb in tqdm(combos, desc="fits", unit="fit"):
        if split != cur_split:
            df = pd.read_parquet(path, columns=[group_col, GEN_COL, *embeddings])
            cur_split = split
        X, y, groups = _build_xyg(df, emb, group_col, human_label)
        variant = Variant(embedding=emb, alg_name=ALG, alg_label=ALG,
                          alg_params=ALG_PARAMS)
        seed_rows = train_eval(variant, X, y, groups, seeds, test_size, fpr)
        for r in seed_rows:
            per_seed.append({"embedding": emb, "split": split, **r})
        summary.append({
            "embedding": emb, "split": split,
            **aggregate_seeds(seed_rows, METRIC_KEYS),
        })

    for emb in embeddings:
        cells = [s for s in summary if s["embedding"] == emb]
        row = {"embedding": emb, "split": "mean", "n_seeds": len(seeds)}
        for key in METRIC_KEYS:
            vals = np.asarray([c[f"{key}_mean"] for c in cells], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(vals))
            row[f"{key}_std"] = float(np.nanstd(vals))
        summary.append(row)

    return per_seed, summary


def _print_ranking(summary: list[dict], fpr: float, n_splits: int) -> None:
    mean = {r["embedding"]: r for r in summary if r["split"] == "mean"}
    order = sorted(mean, key=lambda e: mean[e]["auroc_mean"], reverse=True)
    w = max((len(e) for e in order), default=10) + 2
    pct = f"{fpr * 100:g}"
    print(f"\n{'=' * (w + 40)}")
    print(f"  EMBEDDING RANKING — mean over {n_splits} SCRIBE splits, logistic regression")
    print(f"{'=' * (w + 40)}")
    print(f"{'embedding':<{w}}{'AUROC':>16}{f'TPR@{pct}%FPR':>18}")
    print("-" * (w + 40))
    for e in order:
        r = mean[e]
        print(f"{e:<{w}}"
              f"{r['auroc_mean'] * 100:>10.1f} ± {r['auroc_std'] * 100:<3.1f}"
              f"{r['tpr_at_fpr_mean'] * 100:>11.1f} ± {r['tpr_at_fpr_std'] * 100:<3.1f}")
    print("-" * (w + 40))
    print("  ± is the spread ACROSS SCRIBE splits (consistency), not seed noise.")
    print(f"  best embedding: {order[0]}")


def _to_frame(rows: list[dict], fields: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=fields)


def plots_from_csv(summary_csv: Path, meta: dict, fig_dir: Path) -> None:
    rows = pd.read_csv(summary_csv).to_dict("records")
    from epai.ai_detection.trainv2.plots import emb_by_split, emb_mean_ranking

    fig_dir.mkdir(parents=True, exist_ok=True)
    emb_mean_ranking(rows, meta, fig_dir / "ranking.png")
    emb_by_split(rows, meta, fig_dir / "by_split.png")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--splits", nargs="+", default=None, metavar="KEY",
                    help="SCRIBE splits to run (default: all)")
    ap.add_argument("--embeddings", nargs="+", default=None,
                    help="emb_* columns (default: every one common to the splits)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--human-label", default="human")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", type=Path, default=None, metavar="SUMMARY_CSV",
                    help="skip training; redraw the PNGs from an existing summary.csv")
    args = ap.parse_args()

    splits = resolve_splits(args.splits or [], PARQUET_DIR)
    meta = {
        "fpr": args.fpr, "n_seeds": len(args.seeds), "test_size": args.test_size,
        "splits": [s for s, _ in splits],
    }

    if args.replot is not None:
        plots_from_csv(args.replot, meta, args.replot.parent)
        print(f"replotted from {args.replot}")
        return

    missing = [str(p) for _, p in splits if not p.exists()]
    if missing:
        raise SystemExit(f"parquet(s) not found: {missing}")
    embeddings = args.embeddings or _discover_embeddings([p for _, p in splits])
    if not embeddings:
        raise SystemExit("no emb_* column common to the chosen splits")

    print(f"splits     : {[s for s, _ in splits]}")
    print(f"embeddings : {embeddings}")
    print(f"algorithm  : {ALG} {ALG_PARAMS}")
    print(f"seeds      : {args.seeds}   split: {int(round((1 - args.test_size) * 100))}/"
          f"{int(round(args.test_size * 100))} grouped on {args.group_col!r}")
    print(f"metrics    : AUROC, TPR@{args.fpr * 100:g}%FPR (oracle)")
    print(f"-> {len(splits) * len(embeddings) * len(args.seeds)} fits\n")

    per_seed, summary = compute(
        splits, embeddings, args.seeds, args.test_size, args.fpr,
        args.group_col, args.human_label,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_csv = args.out_dir / "per_seed.csv"
    summary_csv = args.out_dir / "summary.csv"
    _to_frame(per_seed, PER_SEED_FIELDS).to_csv(per_seed_csv, index=False)
    _to_frame(summary, SUMMARY_FIELDS).to_csv(summary_csv, index=False)
    print(f"wrote {per_seed_csv}")
    print(f"wrote {summary_csv}")

    _print_ranking(summary, args.fpr, len(splits))

    if not args.no_plots:
        plots_from_csv(summary_csv, meta, args.out_dir)


if __name__ == "__main__":
    main()

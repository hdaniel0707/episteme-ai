r"""Concatenate several embedding (or feature) columns and PCA the combined space.

The multi-feature sibling of ``analyse_domain_pca.py``: that script projects
ONE column (``--feature``); this one concatenates several (``--features``)
into a single wider matrix first -- the same "early fusion" idea evaluated
quantitatively by :mod:`epai.algs.emb_early_fusion`, but as a picture rather
than a probe. Same statistics, same PCA, same by-generator / binary colour
scheme, same figures: everything after "the matrix is built" is
:func:`analyse_domain_pca.analyse_corpus`, shared verbatim, so a result here
is read exactly the way a single-embedding one is.

WHY BLOCKS ARE STANDARDISED BEFORE CONCATENATION
-------------------------------------------------
A 768-dim dense embedding and an 11-dim 0/1 cue vector do not live at the same
natural scale. Concatenate them raw and PCA finds whichever block happens to
have the larger raw variance, which is an artifact of units, not a fact about
which embedding actually separates human from machine. :func:`stack_features`
standardises each block (zero mean, unit variance, per dimension) BEFORE
concatenation by default -- the same discipline
:class:`epai.algs.emb_early_fusion.EmbEarlyFusionAlg` applies before its own
PCA step. ``--no-standardize-blocks`` is the raw-concatenation control.
``--normalize`` is a SECOND, separate knob (identical to
``analyse_domain_pca.py``'s): it acts on the already-concatenated matrix,
after block standardisation, before PCA.

Usage:
    uv run python epai/ai_detection/analyse/analyse_multi_emb_pca.py \
        data/parquet/manugram_aiscientist.parquet \
        --features emb_roberta emb_luar_crud llm_judge_cuevec_gpt56luna_v3 \
        --jitter
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from epai.ai_detection.analyse.analyse_domain_pca import (
    BASE_DIR,
    DEFAULT_JITTER_SCALE,
    NORMALIZE_MODES,
    Report,
    analyse_corpus,
)
from epai.utils.corpus_utils import DOMAIN_COL
from epai.utils.parquet_utils import load_parquet
from epai.utils.train_utils import CLF_TARGET_COL, drop_missing_rows

RESULTS_DIR = BASE_DIR / "results" / "analyse_multi_emb_pca"


def _short(col: str) -> str:
    """A compact display name: ``emb_luar_crud`` -> ``luar_crud``, everything
    else (a judge cue-vector column, say) unchanged."""
    return col[4:] if col.startswith("emb_") else col


def stack_features(
    df: pd.DataFrame, features: list[str], standardize: bool, report: Report,
) -> tuple[np.ndarray, list[int]]:
    """Concatenate ``features``'s columns into one matrix, in the given order.

    Each block is standardised (per-dimension zero mean, unit variance) before
    concatenation when ``standardize`` is set (the default) -- see the module
    docstring for why. Returns ``(X, dims)``, ``dims`` the pre-standardisation
    block widths, for the report.
    """
    blocks: list[np.ndarray] = []
    dims: list[int] = []
    for f in features:
        block = np.asarray(np.vstack(df[f].to_numpy()), dtype=np.float64)
        dims.append(block.shape[1])
        if standardize:
            mean, sd = block.mean(axis=0), block.std(axis=0)
            sd[sd == 0] = 1.0
            block = (block - mean) / sd
        blocks.append(block)
    report("  Blocks  : " + ", ".join(f"{f} ({d}d)" for f, d in zip(features, dims)))
    report("  Block standardisation: "
           + ("ON (each block zero mean / unit variance before concatenation)"
              if standardize else "OFF (raw concatenation)"))
    return np.concatenate(blocks, axis=1), dims


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("parquet", type=Path,
                   help="Combined parquet, e.g. data/parquet/manugram_aiscientist.parquet")
    p.add_argument("--features", nargs="+", required=True, metavar="COL",
                   help="Columns to concatenate before PCA, in order, e.g. "
                        "emb_roberta emb_luar_crud llm_judge_cuevec_gpt56luna_v3.")
    p.add_argument("--no-standardize-blocks", action="store_true",
                   help="Skip the per-block standardisation normally applied "
                        "before concatenation (see the module docstring): "
                        "combine the raw columns at their native scale.")
    p.add_argument("--normalize", choices=NORMALIZE_MODES, default="none",
                   help="Vector preprocessing of the CONCATENATED matrix, "
                        "after block standardisation, before PCA -- the same "
                        "knob as analyse_domain_pca.py.")
    p.add_argument("--by-generator", action="store_true",
                   help="Force the per-generator view (blue = human, one colour "
                   "per AI generator, marker shape per domain). Automatic when "
                   "the file has more than one generator.")
    p.add_argument("--binary", action="store_true",
                   help="Force the cold-human / warm-machine view even on a file "
                   "with several generators: all AI collapses to one 'machine' "
                   "group per domain.")
    p.add_argument("--drop-classes", nargs="*", default=[], metavar="CLS",
                   help="generated_by labels to exclude before analysing.")
    p.add_argument("--sample", type=int, default=None, metavar="N",
                   help="Subsample to N rows before plotting (the 3D HTML gets "
                   "heavy past ~20k points).")
    p.add_argument("--jitter", action="store_true",
                   help="Spread rows that share a plotted location over a blob "
                   "whose area is proportional to how many they are -- see "
                   "analyse_domain_pca.py's docstring. Worth using whenever a "
                   "low-cardinality feature (a judge cue vector, say) is one of "
                   "--features. Figures only.")
    p.add_argument("--jitter-scale", type=float, default=DEFAULT_JITTER_SCALE,
                   metavar="F",
                   help="Radius of the busiest blob, in multiples of the typical "
                   "gap between distinct locations.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for --sample and for the --jitter layout.")
    p.add_argument("--no-plots", action="store_true",
                   help="Statistics only; skip the four figures.")
    p.add_argument("--no-html", action="store_true",
                   help="Write the PNGs but skip the interactive HTML.")
    p.add_argument("--plotly-cdn", action="store_true",
                   help="Load plotly.js from a CDN instead of embedding it: "
                   "~3 MB smaller per HTML, but needs a network connection to "
                   "open.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory. Default: "
                   "results/analyse_multi_emb_pca/<parquet stem>_<features>/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = Report()

    parquet_path = (args.parquet if args.parquet.is_absolute()
                    else BASE_DIR / args.parquet)
    feature_label = "+".join(_short(f) for f in args.features)
    out_dir = args.out_dir or RESULTS_DIR / f"{parquet_path.stem}_{feature_label}"
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report.rule(f"analyse_multi_emb_pca.py — {datetime.now():%Y-%m-%d %H:%M:%S}")
    report(f"  Parquet  : {parquet_path}")
    report(f"  Features : {args.features}")
    report(f"  Output   : {out_dir}")

    df = load_parquet(parquet_path)
    for col in (DOMAIN_COL, CLF_TARGET_COL):
        if col not in df.columns:
            raise SystemExit(
                f"{parquet_path.name}: no {col!r} column. This script expects the "
                "output of prepare/combine_domains.py."
            )
    missing_cols = [f for f in args.features if f not in df.columns]
    if missing_cols:
        available = sorted(c for c in df.columns
                           if c.startswith("emb_") or "cuevec" in c)
        raise SystemExit(f"{parquet_path.name}: no column(s) {missing_cols}. "
                         f"Available: {available}")
    if len(args.features) < 2:
        raise SystemExit("--features needs at least 2 columns to concatenate; "
                         "use analyse_domain_pca.py for a single embedding")

    df = drop_missing_rows(df, [*args.features, CLF_TARGET_COL])
    if args.drop_classes:
        df = df[~df[CLF_TARGET_COL].isin(args.drop_classes)].reset_index(drop=True)
        report(f"  Dropped classes: {args.drop_classes}")
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed).reset_index(drop=True)
        report(f"  Subsampled to {len(df)} rows (seed {args.seed})")
    df = df.reset_index(drop=True)

    X, dims = stack_features(df, args.features, not args.no_standardize_blocks, report)
    report(f"  Rows    : {len(df)}   combined dim: {X.shape[1]} "
           f"({' + '.join(str(d) for d in dims)})")

    analyse_corpus(df, X, feature_label, parquet_path, out_dir, report, args)


if __name__ == "__main__":
    main()

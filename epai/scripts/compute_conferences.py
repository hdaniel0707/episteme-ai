"""Compute the conference-abstract results as CSVs (no plotting).

Reads the parquet built by ``build_conferences_parquet.py`` and scored by
``run_conferences_judge_pipeline.py``, optionally filters rows to one or more
``conference_id`` values, and writes to ``results/conferences/<tag>/``:

  - ``filter_counts.csv``: per year, how many rows each admission rule dropped
    and how many were kept (see FILTER below).
  - ``calibration.csv``: per detector, the 1%-FPR threshold set on the MG-Sci
    human abstracts (the paper's oracle rule, ``cd_core.roc_point_at_fpr``),
    with the FPR it achieves on those human abstracts and the TPR on the MG-Sci
    machine abstracts, plus the next score value above the threshold and the
    FPR it would give (why a tied readout stops below 1%).
  - ``year_flag_rates.csv``: per (year, detector), the share of kept abstracts
    at or below that threshold (flagged as machine) with a Wilson 95% CI, plus
    the mean (+/- SEM) and median score. Every score is P(human) in [0, 1].
  - ``year_score_hist.csv``: per (year, detector), score counts and densities
    over 50 equal bins on [0, 1].

FILTER. OpenAlex lists placeholders under ICLR ("International audience",
"status: Published online", dataset/slides records). Rows are kept only if they
pass the same row-local admission rules as the MG-Sci human abstracts
(``science_data.rules.check_record`` on ``abst_standardized``: 80-400 words,
title of 3+ words, not truncated, not boilerplate, English), with the
metadata-only rules switched off (year, citation count, FWCI, OpenAlex language
label: not in this parquet), then deduplicated on the normalised abstract,
keeping the earliest year. Rows the judge did not score are dropped last.

    .venv/bin/python epai/scripts/compute_conferences.py --conference ICLR

    .venv/bin/python epai/scripts/compute_conferences.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epai.ai_detection.trainv2.calibrate_common import GEN_COL
from epai.ai_detection.trainv2.splits import resolve as resolve_splits
from epai.science_data.rules import FilterConfig, check_record, normalise_abstract
from epai.utils.cd_core import roc_point_at_fpr
from epai.utils.parquet_utils import load_parquet

DEFAULT_PARQUET = Path("data/parquet/conferences.parquet")
PARQUET_DIR = Path("data/parquet")
RESULTS_DIR = Path("results/conferences")
CALIBRATION_SPLIT = "mg_sci"
FPR = 0.01
HUMAN_LABEL = "human"
N_BINS = 50

# (column, canonical detector name); all three are P(human), so LOW = machine.
SCORE_COLUMNS = [
    ("llm_judge_score_gpt56luna_v3", "BaseJudge"),
    ("logjudge_ord2_gpt56luna_v3", "LogJudge"),
    ("fusionjudge_score", "FusionJudge"),
]

# MG-Sci's rules minus the ones that need metadata this parquet does not carry.
FILTER_CFG = FilterConfig(
    year_max=9999, require_citation_count=False, require_fwci=False, require_language=""
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet", type=Path, default=DEFAULT_PARQUET, help="input parquet (default: %(default)s)"
    )
    parser.add_argument(
        "--conference",
        nargs="+",
        default=None,
        metavar="CONFERENCE_ID",
        help="filter to one or more conference_id values (default: all)",
    )
    return parser.parse_args()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n``."""
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def apply_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Admission rules + dedup + scored; returns (kept rows, per-year counts)."""
    df = df.copy()
    df["drop_reason"] = [
        check_record(
            {"paper_id": r.id, "title": r.title, "abstract": r.abst_standardized, "year": r.conference_year},
            FILTER_CFG,
        )
        for r in df.itertuples()
    ]
    admitted = df["drop_reason"].isna()
    df.loc[admitted, "_key"] = df.loc[admitted, "abst_standardized"].map(normalise_abstract)
    order = df.loc[admitted].sort_values("conference_year", kind="stable").index
    dup = df.loc[order, "_key"].duplicated()
    df.loc[dup[dup].index, "drop_reason"] = "duplicate_abstract"
    unscored = df["drop_reason"].isna() & df[[c for c, _ in SCORE_COLUMNS]].isna().any(axis=1)
    df.loc[unscored, "drop_reason"] = "unscored"

    counts = (
        df.assign(drop_reason=df["drop_reason"].fillna("kept"))
        .groupby(["conference_year", "drop_reason"]).size()
        .unstack(fill_value=0)
    )
    counts.insert(0, "total", counts.sum(axis=1))
    return df[df["drop_reason"].isna()].drop(columns=["drop_reason", "_key"]), counts


def calibrate() -> pd.DataFrame:
    """1%-FPR threshold per detector on the MG-Sci human abstracts."""
    cols = [c for c, _ in SCORE_COLUMNS]
    ref = pd.concat(
        [pd.read_parquet(p, columns=[GEN_COL, *cols]) for _, p in resolve_splits([CALIBRATION_SPLIT], PARQUET_DIR)],
        ignore_index=True,
    )
    rows = []
    for col, name in SCORE_COLUMNS:
        sub = ref[[col, GEN_COL]].dropna()
        s = sub[col].to_numpy(dtype=float)
        is_machine = (sub[GEN_COL].astype(str) != HUMAN_LABEL).to_numpy()
        tpr, thr = roc_point_at_fpr(is_machine.astype(int), -s, FPR)
        thr = -thr  # machine iff P(human) <= thr
        # the next score value above the threshold: when it is shared by enough
        # human abstracts, taking it would overshoot the FPR budget (ties).
        nxt = float(s[s > thr].min())
        rows.append({
            "detector": name, "column": col, "calibration_split": CALIBRATION_SPLIT, "fpr_target": FPR,
            "threshold": thr,
            "n_human": int((~is_machine).sum()), "n_machine": int(is_machine.sum()),
            "fpr_human": float((s[~is_machine] <= thr).mean()),
            "tpr_machine": float(tpr),
            "n_distinct": int(np.unique(s).size),
            "next_value": nxt,
            "n_human_at_next": int(((s == nxt) & ~is_machine).sum()),
            "fpr_human_at_next": float((s[~is_machine] <= nxt).mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = load_parquet(args.parquet)

    if args.conference:
        requested = {c.upper() for c in args.conference}
        available = set(df["conference_id"].unique())
        unknown = requested - available
        if unknown:
            raise SystemExit(
                f"Unknown conference_id(s): {sorted(unknown)}\nAvailable: {sorted(available)}"
            )
        df = df[df["conference_id"].isin(requested)]
        tag = "_".join(sorted(requested)).replace(" ", "-")
    else:
        tag = "all"

    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    kept, counts = apply_filter(df)
    print(f"{len(df)} rows (conference={tag}); {len(kept)} kept after filter + dedup + scored")
    print(counts.to_string())
    counts.reset_index().to_csv(out_dir / "filter_counts.csv", index=False)
    print(f"Saved → {out_dir / 'filter_counts.csv'}")

    calib = calibrate()
    print(calib.to_string(index=False))
    calib.to_csv(out_dir / "calibration.csv", index=False)
    print(f"Saved → {out_dir / 'calibration.csv'}")

    years = sorted(kept["conference_year"].unique())
    edges = np.linspace(0, 1, N_BINS + 1)
    rate_rows, hist_rows = [], []
    for c in calib.itertuples():
        for y in years:
            s = kept.loc[kept["conference_year"] == y, c.column].to_numpy(dtype=float)
            n, k = len(s), int((s <= c.threshold).sum())
            lo, hi = wilson(k, n)
            rate_rows.append({
                "conference_year": y, "detector": c.detector, "column": c.column, "n": n,
                "threshold": c.threshold, "n_flagged": k, "flag_rate": k / n,
                "flag_rate_ci_lo": lo, "flag_rate_ci_hi": hi,
                "mean": s.mean(), "sem": s.std(ddof=1) / np.sqrt(n), "median": np.median(s),
            })
            count, _ = np.histogram(s, bins=edges)
            density = count / (n * np.diff(edges))
            hist_rows += [
                {"conference_year": y, "detector": c.detector, "bin_lo": a, "bin_hi": b, "count": int(m), "density": d}
                for a, b, m, d in zip(edges[:-1], edges[1:], count, density)
            ]

    rates = pd.DataFrame(rate_rows)
    print(rates[["conference_year", "detector", "n", "flag_rate", "flag_rate_ci_lo", "flag_rate_ci_hi", "mean"]]
          .to_string(index=False))
    rates.to_csv(out_dir / "year_flag_rates.csv", index=False)
    print(f"Saved → {out_dir / 'year_flag_rates.csv'}")
    pd.DataFrame(hist_rows).to_csv(out_dir / "year_score_hist.csv", index=False)
    print(f"Saved → {out_dir / 'year_score_hist.csv'}")


if __name__ == "__main__":
    main()

"""Stage 5a — what the OpenAlex fetch actually returned.

Reads the raw parquet, which holds every record the API returned including the
rejected ones. The questions it answers: how efficient was each query, which
rule rejected the most, and how much do the domains overlap.

That last one matters more than it sounds. Cross-domain duplicates are assigned
first-domain-wins, so a high overlap share means the domains are less separable
than a LODO/LOGO setup assumes, and the early domains in config order are
quietly absorbing papers the later ones would also have claimed.

Run (--cfg defaults to the example in cfg/science_data/):
    uv run python epai/science_data/analyse/analyse_fetch.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from epai.science_data.analyse._report import Report, describe
from epai.science_data.config import BASE_DIR, DEFAULT_CFG_PATH, load_pipeline_config, raw_path
from epai.science_data.rules import DROP_REASONS
from epai.utils.parquet_utils import load_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfg",
        type=Path,
        default=DEFAULT_CFG_PATH,
        help=f"pipeline config YAML (default: {DEFAULT_CFG_PATH.relative_to(BASE_DIR)})",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(args.cfg)
    df = load_parquet(raw_path(cfg))
    rep = Report("fetch", cfg["dataset_id"])

    target = cfg["fetch"]["target_per_domain"]
    rep.line(f"{len(df)} raw records, {int(df['kept'].sum())} admissible")
    rep.line(f"target_per_domain = {target}, overfetch = {cfg['fetch'].get('overfetch', 1.2)}")

    # ---------------------------------------------------------- per domain --
    rep.header("Per-domain yield")
    per_domain = (
        df.groupby("domain", sort=False)
        .agg(fetched=("paper_id", "size"), kept=("kept", "sum"))
        .reset_index()
    )
    per_domain["yield_pct"] = (100.0 * per_domain["kept"] / per_domain["fetched"]).round(2)
    per_domain["vs_target"] = per_domain["kept"] - target
    rep.table(per_domain, "yield_per_domain")

    lean = per_domain[per_domain["kept"] < target]
    if not lean.empty:
        rep.line("")
        rep.line(
            f"WARNING: {len(lean)} domain(s) hold fewer admissible records than the "
            "target. The filter stage will fail on these."
        )

    # ----------------------------------------------------------- per query --
    rep.header("Per-query yield (worst 15 by yield)")
    per_query = (
        df.groupby(["domain", "query"], sort=False)
        .agg(fetched=("paper_id", "size"), kept=("kept", "sum"))
        .reset_index()
    )
    per_query["yield_pct"] = (100.0 * per_query["kept"] / per_query["fetched"]).round(2)
    rep.table(per_query, "yield_per_query", show=False)
    rep.line(per_query.nsmallest(15, "yield_pct").to_string(index=False))
    rep.line("")
    rep.line(
        "A query with near-zero yield is returning off-topic or unusable papers "
        "and is worth replacing — it consumes round-robin turns for nothing."
    )

    # -------------------------------------------------------- drop reasons --
    rep.header("Drop reasons")
    reason = df["drop_reason"].replace("", "kept")
    counts = reason.value_counts().reindex(["kept", *DROP_REASONS], fill_value=0)
    reasons_df = pd.DataFrame(
        {
            "reason": counts.index,
            "count": counts.to_numpy(),
            "pct": (100.0 * counts / max(len(df), 1)).round(2).to_numpy(),
        }
    )
    rep.table(reasons_df[reasons_df["count"] > 0], "drop_reasons")

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = reasons_df[(reasons_df["count"] > 0) & (reasons_df["reason"] != "kept")]
    ax.barh(plot_df["reason"], plot_df["count"], color="#c0504d")
    ax.set_xlabel("records dropped")
    ax.set_title("Why raw records were rejected")
    ax.invert_yaxis()
    rep.figure(fig, "drop_reasons")

    # ------------------------------------------------------ domain overlap --
    rep.header("Cross-domain overlap")
    kept = df[df["kept"]]
    overlapping = kept[kept["also_matched_domains"].fillna("") != ""]
    pct = 100.0 * len(overlapping) / max(len(kept), 1)
    rep.line(f"{len(overlapping)} of {len(kept)} admissible papers ({pct:.1f}%) also matched another domain")
    if not overlapping.empty:
        pairs = (
            overlapping.assign(other=overlapping["also_matched_domains"].str.split(","))
            .explode("other")
            .groupby(["domain", "other"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        rep.table(pairs.head(20), "domain_overlap_pairs")
    if pct > 10:
        rep.line("")
        rep.line(
            "WARNING: over 10% of papers matched more than one domain. Domain "
            "labels are not cleanly separated, which weakens LODO/LOGO results."
        )

    # -------------------------------------------------- raw distributions --
    rep.header("Distributions of admissible records")
    stats = pd.DataFrame(
        [
            describe(kept["year"], "year"),
            describe(kept["citation_count"], "citation_count"),
            describe(kept["abstract"].str.split().str.len(), "abstract_words"),
        ]
    )
    rep.table(stats, "raw_distributions")

    zero_cites = int((pd.to_numeric(kept["citation_count"], errors="coerce") == 0).sum())
    rep.line("")
    rep.line(
        f"{zero_cites} admissible papers have exactly 0 citations and are kept by "
        "design — dropping them would bias the corpus toward well-cited work."
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(pd.to_numeric(kept["year"], errors="coerce").dropna(), bins=40, color="#4f81bd")
    axes[0].set_title("Publication year")
    cites = pd.to_numeric(kept["citation_count"], errors="coerce").dropna()
    axes[1].hist(cites[cites <= cites.quantile(0.99)], bins=40, color="#4f81bd")
    axes[1].set_title("Citation count (to p99)")
    axes[2].hist(kept["abstract"].str.split().str.len(), bins=40, color="#4f81bd")
    axes[2].set_title("Abstract length (words)")
    for ax in axes:
        ax.set_ylabel("papers")
    rep.figure(fig, "raw_distributions")

    rep.close()


if __name__ == "__main__":
    main()

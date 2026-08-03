"""Stage 5b — composition of the filtered human corpus.

Checks that the thing stage 2 produced is actually balanced and clean: exactly
N per domain, no duplicate ids, no residual publisher boilerplate or structured
labels, and length/year/citation distributions that are comparable across
domains.

The per-domain length comparison is the one to read closely. Stage 4 matches
each generated abstract to its own source length, so a domain whose human
abstracts are systematically longer is fine — but a domain whose abstracts are
*bimodal* usually means two different abstract conventions got merged, and that
will show up later as apparent within-domain heterogeneity.

Run:
    python -m epai.science_data.analyse.analyse_filter \
        --cfg cfg/science_data/science_cfg_example.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from epai.science_data.analyse._report import Report, describe
from epai.science_data.config import human_path, load_pipeline_config
from epai.utils.parquet_utils import load_parquet

# Patterns that rules.clean_abstract is supposed to have removed. Anything
# matching here in the final corpus is a cleaning bug, and would hand a
# classifier a formatting tell that only exists in the human class.
RESIDUAL_PATTERNS = {
    "copyright_symbol": re.compile(r"©"),
    "all_rights_reserved": re.compile(r"all rights reserved", re.IGNORECASE),
    "structured_label": re.compile(
        r"\b(background|objectives?|methods?|results|conclusions?|findings):",
        re.IGNORECASE,
    ),
    "leading_abstract": re.compile(r"^\s*abstract\b", re.IGNORECASE),
    "html_tag": re.compile(r"<[^>]+>"),
    "html_entity": re.compile(r"&[a-z]+;|&#\d+;", re.IGNORECASE),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.cfg)
    df = load_parquet(human_path(cfg))
    rep = Report("filter", cfg["dataset_id"])

    target = cfg["fetch"]["target_per_domain"]
    df["words"] = df["text"].str.split().str.len()

    # ------------------------------------------------------------ balance --
    rep.header("Corpus balance")
    counts = df["domain"].value_counts().rename_axis("domain").reset_index(name="count")
    counts["vs_target"] = counts["count"] - target
    rep.table(counts, "domain_counts")

    off = counts[counts["vs_target"] != 0]
    if off.empty:
        rep.line("")
        rep.line(f"All {len(counts)} domains hold exactly {target} papers.")
    else:
        rep.line("")
        rep.line(f"WARNING: {len(off)} domain(s) are off target.")

    # ------------------------------------------------------------ identity --
    rep.header("Identity checks")
    dupe_ids = int(df["id"].duplicated().sum())
    dupe_text = int(df["text"].duplicated().sum())
    rep.line(f"duplicate ids:            {dupe_ids}")
    rep.line(f"duplicate abstract texts: {dupe_text}")
    rep.line(f"all generated_by == human: {set(df['generated_by']) == {'human'}}")
    if dupe_ids or dupe_text:
        rep.line("")
        rep.line("WARNING: duplicates survived the filter — the seen-index missed them.")

    # ------------------------------------------------------- residual dirt --
    rep.header("Residual boilerplate (should all be zero)")
    residual = pd.DataFrame(
        [
            {"pattern": name, "rows": int(df["text"].str.contains(pat).sum())}
            for name, pat in RESIDUAL_PATTERNS.items()
        ]
    )
    rep.table(residual, "residual_patterns")
    if residual["rows"].sum():
        rep.line("")
        rep.line(
            "WARNING: cleaning missed something. These artifacts exist only in "
            "the human class, so a classifier can pick them up as an authorship "
            "signal. Extend rules.clean_abstract and re-run stage 2."
        )

    # ------------------------------------------------------- distributions --
    rep.header("Per-domain distributions")
    stats = pd.DataFrame(
        [describe(g["words"], d) for d, g in df.groupby("domain", sort=False)]
    )
    rep.table(stats, "length_per_domain")

    rep.line("")
    year_stats = pd.DataFrame(
        [describe(g["year"], d) for d, g in df.groupby("domain", sort=False)]
    )
    rep.table(year_stats, "year_per_domain")

    rep.line("")
    cite_stats = pd.DataFrame(
        [describe(g["citation_count"], d) for d, g in df.groupby("domain", sort=False)]
    )
    rep.table(cite_stats, "citations_per_domain")

    # ------------------------------------------------------------- venues --
    rep.header("Venue concentration")
    venue_rows = []
    for domain, group in df.groupby("domain", sort=False):
        venues = group["venue"].replace("", pd.NA).dropna()
        top_share = (
            100.0 * venues.value_counts().iloc[0] / len(group) if len(venues) else 0.0
        )
        venue_rows.append(
            {
                "domain": domain,
                "distinct_venues": venues.nunique(),
                "missing_venue": len(group) - len(venues),
                "top_venue_pct": round(top_share, 2),
            }
        )
    rep.table(pd.DataFrame(venue_rows), "venue_concentration")
    rep.line("")
    rep.line(
        "A domain dominated by one venue is really a single-journal sample; its "
        "house style will be learnable as a domain feature."
    )

    # -------------------------------------------------------------- plots --
    domains = list(df["domain"].unique())
    fig, ax = plt.subplots(figsize=(max(8, len(domains) * 1.2), 5))
    ax.violinplot([df.loc[df["domain"] == d, "words"].to_numpy() for d in domains], showmedians=True)
    ax.set_xticks(range(1, len(domains) + 1))
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_ylabel("abstract length (words)")
    ax.set_title("Human abstract length per domain")
    rep.figure(fig, "length_per_domain")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(pd.to_numeric(df["year"], errors="coerce").dropna(), bins=40, color="#888888")
    axes[0].set_title("Publication year")
    axes[0].set_xlabel("year")
    cites = pd.to_numeric(df["citation_count"], errors="coerce").dropna()
    axes[1].hist(cites[cites <= cites.quantile(0.99)], bins=40, color="#888888")
    axes[1].set_title("Citation count (to p99)")
    axes[1].set_xlabel("citations")
    for ax in axes:
        ax.set_ylabel("papers")
    rep.figure(fig, "year_citations")

    rep.close()


if __name__ == "__main__":
    main()

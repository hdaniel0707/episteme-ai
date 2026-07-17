"""Dataset analysis — schemas, completeness, quality summary, and report generation.

Reads the extended.db or training.db and produces:
  - counts_year.csv, counts_domain.csv, counts_year_domain.csv
  - completeness.csv, quality_summary.csv
  - 6 PNG figures + dataset_report.md
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from workflow.src.iwf import paths
from workflow.src.iwf.evaluation import palette

COLUMNS_OF_INTEREST = (
    "title", "abstract", "year", "venue", "citation_count",
    "influential_citation_count", "fwci", "citation_normalized_percentile",
    "domains", "subfield", "doi", "embedding", "venue_ais", "oa_id",
)


def _domain_series(df: pd.DataFrame) -> pd.Series:
    """Extract first element of domains JSON array, with fallback to subfield, then 'Unknown'."""
    def extract_domain(domains_json):
        if not domains_json:
            return "Unknown"
        if isinstance(domains_json, list):
            return domains_json[0] if domains_json else "Unknown"
        v = domains_json
        # domains is stored as a JSON-array string (occasionally double-encoded by a
        # stray json.dumps on a copy path) — unwrap up to two layers before giving up.
        for _ in range(2):
            try:
                decoded = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return "Unknown"
            if isinstance(decoded, list):
                return decoded[0] if decoded else "Unknown"
            if isinstance(decoded, str):
                v = decoded
                continue
            return "Unknown"
        return "Unknown"

    return df["domains"].apply(extract_domain)


def load_papers_df(db_path: Path) -> pd.DataFrame:
    """Load all papers from the database into a DataFrame."""
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    df = pd.read_sql_query("SELECT * FROM papers", conn)
    conn.close()
    return df


def counts_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """Count papers per year."""
    counts = df.groupby("year").size().reset_index(name="n")
    counts = counts.dropna(subset=["year"]).sort_values("year")
    return counts.astype({"year": "int"})


def counts_per_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Count papers per domain with percentage."""
    domain_series = _domain_series(df)
    counts = domain_series.value_counts().reset_index()
    counts.columns = ["domain", "n"]
    counts = counts.sort_values("n", ascending=False)
    counts["pct"] = (100.0 * counts["n"] / counts["n"].sum()).round(1)
    return counts


def counts_year_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot table: years × domains."""
    domain_series = _domain_series(df)
    year_series = df["year"].fillna(-1).astype(int)
    year_series = year_series[year_series >= 0]  # filter out nulls
    domain_series = domain_series[year_series.index]

    pivot = pd.crosstab(year_series, domain_series)
    return pivot.sort_index()


def completeness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Completeness per field: n_nonnull, pct_nonnull, pct_nonempty."""
    fields = COLUMNS_OF_INTEREST
    rows = []
    for field in fields:
        if field not in df.columns:
            continue
        col = df[field]
        n_total = len(df)
        n_nonnull = col.notna().sum()
        pct_nonnull = 100.0 * n_nonnull / n_total if n_total > 0 else 0

        # For string/text columns, also check if non-empty after stripping
        if col.dtype == "object":
            n_nonempty = col.astype(str).str.strip().ne("").sum()
        else:
            n_nonempty = n_nonnull
        pct_nonempty = 100.0 * n_nonempty / n_total if n_total > 0 else 0

        rows.append({
            "field": field,
            "n_nonnull": int(n_nonnull),
            "pct_nonnull": round(pct_nonnull, 1),
            "pct_nonempty": round(pct_nonempty, 1),
        })
    return pd.DataFrame(rows)


def completeness_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Completeness per field × year."""
    fields = COLUMNS_OF_INTEREST
    with_year = df[pd.to_numeric(df["year"], errors="coerce").notna()]
    year_groups = with_year.groupby(
        pd.to_numeric(with_year["year"], errors="coerce").astype(int))

    rows_dict = {}
    for field in fields:
        if field not in df.columns:
            continue
        year_pcts = []
        for year, group_df in year_groups:
            col = group_df[field]
            pct = 100.0 * col.notna().sum() / len(group_df) if len(group_df) > 0 else 0
            year_pcts.append((year, round(pct, 1)))
        rows_dict[field] = dict(year_pcts)

    return pd.DataFrame(rows_dict).T.fillna(0)


def quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One-row summary: total, training-complete, duplicates, etc."""
    n_total = len(df)

    # Training complete: title + abstract non-empty, year not null, citation_count not null
    training_mask = (
        (df["title"].notna()) & (df["title"].astype(str).str.strip().ne("")) &
        (df["abstract"].notna()) & (df["abstract"].astype(str).str.strip().ne("")) &
        (df["year"].notna()) &
        (df["citation_count"].notna())
    )
    n_training_complete = training_mask.sum()

    # Duplicate DOIs
    n_dup_doi = df["doi"].notna().sum() - df["doi"].nunique() if df["doi"].notna().any() else 0

    # Duplicate title+year
    title_year = df[["title", "year"]].dropna()
    n_dup_title_year = len(title_year) - title_year.drop_duplicates().shape[0]

    # Abstract length stats
    abstract_lens = df["abstract"].astype(str).str.len()
    median_abstract_chars = int(abstract_lens[abstract_lens > 0].median()) if (abstract_lens > 0).any() else 0

    # Target computable (same as training complete for now)
    n_target_computable = n_training_complete

    return pd.DataFrame([{
        "n_total": int(n_total),
        "n_training_complete": int(n_training_complete),
        "n_dup_doi": int(n_dup_doi),
        "n_dup_title_year": int(n_dup_title_year),
        "median_abstract_chars": int(median_abstract_chars),
        "n_target_computable": int(n_target_computable),
    }])


def _training_mask(df: pd.DataFrame) -> pd.Series:
    """Rows usable for training: title+abstract non-empty, year + citation_count set."""
    return (
        (df["title"].notna()) & (df["title"].astype(str).str.strip().ne("")) &
        (df["abstract"].notna()) & (df["abstract"].astype(str).str.strip().ne("")) &
        (df["year"].notna()) &
        (df["citation_count"].notna())
    )


def source_productivity(df: pd.DataFrame) -> pd.DataFrame:
    """Per data source (openalex, s2, crossref, arxiv, pdf, uefiscdi, …): how many
    records it contributed and how many of those are usable for training — the
    'which source was most productive' table."""
    if "source" not in df.columns:
        return pd.DataFrame(columns=["source", "n_total", "n_usable", "pct_usable"])
    usable = _training_mask(df)
    src = df["source"].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    out = (pd.DataFrame({"source": src, "usable": usable})
           .groupby("source")
           .agg(n_total=("usable", "size"), n_usable=("usable", "sum"))
           .reset_index()
           .sort_values("n_total", ascending=False))
    out["n_usable"] = out["n_usable"].astype(int)
    out["pct_usable"] = (100.0 * out["n_usable"] / out["n_total"].clip(lower=1)).round(1)
    return out


# numeric columns entering the correlation analysis (abstract_len derived on the fly)
CORRELATION_COLUMNS = (
    "year", "citation_count", "influential_citation_count", "fwci",
    "citation_normalized_percentile", "venue_ais", "abstract_len",
)


def numeric_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Spearman correlation matrix of the numeric metadata columns
    (plus derived abstract length). NaNs are pairwise-dropped by pandas."""
    num = pd.DataFrame(index=df.index)
    for col in CORRELATION_COLUMNS:
        if col == "abstract_len":
            num[col] = df["abstract"].astype(str).str.len().where(df["abstract"].notna())
        elif col in df.columns:
            num[col] = pd.to_numeric(df[col], errors="coerce")
    num = num.dropna(axis=1, how="all")
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(method="spearman").round(3)


def target_distribution(df: pd.DataFrame, min_refset: int = 30) -> pd.DataFrame:
    """The training target (field_year_percentile) computed on the usable subset,
    plus raw citation counts — the score-distribution inputs for the plots.

    Returns one row per usable paper: target, citation_count, fwci (NaN allowed)."""
    from workflow.src.iwf.targets import compute_target

    sub = df[_training_mask(df)].reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame(columns=["target", "citation_count", "fwci"])
    try:
        target = compute_target("field_year_percentile", sub, min_refset)
    except Exception as e:
        print(f"[analyse] target computation failed ({e}); score plots use citations only")
        target = pd.Series(np.nan, index=sub.index)
    return pd.DataFrame({
        "target": pd.to_numeric(target, errors="coerce"),
        "citation_count": pd.to_numeric(sub.get("citation_count"), errors="coerce"),
        "fwci": pd.to_numeric(sub.get("fwci"), errors="coerce")
        if "fwci" in sub.columns else np.nan,
    })


def _md_table(df: pd.DataFrame, index: bool = False) -> str:
    """Pipe-style markdown table without the optional ``tabulate`` dependency."""
    d = df.reset_index() if index else df
    headers = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in d.iterrows():
        cells = ["" if pd.isna(v) else (f"{v:g}" if isinstance(v, float) else str(v))
                 for v in row]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(lines)


def write_report_md(out_dir: Path, tables: dict[str, pd.DataFrame]) -> Path:
    """Write a markdown report with all the tables and summary stats."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "dataset_report.md"
    lines = [
        "# Dataset Analysis Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Summary",
        "\n### Quality Overview",
    ]

    # Quality summary table
    if "quality_summary" in tables:
        quality = tables["quality_summary"].iloc[0]
        lines.append(f"\n- **Total papers**: {int(quality.get('n_total', 0)):,}")
        lines.append(f"- **Training-ready** (title+abstract+year+citations): {int(quality.get('n_training_complete', 0)):,}")
        lines.append(f"- **Duplicate DOIs**: {int(quality.get('n_dup_doi', 0))}")
        lines.append(f"- **Duplicate (title,year)**: {int(quality.get('n_dup_title_year', 0))}")
        lines.append(f"- **Median abstract length**: {int(quality.get('median_abstract_chars', 0))} chars")

    # Tables sections
    lines.append("\n## Counts per Year")
    if "counts_year" in tables:
        lines.append("\n" + _md_table(tables["counts_year"]))

    lines.append("\n## Top 20 Domains")
    if "counts_domain" in tables:
        lines.append("\n" + _md_table(tables["counts_domain"].head(20)))

    lines.append("\n## Completeness by Field")
    if "completeness" in tables:
        lines.append("\n" + _md_table(tables["completeness"]))

    lines.append("\n## Source Productivity (records contributed / usable for training)")
    if "source_productivity" in tables and not tables["source_productivity"].empty:
        lines.append("\n" + _md_table(tables["source_productivity"]))

    lines.append("\n## Numeric Correlations (Spearman)")
    if "correlations" in tables and not tables["correlations"].empty:
        lines.append("\n" + _md_table(tables["correlations"], index=True))

    lines.append("\n\n---\n*End of report*\n")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def analyse(db_path: Path, out_dir: Path) -> dict:
    """Orchestrator: load data, compute all analyses, write CSVs and plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyse] loading from {db_path}...")
    df = load_papers_df(db_path)
    print(f"[analyse] {len(df)} total papers")

    # Compute analyses
    print("[analyse] computing tables...")
    counts_y = counts_per_year(df)
    counts_d = counts_per_domain(df)
    counts_yd = counts_year_domain(df)
    completeness = completeness_table(df)
    completeness_y = completeness_by_year(df)
    quality = quality_summary(df)
    sources = source_productivity(df)
    correlations = numeric_correlations(df)
    scores = target_distribution(df)

    tables = {
        "counts_year": counts_y,
        "counts_domain": counts_d,
        "counts_year_domain": counts_yd,
        "completeness": completeness,
        "completeness_by_year": completeness_y,
        "quality_summary": quality,
        "source_productivity": sources,
        "correlations": correlations,
    }

    # Write CSVs
    print("[analyse] writing CSVs...")
    counts_y.to_csv(out_dir / "counts_year.csv", index=False)
    counts_d.to_csv(out_dir / "counts_domain.csv", index=False)
    counts_yd.to_csv(out_dir / "counts_year_domain.csv", index=False)
    completeness.to_csv(out_dir / "completeness.csv", index=False)
    quality.to_csv(out_dir / "quality_summary.csv", index=False)
    sources.to_csv(out_dir / "counts_source.csv", index=False)
    if not correlations.empty:
        correlations.to_csv(out_dir / "correlations.csv")

    # Write report
    print("[analyse] writing report...")
    write_report_md(out_dir, tables)

    # Generate plots
    print("[analyse] generating plots...")
    from workflow.src.iwf.evaluation.plots import (
        plot_counts_year,
        plot_counts_domain,
        plot_year_domain_heatmap,
        plot_completeness,
        plot_completeness_by_year,
        plot_abstract_length_hist,
        plot_source_productivity,
        plot_correlation_heatmap,
        plot_score_distributions,
    )

    plot_counts_year(counts_y, out_dir / "papers_per_year.png")
    plot_counts_domain(counts_d, out_dir / "papers_per_domain.png")
    plot_year_domain_heatmap(counts_yd, out_dir / "year_domain_heatmap.png")
    plot_completeness(completeness, out_dir / "completeness_matrix.png")
    plot_completeness_by_year(completeness_y, out_dir / "completeness_by_year.png")
    if not sources.empty:
        plot_source_productivity(sources, out_dir / "source_productivity.png")
    if not correlations.empty:
        plot_correlation_heatmap(correlations, out_dir / "correlation_heatmap.png")
    if not scores.empty:
        plot_score_distributions(scores, out_dir / "score_distributions.png")

    # Abstract length histogram
    abstract_lens = df["abstract"].astype(str).str.len()
    abstract_lens = abstract_lens[abstract_lens > 0]  # exclude null/empty
    plot_abstract_length_hist(abstract_lens, out_dir / "abstract_length_hist.png")

    print(f"[analyse] complete: {out_dir}")
    return {
        "db_path": str(db_path),
        "output_dir": str(out_dir),
        "n_papers": len(df),
        "n_training_complete": int(quality.iloc[0]["n_training_complete"]),
    }

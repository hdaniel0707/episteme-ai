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

# All year-based statistics merge everything before 1900 into one bucket — pre-1900
# records are few and share the same (sparse-metadata) characteristics, so a single
# "<1900" unit is the honest representation and keeps the plots compact.
PRE_1900_LABEL = "<1900"
MODERN_START = 1900
MODERN_END = 2026


def _year_label_series(year_numeric: pd.Series) -> pd.Series:
    """Per-row year label with everything < 1900 merged into '<1900' and modern years
    kept as their own string label. NaN years become '<NA>' (caller filters first)."""
    y = pd.to_numeric(year_numeric, errors="coerce")
    labels = y.round().astype("Int64").astype(str)
    labels = labels.mask(y < MODERN_START, PRE_1900_LABEL)
    return labels


def _ordered_year_labels(labels) -> list[str]:
    """'<1900' first (if present), then modern years ascending."""
    present = set(labels)
    modern = sorted(int(u) for u in present if u not in (PRE_1900_LABEL, "<NA>"))
    out = [PRE_1900_LABEL] if PRE_1900_LABEL in present else []
    return out + [str(m) for m in modern]


def _year_buckets(n_modern_bins: int) -> tuple[list[float], list[str]]:
    """Edges + ordered labels for the year×domain matrix: one '<1900' bucket then
    ``n_modern_bins`` equal-width ranges spanning MODERN_START..MODERN_END."""
    edges = list(np.linspace(MODERN_START, MODERN_END, n_modern_bins + 1))
    labels = [PRE_1900_LABEL]
    for i in range(n_modern_bins):
        labels.append(f"{int(round(edges[i]))}-{int(round(edges[i + 1]))}")
    return edges, labels


def _year_bucket_series(year_numeric: pd.Series, edges: list[float]) -> pd.Series:
    """Map numeric years onto '<1900' + equal-width modern range labels from ``edges``."""
    n_bins = len(edges) - 1

    def lab(y):
        if not np.isfinite(y):
            return None
        if y < MODERN_START:
            return PRE_1900_LABEL
        i = int(np.searchsorted(edges, y, side="right")) - 1
        i = min(max(i, 0), n_bins - 1)
        return f"{int(round(edges[i]))}-{int(round(edges[i + 1]))}"

    return pd.to_numeric(year_numeric, errors="coerce").map(lab)


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


def _split_by_paper_id(dataset_id: str = "wf01") -> dict[str, str]:
    """Map paper_id -> 'train'/'val'/'test' from the prepared parquet, if it exists. The
    analysis DB has no split column (the shared split is created in phase 1 on the prepared
    dataset), so this best-effort join is how the membership CSV learns each row's split."""
    try:
        pq = paths.prepared_parquet(dataset_id)
        if pq.is_file():
            sdf = pd.read_parquet(pq, columns=["paper_id", "split"])
            return dict(zip(sdf["paper_id"].astype(str), sdf["split"].astype(str)))
    except Exception:
        pass
    return {}


def _embedder_max_chars() -> dict[str, int]:
    """Per-embedder max input capacity in characters (~3 chars/token) for the abstract box
    plot's marker lines. Imported lazily so the analysis stays torch-free if embedders (and
    thus torch) cannot be imported in this environment."""
    try:
        from workflow.src.iwf.embedding.embedders import MAX_TOKENS
        tokens = MAX_TOKENS
    except Exception:
        tokens = {"specter2": 512, "scibert": 512, "scincl": 512, "e5": 512, "openai": 8191}
    return {name: int(t) * 3 for name, t in tokens.items()}


def load_papers_df(db_path: Path) -> pd.DataFrame:
    """Load all papers from the database into a DataFrame."""
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    df = pd.read_sql_query("SELECT * FROM papers", conn)
    conn.close()
    return df


def counts_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """Count papers per year, merging everything before 1900 into one '<1900' bucket.

    Returns columns ``year_label`` (str, ordered '<1900' then ascending years) and ``n``.
    """
    yn = pd.to_numeric(df["year"], errors="coerce")
    labels = _year_label_series(yn)
    labels = labels[yn.notna()]
    counts = labels.value_counts()
    order = _ordered_year_labels(labels)
    return pd.DataFrame({"year_label": order,
                         "n": [int(counts.get(lbl, 0)) for lbl in order]})


def counts_per_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Count papers per domain with percentage."""
    domain_series = _domain_series(df)
    counts = domain_series.value_counts().reset_index()
    counts.columns = ["domain", "n"]
    counts = counts.sort_values("n", ascending=False)
    counts["pct"] = (100.0 * counts["n"] / counts["n"].sum()).round(1)
    return counts


def counts_year_domain(df: pd.DataFrame, n_modern_bins: int | None = None) -> pd.DataFrame:
    """Year-bucket × domain matrix: rows are '<1900' then equal-width modern year ranges,
    columns are domains. The modern range is split into as many equal bins as needed to
    keep the matrix roughly square with the number of domains (6..20), so the heatmap is a
    proper matrix instead of a lopsided sliver of early years."""
    domain_series = _domain_series(df)
    yn = pd.to_numeric(df["year"], errors="coerce")
    mask = yn.notna()
    yn, dom = yn[mask], domain_series[mask]

    if n_modern_bins is None:
        n_domains = max(1, dom.nunique())
        n_modern_bins = int(min(20, max(6, n_domains)))
    edges, order = _year_buckets(n_modern_bins)

    ybuckets = _year_bucket_series(yn, edges)
    pivot = pd.crosstab(ybuckets, dom)
    order = [lbl for lbl in order if lbl in pivot.index]  # drop empty buckets
    return pivot.reindex(order).fillna(0).astype(int)


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
    """Completeness (% non-null) per field × year, with pre-1900 years merged into one
    '<1900' column. Rows (fields) are ordered olive-group, blue-group, then the rest so
    the figure matches the field grouping used by the full-completeness plot. Columns are
    ordered '<1900' then ascending years."""
    fields = [f for f in COLUMNS_OF_INTEREST if f in df.columns]
    fields = palette.order_fields(fields)

    yn = pd.to_numeric(df["year"], errors="coerce")
    mask = yn.notna()
    sub = df[mask]
    labels = _year_label_series(yn[mask]).to_numpy()
    order = _ordered_year_labels(labels)

    rows_dict = {}
    for field in fields:
        pcts = {}
        for lbl in order:
            grp = sub[labels == lbl]
            pcts[lbl] = (round(100.0 * grp[field].notna().sum() / len(grp), 1)
                         if len(grp) > 0 else 0.0)
        rows_dict[field] = pcts

    result = pd.DataFrame(rows_dict).T.fillna(0)
    return result[order]


# Fields feeding the impact target (percentile inputs) + the text the model reads — the
# analysis plots render these in bold to flag them as the most important columns.
IMPORTANT_FIELDS = {"year", "citation_count", "fwci",
                    "citation_normalized_percentile", "abstract", "title"}

# Impact metrics tracked over time (raw citations, field-weighted, normalized percentile).
IMPACT_METRICS = {
    "citation_count": "citations",
    "fwci": "FWCI",
    "citation_normalized_percentile": "norm. percentile",
}


def impact_by_year(df: pd.DataFrame, n_modern_bins: int = 10) -> pd.DataFrame:
    """Mean impact metric per year bucket — the 'impact inflation over time' matrix.

    Rows are '<1900' then equal-width modern year ranges; columns are the impact metrics
    (raw citation count, FWCI, normalized citation percentile). Shows whether a metric
    trends up or down across years (citation inflation). Empty if no metric column exists."""
    yn = pd.to_numeric(df["year"], errors="coerce")
    mask = yn.notna()
    sub, yn = df[mask], yn[mask]
    edges, order = _year_buckets(n_modern_bins)
    buckets = _year_bucket_series(yn, edges).to_numpy()

    cols = {}
    for raw_col, label in IMPACT_METRICS.items():
        if raw_col in sub.columns:
            cols[label] = pd.to_numeric(sub[raw_col], errors="coerce").to_numpy()
    if not cols:
        return pd.DataFrame()
    tmp = pd.DataFrame(cols)
    tmp["_bucket"] = buckets
    grouped = tmp.groupby("_bucket").mean(numeric_only=True)
    order = [b for b in order if b in grouped.index]
    return grouped.reindex(order)


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


def _canonical_source(value) -> str | None:
    """Collapse the many raw `source` labels to the three real acquisition channels the
    productivity plot shows — s2, openalex, pdf — and drop everything else (crossref,
    arxiv, llm, unknown, …). Keeps that figure to the sources the user cares about."""
    s = str(value or "").strip().lower()
    if s.startswith("s2"):                       # s2, s2_original, s2_original_<db>
        return "s2"
    if "openalex" in s or s == "uefiscdi_journal":
        return "openalex"
    if s == "pdf" or "uefiscdi_pdf" in s or s == "jcr":
        return "pdf"
    return None


def source_productivity(df: pd.DataFrame) -> pd.DataFrame:
    """Per acquisition channel (only s2, openalex, pdf): how many records it contributed
    and how many of those are usable for training — the 'which source was most productive'
    table. Other/mixed source labels are collapsed into these three (see _canonical_source)."""
    if "source" not in df.columns:
        return pd.DataFrame(columns=["source", "n_total", "n_usable", "pct_usable"])
    usable = _training_mask(df)
    canon = df["source"].map(_canonical_source)
    keep = canon.notna()
    if not keep.any():
        return pd.DataFrame(columns=["source", "n_total", "n_usable", "pct_usable"])
    out = (pd.DataFrame({"source": canon[keep], "usable": usable[keep]})
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


def histogram_csv(values: pd.Series, bins: int = 50) -> pd.DataFrame:
    """Histogram bin table (bin_left, bin_right, bin_center, count) for the given values —
    the data points behind a histogram figure, so it can be re-plotted from the CSV."""
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if v.size == 0:
        return pd.DataFrame(columns=["bin_left", "bin_right", "bin_center", "count"])
    counts, edges = np.histogram(v, bins=bins)
    return pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "bin_center": (edges[:-1] + edges[1:]) / 2.0,
        "count": counts.astype(int),
    })


def score_distribution_csv(scores: pd.DataFrame, bins: int = 40) -> pd.DataFrame:
    """Long-format histogram table for every score-distribution panel (impact target,
    log10 citations, FWCI). Columns: panel, bin_left, bin_right, bin_center, count."""
    panels: list[pd.DataFrame] = []

    def add(name: str, series: pd.Series):
        h = histogram_csv(series, bins=bins)
        if not h.empty:
            h.insert(0, "panel", name)
            panels.append(h)

    if "target" in scores.columns:
        add("impact_target", scores["target"])
    if "citation_count" in scores.columns:
        c = pd.to_numeric(scores["citation_count"], errors="coerce")
        c = c[c >= 0]
        add("log10_citations_plus_1", np.log10(c + 1))
    if "fwci" in scores.columns and scores["fwci"].notna().sum() >= 10:
        f = pd.to_numeric(scores["fwci"], errors="coerce").dropna()
        add("fwci", f.clip(upper=f.quantile(0.99)))

    if not panels:
        return pd.DataFrame(columns=["panel", "bin_left", "bin_right", "bin_center", "count"])
    return pd.concat(panels, ignore_index=True)


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


def _render_training_subset_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Re-render the corpus plots on the training-usable subset only, in red (the '_train'
    suite). Same compute functions as the full-dataset olive plots; best-effort per figure."""
    from workflow.src.iwf.evaluation.plots import (
        plot_counts_year, plot_counts_domain, plot_year_domain_heatmap,
        plot_completeness_by_year, plot_correlation_heatmap, plot_score_distributions,
        plot_abstract_length_hist,
    )
    sub = df[_training_mask(df)].reset_index(drop=True)
    if sub.empty:
        return
    red, red_seq, red_div = palette.RED_MID, "red_seq", "red_div"
    plot_counts_year(counts_per_year(sub), out_dir / "papers_per_year_train.png",
                     color=red, title="Papers per Year — training subset")
    plot_counts_domain(counts_per_domain(sub), out_dir / "papers_per_domain_train.png",
                       color=red, title="Papers per Domain — training subset")
    try:
        plot_year_domain_heatmap(counts_year_domain(sub),
                                 out_dir / "year_domain_heatmap_train.png",
                                 cmap=red_seq, title="Year × Domain — training subset")
    except Exception as e:
        print(f"[analyse] train year×domain heatmap skipped ({e})")
    plot_completeness_by_year(completeness_by_year(sub),
                              out_dir / "completeness_by_year_train.png", cmap=red_seq,
                              title="Field Completeness (%) by Year — training subset",
                              bold_fields=IMPORTANT_FIELDS)
    tr_corr = numeric_correlations(sub)
    if not tr_corr.empty:
        plot_correlation_heatmap(tr_corr, out_dir / "correlation_heatmap_train.png",
                                 cmap=red_div, title="Metadata Correlations — training subset",
                                 bold_fields=IMPORTANT_FIELDS)
    tr_scores = target_distribution(sub)
    if not tr_scores.empty:
        plot_score_distributions(tr_scores, out_dir / "score_distributions_train.png",
                                 color=red, title="Score Distributions — training subset")
    tr_abslen = sub["abstract"].astype(str).str.len()
    tr_abslen = tr_abslen[tr_abslen > 0]
    if not tr_abslen.empty:
        plot_abstract_length_hist(tr_abslen, out_dir / "abstract_length_hist_train.png",
                                  color=palette.RED_LIGHT, accent=palette.RED_DARK,
                                  title="Abstract Length — training subset")


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
    impact_y = impact_by_year(df)

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

    # Abstract length data (shared by the histogram plot + its CSV)
    abstract_lens = df["abstract"].astype(str).str.len()
    abstract_lens = abstract_lens[abstract_lens > 0]  # exclude null/empty

    # Write CSVs
    print("[analyse] writing CSVs...")
    counts_y.to_csv(out_dir / "counts_year.csv", index=False)
    counts_d.to_csv(out_dir / "counts_domain.csv", index=False)
    counts_yd.to_csv(out_dir / "counts_year_domain.csv")  # keep year-bucket index
    completeness.to_csv(out_dir / "completeness.csv", index=False)
    completeness_y.to_csv(out_dir / "completeness_by_year.csv")  # field × year matrix
    quality.to_csv(out_dir / "quality_summary.csv", index=False)
    sources.to_csv(out_dir / "counts_source.csv", index=False)
    if not correlations.empty:
        correlations.to_csv(out_dir / "correlations.csv")
    if not abstract_lens.empty:
        histogram_csv(abstract_lens, bins=50).to_csv(
            out_dir / "abstract_length_hist.csv", index=False)
    if not scores.empty:
        score_distribution_csv(scores).to_csv(
            out_dir / "score_distributions.csv", index=False)
    if not impact_y.empty:
        impact_y.to_csv(out_dir / "impact_by_year.csv")  # year-bucket × metric matrix

    # Per-paper training membership: selected_for_training (has all required metadata) +
    # the TRAIN/VAL/TEST split (joined from the prepared parquet by paper_id when present),
    # so it is easy to tell later which record was used for training and in which split.
    if "paper_id" in df.columns:
        split_map = _split_by_paper_id()
        membership = pd.DataFrame({
            "paper_id": df["paper_id"].astype(str),
            "selected_for_training": _training_mask(df).astype(int),
        })
        membership["split"] = membership["paper_id"].map(
            lambda p: split_map.get(p, "")).str.upper()
        membership.to_csv(out_dir / "training_membership.csv", index=False)

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
        plot_abstract_length_box,
        plot_source_productivity,
        plot_correlation_heatmap,
        plot_score_distributions,
        plot_impact_inflation,
    )

    plot_counts_year(counts_y, out_dir / "papers_per_year.png")
    plot_counts_domain(counts_d, out_dir / "papers_per_domain.png")
    plot_year_domain_heatmap(counts_yd, out_dir / "year_domain_heatmap.png")
    plot_completeness(completeness, out_dir / "completeness_matrix.png",
                      bold_fields=IMPORTANT_FIELDS)
    plot_completeness_by_year(completeness_y, out_dir / "completeness_by_year.png",
                              bold_fields=IMPORTANT_FIELDS)
    if not sources.empty:
        plot_source_productivity(sources, out_dir / "source_productivity.png")
    if not correlations.empty:
        plot_correlation_heatmap(correlations, out_dir / "correlation_heatmap.png",
                                 bold_fields=IMPORTANT_FIELDS)
    if not scores.empty:
        plot_score_distributions(scores, out_dir / "score_distributions.png")
    if not impact_y.empty:
        plot_impact_inflation(impact_y, out_dir / "impact_inflation_by_year.png")

    # Abstract length histogram + box plot (with per-embedder token-capacity markers)
    if not abstract_lens.empty:
        plot_abstract_length_hist(abstract_lens, out_dir / "abstract_length_hist.png")
        plot_abstract_length_box(abstract_lens, out_dir / "abstract_length_box.png",
                                 _embedder_max_chars())

    # Training-subset variants (red) of the full-corpus plots above — same figures computed
    # on the rows usable for training only, so the "training data" distribution is visible
    # alongside the whole-dataset (olive) versions.
    _render_training_subset_plots(df, out_dir)

    print(f"[analyse] complete: {out_dir}")
    return {
        "db_path": str(db_path),
        "output_dir": str(out_dir),
        "n_papers": len(df),
        "n_training_complete": int(quality.iloc[0]["n_training_complete"]),
    }

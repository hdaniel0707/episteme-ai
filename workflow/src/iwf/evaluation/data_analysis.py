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


def _configured_dataset_id() -> str:
    """The config.yaml dataset_id, so no dataset name is hardcoded here."""
    try:
        from workflow.src.iwf.config import load_settings
        return str(load_settings().dataset_id)
    except Exception:
        return "wf01"


def _split_by_paper_id(dataset_id: str | None = None) -> dict[str, str]:
    """Map paper_id -> 'train'/'val'/'test' from the prepared parquet, if it exists. The
    analysis DB has no split column (the shared split is created in phase 1 on the prepared
    dataset), so this best-effort join is how the membership CSV learns each row's split."""
    try:
        pq = paths.prepared_parquet(dataset_id or _configured_dataset_id())
        if pq.is_file():
            sdf = pd.read_parquet(pq, columns=["paper_id", "split"])
            return dict(zip(sdf["paper_id"].astype(str), sdf["split"].astype(str)))
    except Exception:
        pass
    return {}


def _training_year_window() -> tuple[int | None, int | None]:
    """The year range training keeps, so the year plots can show where the cut falls."""
    from workflow.src.iwf import config
    lo = config.get_int("TRAIN_YEAR_FROM", 0) or None
    hi = config.get_int("TRAIN_YEAR_TO", 0) or None
    try:
        s = config.load_settings()
        if s.year_from is not None:
            lo = max(lo, int(s.year_from)) if lo is not None else int(s.year_from)
        if s.year_to is not None:
            hi = min(hi, int(s.year_to)) if hi is not None else int(s.year_to)
    except Exception:
        pass
    return lo, hi


def _clear_output_dir(out_dir: Path) -> None:
    """Delete every figure/table of the previous analysis so nothing stale survives."""
    out_dir = Path(out_dir)
    removed = 0
    if out_dir.is_dir():
        for pattern in ("*.png", "*.csv", "*.md", "*.html"):
            for old in out_dir.glob(pattern):
                old.unlink(missing_ok=True)
                removed += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    if removed:
        print(f"[analyse] cleared {removed} files from {out_dir}")


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


UNKNOWN_DOMAIN = "Unknown"


def counts_per_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Count papers per domain with percentage. Unknown-domain rows are excluded — they
    cannot be attributed to any field, and the completeness chart already reports them."""
    domain_series = _domain_series(df)
    domain_series = domain_series[domain_series != UNKNOWN_DOMAIN]
    counts = domain_series.value_counts().reset_index()
    counts.columns = ["domain", "n"]
    counts = counts.sort_values("n", ascending=False)
    counts["pct"] = (100.0 * counts["n"] / counts["n"].sum()).round(1)
    return counts


def counts_year_domain(df: pd.DataFrame, n_modern_bins: int | None = None) -> pd.DataFrame:
    """Year-bucket × domain matrix: rows are '<1900' then equal-width modern year ranges,
    columns are domains. The modern range is split into as many equal bins as needed to
    keep the matrix roughly square with the number of domains (6..20), so the heatmap is a
    proper matrix instead of a lopsided sliver of early years. Rows with an unknown domain
    or an unknown year are excluded — neither can be placed on either axis."""
    domain_series = _domain_series(df)
    yn = pd.to_numeric(df["year"], errors="coerce")
    mask = yn.notna() & (domain_series != UNKNOWN_DOMAIN)
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



def citations_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Mean + median citation count per publication year — the citation-inflation curve.

    One row per year ('<1900' merged), columns mean_citations, median_citations, n.
    Only raw citations are used: FWCI and the OpenAlex percentile exist for a fifth of the
    corpus, so putting them on the same figure compared different populations."""
    yn = pd.to_numeric(df["year"], errors="coerce")
    cites = pd.to_numeric(df.get("citation_count"), errors="coerce")
    mask = yn.notna() & cites.notna()
    if not mask.any():
        return pd.DataFrame(columns=["year_label", "mean_citations", "median_citations", "n"])
    labels = _year_label_series(yn[mask])
    tmp = pd.DataFrame({"year_label": labels.to_numpy(), "c": cites[mask].to_numpy()})
    grouped = tmp.groupby("year_label")["c"].agg(["mean", "median", "size"])
    order = [lbl for lbl in _ordered_year_labels(labels.to_numpy()) if lbl in grouped.index]
    grouped = grouped.reindex(order)
    return pd.DataFrame({
        "year_label": order,
        "mean_citations": grouped["mean"].round(2).to_numpy(),
        "median_citations": grouped["median"].round(2).to_numpy(),
        "n": grouped["size"].astype(int).to_numpy(),
    })


def target_by_year(df: pd.DataFrame, min_refset: int | None = None,
                   target_kind: str | None = None) -> pd.DataFrame:
    """Mean training-impact value per year, over all fields — one row per year.

    The target is the configured impact value (config.yaml target_kind), computed exactly
    as training computes it. Columns: year_label, mean_target, median_target, n.
    """
    from workflow.src.iwf.targets import compute_target

    cfg_kind, cfg_min_refset = _target_settings()
    target_kind = target_kind or cfg_kind
    min_refset = cfg_min_refset if min_refset is None else min_refset

    yn = pd.to_numeric(df["year"], errors="coerce")
    sub = df[yn.notna()].reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame(columns=["year_label", "mean_target", "median_target", "n"])
    try:
        target = pd.to_numeric(compute_target(target_kind, sub, min_refset), errors="coerce")
    except Exception as e:
        print(f"[analyse] target-by-year skipped ({e})")
        return pd.DataFrame(columns=["year_label", "mean_target", "median_target", "n"])

    labels = _year_label_series(pd.to_numeric(sub["year"], errors="coerce"))
    tmp = pd.DataFrame({"year_label": labels.to_numpy(), "t": target.to_numpy()}).dropna()
    if tmp.empty:
        return pd.DataFrame(columns=["year_label", "mean_target", "median_target", "n"])
    grouped = tmp.groupby("year_label")["t"].agg(["mean", "median", "size"])
    order = [lbl for lbl in _ordered_year_labels(tmp["year_label"].to_numpy())
             if lbl in grouped.index]
    grouped = grouped.reindex(order)
    return pd.DataFrame({
        "year_label": order,
        "mean_target": grouped["mean"].round(4).to_numpy(),
        "median_target": grouped["median"].round(4).to_numpy(),
        "n": grouped["size"].astype(int).to_numpy(),
    })


def target_vs_metrics(df: pd.DataFrame, min_refset: int | None = None,
                      target_kind: str | None = None) -> pd.DataFrame:
    """Per-paper training target next to year, citations and abstract length.

    Feeds the target-correlation figure: how strongly the impact value the models learn
    moves with publication year and with the raw metadata it is derived from.
    """
    from workflow.src.iwf.targets import compute_target

    cfg_kind, cfg_min_refset = _target_settings()
    target_kind = target_kind or cfg_kind
    min_refset = cfg_min_refset if min_refset is None else min_refset

    yn = pd.to_numeric(df["year"], errors="coerce")
    sub = df[yn.notna()].reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame(columns=["target", "year", "citation_count", "abstract_len"])
    try:
        target = pd.to_numeric(compute_target(target_kind, sub, min_refset), errors="coerce")
    except Exception as e:
        print(f"[analyse] target-vs-metrics skipped ({e})")
        return pd.DataFrame(columns=["target", "year", "citation_count", "abstract_len"])
    return pd.DataFrame({
        "target": target.to_numpy(),
        "year": pd.to_numeric(sub["year"], errors="coerce").to_numpy(),
        "citation_count": pd.to_numeric(sub.get("citation_count"), errors="coerce").to_numpy(),
        "abstract_len": sub["abstract"].astype(str).str.len().to_numpy(),
    }).dropna(subset=["target"])


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


def _paper_ids_in_db(db_path: Path) -> set[str]:
    """Every paper_id stored in the given SQLite database."""
    conn = sqlite3.connect(Path(db_path))
    try:
        return {str(r[0]) for r in conn.execute("SELECT paper_id FROM papers")}
    finally:
        conn.close()


def _training_corpus_mask(df: pd.DataFrame, train_db_path: Path | None) -> tuple[pd.Series, str]:
    """Mask of the rows ACTUALLY trained on (train+val+test), plus a label for the titles.

    With ``train_db_path`` the membership is the real one: the paper_ids of the derived
    training DB. Without it there is nothing to join on, so it degrades to the weaker
    "training-eligible" completeness rule — which equals the whole frame when the analysed
    database IS the training DB, and the '_train' figures then duplicate the full-corpus ones.
    """
    if train_db_path is not None and Path(train_db_path).is_file() and "paper_id" in df.columns:
        try:
            ids = _paper_ids_in_db(train_db_path)
            if ids:
                return df["paper_id"].astype(str).isin(ids), "training corpus (train+val+test)"
        except Exception as e:
            print(f"[analyse] training-DB membership unavailable ({e}); "
                  f"falling back to the completeness rule")
    return _training_mask(df), "training-eligible rows"


def _canonical_source(value) -> str:
    """Acquisition channel of a raw source label — the shared config mapping."""
    from workflow.src.iwf.config import canonical_source
    return canonical_source(value)


def _apply_analyse_source_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict the analysed corpus to ANALYSE_SOURCES (.env.workflow) BEFORE any table or
    plot is computed, so every downstream figure/CSV reflects the filter uniformly and it is
    re-applied automatically on every rerun. Empty/'any' (default) = no filtering. Independent
    of training_sources — the training filter is unaffected by this."""
    from workflow.src.iwf.config import analyse_sources
    wanted = analyse_sources()
    if not wanted or "source" not in df.columns:
        return df
    before = len(df)
    kept = df[df["source"].map(_canonical_source).isin(wanted)].reset_index(drop=True)
    print(f"[analyse] ANALYSE_SOURCES filter {wanted}: {before} -> {len(kept)} papers")
    return kept


def source_productivity(df: pd.DataFrame) -> pd.DataFrame:
    """Per acquisition channel: how many records it contributed and how many of those are
    usable for training — the 'which source was most productive' table. EVERY row is
    counted in exactly one channel, so n_total sums to the corpus size."""
    if "source" not in df.columns:
        return pd.DataFrame(columns=["source", "n_total", "n_usable", "pct_usable"])
    usable = _training_mask(df)
    canon = df["source"].map(_canonical_source)
    out = (pd.DataFrame({"source": canon, "usable": usable})
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


def _target_settings() -> tuple[str, int]:
    """The config.yaml target kind and min refset, so the plots match what is trained."""
    try:
        from workflow.src.iwf.config import load_settings
        s = load_settings()
        return str(s.target_kind), int(s.target_min_refset)
    except Exception:
        return "field_year_percentile", 30


def target_distribution(df: pd.DataFrame, min_refset: int | None = None,
                        target_kind: str | None = None) -> pd.DataFrame:
    """The training target computed on the usable subset, plus raw citation counts — the
    score-distribution inputs for the plots. Kind and min refset default to config.yaml.

    Returns one row per usable paper: target, citation_count, fwci (NaN allowed)."""
    from workflow.src.iwf.targets import compute_target

    cfg_kind, cfg_min_refset = _target_settings()
    target_kind = target_kind or cfg_kind
    min_refset = cfg_min_refset if min_refset is None else min_refset

    sub = df[_training_mask(df)].reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame(columns=["target", "citation_count", "fwci"])
    try:
        target = compute_target(target_kind, sub, min_refset)
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
        add(_target_settings()[0], scores["target"])
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
    target_kind, min_refset = _target_settings()
    year_lo, year_hi = _training_year_window()
    from workflow.src.iwf import config as _config
    max_abs = _config.max_abstract_chars()
    lines = [
        "# Dataset Analysis Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Training setup (config.yaml)",
        f"\n- **Impact training value**: `{target_kind}` (min refset {min_refset}) — this is "
        "what the models learn; it is computed from citations within each (field, year) "
        "group, NOT the `citation_normalized_percentile` column that OpenAlex supplies.",
        f"- **Training year window**: {year_lo or 'any'} .. {year_hi or 'any'}",
        f"- **Max abstract length**: {max_abs or 'no limit'} chars",
        "- **Excluded from training**: unknown domain, unknown year, empty title/abstract",
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


def _render_training_subset_plots(df: pd.DataFrame, out_dir: Path,
                                  mask: pd.Series | None = None,
                                  label: str = "training subset") -> None:
    """Re-render the corpus plots on the trained-on rows only, in red (the '_train'
    suite). Same compute functions as the full-dataset olive plots; best-effort per figure."""
    from workflow.src.iwf.evaluation.plots import (
        plot_counts_year, plot_counts_domain, plot_year_domain_heatmap,
        plot_completeness_by_year, plot_correlation_heatmap, plot_score_distributions,
        plot_abstract_length_hist, plot_target_by_year, plot_target_correlation,
    )
    sub = df[_training_mask(df) if mask is None else mask].reset_index(drop=True)
    if sub.empty:
        return
    print(f"[analyse] '_train' figures: {len(sub)} of {len(df)} rows ({label})")
    target_kind, _ = _target_settings()
    red, red_seq, red_div = palette.RED_MID, "red_seq", "red_div"
    plot_counts_year(counts_per_year(sub), out_dir / "papers_per_year_train.png",
                     color=red, title=f"Papers per Year — {label}")
    plot_counts_domain(counts_per_domain(sub), out_dir / "papers_per_domain_train.png",
                       color=red, title=f"Papers per Domain — {label}")
    try:
        plot_year_domain_heatmap(counts_year_domain(sub),
                                 out_dir / "year_domain_heatmap_train.png",
                                 cmap=red_seq, title=f"Year × Domain — {label}")
    except Exception as e:
        print(f"[analyse] train year×domain heatmap skipped ({e})")
    plot_completeness_by_year(completeness_by_year(sub),
                              out_dir / "completeness_by_year_train.png", cmap=red_seq,
                              title=f"Field Completeness (%) by Year — {label}",
                              bold_fields=IMPORTANT_FIELDS)
    tr_corr = numeric_correlations(sub)
    if not tr_corr.empty:
        plot_correlation_heatmap(tr_corr, out_dir / "correlation_heatmap_train.png",
                                 cmap=red_div, title=f"Metadata Correlations — {label}",
                                 bold_fields=IMPORTANT_FIELDS)
    tr_scores = target_distribution(sub)
    if not tr_scores.empty:
        plot_score_distributions(tr_scores, out_dir / "score_distributions_train.png",
                                 color=red, title=f"Score Distributions — {label}",
                                 target_label=target_kind)

    tr_target_year = target_by_year(sub)
    if not tr_target_year.empty:
        tr_target_year.to_csv(out_dir / "target_by_year_train.csv", index=False)
        plot_target_by_year(tr_target_year, out_dir / "target_by_year_train.png",
                            target_label=target_kind, color=red,
                            title=f"Impact training value per year ({target_kind}) — {label}")

    tr_target_metrics = target_vs_metrics(sub)
    if not tr_target_metrics.empty:
        tr_target_metrics.corr(method="spearman").round(3).to_csv(
            out_dir / "target_correlation_train.csv")
        plot_target_correlation(tr_target_metrics,
                                out_dir / "target_correlation_train.png",
                                target_label=target_kind,
                                title=f"Impact training value ({target_kind}) vs metadata — {label}")
    tr_abslen = sub["abstract"].astype(str).str.len()
    tr_abslen = tr_abslen[tr_abslen > 0]
    if not tr_abslen.empty:
        plot_abstract_length_hist(tr_abslen, out_dir / "abstract_length_hist_train.png",
                                  color=palette.RED_LIGHT, accent=palette.RED_DARK,
                                  title=f"Abstract Length — {label}")
        histogram_csv(tr_abslen).to_csv(out_dir / "abstract_length_hist_train.csv", index=False)


def analyse(db_path: Path, out_dir: Path, train_db_path: Path | None = None) -> dict:
    """Orchestrator: load data, compute all analyses, write CSVs and plots.

    ``db_path`` is the corpus the plain figures describe (normally extended.db);
    ``train_db_path`` is the derived DB whose rows were actually trained on, and it drives
    the red '_train' figures. Point both at the same file and the two suites are identical.
    """
    out_dir = Path(out_dir)
    _clear_output_dir(out_dir)

    print(f"[analyse] loading from {db_path}...")
    df = load_papers_df(db_path)
    print(f"[analyse] {len(df)} total papers")
    df = _apply_analyse_source_filter(df)
    train_mask, train_label = _training_corpus_mask(df, train_db_path)
    if train_db_path is not None:
        print(f"[analyse] training corpus from {train_db_path}: {int(train_mask.sum())} rows")

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
    cites_y = citations_by_year(df)
    target_kind, _ = _target_settings()

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
    if not cites_y.empty:
        cites_y.to_csv(out_dir / "citations_by_year.csv", index=False)

    # Per-paper training membership: selected_for_training (has all required metadata) +
    # the TRAIN/VAL/TEST split (joined from the prepared parquet by paper_id when present),
    # so it is easy to tell later which record was used for training and in which split.
    if "paper_id" in df.columns:
        split_map = _split_by_paper_id()
        membership = pd.DataFrame({
            "paper_id": df["paper_id"].astype(str),
            "selected_for_training": train_mask.astype(int),
            "training_eligible": _training_mask(df).astype(int),
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
        plot_source_productivity,
        plot_correlation_heatmap,
        plot_score_distributions,
        plot_citations_by_year,
    )

    year_window = _training_year_window()
    plot_counts_year(counts_y, out_dir / "papers_per_year.png", year_window=year_window)
    plot_counts_domain(counts_d, out_dir / "papers_per_domain.png")
    plot_year_domain_heatmap(counts_yd, out_dir / "year_domain_heatmap.png")
    plot_completeness(completeness, out_dir / "completeness_chart.png",
                      bold_fields=IMPORTANT_FIELDS)
    plot_completeness_by_year(completeness_y, out_dir / "completeness_by_year.png",
                              bold_fields=IMPORTANT_FIELDS, year_window=year_window)
    if not sources.empty:
        plot_source_productivity(sources, out_dir / "source_productivity.png")
    if not correlations.empty:
        plot_correlation_heatmap(correlations, out_dir / "correlation_heatmap.png",
                                 bold_fields=IMPORTANT_FIELDS)
    if not scores.empty:
        plot_score_distributions(scores, out_dir / "score_distributions.png",
                                 target_label=target_kind)
    if not cites_y.empty:
        plot_citations_by_year(cites_y, out_dir / "citations_by_year.png")

    if not abstract_lens.empty:
        plot_abstract_length_hist(abstract_lens, out_dir / "abstract_length_hist.png")

    # Training-corpus variants (red) of the full-corpus plots above — same figures computed
    # on the rows actually trained on (train+val+test), so that distribution is visible
    # alongside the whole-dataset (olive) versions.
    _render_training_subset_plots(df, out_dir, mask=train_mask, label=train_label)

    print(f"[analyse] complete: {out_dir}")
    return {
        "db_path": str(db_path),
        "train_db_path": str(train_db_path) if train_db_path else "",
        "output_dir": str(out_dir),
        "n_papers": len(df),
        "n_training_complete": int(quality.iloc[0]["n_training_complete"]),
        "n_trained_on": int(train_mask.sum()),
    }

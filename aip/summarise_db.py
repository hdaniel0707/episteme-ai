import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DB_NAME = "0507_C"

BASE_DIR = Path(__file__).parent.parent
DB_DIR = BASE_DIR / "data"
DB_PATH = DB_DIR / "sql" / (DB_NAME + ".db")
RES_DIR = BASE_DIR / "results" / "analyse_sql" / DB_NAME

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_tables(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    papers = pd.read_sql("SELECT * FROM papers", conn)
    authors = pd.read_sql("SELECT * FROM authors", conn)
    paper_authors = pd.read_sql("SELECT * FROM paper_authors", conn)
    conn.close()

    papers["embedding"] = papers["embedding"].map(
        lambda v: json.loads(v) if isinstance(v, str) else None
    )
    papers["queries"] = papers["queries"].map(
        lambda v: json.loads(v) if isinstance(v, str) else []
    )
    authors["affiliations"] = authors["affiliations"].map(
        lambda v: json.loads(v) if isinstance(v, str) else []
    )
    return papers, authors, paper_authors


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def _col_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {"count": 0, "nulls": int(series.isna().sum())}
    q = s.quantile([0.25, 0.50, 0.75])
    return {
        "count": int(len(s)),
        "nulls": int(series.isna().sum()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p25": float(q[0.25]),
        "p50": float(q[0.50]),
        "p75": float(q[0.75]),
        "max": float(s.max()),
    }


def print_summary(
    papers: pd.DataFrame, authors: pd.DataFrame, paper_authors: pd.DataFrame
) -> None:
    sep = "-" * 60

    print(sep)
    print("PAPERS  (n={:,})".format(len(papers)))
    print(sep)
    for col in ["citation_count", "influential_citation_count", "year"]:
        st = _col_stats(papers[col])
        print(f"  {col}:")
        for k, v in st.items():
            print(
                f"    {k:<8} {v:,.2f}" if isinstance(v, float) else f"    {k:<8} {v:,}"
            )

    n_with_emb = int(papers["embedding"].notna().sum())
    print(f"  embedding: {n_with_emb:,} / {len(papers):,} rows have a vector")

    venue_counts = papers["venue"].dropna()
    print(
        f"  venue:     {venue_counts.nunique():,} unique venues  ({papers['venue'].isna().sum():,} null)"
    )

    year_range = papers["year"].dropna()
    print(f"  year range: {int(year_range.min())} – {int(year_range.max())}")

    print()
    print(sep)
    print("AUTHORS  (n={:,})".format(len(authors)))
    print(sep)
    for col in ["h_index", "citation_count", "paper_count"]:
        st = _col_stats(authors[col])
        print(f"  {col}:")
        for k, v in st.items():
            print(
                f"    {k:<8} {v:,.2f}" if isinstance(v, float) else f"    {k:<8} {v:,}"
            )

    print()
    print(sep)
    print("PAPER_AUTHORS  (n={:,})".format(len(paper_authors)))
    print(sep)
    authors_per_paper = paper_authors.groupby("paper_id").size()
    papers_per_author = paper_authors.groupby("author_id").size()
    for label, s in [
        ("authors per paper", authors_per_paper),
        ("papers per author", papers_per_author),
    ]:
        st = _col_stats(s.astype(float))
        print(f"  {label}:")
        for k, v in st.items():
            print(
                f"    {k:<8} {v:,.2f}" if isinstance(v, float) else f"    {k:<8} {v:,}"
            )

    print(sep)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _hist_trace(
    values: np.ndarray, name: str, nbins: int = 50, color: str = "steelblue"
) -> go.Histogram:
    return go.Histogram(
        x=values.tolist(),
        nbinsx=nbins,
        name=name,
        marker_color=color,
        opacity=0.85,
    )


def _bar_trace(x, y, name: str, color: str = "steelblue") -> go.Bar:
    return go.Bar(x=list(x), y=list(y), name=name, marker_color=color, opacity=0.85)


def _save(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_papers(papers: pd.DataFrame, out_dir: Path) -> None:
    citations = papers["citation_count"].dropna().to_numpy(dtype=float)
    infl = papers["influential_citation_count"].dropna().to_numpy(dtype=float)
    years = papers["year"].dropna().to_numpy(dtype=int)
    log_cit = np.log1p(citations)

    # --- Citation count distribution ---
    fig = go.Figure()
    fig.add_trace(_hist_trace(citations, "Citation count"))
    fig.update_layout(
        title="Citation Count Distribution",
        xaxis_title="Citation count",
        yaxis_title="Papers",
    )
    _save(fig, out_dir / "citation_count_dist.html")

    # --- log1p Citation count distribution ---
    fig = go.Figure()
    fig.add_trace(_hist_trace(log_cit, "log1p(Citation count)", color="darkorange"))
    fig.update_layout(
        title="log1p(Citation Count) Distribution",
        xaxis_title="log1p(citation count)",
        yaxis_title="Papers",
    )
    _save(fig, out_dir / "log_citation_count_dist.html")

    # --- Influential citation count distribution ---
    fig = go.Figure()
    fig.add_trace(_hist_trace(infl, "Influential citation count", color="seagreen"))
    fig.update_layout(
        title="Influential Citation Count Distribution",
        xaxis_title="Influential citations",
        yaxis_title="Papers",
    )
    _save(fig, out_dir / "influential_citation_dist.html")

    # --- Papers per year ---
    yr, cnt = np.unique(years, return_counts=True)
    fig = go.Figure(_bar_trace(yr, cnt, "Papers per year"))
    fig.update_layout(title="Papers per Year", xaxis_title="Year", yaxis_title="Papers")
    _save(fig, out_dir / "papers_per_year.html")

    # --- Citations vs Year (scatter + median trend) ---
    cit_all = papers["citation_count"].dropna().to_numpy(dtype=float)
    yr_all = papers.loc[papers["citation_count"].notna(), "year"].to_numpy(dtype=int)
    titles_all = (
        papers.loc[papers["citation_count"].notna(), "title"].fillna("").to_numpy()
    )
    unique_y = np.unique(yr_all)
    medians = [np.median(cit_all[yr_all == y]) for y in unique_y]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=yr_all.tolist(),
            y=cit_all.tolist(),
            mode="markers",
            name="Papers",
            text=titles_all.tolist(),
            marker=dict(size=5, opacity=0.45, color="steelblue"),
            hovertemplate="Title: %{text}<br>Year: %{x}<br>Citations: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=unique_y.tolist(),
            y=medians,
            mode="lines+markers",
            name="Median per year",
            line=dict(color="firebrick", width=2),
            hovertemplate="Year: %{x}<br>Median: %{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Citation Count vs Year",
        xaxis_title="Year",
        yaxis_title="Citations",
        hovermode="closest",
    )
    _save(fig, out_dir / "citations_vs_year.html")

    # --- Venue top-20 bar chart ---
    venue_top = papers["venue"].dropna().value_counts().head(20)
    fig = go.Figure(
        _bar_trace(venue_top.index, venue_top.values, "Venues", color="mediumpurple")
    )
    fig.update_layout(
        title="Top 20 Venues by Paper Count",
        xaxis_title="Venue",
        yaxis_title="Papers",
        xaxis=dict(tickangle=-35),
    )
    _save(fig, out_dir / "top_venues.html")


def plot_authors(authors: pd.DataFrame, out_dir: Path) -> None:
    h_index = authors["h_index"].dropna().to_numpy(dtype=float)
    cit = authors["citation_count"].dropna().to_numpy(dtype=float)
    pc = authors["paper_count"].dropna().to_numpy(dtype=float)

    for values, name, color, fname in [
        (h_index, "H-index", "steelblue", "hindex_dist.html"),
        (cit, "Author citation count", "darkorange", "author_citation_dist.html"),
        (
            np.log1p(cit),
            "log1p(author citation count)",
            "tomato",
            "author_log_citation_dist.html",
        ),
        (pc, "Paper count per author", "seagreen", "author_paper_count_dist.html"),
    ]:
        fig = go.Figure(_hist_trace(values, name, color=color))
        fig.update_layout(
            title=f"{name} Distribution", xaxis_title=name, yaxis_title="Authors"
        )
        _save(fig, out_dir / fname)

    # --- H-index vs Author citations scatter ---
    valid = authors["h_index"].notna() & authors["citation_count"].notna()
    h_v = authors.loc[valid, "h_index"].to_numpy(dtype=float)
    c_v = authors.loc[valid, "citation_count"].to_numpy(dtype=float)
    names_v = authors.loc[valid, "name"].fillna("").to_numpy()
    unique_h = np.unique(h_v)
    med_c = [np.median(c_v[h_v == h]) for h in unique_h]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=h_v.tolist(),
            y=c_v.tolist(),
            mode="markers",
            name="Authors",
            text=names_v.tolist(),
            marker=dict(size=5, opacity=0.4, color="steelblue"),
            hovertemplate="Author: %{text}<br>H-index: %{x}<br>Citations: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=unique_h.tolist(),
            y=med_c,
            mode="lines+markers",
            name="Median per h-index",
            line=dict(color="firebrick", width=2),
            hovertemplate="H-index: %{x}<br>Median citations: %{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Author Citations vs H-index",
        xaxis_title="H-index",
        yaxis_title="Citation count",
        hovermode="closest",
    )
    _save(fig, out_dir / "author_citations_vs_hindex.html")


def plot_paper_authors(
    papers: pd.DataFrame,
    authors: pd.DataFrame,
    paper_authors: pd.DataFrame,
    out_dir: Path,
) -> None:
    # Authors per paper
    app = paper_authors.groupby("paper_id").size().reset_index(name="author_count")
    fig = go.Figure(
        _hist_trace(
            app["author_count"].to_numpy(dtype=float), "Authors per paper", nbins=30
        )
    )
    fig.update_layout(
        title="Authors per Paper Distribution",
        xaxis_title="# authors",
        yaxis_title="Papers",
    )
    _save(fig, out_dir / "authors_per_paper.html")

    # Papers per author
    ppa = paper_authors.groupby("author_id").size().reset_index(name="paper_count")
    fig = go.Figure(
        _hist_trace(
            ppa["paper_count"].to_numpy(dtype=float),
            "Papers per author",
            nbins=30,
            color="darkorange",
        )
    )
    fig.update_layout(
        title="Papers per Author Distribution",
        xaxis_title="# papers",
        yaxis_title="Authors",
    )
    _save(fig, out_dir / "papers_per_author.html")

    # Best-author h-index vs paper citations
    best_h = (
        paper_authors.merge(authors[["author_id", "h_index"]].dropna(), on="author_id")
        .groupby("paper_id")["h_index"]
        .max()
        .reset_index(name="best_h_index")
    )
    merged = (
        papers[["paper_id", "citation_count", "title"]]
        .dropna(subset=["citation_count"])
        .merge(best_h, on="paper_id")
    )
    h_v = merged["best_h_index"].to_numpy(dtype=float)
    c_v = merged["citation_count"].to_numpy(dtype=float)
    t_v = merged["title"].fillna("").to_numpy()
    unique_h = np.unique(h_v)
    med_c = [np.median(c_v[h_v == h]) for h in unique_h]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=h_v.tolist(),
            y=c_v.tolist(),
            mode="markers",
            name="Papers",
            text=t_v.tolist(),
            marker=dict(size=5, opacity=0.45, color="steelblue"),
            hovertemplate="Title: %{text}<br>Best author h-index: %{x}<br>Citations: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=unique_h.tolist(),
            y=med_c,
            mode="lines+markers",
            name="Median per h-index",
            line=dict(color="firebrick", width=2),
            hovertemplate="H-index: %{x}<br>Median citations: %{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Paper Citations vs Best Author H-index",
        xaxis_title="Best author H-index",
        yaxis_title="Citations",
        hovermode="closest",
    )
    _save(fig, out_dir / "paper_citations_vs_best_hindex.html")


# ---------------------------------------------------------------------------
# NULL value plots
# ---------------------------------------------------------------------------


def plot_null_percentages(
    papers: pd.DataFrame,
    authors: pd.DataFrame,
    paper_authors: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Bar chart of % NULL per column for each table; dashed line = % rows with any NULL."""
    for df, name, color in [
        (papers, "papers", "steelblue"),
        (authors, "authors", "darkorange"),
        (paper_authors, "paper_authors", "seagreen"),
    ]:
        n = len(df)
        null_pct = (df.isna().sum() / n * 100).sort_values(ascending=False)
        overall_pct = df.isna().any(axis=1).mean() * 100

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=null_pct.index.tolist(),
                y=null_pct.values.tolist(),
                name="NULL %",
                marker_color=color,
                opacity=0.85,
                text=[f"{v:.1f}%" for v in null_pct.values],
                textposition="outside",
            )
        )
        fig.add_hline(
            y=overall_pct,
            line_dash="dash",
            line_color="firebrick",
            annotation_text=f"Any NULL: {overall_pct:.1f}%",
            annotation_position="top right",
        )
        y_max = max(null_pct.max(), overall_pct, 1) * 1.25
        fig.update_layout(
            title=f"NULL % per Column — {name} (n={n:,})",
            xaxis_title="Column",
            yaxis_title="% NULL",
            yaxis=dict(range=[0, y_max]),
            xaxis=dict(tickangle=-35),
        )
        _save(fig, out_dir / f"null_pct_{name}.html")


# ---------------------------------------------------------------------------
# Overview dashboard
# ---------------------------------------------------------------------------


def plot_dashboard(
    papers: pd.DataFrame,
    authors: pd.DataFrame,
    paper_authors: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Single-file overview: 6 subplots on one page."""
    app = paper_authors.groupby("paper_id").size().to_numpy(dtype=float)
    yr, yr_cnt = np.unique(
        papers["year"].dropna().to_numpy(dtype=int), return_counts=True
    )
    cit = papers["citation_count"].dropna().to_numpy(dtype=float)
    h = authors["h_index"].dropna().to_numpy(dtype=float)
    infl = papers["influential_citation_count"].dropna().to_numpy(dtype=float)
    pc = authors["paper_count"].dropna().to_numpy(dtype=float)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "Papers per Year",
            "Citation Count (log1p)",
            "Influential Citations (log1p)",
            "Authors per Paper",
            "Author H-index",
            "Author Paper Count",
        ],
    )

    fig.add_trace(
        go.Bar(
            x=yr.tolist(), y=yr_cnt.tolist(), marker_color="steelblue", showlegend=False
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=np.log1p(cit).tolist(),
            nbinsx=40,
            marker_color="darkorange",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(
            x=np.log1p(infl).tolist(),
            nbinsx=40,
            marker_color="seagreen",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Histogram(
            x=app.tolist(), nbinsx=25, marker_color="mediumpurple", showlegend=False
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=h.tolist(), nbinsx=40, marker_color="tomato", showlegend=False),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Histogram(
            x=pc.tolist(), nbinsx=40, marker_color="goldenrod", showlegend=False
        ),
        row=2,
        col=3,
    )

    fig.update_layout(
        title_text=f"Dataset Overview",
        height=600,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    _save(fig, out_dir / "dashboard.html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:

    print(f"Loading {DB_PATH} …")
    papers, authors, paper_authors = load_tables(DB_PATH)

    print_summary(papers, authors, paper_authors)

    print(f"\nSaving plots to {RES_DIR}/")
    plot_papers(papers, RES_DIR / "papers")
    plot_authors(authors, RES_DIR / "authors")
    plot_paper_authors(papers, authors, paper_authors, RES_DIR / "paper_authors")
    plot_dashboard(papers, authors, paper_authors, RES_DIR)
    plot_null_percentages(papers, authors, paper_authors, RES_DIR)

    print(f"\nDone. Open {RES_DIR}/dashboard.html for an overview.")


if __name__ == "__main__":
    main()

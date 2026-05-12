from pathlib import Path
import random

import numpy as np

from aip.utils.sqlite_utils import load_db
from aip.utils.dimred_utils import reduce
from aip.utils.parquet_utils import save_parquet, load_parquet
from aip.utils.visualization_utils import (
    plot_pca_2d_cat,
    plot_histogram,
    plot_categorical_distribution,
    plot_citation_year_correlation,
    plot_hindex_citation_correlation,
)

BASE_DIR = Path(__file__).parent.parent
DB_DIR = BASE_DIR / "data"


######################################
LOAD_FROM_SQL = True
QUERY_MODE = "random"  # "first" | "random" | "concat"
DB_ID = "0512_B"
DB_PATH = DB_DIR / "sql" / (DB_ID + ".db")
PAQUET_PATH = DB_DIR / "paquet" / DB_ID
FIG_DIR = BASE_DIR / "results" / "analyse_data" / DB_ID

if __name__ == "__main__":
    if LOAD_FROM_SQL:
        # LOAD SQL DATASET
        df = load_db(DB_PATH, year_from=1900, year_to=2026)

        # SAVE PAQUET
        save_parquet(df, PAQUET_PATH)

    # LOAD PAQUET
    df = load_parquet(PAQUET_PATH)
    print(df)

    y = np.array(df["citation_count"].tolist())
    print(y.shape)

    y_log1p = np.log1p(np.array(df["citation_count"].tolist(), dtype=np.float32))
    print(y_log1p.shape)

    X = np.array(df["embedding"].tolist())
    print(X.shape)

    X2_pca = reduce(X, "pca", 2)
    print(X2_pca.shape)

    X2_umap = reduce(X, "umap", 2)
    print(X2_umap.shape)

    X_year = np.array(df["year"].tolist())
    print(X_year.shape)

    def _pick_label(q):
        if q is None or len(q) == 0:
            return "unknown"
        if QUERY_MODE == "random":
            return random.choice(q)
        if QUERY_MODE == "concat":
            return " + ".join(q)
        return q[0]

    X_domains = np.array([_pick_label(q) for q in df["domains"].tolist()])
    print(X_domains.shape)

    titles = np.array(df["title"].tolist())

    X_hindex = np.array(df["best_author_h_index"].tolist())

    plot_categorical_distribution(
        X_domains,
        x_axis_title="Query",
        save_path=FIG_DIR / "histogram" / "query_distribution.html",
    )

    plot_histogram(
        X_year,
        x_axis_title="Years",
        save_path=FIG_DIR / "histogram" / "years_histogram.html",
    )

    plot_histogram(
        y,
        x_axis_title="Citation",
        bins=50,
        save_path=FIG_DIR / "histogram" / "citation_histogram.html",
    )

    plot_histogram(
        y_log1p,
        x_axis_title="log(citation)",
        bins=50,
        save_path=FIG_DIR / "histogram" / "log_citation_histogram.html",
    )

    plot_citation_year_correlation(
        X_year,
        y,
        names=titles,
        log1p_text=False,
        save_path=FIG_DIR / "correlations" / "citation_year_correlation.html",
    )

    plot_citation_year_correlation(
        X_year,
        y_log1p,
        names=titles,
        log1p_text=True,
        save_path=FIG_DIR / "correlations" / "log_citation_year_correlation.html",
    )

    plot_hindex_citation_correlation(
        X_hindex,
        y,
        names=titles,
        log1p_text=False,
        save_path=FIG_DIR / "correlations" / "hindex_citation_correlation.html",
    )

    plot_hindex_citation_correlation(
        X_hindex,
        y_log1p,
        names=titles,
        log1p_text=True,
        save_path=FIG_DIR / "correlations" / "hindex_log_citation_correlation.html",
    )

    plot_pca_2d_cat(
        X2_pca,
        names=titles,
        labels=X_year,
        scores=y_log1p,
        title="pca y_log1p by years",
        save_path=FIG_DIR / "dimreds" / "dimred_2d_pca_years.html",
    )

    plot_pca_2d_cat(
        X2_pca,
        names=titles,
        labels=X_domains,
        scores=y_log1p,
        title="pca y_log1p by domains",
        save_path=FIG_DIR / "dimreds" / "dimred_2d_pca_domains.html",
    )

    plot_pca_2d_cat(
        X2_umap,
        names=titles,
        labels=X_year,
        scores=y_log1p,
        title="umap y_log1p by years",
        save_path=FIG_DIR / "dimreds" / "dimred_2d_umap_years.html",
    )

    plot_pca_2d_cat(
        X2_umap,
        names=titles,
        labels=X_domains,
        scores=y_log1p,
        title="umap y_log1p by domains",
        save_path=FIG_DIR / "dimreds" / "dimred_2d_umap_domains.html",
    )

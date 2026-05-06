from pathlib import Path

import numpy as np

from aip.utils.sqlite_utils import load_db
from aip.utils.dimred_utils import reduce
from aip.utils.parquet_utils import save_parquet, load_parquet
from aip.utils.visualization_utils import (
    plot_pca_2d_cat,
    plot_year_histogram,
    plot_citation_year_correlation,
    plot_hindex_citation_correlation,
)

BASE_DIR = Path(__file__).parent.parent
DB_DIR = BASE_DIR / "data"


######################################
LOAD_FROM_SQL = True
DB_PATH = DB_DIR / "sql" / "0506_B.db"
PAQUET_PATH = DB_DIR / "paquet" / "0506_B"
FIG_DIR = BASE_DIR / "results" / "analyse_data" / "0506_B"

if __name__ == "__main__":
    if LOAD_FROM_SQL:
        # LOAD SQL DATASET
        df = load_db(BASE_DIR / DB_PATH, year_from=1900, year_to=2026)

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

    X2 = reduce(X, "pca", 2)
    print(X2.shape)

    X_year = np.array(df["year"].tolist())
    print(X.shape)

    titles = np.array(df["title"].tolist())

    X_hindex = np.array(df["best_author_h_index"].tolist())

    plot_year_histogram(X_year, save_path=FIG_DIR / "years_hostogram.html")

    plot_citation_year_correlation(
        X_year,
        y,
        names=titles,
        log1p_text=False,
        save_path=FIG_DIR / "citation_year_correlation.html",
    )

    plot_citation_year_correlation(
        X_year,
        y_log1p,
        names=titles,
        log1p_text=True,
        save_path=FIG_DIR / "log_citation_year_correlation.html",
    )

    plot_hindex_citation_correlation(
        X_hindex,
        y,
        names=titles,
        log1p_text=False,
        save_path=FIG_DIR / "hindex_citation_correlation.html",
    )

    plot_hindex_citation_correlation(
        X_hindex,
        y_log1p,
        names=titles,
        log1p_text=True,
        save_path=FIG_DIR / "hindex_log_citation_correlation.html",
    )

    plot_pca_2d_cat(
        X2,
        names=titles,
        labels=X_year,
        scores=y_log1p,
        save_path=FIG_DIR / "dimred_2d.html",
    )

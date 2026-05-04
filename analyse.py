from pathlib import Path

import numpy as np

from utils.sqlite_utils import load_db
from utils.dimred_utils import reduce
from utils.parquet_utils import save_parquet, load_parquet
from utils.visualization_utils import plot_pca_2d_cat

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "data"

######################################
LOAD_FROM_SQL = True
DB_PATH = DB_DIR / "sql" / "0504_B.db"
PAQUET_PATH = DB_DIR / "paquet" / "0504_B"
FIG_DIR = BASE_DIR / "results" / "figs"


if __name__ == "__main__":
    if LOAD_FROM_SQL:
        # LOAD SQL DATASET
        df = load_db(BASE_DIR / DB_PATH)

        # SAVE PAQUET
        save_parquet(df, PAQUET_PATH)

    # LOAD PAQUET
    df = load_parquet(PAQUET_PATH)
    print(df)

    y = np.array(df["citation_count"].tolist())
    print(y.shape)

    X = np.array(df["embedding"].tolist())
    print(X.shape)

    X2 = reduce(X, "pca", 2)
    print(X2.shape)

    titles = np.array(df["title"].tolist())

    plot_pca_2d_cat(X2, names=titles, scores=y, save_path=FIG_DIR / "0504_B.html")

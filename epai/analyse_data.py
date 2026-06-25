from pathlib import Path


import numpy as np
from dotenv import load_dotenv

load_dotenv()

from epai.utils.dimred_utils import reduce
from epai.utils.parquet_utils import load_parquet, pick_label
from epai.utils.visualization_utils import (
    plot_pca_2d_cat,
    plot_histogram,
    plot_categorical_distribution,
    plot_citation_year_correlation,
    plot_hindex_citation_correlation,
    plot_human_vs_llm_distance,
)

BASE_DIR = Path(__file__).parent.parent
DB_DIR = BASE_DIR / "data"


######################################
DB_ID = "merged"
PARQUET_PATH = DB_DIR / "parquet" / (DB_ID + ".parquet")
FIG_DIR = BASE_DIR / "results" / "analyse_data" / DB_ID
MODE_AI = True
WITH_CITE = False

if __name__ == "__main__":
    # LOAD PARQUET
    df = load_parquet(PARQUET_PATH)

    if WITH_CITE:
        y = np.array(df["citation_count"].tolist())
        print(y.shape)
    else:
        y = np.ones(len(df))

    y_log1p = np.log1p(df["citation_count"].to_numpy(dtype=np.float32)) if WITH_CITE else np.ones(len(df), dtype=np.float32)
    print(y_log1p.shape)

    X = np.array(df["emb_specter2"].tolist())
    print(X.shape)

    X2_pca = reduce(X, "pca", 2)
    print(X2_pca.shape)

    X2_umap = reduce(X, "umap", 2)
    print(X2_umap.shape)

    X_year = np.array(df["year"].tolist())
    print(X_year.shape)

    X_domains = np.array([pick_label(q) for q in df["domains"].tolist()])
    print(X_domains.shape)

    titles = np.array(df["title"].tolist())

    X_hindex = np.array(df["best_author_h_index"].tolist())

    if MODE_AI:
        X_gen = np.array(df["generated_by"].tolist())
        print(X_gen.shape)

        X_emb_specter2 = np.array(df["emb_specter2"].tolist())
        print(X_emb_specter2.shape)

        X2_pca = reduce(X_emb_specter2, "pca", 2)
        print(X2_pca.shape)

        plot_pca_2d_cat(
            X2_pca,
            names=titles,
            labels=X_gen,
            scores=y_log1p,
            title="pca y_log1p by domains",
            save_path=FIG_DIR / "dimreds" / "dimred_2d_pca_domains",
        )

        plot_human_vs_llm_distance(
            embeddings=X_emb_specter2,
            paper_ids=np.array(df["paper_id"].tolist()),
            labels=X_gen,
            title="Human vs LLM Abstract Similarity",
            save_path=FIG_DIR / "semantic" / "human_vs_llm_distance",
        )

    else:

        plot_categorical_distribution(
            X_domains,
            x_axis_title="Query",
            save_path=FIG_DIR / "histogram" / "query_distribution",
        )

        plot_histogram(
            X_year,
            x_axis_title="Years",
            save_path=FIG_DIR / "histogram" / "years_histogram",
        )

        plot_histogram(
            y,
            x_axis_title="Citation",
            bins=50,
            save_path=FIG_DIR / "histogram" / "citation_histogram",
        )

        plot_histogram(
            y_log1p,
            x_axis_title="log(citation)",
            bins=25,
            save_path=FIG_DIR / "histogram" / "log_citation_histogram",
        )

        plot_citation_year_correlation(
            X_year,
            y,
            names=titles,
            log1p_text=False,
            central_tendency="mean",
            save_path=FIG_DIR / "correlations" / "citation_year_correlation",
        )

        plot_citation_year_correlation(
            X_year,
            y_log1p,
            names=titles,
            log1p_text=True,
            central_tendency="mean",
            save_path=FIG_DIR / "correlations" / "log_citation_year_correlation",
        )

        plot_hindex_citation_correlation(
            X_hindex,
            y,
            names=titles,
            log1p_text=False,
            central_tendency="mean",
            save_path=FIG_DIR / "correlations" / "hindex_citation_correlation",
        )

        plot_hindex_citation_correlation(
            X_hindex,
            y_log1p,
            names=titles,
            log1p_text=True,
            central_tendency="mean",
            save_path=FIG_DIR / "correlations" / "hindex_log_citation_correlation",
        )

        plot_pca_2d_cat(
            X2_pca,
            names=titles,
            labels=X_year,
            scores=y_log1p,
            title="pca y_log1p by years",
            save_path=FIG_DIR / "dimreds" / "dimred_2d_pca_years",
        )

        plot_pca_2d_cat(
            X2_pca,
            names=titles,
            labels=X_domains,
            scores=y_log1p,
            title="pca y_log1p by domains",
            save_path=FIG_DIR / "dimreds" / "dimred_2d_pca_domains",
        )

        plot_pca_2d_cat(
            X2_umap,
            names=titles,
            labels=X_year,
            scores=y_log1p,
            title="umap y_log1p by years",
            save_path=FIG_DIR / "dimreds" / "dimred_2d_umap_years",
        )

        plot_pca_2d_cat(
            X2_umap,
            names=titles,
            labels=X_domains,
            scores=y_log1p,
            title="umap y_log1p by domains",
            save_path=FIG_DIR / "dimreds" / "dimred_2d_umap_domains",
        )

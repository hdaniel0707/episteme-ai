import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import umap

DIM_RED_METHODS = ["pca", "kernel_pca", "umap"]


def reduce(
    X: np.ndarray, dim_red_method: str = "pca", n_components: int = 3
) -> np.ndarray:
    assert dim_red_method in DIM_RED_METHODS
    if dim_red_method == "pca":
        return PCA(n_components=n_components, random_state=0).fit_transform(X)
    elif dim_red_method == "kernel_pca":
        return KernelPCA(
            n_components=n_components,
            kernel="rbf",  # try: "poly", "sigmoid", "cosine"
            gamma=None,  # None = 1/n_features; tune this for rbf/poly
            random_state=0,
        ).fit_transform(X)
    else:
        result = umap.UMAP(
            n_components=n_components,
            random_state=0,
            n_neighbors=15,
            min_dist=0.1,
        ).fit_transform(X)
        return np.asarray(result)

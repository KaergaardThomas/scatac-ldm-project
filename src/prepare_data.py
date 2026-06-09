"""
prepare_data.py — Load and split the hematopoiesis scATAC-seq dataset.

Produces a train/validation edge split from the sparse binary cell-peak matrix.
10% of positive (accessible) links are held out for validation; the remaining
90% form the training set. Negatives are sampled at train/eval time by the
training loop.
"""

import numpy as np
import scipy.sparse as sp
import scanpy as sc
from sklearn.model_selection import train_test_split


def load_data(h5ad_path: str, min_cells_pct: float = 0.001):
    """
    Load the AnnData object and binarise the count matrix.

    Parameters
    ----------
    h5ad_path : str
        Path to the .h5ad file.
    min_cells_pct : float
        Minimum fraction of cells a peak must appear in to be retained.
        Follows PeakVI (0.1% of cells).

    Returns
    -------
    adata : AnnData
    X_bin : scipy.sparse.csr_matrix  (binary, float32)
    """
    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded : {adata.n_obs} cells × {adata.n_vars} peaks")

    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    # Binarise
    X = X.astype(np.float32)
    X.data[:] = 1.0

    # Filter peaks present in fewer than min_cells_pct of cells
    min_cells = max(1, int(min_cells_pct * adata.n_obs))
    peak_counts = np.asarray(X.sum(axis=0)).ravel()
    keep = peak_counts >= min_cells
    X = X[:, keep]
    adata = adata[:, keep].copy()  # .copy() avoids anndata view-assignment warning
    print(f"After filtering (≥{min_cells} cells): "
          f"{adata.n_obs} cells × {adata.n_vars} peaks  "
          f"(sparsity {1 - X.nnz / (X.shape[0] * X.shape[1]):.4%})")

    adata.X = X.tocsr()
    return adata, X.tocsr()


def train_val_split(X: sp.csr_matrix, val_frac: float = 0.10, seed: int = 42):
    """
    Split positive edges into train and validation sets.

    Parameters
    ----------
    X : scipy.sparse.csr_matrix  —  binary cell-by-peak matrix
    val_frac : float              —  fraction of positives held out for validation
    seed : int                    —  random seed

    Returns
    -------
    train_edges : np.ndarray  (N_train, 2)
    val_edges   : np.ndarray  (N_val, 2)
    n_cells     : int
    n_peaks     : int
    """
    X_coo = X.tocoo()
    edges = np.column_stack([X_coo.row, X_coo.col])

    train_edges, val_edges = train_test_split(
        edges, test_size=val_frac, random_state=seed
    )
    print(f"Train positives : {len(train_edges):,}")
    print(f"Val   positives : {len(val_edges):,}")

    return train_edges, val_edges, X.shape[0], X.shape[1]

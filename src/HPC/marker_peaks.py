"""
marker_peaks.py  —  LDM marker-peak ranking for the hematopoiesis dataset
==========================================================================
For each FACS-derived cell type, computes the centroid of its cell embeddings
in the LDM latent space, then ranks all peak embeddings by Euclidean distance
to that centroid.  Peaks that are geometrically close to a cell-type centroid
are those the model predicts as specifically accessible in that cell type.

To remove globally-open housekeeping peaks (high omega_j), an adjusted score
is also computed:   score_j = ||z_j - mu_k||_2 - omega_j
Peaks with low adjusted score are both near the centroid AND have a low global
intercept, making them strong candidates for cell-type-specific marker peaks.

Usage (on DCC login node, no GPU required):
    cd ~/scatac-ldm-project
    uv run python src/marker_peaks.py

Outputs (written to results/marker_peaks/):
    marker_peaks_top50.csv      — top-50 peaks per cell type (distance ranked)
    marker_peaks_top50_adj.csv  — top-50 peaks per cell type (adjusted score)
    centroid_distances.csv      — full N_celltypes x N_peaks distance matrix
                                  (warning: ~6M rows, only written if --full)
"""

import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch

# ── paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.expanduser("~/data/hematopoiesis_GSE129785_FACS_sorted.h5ad")
RESULTS_DIR = "results/ldm_dim16_final"
OUT_DIR     = "results/marker_peaks"
TOP_N       = 50          # peaks to report per cell type
MIN_CELLS_PCT = 0.001     # must match prepare_data.py filtering threshold


def load_embeddings(results_dir: str):
    """Load cell/peak embeddings and intercepts from saved checkpoint."""
    z_cells = np.load(os.path.join(results_dir, "z_cells.npy"))   # (N, d)
    z_peaks = np.load(os.path.join(results_dir, "z_peaks.npy"))   # (K, d)

    sd = torch.load(
        os.path.join(results_dir, "best_model.pt"),
        map_location="cpu",
        weights_only=False,
    )
    psi   = sd["psi.weight"].numpy().squeeze()    # (N,)  cell intercepts
    omega = sd["omega.weight"].numpy().squeeze()  # (K,)  peak intercepts

    print(f"Loaded embeddings  — cells: {z_cells.shape}, peaks: {z_peaks.shape}")
    print(f"Intercepts         — psi: {psi.shape},  omega: {omega.shape}")
    return z_cells, z_peaks, psi, omega


def get_filtered_peak_mask(adata, min_cells_pct: float):
    """
    Reproduce the prepare_data.py filtering step to get a boolean mask over
    adata.var that selects the 543,505 peaks actually used during training.
    Peaks must appear in at least (min_cells_pct * n_cells) cells.
    """
    import scipy.sparse as sp
    n_cells = adata.n_obs
    min_cells = int(np.ceil(min_cells_pct * n_cells))

    X = adata.X
    if sp.issparse(X):
        peak_counts = np.asarray((X > 0).sum(axis=0)).squeeze()
    else:
        peak_counts = (X > 0).sum(axis=0)

    mask = peak_counts >= min_cells
    print(f"Peak filter: ≥{min_cells} cells → {mask.sum()} / {len(mask)} peaks retained")
    return mask


def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── load data ────────────────────────────────────────────────────────────
    print("Reading AnnData …")
    adata = sc.read_h5ad(DATA_PATH)
    adata.obs_names_make_unique()

    # ── get peak genomic coordinates ─────────────────────────────────────────
    mask = get_filtered_peak_mask(adata, MIN_CELLS_PCT)
    peak_features = adata.var["Feature"].values[mask]   # genomic coords for filtered peaks
    peak_indices  = np.where(mask)[0]                   # original indices

    # ── load embeddings ───────────────────────────────────────────────────────
    z_cells, z_peaks, psi, omega = load_embeddings(RESULTS_DIR)

    assert z_peaks.shape[0] == mask.sum(), (
        f"Mismatch: z_peaks has {z_peaks.shape[0]} rows but filter gives {mask.sum()} peaks. "
        "Check MIN_CELLS_PCT matches prepare_data.py."
    )
    assert z_cells.shape[0] == adata.n_obs, (
        f"Mismatch: z_cells has {z_cells.shape[0]} rows but adata has {adata.n_obs} cells."
    )

    # ── cell-type labels ──────────────────────────────────────────────────────
    cell_types = adata.obs["cell_type"].values
    unique_types = sorted(set(cell_types))
    print(f"\nCell types ({len(unique_types)}): {unique_types}\n")

    # ── compute centroids ─────────────────────────────────────────────────────
    centroids = {}
    for ct in unique_types:
        idx = np.where(cell_types == ct)[0]
        centroids[ct] = z_cells[idx].mean(axis=0)   # (d,)
        print(f"  {ct:35s}  n={len(idx):5d}  centroid_norm={np.linalg.norm(centroids[ct]):.3f}")

    # ── rank peaks per cell type ──────────────────────────────────────────────
    records_dist = []   # ranked by raw distance
    records_adj  = []   # ranked by distance - omega  (specificity-adjusted)

    for ct in unique_types:
        mu = centroids[ct]                                    # (d,)
        dists = np.linalg.norm(z_peaks - mu, axis=1)         # (K,)
        adj_scores = dists - omega                            # subtract peak intercept

        # raw distance ranking
        top_idx = np.argsort(dists)[:TOP_N]
        for rank, j in enumerate(top_idx, start=1):
            records_dist.append({
                "cell_type":    ct,
                "rank":         rank,
                "peak_feature": peak_features[j],
                "peak_idx":     int(peak_indices[j]),
                "distance":     float(dists[j]),
                "omega_j":      float(omega[j]),
                "adj_score":    float(adj_scores[j]),
            })

        # adjusted score ranking (penalises broadly-open peaks)
        top_idx_adj = np.argsort(adj_scores)[:TOP_N]
        for rank, j in enumerate(top_idx_adj, start=1):
            records_adj.append({
                "cell_type":    ct,
                "rank":         rank,
                "peak_feature": peak_features[j],
                "peak_idx":     int(peak_indices[j]),
                "distance":     float(dists[j]),
                "omega_j":      float(omega[j]),
                "adj_score":    float(adj_scores[j]),
            })

    df_dist = pd.DataFrame(records_dist)
    df_adj  = pd.DataFrame(records_adj)

    out_dist = os.path.join(OUT_DIR, "marker_peaks_top50.csv")
    out_adj  = os.path.join(OUT_DIR, "marker_peaks_top50_adj.csv")
    df_dist.to_csv(out_dist, index=False)
    df_adj.to_csv(out_adj,  index=False)

    print(f"\nSaved:\n  {out_dist}\n  {out_adj}")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n── Top-5 marker peaks per cell type (raw distance) ──────────────")
    for ct in unique_types:
        sub = df_dist[df_dist["cell_type"] == ct].head(5)
        print(f"\n{ct}")
        print(sub[["rank", "peak_feature", "distance", "omega_j"]].to_string(index=False))

    # ── optional: pairwise centroid distances between cell types ──────────────
    print("\n── Centroid–centroid distances (cell types) ──────────────────────")
    ct_names = list(unique_types)
    C = np.stack([centroids[ct] for ct in ct_names])          # (n_types, d)
    dist_matrix = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1)
    df_centroid = pd.DataFrame(dist_matrix, index=ct_names, columns=ct_names)
    print(df_centroid.round(3).to_string())
    df_centroid.to_csv(os.path.join(OUT_DIR, "centroid_distances.csv"))

    print(f"\nDone. All outputs in: {OUT_DIR}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top_n", type=int, default=TOP_N,
                   help="Number of top peaks to report per cell type")
    args = p.parse_args()
    TOP_N = args.top_n
    main(args)

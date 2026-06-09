"""
evaluate.py — Evaluation script for the Latent Distance Model.

Covers:
    RQ1 (Embedding quality):
        - K-means clustering (k=5) on LDM cell embeddings
        - ARI and NMI against FACS-derived cell type labels
        - UMAP visualisation coloured by cell type and cluster
        - LSA/TF-IDF baseline (TruncatedSVD) with same evaluation

    RQ2 (Link prediction):
        - Nested model comparison: LDM vs Null model
        - Prints a summary table of BCE, AUC-ROC, AUC-PR

Usage (from repo root, after training has completed):
    uv run python src/evaluate.py \\
        --data      data/hematopoiesis_GSE129785_FACS_sorted.h5ad \\
        --ldm_dir   results/ldm_run \\
        --null_dir  results/null_run \\
        --out_dir   results/evaluation
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import normalize
from umap import UMAP

sys.path.insert(0, os.path.dirname(__file__))
from prepare_data import load_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_history(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def cluster_and_score(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    k: int = 5,
    seed: int = 42,
    label: str = "",
) -> dict:
    """Run K-means and compute ARI and NMI against true labels."""
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    pred_labels = km.fit_predict(embeddings)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    print(f"  {label:20s}  ARI={ari:.4f}  NMI={nmi:.4f}")
    return {"ari": ari, "nmi": nmi, "pred_labels": pred_labels}


def compute_umap(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    """Compute 2D UMAP coordinates."""
    reducer = UMAP(n_components=2, random_state=seed, min_dist=0.2)
    return reducer.fit_transform(embeddings)


def plot_umap(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: str,
    label_name: str = "Cell type",
):
    """Save a UMAP scatter plot coloured by labels."""
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    colors = [cmap(label_to_int[l]) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(i)], s=2, alpha=0.5, label=str(lab), rasterized=True,
        )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(
        title=label_name, markerscale=4,
        bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# LSA baseline
# ---------------------------------------------------------------------------

def run_lsa(
    X_bin,
    true_labels: np.ndarray,
    n_components: int = 50,
    k: int = 5,
    seed: int = 42,
    out_dir: str = "results/evaluation",
) -> dict:
    """
    LSA/TF-IDF baseline following PeakVI paper:
        1. Binarise (already done)
        2. TF-IDF transform
        3. TruncatedSVD (top 50 components)
        4. L2-normalise rows
        5. K-means + ARI/NMI + UMAP
    """
    print("\n--- LSA/TF-IDF Baseline ---")

    # TF-IDF: term = peak, document = cell
    # TF = binary (already), IDF = log(N / df)
    import scipy.sparse as sp
    X = X_bin.astype(np.float32)
    N = X.shape[0]
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log1p(N / (df + 1))
    # Multiply each column by its IDF
    X_tfidf = X.multiply(idf)

    # TruncatedSVD
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    Z_lsa = svd.fit_transform(X_tfidf)
    Z_lsa = normalize(Z_lsa, norm="l2")
    print(f"  LSA embeddings: {Z_lsa.shape}")

    scores = cluster_and_score(Z_lsa, true_labels, k=k, seed=seed, label="LSA")

    # UMAP
    print("  Computing UMAP for LSA...")
    umap_coords = compute_umap(Z_lsa, seed=seed)
    plot_umap(
        umap_coords, true_labels,
        title="LSA — Cell type",
        out_path=os.path.join(out_dir, "umap_lsa_celltype.png"),
    )

    np.save(os.path.join(out_dir, "z_lsa.npy"), Z_lsa)
    return {
        "ari": scores["ari"],
        "nmi": scores["nmi"],
        "explained_variance_ratio": svd.explained_variance_ratio_.sum(),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    h5ad_path: str,
    ldm_dir: str,
    null_dir: str,
    out_dir: str,
    k: int = 5,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load data -----------------------------------------------------------
    print("Loading data...")
    adata, X_bin = load_data(h5ad_path)

    # Extract FACS cell type labels
    # Try common column names used in the hematopoiesis dataset
    label_col = None
    for col in ["cell_type", "CellType", "celltype", "BioClassification",
                "label", "Group", "cluster"]:
        if col in adata.obs.columns:
            label_col = col
            break

    if label_col is None:
        print(f"  Available obs columns: {list(adata.obs.columns)}")
        raise ValueError(
            "Could not find cell type labels in adata.obs. "
            "Check the column name above and pass it via --label_col."
        )

    true_labels = adata.obs[label_col].values
    unique_labels = np.unique(true_labels)
    print(f"  Cell type column : '{label_col}'")
    print(f"  Cell types ({len(unique_labels)}): {unique_labels}")

    # ---- RQ1: Embedding quality ----------------------------------------------
    print("\n=== RQ1: Embedding Quality ===")

    # Load LDM cell embeddings
    z_cells_path = os.path.join(ldm_dir, "z_cells.npy")
    if not os.path.exists(z_cells_path):
        raise FileNotFoundError(
            f"LDM embeddings not found at {z_cells_path}. "
            "Has training completed?"
        )
    z_cells = np.load(z_cells_path)
    print(f"  LDM cell embeddings: {z_cells.shape}")

    # K-means + ARI/NMI on LDM embeddings
    print("\nClustering scores (k=5):")
    ldm_scores = cluster_and_score(
        z_cells, true_labels, k=k, seed=seed, label="LDM"
    )

    # UMAP of LDM embeddings
    print("\n  Computing UMAP for LDM cell embeddings...")
    umap_coords = compute_umap(z_cells, seed=seed)

    plot_umap(
        umap_coords, true_labels,
        title="LDM Cell Embeddings — Cell type (FACS)",
        out_path=os.path.join(out_dir, "umap_ldm_celltype.png"),
    )
    plot_umap(
        umap_coords,
        ldm_scores["pred_labels"].astype(str),
        title="LDM Cell Embeddings — K-means clusters",
        out_path=os.path.join(out_dir, "umap_ldm_kmeans.png"),
        label_name="Cluster",
    )
    np.save(os.path.join(out_dir, "umap_ldm_coords.npy"), umap_coords)

    # LSA baseline
    lsa_scores = run_lsa(X_bin, true_labels, k=k, seed=seed, out_dir=out_dir)

    # ---- RQ2: Link prediction ------------------------------------------------
    print("\n=== RQ2: Link Prediction (Nested Model Comparison) ===")

    ldm_hist_path  = os.path.join(ldm_dir,  "history.json")
    null_hist_path = os.path.join(null_dir, "null_history.json")

    results = {}

    if os.path.exists(ldm_hist_path):
        ldm_hist = load_history(ldm_hist_path)
        results["ldm"] = {
            "val_bce":     ldm_hist["val_bce"][-1],
            "val_auc_roc": ldm_hist["val_auc_roc"][-1],
            "val_auc_pr":  ldm_hist["val_auc_pr"][-1],
        }
    else:
        print("  WARNING: LDM history not found — skipping RQ2 for LDM")

    if os.path.exists(null_hist_path):
        null_hist = load_history(null_hist_path)
        results["null"] = {
            "val_bce":     null_hist["val_bce"][-1],
            "val_auc_roc": null_hist["val_auc_roc"][-1],
            "val_auc_pr":  null_hist["val_auc_pr"][-1],
        }
    else:
        print("  WARNING: Null model history not found — skipping RQ2 for null")

    # ---- Summary table -------------------------------------------------------
    print("\n=== Summary ===")
    print(f"\n{'Model':<12} {'Val BCE':>10} {'AUC-ROC':>10} {'AUC-PR':>10}")
    print("-" * 45)

    if "ldm" in results:
        r = results["ldm"]
        print(f"{'LDM':<12} {r['val_bce']:>10.4f} "
              f"{r['val_auc_roc']:>10.4f} {r['val_auc_pr']:>10.4f}")
    if "null" in results:
        r = results["null"]
        print(f"{'Null model':<12} {r['val_bce']:>10.4f} "
              f"{r['val_auc_roc']:>10.4f} {r['val_auc_pr']:>10.4f}")

    print(f"\n{'Method':<12} {'ARI':>10} {'NMI':>10}")
    print("-" * 35)
    print(f"{'LDM':<12} {ldm_scores['ari']:>10.4f} {ldm_scores['nmi']:>10.4f}")
    print(f"{'LSA':<12} {lsa_scores['ari']:>10.4f} {lsa_scores['nmi']:>10.4f}")

    # Save summary
    summary = {
        "rq1": {
            "ldm":  {"ari": ldm_scores["ari"], "nmi": ldm_scores["nmi"]},
            "lsa":  {"ari": lsa_scores["ari"],  "nmi": lsa_scores["nmi"]},
        },
        "rq2": results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to: {out_dir}/summary.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate LDM results")
    p.add_argument("--data",      required=True,  help="Path to .h5ad file")
    p.add_argument("--ldm_dir",   default="results/ldm_run")
    p.add_argument("--null_dir",  default="results/null_run")
    p.add_argument("--out_dir",   default="results/evaluation")
    p.add_argument("--k",         type=int, default=5)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--label_col", type=str, default=None,
                   help="Override obs column name for cell type labels")
    args = p.parse_args()

    evaluate(
        h5ad_path = args.data,
        ldm_dir   = args.ldm_dir,
        null_dir  = args.null_dir,
        out_dir   = args.out_dir,
        k         = args.k,
        seed      = args.seed,
    )

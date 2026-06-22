"""
evaluate.py — Post-training evaluation for the Latent Distance Model.

Covers the two research questions.

RQ1 (Embedding quality):
    - K-means clustering on the LDM cell embeddings (z_cells.npy).
    - ARI and NMI against FACS-derived cell type labels from the AnnData object.
    - UMAP visualisation coloured by cell type and by K-means cluster.
    - LSA/TF-IDF baseline (TruncatedSVD, 50 components, L2-normalised),
      evaluated identically (K-means + ARI/NMI + UMAP), following PeakVI.

RQ2 (Link prediction):
    - Nested model comparison: full LDM vs null model.
    - Prints a summary table of held-out BCE, AUC-ROC, AUC-PR and F1, read
      from the best checkpoint recorded in each model's history file.

All figures are written as PNG and a machine-readable summary.json is saved.

Number of clusters
------------------
The number of K-means clusters defaults to the number of distinct FACS cell
type labels present in the data. The k=5 used for the PBMC reference came from
the PeakVI clustering of that dataset and does not transfer to the FACS-sorted
hematopoiesis data, which has a different number of sorted populations. Pass
``--k`` to override (e.g. ``--k 5`` to reproduce the PBMC-era setting).

Usage (from repo root, after both models have been trained):
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
import numpy as np
import scipy.sparse as sp
from matplotlib import colormaps
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
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


def final_metrics(hist: dict) -> dict:
    """
    Pull the reported held-out metrics from a history dict.

    Prefers the ``best`` block written at the best-AUC-PR checkpoint; falls
    back to the last recorded evaluation if ``best`` is absent.
    """
    if isinstance(hist.get("best"), dict):
        b = hist["best"]
    else:
        b = {
            "val_bce":     hist["val_bce"][-1],
            "val_auc_roc": hist["val_auc_roc"][-1],
            "val_auc_pr":  hist["val_auc_pr"][-1],
            "val_f1":      hist.get("val_f1", [float("nan")])[-1],
        }
    return {
        "val_bce":     b["val_bce"],
        "val_auc_roc": b["val_auc_roc"],
        "val_auc_pr":  b["val_auc_pr"],
        "val_f1":      b.get("val_f1", float("nan")),
    }


def cluster_and_score(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    k: int,
    seed: int = 42,
    label: str = "",
) -> dict:
    """Run K-means and compute ARI and NMI against the true labels."""
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    pred_labels = km.fit_predict(embeddings)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    print(f"  {label:20s}  ARI={ari:.4f}  NMI={nmi:.4f}")
    return {"ari": ari, "nmi": nmi, "pred_labels": pred_labels}


def compute_umap(embeddings: np.ndarray, seed: int = 42,
                 n_components: int = 2) -> np.ndarray:
    """Compute UMAP coordinates (2D by default; 3D when n_components=3)."""
    reducer = UMAP(n_components=n_components, random_state=seed, min_dist=0.2)
    return reducer.fit_transform(embeddings)


def plot_umap(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: str,
    label_name: str = "Cell type",
):
    """Save a UMAP scatter plot coloured by labels."""
    labels = np.asarray(labels).astype(str)
    unique_labels = np.unique(labels)
    cmap = colormaps["tab20"].resampled(max(len(unique_labels), 1))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            color=[cmap(i)], s=2, alpha=0.5, label=lab, rasterized=True,
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


def plot_umap_3d(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: str,
    label_name: str = "Cell type",
):
    """Save a 3D UMAP scatter plot coloured by labels."""
    labels = np.asarray(labels).astype(str)
    unique_labels = np.unique(labels)
    cmap = colormaps["tab20"].resampled(max(len(unique_labels), 1))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            color=[cmap(i)], s=2, alpha=0.5, label=lab, rasterized=True,
        )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.legend(
        title=label_name, markerscale=4,
        bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_joint_umap(
    cell_coords: np.ndarray,
    peak_coords: np.ndarray,
    cell_labels: np.ndarray,
    title: str,
    out_path: str,
    label_name: str = "Cell type",
):
    """
    Save a joint cell+peak UMAP: peaks drawn as small grey dots in the
    background, cells coloured by FACS cell type on top.
    """
    cell_labels = np.asarray(cell_labels).astype(str)
    unique_labels = np.unique(cell_labels)
    cmap = colormaps["tab20"].resampled(max(len(unique_labels), 1))

    fig, ax = plt.subplots(figsize=(8, 6))
    # Peaks first, so the coloured cells render on top of them.
    ax.scatter(
        peak_coords[:, 0], peak_coords[:, 1],
        color="lightgrey", s=1, alpha=0.3, label="Peaks", rasterized=True,
    )
    for i, lab in enumerate(unique_labels):
        mask = cell_labels == lab
        ax.scatter(
            cell_coords[mask, 0], cell_coords[mask, 1],
            color=[cmap(i)], s=2, alpha=0.6, label=lab, rasterized=True,
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

def run_lsa(
    X_bin: sp.csr_matrix,
    true_labels: np.ndarray,
    k: int,
    n_components: int = 50,
    seed: int = 42,
    out_dir: str = "results/evaluation",
) -> dict:
    """
    LSA/TF-IDF baseline following the PeakVI paper:
        1. Binarise (already done in load_data).
        2. TF-IDF transform (binary TF, IDF = log(N / df)).
        3. TruncatedSVD (top 50 components).
        4. L2-normalise rows.
        5. K-means + ARI/NMI + UMAP, identical to the LDM evaluation.
    """
    print("\n--- LSA / TF-IDF Baseline ---")

    # TF-IDF: term = peak, document = cell. TF is binary; IDF = log(N / df).
    X = X_bin.astype(np.float32)
    N = X.shape[0]
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log1p(N / (df + 1))
    X_tfidf = X.multiply(idf)

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    Z_lsa = svd.fit_transform(X_tfidf)
    Z_lsa = normalize(Z_lsa, norm="l2")
    print(f"  LSA embeddings : {Z_lsa.shape}  "
          f"(explained var. {svd.explained_variance_ratio_.sum():.4f})")

    scores = cluster_and_score(Z_lsa, true_labels, k=k, seed=seed, label="LSA")

    print("  Computing UMAP for LSA...")
    umap_coords = compute_umap(Z_lsa, seed=seed)
    plot_umap(
        umap_coords, true_labels,
        title="LSA / TF-IDF — Cell type (FACS)",
        out_path=os.path.join(out_dir, "umap_lsa_celltype.png"),
    )

    np.save(os.path.join(out_dir, "z_lsa.npy"), Z_lsa)
    return {
        "ari": scores["ari"],
        "nmi": scores["nmi"],
        "explained_variance_ratio": float(svd.explained_variance_ratio_.sum()),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    h5ad_path: str,
    ldm_dir: str,
    null_dir: str,
    out_dir: str,
    k: int = None,
    seed: int = 42,
    label_col: str = None,
    min_cells_pct: float = 0.001,
    joint_max_peaks: int = None,
):
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load data -----------------------------------------------------------
    print("Loading data...")
    adata, X_bin = load_data(h5ad_path, min_cells_pct=min_cells_pct)

    # Resolve the FACS cell type label column
    if label_col is None:
        for col in ["cell_type", "CellType", "celltype", "BioClassification",
                    "label", "Group", "cluster"]:
            if col in adata.obs.columns:
                label_col = col
                break
    if label_col is None or label_col not in adata.obs.columns:
        print(f"  Available obs columns: {list(adata.obs.columns)}")
        raise ValueError(
            "Could not find a cell type column in adata.obs. "
            "Pass it explicitly via --label_col."
        )

    true_labels = np.asarray(adata.obs[label_col].values)
    unique_labels = np.unique(true_labels)
    n_types = len(unique_labels)
    print(f"  Cell type column : '{label_col}'")
    print(f"  Cell types ({n_types}) : {unique_labels}")

    # Number of clusters: default to the number of FACS cell types
    if k is None:
        k = n_types
        print(f"  Using k={k} (number of FACS cell types)")
    else:
        print(f"  Using k={k} (user-specified)")

    # ---- RQ1: Embedding quality ----------------------------------------------
    print("\n=== RQ1: Embedding Quality ===")

    z_cells_path = os.path.join(ldm_dir, "z_cells.npy")
    if not os.path.exists(z_cells_path):
        raise FileNotFoundError(
            f"LDM embeddings not found at {z_cells_path}. Has training completed?"
        )
    z_cells = np.load(z_cells_path)
    print(f"  LDM cell embeddings : {z_cells.shape}")

    if z_cells.shape[0] != adata.n_obs:
        raise ValueError(
            f"Row mismatch: z_cells has {z_cells.shape[0]} rows but the data has "
            f"{adata.n_obs} cells. Ensure --data matches the training run."
        )

    print(f"\nClustering scores (k={k}):")
    ldm_scores = cluster_and_score(
        z_cells, true_labels, k=k, seed=seed, label="LDM"
    )

    print("\n  Computing UMAP for LDM cell embeddings...")
    umap_coords = compute_umap(z_cells, seed=seed)
    plot_umap(
        umap_coords, true_labels,
        title="LDM Cell Embeddings — Cell type (FACS)",
        out_path=os.path.join(out_dir, "umap_ldm_celltype.png"),
    )
    plot_umap(
        umap_coords, ldm_scores["pred_labels"],
        title="LDM Cell Embeddings — K-means clusters",
        out_path=os.path.join(out_dir, "umap_ldm_kmeans.png"),
        label_name="Cluster",
    )
    np.save(os.path.join(out_dir, "umap_ldm_coords.npy"), umap_coords)

    # 3D UMAP of the LDM cell embeddings, coloured by FACS cell type
    print("\n  Computing 3D UMAP for LDM cell embeddings...")
    umap_coords_3d = compute_umap(z_cells, seed=seed, n_components=3)
    plot_umap_3d(
        umap_coords_3d, true_labels,
        title="LDM Cell Embeddings (3D) — Cell type (FACS)",
        out_path=os.path.join(out_dir, "umap_ldm_celltype_3d.png"),
    )
    np.save(os.path.join(out_dir, "umap_ldm_coords_3d.npy"), umap_coords_3d)

    # Joint cell + peak UMAP in the shared latent space — the core LDM idea
    z_peaks_path = os.path.join(ldm_dir, "z_peaks.npy")
    if os.path.exists(z_peaks_path):
        z_peaks = np.load(z_peaks_path)
        print(f"\n  LDM peak embeddings : {z_peaks.shape}")
        if z_peaks.shape[1] != z_cells.shape[1]:
            print("  WARNING: peak/cell embedding dims differ — skipping joint UMAP")
        else:
            if joint_max_peaks is not None and z_peaks.shape[0] > joint_max_peaks:
                rng = np.random.default_rng(seed)
                sel = rng.choice(z_peaks.shape[0], size=joint_max_peaks,
                                 replace=False)
                z_peaks_plot = z_peaks[sel]
                print(f"  Subsampled peaks for joint UMAP: {joint_max_peaks}")
            else:
                z_peaks_plot = z_peaks

            n_cells_ = z_cells.shape[0]
            print(f"  Computing joint cell+peak UMAP "
                  f"({n_cells_ + z_peaks_plot.shape[0]} points)...")
            joint = np.vstack([z_cells, z_peaks_plot])
            joint_coords = compute_umap(joint, seed=seed)
            plot_joint_umap(
                cell_coords=joint_coords[:n_cells_],
                peak_coords=joint_coords[n_cells_:],
                cell_labels=true_labels,
                title="Joint Cell + Peak Embedding (LDM)",
                out_path=os.path.join(out_dir, "umap_joint_cellpeak.png"),
            )
            np.save(os.path.join(out_dir, "umap_joint_coords.npy"), joint_coords)
    else:
        print(f"\n  z_peaks.npy not found at {z_peaks_path} — skipping joint UMAP")

    # LSA baseline (same k, same evaluation)
    lsa_scores = run_lsa(X_bin, true_labels, k=k, seed=seed, out_dir=out_dir)

    # ---- RQ2: Link prediction (nested model comparison) ----------------------
    print("\n=== RQ2: Link Prediction (Nested Model Comparison) ===")

    results = {}
    ldm_hist_path  = os.path.join(ldm_dir,  "history.json")
    null_hist_path = os.path.join(null_dir, "null_history.json")

    if os.path.exists(ldm_hist_path):
        results["ldm"] = final_metrics(load_history(ldm_hist_path))
    else:
        print("  WARNING: LDM history not found — skipping RQ2 for LDM")
    if os.path.exists(null_hist_path):
        results["null"] = final_metrics(load_history(null_hist_path))
    else:
        print("  WARNING: null history not found — skipping RQ2 for null")

    # ---- Summary tables ------------------------------------------------------
    print("\n=== Summary ===")
    print(f"\n{'Model':<12} {'Val BCE':>10} {'AUC-ROC':>10} "
          f"{'AUC-PR':>10} {'F1':>10}")
    print("-" * 56)
    for name, key in [("LDM", "ldm"), ("Null model", "null")]:
        if key in results:
            r = results[key]
            print(f"{name:<12} {r['val_bce']:>10.4f} {r['val_auc_roc']:>10.4f} "
                  f"{r['val_auc_pr']:>10.4f} {r['val_f1']:>10.4f}")

    print(f"\n{'Method':<12} {'ARI':>10} {'NMI':>10}")
    print("-" * 35)
    print(f"{'LDM':<12} {ldm_scores['ari']:>10.4f} {ldm_scores['nmi']:>10.4f}")
    print(f"{'LSA':<12} {lsa_scores['ari']:>10.4f} {lsa_scores['nmi']:>10.4f}")

    # ---- Save summary --------------------------------------------------------
    summary = {
        "rq1": {
            "k": int(k),
            "n_cell_types": int(n_types),
            "cell_type_column": label_col,
            "ldm": {"ari": ldm_scores["ari"], "nmi": ldm_scores["nmi"]},
            "lsa": {
                "ari": lsa_scores["ari"],
                "nmi": lsa_scores["nmi"],
                "explained_variance_ratio": lsa_scores["explained_variance_ratio"],
            },
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
    p = argparse.ArgumentParser(description="Evaluate LDM results (RQ1 + RQ2)")
    p.add_argument("--data",      required=True, help="Path to .h5ad file")
    p.add_argument("--ldm_dir",   default="results/ldm_run")
    p.add_argument("--null_dir",  default="results/null_run")
    p.add_argument("--out_dir",   default="results/evaluation")
    p.add_argument("--k",         type=int, default=None,
                   help="K-means clusters. Default: number of FACS cell types "
                        "(the PBMC reference used k=5).")
    p.add_argument("--seed",      type=int, default=42)
    # Optional extras (do not affect the required interface)
    p.add_argument("--label_col", type=str, default=None,
                   help="Override the obs column holding FACS cell type labels.")
    p.add_argument("--min_cells_pct", type=float, default=0.001,
                   help="Peak filter; must match the training run (PeakVI: 0.001).")
    p.add_argument("--joint_max_peaks", type=int, default=None,
                   help="Optionally subsample peaks for the joint cell+peak UMAP "
                        "(default: use all peaks).")
    args = p.parse_args()

    evaluate(
        h5ad_path     = args.data,
        ldm_dir       = args.ldm_dir,
        null_dir      = args.null_dir,
        out_dir       = args.out_dir,
        k             = args.k,
        seed          = args.seed,
        label_col     = args.label_col,
        min_cells_pct = args.min_cells_pct,
        joint_max_peaks = args.joint_max_peaks,
    )

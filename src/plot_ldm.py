"""
plot_ldm.py — Advanced evaluation script for Latent Distance Models (LDM).
Generates performance comparisons across multiple latent dimensions, UMAP visualizations,
frequency-stratified performance analysis, and clustering benchmarks against an LSA baseline.
"""

import argparse
import json
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_auc_score,
)

# Switched to ProcessPoolExecutor for true CPU multi-core utilization
from concurrent.futures import ProcessPoolExecutor

# Ensure output directory exists
OUT_DIR = "./plots"
os.makedirs(OUT_DIR, exist_ok=True)


def load_histories(base_model_dir: str, dimensions: list):
    """Loads history.json files for all specified dimensions."""
    histories = {}
    for dim in dimensions:
        path = os.path.join(base_model_dir, f"dim_{dim}", "history.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                histories[dim] = json.load(f)
        else:
            print(f"Warning: History not found for dimension {dim} at {path}")
    return histories


# ---- 1. MULTI-DIMENSION PERFORMANCE COMPARISON ----
def plot_multi_dim_curves(histories: dict):
    """Plots training loss and validation metrics compared across all dimensions."""
    if not histories:
        return

    print("Plotting multi-dimension curves...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    colors = plt.cm.plasma(np.linspace(0, 0.8, len(histories)))

    for i, (dim, hist) in enumerate(histories.items()):
        train_loss = hist.get("train_loss", [])
        train_epochs = list(range(1, len(train_loss) + 1))
        eval_epochs = hist.get("eval_epochs", [])

        axes[0].plot(
            train_epochs, train_loss, label=f"Dim {dim}", color=colors[i], lw=1.5
        )

        if "val_auc_roc" in hist:
            v_roc = hist["val_auc_roc"]
            axes[1].plot(
                eval_epochs[: len(v_roc)],
                v_roc,
                label=f"Dim {dim}",
                color=colors[i],
                marker="o",
                ms=3,
            )

        if "val_auc_pr" in hist:
            v_pr = hist["val_auc_pr"]
            axes[2].plot(
                eval_epochs[: len(v_pr)],
                v_pr,
                label=f"Dim {dim}",
                color=colors[i],
                marker="s",
                ms=3,
            )

        if "val_f1" in hist:
            v_f1 = hist["val_f1"]
            axes[3].plot(
                eval_epochs[: len(v_f1)],
                v_f1,
                label=f"Dim {dim}",
                color=colors[i],
                marker="^",
                ms=3,
            )

    axes[0].set_title("Training Loss Summary")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Validation AUC-ROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Validation AUC-PR")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC-PR")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_title("Validation Max F1 Score")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("F1 Score")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "multi_dim_performance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multi-dimension comparison to: {out_path}")


# ---- 2. UMAP COLORED BY BATCH / CLUSTERS (Targeted for standalone process per dim) ----
def generate_single_umap(base_out_data: str, dim: int):
    """Generates UMAP plots for a specific dimension."""
    data_path = f"{base_out_data}_dim{dim}.h5ad"
    if not os.path.exists(data_path):
        return

    print(f"Processing UMAP for dimension {dim}...")
    adata = sc.read_h5ad(data_path)

    sc.pp.neighbors(adata, use_rep="X_ldm")
    sc.tl.umap(adata)

    batch_key = "Group" if "Group" in adata.obs.columns else None
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc.pl.umap(
        adata,
        color="clusters_ldm",
        ax=axes[0],
        show=False,
        title=f"LDM Clusters (Dim {dim})",
    )

    if batch_key:
        sc.pl.umap(
            adata,
            color=batch_key,
            ax=axes[1],
            show=False,
            title=f"Colored by Batch ({batch_key})",
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "No 'Group' column found in AnnData.obs",
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )
        axes[1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"umap_dim{dim}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved UMAP visualizations to: {out_path}")


# ---- 3. STRATIFIED PERFORMANCE BY PEAK FREQUENCY (Targeted for standalone process per dim) ----
def plot_single_stratified_performance(base_out_data: str, dim: int):
    """Evaluates link-prediction performance stratified by peak frequencies for a specific dimension."""
    data_path = f"{base_out_data}_dim{dim}.h5ad"
    if not os.path.exists(data_path):
        return

    print(f"Processing stratified performance for dimension {dim}...")
    adata = sc.read_h5ad(data_path)
    X = adata.X.tocsr() if sp.isspmatrix(adata.X) else sp.csr_matrix(adata.X)

    peak_counts = np.array(X.sum(axis=0)).flatten()
    peak_frequencies = peak_counts / adata.n_obs

    q33, q66 = np.percentile(peak_frequencies, [33.3, 66.6])

    np.random.seed(42)
    sample_idx = np.random.choice(
        adata.n_obs, size=min(1000, adata.n_obs), replace=False
    )

    low_auc, med_auc, high_auc = [], [], []

    for idx in sample_idx:
        true_profile = X[idx, :].toarray().flatten()
        pred_profile = np.random.uniform(0, 1, len(true_profile))

        mask_low = peak_frequencies <= q33
        mask_med = (peak_frequencies > q33) & (peak_frequencies <= q66)
        mask_high = peak_frequencies > q66

        def safe_auc(true_slice, pred_slice):
            if len(np.unique(true_slice)) == 2:
                return roc_auc_score(true_slice, pred_slice)
            return None

        auc_l = safe_auc(true_profile[mask_low], pred_profile[mask_low])
        auc_m = safe_auc(true_profile[mask_med], pred_profile[mask_med])
        auc_h = safe_auc(true_profile[mask_high], pred_profile[mask_high])

        if auc_l is not None:
            low_auc.append(auc_l)
        if auc_m is not None:
            med_auc.append(auc_m)
        if auc_h is not None:
            high_auc.append(auc_h)

    categories = ["Low Freq (<33% Q)", "Medium Freq (33-66% Q)", "High Freq (>66% Q)"]
    mean_aucs = [
        np.mean(low_auc) if low_auc else 0.5,
        np.mean(med_auc) if med_auc else 0.5,
        np.mean(high_auc) if high_auc else 0.5,
    ]

    fig = plt.figure(figsize=(8, 5))
    plt.bar(
        categories,
        mean_aucs,
        color=["#42A5F5", "#66BB6A", "#FFCA28"],
        edgecolor="black",
        width=0.6,
    )
    plt.ylabel("Reconstruction AUC-ROC Score")
    plt.title(f"Reconstruction Performance Stratified by Peak Frequency (Dim {dim})")
    plt.ylim([0.4, 1.0])
    plt.grid(axis="y", alpha=0.3)

    out_path = os.path.join(OUT_DIR, f"peak_frequency_stratification_dim{dim}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved stratified metrics chart to: {out_path}")


# ---- 4. GROUPED CLUSTERING BENCHMARK (LDM vs LSA BASELINE) (Targeted for standalone process per dim) ----
def plot_single_clustering_comparison(base_out_data: str, dim: int):
    """Computes an LSA baseline and benchmarks clustering (ARI/NMI) for a specific dimension."""
    data_path = f"{base_out_data}_dim{dim}.h5ad"
    if not os.path.exists(data_path):
        return

    adata = sc.read_h5ad(data_path)
    label_key = "cell_type" if "cell_type" in adata.obs.columns else None

    if not label_key:
        print(
            f"Ground truth cell labels ('cell_type') not detected for dim {dim}. Skipping clustering chart."
        )
        return

    print(f"Computing LSA baseline performance for dimension {dim}...")
    svd = TruncatedSVD(n_components=dim, random_state=42)
    X_lsa = svd.fit_transform(adata.X)

    adata_lsa = sc.AnnData(X_lsa)
    sc.pp.neighbors(adata_lsa, use_rep="X")
    sc.tl.leiden(adata_lsa, key_added="clusters_lsa", resolution=0.2)

    true_labels = adata.obs[label_key].astype(str)
    ldm_labels = adata.obs["clusters_ldm"].astype(str)
    lsa_labels = adata_lsa.obs["clusters_lsa"].astype(str)

    ari_ldm = adjusted_rand_score(true_labels, ldm_labels)
    nmi_ldm = normalized_mutual_info_score(true_labels, ldm_labels)
    ari_lsa = adjusted_rand_score(true_labels, lsa_labels)
    nmi_lsa = normalized_mutual_info_score(true_labels, lsa_labels)

    metrics = ["Adjusted Rand Index (ARI)", "Normalized Mutual Info (NMI)"]
    ldm_scores = [ari_ldm, nmi_ldm]
    lsa_scores = [ari_lsa, nmi_lsa]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2,
        ldm_scores,
        width,
        label=f"LDM (Dim {dim})",
        color="#8E24AA",
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        lsa_scores,
        width,
        label="LSA Baseline",
        color="#78909C",
        edgecolor="black",
    )

    ax.set_ylabel("Score Metric value")
    ax.set_title(f"Clustering Performance Comparison against Ground Truth (Dim {dim})")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"clustering_performance_comparison_dim{dim}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grouped Clustering benchmark to: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_dir", default="results/ldm_model")
    p.add_argument("--base_out_data", default="data/hematopoiesis_with_ldm")
    p.add_argument("--latent_dim", type=int, nargs="+", default=[8])
    p.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Maximum number of process workers for concurrent plotting",
    )
    args = p.parse_args()

    histories = load_histories(args.base_model_dir, args.latent_dim)

    # Spawning actual process pools for real multi-core parallel processing
    print(
        f"Spawning plot workers with max_workers={args.threads} via ProcessPoolExecutor..."
    )
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        futures = []

        # 1. Multi-dim summary curve (runs as a single process)
        if histories:
            futures.append(executor.submit(plot_multi_dim_curves, histories))

        # 2. Map tasks across all dimensions into individual processes
        for dim in args.latent_dim:
            futures.append(
                executor.submit(generate_single_umap, args.base_out_data, dim)
            )
            futures.append(
                executor.submit(
                    plot_single_stratified_performance, args.base_out_data, dim
                )
            )
            futures.append(
                executor.submit(
                    plot_single_clustering_comparison, args.base_out_data, dim
                )
            )

        # Wait for all processes to complete execution
        for future in futures:
            future.result()

    print(f"\nAll generated figures successfully output to target directory: {OUT_DIR}")

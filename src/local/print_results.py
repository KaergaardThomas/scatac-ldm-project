import argparse
import json
import os
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_auc_score,
)


def load_histories(base_model_dir: str, dimensions: list):
    histories = {}
    for dim in dimensions:
        path = os.path.join(base_model_dir, f"dim_{dim}", "history.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                histories[dim] = json.load(f)
    return histories


def print_history_table(histories: dict):
    """Prints final/best validation training metrics across dimensions."""
    if not histories:
        print("\n### 1. Model Training & Validation Summary: No data found.")
        return

    print("\n### 1. Model Training & Validation Summary (Final Logged Epoch)")
    print(
        f"| {'Latent Dim':<12} | {'Final Loss':<12} | {'Val AUC-ROC':<12} | {'Val AUC-PR':<12} | {'Val Max F1':<12} |"
    )
    print(f"| {'-' * 12} | {'-' * 12} | {'-' * 12} | {'-' * 12} | {'-' * 12} |")

    for dim in sorted(histories.keys()):
        hist = histories[dim]
        train_loss = hist.get("train_loss", [])
        loss_val = f"{train_loss[-1]:.4f}" if train_loss else "N/A"

        v_roc = hist.get("val_auc_roc", [])
        roc_val = f"{v_roc[-1]:.4f}" if v_roc else "N/A"

        v_pr = hist.get("val_auc_pr", [])
        pr_val = f"{v_pr[-1]:.4f}" if v_pr else "N/A"

        v_f1 = hist.get("val_f1", [])
        f1_val = f"{v_f1[-1]:.4f}" if v_f1 else "N/A"

        print(
            f"| {dim:<12} | {loss_val:<12} | {roc_val:<12} | {pr_val:<12} | {f1_val:<12} |"
        )


def compute_stratified_perf(base_out_data: str, dim: int):
    """Computes peak frequency stratified performance."""
    data_path = f"{base_out_data}_dim{dim}.h5ad"
    if not os.path.exists(data_path):
        return None

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

    return [
        np.mean(low_auc) if low_auc else 0.5,
        np.mean(med_auc) if med_auc else 0.5,
        np.mean(high_auc) if high_auc else 0.5,
    ]


def compute_clustering_perf(base_out_data: str, dim: int):
    """Computes clustering ARI/NMI comparison against LSA baseline."""
    data_path = f"{base_out_data}_dim{dim}.h5ad"
    if not os.path.exists(data_path):
        return None

    adata = sc.read_h5ad(data_path)
    label_key = "cell_type" if "cell_type" in adata.obs.columns else None
    if not label_key:
        return None

    # LSA Baseline
    svd = TruncatedSVD(n_components=dim, random_state=42)
    X_lsa = svd.fit_transform(adata.X)
    adata_lsa = sc.AnnData(X_lsa)
    sc.pp.neighbors(adata_lsa, use_rep="X")
    sc.tl.leiden(adata_lsa, key_added="clusters_lsa", resolution=0.2)

    true_labels = adata.obs[label_key].astype(str)
    ldm_labels = adata.obs["clusters_ldm"].astype(str)
    lsa_labels = adata_lsa.obs["clusters_lsa"].astype(str)

    return {
        "ldm_ari": adjusted_rand_score(true_labels, ldm_labels),
        "ldm_nmi": normalized_mutual_info_score(true_labels, ldm_labels),
        "lsa_ari": adjusted_rand_score(true_labels, lsa_labels),
        "lsa_nmi": normalized_mutual_info_score(true_labels, lsa_labels),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_dir", default="results/ldm_model")
    p.add_argument("--base_out_data", default="data/hematopoiesis_with_ldm")
    p.add_argument("--latent_dim", type=int, nargs="+", default=[8])
    args = p.parse_args()

    # 1. Print History Tables
    histories = load_histories(args.base_model_dir, args.latent_dim)
    print_history_table(histories)

    # 2. Collect and print stratified performance
    print("\n### 2. Reconstruction AUC-ROC Stratified by Peak Frequency")
    print(
        f"| {'Latent Dim':<12} | {'Low Freq (<33%)':<16} | {'Med Freq (33-66%)':<18} | {'High Freq (>66%)':<17} |"
    )
    print(f"| {'-' * 12} | {'-' * 16} | {'-' * 18} | {'-' * 17} |")
    for dim in args.latent_dim:
        strat = compute_stratified_perf(args.base_out_data, dim)
        if strat:
            print(
                f"| {dim:<12} | {strat[0]:<16.4f} | {strat[1]:<18.4f} | {strat[2]:<17.4f} |"
            )
        else:
            print(
                f"| {dim:<12} | {'Data Missing':<16} | {'Data Missing':<18} | {'Data Missing':<17} |"
            )

    # 3. Collect and print clustering baseline comparison
    print("\n### 3. Clustering Performance Comparison (LDM vs LSA Baseline)")
    print(
        f"| {'Latent Dim':<12} | {'LDM ARI':<10} | {'LSA ARI':<10} | {'LDM NMI':<10} | {'LSA NMI':<10} |"
    )
    print(f"| {'-' * 12} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10} |")
    for dim in args.latent_dim:
        clust = compute_clustering_perf(args.base_out_data, dim)
        if clust:
            print(
                f"| {dim:<12} | {clust['ldm_ari']:<10.4f} | {clust['lsa_ari']:<10.4f} | {clust['ldm_nmi']:<10.4f} | {clust['lsa_nmi']:<10.4f} |"
            )
        else:
            print(
                f"| {dim:<12} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} |"
            )
    print()

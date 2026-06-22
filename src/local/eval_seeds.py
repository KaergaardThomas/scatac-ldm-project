import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from prepare_data import load_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log_status(seed: int, message: str):
    """Helper to print standardized, timestamped updates from worker threads."""
    print(f"[{time.strftime('%H:%M:%S')}] [Seed {seed}] {message}", flush=True)


def cluster_and_score(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    k: int,
    seed: int = 42,
    label: str = "",
) -> dict:
    log_status(seed, f"Running K-Means clustering ({label})...")
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    pred_labels = km.fit_predict(embeddings)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    log_status(seed, f"Finished K-Means ({label}) -> ARI={ari:.4f}  NMI={nmi:.4f}")
    return {"ari": ari, "nmi": nmi}


def run_lsa(
    X_bin: sp.csr_matrix,
    true_labels: np.ndarray,
    k: int,
    seed: int,
    n_components: int = 50,
) -> dict:
    log_status(seed, "Starting LSA baseline (TF-IDF + SVD)...")
    N = X_bin.shape[0]
    df = np.asarray(X_bin.sum(axis=0)).ravel()
    idf = np.log1p(N / (df + 1))
    X_tfidf = X_bin.multiply(idf).astype(np.float32)

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    Z_lsa = svd.fit_transform(X_tfidf)
    Z_lsa = normalize(Z_lsa, norm="l2")

    scores = cluster_and_score(Z_lsa, true_labels, k=k, seed=seed, label="LSA")
    return scores


# ---------------------------------------------------------------------------
# Single Seed Evaluation Task (Concurrently Executed)
# ---------------------------------------------------------------------------


def evaluate_single_seed(
    seed: int,
    adata,
    X_bin,
    true_labels: np.ndarray,
    n_types: int,
    label_col: str,
    base_ldm_dir: str,
    base_null_dir: str,
    base_out_dir: str,
    k: int,
    joint_max_peaks: int = None,
):
    seed_suffix = f"seed_{seed}_dim_16"
    ldm_dir = os.path.join(base_ldm_dir, seed_suffix)
    log_status(seed, "Initialization started.")

    z_cells_path = os.path.join(ldm_dir, "z_cells.npy")
    if not os.path.exists(z_cells_path):
        log_status(seed, f"[Warning]: Files missing at {ldm_dir}. Skipping.")
        return None

    z_cells = np.load(z_cells_path)

    if z_cells.shape[0] != adata.n_obs:
        log_status(seed, "[Error]: Row mismatch with AnnData object.")
        return None

    # 1. Compute LDM Metrics
    ldm_scores = cluster_and_score(z_cells, true_labels, k=k, seed=seed, label="LDM")

    # 2. Compute LSA Baseline Metrics
    lsa_scores = run_lsa(X_bin, true_labels, k=k, seed=seed)

    return {
        "seed": seed,
        "ldm_ari": ldm_scores["ari"],
        "ldm_nmi": ldm_scores["nmi"],
        "lsa_ari": lsa_scores["ari"],
        "lsa_nmi": lsa_scores["nmi"],
    }


# ---------------------------------------------------------------------------
# Main Execution Entrypoint
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Evaluate LDM clustering performance across multiple seeds"
    )
    p.add_argument("--data", required=True, help="Path to .h5ad file")
    p.add_argument("--ldm_dir", default="results/ldm_model")
    p.add_argument("--null_dir", default=None)
    p.add_argument("--out_dir", default="results/evaluation")
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[42], help="List of seeds to process."
    )
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--label_col", type=str, default=None)
    p.add_argument("--min_cells_pct", type=float, default=0.001)
    p.add_argument("--joint_max_peaks", type=int, default=None)
    p.add_argument(
        "--max_workers", type=int, default=4, help="Number of concurrent threads."
    )
    args = p.parse_args()

    print("Loading shared dataset...")
    adata, X_bin = load_data(args.data, min_cells_pct=args.min_cells_pct)

    label_col = args.label_col
    if label_col is None:
        for col in [
            "cell_type",
            "CellType",
            "celltype",
            "BioClassification",
            "label",
            "Group",
            "cluster",
        ]:
            if col in adata.obs.columns:
                label_col = col
                break
    if label_col is None or label_col not in adata.obs.columns:
        raise ValueError("Could not find a valid cell type column in adata.obs.")

    true_labels = np.asarray(adata.obs[label_col].values)
    n_types = len(np.unique(true_labels))
    k = n_types if args.k is None else args.k

    print(f"Dataset loaded. Cells: {adata.n_obs}, Base K-means target clusters: {k}")
    print(f"Running evaluation concurrently for seeds: {args.seeds}\n")

    results = []

    # ---- Multithreaded Evaluation Loop ---------------------------------------
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_single_seed,
                seed=seed,
                adata=adata,
                X_bin=X_bin,
                true_labels=true_labels,
                n_types=n_types,
                label_col=label_col,
                base_ldm_dir=args.ldm_dir,
                base_null_dir=args.null_dir,
                base_out_dir=args.out_dir,
                k=k,
                joint_max_peaks=args.joint_max_peaks,
            ): seed
            for seed in args.seeds
        }

        with tqdm(total=len(futures), desc="Evaluating Seeds", unit="seed") as pbar:
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    res = future.result()
                    if res is not None:
                        results.append(res)
                except Exception as exc:
                    print(f"\n[ERROR] Seed {seed} generated an exception: {exc}\n")
                pbar.update(1)

    if not results:
        print("\n[Error]: No valid results computed. Exiting.")
        return

    # Sort results by seed value for presentation consistency
    results = sorted(results, key=lambda x: x["seed"])

    # ---- 1. Print Performance Table ------------------------------------------
    print("\n" + "=" * 65)
    print(
        f"{'Seed':<10} | {'LDM ARI':<11} | {'LDM NMI':<11} | {'LSA ARI':<11} | {'LSA NMI':<11}"
    )
    print("=" * 65)
    for res in results:
        print(
            f"{res['seed']:<10} | {res['ldm_ari']:<11.4f} | {res['ldm_nmi']:<11.4f} | {res['lsa_ari']:<11.4f} | {res['lsa_nmi']:<11.4f}"
        )
    print("-" * 65)

    ldm_aris = [r["ldm_ari"] for r in results]
    ldm_nmis = [r["ldm_nmi"] for r in results]
    lsa_aris = [r["lsa_ari"] for r in results]
    lsa_nmis = [r["lsa_nmi"] for r in results]

    mean_ldm_ari, mean_ldm_nmi = np.mean(ldm_aris), np.mean(ldm_nmis)
    mean_lsa_ari, mean_lsa_nmi = np.mean(lsa_aris), np.mean(lsa_nmis)

    print(
        f"{'Mean':<10} | {mean_ldm_ari:<11.4f} | {mean_ldm_nmi:<11.4f} | {mean_lsa_ari:<11.4f} | {mean_lsa_nmi:<11.4f}"
    )
    print("=" * 65 + "\n")

    # ---- 2. Save Performance Metrics to Master JSON --------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    json_output_path = os.path.join(args.out_dir, "evaluation_summary.json")

    def calculate_ci95(data):
        n = len(data)
        if n <= 1:
            return 0.0
        # 95% Confidence Interval based on Student's t-distribution
        return float(stats.sem(data) * stats.t.ppf((1 + 0.95) / 2.0, n - 1))

    ldm_ari_ci = calculate_ci95(ldm_aris)
    ldm_nmi_ci = calculate_ci95(ldm_nmis)
    lsa_ari_ci = calculate_ci95(lsa_aris)
    lsa_nmi_ci = calculate_ci95(lsa_nmis)

    summary_data = {
        "raw_results_by_seed": results,
        "summary_statistics": {
            "ldm": {
                "ari_mean": float(mean_ldm_ari),
                "ari_ci95": ldm_ari_ci,
                "nmi_mean": float(mean_ldm_nmi),
                "nmi_ci95": ldm_nmi_ci,
            },
            "lsa": {
                "ari_mean": float(mean_lsa_ari),
                "ari_ci95": lsa_ari_ci,
                "nmi_mean": float(mean_lsa_nmi),
                "nmi_ci95": lsa_nmi_ci,
            },
        },
    }

    with open(json_output_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved performance metrics to JSON: {json_output_path}")

    # ---- 3. Single-Threaded Performance Graph Generation ---------------------
    categories = ["ARI", "NMI"]
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    ldm_means = [mean_ldm_ari, mean_ldm_nmi]
    ldm_cis = [ldm_ari_ci, ldm_nmi_ci]

    lsa_means = [mean_lsa_ari, mean_lsa_nmi]
    lsa_cis = [lsa_ari_ci, lsa_nmi_ci]

    rects1 = ax.bar(
        x - width / 2,
        ldm_means,
        width,
        yerr=ldm_cis,
        label="LDM",
        capsize=6,
        color="#1f77b4",
        edgecolor="black",
        alpha=0.85,
    )
    rects2 = ax.bar(
        x + width / 2,
        lsa_means,
        width,
        yerr=lsa_cis,
        label="LSA Baseline",
        capsize=6,
        color="#ff7f0e",
        edgecolor="black",
        alpha=0.85,
    )

    ax.set_ylabel("Score Value", fontsize=11)
    ax.set_title(
        "Clustering Performance Comparison\n(Mean with 95% Confidence Intervals)",
        fontsize=13,
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="upper right")

    # Annotate bar heights
    def label_bars(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),  # 4 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    label_bars(rects1)
    label_bars(rects2)

    plt.tight_layout()
    graph_output_path = os.path.join(args.out_dir, "performance_comparison.png")
    plt.savefig(graph_output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved performance comparison graph to: {graph_output_path}")
    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()

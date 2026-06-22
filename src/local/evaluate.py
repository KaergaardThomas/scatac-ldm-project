import argparse
import json
import os
import sys
import time  # Added for timestamped tracking
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import scipy.sparse as sp
from matplotlib import colormaps
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from tqdm import tqdm  # Added for the master progress bar
from umap import UMAP

sys.path.insert(0, os.path.dirname(__file__))
from prepare_data import load_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log_status(seed: int, message: str):
    """Helper to print standardized, timestamped updates from worker threads."""
    print(f"[{time.strftime('%H:%M:%S')}] [Seed {seed}] {message}", flush=True)


def load_history(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def final_metrics(hist: dict) -> dict:
    if isinstance(hist.get("best"), dict):
        b = hist["best"]
    else:
        b = {
            "val_bce": hist["val_bce"][-1],
            "val_auc_roc": hist["val_auc_roc"][-1],
            "val_auc_pr": hist["val_auc_pr"][-1],
            "val_f1": hist.get("val_f1", [float("nan")])[-1],
        }
    return {
        "val_bce": b["val_bce"],
        "val_auc_roc": b["val_auc_roc"],
        "val_auc_pr": b["val_auc_pr"],
        "val_f1": b.get("val_f1", float("nan")),
    }


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
    return {"ari": ari, "nmi": nmi, "pred_labels": pred_labels}


def compute_umap(
    embeddings: np.ndarray, seed: int = 42, n_components: int = 2
) -> np.ndarray:
    log_status(seed, f"Computing {n_components}D UMAP (this will take a moment)...")
    reducer = UMAP(n_components=n_components, min_dist=0.2, n_jobs=1, random_state=seed)
    coords = reducer.fit_transform(embeddings)
    log_status(seed, f"Finished {n_components}D UMAP reduction.")
    return coords


def plot_umap(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: str,
    label_name: str = "Cell type",
):
    labels = np.asarray(labels).astype(str)
    unique_labels = np.unique(labels)
    cmap = colormaps["tab20"].resampled(max(len(unique_labels), 1))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=[cmap(i)],
            s=2,
            alpha=0.5,
            label=lab,
            rasterized=True,
        )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(
        title=label_name,
        markerscale=4,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_umap_3d(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: str,
    label_name: str = "Cell type",
):
    labels = np.asarray(labels).astype(str)
    unique_labels = np.unique(labels)
    cmap = colormaps["tab20"].resampled(max(len(unique_labels), 1))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            coords[mask, 2],
            color=[cmap(i)],
            s=2,
            alpha=0.5,
            label=lab,
            rasterized=True,
        )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.legend(
        title=label_name,
        markerscale=4,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_joint_umap(
    cell_coords: np.ndarray,
    peak_coords: np.ndarray,
    cell_labels: np.ndarray,
    title: str,
    out_path: str,
    label_name: str = "Cell type",
):
    cell_labels = np.asarray(cell_labels).astype(str)
    unique_labels = np.unique(cell_labels)
    cmap = colormaps["tab20"].resampled(max(len(unique_labels), 1))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        peak_coords[:, 0],
        peak_coords[:, 1],
        color="lightgrey",
        s=1,
        alpha=0.3,
        label="Peaks",
        rasterized=True,
    )
    for i, lab in enumerate(unique_labels):
        mask = cell_labels == lab
        ax.scatter(
            cell_coords[mask, 0],
            cell_coords[mask, 1],
            color=[cmap(i)],
            s=2,
            alpha=0.6,
            label=lab,
            rasterized=True,
        )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(
        title=label_name,
        markerscale=4,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_lsa(
    X_bin: sp.csr_matrix,
    true_labels: np.ndarray,
    k: int,
    seed: int,
    out_dir: str,
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
    umap_coords = compute_umap(Z_lsa, seed=seed)

    log_status(seed, "Saving LSA plots and arrays...")
    plot_umap(
        umap_coords,
        true_labels,
        title=f"LSA / TF-IDF (Seed {seed}) — Cell type",
        out_path=os.path.join(out_dir, "umap_lsa_celltype.png"),
    )

    np.save(os.path.join(out_dir, "z_lsa.npy"), Z_lsa)
    return {
        "ari": scores["ari"],
        "nmi": scores["nmi"],
        "explained_variance_ratio": float(svd.explained_variance_ratio_.sum()),
    }


# ---------------------------------------------------------------------------
# Single Seed Evaluation Task
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
    null_dir = (
        os.path.join(base_null_dir, f"seed_{seed}_dim_16") if base_null_dir else None
    )
    out_dir = os.path.join(base_out_dir, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    log_status(seed, "Initialization started.")

    # ---- RQ1: Embedding quality ----------------------------------------------
    z_cells_path = os.path.join(ldm_dir, "z_cells.npy")
    if not os.path.exists(z_cells_path):
        log_status(seed, f"[Warning]: Files missing at {ldm_dir}. Skipping.")
        return

    z_cells = np.load(z_cells_path)

    if z_cells.shape[0] != adata.n_obs:
        log_status(seed, "[Error]: Row mismatch with AnnData object.")
        return

    # 1. K-Means
    ldm_scores = cluster_and_score(z_cells, true_labels, k=k, seed=seed, label="LDM")

    # 2. 2D UMAP
    umap_coords = compute_umap(z_cells, seed=seed)
    log_status(seed, "Generating 2D UMAP plots...")
    plot_umap(
        umap_coords,
        true_labels,
        title=f"LDM Cell Embeddings (Seed {seed}) — Cell type",
        out_path=os.path.join(out_dir, "umap_ldm_celltype.png"),
    )
    plot_umap(
        umap_coords,
        ldm_scores["pred_labels"],
        title=f"LDM Cell Embeddings (Seed {seed}) — K-means",
        out_path=os.path.join(out_dir, "umap_ldm_kmeans.png"),
        label_name="Cluster",
    )
    np.save(os.path.join(out_dir, "umap_ldm_coords.npy"), umap_coords)

    # 3. 3D UMAP
    umap_coords_3d = compute_umap(z_cells, seed=seed, n_components=3)
    log_status(seed, "Generating 3D UMAP plots...")
    plot_umap_3d(
        umap_coords_3d,
        true_labels,
        title=f"LDM Cell Embeddings 3D (Seed {seed}) — Cell type",
        out_path=os.path.join(out_dir, "umap_ldm_celltype_3d.png"),
    )
    np.save(os.path.join(out_dir, "umap_ldm_coords_3d.npy"), umap_coords_3d)

    # 4. Joint UMAP (Cells + Peaks)
    z_peaks_path = os.path.join(ldm_dir, "z_peaks.npy")
    if os.path.exists(z_peaks_path):
        z_peaks = np.load(z_peaks_path)
        if z_peaks.shape[1] == z_cells.shape[1]:
            if joint_max_peaks is not None and z_peaks.shape[0] > joint_max_peaks:
                rng = np.random.default_rng(seed)
                sel = rng.choice(z_peaks.shape[0], size=joint_max_peaks, replace=False)
                z_peaks_plot = z_peaks[sel]
            else:
                z_peaks_plot = z_peaks

            n_cells_ = z_cells.shape[0]
            joint = np.vstack([z_cells, z_peaks_plot])

            log_status(seed, "Processing Joint Cell + Peak UMAP...")
            joint_coords = compute_umap(joint, seed=seed)
            log_status(seed, "Generating Joint UMAP plot...")
            plot_joint_umap(
                cell_coords=joint_coords[:n_cells_],
                peak_coords=joint_coords[n_cells_:],
                cell_labels=true_labels,
                title=f"Joint Cell + Peak Embedding (Seed {seed})",
                out_path=os.path.join(out_dir, "umap_joint_cellpeak.png"),
            )
            np.save(os.path.join(out_dir, "umap_joint_coords.npy"), joint_coords)

    # 5. LSA Baseline
    lsa_scores = run_lsa(X_bin, true_labels, k=k, seed=seed, out_dir=out_dir)

    # ---- RQ2: Link prediction ----------------------------------------------
    log_status(seed, "Gathering link prediction metrics...")
    results = {}
    ldm_hist_path = os.path.join(ldm_dir, "history.json")
    if os.path.exists(ldm_hist_path):
        results["ldm"] = final_metrics(load_history(ldm_hist_path))

    if null_dir:
        null_hist_path = os.path.join(null_dir, "null_history.json")
        if os.path.exists(null_hist_path):
            results["null"] = final_metrics(load_history(null_hist_path))

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
    log_status(seed, "Task completed successfully. Summary saved.")


# ---------------------------------------------------------------------------
# Main Execution Entrypoint
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Evaluate LDM results across multiple seeds"
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

    # ---- Multithreaded Execution Loop with Progress Tracking ----------------
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all evaluations to the thread pool
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

        # Use tqdm to track seed level progress as threads finish
        with tqdm(total=len(futures), desc="Evaluating Seeds", unit="seed") as pbar:
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"\n[ERROR] Seed {seed} generated an exception: {exc}\n")
                pbar.update(1)

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()

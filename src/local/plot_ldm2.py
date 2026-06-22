"""
plot_ldm.py — Advanced evaluation script for Latent Distance Models (LDM).
Generates performance comparisons across multiple latent dimensions and UMAP visualizations.
"""

import argparse
import json
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from umap import UMAP

# ProcessPoolExecutor for true CPU multi-core utilization
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


# ---- 2. UMAP GENERATION ----
def generate_single_umap(base_out_data: str, dim: int):
    """Generates LDM UMAP and compares it side-by-side with original publication coordinates."""
    data_path = f"{base_out_data}_dim{dim}.h5ad"
    if not os.path.exists(data_path):
        return

    print(f"Processing UMAP for dimension {dim}...")
    adata = sc.read_h5ad(data_path)

    if "X_ldm" in adata.obsm:
        embeddings = adata.obsm["X_ldm"]
    else:
        print(f"Warning: 'X_ldm' not found in adata.obsm for dim {dim}. Skipping.")
        return

    # Compute UMAP coordinates directly using the exact setup from evaluate.py
    reducer = UMAP(n_components=2, random_state=42, min_dist=0.2)
    adata.obsm["X_umap"] = reducer.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: LDM Coordinate Space Colored by Latent Clusters
    sc.pl.umap(
        adata,
        color="clusters_ldm",
        ax=axes[0],
        show=False,
        title=f"LDM Clusters (Dim {dim})",
    )

    # Panel 2: Original Publication UMAP Coordinates
    if "UMAP1" in adata.obs.columns and "UMAP2" in adata.obs.columns:
        adata.obsm["X_original_umap"] = adata.obs[["UMAP1", "UMAP2"]].to_numpy()
        color_key = "cell_type" if "cell_type" in adata.obs.columns else "clusters_ldm"

        sc.pl.embedding(
            adata,
            basis="X_original_umap",
            color=color_key,
            ax=axes[1],
            show=False,
            title=f"Original Coordinates Space ({color_key})",
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "Original UMAP1/UMAP2 columns not found in metadata",
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

    print(
        f"Spawning plot workers with max_workers={args.threads} via ProcessPoolExecutor..."
    )
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        futures = []

        # 1. Multi-dim summary curve
        if histories:
            futures.append(executor.submit(plot_multi_dim_curves, histories))

        # 2. Map UMAP generation across dimensions
        for dim in args.latent_dim:
            futures.append(
                executor.submit(generate_single_umap, args.base_out_data, dim)
            )

        for future in futures:
            future.result()

    print(f"\nAll generated figures successfully output to target directory: {OUT_DIR}")

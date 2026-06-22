import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def analyze_depth_batch_effects(h5ad_path, ldm_dir, out_dir):
    """
    Evaluates whether sequencing depth acts as a confounding batch effect
    in the learned latent space embeddings.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load data and embeddings
    print("Loading AnnData and LDM embeddings...")
    adata = sc.read_h5ad(h5ad_path)
    z_cells = np.load(os.path.join(ldm_dir, "z_cells.npy"))

    if "depth" not in adata.obs.columns:
        raise KeyError("The metadata column 'depth' was not found in adata.obs!")

    depths = adata.obs["depth"].values

    # 2. Compute UMAP coordinates if not already saved, or reuse them
    umap_path = os.path.join(out_dir, "umap_ldm_coords.npy")
    if os.path.exists(umap_path):
        print("Reusing existing UMAP coordinates...")
        umap_coords = np.load(umap_path)
    else:
        print("Computing UMAP for depth analysis...")
        from umap import UMAP

        reducer = UMAP(n_components=2, random_state=42, min_dist=0.2)
        umap_coords = reducer.fit_transform(z_cells)

    # 3. Visual Analysis: Continuous Depth UMAP Scatter
    print("Plotting depth gradient UMAP...")
    fig, ax = plt.subplots(figsize=(8, 6))
    scat = ax.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=depths,
        cmap="viridis",
        s=2,
        alpha=0.6,
        rasterized=True,
    )
    cbar = fig.colorbar(scat, ax=ax)
    cbar.set_label("Sequencing Depth (Reads per Cell)")
    ax.set_title("LDM Latent Space Colored by Sequencing Depth")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    depth_plot_path = os.path.join(out_dir, "umap_ldm_depth_gradient.png")
    plt.tight_layout()
    plt.savefig(depth_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Quantitative Analysis: Silhouette Score based on Depth Bins
    print("Calculating Silhouette Score for depth mixability...")
    # Stratify continuous depths into 5 equal-sized frequency quantiles (batches)
    depth_bins = pd.qcut(
        depths, q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )

    # Downsample points for Silhouette computation to speed up calculation if needed
    if len(z_cells) > 10000:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(z_cells), size=10000, replace=False)
        sub_z = z_cells[indices]
        sub_bins = depth_bins[indices]
    else:
        sub_z = z_cells
        sub_bins = depth_bins

    sil_score = silhouette_score(sub_z, sub_bins)
    print(f"  >>> Depth Batch Silhouette Score: {sil_score:.4f}")
    print("  (Note: Closer to 0 is ideal, meaning low depth-induced separation)")

    # 5. Export results
    summary_metrics = {
        "depth_silhouette_score": float(sil_score),
        "mean_depth": float(np.mean(depths)),
        "median_depth": float(np.median(depths)),
    }

    with open(os.path.join(out_dir, "depth_analysis.json"), "w") as f:
        json.dump(summary_metrics, f, indent=4)

    print(f"Analysis saved to {out_dir}/depth_analysis.json and plots generated.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--ldm_dir", default="results/ldm_model/seed_42_dim_8")
    parser.add_argument("--out_dir", default="results/evaluation")
    args = parser.parse_args()

    analyze_depth_batch_effects(args.data, args.ldm_dir, args.out_dir)

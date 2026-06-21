import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.decomposition import PCA


def plot_history(history_path, eval_every, out_dir):
    """Plots Training Loss, Validation Loss, and AUC metrics."""
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_bce"]) + 1)

    # Reconstruct the epochs where evaluation happened
    eval_epochs = [1] + [e for e in epochs if e % eval_every == 0]

    # Ensure lengths match in case of interrupted training
    eval_epochs = eval_epochs[: len(history["val_bce"])]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss Curve
    axes[0].plot(
        epochs, history["train_bce"], label="Train BCE", color="blue", alpha=0.7
    )
    axes[0].plot(
        eval_epochs, history["val_bce"], label="Val BCE", marker="o", color="red"
    )
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. AUC-ROC Curve
    axes[1].plot(eval_epochs, history["val_auc_roc"], marker="o", color="green")
    axes[1].set_title("Validation AUC-ROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. AUC-PR Curve
    axes[2].plot(eval_epochs, history["val_auc_pr"], marker="o", color="purple")
    axes[2].set_title("Validation AUC-PR")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC-PR")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    metrics_plot_path = os.path.join(out_dir, "metrics_plot.png")
    plt.savefig(metrics_plot_path, dpi=300)
    print(f"Saved metrics plot to: {metrics_plot_path}")
    plt.close()


def plot_latent_space(cells_path, peaks_path, out_dir, sample_size=5000):
    """Reduces dimensionality of Latent Embeddings and plots them."""
    if not os.path.exists(cells_path) or not os.path.exists(peaks_path):
        print("Latent embedding files not found. Skipping latent space visualization.")
        return

    z_cells = np.load(cells_path)
    z_peaks = np.load(peaks_path)

    # Subsample for visualization speed if the dataset is massive
    if len(z_cells) > sample_size:
        z_cells = z_cells[
            np.random.choice(z_cells.shape[0], sample_size, replace=False)
        ]
    if len(z_peaks) > sample_size:
        z_peaks = z_peaks[
            np.random.choice(z_peaks.shape[0], sample_size, replace=False)
        ]

    print("Reducing dimensionality for visualization...")
    if HAS_UMAP:
        print("Using UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    else:
        print("UMAP not installed. Using PCA...")
        reducer = PCA(n_components=2, random_state=42)

    # Combine temporarily to map to the same 2D space
    combined_z = np.vstack([z_cells, z_peaks])
    z_2d = reducer.fit_transform(combined_z)

    # Split back
    z_cells_2d = z_2d[: len(z_cells)]
    z_peaks_2d = z_2d[len(z_cells) :]

    plt.figure(figsize=(10, 8))
    plt.scatter(
        z_cells_2d[:, 0],
        z_cells_2d[:, 1],
        s=2,
        alpha=0.5,
        label="Cells",
        color="royalblue",
    )
    plt.scatter(
        z_peaks_2d[:, 0],
        z_peaks_2d[:, 1],
        s=2,
        alpha=0.5,
        label="Peaks",
        color="darkorange",
    )

    plt.title(f"Latent Space Visualization ({'UMAP' if HAS_UMAP else 'PCA'})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # Using legend markers with larger size for visibility
    lgnd = plt.legend()
    for handle in lgnd.legend_handles:
        handle.set_sizes([30.0])

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    latent_plot_path = os.path.join(out_dir, "latent_space_plot.png")
    plt.savefig(latent_plot_path, dpi=300)
    print(f"Saved latent space plot to: {latent_plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LDM outputs")
    parser.add_argument(
        "--res_dir",
        type=str,
        default="results/ldm_run",
        help="Directory containing the training outputs",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=10,
        help="Matches the --eval_every argument used during training",
    )
    args = parser.parse_args()

    history_file = os.path.join(args.res_dir, "history.json")
    cells_file = os.path.join(args.res_dir, "z_cells.npy")
    peaks_file = os.path.join(args.res_dir, "z_peaks.npy")

    if os.path.exists(history_file):
        plot_history(history_file, args.eval_every, args.res_dir)
    else:
        print(f"Could not find history file at: {history_file}")

    plot_latent_space(cells_file, peaks_file, args.res_dir)

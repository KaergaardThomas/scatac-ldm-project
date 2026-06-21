"""
src/download_peakvi.py — High-performance, lower-VRAM training script for PeakVI.
"""

import os

# 1. Force PyTorch to manage memory allocation chunks more dynamically.
# This prevents memory fragmentation errors on large-width single-cell models.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import scanpy as sc
import scipy.sparse as sp
import scvi
import torch


def train_and_save_peakvi(h5ad_path: str, save_dir: str = "results/peakvi_model"):
    os.makedirs(save_dir, exist_ok=True)

    # Tensor Core Optimization
    torch.set_float32_matmul_precision("high")

    # 2. Load dataset
    print(f"Loading data from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)

    # 3. Handle duplicates
    if not adata.obs_names.is_unique:
        print("Making observation names unique...")
        adata.obs_names_make_unique()

    # 4. Strict CSR configuration for performance
    if not isinstance(adata.X, sp.csr_matrix):
        print("Formatting sparse matrix explicitly to CSR...")
        adata.X = sp.csr_matrix(adata.X)

    # 5. Binarize
    print("Binarizing sparse matrix indices...")
    adata.X.data = (adata.X.data > 0).astype("float32")

    # 6. Setup and Initialize PeakVI
    print("Setting up AnnData for PeakVI...")
    scvi.model.PEAKVI.setup_anndata(adata)
    model = scvi.model.PEAKVI(adata)

    # 7. Train PeakVI on a SINGLE GPU with safer VRAM batch sizing
    print("Training PeakVI on a single GPU...")
    # NOTE: Lowering the batch size to 256 shrinks the activation tensors
    # dramatically, allowing it to easily fit inside your 16GB VRAM card.
    model.train(
        max_epochs=100,
        batch_size=256,  # Slashed to clear the VRAM threshold
        accelerator="cuda",
        devices=1,
    )

    # 8. Save the model files
    print(f"Saving PeakVI model folder to {save_dir}...")
    model.save(save_dir, overwrite=True)
    print("Training and storage completed successfully!")


if __name__ == "__main__":
    train_and_save_peakvi(
        h5ad_path="data/hematopoiesis_GSE129785_FACS_sorted.h5ad",
        save_dir="results/peakvi_model",
    )

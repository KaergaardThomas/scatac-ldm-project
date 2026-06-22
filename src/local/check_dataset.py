"""
check_dataset.py — Inspects AnnData structure and metadata columns.
"""

import os
import scanpy as sc

# Target dataset path based on train_ldm.py configuration
DATA_PATH = "data/hematopoiesis_GSE129785_FACS_sorted.h5ad"


def inspect_dataset():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find dataset file at '{DATA_PATH}'")
        return

    print("Loading dataset...")
    adata = sc.read_h5ad(DATA_PATH)

    print("\n" + "=" * 40)
    print("DATASET OVERVIEW")
    print("=" * 40)
    print(f"Number of cells (Rows): {adata.n_obs}")
    print(f"Number of peaks (Columns): {adata.n_vars}")

    print("\n" + "=" * 40)
    print("AVAILABLE METADATA COLUMNS (adata.obs)")
    print("=" * 40)
    if len(adata.obs.columns) == 0:
        print("❌ WARNING: The metadata table (adata.obs) is completely empty!")
    else:
        for col in adata.obs.columns:
            print(f" - {col} (Type: {adata.obs[col].dtype})")

    print("\n" + "=" * 40)
    print("FIRST 5 ROWS OF METADATA")
    print("=" * 40)
    if len(adata.obs.columns) > 0:
        print(adata.obs.head(5))
    else:
        print("No metadata dataframes to display.")

    print("\n" + "=" * 40)
    print("UNSTRUCTURED KEYS (adata.uns)")
    print("=" * 40)
    print(list(adata.uns.keys()))


if __name__ == "__main__":
    inspect_dataset()

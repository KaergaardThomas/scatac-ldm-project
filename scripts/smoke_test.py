"""
smoke_test.py — Verify the full pipeline on synthetic data (no real data needed).

Creates a tiny synthetic scATAC-seq AnnData (500 cells × 2000 peaks, ~2% density)
and runs 5 training epochs end-to-end for both the full LDM and the null model,
checking all outputs.

Usage (from repo root):
    uv run python scripts/smoke_test.py
    uv run python scripts/smoke_test.py --epochs 20 --latent_dim 8
"""

import argparse
import os
import sys
import tempfile

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train import train
from train_null import train_null


def make_synthetic_h5ad(
    n_cells: int   = 500,
    n_peaks: int   = 2000,
    density: float = 0.02,
    seed: int      = 0,
) -> str:
    """Write a synthetic binary scATAC-seq AnnData to a temp .h5ad and return path."""
    rng = np.random.default_rng(seed)
    X = sp.random(
        n_cells, n_peaks, density=density, format="csr",
        random_state=int(rng.integers(0, 2**31)),
    ).astype(np.float32)
    X.data[:] = 1.0

    adata = anndata.AnnData(X=X)
    adata.obs = pd.DataFrame(
        {"cell_type": rng.choice(["CMP", "GMP", "MEP", "HSC", "LMPP"], size=n_cells)},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    adata.var = pd.DataFrame(
        {"chr": [f"chr{rng.integers(1, 23)}" for _ in range(n_peaks)]},
        index=[f"peak_{j}" for j in range(n_peaks)],
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
    adata.write_h5ad(tmp.name)
    print(f"Synthetic data : {n_cells} cells × {n_peaks} peaks  "
          f"(density={density:.1%})  →  {tmp.name}")
    return tmp.name


def run_smoke_test(epochs: int = 5, latent_dim: int = 4):
    print("=" * 60)
    print("SMOKE TEST — scatac-ldm pipeline")
    print("=" * 60)

    h5ad_path = make_synthetic_h5ad()
    out_dir      = os.path.join(tempfile.gettempdir(), "scatac_ldm_smoke")
    null_out_dir = os.path.join(tempfile.gettempdir(), "scatac_null_smoke")

    # ---- Full LDM ------------------------------------------------------------
    print("\n--- Full LDM ---")
    history = train(
        h5ad_path    = h5ad_path,
        latent_dim   = latent_dim,
        epochs       = epochs,
        batch_size   = 512,
        neg_ratio    = 5,
        lr           = 1e-3,
        weight_decay = 1e-4,
        seed         = 42,
        out_dir      = out_dir,
        val_frac     = 0.10,
        eval_every   = 1,
    )

    assert len(history["train_bce"])  == epochs
    assert len(history["val_auc_pr"]) == epochs
    assert all(0.0 < v < 1.5 for v in history["train_bce"]),     "LDM BCE looks wrong"
    assert all(0.0 <= v <= 1.0 for v in history["val_auc_roc"]), "LDM AUC-ROC out of range"
    assert all(0.0 <= v <= 1.0 for v in history["val_auc_pr"]),  "LDM AUC-PR out of range"

    for fname in ["best_model.pt", "final_model.pt",
                  "z_cells.npy", "z_peaks.npy", "history.json"]:
        assert os.path.exists(os.path.join(out_dir, fname)), f"Missing LDM output: {fname}"

    z_cells = np.load(os.path.join(out_dir, "z_cells.npy"))
    z_peaks = np.load(os.path.join(out_dir, "z_peaks.npy"))
    assert z_cells.shape[1] == latent_dim, "Cell embedding dim mismatch"
    assert z_peaks.shape[1] == latent_dim, "Peak embedding dim mismatch"

    # ---- Null model ----------------------------------------------------------
    print("\n--- Null Model ---")
    null_history = train_null(
        h5ad_path  = h5ad_path,
        epochs     = epochs,
        batch_size = 512,
        neg_ratio  = 5,
        lr         = 1e-3,
        seed       = 42,
        out_dir    = null_out_dir,
        val_frac   = 0.10,
        eval_every = 1,
    )

    assert len(null_history["train_bce"])  == epochs
    assert len(null_history["val_auc_pr"]) == epochs
    assert all(0.0 < v < 1.5 for v in null_history["train_bce"]),     "Null BCE looks wrong"
    assert all(0.0 <= v <= 1.0 for v in null_history["val_auc_pr"]),  "Null AUC-PR out of range"

    for fname in ["best_null_model.pt", "final_null_model.pt", "null_history.json"]:
        assert os.path.exists(os.path.join(null_out_dir, fname)), f"Missing null output: {fname}"

    # ---- Summary -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("✓  ALL CHECKS PASSED")
    print(f"\n  Full LDM:")
    print(f"   Final train BCE : {history['train_bce'][-1]:.4f}")
    print(f"   Final AUC-ROC   : {history['val_auc_roc'][-1]:.4f}")
    print(f"   Final AUC-PR    : {history['val_auc_pr'][-1]:.4f}")
    print(f"   z_cells shape   : {z_cells.shape}")
    print(f"   z_peaks shape   : {z_peaks.shape}")
    print(f"\n  Null Model:")
    print(f"   Final train BCE : {null_history['train_bce'][-1]:.4f}")
    print(f"   Final AUC-ROC   : {null_history['val_auc_roc'][-1]:.4f}")
    print(f"   Final AUC-PR    : {null_history['val_auc_pr'][-1]:.4f}")
    print("=" * 60)

    os.unlink(h5ad_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int, default=5)
    p.add_argument("--latent_dim", type=int, default=4)
    args = p.parse_args()
    run_smoke_test(epochs=args.epochs, latent_dim=args.latent_dim)

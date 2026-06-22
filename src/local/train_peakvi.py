"""
train_peakvi.py — Training script using scvi-tools PEAKVI.
"""

import argparse
from importlib.metadata import entry_points
import json
import os
import threading
import scanpy as sc
import scipy.sparse as sp
import scvi
import numpy as np
import anndata
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

# Handle lightning import depending on scvi-tools version
try:
    from lightning.pytorch import Callback
except ImportError:
    from pytorch_lightning import Callback


class EvalMetricsCallback(Callback):
    def __init__(
        self,
        model,
        adata,
        labels_key=None,
        resolution=0.2,
        eval_every=1,
        sample_size=1000,
    ):
        super().__init__()
        self.model = model
        self.labels_key = labels_key
        self.resolution = resolution
        self.eval_every = eval_every
        self.sample_size = min(sample_size, adata.n_obs)

        self.eval_epochs = []
        self.val_auc_roc = []
        self.val_auc_pr = []
        self.val_f1 = []
        self.val_ari = []
        self.val_nmi = []

        # Keep track of the background evaluation thread
        self.eval_thread = None

        # OPTIMIZATION 1: Create a static validation set ONCE
        print(f"Creating static validation subset of {self.sample_size} cells...")
        idx = np.random.choice(adata.n_obs, self.sample_size, replace=False)
        self.adata_sub = adata[idx].copy()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        total_batches = trainer.num_training_batches
        # Smooth live-update on the same line
        print(
            f"Epoch {trainer.current_epoch + 1} | Batch {batch_idx + 1}/{total_batches}",
            end="\r",
            flush=True,
        )

    def _async_metrics_worker(
        self, epoch, y_true_eval, y_probs_eval, latent_array, labels_true
    ):
        """Runs on a background thread to prevent sklearn/scanpy from blocking training."""
        # 1. Link Prediction Metrics (AUC-ROC, AUC-PR, F1)
        self.val_auc_roc.append(float(roc_auc_score(y_true_eval, y_probs_eval)))
        self.val_auc_pr.append(
            float(average_precision_score(y_true_eval, y_probs_eval))
        )

        y_pred_eval = (y_probs_eval > 0.5).astype(int)
        self.val_f1.append(float(f1_score(y_true_eval, y_pred_eval)))

        # 2. Clustering Metrics (ARI, NMI)
        if latent_array is not None and labels_true is not None:
            # Create a lightweight temporary AnnData to avoid mutating the main one across threads
            temp_adata = anndata.AnnData(X=np.empty((latent_array.shape[0], 1)))
            temp_adata.obsm["X_peakvi"] = latent_array

            sc.pp.neighbors(temp_adata, use_rep="X_peakvi")
            try:
                sc.tl.leiden(
                    temp_adata,
                    resolution=self.resolution,
                    flavor="igraph",
                    n_iterations=2,
                    directed=False,
                )
            except Exception:
                sc.tl.leiden(temp_adata, resolution=self.resolution)

            labels_pred = temp_adata.obs["leiden"]
            self.val_ari.append(float(adjusted_rand_score(labels_true, labels_pred)))
            self.val_nmi.append(
                float(normalized_mutual_info_score(labels_true, labels_pred))
            )
        print(f"Epoch {epoch} metrics done")

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.eval_every != 0:
            return

        self.eval_epochs.append(epoch)

        # --- SYNCHRONOUS GPU OPERATIONS ---
        # We must grab the predictions while the main thread holds the model state
        probs = self.model.get_normalized_accessibility(self.adata_sub)

        y_true = (
            self.adata_sub.X.toarray().flatten()
            if sp.issparse(self.adata_sub.X)
            else self.adata_sub.X.flatten()
        )
        y_true = (y_true > 0).astype(int)
        y_probs = np.asarray(probs).flatten()

        MAX_EVAL_POINTS = 500_000
        if len(y_true) > MAX_EVAL_POINTS:
            mask = np.random.choice(len(y_true), MAX_EVAL_POINTS, replace=False)
            y_true_eval = y_true[mask]
            y_probs_eval = y_probs[mask]
        else:
            y_true_eval = y_true
            y_probs_eval = y_probs

        latent_array = None
        labels_true = None
        if self.labels_key and self.labels_key in self.adata_sub.obs:
            latent_array = self.model.get_latent_representation(self.adata_sub)
            labels_true = self.adata_sub.obs[self.labels_key].values

        # --- ASYNCHRONOUS CPU THREADING ---
        # Ensure the previous evaluation thread has finished before starting a new one
        if self.eval_thread is not None and self.eval_thread.is_alive():
            self.eval_thread.join()

        # Spin off the sklearn/scanpy heavy lifting to a background thread
        self.eval_thread = threading.Thread(
            target=self._async_metrics_worker,
            args=(epoch, y_true_eval, y_probs_eval, latent_array, labels_true),
        )
        self.eval_thread.start()

    def on_train_end(self, trainer, pl_module):
        """Ensure the final thread finishes before the script attempts to save the history JSON."""
        if self.eval_thread is not None and self.eval_thread.is_alive():
            print("\nWaiting for the final background evaluation thread to complete...")
            self.eval_thread.join()


def run_train_pipeline(
    data_path: str,
    out_data_path: str,
    model_dir: str,
    accelerator: str,
    devices: int,
    resolution: float,
    min_cells_fraction: float,
    labels_key: str,
    epochs: int,
    batch_size: int,
):
    out_data_dir = os.path.dirname(os.path.abspath(out_data_path))
    if out_data_dir:
        os.makedirs(out_data_dir, exist_ok=True)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    print("Loading data...")
    try:
        adata = sc.read_h5ad(data_path)
    except Exception as e:
        print(f"File not found locally. Attempting to download to {data_path}...")
        adata = sc.read(
            data_path, backup_url="https://figshare.com/ndownloader/files/52859288"
        )

    adata.obs_names_make_unique()
    print(f"Loaded raw shape: {adata.shape}")

    if not sp.isspmatrix_csr(adata.X):
        print("Converting sparse matrix to CSR format to save memory...")
        adata.X = sp.csr_matrix(adata.X)

    min_cells = int(adata.n_obs * min_cells_fraction)
    print(
        f"Filtering peaks present in fewer than {min_cells} cells ({min_cells_fraction * 100}%)..."
    )
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"Data shape after filtering: {adata.shape}")

    print("\nSetting up AnnData for PEAKVI...")
    scvi.model.PEAKVI.setup_anndata(adata)

    print(
        f"Initializing PEAKVI model and training (accelerator='{accelerator}', devices={devices})..."
    )
    model = scvi.model.PEAKVI(adata)

    strategy = (
        "ddp" if (accelerator in ["cuda", "gpu", "auto"] and devices > 1) else "auto"
    )

    # --- Setup Custom Callback ---
    eval_callback = EvalMetricsCallback(
        model=model,
        adata=adata,
        labels_key=labels_key,
        resolution=resolution,
        eval_every=1,  # Evaluate every epoch
    )

    model.train(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        batch_size=batch_size,
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        callbacks=[eval_callback],
    )

    print(f"\nSaving model to {model_dir}...")
    model.save(model_dir, overwrite=True)

    # --- Save History JSON ---
    def get_hist(key):
        if key in model.history and not model.history[key].empty:
            return model.history[key][key].tolist()
        return []

    history = {
        "train_bce": get_hist("reconstruction_loss_train"),
        "val_bce": get_hist("reconstruction_loss_validation"),
        "eval_epochs": eval_callback.eval_epochs,
        "val_auc_roc": eval_callback.val_auc_roc,
        "val_auc_pr": eval_callback.val_auc_pr,
        "val_f1": eval_callback.val_f1,
    }

    if eval_callback.val_ari:
        history["val_ari"] = eval_callback.val_ari
    if eval_callback.val_nmi:
        history["val_nmi"] = eval_callback.val_nmi

    hist_path = os.path.join(model_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Saved evaluation metrics to {hist_path}")

    # --- Latent & Clustering ---
    print("\nExtracting latent representation...")
    PEAKVI_LATENT_KEY = "X_peakvi"
    latent = model.get_latent_representation()
    adata.obsm[PEAKVI_LATENT_KEY] = latent

    print(f"\nComputing k-nearest-neighbor graph (use_rep='{PEAKVI_LATENT_KEY}')...")
    sc.pp.neighbors(adata, use_rep=PEAKVI_LATENT_KEY)
    PEAKVI_CLUSTERS_KEY = "clusters_peakvi"

    try:
        sc.tl.leiden(
            adata,
            key_added=PEAKVI_CLUSTERS_KEY,
            resolution=resolution,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    except Exception as e:
        sc.tl.leiden(adata, key_added=PEAKVI_CLUSTERS_KEY, resolution=resolution)

    print(f"\nWriting updated AnnData object to {out_data_path}...")
    adata.write_h5ad(out_data_path)
    print("Training pipeline complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train PEAKVI and generate cluster labels")
    p.add_argument("--data", type=str, default="data/lung_atlas_preprocessed.h5ad")
    p.add_argument(
        "--out_data", type=str, default="data/lung_atlas_preprocessed_with_peakvi.h5ad"
    )
    p.add_argument("--model_dir", type=str, default="results/peakvi_model")
    p.add_argument(
        "--accelerator",
        type=str,
        default="cuda",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    p.add_argument("--devices", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--resolution", type=float, default=0.2)
    p.add_argument("--min_cells_fraction", type=float, default=0.01)

    p.add_argument(
        "--labels_key",
        type=str,
        default=None,
        help="Column in adata.obs for ground truth labels (e.g., cell_type)",
    )
    args = p.parse_args()

    run_train_pipeline(
        data_path=args.data,
        out_data_path=args.out_data,
        model_dir=args.model_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        resolution=args.resolution,
        min_cells_fraction=args.min_cells_fraction,
        labels_key=args.labels_key,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

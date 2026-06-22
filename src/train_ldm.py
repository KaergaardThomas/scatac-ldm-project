import argparse
import json
import os
import time
import datetime
import scanpy as sc
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecisionRecallCurve,
)

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback

from src.ldm import LDM


# ---------------------------------------------------------------------------
# Fixed Evaluation Set Generators (Ported from train.py)
# ---------------------------------------------------------------------------


def edges_to_keys(edges: np.ndarray, n_peaks: int) -> np.ndarray:
    return edges[:, 0].astype(np.int64) * np.int64(n_peaks) + edges[:, 1].astype(
        np.int64
    )


def _isin_sorted(keys: np.ndarray, sorted_arr: np.ndarray) -> np.ndarray:
    if sorted_arr.size == 0:
        return np.zeros(keys.shape, dtype=bool)
    idx = np.searchsorted(sorted_arr, keys)
    idx = np.clip(idx, 0, sorted_arr.size - 1)
    return sorted_arr[idx] == keys


def sample_negatives(
    pos_edges: np.ndarray,
    n_cells: int,
    n_peaks: int,
    neg_ratio: int,
    rng: np.random.Generator,
) -> np.ndarray:
    N = len(pos_edges)
    total = N * neg_ratio
    repeated = np.tile(pos_edges, (neg_ratio, 1))

    corrupt_side = rng.integers(0, 2, size=total)
    random_cells = rng.integers(0, n_cells, size=total)
    random_peaks = rng.integers(0, n_peaks, size=total)

    neg = repeated.copy()
    cell_mask = corrupt_side == 0
    neg[cell_mask, 0] = random_cells[cell_mask]
    neg[~cell_mask, 1] = random_peaks[~cell_mask]
    return neg


import numpy as np
import scipy.sparse as sp
from tqdm.auto import tqdm  # Add this import at the top of your file


def build_eval_set(
    val_edges: np.ndarray,
    n_cells: int,
    n_peaks: int,
    neg_ratio: int,
    seed: int,
    observed_csr: sp.csr_matrix = None,
    eval_max_pos: int = 1_000_000,
):
    rng = np.random.default_rng(seed + 12345)

    pos = val_edges
    if eval_max_pos and len(pos) > eval_max_pos:
        sel = rng.choice(len(pos), size=eval_max_pos, replace=False)
        pos = pos[sel]

    # 1. Sample initial negatives
    neg = sample_negatives(pos, n_cells, n_peaks, neg_ratio, rng)

    if observed_csr is not None:
        # Fast O(1) collision checking using SciPy sparse matrix indexing
        collisions = np.asarray(observed_csr[neg[:, 0], neg[:, 1]]).flatten()
        bad = collisions == True
        n_bad = int(bad.sum())

        # Initialize the loading bar based on total negatives to process
        total_neg = len(neg)
        pbar = tqdm(total=total_neg, desc="Filtering Collisions", unit=" edges")

        # Immediately update the bar with the edges that were valid on the first try
        pbar.update(total_neg - n_bad)

        tries = 0
        while n_bad > 0 and tries < 10:
            neg[bad, 0] = rng.integers(0, n_cells, size=n_bad)
            neg[bad, 1] = rng.integers(0, n_peaks, size=n_bad)

            # Re-check the whole array for simplicity
            collisions = np.asarray(observed_csr[neg[:, 0], neg[:, 1]]).flatten()
            bad = collisions == True
            new_n_bad = int(bad.sum())

            # Advance the bar by the amount of collisions we successfully resolved
            resolved_this_round = n_bad - new_n_bad
            if resolved_this_round > 0:
                pbar.update(resolved_this_round)

            n_bad = new_n_bad
            tries += 1

        pbar.close()  # Clean up the bar when done

        if n_bad > 0:
            print(f"Warning: {n_bad} collisions could not be resolved after 10 tries.")

    cell = np.concatenate([pos[:, 0], neg[:, 0]]).astype(np.int64)
    peak = np.concatenate([pos[:, 1], neg[:, 1]]).astype(np.int64)
    label = np.concatenate(
        [
            np.ones(len(pos), dtype=np.float32),
            np.zeros(len(neg), dtype=np.float32),
        ]
    )

    return cell, peak, label


# ---------------------------------------------------------------------------
# Training Dataset (Preserved from train_ldm.py)
# ---------------------------------------------------------------------------


class BipartiteEdgeDataset(Dataset):
    def __init__(self, cells, peaks, batch_size):
        super().__init__()
        self.cells = torch.tensor(cells, dtype=torch.long)
        self.peaks = torch.tensor(peaks, dtype=torch.long)

        perm = torch.randperm(len(self.cells))
        self.cells = self.cells[perm]
        self.peaks = self.peaks[perm]

        self.num_edges = len(self.cells)
        self.batch_size = batch_size

    def __len__(self):
        return (self.num_edges + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_edges)
        return self.cells[start_idx:end_idx], self.peaks[start_idx:end_idx]


# ---------------------------------------------------------------------------
# PyTorch Lightning Module
# ---------------------------------------------------------------------------


class LightningLDM(pl.LightningModule):
    def __init__(self, n_cells, n_peaks, latent_dim=8, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = LDM(n_cells=n_cells, n_peaks=n_peaks, latent_dim=latent_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_peaks = n_peaks

        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.val_pr_curve = BinaryPrecisionRecallCurve()
        self.train_start_time = None

    def on_train_start(self):
        self.train_start_time = time.time()

    def forward(self, cell_idx, peak_idx):
        return self.model(cell_idx, peak_idx)

    def training_step(self, batch, batch_idx):
        # Preserved exactly as train_ldm.py: 1:1 on-the-fly peak-side corruption
        pos_c, pos_p = batch
        neg_c = pos_c
        neg_p = torch.randint(0, self.n_peaks, size=pos_p.shape, device=self.device)

        pos_logits = self(pos_c, pos_p)
        neg_logits = self(neg_c, neg_p)

        loss = self.loss_fn(pos_logits, torch.ones_like(pos_logits)) + self.loss_fn(
            neg_logits, torch.zeros_like(neg_logits)
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero and self.train_start_time is not None:
            epochs_completed = self.current_epoch + 1
            elapsed_time = time.time() - self.train_start_time
            avg_time_per_epoch = elapsed_time / epochs_completed

            remaining_epochs = self.trainer.max_epochs - epochs_completed
            eta_seconds = remaining_epochs * avg_time_per_epoch
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

            print(
                f"\n[Epoch {epochs_completed}/{self.trainer.max_epochs}] Done. ETA: {eta_str}"
            )

    def validation_step(self, batch, batch_idx):
        # Updated to process the fixed validation batch from train.py
        c, p, y = batch

        logits = self(c, p)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())

        self.log("val_bce", loss, sync_dist=True)

        probs = torch.sigmoid(logits)
        self.val_auroc.update(probs, y.long())
        self.val_auprc.update(probs, y.long())
        self.val_pr_curve.update(probs, y.long())

        return loss

    def on_validation_epoch_end(self):
        auroc = self.val_auroc.compute()
        auprc = self.val_auprc.compute()
        precision, recall, _ = self.val_pr_curve.compute()

        f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
        best_f1 = torch.max(f1_scores)

        self.log("val_auc_roc", auroc, sync_dist=True)
        self.log("val_auc_pr", auprc, sync_dist=True)
        self.log("val_f1", best_f1, sync_dist=True)

        if self.trainer.is_global_zero:
            print(
                f"--- Val metrics -> ROC: {auroc:.4f} | PR: {auprc:.4f} | F1: {best_f1:.4f} ---"
            )

        self.val_auroc.reset()
        self.val_auprc.reset()
        self.val_pr_curve.reset()

    def configure_optimizers(self):
        # Updated to use train.py's selective weight decay
        embed_params = [self.model.z_i.weight, self.model.z_j.weight]
        bias_params = [
            p
            for n, p in self.model.named_parameters()
            if n not in ("z_i.weight", "z_j.weight")
        ]

        return torch.optim.Adam(
            [
                {"params": embed_params, "weight_decay": self.weight_decay},
                {"params": bias_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )


class MetricHistoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.history = {
            "train_loss": [],
            "val_bce": [],
            "val_auc_roc": [],
            "val_auc_pr": [],
            "val_f1": [],
            "eval_epochs": [],
        }

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.history["train_loss"].append(metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_bce" not in metrics:
            return

        self.history["eval_epochs"].append(trainer.current_epoch + 1)
        self.history["val_bce"].append(metrics["val_bce"].item())

        if "val_auc_roc" in metrics:
            self.history["val_auc_roc"].append(metrics["val_auc_roc"].item())
        if "val_auc_pr" in metrics:
            self.history["val_auc_pr"].append(metrics["val_auc_pr"].item())
        if "val_f1" in metrics:
            self.history["val_f1"].append(metrics["val_f1"].item())


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
class FixedEvalDataset(Dataset):
    def __init__(self, c, p, y, batch_size):
        super().__init__()
        self.c = torch.as_tensor(c, dtype=torch.long)
        self.p = torch.as_tensor(p, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.batch_size = batch_size
        self.num_edges = len(self.c)

    def __len__(self):
        return (self.num_edges + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        # Native slicing avoids PyTorch's heavy collate overhead
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_edges)
        return (
            self.c[start_idx:end_idx],
            self.p[start_idx:end_idx],
            self.y[start_idx:end_idx],
        )


def run_train_pipeline(
    data_path,
    out_data_path,
    model_dir,
    accelerator,
    device_list,
    resolution,
    min_cells_fraction,
    epochs,
    batch_size,
    latent_dim,
    weight_decay,
    val_split,
    check_val_every_n_epoch,
    seed,
):
    os.makedirs(os.path.dirname(os.path.abspath(out_data_path)), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f">> Starting: seed={seed}, latent_dim={latent_dim}")
    print("loading h5ad data...")
    adata = sc.read_h5ad(data_path)
    adata.obs_names_make_unique()

    min_cells = int(adata.n_obs * min_cells_fraction)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    n_cells, n_peaks = adata.shape

    X_coo = (
        adata.X.tocoo() if sp.isspmatrix(adata.X) else sp.csr_matrix(adata.X).tocoo()
    )
    cells, peaks = X_coo.row, X_coo.col

    # 1. Edge Split Logic (Replacing cell isolation split)
    num_edges = len(cells)
    perm = np.random.permutation(num_edges)
    n_val = int(num_edges * val_split)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_cells, train_peaks = cells[train_idx], peaks[train_idx]
    val_cells, val_peaks = cells[val_idx], peaks[val_idx]

    train_edges = np.column_stack((train_cells, train_peaks))
    val_edges = np.column_stack((val_cells, val_peaks))

    print(f"split stats: train_edges={len(train_edges)}, val_edges={len(val_edges)}")
    import gc  # Ensure gc is imported at the top of your file

    # ---------------------------------------------------------
    # The Block You Shared Begins Here
    # ---------------------------------------------------------

    # 2. Build Fixed Eval Set (Optimized)
    all_pos = np.vstack([train_edges, val_edges])

    # Create a sparse matrix instead of unique int64 keys
    data = np.ones(len(all_pos), dtype=bool)
    observed_csr = sp.csr_matrix(
        (data, (all_pos[:, 0], all_pos[:, 1])), shape=(n_cells, n_peaks)
    )

    val_c, val_p, val_y = build_eval_set(
        val_edges,
        n_cells,
        n_peaks,
        neg_ratio=10,
        seed=seed,
        observed_csr=observed_csr,
    )

    print(f"built eval set")

    # Aggressively free up System RAM before PyTorch takes over
    del all_pos
    del observed_csr
    del train_edges
    del val_edges
    gc.collect()

    # 3. Dataloaders Setup
    train_dataset = BipartiteEdgeDataset(
        train_cells, train_peaks, batch_size=batch_size
    )

    # Use our custom native batching dataset instead of TensorDataset
    val_dataset = FixedEvalDataset(val_c, val_p, val_y, batch_size=batch_size)
    print(f"build 2")

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,  # None because dataset returns batches natively
        shuffle=True,
        num_workers=0,
        pin_memory=True if accelerator in ["cuda", "gpu"] else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=None,  # MUST be None to bypass the RAM-heavy collate function
        shuffle=False,
        num_workers=0,
        pin_memory=True if accelerator in ["cuda", "gpu"] else False,
    )

    lightning_model = LightningLDM(
        n_cells=n_cells,
        n_peaks=n_peaks,
        latent_dim=latent_dim,
        weight_decay=weight_decay,
    )
    eval_callback = MetricHistoryCallback()

    strategy = (
        "ddp" if (accelerator in ["cuda", "gpu"] and len(device_list) > 1) else "auto"
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=device_list,
        strategy=strategy,
        max_epochs=epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[eval_callback],
        enable_checkpointing=False,
        logger=False,
    )

    print("training...")
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    if trainer.is_global_zero:
        print(f"saving weights to {model_dir}")
        torch.save(
            lightning_model.model.state_dict(),
            os.path.join(model_dir, "ldm_weights.pt"),
        )

        hist_path = os.path.join(model_dir, "history.json")
        with open(hist_path, "w") as f:
            json.dump(eval_callback.history, f, indent=4)

        print("extracting cell embeddings...")
        with torch.no_grad():
            all_cells = torch.arange(n_cells)
            if lightning_model.device.type != "cpu":
                all_cells = all_cells.to(lightning_model.device)
            latent = lightning_model.model.z_i(all_cells).cpu().numpy()

        np.save(os.path.join(model_dir, "z_cells.npy"), latent)
        adata.obsm["X_ldm"] = latent

        print("running clustering...")
        sc.pp.neighbors(adata, use_rep="X_ldm")
        try:
            sc.tl.leiden(
                adata,
                key_added="clusters_ldm",
                resolution=resolution,
                flavor="igraph",
                n_iterations=2,
                directed=False,
            )
        except Exception:
            sc.tl.leiden(adata, key_added="clusters_ldm", resolution=resolution)

        print(f"saving updated anndata -> {out_data_path}")
        adata.write_h5ad(out_data_path)
        print("done.\n")

    if trainer.strategy.launcher is not None or len(device_list) > 1:
        trainer.strategy.barrier("pipeline_cleanup")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--accelerator",
        type=str,
        default="cuda",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    p.add_argument("--device", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument(
        "--weight_decay", type=float, default=1e-4
    )  # Added to align with train.py
    p.add_argument("--resolution", type=float, default=0.2)
    p.add_argument("--min_cells_fraction", type=float, default=0.01)
    p.add_argument("--latent_dim", type=int, nargs="+", default=[8])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--check_val_every_n_epoch", type=int, default=1)
    args = p.parse_args()

    BASE_DATA_PATH = "data/hematopoiesis_GSE129785_FACS_sorted.h5ad"
    BASE_OUT_DATA = "data/hematopoiesis_with_ldm"
    BASE_MODEL_DIR = "results/ldm_model"

    for seed in args.seeds:
        pl.seed_everything(seed, workers=True)

        for dim in args.latent_dim:
            current_out_data = f"{BASE_OUT_DATA}_seed{seed}_dim{dim}.h5ad"
            current_model_dir = os.path.join(BASE_MODEL_DIR, f"seed_{seed}_dim_{dim}")

            run_train_pipeline(
                data_path=BASE_DATA_PATH,
                out_data_path=current_out_data,
                model_dir=current_model_dir,
                accelerator=args.accelerator,
                device_list=args.device,
                resolution=args.resolution,
                min_cells_fraction=args.min_cells_fraction,
                epochs=args.epochs,
                batch_size=args.batch_size,
                latent_dim=dim,
                weight_decay=args.weight_decay,
                val_split=args.val_split,
                check_val_every_n_epoch=args.check_val_every_n_epoch,
                seed=seed,
            )

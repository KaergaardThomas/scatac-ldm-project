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
from torch.utils.data import Dataset, DataLoader

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


class LightningLDM(pl.LightningModule):
    def __init__(self, n_cells, n_peaks, latent_dim=8, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = LDM(n_cells=n_cells, n_peaks=n_peaks, latent_dim=latent_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr
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
        pos_c, pos_p = batch
        neg_c = pos_c
        neg_p = torch.randint(0, self.n_peaks, size=pos_p.shape, device=self.device)

        pos_logits = self(pos_c, pos_p)
        neg_logits = self(neg_c, neg_p)

        loss = self.loss_fn(pos_logits, torch.ones_like(pos_logits)) + self.loss_fn(
            neg_logits, torch.zeros_like(neg_logits)
        )
        self.log("val_bce", loss, sync_dist=True)

        pos_probs = torch.sigmoid(pos_logits)
        neg_probs = torch.sigmoid(neg_logits)

        preds = torch.cat([pos_probs, neg_probs])
        targets = torch.cat(
            [torch.ones_like(pos_probs), torch.zeros_like(neg_probs)]
        ).long()

        self.val_auroc.update(preds, targets)
        self.val_auprc.update(preds, targets)
        self.val_pr_curve.update(preds, targets)

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
        return torch.optim.Adam(self.parameters(), lr=self.lr)


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

    total_cells = adata.n_obs
    cell_perm = np.random.permutation(total_cells)
    n_val_cells = int(total_cells * val_split)
    val_cell_ids = set(cell_perm[:n_val_cells])

    val_mask = np.array([c in val_cell_ids for c in cells])
    train_mask = ~val_mask

    train_cells, train_peaks = cells[train_mask], peaks[train_mask]
    val_cells, val_peaks = cells[val_mask], peaks[val_mask]

    print(f"split stats: train_edges={len(train_cells)}, val_edges={len(val_cells)}")

    train_dataset = BipartiteEdgeDataset(
        train_cells, train_peaks, batch_size=batch_size
    )
    val_dataset = BipartiteEdgeDataset(val_cells, val_peaks, batch_size=batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=0,
        pin_memory=True if accelerator in ["cuda", "gpu"] else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        pin_memory=True if accelerator in ["cuda", "gpu"] else False,
    )

    lightning_model = LightningLDM(
        n_cells=n_cells, n_peaks=n_peaks, latent_dim=latent_dim
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
    p.add_argument("--batch_size", type=int, default=2048)
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
                val_split=args.val_split,
                check_val_every_n_epoch=args.check_val_every_n_epoch,
                seed=seed,
            )

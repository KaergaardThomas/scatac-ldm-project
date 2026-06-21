"""
train.py — Training loop for the Latent Distance Model.

Edge split contract
-------------------
- train_edges : positive (cell, peak) pairs used for optimisation.
- val_edges   : held-out positives used only for evaluation.
- Negatives are sampled fresh each batch directly on the GPU.

Usage
-----
    uv run python src/train.py --data data/hematopoiesis.h5ad
"""

import argparse
import json
import os
import random
import sys
import time

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

# Allow running from repo root or from src/
sys.path.insert(0, os.path.dirname(__file__))
from ldm import LDM


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Extraction Utilities
# ---------------------------------------------------------------------------


def extract_edges_from_chunk(chunk, row_offset):
    """Safely extracts (row, col) coordinates from a generic sparse/dense chunk."""
    if hasattr(chunk, "tocoo"):
        chunk_coo = chunk.tocoo()
        rows, cols = chunk_coo.row + row_offset, chunk_coo.col
    elif hasattr(chunk, "toarray"):
        rows, cols = np.nonzero(chunk.toarray())
        rows += row_offset
    else:
        rows, cols = np.nonzero(chunk)
        rows += row_offset
    return rows, cols


def load_validation_edges(adata, start_cell, end_cell, chunk_size=5000):
    """Loads only the validation subset into RAM."""
    print(f"Extracting validation edges from cells {start_cell} to {end_cell}...")
    val_edges = []
    for i in tqdm(range(start_cell, end_cell, chunk_size), desc="Val Chunks"):
        chunk_end = min(i + chunk_size, end_cell)
        chunk = adata.X[i:chunk_end]
        rows, cols = extract_edges_from_chunk(chunk, i)
        if len(rows) > 0:
            val_edges.append(np.column_stack((rows, cols)))

    if len(val_edges) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.vstack(val_edges)


# ---------------------------------------------------------------------------
# Disk-to-GPU Streaming Generator
# ---------------------------------------------------------------------------


def stream_train_epochs(
    adata,
    n_train_cells: int,
    n_cells: int,
    n_peaks: int,
    neg_ratio: int,
    batch_size: int,
    device: torch.device,
    disk_chunk: int = 10000,
):
    """
    Streams data straight from disk to VRAM.
    System RAM never holds more than `disk_chunk` cells at a time.
    """
    for chunk_start in range(0, n_train_cells, disk_chunk):
        chunk_end = min(chunk_start + disk_chunk, n_train_cells)

        # 1. Read small chunk from disk (RAM usage goes up slightly)
        chunk = adata.X[chunk_start:chunk_end]
        rows, cols = extract_edges_from_chunk(chunk, chunk_start)

        if len(rows) == 0:
            continue

        chunk_edges = np.column_stack((rows, cols))

        # Local shuffle to ensure mixed batches
        np.random.shuffle(chunk_edges)

        # 2. Move directly to GPU
        gpu_pos_edges = torch.from_numpy(chunk_edges).long().to(device)
        n_edges_in_chunk = gpu_pos_edges.size(0)

        # 3. Process mini-batches on the GPU
        for i in range(0, n_edges_in_chunk, batch_size):
            pos_batch = gpu_pos_edges[i : i + batch_size]
            B = pos_batch.size(0)
            total_neg = B * neg_ratio

            # Replicate positives
            neg_edges = pos_batch.repeat(neg_ratio, 1)

            # GPU-accelerated corruption coin-flip
            corrupt_side = torch.randint(
                0, 2, (total_neg,), device=device, dtype=torch.bool
            )
            random_cells = torch.randint(0, n_cells, (total_neg,), device=device)
            random_peaks = torch.randint(0, n_peaks, (total_neg,), device=device)

            neg_edges[corrupt_side, 0] = random_cells[corrupt_side]
            neg_edges[~corrupt_side, 1] = random_peaks[~corrupt_side]

            # Concatenate pos and neg
            batch_edges = torch.cat([pos_batch, neg_edges], dim=0)
            batch_labels = torch.cat(
                [
                    torch.ones(B, dtype=torch.float32, device=device),
                    torch.zeros(total_neg, dtype=torch.float32, device=device),
                ],
                dim=0,
            )

            # Local shuffle to mix pos and neg
            batch_perm = torch.randperm(batch_edges.size(0), device=device)
            yield (
                batch_edges[batch_perm, 0],
                batch_edges[batch_perm, 1],
                batch_labels[batch_perm],
            )

        # 4. RAM / VRAM is automatically freed when the loop moves to the next chunk


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: LDM,
    val_edges: np.ndarray,
    n_cells: int,
    n_peaks: int,
    neg_ratio: int,
    device: torch.device,
    batch_size: int = 8192,
) -> dict:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_logits, all_true = [], []
    total_loss, n_batches = 0.0, 0

    gpu_val_edges = torch.from_numpy(val_edges).long().to(device)

    for i in range(0, gpu_val_edges.size(0), batch_size):
        pos_batch = gpu_val_edges[i : i + batch_size]
        B = pos_batch.size(0)
        total_neg = B * neg_ratio

        neg_edges = pos_batch.repeat(neg_ratio, 1)
        corrupt_side = torch.randint(
            0, 2, (total_neg,), device=device, dtype=torch.bool
        )
        random_cells = torch.randint(0, n_cells, (total_neg,), device=device)
        random_peaks = torch.randint(0, n_peaks, (total_neg,), device=device)

        neg_edges[corrupt_side, 0] = random_cells[corrupt_side]
        neg_edges[~corrupt_side, 1] = random_peaks[~corrupt_side]

        batch_edges = torch.cat([pos_batch, neg_edges], dim=0)
        y = torch.cat(
            [
                torch.ones(B, dtype=torch.float32, device=device),
                torch.zeros(total_neg, dtype=torch.float32, device=device),
            ],
            dim=0,
        )

        c_idx, p_idx = batch_edges[:, 0], batch_edges[:, 1]

        logits = model(c_idx, p_idx)
        total_loss += criterion(logits, y).item()
        n_batches += 1
        all_logits.append(logits.cpu().numpy())
        all_true.append(y.cpu().numpy())

    logits_np = np.concatenate(all_logits)
    true_np = np.concatenate(all_true)
    probs = torch.sigmoid(torch.from_numpy(logits_np)).numpy()

    return {
        "val_bce": total_loss / n_batches,
        "val_auc_roc": roc_auc_score(true_np, probs),
        "val_auc_pr": average_precision_score(true_np, probs),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    h5ad_path: str,
    latent_dim: int = 8,
    epochs: int = 200,
    batch_size: int = 4096,
    neg_ratio: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    out_dir: str = "results/ldm_run",
    val_frac: float = 0.10,
    eval_every: int = 10,
    disk_chunk: int = 10000,
) -> dict:
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ---- Memory Safe Setup ---------------------------------------------------
    print(f"Opening {h5ad_path} in disk-backed mode...")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    n_cells, n_peaks = adata.shape

    n_val_cells = int(n_cells * val_frac)
    n_train_cells = n_cells - n_val_cells

    print(f"Dataset split by row: {n_train_cells} train cells, {n_val_cells} val cells")

    # Pre-load only the validation edges into RAM (usually <1GB for 10%)
    val_edges = load_validation_edges(
        adata, start_cell=n_train_cells, end_cell=n_cells, chunk_size=disk_chunk
    )
    print(f"Loaded {len(val_edges):,} validation edges.")

    # ---- Model Setup ---------------------------------------------------------
    model = LDM(n_cells=n_cells, n_peaks=n_peaks, latent_dim=latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model  : {n_cells} cells, {n_peaks} peaks, dim={latent_dim}, params={n_params:,}"
    )

    optimizer = torch.optim.Adam(
        [
            {
                "params": [model.z_i.weight, model.z_j.weight],
                "weight_decay": weight_decay,
            },
            {"params": [model.psi.weight, model.omega.weight], "weight_decay": 0.0},
        ],
        lr=lr,
    )

    criterion = nn.BCEWithLogitsLoss()

    history = {
        "train_bce": [],
        "val_bce": [],
        "val_auc_roc": [],
        "val_auc_pr": [],
        "epoch_time_s": [],
    }
    best_val_auc_pr = -1.0

    # ---- Loop ----------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        epoch_loss, n_batches = 0.0, 0

        # We don't know exact total batches upfront due to sparsity differences, so we track chunks
        progress_bar = tqdm(
            stream_train_epochs(
                adata,
                n_train_cells,
                n_cells,
                n_peaks,
                neg_ratio,
                batch_size,
                device,
                disk_chunk,
            ),
            desc=f"Epoch {epoch:03d}/{epochs}",
            leave=False,
        )

        for c_idx, p_idx, y in progress_bar:
            optimizer.zero_grad()
            loss = criterion(model(c_idx, p_idx), y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # Update terminal roughly every 50 batches to avoid IO bottleneck
            if n_batches % 50 == 0:
                progress_bar.set_postfix({"bce": f"{loss.item():.4f}"})

        train_bce = epoch_loss / max(1, n_batches)
        elapsed = time.time() - t0
        history["train_bce"].append(train_bce)
        history["epoch_time_s"].append(elapsed)

        if epoch % eval_every == 0 or epoch == 1:
            metrics = evaluate(
                model, val_edges, n_cells, n_peaks, neg_ratio, device, batch_size
            )
            history["val_bce"].append(metrics["val_bce"])
            history["val_auc_roc"].append(metrics["val_auc_roc"])
            history["val_auc_pr"].append(metrics["val_auc_pr"])

            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"train_bce={train_bce:.4f} | "
                f"val_bce={metrics['val_bce']:.4f} | "
                f"AUC-ROC={metrics['val_auc_roc']:.4f} | "
                f"AUC-PR={metrics['val_auc_pr']:.4f} | "
                f"({elapsed:.1f}s)"
            )

            if metrics["val_auc_pr"] > best_val_auc_pr:
                best_val_auc_pr = metrics["val_auc_pr"]
                torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
                print(f"  ✓ New best saved (AUC-PR={best_val_auc_pr:.4f})")
        else:
            print(
                f"Epoch {epoch:4d}/{epochs} | train_bce={train_bce:.4f} | ({elapsed:.1f}s)"
            )

    # ---- Save outputs --------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(out_dir, "final_model.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    model.eval()
    with torch.no_grad():
        np.save(os.path.join(out_dir, "z_cells.npy"), model.z_i.weight.cpu().numpy())
        np.save(os.path.join(out_dir, "z_peaks.npy"), model.z_j.weight.cpu().numpy())

    print(f"\nDone. Results saved to: {out_dir}")
    return history


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the Latent Distance Model")
    p.add_argument("--data", required=True, help="Path to .h5ad file")
    p.add_argument("--latent_dim", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--neg_ratio", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="results/ldm_run")
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument(
        "--disk_chunk",
        type=int,
        default=10000,
        help="Cells to load per chunk from disk",
    )
    args = p.parse_args()

    train(
        h5ad_path=args.data,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        neg_ratio=args.neg_ratio,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        out_dir=args.out_dir,
        val_frac=args.val_frac,
        eval_every=args.eval_every,
        disk_chunk=args.disk_chunk,
    )

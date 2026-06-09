"""
train.py — Training loop for the Latent Distance Model.

Edge split contract
-------------------
- train_edges : positive (cell, peak) pairs used for optimisation.
- val_edges   : held-out positives used only for evaluation.
- Negatives are sampled fresh each epoch by corrupting either the cell or the
  peak index of a positive edge (symmetric coin-flip, following SIMBA).
  False-negative filtering is deliberately omitted; see report Discussion.

Usage
-----
    uv run python src/train.py --data data/hematopoiesis.h5ad
    uv run python src/train.py --data data/hematopoiesis.h5ad --epochs 200 --latent_dim 8
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Allow running from repo root or from src/
sys.path.insert(0, os.path.dirname(__file__))
from ldm import LDM
from prepare_data import load_data, train_val_split


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
# Negative sampling
# ---------------------------------------------------------------------------

def sample_negatives(
    pos_edges: np.ndarray,
    n_cells: int,
    n_peaks: int,
    neg_ratio: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Corrupt one side of each positive edge to generate negatives.
    The corrupted side (cell or peak) is chosen by a symmetric coin flip.
    """
    N = len(pos_edges)
    total = N * neg_ratio
    repeated = np.tile(pos_edges, (neg_ratio, 1))

    corrupt_side = rng.integers(0, 2, size=total)   # 0 = corrupt cell, 1 = corrupt peak
    random_cells = rng.integers(0, n_cells, size=total)
    random_peaks = rng.integers(0, n_peaks, size=total)

    neg = repeated.copy()
    cell_mask = corrupt_side == 0
    neg[cell_mask,  0] = random_cells[cell_mask]
    neg[~cell_mask, 1] = random_peaks[~cell_mask]

    return neg


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
    rng: np.random.Generator,
    batch_size: int = 8192,
) -> dict:
    """
    Evaluate on held-out positives + matched sampled negatives.

    Returns dict with: val_bce, val_auc_roc, val_auc_pr
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    neg_edges = sample_negatives(val_edges, n_cells, n_peaks, neg_ratio, rng)
    all_edges  = np.vstack([val_edges, neg_edges])
    all_labels = np.concatenate([
        np.ones(len(val_edges),  dtype=np.float32),
        np.zeros(len(neg_edges), dtype=np.float32),
    ])

    dataset = TensorDataset(
        torch.from_numpy(all_edges[:, 0]).long(),
        torch.from_numpy(all_edges[:, 1]).long(),
        torch.from_numpy(all_labels),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits, all_true = [], []
    total_loss, n_batches = 0.0, 0

    for c_idx, p_idx, y in loader:
        c_idx, p_idx, y = c_idx.to(device), p_idx.to(device), y.to(device)
        logits = model(c_idx, p_idx)
        total_loss += criterion(logits, y).item()
        n_batches  += 1
        all_logits.append(logits.cpu().numpy())
        all_true.append(y.cpu().numpy())

    logits_np = np.concatenate(all_logits)
    true_np   = np.concatenate(all_true)
    probs     = torch.sigmoid(torch.from_numpy(logits_np)).numpy()

    return {
        "val_bce":     total_loss / n_batches,
        "val_auc_roc": roc_auc_score(true_np, probs),
        "val_auc_pr":  average_precision_score(true_np, probs),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    h5ad_path: str,
    latent_dim: int     = 8,
    epochs: int         = 200,
    batch_size: int     = 4096,
    neg_ratio: int      = 10,
    lr: float           = 1e-3,
    weight_decay: float = 1e-4,
    seed: int           = 42,
    out_dir: str        = "results/ldm_run",
    val_frac: float     = 0.10,
    eval_every: int     = 10,
) -> dict:
    """Full training run. Returns history dict."""

    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ---- Data ----------------------------------------------------------------
    adata, X_bin = load_data(h5ad_path)
    train_edges, val_edges, n_cells, n_peaks = train_val_split(
        X_bin, val_frac=val_frac, seed=seed
    )

    # ---- Model ---------------------------------------------------------------
    model = LDM(n_cells=n_cells, n_peaks=n_peaks, latent_dim=latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model  : {n_cells} cells, {n_peaks} peaks, "
          f"dim={latent_dim}, params={n_params:,}")

    # Weight decay on embeddings only, not on intercepts
    optimizer = torch.optim.Adam([
        {"params": [model.z_i.weight, model.z_j.weight], "weight_decay": weight_decay},
        {"params": [model.psi.weight, model.omega.weight], "weight_decay": 0.0},
    ], lr=lr)

    criterion = nn.BCEWithLogitsLoss()

    history = {
        "train_bce":    [],
        "val_bce":      [],
        "val_auc_roc":  [],
        "val_auc_pr":   [],
        "epoch_time_s": [],
    }
    best_val_auc_pr = -1.0

    # ---- Loop ----------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        neg_edges  = sample_negatives(train_edges, n_cells, n_peaks, neg_ratio, rng)
        all_edges  = np.vstack([train_edges, neg_edges])
        all_labels = np.concatenate([
            np.ones(len(train_edges),  dtype=np.float32),
            np.zeros(len(neg_edges), dtype=np.float32),
        ])
        perm = rng.permutation(len(all_edges))
        all_edges, all_labels = all_edges[perm], all_labels[perm]

        dataset = TensorDataset(
            torch.from_numpy(all_edges[:, 0]).long(),
            torch.from_numpy(all_edges[:, 1]).long(),
            torch.from_numpy(all_labels),
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )

        epoch_loss, n_batches = 0.0, 0
        for c_idx, p_idx, y in loader:
            c_idx, p_idx, y = c_idx.to(device), p_idx.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(c_idx, p_idx), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        train_bce = epoch_loss / n_batches
        elapsed   = time.time() - t0
        history["train_bce"].append(train_bce)
        history["epoch_time_s"].append(elapsed)

        if epoch % eval_every == 0 or epoch == 1:
            metrics = evaluate(
                model, val_edges, n_cells, n_peaks, neg_ratio, device, rng
            )
            history["val_bce"].append(metrics["val_bce"])
            history["val_auc_roc"].append(metrics["val_auc_roc"])
            history["val_auc_pr"].append(metrics["val_auc_pr"])

            print(
                f"Epoch {epoch:4d}/{epochs}  "
                f"train_bce={train_bce:.4f}  "
                f"val_bce={metrics['val_bce']:.4f}  "
                f"AUC-ROC={metrics['val_auc_roc']:.4f}  "
                f"AUC-PR={metrics['val_auc_pr']:.4f}  "
                f"({elapsed:.1f}s)"
            )

            if metrics["val_auc_pr"] > best_val_auc_pr:
                best_val_auc_pr = metrics["val_auc_pr"]
                torch.save(model.state_dict(),
                           os.path.join(out_dir, "best_model.pt"))
                print(f"  ✓ New best saved (AUC-PR={best_val_auc_pr:.4f})")
        else:
            print(f"Epoch {epoch:4d}/{epochs}  "
                  f"train_bce={train_bce:.4f}  ({elapsed:.1f}s)")

    # ---- Save outputs --------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(out_dir, "final_model.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    model.eval()
    with torch.no_grad():
        np.save(os.path.join(out_dir, "z_cells.npy"),
                model.z_i.weight.cpu().numpy())
        np.save(os.path.join(out_dir, "z_peaks.npy"),
                model.z_j.weight.cpu().numpy())

    print(f"\nDone. Results saved to: {out_dir}")
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the Latent Distance Model")
    p.add_argument("--data",          required=True,  help="Path to .h5ad file")
    p.add_argument("--latent_dim",    type=int,   default=8)
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--batch_size",    type=int,   default=4096)
    p.add_argument("--neg_ratio",     type=int,   default=10)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--out_dir",       type=str,   default="results/ldm_run")
    p.add_argument("--val_frac",      type=float, default=0.10)
    p.add_argument("--eval_every",    type=int,   default=10)
    args = p.parse_args()

    train(
        h5ad_path    = args.data,
        latent_dim   = args.latent_dim,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        neg_ratio    = args.neg_ratio,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        seed         = args.seed,
        out_dir      = args.out_dir,
        val_frac     = args.val_frac,
        eval_every   = args.eval_every,
    )

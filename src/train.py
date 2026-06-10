"""
train.py — Training loop for the Latent Distance Model.

Edge split contract
-------------------
- train_edges : positive (cell, peak) pairs used for optimisation.
- val_edges   : held-out positives used only for evaluation.
- Training negatives are sampled fresh each batch by corrupting either the
  cell or the peak index of a positive edge (symmetric coin-flip, following
  SIMBA). Training-time false-negative filtering is deliberately omitted; see
  report Discussion.

Evaluation contract
-------------------
- A single fixed evaluation set (held-out positives + sampled negatives) is
  built ONCE, with its own seed, and reused for every evaluation call. This
  keeps validation metrics comparable across epochs and — because the null
  model uses the identical builder, split seed and negative-sampling seed —
  identical between the full LDM and the null model, so the nested comparison
  isolates the contribution of the distance term.
- Unlike training, evaluation negatives are filtered against the observed
  positive set (train ∪ val) by default, so held-out metrics are not biased by
  mislabelling true edges as negatives. The SIMBA no-filtering convention is
  therefore retained only where it belongs (the training objective).

Usage
-----
    uv run python src/train.py --data data/hematopoiesis.h5ad
    uv run python src/train.py --data data/hematopoiesis.h5ad --epochs 200 --latent_dim 8
"""

import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

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
# Edge / key helpers
# ---------------------------------------------------------------------------

def edges_to_keys(edges: np.ndarray, n_peaks: int) -> np.ndarray:
    """Encode (cell, peak) pairs as unique int64 keys: cell * n_peaks + peak."""
    return edges[:, 0].astype(np.int64) * np.int64(n_peaks) + edges[:, 1].astype(np.int64)


def _isin_sorted(keys: np.ndarray, sorted_arr: np.ndarray) -> np.ndarray:
    """Vectorised membership test of `keys` in the sorted unique `sorted_arr`."""
    if sorted_arr.size == 0:
        return np.zeros(keys.shape, dtype=bool)
    idx = np.searchsorted(sorted_arr, keys)
    idx = np.clip(idx, 0, sorted_arr.size - 1)
    return sorted_arr[idx] == keys


# ---------------------------------------------------------------------------
# Negative sampling (numpy; used to build the fixed evaluation set)
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
    neg[cell_mask, 0] = random_cells[cell_mask]
    neg[~cell_mask, 1] = random_peaks[~cell_mask]
    return neg


def build_eval_set(
    val_edges: np.ndarray,
    n_cells: int,
    n_peaks: int,
    neg_ratio: int,
    seed: int,
    observed_keys_sorted: np.ndarray = None,
    eval_max_pos: int = 1_000_000,
):
    """
    Build a single fixed evaluation set reused across epochs and models.

    Returns a tuple of CPU tensors (cell_idx, peak_idx, label).

    The negative-sampling RNG is seeded from `seed` (offset by a constant) so
    that, given the same split seed, the full LDM and the null model evaluate
    on identical (cell, peak) pairs. If `observed_keys_sorted` is provided,
    sampled negatives that collide with an observed positive are re-drawn as
    uniform random non-edges.
    """
    rng = np.random.default_rng(seed + 12345)

    pos = val_edges
    if eval_max_pos and len(pos) > eval_max_pos:
        sel = rng.choice(len(pos), size=eval_max_pos, replace=False)
        pos = pos[sel]

    neg = sample_negatives(pos, n_cells, n_peaks, neg_ratio, rng)

    if observed_keys_sorted is not None:
        neg_keys = edges_to_keys(neg, n_peaks)
        bad = _isin_sorted(neg_keys, observed_keys_sorted)
        n_bad = int(bad.sum())
        tries = 0
        while n_bad > 0 and tries < 10:
            neg[bad, 0] = rng.integers(0, n_cells, size=n_bad)
            neg[bad, 1] = rng.integers(0, n_peaks, size=n_bad)
            neg_keys = edges_to_keys(neg, n_peaks)
            bad = _isin_sorted(neg_keys, observed_keys_sorted)
            n_bad = int(bad.sum())
            tries += 1

    cell = np.concatenate([pos[:, 0], neg[:, 0]]).astype(np.int64)
    peak = np.concatenate([pos[:, 1], neg[:, 1]]).astype(np.int64)
    label = np.concatenate([
        np.ones(len(pos), dtype=np.float32),
        np.zeros(len(neg), dtype=np.float32),
    ])
    print(f"Fixed eval set : {len(pos):,} positives + {len(neg):,} negatives "
          f"(prevalence {len(pos) / (len(pos) + len(neg)):.4f})")

    return (
        torch.from_numpy(cell),
        torch.from_numpy(peak),
        torch.from_numpy(label),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def f1_optimal(true: np.ndarray, probs: np.ndarray):
    """Return (best_F1, best_threshold) maximising F1 over the PR curve."""
    prec, rec, thr = precision_recall_curve(true, probs)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    if thr.size == 0:
        return float(f1[0]) if f1.size else 0.0, 0.5
    best = int(np.nanargmax(f1[:-1]))  # final PR point has no threshold
    return float(f1[best]), float(thr[best])


@torch.no_grad()
def evaluate(model, eval_set, device, eval_batch_size: int = 1_048_576) -> dict:
    """
    Evaluate on the fixed held-out set.

    Returns dict with: val_bce (sample-mean BCE / held-out log-loss),
    val_auc_roc, val_auc_pr, val_f1, val_threshold.
    """
    model.eval()
    cell, peak, label = eval_set
    n = cell.shape[0]

    total_loss = 0.0
    probs_chunks = []
    for s in range(0, n, eval_batch_size):
        e = min(n, s + eval_batch_size)
        c = cell[s:e].to(device)
        p = peak[s:e].to(device)
        y = label[s:e].to(device)
        logits = model(c, p)
        total_loss += F.binary_cross_entropy_with_logits(
            logits, y, reduction="sum"
        ).item()
        probs_chunks.append(torch.sigmoid(logits).cpu())

    probs = torch.cat(probs_chunks).numpy()
    true = label.numpy()
    f1, thr = f1_optimal(true, probs)

    return {
        "val_bce":     total_loss / n,
        "val_auc_roc": roc_auc_score(true, probs),
        "val_auc_pr":  average_precision_score(true, probs),
        "val_f1":      f1,
        "val_threshold": thr,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    h5ad_path: str,
    latent_dim: int       = 8,
    epochs: int           = 200,
    batch_size: int       = 4096,
    neg_ratio: int        = 10,
    lr: float             = 1e-3,
    weight_decay: float   = 1e-4,
    seed: int             = 42,
    out_dir: str          = "results/ldm_run",
    val_frac: float       = 0.10,
    eval_every: int       = 10,
    min_cells_pct: float  = 0.001,
    eval_max_pos: int     = 1_000_000,
    filter_eval_neg: bool = True,
) -> dict:
    """Full training run. Returns history dict."""

    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ---- Data ----------------------------------------------------------------
    adata, X_bin = load_data(h5ad_path, min_cells_pct=min_cells_pct)
    train_edges, val_edges, n_cells, n_peaks = train_val_split(
        X_bin, val_frac=val_frac, seed=seed
    )

    # ---- Fixed evaluation set (shared, comparable across models) -------------
    observed_keys = None
    if filter_eval_neg:
        all_pos = np.vstack([train_edges, val_edges])
        observed_keys = np.unique(edges_to_keys(all_pos, n_peaks))
    eval_set = build_eval_set(
        val_edges, n_cells, n_peaks, neg_ratio, seed,
        observed_keys_sorted=observed_keys, eval_max_pos=eval_max_pos,
    )

    # ---- Model ---------------------------------------------------------------
    model = LDM(n_cells=n_cells, n_peaks=n_peaks, latent_dim=latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model  : {n_cells} cells, {n_peaks} peaks, "
          f"dim={latent_dim}, params={n_params:,}")

    # Weight decay (L2 prior, following SIMBA) on embeddings only, not intercepts
    optimizer = torch.optim.Adam([
        {"params": [model.z_i.weight, model.z_j.weight], "weight_decay": weight_decay},
        {"params": [model.psi.weight, model.omega.weight], "weight_decay": 0.0},
    ], lr=lr)

    # Positive edges resident on the device; negatives sampled per batch on device
    train_pos = torch.as_tensor(train_edges, dtype=torch.long, device=device)
    N = train_pos.shape[0]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    history = {
        "train_bce":    [],
        "val_bce":      [],
        "val_auc_roc":  [],
        "val_auc_pr":   [],
        "val_f1":       [],
        "val_threshold": [],
        "eval_epochs":  [],
        "epoch_time_s": [],
    }
    best_val_auc_pr = -1.0
    best_state = None
    best_metrics = None
    best_epoch = None

    # ---- Loop ----------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        perm = torch.randperm(N, generator=gen, device=device)
        total_loss, total_n = 0.0, 0

        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            pos_c = train_pos[idx, 0]
            pos_p = train_pos[idx, 1]
            B = idx.shape[0]
            M = B * neg_ratio

            rep_c = pos_c.repeat(neg_ratio)
            rep_p = pos_p.repeat(neg_ratio)
            corrupt = torch.randint(0, 2, (M,), device=device, generator=gen)
            rand_c = torch.randint(0, n_cells, (M,), device=device, generator=gen)
            rand_p = torch.randint(0, n_peaks, (M,), device=device, generator=gen)
            neg_c = torch.where(corrupt == 0, rand_c, rep_c)
            neg_p = torch.where(corrupt == 0, rep_p, rand_p)

            all_c = torch.cat([pos_c, neg_c])
            all_p = torch.cat([pos_p, neg_p])
            y = torch.cat([
                torch.ones(B, device=device),
                torch.zeros(M, device=device),
            ])

            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(all_c, all_p), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * (B + M)
            total_n += (B + M)

        train_bce = total_loss / total_n
        elapsed = time.time() - t0
        history["train_bce"].append(train_bce)
        history["epoch_time_s"].append(elapsed)

        if epoch % eval_every == 0 or epoch == 1:
            metrics = evaluate(model, eval_set, device)
            history["val_bce"].append(metrics["val_bce"])
            history["val_auc_roc"].append(metrics["val_auc_roc"])
            history["val_auc_pr"].append(metrics["val_auc_pr"])
            history["val_f1"].append(metrics["val_f1"])
            history["val_threshold"].append(metrics["val_threshold"])
            history["eval_epochs"].append(epoch)

            print(
                f"Epoch {epoch:4d}/{epochs}  "
                f"train_bce={train_bce:.4f}  "
                f"val_bce={metrics['val_bce']:.4f}  "
                f"AUC-ROC={metrics['val_auc_roc']:.4f}  "
                f"AUC-PR={metrics['val_auc_pr']:.4f}  "
                f"F1={metrics['val_f1']:.4f}  "
                f"({elapsed:.1f}s)"
            )

            if metrics["val_auc_pr"] > best_val_auc_pr:
                best_val_auc_pr = metrics["val_auc_pr"]
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = dict(metrics)
                best_epoch = epoch
                torch.save(best_state, os.path.join(out_dir, "best_model.pt"))
                print(f"  ✓ New best saved (AUC-PR={best_val_auc_pr:.4f})")
        else:
            print(f"Epoch {epoch:4d}/{epochs}  "
                  f"train_bce={train_bce:.4f}  ({elapsed:.1f}s)")

    # ---- Save final weights --------------------------------------------------
    torch.save(model.state_dict(), os.path.join(out_dir, "final_model.pt"))

    # ---- Export embeddings from the BEST checkpoint --------------------------
    # RQ1 (embeddings) and RQ2 (reported metrics) then refer to the same model.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        np.save(os.path.join(out_dir, "z_cells.npy"),
                model.z_i.weight.cpu().numpy())
        np.save(os.path.join(out_dir, "z_peaks.npy"),
                model.z_j.weight.cpu().numpy())

    history["best"] = best_metrics
    history["best_epoch"] = best_epoch
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best epoch {best_epoch} (AUC-PR={best_val_auc_pr:.4f}). "
          f"Results saved to: {out_dir}")
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
    p.add_argument("--min_cells_pct", type=float, default=0.001,
                   help="Min fraction of cells a peak must appear in (PeakVI: 0.001).")
    p.add_argument("--eval_max_pos",  type=int,   default=1_000_000,
                   help="Cap on held-out positives used for the fixed eval set.")
    p.add_argument("--no_filter_eval_neg", action="store_true",
                   help="Disable filtering observed positives out of eval negatives.")
    args = p.parse_args()

    train(
        h5ad_path     = args.data,
        latent_dim    = args.latent_dim,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        neg_ratio     = args.neg_ratio,
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        seed          = args.seed,
        out_dir       = args.out_dir,
        val_frac      = args.val_frac,
        eval_every    = args.eval_every,
        min_cells_pct = args.min_cells_pct,
        eval_max_pos  = args.eval_max_pos,
        filter_eval_neg = not args.no_filter_eval_neg,
    )

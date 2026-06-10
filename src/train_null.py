"""
train_null.py — Training loop for the Null LDM (intercepts only).

The null model contains only per-cell (ψ_i) and per-peak (ω_j) intercepts,
with no latent embeddings and no distance term:

    η⁰_ij = ψ_i + ω_j

It serves as the nested baseline for the full LDM. The difference in held-out
log-likelihood (BCE), AUC-PR, AUC-ROC and F1 between the full model and this
baseline quantifies the contribution of the latent geometric structure
(nested model comparison, following Hoff et al., 2002).

Shared evaluation contract
--------------------------
This script imports `set_seed`, `edges_to_keys`, `sample_negatives`,
`build_eval_set` and `evaluate` directly from `train.py`. Because the
train/validation split (`train_val_split`) is deterministic given the seed and
`build_eval_set` seeds its negative sampler from the same `seed`, the null
model is evaluated on the *identical* fixed set of (cell, peak) pairs as the
full LDM. The nested comparison is therefore fair: only the model differs, not
the held-out positives, the sampled negatives, or the evaluation code path.

As in `train.py`, training-time negatives are sampled fresh each batch by
symmetric coin-flip corruption (SIMBA convention, no false-negative filtering),
while evaluation negatives are filtered against the observed positive set.

The intercept parameters ψ_i and ω_j are not regularised: shrinking them toward
zero would interfere with their role of correcting for sequencing depth and
peak width. The optimiser therefore applies no weight decay.

Usage
-----
    uv run python src/train_null.py --data data/hematopoiesis.h5ad
    uv run python src/train_null.py --data data/hematopoiesis.h5ad --epochs 200
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from ldm import NullLDM
from prepare_data import load_data, train_val_split
from train import build_eval_set, edges_to_keys, evaluate, sample_negatives, set_seed


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_null(
    h5ad_path: str,
    epochs: int           = 200,
    batch_size: int       = 4096,
    neg_ratio: int        = 10,
    lr: float             = 1e-3,
    seed: int             = 42,
    out_dir: str          = "results/null_run",
    val_frac: float       = 0.10,
    eval_every: int       = 10,
    min_cells_pct: float  = 0.001,
    eval_max_pos: int     = 1_000_000,
    filter_eval_neg: bool = True,
) -> dict:
    """Full null model training run. Returns history dict."""

    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ---- Data ----------------------------------------------------------------
    adata, X_bin = load_data(h5ad_path, min_cells_pct=min_cells_pct)
    train_edges, val_edges, n_cells, n_peaks = train_val_split(
        X_bin, val_frac=val_frac, seed=seed
    )

    # ---- Fixed evaluation set (identical builder/seed to the full LDM) -------
    observed_keys = None
    if filter_eval_neg:
        all_pos = np.vstack([train_edges, val_edges])
        observed_keys = np.unique(edges_to_keys(all_pos, n_peaks))
    eval_set = build_eval_set(
        val_edges, n_cells, n_peaks, neg_ratio, seed,
        observed_keys_sorted=observed_keys, eval_max_pos=eval_max_pos,
    )

    # ---- Model ---------------------------------------------------------------
    model = NullLDM(n_cells=n_cells, n_peaks=n_peaks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Null model : {n_cells} cells, {n_peaks} peaks, params={n_params:,}")

    # Intercepts are not regularised → no weight decay.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
                torch.save(best_state, os.path.join(out_dir, "best_null_model.pt"))
                print(f"  ✓ New best saved (AUC-PR={best_val_auc_pr:.4f})")
        else:
            print(f"Epoch {epoch:4d}/{epochs}  "
                  f"train_bce={train_bce:.4f}  ({elapsed:.1f}s)")

    # ---- Save ----------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(out_dir, "final_null_model.pt"))

    history["best"] = best_metrics
    history["best_epoch"] = best_epoch
    with open(os.path.join(out_dir, "null_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best epoch {best_epoch} (AUC-PR={best_val_auc_pr:.4f}). "
          f"Results saved to: {out_dir}")
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the Null LDM (intercepts only)")
    p.add_argument("--data",          required=True,  help="Path to .h5ad file")
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--batch_size",    type=int,   default=4096)
    p.add_argument("--neg_ratio",     type=int,   default=10)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--out_dir",       type=str,   default="results/null_run")
    p.add_argument("--val_frac",      type=float, default=0.10)
    p.add_argument("--eval_every",    type=int,   default=10)
    p.add_argument("--min_cells_pct", type=float, default=0.001,
                   help="Min fraction of cells a peak must appear in (PeakVI: 0.001).")
    p.add_argument("--eval_max_pos",  type=int,   default=1_000_000,
                   help="Cap on held-out positives used for the fixed eval set.")
    p.add_argument("--no_filter_eval_neg", action="store_true",
                   help="Disable filtering observed positives out of eval negatives.")
    args = p.parse_args()

    train_null(
        h5ad_path     = args.data,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        neg_ratio     = args.neg_ratio,
        lr            = args.lr,
        seed          = args.seed,
        out_dir       = args.out_dir,
        val_frac      = args.val_frac,
        eval_every    = args.eval_every,
        min_cells_pct = args.min_cells_pct,
        eval_max_pos  = args.eval_max_pos,
        filter_eval_neg = not args.no_filter_eval_neg,
    )

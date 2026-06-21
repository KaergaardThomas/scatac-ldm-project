"""
visualize_null.py — Component for comparing the Null model and full LDM.
Generates performance curves and intercepts calibration diagnostics.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
)

# Structural path resolution to completely bypass cross-import hijacking
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ldm import LDM, NullLDM
from src.train import load_data_to_ram_safely, gpu_accelerated_batches


@torch.no_grad()
def get_predictions(model, edges, n_cells, n_peaks, neg_ratio, device):
    """Compute exact probability predictions and ground truth from generator."""
    model.eval()
    _, loader = gpu_accelerated_batches(
        edges, n_cells, n_peaks, neg_ratio, batch_size=8192, device=device
    )

    all_probs, all_true = [], []
    for c_idx, p_idx, y in loader:
        logits = model(c_idx, p_idx)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_true.append(y.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_true)


def main(h5ad_path, ldm_dir, null_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ldm_dir, exist_ok=True)

    # ---- 1. Load Data Elements Natively -------------------------------------
    print("Extracting empirical array data...")
    train_edges, val_edges, n_cells, n_peaks = load_data_to_ram_safely(
        h5ad_path, val_frac=0.10, seed=42
    )

    # ---- 2. Initialize Models & Load Trained Parameters ---------------------
    print("\nLoading model checkpoints...")

    # Check for hyperparameter configurations in history if they exist
    latent_dim = 8
    history_path = os.path.join(ldm_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            hist = json.load(f)
            if "latent_dim" in hist:
                latent_dim = hist["latent_dim"]

    full_model = LDM(n_cells=n_cells, n_peaks=n_peaks, latent_dim=latent_dim).to(device)
    full_weights = os.path.join(ldm_dir, "best_model.pt")
    if os.path.exists(full_weights):
        full_model.load_state_dict(torch.load(full_weights, map_location=device))
        print(f"  ✓ Loaded Full LDM from {full_weights}")
    else:
        print(
            f"  ⚠️ Full LDM weights not found at {full_weights}. Using random initializations."
        )

    null_model = NullLDM(n_cells=n_cells, n_peaks=n_peaks).to(device)
    null_weights = os.path.join(null_dir, "best_null_model.pt")
    if os.path.exists(null_weights):
        null_model.load_state_dict(torch.load(null_weights, map_location=device))
        print(f"  ✓ Loaded Null Model from {null_weights}")
    else:
        # Fall back to final model if best model doesn't exist
        null_weights_final = os.path.join(null_dir, "final_null_model.pt")
        if os.path.exists(null_weights_final):
            null_model.load_state_dict(
                torch.load(null_weights_final, map_location=device)
            )
            print(f"  ✓ Loaded Null Model from {null_weights_final}")
        else:
            print(f"  ⚠️ Null Model weights not found. Using random initializations.")

    # ---- 3. Extract Validation Predictions ----------------------------------
    print("\nGenerating model evaluation predictions (this may take a moment)...")
    # Using a negative ratio of 10 to exactly match training conditions
    full_probs, true_labels = get_predictions(
        full_model, val_edges, n_cells, n_peaks, 10, device
    )
    null_probs, _ = get_predictions(null_model, val_edges, n_cells, n_peaks, 10, device)

    # ---- 4. Plot Performance Validation Curves ------------------------------
    print("Calculating metrics and curves...")
    n_fpr, n_tpr, _ = roc_curve(true_labels, null_probs)
    f_fpr, f_tpr, _ = roc_curve(true_labels, full_probs)
    n_prec, n_rec, _ = precision_recall_curve(true_labels, null_probs)
    f_prec, f_rec, _ = precision_recall_curve(true_labels, full_probs)

    n_auc_roc = auc(n_fpr, n_tpr)
    f_auc_roc = auc(f_fpr, f_tpr)
    n_auc_pr = average_precision_score(true_labels, null_probs)
    f_auc_pr = average_precision_score(true_labels, full_probs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left Panel: ROC
    axes[0].plot(
        n_fpr,
        n_tpr,
        color="darkgray",
        linestyle="--",
        linewidth=1.5,
        label=f"Null Model (AUC = {n_auc_roc:.4f})",
    )
    axes[0].plot(
        f_fpr,
        f_tpr,
        color="royalblue",
        linewidth=2.5,
        label=f"Full LDM (AUC = {f_auc_roc:.4f})",
    )
    axes[0].plot([0, 1], [0, 1], color="black", linestyle=":", alpha=0.5)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve Comparison")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.25)

    # Right Panel: Precision-Recall
    axes[1].plot(
        n_rec,
        n_prec,
        color="darkgray",
        linestyle="--",
        linewidth=1.5,
        label=f"Null Model (PR-AUC = {n_auc_pr:.4f})",
    )
    axes[1].plot(
        f_rec,
        f_prec,
        color="royalblue",
        linewidth=2.5,
        label=f"Full LDM (PR-AUC = {f_auc_pr:.4f})",
    )
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve Comparison")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    curves_out = os.path.join(ldm_dir, "null_comparison_curves.png")
    plt.savefig(curves_out, dpi=200)
    print(f"  ✓ Saved performance comparison curves to: {curves_out}")

    # ---- 5. Plot Intercept Calibration Figures -----------------------------
    print("Generating intercept calibration analysis...")
    psi_weights = null_model.psi.weight.detach().cpu().numpy().flatten()
    omega_weights = null_model.omega.weight.detach().cpu().numpy().flatten()

    # Calculate global empirical frequencies directly from positive train/val counts
    combined_edges = np.vstack([train_edges, val_edges])
    cell_counts = np.bincount(combined_edges[:, 0], minlength=n_cells)
    peak_counts = np.bincount(combined_edges[:, 1], minlength=n_peaks)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Cell Intercepts vs Library Depth
    log_cell_depth = np.log10(cell_counts + 1)
    axes[0].scatter(
        log_cell_depth, psi_weights, alpha=0.3, color="teal", s=3, rasterized=True
    )
    axes[0].set_xlabel(r"Empirical Cell Library Depth ($\log_{10}$ Peak Count)")
    axes[0].set_ylabel(r"Null Model Intercept ($\psi_i$)")
    axes[0].set_title("Cell Intercept Calibration Profiles")
    axes[0].grid(True, alpha=0.25)

    # Peak Intercepts vs Global Prevalence
    log_peak_prev = np.log10(peak_counts + 1)
    axes[1].scatter(
        log_peak_prev, omega_weights, alpha=0.3, color="purple", s=1, rasterized=True
    )
    axes[1].set_xlabel(r"Empirical Peak Prevalence ($\log_{10}$ Cell Count)")
    axes[1].set_ylabel(r"Null Model Intercept ($\omega_j$)")
    axes[1].set_title("Peak Intercept Calibration Profiles")
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    calib_out = os.path.join(ldm_dir, "null_intercept_calibration.png")
    plt.savefig(calib_out, dpi=200)
    print(f"  ✓ Saved intercept parameter tracking to: {calib_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate and visualize LDM versus Null baseline"
    )
    p.add_argument("--data", required=True, help="Path to .h5ad single-cell file")
    p.add_argument(
        "--ldm_dir",
        type=str,
        default="results/ldm_run",
        help="Output folder of full LDM model run",
    )
    p.add_argument(
        "--null_dir",
        type=str,
        default="results/null_run",
        help="Output folder of Null model run",
    )
    args = p.parse_args()

    main(h5ad_path=args.data, ldm_dir=args.ldm_dir, null_dir=args.null_dir)

"""
plot_history.py — Plot training curves from a model's history.json.

Usage (from repo root):
    uv run python src/plot_history.py --ldm_dir results/ldm_dim16 --out_dir results/evaluation_dim16
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_history(ldm_dir: str, null_dir: str = None, out_dir: str = None):
    out_dir = out_dir or ldm_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load LDM history ----------------------------------------------------
    with open(os.path.join(ldm_dir, "history.json")) as f:
        hist = json.load(f)

    epochs = list(range(1, len(hist["train_bce"]) + 1))
    eval_epochs = hist.get("eval_epochs", [])

    # ---- Load null history (optional) ----------------------------------------
    null_hist = None
    if null_dir:
        null_path = os.path.join(null_dir, "null_history.json")
        if os.path.exists(null_path):
            with open(null_path) as f:
                null_hist = json.load(f)

    # ---- Figure 1: Training BCE ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, hist["train_bce"], label="Train BCE", color="#2196F3", lw=1.5)
    if eval_epochs and hist.get("val_bce"):
        ax.plot(eval_epochs, hist["val_bce"], label="Val BCE",
                color="#F44336", lw=1.5, linestyle="--", marker="o", markersize=3)
    if null_hist and null_hist.get("val_bce"):
        null_eval = null_hist.get("eval_epochs", [])
        ax.plot(null_eval, null_hist["val_bce"], label="Null val BCE",
                color="#9E9E9E", lw=1.5, linestyle=":", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_loss.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ---- Figure 2: AUC-ROC and AUC-PR ----------------------------------------
    if eval_epochs and hist.get("val_auc_roc"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(eval_epochs, hist["val_auc_roc"],
                     color="#4CAF50", lw=1.5, marker="o", markersize=3)
        if null_hist and null_hist.get("val_auc_roc"):
            null_eval = null_hist.get("eval_epochs", [])
            axes[0].plot(null_eval, null_hist["val_auc_roc"],
                         color="#9E9E9E", lw=1.5, linestyle=":", marker="s",
                         markersize=3, label="Null")
            axes[0].legend(["LDM", "Null"])
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("AUC-ROC")
        axes[0].set_title("Validation AUC-ROC")
        axes[0].set_ylim([0.5, 1.0])
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(eval_epochs, hist["val_auc_pr"],
                     color="#FF9800", lw=1.5, marker="o", markersize=3)
        if null_hist and null_hist.get("val_auc_pr"):
            null_eval = null_hist.get("eval_epochs", [])
            axes[1].plot(null_eval, null_hist["val_auc_pr"],
                         color="#9E9E9E", lw=1.5, linestyle=":", marker="s",
                         markersize=3, label="Null")
            axes[1].legend(["LDM", "Null"])
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC-PR")
        axes[1].set_title("Validation AUC-PR")
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("Link Prediction Performance (RQ2)", fontsize=13)
        plt.tight_layout()
        path = os.path.join(out_dir, "validation_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    # ---- Figure 3: F1 score --------------------------------------------------
    if eval_epochs and hist.get("val_f1"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(eval_epochs, hist["val_f1"],
                color="#9C27B0", lw=1.5, marker="o", markersize=3, label="LDM")
        if null_hist and null_hist.get("val_f1"):
            null_eval = null_hist.get("eval_epochs", [])
            ax.plot(null_eval, null_hist["val_f1"],
                    color="#9E9E9E", lw=1.5, linestyle=":", marker="s",
                    markersize=3, label="Null")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 Score")
        ax.set_title("Validation F1 Score (optimal threshold)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, "validation_f1.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot training curves from history.json")
    p.add_argument("--ldm_dir",  required=True, help="Directory with history.json")
    p.add_argument("--null_dir", default=None,  help="Directory with null_history.json (optional)")
    p.add_argument("--out_dir",  default=None,  help="Output directory for plots")
    args = p.parse_args()

    plot_history(
        ldm_dir  = args.ldm_dir,
        null_dir = args.null_dir,
        out_dir  = args.out_dir,
    )

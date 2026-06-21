"""
plot_peakvi.py — Plot training curves from a PeakVI model's history.json.

Uses per-step BCE when available (requires --log_every N during training),
otherwise falls back to per-epoch BCE.

Usage (from repo root):
    uv run python src/plot_peakvi.py \
        --peakvi_dir results/peakvi_dim16 \
        --null_dir   results/null_nr70 \
        --out_dir    results/evaluation_peakvi_dim16
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_peakvi(peakvi_dir: str, null_dir: str = None, out_dir: str = None):
    out_dir = out_dir or peakvi_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(peakvi_dir, "history.json")) as f:
        hist = json.load(f)

    epochs = list(range(1, len(hist["train_bce"]) + 1))
    eval_epochs = hist.get("eval_epochs", [])

    # Per-step data (only present if --log_every was set during training)
    step_bce = hist.get("step_bce", [])
    step_global = hist.get("step_global", [])

    null_hist = None
    if null_dir:
        null_path = os.path.join(null_dir, "null_history.json")
        if os.path.exists(null_path):
            with open(null_path) as f:
                null_hist = json.load(f)

    # ---- Figure 1: Training BCE (step-level if available, else per-epoch) ----
    fig, ax = plt.subplots(figsize=(10, 4))

    if step_bce:
        ax.plot(
            step_global,
            step_bce,
            label="Train BCE (per 100 steps)",
            color="#2196F3",
            lw=1.0,
            alpha=0.8,
        )
        x_label = "Global step"
        # Mark epoch boundaries
        steps_per_epoch = step_global[-1] / len(epochs) if epochs else None
        if steps_per_epoch:
            for e in epochs:
                ax.axvline(
                    e * steps_per_epoch, color="grey", lw=0.4, linestyle="--", alpha=0.4
                )
    else:
        ax.plot(
            epochs,
            hist["train_bce"],
            label="Train BCE (per epoch)",
            color="#2196F3",
            lw=1.5,
        )
        x_label = "Epoch"

    if eval_epochs and hist.get("val_bce"):
        # Convert eval epochs to steps if using step x-axis
        x_eval = (
            [e * steps_per_epoch for e in eval_epochs]
            if step_bce and steps_per_epoch
            else eval_epochs
        )
        ax.plot(
            x_eval,
            hist["val_bce"],
            label="Val BCE",
            color="#F44336",
            lw=1.5,
            linestyle="--",
            marker="o",
            markersize=4,
        )

    if null_hist and null_hist.get("val_bce"):
        null_eval = null_hist.get("eval_epochs", [])
        ax.plot(
            null_eval,
            null_hist["val_bce"],
            label="Null val BCE",
            color="#9E9E9E",
            lw=1.5,
            linestyle=":",
            marker="s",
            markersize=4,
        )

    ax.set_xlabel(x_label)
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

        axes[0].plot(
            eval_epochs,
            hist["val_auc_roc"],
            color="#4CAF50",
            lw=1.5,
            marker="o",
            markersize=3,
            label="PeakVI",
        )
        if null_hist and null_hist.get("val_auc_roc"):
            axes[0].plot(
                null_hist.get("eval_epochs", []),
                null_hist["val_auc_roc"],
                color="#9E9E9E",
                lw=1.5,
                linestyle=":",
                marker="s",
                markersize=3,
                label="Null",
            )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("AUC-ROC")
        axes[0].set_title("Validation AUC-ROC")
        axes[0].set_ylim([0.5, 1.0])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            eval_epochs,
            hist["val_auc_pr"],
            color="#FF9800",
            lw=1.5,
            marker="o",
            markersize=3,
            label="PeakVI",
        )
        if null_hist and null_hist.get("val_auc_pr"):
            axes[1].plot(
                null_hist.get("eval_epochs", []),
                null_hist["val_auc_pr"],
                color="#9E9E9E",
                lw=1.5,
                linestyle=":",
                marker="s",
                markersize=3,
                label="Null",
            )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC-PR")
        axes[1].set_title("Validation AUC-PR")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("Link Prediction Performance (RQ2)", fontsize=13)
        plt.tight_layout()
        path = os.path.join(out_dir, "validation_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    # ---- Figure 3: F1 --------------------------------------------------------
    if eval_epochs and hist.get("val_f1"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            eval_epochs,
            hist["val_f1"],
            color="#9C27B0",
            lw=1.5,
            marker="o",
            markersize=3,
            label="PeakVI",
        )
        if null_hist and null_hist.get("val_f1"):
            ax.plot(
                null_hist.get("eval_epochs", []),
                null_hist["val_f1"],
                color="#9E9E9E",
                lw=1.5,
                linestyle=":",
                marker="s",
                markersize=3,
                label="Null",
            )
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

    # ---- Figure 4: ARI and NMI -----------------------------------------------
    if eval_epochs and (hist.get("val_ari") or hist.get("val_nmi")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        if hist.get("val_ari"):
            axes[0].plot(
                eval_epochs,
                hist["val_ari"],
                color="#E91E63",
                lw=1.5,
                marker="o",
                markersize=3,
                label="PeakVI",
            )
            if null_hist and null_hist.get("val_ari"):
                axes[0].plot(
                    null_hist.get("eval_epochs", []),
                    null_hist["val_ari"],
                    color="#9E9E9E",
                    lw=1.5,
                    linestyle=":",
                    marker="s",
                    markersize=3,
                    label="Null",
                )
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("ARI")
            axes[0].set_title("Validation ARI")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        if hist.get("val_nmi"):
            axes[1].plot(
                eval_epochs,
                hist["val_nmi"],
                color="#00BCD4",
                lw=1.5,
                marker="o",
                markersize=3,
                label="PeakVI",
            )
            if null_hist and null_hist.get("val_nmi"):
                axes[1].plot(
                    null_hist.get("eval_epochs", []),
                    null_hist["val_nmi"],
                    color="#9E9E9E",
                    lw=1.5,
                    linestyle=":",
                    marker="s",
                    markersize=3,
                    label="Null",
                )
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("NMI")
            axes[1].set_title("Validation NMI")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.suptitle("Clustering Performance (RQ1)", fontsize=13)
        plt.tight_layout()
        path = os.path.join(out_dir, "validation_clustering.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--peakvi_dir", required=True)
    p.add_argument("--null_dir", default=None)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    plot_peakvi(
        peakvi_dir=args.peakvi_dir, null_dir=args.null_dir, out_dir=args.out_dir
    )

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t


def print_markdown_table(headers, rows):
    """Prints a clean, formatted text table in the console."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    header_line = " | ".join(f"{str(h):<{widths[i]}}" for i, h in enumerate(headers))
    sep_line = "-|-".join("-" * widths[i] for i in range(len(headers)))
    print(f"| {header_line} |")
    print(f"| {sep_line} |")
    for row in rows:
        row_line = " | ".join(f"{str(val):<{widths[i]}}" for i, val in enumerate(row))
        print(f"| {row_line} |")


def plot_multiseed_performance(base_model_dir, seeds, dim=16, out_dir="./plots"):
    metrics_to_plot = ["train_loss", "val_bce", "val_auc_roc", "val_auc_pr", "val_f1"]
    data = {m: [] for m in metrics_to_plot}

    train_epochs = None
    eval_epochs = None
    valid_seeds = []

    for seed in seeds:
        hist_path = os.path.join(
            base_model_dir, f"seed_{seed}_dim_{dim}", "history.json"
        )
        if not os.path.exists(hist_path):
            continue

        with open(hist_path, "r") as f:
            history = json.load(f)

        valid_seeds.append(seed)
        for m in metrics_to_plot:
            data[m].append(history[m])

        if train_epochs is None:
            train_epochs = list(range(1, len(history["train_loss"]) + 1))
            eval_epochs = history["eval_epochs"]

    if not valid_seeds:
        print("Error: No history data found for any of the specified seeds.")
        return

    n_seeds = len(valid_seeds)
    print(f"\n=========================================================")
    print(f" METRIC ANALYSIS FOR {n_seeds} SEEDS (LATENT DIM: {dim})")
    print(f"=========================================================\n")

    # Convert to numpy arrays & align sizes
    aligned_data = {}
    x_axes = {}
    t_crit = t.ppf(0.975, df=n_seeds - 1) if n_seeds > 1 else 0

    for metric in metrics_to_plot:
        raw_x = np.array(train_epochs if metric == "train_loss" else eval_epochs)
        raw_y = np.array(data[metric])
        min_len = min(len(raw_x), raw_y.shape[1])

        x_axes[metric] = raw_x[:min_len]
        aligned_data[metric] = raw_y[:, :min_len]

    # ---------------------------------------------------------
    # TABLE 1: FINAL EPOCH SUMMARY STATISTICS
    # ---------------------------------------------------------
    print("### TABLE 1: Final Epoch Performance Summary (Aggregated)")
    final_headers = ["Metric", "Mean", "Std Dev", "95% CI Lower", "95% CI Upper"]
    final_rows = []

    for m in metrics_to_plot:
        y_final = aligned_data[m][:, -1]
        mean_val = np.mean(y_final)
        std_val = np.std(y_final, ddof=1) if n_seeds > 1 else 0
        sem_val = std_val / np.sqrt(n_seeds) if n_seeds > 1 else 0
        ci_bound = t_crit * sem_val

        final_rows.append(
            [
                m.upper(),
                f"{mean_val:.4f}",
                f"{std_val:.4f}",
                f"{(mean_val - ci_bound):.4f}",
                f"{(mean_val + ci_bound):.4f}",
            ]
        )
    print_markdown_table(final_headers, final_rows)
    print("\n" + "-" * 80 + "\n")

    # ---------------------------------------------------------
    # TABLE 2: SEED-BY-SEED FINAL PERFORMANCE BREAKDOWN
    # ---------------------------------------------------------
    print("### TABLE 2: Final Performance Matrix by Individual Seed")
    seed_headers = ["Seed Key"] + [
        m.replace("val_", "").upper() for m in metrics_to_plot
    ]
    seed_rows = []

    for idx, seed in enumerate(valid_seeds):
        row = [f"Seed {seed}"]
        for m in metrics_to_plot:
            final_seed_val = aligned_data[m][
                idx, -1
            ]  # Get final epoch value for this specific seed
            row.append(f"{final_seed_val:.4f}")
        seed_rows.append(row)

    print_markdown_table(seed_headers, seed_rows)
    print("\n" + "-" * 80 + "\n")

    # ---------------------------------------------------------
    # TABLE 3: MEAN TRAJECTORY PROFILE (Selected Epochs Breakdown)
    # ---------------------------------------------------------
    print("### TABLE 3: Trajectory Snapshot (Mean Values Across Training)")
    total_val_steps = len(x_axes["val_bce"])
    sample_indices = sorted(
        list(
            set(np.linspace(0, total_val_steps - 1, min(6, total_val_steps), dtype=int))
        )
    )

    traj_headers = ["Epoch"] + [m.replace("val_", "").upper() for m in metrics_to_plot]
    traj_rows = []

    for idx in sample_indices:
        epoch_num = x_axes["val_bce"][idx]
        row = [f"Epoch {epoch_num}"]

        for m in metrics_to_plot:
            if m == "train_loss":
                train_total = aligned_data[m].shape[1]
                step_idx = min(
                    int(idx * (train_total / total_val_steps)), train_total - 1
                )
            else:
                step_idx = min(idx, aligned_data[m].shape[1] - 1)

            mean_val = np.mean(aligned_data[m][:, step_idx])
            row.append(f"{mean_val:.4f}")
        traj_rows.append(row)

    print_markdown_table(traj_headers, traj_rows)
    print("\n")

    # ---------------------------------------------------------
    # MATPLOTLIB PLOTTING ENGINE
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    colors = {
        "train_loss": "#1f77b4",
        "val_bce": "#ff7f0e",
        "val_auc_roc": "#2ca02c",
        "val_auc_pr": "#d62728",
        "val_f1": "#9467bd",
    }

    for i, m in enumerate(metrics_to_plot):
        ax = axes[i]
        x = x_axes[m]
        y_mat = aligned_data[m]

        mean_curve = np.mean(y_mat, axis=0)
        std_curve = np.std(y_mat, axis=0, ddof=1) if n_seeds > 1 else 0
        ci_bound = t_crit * (std_curve / np.sqrt(n_seeds)) if n_seeds > 1 else 0

        ax.plot(x, mean_curve, color=colors[m], label="Mean", lw=2)
        if n_seeds > 1:
            ax.fill_between(
                x,
                mean_curve - ci_bound,
                mean_curve + ci_bound,
                color=colors[m],
                alpha=0.2,
                label="95% CI",
            )

        ax.set_title(m.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Epochs")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

    fig.delaxes(axes[-1])
    plt.suptitle(
        f"LDM Convergence Curves with 95% Confidence Intervals",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_img_path = os.path.join(out_dir, f"ldm_performance_dim{dim}.png")
    plt.savefig(out_img_path, dpi=300)
    print(f"Saved figure to: {out_img_path}\n")


if __name__ == "__main__":
    SEEDS = [42, 237, 284, 156, 708, 844, 444, 397, 431, 437]
    BASE_MODEL_DIR = "results/ldm_model"
    LATENT_DIM = 16

    plot_multiseed_performance(BASE_MODEL_DIR, SEEDS, dim=LATENT_DIM, out_dir="./plots")

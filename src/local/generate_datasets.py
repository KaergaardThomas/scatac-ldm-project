import argparse
import os
import gc
import scanpy as sc
import scipy.sparse as sp
import numpy as np
import concurrent.futures
from tqdm.auto import tqdm


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


def build_eval_set(
    val_edges: np.ndarray,
    n_cells: int,
    n_peaks: int,
    neg_ratio: int,
    seed: int,
    observed_csr: sp.csr_matrix = None,
    eval_max_pos: int = 1_000_000,
    disable_tqdm: bool = False,
):
    rng = np.random.default_rng(seed + 12345)

    pos = val_edges
    if eval_max_pos and len(pos) > eval_max_pos:
        sel = rng.choice(len(pos), size=eval_max_pos, replace=False)
        pos = pos[sel]

    # 1. Sample initial negatives
    neg = sample_negatives(pos, n_cells, n_peaks, neg_ratio, rng)

    if observed_csr is not None:
        collisions = np.asarray(observed_csr[neg[:, 0], neg[:, 1]]).flatten()
        bad = collisions == True
        n_bad = int(bad.sum())

        total_neg = len(neg)

        # tqdm disabled by default in threads to prevent console scrambling
        pbar = tqdm(
            total=total_neg,
            desc=f"Filtering Collisions (Seed {seed})",
            unit=" edges",
            disable=disable_tqdm,
        )
        pbar.update(total_neg - n_bad)

        tries = 0
        while n_bad > 0 and tries < 10:
            neg[bad, 0] = rng.integers(0, n_cells, size=n_bad)
            neg[bad, 1] = rng.integers(0, n_peaks, size=n_bad)

            collisions = np.asarray(observed_csr[neg[:, 0], neg[:, 1]]).flatten()
            bad = collisions == True
            new_n_bad = int(bad.sum())

            resolved_this_round = n_bad - new_n_bad
            if resolved_this_round > 0:
                pbar.update(resolved_this_round)

            n_bad = new_n_bad
            tries += 1

        pbar.close()

        if n_bad > 0:
            print(
                f"Warning: {n_bad} collisions could not be resolved after 10 tries for seed {seed}."
            )

    cell = np.concatenate([pos[:, 0], neg[:, 0]]).astype(np.int64)
    peak = np.concatenate([pos[:, 1], neg[:, 1]]).astype(np.int64)
    label = np.concatenate(
        [
            np.ones(len(pos), dtype=np.float32),
            np.zeros(len(neg), dtype=np.float32),
        ]
    )

    return cell, peak, label


def process_seed_task(seed, cells, peaks, num_edges, observed_csr, args):
    """Worker function to process a single seed."""
    print(f"[{seed}] Starting dataset generation...")
    n_cells, n_peaks = observed_csr.shape

    # Use isolated random generator for thread safety
    rng = np.random.default_rng(seed)

    # 1. Edge Split Logic
    perm = rng.permutation(num_edges)
    n_val = int(num_edges * args.val_split)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_cells, train_peaks = cells[train_idx], peaks[train_idx]
    val_cells, val_peaks = cells[val_idx], peaks[val_idx]

    val_edges = np.column_stack((val_cells, val_peaks))

    # 2. Build Fixed Eval Set using shared, pre-computed observed_csr
    val_c, val_p, val_y = build_eval_set(
        val_edges,
        n_cells,
        n_peaks,
        neg_ratio=10,
        seed=seed,
        observed_csr=observed_csr,
        disable_tqdm=True,  # Disable TQDM in threads to avoid messy logs
    )

    # 3. Save to disk
    out_file = os.path.join(args.out_dir, f"dataset_seed_{seed}.npz")
    np.savez_compressed(
        out_file,
        train_cells=train_cells,
        train_peaks=train_peaks,
        val_c=val_c,
        val_p=val_p,
        val_y=val_y,
    )

    # Local arrays will be automatically garbage collected
    return f"[{seed}] Saved dataset to {out_file}"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_path", type=str, default="data/hematopoiesis_GSE129785_FACS_sorted.h5ad"
    )
    p.add_argument("--out_dir", type=str, default="data/pregenerated_sets")
    p.add_argument("--min_cells_fraction", type=float, default=0.01)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 237, 284, 156, 708, 844, 444, 397, 431, 437],
    )
    p.add_argument(
        "--threads", type=int, default=8, help="Number of concurrent threads to use"
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    filtered_adata_path = os.path.join(args.out_dir, "filtered_base_data.h5ad")

    print(f"Loading data from {args.data_path}...")
    adata = sc.read_h5ad(args.data_path)
    adata.obs_names_make_unique()

    # Apply global filtering once
    min_cells = int(adata.n_obs * args.min_cells_fraction)
    print(f"Filtering genes present in less than {min_cells} cells...")
    sc.pp.filter_genes(adata, min_cells=min_cells)

    print(f"Saving filtered base anndata to {filtered_adata_path}...")
    adata.write_h5ad(filtered_adata_path)

    n_cells, n_peaks = adata.shape
    print(f"Final shape: {n_cells} cells x {n_peaks} peaks")

    # Get COO coordinates
    X_coo = (
        adata.X.tocoo() if sp.isspmatrix(adata.X) else sp.csr_matrix(adata.X).tocoo()
    )
    cells, peaks = X_coo.row, X_coo.col
    num_edges = len(cells)

    del adata
    gc.collect()

    print("\nBuilding global CSR matrix for collision detection (Optimization)...")
    # All splits contain exactly the same edges as the base graph.
    # Building this ONCE saves memory relative to building one per thread.
    global_data = np.ones(num_edges, dtype=bool)
    shared_observed_csr = sp.csr_matrix(
        (global_data, (cells, peaks)), shape=(n_cells, n_peaks)
    )
    del global_data
    gc.collect()

    print(f"\n>> Spawning thread pool with {args.threads} workers...")

    # Use ThreadPoolExecutor so memory arrays (cells, peaks, csr) are shared natively
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                process_seed_task,
                seed,
                cells,
                peaks,
                num_edges,
                shared_observed_csr,
                args,
            ): seed
            for seed in args.seeds
        }

        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                result_msg = future.result()
                print(result_msg)
            except Exception as e:
                print(f"[{seed}] Failed with error: {e}")

    print("\nAll datasets generated successfully!")

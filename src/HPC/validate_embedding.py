#!/usr/bin/env python3
"""
Biological validation of the joint cell+peak embedding.

For the latent dimensionality chosen from the sweep, this computes each
cell-type centroid in the shared latent space and ranks peaks by proximity
to it, two ways:

  * nearest   -- plainly closest peaks to the centroid (the supervisor's ask)
  * specific  -- peaks closer to THIS centroid than to the others (a contrast
                 that controls for ubiquitously-accessible peaks)

It writes a CSV of the top peaks per cell type, carrying along any gene
annotation already present on ad.var, so the result can be checked against
the marker-gene literature.

Usage:
    uv run python src/validate_embedding.py --dim 16 --topk 25
"""
import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, required=True,
                    help="Latent dim chosen from the sweep -> results/ldm_dim{DIM}.")
    ap.add_argument("--data", default=os.path.expanduser(
        "~/data/hematopoiesis_GSE129785_FACS_sorted.h5ad"))
    ap.add_argument("--results_dir", default=None,
                    help="Defaults to results/ldm_dim{DIM}.")
    ap.add_argument("--min_cells_pct", type=float, default=0.001,
                    help="MUST match training (keeps peak order aligned with z_peaks).")
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--label_col", default=None)
    args = ap.parse_args()

    d = args.results_dir or f"results/ldm_dim{args.dim}"
    zc = np.load(f"{d}/z_cells.npy")
    zp = np.load(f"{d}/z_peaks.npy")
    print(f"Loaded z_cells {zc.shape}, z_peaks {zp.shape} from {d}")

    ad = sc.read_h5ad(args.data)
    col = args.label_col or next(
        c for c in ["cell_type", "CellType", "celltype", "BioClassification",
                    "label", "Group", "cluster"] if c in ad.obs.columns)
    labels = np.asarray(ad.obs[col].values)
    print(f"Label column: {col}  ({len(np.unique(labels))} types)")

    # --- Reproduce the training peak filter so names align with z_peaks rows ---
    min_cells = int(args.min_cells_pct * ad.n_obs)
    counts = np.asarray((ad.X > 0).sum(axis=0)).ravel()
    keep = counts >= min_cells
    peak_names = np.asarray(ad.var_names)[keep]
    var_kept = ad.var.loc[keep].reset_index(drop=True)
    assert peak_names.shape[0] == zp.shape[0], (
        f"Peak mismatch: {peak_names.shape[0]} names vs {zp.shape[0]} embeddings -- "
        f"check that --min_cells_pct matches training.")

    annot_cols = [c for c in ad.var.columns
                  if any(k in c.lower() for k in
                         ["gene", "symbol", "annot", "nearest"])]
    print(f"Gene-annotation columns found on ad.var: {annot_cols or 'NONE'}")

    # --- Centroids in the shared latent space ---------------------------------
    types = np.unique(labels)
    centroids = np.stack([zc[labels == t].mean(0) for t in types])  # (T, dim)

    # Peak -> centroid distances (per-centroid loop keeps memory small)
    Dm = np.empty((zp.shape[0], len(types)), dtype=np.float32)
    for ti in range(len(types)):
        Dm[:, ti] = np.linalg.norm(zp - centroids[ti], axis=1)

    # Specificity: how much closer to t than to the mean of the other centroids
    spec = Dm.mean(axis=1, keepdims=True) - Dm   # high -> specific to t

    rows = []
    for ti, t in enumerate(types):
        nearest = np.argsort(Dm[:, ti])[:args.topk]
        specific = np.argsort(-spec[:, ti])[:args.topk]
        for rank, (jn, js) in enumerate(zip(nearest, specific), 1):
            row = {"cell_type": t, "rank": rank,
                   "nearest_peak": peak_names[jn], "nearest_dist": float(Dm[jn, ti]),
                   "specific_peak": peak_names[js], "spec_score": float(spec[js, ti])}
            for c in annot_cols:
                row[f"nearest::{c}"] = var_kept.iloc[jn][c]
                row[f"specific::{c}"] = var_kept.iloc[js][c]
            rows.append(row)

    out = pd.DataFrame(rows)
    out_path = f"{d}/centroid_nearest_peaks.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(out)} rows)\n")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out.groupby("cell_type").head(5).to_string(index=False))


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import scipy.sparse as sp

DATA_PATH = "data/hematopoiesis.h5ad"

adata = sc.read(DATA_PATH)

# Binarize
X_binary = (adata.X > 0).astype(int)
if sp.issparse(X_binary):
    X_binary_sparse = X_binary.tocsr()
else:
    X_binary_sparse = sp.csr_matrix(X_binary)

n_cells, n_peaks = X_binary_sparse.shape
n_edges = X_binary_sparse.nnz
total_entries = n_cells * n_peaks
density = n_edges / total_entries

print(f"\nBinary Matrix Statistics:")
print(f"Cells (N):         {n_cells:,}")
print(f"Peaks (K):         {n_peaks:,}")
print(f"Edges (ones):      {n_edges:,}")
print(f"Total entries:     {total_entries:,}")
print(f"Density:           {density:.4f} ({density*100:.1f}%)")
print(f"Sparsity:          {1 - density:.4f} ({(1-density)*100:.1f}%)")

peaks_per_cell = np.array(X_binary_sparse.sum(axis=1)).flatten()
cells_per_peak = np.array(X_binary_sparse.sum(axis=0)).flatten()

print(f"\nPeaks per cell:    mean={peaks_per_cell.mean():.0f}, "
      f"median={np.median(peaks_per_cell):.0f}, "
      f"min={peaks_per_cell.min()}, max={peaks_per_cell.max()}")
print(f"Cells per peak:    mean={cells_per_peak.mean():.0f}, "
      f"median={np.median(cells_per_peak):.0f}, "
      f"min={cells_per_peak.min()}, max={cells_per_peak.max()}")

# Figure 2: Sparsity heatmap (subsampled, sorted by marginals)
np.random.seed(42)
n_sub_cells = 200
n_sub_peaks = 500

# Subsample random cells and peaks
cell_idx = np.random.choice(n_cells, n_sub_cells, replace=False)
peak_idx = np.random.choice(n_peaks, n_sub_peaks, replace=False)
sub_matrix = X_binary_sparse[cell_idx][:, peak_idx].toarray()

# Sort rows by row sum (descending: densest cells at the top)
row_order = np.argsort(-sub_matrix.sum(axis=1))
sub_matrix = sub_matrix[row_order]

# Sort columns by column sum (descending: most common peaks on the left)
col_order = np.argsort(-sub_matrix.sum(axis=0))
sub_matrix = sub_matrix[:, col_order]

fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(sub_matrix, aspect='auto', cmap='Greys', interpolation='none')
ax.set_xlabel("Peaks (sorted by frequency)", fontsize=12)
ax.set_ylabel("Cells (sorted by total accessibility)", fontsize=12)
ax.set_title("Binarized cell-by-peak matrix (200 cells × 500 peaks)", fontsize=13)
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.savefig("fig2_sparsity_heatmap2.png", dpi=300, bbox_inches='tight')
plt.savefig("fig2_sparsity_heatmap2.pdf", bbox_inches='tight')
plt.close()


# Figure 3: Marginal distributions

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: peaks per cell
axes[0].hist(peaks_per_cell, bins=60, color='steelblue', edgecolor='white', linewidth=0.5)
axes[0].set_xlabel("Number of accessible peaks", fontsize=12)
axes[0].set_ylabel("Number of cells", fontsize=12)
axes[0].set_title("Accessible peaks per cell", fontsize=13)
axes[0].axvline(peaks_per_cell.mean(), color='darkred', linestyle='--', linewidth=1.5,
                label=f'Mean = {peaks_per_cell.mean():.0f}')
axes[0].legend(fontsize=10)

# Right: cells per peak
axes[1].hist(cells_per_peak, bins=60, color='coral', edgecolor='white', linewidth=0.5)
axes[1].set_xlabel("Number of cells", fontsize=12)
axes[1].set_ylabel("Number of peaks", fontsize=12)
axes[1].set_title("Cells per peak", fontsize=13)
axes[1].axvline(cells_per_peak.mean(), color='darkred', linestyle='--', linewidth=1.5,
                label=f'Mean = {cells_per_peak.mean():.0f}')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig("fig3_marginal_distributionsFinal.png", dpi=300, bbox_inches='tight')
plt.savefig("fig3_marginal_distributionsFinal.pdf", bbox_inches='tight')
plt.close()


# Figure 4: Chromosome distribution of peaks

if 'chr' in adata.var.columns:
    chr_counts = adata.var['chr'].value_counts()

    # Sort chromosomes naturally (chr1, chr2, ..., chr22, chrX, chrY)
    def chr_sort_key(c):
        c = str(c).replace('chr', '')
        if c == 'X': return 23
        if c == 'Y': return 24
        if c == 'M' or c == 'MT': return 25
        try: return int(c)
        except: return 99

    chr_order = sorted(chr_counts.index, key=chr_sort_key)
    chr_counts = chr_counts[chr_order]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(range(len(chr_counts)), chr_counts.values, color='steelblue',
           edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(chr_counts)))
    ax.set_xticklabels(chr_counts.index, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel("Chromosome", fontsize=12)
    ax.set_ylabel("Number of peaks", fontsize=12)
    ax.set_title("Distribution of peaks across chromosomes", fontsize=13)
    plt.tight_layout()
    plt.savefig("fig_chr_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig("fig_chr_distribution.pdf", bbox_inches='tight')
    plt.close()


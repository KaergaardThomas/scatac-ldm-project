"""
ldm.py — Latent Distance Model for scATAC-seq bipartite graph data.

Model
-----
Each cell i is assigned an embedding z_i ∈ R^d and a scalar intercept ψ_i.
Each peak j is assigned an embedding z_j ∈ R^d and a scalar intercept ω_j.

The log-odds of observing a link between cell i and peak j is:

    η_ij = ψ_i + ω_j − ‖z_i − z_j‖₂

The probability of accessibility is:

    π_ij = σ(η_ij)

where σ is the sigmoid function.

Note
----
False-negative filtering is deliberately omitted following the SIMBA convention.
Zeros in the training set include genuine negatives and technical dropouts alike;
this choice is discussed in the project report.
"""

import torch
import torch.nn as nn


class LDM(nn.Module):
    """
    Latent Distance Model for a bipartite cell-peak graph.

    Parameters
    ----------
    n_cells : int
        Number of cells (rows of the accessibility matrix).
    n_peaks : int
        Number of genomic peaks (columns of the accessibility matrix).
    latent_dim : int
        Dimensionality d of the shared latent embedding space.
    """

    def __init__(self, n_cells: int, n_peaks: int, latent_dim: int = 8):
        super().__init__()
        self.n_cells = n_cells
        self.n_peaks = n_peaks
        self.latent_dim = latent_dim

        # Cell embeddings: z_i ∈ R^d
        self.z_i = nn.Embedding(n_cells, latent_dim)
        # Peak embeddings: z_j ∈ R^d
        self.z_j = nn.Embedding(n_peaks, latent_dim)
        # Cell intercepts: ψ_i ∈ R
        self.psi = nn.Embedding(n_cells, 1)
        # Peak intercepts: ω_j ∈ R
        self.omega = nn.Embedding(n_peaks, 1)

        nn.init.normal_(self.z_i.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.z_j.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.psi.weight)
        nn.init.zeros_(self.omega.weight)

    def forward(
        self,
        cell_idx: torch.Tensor,
        peak_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute predicted log-odds η_ij for a batch of (cell, peak) pairs.

        Parameters
        ----------
        cell_idx : LongTensor of shape (B,)
        peak_idx : LongTensor of shape (B,)

        Returns
        -------
        eta : FloatTensor of shape (B,)
            η_ij = ψ_i + ω_j − ‖z_i − z_j‖₂
        """
        z_i = self.z_i(cell_idx)                 # (B, d)
        z_j = self.z_j(peak_idx)                 # (B, d)
        psi = self.psi(cell_idx).squeeze(-1)      # (B,)
        omega = self.omega(peak_idx).squeeze(-1)  # (B,)

        dist = torch.norm(z_i - z_j, p=2, dim=-1)  # (B,)
        eta = psi + omega - dist                     # (B,)
        return eta

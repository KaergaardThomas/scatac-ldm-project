"""
ldm.py — Latent Distance Model and Null Model for scATAC-seq bipartite graph data.

Models
------
LDM (full model):
    Each cell i has embedding z_i ∈ R^d and intercept ψ_i.
    Each peak j has embedding z_j ∈ R^d and intercept ω_j.

    Log-odds:  η_ij = ψ_i + ω_j − ‖z_i − z_j‖₂
    Probability: π_ij = σ(η_ij)

NullLDM (intercepts-only baseline):
    Same intercepts as the full model, but no distance term.

    Log-odds:  η⁰_ij = ψ_i + ω_j
    Probability: π⁰_ij = σ(η⁰_ij)

    The null model captures only the marginal accessibility rates per cell
    and per peak, without any geometric structure. Comparing held-out
    log-likelihood of LDM vs NullLDM quantifies how much explanatory
    power the latent distance term contributes (nested model comparison,
    following Hoff et al., 2002).

Distance term
-------------
The Euclidean norm ‖z_i − z_j‖₂ matches Equation (1) of the report and the
distance parameterisation of Hoff et al. (2002). It is computed as
    ‖z_i − z_j‖₂ = sqrt( Σ_d (z_i − z_j)_d² + ε )
The small ε guards the gradient of the square root at z_i = z_j, where the
plain Euclidean norm has an undefined (0/0) gradient that otherwise produces
NaNs during optimisation. With ε = 1e-8 the value is indistinguishable from
the true norm for any non-degenerate separation.

Note
----
False-negative filtering is deliberately omitted at training time following
the SIMBA convention. Zeros in the training set include genuine negatives and
technical dropouts alike; this choice is discussed in the project report.
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
    eps : float
        Numerical-stability constant added under the square root of the
        Euclidean distance to keep its gradient finite at zero separation.
    """

    def __init__(self, n_cells: int, n_peaks: int, latent_dim: int = 8,
                 eps: float = 1e-8, use_intercepts: bool = True,
                 init_std: float = 0.1):
        super().__init__()
        self.n_cells = n_cells
        self.n_peaks = n_peaks
        self.latent_dim = latent_dim
        self.eps = eps
        self.use_intercepts = use_intercepts

        # Cell embeddings: z_i ∈ R^d  and  peak embeddings: z_j ∈ R^d
        self.z_i = nn.Embedding(n_cells, latent_dim)
        self.z_j = nn.Embedding(n_peaks, latent_dim)
        nn.init.normal_(self.z_i.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.z_j.weight, mean=0.0, std=init_std)

        if use_intercepts:
            # Per-node intercepts ψ_i, ω_j — capture marginal cell/peak rates.
            self.psi = nn.Embedding(n_cells, 1)
            self.omega = nn.Embedding(n_peaks, 1)
            nn.init.zeros_(self.psi.weight)
            nn.init.zeros_(self.omega.weight)
        else:
            # Ablation: a single global bias replaces the per-node intercepts,
            # so the marginal-rate "shortcut" is removed and the distance term
            # has to carry the signal. The scalar keeps probabilities
            # calibratable (η = β₀ − ‖z_i − z_j‖, so π is not capped at 0.5).
            self.bias = nn.Parameter(torch.zeros(1))

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
        z_i = self.z_i(cell_idx)                  # (B, d)
        z_j = self.z_j(peak_idx)                  # (B, d)

        # Euclidean distance with epsilon under the root for a finite gradient
        sq_dist = torch.sum((z_i - z_j) ** 2, dim=-1)   # (B,)
        dist = torch.sqrt(sq_dist + self.eps)           # (B,)

        if self.use_intercepts:
            psi = self.psi(cell_idx).squeeze(-1)      # (B,)
            omega = self.omega(peak_idx).squeeze(-1)  # (B,)
            return psi + omega - dist                 # (B,)
        return self.bias - dist                       # (B,)


class NullLDM(nn.Module):
    """
    Null model for nested comparison with LDM.

    Contains only per-cell (ψ_i) and per-peak (ω_j) intercepts — no
    latent embeddings and no distance term.

    Log-odds:  η⁰_ij = ψ_i + ω_j

    This model captures the marginal accessibility rates of each cell and
    each peak, but encodes no geometric structure. By comparing the
    held-out log-likelihood (or AUC-PR) of the full LDM against this
    baseline, the contribution of the latent distance term can be directly
    quantified (Hoff et al., 2002).

    Parameters
    ----------
    n_cells : int
    n_peaks : int
    """

    def __init__(self, n_cells: int, n_peaks: int):
        super().__init__()
        self.n_cells = n_cells
        self.n_peaks = n_peaks

        # Cell intercepts: ψ_i ∈ R
        self.psi = nn.Embedding(n_cells, 1)
        # Peak intercepts: ω_j ∈ R
        self.omega = nn.Embedding(n_peaks, 1)

        nn.init.zeros_(self.psi.weight)
        nn.init.zeros_(self.omega.weight)

    def forward(
        self,
        cell_idx: torch.Tensor,
        peak_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute predicted log-odds η⁰_ij for a batch of (cell, peak) pairs.

        Parameters
        ----------
        cell_idx : LongTensor of shape (B,)
        peak_idx : LongTensor of shape (B,)

        Returns
        -------
        eta : FloatTensor of shape (B,)
            η⁰_ij = ψ_i + ω_j
        """
        psi = self.psi(cell_idx).squeeze(-1)      # (B,)
        omega = self.omega(peak_idx).squeeze(-1)  # (B,)
        return psi + omega

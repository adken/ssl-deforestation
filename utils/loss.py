"""
VICReg loss components.

References
----------
Bardes et al. (2022) "VICReg: Variance-Invariance-Covariance Regularization
for Self-Supervised Learning." https://arxiv.org/abs/2105.04906

off_diagonal() adapted from:
https://github.com/facebookresearch/barlowtwins
"""

import torch
import torch.nn.functional as F

# ── Invariance loss ──────────────────────────────────────────────────────────

def sim_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Mean squared error between the two embedding views (invariance term)."""
    return F.mse_loss(z_a, z_b)


# ── Variance loss ────────────────────────────────────────────────────────────

def std_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """
    Penalises collapsed dimensions by encouraging per-dimension std > 1.
    Operates over the batch dimension.
    """
    std_a = torch.sqrt(z_a.var(dim=0) + 1e-4)
    std_b = torch.sqrt(z_b.var(dim=0) + 1e-4)
    return torch.mean(F.relu(1 - std_a)) + torch.mean(F.relu(1 - std_b))


# ── Covariance loss ──────────────────────────────────────────────────────────

def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return a flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m, f"Expected square matrix, got ({n}, {m})"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def cov_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """
    Penalises off-diagonal covariance (decorrelation term).
    Decorrelates the embedding dimensions to prevent information collapse.
    """
    N, D = z_a.shape

    z_a = z_a - z_a.mean(dim=0)
    z_b = z_b - z_b.mean(dim=0)

    cov_a = (z_a.T @ z_a) / (N - 1)
    cov_b = (z_b.T @ z_b) / (N - 1)

    return (
        off_diagonal(cov_a).pow_(2).sum() / D
        + off_diagonal(cov_b).pow_(2).sum() / D
    )

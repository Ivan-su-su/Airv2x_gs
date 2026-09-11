"""Small batched tensor helpers for Gaussian initialization."""

from __future__ import annotations

from typing import Tuple

import torch

EPS = 1.0e-6


def empty_long(device: torch.device) -> torch.Tensor:
    """Return an empty 1-D long tensor on ``device``."""
    return torch.empty((0,), dtype=torch.long, device=device)


def safe_normalize(vectors: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """Normalize ``vectors`` along ``dim`` with a clamped norm.

    Args:
        vectors: Input tensor.
        dim: Reduction dimension.
        eps: Floor for the L2 norm.

    Returns:
        Unit vectors with the same shape as ``vectors``.
    """
    norm = torch.linalg.norm(vectors, dim=dim, keepdim=True).clamp_min(eps)
    return vectors / norm


def principal_eigenvector_2x2(
    cxx: torch.Tensor,
    cxy: torch.Tensor,
    cyy: torch.Tensor,
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Largest-eigenvalue eigenvector of packed 2x2 covariances.

    Args:
        cxx: ``C_xx``, shape ``[N]``.
        cxy: ``C_xy``, shape ``[N]``.
        cyy: ``C_yy``, shape ``[N]``.
        eps: Numerical floor.

    Returns:
        ``(vx, vy, lambda_max)`` each ``[N]``. ``(vx, vy)`` is unit length
        in image ``(x, y) = (col, row)`` coordinates.
    """
    trace = cxx + cyy
    diff = cxx - cyy
    disc = torch.sqrt((diff * diff + 4.0 * cxy * cxy).clamp_min(0.0))
    lam1 = 0.5 * (trace + disc)
    use_xy = cxy.abs() >= (lam1 - cxx).abs()
    vx = torch.where(use_xy, cxy, lam1 - cyy)
    vy = torch.where(use_xy, lam1 - cxx, cxy)
    n = torch.sqrt(vx * vx + vy * vy).clamp_min(eps)
    return vx / n, vy / n, lam1

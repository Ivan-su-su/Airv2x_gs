"""Heatmap PCA orientation. Eigenvectors only; eigenvalues are ignored."""

from __future__ import annotations

from typing import Tuple

import torch
from torch.nn import functional as F

from opencood.models.gaussian_modules_0822.base.utils import (
    EPS,
    principal_eigenvector_2x2,
)

PCA_RADIUS_PX = 40
MIN_PCA_WEIGHT = 1.0e-3


def pca_cell_radius(
    image_hw: Tuple[int, int],
    feature_hw: Tuple[int, int],
    radius_px: float = PCA_RADIUS_PX,
) -> int:
    """Convert a fixed image-space PCA radius into feature cells.

    Args:
        image_hw: Source image ``(H, W)``.
        feature_hw: Heatmap / F90 ``(h, w)``.
        radius_px: Half-width in source-image pixels.

    Returns:
        Integer cell radius ``max(1, round(radius_px / stride))`` where
        ``stride = image_H / feature_h``.
    """
    stride = float(image_hw[0]) / float(feature_hw[0])
    return max(1, int(round(float(radius_px) / stride)))


def heatmap_pca_direction(
    p_fg: torch.Tensor,
    view_index: torch.Tensor,
    y_indices: torch.Tensor,
    x_indices: torch.Tensor,
    radius: int,
    min_weight: float = MIN_PCA_WEIGHT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Soft-weighted PCA on a local ``p_fg`` patch around each seed.

    Weights are the raw local foreground probabilities (no hard FG mask).
    The principal eigenvector is the tangent direction in feature-cell
    ``(x, y)`` coordinates. Eigenvalues are discarded.

    Args:
        p_fg: ``[N, H, W]``.
        view_index: Seed camera index ``[M]``.
        y_indices: Seed rows ``[M]``.
        x_indices: Seed columns ``[M]``.
        radius: Half-width of the square patch in feature cells.
        min_weight: Fallback to identity ``(1, 0)`` when mass is below this.

    Returns:
        ``(v_major, v_minor)`` each ``[M, 2]``. Identity if mass is too small.
    """
    n_seeds = int(view_index.shape[0])
    device = p_fg.device
    dtype = p_fg.dtype
    if n_seeds == 0:
        empty = p_fg.new_empty((0, 2))
        return empty, empty

    kernel = int(radius) * 2 + 1
    patches = F.unfold(
        p_fg.unsqueeze(1).to(dtype=dtype),
        kernel_size=kernel,
        padding=int(radius),
    )
    _, feat_h, feat_w = p_fg.shape
    linear = y_indices.long() * int(feat_w) + x_indices.long()
    weights = patches[view_index.long(), :, linear].clamp_min(0.0)
    yy, xx = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=device, dtype=dtype),
        torch.arange(-radius, radius + 1, device=device, dtype=dtype),
        indexing="ij",
    )
    rel_x = xx.reshape(1, -1)
    rel_y = yy.reshape(1, -1)
    wsum = weights.sum(dim=-1, keepdim=True)
    fallback = (wsum.squeeze(-1) < float(min_weight))
    wn = weights / wsum.clamp_min(EPS)
    cx = (wn * rel_x).sum(dim=-1, keepdim=True)
    cy = (wn * rel_y).sum(dim=-1, keepdim=True)
    dx = rel_x - cx
    dy = rel_y - cy
    cxx = (wn * dx * dx).sum(dim=-1)
    cyy = (wn * dy * dy).sum(dim=-1)
    cxy = (wn * dx * dy).sum(dim=-1)
    vx, vy, lam = principal_eigenvector_2x2(cxx, cxy, cyy)
    identity = fallback | (lam < EPS)
    vx = torch.where(identity, torch.ones_like(vx), vx)
    vy = torch.where(identity, torch.zeros_like(vy), vy)
    v_major = torch.stack([vx, vy], dim=-1)
    v_minor = torch.stack([-vy, vx], dim=-1)
    return v_major, v_minor

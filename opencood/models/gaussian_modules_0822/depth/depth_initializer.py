"""Agent-specific depth aggregation for a single Gaussian per seed.

Vehicle:
    local window around the peak (``peak +/- 10 m``) for both mean and variance.

RSU:
    global softmax expectation for the center; variance in a window of
    the same radius centered on that global mean.

Drone:
    continuous predicted depth; fixed ``sigma_D = 0.5 m``.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.nn import functional as F

from opencood.models.gaussian_modules_0822.base.utils import EPS
from opencood.models.gaussian_modules_0822.depth.uncertainty import (
    LOCAL_WINDOW_M,
    SIGMA_D_DRONE,
)


def _gather_logits(
    depth_logits: torch.Tensor,
    view_index: torch.Tensor,
    y_indices: torch.Tensor,
    x_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather per-seed categorical logits ``[M, D]``."""
    return depth_logits[view_index.long(), :, y_indices.long(), x_indices.long()]


def _local_mask(z_bins: torch.Tensor, center_z: torch.Tensor, window_m: float) -> torch.Tensor:
    """Boolean mask ``[M, D]`` of bins within ``window_m`` of ``center_z``."""
    return (z_bins.unsqueeze(0) - center_z.unsqueeze(1)).abs() <= float(window_m)


def _renormalize(prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero outside ``mask`` and renormalize over D. Fallback to one-hot peak."""
    masked = prob * mask.to(dtype=prob.dtype)
    mass = masked.sum(dim=-1, keepdim=True)
    local = masked / mass.clamp_min(EPS)
    empty = mass.squeeze(-1) < EPS
    if bool(empty.any()):
        peak = torch.argmax(prob, dim=-1)
        one_hot = F.one_hot(peak, num_classes=int(prob.shape[-1])).to(dtype=prob.dtype)
        local = torch.where(empty.unsqueeze(-1), one_hot, local)
    return local


def categorical_depth_moments(
    depth_logits: torch.Tensor,
    z_bins: torch.Tensor,
    view_index: torch.Tensor,
    y_indices: torch.Tensor,
    x_indices: torch.Tensor,
    agent: str,
    window_m: float = LOCAL_WINDOW_M,
) -> Dict[str, torch.Tensor]:
    """Vehicle / RSU depth mean and variance at FG seeds.

    Vehicle uses a peak-centered window for both the center and the
    variance. RSU uses the global categorical mean as the center and
    the same metric radius around that mean for the variance.

    Args:
        depth_logits: ``[N, D, H, W]``.
        z_bins: Metric bin centers ``[D]``.
        view_index: ``[M]``.
        y_indices: ``[M]``.
        x_indices: ``[M]``.
        agent: ``vehicle`` or ``rsu``.
        window_m: Local window radius in meters.

    Returns:
        ``depth_mean [M]``, ``depth_var [M]``, ``peak_z [M]``.
    """
    agent = str(agent).lower()
    if agent not in ("vehicle", "rsu"):
        raise ValueError(f"categorical_depth_moments expects vehicle/rsu, got {agent}")
    n_seeds = int(view_index.shape[0])
    bins = z_bins.reshape(-1).to(dtype=depth_logits.dtype, device=depth_logits.device)
    if n_seeds == 0:
        empty = depth_logits.new_empty((0,))
        return {"depth_mean": empty, "depth_var": empty, "peak_z": empty}

    logits = _gather_logits(depth_logits, view_index, y_indices, x_indices)
    if int(logits.shape[-1]) != int(bins.shape[0]):
        raise ValueError(
            f"depth_logits D={logits.shape[-1]} != z_bins {int(bins.shape[0])}"
        )
    prob = torch.softmax(logits, dim=-1)
    peak_z = bins[torch.argmax(prob, dim=-1)]
    z_global = (prob * bins.unsqueeze(0)).sum(dim=-1)
    if agent == "vehicle":
        local_prob = _renormalize(prob, _local_mask(bins, peak_z, window_m))
        depth_mean = (local_prob * bins.unsqueeze(0)).sum(dim=-1)
    else:
        masked = prob * _local_mask(bins, z_global, window_m).to(dtype=prob.dtype)
        local_prob = masked / masked.sum(dim=-1, keepdim=True).clamp_min(EPS)
        depth_mean = z_global
    depth_var = (
        local_prob * (bins.unsqueeze(0) - depth_mean.unsqueeze(1)).pow(2)
    ).sum(dim=-1)
    return {
        "depth_mean": depth_mean,
        "depth_var": depth_var.clamp_min(0.0),
        "peak_z": peak_z,
    }


def drone_depth_moments(
    depth_z: torch.Tensor,
    view_index: torch.Tensor,
    y_indices: torch.Tensor,
    x_indices: torch.Tensor,
    sigma_d: float = SIGMA_D_DRONE,
) -> Dict[str, torch.Tensor]:
    """Continuous drone depth: predicted ``z``, fixed ``sigma_D``.

    Args:
        depth_z: ``[N, H, W]`` or ``[N, 1, H, W]`` predicted optical depth.
        view_index: ``[M]``.
        y_indices: ``[M]``.
        x_indices: ``[M]``.
        sigma_d: Fixed ray scale in meters.

    Returns:
        ``depth_mean [M]``, ``depth_var [M]``, ``peak_z [M]`` (same as mean).
    """
    if depth_z.dim() == 4:
        depth_z = depth_z[:, 0]
    if depth_z.dim() != 3:
        raise ValueError(f"drone depth_z must be [N,H,W], got {tuple(depth_z.shape)}")
    n_seeds = int(view_index.shape[0])
    if n_seeds == 0:
        empty = depth_z.new_empty((0,))
        return {"depth_mean": empty, "depth_var": empty, "peak_z": empty}
    z = depth_z[view_index.long(), y_indices.long(), x_indices.long()]
    var = torch.full_like(z, float(sigma_d) ** 2)
    return {"depth_mean": z, "depth_var": var, "peak_z": z}

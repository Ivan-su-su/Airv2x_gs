"""Foreground heatmap seeds. Location only; no NMS / clustering / top-k."""

from __future__ import annotations

from typing import Dict

import torch

from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    PRIMARY_OBJECTNESS_THRESHOLD,
)

FG_THRESHOLD = float(PRIMARY_OBJECTNESS_THRESHOLD)


def foreground_probability(heatmap_logits: torch.Tensor) -> torch.Tensor:
    """Softmax foreground channel from 2-class heatmap logits.

    Args:
        heatmap_logits: ``[N, 2, H, W]``.

    Returns:
        ``p_fg`` of shape ``[N, H, W]``.
    """
    if heatmap_logits.dim() != 4 or int(heatmap_logits.shape[1]) != 2:
        raise ValueError(
            f"heatmap_logits must be [N, 2, H, W], got {tuple(heatmap_logits.shape)}"
        )
    return torch.softmax(heatmap_logits, dim=1)[:, 1]


def generate_seeds(
    p_fg: torch.Tensor,
    features: torch.Tensor,
    fg_threshold: float = FG_THRESHOLD,
) -> Dict[str, torch.Tensor]:
    """Emit one seed at every cell with ``p_fg > threshold``.

    Args:
        p_fg: ``[N, H, W]`` foreground probability.
        features: ``[N, C, H, W]`` image features aligned with ``p_fg``.
        fg_threshold: Occupancy cut, default ``0.3``.

    Returns:
        Dict with:
            ``view_index [M]``, ``y_indices [M]``, ``x_indices [M]``,
            ``uv [M, 2]`` feature ``(x, y)``, ``feature [M, C]``,
            ``p_fg [M]``.
    """
    if p_fg.dim() != 3:
        raise ValueError(f"p_fg must be [N, H, W], got {tuple(p_fg.shape)}")
    if features.dim() != 4:
        raise ValueError(f"features must be [N, C, H, W], got {tuple(features.shape)}")
    if tuple(p_fg.shape) != (int(features.shape[0]), int(features.shape[2]), int(features.shape[3])):
        raise ValueError(
            f"p_fg {tuple(p_fg.shape)} vs features {tuple(features.shape)}"
        )

    mask = p_fg > float(fg_threshold)
    selected = torch.nonzero(mask, as_tuple=False)
    device = p_fg.device
    n_channels = int(features.shape[1])
    if selected.numel() == 0:
        return {
            "view_index": torch.empty((0,), dtype=torch.long, device=device),
            "y_indices": torch.empty((0,), dtype=torch.long, device=device),
            "x_indices": torch.empty((0,), dtype=torch.long, device=device),
            "uv": torch.empty((0, 2), dtype=features.dtype, device=device),
            "feature": torch.empty((0, n_channels), dtype=features.dtype, device=device),
            "p_fg": torch.empty((0,), dtype=p_fg.dtype, device=device),
        }

    view_index = selected[:, 0].long()
    y_indices = selected[:, 1].long()
    x_indices = selected[:, 2].long()
    feat = features[view_index, :, y_indices, x_indices]
    return {
        "view_index": view_index,
        "y_indices": y_indices,
        "x_indices": x_indices,
        "uv": torch.stack([x_indices.to(features.dtype), y_indices.to(features.dtype)], dim=-1),
        "feature": feat,
        "p_fg": p_fg[view_index, y_indices, x_indices],
    }

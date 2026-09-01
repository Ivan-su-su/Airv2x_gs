"""Image-space support covariance. Diagnostic only; not in the P1 train graph.

Map background id 0 → ``BACKGROUND_LABEL_ID`` before calling.
Uses fixed major/minor scales.

Removed from this P1 copy (still present in the old
``uncertain_gaussian_modules`` path, not here):

- ``lidar_coverage_mask`` / uncovered-LiDAR filtering
- ``resize_bool_mask`` (only used to resize that LiDAR mask)
- ``sampled_strength`` modulation (old ``strength=0`` collapsed axes)

No confidence scaling. No LiDAR. No training loss.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

BACKGROUND_LABEL_ID = -1


def reshape_label_map(label_map: torch.Tensor, num_views: int) -> torch.Tensor:
    """Validate ``[B, num_views, H, W]`` and return long ids."""
    if label_map.ndim != 4:
        raise ValueError("label_map must have shape [B, num_views, H, W].")
    if int(label_map.shape[1]) != num_views:
        raise ValueError(
            f"label_map num_views mismatch: got {label_map.shape[1]}, expected {num_views}."
        )
    return label_map.long()


def feature_indices_to_lss_normalized_coords(
    x_indices: torch.Tensor,
    y_indices: torch.Tensor,
    feature_hw: Tuple[int, int],
    image_hw: Tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map feature-grid indices to LSS image-plane normalized coordinates."""
    feature_height, feature_width = feature_hw
    image_height, image_width = image_hw
    x_indices = x_indices.to(dtype)
    y_indices = y_indices.to(dtype)
    if feature_width <= 1:
        pixel_x = torch.zeros_like(x_indices)
    else:
        pixel_x = x_indices * (float(image_width - 1) / float(feature_width - 1))
    if feature_height <= 1:
        pixel_y = torch.zeros_like(y_indices)
    else:
        pixel_y = y_indices * (float(image_height - 1) / float(feature_height - 1))
    return torch.stack(
        [pixel_x / float(image_width), pixel_y / float(image_height)], dim=-1
    )


def select_foreground_label_points(
    label_map: torch.Tensor,
    image_hw: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    """Select all foreground label cells. Background = ``BACKGROUND_LABEL_ID``."""
    _, _, feature_height, feature_width = label_map.shape
    selected = torch.nonzero(label_map.ne(BACKGROUND_LABEL_ID), as_tuple=False)
    if selected.numel() == 0:
        device = label_map.device
        empty_long = torch.empty((0,), dtype=torch.long, device=device)
        empty_coords = torch.empty((0, 2), dtype=torch.float32, device=device)
        return {
            "local_agent_ids": empty_long,
            "view_ids": empty_long,
            "y_indices": empty_long,
            "x_indices": empty_long,
            "labels": empty_long,
            "normalized_coords": empty_coords,
        }
    local_agent_ids = selected[:, 0].long()
    view_ids = selected[:, 1].long()
    y_indices = selected[:, 2].long()
    x_indices = selected[:, 3].long()
    normalized_coords = feature_indices_to_lss_normalized_coords(
        x_indices=x_indices,
        y_indices=y_indices,
        feature_hw=(feature_height, feature_width),
        image_hw=image_hw,
        dtype=torch.float32,
    )
    return {
        "local_agent_ids": local_agent_ids,
        "view_ids": view_ids,
        "y_indices": y_indices,
        "x_indices": x_indices,
        "labels": label_map[local_agent_ids, view_ids, y_indices, x_indices].long(),
        "normalized_coords": normalized_coords,
    }


def estimate_local_patch_covariance(
    label_map: torch.Tensor,
    point_labels: torch.Tensor,
    point_coords: torch.Tensor,
    local_agent_ids: torch.Tensor,
    view_ids: torch.Tensor,
    x_indices: torch.Tensor,
    y_indices: torch.Tensor,
    image_hw: Tuple[int, int],
    patch_size: int,
    major_scale: float,
    minor_scale: float,
    eps: float,
) -> torch.Tensor:
    """Fixed-axis covariance from local same-label support.

    ``major = major_scale``, ``minor = minor_scale``. No strength modulation.

    Returns:
        ``[P, 2, 2]`` covariances.
    """
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer.")
    if point_coords.numel() == 0:
        return point_coords.new_empty((0, 2, 2))

    _, num_views, feature_height, feature_width = label_map.shape
    flat_views = label_map.shape[0] * num_views
    label_map_flat = label_map.view(flat_views, feature_height, feature_width)
    pad = patch_size // 2
    label_tensor = F.pad(
        label_map_flat.unsqueeze(1).float(),
        (pad, pad, pad, pad),
        value=float(BACKGROUND_LABEL_ID),
    )
    x_grid = torch.arange(
        feature_width, device=label_map.device, dtype=torch.float32
    ).view(1, 1, feature_width).expand(flat_views, feature_height, feature_width)
    y_grid = torch.arange(
        feature_height, device=label_map.device, dtype=torch.float32
    ).view(1, feature_height, 1).expand(flat_views, feature_height, feature_width)
    x_tensor = F.pad(x_grid.unsqueeze(1), (pad, pad, pad, pad), value=0.0)
    y_tensor = F.pad(y_grid.unsqueeze(1), (pad, pad, pad, pad), value=0.0)

    label_patches = F.unfold(label_tensor, kernel_size=patch_size).transpose(1, 2)
    x_patches = F.unfold(x_tensor, kernel_size=patch_size).transpose(1, 2)
    y_patches = F.unfold(y_tensor, kernel_size=patch_size).transpose(1, 2)

    flat_view_ids = local_agent_ids.long() * num_views + view_ids.long()
    linear_indices = y_indices.long() * feature_width + x_indices.long()
    local_labels = label_patches[flat_view_ids, linear_indices].long()
    same_label_mask = local_labels.eq(point_labels.long().unsqueeze(1))
    same_label_mask = same_label_mask & local_labels.ne(BACKGROUND_LABEL_ID)

    local_coords = feature_indices_to_lss_normalized_coords(
        x_indices=x_patches[flat_view_ids, linear_indices],
        y_indices=y_patches[flat_view_ids, linear_indices],
        feature_hw=(feature_height, feature_width),
        image_hw=image_hw,
        dtype=point_coords.dtype,
    )
    weights = same_label_mask.to(point_coords.dtype)
    weight_sum = weights.sum(dim=1, keepdim=True)
    local_centers = (weights.unsqueeze(-1) * local_coords).sum(dim=1) / weight_sum.clamp_min(
        1.0
    )
    local_centers = torch.where(weight_sum > 0, local_centers, point_coords)

    trend_vectors = local_centers - point_coords
    trend_norm = torch.norm(trend_vectors, dim=-1, keepdim=True)
    default_unit = torch.zeros_like(trend_vectors)
    default_unit[:, 0] = 1.0
    unit_vectors = torch.where(
        trend_norm > eps,
        trend_vectors / trend_norm.clamp_min(eps),
        default_unit,
    )
    ortho_vectors = torch.cat([-unit_vectors[:, 1:2], unit_vectors[:, 0:1]], dim=-1)
    major_cov = (major_scale ** 2) * torch.matmul(
        unit_vectors.unsqueeze(-1), unit_vectors.unsqueeze(-2)
    )
    minor_cov = (minor_scale ** 2) * torch.matmul(
        ortho_vectors.unsqueeze(-1), ortho_vectors.unsqueeze(-2)
    )
    return major_cov + minor_cov

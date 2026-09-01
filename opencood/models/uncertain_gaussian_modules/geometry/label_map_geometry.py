from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

BACKGROUND_LABEL_ID = -1


def reshape_label_map(label_map: torch.Tensor, num_views: int) -> torch.Tensor:
    """Validate one agent-local categorical label map."""
    if label_map.ndim != 4:
        raise ValueError("label_map must have shape [B, num_views, H, W].")
    if int(label_map.shape[1]) != num_views:
        raise ValueError(
            f"label_map num_views mismatch: got {label_map.shape[1]}, expected {num_views}."
        )
    return label_map.long()


def resize_bool_mask(
    mask: Optional[torch.Tensor], target_hw: Tuple[int, int]
) -> Optional[torch.Tensor]:
    """Resize a boolean mask to the requested feature-map size."""
    if mask is None:
        return None
    if mask.shape[-2:] == target_hw:
        return mask.bool()
    if mask.ndim != 4:
        raise ValueError("mask must have shape [B, num_views, H, W].")
    batch_size, num_views = mask.shape[:2]
    resized_mask = F.interpolate(
        mask.float().view(batch_size * num_views, 1, *mask.shape[-2:]),
        size=target_hw,
        mode="nearest",
    )
    return resized_mask.view(batch_size, num_views, *target_hw).bool()


def feature_indices_to_lss_normalized_coords(
    x_indices: torch.Tensor,
    y_indices: torch.Tensor,
    feature_hw: Tuple[int, int],
    image_hw: Tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map feature-grid indices to the LSS image-plane normalized coordinates."""
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
    normalized_x = pixel_x / float(image_width)
    normalized_y = pixel_y / float(image_height)
    return torch.stack([normalized_x, normalized_y], dim=-1)


def lss_normalized_coords_to_feature_grid(
    normalized_coords: torch.Tensor,
    feature_hw: Tuple[int, int],
    image_hw: Tuple[int, int],
) -> torch.Tensor:
    """Map LSS image-plane normalized coordinates back to the feature grid."""
    feature_height, feature_width = feature_hw
    image_height, image_width = image_hw
    grid_x = normalized_coords[:, 0].clamp(0.0, 1.0) * float(image_width)
    grid_y = normalized_coords[:, 1].clamp(0.0, 1.0) * float(image_height)
    if image_width <= 1 or feature_width <= 1:
        feature_x = torch.zeros_like(grid_x)
    else:
        feature_x = grid_x * (float(feature_width - 1) / float(image_width - 1))
    if image_height <= 1 or feature_height <= 1:
        feature_y = torch.zeros_like(grid_y)
    else:
        feature_y = grid_y * (float(feature_height - 1) / float(image_height - 1))
    return torch.stack([feature_x, feature_y], dim=-1)


def select_foreground_label_points(
    label_map: torch.Tensor,
    lidar_coverage_mask: Optional[torch.Tensor],
    image_hw: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    """Select all uncovered foreground label cells in parallel."""
    batch_size, num_views, feature_height, feature_width = label_map.shape
    resized_mask = resize_bool_mask(lidar_coverage_mask, (feature_height, feature_width))
    valid_mask = label_map.ne(BACKGROUND_LABEL_ID)
    if resized_mask is not None:
        valid_mask = valid_mask & (~resized_mask)
    selected = torch.nonzero(valid_mask, as_tuple=False)
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
    sampled_strength: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Estimate a local same-label support covariance around each candidate."""
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("local_patch_size must be a positive odd integer.")
    if point_coords.numel() == 0:
        return point_coords.new_empty((0, 2, 2))

    batch_size, num_views, feature_height, feature_width = label_map.shape
    flat_views = batch_size * num_views
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
    local_x = x_patches[flat_view_ids, linear_indices]
    local_y = y_patches[flat_view_ids, linear_indices]
    same_label_mask = local_labels.eq(point_labels.long().unsqueeze(1))
    same_label_mask = same_label_mask & local_labels.ne(BACKGROUND_LABEL_ID)

    local_coords = feature_indices_to_lss_normalized_coords(
        x_indices=local_x,
        y_indices=local_y,
        feature_hw=(feature_height, feature_width),
        image_hw=image_hw,
        dtype=point_coords.dtype,
    )
    weights = same_label_mask.to(point_coords.dtype)
    weight_sum = weights.sum(dim=1, keepdim=True)
    safe_weight_sum = weight_sum.clamp_min(1.0)
    local_centers = (weights.unsqueeze(-1) * local_coords).sum(dim=1) / safe_weight_sum
    has_support = weight_sum > 0
    local_centers = torch.where(has_support, local_centers, point_coords)

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
    if sampled_strength is None:
        sampled_strength = torch.ones_like(trend_norm)
    major = major_scale * (1.0 + sampled_strength)
    minor = minor_scale * torch.ones_like(major)
    major_cov = (major.square().unsqueeze(-1)) * torch.matmul(
        unit_vectors.unsqueeze(-1), unit_vectors.unsqueeze(-2)
    )
    minor_cov = (minor.square().unsqueeze(-1)) * torch.matmul(
        ortho_vectors.unsqueeze(-1), ortho_vectors.unsqueeze(-2)
    )
    return major_cov + minor_cov

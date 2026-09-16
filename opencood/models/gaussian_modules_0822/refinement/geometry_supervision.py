"""Stage-2 delta-mean auxiliary loss against GT ego points.

No learnable parameters. Reuses Gaussian provenance (uv / view_index),
dataset GT depth in ``imgs[:, 3]``, and ``lift_optical_z_to_lidar``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn.functional as F

from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.base.projection import (
    CAMERA_GEOMETRY_KEY,
    lift_optical_z_to_lidar,
)


def _flatten_imgs(imgs: torch.Tensor) -> torch.Tensor:
    """Return ``[n_flat, C, H, W]`` from ``[B,V,C,H,W]`` or ``[N,C,H,W]``."""
    if imgs.dim() == 5:
        return imgs.reshape(-1, *imgs.shape[-3:])
    if imgs.dim() == 4:
        return imgs
    raise ValueError(f"camera imgs must be 4D/5D, got {tuple(imgs.shape)}")


def _zero_like(gaussians_by_agent: Mapping[str, GaussianSet]) -> torch.Tensor:
    """Differentiable zero attached to a Stage-2 mean tensor."""
    for gs in gaussians_by_agent.values():
        return gs.mean.sum() * 0.0
    return torch.zeros(())


def _sample_gt_depth(
    gs: GaussianSet, imgs_flat: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest-pixel GT optical-z at each Gaussian's augmented uv."""
    depth_maps = imgs_flat[:, 3].to(device=gs.mean.device, dtype=gs.mean.dtype)
    n_flat, height, width = (
        int(depth_maps.shape[0]),
        int(depth_maps.shape[1]),
        int(depth_maps.shape[2]),
    )
    view = gs.view_index.long()
    uv = gs.uv.to(dtype=gs.mean.dtype)
    u_int = torch.round(uv[:, 0]).long().clamp(0, width - 1)
    v_int = torch.round(uv[:, 1]).long().clamp(0, height - 1)
    in_view = (view >= 0) & (view < n_flat)
    safe_view = view.clamp(0, max(n_flat - 1, 0))
    gt_depth = depth_maps[safe_view, v_int, u_int]
    gt_depth = torch.where(in_view, gt_depth, gt_depth.new_full((), float("nan")))
    uv_px = torch.stack(
        [u_int.to(dtype=gs.mean.dtype), v_int.to(dtype=gs.mean.dtype)], dim=-1
    )
    return gt_depth, view, uv_px


def _gt_xyz_ego(
    gs: GaussianSet,
    geometry: Mapping[str, Any],
    uv: torch.Tensor,
    view: torch.Tensor,
    gt_depth: torch.Tensor,
    loss_idx: torch.Tensor,
) -> torch.Tensor:
    """Lift valid GT depth to ego XYZ via existing camera geometry."""
    vi = view[loss_idx]
    uv_i = uv[loss_idx].to(dtype=gs.mean.dtype)
    depth_i = gt_depth[loss_idx]
    dtype = gs.mean.dtype
    xyz_agent, _, _ = lift_optical_z_to_lidar(
        uv_aug=uv_i,
        depth_z=depth_i,
        intrinsic=geometry["intrinsics"][vi].to(dtype=dtype),
        cam2lidar_rot=geometry["cam2lidar_rot"][vi].to(dtype=dtype),
        cam2lidar_trans=geometry["cam2lidar_trans"][vi].to(dtype=dtype),
        post_rot=geometry["post_rots"][vi].to(dtype=dtype),
        post_trans=geometry["post_trans"][vi].to(dtype=dtype),
    )
    transform = geometry["agent_to_ego"][vi].to(dtype=dtype)
    rotation = transform[:, :3, :3]
    translation = transform[:, :3, 3]
    return torch.matmul(rotation, xyz_agent.unsqueeze(-1)).squeeze(-1) + translation


def compute_stage2_delta_mean_loss(
    stage1_gaussians_by_agent: Mapping[str, GaussianSet],
    stage2_gaussians_by_agent: Mapping[str, GaussianSet],
    data_dict: Mapping[str, Any],
    cross_valid_by_agent: Mapping[str, torch.Tensor],
    depth_ranges: Mapping[str, Tuple[float, float]],
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth-L1 on Stage-2 mean residual vs GT ego point, all valid coords.

    Args:
        stage1_gaussians_by_agent: Per-agent Stage-1 Gaussian sets.
        stage2_gaussians_by_agent: Per-agent Stage-2 Gaussian sets.
        data_dict: Forward payload; GT depth is ``imgs`` channel 3.
        cross_valid_by_agent: Stage-2 cross-agent valid mask per query agent.
        depth_ranges: ``{agent: (d_min, d_max)}`` from LSS ``ddiscr``.
        beta: Smooth-L1 transition point.

    Returns:
        Scalar geometry loss, or a differentiable zero if nothing is valid.
    """
    loss_sum = _zero_like(stage2_gaussians_by_agent)
    num_coords = 0
    for agent, s2 in stage2_gaussians_by_agent.items():
        s1 = stage1_gaussians_by_agent.get(agent)
        if s1 is None or s2.n_gaussians == 0:
            continue
        payload = data_dict.get(agent)
        if not isinstance(payload, dict):
            continue
        if s1.uv is None or s1.view_index is None:
            continue
        cam_inputs = payload.get("batch_merged_cam_inputs") or {}
        imgs = cam_inputs.get("imgs")
        if imgs is None or int(imgs.shape[-3]) < 4:
            continue
        geometry = payload.get(CAMERA_GEOMETRY_KEY)
        if geometry is None:
            continue

        imgs_flat = _flatten_imgs(imgs)
        gt_depth, view, uv_px = _sample_gt_depth(s1, imgs_flat)
        d_min, d_max = depth_ranges.get(agent, (float("nan"), float("nan")))
        depth_valid = (
            torch.isfinite(gt_depth) & (gt_depth >= d_min) & (gt_depth <= d_max)
        )
        cross_valid = cross_valid_by_agent.get(agent)
        if cross_valid is None:
            cross_valid = torch.zeros(
                s2.n_gaussians, dtype=torch.bool, device=s2.mean.device
            )
        else:
            cross_valid = cross_valid.to(device=s2.mean.device)
            if int(cross_valid.numel()) != int(s2.n_gaussians):
                continue
        loss_mask = depth_valid & cross_valid
        if int(loss_mask.sum().item()) == 0:
            continue

        loss_idx = loss_mask.nonzero(as_tuple=False).squeeze(1)
        gt_xyz_ego = _gt_xyz_ego(s1, geometry, uv_px, view, gt_depth, loss_idx)
        base_mean = s1.mean.detach()[loss_idx]
        delta_pred = s2.mean[loss_idx] - base_mean
        delta_gt = gt_xyz_ego - base_mean
        loss_sum = loss_sum + F.smooth_l1_loss(
            delta_pred, delta_gt, beta=beta, reduction="sum"
        )
        num_coords += int(loss_idx.numel()) * 3
    return loss_sum / max(num_coords, 1)

"""Depth class targets and evaluation z maps. No loss, no metrics.

R90 classification target: official ``bin_depths`` then stride-4 centers
``[:, 2::4, 2::4]``. Same cell index as heatmap ``(i,j)`` with
``(u,v)=(4j+2, 4i+2)``. Does not call ``CamEncode.get_gt_depth_dist``
and does not build a downsample=4 CamEncode.
"""

from __future__ import annotations

import torch

from opencood.models.gaussian_modules_0822.image_frontend import flatten_camera_images
from opencood.models.gaussian_modules_0822.p1_layout import SPATIAL_STRIDE
from opencood.models.sub_modules.lss_submodule import CamEncode
from opencood.utils.camera_utils import bin_depths


def _camera_z_from_imgs(imgs: torch.Tensor) -> torch.Tensor:
    """Clone optical-axis z from the depth channel. ``[N, H, W]``."""
    flat_imgs, _ = flatten_camera_images(imgs)
    if int(flat_imgs.shape[1]) < 4:
        raise ValueError(
            f"depth target requires a depth channel in imgs, got C={flat_imgs.shape[1]}"
        )
    return flat_imgs[:, 3, :, :].clone()


def build_depth_class_target(camencode: CamEncode, imgs: torch.Tensor) -> torch.Tensor:
    """Official LID/UD bins at aligned block centers.

    Clones camera-z, applies official ``clamp_max_(d_max)``, then
    ``camera_utils.bin_depths`` with ``target=camencode.training``.

    Args:
        camencode: Live encoder (``d_min``, ``d_max``, ``num_bins``, ``mode``,
            ``training``). ``downsample`` is not used for spatial sampling.
        imgs: ``[B_a, V, C, H, W]`` or ``[N, C, H, W]``, C>=4.

    Returns:
        ``depth_z_indices_gt`` of shape ``[N, 90, 160]``.
    """
    camera_z = _camera_z_from_imgs(imgs)
    torch.clamp_max_(camera_z, camencode.d_max)
    depth_indices, _mask = bin_depths(
        camera_z,
        camencode.mode,
        camencode.d_min,
        camencode.d_max,
        camencode.num_bins,
        target=bool(camencode.training),
    )
    offset = SPATIAL_STRIDE // 2
    sampled = depth_indices[:, offset::SPATIAL_STRIDE, offset::SPATIAL_STRIDE]
    return sampled.long()


def extract_camera_z_gt(imgs: torch.Tensor) -> torch.Tensor:
    """Unclamped camera-z at R90 block centers. MAE/RMSE only.

    Args:
        imgs: Camera tensor with depth channel.

    Returns:
        ``camera_z_gt`` of shape ``[N, 90, 160]``.
    """
    camera_z_full = _camera_z_from_imgs(imgs)
    offset = SPATIAL_STRIDE // 2
    return camera_z_full[:, offset::SPATIAL_STRIDE, offset::SPATIAL_STRIDE]

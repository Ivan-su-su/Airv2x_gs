# -*- coding: utf-8 -*-
"""Pairwise transform adapters for AirV2X PyramidFusion and NegoCollab.

AirV2X HEAL / STAMP currently feed PyramidFusion a raw 2x3 slice of
`img_pairwise_t_matrix_collab` (no `normalize_pairwise_tfm`). That behavior
is preserved by `to_airv2x_pyramid_affine`.

Official NegoCollab `ComminPub.receive` / `warp_affine_simple` expects the
normalized affine produced by `normalize_pairwise_tfm`. Use
`to_normalized_affine` on that path only.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.utils.transformation_utils import normalize_pairwise_tfm


def to_airv2x_pyramid_affine(pairwise_t_matrix: Tensor) -> Tensor:
    """Extract the 2x3 affine slice used by current AirV2X PyramidFusion.

    Args:
        pairwise_t_matrix: `[B, L, L, 4, 4]` (or already `[B, L, L, 2, 3]`).

    Returns:
        `[B, L, L, 2, 3]` slice of the xy rotation / translation rows.
    """
    if pairwise_t_matrix.dim() != 5:
        raise ValueError(
            f"Expected 5D pairwise matrix, got shape {tuple(pairwise_t_matrix.shape)}"
        )
    if pairwise_t_matrix.shape[-2:] == (2, 3):
        return pairwise_t_matrix
    if pairwise_t_matrix.shape[-2:] != (4, 4):
        raise ValueError(
            f"Expected 4x4 or 2x3 pairwise matrix, got {tuple(pairwise_t_matrix.shape)}"
        )
    return pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]


def to_normalized_affine(
    pairwise_t_matrix: Tensor,
    feature_hw: Optional[Tuple[int, int]] = None,
    cav_lidar_range: Optional[Sequence[float]] = None,
    discrete_ratio: float = 1.0,
    downsample_rate: float = 1.0,
) -> Tensor:
    """Convert AirV2X 4x4 pairwise matrices to NegoCollab affine grids.

    Official NegoCollab uses physical BEV extents as H/W with
    `fake_voxel_size=1`:

        H = cav_range[4] - cav_range[1]
        W = cav_range[3] - cav_range[0]

    Args:
        pairwise_t_matrix: `[B, L, L, 4, 4]` rigid transforms, source→target
            convention matching AirV2X `get_pairwise_transformation`
            (`T_j^{-1} T_i`, i.e. i → j).
        feature_hw: Unused placeholder kept for call-site clarity. Official
            NegoCollab does **not** pass feature-map pixels here.
        cav_lidar_range: `[x_min, y_min, z_min, x_max, y_max, z_max]`.
        discrete_ratio: Passed to `normalize_pairwise_tfm`. Official value is 1.
        downsample_rate: Passed to `normalize_pairwise_tfm`.

    Returns:
        `[B, L, L, 2, 3]` affine matrices for `F.affine_grid`.
    """
    del feature_hw  # official path uses physical extents, not pixels
    if cav_lidar_range is None:
        raise ValueError("cav_lidar_range is required for NegoCollab normalization")
    height = float(cav_lidar_range[4] - cav_lidar_range[1])
    width = float(cav_lidar_range[3] - cav_lidar_range[0])
    return normalize_pairwise_tfm(
        pairwise_t_matrix,
        height,
        width,
        discrete_ratio,
        downsample_rate,
    )


def warp_with_affine(
    features: Tensor,
    affine_2x3: Tensor,
    align_corners: bool = False,
) -> Tensor:
    """Warp `[N, C, H, W]` features with matching `[N, 2, 3]` affines."""
    _, _, height, width = features.shape
    return warp_affine_simple(
        features, affine_2x3, (height, width), align_corners=align_corners
    )


def identity_affine(batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return `[B, 2, 3]` identity affines."""
    eye = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype
    )
    return eye.unsqueeze(0).expand(batch, -1, -1).contiguous()

"""MambaFusion CUDA 3D-to-image projection wrapper.

Does not implement a second projector. Calls the existing
``map_points`` CUDA kernel (with a matching PyTorch fallback for CPU).

Handles ego-frame points, camera extrinsics / intrinsics, image
augmentation, feature-map scaling, and a validity mask.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from opencood.models.gaussian_modules_0822.base.projection import projection_mats
from opencood.models.gaussian_modules_0822.base.utils import EPS

try:
    from opencood.models.mambafusion_modules.ops.win_coors.flattened_window_cuda import (
        map_points as map_points_cuda,
    )
except ImportError:
    map_points_cuda = None


def _map_points_pytorch(
    points: torch.Tensor,
    lidar2image: torch.Tensor,
    image_aug_matrix: torch.Tensor,
    batch_size: int,
    image_hw: Tuple[int, int],
    expand_scale: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same geometry as MambaFusion ``unitr_utils.map_points`` (CPU / CUDA).

    Args:
        points: ``[grid, sample, 3]``.
        lidar2image: ``[B, V, 4, 4]``.
        image_aug_matrix: ``[B, V, 4, 4]``.
        batch_size: Leading batch of the camera matrices.
        image_hw: ``(height, width)`` of the augmented image.
        expand_scale: Extra normalized margin.

    Returns:
        ``(points_h, uv_norm, valid_mask)``.
    """
    del batch_size
    points_h = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    grid_num, sample_num = int(points.shape[0]), int(points.shape[1])
    num_view = int(lidar2image.shape[1])
    points_h = (
        points_h.view(1, 1, grid_num, sample_num, 4, 1)
        .expand(lidar2image.shape[0], num_view, -1, -1, -1, -1)
    )
    lidar2image_exp = lidar2image.view(
        lidar2image.shape[0], num_view, 1, 1, 4, 4
    ).expand(-1, -1, grid_num, sample_num, -1, -1)
    aug_exp = image_aug_matrix.view(
        image_aug_matrix.shape[0], num_view, 1, 1, 4, 4
    ).expand(-1, -1, grid_num, sample_num, -1, -1)
    cam_h = torch.matmul(lidar2image_exp, points_h).squeeze(-1)
    valid = cam_h[..., 2] > EPS
    denom = torch.maximum(cam_h[..., 2:3], cam_h.new_full((1,), EPS))
    uv = cam_h[..., :2] / denom
    ones = torch.ones_like(uv[..., :1])
    uv_h = torch.cat([uv, ones, ones], dim=-1).unsqueeze(-1)
    uv_aug = torch.matmul(aug_exp, uv_h).squeeze(-1)[..., :2]
    height, width = float(image_hw[0]), float(image_hw[1])
    uv_norm = torch.stack(
        [uv_aug[..., 0] / width, uv_aug[..., 1] / height], dim=-1
    )
    margin = float(expand_scale)
    valid = (
        valid
        & (uv_norm[..., 0] > 0.0 - margin)
        & (uv_norm[..., 0] < 1.0 + margin)
        & (uv_norm[..., 1] > 0.0 - margin)
        & (uv_norm[..., 1] < 1.0 + margin)
    )
    valid = torch.nan_to_num(valid.to(dtype=torch.float32)).bool()
    return points_h.squeeze(-1), uv_norm, valid


def map_points(
    points: torch.Tensor,
    lidar2image: torch.Tensor,
    image_aug_matrix: torch.Tensor,
    image_hw: Tuple[int, int],
    expand_scale: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call MambaFusion CUDA ``map_points`` when tensors are on GPU.

    Args:
        points: ``[grid, sample, 3]``.
        lidar2image: ``[B, V, 4, 4]``.
        image_aug_matrix: ``[B, V, 4, 4]``.
        image_hw: ``(height, width)``.
        expand_scale: Extra normalized margin.

    Returns:
        ``(points_h, uv_norm [B,V,grid,sample,2], valid [B,V,grid,sample])``.
    """
    batch_size = int(lidar2image.shape[0])
    use_cuda = (
        map_points_cuda is not None
        and points.is_cuda
        and lidar2image.is_cuda
        and image_aug_matrix.is_cuda
    )
    if use_cuda:
        _points_h, uv_norm, valid = map_points_cuda(
            points.contiguous().float(),
            lidar2image.contiguous().float(),
            image_aug_matrix.contiguous().float(),
            batch_size,
            int(image_hw[0]),
            int(image_hw[1]),
            float(expand_scale),
        )
        if valid.dim() == 5:
            valid = valid.squeeze(-1)
        return _points_h, uv_norm, valid.bool()
    return _map_points_pytorch(
        points.float(),
        lidar2image.float(),
        image_aug_matrix.float(),
        batch_size,
        image_hw,
        expand_scale,
    )


def flatten_view_tokens(
    sampled_features: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collapse views and samples to attention tokens ``[N, V·S, C]``.

    Args:
        sampled_features: ``[B, V, N, S, C]``.
        valid_mask: ``[B, V, N, S]``.

    Returns:
        ``(tokens [N, V·S, C], mask [N, V·S])``.
    """
    if sampled_features.dim() != 5:
        raise ValueError(
            f"sampled_features must be [B,V,N,S,C], got {tuple(sampled_features.shape)}"
        )
    batch, n_views, n_pts, n_samples, channels = sampled_features.shape
    n_tokens = batch * n_views * n_samples
    tokens = sampled_features.permute(2, 0, 1, 3, 4).reshape(
        n_pts, n_tokens, channels
    )
    mask = valid_mask.permute(2, 0, 1, 3).reshape(n_pts, n_tokens)
    return tokens, mask


class MambaProjectionWrapper(nn.Module):
    """Project 3D points with precomputed MambaFusion ``map_points`` matrices.

    ``lidar2image`` / ``img_aug_matrix`` must already live on
    ``camera_geometry`` (built once by ``attach_camera_geometry``).

    Args:
        expand_scale: Extra normalized image margin passed to ``map_points``.
        align_corners: ``grid_sample`` alignment when sampling features.
    """

    def __init__(
        self, expand_scale: float = 0.0, align_corners: bool = False
    ) -> None:
        super().__init__()
        self.expand_scale = float(expand_scale)
        self.align_corners = bool(align_corners)
        self._sampling_coord_cache: Dict[
            Tuple[int, int, int, int, bool],
            Tuple[float, float, float, float, float, float],
        ] = {}

    def _sampling_coord_params(
        self,
        image_hw: Tuple[int, int],
        feature_hw: Tuple[int, int],
    ) -> Tuple[float, float, float, float, float, float]:
        """Return cached LSS-pixel → feature/grid affine parameters.

        ``map_points`` returns ``uv_norm = (u / W_img, v / H_img)``.
        AirV2X/LSS places feature cells with endpoint linspace coordinates:

            x_feat = u * (W_feat - 1) / (W_img - 1)
            y_feat = v * (H_feat - 1) / (H_img - 1)

        ``grid_sample`` then needs those feature coordinates converted using
        its own ``align_corners`` convention. Image/feature resolutions are
        static for a run, so the scalar affine coefficients are cached.
        """
        image_h, image_w = int(image_hw[0]), int(image_hw[1])
        feat_h, feat_w = int(feature_hw[0]), int(feature_hw[1])
        if min(image_h, image_w, feat_h, feat_w) <= 1:
            raise ValueError(
                "image/feature dimensions must be > 1, got "
                f"image_hw={image_hw}, feature_hw={feature_hw}"
            )

        key = (image_h, image_w, feat_h, feat_w, self.align_corners)
        cached = self._sampling_coord_cache.get(key)
        if cached is not None:
            return cached

        feat_x_scale = (
            float(image_w) * float(feat_w - 1) / float(image_w - 1)
        )
        feat_y_scale = (
            float(image_h) * float(feat_h - 1) / float(image_h - 1)
        )

        if self.align_corners:
            grid_x_scale = 2.0 * float(image_w) / float(image_w - 1)
            grid_x_bias = -1.0
            grid_y_scale = 2.0 * float(image_h) / float(image_h - 1)
            grid_y_bias = -1.0
        else:
            grid_x_scale = 2.0 * feat_x_scale / float(feat_w)
            grid_x_bias = 1.0 / float(feat_w) - 1.0
            grid_y_scale = 2.0 * feat_y_scale / float(feat_h)
            grid_y_bias = 1.0 / float(feat_h) - 1.0

        params = (
            feat_x_scale, feat_y_scale,
            grid_x_scale, grid_x_bias,
            grid_y_scale, grid_y_bias,
        )
        self._sampling_coord_cache[key] = params
        return params

    def project(
        self,
        points: torch.Tensor,
        lidar2image: torch.Tensor,
        image_aug_matrix: torch.Tensor,
        image_hw: Tuple[int, int],
        feature_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Project ``[N, S, 3]`` (shared across B) or ``[B, N, S, 3]`` (per CAV)."""
        if points.dim() == 2:
            points = points.unsqueeze(1)
        if points.dim() == 4:
            points_h = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
            cam_h = torch.einsum("bvij,bnsj->bvnsi", lidar2image.float(), points_h.float())
            valid_mask = cam_h[..., 2] > EPS
            denom = torch.maximum(cam_h[..., 2:3], cam_h.new_full((1,), EPS))
            uv = cam_h[..., :2] / denom
            ones = torch.ones_like(uv[..., :1])
            uv_h = torch.cat([uv, ones, ones], dim=-1)
            uv_aug = torch.einsum(
                "bvij,bvnsj->bvnsi", image_aug_matrix.float(), uv_h
            )[..., :2]
            height, width = float(image_hw[0]), float(image_hw[1])
            uv_norm = torch.stack(
                [uv_aug[..., 0] / width, uv_aug[..., 1] / height], dim=-1
            )
            margin = float(self.expand_scale)
            valid_mask = (
                valid_mask
                & (uv_norm[..., 0] > 0.0 - margin)
                & (uv_norm[..., 0] < 1.0 + margin)
                & (uv_norm[..., 1] > 0.0 - margin)
                & (uv_norm[..., 1] < 1.0 + margin)
            )
            valid_mask = torch.nan_to_num(valid_mask.to(dtype=torch.float32)).bool()
        elif points.dim() == 3:
            n_pts, n_samples = int(points.shape[0]), int(points.shape[1])
            batch, n_views = int(lidar2image.shape[0]), int(lidar2image.shape[1])
            if n_pts == 0:
                empty_uv = points.new_empty((batch, n_views, 0, n_samples, 2))
                empty_mask = points.new_empty(
                    (batch, n_views, 0, n_samples), dtype=torch.bool
                )
                return {
                    "uv_norm": empty_uv,
                    "uv_feat": empty_uv,
                    "uv_grid": empty_uv,
                    "valid_mask": empty_mask,
                }
            _points_h, uv_norm, valid_mask = map_points(
                points,
                lidar2image,
                image_aug_matrix,
                image_hw,
                expand_scale=self.expand_scale,
            )
            valid_mask = valid_mask.bool()
        else:
            raise ValueError(
                f"points must be [N,3], [N,S,3], or [B,N,S,3], got {tuple(points.shape)}"
            )

        feat_h, feat_w = feature_hw if feature_hw is not None else image_hw
        (
            feat_x_scale,
            feat_y_scale,
            grid_x_scale,
            grid_x_bias,
            grid_y_scale,
            grid_y_bias,
        ) = self._sampling_coord_params(
            image_hw=image_hw,
            feature_hw=(int(feat_h), int(feat_w)),
        )
        uv_feat = torch.stack(
            [
                uv_norm[..., 0] * feat_x_scale,
                uv_norm[..., 1] * feat_y_scale,
            ],
            dim=-1,
        )
        uv_grid = torch.stack(
            [
                uv_norm[..., 0] * grid_x_scale + grid_x_bias,
                uv_norm[..., 1] * grid_y_scale + grid_y_bias,
            ],
            dim=-1,
        )
        return {
            "uv_norm": uv_norm,
            "uv_feat": uv_feat,
            "uv_grid": uv_grid,
            "valid_mask": valid_mask,
        }

    def sample_features(
        self,
        feature_maps: torch.Tensor,
        uv_grid: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Bilinear-sample ``[B, V, C, Hf, Wf]`` (or ``[V, C, Hf, Wf]``) maps.

        Args:
            feature_maps: Flattened-camera or batched feature maps.
            uv_grid: ``[B, V, N, S, 2]`` in ``[-1, 1]``.
            valid_mask: Optional ``[B, V, N, S]``. Invalid locations are
                zeroed after sampling.

        Returns:
            Sampled features ``[B, V, N, S, C]``.
        """
        if uv_grid.dim() != 5:
            raise ValueError(f"uv_grid must be [B,V,N,S,2], got {tuple(uv_grid.shape)}")
        maps = feature_maps
        if maps.dim() == 4:
            maps = maps.unsqueeze(0)
        if maps.dim() != 5:
            raise ValueError(
                f"feature_maps must be [V,C,H,W] or [B,V,C,H,W], got {tuple(maps.shape)}"
            )
        batch, n_views, n_pts, n_samples, _ = uv_grid.shape
        channels = int(maps.shape[2])
        if int(maps.shape[0]) != batch or int(maps.shape[1]) != n_views:
            raise ValueError(
                f"feature_maps {tuple(maps.shape)} vs uv_grid {tuple(uv_grid.shape)}"
            )
        if n_pts == 0:
            return maps.new_empty((batch, n_views, 0, n_samples, channels))

        flat_maps = maps.reshape(batch * n_views, channels, maps.shape[-2], maps.shape[-1])
        grid = uv_grid.reshape(batch * n_views, n_pts * n_samples, 1, 2).to(dtype=maps.dtype)
        sampled = F.grid_sample(
            flat_maps,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )
        sampled = (
            sampled.squeeze(-1)
            .transpose(1, 2)
            .reshape(batch, n_views, n_pts, n_samples, channels)
        )
        if valid_mask is not None:
            sampled = sampled * valid_mask.to(dtype=sampled.dtype).unsqueeze(-1)
        return sampled

    def forward(
        self,
        points: torch.Tensor,
        geometry: Mapping[str, Any],
        points_frame: str = "ego",
        feature_maps: Optional[torch.Tensor] = None,
        feature_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Project points with packed geometry; optionally sample features.

        Args:
            points: ``[N, 3]`` or ``[N, S, 3]``.
            geometry: ``camera_geometry`` from ``attach_camera_geometry``.
            points_frame: ``ego`` or ``agent``.
            feature_maps: Optional ``[V, C, Hf, Wf]`` or ``[B, V, C, Hf, Wf]``.
            feature_hw: Optional ``(Hf, Wf)``. Inferred from ``feature_maps``.

        Returns:
            Projection dict, plus ``sampled_features`` when maps are given.
        """
        lidar2image, image_aug, image_hw = projection_mats(geometry, points_frame)
        if feature_hw is None and feature_maps is not None:
            feature_hw = (int(feature_maps.shape[-2]), int(feature_maps.shape[-1]))
        projected = self.project(
            points, lidar2image, image_aug, image_hw, feature_hw=feature_hw
        )
        if feature_maps is not None:
            projected["sampled_features"] = self.sample_features(
                feature_maps, projected["uv_grid"], projected["valid_mask"]
            )
        return projected

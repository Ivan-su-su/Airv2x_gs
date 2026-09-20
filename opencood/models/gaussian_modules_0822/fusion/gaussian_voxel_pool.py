"""Per-agent opacity-aware voxel pooling for Stage-3.

Backward compatible:
- weights=None: exact legacy mean pooling behavior.
- weights=[N]: normalized opacity-weighted pooling.

Geometry stays in scale/quaternion form. We deliberately do NOT pool a full
covariance and eigendecompose it during training, because backward through
eigenvectors becomes unstable for repeated / nearly repeated eigenvalues.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.base.gaussian_transform import (
    normalize_quaternion,
)

DEFAULT_VOXEL_SIZE = (0.4, 0.4, 0.125)
DEFAULT_POINT_CLOUD_RANGE = (-140.8, -40.0, -150.0, 140.8, 40.0, 30.0)


def scatter_mean(
    values: torch.Tensor,
    inverse: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    sums = values.new_zeros((n_groups, values.shape[1]))
    sums.index_add_(0, inverse, values)
    counts = torch.bincount(inverse, minlength=n_groups).to(values.dtype)
    return sums / counts.clamp_min(1).unsqueeze(-1)


def scatter_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    inverse: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    if weights.dim() != 1 or int(weights.shape[0]) != int(values.shape[0]):
        raise ValueError(
            f"weights must be [N], got {tuple(weights.shape)} "
            f"for values {tuple(values.shape)}"
        )

    w = weights.to(device=values.device, dtype=values.dtype)
    sums = values.new_zeros((n_groups, values.shape[1]))
    sums.index_add_(0, inverse, values * w.unsqueeze(-1))

    denom = values.new_zeros((n_groups,))
    denom.index_add_(0, inverse, w)
    return sums / denom.clamp_min(1.0e-8).unsqueeze(-1)


class GaussianVoxelPool:
    def __init__(
        self,
        voxel_size: Any = DEFAULT_VOXEL_SIZE,
        point_cloud_range: Any = DEFAULT_POINT_CLOUD_RANGE,
    ) -> None:
        self.voxel_size = torch.tensor([float(v) for v in voxel_size])
        self.pc_min = torch.tensor([float(v) for v in point_cloud_range[:3]])
        pc_max = torch.tensor([float(v) for v in point_cloud_range[3:]])
        self.grid = torch.floor(
            (pc_max - self.pc_min) / self.voxel_size
        ).long()

    def __call__(
        self,
        gaussians: GaussianSet,
        weights: Optional[torch.Tensor] = None,
    ) -> GaussianSet:
        n = gaussians.n_gaussians
        if n == 0:
            return gaussians

        if weights is not None:
            if weights.dim() != 1 or int(weights.shape[0]) != n:
                raise ValueError(
                    f"weights must be [N={n}], got {tuple(weights.shape)}"
                )

        device = gaussians.mean.device
        dtype = gaussians.mean.dtype
        voxel_size = self.voxel_size.to(device=device, dtype=dtype)
        pc_min = self.pc_min.to(device=device, dtype=dtype)
        grid = self.grid.to(device)

        idx = torch.floor(
            (gaussians.mean - pc_min) / voxel_size
        ).long()
        valid = ((idx >= 0) & (idx < grid)).all(dim=-1)

        kept = gaussians.index_select(valid)
        if kept.n_gaussians == 0:
            return kept

        kept_weights = None if weights is None else weights[valid]
        idx = idx[valid]

        batch = (
            kept.batch_index
            if kept.batch_index is not None
            else torch.zeros(
                kept.n_gaussians,
                dtype=torch.long,
                device=device,
            )
        )

        gx, gy, gz = (int(v) for v in grid.tolist())
        code = (
            (batch * gx + idx[:, 0]) * gy + idx[:, 1]
        ) * gz + idx[:, 2]

        unq_code, inverse = torch.unique(
            code,
            return_inverse=True,
        )
        inverse = inverse.long()
        m = int(unq_code.numel())
        pooled_batch = unq_code // (gx * gy * gz)

        def _pool(values: torch.Tensor) -> torch.Tensor:
            if kept_weights is None:
                return scatter_mean(values, inverse, m)
            return scatter_weighted_mean(
                values,
                kept_weights,
                inverse,
                m,
            )

        # Stable quaternion pooling:
        # align q/-q against the first quaternion in each voxel,
        # then take the same normalized opacity-weighted mean.
        q = kept.quaternion
        order = torch.argsort(inverse, stable=True)
        sorted_inv = inverse[order]
        boundary = torch.ones_like(
            sorted_inv,
            dtype=torch.bool,
        )
        boundary[1:] = sorted_inv[1:] != sorted_inv[:-1]
        first_rows = order[boundary]
        reference = q[first_rows][inverse]

        sign = torch.where(
            (q * reference).sum(-1, keepdim=True) < 0.0,
            -1.0,
            1.0,
        )
        pooled_quat = normalize_quaternion(
            _pool(q * sign)
        )

        # Scale is pooled directly, exactly like the old stable baseline,
        # except opacity supplies normalized weights when enabled.
        # This remains a convex mean, so the number of Gaussians in a voxel
        # cannot by itself make scale explode.
        pooled_scale = _pool(kept.scale)

        return GaussianSet(
            mean=_pool(kept.mean),
            scale=pooled_scale,
            quaternion=pooled_quat,
            feature=_pool(kept.feature),
            p_fg=(
                _pool(kept.p_fg.unsqueeze(-1)).squeeze(-1)
                if kept.p_fg is not None
                else None
            ),
            batch_index=pooled_batch,
            agent=kept.agent,
            frame=kept.frame,
        )

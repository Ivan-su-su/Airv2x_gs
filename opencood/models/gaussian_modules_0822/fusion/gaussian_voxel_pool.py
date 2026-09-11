"""Per-agent voxel pooling for Stage-3.

Groups a Stage-2 ``GaussianSet`` (one agent at a time) into voxels by
``(batch, x, y, z)`` and mean-pools feature / mean / scale. Quaternions
are antipodally aligned to one reference per group before averaging.
Grouping is computed once and reused for every pooled attribute.

Out-of-range Gaussians are dropped by the in-grid validity mask, so no
separate range filter is needed before pooling (Stage-1 already filtered
by ``grid_conf``; this mask bounds means to the Stage-3 voxel range).
"""

from __future__ import annotations

from typing import Any

import torch

from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.base.gaussian_transform import (
    normalize_quaternion,
)

DEFAULT_VOXEL_SIZE = (0.4, 0.4, 0.125)
# Union of agent zbounds (veh [-10,10], rsu [-30,30], drone [-150,-6]);
# x/y match the detection cav_lidar_range [-140.8, -40, -200, 140.8, 40, 200].
DEFAULT_POINT_CLOUD_RANGE = (-140.8, -40.0, -150.0, 140.8, 40.0, 30.0)


def scatter_mean(values: torch.Tensor, inverse: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Group-wise mean via one ``index_add`` plus a count divide.

    Args:
        values: ``[N, D]``.
        inverse: ``[N]`` group id per row.
        n_groups: Number of groups ``M``.

    Returns:
        ``[M, D]`` group means.
    """
    sums = values.new_zeros((n_groups, values.shape[1]))
    sums.index_add_(0, inverse, values)
    counts = torch.bincount(inverse, minlength=n_groups).to(values.dtype)
    return sums / counts.unsqueeze(-1)


class GaussianVoxelPool:
    """MeanVFE-style pooling of one agent's Gaussians.

    Args:
        voxel_size: ``(vx, vy, vz)`` in meters, from the Stage-3 cfg.
        point_cloud_range: ``(xmin, ymin, zmin, xmax, ymax, zmax)``.
    """

    def __init__(
        self,
        voxel_size: Any = DEFAULT_VOXEL_SIZE,
        point_cloud_range: Any = DEFAULT_POINT_CLOUD_RANGE,
    ) -> None:
        self.voxel_size = torch.tensor([float(v) for v in voxel_size])
        self.pc_min = torch.tensor([float(v) for v in point_cloud_range[:3]])
        pc_max = torch.tensor([float(v) for v in point_cloud_range[3:]])
        self.grid = torch.floor((pc_max - self.pc_min) / self.voxel_size).long()

    def __call__(self, gaussians: GaussianSet) -> GaussianSet:
        """Pool Gaussians into one token per occupied voxel.

        Args:
            gaussians: One agent's Stage-2 set; means must be in the
                Stage-3 voxel frame (ego).

        Returns:
            Pooled ``GaussianSet`` with mean-pooled mean / scale / feature,
            antipodally averaged quaternion, and per-voxel ``batch_index``.
        """
        n = gaussians.n_gaussians
        if n == 0:
            return gaussians
        device, dtype = gaussians.mean.device, gaussians.mean.dtype
        voxel_size = self.voxel_size.to(device=device, dtype=dtype)
        pc_min = self.pc_min.to(device=device, dtype=dtype)
        grid = self.grid.to(device)

        # One voxel assignment, shared by every pooled attribute.
        idx = torch.floor((gaussians.mean - pc_min) / voxel_size).long()
        valid = ((idx >= 0) & (idx < grid)).all(dim=-1)
        kept = gaussians.index_select(valid)
        if kept.n_gaussians == 0:
            return kept
        idx = idx[valid]
        batch = (
            kept.batch_index
            if kept.batch_index is not None
            else torch.zeros(kept.n_gaussians, dtype=torch.long, device=device)
        )
        gx, gy, gz = (int(v) for v in grid.tolist())
        code = ((batch * gx + idx[:, 0]) * gy + idx[:, 1]) * gz + idx[:, 2]
        unq_code, inverse = torch.unique(code, return_inverse=True)
        inverse = inverse.long()
        m = int(unq_code.numel())
        pooled_batch = unq_code // (gx * gy * gz)

        # Quaternion pooling: flip q/-q against each group's first member,
        # then average and renormalize. Group starts come from one stable
        # sort — no Python loop.
        q = kept.quaternion
        order = torch.argsort(inverse, stable=True)
        sorted_inv = inverse[order]
        boundary = torch.ones_like(sorted_inv, dtype=torch.bool)
        boundary[1:] = sorted_inv[1:] != sorted_inv[:-1]
        first_rows = order[boundary]  # first row of each group, id order
        reference = q[first_rows][inverse]
        sign = torch.where((q * reference).sum(-1, keepdim=True) < 0.0, -1.0, 1.0)
        pooled_quat = normalize_quaternion(scatter_mean(q * sign, inverse, m))

        pooled = GaussianSet(
            mean=scatter_mean(kept.mean, inverse, m),
            scale=scatter_mean(kept.scale, inverse, m),
            quaternion=pooled_quat,
            feature=scatter_mean(kept.feature, inverse, m),
            p_fg=scatter_mean(kept.p_fg.unsqueeze(-1), inverse, m).squeeze(-1)
            if kept.p_fg is not None
            else None,
            batch_index=pooled_batch,
            agent=kept.agent,
            frame=kept.frame,
        )
        return pooled

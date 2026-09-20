"""Sparse Gaussian-to-BEV splatting for Stage-3.

Each pooled Gaussian writes its feature into BEV cells inside its own
n-sigma AABB of the ego-XY marginal, weighted by the Mahalanobis kernel
of ``Σ_xy = Σ_3d[:2, :2]``. Neighborhoods are generated per
``(radius_x, radius_y)`` bucket so one large Gaussian cannot enlarge the
offset grid of the others.

Aggregation is an unnormalized direct Gaussian weighted sum:
``BEV(p) = Σ_i feature_i * exp(-0.5 * Mahalanobis²(i, p))``. The kernel
keeps its absolute radial decay; contributions are NOT divided by the
accumulated weight (a normalized average would flatten the kernel to a
plateau for singly-covered cells).

BEV layout matches ``PointPillarScatter``: ``[B, C, ny, nx]`` with rows
along y and columns along x.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import time

import torch
from torch import nn

from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.fusion.gaussian_voxel_pool import (
    DEFAULT_POINT_CLOUD_RANGE,
    DEFAULT_VOXEL_SIZE,
)

_COV_EPS = 1.0e-4
_DET_EPS = 1.0e-6


class GaussianBEVSplat(nn.Module):
    """Splat pooled Gaussian features onto a BEV grid.

    Args:
        feature_dim: Gaussian feature width ``C``.
        voxel_size: BEV cell size ``(vx, vy, ...)`` meters.
        point_cloud_range: ``(xmin, ymin, zmin, xmax, ymax, zmax)``; only
            x/y bounds are used for the BEV grid.
        n_sigma: Support radius in standard deviations.
    """

    def __init__(
        self,
        feature_dim: int,
        voxel_size: Any = DEFAULT_VOXEL_SIZE,
        point_cloud_range: Any = DEFAULT_POINT_CLOUD_RANGE,
        n_sigma: float = 3.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.n_sigma = float(n_sigma)
        vx, vy = float(voxel_size[0]), float(voxel_size[1])
        self.register_buffer("cell", torch.tensor([vx, vy]), persistent=False)
        self.register_buffer(
            "xy_min",
            torch.tensor(
                [float(point_cloud_range[0]), float(point_cloud_range[1])]
            ),
            persistent=False,
        )
        ny = int((float(point_cloud_range[4]) - float(point_cloud_range[1])) / vy)
        nx = int((float(point_cloud_range[3]) - float(point_cloud_range[0])) / vx)
        self.register_buffer("grid_hw", torch.tensor([ny, nx]), persistent=False)
        self.last_stats: Dict[str, Any] = {}

    def forward(self, gaussians: GaussianSet) -> torch.Tensor:
        """Splat features and return the BEV map.

        Means / quaternions are already in ego; this does not apply another
        rigid transform. ``Σ_3d`` comes from ``GaussianSet.covariance()``.

        Args:
            gaussians: Merged pooled Gaussians. ``batch_index`` may be
                ``None`` for a single-sample set.

        Returns:
            ``[B, C, ny, nx]`` BEV feature map (rows y, columns x).
        """
        n = gaussians.n_gaussians
        device = gaussians.mean.device
        dtype = gaussians.feature.dtype
        ny, nx = (int(v) for v in self.grid_hw.tolist())
        if n == 0:
            self.last_stats = {"n_gaussians": 0, "n_candidate": 0, "n_valid": 0}
            return torch.zeros((1, self.feature_dim, ny, nx), device=device, dtype=dtype)
        if gaussians.batch_index is None:
            batch_index = torch.zeros(n, dtype=torch.long, device=device)
            n_batch = 1
        else:
            batch_index = gaussians.batch_index.long()
            n_batch = int(batch_index.max().item()) + 1

        t0 = time.perf_counter()
        cov_xy = gaussians.covariance()[:, :2, :2]
        cov_xy = cov_xy + torch.eye(2, device=device, dtype=dtype) * _COV_EPS
        cell = self.cell.to(device=device, dtype=dtype)
        xy_min = self.xy_min.to(device=device, dtype=dtype)
        var_xy = cov_xy.diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
        radius = torch.clamp(
            torch.ceil(self.n_sigma * var_xy.sqrt() / cell).long(), min=1
        )
        mean_xy = gaussians.mean[:, :2]
        center_idx = torch.floor((mean_xy - xy_min) / cell).long()

        g_idx, ix, iy, delta = self._candidate_pairs(
            center_idx, radius, mean_xy, xy_min, cell, nx, ny
        )
        n_candidate = int(g_idx.shape[0])
        if n_candidate == 0:
            self.last_stats = {
                "n_gaussians": n,
                "n_candidate": 0,
                "n_valid": 0,
                **_radius_summary(radius),
            }
            return torch.zeros(
                (n_batch, self.feature_dim, ny, nx), device=device, dtype=dtype
            )

        d2 = _mahalanobis2(cov_xy[g_idx], delta)
        support = d2 <= (self.n_sigma * self.n_sigma)
        g_idx = g_idx[support]
        ix = ix[support]
        iy = iy[support]
        weight = torch.exp(-0.5 * d2[support])
        if gaussians.opacity is not None:
            weight = weight * gaussians.opacity[g_idx].to(
                device=device, dtype=weight.dtype
            )
        n_valid = int(g_idx.shape[0])

        # Preserve the absolute Gaussian spatial response.
        # Each Gaussian contributes feature * exp(-0.5 * Mahalanobis^2)
        # and overlapping Gaussian contributions are summed directly.
        # Do NOT normalize by the accumulated Gaussian weights here:
        # for a singly-covered cell that would compute
        # (f * w) / w = f and cancel the radial decay entirely,
        # turning the kernel into a flat plateau inside the support.
        flat = batch_index[g_idx] * (ny * nx) + iy * nx + ix
        bev = gaussians.feature.new_zeros((n_batch * ny * nx, self.feature_dim))
        bev.index_add_(0, flat, gaussians.feature[g_idx] * weight.unsqueeze(-1))
        splat_ms = (time.perf_counter() - t0) * 1000.0
        self.last_stats = {
            "n_gaussians": n,
            "n_candidate": n_candidate,
            "n_valid": n_valid,
            "splat_ms": splat_ms,
            **_radius_summary(radius),
        }
        return (
            bev.view(n_batch, ny, nx, self.feature_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def _candidate_pairs(
        self,
        center_idx: torch.Tensor,
        radius: torch.Tensor,
        mean_xy: torch.Tensor,
        xy_min: torch.Tensor,
        cell: torch.Tensor,
        nx: int,
        ny: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build per-Gaussian AABB candidates, bucketed by ``(rx, ry)``.

        Args:
            center_idx: Occupied BEV cell ``[N, 2]`` ``(ix, iy)``.
            radius: Per-Gaussian cell radii ``[N, 2]``.
            mean_xy: Ego-frame means ``[N, 2]``.
            xy_min: BEV origin.
            cell: Cell size ``(vx, vy)``.
            nx: Grid width (x).
            ny: Grid height (y).

        Returns:
            ``g_idx, ix, iy, delta`` for in-grid AABB cells. ``delta`` is
            cell-center minus ``mean_xy``.
        """
        unique_r, inverse = torch.unique(radius, dim=0, return_inverse=True)
        g_parts: List[torch.Tensor] = []
        ix_parts: List[torch.Tensor] = []
        iy_parts: List[torch.Tensor] = []
        delta_parts: List[torch.Tensor] = []
        for bucket in range(int(unique_r.shape[0])):
            rx = int(unique_r[bucket, 0].item())
            ry = int(unique_r[bucket, 1].item())
            g_ids = (inverse == bucket).nonzero(as_tuple=True)[0]
            ox = torch.arange(-rx, rx + 1, device=center_idx.device)
            oy = torch.arange(-ry, ry + 1, device=center_idx.device)
            oy_grid, ox_grid = torch.meshgrid(oy, ox, indexing="ij")
            ox_flat = ox_grid.reshape(-1)
            oy_flat = oy_grid.reshape(-1)
            ix = center_idx[g_ids, 0].unsqueeze(1) + ox_flat.unsqueeze(0)
            iy = center_idx[g_ids, 1].unsqueeze(1) + oy_flat.unsqueeze(0)
            inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            gi, ki = inside.nonzero(as_tuple=True)
            if int(gi.numel()) == 0:
                continue
            hit = g_ids[gi]
            hit_ix = ix[gi, ki]
            hit_iy = iy[gi, ki]
            center = xy_min + (torch.stack([hit_ix, hit_iy], dim=-1).to(
                dtype=mean_xy.dtype
            ) + 0.5) * cell
            g_parts.append(hit)
            ix_parts.append(hit_ix)
            iy_parts.append(hit_iy)
            delta_parts.append(center - mean_xy[hit])
        if not g_parts:
            empty_i = center_idx.new_empty((0,), dtype=torch.long)
            empty_d = mean_xy.new_empty((0, 2))
            return empty_i, empty_i, empty_i, empty_d
        return (
            torch.cat(g_parts),
            torch.cat(ix_parts),
            torch.cat(iy_parts),
            torch.cat(delta_parts),
        )


def _mahalanobis2(cov_xy: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """``delta^T Σ^{-1} delta`` for batched 2x2 SPD covariances.

    Args:
        cov_xy: ``[K, 2, 2]``.
        delta: ``[K, 2]``.

    Returns:
        Squared Mahalanobis distance ``[K]``.
    """
    a = cov_xy[:, 0, 0]
    b = cov_xy[:, 0, 1]
    c = cov_xy[:, 1, 1]
    det = (a * c - b * b).clamp_min(_DET_EPS)
    inv00 = c / det
    inv01 = -b / det
    inv11 = a / det
    dx = delta[:, 0]
    dy = delta[:, 1]
    return dx * (inv00 * dx + inv01 * dy) + dy * (inv01 * dx + inv11 * dy)


def _radius_summary(radius: torch.Tensor) -> Dict[str, float]:
    """Median / P90 / P99 / max of per-axis cell radii."""

    def _q(values: torch.Tensor, q: float) -> float:
        return float(torch.quantile(values.float(), q).item())

    rx = radius[:, 0]
    ry = radius[:, 1]
    return {
        "radius_x_median": _q(rx, 0.5),
        "radius_x_p90": _q(rx, 0.9),
        "radius_x_p99": _q(rx, 0.99),
        "radius_x_max": float(rx.max().item()),
        "radius_y_median": _q(ry, 0.5),
        "radius_y_p90": _q(ry, 0.9),
        "radius_y_p99": _q(ry, 0.99),
        "radius_y_max": float(ry.max().item()),
    }

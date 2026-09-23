"""Shared Gaussian geometry encoder with position normalization.

Offset sampler, Stage-1 refiner, and cross-view interaction all use this
module. Fusion MLPs stay task-specific elsewhere.
"""

from __future__ import annotations

import torch
from torch import nn

GEO_INPUT_DIM = 10
DEFAULT_GEO_DIM = 64
DEFAULT_GEO_HIDDEN = 64


class GaussianGeometryEncoder(nn.Module):
    """MLP over ``[normalized mean, scale, quaternion]`` (dim 10).

    Only the Gaussian center is normalized. ``scale`` remains in meters and
    ``quaternion`` keeps its existing representation. The position
    normalization constants are computed once in ``__init__`` and stored as
    buffers so all users of this shared encoder reuse the same values.

    Args:
        geo_dim: Output embedding width.
        hidden_dim: Hidden width.
        cav_range: Optional
            ``[xmin, ymin, zmin, xmax, ymax, zmax]`` range in meters.
            When omitted, mean normalization is the identity transform for
            backward compatibility with older callers.
    """

    def __init__(
        self,
        geo_dim: int = DEFAULT_GEO_DIM,
        hidden_dim: int = DEFAULT_GEO_HIDDEN,
        cav_range=None,
    ) -> None:
        super().__init__()
        self.geo_dim = int(geo_dim)

        if cav_range is None:
            mean_center = torch.zeros(3, dtype=torch.float32)
            mean_inv_half_extent = torch.ones(3, dtype=torch.float32)
        else:
            if len(cav_range) != 6:
                raise ValueError(
                    "cav_range must be "
                    "[xmin, ymin, zmin, xmax, ymax, zmax]"
                )
            cav = torch.as_tensor(cav_range, dtype=torch.float32)
            xyz_min, xyz_max = cav[:3], cav[3:]
            half_extent = 0.5 * (xyz_max - xyz_min)
            if not torch.isfinite(cav).all() or torch.any(half_extent <= 0):
                raise ValueError(f"invalid cav_range: {list(cav_range)}")
            mean_center = 0.5 * (xyz_min + xyz_max)
            mean_inv_half_extent = half_extent.reciprocal()

        self.register_buffer(
            "_mean_center",
            mean_center,
            persistent=False,
        )
        self.register_buffer(
            "_mean_inv_half_extent",
            mean_inv_half_extent,
            persistent=False,
        )

        self.mlp = nn.Sequential(
            nn.Linear(GEO_INPUT_DIM, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), self.geo_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor,
        quaternion: torch.Tensor,
    ) -> torch.Tensor:
        """Encode Gaussian geometry.

        Args:
            mean: ``[N, 3]``.
            scale: ``[N, 3]``.
            quaternion: ``[N, 4]`` ``wxyz``.

        Returns:
            ``[N, geo_dim]``.
        """
        mean_normalized = (
            mean - self._mean_center
        ) * self._mean_inv_half_extent
        geometry = torch.cat([mean_normalized, scale, quaternion], dim=-1)
        return self.mlp(geometry)

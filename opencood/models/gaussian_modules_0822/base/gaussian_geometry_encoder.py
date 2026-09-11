"""Shared Gaussian geometry encoder: concat(mean, scale, quaternion) → embedding.

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
    """MLP over ``[mean, scale, quaternion]`` (dim 10).

    Args:
        geo_dim: Output embedding width.
        hidden_dim: Hidden width.
    """

    def __init__(
        self,
        geo_dim: int = DEFAULT_GEO_DIM,
        hidden_dim: int = DEFAULT_GEO_HIDDEN,
    ) -> None:
        super().__init__()
        self.geo_dim = int(geo_dim)
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
        return self.mlp(torch.cat([mean, scale, quaternion], dim=-1))

"""Geometry-aware 3D offset sampler for scale-quaternion Gaussians.

Predicts ``K`` coefficients ``(a, b, c) ∈ (0, 1)`` in the canonical frame,
scales by stored ``scale``, rotates by the stored quaternion, and emits
``{mean ± delta_k} ∪ {mean}`` as ``[N, 2K+1, 3]``.

Geometry encoding is the shared :class:`GaussianGeometryEncoder`. Offset
fusion MLPs are task-specific and not reused by refinement.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from opencood.models.gaussian_modules_0822.base.gaussian_geometry_encoder import (
    GaussianGeometryEncoder,
)
from opencood.models.gaussian_modules_0822.base.gaussian_transform import (
    quaternion_to_rotation_matrix,
)
from opencood.models.gaussian_modules_0822.fusion.offset_fusion import OffsetFusion
from opencood.models.gaussian_modules_0822.p1_layout import AGENT_TYPES, F90_CHANNELS

DEFAULT_NUM_OFFSETS = 3


class GaussianOffsetSampler(nn.Module):
    """Predict intra-ellipsoid sample offsets from geometry + appearance.

    Args:
        feature_dim: Gaussian feature width ``C``.
        num_offsets: ``K`` learned offset directions.
        geometry_encoder: Shared encoder. Created if omitted.
        hidden_dim: Offset fusion hidden width.
    """

    def __init__(
        self,
        feature_dim: int = F90_CHANNELS,
        num_offsets: int = DEFAULT_NUM_OFFSETS,
        geometry_encoder: Optional[GaussianGeometryEncoder] = None,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_offsets = int(num_offsets)
        self.geometry_encoder = geometry_encoder or GaussianGeometryEncoder()
        self.offset_fusion = OffsetFusion(
            feature_dim=int(feature_dim),
            geo_dim=self.geometry_encoder.geo_dim,
            hidden_dim=int(hidden_dim),
        )
        self.offset_heads = nn.ModuleDict(
            {
                agent: nn.Linear(self.offset_fusion.hidden_dim, self.num_offsets * 3)
                for agent in AGENT_TYPES
            }
        )

    def forward(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor,
        quaternion: torch.Tensor,
        feature: torch.Tensor,
        agent: str,
    ) -> Dict[str, torch.Tensor]:
        """Sample ``2K+1`` 3D points per Gaussian.

        Args:
            mean: ``[N, 3]``.
            scale: ``[N, 3]``.
            quaternion: ``[N, 4]``.
            feature: ``[N, C]``.
            agent: ``vehicle`` / ``rsu`` / ``drone``.

        Returns:
            ``samples [N, 2K+1, 3]``, ``coefficients [N, K, 3]``,
            ``delta [N, K, 3]``.
        """
        agent_key = str(agent).lower()
        n_out = 2 * self.num_offsets + 1
        if int(mean.shape[0]) == 0:
            return {
                "samples": mean.new_empty((0, n_out, 3)),
                "coefficients": mean.new_empty((0, self.num_offsets, 3)),
                "delta": mean.new_empty((0, self.num_offsets, 3)),
            }

        geo_embedding = self.geometry_encoder(mean, scale, quaternion)
        raw = self.offset_heads[agent_key](
            self.offset_fusion(feature, geo_embedding, agent_key)
        )
        coefficients = torch.sigmoid(raw).view(-1, self.num_offsets, 3)
        rotation = quaternion_to_rotation_matrix(quaternion)
        delta = torch.matmul(
            rotation, (coefficients * scale.unsqueeze(1)).transpose(-1, -2)
        ).transpose(-1, -2)
        center = mean.unsqueeze(1)
        return {
            "samples": torch.cat([center + delta, center - delta, center], dim=1),
            "coefficients": coefficients,
            "delta": delta,
        }

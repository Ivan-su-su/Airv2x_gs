"""Task-specific fusion for Stage-1 Gaussian parameter refinement.

Do not share these MLPs with offset sampling.
"""

from __future__ import annotations

import torch
from torch import nn

from opencood.models.gaussian_modules_0822.p1_layout import AGENT_TYPES


class RefinementFusion(nn.Module):
    """Agent-specific fusion of refined features and geometry.

    Args:
        feature_dim: Refined Gaussian feature width ``C``.
        geo_dim: Geometry embedding width ``G``.
        hidden_dim: Fusion MLP width ``H``.
    """

    def __init__(
        self,
        feature_dim: int,
        geo_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        fused_dim = int(feature_dim) + int(geo_dim)
        self.fusions = nn.ModuleDict(
            {
                agent: nn.Sequential(
                    nn.Linear(fused_dim, self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for agent in AGENT_TYPES
            }
        )

    def forward(
        self,
        feature: torch.Tensor,
        geo_embedding: torch.Tensor,
        agent: str,
    ) -> torch.Tensor:
        """Fuse Stage-1 features with geometry.

        Args:
            feature: ``[N, C]``.
            geo_embedding: ``[N, G]``.
            agent: ``vehicle`` / ``rsu`` / ``drone``.

        Returns:
            ``[N, H]``.
        """
        return self.fusions[str(agent).lower()](
            torch.cat([feature, geo_embedding], dim=-1)
        )

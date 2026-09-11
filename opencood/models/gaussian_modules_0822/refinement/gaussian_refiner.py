"""Stage-1 Gaussian parameter refiner.

Runs after intra-view image aggregation. Updates mean / scale / quaternion
from the refined feature. Never builds Σ.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional

import torch
from torch import nn

from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.base.gaussian_geometry_encoder import (
    GaussianGeometryEncoder,
)
from opencood.models.gaussian_modules_0822.fusion.refinement_fusion import (
    RefinementFusion,
)
from opencood.models.gaussian_modules_0822.p1_layout import AGENT_TYPES, F90_CHANNELS
from opencood.models.gaussian_modules_0822.refinement.refine_heads import (
    DEFAULT_UNIT_XYZ,
    GaussianRefineHeads,
)


class GaussianRefiner(nn.Module):
    """Refine Gaussian geometry after Stage-1 feature aggregation.

    Args:
        geometry_encoder: Shared ``mean/scale/quat`` encoder. Same instance
            as the offset sampler when both run in one model.
        feature_dim: Gaussian feature width ``C``.
        cfg: Optional ``gaussian_refinement`` yaml block.
    """

    def __init__(
        self,
        geometry_encoder: GaussianGeometryEncoder,
        feature_dim: int = F90_CHANNELS,
        cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        cfg = dict(cfg or {})
        hidden_dim = int((cfg.get("fusion") or {}).get("hidden_dim", 256))
        agent_cfg = dict(cfg.get("agents") or {})
        unit_xyz = {
            agent: list(
                dict(agent_cfg.get(agent) or {}).get("unit_xyz", DEFAULT_UNIT_XYZ[agent])
            )
            for agent in AGENT_TYPES
        }
        self.geometry_encoder = geometry_encoder
        self.fusion = RefinementFusion(
            feature_dim=int(feature_dim),
            geo_dim=geometry_encoder.geo_dim,
            hidden_dim=hidden_dim,
        )
        scale_cfg = dict(cfg.get("scale") or {})
        delta_range = scale_cfg.get("delta_range", (-0.20, 0.05))
        self.heads = GaussianRefineHeads(
            hidden_dim=hidden_dim,
            unit_xyz=unit_xyz,
            scale_delta_range=(float(delta_range[0]), float(delta_range[1])),
            rotation_max_angle=float(
                (cfg.get("rotation") or {}).get("max_angle", 0.2)
            ),
        )

    def forward(
        self,
        gaussians: GaussianSet,
        refined_feature: torch.Tensor,
        agent_type: Optional[str] = None,
    ) -> GaussianSet:
        """Update mean / scale / rotation; keep the refined feature.

        Args:
            gaussians: Current Gaussian set.
            refined_feature: Stage-1 aggregated feature ``[N, C]``.
            agent_type: ``vehicle`` / ``rsu`` / ``drone``. Defaults to
                ``gaussians.agent``.

        Returns:
            ``GaussianSet`` with updated geometry and ``refined_feature``.
        """
        if int(gaussians.n_gaussians) == 0:
            return replace(gaussians, feature=refined_feature)
        agent = str(agent_type or gaussians.agent).lower()
        geo_embedding = self.geometry_encoder(
            gaussians.mean, gaussians.scale, gaussians.quaternion
        )
        updated = self.heads(
            self.fusion(refined_feature, geo_embedding, agent),
            gaussians.mean,
            gaussians.scale,
            gaussians.quaternion,
            agent,
        )
        return replace(
            gaussians,
            mean=updated["mean"],
            scale=updated["scale"],
            quaternion=updated["quaternion"],
            feature=refined_feature,
        )

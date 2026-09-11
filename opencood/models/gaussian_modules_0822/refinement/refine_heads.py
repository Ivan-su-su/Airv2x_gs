"""Agent-specific Gaussian parameter refinement heads.

Predict bounded residuals for mean, scale, and rotation. Never builds Σ.
Hyperparameters come from the constructor, not literals in the update.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
from torch import nn

from opencood.models.gaussian_modules_0822.p1_layout import AGENT_TYPES
from opencood.models.gaussian_modules_0822.refinement.rotation_utils import (
    compose_rotation_residual,
)

DEFAULT_UNIT_XYZ: Dict[str, Sequence[float]] = {
    "vehicle": (4.0, 4.0, 1.0),
    "rsu": (5.0, 5.0, 2.0),
    "drone": (5.0, 5.0, 3.0),
}


class AgentRefineHeads(nn.Module):
    """Three residual branches for one agent type.

    Args:
        hidden_dim: Fusion feature width ``H``.
        unit_xyz: Per-axis mean residual bound in meters.
        scale_delta_range: Log-scale residual ``[lo, hi]``. Head logit 0
            maps to ``δ=0`` (multiplier 1). Asymptotes are ``exp(lo)`` and
            ``exp(hi)``.
        rotation_max_angle: Axis-angle bound (radians).
    """

    def __init__(
        self,
        hidden_dim: int,
        unit_xyz: Sequence[float],
        scale_delta_range: Sequence[float],
        rotation_max_angle: float,
    ) -> None:
        super().__init__()
        self.mean_head = nn.Linear(int(hidden_dim), 3)
        self.scale_head = nn.Linear(int(hidden_dim), 3)
        self.rotation_head = nn.Linear(int(hidden_dim), 3)
        self.register_buffer(
            "unit_xyz",
            torch.tensor([float(v) for v in unit_xyz], dtype=torch.float32),
        )
        lo, hi = float(scale_delta_range[0]), float(scale_delta_range[1])
        self.register_buffer(
            "scale_delta_range",
            torch.tensor([lo, hi], dtype=torch.float32),
        )
        self.rotation_max_angle = float(rotation_max_angle)

    def forward(
        self,
        fusion_feature: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        quaternion: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Apply bounded residuals to mean / scale / quaternion.

        Args:
            fusion_feature: ``[N, H]``.
            mean: ``[N, 3]``.
            scale: ``[N, 3]``.
            quaternion: ``[N, 4]``.

        Returns:
            Updated geometry tensors.
        """
        unit_xyz = self.unit_xyz.to(device=mean.device, dtype=mean.dtype)
        delta_mean = (2.0 * torch.sigmoid(self.mean_head(fusion_feature)) - 1.0) * unit_xyz
        scale_prob = torch.sigmoid(self.scale_head(fusion_feature))
        lo, hi = self.scale_delta_range[0], self.scale_delta_range[1]
        delta_scale = torch.where(
            scale_prob <= 0.5,
            lo * (1.0 - 2.0 * scale_prob),
            hi * (2.0 * scale_prob - 1.0),
        )
        new_scale = scale * torch.exp(delta_scale)
        delta_rotation = self.rotation_max_angle * torch.tanh(
            self.rotation_head(fusion_feature)
        )
        return {
            "mean": mean + delta_mean,
            "scale": new_scale,
            "quaternion": compose_rotation_residual(delta_rotation, quaternion),
            "delta_log_scale": delta_scale,
        }


class GaussianRefineHeads(nn.Module):
    """Independent refinement heads for vehicle / RSU / drone.

    Args:
        hidden_dim: Fusion feature width.
        unit_xyz: ``{agent: [dx, dy, dz]}``. Missing agents use defaults.
        scale_delta_range: Shared log-scale residual ``[lo, hi]``.
        rotation_max_angle: Shared axis-angle bound.
    """

    def __init__(
        self,
        hidden_dim: int,
        unit_xyz: Mapping[str, Sequence[float]],
        scale_delta_range: Sequence[float] = (-0.20, 0.05),
        rotation_max_angle: float = 0.2,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleDict()
        for agent in AGENT_TYPES:
            self.heads[agent] = AgentRefineHeads(
                hidden_dim=hidden_dim,
                unit_xyz=unit_xyz[agent],
                scale_delta_range=scale_delta_range,
                rotation_max_angle=rotation_max_angle,
            )

    def forward(
        self,
        fusion_feature: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        quaternion: torch.Tensor,
        agent: str,
    ) -> Dict[str, torch.Tensor]:
        """Dispatch to the agent-specific head.

        Args:
            fusion_feature: ``[N, H]``.
            mean: ``[N, 3]``.
            scale: ``[N, 3]``.
            quaternion: ``[N, 4]``.
            agent: ``vehicle`` / ``rsu`` / ``drone``.

        Returns:
            Updated geometry tensors.
        """
        return self.heads[str(agent).lower()](
            fusion_feature, mean, scale, quaternion
        )

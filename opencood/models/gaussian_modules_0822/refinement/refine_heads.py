"""Agent-specific Gaussian parameter refinement heads.

Predict bounded residuals for mean, scale, and rotation. Never builds Σ.
Hyperparameters come from the constructor, not literals in the update.

Mean update has two modes:

* ``"xyz"`` (legacy/default): bounded free XYZ residual via ``mean_head``.
* ``"ray"``: scalar depth residual ``Δd = d_max·tanh(h_ray)`` along the
  initializer's camera observation ray, i.e. ``Δmean = Δd · unit_ray``.
  The legacy ``mean_head`` is still created so old checkpoints stay
  loadable, but is unused in this mode.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

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
        unit_xyz: Per-axis mean residual bound in meters (xyz mode).
        scale_delta_range: Log-scale residual ``[lo, hi]``. Head logit 0
            maps to ``δ=0`` (multiplier 1). Asymptotes are ``exp(lo)`` and
            ``exp(hi)``.
        rotation_max_angle: Axis-angle bound (radians).
        mean_update_mode: ``"xyz"`` (legacy bounded XYZ) or ``"ray"``
            (scalar depth residual along the observation ray).
        ray_delta_max: Depth residual bound ``d_max`` in meters (ray mode).
    """

    def __init__(
        self,
        hidden_dim: int,
        unit_xyz: Sequence[float],
        scale_delta_range: Sequence[float],
        rotation_max_angle: float,
        mean_update_mode: str = "xyz",
        ray_delta_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.mean_update_mode = str(mean_update_mode).lower()
        if self.mean_update_mode not in ("xyz", "ray"):
            raise ValueError(
                "mean_update_mode must be 'xyz' or 'ray', got "
                f"{self.mean_update_mode!r}"
            )
        # Legacy head: kept in both modes (name/shape unchanged) so old
        # checkpoints stay loadable and xyz mode is bit-identical.
        self.mean_head = nn.Linear(int(hidden_dim), 3)
        if self.mean_update_mode == "ray":
            self.ray_head = nn.Linear(int(hidden_dim), 1)
            # Zero-init so Δd = 0 at start: a fresh ray model resumes from
            # an old checkpoint without randomly shoving centers along rays.
            nn.init.zeros_(self.ray_head.weight)
            nn.init.zeros_(self.ray_head.bias)
        self.scale_head = nn.Linear(int(hidden_dim), 3)
        self.rotation_head = nn.Linear(int(hidden_dim), 3)
        self.register_buffer(
            "unit_xyz",
            torch.tensor([float(v) for v in unit_xyz], dtype=torch.float32),
        )
        self.register_buffer(
            "ray_delta_max",
            torch.tensor(float(ray_delta_max), dtype=torch.float32),
            # Non-persistent: a config hyperparameter, not a learned state.
            # Keeps old-YAML strict checkpoint loads free of any new
            # mandatory ray key (spec §2 / Test B).
            persistent=False,
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
        ray_dir: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply bounded residuals to mean / scale / quaternion.

        Args:
            fusion_feature: ``[N, H]``.
            mean: ``[N, 3]``.
            scale: ``[N, 3]``.
            quaternion: ``[N, 4]``.
            ray_dir: Optional ``[N, 3]`` unit observation ray from
                ``GaussianSet.ray_dir``. Required by ray mode, ignored by
                xyz mode.

        Returns:
            Updated geometry tensors (plus diagnostic
            ``delta_mean`` / ``delta_depth``).

        Raises:
            ValueError: Ray mode without ``ray_dir`` or with a shape
                mismatch, or unknown mode.
        """
        if self.mean_update_mode == "xyz":
            # Legacy path, kept byte-for-byte identical.
            unit_xyz = self.unit_xyz.to(device=mean.device, dtype=mean.dtype)
            delta_mean = (2.0 * torch.sigmoid(self.mean_head(fusion_feature)) - 1.0) * unit_xyz
            delta_depth = None
        elif self.mean_update_mode == "ray":
            if ray_dir is None:
                raise ValueError(
                    "ray mean update requires GaussianSet.ray_dir"
                )
            if tuple(ray_dir.shape) != tuple(mean.shape):
                raise ValueError(
                    "ray_dir shape "
                    f"{tuple(ray_dir.shape)} != mean shape {tuple(mean.shape)}"
                )
            ray = ray_dir.to(device=mean.device, dtype=mean.dtype)
            ray = ray / torch.linalg.norm(ray, dim=-1, keepdim=True).clamp_min(1.0e-8)
            ray_delta_max = self.ray_delta_max.to(device=mean.device, dtype=mean.dtype)
            delta_depth = ray_delta_max * torch.tanh(self.ray_head(fusion_feature))
            delta_mean = delta_depth * ray
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(
                f"unknown mean_update_mode {self.mean_update_mode!r}"
            )
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
        out = {
            "mean": mean + delta_mean,
            "scale": new_scale,
            "quaternion": compose_rotation_residual(delta_rotation, quaternion),
            "delta_log_scale": delta_scale,
            "delta_mean": delta_mean,
        }
        if delta_depth is not None:
            # Debug/diagnostic only; production must not depend on it.
            out["delta_depth"] = delta_depth.squeeze(-1)
        return out


class GaussianRefineHeads(nn.Module):
    """Independent refinement heads for vehicle / RSU / drone.

    Args:
        hidden_dim: Fusion feature width.
        unit_xyz: ``{agent: [dx, dy, dz]}``. Missing agents use defaults.
        scale_delta_range: Shared log-scale residual ``[lo, hi]``.
        rotation_max_angle: Shared axis-angle bound.
        mean_update_mode: ``"xyz"`` (legacy) or ``"ray"``. See
            ``AgentRefineHeads``.
        ray_delta_max: Optional ``{agent: d_max}`` depth residual bounds
            (ray mode). Missing agents default to ``1.0``.
    """

    def __init__(
        self,
        hidden_dim: int,
        unit_xyz: Mapping[str, Sequence[float]],
        scale_delta_range: Sequence[float] = (-0.20, 0.05),
        rotation_max_angle: float = 0.2,
        mean_update_mode: str = "xyz",
        ray_delta_max: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        ray_delta_max = dict(ray_delta_max or {})
        self.heads = nn.ModuleDict()
        for agent in AGENT_TYPES:
            self.heads[agent] = AgentRefineHeads(
                hidden_dim=hidden_dim,
                unit_xyz=unit_xyz[agent],
                scale_delta_range=scale_delta_range,
                rotation_max_angle=rotation_max_angle,
                mean_update_mode=mean_update_mode,
                ray_delta_max=float(ray_delta_max.get(agent, 1.0)),
            )

    def forward(
        self,
        fusion_feature: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        quaternion: torch.Tensor,
        agent: str,
        ray_dir: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Dispatch to the agent-specific head.

        Args:
            fusion_feature: ``[N, H]``.
            mean: ``[N, 3]``.
            scale: ``[N, 3]``.
            quaternion: ``[N, 4]``.
            agent: ``vehicle`` / ``rsu`` / ``drone``.
            ray_dir: Optional ``[N, 3]`` observation ray forwarded to the
                per-agent head (used only in ray mode).

        Returns:
            Updated geometry tensors.
        """
        return self.heads[str(agent).lower()](
            fusion_feature, mean, scale, quaternion, ray_dir=ray_dir
        )

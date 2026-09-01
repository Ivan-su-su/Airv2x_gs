from typing import Dict, Optional

import torch
import torch.nn as nn

from opencood.models.uncertain_gaussian_modules.gaussian_refine.gaussian_geometry_utils import (
    quaternion_to_rotation_matrix,
)


class GaussianKeyPointGenerator(nn.Module):
    """Generate Gaussian-centric 3D key points from `mean + scale + quat + feature`."""

    def __init__(self, model_cfg: Optional[Dict] = None) -> None:
        """Initialize the key-point generator."""
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.learnable_fixed_scale = float(
            self.model_cfg.get("learnable_fixed_scale", 1.0)
        )
        self.use_sigmoid_for_learnable = bool(
            self.model_cfg.get("use_sigmoid_for_learnable", True)
        )
        fixed_scales = self.model_cfg.get(
            "fix_scale",
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
        )
        self.register_buffer(
            "fixed_scales",
            torch.tensor(fixed_scales, dtype=torch.float32),
            persistent=False,
        )
        self.num_learnable_pts = int(self.model_cfg.get("num_learnable_pts", 0))
        self.num_pts = int(self.fixed_scales.shape[0]) + self.num_learnable_pts
        if self.num_learnable_pts > 0:
            self.learnable_fc = nn.LazyLinear(self.num_learnable_pts * 3)
        else:
            self.learnable_fc = None

    def _build_learnable_offsets(
        self,
        feature: torch.Tensor,
        axis_scales: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict additional key points in the local principal-axis frame."""
        if self.learnable_fc is None:
            return None
        learnable = self.learnable_fc(feature).view(
            feature.shape[0], self.num_learnable_pts, 3
        )
        if self.use_sigmoid_for_learnable:
            learnable = torch.sigmoid(learnable) - 0.5
        learnable = learnable * self.learnable_fixed_scale
        return learnable * axis_scales.unsqueeze(1)

    def forward(
        self,
        mean: torch.Tensor,
        axis_scales: torch.Tensor,
        rotation: torch.Tensor,
        feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return Gaussian-aligned 3D key points from `mean + scale + quat + feature`."""
        if mean.numel() == 0:
            empty_points = mean.new_empty((0, self.num_pts, 3))
            empty_rot = mean.new_empty((0, 4))
            empty_scale = mean.new_empty((0, 3))
            return {
                "key_points": empty_points,
                "rotation": empty_rot,
                "axis_scales": empty_scale,
            }

        rotation_matrix = quaternion_to_rotation_matrix(rotation)

        fixed_offsets = (
            self.fixed_scales.to(dtype=mean.dtype, device=mean.device)
            .unsqueeze(0)
            .expand(mean.shape[0], -1, -1)
        )
        fixed_offsets = fixed_offsets * axis_scales.unsqueeze(1)

        learnable_offsets = self._build_learnable_offsets(feature, axis_scales)
        if learnable_offsets is not None:
            local_offsets = torch.cat([fixed_offsets, learnable_offsets], dim=1)
        else:
            local_offsets = fixed_offsets

        world_offsets = torch.matmul(
            rotation_matrix.unsqueeze(1), local_offsets.unsqueeze(-1)
        ).squeeze(-1)
        key_points = world_offsets + mean.unsqueeze(1)
        return {
            "key_points": key_points,
            "rotation": rotation,
            "axis_scales": axis_scales,
        }

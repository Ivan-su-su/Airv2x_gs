"""Learned delta mean / scale / quaternion from Gaussian features."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from opencood.models.uncertain_gaussian_modules.gaussian_refine.gaussian_geometry_utils import (
    normalize_quaternion,
    quaternion_multiply,
    reconstruct_covariance_from_scale_quaternion,
)


class FeatureDrivenGaussianGeometryUpdate(nn.Module):
    """Predict `Δm + Δs + Δquat` from Gaussian features with one shared decoder."""

    def __init__(self, model_cfg: Optional[Dict] = None) -> None:
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.hidden_dim = int(self.model_cfg.get("hidden_dim", 128))
        self.num_layers = int(self.model_cfg.get("num_layers", 3))
        self.dropout = float(self.model_cfg.get("dropout", 0.0))
        self.mean_delta_scale = float(self.model_cfg.get("mean_delta_scale", 1.0))
        self.scale_delta_scale = float(self.model_cfg.get("scale_delta_scale", 0.1))
        self.min_scale = float(self.model_cfg.get("min_scale", 1e-4))
        layers = []
        for layer_idx in range(self.num_layers):
            if layer_idx == 0:
                layers.append(nn.LazyLinear(self.hidden_dim))
            else:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.GELU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
        self.decoder = nn.Sequential(*layers)
        self.param_head = nn.Linear(self.hidden_dim, 10)

    def forward(
        self,
        feature: torch.Tensor,
        mean: torch.Tensor,
        axis_scales: torch.Tensor,
        rotation: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Apply feature-conditioned geometry deltas to `m/s/r`."""
        if feature.numel() == 0:
            empty = feature.new_empty((0, 3))
            empty_cov = feature.new_empty((0, 3, 3))
            empty_rot = feature.new_empty((0, 4))
            return {
                "mean": empty,
                "covariance": empty_cov,
                "axis_scales": empty,
                "rotation": empty_rot,
            }

        decoded = self.decoder(feature)
        params = self.param_head(decoded)
        delta_mean = torch.tanh(params[:, :3]) * self.mean_delta_scale
        delta_scale = torch.tanh(params[:, 3:6]) * self.scale_delta_scale
        delta_quat = normalize_quaternion(params[:, 6:10])
        mean_new = mean + delta_mean
        axis_new = torch.clamp(axis_scales + delta_scale, min=self.min_scale)
        rotation_new = normalize_quaternion(quaternion_multiply(rotation, delta_quat))
        covariance_new = reconstruct_covariance_from_scale_quaternion(
            axis_new, rotation_new
        )
        covariance_new = 0.5 * (covariance_new + covariance_new.transpose(-1, -2))
        return {
            "mean": mean_new,
            "covariance": covariance_new,
            "axis_scales": axis_new,
            "rotation": rotation_new,
        }

# Agent-specific learned Gaussian opacity for Stage-3.

from __future__ import annotations

import math

import torch
from torch import nn

from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.base.gaussian_transform import (
    reconstruct_covariance,
)


class GaussianOpacityCalibrator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 16,  # YAML compatibility only
        init_prob: float = 0.95,
        detach_covariance: bool = True,
        cell_x: float = 0.4,
        cell_y: float = 0.4,
        prob_eps: float = 1.0e-4,
        var_eps: float = 1.0e-6,
        feature_conditioned: bool = False,
        feature_hidden_dim: int = 32,
        feature_embed_dim: int = 8,
        fusion_hidden_dim: int = 8,
    ) -> None:
        super().__init__()
        del hidden_dim

        if not 0.0 < init_prob < 1.0:
            raise ValueError(f"init_prob must be in (0,1), got {init_prob}")
        if int(feature_dim) <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")

        self.feature_dim = int(feature_dim)
        self.detach_covariance = bool(detach_covariance)
        self.cell_x = float(cell_x)
        self.cell_y = float(cell_y)
        self.prob_eps = float(prob_eps)
        self.var_eps = float(var_eps)
        self.feature_conditioned = bool(feature_conditioned)
        self.feature_hidden_dim = int(feature_hidden_dim)
        self.feature_embed_dim = int(feature_embed_dim)
        self.fusion_hidden_dim = int(fusion_hidden_dim)

        # Legacy reliability MLP. Module name and shapes stay fixed so
        # checkpoints keyed as opacity_heads.<agent>.net.* still load.
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.SiLU(),
            nn.Linear(16, 32),
            nn.SiLU(),
            nn.Linear(32, 8),
            nn.SiLU(),
            nn.Linear(8, 1),
        )

        last = self.net[-1]
        nn.init.normal_(last.weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(
            last.bias,
            math.log(float(init_prob) / (1.0 - float(init_prob))),
        )

        if self.feature_conditioned:
            if self.feature_hidden_dim <= 0 or self.feature_embed_dim <= 0:
                raise ValueError(
                    "feature_hidden_dim and feature_embed_dim must be positive, "
                    f"got {self.feature_hidden_dim}, {self.feature_embed_dim}"
                )
            if self.fusion_hidden_dim <= 0:
                raise ValueError(
                    f"fusion_hidden_dim must be positive, got {self.fusion_hidden_dim}"
                )
            self.feature_mlp = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, self.feature_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.feature_hidden_dim, self.feature_embed_dim),
                nn.SiLU(),
            )
            self.fusion_mlp = nn.Sequential(
                nn.Linear(1 + self.feature_embed_dim, self.fusion_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.fusion_hidden_dim, 1),
            )

    def forward(self, gaussians: GaussianSet) -> torch.Tensor:
        if gaussians.n_gaussians == 0:
            return gaussians.feature.new_empty((0,))
        if gaussians.p_fg is None:
            raise ValueError("opacity calibration requires GaussianSet.p_fg")

        p = gaussians.p_fg.float().clamp(
            min=self.prob_eps,
            max=1.0 - self.prob_eps,
        )
        p_logit = torch.log(p) - torch.log1p(-p)

        scale = gaussians.scale.float()
        quaternion = gaussians.quaternion.float()
        if self.detach_covariance:
            scale = scale.detach()
            quaternion = quaternion.detach()

        cov = reconstruct_covariance(scale, quaternion)
        xx = cov[:, 0, 0].clamp_min(self.var_eps)
        yy = cov[:, 1, 1].clamp_min(self.var_eps)
        xy = cov[:, 0, 1]

        log_var_x = torch.log(xx / (self.cell_x * self.cell_x))
        log_var_y = torch.log(yy / (self.cell_y * self.cell_y))
        corr_xy = xy / torch.sqrt((xx * yy).clamp_min(self.var_eps))
        corr_xy = corr_xy.clamp(-1.0, 1.0)

        x = torch.stack(
            [p_logit, log_var_x, log_var_y, corr_xy],
            dim=-1,
        )

        if not torch.isfinite(x).all():
            n_bad = int((~torch.isfinite(x)).any(dim=-1).sum().item())
            raise RuntimeError(f"non-finite opacity inputs: n_bad={n_bad}")

        # self.net(x) is the base reliability logit. Sigmoid is applied
        # only after the optional feature fusion.
        base_logit = self.net(x).squeeze(-1)
        if self.feature_conditioned:
            feat = gaussians.feature.float()
            if int(feat.shape[-1]) != self.feature_dim:
                raise ValueError(
                    f"feature dim {int(feat.shape[-1])} != {self.feature_dim}"
                )
            feature_embed = self.feature_mlp(feat)
            fusion_input = torch.cat(
                [base_logit.unsqueeze(-1), feature_embed],
                dim=-1,
            )
            final_logit = self.fusion_mlp(fusion_input).squeeze(-1)
            opacity = torch.sigmoid(final_logit)
        else:
            opacity = torch.sigmoid(base_logit)
        return opacity.to(dtype=gaussians.feature.dtype)

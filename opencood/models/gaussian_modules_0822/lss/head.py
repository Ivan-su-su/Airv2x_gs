"""Agent-specific categorical depth head. Logits only."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn

AGENT_TYPES = ("vehicle", "rsu", "drone")
CATEGORICAL_DEPTH_AGENTS = ("vehicle", "rsu")
HEIGHT_EMBED_DIM = 16
HEIGHT_SCALE_M = 100.0
HEIGHT_SCALE_M = HEIGHT_SCALE_M


class DepthHead(nn.Module):
    """Lightweight depth classifier on 64-channel F90: 3x3 64→64, ReLU, 1x1 64→D.

    No BatchNorm. No extra encoder/FPN. Each agent owns its own head and D.

    Args:
        num_bins: Agent-specific depth class count ``D``.
        in_channels: Shared ``F90`` channels (64).
    """

    def __init__(self, num_bins: int, in_channels: int = 64) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.spatial = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pred = nn.Conv2d(in_channels, self.num_bins, kernel_size=1)

    def forward(self, f90: torch.Tensor) -> torch.Tensor:
        """Predict per-cell depth logits from shared ``F90``.

        Args:
            f90: ``[N, 64, 90, 160]``.

        Returns:
            ``depth_logits`` of shape ``[N, D, 90, 160]``.
        """
        return self.pred(self.relu(self.spatial(f90)))


class HeightEmbedding(nn.Module):
    """Scalar camera height → 16-d embedding, then spatial broadcast.

    Height is divided by ``HEIGHT_SCALE_M`` (100 m) so typical drone
    altitudes map to an O(1) range. No batch statistics.

    Args:
        embed_dim: Embedding width (16).
        height_scale_m: Deterministic meters-to-O(1) divisor.
    """

    def __init__(
        self, embed_dim: int = HEIGHT_EMBED_DIM, height_scale_m: float = HEIGHT_SCALE_M
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.height_scale_m = float(height_scale_m)
        self.net = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def forward(self, camera_world_z: torch.Tensor, spatial_hw: Tuple[int, int]) -> torch.Tensor:
        """Embed ``[N]`` heights and broadcast to ``[N, 16, H, W]``.

        Args:
            camera_world_z: Camera world-z in meters, shape ``[N]`` or ``[N, 1]``.
            spatial_hw: Feature map ``(H, W)``.

        Returns:
            Height embedding map ``[N, embed_dim, H, W]``.
        """
        height = camera_world_z.reshape(-1, 1).to(dtype=torch.float32)
        embed = self.net(height / self.height_scale_m)
        feat_h, feat_w = spatial_hw
        return embed[:, :, None, None].expand(-1, -1, feat_h, feat_w)


class DeltaHead(nn.Module):
    """Height-conditioned residual: concat F90+embed, 3x3 80→64, ReLU, 1x1 64→1.

    No BatchNorm. No residual block. No extra tower.

    Args:
        feat_channels: F90 channels (64).
        embed_channels: Height embedding channels (16).
    """

    def __init__(self, feat_channels: int = 64, embed_channels: int = HEIGHT_EMBED_DIM) -> None:
        super().__init__()
        in_channels = int(feat_channels) + int(embed_channels)
        self.spatial = nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pred = nn.Conv2d(feat_channels, 1, kernel_size=1)

    def forward(self, f90: torch.Tensor, height_embed: torch.Tensor) -> torch.Tensor:
        """Predict ``delta`` on the canonical 90x160 grid.

        Args:
            f90: ``[N, 64, 90, 160]``.
            height_embed: ``[N, 16, 90, 160]``.

        Returns:
            ``delta_pred`` of shape ``[N, 90, 160]``.
        """
        fused = torch.cat([f90, height_embed], dim=1)
        return self.pred(self.relu(self.spatial(fused))).squeeze(1)


def build_depth_heads(model_cfg: Dict[str, Any]) -> nn.ModuleDict:
    """Build categorical DepthHeads for vehicle and RSU only."""
    heads = nn.ModuleDict()
    for agent_type in CATEGORICAL_DEPTH_AGENTS:
        ddiscr = model_cfg[agent_type]["cam"]["grid_conf"]["ddiscr"]
        heads[agent_type] = DepthHead(num_bins=int(ddiscr[2]))
    return heads

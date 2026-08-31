"""Concat fusion: native R2 + decoded F45 → shared F90."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from opencood.models.gaussian_modules_0822.p1_layout import F90_CHANNELS


class HighResFusion(nn.Module):
    """cat(R2, bilinear-up F45) → Conv3x3 280→128 → ReLU → Conv3x3 128→128 → ReLU."""

    def __init__(
        self,
        r2_channels: int = 24,
        f45_channels: int = 256,
        out_channels: int = F90_CHANNELS,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.conv1 = nn.Conv2d(
            int(r2_channels) + int(f45_channels), self.out_channels, 3, padding=1
        )
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, r2: torch.Tensor, f45: torch.Tensor) -> torch.Tensor:
        f45_up = F.interpolate(
            f45, size=r2.shape[-2:], mode="bilinear", align_corners=True
        )
        return self.relu(self.conv2(self.relu(self.conv1(torch.cat([r2, f45_up], 1)))))

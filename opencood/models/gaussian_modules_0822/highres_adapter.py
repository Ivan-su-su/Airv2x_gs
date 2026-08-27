"""Shallow FPN-style 90x160 fusion: R2 + decoded F45, 64-channel sum.

R2 is native stride-4 detail. F45 is the up1/up2 decoded deep feature,
not raw R3. Both laterals use default Conv initialization so R2 participates
from step 0. Interpolation matches the previous P1 bilinear convention.

One HighResFusion per agent. Heatmap and Depth consume the same ``F90``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HighResFusion(nn.Module):
    """``F90 = Conv1x1(R2, 24→64) + upsample(Conv1x1(F45, 256→64))``.

    Upsample: bilinear, ``align_corners=True`` (same as the previous P1
    ``F.interpolate`` on F45). Exact spatial size is taken from R2.

    Args:
        r2_channels: Official ``reduction_2`` channels (EfficientNet-b0: 24).
        f45_channels: Official up2 / CamEncode chain channels (256).
        out_channels: Shared F90 width (64).
    """

    def __init__(
        self,
        r2_channels: int = 24,
        f45_channels: int = 256,
        out_channels: int = 64,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.r2_lateral = nn.Conv2d(r2_channels, out_channels, kernel_size=1, bias=True)
        self.f45_lateral = nn.Conv2d(f45_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, r2: torch.Tensor, f45: torch.Tensor) -> torch.Tensor:
        """Fuse native R2 with decoded F45.

        Args:
            r2: ``[N, 24, 90, 160]``.
            f45: ``[N, 256, 45, 80]``.

        Returns:
            ``f90`` of shape ``[N, 64, 90, 160]``.
        """
        r2_64 = self.r2_lateral(r2)
        f45_64 = self.f45_lateral(f45)
        f45_up = F.interpolate(
            f45_64,
            size=r2_64.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        return r2_64 + f45_up

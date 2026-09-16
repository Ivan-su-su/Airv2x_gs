# -*- coding: utf-8 -*-
"""Channel / spatial resizer used by official NegoCollab ComminPub."""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResizeNet(nn.Module):
    """Resize a BEV feature to a target `(C, H, W)`."""

    def __init__(self, input_dim: int, out_dim: int, unify_method: str = "conv") -> None:
        super().__init__()
        if unify_method == "conv":
            self.channel_unify = nn.Conv2d(input_dim, out_dim, kernel_size=1)
        elif unify_method == "conv3":
            self.channel_unify = nn.Conv2d(input_dim, out_dim, kernel_size=3, padding=1)
        elif unify_method == "identity":
            self.channel_unify = nn.Identity()
        else:
            raise ValueError(f"Unsupported unify_method: {unify_method}")

    def forward(self, x: Tensor, target_size: Sequence[int]) -> Tensor:
        """Interpolate spatially then align channels if needed."""
        _batch, channels, height, width = x.size()
        target_c, target_h, target_w = int(target_size[0]), int(target_size[1]), int(target_size[2])
        if (height, width) != (target_h, target_w):
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        if channels != target_c:
            x = self.channel_unify(x)
        return x

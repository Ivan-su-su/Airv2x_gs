"""Agent-specific objectness head. Raw logits only."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from opencood.models.gaussian_modules_0822.p1_layout import F90_CHANNELS, NUM_CLASSES


class HeatmapHead(nn.Module):
    """Lightweight classifier on 128-channel F90: 3x3 128→128, ReLU, 1x1 128→2.

    No BatchNorm. No extra tower. No residual block. No sigmoid.

    Args:
        in_channels: Shared ``F90`` channels (128).
        num_classes: Objectness classes including background (0/1).
    """

    def __init__(
        self,
        in_channels: int = F90_CHANNELS,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        if int(num_classes) != int(NUM_CLASSES):
            raise ValueError(
                f"P1 HeatmapHead is 2-class objectness, got num_classes={num_classes}"
            )
        self.spatial = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cls = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, f90: torch.Tensor) -> torch.Tensor:
        """Predict per-cell objectness logits from shared ``F90``.

        Args:
            f90: ``[N, 128, 90, 160]``.

        Returns:
            ``heatmap_logits`` of shape ``[N, 2, 90, 160]``.
        """
        return self.cls(self.relu(self.spatial(f90)))


def build_heatmap_heads(model_cfg: Dict[str, Any]) -> nn.ModuleDict:
    """Build one HeatmapHead per agent type."""
    num_classes = int(model_cfg.get("heatmap_num_classes", NUM_CLASSES))
    heads = nn.ModuleDict()
    for agent_type in ("vehicle", "rsu", "drone"):
        heads[agent_type] = HeatmapHead(num_classes=num_classes)
    return heads

"""Per-agent residual feature adapter for cross-agent domain alignment."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from opencood.models.gaussian_modules_0822.p1_layout import AGENT_TYPES, F90_CHANNELS

DEFAULT_BOTTLENECK = 64


class AgentResidualAdapter(nn.Module):
    """Bottleneck residual MLP, one independent set per agent.

    Stage-2 (default): ``output = x + MLP_agent(x)``.

    Stage-3 mode (``gamma_init`` set): ``output = x + gamma * LN0(MLP(LN(x)))``
    where ``LN0`` is a non-affine LayerNorm and ``gamma`` is a learnable
    per-channel scale initialized to ``gamma_init``.

    Args:
        feature_dim: Gaussian feature width ``C``.
        bottleneck: Bottleneck width.
        gamma_init: If set, enable Stage-3 mode with this initial gamma.
    """

    def __init__(
        self,
        feature_dim: int = F90_CHANNELS,
        bottleneck: int = DEFAULT_BOTTLENECK,
        gamma_init: Optional[float] = None,
    ) -> None:
        super().__init__()
        width = int(feature_dim)
        self.adapters = nn.ModuleDict(
            {
                agent: nn.Sequential(
                    nn.Linear(width, int(bottleneck)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(bottleneck), width),
                )
                for agent in AGENT_TYPES
            }
        )
        self.scaled = gamma_init is not None
        self.gamma: Optional[nn.Parameter]
        self.input_norm: Optional[nn.LayerNorm]
        self.delta_norm: Optional[nn.LayerNorm]
        if self.scaled:
            self.gamma = nn.Parameter(torch.full((width,), float(gamma_init)))
            self.input_norm = nn.LayerNorm(width)
            self.delta_norm = nn.LayerNorm(width, elementwise_affine=False)
        else:
            self.gamma = None
            self.input_norm = None
            self.delta_norm = None

    def forward(self, feature: torch.Tensor, agent: str) -> torch.Tensor:
        """Residually adapt features in the last dimension.

        Args:
            feature: ``[..., C]`` source tokens or pooled Gaussian features.
            agent: ``vehicle`` / ``rsu`` / ``drone``.

        Returns:
            Same shape as ``feature``. Stage-2: ``x + MLP(x)``. Stage-3:
            ``x + gamma * LN0(MLP(LN(x)))``.
        """
        mlp = self.adapters[str(agent).lower()]
        if self.input_norm is None or self.gamma is None or self.delta_norm is None:
            return feature + mlp(feature)
        delta = self.delta_norm(mlp(self.input_norm(feature)))
        gamma = self.gamma.to(device=feature.device, dtype=feature.dtype)
        return feature + gamma * delta

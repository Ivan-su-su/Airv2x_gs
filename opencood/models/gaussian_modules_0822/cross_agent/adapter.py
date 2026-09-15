"""Per-agent residual feature adapter for cross-agent domain alignment."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from opencood.models.gaussian_modules_0822.p1_layout import AGENT_TYPES, F90_CHANNELS

DEFAULT_BOTTLENECK = 64


class AgentResidualAdapter(nn.Module):
    """Bottleneck residual MLP, one independent set per agent.

    Stage-2 (``prenorm=False``, default): ``output = x + MLP(x)``.

    Stage-3 (``prenorm=True``): standard PreNorm residual
    ``output = x + MLP(LN(x))``. The old gated form
    ``x + gamma * LN0(MLP(LN(x)))`` (learnable ``gamma``, output-side
    non-affine LN) is removed; see git history to roll back.
    """

    def __init__(
        self,
        feature_dim: int = F90_CHANNELS,
        bottleneck: int = DEFAULT_BOTTLENECK,
        prenorm: bool = False,
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
        self.prenorm = bool(prenorm)
        self.input_norm: Optional[nn.LayerNorm]
        if self.prenorm:
            self.input_norm = nn.LayerNorm(width)
        else:
            self.input_norm = None

    def forward(self, feature: torch.Tensor, agent: str) -> torch.Tensor:
        """Residually adapt features in the last dimension.

        Args:
            feature: ``[..., C]`` source tokens or pooled Gaussian features.
            agent: ``vehicle`` / ``rsu`` / ``drone``.

        Returns:
            Same shape as ``feature``. Stage-2: ``x + MLP(x)``. Stage-3:
            ``x + MLP(LN(x))``.
        """
        mlp = self.adapters[str(agent).lower()]
        if self.input_norm is None:
            return feature + mlp(feature)
        return feature + mlp(self.input_norm(feature))

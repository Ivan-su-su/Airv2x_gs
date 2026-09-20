"""Stage-3 contextual Gaussian interaction and BEV formation.

pooled Gaussian → shared C→2C split MLP
    ├─ self: intra-agent attention at real length
    └─ cross: Stage-3 agent residual adapter → cross-agent attention
concat → 2C→C + pooled residual → BEV splat.

Feature-only: mean / scale / quaternion are never touched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn

from opencood.models.gaussian_modules_0822.attention.interaction_block import (
    GaussianInteractionBlock,
)
from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.cross_agent.adapter import (
    AgentResidualAdapter,
)
from opencood.models.gaussian_modules_0822.fusion.gaussian_bev_splat import (
    GaussianBEVSplat,
)
from opencood.models.gaussian_modules_0822.fusion.gaussian_opacity import (
    GaussianOpacityCalibrator,
)
from opencood.models.gaussian_modules_0822.fusion.gaussian_voxel_pool import (
    DEFAULT_POINT_CLOUD_RANGE,
    DEFAULT_VOXEL_SIZE,
    GaussianVoxelPool,
)
from opencood.models.gaussian_modules_0822.p1_layout import F90_CHANNELS

DEFAULT_NUM_HEADS = 4
DEFAULT_ADAPTER_BOTTLENECK = 64


def _blocks_by_batch_agent(
    batch_index: torch.Tensor,
    n_batch: int,
    agent_slices: Mapping[str, Tuple[int, int]],
) -> List[Tuple[int, str, torch.Tensor]]:
    blocks = []
    for b in range(n_batch):
        for agent, (start, end) in agent_slices.items():
            rows = torch.arange(start, end, device=batch_index.device)[
                batch_index[start:end] == b
            ]
            if int(rows.numel()) > 0:
                blocks.append((b, agent, rows))
    return blocks


class GaussianContextInteraction(nn.Module):
    """Stage-3 orchestrator operating on pooled Gaussians."""

    def __init__(
        self,
        feature_dim: int = F90_CHANNELS,
        num_heads: int = DEFAULT_NUM_HEADS,
        adapter_bottleneck: int = DEFAULT_ADAPTER_BOTTLENECK,
        voxel_size: Any = DEFAULT_VOXEL_SIZE,
        point_cloud_range: Any = DEFAULT_POINT_CLOUD_RANGE,
        n_sigma: float = 3.0,
        adapter: Optional[AgentResidualAdapter] = None,
        splat: Optional[GaussianBEVSplat] = None,
        attn_cfg: Optional[Mapping[str, Any]] = None,
        opacity_cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        attn = dict(attn_cfg or {})
        heads = int(attn.get("heads", num_heads))
        ffn_dim = int(attn.get("ffn_dim", 2 * feature_dim))
        n_self = int(attn.get("stage3_self_layers", 1))
        n_cross = int(attn.get("stage3_cross_layers", 1))
        if feature_dim % heads:
            raise ValueError(
                f"feature_dim {feature_dim} not divisible by heads {heads}"
            )
        self.feature_dim = int(feature_dim)
        self.pool = GaussianVoxelPool(
            voxel_size=voxel_size, point_cloud_range=point_cloud_range
        )
        opacity = dict(opacity_cfg or {})
        self.opacity_enabled = bool(opacity.get("enabled", False))
        self.opacity_agent_specific = bool(opacity.get("agent_specific", True))
        self.opacity_heads = nn.ModuleDict()
        if self.opacity_enabled:
            agents = [str(a) for a in opacity.get("agents", ("vehicle", "rsu", "drone"))]
            keys = agents if self.opacity_agent_specific else ["shared"]
            for key in keys:
                self.opacity_heads[key] = GaussianOpacityCalibrator(
                    hidden_dim=int(opacity.get("hidden_dim", 16)),
                    init_prob=float(opacity.get("init_prob", 0.95)),
                    detach_covariance=bool(opacity.get("detach_covariance", True)),
                    cell_x=float(voxel_size[0]),
                    cell_y=float(voxel_size[1]),
                )
        self.split_mlp = nn.Linear(feature_dim, 2 * feature_dim)
        self.self_blocks = nn.ModuleList(
            [GaussianInteractionBlock(feature_dim, heads, ffn_dim) for _ in range(n_self)]
        )
        self.cross_blocks = nn.ModuleList(
            [GaussianInteractionBlock(feature_dim, heads, ffn_dim) for _ in range(n_cross)]
        )
        self.adapter = adapter or AgentResidualAdapter(
            feature_dim=feature_dim,
            bottleneck=adapter_bottleneck,
            prenorm=True,
        )
        self.self_fusion_norm = nn.LayerNorm(feature_dim)
        self.cross_fusion_norm = nn.LayerNorm(feature_dim)
        self.fusion_mlp = nn.Linear(2 * feature_dim, feature_dim)
        self.splat = splat or GaussianBEVSplat(
            feature_dim=feature_dim,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            n_sigma=n_sigma,
        )

    def _calibrate_opacity(
        self, agent: str, gaussians: GaussianSet
    ) -> torch.Tensor:
        key = agent if self.opacity_agent_specific else "shared"
        if key not in self.opacity_heads:
            raise KeyError(
                f"opacity head for {agent!r} is missing; available={list(self.opacity_heads.keys())}"
            )
        return self.opacity_heads[key](gaussians)

    def forward(
        self,
        gaussians_by_agent: Mapping[str, GaussianSet],
    ) -> Tuple[GaussianSet, torch.Tensor]:
        pooled: Dict[str, GaussianSet] = {}
        for agent, gs in gaussians_by_agent.items():
            if gs.n_gaussians <= 0:
                continue
            if self.opacity_enabled:
                alpha_pre = self._calibrate_opacity(agent, gs)
                pooled_gs = self.pool(
                    gs,
                    weights=alpha_pre,
                )
                alpha_final = self._calibrate_opacity(agent, pooled_gs)
                pooled_gs = pooled_gs.replace_opacity(alpha_final)
            else:
                pooled_gs = self.pool(gs)
            pooled[agent] = pooled_gs
        sample = next(iter(gaussians_by_agent.values()))
        if not pooled:
            empty = GaussianSet.empty(
                sample.mean.device, sample.feature.dtype, self.feature_dim
            )
            return empty, self.splat(empty)

        merged, agent_slices = self._merge(pooled)
        pooled_feature = merged.feature
        split = self.split_mlp(pooled_feature)
        self_feature = split[:, : self.feature_dim]
        cross_feature = split[:, self.feature_dim :]

        batch = merged.batch_index
        n_batch = int(batch.max().item()) + 1
        groups = _blocks_by_batch_agent(batch, n_batch, agent_slices)

        self_out = self_feature.clone()
        for _b, _agent, rows in groups:
            x = self_feature[rows].unsqueeze(0)
            for block in self.self_blocks:
                x = block(x)
            self_out[rows] = x.squeeze(0)

        adapted = cross_feature.clone()
        for agent, (start, end) in agent_slices.items():
            if end <= start:
                continue
            adapted[start:end] = self.adapter(cross_feature[start:end], agent)

        cross_out = cross_feature.clone()
        for _b, agent, rows in groups:
            others = [
                (a2, r2) for b2, a2, r2 in groups if b2 == _b and a2 != agent
            ]
            if not others:
                continue
            query = adapted[rows].unsqueeze(0)
            kv = torch.cat([adapted[r2] for _a2, r2 in others], dim=0).unsqueeze(0)
            for block in self.cross_blocks:
                query = block(query, kv)
            cross_out[rows] = query.squeeze(0)

        fusion_input = torch.cat(
            [self.self_fusion_norm(self_out), self.cross_fusion_norm(cross_out)],
            dim=-1,
        )
        delta = self.fusion_mlp(fusion_input)
        fused = pooled_feature + delta
        out = merged.replace_feature(fused)
        return out, self.splat(out)

    def _merge(
        self, pooled: Mapping[str, GaussianSet]
    ) -> Tuple[GaussianSet, Dict[str, Tuple[int, int]]]:
        sets = list(pooled.values())
        device = sets[0].mean.device
        merged = GaussianSet(
            mean=torch.cat([gs.mean for gs in sets]),
            scale=torch.cat([gs.scale for gs in sets]),
            quaternion=torch.cat([gs.quaternion for gs in sets]),
            feature=torch.cat([gs.feature for gs in sets]),
            p_fg=(
                torch.cat([gs.p_fg for gs in sets])
                if all(gs.p_fg is not None for gs in sets)
                else None
            ),
            opacity=(
                torch.cat([gs.opacity for gs in sets])
                if all(gs.opacity is not None for gs in sets)
                else None
            ),
            batch_index=torch.cat(
                [
                    gs.batch_index
                    if gs.batch_index is not None
                    else torch.zeros(
                        gs.n_gaussians, dtype=torch.long, device=device
                    )
                    for gs in sets
                ]
            ),
            agent="mixed",
            frame=sets[0].frame,
        )
        agent_slices: Dict[str, Tuple[int, int]] = {}
        start = 0
        for agent, gs in pooled.items():
            agent_slices[agent] = (start, start + gs.n_gaussians)
            start += gs.n_gaussians
        return merged, agent_slices


def build_gaussian_context_interaction(
    feature_dim: int = F90_CHANNELS,
    stage3_cfg: Optional[Mapping[str, Any]] = None,
    adapter: Optional[AgentResidualAdapter] = None,
    attn_cfg: Optional[Mapping[str, Any]] = None,
) -> GaussianContextInteraction:
    cfg = dict(stage3_cfg or {})
    attn = dict(attn_cfg or {})
    if "heads" not in attn and "num_heads" in cfg:
        attn["heads"] = cfg["num_heads"]
    return GaussianContextInteraction(
        feature_dim=feature_dim,
        num_heads=int(attn.get("heads", DEFAULT_NUM_HEADS)),
        adapter_bottleneck=int(
            cfg.get("adapter_bottleneck", DEFAULT_ADAPTER_BOTTLENECK)
        ),
        voxel_size=cfg.get("voxel_size", DEFAULT_VOXEL_SIZE),
        point_cloud_range=cfg.get("point_cloud_range", DEFAULT_POINT_CLOUD_RANGE),
        n_sigma=float(cfg.get("bev_n_sigma", 3.0)),
        adapter=adapter,
        attn_cfg=attn,
        opacity_cfg=cfg.get("opacity"),
    )

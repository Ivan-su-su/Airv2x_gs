"""Stage-2 cross-agent Gaussian interaction.

Query Gaussians (Stage-1 output, ego frame) sample the source agent's
cameras, align SOURCE image tokens with an agent residual adapter, attend
onto those tokens from the original Gaussian feature, then refine geometry.

Supported interactions (query ← source):

* ``drone_from_ground``:  drone ← vehicle + rsu
* ``vehicle_from_drone``: vehicle ← drone
* ``rsu_from_drone``:     rsu ← drone

Queries with zero valid source tokens are exact identity and skip the
adapter / Transformer / Refiner.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn

from opencood.models.gaussian_modules_0822.attention.interaction_block import (
    GaussianInteractionBlock,
)
from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.base.gaussian_geometry_encoder import (
    GaussianGeometryEncoder,
)
from opencood.models.gaussian_modules_0822.cross_agent.adapter import (
    AgentResidualAdapter,
)
from opencood.models.gaussian_modules_0822.p1_layout import F90_CHANNELS
from opencood.models.gaussian_modules_0822.projection.mamba_projection_wrapper import (
    MambaProjectionWrapper,
    flatten_view_tokens,
)
from opencood.models.gaussian_modules_0822.refinement.gaussian_refiner import (
    GaussianRefiner,
)
from opencood.models.gaussian_modules_0822.sampling.gaussian_offset_sampler import (
    DEFAULT_NUM_OFFSETS,
    GaussianOffsetSampler,
)

INTERACTION_TYPES: Tuple[str, ...] = (
    "drone_from_ground",
    "vehicle_from_drone",
    "rsu_from_drone",
)


def interaction_sources(interaction_type: str) -> Tuple[str, Tuple[str, ...]]:
    """Map an interaction name to ``(query_agent, source_agents)``."""
    if interaction_type == "drone_from_ground":
        return "drone", ("vehicle", "rsu")
    if interaction_type == "vehicle_from_drone":
        return "vehicle", ("drone",)
    if interaction_type == "rsu_from_drone":
        return "rsu", ("drone",)
    raise KeyError(
        f"unknown interaction '{interaction_type}'; "
        f"expected one of {list(INTERACTION_TYPES)}"
    )


def _interaction_stacks(
    feature_dim: int, attn_cfg: Optional[Mapping[str, Any]]
) -> nn.ModuleDict:
    attn = dict(attn_cfg or {})
    heads = int(attn.get("heads", 4))
    ffn_dim = int(attn.get("ffn_dim", 2 * feature_dim))
    n_layers = int(attn.get("stage2_layers", 2))
    return nn.ModuleDict(
        {
            itype: nn.ModuleList(
                [
                    GaussianInteractionBlock(feature_dim, heads, ffn_dim)
                    for _ in range(n_layers)
                ]
            )
            for itype in INTERACTION_TYPES
        }
    )


class CrossAgentInteraction(nn.Module):
    """All Stage-2 directions: sample once → source adapter → blocks → refine."""

    def __init__(
        self,
        feature_dim: int = F90_CHANNELS,
        num_offsets: int = DEFAULT_NUM_OFFSETS,
        geometry_encoder: Optional[GaussianGeometryEncoder] = None,
        sampler: Optional[GaussianOffsetSampler] = None,
        projector: Optional[MambaProjectionWrapper] = None,
        adapter: Optional[AgentResidualAdapter] = None,
        attention: Optional[nn.ModuleDict] = None,
        refiner: Optional[GaussianRefiner] = None,
        refinement_cfg: Optional[Mapping[str, Any]] = None,
        attn_cfg: Optional[Mapping[str, Any]] = None,
        refine: bool = True,
    ) -> None:
        super().__init__()
        cfg = dict(refinement_cfg or {})
        self.refine = bool(refine) and bool(cfg.get("enabled", True))
        if sampler is not None:
            self.sampler = sampler
            self.geometry_encoder = sampler.geometry_encoder
        else:
            self.geometry_encoder = geometry_encoder or GaussianGeometryEncoder(
                geo_dim=int(dict(cfg.get("geometry") or {}).get("geo_dim", 64)),
                hidden_dim=int(dict(cfg.get("geometry") or {}).get("hidden_dim", 64)),
            )
            self.sampler = GaussianOffsetSampler(
                feature_dim=feature_dim,
                num_offsets=num_offsets,
                geometry_encoder=self.geometry_encoder,
            )
        self.projector = projector or MambaProjectionWrapper()
        self.adapter = adapter or AgentResidualAdapter(feature_dim=feature_dim)
        self.attention = attention or _interaction_stacks(feature_dim, attn_cfg)
        self.refiner: Optional[GaussianRefiner] = None
        if self.refine:
            self.refiner = refiner or GaussianRefiner(
                self.geometry_encoder, feature_dim=feature_dim, cfg=cfg
            )
        self.last_stats: Dict[str, Any] = {}

    def _sample_tokens(
        self,
        points: torch.Tensor,
        source_features: torch.Tensor,
        source_geometry: Mapping[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if int(points.shape[0]) == 0:
            n_tokens = int(source_features.shape[0]) * int(points.shape[1])
            empty = source_features.new_empty(
                (0, n_tokens, int(source_features.shape[1]))
            )
            return empty, torch.zeros(
                (0, n_tokens), dtype=torch.bool, device=empty.device
            )
        projected = self.projector(
            points,
            source_geometry,
            points_frame="ego",
            feature_maps=source_features,
        )
        return flatten_view_tokens(
            projected["sampled_features"], projected["valid_mask"]
        )

    def _interact(
        self,
        gaussians: GaussianSet,
        interaction_type: str,
        sources: Mapping[str, Mapping[str, Any]],
    ) -> GaussianSet:
        query_agent, source_agents = interaction_sources(interaction_type)
        if gaussians.agent and gaussians.agent.lower() != query_agent:
            raise ValueError(
                f"interaction '{interaction_type}' queries '{query_agent}' "
                f"but the set is '{gaussians.agent}'"
            )
        if gaussians.n_gaussians == 0:
            return gaussians
        present_sources = tuple(source for source in source_agents if source in sources)
        if not present_sources:
            return gaussians

        points = self.sampler(
            gaussians.mean,
            gaussians.scale,
            gaussians.quaternion,
            gaussians.feature,
            query_agent,
        )["samples"]
        token_groups: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        mask_groups: List[torch.Tensor] = []
        for source in present_sources:
            payload = sources[source]
            tokens, mask = self._sample_tokens(
                points, payload["features"], payload["geometry"]
            )
            token_groups.append((source, tokens, mask))
            mask_groups.append(mask)
        mask = torch.cat(mask_groups, dim=1)
        valid = mask.any(dim=1)
        n_total = int(gaussians.n_gaussians)
        n_valid = int(valid.sum().item())
        self.last_stats[interaction_type] = {
            "n_query": n_total,
            "n_valid": n_valid,
            "n_invalid": n_total - n_valid,
            "skip_ratio": 0.0 if n_total == 0 else 1.0 - n_valid / n_total,
        }
        if n_valid == 0:
            return gaussians

        idx = valid.nonzero(as_tuple=False).squeeze(1)
        sub = gaussians.index_select(idx)
        adapted: List[torch.Tensor] = []
        for source, tokens, _mask in token_groups:
            adapted.append(self.adapter(tokens.index_select(0, idx), source))
        tokens_v = torch.cat(adapted, dim=1)
        mask_v = mask.index_select(0, idx)
        attended = sub.feature
        for block in self.attention[interaction_type]:
            attended = block(attended, tokens_v, mask_v)
        if self.refiner is None:
            updated = sub.replace_feature(attended)
        else:
            updated = self.refiner(sub, attended, agent_type=query_agent)
        if n_valid == n_total:
            return updated
        return gaussians.scatter_replace(idx, updated)

    def forward(
        self,
        gaussians: Mapping[str, GaussianSet],
        sources: Mapping[str, Mapping[str, Any]],
        interaction_type: Optional[str] = None,
    ) -> Dict[str, GaussianSet]:
        """Run one or all Stage-2 directions.

        Args:
            gaussians: Per-agent Stage-1 sets.
            sources: ``{agent: {features, geometry}}`` image payloads.
            interaction_type: If set, only that direction; else all three.
        """
        out = dict(gaussians)
        types = (interaction_type,) if interaction_type else INTERACTION_TYPES
        self.last_stats = {}
        for itype in types:
            query_agent, _sources = interaction_sources(itype)
            if query_agent not in out:
                continue
            out[query_agent] = self._interact(out[query_agent], itype, sources)
        total_q = total_v = 0
        for key, value in self.last_stats.items():
            if isinstance(value, dict) and "n_query" in value:
                total_q += int(value["n_query"])
                total_v += int(value["n_valid"])
        self.last_stats["total"] = {
            "n_query": total_q,
            "n_valid": total_v,
            "n_invalid": total_q - total_v,
            "skip_ratio": 0.0 if total_q == 0 else 1.0 - total_v / total_q,
        }
        return out


def build_cross_agent_interactions(
    feature_dim: int = F90_CHANNELS,
    num_offsets: int = DEFAULT_NUM_OFFSETS,
    refinement_cfg: Optional[Mapping[str, Any]] = None,
    refine: bool = True,
    geometry_encoder: Optional[GaussianGeometryEncoder] = None,
    sampler: Optional[GaussianOffsetSampler] = None,
    projector: Optional[MambaProjectionWrapper] = None,
    adapter: Optional[AgentResidualAdapter] = None,
    attention: Optional[nn.ModuleDict] = None,
    refiner: Optional[GaussianRefiner] = None,
    attn_cfg: Optional[Mapping[str, Any]] = None,
) -> CrossAgentInteraction:
    """Build one Stage-2 module owning adapter, attention stacks, and refiner."""
    return CrossAgentInteraction(
        feature_dim=feature_dim,
        num_offsets=num_offsets,
        geometry_encoder=geometry_encoder,
        sampler=sampler,
        projector=projector or MambaProjectionWrapper(),
        adapter=adapter or AgentResidualAdapter(feature_dim=feature_dim),
        attention=attention,
        refiner=refiner,
        refinement_cfg=refinement_cfg,
        attn_cfg=attn_cfg,
        refine=refine,
    )

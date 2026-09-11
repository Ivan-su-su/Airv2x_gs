"""Stage-1 same-CAV multi-view interaction.

Gaussian from CAV X attends that CAV's cameras only, then refines geometry.
Offset sampling runs once. Projection and grid_sample are packed to
``[n_cav, n_max]`` so the CUDA/einsum path is one launch. Transformer
blocks see only the real Gaussians, with same-CAV tokens ``V_cav * S``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn

from opencood.models.gaussian_modules_0822.attention.interaction_block import (
    GaussianInteractionBlock,
)
from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.base.gaussian_geometry_encoder import (
    GaussianGeometryEncoder,
)
from opencood.models.gaussian_modules_0822.base.projection import projection_mats
from opencood.models.gaussian_modules_0822.p1_layout import F90_CHANNELS
from opencood.models.gaussian_modules_0822.projection.mamba_projection_wrapper import (
    MambaProjectionWrapper,
)
from opencood.models.gaussian_modules_0822.refinement.gaussian_refiner import (
    GaussianRefiner,
)
from opencood.models.gaussian_modules_0822.sampling.gaussian_offset_sampler import (
    DEFAULT_NUM_OFFSETS,
    GaussianOffsetSampler,
)


class IntraViewInteraction(nn.Module):
    """Same-CAV image tokens → interaction blocks → one geometry refine."""

    def __init__(
        self,
        feature_dim: int = F90_CHANNELS,
        num_offsets: int = DEFAULT_NUM_OFFSETS,
        geometry_encoder: Optional[GaussianGeometryEncoder] = None,
        sampler: Optional[GaussianOffsetSampler] = None,
        projector: Optional[MambaProjectionWrapper] = None,
        refiner: Optional[GaussianRefiner] = None,
        refinement_cfg: Optional[Mapping[str, Any]] = None,
        attn_cfg: Optional[Mapping[str, Any]] = None,
        refine: bool = True,
        points_frame: str = "ego",
    ) -> None:
        super().__init__()
        cfg = dict(refinement_cfg or {})
        attn = dict(attn_cfg or {})
        self.points_frame = str(points_frame)
        self.refine = bool(refine) and bool(cfg.get("enabled", True))
        if sampler is not None:
            self.sampler = sampler
            self.geometry_encoder = sampler.geometry_encoder
        else:
            geo_cfg = dict(cfg.get("geometry") or {})
            self.geometry_encoder = geometry_encoder or GaussianGeometryEncoder(
                geo_dim=int(geo_cfg.get("geo_dim", 64)),
                hidden_dim=int(geo_cfg.get("hidden_dim", 64)),
            )
            self.sampler = GaussianOffsetSampler(
                feature_dim=feature_dim,
                num_offsets=num_offsets,
                geometry_encoder=self.geometry_encoder,
            )
        self.projector = projector or MambaProjectionWrapper()
        heads = int(attn.get("heads", 4))
        ffn_dim = int(attn.get("ffn_dim", 2 * feature_dim))
        n_layers = int(attn.get("stage1_layers", 2))
        self.blocks = nn.ModuleList(
            [
                GaussianInteractionBlock(feature_dim, heads, ffn_dim)
                for _ in range(n_layers)
            ]
        )
        self.refiner: Optional[GaussianRefiner] = None
        if self.refine:
            self.refiner = refiner or GaussianRefiner(
                self.geometry_encoder, feature_dim=feature_dim, cfg=cfg
            )
        self.last_stats: Dict[str, Any] = {}

    def forward(
        self,
        gaussians: GaussianSet,
        image_features: torch.Tensor,
        geometry: Mapping[str, Any],
        agent: Optional[str] = None,
        points_frame: Optional[str] = None,
    ) -> GaussianSet:
        if gaussians.n_gaussians == 0:
            self.last_stats = {"n_real": 0, "n_padded": 0, "n_tokens": 0}
            return gaussians
        if gaussians.view_index is None:
            raise ValueError("Stage-1 intra-view requires view_index")
        agent_key = str(agent or gaussians.agent).lower()
        n_cav = int(geometry["n_cav"])
        n_views = int(geometry["n_flat"]) // n_cav
        lidar2image, image_aug, image_hw = projection_mats(
            geometry, str(points_frame or self.points_frame)
        )
        lidar2image = lidar2image.reshape(n_cav, n_views, 4, 4)
        image_aug = image_aug.reshape(n_cav, n_views, 4, 4)
        maps = image_features.reshape(n_cav, n_views, *image_features.shape[1:])
        feat_hw = (int(maps.shape[-2]), int(maps.shape[-1]))

        cav_id = gaussians.view_index.long() // n_views
        sampled_xyz = self.sampler(
            gaussians.mean,
            gaussians.scale,
            gaussians.quaternion,
            gaussians.feature,
            agent_key,
        )["samples"]
        n_gaussians = int(cav_id.shape[0])
        n_samples = int(sampled_xyz.shape[1])
        channels = int(gaussians.feature.shape[-1])
        n_tokens = n_views * n_samples
        counts = torch.bincount(cav_id, minlength=n_cav)
        n_max = int(counts.max().item())
        order = torch.argsort(cav_id, stable=True)
        sorted_id = cav_id[order]
        starts = torch.ones(n_gaussians, dtype=torch.bool, device=cav_id.device)
        starts[1:] = sorted_id[1:] != sorted_id[:-1]
        start_pos = torch.arange(n_gaussians, device=cav_id.device)[starts]
        group = starts.to(dtype=torch.long).cumsum(0) - 1
        local = torch.empty(n_gaussians, dtype=torch.long, device=cav_id.device)
        local[order] = torch.arange(n_gaussians, device=cav_id.device) - start_pos[group]
        packed_xyz = sampled_xyz.new_zeros(n_cav, n_max, n_samples, 3)
        packed_xyz[cav_id, local] = sampled_xyz
        projected = self.projector.project(
            packed_xyz, lidar2image, image_aug, image_hw, feature_hw=feat_hw
        )
        sampled = self.projector.sample_features(
            maps, projected["uv_grid"], projected["valid_mask"]
        )
        tokens = sampled.permute(0, 2, 1, 3, 4).reshape(
            n_cav, n_max, n_tokens, channels
        )[cav_id, local]
        token_mask = projected["valid_mask"].permute(0, 2, 1, 3).reshape(
            n_cav, n_max, n_tokens
        )[cav_id, local]
        feat = gaussians.feature
        for block in self.blocks:
            feat = block(feat, tokens, token_mask)
        updated_feature = feat
        self.last_stats = {
            "agent": agent_key,
            "n_real": n_gaussians,
            "n_padded": int(n_cav * n_max),
            "n_cav": n_cav,
            "n_cav_nonempty": int((counts > 0).sum().item()),
            "n_tokens": n_tokens,
        }
        if self.refiner is None:
            return gaussians.replace_feature(updated_feature)
        return self.refiner(gaussians, updated_feature, agent_type=agent_key)

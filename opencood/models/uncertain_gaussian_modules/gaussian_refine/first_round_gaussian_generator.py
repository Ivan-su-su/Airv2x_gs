from typing import Dict, List, Optional

import torch
import torch.nn as nn

from opencood.models.uncertain_gaussian_modules.agent_inner_fusion import (
    DeformableGaussianCrossAttention,
)
from opencood.models.uncertain_gaussian_modules.gaussian_refine.gaussian_geometry_utils import (
    decompose_covariance_to_scale_rotation,
)
from opencood.models.uncertain_gaussian_modules.gaussian_refine.multiview_gaussian_fuser import (
    MultiViewGaussianFuser,
)


class FirstRoundGaussianGenerator(nn.Module):
    """Generate first-round gaussians from voxel queries and view observations."""

    def __init__(
        self,
        model_cfg=None,
        multiview_fuser: Optional[MultiViewGaussianFuser] = None,
    ):
        super().__init__()
        self.model_cfg = model_cfg or {}
        cross_attention_cfg = self.model_cfg.get("cross_attention", self.model_cfg)
        self.feature_alignment_dim = int(cross_attention_cfg.get("attention_dim", 128))
        self.lidar_feature_align = nn.Sequential(
            nn.LazyLinear(self.feature_alignment_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_alignment_dim, self.feature_alignment_dim),
        )
        self.image_feature_align = nn.Sequential(
            nn.LazyLinear(self.feature_alignment_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_alignment_dim, self.feature_alignment_dim),
        )
        self.cross_attention = DeformableGaussianCrossAttention(
            cross_attention_cfg
        )
        self.multiview_fuser = (
            multiview_fuser
            if multiview_fuser is not None
            else MultiViewGaussianFuser(
                self.model_cfg.get("multiview_fuser", self.model_cfg)
            )
        )

    def _get_agent_batch(self, batch_dict: Dict, agent: Optional[str]) -> Dict:
        """Read the current agent sub-dict when the caller forwards agent explicitly."""
        if agent is not None and agent in batch_dict and isinstance(batch_dict[agent], dict):
            return batch_dict[agent]
        return batch_dict

    def _get_num_views(self, agent_batch: Dict) -> int:
        """Infer the current agent-local number of camera views."""
        image_feature = agent_batch.get("image_feature")
        assert torch.is_tensor(image_feature), "image_feature is required."
        assert image_feature.ndim == 5, "image_feature must have shape [B, num_views, C, H, W]."
        return int(image_feature.shape[1])

    def _align_query_features(
        self,
        query_features: torch.Tensor,
        source_is_image_only: torch.Tensor,
    ) -> torch.Tensor:
        """Align image-born and lidar-born query features before DA."""
        assert torch.is_tensor(source_is_image_only), "source_is_image_only is required."
        lidar_aligned = self.lidar_feature_align(query_features)
        image_aligned = self.image_feature_align(query_features)
        image_mask = source_is_image_only.bool().unsqueeze(-1)
        return torch.where(image_mask, image_aligned, lidar_aligned)

    def _set_empty_first_round(
        self, gp: Dict, current_agent: str, batch_dict: Dict
    ) -> Dict:
        """Store an empty first-round result and return the batch dict."""
        gp.setdefault("first_round_gaussians", {})
        gp["first_round_gaussians"][current_agent] = None
        return batch_dict

    def _forward_single_agent(
        self,
        batch_dict: Dict,
        current_agent: str,
    ) -> Dict:
        """Generate first-round gaussians for one agent."""
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        candidate_entries = gp.get("gaussian_candidates", {}).get(current_agent, [])
        if len(candidate_entries) == 0:
            return self._set_empty_first_round(gp, current_agent, batch_dict)
        merged_candidates: Dict[str, torch.Tensor] = {}
        all_keys = set().union(*(entry.keys() for entry in candidate_entries))
        for key in all_keys:
            values = [entry.get(key) for entry in candidate_entries if entry.get(key) is not None]
            if len(values) == 0:
                continue
            merged_candidates[key] = torch.cat(values, dim=0)

        agent_batch = self._get_agent_batch(batch_dict, current_agent)
        image_feature = agent_batch.get("image_feature")
        query_features = merged_candidates.get("feature")
        group_ids = merged_candidates.get("group_ids")
        hit_points_3d = merged_candidates.get("mean")
        normalized_coords = merged_candidates.get("normalized_coords")
        local_agent_ids = merged_candidates.get("local_agent_ids")
        view_ids = merged_candidates.get("view_ids")
        support_covariance_2d = merged_candidates.get("support_covariance_2d")
        sigma_3d = merged_candidates.get("sigma_3d")
        source_is_image_only = merged_candidates.get("source_is_image_only")

        if (
            image_feature is None
            or query_features is None
            or group_ids is None
            or hit_points_3d is None
            or normalized_coords is None
            or local_agent_ids is None
            or view_ids is None
            or sigma_3d is None
        ):
            return self._set_empty_first_round(gp, current_agent, batch_dict)

        if group_ids.numel() == 0:
            return self._set_empty_first_round(gp, current_agent, batch_dict)

        aligned_query_features = self._align_query_features(
            query_features=query_features,
            source_is_image_only=source_is_image_only,
        )
        num_views = self._get_num_views(agent_batch)
        cross_attn_dict = self.cross_attention(
            voxel_queries=aligned_query_features,
            image_feature_map=image_feature,
            normalized_coords=normalized_coords,
            local_agent_ids=local_agent_ids,
            view_ids=view_ids,
            num_views=num_views,
            support_covariance_2d=support_covariance_2d,
        )
        fused_gaussians = self.multiview_fuser.fuse_groups(
            source_group_ids=group_ids.long(),
            hit_points_3d=hit_points_3d,
            updated_view_features=cross_attn_dict["updated_view_features"],
            attended_view_tokens=cross_attn_dict["attended_view_tokens"],
            sigma_3d=sigma_3d,
            view_ids=view_ids,
            local_agent_ids=local_agent_ids,
            num_local_agents=int(image_feature.shape[0]),
        )
        covariance_state = decompose_covariance_to_scale_rotation(
            fused_gaussians["covariance"]
        )
        fused_gaussians["scale"] = covariance_state["axis_scales"]
        fused_gaussians["rotation"] = covariance_state["rotation"]
        gp.setdefault("first_round_gaussians", {})
        gp["first_round_gaussians"][current_agent] = fused_gaussians
        return batch_dict

    def forward(
        self,
        batch_dict: Dict,
        available_agents: Optional[List[str]] = None,
        agent: Optional[str] = None,
    ) -> Dict:
        """Run first-round gaussian generation for one or multiple agents."""
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        gp["available_agents"] = available_agents or gp.get("available_agents", [])

        if agent is not None:
            return self._forward_single_agent(batch_dict, agent)

        target_agents = [
            current_agent
            for current_agent in gp["available_agents"]
            if current_agent in batch_dict and isinstance(batch_dict[current_agent], dict)
        ]
        if len(target_agents) == 0 and (
            "image_feature" in batch_dict or "voxel_features" in batch_dict
        ):
            target_agents = ["vehicle"]

        for current_agent in target_agents:
            batch_dict = self._forward_single_agent(batch_dict, current_agent)
        return batch_dict

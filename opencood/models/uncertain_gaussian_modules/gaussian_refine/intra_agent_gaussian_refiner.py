from typing import Dict, List, Optional

import torch
import torch.nn as nn

from opencood.models.uncertain_gaussian_modules.agent_inner_fusion import (
    DeformableGaussianCrossAttention,
)
from opencood.models.uncertain_gaussian_modules.gaussian_refine.feature_driven_gaussian_geometry_update import (
    FeatureDrivenGaussianGeometryUpdate,
)
from opencood.models.uncertain_gaussian_modules.gaussian_refine.gaussian_keypoint_generator import (
    GaussianKeyPointGenerator,
)
from opencood.models.uncertain_gaussian_modules.gaussian_refine.multiview_gaussian_fuser import (
    MultiViewGaussianFuser,
)
from opencood.models.uncertain_gaussian_modules.geometry import (
    GaussianToImageProjector,
)


class IntraAgentGaussianRefiner(nn.Module):
    """执行第二轮 agent 内 refinement 与协方差融合。"""

    def __init__(
        self,
        model_cfg=None,
        multiview_fuser: Optional[MultiViewGaussianFuser] = None,
        feature_geometry_update: Optional[FeatureDrivenGaussianGeometryUpdate] = None,
    ):
        super().__init__()
        self.model_cfg = model_cfg or {}
        keypoint_cfg = self.model_cfg.get("keypoint_generator", self.model_cfg)
        projector_cfg = self.model_cfg.get("gaussian_to_image_projector", self.model_cfg)
        cross_attention_cfg = self.model_cfg.get("cross_attention", self.model_cfg)
        multiview_fuser_cfg = self.model_cfg.get("multiview_fuser", self.model_cfg)
        self.keypoint_generator = GaussianKeyPointGenerator(keypoint_cfg)
        self.gaussian_to_image_projector = GaussianToImageProjector(projector_cfg)
        self.cross_attention = DeformableGaussianCrossAttention(cross_attention_cfg)
        self.multiview_fuser = (
            multiview_fuser
            if multiview_fuser is not None
            else MultiViewGaussianFuser(multiview_fuser_cfg)
        )
        fgu_cfg = self.model_cfg.get("feature_geometry_update")
        if feature_geometry_update is not None:
            self.feature_geometry_update = feature_geometry_update
        elif isinstance(fgu_cfg, dict):
            self.feature_geometry_update = FeatureDrivenGaussianGeometryUpdate(fgu_cfg)
        else:
            self.feature_geometry_update = None

    def _get_agent_batch(self, batch_dict: Dict, agent: Optional[str]) -> Dict:
        """Read the current agent-local batch dict."""
        if agent is not None and agent in batch_dict and isinstance(batch_dict[agent], dict):
            return batch_dict[agent]
        return batch_dict

    def _get_num_views(self, agent_batch: Dict) -> int:
        """Infer the current agent-local number of views."""
        image_feature = agent_batch.get("image_feature")
        assert torch.is_tensor(image_feature), "image_feature is required."
        assert image_feature.ndim == 5, "image_feature must have shape [B, V, C, H, W]."
        return int(image_feature.shape[1])

    def _derive_local_agent_mask(self, first_round: Dict, agent_batch: Dict) -> torch.Tensor:
        """Derive or validate one `[G, B]` local-agent membership mask."""
        local_agent_mask = first_round.get("local_agent_mask")
        if torch.is_tensor(local_agent_mask):
            return local_agent_mask.bool()

        local_agent_ids = first_round.get("local_agent_ids")
        group_indices = first_round.get("group_indices")
        source_group_ids = first_round.get("source_group_ids")
        image_feature = agent_batch.get("image_feature")
        assert torch.is_tensor(image_feature), "image_feature is required to derive local_agent_mask."
        if (
            not torch.is_tensor(local_agent_ids)
            or not torch.is_tensor(group_indices)
            or not torch.is_tensor(source_group_ids)
        ):
            raise KeyError("first_round_gaussians must provide local_agent_mask or hit-level grouping fields.")
        local_agent_mask = torch.zeros(
            int(source_group_ids.shape[0]),
            int(image_feature.shape[0]),
            dtype=torch.bool,
            device=local_agent_ids.device,
        )
        local_agent_mask[group_indices.long(), local_agent_ids.long()] = True
        return local_agent_mask

    def _identity_second_round_output(
        self,
        first_round: Dict,
        local_agent_mask: torch.Tensor,
    ) -> Dict:
        """Return one identity refinement result when no second-round hit is valid."""
        output = {
            "mean": first_round["mean"].clone(),
            "scale": first_round["scale"].clone(),
            "rotation": first_round["rotation"].clone(),
            "feature": first_round["feature"].clone(),
            "covariance": first_round["covariance"].clone(),
            "local_agent_mask": local_agent_mask.clone(),
            "num_valid_views": torch.zeros(
                first_round["mean"].shape[0],
                dtype=torch.long,
                device=first_round["mean"].device,
            ),
            "multi_view_group_mask": torch.zeros(
                first_round["mean"].shape[0],
                dtype=torch.bool,
                device=first_round["mean"].device,
            ),
            "view_weights": first_round["mean"].new_empty((0,)),
            "group_indices": first_round["mean"].new_empty((0,), dtype=torch.long),
        }
        if torch.is_tensor(first_round.get("source_group_ids")):
            output["source_group_ids"] = first_round["source_group_ids"].clone()
        output["gaussian_indices"] = torch.arange(
            first_round["mean"].shape[0],
            device=first_round["mean"].device,
            dtype=torch.long,
        )
        return output

    def _forward_single_agent(self, batch_dict: Dict, current_agent: str) -> Dict:
        """Run the second-round intra-agent refinement for one agent."""
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        first_round = gp.get("first_round_gaussians", {}).get(current_agent)
        gp.setdefault("second_round_gaussians", {})
        if first_round is None:
            gp["second_round_gaussians"][current_agent] = None
            return batch_dict

        agent_batch = self._get_agent_batch(batch_dict, current_agent)
        image_feature = agent_batch.get("image_feature")
        assert torch.is_tensor(image_feature), "image_feature is required."
        local_agent_mask = self._derive_local_agent_mask(first_round, agent_batch)
        keypoint_dict = self.keypoint_generator(
            mean=first_round["mean"],
            axis_scales=first_round["scale"],
            rotation=first_round["rotation"],
            feature=first_round["feature"],
        )
        projection_dict = self.gaussian_to_image_projector(
            agent_batch=agent_batch,
            gaussian_key_points=keypoint_dict["key_points"],
            local_agent_mask=local_agent_mask,
        )
        gaussian_ids = projection_dict["gaussian_ids"]
        if gaussian_ids.numel() == 0:
            identity_output = self._identity_second_round_output(first_round, local_agent_mask)
            gp["second_round_gaussians"][current_agent] = identity_output
            return batch_dict

        num_views = self._get_num_views(agent_batch)
        gaussian_features = first_round["feature"][gaussian_ids.long()]
        cross_attn_dict = self.cross_attention(
            voxel_queries=gaussian_features,
            image_feature_map=image_feature,
            normalized_coords=projection_dict["normalized_coords"],
            local_agent_ids=projection_dict["local_agent_ids"],
            view_ids=projection_dict["view_ids"],
            num_views=num_views,
            support_covariance_2d=None,
            external_sampling_coords=projection_dict["sampling_coords"],
            sampling_valid_mask=projection_dict["sampling_valid_mask"],
        )

        second_round = self.multiview_fuser.fuse_groups(
            source_group_ids=gaussian_ids.long(),
            hit_points_3d=first_round["mean"][gaussian_ids.long()],
            updated_view_features=cross_attn_dict["updated_view_features"],
            attended_view_tokens=cross_attn_dict["attended_view_tokens"],
            sigma_3d=first_round["covariance"][gaussian_ids.long()],
            view_ids=projection_dict["view_ids"],
            local_agent_ids=projection_dict["local_agent_ids"],
            num_local_agents=int(image_feature.shape[0]),
        )
        identity_output = self._identity_second_round_output(first_round, local_agent_mask)
        gaussian_indices = second_round["source_group_ids"].clone()
        identity_output["feature"][gaussian_indices] = second_round["feature"]
        identity_output["num_valid_views"][gaussian_indices] = second_round["num_valid_views"]
        identity_output["multi_view_group_mask"][gaussian_indices] = (
            second_round["multi_view_group_mask"]
        )
        identity_output["refined_gaussian_mask"] = torch.zeros(
            first_round["mean"].shape[0],
            dtype=torch.bool,
            device=first_round["mean"].device,
        )
        identity_output["refined_gaussian_mask"][gaussian_indices] = True
        identity_output["view_weights"] = second_round["view_weights"]
        identity_output["group_indices"] = second_round["group_indices"]
        if self.feature_geometry_update is not None and gaussian_indices.numel() > 0:
            refined_features = identity_output["feature"][gaussian_indices]
            base_mean = first_round["mean"][gaussian_indices]
            axis_scales = first_round["scale"][gaussian_indices]
            rotation = first_round["rotation"][gaussian_indices]
            geometry_dict = self.feature_geometry_update(
                refined_features,
                base_mean,
                axis_scales,
                rotation,
            )
            identity_output["mean"][gaussian_indices] = geometry_dict["mean"]
            identity_output["scale"][gaussian_indices] = geometry_dict["axis_scales"]
            identity_output["rotation"][gaussian_indices] = geometry_dict["rotation"]
            identity_output["covariance"][gaussian_indices] = geometry_dict["covariance"]
        gp["second_round_gaussians"][current_agent] = identity_output
        return batch_dict

    def run_second_round_refinement(
        self, batch_dict: Dict, available_agents: Optional[List[str]] = None
    ) -> Dict:
        """Run second-round re-projection refinement for one or more agents."""
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        gp.setdefault("second_round_gaussians", {})
        target_agents = [
            current_agent
            for current_agent in (available_agents or gp.get("available_agents", []))
            if current_agent in batch_dict and isinstance(batch_dict[current_agent], dict)
        ]
        for current_agent in target_agents:
            batch_dict = self._forward_single_agent(batch_dict, current_agent)
        return batch_dict

    def forward(
        self, batch_dict: Dict, available_agents: Optional[List[str]] = None
    ) -> Dict:
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        gp["available_agents"] = available_agents or gp.get("available_agents", [])
        batch_dict = self.run_second_round_refinement(batch_dict, available_agents)
        return batch_dict

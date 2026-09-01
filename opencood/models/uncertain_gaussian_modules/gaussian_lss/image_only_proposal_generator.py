from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from opencood.models.uncertain_gaussian_modules.gaussian_lss.shared_depth_predictor import (
    SharedDepthPredictor,
)
from opencood.models.uncertain_gaussian_modules.geometry import (
    GaussianCovarianceBuilder,
    MultiViewGridSampler,
    estimate_local_patch_covariance,
    reshape_label_map,
    select_foreground_label_points,
)


class ImageOnlyProposalGenerator(nn.Module):
    """Generate minimal image-only Gaussian proposals on uncovered image cells."""

    def __init__(
        self,
        model_cfg=None,
        depth_predictor: Optional[SharedDepthPredictor] = None,
    ):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.heatmap_major_scale = float(self.model_cfg.get("heatmap_major_scale", 0.10))
        self.heatmap_minor_scale = float(self.model_cfg.get("heatmap_minor_scale", 0.04))
        self.heatmap_center_eps = float(self.model_cfg.get("heatmap_center_eps", 1e-6))
        self.local_patch_size = int(self.model_cfg.get("local_patch_size", 5))
        self.depth_predictor = depth_predictor or SharedDepthPredictor(self.model_cfg)
        self.gaussian_covariance_builder = GaussianCovarianceBuilder(
            self.model_cfg.get("gaussian_covariance_builder", self.model_cfg)
        )
        self.feature_sampler = MultiViewGridSampler(
            self.model_cfg.get("depth_feature_sampler", self.model_cfg)
        )

    def _get_agent_batch(self, batch_dict: Dict, agent: Optional[str]) -> Dict:
        """Read the current agent sub-dict when the caller forwards agent explicitly."""
        if agent is not None and agent in batch_dict and isinstance(batch_dict[agent], dict):
            return batch_dict[agent]
        return batch_dict

    def _get_camera_imgs(self, agent_batch: Dict) -> Optional[torch.Tensor]:
        """Read multiview images from one agent batch."""
        cam_inputs = agent_batch.get("batch_merged_cam_inputs")
        assert isinstance(cam_inputs, dict), "batch_merged_cam_inputs is required."
        imgs = cam_inputs.get("imgs")
        assert torch.is_tensor(imgs), "batch_merged_cam_inputs['imgs'] is required."
        return imgs

    def _gather_view_geometry(
        self,
        agent_batch: Dict,
        local_agent_ids: torch.Tensor,
        view_ids: torch.Tensor,
        image_shape_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Gather candidate-aligned camera parameters for 3D lifting."""
        cam_inputs = agent_batch.get("batch_merged_cam_inputs")
        if not isinstance(cam_inputs, dict):
            return {}
        intrinsics = cam_inputs.get("intrinsics")
        extrinsics = cam_inputs.get("extrinsics")
        post_rots = cam_inputs.get("post_rots")
        post_trans = cam_inputs.get("post_trans")
        assert torch.is_tensor(intrinsics), "batch_merged_cam_inputs['intrinsics'] is required."
        assert torch.is_tensor(extrinsics), "batch_merged_cam_inputs['extrinsics'] is required."
        assert torch.is_tensor(post_rots), "batch_merged_cam_inputs['post_rots'] is required."
        assert torch.is_tensor(post_trans), "batch_merged_cam_inputs['post_trans'] is required."

        return {
            "hit_intrinsics": intrinsics[local_agent_ids.long(), view_ids.long()],
            "hit_extrinsics": extrinsics[local_agent_ids.long(), view_ids.long()],
            "hit_post_rots": post_rots[local_agent_ids.long(), view_ids.long()],
            "hit_post_trans": post_trans[local_agent_ids.long(), view_ids.long()],
            "image_shape_hw": torch.tensor(
                image_shape_hw,
                device=view_ids.device,
                dtype=intrinsics.dtype,
            ),
        }

    def _backproject_points(
        self,
        normalized_coords: torch.Tensor,
        depth: torch.Tensor,
        hit_intrinsics: torch.Tensor,
        hit_extrinsics: torch.Tensor,
        hit_post_rots: torch.Tensor,
        hit_post_trans: torch.Tensor,
        image_shape_hw: torch.Tensor,
    ) -> torch.Tensor:
        """Backproject image-plane candidates into lidar coordinates."""
        if normalized_coords.numel() == 0:
            return normalized_coords.new_empty((0, 3))

        intrinsics_3 = self.gaussian_covariance_builder._ensure_intrinsics_3x3(hit_intrinsics)
        extrinsics_4 = self.gaussian_covariance_builder._ensure_extrinsics_4x4(hit_extrinsics)
        post_rot_2 = self.gaussian_covariance_builder._ensure_post_rot_2x2(hit_post_rots)
        post_trans_2 = self.gaussian_covariance_builder._ensure_post_trans_2(hit_post_trans)
        post_rot_inv = torch.inverse(post_rot_2)
        k_inv = torch.inverse(intrinsics_3)
        height, width = image_shape_hw.unbind()
        pixel_scale = torch.tensor(
            [width, height],
            device=normalized_coords.device,
            dtype=normalized_coords.dtype,
        ).view(1, 2)
        pixel_coords_aug = normalized_coords * pixel_scale
        pixel_coords = torch.matmul(
            post_rot_inv,
            (pixel_coords_aug - post_trans_2).unsqueeze(-1),
        ).squeeze(-1)
        homogeneous_pixels = torch.cat(
            [
                pixel_coords,
                torch.ones(
                    pixel_coords.shape[0],
                    1,
                    device=pixel_coords.device,
                    dtype=pixel_coords.dtype,
                ),
            ],
            dim=-1,
        )
        camera_rays = torch.matmul(k_inv, homogeneous_pixels.unsqueeze(-1)).squeeze(-1)
        camera_points = camera_rays * depth.clamp_min(self.gaussian_covariance_builder.eps)
        rotation_cam_to_lidar = extrinsics_4[:, :3, :3]
        translation_cam_to_lidar = extrinsics_4[:, :3, 3]
        return (
            torch.matmul(rotation_cam_to_lidar, camera_points.unsqueeze(-1)).squeeze(-1)
            + translation_cam_to_lidar
        )

    def _build_single_agent_proposals(
        self,
        batch_dict: Dict,
        current_agent: Optional[str],
    ) -> Dict:
        """Generate image-only proposals for one agent and append one candidate entry."""
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        agent_name = current_agent or "vehicle"
        agent_batch = self._get_agent_batch(batch_dict, current_agent)
        image_feature = agent_batch.get("image_feature")
        heatmap_feature = agent_batch.get("heatmap_feature")
        label_map = agent_batch.get("label_map")
        camera_imgs = self._get_camera_imgs(agent_batch)

        assert torch.is_tensor(image_feature), "image_feature is required."
        assert torch.is_tensor(label_map), "label_map is required."

        num_views = int(camera_imgs.shape[1])
        label_map = reshape_label_map(label_map, num_views)
        image_hw = tuple(int(v) for v in camera_imgs.shape[-2:])
        lidar_coverage_mask = (
            gp.get("projection_masks", {})
            .get(agent_name, {})
            .get("lidar_coverage_mask")
        )
        proposal_indices = select_foreground_label_points(
            label_map=label_map,
            lidar_coverage_mask=lidar_coverage_mask,
            image_hw=image_hw,
        )
        local_agent_ids = proposal_indices["local_agent_ids"]
        normalized_coords = proposal_indices["normalized_coords"]
        view_ids = proposal_indices["view_ids"]
        x_indices = proposal_indices["x_indices"]
        y_indices = proposal_indices["y_indices"]
        point_labels = proposal_indices["labels"]
        if normalized_coords.numel() == 0:
            return batch_dict

        sampled_features = self.feature_sampler.sample_feature_dict(
            {
                "image_feature": image_feature,
                "heatmap_feature": heatmap_feature,
            },
            normalized_coords,
            local_agent_ids,
            view_ids,
            num_views,
        )
        sampled_image_features = sampled_features["image_feature"]
        assert torch.is_tensor(sampled_image_features), "sampled image features are required."
        geometry_dict = self._gather_view_geometry(
            agent_batch=agent_batch,
            local_agent_ids=local_agent_ids,
            view_ids=view_ids,
            image_shape_hw=camera_imgs.shape[-2:],
        )
        depth_logits = self.depth_predictor.predict_agent_depth_logits(
            sampled_image_features, agent_name
        )
        depth_stats = self.depth_predictor.predict_depth_distribution(
            depth_logits, agent_name
        )
        pred_depth_mean = depth_stats["soft_depth_mean"]
        depth_variance = depth_stats["depth_variance"]
        sampled_heatmap_features = sampled_features["heatmap_feature"]
        sampled_strength = None
        if torch.is_tensor(sampled_heatmap_features):
            sampled_strength = sampled_heatmap_features.abs().mean(dim=-1, keepdim=True)
        support_covariance_2d = estimate_local_patch_covariance(
            label_map=label_map,
            point_labels=point_labels,
            point_coords=normalized_coords,
            local_agent_ids=local_agent_ids,
            view_ids=view_ids,
            x_indices=x_indices,
            y_indices=y_indices,
            image_hw=image_hw,
            patch_size=self.local_patch_size,
            major_scale=self.heatmap_major_scale,
            minor_scale=self.heatmap_minor_scale,
            eps=self.heatmap_center_eps,
            sampled_strength=sampled_strength,
        )
        base_mean_3d = self._backproject_points(
            normalized_coords=normalized_coords,
            depth=pred_depth_mean,
            hit_intrinsics=geometry_dict["hit_intrinsics"],
            hit_extrinsics=geometry_dict["hit_extrinsics"],
            hit_post_rots=geometry_dict["hit_post_rots"],
            hit_post_trans=geometry_dict["hit_post_trans"],
            image_shape_hw=geometry_dict["image_shape_hw"],
        )
        sigma_dict = self.gaussian_covariance_builder.build_view_covariance(
            support_covariance_2d=support_covariance_2d,
            depth_variance=depth_variance,
            anchor_depth=pred_depth_mean,
            normalized_coords=normalized_coords,
            hit_intrinsics=geometry_dict["hit_intrinsics"],
            hit_extrinsics=geometry_dict["hit_extrinsics"],
            hit_post_rots=geometry_dict["hit_post_rots"],
            hit_post_trans=geometry_dict["hit_post_trans"],
            image_shape_hw=geometry_dict["image_shape_hw"],
        )

        group_offset = 0
        existing_candidates = gp.get("gaussian_candidates", {}).get(agent_name, [])
        for existing_candidate in existing_candidates:
            existing_group_ids = existing_candidate.get("group_ids")
            if torch.is_tensor(existing_group_ids) and existing_group_ids.numel() > 0:
                group_offset = max(group_offset, int(existing_group_ids.max().item()) + 1)
        group_ids = torch.arange(
            group_offset,
            group_offset + normalized_coords.shape[0],
            device=normalized_coords.device,
            dtype=torch.long,
        )
        candidate_payload = {
            "feature": sampled_image_features,
            "normalized_coords": normalized_coords,
            "local_agent_ids": local_agent_ids.long(),
            "view_ids": view_ids.long(),
            "mean": base_mean_3d,
            "sigma_3d": sigma_dict["sigma_3d"],
            "support_covariance_2d": support_covariance_2d,
            "group_ids": group_ids,
            "source_is_image_only": torch.ones(
                normalized_coords.shape[0],
                dtype=torch.bool,
                device=normalized_coords.device,
            ),
        }
        gp.setdefault("gaussian_candidates", {})
        gp["gaussian_candidates"].setdefault(agent_name, [])
        gp["gaussian_candidates"][agent_name].append(candidate_payload)
        return batch_dict

    def build_image_only_proposals(
        self,
        batch_dict: Dict,
        available_agents: Optional[List[str]] = None,
        agent: Optional[str] = None,
    ) -> Dict:
        """Build image-only proposal candidates for one or multiple agents."""
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        gp["available_agents"] = available_agents or gp.get("available_agents", [])

        target_agents: List[Optional[str]]
        if agent is not None:
            target_agents = [agent]
        elif (
            "image_feature" in batch_dict
            or "camera_imgs" in batch_dict
            or "batch_merged_cam_inputs" in batch_dict
        ):
            target_agents = [None]
        else:
            target_agents = [
                current_agent
                for current_agent in gp["available_agents"]
                if current_agent in batch_dict and isinstance(batch_dict[current_agent], dict)
            ]

        for current_agent in target_agents:
            batch_dict = self._build_single_agent_proposals(batch_dict, current_agent)
        return batch_dict

    def forward(
        self, batch_dict: Dict, available_agents: Optional[List[str]] = None
    ) -> Dict:
        """Run image-only proposal generation for the available agents."""
        return self.build_image_only_proposals(
            batch_dict=batch_dict,
            available_agents=available_agents,
        )

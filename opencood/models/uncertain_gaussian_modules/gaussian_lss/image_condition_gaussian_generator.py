from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from opencood.models.uncertain_gaussian_modules.geometry import (
    GaussianCovarianceBuilder,
    LidarToImageProjector,
    MultiViewGridSampler,
    estimate_local_patch_covariance,
    lss_normalized_coords_to_feature_grid,
    reshape_label_map,
)
from opencood.models.uncertain_gaussian_modules.gaussian_lss.shared_depth_predictor import (
    SharedDepthPredictor,
)


class ImageConditionGaussianGenerator(nn.Module):
    """按 GaussianLSS 风格完成图像条件高斯生成。"""

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
        self.lidar_to_image_projector = LidarToImageProjector(
            self.model_cfg.get("lidar_to_image_projector", self.model_cfg)
        )
        self.gaussian_covariance_builder = GaussianCovarianceBuilder(
            self.model_cfg.get("gaussian_covariance_builder", self.model_cfg)
        )
        self.depth_feature_sampler = MultiViewGridSampler(
            self.model_cfg.get("depth_feature_sampler", self.model_cfg)
        )

    def _get_agent_batch(self, batch_dict: Dict, agent: Optional[str]) -> Dict:
        """Read the current agent sub-dict when the caller forwards agent explicitly."""
        if agent is not None and agent in batch_dict and isinstance(batch_dict[agent], dict):
            return batch_dict[agent]
        return batch_dict

    def _get_camera_imgs(self, agent_batch: Dict) -> Optional[torch.Tensor]:
        cam_inputs = agent_batch.get("batch_merged_cam_inputs")
        assert isinstance(cam_inputs, dict), "batch_merged_cam_inputs is required."
        imgs = cam_inputs.get("imgs")
        assert torch.is_tensor(imgs), "batch_merged_cam_inputs['imgs'] is required."
        return imgs

    def _filter_agent_batch(self, agent_batch, voxel_mask):
        filtered_agent_batch = dict(agent_batch)
        for key in ("voxel_coords", "voxel_features", "ori_coords_height"):
            value = agent_batch.get(key)
            if torch.is_tensor(value) and value.shape[0] == voxel_mask.shape[0]:
                filtered_agent_batch[key] = value[voxel_mask]
        return filtered_agent_batch

    def build_lidar_coverage_mask(
        self,
        normalized_coords: torch.Tensor,
        local_agent_ids: torch.Tensor,
        view_ids: torch.Tensor,
        batch_size: int,
        num_views: int,
        image_hw: Tuple[int, int],
        feature_map_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Mark the 2x2 integer neighborhood around each projected lidar hit."""
        feature_height, feature_width = feature_map_hw
        coverage_mask = torch.zeros(
            batch_size,
            num_views,
            feature_height,
            feature_width,
            dtype=torch.bool,
            device=normalized_coords.device,
        )
        if normalized_coords.numel() == 0:
            return coverage_mask

        feature_grid = lss_normalized_coords_to_feature_grid(
            normalized_coords=normalized_coords,
            feature_hw=feature_map_hw,
            image_hw=image_hw,
        )
        grid_x = feature_grid[:, 0]
        grid_y = feature_grid[:, 1]
        x0 = torch.floor(grid_x).long().clamp(0, max(feature_width - 1, 0))
        y0 = torch.floor(grid_y).long().clamp(0, max(feature_height - 1, 0))
        x1 = (x0 + 1).clamp(0, max(feature_width - 1, 0))
        y1 = (y0 + 1).clamp(0, max(feature_height - 1, 0))
        hit_local_agent_ids = local_agent_ids.long()
        hit_view_ids = view_ids.long()

        coverage_mask[hit_local_agent_ids, hit_view_ids, y0, x0] = True
        coverage_mask[hit_local_agent_ids, hit_view_ids, y0, x1] = True
        coverage_mask[hit_local_agent_ids, hit_view_ids, y1, x0] = True
        coverage_mask[hit_local_agent_ids, hit_view_ids, y1, x1] = True
        return coverage_mask

    def sample_projected_depth_features(
        self, batch_dict: Dict, agent: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Project current-agent voxels to images and sample per-hit depth features."""
        agent_batch = self._get_agent_batch(batch_dict, agent)
        image_feature = agent_batch.get("image_feature")
        heatmap_feature = agent_batch.get("heatmap_feature")
        label_map = agent_batch.get("semantic_feature")
        camera_imgs = self._get_camera_imgs(agent_batch)
        assert torch.is_tensor(image_feature), "image_feature is required."
        assert torch.is_tensor(label_map), "label_map is required."

        voxel_mask = agent_batch.get("instance_voxel_mask").bool()
        if voxel_mask is None or voxel_mask.sum() == 0:
            raise ValueError(f"No valid voxel mask found for agent {agent}")
        filtered_agent_batch = self._filter_agent_batch(agent_batch, voxel_mask)
        projection_dict = self.lidar_to_image_projector(filtered_agent_batch)

        import pdb; pdb.set_trace()
        batch_size = int(camera_imgs.shape[0])
        num_views = int(camera_imgs.shape[1])
        label_map = label_map.squeeze(2)
        label_map = reshape_label_map(label_map, num_views)
        projection_dict.update(
            self.gather_hit_camera_geometry(
                agent_batch=agent_batch,
                projection_dict=projection_dict,
                image_shape=camera_imgs.shape[-2:],
            )
        )
        sampled_features = self.depth_feature_sampler.sample_feature_dict(
            {
                "image_feature": image_feature,
                "heatmap_feature": heatmap_feature,
            },
            projection_dict["normalized_coords"],
            projection_dict["local_agent_ids"],
            projection_dict["view_ids"],
            num_views,
        )
        sampled_image_features = sampled_features["image_feature"]
        assert torch.is_tensor(sampled_image_features), "sampled image features are required."

        coverage_mask_hw = (int(label_map.shape[-2]), int(label_map.shape[-1]))
        lidar_coverage_mask = self.build_lidar_coverage_mask(
            normalized_coords=projection_dict["normalized_coords"],
            local_agent_ids=projection_dict["local_agent_ids"],
            view_ids=projection_dict["view_ids"],
            batch_size=batch_size,
            num_views=num_views,
            image_hw=tuple(int(v) for v in camera_imgs.shape[-2:]),
            feature_map_hw=coverage_mask_hw,
        )

        projection_dict["sampled_image_features"] = sampled_image_features
        projection_dict["sampled_heatmap_features"] = sampled_features["heatmap_feature"]
        projection_dict["lidar_coverage_mask"] = lidar_coverage_mask
        return projection_dict

    def gather_hit_camera_geometry(
        self,
        agent_batch: Dict,
        projection_dict: Dict[str, torch.Tensor],
        image_shape,
    ) -> Dict[str, torch.Tensor]:
        """Gather hit-aligned camera parameters for explicit covariance lifting."""
        cam_inputs = agent_batch.get("batch_merged_cam_inputs")
        local_agent_ids = projection_dict["local_agent_ids"].long()
        view_ids = projection_dict["view_ids"].long()
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

        geometry_dict: Dict[str, torch.Tensor] = {
            "hit_intrinsics": intrinsics[local_agent_ids, view_ids],
            "hit_extrinsics": extrinsics[local_agent_ids, view_ids],
            "hit_post_rots": post_rots[local_agent_ids, view_ids],
            "hit_post_trans": post_trans[local_agent_ids, view_ids],
            "image_shape_hw": torch.tensor(
                image_shape,
                device=view_ids.device,
                dtype=intrinsics.dtype,
            ),
        }
        return geometry_dict

    def build_anchor_depth(self, projection_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build anchor depth as the current lidar anchor's camera-frame z depth."""
        hit_points_3d = projection_dict.get("hit_points_3d")
        hit_extrinsics = projection_dict.get("hit_extrinsics")
        assert torch.is_tensor(hit_points_3d), "Projection dict must contain hit_points_3d."
        assert torch.is_tensor(hit_extrinsics), "Projection dict must contain hit_extrinsics."
        assert hit_extrinsics.shape[-2:] == (4, 4), "hit_extrinsics must have shape [..., 4, 4]."

        lidar_to_camera = torch.inverse(hit_extrinsics)
        hit_points_h = torch.cat(
            [
                hit_points_3d,
                torch.ones(
                    hit_points_3d.shape[0],
                    1,
                    device=hit_points_3d.device,
                    dtype=hit_points_3d.dtype,
                ),
            ],
            dim=-1,
        )
        points_cam = torch.matmul(lidar_to_camera, hit_points_h.unsqueeze(-1)).squeeze(-1)
        return points_cam[:, 2:3]

    def _estimate_support_covariance_from_projection(
        self,
        agent_batch: Dict,
        projection_dict: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Estimate heatmap-guided 2D support for one projected lidar observation set."""
        label_map = agent_batch.get("label_map")
        sampled_heatmap_features = projection_dict.get("sampled_heatmap_features")
        normalized_coords = projection_dict.get("normalized_coords")
        view_ids = projection_dict.get("view_ids")
        camera_imgs = self._get_camera_imgs(agent_batch)
        assert torch.is_tensor(label_map), "label_map is required."
        assert torch.is_tensor(normalized_coords), "normalized_coords are required."
        assert torch.is_tensor(view_ids), "view_ids are required."

        num_views = int(camera_imgs.shape[1])
        label_map = reshape_label_map(label_map, num_views)
        feature_hw = (int(label_map.shape[-2]), int(label_map.shape[-1]))
        image_hw = tuple(int(v) for v in camera_imgs.shape[-2:])
        local_agent_ids = projection_dict["local_agent_ids"].long()
        feature_grid = lss_normalized_coords_to_feature_grid(
            normalized_coords=normalized_coords,
            feature_hw=feature_hw,
            image_hw=image_hw,
        )
        x_indices = feature_grid[:, 0].round().long().clamp(0, max(feature_hw[1] - 1, 0))
        y_indices = feature_grid[:, 1].round().long().clamp(0, max(feature_hw[0] - 1, 0))
        point_labels = label_map[local_agent_ids, view_ids.long(), y_indices, x_indices]
        sampled_strength = None
        if torch.is_tensor(sampled_heatmap_features):
            sampled_strength = sampled_heatmap_features.abs().mean(dim=-1, keepdim=True)
        return estimate_local_patch_covariance(
            label_map=label_map,
            point_labels=point_labels,
            point_coords=normalized_coords,
            local_agent_ids=local_agent_ids,
            view_ids=view_ids.long(),
            x_indices=x_indices,
            y_indices=y_indices,
            image_hw=image_hw,
            patch_size=self.local_patch_size,
            major_scale=self.heatmap_major_scale,
            minor_scale=self.heatmap_minor_scale,
            eps=self.heatmap_center_eps,
            sampled_strength=sampled_strength,
        )

    def _build_lidar_candidate_entry(
        self,
        batch_dict: Dict,
        agent: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build one minimal lidar candidate entry without storing bulky intermediates."""
        agent_batch = self._get_agent_batch(batch_dict, agent)
        projection_dict = self.sample_projected_depth_features(batch_dict, agent)
        sampled_image_features = projection_dict["sampled_image_features"]
        assert sampled_image_features.ndim == 2, "sampled_image_features must have shape [num_hits, C]."
        if sampled_image_features.numel() == 0:
            return None

        depth_logits = self.depth_predictor.predict_agent_depth_logits(
            sampled_image_features, agent
        )
        depth_stats = self.depth_predictor.predict_depth_distribution(
            depth_logits, agent
        )
        anchor_depth = self.build_anchor_depth(projection_dict)
        support_covariance_2d = self._estimate_support_covariance_from_projection(
            agent_batch=agent_batch,
            projection_dict=projection_dict,
        )
        sigma_dict = self.gaussian_covariance_builder.build_view_covariance(
            support_covariance_2d=support_covariance_2d,
            depth_variance=depth_stats["depth_variance"],
            anchor_depth=anchor_depth,
            normalized_coords=projection_dict["normalized_coords"],
            hit_intrinsics=projection_dict["hit_intrinsics"],
            hit_extrinsics=projection_dict["hit_extrinsics"],
            hit_post_rots=projection_dict["hit_post_rots"],
            hit_post_trans=projection_dict["hit_post_trans"],
            image_shape_hw=projection_dict["image_shape_hw"],
        )

        voxel_features = agent_batch.get("voxel_features")
        voxel_ids = projection_dict["voxel_ids"]
        assert torch.is_tensor(voxel_features), "voxel_features is required."
        if voxel_ids.numel() == 0:
            return None
        query_features = voxel_features[voxel_ids.long()]
        return {
            "feature": query_features,
            "normalized_coords": projection_dict["normalized_coords"],
            "local_agent_ids": projection_dict["local_agent_ids"].long(),
            "view_ids": projection_dict["view_ids"].long(),
            "mean": projection_dict["hit_points_3d"],
            "sigma_3d": sigma_dict["sigma_3d"],
            "support_covariance_2d": support_covariance_2d,
            "group_ids": voxel_ids.long(),
            "source_is_image_only": torch.zeros(
                voxel_ids.shape[0],
                dtype=torch.bool,
                device=voxel_ids.device,
            ),
            "lidar_coverage_mask": projection_dict["lidar_coverage_mask"],
        }

    def build_view_observations(
        self,
        batch_dict: Dict,
        agent: str,
    ) -> Dict:
        """Build one minimal lidar candidate entry and persist only cross-file outputs."""
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        
        candidate_entry = self._build_lidar_candidate_entry(batch_dict, agent=agent)
        if candidate_entry is None:
            return batch_dict
        gp.setdefault("projection_masks", {})
        gp["projection_masks"][agent] = {
            "lidar_coverage_mask": candidate_entry.pop("lidar_coverage_mask"),
        }
        gp.setdefault("gaussian_candidates", {})
        gp["gaussian_candidates"].setdefault(agent, [])
        gp["gaussian_candidates"][agent].append(candidate_entry)
        return batch_dict

    def forward(
        self,
        batch_dict: Dict,
        available_agents: Optional[List[str]] = None,
        agent: Optional[str] = None,
    ) -> Dict:
        # import pdb; pdb.set_trace()
        available_agents = available_agents or batch_dict.get("available_agents")
        
        for current_agent in available_agents:
            batch_dict = self.build_view_observations(batch_dict, agent=current_agent)
        return batch_dict

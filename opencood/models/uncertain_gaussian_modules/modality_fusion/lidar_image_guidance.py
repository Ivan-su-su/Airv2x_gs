from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.uncertain_gaussian_modules.geometry.lidar_to_image_projector import LidarToImageProjector


class SpatialGuidanceAdapter(nn.Module):
    """Apply a lightweight spatial residual update using shared guidance."""

    def __init__(self, guidance_channels: int, strength: float = 1.0):
        super().__init__()
        hidden_channels = max(guidance_channels, 8)
        self.strength = float(strength)
        self.gate_net = nn.Sequential(
            nn.Conv2d(guidance_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.bias_net = nn.Sequential(
            nn.Conv2d(guidance_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feature_map: torch.Tensor, guidance_feature: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4:
            raise ValueError("SpatialGuidanceAdapter expects [B*V, C, H, W] inputs.")
        resized_guidance = F.interpolate(
            guidance_feature,
            size=feature_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gate = torch.sigmoid(self.gate_net(resized_guidance))
        bias = self.bias_net(resized_guidance)
        return feature_map * (1.0 + self.strength * gate) + self.strength * bias


class LidarImageGuidanceModule(nn.Module):
    """将 LiDAR 投影到图像平面并生成图像侧深度提示。"""

    def __init__(self, model_cfg=None):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.reuse_existing_features = bool(self.model_cfg.get("reuse_existing_features", True))
        self.guidance_map_channels = int(self.model_cfg.get("guidance_map_channels", 4))
        self.guidance_hidden_dim = int(self.model_cfg.get("guidance_hidden_dim", 16))
        self.image_feature_dim = int(self.model_cfg.get("image_feature_dim", 64))
        self.heatmap_feature_dim = int(self.model_cfg.get("heatmap_feature_dim", 7))
        # TODO: 肯定不能直接这样获得语义标签，为了先跑通代码，简单处理
        self.semantic_feature_dim = int(self.model_cfg.get("semantic_feature_dim", 1))
        self.enable_semantic_branch = bool(self.model_cfg.get("enable_semantic_branch", True))
        stem_hidden_dim = int(self.model_cfg.get("image_stem_hidden_dim", 32))
        self.image_feature_extractor = nn.Sequential(
            nn.Conv2d(4, stem_hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_hidden_dim, self.image_feature_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.image_feature_dim, self.image_feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(self.image_feature_dim, self.image_feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.image_feature_dim, self.heatmap_feature_dim, kernel_size=1),
        )
        self.semantic_head = nn.Sequential(
            nn.Conv2d(self.image_feature_dim, self.image_feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.image_feature_dim, self.semantic_feature_dim, kernel_size=1),
        )
        self.lidar_to_image_projector = LidarToImageProjector(
            self.model_cfg.get("lidar_to_image_projector", self.model_cfg)
        )
        self.guidance_encoder = nn.Sequential(
            nn.Conv2d(self.guidance_map_channels, self.guidance_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.guidance_hidden_dim, self.guidance_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.image_adapter = SpatialGuidanceAdapter(
            self.guidance_hidden_dim,
            strength=float(self.model_cfg.get("image_guidance_strength", 1.0)),
        )
        self.heatmap_adapter = SpatialGuidanceAdapter(
            self.guidance_hidden_dim,
            strength=float(self.model_cfg.get("heatmap_guidance_strength", 0.75)),
        )
        self.semantic_adapter = SpatialGuidanceAdapter(
            self.guidance_hidden_dim,
            strength=float(self.model_cfg.get("semantic_guidance_strength", 0.5)),
        )

    def _get_agent_names(
        self, batch_dict: Dict, available_agents: Optional[List[str]] = None
    ) -> List[str]:
        if available_agents:
            return list(available_agents)
        gp = batch_dict.get("gaussian_pipeline", {})
        if gp.get("available_agents"):
            return list(gp["available_agents"])
        return [
            agent_name
            for agent_name, agent_batch in batch_dict.items()
            if isinstance(agent_batch, dict)
            and (
                "voxel_coords" in agent_batch
                or "voxel_features" in agent_batch
                or "batch_merged_cam_inputs" in agent_batch
            )
        ]

    def _get_camera_imgs(self, agent_batch: Dict) -> Optional[torch.Tensor]:
        if "camera_imgs" in agent_batch:
            return agent_batch["camera_imgs"]
        cam_inputs = agent_batch.get("batch_merged_cam_inputs")
        if isinstance(cam_inputs, dict):
            return cam_inputs.get("imgs")
        return None

    def _build_base_image_features(self, agent_batch):
        camera_imgs = self._get_camera_imgs(agent_batch)
        
        has_existing_image = agent_batch.get("image_feature") is not None
        has_existing_heatmap = agent_batch.get("heatmap_feature") is not None
        has_existing_semantic = (agent_batch.get("semantic_feature") is not None) or (
            not self.enable_semantic_branch
        )
        if self.reuse_existing_features and has_existing_image and has_existing_heatmap and has_existing_semantic:
            return {
                "image_feature": agent_batch.get("image_feature"),
                "heatmap_feature": agent_batch.get("heatmap_feature"),
                "semantic_feature": agent_batch.get("semantic_feature"),
            }
        
        batch_size, num_views, channels, height, width = camera_imgs.shape
        flat_images = camera_imgs.reshape(batch_size * num_views, channels, height, width).float()
        image_feature = self.image_feature_extractor(flat_images)
        heatmap_feature = self.heatmap_head(image_feature)
        semantic_feature = self.semantic_head(image_feature) if self.enable_semantic_branch else None
        image_feature = image_feature.reshape(
            batch_size,
            num_views,
            image_feature.shape[1],
            image_feature.shape[2],
            image_feature.shape[3],
        )
        heatmap_feature = heatmap_feature.reshape(
            batch_size,
            num_views,
            heatmap_feature.shape[1],
            heatmap_feature.shape[2],
            heatmap_feature.shape[3],
        )
        if semantic_feature is not None:
            semantic_feature = semantic_feature.reshape(
                batch_size,
                num_views,
                semantic_feature.shape[1],
                semantic_feature.shape[2],
                semantic_feature.shape[3],
            )
        return {
            "image_feature": image_feature,
            "heatmap_feature": heatmap_feature,
            "semantic_feature": semantic_feature,
        }

    def _flatten_feature_map(
        self, feature_map: Optional[torch.Tensor], total_views: int
    ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[str, Tuple[int, ...]]]]:
        if feature_map is None:
            return None, None
        if feature_map.ndim == 5:
            batch_size, num_views = feature_map.shape[:2]
            return (
                feature_map.reshape(batch_size * num_views, *feature_map.shape[2:]),
                ("5d", tuple(feature_map.shape)),
            )
        if feature_map.ndim != 4:
            raise ValueError("Expected a 4D or 5D feature map in LidarImageGuidanceModule.")
        if feature_map.shape[0] == total_views:
            return feature_map, ("4d", tuple(feature_map.shape))
        raise ValueError("Cannot flatten feature map with the inferred camera-view count.")

    def _restore_feature_map(
        self, flat_feature_map: Optional[torch.Tensor], meta: Optional[Tuple[str, Tuple[int, ...]]]
    ) -> Optional[torch.Tensor]:
        if flat_feature_map is None or meta is None:
            return None
        layout, original_shape = meta
        if layout == "5d":
            batch_size, num_views, channels, height, width = original_shape
            return flat_feature_map.reshape(batch_size, num_views, channels, height, width)
        if layout == "4d":
            return flat_feature_map.reshape(*original_shape)
        raise ValueError(f"Unsupported feature layout metadata: {layout}")

    def _filter_agent_batch(self, agent_batch, voxel_mask):
        filtered_agent_batch = dict(agent_batch)
        for key in ("voxel_coords", "voxel_features", "ori_coords_height"):
            value = agent_batch.get(key)
            if torch.is_tensor(value) and value.shape[0] == voxel_mask.shape[0]:
                filtered_agent_batch[key] = value[voxel_mask]
        return filtered_agent_batch

    def _get_base_feature_shape(
        self, agent_batch: Dict, total_views: int
    ) -> Optional[Tuple[int, int, torch.device, torch.dtype]]:
        for key in ("image_feature", "heatmap_feature", "semantic_feature"):
            flat_feature_map, _ = self._flatten_feature_map(agent_batch.get(key), total_views)
            if flat_feature_map is not None:
                return (
                    int(flat_feature_map.shape[-2]),
                    int(flat_feature_map.shape[-1]),
                    flat_feature_map.device,
                    flat_feature_map.dtype,
                )
        return None

    def _build_anchor_depth(self, agent_batch: Dict, projection_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        hit_points_3d = projection_dict.get("hit_points_3d")
        view_ids = projection_dict.get("view_ids")
        if hit_points_3d is None or view_ids is None or hit_points_3d.numel() == 0:
            device = hit_points_3d.device if torch.is_tensor(hit_points_3d) else torch.device("cpu")
            return torch.empty((0, 1), device=device)
        cam_inputs = agent_batch.get("batch_merged_cam_inputs")
        extrinsics = cam_inputs.get("extrinsics") if isinstance(cam_inputs, dict) else None
        if not torch.is_tensor(extrinsics):
            return torch.norm(hit_points_3d, dim=-1, keepdim=True)
        flat_extrinsics = extrinsics.reshape(-1, *extrinsics.shape[-2:])
        hit_extrinsics = flat_extrinsics[view_ids.long()]
        if hit_extrinsics.shape[-2:] == (3, 4):
            extrinsics_4 = torch.eye(
                4, device=hit_extrinsics.device, dtype=hit_extrinsics.dtype
            ).unsqueeze(0).repeat(hit_extrinsics.shape[0], 1, 1)
            extrinsics_4[:, :3, :4] = hit_extrinsics
        elif hit_extrinsics.shape[-2:] == (4, 4):
            extrinsics_4 = hit_extrinsics
        else:
            return torch.norm(hit_points_3d, dim=-1, keepdim=True)
        lidar_to_camera = torch.inverse(extrinsics_4)
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

    def _build_relative_depth(
        self, anchor_depth: torch.Tensor, view_ids: torch.Tensor, total_views: int
    ) -> torch.Tensor:
        if anchor_depth.numel() == 0:
            return anchor_depth
        depth_sum = anchor_depth.new_zeros((total_views, 1))
        depth_count = anchor_depth.new_zeros((total_views, 1))
        depth_sum.index_add_(0, view_ids, anchor_depth)
        depth_count.index_add_(0, view_ids, torch.ones_like(anchor_depth))
        depth_mean = depth_sum / depth_count.clamp_min(1.0)
        return anchor_depth - depth_mean[view_ids]

    def _build_voxel_feature_scalar(self, projection_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        selected_voxel_features = projection_dict.get("selected_voxel_features")
        voxel_hit_indices = projection_dict.get("voxel_hit_indices")
        if not torch.is_tensor(voxel_hit_indices):
            # LidarToImageProjector 写入的是 voxel_ids，与每条投影命中一一对应
            voxel_hit_indices = projection_dict.get("voxel_ids")
        if (
            not torch.is_tensor(selected_voxel_features)
            or not torch.is_tensor(voxel_hit_indices)
            or voxel_hit_indices.numel() == 0
        ):
            device = voxel_hit_indices.device if torch.is_tensor(voxel_hit_indices) else torch.device("cpu")
            return torch.empty((0, 1), device=device)
        hit_voxel_features = selected_voxel_features[voxel_hit_indices.long()]
        if hit_voxel_features.ndim == 1:
            return hit_voxel_features.unsqueeze(-1)
        return hit_voxel_features.mean(dim=-1, keepdim=True)

    def _rasterize_guidance_maps(
        self,
        normalized_coords: torch.Tensor,
        view_ids: torch.Tensor,
        sparse_depth: torch.Tensor,
        relative_depth: torch.Tensor,
        voxel_feature_scalar: torch.Tensor,
        total_views: int,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        device = sparse_depth.device
        sparse_depth_map = torch.zeros((total_views, 1, height, width), device=device, dtype=dtype)
        relative_depth_map = torch.zeros((total_views, 1, height, width), device=device, dtype=dtype)
        hit_count_map = torch.zeros((total_views, 1, height, width), device=device, dtype=dtype)
        voxel_feature_map = torch.zeros((total_views, 1, height, width), device=device, dtype=dtype)
        if normalized_coords.numel() == 0:
            return {
                "sparse_depth_map": sparse_depth_map,
                "relative_depth_map": relative_depth_map,
                "hit_count_map": hit_count_map,
                "hit_mask": hit_count_map > 0,
                "voxel_feature_map": voxel_feature_map,
            }
        x_coords = (normalized_coords[:, 0] * (width - 1)).round().long().clamp_(0, width - 1)
        y_coords = (normalized_coords[:, 1] * (height - 1)).round().long().clamp_(0, height - 1)
        flat_indices = view_ids.long() * (height * width) + y_coords * width + x_coords
        flat_size = total_views * height * width

        def scatter_average(values: torch.Tensor) -> torch.Tensor:
            flat_sum = torch.zeros((flat_size, 1), device=device, dtype=dtype)
            flat_count = torch.zeros((flat_size, 1), device=device, dtype=dtype)
            flat_sum.index_add_(0, flat_indices, values.to(dtype))
            flat_count.index_add_(0, flat_indices, torch.ones_like(values, dtype=dtype))
            averaged = flat_sum / flat_count.clamp_min(1.0)
            return averaged.view(total_views, 1, height, width)

        sparse_depth_map = scatter_average(sparse_depth)
        relative_depth_map = scatter_average(relative_depth)
        voxel_feature_map = scatter_average(voxel_feature_scalar)
        flat_hits = torch.zeros((flat_size, 1), device=device, dtype=dtype)
        flat_hits.index_add_(0, flat_indices, torch.ones_like(sparse_depth, dtype=dtype))
        hit_count_map = flat_hits.view(total_views, 1, height, width)
        return {
            "sparse_depth_map": sparse_depth_map,
            "relative_depth_map": relative_depth_map,
            "hit_count_map": hit_count_map,
            "hit_mask": hit_count_map > 0,
            "voxel_feature_map": voxel_feature_map,
        }

    def _reshape_guidance_tensor(self, tensor: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        return tensor.reshape(batch_size, num_views, *tensor.shape[1:])

    def _build_guidance_for_agent(self, agent_batch, projection_dict):
        camera_imgs = self._get_camera_imgs(agent_batch)
        
        batch_size, num_views = camera_imgs.shape[:2]
        total_views = batch_size * num_views
        base_shape = self._get_base_feature_shape(agent_batch, total_views)
        
        height, width, device, dtype = base_shape

        normalized_coords = projection_dict["normalized_coords"]
        view_ids = projection_dict["view_ids"].long()
        anchor_depth = self._build_anchor_depth(agent_batch, projection_dict).to(device=device, dtype=dtype)
        relative_depth = self._build_relative_depth(anchor_depth, view_ids, total_views).to(
            device=device, dtype=dtype
        )
        voxel_feature_scalar = self._build_voxel_feature_scalar(projection_dict).to(
            device=device, dtype=dtype
        )
        rasterized_maps = self._rasterize_guidance_maps(
            normalized_coords=normalized_coords,
            view_ids=view_ids,
            sparse_depth=anchor_depth,
            relative_depth=relative_depth,
            voxel_feature_scalar=voxel_feature_scalar,
            total_views=total_views,
            height=height,
            width=width,
            dtype=dtype,
        )
        flat_guidance_input = torch.cat(
            [
                rasterized_maps["sparse_depth_map"],
                rasterized_maps["relative_depth_map"],
                rasterized_maps["hit_mask"].to(dtype),
                rasterized_maps["voxel_feature_map"],
            ],
            dim=1,
        )
        flat_guidance_feature = self.guidance_encoder(flat_guidance_input)
        return {
            "sparse_depth_map": self._reshape_guidance_tensor(
                rasterized_maps["sparse_depth_map"], batch_size, num_views
            ),
            "relative_depth_map": self._reshape_guidance_tensor(
                rasterized_maps["relative_depth_map"], batch_size, num_views
            ),
            "hit_count_map": self._reshape_guidance_tensor(
                rasterized_maps["hit_count_map"], batch_size, num_views
            ),
            "hit_mask": self._reshape_guidance_tensor(
                rasterized_maps["hit_mask"], batch_size, num_views
            ),
            "voxel_feature_map": self._reshape_guidance_tensor(
                rasterized_maps["voxel_feature_map"], batch_size, num_views
            ),
            "guidance_input": self._reshape_guidance_tensor(
                flat_guidance_input, batch_size, num_views
            ),
            "guidance_feature": self._reshape_guidance_tensor(
                flat_guidance_feature, batch_size, num_views
            ),
            "anchor_depth": anchor_depth,
            "relative_depth_values": relative_depth,
        }

    def _update_feature_branch(
        self,
        feature_map: Optional[torch.Tensor],
        total_views: int,
        flat_guidance_feature: torch.Tensor,
        adapter: SpatialGuidanceAdapter,
    ) -> Optional[torch.Tensor]:
        flat_feature_map, feature_meta = self._flatten_feature_map(feature_map, total_views)
        if flat_feature_map is None:
            return None
        updated_feature_map = adapter(flat_feature_map, flat_guidance_feature)
        return self._restore_feature_map(updated_feature_map, feature_meta)

    def project_lidar_to_image(self, batch_dict, available_agents):
        """Project each agent's masked voxel subset to its own image views."""
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        gp.setdefault("lidar_image_projection", {})
        for agent_name in available_agents:
            agent_batch = batch_dict.get(agent_name)
            
            voxel_mask = agent_batch.get("instance_voxel_mask").bool()
            if voxel_mask is None or voxel_mask.sum() == 0:
                raise ValueError(f"No valid voxel mask found for agent {agent_name}")
            
            filtered_agent_batch = self._filter_agent_batch(agent_batch, voxel_mask)
            projection_dict = self.lidar_to_image_projector(filtered_agent_batch)
            
            projection_dict["selected_voxel_mask"] = voxel_mask
            projection_dict["selected_voxel_indices"] = torch.nonzero(
                voxel_mask, as_tuple=False
            ).squeeze(-1)
            projection_dict["selected_voxel_features"] = filtered_agent_batch.get("voxel_features")
            projection_dict["selected_voxel_coords"] = filtered_agent_batch.get("voxel_coords")
            gp["lidar_image_projection"][agent_name] = projection_dict
        return batch_dict

    def build_image_feature_branches(self, batch_dict, available_agents):
        """Create the base image / heatmap / semantic features from raw camera images."""

        for agent_name in available_agents:
            agent_batch = batch_dict.get(agent_name)
            
            feature_dict = self._build_base_image_features(agent_batch)

            agent_batch["image_feature"] = feature_dict["image_feature"]
            agent_batch["heatmap_feature"] = feature_dict["heatmap_feature"]
            agent_batch["semantic_feature"] = feature_dict["semantic_feature"]
            
        return batch_dict

    def build_depth_guidance_features(self, batch_dict, available_agents):
        """Build shared 2D guidance maps and write guided output keys."""
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        
        lidar_image_projection = gp.get("lidar_image_projection", {})
        for agent_name in available_agents:
            agent_batch = batch_dict.get(agent_name)
            
            guidance_dict = self._build_guidance_for_agent(
                agent_batch,
                lidar_image_projection.get(agent_name),
            )
            
            if guidance_dict is None:
                agent_batch["guided_image_feature"] = agent_batch.get("image_feature")
                agent_batch["guided_heatmap_feature"] = agent_batch.get("heatmap_feature")
                agent_batch["guided_semantic_feature"] = agent_batch.get("semantic_feature")
                agent_batch["lss_ready_depth_feature"] = agent_batch.get("image_feature")
                continue
            batch_size, num_views = guidance_dict["guidance_feature"].shape[:2]
            total_views = batch_size * num_views
            flat_guidance_feature = guidance_dict["guidance_feature"].reshape(
                total_views,
                self.guidance_hidden_dim,
                guidance_dict["guidance_feature"].shape[-2],
                guidance_dict["guidance_feature"].shape[-1],
            )
            agent_batch["guided_image_feature"] = self._update_feature_branch(
                feature_map=agent_batch.get("image_feature"),
                total_views=total_views,
                flat_guidance_feature=flat_guidance_feature,
                adapter=self.image_adapter,
            )
            agent_batch["guided_heatmap_feature"] = self._update_feature_branch(
                feature_map=agent_batch.get("heatmap_feature"),
                total_views=total_views,
                flat_guidance_feature=flat_guidance_feature,
                adapter=self.heatmap_adapter,
            )
            agent_batch["guided_semantic_feature"] = self._update_feature_branch(
                feature_map=agent_batch.get("semantic_feature"),
                total_views=total_views,
                flat_guidance_feature=flat_guidance_feature,
                adapter=self.semantic_adapter,
            )
            agent_batch["lss_ready_depth_feature"] = agent_batch.get("guided_image_feature")
        return batch_dict

    def forward(self, batch_dict, available_agents=None):
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        gp["available_agents"] = available_agents
        batch_dict = self.build_image_feature_branches(batch_dict, available_agents)
        batch_dict = self.project_lidar_to_image(batch_dict, available_agents)
        batch_dict = self.build_depth_guidance_features(batch_dict, available_agents)
        return batch_dict

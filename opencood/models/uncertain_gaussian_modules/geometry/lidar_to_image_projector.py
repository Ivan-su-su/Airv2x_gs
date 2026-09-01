from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from opencood.models.uncertain_gaussian_modules.ops.win_coors.flattened_window_cuda import (
    map_points as map_points_cuda,
)


def get_points(pc_range, sample_num, space_shape, coords=None):
    """Generate reference points inside the requested voxels."""
    sx, sy, sz = space_shape
    x1, y1, z1, x2, y2, z2 = pc_range
    if coords is None:
        coord_x = torch.linspace(0, sx - 1, sx).view(1, -1, 1, 1).repeat(1, 1, sy, sz)
        coord_y = torch.linspace(0, sy - 1, sy).view(1, 1, -1, 1).repeat(1, sx, 1, sz)
        coord_z = torch.linspace(0, sz - 1, sz).view(1, 1, 1, -1).repeat(1, sx, sy, 1)
        coords = torch.stack((coord_x, coord_y, coord_z), -1).view(-1, 3)

    points = coords.clone().float()
    points[..., 0] = ((points[..., 0] + 0.5) / sx) * (x2 - x1) + x1
    points[..., 1] = ((points[..., 1] + 0.5) / sy) * (y2 - y1) + y1
    points[..., 2] = ((points[..., 2] + 0.5) / sz) * (z2 - z1) + z1

    if sample_num == 1:
        return points.unsqueeze(1)

    points = points.unsqueeze(1).repeat(1, sample_num, 1)
    points[..., 2] = torch.linspace(z1, z2, sample_num, device=points.device).unsqueeze(0)
    return points


class LidarToImageProjector(nn.Module):
    """Project voxel-aligned lidar points to all valid image views."""

    def __init__(self, model_cfg=None):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.pc_range = self.model_cfg.get(
            "point_cloud_range", [-54.0, -54.0, -10.0, 54.0, 54.0, 10.0]
        )
        self.voxel_size = self.model_cfg.get("voxel_size", [0.3, 0.3, 20.0])
        self.sample_num = int(self.model_cfg.get("sample_num", 1))
        self.space_shape = [
            int((self.pc_range[i + 3] - self.pc_range[i]) / self.voxel_size[i]) for i in range(3)
        ]

    def _get_camera_imgs(self, batch_dict: Dict) -> Optional[torch.Tensor]:
        cam_inputs = batch_dict.get("batch_merged_cam_inputs")
        assert isinstance(cam_inputs, dict), "batch_merged_cam_inputs is required."
        imgs = cam_inputs.get("imgs")
        assert torch.is_tensor(imgs), "batch_merged_cam_inputs['imgs'] is required."
        return imgs

    def _build_lidar2image_and_aug(
        self, agent_batch: Dict
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build per-agent lidar-to-image and image augmentation matrices."""
        cam_inputs = agent_batch.get("batch_merged_cam_inputs")
        assert isinstance(cam_inputs, dict), "batch_merged_cam_inputs is required."

        imgs = cam_inputs.get("imgs")
        intrinsics = cam_inputs.get("intrinsics")
        extrinsics = cam_inputs.get("extrinsics")
        post_rots = cam_inputs.get("post_rots")
        post_trans = cam_inputs.get("post_trans")
        assert torch.is_tensor(imgs), "batch_merged_cam_inputs['imgs'] is required."
        assert torch.is_tensor(intrinsics), "batch_merged_cam_inputs['intrinsics'] is required."
        assert torch.is_tensor(extrinsics), "batch_merged_cam_inputs['extrinsics'] is required."
        assert torch.is_tensor(post_rots), "batch_merged_cam_inputs['post_rots'] is required."
        assert torch.is_tensor(post_trans), "batch_merged_cam_inputs['post_trans'] is required."

        batch_size, num_views = imgs.shape[:2]
        device = imgs.device
        dtype = intrinsics.dtype

        if intrinsics.shape[-2:] == (3, 3):
            intrinsics_4 = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(
                batch_size, num_views, 1, 1
            )
            intrinsics_4[:, :, :3, :3] = intrinsics
        else:
            intrinsics_4 = intrinsics

        if extrinsics.shape[-2:] != (4, 4):
            extrinsics_4 = torch.eye(4, device=device, dtype=extrinsics.dtype).view(
                1, 1, 4, 4
            ).repeat(batch_size, num_views, 1, 1)
            extrinsics_4[:, :, :3, :3] = extrinsics[..., :3]
            extrinsics_4[:, :, :3, 3] = extrinsics[..., 3]
        else:
            extrinsics_4 = extrinsics
        extrinsics_4 = torch.inverse(extrinsics_4)

        aug_matrix = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(
            batch_size, num_views, 1, 1
        )
        aug_matrix[:, :, :3, :3] = post_rots
        aug_matrix[:, :, :3, 3] = post_trans

        lidar2image = torch.matmul(intrinsics_4, extrinsics_4)
        agent_to_ego = agent_batch.get("agent_to_ego_transform")
        if agent_to_ego is not None:
            ego_to_agent = torch.inverse(agent_to_ego)
            lidar2image = torch.matmul(lidar2image, ego_to_agent.unsqueeze(1))
        return lidar2image, aug_matrix

    def _get_reference_points(self, batch_dict: Dict) -> torch.Tensor:
        coords = batch_dict["voxel_coords"][:, [0, 3, 2, 1]].clone()
        if "ori_coords_height" in batch_dict:
            ori_coords_height = batch_dict["ori_coords_height"].reshape(-1, 1)
            coords = torch.cat([coords[:, :-1], ori_coords_height], dim=1)
        return get_points(self.pc_range, self.sample_num, self.space_shape, coords[:, 1:])

    def _project_single_agent(
        self,
        points: torch.Tensor,
        lidar2image: torch.Tensor,
        img_aug_matrix: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Project one local agent's selected voxels to all valid views."""
        _, points_2d, map_mask = map_points_cuda(
            points,
            lidar2image.float(),
            img_aug_matrix.float(),
            1,
            image_shape[0],
            image_shape[1],
            0,
        )
        points_2d = points_2d.squeeze(0)
        map_mask = map_mask.squeeze(0)
        hit_indices = torch.nonzero(map_mask, as_tuple=False)
        if hit_indices.numel() == 0:
            device = points.device
            empty_long = torch.empty((0,), device=device, dtype=torch.long)
            return {
                "normalized_coords": torch.empty((0, 2), device=device),
                "view_ids": empty_long,
                "local_voxel_ids": empty_long,
                "hit_points_3d": points.new_empty((0, 3)),
            }

        view_ids = hit_indices[:, 0].long()
        grid_ids = hit_indices[:, 1].long()
        sample_ids = hit_indices[:, 2].long()
        return {
            "normalized_coords": points_2d[view_ids, grid_ids, sample_ids].clamp(0.0, 1.0),
            "view_ids": view_ids,
            "local_voxel_ids": grid_ids.long(),
            "hit_points_3d": points[grid_ids, sample_ids],
        }

    def project(
        self,
        agent_batch: Dict,
        lidar_mask: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        camera_imgs = self._get_camera_imgs(agent_batch)
        lidar2image, img_aug_matrix = self._build_lidar2image_and_aug(agent_batch)
        voxel_coords = agent_batch.get("voxel_coords")
        assert torch.is_tensor(voxel_coords), "voxel_coords is required."

        points = self._get_reference_points(agent_batch).float()
        image_shape = camera_imgs.shape[-2:]
        batch_size, num_views = camera_imgs.shape[:2]
        num_voxels = voxel_coords.shape[0]
        if lidar_mask is not None and lidar_mask.shape != (batch_size, num_voxels):
            raise ValueError(
                "lidar_mask must have shape [B, num_voxels] aligned with agent-local cameras."
            )

        normalized_coords_list = []
        view_ids_list = []
        voxel_ids_list = []
        local_agent_ids_list = []
        hit_points_3d_list = []

        device = voxel_coords.device
        for local_agent_id in range(batch_size):
            if lidar_mask is None:
                selected_voxel_indices = torch.arange(
                    num_voxels, device=device, dtype=torch.long
                )
            else:
                selected_voxel_indices = torch.nonzero(
                    lidar_mask[local_agent_id].bool(), as_tuple=False
                ).squeeze(-1)
            if selected_voxel_indices.numel() == 0:
                continue

            projected = self._project_single_agent(
                points=points[selected_voxel_indices],
                lidar2image=lidar2image[local_agent_id : local_agent_id + 1],
                img_aug_matrix=img_aug_matrix[local_agent_id : local_agent_id + 1],
                image_shape=image_shape,
            )
            if projected["view_ids"].numel() == 0:
                continue

            hit_global_voxel_ids = selected_voxel_indices[projected["local_voxel_ids"]]
            normalized_coords_list.append(projected["normalized_coords"])
            view_ids_list.append(projected["view_ids"])
            voxel_ids_list.append(hit_global_voxel_ids.long())
            local_agent_ids_list.append(
                torch.full_like(projected["view_ids"], local_agent_id)
            )
            hit_points_3d_list.append(projected["hit_points_3d"])

        if len(normalized_coords_list) == 0:
            empty_long = torch.empty((0,), device=device, dtype=torch.long)
            return {
                "normalized_coords": torch.empty((0, 2), device=device),
                "view_ids": empty_long,
                "local_agent_ids": empty_long,
                "voxel_ids": empty_long,
                "hit_points_3d": points.new_empty((0, 3)),
            }

        return {
            "normalized_coords": torch.cat(normalized_coords_list, dim=0),
            "view_ids": torch.cat(view_ids_list, dim=0),
            "local_agent_ids": torch.cat(local_agent_ids_list, dim=0),
            "voxel_ids": torch.cat(voxel_ids_list, dim=0),
            "hit_points_3d": torch.cat(hit_points_3d_list, dim=0),
        }

    def forward(
        self,
        agent_batch: Dict,
        lidar_mask: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        return self.project(agent_batch, lidar_mask)

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class GaussianToImageProjector(nn.Module):
    """Project Gaussian key points to the current agent-local image views."""

    def __init__(self, model_cfg: Optional[Dict] = None) -> None:
        """Initialize the projector with the shared camera interface contract."""
        super().__init__()
        self.model_cfg = model_cfg or {}

    def _get_camera_imgs(self, agent_batch: Dict) -> torch.Tensor:
        """Read the current agent-local image tensor."""
        cam_inputs = agent_batch.get("batch_merged_cam_inputs")
        assert isinstance(cam_inputs, dict), "batch_merged_cam_inputs is required."
        imgs = cam_inputs.get("imgs")
        assert torch.is_tensor(imgs), "batch_merged_cam_inputs['imgs'] is required."
        return imgs

    def _build_projection_and_image_wh(
        self, agent_batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the current agent-local lidar-to-image matrices and image sizes."""
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
            intrinsics_4[..., :3, :3] = intrinsics
        elif intrinsics.shape[-2:] == (4, 4):
            intrinsics_4 = intrinsics
        else:
            raise ValueError("Unsupported intrinsics shape in GaussianToImageProjector.")

        if extrinsics.shape[-2:] != (4, 4):
            raise ValueError("GaussianToImageProjector expects 4x4 extrinsics.")

        if post_rots.shape[-2:] == (3, 3):
            post_rots_4 = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(
                batch_size, num_views, 1, 1
            )
            post_rots_4[..., :3, :3] = post_rots
        else:
            raise ValueError("GaussianToImageProjector expects 3x3 post_rots.")

        if post_trans.shape[-1] < 3:
            raise ValueError("GaussianToImageProjector expects 3D post_trans.")

        img_aug_matrix = post_rots_4.clone()
        img_aug_matrix[..., :3, 3] = post_trans[..., :3]
        lidar2image = torch.matmul(img_aug_matrix, torch.matmul(intrinsics_4, torch.inverse(extrinsics)))

        image_wh = torch.tensor(
            [imgs.shape[-1], imgs.shape[-2]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 2).repeat(batch_size, num_views, 1)
        return lidar2image, image_wh

    def forward(
        self,
        agent_batch: Dict,
        gaussian_key_points: torch.Tensor,
        local_agent_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Project Gaussian key points to valid `(gaussian, B, view)` image hits."""
        camera_imgs = self._get_camera_imgs(agent_batch)
        batch_size, num_views = camera_imgs.shape[:2]
        device = gaussian_key_points.device
        num_pts = int(gaussian_key_points.shape[1])
        if gaussian_key_points.numel() == 0:
            empty_long = torch.empty((0,), device=device, dtype=torch.long)
            empty_float = torch.empty((0, 2), device=device, dtype=gaussian_key_points.dtype)
            empty_points = torch.empty((0, num_pts, 2), device=device, dtype=gaussian_key_points.dtype)
            empty_mask = torch.empty((0, num_pts), device=device, dtype=torch.bool)
            return {
                "gaussian_ids": empty_long,
                "local_agent_ids": empty_long,
                "view_ids": empty_long,
                "normalized_coords": empty_float,
                "sampling_coords": empty_points,
                "sampling_valid_mask": empty_mask,
            }

        lidar2image, image_wh = self._build_projection_and_image_wh(agent_batch)
        valid_pairs = torch.nonzero(local_agent_mask.bool(), as_tuple=False)
        if valid_pairs.numel() == 0:
            empty_long = torch.empty((0,), device=device, dtype=torch.long)
            empty_float = torch.empty((0, 2), device=device, dtype=gaussian_key_points.dtype)
            empty_points = torch.empty((0, num_pts, 2), device=device, dtype=gaussian_key_points.dtype)
            empty_mask = torch.empty((0, num_pts), device=device, dtype=torch.bool)
            return {
                "gaussian_ids": empty_long,
                "local_agent_ids": empty_long,
                "view_ids": empty_long,
                "normalized_coords": empty_float,
                "sampling_coords": empty_points,
                "sampling_valid_mask": empty_mask,
            }

        gaussian_ids = valid_pairs[:, 0].long()
        local_agent_ids = valid_pairs[:, 1].long()
        assert int(local_agent_mask.shape[1]) == batch_size, "local_agent_mask second dim must equal B."

        selected_points = gaussian_key_points[gaussian_ids]
        points_h = torch.cat(
            [
                selected_points,
                torch.ones(
                    selected_points.shape[0],
                    selected_points.shape[1],
                    1,
                    device=device,
                    dtype=selected_points.dtype,
                ),
            ],
            dim=-1,
        )
        projection = lidar2image[local_agent_ids]
        projected = torch.matmul(
            projection[:, :, None, :, :],
            points_h[:, None, :, :, None],
        ).squeeze(-1)
        depth = projected[..., 2]
        coords = projected[..., :2] / torch.clamp(projected[..., 2:3], min=1e-5)
        normalized_coords = coords / image_wh[local_agent_ids, :, None, :]
        valid_mask = (
            (depth > 1e-5)
            & (normalized_coords[..., 0] > 0.0)
            & (normalized_coords[..., 0] < 1.0)
            & (normalized_coords[..., 1] > 0.0)
            & (normalized_coords[..., 1] < 1.0)
        )
        hit_mask = valid_mask.any(dim=-1)
        hit_indices = torch.nonzero(hit_mask, as_tuple=False)
        if hit_indices.numel() == 0:
            empty_long = torch.empty((0,), device=device, dtype=torch.long)
            empty_float = torch.empty((0, 2), device=device, dtype=gaussian_key_points.dtype)
            empty_points = torch.empty((0, num_pts, 2), device=device, dtype=gaussian_key_points.dtype)
            empty_mask = torch.empty((0, num_pts), device=device, dtype=torch.bool)
            return {
                "gaussian_ids": empty_long,
                "local_agent_ids": empty_long,
                "view_ids": empty_long,
                "normalized_coords": empty_float,
                "sampling_coords": empty_points,
                "sampling_valid_mask": empty_mask,
            }

        pair_ids = hit_indices[:, 0].long()
        view_ids = hit_indices[:, 1].long()
        hit_sampling_coords = normalized_coords[pair_ids, view_ids].clamp(0.0, 1.0)
        hit_sampling_valid_mask = valid_mask[pair_ids, view_ids]
        center_coords = hit_sampling_coords[:, 0]
        return {
            "gaussian_ids": gaussian_ids[pair_ids],
            "local_agent_ids": local_agent_ids[pair_ids],
            "view_ids": view_ids,
            "normalized_coords": center_coords,
            "sampling_coords": hit_sampling_coords,
            "sampling_valid_mask": hit_sampling_valid_mask,
        }

from typing import Dict

import torch
import torch.nn as nn


class GaussianCovarianceBuilder(nn.Module):
    """Build ray/tangent/3D covariance from view geometry."""

    def __init__(self, model_cfg=None):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.eps = float(self.model_cfg.get("covariance_eps", 1e-6))

    def _symmetrize(self, covariance: torch.Tensor) -> torch.Tensor:
        return 0.5 * (covariance + covariance.transpose(-1, -2))

    def build_ray_covariance(
        self,
        depth_variance: torch.Tensor,
        normalized_coords: torch.Tensor,
        hit_intrinsics: torch.Tensor,
        hit_extrinsics: torch.Tensor,
        hit_post_rots: torch.Tensor,
        hit_post_trans: torch.Tensor,
        image_shape_hw: torch.Tensor,
    ) -> torch.Tensor:
        """Lift z-depth variance with the explicit depth Jacobian."""
        if depth_variance.ndim == 1:
            depth_variance = depth_variance.unsqueeze(-1)
        depth_jacobian = self.build_depth_jacobian(
            normalized_coords=normalized_coords,
            hit_intrinsics=hit_intrinsics,
            hit_extrinsics=hit_extrinsics,
            hit_post_rots=hit_post_rots,
            hit_post_trans=hit_post_trans,
            image_shape_hw=image_shape_hw,
        )
        sigma_ray = depth_variance.unsqueeze(-1) * torch.matmul(
            depth_jacobian, depth_jacobian.transpose(-1, -2)
        )
        return self._symmetrize(sigma_ray)

    def _ensure_intrinsics_3x3(self, intrinsics: torch.Tensor) -> torch.Tensor:
        if intrinsics.shape[-2:] == (3, 3):
            return intrinsics
        if intrinsics.shape[-2:] == (4, 4):
            return intrinsics[..., :3, :3]
        raise ValueError("Unsupported intrinsics shape for covariance lifting.")

    def _ensure_extrinsics_4x4(self, extrinsics: torch.Tensor) -> torch.Tensor:
        if extrinsics.shape[-2:] == (4, 4):
            return extrinsics
        if extrinsics.shape[-2:] == (3, 4):
            extrinsics_4 = torch.eye(
                4, device=extrinsics.device, dtype=extrinsics.dtype
            ).view(1, 4, 4).repeat(extrinsics.shape[0], 1, 1)
            extrinsics_4[:, :3, :4] = extrinsics
            return extrinsics_4
        raise ValueError("Unsupported extrinsics shape for covariance lifting.")

    def _ensure_post_rot_2x2(self, post_rots: torch.Tensor) -> torch.Tensor:
        if post_rots.shape[-2:] == (2, 2):
            return post_rots
        if post_rots.shape[-2:] == (3, 3):
            return post_rots[..., :2, :2]
        raise ValueError("Unsupported post rotation shape for covariance lifting.")

    def _ensure_post_trans_2(self, post_trans: torch.Tensor) -> torch.Tensor:
        if post_trans.shape[-1] == 2:
            return post_trans
        if post_trans.shape[-1] >= 3:
            return post_trans[..., :2]
        raise ValueError("Unsupported post translation shape for covariance lifting.")

    def _pixel_jacobian_wrt_normalized(
        self, hit_post_rots: torch.Tensor, image_shape_hw: torch.Tensor
    ) -> torch.Tensor:
        """∂p_aug/∂(u,v) with p = post_rot^{-1} @ ((u*W, v*H) - t); translation drops out."""
        post_rot_2 = self._ensure_post_rot_2x2(hit_post_rots)
        post_rot_inv = torch.inverse(post_rot_2)
        height, width = image_shape_hw.unbind()
        n = hit_post_rots.shape[0]
        diag_wh = torch.zeros(n, 2, 2, device=post_rot_2.device, dtype=post_rot_2.dtype)
        diag_wh[:, 0, 0] = width
        diag_wh[:, 1, 1] = height
        return torch.matmul(post_rot_inv, diag_wh)

    def build_depth_jacobian(
        self,
        normalized_coords: torch.Tensor,
        hit_intrinsics: torch.Tensor,
        hit_extrinsics: torch.Tensor,
        hit_post_rots: torch.Tensor,
        hit_post_trans: torch.Tensor,
        image_shape_hw: torch.Tensor,
    ) -> torch.Tensor:
        """Build the Jacobian of lidar coordinates w.r.t. camera-frame z depth."""
        intrinsics_3 = self._ensure_intrinsics_3x3(hit_intrinsics)
        extrinsics_4 = self._ensure_extrinsics_4x4(hit_extrinsics)
        post_rot_2 = self._ensure_post_rot_2x2(hit_post_rots)
        post_trans_2 = self._ensure_post_trans_2(hit_post_trans)

        k_inv = torch.inverse(intrinsics_3)
        post_rot_inv = torch.inverse(post_rot_2)
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
        camera_rays = torch.matmul(k_inv, homogeneous_pixels.unsqueeze(-1))
        rotation_cam_to_lidar = extrinsics_4[:, :3, :3]
        return torch.matmul(rotation_cam_to_lidar, camera_rays)

    def build_backprojection_jacobian(
        self,
        hit_intrinsics: torch.Tensor,
        hit_extrinsics: torch.Tensor,
        hit_post_rots: torch.Tensor,
        hit_post_trans: torch.Tensor,
        anchor_depth: torch.Tensor,
        image_shape_hw: torch.Tensor,
    ) -> torch.Tensor:
        """Build ∂X_lidar/∂(u,v); aug matches build_depth_jacobian (post_rot + post_trans on pixel)."""
        if anchor_depth.ndim == 1:
            anchor_depth = anchor_depth.unsqueeze(-1)

        _ = self._ensure_post_trans_2(hit_post_trans)
        intrinsics_3 = self._ensure_intrinsics_3x3(hit_intrinsics)
        extrinsics_4 = self._ensure_extrinsics_4x4(hit_extrinsics)

        k_inv = torch.inverse(intrinsics_3)
        dp_duv = self._pixel_jacobian_wrt_normalized(hit_post_rots, image_shape_hw)

        jacobian_cam = anchor_depth.unsqueeze(-1) * torch.matmul(k_inv[:, :, :2], dp_duv)
        rotation_cam_to_lidar = extrinsics_4[:, :3, :3]
        return torch.matmul(rotation_cam_to_lidar, jacobian_cam)

    def lift_support_covariance_to_tangent_3d(
        self,
        support_covariance_2d: torch.Tensor,
        hit_intrinsics: torch.Tensor,
        hit_extrinsics: torch.Tensor,
        hit_post_rots: torch.Tensor,
        hit_post_trans: torch.Tensor,
        anchor_depth: torch.Tensor,
        image_shape_hw: torch.Tensor,
    ) -> torch.Tensor:
        """Lift image-plane support covariance with an explicit camera Jacobian."""
        jacobian = self.build_backprojection_jacobian(
            hit_intrinsics=hit_intrinsics,
            hit_extrinsics=hit_extrinsics,
            hit_post_rots=hit_post_rots,
            hit_post_trans=hit_post_trans,
            anchor_depth=anchor_depth.clamp_min(self.eps),
            image_shape_hw=image_shape_hw,
        )
        sigma_tan = torch.matmul(
            jacobian, torch.matmul(support_covariance_2d, jacobian.transpose(-1, -2))
        )
        return self._symmetrize(sigma_tan)

    def build_3d_covariance(
        self, sigma_tan: torch.Tensor, sigma_ray: torch.Tensor
    ) -> torch.Tensor:
        """Fuse tangent and ray covariance into one 3D covariance."""
        return self._symmetrize(sigma_tan + sigma_ray)

    def build_view_covariance(
        self,
        support_covariance_2d: torch.Tensor,
        depth_variance: torch.Tensor,
        anchor_depth: torch.Tensor,
        normalized_coords: torch.Tensor,
        hit_intrinsics: torch.Tensor,
        hit_extrinsics: torch.Tensor,
        hit_post_rots: torch.Tensor,
        hit_post_trans: torch.Tensor,
        image_shape_hw: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build the minimal per-view covariance tuple used by LSS."""
        sigma_ray = self.build_ray_covariance(
            depth_variance=depth_variance,
            normalized_coords=normalized_coords,
            hit_intrinsics=hit_intrinsics,
            hit_extrinsics=hit_extrinsics,
            hit_post_rots=hit_post_rots,
            hit_post_trans=hit_post_trans,
            image_shape_hw=image_shape_hw,
        )
        sigma_tan = self.lift_support_covariance_to_tangent_3d(
            support_covariance_2d=support_covariance_2d,
            hit_intrinsics=hit_intrinsics,
            hit_extrinsics=hit_extrinsics,
            hit_post_rots=hit_post_rots,
            hit_post_trans=hit_post_trans,
            anchor_depth=anchor_depth,
            image_shape_hw=image_shape_hw,
        )
        sigma_3d = self.build_3d_covariance(sigma_tan, sigma_ray)
        return {
            "sigma_ray": sigma_ray,
            "sigma_tan": sigma_tan,
            "sigma_3d": sigma_3d,
        }

"""LSS splat/pool using frozen Griffin-P1 image features.

Vehicle uses the trained categorical P1 depth distribution.
Drone uses dataset-provided analytic bottom-camera ideal_depth_z.
No Gaussian heatmap is consumed by this baseline path.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch
from torch import nn

from opencood.models.common_modules.airv2x_encoder import LiftSplatShootEncoder
from opencood.models.sub_modules.lss_submodule import BevEncode
from opencood.utils.camera_utils import depth_discretization, gen_dx_bx


class P1SplatEncoder(nn.Module):
    """Pool frozen P1 F90 into an agent-local BEV."""

    get_geometry = LiftSplatShootEncoder.get_geometry
    voxel_pooling = LiftSplatShootEncoder.voxel_pooling

    def __init__(
        self,
        args: Dict[str, Any],
        agent_type: str,
        encode_fn: Callable[[Dict[str, Any]], Dict[str, Dict[str, torch.Tensor]]],
    ) -> None:
        super().__init__()
        self.agent_type = str(agent_type)
        self._encode = encode_fn
        self.lift = str(args.get("lift", "categorical"))
        self.feature_sample_mode = str(
            args.get("feature_sample_mode", "lss_endpoint")
        )
        if self.feature_sample_mode not in ("lss_endpoint", "cell_center"):
            raise ValueError(
                "feature_sample_mode must be lss_endpoint/cell_center, got "
                f"{self.feature_sample_mode}"
            )

        self.grid_conf = args["grid_conf"]
        self.data_aug_conf = args["data_aug_conf"]
        self.bevout_feature = int(args["bevout_feature"])
        self.downsample = int(args["img_downsample"])
        self.camC = int(args["img_features"])

        dx, bx, nx = gen_dx_bx(
            self.grid_conf["xbound"],
            self.grid_conf["ybound"],
            self.grid_conf["zbound"],
        )
        self.register_buffer("dx", dx.clone().detach(), persistent=False)
        self.register_buffer("bx", bx.clone().detach(), persistent=False)
        self.register_buffer("nx", nx.clone().detach(), persistent=False)

        self.register_buffer(
            "frustum", self.create_frustum(), persistent=False
        )
        self.D = int(self.frustum.shape[0])
        self.bevencode = BevEncode(inC=self.camC, outC=self.bevout_feature)
        self.use_quickcumsum = True

    def create_frustum(self) -> torch.Tensor:
        final_h, final_w = [
            int(x) for x in self.data_aug_conf["final_dim"]
        ]
        feat_h = final_h // self.downsample
        feat_w = final_w // self.downsample

        if self.lift == "ideal_ground":
            depth_values = torch.tensor([1.0], dtype=torch.float32)
        else:
            depth_values = torch.tensor(
                depth_discretization(
                    *self.grid_conf["ddiscr"],
                    self.grid_conf["mode"],
                ),
                dtype=torch.float32,
            )

        if self.feature_sample_mode == "cell_center":
            xs = (torch.arange(feat_w, dtype=torch.float32) + 0.5) * (
                float(final_w) / float(feat_w)
            )
            ys = (torch.arange(feat_h, dtype=torch.float32) + 0.5) * (
                float(final_h) / float(feat_h)
            )
        else:
            xs = torch.linspace(0, final_w - 1, feat_w, dtype=torch.float32)
            ys = torch.linspace(0, final_h - 1, feat_h, dtype=torch.float32)

        ds = depth_values.view(-1, 1, 1).expand(-1, feat_h, feat_w)
        xs = xs.view(1, 1, feat_w).expand(len(depth_values), feat_h, feat_w)
        ys = ys.view(1, feat_h, 1).expand(len(depth_values), feat_h, feat_w)
        return torch.stack((xs, ys, ds), dim=-1)

    def get_geometry_from_z(
        self,
        rots: torch.Tensor,
        trans: torch.Tensor,
        intrins: torch.Tensor,
        post_rots: torch.Tensor,
        post_trans: torch.Tensor,
        z_map: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_cam, _ = trans.shape
        feat_h, feat_w = int(self.frustum.shape[1]), int(self.frustum.shape[2])
        xy = self.frustum[..., :2].to(
            device=z_map.device, dtype=z_map.dtype
        ).view(1, 1, 1, feat_h, feat_w, 2)
        xyz = torch.cat(
            [
                xy.expand(batch_size, num_cam, 1, feat_h, feat_w, 2),
                z_map[:, :, None, :, :, None],
            ],
            dim=-1,
        )
        points = xyz - post_trans.view(
            batch_size, num_cam, 1, 1, 1, 3
        )
        points = (
            torch.inverse(post_rots)
            .view(batch_size, num_cam, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        points = torch.cat(
            (
                points[..., :2] * points[..., 2:3],
                points[..., 2:3],
            ),
            dim=5,
        )
        combine = rots.matmul(torch.inverse(intrins))
        points = (
            combine.view(batch_size, num_cam, 1, 1, 1, 3, 3)
            .matmul(points)
            .squeeze(-1)
        )
        points += trans.view(batch_size, num_cam, 1, 1, 1, 3)
        return points

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pack = self._encode(data_dict)[self.agent_type]
        cam = data_dict[self.agent_type]["batch_merged_cam_inputs"]
        imgs = cam["imgs"]
        if imgs.dim() != 5:
            raise ValueError(
                f"{self.agent_type} imgs must be [B,N,C,H,W], got {tuple(imgs.shape)}"
            )

        batch_size, num_cam = int(imgs.shape[0]), int(imgs.shape[1])
        f90 = pack["f90"]
        feat_c, feat_h, feat_w = (
            int(f90.shape[1]),
            int(f90.shape[2]),
            int(f90.shape[3]),
        )
        f90 = f90.view(batch_size, num_cam, feat_c, feat_h, feat_w)
        feat = f90.permute(0, 1, 3, 4, 2).unsqueeze(2)

        if self.lift == "ideal_ground":
            z_map = data_dict[self.agent_type].get("ideal_depth_z")
            if not torch.is_tensor(z_map):
                raise KeyError(
                    f"{self.agent_type} ideal_ground lift requires ideal_depth_z"
                )
            if z_map.dim() == 3:
                z_map = z_map.unsqueeze(0)
            if tuple(z_map.shape) != (
                batch_size,
                num_cam,
                feat_h,
                feat_w,
            ):
                raise ValueError(
                    f"{self.agent_type} ideal_depth_z {tuple(z_map.shape)} "
                    f"expected {(batch_size, num_cam, feat_h, feat_w)}"
                )
            z_map = z_map.to(device=f90.device, dtype=f90.dtype)
            valid_depth = torch.isfinite(z_map) & (z_map > 0.0)
            # Invalid bottom-camera rays must contribute neither geometry nor
            # features. A large finite z keeps the inherited integer voxel
            # conversion deterministic and guarantees range rejection.
            z_safe = torch.where(
                valid_depth, z_map, torch.full_like(z_map, 1.0e6)
            )
            geom = self.get_geometry_from_z(
                cam["rots"],
                cam["trans"],
                cam["intrinsics"],
                cam["post_rots"],
                cam["post_trans"],
                z_safe,
            )
            x_img = feat * valid_depth[:, :, None, :, :, None].to(
                dtype=feat.dtype
            )
        else:
            depth = pack.get("depth")
            if not torch.is_tensor(depth):
                raise KeyError(
                    f"{self.agent_type} categorical lift requires P1 depth"
                )
            depth = depth.view(
                batch_size, num_cam, -1, feat_h, feat_w
            )
            x_img = depth.unsqueeze(-1) * feat
            geom = self.get_geometry(
                cam["rots"],
                cam["trans"],
                cam["intrinsics"],
                cam["post_rots"],
                cam["post_trans"],
            )

        voxels = self.voxel_pooling(geom, x_img)
        bev = self.bevencode(voxels)
        return {
            "spatial_features": bev,
            "spatial_features_3d": bev.unsqueeze(2),
        }

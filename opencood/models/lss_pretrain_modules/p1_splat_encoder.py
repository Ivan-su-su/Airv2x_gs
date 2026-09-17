"""LSS splat/pool/BEV using frozen P1 features. Does not build CamEncode.

The P1 path (matching Airv2xGaussian0822) is::

    CamEncode (RGB only) -> R2 / F45 -> (v45: r2_downsample) -> HighResFusion
        -> F90 [N,128,fH,fW]
        -> veh/rsu: DepthHead  -> categorical depth probs [N,D,fH,fW]
        -> drone:   HeightEmbed + DeltaHead -> per-pixel z [N,fH,fW]

voxel_pooling then consumes F90 features weighted by the *trained* P1 depth
instead of the never-trained official ``camencode.depth_head``. ``bevencode``
stays a normal trainable/random module.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch
from torch import nn

from opencood.models.common_modules.airv2x_encoder import LiftSplatShootEncoder
from opencood.models.sub_modules.lss_submodule import BevEncode
from opencood.utils.camera_utils import gen_dx_bx


class P1SplatEncoder(nn.Module):
    """Voxel-pool P1 F90 with categorical depth (veh/rsu) or delta-z (drone)."""

    # Reuse the official geometry / pooling math verbatim.
    create_frustum = LiftSplatShootEncoder.create_frustum
    get_geometry = LiftSplatShootEncoder.get_geometry
    voxel_pooling = LiftSplatShootEncoder.voxel_pooling

    def __init__(
        self,
        args: Dict[str, Any],
        agent_type: str,
        encode_fn: Callable[[Dict[str, Any]], Dict[str, Dict[str, torch.Tensor]]],
    ) -> None:
        super().__init__()
        self.agent_type = agent_type
        self._encode = encode_fn
        self.lift = str(args.get("lift", "categorical"))
        self.grid_conf = args["grid_conf"]
        self.data_aug_conf = args["data_aug_conf"]
        self.bevout_feature = args["bevout_feature"]
        self.downsample = int(args["img_downsample"])
        self.camC = int(args["img_features"])
        dx, bx, nx = gen_dx_bx(
            self.grid_conf["xbound"],
            self.grid_conf["ybound"],
            self.grid_conf["zbound"],
        )
        device = torch.device("cuda")
        self.dx = dx.clone().detach().to(device)
        self.bx = bx.clone().detach().to(device)
        self.nx = nx.clone().detach().to(device)
        if self.lift == "delta":
            # delta lift: frustum xy only, real z comes from the z-map.
            saved = self.grid_conf
            self.grid_conf = {**saved, "ddiscr": [0.0, 1.0, 1], "mode": "UD"}
            frustum = self.create_frustum()
            self.grid_conf = saved
        else:
            frustum = self.create_frustum()
        self.frustum = frustum.clone().detach().to(device)
        self.D = int(self.frustum.shape[0])
        self.bevencode = BevEncode(inC=self.camC, outC=self.bevout_feature)
        self.use_quickcumsum = True

    def get_geometry_from_z(
        self,
        rots: torch.Tensor,
        trans: torch.Tensor,
        intrins: torch.Tensor,
        post_rots: torch.Tensor,
        post_trans: torch.Tensor,
        z_map: torch.Tensor,
    ) -> torch.Tensor:
        """Same unproject as ``get_geometry``, with per-pixel optical z."""
        batch_size, num_cam, _ = trans.shape
        feat_h, feat_w = int(self.frustum.shape[1]), int(self.frustum.shape[2])
        xy = self.frustum[..., :2].view(1, 1, 1, feat_h, feat_w, 2)
        xyz = torch.cat(
            [
                xy.expand(batch_size, num_cam, 1, feat_h, feat_w, 2),
                z_map[:, :, None, :, :, None],
            ],
            dim=-1,
        )
        points = xyz - post_trans.view(batch_size, num_cam, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(batch_size, num_cam, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(batch_size, num_cam, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(batch_size, num_cam, 1, 1, 1, 3)
        return points

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pack = self._encode(data_dict)[self.agent_type]
        cam = data_dict[self.agent_type]["batch_merged_cam_inputs"]
        imgs = cam["imgs"]
        batch_size, num_cam = int(imgs.shape[0]), int(imgs.shape[1])
        f90 = pack["f90"]
        feat_c, feat_h, feat_w = int(f90.shape[1]), int(f90.shape[2]), int(f90.shape[3])
        f90 = f90.view(batch_size, num_cam, feat_c, feat_h, feat_w)
        feat = f90.permute(0, 1, 3, 4, 2).unsqueeze(2)
        if self.lift == "delta":
            z_map = pack["z"].view(batch_size, num_cam, feat_h, feat_w)
            geom = self.get_geometry_from_z(
                cam["rots"],
                cam["trans"],
                cam["intrinsics"],
                cam["post_rots"],
                cam["post_trans"],
                z_map,
            )
            x_img = feat
        else:
            depth = pack["depth"].view(batch_size, num_cam, -1, feat_h, feat_w)
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
        return {"spatial_features": bev, "spatial_features_3d": bev.unsqueeze(2)}

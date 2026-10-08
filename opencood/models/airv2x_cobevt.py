from typing import Any, Dict

import torch
import torch.nn as nn
from einops import rearrange, repeat

from opencood.models.cobevt_modules.fuse_utils import regroup
from opencood.models.common_modules.torch_transformation_utils import (
    warp_affine_simple,
)
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.cobevt_modules.swap_fusion_modules import SwapFusionEncoder
from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.common_modules.airv2x_base_model import Airv2xBase
from opencood.models.common_modules.airv2x_encoder import LiftSplatShootEncoder
from opencood.models.lss_pretrain_modules.p1_mixin import P1CamMixin
from opencood.models.task_heads.segmentation_head import BevSegHead


class Airv2xCoBEVT(P1CamMixin, Airv2xBase):
    def __init__(self, args):
        super().__init__(args)

        self.args = args

        self.collaborators = args["collaborators"]
        self.active_sensors = args["active_sensors"]
        max_cav = args["max_cav"]
        self.max_cav_num = sum(max_cav.values())

        # if "vehicle" in self.collaborators:
        #     self.veh_model = LiftSplatShootEncoder(args, agent_type="vehicle")

        # if "rsu" in self.collaborators:
        #     self.rsu_model = LiftSplatShootEncoder(args, agent_type="rsu")

        # if "drone" in self.collaborators:
        #     self.drone_model = LiftSplatShootEncoder(args, agent_type="drone")
        
        if args.get("p1_checkpoint"):
            # Frozen Griffin-P1 LSS frontend (same recipe as where2comm /
            # v2xvit Griffin baselines): camera-only collaborators.
            P1CamMixin.init_encoders(self, args)
        else:
            Airv2xBase.init_encoders(self, args)

        self.backbone = BaseBEVBackbone(args["base_bev_backbone"], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if "shrink_header" in args and args["shrink_header"]["use"]:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args["shrink_header"])

        self.compression = False
        if args["compression"] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args["compression"])

        args['fax_fusion']['agent_size'] = self.max_cav_num
        self.fusion_net = SwapFusionEncoder(args["fax_fusion"])

        self.outC = args["outC"]
        if args["task"] == "det":
            self.cls_head = nn.Conv2d(
                self.outC, args["anchor_number"] * args["num_class"], kernel_size=1
            )
            self.reg_head = nn.Conv2d(
                self.outC, 7 * args["anchor_number"], kernel_size=1
            )
            if args["obj_head"]:
                self.obj_head = nn.Conv2d(
                    self.outC, args["anchor_number"], kernel_size=1
                )

        elif args["task"] == "seg":
            self.seg_head = BevSegHead(
                args["seg_branch"], args["seg_hw"], args["seg_hw"], 256, args["dynamic_class"], args["static_class"],
                seg_res=args["seg_res"], cav_range=args["cav_range"]
            )

        if args["backbone_fix"]:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        if "vehicle" in self.collaborators:
            for p in self.veh_model.parameters():
                p.requires_grad = False
        if "rsu" in self.collaborators:
            for p in self.rsu_model.parameters():
                p.requires_grad = False
        if "drone" in self.collaborators:
            for p in self.drone_model.parameters():
                p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        if self.args["task"] == "det":
            for p in self.cls_head.parameters():
                p.requires_grad = False
            for p in self.reg_head.parameters():
                p.requires_grad = False

            if self.args["obj_head"]:
                for p in self.obj_head.parameters():
                    p.requires_grad = False
        elif self.args["task"] == "seg":
            for p in self.seg_head.parameters():
                p.requires_grad = False

    def _align_agents_to_ego(
        self,
        regroup_feature: torch.Tensor,
        pairwise: torch.Tensor,
    ) -> torch.Tensor:
        """Warp each agent BEV onto the ego grid before swap fusion.

        Camera LSS features are agent-local. ``pairwise[:, 0, i]`` is
        ego → agent ``i``, which is the sampling map ``affine_grid`` needs.
        Translation is converted from meters into the normalized frame of
        this feature map.

        Args:
            regroup_feature: Agent features, shape ``[B, L, C, H, W]``.
            pairwise: Rigid transforms, shape ``[B, L, L, 4, 4]``.

        Returns:
            Features in the ego frame, same shape as ``regroup_feature``.
        """
        if pairwise.dim() == 4:
            pairwise = pairwise.unsqueeze(0)
        batch, agents, channels, height, width = regroup_feature.shape
        cav_range = self.args["cav_range"]
        x_extent = float(cav_range[3] - cav_range[0])
        y_extent = float(cav_range[4] - cav_range[1])
        if abs(x_extent / width - y_extent / height) > 1e-3:
            raise ValueError(
                "CoBEVT ego alignment expects square meters-per-pixel, got "
                f"x {x_extent}/{width} vs y {y_extent}/{height}"
            )
        discrete_ratio = x_extent / float(width)
        affine = normalize_pairwise_tfm(
            pairwise, height, width, discrete_ratio, downsample_rate=1
        )
        theta = affine[:, 0].reshape(batch * agents, 2, 3).to(
            device=regroup_feature.device, dtype=regroup_feature.dtype
        )
        warped = warp_affine_simple(
            regroup_feature.reshape(batch * agents, channels, height, width),
            theta,
            (height, width),
        )
        return warped.reshape(batch, agents, channels, height, width)

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        batch_output_dict, batch_record_len = self.extract_features(data_dict)

        batch_dict_output = self.backbone(batch_output_dict)

        spatial_features_2d = batch_dict_output["spatial_features_2d"]
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # N, C, H, W -> B,  L, C, H, W
        # TODO(YH): bug here
        regroup_feature, mask = regroup(
            spatial_features_2d, batch_record_len, self.max_cav_num
        )
        regroup_feature = self._align_agents_to_ego(
            regroup_feature, data_dict["img_pairwise_t_matrix_collab"]
        )

        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        com_mask = repeat(
            com_mask,
            "b h w c l -> b (h new_h) (w new_w) c l",
            new_h=regroup_feature.shape[3],
            new_w=regroup_feature.shape[4],
        )

        fused_feature = self.fusion_net(regroup_feature, com_mask)

        output_dict = {}
        if self.args["task"] == "det":
            psm = self.cls_head(fused_feature)
            rm = self.reg_head(fused_feature)

            output_dict = {"psm": psm, "rm": rm}
            if self.args["obj_head"]:
                obj = self.obj_head(fused_feature)
                output_dict.update({"obj": obj})
        elif self.args["task"] == "seg":
            # seg_logits = self.seg_head(fused_feature)
            # output_dict = {"seg_logits": seg_logits}
            seg_output_dict = self.seg_head(fused_feature)
            output_dict.update(seg_output_dict)

        return output_dict

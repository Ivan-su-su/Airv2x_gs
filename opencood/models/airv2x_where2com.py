# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# Modified by: Yuheng Wu <yuhengwu@kaist.ac.kr>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn

from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.common_modules.airv2x_encoder import LiftSplatShootEncoder
from opencood.models.common_modules.airv2x_base_model import Airv2xBase
from opencood.models.lss_pretrain_modules.p1_mixin import P1CamMixin
from opencood.models.where2comm_modules.where2comm_fuse import Where2comm
from opencood.models.task_heads.segmentation_head import BevSegHead 


class Airv2xWhere2com(P1CamMixin, Airv2xBase):
    def __init__(self, args):
        super().__init__(args)

        self.args = args

        # here we use image encoder LSS instead of lidar
        self.collaborators = args["collaborators"]
        self.active_sensors = args["active_sensors"]

        # if "vehicle" in self.collaborators:
        #     self.veh_model = LiftSplatShootEncoder(args, agent_type="vehicle")

        # if "rsu" in self.collaborators:
        #     self.rsu_model = LiftSplatShootEncoder(args, agent_type="rsu")

        # if "drone" in self.collaborators:
        #     self.drone_model = LiftSplatShootEncoder(args, agent_type="drone")
        
        if args.get("p1_checkpoint"):
            P1CamMixin.init_encoders(self, args)
        else:
            Airv2xBase.init_encoders(self, args)

        modality_args = args["modality_fusion"]
        self.backbone = BaseBEVBackbone(modality_args["base_bev_backbone"], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if "shrink_header" in modality_args and modality_args["shrink_header"]["use"]:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(modality_args["shrink_header"])
        self.compression = False

        if modality_args["compression"] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args["compression"])

        self.fusion_net = Where2comm(args["where2com_fusion"])
        self.multi_scale = args["where2com_fusion"]["multi_scale"]

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
                args["seg_branch"], args["seg_hw"], args["seg_hw"], self.outC, args["dynamic_class"], args["static_class"],
                seg_res=args["seg_res"], cav_range=args["cav_range"]
            )

        if args["backbone_fix"]:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        if "vehicle" in self.collaborators:
            for p in self.veh_models.parameters():
                p.requires_grad = False
        if "rsu" in self.collaborators:
            for p in self.rsu_models.parameters():
                p.requires_grad = False
        if "drone" in self.collaborators:
            for p in self.drone_models.parameters():
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

    def forward(self, data_dict):
        batch_output_dict, batch_record_len = self.extract_features(data_dict)
        batch_spatial_features = batch_output_dict["spatial_features"]
        comm_rates = batch_spatial_features.count_nonzero().item()

        # One explicit backbone pass for the single-agent confidence map.
        batch_dict = self.backbone(batch_output_dict)
        batch_spatial_features_2d = batch_dict["spatial_features_2d"]
        pairwise_t_matrix = data_dict["img_pairwise_t_matrix_collab"]

        if self.shrink_flag:
            batch_spatial_features_2d = self.shrink_conv(
                batch_spatial_features_2d
            )

        output_dict = {}
        if self.args["task"] == "det":
            psm_single = self.cls_head(batch_spatial_features_2d)

            if self.compression:
                batch_spatial_features_2d = self.naive_compressor(
                    batch_spatial_features_2d
                )

            if self.multi_scale:
                fused_feature, communication_rates = self.fusion_net(
                    batch_dict["spatial_features"],
                    psm_single,
                    batch_record_len,
                    pairwise_t_matrix,
                    self.backbone,
                )
                if self.shrink_flag:
                    fused_feature = self.shrink_conv(fused_feature)
            else:
                fused_feature, communication_rates = self.fusion_net(
                    batch_spatial_features_2d,
                    psm_single,
                    batch_record_len,
                    pairwise_t_matrix,
                )

            output_dict["psm"] = self.cls_head(fused_feature)
            output_dict["rm"] = self.reg_head(fused_feature)
            if self.args["obj_head"]:
                output_dict["obj"] = self.obj_head(fused_feature)
            output_dict.update(
                {
                    "mask": 0,
                    "com": communication_rates,
                    "comm_rate": comm_rates,
                }
            )
            return output_dict

        if self.args["task"] == "seg":
            _, ori_x = self.seg_head(batch_spatial_features_2d, True)
            if self.compression:
                batch_spatial_features_2d = self.naive_compressor(
                    batch_spatial_features_2d
                )
            if self.multi_scale:
                fused_feature, communication_rates = self.fusion_net(
                    batch_dict["spatial_features"],
                    ori_x,
                    batch_record_len,
                    pairwise_t_matrix,
                    self.backbone,
                )
                if self.shrink_flag:
                    fused_feature = self.shrink_conv(fused_feature)
            else:
                fused_feature, communication_rates = self.fusion_net(
                    batch_spatial_features_2d,
                    ori_x,
                    batch_record_len,
                    pairwise_t_matrix,
                )
            output_dict.update(self.seg_head(fused_feature))
            output_dict.update(
                {
                    "mask": 0,
                    "com": communication_rates,
                    "comm_rate": comm_rates,
                }
            )
            return output_dict

        raise ValueError(f"Unsupported task: {self.args['task']}")


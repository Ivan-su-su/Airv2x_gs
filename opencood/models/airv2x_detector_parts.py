# -*- coding: utf-8 -*-
"""Shared camera detector parts for homogeneous / HEAL / STAMP / NegoCollab."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from opencood.models.common_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.task_heads.segmentation_head import BevSegHead
from opencood.utils.negocollab_transform import to_airv2x_pyramid_affine


class Airv2xSharedDetector(nn.Module):
    """ResNet BEV backbone + PyramidFusion + optional shrink + det heads.

    This is the Stage-0 homogeneous detector body reused as:
    * the trainable body of `Airv2xHomoBase`
    * the shared vehicle-domain detector of AirV2X HEAL / STAMP
    * each local branch of NegoCollab
    """

    def __init__(self, args: Dict[str, Any], backbone_in_channels: int = 64) -> None:
        super().__init__()
        self.args = args
        modality_args = args["modality_fusion"]
        self.backbone = ResNetBEVBackbone(
            modality_args["base_bev_backbone"], backbone_in_channels
        )
        self.shrink_flag = bool(
            "shrink_header" in modality_args and modality_args["shrink_header"].get("use")
        )
        if self.shrink_flag:
            self.shrink_conv = DownsampleConv(modality_args["shrink_header"])
        else:
            self.shrink_conv = None
        self.pyramid_backbone = PyramidFusion(args["fusion_backbone"])
        self.task = args.get("task", "det")
        self.use_obj_head = bool(args.get("obj_head", False))
        if self.task == "det":
            in_head = args["in_head"]
            self.cls_head = nn.Conv2d(
                in_head, args["anchor_number"] * args["num_class"], kernel_size=1
            )
            self.reg_head = nn.Conv2d(
                in_head, 7 * args["anchor_number"], kernel_size=1
            )
            if self.use_obj_head:
                self.obj_head = nn.Conv2d(
                    in_head, args["anchor_number"], kernel_size=1
                )
            else:
                self.obj_head = None
            self.seg_head = None
        elif self.task == "seg":
            self.seg_head = BevSegHead(
                args["seg_branch"],
                args["seg_hw"],
                args["seg_hw"],
                args["in_head"],
                args["dynamic_class"],
                args["static_class"],
                seg_res=args["seg_res"],
                cav_range=args["cav_range"],
            )
            self.cls_head = None
            self.reg_head = None
            self.obj_head = None
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def encode_bev(self, spatial_features):
        """Run the BEV backbone on LSS / pillar `spatial_features`."""
        return self.backbone({"spatial_features": spatial_features})["spatial_features_2d"]

    def fuse_and_decode(self, spatial_features_2d, record_len, pairwise_t_matrix):
        """Pyramid collab fusion + optional shrink.

        `pairwise_t_matrix` may be `[B, L, L, 4, 4]` or `[B, L, L, 2, 3]`.
        """
        affine = to_airv2x_pyramid_affine(pairwise_t_matrix)
        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
            spatial_features_2d, record_len, affine
        )
        if self.shrink_flag and self.shrink_conv is not None:
            fused_feature = self.shrink_conv(fused_feature)
        return fused_feature, occ_outputs

    def decode_heads(self, fused_feature) -> Dict[str, Any]:
        """Apply detection / segmentation heads."""
        output: Dict[str, Any] = {}
        if self.task == "det":
            output["psm"] = self.cls_head(fused_feature)
            output["rm"] = self.reg_head(fused_feature)
            if self.use_obj_head and self.obj_head is not None:
                output["obj"] = self.obj_head(fused_feature)
        else:
            output.update(self.seg_head(fused_feature))
        return output

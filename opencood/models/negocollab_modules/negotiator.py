# -*- coding: utf-8 -*-
"""Official NegoCollab Negotiator: pyramid occupancy-weighted mean of modalities."""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import Tensor, nn

from opencood.models.common_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.resblock import Bottleneck, ResNetModified


class Negotiator(ResNetBEVBackbone):
    """Build a common representation from a list of modality BEV features."""

    def __init__(self, model_cfg, input_channels: int = 64) -> None:
        backbone_cfg = model_cfg["backbone_args"]
        super().__init__(backbone_cfg, input_channels)
        if backbone_cfg.get("resnext"):
            Bottleneck.expansion = 1
            self.resnet = ResNetModified(
                Bottleneck,
                backbone_cfg["layer_nums"],
                backbone_cfg["layer_strides"],
                backbone_cfg["num_filters"],
                inplanes=model_cfg.get("inplanes", backbone_cfg.get("inplanes", 64)),
                groups=32,
                width_per_group=4,
            )
        self.align_corners = backbone_cfg.get("align_corners", False)
        for i in range(self.resnet.layernum):
            setattr(
                self,
                f"estimator_for_layer{i}",
                nn.Conv2d(backbone_cfg["num_filters"][i], 1, kernel_size=1),
            )
        self.shrink_header = DownsampleConv(model_cfg["shrink_header_args"])

    def _estimator(self, level: int) -> nn.Module:
        return getattr(self, f"estimator_for_layer{level}")

    def forward(self, feature_list: Sequence[Tensor]) -> Tensor:
        """Negotiate one common BEV from same-shaped modality features."""
        if not feature_list:
            raise ValueError("Negotiator requires at least one modality feature")
        pyramid_feature_list: List[List[Tensor]] = [[] for _ in range(self.num_levels)]
        for mfeat in feature_list:
            mfeat_list = self.get_multiscale_feature(mfeat)
            for level in range(self.num_levels):
                occ_map = self._estimator(level)(mfeat_list[level])
                pyramid_feature_list[level].append(mfeat_list[level] * occ_map)
        fused_levels = [
            torch.mean(torch.stack(level_feats, dim=0), dim=0)
            for level_feats in pyramid_feature_list
        ]
        consensus = self.decode_multiscale_feature(fused_levels)
        return self.shrink_header(consensus)

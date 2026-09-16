# -*- coding: utf-8 -*-
"""Homogeneous collaborative Stage-0 detector for one AirV2X agent type."""

from __future__ import annotations

from typing import Any, Dict, List

from opencood.models.airv2x_detector_parts import Airv2xSharedDetector
from opencood.models.common_modules.airv2x_base_model import Airv2xBase
from opencood.utils.airv2x_freeze import apply_named_freeze_policy, assert_lss_frozen


class Airv2xHomoBase(Airv2xBase):
    """Homogeneous collaborative detector for exactly one agent type.

    Graph::

        1–N same-type agents
            → frozen pretrained LSS
            → ResNet BEV backbone
            → PyramidFusion
            → optional shrink
            → cls / reg / obj heads

    Vehicle Stage-0 uses 1–3 vehicles, RSU 1–2 RSUs, drone 1–2 drones.
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)
        self.args = args
        self.homo_agent_type = args.get("homo_agent_type")
        if self.homo_agent_type is None:
            if len(self.collaborators) != 1:
                raise ValueError(
                    "Airv2xHomoBase requires exactly one collaborator or an "
                    "explicit homo_agent_type"
                )
            self.homo_agent_type = self.collaborators[0]
        if self.homo_agent_type not in ("vehicle", "rsu", "drone"):
            raise ValueError(f"Unknown homo_agent_type: {self.homo_agent_type}")
        if self.collaborators != [self.homo_agent_type]:
            raise ValueError(
                "Homo Stage-0 collaborators must be "
                f"['{self.homo_agent_type}'], got {self.collaborators}"
            )

        self.init_encoders(args)
        self.detector = Airv2xSharedDetector(args)
        # Expose the same attribute names as HEAL so Stage-0 checkpoints can
        # be remapped by prefix (`backbone.*`, `pyramid_backbone.*`, ...).
        self.backbone = self.detector.backbone
        self.pyramid_backbone = self.detector.pyramid_backbone
        self.shrink_flag = self.detector.shrink_flag
        self.shrink_conv = self.detector.shrink_conv
        self.cls_head = self.detector.cls_head
        self.reg_head = self.detector.reg_head
        self.obj_head = self.detector.obj_head
        self.seg_head = self.detector.seg_head

        freeze_modules = {
            "backbone": bool(args.get("freeze_bev_backbone", False)),
            "pyramid_backbone": bool(args.get("freeze_fusion", False)),
            "cls_head": bool(args.get("freeze_heads", False)),
            "reg_head": bool(args.get("freeze_heads", False)),
        }
        if self.shrink_conv is not None:
            freeze_modules["shrink_conv"] = bool(args.get("freeze_fusion", False))
        if self.obj_head is not None:
            freeze_modules["obj_head"] = bool(args.get("freeze_heads", False))
        if self.seg_head is not None:
            freeze_modules["seg_head"] = bool(args.get("freeze_heads", False))

        apply_named_freeze_policy(
            self,
            freeze_lss=bool(args.get("freeze_lss", True)),
            freeze_modules=freeze_modules,
            assembled_inference=bool(args.get("assembled_inference", False)),
        )
        assert_lss_frozen(self)

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run homogeneous collaborative detection."""
        batch_output_dict, batch_record_len = self.extract_features(data_dict)
        spatial_features = batch_output_dict["spatial_features"]
        comm_rates = spatial_features.count_nonzero().item()
        spatial_features_2d = self.detector.encode_bev(spatial_features)
        fused_feature, _occ = self.detector.fuse_and_decode(
            spatial_features_2d,
            batch_record_len,
            data_dict["img_pairwise_t_matrix_collab"],
        )
        output_dict = self.detector.decode_heads(fused_feature)
        output_dict["comm_rate"] = comm_rates
        output_dict["pyramid"] = "collab"
        return output_dict

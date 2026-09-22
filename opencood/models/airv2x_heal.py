"""AirV2X HEAL: three agent-specific frozen LSS encoders + shared vehicle-domain detector."""

from __future__ import annotations

from typing import Any, Dict

from opencood.models.airv2x_detector_parts import Airv2xSharedDetector
from opencood.models.common_modules.airv2x_base_model import Airv2xBase
from opencood.models.lss_pretrain_modules.p1_mixin import P1CamMixin
from opencood.utils.airv2x_freeze import (
    apply_named_freeze_policy, assert_lss_frozen, set_module_trainable,
)


class Airv2xHEAL(P1CamMixin, Airv2xBase):
    """AirV2X adaptation of HEAL.

    Graph::

        Vehicle LSS ─┐
        RSU LSS ─────┼→ shared BEV backbone (vehicle domain)
        Drone LSS ───┘
                         ↓
                 shared PyramidFusion
                         ↓
                    shared heads

    Stage-0 initialization is explicit (not load-order dependent):

    * `veh_models` ← vehicle base encoder
    * `rsu_models` ← rsu base encoder
    * `drone_models` ← drone base encoder
    * shared backbone / fusion / heads ← vehicle base
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)
        self.args = args
        self.collaborators = args["collaborators"]
        self.active_sensors = args["active_sensors"]
        if not args.get("p1_checkpoint"):
            raise ValueError("HEAL requires --p1_checkpoint for the P1 homo bases")
        self.init_encoders(args)

        self.detector = Airv2xSharedDetector(args)
        self.backbone = self.detector.backbone
        self.pyramid_backbone = self.detector.pyramid_backbone
        self.shrink_flag = self.detector.shrink_flag
        self.shrink_conv = self.detector.shrink_conv
        self.cls_head = self.detector.cls_head
        self.reg_head = self.detector.reg_head
        self.obj_head = self.detector.obj_head
        self.seg_head = self.detector.seg_head

        # `backbone_fix` is legacy. Prefer explicit freeze_* flags.
        legacy_fix = args.get("backbone_fix", False)
        freeze_bev = bool(args.get("freeze_bev_backbone", False))
        freeze_fusion = bool(args.get("freeze_fusion", False))
        freeze_heads = bool(args.get("freeze_heads", False))
        if legacy_fix and not any(
            k in args for k in ("freeze_bev_backbone", "freeze_fusion", "freeze_heads")
        ):
            # Legacy HEAL collab used backbone_fix to freeze RSU/Drone LSS
            # AND the shared detector. LSS is always frozen now; keep the
            # shared detector trainable unless freeze_* is set.
            print(
                "[HEAL] Ignoring legacy backbone_fix for LSS (all LSS are "
                "frozen). Shared detector trainability is controlled by "
                "freeze_bev_backbone / freeze_fusion / freeze_heads."
            )

        freeze_modules = {
            "backbone": freeze_bev,
            "pyramid_backbone": freeze_fusion,
            "cls_head": freeze_heads,
            "reg_head": freeze_heads,
        }
        if self.shrink_conv is not None:
            freeze_modules["shrink_conv"] = freeze_fusion
        if self.obj_head is not None:
            freeze_modules["obj_head"] = freeze_heads
        if self.seg_head is not None:
            freeze_modules["seg_head"] = freeze_heads

        apply_named_freeze_policy(
            self,
            freeze_lss=bool(args.get("freeze_lss", True)),
            freeze_modules=freeze_modules,
            assembled_inference=bool(args.get("assembled_inference", False)),
        )
        for attr in ("veh_models", "rsu_models", "drone_models"):
            set_module_trainable(getattr(self, attr), False)
        assert_lss_frozen(self)

    def train(self, mode: bool = True) -> "Airv2xHEAL":
        super().train(mode)
        for attr in ("veh_models", "rsu_models", "drone_models"):
            getattr(self, attr).eval()
        return self

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Heterogeneous collab detection in the vehicle domain."""
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

"""AirV2X STAMP: frozen shared detector + agent-type adapters before fusion."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from opencood.models.airv2x_detector_parts import Airv2xSharedDetector
from opencood.models.common_modules.airv2x_base_model import Airv2xBase
from opencood.models.fuse_modules.adapter import Adapter
from opencood.utils.airv2x_agent_repack import infer_batch_size, repack_agent_features
from opencood.utils.airv2x_freeze import (
    apply_named_freeze_policy,
    assert_lss_frozen,
    set_module_trainable,
)


class Airv2xSTAMP(Airv2xBase):
    """AirV2X STAMP baseline (not the full original STAMP protocol).

    Graph::

        agent LSS (frozen, agent-specific)
            → shared BEV backbone (frozen, vehicle domain)
            → agent-specific adapter  ← only trainable modules
            → shared PyramidFusion (frozen)
            → shared detection head (frozen)

    The adapter output is packed in scene-agent order and is the tensor
    that PyramidFusion actually consumes.
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)
        self.args = args
        self.collaborators = args["collaborators"]
        self.active_sensors = args["active_sensors"]
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

        self.adapters = torch.nn.ModuleDict()
        for agent_type in self.collaborators:
            adapter_cfg = args[agent_type].get("adapter")
            if adapter_cfg is None:
                raise KeyError(f"STAMP requires args['{agent_type}']['adapter']")
            self.adapters[agent_type] = Adapter(adapter_cfg)

        freeze_modules = {
            "backbone": True,
            "pyramid_backbone": True,
            "cls_head": True,
            "reg_head": True,
        }
        if self.shrink_conv is not None:
            freeze_modules["shrink_conv"] = True
        if self.obj_head is not None:
            freeze_modules["obj_head"] = True
        if self.seg_head is not None:
            freeze_modules["seg_head"] = True

        apply_named_freeze_policy(
            self,
            freeze_lss=bool(args.get("freeze_lss", True)),
            freeze_modules=freeze_modules,
            assembled_inference=bool(args.get("assembled_inference", False)),
            allow_zero_trainable=True,
        )
        for adapter in self.adapters.values():
            set_module_trainable(adapter, True)
            adapter.train()
        assert_lss_frozen(self)
        n_adapter = sum(p.numel() for p in self.adapters.parameters() if p.requires_grad)
        if n_adapter == 0 and not args.get("assembled_inference", False):
            raise RuntimeError(
                "STAMP has no trainable adapter parameters. Identity adapters "
                "cannot be the only adapters during adapter training."
            )
        print(f"[STAMP] trainable adapter parameters: {n_adapter:,}")

    def _encode_type(self, data_dict: Dict[str, Any], agent_type: str) -> Optional[Tensor]:
        """LSS → shared backbone → adapter for one agent type."""
        agent = data_dict.get(agent_type)
        if not isinstance(agent, dict) or len(agent.get("batch_idxs") or []) == 0:
            return None
        models = {
            "vehicle": self.veh_models,
            "rsu": self.rsu_models,
            "drone": self.drone_models,
        }[agent_type]
        if models is None:
            raise RuntimeError(f"{agent_type} encoder is not initialized")
        encoded = self.fuse_bev([encoder(data_dict) for encoder in models])
        bev = self.backbone(encoded)["spatial_features_2d"]
        return self.adapters[agent_type](bev)

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Heterogeneous collab detection with adapted BEV features."""
        feature_by_type = {}
        for agent_type in self.collaborators:
            feat = self._encode_type(data_dict, agent_type)
            if feat is not None:
                feature_by_type[agent_type] = feat
        if not feature_by_type:
            raise RuntimeError("STAMP received a batch with no collaborating agents")

        packed, record_len = repack_agent_features(
            feature_by_type, data_dict, batch_size=infer_batch_size(data_dict)
        )
        comm_rates = packed.count_nonzero().item()
        fused_feature, _occ = self.detector.fuse_and_decode(
            packed,
            record_len,
            data_dict["img_pairwise_t_matrix_collab"],
        )
        output_dict = self.detector.decode_heads(fused_feature)
        output_dict["comm_rate"] = comm_rates
        output_dict["pyramid"] = "collab"
        output_dict["stamp_adapted_features"] = packed
        return output_dict

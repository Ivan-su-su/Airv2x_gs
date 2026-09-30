# -*- coding: utf-8 -*-
"""Griffin STAMP Stage-1: frozen shared detector + trainable agent adapters.

Graph (vehicle domain)::

    vehicle P1 → BevEncode (frozen) → backbone (frozen) → adapter ─┐
                                                                    ├→ PyramidFusion (frozen*)
    drone P1 → BevEncode (frozen) → backbone (frozen) → adapter ───┘       ↓
                                                            shrink + heads (frozen)

(*) ``pyramid_backbone.single_head_*`` is released for training: Stage-0
    N=1 left those occupancy heads at zero gradient, and freezing them
    here would lock in random agent-confidence weights (user decision).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from opencood.models.fuse_modules.adapter import Adapter
from opencood.models.griffin_collab_base import GriffinCollabBase
from opencood.utils.airv2x_agent_repack import infer_batch_size, repack_agent_features
from opencood.utils.airv2x_freeze import set_module_trainable


class GriffinSTAMPCollab(GriffinCollabBase):
    """STAMP collab: only adapters (+ released single_head_*) train."""

    def __init__(self, args: Dict[str, Any]) -> None:
        args = dict(args)
        args.setdefault("freeze_bev_backbone", True)
        args.setdefault("freeze_fusion", True)
        args.setdefault("freeze_heads", True)
        super().__init__(args)

        self.adapters = torch.nn.ModuleDict()
        for agent_type in self.collaborators:
            adapter_cfg = args[agent_type].get("adapter")
            if adapter_cfg is None:
                raise KeyError(
                    f"GriffinSTAMPCollab requires args['{agent_type}']['adapter']"
                )
            self.adapters[agent_type] = Adapter(adapter_cfg)

        # Adapters are the main STAMP trainable body.
        for adapter in self.adapters.values():
            set_module_trainable(adapter, True)

        n_adapter = sum(
            p.numel() for p in self.adapters.parameters() if p.requires_grad
        )
        print(f"[griffin-stamp] trainable adapter parameters: {n_adapter:,}")

    def train(self, mode: bool = True) -> "GriffinSTAMPCollab":
        """Keep frozen detector BN in eval; only adapters stay in train."""
        super().train(mode)
        if mode:
            self.detector.eval()
            for adapter in self.adapters.values():
                adapter.train()
        return self

    def _encode_type(self, data_dict: Dict[str, Any], agent_type: str) -> Optional[Tensor]:
        """Frozen LSS → frozen shared backbone → adapter for one agent type."""
        agent = data_dict.get(agent_type)
        if not isinstance(agent, dict) or len(agent.get("batch_idxs") or []) == 0:
            return None
        models = {
            "vehicle": self.veh_models,
            "drone": self.drone_models,
        }[agent_type]
        encoded = self.fuse_bev([encoder(data_dict) for encoder in models])
        bev = self.backbone(encoded)["spatial_features_2d"]
        return self.adapters[agent_type](bev)

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        feature_by_type = {}
        for agent_type in self.collaborators:
            feat = self._encode_type(data_dict, agent_type)
            if feat is not None:
                feature_by_type[agent_type] = feat
        if not feature_by_type:
            raise RuntimeError(
                "GriffinSTAMPCollab received a batch with no collaborating agents"
            )

        packed, record_len = repack_agent_features(
            feature_by_type, data_dict, batch_size=infer_batch_size(data_dict)
        )
        comm_rates = packed.count_nonzero().item()
        fused_feature, _occ = self.detector.fuse_and_decode(
            packed,
            record_len,
            self.collab_affine(packed, data_dict["img_pairwise_t_matrix_collab"]),
        )
        output_dict = self.detector.decode_heads(fused_feature)
        output_dict["comm_rate"] = comm_rates
        output_dict["pyramid"] = "collab"
        output_dict["stamp_adapted_features"] = packed
        return output_dict

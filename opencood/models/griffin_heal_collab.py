# -*- coding: utf-8 -*-
"""Griffin HEAL Stage-1: two frozen P1 frontends + trainable shared detector.

Graph (vehicle domain, unified Griffin detection range)::

    vehicle P1 → BevEncode (frozen) ─┐
                                     ├→ shared backbone (trainable)
    drone P1 → BevEncode (frozen) ───┘        ↓
                                    PyramidFusion (trainable, N=2)
                                              ↓
                                     shrink + det heads (trainable)
"""

from __future__ import annotations

from typing import Any, Dict

from opencood.models.griffin_collab_base import GriffinCollabBase


class GriffinHEALCollab(GriffinCollabBase):
    """HEAL collab: LSS frozen, shared detector fully trainable."""

    def __init__(self, args: Dict[str, Any]) -> None:
        args = dict(args)
        args.setdefault("freeze_bev_backbone", False)
        args.setdefault("freeze_fusion", False)
        args.setdefault("freeze_heads", False)
        super().__init__(args)

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        batch_output_dict, batch_record_len = self.extract_features(data_dict)
        spatial_features = batch_output_dict["spatial_features"]
        comm_rates = spatial_features.count_nonzero().item()
        spatial_features_2d = self.detector.encode_bev(spatial_features)
        fused_feature, _occ = self.detector.fuse_and_decode(
            spatial_features_2d,
            batch_record_len,
            self.collab_affine(spatial_features_2d, data_dict["img_pairwise_t_matrix_collab"]),
        )
        output_dict = self.detector.decode_heads(fused_feature)
        output_dict["comm_rate"] = comm_rates
        output_dict["pyramid"] = "collab"
        return output_dict

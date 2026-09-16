# -*- coding: utf-8 -*-
"""NegoCollab losses for AirV2X detection.

Stage 1 (`nego`) uses official-style cycle / unify / occupancy-pragma
losses. Stage 2 (`ft`) reuses the existing AirV2X detection loss so
anchors, GT assignment and the evaluator stay unchanged.

AirV2X adaptation vs official `nego_loss_distill-o.py`:
* `unify_stru` / semantic distill are replaced by occupancy-weighted MSE
  (same tensors, simpler formula) because AirV2X scenes have unequal
  agent counts per type and we negotiate type-mean prototypes.
* Detection pragma in stage 1 uses a single occupancy head on the
  common representation instead of a full pyramid detector.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from opencood.loss.point_pillar_loss_multiclass import PointPillarLossMultiClass


class Airv2xNegoCollabLoss(nn.Module):
    """Stage-aware NegoCollab loss wrapper."""

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__()
        self.stage = str(args.get("stage", "ft"))
        self.cycle_ratio = float(args.get("cycle_ratio", 1.0))
        self.unify_dis_ratio = float(args.get("unify_dis_ratio", 1.0))
        self.pragma_ratio = float(args.get("pragma_ratio", 0.5))
        det_args = dict(args.get("detection") or {})
        if "num_class" not in det_args and "num_class" in args:
            det_args["num_class"] = args["num_class"]
        if "cls_weight" not in det_args:
            det_args["cls_weight"] = args.get("cls_weight", 1.0)
        if "reg" not in det_args:
            det_args["reg"] = args.get("reg", 2.0)
        self.det_loss = PointPillarLossMultiClass(det_args)
        self.loss_dict: Dict[str, float] = {}

    def forward(self, output_dict: Dict[str, Any], target_dict: Dict[str, Any], prefix: str = "") -> Tensor:
        stage = str(output_dict.get("nego_stage", self.stage))
        if stage == "nego":
            return self._nego_forward(output_dict, target_dict)
        return self._ft_forward(output_dict, target_dict, prefix)

    def _ft_forward(self, output_dict: Dict[str, Any], target_dict: Dict[str, Any], prefix: str) -> Tensor:
        loss = self.det_loss(output_dict, target_dict, prefix=prefix)
        self.loss_dict = dict(self.det_loss.loss_dict)
        return loss

    def _occupancy_target(self, target_dict: Dict[str, Any], height: int, width: int) -> Tensor:
        pos = target_dict["pos_equal_one"]
        if pos.dim() == 4:
            occ = torch.logical_or(pos[..., 0], pos[..., 1]).unsqueeze(1).float()
        else:
            occ = pos.float()
            if occ.dim() == 3:
                occ = occ.unsqueeze(1)
        return F.interpolate(occ, size=(height, width), mode="bilinear", align_corners=False)

    def _nego_forward(self, output_dict: Dict[str, Any], target_dict: Dict[str, Any]) -> Tensor:
        before = output_dict["fc_before_send"]
        after = output_dict["fc_after_receive"]
        cycle = F.mse_loss(after, before)

        common_expanded = output_dict["common_expanded"]
        packed_sent = output_dict["packed_sent"]
        if common_expanded.shape[-2:] != packed_sent.shape[-2:]:
            common_expanded = F.interpolate(
                common_expanded,
                size=packed_sent.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        unify = F.mse_loss(packed_sent, common_expanded)

        occ_pred = output_dict["occ_preds_nego"]
        occ_tgt = self._occupancy_target(target_dict, occ_pred.shape[-2], occ_pred.shape[-1])
        pragma = F.binary_cross_entropy_with_logits(occ_pred, occ_tgt)

        total = (
            self.cycle_ratio * cycle
            + self.unify_dis_ratio * unify
            + self.pragma_ratio * pragma
        )
        self.loss_dict = {
            "total_loss": float(total.item()),
            "cycle_loss": float(cycle.item()),
            "unify_dis_loss": float(unify.item()),
            "pragma_loss": float(pragma.item()),
        }
        return total

    def logging(self, epoch, batch_id, batch_len, writer=None, pbar=None, suffix=""):
        """Match AirV2X detection-loss logging contract."""
        if self.loss_dict.get("cycle_loss") is not None and "cls_loss" not in self.loss_dict:
            msg = (
                f"[epoch {epoch}][{batch_id + 1}/{batch_len}]{suffix} || "
                f"Loss: {self.loss_dict.get('total_loss', 0):.4f} || "
                f"Cycle: {self.loss_dict.get('cycle_loss', 0):.4f} || "
                f"Unify: {self.loss_dict.get('unify_dis_loss', 0):.4f} || "
                f"Pragma: {self.loss_dict.get('pragma_loss', 0):.4f}"
            )
            if writer is not None:
                step = epoch * batch_len + batch_id
                for key, value in self.loss_dict.items():
                    writer.add_scalar(key + suffix, value, step)
            if pbar is None:
                print(msg)
            else:
                pbar.set_description(msg)
            return msg
        return self.det_loss.logging(epoch, batch_id, batch_len, writer)

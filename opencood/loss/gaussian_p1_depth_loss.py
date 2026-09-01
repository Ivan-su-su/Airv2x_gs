"""Heterogeneous P1 depth loss: Focal for vehicle/RSU, SmoothL1 for drone."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from opencood.loss.point_pillar_depth_loss import FocalLoss
from opencood.models.gaussian_modules_0822.lss.target import depth_valid_mask

DRONE_Z_MIN = 6.0
DRONE_Z_MAX = 150.0


class GaussianP1DepthLoss(nn.Module):
    """Vehicle: global in-range Focal. RSU: SAM3 FG Focal. Drone: SmoothL1."""

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__()
        self.depth_loss_func = FocalLoss(
            alpha=float(args.get("alpha", 0.25)),
            gamma=float(args.get("gamma", 2.0)),
            reduction="none",
            smooth_target=bool(args.get("smooth_target", False)),
        )
        self.drone_smooth_l1 = nn.SmoothL1Loss(reduction="none")
        self.drone_z_min = float(args.get("drone_z_min", DRONE_Z_MIN))
        self.drone_z_max = float(args.get("drone_z_max", DRONE_Z_MAX))
        self.smooth_l1_beta = float(self.drone_smooth_l1.beta)
        self.loss_dict: Dict[str, float] = {}
        print(
            f"[P1 depth] vehicle Focal alpha={args.get('alpha', 0.25)} "
            f"gamma={args.get('gamma', 2.0)} on all valid in-range pixels; "
            f"RSU Focal on SAM3 foreground ∩ in-range; "
            f"drone SmoothL1 beta={self.smooth_l1_beta} z in "
            f"[{self.drone_z_min}, {self.drone_z_max}]"
        )

    def _drone_residual_loss(
        self,
        pred: Dict[str, torch.Tensor],
        delta_gt: torch.Tensor,
        semantic_target: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if "delta_pred" not in pred:
            raise KeyError("drone prediction is missing delta_pred")
        height = pred["camera_world_z"].reshape(-1, 1, 1)
        z_gt = delta_gt + height
        fg_mask = semantic_target.ne(0)
        drone_mask = fg_mask & depth_valid_mask(z_gt, self.drone_z_min, self.drone_z_max)
        n_fg = int(fg_mask.sum().item())
        n_valid = int(drone_mask.sum().item())
        self.loss_dict["drone_fg_cells"] = float(n_fg)
        self.loss_dict["drone_valid_cells"] = float(n_valid)
        self.loss_dict["drone_excluded_fg_cells"] = float(n_fg - n_valid)
        if n_valid == 0:
            return None
        return self.drone_smooth_l1(pred["delta_pred"], delta_gt)[drone_mask].mean()

    def _categorical_focal_loss(
        self,
        agent_type: str,
        pred: Dict[str, torch.Tensor],
        depth_gt: torch.Tensor,
        valid: torch.Tensor,
        semantic_target: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Vehicle: all in-range cells. RSU: SAM3 foreground ∩ in-range."""
        if agent_type == "rsu":
            if tuple(semantic_target.shape) != tuple(valid.shape):
                raise AssertionError(
                    f"rsu semantic_target {tuple(semantic_target.shape)} vs "
                    f"valid {tuple(valid.shape)}"
                )
            fg_mask = semantic_target.ne(0)
            valid = valid & fg_mask
            self.loss_dict["rsu_fg_cells"] = float(int(fg_mask.sum().item()))
            self.loss_dict["rsu_valid_cells"] = float(int(valid.sum().item()))
        if int(valid.sum().item()) == 0:
            return None
        loss_map = self.depth_loss_func(pred["depth_logits"], depth_gt)
        return loss_map[valid].mean()

    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        depth_targets: Dict[str, torch.Tensor],
        semantic_targets: Dict[str, torch.Tensor],
        depth_valid_masks: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        agent_losses = []
        zero_ref: Optional[torch.Tensor] = None
        per_agent: Dict[str, float] = {}
        for agent_type, pred in predictions.items():
            if agent_type not in depth_targets or agent_type not in semantic_targets:
                raise KeyError(f"missing depth/semantic target for agent {agent_type}")
            if zero_ref is None:
                ref = pred.get("heatmap_logits", pred.get("depth_z_mean"))
                zero_ref = ref.sum() * 0.0
            if agent_type == "drone":
                agent_loss = self._drone_residual_loss(
                    pred, depth_targets[agent_type], semantic_targets[agent_type]
                )
            else:
                agent_loss = self._categorical_focal_loss(
                    agent_type,
                    pred,
                    depth_targets[agent_type],
                    depth_valid_masks[agent_type],
                    semantic_targets[agent_type],
                )
            if agent_loss is None:
                continue
            agent_losses.append(agent_loss)
            per_agent[agent_type] = float(agent_loss.detach().item())
        if not agent_losses:
            if zero_ref is None:
                raise RuntimeError("depth loss received no present camera agents.")
            depth_loss = zero_ref
        else:
            depth_loss = torch.stack(agent_losses).mean()
        self.loss_dict["depth_loss"] = float(depth_loss.detach().item())
        for agent_type in ("vehicle", "rsu", "drone"):
            key = f"depth_loss_{agent_type}"
            if agent_type in per_agent:
                self.loss_dict[key] = per_agent[agent_type]
            else:
                self.loss_dict.pop(key, None)
        if "rsu" not in predictions:
            self.loss_dict.pop("rsu_fg_cells", None)
            self.loss_dict.pop("rsu_valid_cells", None)
        return depth_loss

    def logging(
        self,
        epoch: int,
        batch_id: int,
        batch_len: int,
        writer: Optional[Any] = None,
        pbar: Optional[Any] = None,
    ) -> str:
        depth_loss = self.loss_dict.get("depth_loss", 0.0)
        parts = ["depth: %.4f" % depth_loss]
        for agent_type in ("vehicle", "rsu", "drone"):
            key = f"depth_loss_{agent_type}"
            if key in self.loss_dict:
                parts.append(f"{agent_type}: {self.loss_dict[key]:.4f}")
        if writer is not None:
            step = epoch * batch_len + batch_id
            writer.add_scalar("Train/depth_loss", depth_loss, step)
            for agent_type in ("vehicle", "rsu", "drone"):
                key = f"depth_loss_{agent_type}"
                if key in self.loss_dict:
                    writer.add_scalar(f"Train/{key}", self.loss_dict[key], step)
        return " | ".join(parts)

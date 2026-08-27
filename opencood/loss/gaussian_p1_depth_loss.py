"""Heterogeneous P1 depth loss: Focal for vehicle/RSU, SmoothL1 for drone."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from opencood.loss.point_pillar_depth_loss import FocalLoss

DRONE_Z_MIN = 6.0
DRONE_Z_MAX = 150.0


class GaussianP1DepthLoss(nn.Module):
    """Vehicle/RSU: official Focal on classes. Drone: SmoothL1 on residual.

    Agent-type losses are each reduced over that type's supervised cells,
    then averaged across present types with a valid loss. No task weights.

    Args:
        args: Loss yaml args. Focal defaults ``alpha=0.25``, ``gamma=2.0``.
            Drone SmoothL1 uses PyTorch default ``beta=1.0``.
    """

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
            f"[P1 depth] vehicle/RSU Focal alpha={args.get('alpha', 0.25)} "
            f"gamma={args.get('gamma', 2.0)}; drone SmoothL1 beta="
            f"{self.smooth_l1_beta} z in [{self.drone_z_min}, {self.drone_z_max}]"
        )

    def _drone_residual_loss(
        self,
        pred: Dict[str, torch.Tensor],
        delta_gt: torch.Tensor,
        semantic_target: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """SmoothL1 on ``delta_pred`` vs ``delta_gt`` with physical mask."""
        if "delta_pred" not in pred:
            raise KeyError("drone prediction is missing delta_pred")
        delta_pred = pred["delta_pred"]
        height = pred["camera_world_z"].reshape(-1, 1, 1)
        z_gt = delta_gt + height
        fg_mask = semantic_target.ne(0)
        valid_z = torch.isfinite(z_gt) & (z_gt >= self.drone_z_min) & (
            z_gt <= self.drone_z_max
        )
        drone_mask = fg_mask & valid_z
        n_fg = int(fg_mask.sum().item())
        n_valid = int(drone_mask.sum().item())
        n_excluded = n_fg - n_valid
        self.loss_dict["drone_fg_cells"] = float(n_fg)
        self.loss_dict["drone_valid_cells"] = float(n_valid)
        self.loss_dict["drone_excluded_fg_cells"] = float(n_excluded)
        if n_valid == 0:
            return None
        loss_map = self.drone_smooth_l1(delta_pred, delta_gt)
        return loss_map[drone_mask].mean()

    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        depth_targets: Dict[str, torch.Tensor],
        semantic_targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """One scalar per present agent type, then mean over those scalars."""
        agent_losses = []
        zero_ref: Optional[torch.Tensor] = None
        per_agent: Dict[str, float] = {}
        for agent_type, pred in predictions.items():
            if agent_type not in depth_targets:
                raise KeyError(f"missing depth target for agent {agent_type}")
            if agent_type not in semantic_targets:
                raise KeyError(f"missing semantic target for agent {agent_type}")
            if zero_ref is None:
                ref = pred.get("heatmap_logits", pred.get("depth_z_mean"))
                zero_ref = ref.sum() * 0.0
            if agent_type == "drone":
                agent_loss = self._drone_residual_loss(
                    pred, depth_targets[agent_type], semantic_targets[agent_type]
                )
            else:
                logits = pred["depth_logits"]
                loss_map = self.depth_loss_func(logits, depth_targets[agent_type])
                fg_mask = semantic_targets[agent_type].ne(0)
                if int(fg_mask.sum().item()) == 0:
                    agent_loss = None
                else:
                    agent_loss = loss_map[fg_mask].mean()
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
        return depth_loss

    def logging(
        self,
        epoch: int,
        batch_id: int,
        batch_len: int,
        writer: Optional[Any] = None,
        pbar: Optional[Any] = None,
    ) -> str:
        """Log aggregated depth loss and per-agent terms. Trainer owns total."""
        depth_loss = self.loss_dict.get("depth_loss", 0.0)
        parts = ["depth: %.4f" % depth_loss]
        for agent_type in ("vehicle", "rsu", "drone"):
            key = f"depth_loss_{agent_type}"
            if key in self.loss_dict:
                parts.append(f"{agent_type}: {self.loss_dict[key]:.4f}")
        msg = " | ".join(parts)
        if writer is not None:
            step = epoch * batch_len + batch_id
            writer.add_scalar("Train/depth_loss", depth_loss, step)
            for agent_type in ("vehicle", "rsu", "drone"):
                key = f"depth_loss_{agent_type}"
                if key in self.loss_dict:
                    writer.add_scalar(f"Train/{key}", self.loss_dict[key], step)
        return msg

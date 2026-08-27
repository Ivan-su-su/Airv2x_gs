"""2-class softmax Focal Loss for P1 objectness. Targets and metrics live elsewhere."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


def softmax_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Mean 2-class softmax focal loss. No alpha / class weights.

    For each cell ``y``:

        CE = cross_entropy(logits, y, reduction="none")
        p_t = softmax(logits)[y]
        FL = (1 - p_t) ** gamma * CE

    Args:
        logits: ``[N, 2, H, W]`` raw logits. Channel 0=bg, 1=fg.
        target: ``[N, H, W]`` long ids in ``{0, 1}``.
        gamma: Focusing parameter. ``gamma=0`` recovers ordinary CE.

    Returns:
        Scalar mean over all cells.

    Raises:
        ValueError: If logits are not 2-class ``[N, 2, H, W]``.
    """
    if logits.dim() != 4 or int(logits.shape[1]) != 2:
        raise ValueError(
            f"softmax_focal_loss expects [N,2,H,W] logits, got {tuple(logits.shape)}"
        )
    ce = F.cross_entropy(logits, target, reduction="none")
    probs = torch.softmax(logits, dim=1)
    p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)
    focal_weight = (1.0 - p_t).pow(float(gamma))
    return (focal_weight * ce).mean()


class GaussianP1SemanticLoss(nn.Module):
    """Per-agent mean softmax focal, then equal mean over present agents.

    Args:
        args: Loss yaml args. ``gamma`` defaults to 2.0. Alpha is not used.
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__()
        self.gamma = float((args or {}).get("gamma", 2.0))
        self.loss_dict: Dict[str, float] = {}

    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """``predictions[agent]['heatmap_logits']`` vs ``targets[agent]``."""
        agent_losses = []
        for agent_type, pred in predictions.items():
            if agent_type not in targets:
                raise KeyError(f"missing semantic target for agent {agent_type}")
            agent_losses.append(
                softmax_focal_loss(
                    pred["heatmap_logits"],
                    targets[agent_type],
                    gamma=self.gamma,
                )
            )
        if not agent_losses:
            raise RuntimeError("semantic loss received no present camera agents.")
        heatmap_loss = torch.stack(agent_losses).mean()
        self.loss_dict = {"heatmap_loss": float(heatmap_loss.detach().item())}
        return heatmap_loss

    def logging(
        self,
        epoch: int,
        batch_id: int,
        batch_len: int,
        writer: Optional[Any] = None,
        pbar: Optional[Any] = None,
    ) -> str:
        """Log heatmap focal only. Trainer owns total_loss logging."""
        heatmap_loss = self.loss_dict.get("heatmap_loss", 0.0)
        msg = "focal: %.4f" % heatmap_loss
        if writer is not None:
            writer.add_scalar("Train/heatmap_loss", heatmap_loss, epoch * batch_len + batch_id)
        return msg

"""Objectness diagnostic metrics. No loss, no target construction, no logging."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

BACKGROUND_CLASS_ID = 0
OBJECTNESS_THRESHOLDS: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)
PRIMARY_OBJECTNESS_THRESHOLD = 0.3


def compute_heatmap_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Foreground precision / recall / F1 at fixed p_fg thresholds.

    Args:
        logits: ``[N, 2, H, W]`` objectness logits.
        target: ``[N, H, W]`` long ids in ``{0, 1}``.

    Returns:
        Dict of scalar floats.
    """
    with torch.no_grad():
        return _compute_heatmap_metrics(logits.detach(), target.detach())


def _compute_heatmap_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Implementation of ``compute_heatmap_metrics`` (already detached)."""
    if logits.dim() != 4 or int(logits.shape[1]) != 2:
        raise ValueError(
            f"objectness metrics expect [N,2,H,W] logits, got {tuple(logits.shape)}"
        )
    probs = torch.softmax(logits, dim=1)
    p_fg = probs[:, 1]
    gt_fg = target.ne(BACKGROUND_CLASS_ID)
    gt_fg_count = int(gt_fg.sum().item())
    metrics: Dict[str, float] = {
        "gt_fg_ratio": float(gt_fg.float().mean().item()),
    }
    if gt_fg_count > 0:
        fg_scores = p_fg[gt_fg]
        metrics["mean_p_fg_gt"] = float(fg_scores.mean().item())
        metrics["median_p_fg_gt"] = float(fg_scores.median().item())
    else:
        metrics["mean_p_fg_gt"] = 0.0
        metrics["median_p_fg_gt"] = 0.0
    for tau in OBJECTNESS_THRESHOLDS:
        pred_fg = p_fg.ge(float(tau))
        tp = float((pred_fg & gt_fg).float().sum().item())
        fp = float((pred_fg & ~gt_fg).float().sum().item())
        fn = float((~pred_fg & gt_fg).float().sum().item())
        recall = tp / max(tp + fn, 1.0)
        precision = tp / max(tp + fp, 1.0)
        f1 = (2.0 * precision * recall) / max(precision + recall, 1.0e-12)
        tag = f"{tau:g}"
        metrics[f"recall@{tag}"] = recall
        metrics[f"precision@{tag}"] = precision
        metrics[f"f1@{tag}"] = f1
        if abs(float(tau) - PRIMARY_OBJECTNESS_THRESHOLD) < 1.0e-8:
            metrics["pred_fg_ratio"] = float(pred_fg.float().mean().item())
    return metrics

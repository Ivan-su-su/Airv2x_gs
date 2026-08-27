"""Camera-z MAE/RMSE. No loss, no binning, no model or batch access."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def _masked_mae_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    """MAE and RMSE on a boolean mask. Zeros if the mask is empty."""
    if int(mask.sum().item()) == 0:
        return 0.0, 0.0
    diff = pred[mask] - target[mask]
    mae = float(diff.abs().mean().item())
    rmse = float(torch.sqrt((diff * diff).mean()).item())
    return mae, rmse


def compute_depth_metrics(
    depth_z_mean: torch.Tensor,
    camera_z_gt: torch.Tensor,
    d_min: float,
    d_max: float,
    foreground_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Valid-range MAE/RMSE, plus foreground subset when a mask is given.

    Args:
        depth_z_mean: Predicted categorical z mean ``[N, fH, fW]``.
        camera_z_gt: Unclamped GT z at feature resolution.
        d_min: Official bin minimum.
        d_max: Official bin maximum.
        foreground_mask: Optional bool ``[N, fH, fW]``.

    Returns:
        ``mae_valid``, ``rmse_valid``; and ``mae_fg``, ``rmse_fg`` if masked.
    """
    with torch.no_grad():
        return _compute_depth_metrics(
            depth_z_mean.detach(),
            camera_z_gt.detach(),
            d_min,
            d_max,
            None if foreground_mask is None else foreground_mask.detach(),
        )


def _compute_depth_metrics(
    depth_z_mean: torch.Tensor,
    camera_z_gt: torch.Tensor,
    d_min: float,
    d_max: float,
    foreground_mask: Optional[torch.Tensor],
) -> Dict[str, float]:
    """Implementation of ``compute_depth_metrics`` (already detached)."""
    valid = (camera_z_gt >= d_min) & (camera_z_gt <= d_max)
    mae_valid, rmse_valid = _masked_mae_rmse(depth_z_mean, camera_z_gt, valid)
    metrics = {"mae_valid": mae_valid, "rmse_valid": rmse_valid}
    if foreground_mask is None:
        return metrics
    mae_fg, rmse_fg = _masked_mae_rmse(
        depth_z_mean, camera_z_gt, valid & foreground_mask
    )
    metrics["mae_fg"] = mae_fg
    metrics["rmse_fg"] = rmse_fg
    return metrics


compute_depth_metrics = compute_depth_metrics

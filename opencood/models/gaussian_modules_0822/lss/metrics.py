"""Camera-z MAE/RMSE. No loss, no binning, no model or batch access."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from opencood.models.gaussian_modules_0822.lss.target import depth_valid_mask


def _masked_mae_rmse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Tuple[float, float]:
    """MAE and RMSE on a boolean mask. Zeros if the mask is empty."""
    if int(mask.sum().item()) == 0:
        return 0.0, 0.0
    diff = pred[mask] - target[mask]
    return float(diff.abs().mean().item()), float(torch.sqrt((diff * diff).mean()).item())


def compute_depth_metrics(
    depth_z_mean: torch.Tensor,
    camera_z_gt: torch.Tensor,
    d_min: float,
    d_max: float,
    foreground_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Valid-range MAE/RMSE, plus foreground subset when a mask is given."""
    with torch.no_grad():
        valid = depth_valid_mask(camera_z_gt, d_min, d_max)
        mae_valid, rmse_valid = _masked_mae_rmse(depth_z_mean, camera_z_gt, valid)
        metrics = {"mae_valid": mae_valid, "rmse_valid": rmse_valid}
        if foreground_mask is not None:
            mae_fg, rmse_fg = _masked_mae_rmse(
                depth_z_mean, camera_z_gt, valid & foreground_mask
            )
            metrics["mae_fg"] = mae_fg
            metrics["rmse_fg"] = rmse_fg
        return metrics

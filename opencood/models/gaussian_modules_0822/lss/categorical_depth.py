"""Categorical LSS camera-z moments from official depth_head logits.

``depth_z_var`` is the variance of the categorical LSS distribution over
optical-axis z bins:

    p = softmax(depth_logits, dim=D)
    depth_z_mean = sum_i p_i * z_i
    depth_z_var  = sum_i p_i * (z_i - depth_z_mean)^2

``z_i`` come from official ``depth_discretization`` (meters, optical-axis z).

It is a categorical-distribution variance baseline.
It is NOT calibrated predictive uncertainty.
It is NOT unit-ray variance.

``depth_z_var`` is differentiable with respect to depth logits, but the
depth objective is ``FocalLoss(depth_logits, depth_z_indices_gt)`` and
does not depend on variance, so no extra gradient is generated through
this branch. Do not imply that ``depth_z_var`` itself is supervised.

Do not name this ``sigma_z`` or ``depth_uncertainty``.
Do not implement an independent variance head here.
Do not implement ``Sigma_ray`` here.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from opencood.utils.camera_utils import depth_discretization


class CategoricalDepthMoments(nn.Module):
    """depth_logits → (depth_z_mean, depth_z_var).

    Args:
        ddiscr: ``[d_min, d_max, num_bins]``.
        mode: Official LSS mode, ``LID`` or ``UD``.
    """

    def __init__(self, ddiscr: Sequence[Union[int, float]], mode: str) -> None:
        super().__init__()
        self.num_bins = int(ddiscr[2])
        self.mode = str(mode)
        z_bins = depth_discretization(
            float(ddiscr[0]), float(ddiscr[1]), self.num_bins, self.mode
        )
        self.register_buffer(
            "z_bins",
            torch.tensor(z_bins, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, depth_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Softmax over z bins, then mean and variance.

        Args:
            depth_logits: ``[N, D, fH, fW]``.

        Returns:
            ``(depth_z_mean, depth_z_var)``, each ``[N, fH, fW]``.
        """
        if depth_logits.dim() != 4:
            raise ValueError(
                f"depth_logits must be [N, D, H, W], got {tuple(depth_logits.shape)}"
            )
        if int(depth_logits.shape[1]) != self.num_bins:
            raise ValueError(
                f"depth_logits D={depth_logits.shape[1]} != num_bins={self.num_bins}"
            )
        depth_prob = F.softmax(depth_logits, dim=1)
        z_bins = self.z_bins.to(device=depth_logits.device, dtype=depth_prob.dtype)
        z_bins = z_bins.view(1, self.num_bins, 1, 1)
        depth_z_mean = (depth_prob * z_bins).sum(dim=1)
        depth_z_var = (depth_prob * (z_bins - depth_z_mean.unsqueeze(1)).pow(2)).sum(
            dim=1
        )
        return depth_z_mean, depth_z_var

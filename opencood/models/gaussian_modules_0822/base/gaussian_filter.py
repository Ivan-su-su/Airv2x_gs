"""Drop Gaussians whose mean is outside the ego BEV xyz box.

Not NMS and not confidence filtering. Bounds are the detection
``cav_lidar_range``, parsed once at init. Call after agent→ego.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet


class GaussianRangeFilter:
    """Keep Gaussians inside the ego ``cav_lidar_range``.

    Args:
        cav_range: ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
        enabled: If False, pass through and only record counts.
    """

    def __init__(self, cav_range: Sequence[float], enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.xyz_min: Tuple[float, float, float] = (
            float(cav_range[0]),
            float(cav_range[1]),
            float(cav_range[2]),
        )
        self.xyz_max: Tuple[float, float, float] = (
            float(cav_range[3]),
            float(cav_range[4]),
            float(cav_range[5]),
        )
        self.last_stats: Dict[str, Any] = {
            "agent": "",
            "before_filter_count": 0,
            "after_filter_count": 0,
            "removed_ratio": 0.0,
        }

    def __call__(
        self,
        gaussians: GaussianSet,
        agent_type: Optional[str] = None,
    ) -> GaussianSet:
        """Index-select Gaussians inside the ego xyz box.

        Args:
            gaussians: Input set in the ego frame. Parameters are not modified.
            agent_type: Log key. Defaults to ``gaussians.agent``.

        Returns:
            Filtered ``GaussianSet``.
        """
        agent = str(agent_type or gaussians.agent).lower()
        before = int(gaussians.n_gaussians)
        if self.enabled and before > 0:
            min_t = gaussians.mean.new_tensor(self.xyz_min)
            max_t = gaussians.mean.new_tensor(self.xyz_max)
            keep = ((gaussians.mean >= min_t) & (gaussians.mean <= max_t)).all(dim=-1)
            gaussians = gaussians.index_select(keep)
        after = int(gaussians.n_gaussians)
        removed_ratio = 0.0 if before == 0 else float(before - after) / float(before)
        self.last_stats = {
            "agent": agent,
            "before_filter_count": before,
            "after_filter_count": after,
            "removed_ratio": removed_ratio,
        }
        return gaussians

"""Agent-specific depth mean / variance for Gaussian initialization."""

from opencood.models.gaussian_modules_0822.depth.depth_initializer import (
    categorical_depth_moments,
    drone_depth_moments,
)
from opencood.models.gaussian_modules_0822.depth.uncertainty import (
    LOCAL_WINDOW_M,
    SIGMA_D_DRONE,
)

__all__ = [
    "LOCAL_WINDOW_M",
    "SIGMA_D_DRONE",
    "categorical_depth_moments",
    "drone_depth_moments",
]

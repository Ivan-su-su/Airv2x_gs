"""Stage-1 Gaussian parameter refinement."""

from opencood.models.gaussian_modules_0822.refinement.gaussian_refiner import (
    GaussianRefiner,
)
from opencood.models.gaussian_modules_0822.refinement.refine_heads import (
    AgentRefineHeads,
    GaussianRefineHeads,
)
from opencood.models.gaussian_modules_0822.refinement.rotation_utils import (
    axis_angle_to_quaternion,
    compose_rotation_residual,
    quaternion_multiply,
)

__all__ = [
    "AgentRefineHeads",
    "GaussianRefineHeads",
    "GaussianRefiner",
    "axis_angle_to_quaternion",
    "compose_rotation_residual",
    "quaternion_multiply",
]

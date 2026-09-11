"""Task-specific fusion MLPs. Geometry encoding is shared elsewhere."""

from opencood.models.gaussian_modules_0822.fusion.gaussian_bev_splat import (
    GaussianBEVSplat,
)
from opencood.models.gaussian_modules_0822.fusion.gaussian_voxel_pool import (
    GaussianVoxelPool,
)
from opencood.models.gaussian_modules_0822.fusion.offset_fusion import OffsetFusion
from opencood.models.gaussian_modules_0822.fusion.refinement_fusion import (
    RefinementFusion,
)

__all__ = [
    "GaussianBEVSplat",
    "GaussianVoxelPool",
    "OffsetFusion",
    "RefinementFusion",
]

"""Fixed R90 spatial contract. There is no grid selector.

Output cell ``(i, j)`` maps to source block ``[4*i:4*i+4, 4*j:4*j+4]``
and image center ``(u, v) = (4*j+2, 4*i+2)``. Heatmap target, depth
target, and both heads share this index.
"""

from __future__ import annotations

from typing import Tuple

FEAT_H = 90
FEAT_W = 160
BLOCK = 4
SPATIAL_STRIDE = 4
# Shared HighResFusion / HeatmapHead / DepthHead / DeltaHead width.
F90_CHANNELS = 128
# HeatmapHead channels: 0=background, 1=foreground. Source image_semantic_gts
# still uses ids 0..6; target construction collapses any id>0 to occupancy.
NUM_CLASSES = 2


def expected_feature_hw() -> Tuple[int, int]:
    """Return the fixed ``(90, 160)`` prediction grid."""
    return FEAT_H, FEAT_W

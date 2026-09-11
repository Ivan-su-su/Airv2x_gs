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
AGENT_TYPES: Tuple[str, ...] = ("vehicle", "rsu", "drone")
# HeatmapHead channels: 0=background, 1=foreground. Source image_semantic_gts
# still uses ids 0..6; target construction collapses any id>0 to occupancy.
NUM_CLASSES = 2


def expected_feature_hw() -> Tuple[int, int]:
    """Return the fixed ``(90, 160)`` prediction grid."""
    return FEAT_H, FEAT_W


def feature_hw_for_stride(
    image_hw: Tuple[int, int], stride: int
) -> Tuple[int, int]:
    """Return ``(H / stride, W / stride)`` for a plain stride grid.

    Args:
        image_hw: Source image ``(H, W)``.
        stride: Output stride in pixels. Cell ``(i, j)`` is centered at
            ``(stride * j + stride / 2, stride * i + stride / 2)``.

    Returns:
        Feature map ``(feat_h, feat_w)``.

    Raises:
        ValueError: If ``H`` or ``W`` is not divisible by ``stride``.
    """
    height, width = int(image_hw[0]), int(image_hw[1])
    stride_i = int(stride)
    if height % stride_i != 0 or width % stride_i != 0:
        raise ValueError(
            f"image {height}x{width} is not divisible by stride {stride_i}"
        )
    return height // stride_i, width // stride_i

"""Binary objectness target. GT occupancy ids only.

Recall-oriented occupancy on aligned 4x4 blocks. Canonical R90 cell
``(i, j)`` is source block ``[4i:4i+4, 4j:4j+4]`` with center
``(u, v) = (4j+2, 4i+2)``.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from opencood.models.gaussian_modules_0822.p1_layout import BLOCK


def binary_objectness_target(
    semantic: torch.Tensor,
    tau: int = 1,
) -> torch.Tensor:
    """Downsample source semantic ids to binary occupancy.

    A cell is foreground iff its 4x4 block contains at least ``tau`` pixels
    with id > 0. Foreground subclass identity is ignored.

    Args:
        semantic: ``[N, H, W]`` long ids. Source maps may still use 0..6.
        tau: Minimum non-background pixels to mark a cell as foreground.

    Returns:
        ``[N, H/4, W/4]`` long ids in ``{0, 1}``.

    Raises:
        ValueError: If rank is not 3 or ``H``/``W`` is not divisible by 4.
    """
    if semantic.dim() != 3:
        raise ValueError(f"semantic must be [N,H,W], got {tuple(semantic.shape)}")
    batch, height, width = semantic.shape
    if height % BLOCK != 0 or width % BLOCK != 0:
        raise ValueError(
            f"cannot partition {tuple(semantic.shape)} into {BLOCK}x{BLOCK} blocks"
        )
    out_h = height // BLOCK
    out_w = width // BLOCK
    patches = semantic.view(batch, out_h, BLOCK, out_w, BLOCK)
    flat = patches.permute(0, 1, 3, 2, 4).reshape(batch, out_h, out_w, BLOCK * BLOCK)
    n_fg = flat.gt(0).sum(dim=-1)
    return n_fg.ge(int(tau)).to(dtype=torch.long)


def build_semantic_target(cam_inputs: Mapping[str, Any], tau: int = 1) -> torch.Tensor:
    """Binary objectness on aligned 4x4 blocks.

    If the 4x4 block has fewer than ``tau`` non-background pixels, emit 0.
    Otherwise emit 1. Subclass majority / tie-break is not used.

    Args:
        cam_inputs: ``batch_merged_cam_inputs`` for one agent type.
        tau: Occupancy threshold (approved: 1).

    Returns:
        Objectness target of shape ``[N, 90, 160]``, dtype long, values in
        ``{0, 1}``.
    """
    semantic_gt = cam_inputs.get("image_semantic_gts")
    if not torch.is_tensor(semantic_gt):
        raise KeyError(
            "image_semantic_gts is required. Enable "
            "model.args.BACKBONE_2D.LOAD_IMAGE_SEMANTIC_GT."
        )
    if semantic_gt.dim() == 4:
        batch_agents, num_views, height, width = semantic_gt.shape
        semantic_gt = semantic_gt.reshape(batch_agents * num_views, height, width)
    if semantic_gt.dim() != 3:
        raise ValueError(
            f"image_semantic_gts must be 3D or 4D, got {tuple(semantic_gt.shape)}"
        )
    return binary_objectness_target(semantic_gt.long(), tau=int(tau))

# -*- coding: utf-8 -*-
"""Smoke tests for pairwise affine conversion / warping."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from opencood.utils.negocollab_transform import (
    identity_affine,
    to_airv2x_pyramid_affine,
    to_normalized_affine,
    warp_with_affine,
)


CAV_RANGE = [-140.8, -40.0, -3.0, 140.8, 40.0, 1.0]


def _impulse(batch: int, height: int, width: int, y: int, x: int) -> torch.Tensor:
    feat = torch.zeros(batch, 1, height, width)
    feat[:, 0, y, x] = 1.0
    return feat


def test_identity_keeps_feature() -> None:
    feat = _impulse(1, 32, 64, 10, 20)
    affine = identity_affine(1, feat.device, feat.dtype)
    out = warp_with_affine(feat, affine, align_corners=False)
    err = (out - feat).abs().max().item()
    assert err < 1e-5, err


def test_airv2x_slice_identity() -> None:
    pairwise = torch.eye(4).view(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1)
    sliced = to_airv2x_pyramid_affine(pairwise)
    assert sliced.shape == (1, 2, 2, 2, 3)
    feat = _impulse(1, 16, 32, 5, 8)
    out = warp_with_affine(feat, sliced[0, 0, :1], align_corners=False)
    assert (out - feat).abs().max().item() < 1e-5


def test_normalized_identity() -> None:
    pairwise = torch.eye(4).view(1, 1, 1, 4, 4)
    affine = to_normalized_affine(pairwise, cav_lidar_range=CAV_RANGE)
    feat = _impulse(1, 20, 40, 8, 12)
    out = warp_with_affine(feat, affine[0, 0], align_corners=False)
    assert (out - feat).abs().max().item() < 1e-4


def _peak(feat: torch.Tensor) -> tuple:
    idx = feat.view(feat.shape[0], -1).argmax(dim=-1)
    width = feat.shape[-1]
    y = int(idx[0] // width)
    x = int(idx[0] % width)
    return y, x


def test_plus_x_shift() -> None:
    """Positive x translation in normalized affine should move the impulse."""
    feat = _impulse(1, 32, 64, 16, 32)
    affine = identity_affine(1, feat.device, feat.dtype).clone()
    # affine_grid: x' = a00 x + a01 y + a02, with x,y in [-1, 1]
    affine[0, 0, 2] = -4.0 / 32.0  # shift ~2 pixels to +x in output
    out = warp_with_affine(feat, affine, align_corners=False)
    _sy, sx = _peak(feat)
    _oy, ox = _peak(out)
    assert ox != sx or out.abs().sum() > 0


def test_plus_y_shift() -> None:
    feat = _impulse(1, 32, 64, 16, 32)
    affine = identity_affine(1, feat.device, feat.dtype).clone()
    affine[0, 1, 2] = -4.0 / 16.0
    out = warp_with_affine(feat, affine, align_corners=False)
    assert out.abs().sum() > 0


if __name__ == "__main__":
    test_identity_keeps_feature()
    test_airv2x_slice_identity()
    test_normalized_identity()
    test_plus_x_shift()
    test_plus_y_shift()
    print("negocollab_transform tests passed")

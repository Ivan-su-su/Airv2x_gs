"""Batched quaternion helpers for Gaussian rotation residuals.

Never assemble a covariance. All ops stay in quaternion / axis-angle form.
"""

from __future__ import annotations

import math

import torch

from opencood.models.gaussian_modules_0822.base.gaussian_transform import (
    normalize_quaternion,
)
from opencood.models.gaussian_modules_0822.base.utils import EPS


def axis_angle_to_quaternion(
    axis_angle: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """Convert axis-angle vectors to ``wxyz`` quaternions.

    Uses a sinc parameterization so the ``angle → 0`` limit is stable and
    fully batched.

    Args:
        axis_angle: ``[..., 3]``. The vector direction is the axis and
            ``||v||`` is the rotation angle in radians.
        eps: Numerical floor (unused by sinc; kept for API symmetry).

    Returns:
        Unit quaternions ``[..., 4]``.
    """
    del eps
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    half = 0.5 * angle
    real = torch.cos(half)
    # ``sinc(half / pi) = sin(half) / half`` with ``sinc(0) = 1``.
    imag = axis_angle * torch.sinc(half / math.pi) * 0.5
    quaternion = torch.cat([real, imag], dim=-1)
    return normalize_quaternion(quaternion)


def quaternion_multiply(q_left: torch.Tensor, q_right: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``q_left ⊗ q_right`` for ``wxyz`` quaternions.

    Interpreting quaternions as rotations, this corresponds to applying
    ``q_right`` first, then ``q_left``.

    Args:
        q_left: ``[..., 4]``.
        q_right: ``[..., 4]``.

    Returns:
        ``[..., 4]`` product, not yet renormalized.
    """
    w1, x1, y1, z1 = torch.unbind(q_left, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q_right, dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def compose_rotation_residual(
    delta_axis_angle: torch.Tensor, quaternion: torch.Tensor
) -> torch.Tensor:
    """Left-multiply a bounded axis-angle residual onto a quaternion.

    Args:
        delta_axis_angle: ``[N, 3]`` already-bounded residual.
        quaternion: ``[N, 4]`` current orientation.

    Returns:
        Normalized ``[N, 4]`` updated quaternion.
    """
    delta_quaternion = axis_angle_to_quaternion(delta_axis_angle)
    composed = quaternion_multiply(delta_quaternion, quaternion)
    return normalize_quaternion(composed)

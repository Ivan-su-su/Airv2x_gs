"""Rigid transform for scale-quaternion Gaussians.

A 3D Gaussian is stored as ``(mean, scale, quaternion)``. Under a rigid
``T = (R, t)``:

* ``mu_new = R @ mu + t``
* ``scale`` is invariant
* ``R_new = R @ R_gaussian``, then converted back to a quaternion

No covariance is assembled or decomposed here.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from opencood.models.gaussian_modules_0822.base.utils import EPS


def normalize_quaternion(
    quaternion: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """Normalize ``[w, x, y, z]`` quaternions and flip to ``w >= 0``.

    Args:
        quaternion: ``[..., 4]``.
        eps: Floor for the L2 norm.

    Returns:
        Unit quaternions with the same shape.
    """
    quaternion = quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True).clamp_min(
        eps
    )
    sign = torch.where(
        quaternion[..., :1] < 0.0,
        quaternion.new_tensor(-1.0),
        quaternion.new_tensor(1.0),
    )
    return quaternion * sign


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert ``[w, x, y, z]`` quaternions to ``[..., 3, 3]`` rotations.

    Args:
        quaternion: ``[..., 4]``.

    Returns:
        Rotation matrices ``[..., 3, 3]``.
    """
    quaternion = normalize_quaternion(quaternion)
    w, x, y, z = (
        quaternion[..., 0],
        quaternion[..., 1],
        quaternion[..., 2],
        quaternion[..., 3],
    )
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    row0 = torch.stack(
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1
    )
    row1 = torch.stack(
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1
    )
    row2 = torch.stack(
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1
    )
    return torch.stack([row0, row1, row2], dim=-2)


def rotation_matrix_to_quaternion(
    rotation: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """Convert ``[N, 3, 3]`` rotations to ``[N, 4]`` ``[w, x, y, z]`` quaternions.

    Fully batched Shepperd conversion. No Python loop over ``N``.

    Args:
        rotation: ``[N, 3, 3]``.
        eps: Numerical floor.

    Returns:
        Unit quaternions ``[N, 4]``.
    """
    if rotation.numel() == 0:
        return rotation.new_empty((0, 4))

    r00 = rotation[:, 0, 0]
    r01 = rotation[:, 0, 1]
    r02 = rotation[:, 0, 2]
    r10 = rotation[:, 1, 0]
    r11 = rotation[:, 1, 1]
    r12 = rotation[:, 1, 2]
    r20 = rotation[:, 2, 0]
    r21 = rotation[:, 2, 1]
    r22 = rotation[:, 2, 2]

    t0 = 1.0 + r00 + r11 + r22
    t1 = 1.0 + r00 - r11 - r22
    t2 = 1.0 - r00 + r11 - r22
    t3 = 1.0 - r00 - r11 + r22
    traces = torch.stack([t0, t1, t2, t3], dim=-1)
    case = traces.argmax(dim=-1)

    q0 = torch.stack([t0, r21 - r12, r02 - r20, r10 - r01], dim=-1)
    q1 = torch.stack([r21 - r12, t1, r01 + r10, r02 + r20], dim=-1)
    q2 = torch.stack([r02 - r20, r01 + r10, t2, r12 + r21], dim=-1)
    q3 = torch.stack([r10 - r01, r02 + r20, r12 + r21, t3], dim=-1)
    candidates = torch.stack([q0, q1, q2, q3], dim=1)

    batch = int(rotation.shape[0])
    index = torch.arange(batch, device=rotation.device)
    selected = candidates[index, case]
    selected_t = traces[index, case].clamp_min(eps)
    quaternion = selected / (2.0 * selected_t.sqrt().unsqueeze(-1))
    return normalize_quaternion(quaternion, eps=eps)


def reconstruct_covariance(
    scale: torch.Tensor, quaternion: torch.Tensor
) -> torch.Tensor:
    """Rebuild ``Σ = R diag(s²) Rᵀ``.

    Args:
        scale: ``[N, 3]`` axis standard deviations.
        quaternion: ``[N, 4]``.

    Returns:
        ``[N, 3, 3]`` covariances.
    """
    if scale.numel() == 0:
        return scale.new_empty((0, 3, 3))
    rotation = quaternion_to_rotation_matrix(quaternion)
    diag = torch.diag_embed(scale.pow(2))
    covariance = torch.matmul(rotation, torch.matmul(diag, rotation.transpose(-1, -2)))
    return 0.5 * (covariance + covariance.transpose(-1, -2))


def _broadcast_transform(
    transform_4x4: torch.Tensor,
    n_gaussians: int,
    view_index: Optional[torch.Tensor],
) -> torch.Tensor:
    """Expand a rigid transform to ``[N, 4, 4]``.

    Args:
        transform_4x4: ``[4, 4]``, ``[N, 4, 4]``, or per-camera ``[n_cam, 4, 4]``.
        n_gaussians: Target ``N``.
        view_index: Camera indices used to gather a per-camera stack.

    Returns:
        ``[N, 4, 4]``.
    """
    if transform_4x4.dim() == 2:
        return transform_4x4.unsqueeze(0).expand(n_gaussians, -1, -1)
    if int(transform_4x4.shape[0]) == n_gaussians:
        return transform_4x4
    if view_index is None:
        raise ValueError(
            f"transform {tuple(transform_4x4.shape)} is not [N,4,4] and "
            "view_index is missing"
        )
    return transform_4x4[view_index.long()]


def transform_gaussians(
    mean: torch.Tensor,
    scale: torch.Tensor,
    quaternion: torch.Tensor,
    transform_4x4: torch.Tensor,
    view_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a rigid transform to scale-quaternion Gaussians.

    Args:
        mean: ``[N, 3]`` in the source frame.
        scale: ``[N, 3]``. Unchanged by a rigid motion.
        quaternion: ``[N, 4]`` source-frame orientation.
        transform_4x4: Source→destination ``[4, 4]``, ``[N, 4, 4]``, or
            per-camera ``[n_cam, 4, 4]``.
        view_index: Gather index when ``transform_4x4`` is per-camera.

    Returns:
        ``(mean_dst, scale, quaternion_dst)``.
    """
    if mean.numel() == 0:
        return mean, scale, quaternion

    transform = _broadcast_transform(
        transform_4x4.to(device=mean.device, dtype=mean.dtype),
        int(mean.shape[0]),
        view_index,
    )
    rotation = transform[:, :3, :3]
    translation = transform[:, :3, 3]
    mean_dst = torch.matmul(rotation, mean.unsqueeze(-1)).squeeze(-1) + translation

    rotation_g = quaternion_to_rotation_matrix(quaternion)
    rotation_new = torch.matmul(rotation, rotation_g)
    quaternion_dst = rotation_matrix_to_quaternion(rotation_new)
    return mean_dst, scale, quaternion_dst

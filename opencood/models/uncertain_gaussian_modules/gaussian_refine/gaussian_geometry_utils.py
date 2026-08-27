"""Utilities for 3D Gaussian covariance in scale-quaternion form."""

from typing import Dict

import torch
import torch.nn.functional as F


def symmetrize_covariance(covariance: torch.Tensor) -> torch.Tensor:
    """Return a symmetric PSD-friendly covariance tensor."""
    return 0.5 * (covariance + covariance.transpose(-1, -2))


def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Normalize quaternions and canonicalize them to non-negative w."""
    quaternion = F.normalize(quaternion, p=2, dim=-1)
    sign = torch.where(
        quaternion[..., :1] < 0.0,
        quaternion.new_tensor(-1.0),
        quaternion.new_tensor(1.0),
    )
    return quaternion * sign


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication for `[w, x, y, z]` tensors."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized `[w, x, y, z]` quaternions to rotation matrices."""
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
    return torch.stack(
        [
            torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1),
            torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1),
            torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )


def rotation_matrix_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to `[w, x, y, z]` quaternions."""
    if rotation.numel() == 0:
        return rotation.new_empty((0, 4))

    batch = int(rotation.shape[0])
    quaternion = rotation.new_zeros((batch, 4))
    trace = rotation[:, 0, 0] + rotation[:, 1, 1] + rotation[:, 2, 2]

    positive_trace = trace > 0.0
    if positive_trace.any():
        t = torch.sqrt(trace[positive_trace] + 1.0) * 2.0
        quaternion[positive_trace, 0] = 0.25 * t
        quaternion[positive_trace, 1] = (
            rotation[positive_trace, 2, 1] - rotation[positive_trace, 1, 2]
        ) / t
        quaternion[positive_trace, 2] = (
            rotation[positive_trace, 0, 2] - rotation[positive_trace, 2, 0]
        ) / t
        quaternion[positive_trace, 3] = (
            rotation[positive_trace, 1, 0] - rotation[positive_trace, 0, 1]
        ) / t

    mask_x = (~positive_trace) & (rotation[:, 0, 0] > rotation[:, 1, 1]) & (
        rotation[:, 0, 0] > rotation[:, 2, 2]
    )
    if mask_x.any():
        t = torch.sqrt(
            1.0 + rotation[mask_x, 0, 0] - rotation[mask_x, 1, 1] - rotation[mask_x, 2, 2]
        ) * 2.0
        quaternion[mask_x, 0] = (rotation[mask_x, 2, 1] - rotation[mask_x, 1, 2]) / t
        quaternion[mask_x, 1] = 0.25 * t
        quaternion[mask_x, 2] = (rotation[mask_x, 0, 1] + rotation[mask_x, 1, 0]) / t
        quaternion[mask_x, 3] = (rotation[mask_x, 0, 2] + rotation[mask_x, 2, 0]) / t

    mask_y = (~positive_trace) & (~mask_x) & (rotation[:, 1, 1] > rotation[:, 2, 2])
    if mask_y.any():
        t = torch.sqrt(
            1.0 + rotation[mask_y, 1, 1] - rotation[mask_y, 0, 0] - rotation[mask_y, 2, 2]
        ) * 2.0
        quaternion[mask_y, 0] = (rotation[mask_y, 0, 2] - rotation[mask_y, 2, 0]) / t
        quaternion[mask_y, 1] = (rotation[mask_y, 0, 1] + rotation[mask_y, 1, 0]) / t
        quaternion[mask_y, 2] = 0.25 * t
        quaternion[mask_y, 3] = (rotation[mask_y, 1, 2] + rotation[mask_y, 2, 1]) / t

    mask_z = (~positive_trace) & (~mask_x) & (~mask_y)
    if mask_z.any():
        t = torch.sqrt(
            1.0 + rotation[mask_z, 2, 2] - rotation[mask_z, 0, 0] - rotation[mask_z, 1, 1]
        ) * 2.0
        quaternion[mask_z, 0] = (rotation[mask_z, 1, 0] - rotation[mask_z, 0, 1]) / t
        quaternion[mask_z, 1] = (rotation[mask_z, 0, 2] + rotation[mask_z, 2, 0]) / t
        quaternion[mask_z, 2] = (rotation[mask_z, 1, 2] + rotation[mask_z, 2, 1]) / t
        quaternion[mask_z, 3] = 0.25 * t

    return normalize_quaternion(quaternion)


def decompose_covariance_to_scale_rotation(
    covariance: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Factorize covariance to axis scales and a quaternion rotation state."""
    cov = symmetrize_covariance(covariance)
    eigen_values, eigen_vectors = torch.linalg.eigh(cov)
    eigen_values = eigen_values.clamp_min(eps)
    axis_scales = eigen_values.sqrt()
    det = torch.det(eigen_vectors)
    eigen_vectors[det < 0.0, :, -1] *= -1.0
    rotation = rotation_matrix_to_quaternion(eigen_vectors)
    return {
        "axis_scales": axis_scales,
        "rotation": rotation,
    }


def reconstruct_covariance_from_scale_rotation(
    axis_scales: torch.Tensor,
    rotation_matrix: torch.Tensor,
) -> torch.Tensor:
    """Build `Σ = R diag(s²) R^T` from axis scales and rotation matrix."""
    s2 = axis_scales.pow(2)
    diag = torch.diag_embed(s2)
    return rotation_matrix @ diag @ rotation_matrix.transpose(-1, -2)


def reconstruct_covariance_from_scale_quaternion(
    axis_scales: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Build covariance from axis scales and quaternion rotation."""
    rotation_matrix = quaternion_to_rotation_matrix(rotation)
    return reconstruct_covariance_from_scale_rotation(axis_scales, rotation_matrix)

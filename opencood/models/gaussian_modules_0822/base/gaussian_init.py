"""Convert the ray-tangent covariance into a scale-quaternion Gaussian.

Construction (once, at initialization):

    Sigma = sigma_depth^2 r r^T + sigma_t1^2 t1 t1^T + sigma_t2^2 t2 t2^T

Then a single batched eigendecomposition:

    Sigma = R diag(lambda) R^T
    scale = sqrt(lambda)
    quaternion = rotation_matrix_to_quaternion(R)

Forward interaction must reuse ``(mean, scale, quaternion, feature)`` and
must not call this module again.
"""

from __future__ import annotations

from typing import Tuple

import torch

from opencood.models.gaussian_modules_0822.base.covariance import assemble_covariance
from opencood.models.gaussian_modules_0822.base.gaussian_transform import (
    rotation_matrix_to_quaternion,
)
from opencood.models.gaussian_modules_0822.base.utils import EPS


def covariance_to_scale_quaternion(
    covariance: torch.Tensor, eps: float = EPS
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eigendecompose ``Sigma`` once into axis scales and a quaternion.

    Args:
        covariance: ``[N, 3, 3]`` SPD matrices.
        eps: Floor for eigenvalues.

    Returns:
        ``(scale [N, 3], quaternion [N, 4])``. Eigenvalues are returned in
        ascending order from ``torch.linalg.eigh``.
    """
    if covariance.numel() == 0:
        return covariance.new_empty((0, 3)), covariance.new_empty((0, 4))

    symmetric = 0.5 * (covariance + covariance.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    eigenvalues = eigenvalues.clamp_min(eps)
    # Columns of ``eigenvectors`` are the principal axes. Flip the last
    # axis when det < 0 so ``R`` is a proper rotation, not a reflection.
    det = torch.det(eigenvectors)
    flip = torch.where(det < 0.0, -1.0, 1.0).to(dtype=eigenvectors.dtype)
    last_axis = eigenvectors[:, :, 2] * flip.unsqueeze(-1)
    rotation = torch.stack(
        [eigenvectors[:, :, 0], eigenvectors[:, :, 1], last_axis], dim=-1
    )
    scale = torch.sqrt(eigenvalues)
    quaternion = rotation_matrix_to_quaternion(rotation, eps=eps)
    return scale, quaternion


def init_gaussian_from_axes(
    unit_ray: torch.Tensor,
    tangent_major: torch.Tensor,
    tangent_minor: torch.Tensor,
    sigma_z_sq: torch.Tensor,
    sigma_t1: float = 1.0,
    sigma_t2: float = 0.5,
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assemble the ray-tangent ``Sigma``, then factor it once.

    Args:
        unit_ray: ``[N, 3]``.
        tangent_major: ``[N, 3]``.
        tangent_minor: ``[N, 3]``.
        sigma_z_sq: ``[N]`` ray variance (meters^2).
        sigma_t1: Major tangent scale in meters.
        sigma_t2: Minor tangent scale in meters.
        eps: Numerical floor.

    Returns:
        ``(scale [N, 3], quaternion [N, 4])``.
    """
    covariance = assemble_covariance(
        unit_ray=unit_ray,
        tangent_major=tangent_major,
        tangent_minor=tangent_minor,
        sigma_z_sq=sigma_z_sq,
        sigma_t1=sigma_t1,
        sigma_t2=sigma_t2,
        eps=eps,
    )
    return covariance_to_scale_quaternion(covariance, eps=eps)

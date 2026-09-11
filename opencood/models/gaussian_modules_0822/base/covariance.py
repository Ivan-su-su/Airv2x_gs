"""Ray + tangent covariance assembly.

Final covariance is ``Sigma_ray + Sigma_tangent`` with a symmetric PSD
repair. Scales are metric (meters) and not learned.
"""

from __future__ import annotations

from typing import Tuple

import torch

from opencood.models.gaussian_modules_0822.base.utils import EPS


def ensure_psd_symmetric(
    covariance: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """Symmetrize and add ``eps I`` so eigenvalues stay non-negative.

    Args:
        covariance: ``[..., 3, 3]``.
        eps: Diagonal jitter.

    Returns:
        Symmetric PSD tensor of the same shape.
    """
    symmetric = 0.5 * (covariance + covariance.transpose(-1, -2))
    eye = torch.eye(3, device=covariance.device, dtype=covariance.dtype)
    return symmetric + eps * eye


def assemble_covariance(
    unit_ray: torch.Tensor,
    tangent_major: torch.Tensor,
    tangent_minor: torch.Tensor,
    sigma_z_sq: torch.Tensor,
    sigma_t1: float = 1.0,
    sigma_t2: float = 0.5,
    eps: float = EPS,
) -> torch.Tensor:
    """Build ``Sigma = sigma_z^2 rr^T + sigma_t1^2 t1 t1^T + sigma_t2^2 t2 t2^T``.

    Args:
        unit_ray: ``[N, 3]`` unit viewing rays in the target frame.
        tangent_major: ``[N, 3]`` unit tangent along heatmap PCA.
        tangent_minor: ``[N, 3]`` unit tangent completing ``(r, t1, t2)``.
        sigma_z_sq: ``[N]`` ray variance (meters^2). For drone this is
            ``sigma_D^2``.
        sigma_t1: Fixed major tangent scale in meters.
        sigma_t2: Fixed minor tangent scale in meters.
        eps: PSD jitter.

    Returns:
        ``[N, 3, 3]`` covariances.
    """
    if unit_ray.numel() == 0:
        return unit_ray.new_empty((0, 3, 3))
    ray = unit_ray.unsqueeze(-1)
    t1 = tangent_major.unsqueeze(-1)
    t2 = tangent_minor.unsqueeze(-1)
    var = sigma_z_sq.reshape(-1, 1, 1)
    sigma_ray = var * torch.matmul(ray, ray.transpose(-1, -2))
    sigma_tan = (sigma_t1 ** 2) * torch.matmul(t1, t1.transpose(-1, -2)) + (
        sigma_t2 ** 2
    ) * torch.matmul(t2, t2.transpose(-1, -2))
    return ensure_psd_symmetric(sigma_ray + sigma_tan, eps=eps)


def transform_covariance(
    covariance: torch.Tensor, rotation: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """Push covariances through a rigid rotation: ``R Σ Rᵀ``.

    Args:
        covariance: ``[N, 3, 3]`` in the source frame.
        rotation: ``[N, 3, 3]`` or ``[3, 3]``.
        eps: PSD jitter after the product.

    Returns:
        ``[N, 3, 3]`` in the destination frame.
    """
    if covariance.numel() == 0:
        return covariance.new_empty((0, 3, 3))
    if rotation.dim() == 2:
        rotation = rotation.unsqueeze(0).expand(covariance.shape[0], -1, -1)
    rotated = torch.matmul(
        rotation, torch.matmul(covariance, rotation.transpose(-1, -2))
    )
    return ensure_psd_symmetric(rotated, eps=eps)


def transform_gaussians(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    transform_4x4: torch.Tensor,
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a rigid ``[N,4,4]`` (or broadcast ``[4,4]``) to means and covariances.

    Args:
        mean: ``[N, 3]``.
        covariance: ``[N, 3, 3]``.
        transform_4x4: ``[N, 4, 4]`` or ``[4, 4]`` source→destination.
        eps: PSD jitter.

    Returns:
        ``(mean_dst, covariance_dst)``.
    """
    if mean.numel() == 0:
        return mean, covariance
    if transform_4x4.dim() == 2:
        transform_4x4 = transform_4x4.unsqueeze(0).expand(mean.shape[0], -1, -1)
    rotation = transform_4x4[:, :3, :3]
    translation = transform_4x4[:, :3, 3]
    mean_dst = torch.matmul(rotation, mean.unsqueeze(-1)).squeeze(-1) + translation
    cov_dst = transform_covariance(covariance, rotation, eps=eps)
    return mean_dst, cov_dst

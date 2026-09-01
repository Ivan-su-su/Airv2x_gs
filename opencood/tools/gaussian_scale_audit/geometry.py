# -*- coding: utf-8 -*-
"""Lift heatmap seeds to ego-frame Gaussians. Diagnostic only.

Active conventions audited from production code:

* R90 cell ``(i, j)`` image center ``(u, v) = (4*j+2, 4*i+2)``
  (``p1_layout.py``, ``lss/target.py``).
* Optical-axis z, lift ``X_cam = z * K^{-1} [u', v', 1]`` after undoing
  ``post_rots`` / ``post_trans`` (LSS ``get_geometry``,
  ``camera_optical_ray_range``). Not ``z * normalize(q)``.
* Vehicle / RSU: categorical ``p = softmax(depth_logits)``,
  ``mu_z = sum p_k z_k``, ``Sigma_depth = sum p_k (x_k-mu)(x_k-mu)^T``.
* Drone: ``z = camera_world_z + delta_pred``; ray sigma is a free
  diagnostic parameter.
* Collated extrinsics after ``ue4_to_lss`` are camera-to-lidar for
  vehicle / RSU / drone. Do not invert RSU.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from opencood.models.gaussian_modules_0822.p1_layout import BLOCK, FEAT_H, FEAT_W

LIDAR2CAM_STORED = {"vehicle": False, "rsu": False, "drone": False}


def r90_pixel_centers(
    feat_h: int = FEAT_H, feat_w: int = FEAT_W, block: int = BLOCK
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(u, v)`` maps of shape ``[feat_h, feat_w]``.

    ``u = block * j + block // 2``, ``v = block * i + block // 2``.
    """
    rows = np.arange(feat_h, dtype=np.float64)
    cols = np.arange(feat_w, dtype=np.float64)
    grid_i, grid_j = np.meshgrid(rows, cols, indexing="ij")
    u_map = block * grid_j + block / 2.0
    v_map = block * grid_i + block / 2.0
    return u_map, v_map


def _as_44(matrix: np.ndarray) -> np.ndarray:
    """Promote a 3x4 / 4x4 extrinsic to 4x4."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = arr
        return out
    raise ValueError(f"extrinsic must be 3x4 or 4x4, got {arr.shape}")


def cam_to_lidar_matrix(extrinsic: np.ndarray, agent_type: str) -> np.ndarray:
    """Camera-to-lidar 4x4. Stored matrices are already cam2lidar for all agents."""
    ext = _as_44(extrinsic)
    if LIDAR2CAM_STORED.get(agent_type, False):
        return np.linalg.inv(ext)
    return ext


def undo_post_pixels(
    u_aug: np.ndarray,
    v_aug: np.ndarray,
    post_rot: np.ndarray,
    post_trans: np.ndarray,
) -> np.ndarray:
    """Undo resize/crop: ``pix_orig = post_R^{-1} (pix_aug - post_t)``.

    Args:
        u_aug, v_aug: Augmented-image pixel coordinates, any broadcastable shape.
        post_rot: 3x3 or 2x2 post-homography.
        post_trans: length-2 or length-3 translation.

    Returns:
        ``[..., 3]`` homogeneous original pixels.
    """
    post_r = np.asarray(post_rot, dtype=np.float64)
    if post_r.shape == (2, 2):
        rot3 = np.eye(3, dtype=np.float64)
        rot3[:2, :2] = post_r
        post_r = rot3
    post_t = np.zeros(3, dtype=np.float64)
    t_in = np.asarray(post_trans, dtype=np.float64).reshape(-1)
    post_t[: t_in.size] = t_in[:3]
    u_f = np.asarray(u_aug, dtype=np.float64)
    v_f = np.asarray(v_aug, dtype=np.float64)
    ones = np.ones_like(u_f)
    pix = np.stack([u_f, v_f, ones], axis=-1)
    inv_r = np.linalg.inv(post_r)
    return (pix - post_t) @ inv_r.T


def ray_dir_cam(
    u_aug: np.ndarray,
    v_aug: np.ndarray,
    intrinsic: np.ndarray,
    post_rot: np.ndarray,
    post_trans: np.ndarray,
) -> np.ndarray:
    """Unnormalized camera ray ``q = K^{-1} [u', v', 1]``. Not unit-normalized."""
    pix_orig = undo_post_pixels(u_aug, v_aug, post_rot, post_trans)
    k_inv = np.linalg.inv(np.asarray(intrinsic, dtype=np.float64))
    return pix_orig @ k_inv.T


def lift_cam(q: np.ndarray, z: np.ndarray) -> np.ndarray:
    """``X_cam = z * q`` with optical-axis z. ``q`` and ``z`` broadcast."""
    z_b = np.asarray(z, dtype=np.float64)[..., None]
    return z_b * np.asarray(q, dtype=np.float64)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to ``[..., 3]`` points."""
    pts = np.asarray(points, dtype=np.float64)
    mat = _as_44(matrix)
    ones = np.ones(pts.shape[:-1] + (1,), dtype=np.float64)
    homo = np.concatenate([pts, ones], axis=-1)
    return (homo @ mat.T)[..., :3]


def rotation_of(matrix: np.ndarray) -> np.ndarray:
    """Upper-left 3x3 of a 4x4 transform."""
    return _as_44(matrix)[:3, :3]


def categorical_depth_covariance(
    q: np.ndarray,
    z_bins: np.ndarray,
    prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """GaussianLSS-style camera-frame depth covariance along one ray.

    Args:
        q: ``[3]`` unnormalized ray.
        z_bins: ``[D]`` optical-axis bin centers.
        prob: ``[D]`` softmax probabilities.

    Returns:
        ``(mu_cam, sigma_direct, sigma_varz_qqT, max_abs_diff)``.
    """
    q = np.asarray(q, dtype=np.float64).reshape(3)
    z_bins = np.asarray(z_bins, dtype=np.float64).reshape(-1)
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    x_k = z_bins[:, None] * q[None, :]
    mu = (prob[:, None] * x_k).sum(axis=0)
    delta = x_k - mu[None, :]
    sigma_direct = (prob[:, None, None] * delta[:, :, None] * delta[:, None, :]).sum(
        axis=0
    )
    z_mean = float((prob * z_bins).sum())
    z_var = float((prob * (z_bins - z_mean) ** 2).sum())
    sigma_eq = z_var * np.outer(q, q)
    max_diff = float(np.max(np.abs(sigma_direct - sigma_eq)))
    return mu, sigma_direct, sigma_eq, max_diff


def drone_ray_covariance(q: np.ndarray, sigma_z: float) -> np.ndarray:
    """Fixed optical-axis variance ``sigma_z^2 q q^T`` (diagnostic)."""
    q = np.asarray(q, dtype=np.float64).reshape(3)
    return float(sigma_z) ** 2 * np.outer(q, q)


def pixel_jacobian(
    z: float,
    intrinsic: np.ndarray,
    post_rot: np.ndarray,
    post_trans: np.ndarray,
) -> np.ndarray:
    """``d X_cam / d[u_aug, v_aug]``, shape ``[3, 2]``.

    ``X_cam = z * K^{-1} post_R^{-1} (pix_aug - post_t)``.
    ``z`` is treated as independent of the pixel (cell depth).
    """
    del post_trans
    post_r = np.asarray(post_rot, dtype=np.float64)
    if post_r.shape == (2, 2):
        rot3 = np.eye(3, dtype=np.float64)
        rot3[:2, :2] = post_r
        post_r = rot3
    k_inv = np.linalg.inv(np.asarray(intrinsic, dtype=np.float64))
    inv_r = np.linalg.inv(post_r)
    return float(z) * k_inv @ inv_r[:, :2]


def tangent_covariance_cam(
    z: float,
    intrinsic: np.ndarray,
    post_rot: np.ndarray,
    post_trans: np.ndarray,
    sigma_parallel_cells: float,
    sigma_perp_cells: float,
    theta: float,
    block: int = BLOCK,
) -> np.ndarray:
    """Lift an oriented 2D image covariance (R90-cell units) into camera 3D."""
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    scale = np.diag(
        [
            float(sigma_parallel_cells) ** 2,
            float(sigma_perp_cells) ** 2,
        ]
    )
    sigma_cells = rot @ scale @ rot.T
    sigma_px = (float(block) ** 2) * sigma_cells
    jac = pixel_jacobian(z, intrinsic, post_rot, post_trans)
    return jac @ sigma_px @ jac.T


def rotate_covariance(sigma: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """``R Sigma R^T``."""
    rot = np.asarray(rotation, dtype=np.float64)
    cov = np.asarray(sigma, dtype=np.float64)
    return rot @ cov @ rot.T


def assemble_sigma_ego(
    sigma_depth_cam: np.ndarray,
    sigma_tangent_cam: np.ndarray,
    rot_cam_to_ego: np.ndarray,
    eps: float,
) -> np.ndarray:
    """``R (Sigma_depth + Sigma_tangent + eps I) R^T`` in ego frame."""
    eye = np.eye(3, dtype=np.float64)
    sigma_cam = (
        np.asarray(sigma_depth_cam, dtype=np.float64)
        + np.asarray(sigma_tangent_cam, dtype=np.float64)
        + float(eps) * eye
    )
    return rotate_covariance(sigma_cam, rot_cam_to_ego)


def cam_to_ego_rt(
    extrinsic: np.ndarray, t_cav2ego: np.ndarray, agent_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Camera→ego 4x4 and its 3x3 rotation."""
    cam2lidar = cam_to_lidar_matrix(extrinsic, agent_type)
    cav2ego = _as_44(t_cav2ego)
    cam2ego = cav2ego @ cam2lidar
    return cam2ego, cam2ego[:3, :3]


def invert_spd(sigma: np.ndarray) -> np.ndarray:
    """Invert a batch of 3x3 covariances. ``sigma`` is ``[G, 3, 3]``."""
    return np.linalg.inv(np.asarray(sigma, dtype=np.float64))


def view_gaussians(
    seed_u: np.ndarray,
    seed_v: np.ndarray,
    z_cell: np.ndarray,
    theta: np.ndarray,
    anisotropy: np.ndarray,
    intrinsic: np.ndarray,
    post_rot: np.ndarray,
    post_trans: np.ndarray,
    extrinsic: np.ndarray,
    t_cav2ego: np.ndarray,
    agent_type: str,
    sigma0_list: Sequence[float],
    z_bins: Optional[np.ndarray],
    depth_prob: Optional[np.ndarray],
    drone_ray_sigma: float,
    eps: float,
    block: int = BLOCK,
) -> Tuple[np.ndarray, Dict[float, np.ndarray], float]:
    """Vectorized ego Gaussians for one camera view.

    Args:
        seed_u, seed_v: ``[P]`` augmented pixels.
        z_cell: ``[P]`` predicted optical-axis z.
        theta, anisotropy: ``[P]`` local heatmap orientation.
        depth_prob: ``[D, P]`` categorical probabilities, or None for drone.

    Returns:
        ``(mu_ego [P,3], sigmas {sigma0: [P,3,3]}, max_abs_varz_qqT_diff)``.
    """
    n_seed = int(np.asarray(seed_u).reshape(-1).size)
    if n_seed == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, {float(s): np.zeros((0, 3, 3)) for s in sigma0_list}, 0.0
    q = ray_dir_cam(seed_u, seed_v, intrinsic, post_rot, post_trans).reshape(n_seed, 3)
    z_cell = np.asarray(z_cell, dtype=np.float64).reshape(n_seed)
    max_diff = 0.0
    if z_bins is not None and depth_prob is not None:
        z_bins = np.asarray(z_bins, dtype=np.float64).reshape(-1)
        prob = np.asarray(depth_prob, dtype=np.float64).reshape(-1, n_seed)
        z_mean = (prob * z_bins[:, None]).sum(axis=0)
        z_var = (prob * (z_bins[:, None] - z_mean[None, :]) ** 2).sum(axis=0)
        mu_cam = z_mean[:, None] * q
        sigma_depth = z_var[:, None, None] * q[:, :, None] * q[:, None, :]
        # Spot-check the first seed against the direct sum formulation.
        _, sig_direct, _, diff = categorical_depth_covariance(
            q[0], z_bins, prob[:, 0]
        )
        max_diff = max(float(diff), float(np.max(np.abs(sigma_depth[0] - sig_direct))))
        z_use = z_mean
    else:
        mu_cam = lift_cam(q, z_cell)
        sigma_depth = drone_ray_covariance(q[0], drone_ray_sigma)[None].repeat(
            n_seed, axis=0
        )
        sigma_depth = (float(drone_ray_sigma) ** 2) * q[:, :, None] * q[:, None, :]
        z_use = z_cell
    cam2ego, rot_c2e = cam_to_ego_rt(extrinsic, t_cav2ego, agent_type)
    mu_ego = transform_points(mu_cam, cam2ego)
    jac0 = pixel_jacobian(1.0, intrinsic, post_rot, post_trans)
    jac = z_use[:, None, None] * jac0[None, :, :]
    theta = np.asarray(theta, dtype=np.float64).reshape(n_seed)
    aniso = np.asarray(anisotropy, dtype=np.float64).reshape(n_seed)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot2 = np.stack(
        [
            np.stack([cos_t, -sin_t], axis=1),
            np.stack([sin_t, cos_t], axis=1),
        ],
        axis=1,
    )
    sigmas: Dict[float, np.ndarray] = {}
    for sigma0 in sigma0_list:
        s_par = float(sigma0) * np.sqrt(np.clip(aniso, 1.0e-8, None))
        s_perp = float(sigma0) / np.sqrt(np.clip(aniso, 1.0e-8, None))
        scale = np.zeros((n_seed, 2, 2), dtype=np.float64)
        scale[:, 0, 0] = s_par ** 2
        scale[:, 1, 1] = s_perp ** 2
        sigma_cells = rot2 @ scale @ np.transpose(rot2, (0, 2, 1))
        sigma_px = (float(block) ** 2) * sigma_cells
        sigma_t = jac @ sigma_px @ np.transpose(jac, (0, 2, 1))
        sigma_cam = sigma_depth + sigma_t + float(eps) * np.eye(3)[None]
        sigmas[float(sigma0)] = (
            rot_c2e[None] @ sigma_cam @ rot_c2e.T[None]
        )
    return mu_ego, sigmas, max_diff


def audit_layout() -> Dict[str, object]:
    """Record the R90 / lift conventions used by this diagnostic."""
    u_map, v_map = r90_pixel_centers()
    return {
        "feat_hw": [FEAT_H, FEAT_W],
        "block": BLOCK,
        "cell_00_uv": [float(u_map[0, 0]), float(v_map[0, 0])],
        "cell_formula": "u=4*j+2, v=4*i+2",
        "lift": "X_cam = z * K^{-1} undo_post([u,v,1])",
        "not_used": "z * normalize(q)",
        "lidar2cam_stored": dict(LIDAR2CAM_STORED),
    }

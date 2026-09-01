# -*- coding: utf-8 -*-
"""GT box association, coverage, and spill. Diagnostic only."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from opencood.utils.box_utils import boxes_to_corners_3d

SEED_BUCKETS = ("0", "1", "2-3", "4-8", ">8")


def seed_bucket(n_seed: int) -> str:
    """Map a seed count to the report bucket label."""
    if n_seed <= 0:
        return "0"
    if n_seed == 1:
        return "1"
    if n_seed <= 3:
        return "2-3"
    if n_seed <= 8:
        return "4-8"
    return ">8"


def boxes_hwl_to_corners(boxes: np.ndarray) -> np.ndarray:
    """``[N, 7]`` hwl boxes → ``[N, 8, 3]`` corners in the same frame."""
    if boxes.size == 0:
        return np.zeros((0, 8, 3), dtype=np.float64)
    corners = boxes_to_corners_3d(np.asarray(boxes, dtype=np.float64), "hwl")
    if hasattr(corners, "numpy"):
        corners = corners.numpy()
    return np.asarray(corners, dtype=np.float64)


def box_frame_axes(box: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center, rotation (x-forward), and half-extents ``(l/2, w/2, h/2)``.

    OpenCOOD ``hwl`` boxes are ``[x, y, z, h, w, l, yaw]``.
    """
    x, y, z, height, width, length, yaw = [float(v) for v in np.asarray(box).reshape(7)]
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot = np.array(
        [[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    center = np.array([x, y, z], dtype=np.float64)
    half = np.array([length / 2.0, width / 2.0, height / 2.0], dtype=np.float64)
    return center, rot, half


def points_in_boxes(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Boolean ``[M, N]``: point m inside OBB n."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    n_pts, n_box = pts.shape[0], boxes.shape[0]
    inside = np.zeros((n_pts, n_box), dtype=bool)
    for box_i in range(n_box):
        center, rot, half = box_frame_axes(boxes[box_i])
        local = (pts - center[None, :]) @ rot
        inside[:, box_i] = np.all(np.abs(local) <= half[None, :] + 1.0e-6, axis=1)
    return inside


def sample_box_points(box: np.ndarray, res: int) -> np.ndarray:
    """Regular grid inside one OBB. ``res`` samples along each axis."""
    n = max(int(res), 2)
    center, rot, half = box_frame_axes(box)
    lin = [np.linspace(-h, h, n) for h in half]
    grid = np.stack(np.meshgrid(*lin, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid @ rot.T + center[None, :]


def sample_expanded_region(
    box: np.ndarray, res: int, expand: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Grid in an expanded AABB around the OBB, plus inside-OBB mask."""
    corners = boxes_hwl_to_corners(np.asarray(box).reshape(1, 7))[0]
    lo = corners.min(axis=0)
    hi = corners.max(axis=0)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * float(expand)
    lo_e, hi_e = center - half, center + half
    n = max(int(res), 2)
    axes = [np.linspace(lo_e[d], hi_e[d], n) for d in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = points_in_boxes(pts, np.asarray(box).reshape(1, 7))[:, 0]
    return pts, inside


def mahalanobis2(
    points: np.ndarray, means: np.ndarray, sigma_inv: np.ndarray
) -> np.ndarray:
    """``(x-mu)^T Sigma^{-1} (x-mu)`` for all pairs, shape ``[M, G]``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mu = np.asarray(means, dtype=np.float64).reshape(-1, 3)
    prec = np.asarray(sigma_inv, dtype=np.float64).reshape(-1, 3, 3)
    if pts.size == 0 or mu.size == 0:
        return np.zeros((pts.shape[0], mu.shape[0]), dtype=np.float64)
    delta = pts[:, None, :] - mu[None, :, :]
    return np.einsum("mgi,gij,mgj->mg", delta, prec, delta)


def coverage_fraction(
    points: np.ndarray,
    means: np.ndarray,
    sigma_inv: np.ndarray,
    k: float,
) -> float:
    """Fraction of points with ``min_i d_i^2 <= k^2``."""
    if points.size == 0:
        return float("nan")
    if means.size == 0:
        return 0.0
    dist = mahalanobis2(points, means, sigma_inv)
    covered = dist.min(axis=1) <= float(k) ** 2
    return float(covered.mean())


def support_precision(
    points: np.ndarray,
    inside: np.ndarray,
    means: np.ndarray,
    sigma_inv: np.ndarray,
    k: float,
) -> Tuple[float, int, int]:
    """Precision of Mahalanobis support on a local sampled region.

    Returns:
        ``(precision, n_covered_inside, n_covered_total)``.
    """
    if points.size == 0 or means.size == 0:
        return float("nan"), 0, 0
    dist = mahalanobis2(points, means, sigma_inv)
    covered = dist.min(axis=1) <= float(k) ** 2
    n_total = int(covered.sum())
    n_in = int((covered & inside).sum())
    if n_total == 0:
        return float("nan"), 0, 0
    return float(n_in / n_total), n_in, n_total


def nearest_box_index(
    points: np.ndarray, boxes: np.ndarray, inside: Optional[np.ndarray] = None
) -> np.ndarray:
    """Assign each point to the nearest box center; prefer containing boxes."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    n_pts = pts.shape[0]
    if n_pts == 0 or boxes.shape[0] == 0:
        return np.full(n_pts, -1, dtype=np.int64)
    centers = boxes[:, :3]
    dist = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
    if inside is None:
        inside = points_in_boxes(pts, boxes)
    assigned = np.full(n_pts, -1, dtype=np.int64)
    for i in range(n_pts):
        hit = np.flatnonzero(inside[i])
        if hit.size:
            assigned[i] = int(hit[np.argmin(dist[i, hit])])
        else:
            assigned[i] = int(np.argmin(dist[i]))
    return assigned


def bucket_histogram(counts: Sequence[int]) -> Dict[str, int]:
    """Count objects in each seed-count bucket."""
    hist = {key: 0 for key in SEED_BUCKETS}
    for n_seed in counts:
        hist[seed_bucket(int(n_seed))] += 1
    return hist

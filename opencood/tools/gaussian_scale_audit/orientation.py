# -*- coding: utf-8 -*-
"""Local heatmap orientation / bounded anisotropy. Diagnostic only."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def local_orientation(
    p_fg: np.ndarray,
    seed_i: np.ndarray,
    seed_j: np.ndarray,
    window: int = 7,
    anisotropy_max: float = 4.0,
    eps: float = 1.0e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PCA of a local p_fg window around each R90 seed.

    Positions are in R90-cell coordinates ``(j, i)`` matching ``(u, v)``.
    Anisotropy ``a = clip(lambda1 / lambda2, 1, anisotropy_max)``.
    Does not decide whether a seed exists.

    Args:
        p_fg: ``[H, W]`` foreground probability.
        seed_i, seed_j: integer R90 indices, shape ``[P]``.
        window: Odd local window size in R90 cells.
        anisotropy_max: Upper clamp on ``a``.
        eps: Numerical floor.

    Returns:
        ``theta, anisotropy, lambda1, lambda2`` each ``[P]``.
    """
    if int(window) < 1 or int(window) % 2 == 0:
        raise ValueError(f"window must be a positive odd int, got {window}")
    heat = np.asarray(p_fg, dtype=np.float64)
    ys = np.asarray(seed_i, dtype=np.int64).reshape(-1)
    xs = np.asarray(seed_j, dtype=np.int64).reshape(-1)
    n_seed = int(ys.size)
    theta = np.zeros(n_seed, dtype=np.float64)
    aniso = np.ones(n_seed, dtype=np.float64)
    lam1 = np.ones(n_seed, dtype=np.float64)
    lam2 = np.ones(n_seed, dtype=np.float64)
    if n_seed == 0:
        return theta, aniso, lam1, lam2
    pad = int(window) // 2
    padded = np.pad(heat, pad, mode="constant", constant_values=0.0)
    offsets = np.arange(-pad, pad + 1, dtype=np.float64)
    dj, di = np.meshgrid(offsets, offsets, indexing="xy")
    rel = np.stack([dj.reshape(-1), di.reshape(-1)], axis=1)
    for idx in range(n_seed):
        patch = padded[
            int(ys[idx]) : int(ys[idx]) + int(window),
            int(xs[idx]) : int(xs[idx]) + int(window),
        ].reshape(-1)
        weight = np.clip(patch, 0.0, None)
        w_sum = float(weight.sum())
        if w_sum <= eps:
            continue
        centroid = (weight[:, None] * rel).sum(axis=0) / w_sum
        delta = rel - centroid[None, :]
        cov = (weight[:, None, None] * delta[:, :, None] * delta[:, None, :]).sum(
            axis=0
        ) / w_sum
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = np.clip(evals[order], 0.0, None)
        evec = evecs[:, order[0]]
        lam1[idx] = float(evals[0])
        lam2[idx] = float(evals[1])
        theta[idx] = float(np.arctan2(evec[1], evec[0]))
        ratio = lam1[idx] / max(lam2[idx], eps)
        aniso[idx] = float(np.clip(ratio, 1.0, float(anisotropy_max)))
    return theta, aniso, lam1, lam2


def tangent_scales(
    sigma0: float, anisotropy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """``sigma_parallel = sigma0 sqrt(a)``, ``sigma_perp = sigma0 / sqrt(a)``."""
    a = np.asarray(anisotropy, dtype=np.float64)
    root = np.sqrt(np.clip(a, 1.0e-8, None))
    s0 = float(sigma0)
    return s0 * root, s0 / root

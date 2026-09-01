# -*- coding: utf-8 -*-
"""Diagnostic panels for Gaussian scale coverage. Not used in training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Polygon, Rectangle
from PIL import Image


def _ellipse_patch(
    mean_xy: np.ndarray, cov_xy: np.ndarray, n_sigma: float, **kwargs
) -> Optional[Ellipse]:
    """2D covariance ellipse in plot coordinates."""
    cov = np.asarray(cov_xy, dtype=np.float64)
    if not np.isfinite(cov).all():
        return None
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0.0, None)
    if float(evals.max()) <= 1.0e-12:
        return None
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    width = 2.0 * float(n_sigma) * float(np.sqrt(evals[0]))
    height = 2.0 * float(n_sigma) * float(np.sqrt(max(evals[1], 0.0)))
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    return Ellipse(
        (float(mean_xy[0]), float(mean_xy[1])),
        width=width,
        height=height,
        angle=angle,
        **kwargs,
    )


def overlay_box_polygons(
    ax: plt.Axes,
    rgb: np.ndarray,
    polygons: Sequence[np.ndarray],
) -> None:
    """RGB with projected GT polygons."""
    ax.imshow(rgb)
    for poly in polygons:
        if poly is None or len(poly) < 3:
            continue
        ax.add_patch(
            Polygon(poly, closed=True, fill=False, edgecolor="lime", linewidth=1.0)
        )
    ax.set_axis_off()


def save_seed_orientation_panel(
    path: Path,
    rgb: np.ndarray,
    polygons: Sequence[np.ndarray],
    p_fg: np.ndarray,
    seed_mask: np.ndarray,
    seed_uv: np.ndarray,
    theta: np.ndarray,
    title: str,
) -> None:
    """RGB | GT boxes | p_fg | binary seeds + orientation ticks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16.4, 3.6))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    overlay_box_polygons(axes[1], rgb, polygons)
    axes[1].set_title("GT projected boxes")
    im = axes[2].imshow(p_fg, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("heatmap p_fg")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    seed_img = (seed_mask > 0).astype(np.float32)
    axes[3].imshow(rgb)
    heat_up = np.array(
        Image.fromarray((seed_img * 255).astype(np.uint8)).resize(
            (rgb.shape[1], rgb.shape[0]), Image.NEAREST
        )
    )
    axes[3].imshow(heat_up, cmap="Reds", alpha=0.35, vmin=0, vmax=255)
    if seed_uv.size:
        axes[3].scatter(seed_uv[:, 0], seed_uv[:, 1], s=8, c="cyan", linewidths=0)
        length = 14.0
        for uv, ang in zip(seed_uv, theta):
            dx = length * np.cos(ang)
            dy = length * np.sin(ang)
            axes[3].plot(
                [uv[0] - dx, uv[0] + dx],
                [uv[1] - dy, uv[1] + dy],
                color="yellow",
                lw=0.8,
            )
    axes[3].set_title("seeds + orientation")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _bev_lim(means: np.ndarray, corners: np.ndarray, pad: float = 8.0) -> Tuple[float, float, float, float]:
    """Shared xy limits from GT corners and Gaussian centers."""
    pts = []
    if corners.size:
        pts.append(corners.reshape(-1, 3)[:, :2])
    if means.size:
        pts.append(means[:, :2])
    if not pts:
        return -40.0, 40.0, -40.0, 40.0
    xy = np.concatenate(pts, axis=0)
    lo = xy.min(axis=0) - pad
    hi = xy.max(axis=0) + pad
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def draw_box_bev(ax: plt.Axes, corners: np.ndarray, color: str = "black") -> None:
    """Draw each 3D box's BEV rectangle from 8 corners."""
    for box in corners.reshape(-1, 8, 3):
        xy = box[:, :2]
        # Convex hull order is not guaranteed; plot the bottom 4 by z then hull.
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(xy)
            poly = xy[hull.vertices]
        except Exception:
            poly = xy[:4]
        ax.add_patch(
            Polygon(poly, closed=True, fill=False, edgecolor=color, linewidth=1.1)
        )


def save_scale_sweep_view(
    path: Path,
    view: str,
    sigma0_list: Sequence[float],
    means: np.ndarray,
    sigmas: Dict[float, np.ndarray],
    corners: np.ndarray,
    title: str,
    max_ellipses: int = 400,
    rng: Optional[np.random.RandomState] = None,
) -> None:
    """Same-frame BEV or side view across ``sigma0`` values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_col = 3
    n_row = int(np.ceil(len(sigma0_list) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.2 * n_col, 4.0 * n_row))
    axes_f = np.atleast_1d(axes).ravel()
    axis_idx = (0, 1) if view == "bev" else (0, 2)
    lim_pts_mu = means[:, list(axis_idx)] if means.size else np.zeros((0, 2))
    lim_pts_gt = (
        corners.reshape(-1, 3)[:, list(axis_idx)] if corners.size else np.zeros((0, 2))
    )
    pts = (
        np.concatenate([lim_pts_mu, lim_pts_gt], axis=0)
        if lim_pts_mu.size or lim_pts_gt.size
        else np.array([[-40.0, -40.0], [40.0, 40.0]])
    )
    lo, hi = pts.min(axis=0) - 8.0, pts.max(axis=0) + 8.0
    n_g = int(means.shape[0]) if means.size else 0
    keep = np.arange(n_g)
    if n_g > max_ellipses:
        rng = rng or np.random.RandomState(0)
        keep = np.sort(rng.choice(n_g, size=max_ellipses, replace=False))
    for ax_i, sigma0 in enumerate(sigma0_list):
        ax = axes_f[ax_i]
        ax.set_aspect("equal")
        if corners.size:
            if view == "bev":
                draw_box_bev(ax, corners, color="black")
            else:
                for box in corners.reshape(-1, 8, 3):
                    xz = box[:, [0, 2]]
                    ax.add_patch(
                        Polygon(
                            xz[np.argsort(xz[:, 0])][:4]
                            if False
                            else xz,
                            closed=True,
                            fill=False,
                            edgecolor="black",
                            linewidth=0.8,
                        )
                    )
                    xs, zs = box[:, 0], box[:, 2]
                    ax.add_patch(
                        Rectangle(
                            (float(xs.min()), float(zs.min())),
                            float(xs.max() - xs.min()),
                            float(zs.max() - zs.min()),
                            fill=False,
                            edgecolor="black",
                            linewidth=0.9,
                        )
                    )
        covs = sigmas[float(sigma0)]
        mu_keep = means[keep] if n_g else means
        cov_keep = covs[keep] if n_g else covs
        for mu, cov in zip(mu_keep, cov_keep):
            mean_2d = mu[list(axis_idx)]
            cov_2d = cov[np.ix_(list(axis_idx), list(axis_idx))]
            e2 = _ellipse_patch(
                mean_2d,
                cov_2d,
                2.0,
                facecolor="tab:orange",
                edgecolor="tab:orange",
                alpha=0.12,
                linewidth=0.4,
            )
            e1 = _ellipse_patch(
                mean_2d,
                cov_2d,
                1.0,
                facecolor="none",
                edgecolor="tab:red",
                alpha=0.7,
                linewidth=0.6,
            )
            if e2 is not None:
                ax.add_patch(e2)
            if e1 is not None:
                ax.add_patch(e1)
        if n_g:
            ax.scatter(
                mu_keep[:, axis_idx[0]],
                mu_keep[:, axis_idx[1]],
                s=4,
                c="deepskyblue",
                zorder=3,
            )
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[1]), float(hi[1]))
        xlab, ylab = ("x (m)", "y (m)") if view == "bev" else ("x (m)", "z (m)")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"sigma0={sigma0:g} R90 cells")
        ax.grid(alpha=0.2)
    for ax in axes_f[len(sigma0_list) :]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_object_closeup(
    path: Path,
    box: np.ndarray,
    means: np.ndarray,
    sigmas: Dict[float, np.ndarray],
    sigma0_list: Sequence[float],
    title: str,
) -> None:
    """Crop around one GT box; BEV ellipses for each sigma0."""
    path.parent.mkdir(parents=True, exist_ok=True)
    corners = box.reshape(8, 3)
    n_col = 3
    n_row = int(np.ceil(len(sigma0_list) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.0 * n_col, 3.8 * n_row))
    axes_f = np.atleast_1d(axes).ravel()
    xy = corners[:, :2]
    pad = 6.0
    lo, hi = xy.min(axis=0) - pad, xy.max(axis=0) + pad
    for ax_i, sigma0 in enumerate(sigma0_list):
        ax = axes_f[ax_i]
        ax.set_aspect("equal")
        draw_box_bev(ax, corners[None], color="black")
        covs = sigmas[float(sigma0)]
        for mu, cov in zip(means, covs):
            e2 = _ellipse_patch(
                mu[:2],
                cov[:2, :2],
                2.0,
                facecolor="tab:orange",
                edgecolor="tab:orange",
                alpha=0.18,
                linewidth=0.5,
            )
            e1 = _ellipse_patch(
                mu[:2],
                cov[:2, :2],
                1.0,
                facecolor="none",
                edgecolor="tab:red",
                alpha=0.8,
                linewidth=0.7,
            )
            if e2 is not None:
                ax.add_patch(e2)
            if e1 is not None:
                ax.add_patch(e1)
        if means.size:
            ax.scatter(means[:, 0], means[:, 1], s=10, c="deepskyblue", zorder=3)
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[1]), float(hi[1]))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"sigma0={sigma0:g}")
        ax.grid(alpha=0.2)
    for ax in axes_f[len(sigma0_list) :]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-frame Drone Gaussian-flow diagnosis for AirV2X timestamp_000035.

This script answers four questions without changing the model:

Q1. Is Drone foreground (FG) heatmap spatially shifted?
    - Saves per-view RGB + predicted p_fg overlay.
    - Projects ego GT 3D boxes into each Drone camera.
    - Measures p_fg / FG coverage inside each projected GT box.

Q2. Is Drone depth prediction causing the 3D shift?
    For EXACTLY THE SAME predicted-FG seed cells:
        predicted FG cell + predicted optical-z
        predicted FG cell + CARLA GT ray-range converted to optical-z
    are backprojected with the repository's own backproject_optical_z().
    Both are then transformed with the SAME T_drone->ego.
    Their 3D/BEV difference therefore isolates depth error.

Q3. Where does Gaussian geometry first move?
    Captures:
        Drone init (after predicted-depth backprojection + agent->ego + range filter)
        Drone Stage1
        Drone Stage2 / cross-agent fusion
        Drone Stage3 voxel-pooled geometry
    plus Vehicle/RSU Stage2 and pooled geometry when present.

Q4. Is the system-level bias already present in the Gaussian evidence?
    Per GT, reports nearest Gaussian and p_fg-weighted local center error for:
        raw predicted-depth seed points
        raw GT-depth seed points
        Drone init
        Drone Stage1
        Drone Stage2
        Drone pooled
        Vehicle Stage2 / pooled
        RSU Stage2 / pooled

Repository contracts used here
------------------------------
- Dataset camera tensor stores RGB + transformed CARLA depth range:
      imgs[..., 0:3, H, W] = ImageNet-normalized RGB
      imgs[..., 3,   H, W] = CARLA camera ray-range in meters
  when depth is available. Before calling backproject_optical_z(), this
  script converts CARLA ray-range rho to camera optical-axis z by:
      z = rho / || K^{-1} [u_orig, v_orig, 1]^T ||
  after undoing post_rot / post_trans.
- Drone P1:
      delta_pred = drone_delta_head(F90, height_embed)
      depth_z_mean = camera_world_z + delta_pred
- Heatmap:
      p_fg = softmax(heatmap_logits, dim=1)[:, 1]
      one seed for every p_fg > fg_threshold (default 0.3)
- Initializer:
      backproject_optical_z(... depth_z_mean ...)
      then GaussianSet.to_ego(T_agent->ego)
- Stage3 is feature-only after per-agent voxel pooling; pooling geometry is
  obtained by calling model.stage3.pool(stage2_gaussians[agent]).

No model weights, thresholds, tensors, or dataset files are modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.gaussian_modules_0822.base.projection import (
    CAMERA_GEOMETRY_KEY,
    backproject_optical_z,
    feature_indices_to_aug_pixels,
    project_lidar_to_image,
)
from opencood.models.gaussian_modules_0822.heatmap.seed_generator import (
    foreground_probability,
    generate_seeds,
)
from opencood.tools import train_utils


DEFAULT_CKPT = (
    "/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
    "airv2x_gaussian_0822_geom_sup/"
    "geom_sup_2026_09_18_00_56_21/net_epoch15.pth"
)
DEFAULT_TIMESTAMP = (
    "/mnt/home/suyi/AirV2X-Perception/test/"
    "2025_05_05_21_46_39/timestamp_000035"
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GT_DEPTH_CHANNEL = 3
GT_DEPTH_MIN_M = 0.1
GT_DEPTH_FAR_M = 999.0

# Explicit per-Drone colors for all BEV Gaussian-flow plots.
# Matplotlib default named colors are used so the same CAV keeps the same
# color across raw seeds / init / Stage1 / Stage2 / pooled plots.
DRONE_CAV_COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
)
MIXED_CAV_COLOR = "tab:gray"


# ---------------------------------------------------------------------
# CLI / loading
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--config", default="")
    p.add_argument("--timestamp-dir", default=DEFAULT_TIMESTAMP)
    p.add_argument("--test-root", default="")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument(
        "--fg-threshold",
        type=float,
        default=None,
        help="diagnostic seed threshold; default = model.initializer.fg_threshold",
    )
    p.add_argument(
        "--local-radius",
        type=float,
        default=4.0,
        help="XY radius around each GT used for local Gaussian center statistics",
    )
    p.add_argument(
        "--max-bev-points",
        type=int,
        default=25000,
        help="plot subsampling cap only; numerical diagnostics still use all points",
    )
    p.add_argument(
        "--out-dir",
        default="opencood/logs/diagnose_timestamp35_gaussian_flow",
    )
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_config(ckpt: Path, explicit: str) -> Path:
    p = Path(explicit) if explicit else ckpt.parent / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"config not found: {p}; pass --config explicitly"
        )
    return p


def unwrap_state_dict(blob):
    if not isinstance(blob, dict):
        raise TypeError(f"checkpoint must be dict-like, got {type(blob)}")
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        if isinstance(blob.get(key), dict):
            return blob[key]
    return blob


def load_checkpoint_exact(model, ckpt: Path, device):
    blob = torch.load(str(ckpt), map_location=device)
    state = dict(unwrap_state_dict(blob))
    if state and next(iter(state)).startswith("module."):
        state = {k[7:]: v for k, v in state.items()}

    current = model.state_dict()
    shape_bad = [
        k for k, v in state.items()
        if k in current and tuple(v.shape) != tuple(current[k].shape)
    ]
    if shape_bad:
        raise RuntimeError(
            f"checkpoint/model shape mismatch: {shape_bad[:20]}"
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch\n"
            f"missing({len(missing)}): {missing[:30]}\n"
            f"unexpected({len(unexpected)}): {unexpected[:30]}"
        )
    print(f"[ckpt] exact-compatible: {ckpt}")


def find_timestamp_index(dataset, timestamp_dir: Path) -> Tuple[int, str]:
    scene_name = timestamp_dir.parent.name
    timestamp_name = timestamp_dir.name

    if not hasattr(dataset, "_metadata_path_for_idx"):
        raise RuntimeError(
            "Current dataset has no _metadata_path_for_idx()."
        )

    hits = []
    for idx in range(len(dataset)):
        meta = str(dataset._metadata_path_for_idx(idx))
        parts = Path(meta).parts
        if scene_name in parts and timestamp_name in parts:
            hits.append((idx, meta))

    if not hits:
        raise RuntimeError(
            f"Cannot find {scene_name}/{timestamp_name} "
            f"in validation dataset of length {len(dataset)}"
        )
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one index for target timestamp, got {hits}"
        )
    print(f"[target] idx={hits[0][0]}")
    print(f"[target] metadata={hits[0][1]}")
    return hits[0]


def make_batch(dataset, idx: int):
    sample = dataset[idx]
    if sample is None:
        raise RuntimeError(f"dataset[{idx}] returned None")
    return dataset.collate_batch_test([sample])


# ---------------------------------------------------------------------
# Stage capture
# ---------------------------------------------------------------------

class GaussianFlowCapture:
    """Capture initializer, Stage1, Stage2, and Stage3 merged output."""

    def __init__(self, model):
        self.model = model
        self.init = None
        self.stage1 = None
        self.stage2 = None
        self.stage3_merged = None

        # GaussianInitializer is intentionally not nn.Module; wrap its
        # bound forward and take a shallow dict copy BEFORE model.forward
        # mutates the returned dict with Stage1 outputs.
        self._orig_initializer_forward = model.initializer.forward

        def wrapped_initializer_forward(ego_batch):
            out = self._orig_initializer_forward(ego_batch)
            self.init = dict(out)
            return out

        model.initializer.forward = wrapped_initializer_forward

        # cross_agent(stage1_gaussians, sources)
        self._h_cross_pre = model.cross_agent.register_forward_pre_hook(
            self._cross_pre
        )
        # stage3(stage2_gaussians)
        self._h_stage3_pre = model.stage3.register_forward_pre_hook(
            self._stage3_pre
        )
        self._h_stage3_post = model.stage3.register_forward_hook(
            self._stage3_post
        )

    def _cross_pre(self, module, inputs):
        if inputs and isinstance(inputs[0], Mapping):
            self.stage1 = dict(inputs[0])

    def _stage3_pre(self, module, inputs):
        if inputs and isinstance(inputs[0], Mapping):
            self.stage2 = dict(inputs[0])

    def _stage3_post(self, module, inputs, output):
        if isinstance(output, (tuple, list)) and len(output) >= 1:
            self.stage3_merged = output[0]

    def close(self):
        self.model.initializer.forward = self._orig_initializer_forward
        self._h_cross_pre.remove()
        self._h_stage3_pre.remove()
        self._h_stage3_post.remove()


# ---------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------

def flatten_images(imgs: torch.Tensor) -> torch.Tensor:
    if imgs.dim() == 5:
        b, v, c, h, w = imgs.shape
        return imgs.reshape(b * v, c, h, w)
    if imgs.dim() == 4:
        return imgs
    raise ValueError(f"imgs must be 4D/5D, got {tuple(imgs.shape)}")


def flatten_semantic(x: torch.Tensor, n_flat: int) -> torch.Tensor:
    if x.dim() == 4 and int(x.shape[0]) * int(x.shape[1]) == n_flat:
        return x.reshape(n_flat, *x.shape[-2:])
    if x.dim() == 3 and int(x.shape[0]) == n_flat:
        return x
    raise ValueError(
        f"semantic GT {tuple(x.shape)} incompatible with n_flat={n_flat}"
    )


def denormalize_rgb(rgb: torch.Tensor) -> np.ndarray:
    x = rgb.detach().float().cpu().clone()
    mean = x.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = x.new_tensor(IMAGENET_STD)[:, None, None]
    x = (x * std + mean).clamp(0.0, 1.0)
    return x.permute(1, 2, 0).numpy()


def finite_stats(values) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "p50": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def vec3_stats(x: torch.Tensor) -> Dict[str, Any]:
    if x.numel() == 0:
        return {"n": 0}
    a = x.detach().float().cpu().numpy()
    norm = np.linalg.norm(a, axis=1)
    return {
        "n": int(a.shape[0]),
        "dx": finite_stats(a[:, 0]),
        "dy": finite_stats(a[:, 1]),
        "dz": finite_stats(a[:, 2]),
        "norm": finite_stats(norm),
    }


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def apply_agent_to_ego(
    points_agent: torch.Tensor,
    agent_to_ego: torch.Tensor,
    view_index: torch.Tensor,
) -> torch.Tensor:
    if points_agent.numel() == 0:
        return points_agent
    t = agent_to_ego[view_index.long()].to(
        device=points_agent.device,
        dtype=points_agent.dtype,
    )
    return (
        torch.matmul(
            t[:, :3, :3],
            points_agent.unsqueeze(-1),
        ).squeeze(-1)
        + t[:, :3, 3]
    )


def sample_gt_depth_at_seed_pixels(
    gt_depth: torch.Tensor,
    uv_aug: torch.Tensor,
    view_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Nearest-pixel sampling of the already-augmented CARLA depth map.
    This avoids interpolating depth across object boundaries.
    """
    h, w = int(gt_depth.shape[-2]), int(gt_depth.shape[-1])
    u = torch.round(uv_aug[:, 0]).long().clamp(0, w - 1)
    v = torch.round(uv_aug[:, 1]).long().clamp(0, h - 1)
    z = gt_depth[view_index.long(), v, u]
    return z, u, v



def carla_range_to_optical_z_at_aug_pixels(
    ray_range_m: torch.Tensor,
    uv_aug: torch.Tensor,
    view_index: torch.Tensor,
    geometry: Mapping[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert CARLA camera ray-range rho to optical-axis z at augmented pixels.

    The repository's ``backproject_optical_z`` expects optical-axis z, while
    CARLA depth encodes Euclidean distance along the camera ray. For each
    augmented-image pixel we first undo image augmentation, then compute

        ray = K^{-1} [u_orig, v_orig, 1]^T
        z   = rho / ||ray||.

    Args:
        ray_range_m: CARLA ray-range [N] in meters.
        uv_aug: Augmented-image pixel coordinates [N,2].
        view_index: Flattened camera index [N].
        geometry: Packed camera geometry.

    Returns:
        optical_z [N] and ray_norm [N].
    """
    if int(ray_range_m.numel()) == 0:
        return ray_range_m, ray_range_m

    dtype = ray_range_m.dtype
    device = ray_range_m.device
    vi = view_index.long()

    ones = torch.ones(
        (int(ray_range_m.shape[0]), 1),
        device=device,
        dtype=dtype,
    )
    pix_aug_h = torch.cat(
        [uv_aug.to(device=device, dtype=dtype), ones],
        dim=-1,
    )

    post_rot = geometry["post_rots"][vi].to(
        device=device, dtype=dtype
    )
    post_trans = geometry["post_trans"][vi].to(
        device=device, dtype=dtype
    )
    pix_orig_h = torch.matmul(
        torch.linalg.inv(post_rot),
        (pix_aug_h - post_trans).unsqueeze(-1),
    ).squeeze(-1)

    intrinsic = geometry["intrinsics"][vi].to(
        device=device, dtype=dtype
    )
    ray = torch.matmul(
        torch.linalg.inv(intrinsic),
        pix_orig_h.unsqueeze(-1),
    ).squeeze(-1)
    ray_norm = torch.linalg.norm(ray, dim=-1).clamp_min(1.0e-8)
    optical_z = ray_range_m / ray_norm
    return optical_z, ray_norm


def carla_range_map_to_optical_z(
    ray_range_maps: torch.Tensor,
    geometry: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Vectorized full-image CARLA ray-range -> optical-z conversion.

    Args:
        ray_range_maps: [N,H,W] already transformed with the same image
            resize/crop augmentation as RGB.
        geometry: Packed camera geometry.

    Returns:
        [N,H,W] optical-axis z in meters.
    """
    if ray_range_maps.dim() != 3:
        raise ValueError(
            f"ray_range_maps must be [N,H,W], got {tuple(ray_range_maps.shape)}"
        )

    n, h, w = map(int, ray_range_maps.shape)
    device = ray_range_maps.device
    dtype = ray_range_maps.dtype

    ys = torch.arange(h, device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pix = torch.stack(
        [gx, gy, torch.ones_like(gx)],
        dim=-1,
    ).reshape(-1, 3)

    out = torch.empty_like(ray_range_maps)
    for view in range(n):
        post_rot = geometry["post_rots"][view].to(
            device=device, dtype=dtype
        )
        post_trans = geometry["post_trans"][view].to(
            device=device, dtype=dtype
        )
        intrinsic = geometry["intrinsics"][view].to(
            device=device, dtype=dtype
        )

        pix_orig = torch.matmul(
            torch.linalg.inv(post_rot),
            (pix - post_trans).unsqueeze(-1),
        ).squeeze(-1)
        ray = torch.matmul(
            torch.linalg.inv(intrinsic),
            pix_orig.unsqueeze(-1),
        ).squeeze(-1)
        norm = torch.linalg.norm(ray, dim=-1).clamp_min(1.0e-8)
        out[view] = (
            ray_range_maps[view].reshape(-1) / norm
        ).reshape(h, w)
    return out


def valid_gt_depth(z: torch.Tensor) -> torch.Tensor:
    return (
        torch.isfinite(z)
        & (z > GT_DEPTH_MIN_M)
        & (z < GT_DEPTH_FAR_M)
    )


def subsample_indices(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    # deterministic uniform subsampling for plots only
    return np.linspace(0, n - 1, cap).astype(np.int64)


# ---------------------------------------------------------------------
# GT projection into Drone images
# ---------------------------------------------------------------------

def ego_points_to_agent(
    xyz_ego: torch.Tensor,
    agent_to_ego_one: torch.Tensor,
) -> torch.Tensor:
    t_inv = torch.linalg.inv(agent_to_ego_one)
    return (
        torch.matmul(
            t_inv[:3, :3],
            xyz_ego.unsqueeze(-1),
        ).squeeze(-1)
        + t_inv[:3, 3]
    )


def project_gt_boxes_to_view(
    gt_corners_ego: torch.Tensor,
    geometry: Mapping[str, torch.Tensor],
    view: int,
    image_hw: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """
    Project ego-frame GT 3D corners through one Drone camera.

    Returns axis-aligned 2D rectangles around visible projected 3D corners.
    These rectangles are used only for FG diagnosis/visualization.
    """
    if gt_corners_ego.numel() == 0:
        return []

    g, k, _ = gt_corners_ego.shape
    xyz_ego = gt_corners_ego.reshape(-1, 3)

    t_ae = geometry["agent_to_ego"][view].to(
        device=xyz_ego.device,
        dtype=xyz_ego.dtype,
    )
    xyz_agent = ego_points_to_agent(xyz_ego, t_ae)

    n = int(xyz_agent.shape[0])
    vi = torch.full(
        (n,), int(view),
        dtype=torch.long,
        device=xyz_agent.device,
    )
    intrinsic = geometry["intrinsics"][vi].to(dtype=xyz_agent.dtype)
    cam_rot = geometry["cam2lidar_rot"][vi].to(dtype=xyz_agent.dtype)
    cam_trans = geometry["cam2lidar_trans"][vi].to(dtype=xyz_agent.dtype)
    post_rot = geometry["post_rots"][vi].to(dtype=xyz_agent.dtype)
    post_trans = geometry["post_trans"][vi].to(dtype=xyz_agent.dtype)

    uv, z, front = project_lidar_to_image(
        xyz_agent,
        intrinsic,
        cam_rot,
        cam_trans,
        post_rot,
        post_trans,
    )
    uv = uv.reshape(g, k, 2)
    front = front.reshape(g, k)

    h, w = image_hw
    rects = []
    for gi in range(g):
        mask = front[gi]
        # A valid 3D box should have multiple visible corners.
        if int(mask.sum().item()) < 2:
            continue
        pts = uv[gi][mask].detach().float().cpu().numpy()
        x0 = float(np.clip(pts[:, 0].min(), 0, w - 1))
        y0 = float(np.clip(pts[:, 1].min(), 0, h - 1))
        x1 = float(np.clip(pts[:, 0].max(), 0, w - 1))
        y1 = float(np.clip(pts[:, 1].max(), 0, h - 1))
        if x1 <= x0 or y1 <= y0:
            continue
        # If all visible corners lie completely outside before clipping,
        # the clipped rectangle can become misleadingly large/small. Require
        # at least one projected corner inside image bounds.
        inside = (
            (pts[:, 0] >= 0)
            & (pts[:, 0] < w)
            & (pts[:, 1] >= 0)
            & (pts[:, 1] < h)
        )
        if not np.any(inside):
            continue
        rects.append(
            {
                "gt_index": gi,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            }
        )
    return rects


def fg_metrics_inside_projected_boxes(
    p_fg_up: torch.Tensor,
    fg_mask_up: torch.Tensor,
    rects: List[Dict[str, Any]],
    view: int,
) -> List[Dict[str, Any]]:
    h, w = int(p_fg_up.shape[-2]), int(p_fg_up.shape[-1])
    rows = []
    for r in rects:
        x0 = max(0, min(w - 1, int(math.floor(r["x0"]))))
        y0 = max(0, min(h - 1, int(math.floor(r["y0"]))))
        x1 = max(x0 + 1, min(w, int(math.ceil(r["x1"])) + 1))
        y1 = max(y0 + 1, min(h, int(math.ceil(r["y1"])) + 1))

        patch_p = p_fg_up[y0:y1, x0:x1]
        patch_m = fg_mask_up[y0:y1, x0:x1]
        rows.append(
            {
                "view_index": int(view),
                "gt_index": int(r["gt_index"]),
                "box_x0": float(r["x0"]),
                "box_y0": float(r["y0"]),
                "box_x1": float(r["x1"]),
                "box_y1": float(r["y1"]),
                "pfg_mean_in_projected_gt": float(
                    patch_p.mean().item()
                ),
                "pfg_max_in_projected_gt": float(
                    patch_p.max().item()
                ),
                "fg_fraction_in_projected_gt": float(
                    patch_m.float().mean().item()
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------
# Per-view image diagnostics
# ---------------------------------------------------------------------

def draw_rects(ax, rects: List[Dict[str, Any]]):
    for r in rects:
        xs = [
            r["x0"], r["x1"], r["x1"], r["x0"], r["x0"]
        ]
        ys = [
            r["y0"], r["y0"], r["y1"], r["y1"], r["y0"]
        ]
        ax.plot(xs, ys, linewidth=1.0)


def save_image(path: Path, image: np.ndarray, title: str):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_scalar_map(
    path: Path,
    value: np.ndarray,
    title: str,
    rects: Optional[List[Dict[str, Any]]] = None,
):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(value)
    if rects:
        draw_rects(ax, rects)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_rgb_fg_overlay(
    path: Path,
    rgb: np.ndarray,
    pfg: np.ndarray,
    rects: List[Dict[str, Any]],
    title: str,
):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.imshow(rgb)
    ax.imshow(pfg, alpha=0.45)
    draw_rects(ax, rects)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_view_diagnostics(
    out_dir: Path,
    flat_imgs: torch.Tensor,
    p_fg: torch.Tensor,
    pred_depth: torch.Tensor,
    gt_range: torch.Tensor,
    camera_world_z: torch.Tensor,
    delta_pred: torch.Tensor,
    fg_threshold: float,
    geometry: Mapping[str, torch.Tensor],
    gt_corners: torch.Tensor,
    n_views_per_cav: int,
    semantic_gt: Optional[torch.Tensor],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Save per-view FG/depth diagnostics.

    ``gt_range`` is CARLA ray-range. It is converted here to optical-axis z
    before comparing with ``pred_depth``.
    """
    n_flat, _, image_h, image_w = flat_imgs.shape
    p_up = F.interpolate(
        p_fg[:, None].float(),
        size=(image_h, image_w),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    m_up = F.interpolate(
        (p_fg > fg_threshold)[:, None].float(),
        size=(image_h, image_w),
        mode="nearest",
    )[:, 0] > 0.5
    d_up = F.interpolate(
        pred_depth[:, None].float(),
        size=(image_h, image_w),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    delta_up = F.interpolate(
        delta_pred[:, None].float(),
        size=(image_h, image_w),
        mode="bilinear",
        align_corners=False,
    )[:, 0]

    gt_optical_z = carla_range_map_to_optical_z(
        gt_range.to(device=pred_depth.device, dtype=torch.float32),
        geometry,
    )

    box_metric_rows = []
    view_rows = []

    view_dir = out_dir / "views"
    view_dir.mkdir(parents=True, exist_ok=True)

    for view in range(n_flat):
        cav_idx = int(view // max(n_views_per_cav, 1))
        cam_idx = int(view % max(n_views_per_cav, 1))
        tag = f"view{view:02d}_cav{cav_idx:02d}_cam{cam_idx:02d}"

        rgb = denormalize_rgb(flat_imgs[view, :3])
        p = p_up[view].detach().cpu().numpy()
        dp = d_up[view].detach().cpu().numpy()
        dg_range = gt_range[view].detach().float().cpu().numpy()
        dg_z = gt_optical_z[view].detach().float().cpu().numpy()
        dd = delta_up[view].detach().float().cpu().numpy()

        rects = project_gt_boxes_to_view(
            gt_corners,
            geometry,
            view,
            (image_h, image_w),
        )

        box_metric_rows.extend(
            fg_metrics_inside_projected_boxes(
                p_up[view],
                m_up[view],
                rects,
                view,
            )
        )

        valid_range = (
            np.isfinite(dg_range)
            & (dg_range > GT_DEPTH_MIN_M)
            & (dg_range < GT_DEPTH_FAR_M)
        )
        valid_z = (
            np.isfinite(dg_z)
            & (dg_z > GT_DEPTH_MIN_M)
            & (dg_z < GT_DEPTH_FAR_M)
        )
        err = np.full_like(dg_z, np.nan, dtype=np.float32)
        err[valid_z] = dp[valid_z] - dg_z[valid_z]

        save_rgb_fg_overlay(
            view_dir / f"{tag}_rgb_pfg_overlay.png",
            rgb,
            p,
            rects,
            (
                f"{tag}: predicted p_fg overlay; "
                f"rectangles=projected GT; thr={fg_threshold:.3f}"
            ),
        )
        save_scalar_map(
            view_dir / f"{tag}_pfg.png",
            p,
            f"{tag}: predicted p_fg",
            rects,
        )
        save_scalar_map(
            view_dir / f"{tag}_delta_pred.png",
            dd,
            (
                f"{tag}: Drone delta_pred (m), "
                f"camera_world_z={float(camera_world_z[view].item()):.3f} m"
            ),
            rects,
        )
        save_scalar_map(
            view_dir / f"{tag}_pred_depth.png",
            np.where(np.isfinite(dp), dp, np.nan),
            f"{tag}: predicted Drone optical-z depth (m)",
            rects,
        )
        save_scalar_map(
            view_dir / f"{tag}_gt_range.png",
            np.where(valid_range, dg_range, np.nan),
            f"{tag}: CARLA GT ray-range rho (m)",
            rects,
        )
        save_scalar_map(
            view_dir / f"{tag}_gt_optical_z.png",
            np.where(valid_z, dg_z, np.nan),
            f"{tag}: CARLA GT converted optical-z (m)",
            rects,
        )
        save_scalar_map(
            view_dir / f"{tag}_pred_minus_gt_optical_z.png",
            err,
            f"{tag}: predicted optical-z - GT optical-z (m)",
            rects,
        )

        semantic_nonzero = False
        if semantic_gt is not None:
            sem = semantic_gt[view].detach().float().cpu().numpy()
            semantic_nonzero = bool(np.any(sem != 0))
            if semantic_nonzero:
                save_scalar_map(
                    view_dir / f"{tag}_semantic_gt.png",
                    sem,
                    f"{tag}: image semantic GT",
                    rects,
                )

        fg_mask = p_fg[view] > fg_threshold
        delta_fg = delta_pred[view][fg_mask]
        t = geometry["agent_to_ego"][view].detach().float().cpu().numpy()
        yaw_deg = float(
            math.degrees(math.atan2(float(t[1, 0]), float(t[0, 0])))
        )

        view_rows.append(
            {
                "view_index": view,
                "cav_index": cav_idx,
                "camera_index_within_cav": cam_idx,
                "num_projected_gt_boxes": len(rects),
                "pfg_mean": float(p_fg[view].mean().item()),
                "pfg_max": float(p_fg[view].max().item()),
                "fg_cells": int(fg_mask.sum().item()),
                "camera_world_z_m": float(camera_world_z[view].item()),
                "delta_pred_fg_mean_m": (
                    float(delta_fg.mean().item())
                    if int(delta_fg.numel()) else float("nan")
                ),
                "delta_pred_fg_p50_m": (
                    float(delta_fg.median().item())
                    if int(delta_fg.numel()) else float("nan")
                ),
                "pred_depth_fg_mean_m": (
                    float(pred_depth[view][fg_mask].mean().item())
                    if int(fg_mask.sum().item()) else float("nan")
                ),
                "agent_to_ego_tx_m": float(t[0, 3]),
                "agent_to_ego_ty_m": float(t[1, 3]),
                "agent_to_ego_tz_m": float(t[2, 3]),
                "agent_to_ego_yaw_deg": yaw_deg,
                "semantic_gt_present_and_nonzero": semantic_nonzero,
            }
        )

    return view_rows, box_metric_rows


# ---------------------------------------------------------------------
# Seed-depth diagnosis
# ---------------------------------------------------------------------

def build_seed_depth_diagnostic(
    drone_payload: Mapping[str, Any],
    fg_threshold: float,
) -> Dict[str, Any]:
    heatmap_logits = drone_payload["heatmap_logits"]
    features = drone_payload["f90"]
    pred_depth = drone_payload["depth_z_mean"]
    delta_pred = drone_payload["delta_pred"]
    camera_world_z = drone_payload["camera_world_z"]
    cam = drone_payload["batch_merged_cam_inputs"]
    geometry = drone_payload[CAMERA_GEOMETRY_KEY]

    flat_imgs = flatten_images(cam["imgs"])
    if int(flat_imgs.shape[1]) <= GT_DEPTH_CHANNEL:
        raise RuntimeError(
            "Drone camera tensor has no GT depth channel. "
            f"imgs shape={tuple(flat_imgs.shape)}; expected C>=4."
        )

    # IMPORTANT: this channel is CARLA ray-range rho, not optical-axis z.
    gt_range = flat_imgs[:, GT_DEPTH_CHANNEL].float()

    p_fg = foreground_probability(heatmap_logits)
    seeds = generate_seeds(
        p_fg,
        features,
        fg_threshold=fg_threshold,
    )
    vi = seeds["view_index"]
    yi = seeds["y_indices"]
    xi = seeds["x_indices"]

    feature_hw = (
        int(heatmap_logits.shape[2]),
        int(heatmap_logits.shape[3]),
    )
    image_hw = (
        int(flat_imgs.shape[-2]),
        int(flat_imgs.shape[-1]),
    )

    uv = feature_indices_to_aug_pixels(
        xi,
        yi,
        feature_hw,
        image_hw,
        dtype=torch.float32,
    ).to(device=pred_depth.device)

    gt_range_seed, gt_u, gt_v = sample_gt_depth_at_seed_pixels(
        gt_range.to(device=pred_depth.device),
        uv,
        vi,
    )
    gt_z, ray_norm = carla_range_to_optical_z_at_aug_pixels(
        gt_range_seed.to(dtype=torch.float32),
        uv.to(dtype=torch.float32),
        vi,
        geometry,
    )

    pred_z = pred_depth[vi, yi, xi].float()
    delta_seed = delta_pred[vi, yi, xi].float()
    height_seed = camera_world_z[vi].float()
    p_seed = seeds["p_fg"].float()

    pred_agent, _, _ = backproject_optical_z(
        x_indices=xi,
        y_indices=yi,
        depth_z=pred_z,
        view_index=vi,
        geometry=geometry,
        feature_hw=feature_hw,
        image_hw=image_hw,
    )
    gt_agent, _, _ = backproject_optical_z(
        x_indices=xi,
        y_indices=yi,
        depth_z=gt_z.to(dtype=pred_z.dtype),
        view_index=vi,
        geometry=geometry,
        feature_hw=feature_hw,
        image_hw=image_hw,
    )

    pred_ego = apply_agent_to_ego(
        pred_agent,
        geometry["agent_to_ego"],
        vi,
    )
    gt_ego = apply_agent_to_ego(
        gt_agent,
        geometry["agent_to_ego"],
        vi,
    )

    valid_range = valid_gt_depth(gt_range_seed)
    valid_gt = valid_range & valid_gt_depth(gt_z)
    valid_pred = (
        torch.isfinite(pred_z)
        & (pred_z > GT_DEPTH_MIN_M)
        & (pred_z < GT_DEPTH_FAR_M)
    )
    valid_pair = valid_gt & valid_pred

    n_views = int(geometry["n_views"])
    cav_id = torch.div(
        vi,
        max(n_views, 1),
        rounding_mode="floor",
    ).long()

    return {
        "p_fg": p_fg,
        "seeds": seeds,
        "view_index": vi,
        "cav_id": cav_id,
        "y_indices": yi,
        "x_indices": xi,
        "uv_aug": uv,
        "gt_u": gt_u,
        "gt_v": gt_v,
        "pred_depth": pred_z,
        "delta_seed": delta_seed,
        "camera_world_z_seed": height_seed,
        "gt_range": gt_range_seed,
        "gt_depth": gt_z,
        "ray_norm": ray_norm,
        "p_seed": p_seed,
        "pred_agent": pred_agent,
        "gt_agent": gt_agent,
        "pred_ego": pred_ego,
        "gt_ego": gt_ego,
        "valid_gt_range": valid_range,
        "valid_gt": valid_gt,
        "valid_pred": valid_pred,
        "valid_pair": valid_pair,
        "gt_range_map": gt_range,
        "gt_depth_map": carla_range_map_to_optical_z(
            gt_range.to(device=pred_depth.device, dtype=torch.float32),
            geometry,
        ),
        "pred_depth_map": pred_depth,
        "delta_pred_map": delta_pred,
        "camera_world_z": camera_world_z,
        "flat_imgs": flat_imgs,
        "geometry": geometry,
        "feature_hw": feature_hw,
        "image_hw": image_hw,
    }


def seed_summary(
    diag: Dict[str, Any],
    n_views_per_cav: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    valid = diag["valid_pair"]
    pred_z = diag["pred_depth"]
    gt_z = diag["gt_depth"]
    gt_range = diag["gt_range"]
    ray_norm = diag["ray_norm"]
    delta_seed = diag["delta_seed"]
    height_seed = diag["camera_world_z_seed"]
    p = diag["p_seed"]
    vi = diag["view_index"]
    delta_ego = diag["pred_ego"] - diag["gt_ego"]

    dz = (pred_z - gt_z)[valid]
    dego = delta_ego[valid]

    summary = {
        "n_fg_seeds": int(pred_z.numel()),
        "n_valid_gt_range_at_fg_seed": int(
            diag["valid_gt_range"].sum().item()
        ),
        "n_valid_gt_depth_at_fg_seed": int(
            diag["valid_gt"].sum().item()
        ),
        "n_valid_pred_gt_pairs": int(valid.sum().item()),
        "pred_depth_optical_z_m": finite_stats(
            pred_z[diag["valid_pred"]].detach().cpu().numpy()
        ),
        # Backward-compatible alias; this is optical-z after conversion.
        "pred_depth_m": finite_stats(
            pred_z[diag["valid_pred"]].detach().cpu().numpy()
        ),
        "gt_carla_ray_range_m": finite_stats(
            gt_range[diag["valid_gt_range"]].detach().cpu().numpy()
        ),
        "gt_optical_z_m": finite_stats(
            gt_z[diag["valid_gt"]].detach().cpu().numpy()
        ),
        # Backward-compatible alias; no longer raw CARLA range.
        "gt_depth_m": finite_stats(
            gt_z[diag["valid_gt"]].detach().cpu().numpy()
        ),
        "ray_norm_Kinv_pixel": finite_stats(
            ray_norm[diag["valid_gt_range"]].detach().cpu().numpy()
        ),
        "camera_world_z_m": finite_stats(
            height_seed.detach().cpu().numpy()
        ),
        "delta_pred_m": finite_stats(
            delta_seed.detach().cpu().numpy()
        ),
        "pred_minus_gt_optical_z_m": finite_stats(
            dz.detach().cpu().numpy()
        ),
        # Backward-compatible alias.
        "pred_minus_gt_depth_m": finite_stats(
            dz.detach().cpu().numpy()
        ),
        "abs_optical_z_error_m": finite_stats(
            dz.abs().detach().cpu().numpy()
        ),
        "abs_depth_error_m": finite_stats(
            dz.abs().detach().cpu().numpy()
        ),
        "pred_backproject_minus_gtdepth_backproject_ego_m": vec3_stats(
            dego
        ),
    }

    if int(valid.sum().item()) >= 3:
        dz_np = dz.detach().cpu().numpy()
        d_np = dego.detach().cpu().numpy()
        summary["depth_error_correlation_with_ego_shift"] = {
            "corr_deptherr_dx": corr_safe(dz_np, d_np[:, 0]),
            "corr_deptherr_dy": corr_safe(dz_np, d_np[:, 1]),
            "corr_deptherr_dz": corr_safe(dz_np, d_np[:, 2]),
            "corr_deptherr_xy_norm": corr_safe(
                dz_np,
                np.linalg.norm(d_np[:, :2], axis=1),
            ),
        }

    view_rows = []
    for view in range(int(diag["p_fg"].shape[0])):
        mv = vi == view
        m = valid & mv
        mg = diag["valid_gt"] & mv
        mgr = diag["valid_gt_range"] & mv
        mp = diag["valid_pred"] & mv
        cav_idx = int(view // max(n_views_per_cav, 1))
        cam_idx = int(view % max(n_views_per_cav, 1))

        if int(m.sum().item()) > 0:
            dz_v = (pred_z - gt_z)[m]
            d_v = delta_ego[m]
            dx_mean = float(d_v[:, 0].mean().item())
            dy_mean = float(d_v[:, 1].mean().item())
            dz3_mean = float(d_v[:, 2].mean().item())
            dxy_mean = float(
                torch.linalg.norm(d_v[:, :2], dim=1).mean().item()
            )
            depth_err_mean = float(dz_v.mean().item())
            depth_abs_mean = float(dz_v.abs().mean().item())
        else:
            dx_mean = dy_mean = dz3_mean = dxy_mean = float("nan")
            depth_err_mean = depth_abs_mean = float("nan")

        delta_v = delta_seed[mv]
        pred_v = pred_z[mv]
        gt_z_v = gt_z[mg]
        gt_range_v = gt_range[mgr]
        ray_v = ray_norm[mgr]

        t = diag["geometry"]["agent_to_ego"][view].detach().float().cpu().numpy()
        yaw_deg = float(
            math.degrees(math.atan2(float(t[1, 0]), float(t[0, 0])))
        )

        view_rows.append(
            {
                "view_index": view,
                "cav_index": cav_idx,
                "camera_index_within_cav": cam_idx,
                "fg_seed_count": int(mv.sum().item()),
                "valid_gt_range_seed_count": int(mgr.sum().item()),
                "valid_gt_depth_seed_count": int(mg.sum().item()),
                "valid_pred_depth_seed_count": int(mp.sum().item()),
                "valid_pair_count": int(m.sum().item()),
                "pfg_seed_mean": float(
                    p[mv].mean().item()
                )
                if int(mv.sum().item()) else float("nan"),
                "camera_world_z_m": float(
                    diag["camera_world_z"][view].item()
                ),
                "delta_pred_fg_mean_m": float(
                    delta_v.mean().item()
                ) if int(delta_v.numel()) else float("nan"),
                "delta_pred_fg_p50_m": float(
                    delta_v.median().item()
                ) if int(delta_v.numel()) else float("nan"),
                "delta_pred_fg_p10_m": float(
                    torch.quantile(delta_v.float(), 0.10).item()
                ) if int(delta_v.numel()) else float("nan"),
                "delta_pred_fg_p90_m": float(
                    torch.quantile(delta_v.float(), 0.90).item()
                ) if int(delta_v.numel()) else float("nan"),
                "pred_optical_z_fg_mean_m": float(
                    pred_v.mean().item()
                ) if int(pred_v.numel()) else float("nan"),
                "gt_carla_range_fg_mean_m": float(
                    gt_range_v.mean().item()
                ) if int(gt_range_v.numel()) else float("nan"),
                "gt_optical_z_fg_mean_m": float(
                    gt_z_v.mean().item()
                ) if int(gt_z_v.numel()) else float("nan"),
                "ray_norm_fg_mean": float(
                    ray_v.mean().item()
                ) if int(ray_v.numel()) else float("nan"),
                "pred_minus_gt_depth_mean_m": depth_err_mean,
                "pred_minus_gt_optical_z_mean_m": depth_err_mean,
                "abs_depth_error_mean_m": depth_abs_mean,
                "abs_optical_z_error_mean_m": depth_abs_mean,
                "pred_vs_gtdepth_backproject_mean_dx_ego_m": dx_mean,
                "pred_vs_gtdepth_backproject_mean_dy_ego_m": dy_mean,
                "pred_vs_gtdepth_backproject_mean_dz_ego_m": dz3_mean,
                "pred_vs_gtdepth_backproject_mean_xy_shift_m": dxy_mean,
                "agent_to_ego_tx_m": float(t[0, 3]),
                "agent_to_ego_ty_m": float(t[1, 3]),
                "agent_to_ego_tz_m": float(t[2, 3]),
                "agent_to_ego_yaw_deg": yaw_deg,
            }
        )

    return summary, view_rows


def seed_rows_for_csv(
    diag: Dict[str, Any],
    n_views_per_cav: int,
) -> List[Dict[str, Any]]:
    vi = diag["view_index"].detach().cpu().numpy()
    cav_id = diag["cav_id"].detach().cpu().numpy()
    yi = diag["y_indices"].detach().cpu().numpy()
    xi = diag["x_indices"].detach().cpu().numpy()
    uv = diag["uv_aug"].detach().float().cpu().numpy()
    p = diag["p_seed"].detach().float().cpu().numpy()
    pred_z = diag["pred_depth"].detach().float().cpu().numpy()
    delta_seed = diag["delta_seed"].detach().float().cpu().numpy()
    height_seed = (
        diag["camera_world_z_seed"].detach().float().cpu().numpy()
    )
    gt_range = diag["gt_range"].detach().float().cpu().numpy()
    gt_z = diag["gt_depth"].detach().float().cpu().numpy()
    ray_norm = diag["ray_norm"].detach().float().cpu().numpy()
    valid_gt_range = diag["valid_gt_range"].detach().cpu().numpy()
    valid_gt = diag["valid_gt"].detach().cpu().numpy()
    valid_pair = diag["valid_pair"].detach().cpu().numpy()
    pa = diag["pred_agent"].detach().float().cpu().numpy()
    ga = diag["gt_agent"].detach().float().cpu().numpy()
    pe = diag["pred_ego"].detach().float().cpu().numpy()
    ge = diag["gt_ego"].detach().float().cpu().numpy()

    rows = []
    for i in range(len(vi)):
        v = int(vi[i])
        cav = int(cav_id[i])
        cam = int(v % max(n_views_per_cav, 1))
        rows.append(
            {
                "seed_index": i,
                "view_index": v,
                "cav_index": cav,
                "camera_index_within_cav": cam,
                "feature_x": int(xi[i]),
                "feature_y": int(yi[i]),
                "aug_u": float(uv[i, 0]),
                "aug_v": float(uv[i, 1]),
                "p_fg": float(p[i]),
                "camera_world_z_m": float(height_seed[i]),
                "delta_pred_m": float(delta_seed[i]),
                "pred_depth_optical_z_m": float(pred_z[i]),
                "gt_carla_ray_range_m": float(gt_range[i]),
                "ray_norm_Kinv_pixel": float(ray_norm[i]),
                "gt_depth_optical_z_m": float(gt_z[i]),
                # Backward-compatible column name, now corrected optical-z.
                "gt_depth_z_m": float(gt_z[i]),
                "valid_gt_range": bool(valid_gt_range[i]),
                "valid_gt_depth": bool(valid_gt[i]),
                "valid_pair": bool(valid_pair[i]),
                "depth_error_pred_minus_gt_optical_z_m": (
                    float(pred_z[i] - gt_z[i])
                    if valid_pair[i] else float("nan")
                ),
                "depth_error_pred_minus_gt_m": (
                    float(pred_z[i] - gt_z[i])
                    if valid_pair[i] else float("nan")
                ),
                "pred_agent_x": float(pa[i, 0]),
                "pred_agent_y": float(pa[i, 1]),
                "pred_agent_z": float(pa[i, 2]),
                "gtdepth_agent_x": float(ga[i, 0]),
                "gtdepth_agent_y": float(ga[i, 1]),
                "gtdepth_agent_z": float(ga[i, 2]),
                "pred_ego_x": float(pe[i, 0]),
                "pred_ego_y": float(pe[i, 1]),
                "pred_ego_z": float(pe[i, 2]),
                "gtdepth_ego_x": float(ge[i, 0]),
                "gtdepth_ego_y": float(ge[i, 1]),
                "gtdepth_ego_z": float(ge[i, 2]),
                "backproject_dx_ego_m": (
                    float(pe[i, 0] - ge[i, 0])
                    if valid_pair[i] else float("nan")
                ),
                "backproject_dy_ego_m": (
                    float(pe[i, 1] - ge[i, 1])
                    if valid_pair[i] else float("nan")
                ),
                "backproject_dz_ego_m": (
                    float(pe[i, 2] - ge[i, 2])
                    if valid_pair[i] else float("nan")
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------
# Gaussian stage flow
# ---------------------------------------------------------------------

def gs_to_np_dict(gs) -> Dict[str, np.ndarray]:
    if gs is None:
        return {}
    out = {
        "mean": gs.mean.detach().float().cpu().numpy(),
        "scale": gs.scale.detach().float().cpu().numpy(),
        "quaternion": gs.quaternion.detach().float().cpu().numpy(),
    }
    for name in (
        "p_fg",
        "uv",
        "view_index",
        "batch_index",
        "depth_mean",
        "sigma_z",
    ):
        value = getattr(gs, name, None)
        if value is not None:
            out[name] = value.detach().float().cpu().numpy()
    return out


def aligned_movement(a, b) -> Dict[str, Any]:
    if a is None or b is None:
        return {"available": False}
    if a.n_gaussians != b.n_gaussians:
        return {
            "available": True,
            "aligned": False,
            "n_a": a.n_gaussians,
            "n_b": b.n_gaussians,
        }
    delta = b.mean - a.mean
    return {
        "available": True,
        "aligned": True,
        "n": a.n_gaussians,
        "delta_m": vec3_stats(delta),
    }


def save_gaussian_npz(
    path: Path,
    capture: GaussianFlowCapture,
    pooled: Dict[str, Any],
):
    arrays = {}
    for stage_name, mapping in (
        ("init", capture.init),
        ("stage1", capture.stage1),
        ("stage2", capture.stage2),
        ("pooled", pooled),
    ):
        if not isinstance(mapping, Mapping):
            continue
        for agent, gs in mapping.items():
            if gs is None:
                continue
            d = gs_to_np_dict(gs)
            for field, value in d.items():
                arrays[f"{stage_name}__{agent}__{field}"] = value

    if capture.stage3_merged is not None:
        d = gs_to_np_dict(capture.stage3_merged)
        for field, value in d.items():
            arrays[f"stage3merged__mixed__{field}"] = value

    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------
# Per-GT local geometry
# ---------------------------------------------------------------------

def weighted_local_center(
    points: torch.Tensor,
    weights: Optional[torch.Tensor],
    gt_xy: torch.Tensor,
    radius: float,
) -> Dict[str, Any]:
    if points is None or points.numel() == 0:
        return {
            "n_local": 0,
            "nearest_xy_m": float("nan"),
            "weighted_dx_m": float("nan"),
            "weighted_dy_m": float("nan"),
            "weighted_xy_error_m": float("nan"),
        }

    xy = points[:, :2]
    d = torch.linalg.norm(xy - gt_xy[None, :], dim=1)
    nearest = float(d.min().item())
    mask = d <= float(radius)
    n = int(mask.sum().item())
    if n == 0:
        return {
            "n_local": 0,
            "nearest_xy_m": nearest,
            "weighted_dx_m": float("nan"),
            "weighted_dy_m": float("nan"),
            "weighted_xy_error_m": float("nan"),
        }

    local = xy[mask]
    if weights is None:
        w = torch.ones(
            n, device=local.device, dtype=local.dtype
        )
    else:
        w = weights[mask].to(dtype=local.dtype).clamp_min(1e-6)

    center = (local * w[:, None]).sum(0) / w.sum()
    delta = center - gt_xy
    return {
        "n_local": n,
        "nearest_xy_m": nearest,
        "weighted_x": float(center[0].item()),
        "weighted_y": float(center[1].item()),
        "weighted_dx_m": float(delta[0].item()),
        "weighted_dy_m": float(delta[1].item()),
        "weighted_xy_error_m": float(
            torch.linalg.norm(delta).item()
        ),
        "weight_sum": float(w.sum().item()),
    }


def per_gt_flow_rows(
    gt_corners: torch.Tensor,
    seed_diag: Dict[str, Any],
    capture: GaussianFlowCapture,
    pooled: Dict[str, Any],
    local_radius: float,
) -> List[Dict[str, Any]]:
    gt_xy = gt_corners[..., :2].mean(dim=1)

    valid_gt = seed_diag["valid_gt"]
    source_sets = {
        "seed_preddepth": (
            seed_diag["pred_ego"],
            seed_diag["p_seed"],
        ),
        "seed_gtdepth": (
            seed_diag["gt_ego"][valid_gt],
            seed_diag["p_seed"][valid_gt],
        ),
    }

    for stage_name, mapping in (
        ("drone_init", capture.init),
        ("drone_stage1", capture.stage1),
        ("drone_stage2", capture.stage2),
        ("drone_pooled", pooled),
    ):
        if isinstance(mapping, Mapping) and "drone" in mapping:
            gs = mapping["drone"]
            source_sets[stage_name] = (gs.mean, gs.p_fg)

    for agent in ("vehicle", "rsu"):
        if isinstance(capture.stage2, Mapping) and agent in capture.stage2:
            gs = capture.stage2[agent]
            source_sets[f"{agent}_stage2"] = (gs.mean, gs.p_fg)
        if isinstance(pooled, Mapping) and agent in pooled:
            gs = pooled[agent]
            source_sets[f"{agent}_pooled"] = (gs.mean, gs.p_fg)

    rows = []
    for gi in range(int(gt_xy.shape[0])):
        row = {
            "gt_index": gi,
            "gt_x": float(gt_xy[gi, 0].item()),
            "gt_y": float(gt_xy[gi, 1].item()),
            "local_radius_m": float(local_radius),
        }
        for name, (points, weights) in source_sets.items():
            s = weighted_local_center(
                points,
                weights,
                gt_xy[gi],
                local_radius,
            )
            for k, v in s.items():
                row[f"{name}__{k}"] = v
        rows.append(row)
    return rows


# ---------------------------------------------------------------------
# BEV plots
# ---------------------------------------------------------------------

def drone_cav_ids_from_gaussians(
    gs,
    n_views_per_cav: int,
) -> Optional[torch.Tensor]:
    """Recover originating Drone CAV id from preserved Gaussian view_index."""
    if gs is None or gs.view_index is None:
        return None
    return torch.div(
        gs.view_index.long(),
        max(int(n_views_per_cav), 1),
        rounding_mode="floor",
    )


def infer_pooled_drone_cav_ids(
    stage2_gs,
    pooled_gs,
    pool,
    n_views_per_cav: int,
) -> Optional[torch.Tensor]:
    """Label each *actual* pooled Drone voxel by source CAV.

    Uses exactly the same voxel code as GaussianVoxelPool. If a pooled voxel
    contains inputs from multiple Drone CAVs, its id is -1 ("mixed") rather
    than pretending it belongs to one CAV.
    """
    if (
        stage2_gs is None
        or pooled_gs is None
        or stage2_gs.view_index is None
        or stage2_gs.n_gaussians == 0
    ):
        return None

    device = stage2_gs.mean.device
    dtype = stage2_gs.mean.dtype
    voxel_size = pool.voxel_size.to(device=device, dtype=dtype)
    pc_min = pool.pc_min.to(device=device, dtype=dtype)
    grid = pool.grid.to(device)

    voxel_idx = torch.floor(
        (stage2_gs.mean - pc_min) / voxel_size
    ).long()
    valid = ((voxel_idx >= 0) & (voxel_idx < grid)).all(dim=-1)
    if int(valid.sum().item()) == 0:
        return torch.empty(
            (0,), dtype=torch.long, device=device
        )

    voxel_idx = voxel_idx[valid]
    kept = stage2_gs.index_select(valid)
    batch = (
        kept.batch_index
        if kept.batch_index is not None
        else torch.zeros(
            kept.n_gaussians,
            dtype=torch.long,
            device=device,
        )
    )
    gx, gy, gz = (int(v) for v in grid.tolist())
    code = (
        (batch * gx + voxel_idx[:, 0]) * gy + voxel_idx[:, 1]
    ) * gz + voxel_idx[:, 2]
    unique_code, inverse = torch.unique(code, return_inverse=True)

    if int(unique_code.numel()) != pooled_gs.n_gaussians:
        raise RuntimeError(
            "Reconstructed pooled group count does not match actual "
            f"pooled Drone set: {int(unique_code.numel())} vs "
            f"{pooled_gs.n_gaussians}"
        )

    source_cav = torch.div(
        kept.view_index.long(),
        max(int(n_views_per_cav), 1),
        rounding_mode="floor",
    )
    out = torch.full(
        (int(unique_code.numel()),),
        -1,
        dtype=torch.long,
        device=device,
    )
    for group in range(int(unique_code.numel())):
        ids = torch.unique(source_cav[inverse == group])
        if int(ids.numel()) == 1:
            out[group] = ids[0]
        # multiple source CAVs -> keep -1 = mixed
    return out


def _cav_color(cav_id: int) -> str:
    if int(cav_id) < 0:
        return MIXED_CAV_COLOR
    return DRONE_CAV_COLORS[int(cav_id) % len(DRONE_CAV_COLORS)]


def _cav_label(cav_id: int, prefix: str = "Drone") -> str:
    if int(cav_id) < 0:
        return f"{prefix} mixed-CAV voxel"
    return f"{prefix} CAV {int(cav_id)}"



def draw_gt_bev(ax, gt_corners: torch.Tensor):
    arr = gt_corners.detach().float().cpu().numpy()
    for box in arr:
        xy = box[:4, :2]
        xy = np.concatenate([xy, xy[:1]], axis=0)
        ax.plot(xy[:, 0], xy[:, 1], linewidth=1.1, color="black", alpha=0.9)


def save_bev_points(
    path: Path,
    gt_corners: torch.Tensor,
    points: torch.Tensor,
    title: str,
    cav_range: Sequence[float],
    max_points: int,
    weights: Optional[torch.Tensor] = None,
    group_ids: Optional[torch.Tensor] = None,
    group_prefix: str = "Drone",
):
    """Plot BEV points, optionally coloring each Drone CAV differently."""
    p_all = points.detach().float().cpu().numpy()
    idx = subsample_indices(len(p_all), max_points)
    p = p_all[idx]

    w = None
    if weights is not None:
        w = weights.detach().float().cpu().numpy()[idx]

    gid = None
    if group_ids is not None:
        gid = group_ids.detach().long().cpu().numpy()[idx]
        if len(gid) != len(p):
            raise RuntimeError(
                f"group_ids length {len(gid)} != plotted points {len(p)}"
            )

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)
    draw_gt_bev(ax, gt_corners)

    if len(p):
        if gid is None:
            size = (
                4.0 + 18.0 * np.clip(w, 0.0, 1.0)
                if w is not None
                else 5.0
            )
            ax.scatter(
                p[:, 0],
                p[:, 1],
                s=size,
                alpha=0.35,
            )
        else:
            for cav in sorted(np.unique(gid).tolist()):
                m = gid == cav
                size = (
                    4.0 + 18.0 * np.clip(w[m], 0.0, 1.0)
                    if w is not None
                    else 6.0
                )
                ax.scatter(
                    p[m, 0],
                    p[m, 1],
                    s=size,
                    alpha=0.42,
                    color=_cav_color(int(cav)),
                    label=_cav_label(int(cav), group_prefix),
                )
            ax.legend(loc="best")

    ax.set_xlim(float(cav_range[0]), float(cav_range[3]))
    ax.set_ylim(float(cav_range[1]), float(cav_range[4]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ego x (m)")
    ax.set_ylabel("ego y (m)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_bev_pred_vs_gtdepth(
    path: Path,
    gt_corners: torch.Tensor,
    diag: Dict[str, Any],
    cav_range: Sequence[float],
    max_points: int,
):
    """Compare predicted-depth vs corrected GT-optical-z by Drone CAV color."""
    valid = diag["valid_gt"]
    pred = diag["pred_ego"][valid].detach().float().cpu().numpy()
    gtd = diag["gt_ego"][valid].detach().float().cpu().numpy()
    p = diag["p_seed"][valid].detach().float().cpu().numpy()
    cav = diag["cav_id"][valid].detach().long().cpu().numpy()

    idx = subsample_indices(len(pred), max_points)
    pred = pred[idx]
    gtd = gtd[idx]
    p = p[idx]
    cav = cav[idx]

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)
    draw_gt_bev(ax, gt_corners)

    if len(pred):
        size = 4.0 + 18.0 * np.clip(p, 0.0, 1.0)

        for cav_id in sorted(np.unique(cav).tolist()):
            m = cav == cav_id
            color = _cav_color(int(cav_id))
            ax.scatter(
                pred[m, 0],
                pred[m, 1],
                s=size[m],
                alpha=0.40,
                marker="o",
                color=color,
                label=f"Drone {int(cav_id)} pred-depth",
            )
            ax.scatter(
                gtd[m, 0],
                gtd[m, 1],
                s=size[m],
                alpha=0.75,
                marker="x",
                color=color,
                label=f"Drone {int(cav_id)} GT-optical-z",
            )

        # Draw sparse displacement segments using each Drone's color.
        seg_idx = subsample_indices(len(pred), min(300, max_points))
        for i in seg_idx:
            ax.plot(
                [gtd[i, 0], pred[i, 0]],
                [gtd[i, 1], pred[i, 1]],
                linewidth=0.45,
                alpha=0.30,
                color=_cav_color(int(cav[i])),
            )

    ax.set_xlim(float(cav_range[0]), float(cav_range[3]))
    ax.set_ylim(float(cav_range[1]), float(cav_range[4]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ego x (m)")
    ax.set_ylabel("ego y (m)")
    ax.set_title(
        "Same predicted-FG seeds: predicted optical-z vs "
        "CARLA ray-range converted to optical-z"
    )
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    opt = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    seed_all(opt.seed)
    torch.cuda.set_device(opt.gpu_id)
    device = torch.device(f"cuda:{opt.gpu_id}")

    ckpt = Path(opt.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    config = resolve_config(ckpt, opt.config)
    hypes = yaml_utils.load_yaml(str(config))

    timestamp_dir = Path(opt.timestamp_dir)
    if not timestamp_dir.exists():
        raise FileNotFoundError(timestamp_dir)

    test_root = (
        Path(opt.test_root)
        if opt.test_root
        else timestamp_dir.parent.parent
    )
    hypes["validate_dir"] = str(test_root)

    print("[config]", config)
    print("[checkpoint]", ckpt)
    print("[validate_dir]", hypes["validate_dir"])
    print("[timestamp]", timestamp_dir)

    dataset = build_dataset(
        hypes,
        visualize=False,
        train=False,
    )
    idx, metadata_path = find_timestamp_index(
        dataset,
        timestamp_dir,
    )

    batch = make_batch(dataset, idx)
    batch = train_utils.to_device(batch, device)
    ego = batch["ego"]
    ego["epoch"] = 0

    model = train_utils.create_model(hypes).to(device)
    load_checkpoint_exact(model, ckpt, device)
    model.eval()

    capture = GaussianFlowCapture(model)
    try:
        with torch.no_grad():
            if opt.amp:
                with torch.cuda.amp.autocast():
                    _output = model(ego)
            else:
                _output = model(ego)
    finally:
        capture.close()

    if not isinstance(capture.init, Mapping):
        raise RuntimeError("Failed to capture initializer output.")
    if not isinstance(capture.stage1, Mapping):
        raise RuntimeError("Failed to capture Stage1 output.")
    if not isinstance(capture.stage2, Mapping):
        raise RuntimeError("Failed to capture Stage2 output.")
    if "drone" not in capture.init:
        raise RuntimeError("No Drone Gaussian set in this timestamp.")

    drone_payload = ego["drone"]
    if "heatmap_logits" not in drone_payload:
        raise RuntimeError("Drone P1 heatmap_logits missing.")
    if "depth_z_mean" not in drone_payload:
        raise RuntimeError("Drone P1 depth_z_mean missing.")
    if CAMERA_GEOMETRY_KEY not in drone_payload:
        raise RuntimeError("Drone camera_geometry missing after forward.")

    fg_threshold = (
        float(opt.fg_threshold)
        if opt.fg_threshold is not None
        else float(model.initializer.fg_threshold)
    )

    # Stage3 per-agent pooling. Stage3 later changes only feature, not geometry.
    pooled = {}
    with torch.no_grad():
        for agent, gs in capture.stage2.items():
            if gs.n_gaussians > 0:
                pooled[agent] = model.stage3.pool(gs)

    # Official ego GT corners.
    gt_corners, gt_labels, gt_tracks = (
        dataset.post_processor.generate_gt_bbx_airv2x(batch)
    )

    out_dir = Path(opt.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Same-FG-cell predicted-depth vs GT-depth backprojection
    # --------------------------------------------------------------
    seed_diag = build_seed_depth_diagnostic(
        drone_payload,
        fg_threshold,
    )

    geometry = seed_diag["geometry"]
    n_flat = int(geometry["n_flat"])
    n_views = int(geometry["n_views"])

    summary_seed, per_view_depth_rows = seed_summary(
        seed_diag,
        n_views,
    )
    seed_csv_rows = seed_rows_for_csv(
        seed_diag,
        n_views,
    )

    # --------------------------------------------------------------
    # FG image diagnostics + optional semantic GT
    # --------------------------------------------------------------
    cam = drone_payload["batch_merged_cam_inputs"]
    semantic_gt = None
    if "image_semantic_gts" in cam:
        try:
            semantic_gt = flatten_semantic(
                cam["image_semantic_gts"],
                n_flat,
            )
        except Exception as exc:
            print(
                "[warning] semantic GT exists but could not be flattened:",
                repr(exc),
            )
            semantic_gt = None

    view_rows, fg_box_rows = save_view_diagnostics(
        out_dir=out_dir,
        flat_imgs=seed_diag["flat_imgs"],
        p_fg=seed_diag["p_fg"],
        pred_depth=seed_diag["pred_depth_map"],
        gt_range=seed_diag["gt_range_map"],
        camera_world_z=seed_diag["camera_world_z"],
        delta_pred=seed_diag["delta_pred_map"],
        fg_threshold=fg_threshold,
        geometry=geometry,
        gt_corners=gt_corners,
        n_views_per_cav=n_views,
        semantic_gt=semantic_gt,
    )

    # Merge depth stats into view rows.
    depth_by_view = {
        int(r["view_index"]): r
        for r in per_view_depth_rows
    }
    for r in view_rows:
        r.update(depth_by_view.get(int(r["view_index"]), {}))

    # --------------------------------------------------------------
    # Gaussian stage movement
    # --------------------------------------------------------------
    stage_flow = {
        "drone_counts": {
            "init": capture.init["drone"].n_gaussians,
            "stage1": capture.stage1["drone"].n_gaussians
            if "drone" in capture.stage1 else 0,
            "stage2": capture.stage2["drone"].n_gaussians
            if "drone" in capture.stage2 else 0,
            "pooled": pooled["drone"].n_gaussians
            if "drone" in pooled else 0,
        },
        "drone_init_to_stage1": aligned_movement(
            capture.init.get("drone"),
            capture.stage1.get("drone"),
        ),
        "drone_stage1_to_stage2": aligned_movement(
            capture.stage1.get("drone"),
            capture.stage2.get("drone"),
        ),
    }

    # Exact source-CAV labels for Drone geometry plots.
    drone_cav_ids = {
        "init": drone_cav_ids_from_gaussians(
            capture.init.get("drone"), n_views
        ),
        "stage1": drone_cav_ids_from_gaussians(
            capture.stage1.get("drone"), n_views
        ),
        "stage2": drone_cav_ids_from_gaussians(
            capture.stage2.get("drone"), n_views
        ),
        "pooled": infer_pooled_drone_cav_ids(
            capture.stage2.get("drone"),
            pooled.get("drone"),
            model.stage3.pool,
            n_views,
        ),
    }

    stage_flow["drone_cav_origin_counts"] = {}
    for stage_name, ids in drone_cav_ids.items():
        if ids is None:
            continue
        unique_ids, counts = torch.unique(
            ids.detach().long().cpu(),
            return_counts=True,
        )
        stage_flow["drone_cav_origin_counts"][stage_name] = {
            ("mixed" if int(k.item()) < 0 else f"cav_{int(k.item())}"):
            int(v.item())
            for k, v in zip(unique_ids, counts)
        }

    # --------------------------------------------------------------
    # Per-GT local evidence flow
    # --------------------------------------------------------------
    gt_flow_rows = per_gt_flow_rows(
        gt_corners=gt_corners,
        seed_diag=seed_diag,
        capture=capture,
        pooled=pooled,
        local_radius=float(opt.local_radius),
    )

    # --------------------------------------------------------------
    # Save raw arrays
    # --------------------------------------------------------------
    save_gaussian_npz(
        out_dir / "gaussian_flow_all_stages.npz",
        capture,
        pooled,
    )

    # Save exact CAV-origin labels separately because Stage3 pooling drops
    # view_index. -1 means a voxel mixed Gaussians from multiple Drone CAVs.
    np.savez_compressed(
        out_dir / "drone_gaussian_cav_ids.npz",
        **{
            name: ids.detach().cpu().numpy()
            for name, ids in drone_cav_ids.items()
            if ids is not None
        },
    )

    valid = seed_diag["valid_gt"]
    np.savez_compressed(
        out_dir / "drone_seed_depth_backprojection.npz",
        view_index=seed_diag["view_index"].detach().cpu().numpy(),
        cav_id=seed_diag["cav_id"].detach().cpu().numpy(),
        y_indices=seed_diag["y_indices"].detach().cpu().numpy(),
        x_indices=seed_diag["x_indices"].detach().cpu().numpy(),
        uv_aug=seed_diag["uv_aug"].detach().float().cpu().numpy(),
        p_fg=seed_diag["p_seed"].detach().float().cpu().numpy(),
        camera_world_z=seed_diag["camera_world_z_seed"].detach().float().cpu().numpy(),
        delta_pred=seed_diag["delta_seed"].detach().float().cpu().numpy(),
        pred_depth_optical_z=seed_diag["pred_depth"].detach().float().cpu().numpy(),
        gt_carla_ray_range=seed_diag["gt_range"].detach().float().cpu().numpy(),
        gt_optical_z=seed_diag["gt_depth"].detach().float().cpu().numpy(),
        ray_norm=seed_diag["ray_norm"].detach().float().cpu().numpy(),
        # compatibility name; corrected optical-z
        gt_depth=seed_diag["gt_depth"].detach().float().cpu().numpy(),
        valid_gt=valid.detach().cpu().numpy(),
        pred_agent=seed_diag["pred_agent"].detach().float().cpu().numpy(),
        gtdepth_agent=seed_diag["gt_agent"].detach().float().cpu().numpy(),
        pred_ego=seed_diag["pred_ego"].detach().float().cpu().numpy(),
        gtdepth_ego=seed_diag["gt_ego"].detach().float().cpu().numpy(),
    )

    # --------------------------------------------------------------
    # BEV plots
    # --------------------------------------------------------------
    cav_range = hypes["postprocess"]["anchor_args"][
        "cav_lidar_range"
    ]

    save_bev_pred_vs_gtdepth(
        out_dir / "bev_same_fg_preddepth_vs_gtdepth.png",
        gt_corners,
        seed_diag,
        cav_range,
        opt.max_bev_points,
    )

    save_bev_points(
        out_dir / "bev_drone_preddepth_raw_seeds.png",
        gt_corners,
        seed_diag["pred_ego"],
        "Drone raw FG seeds backprojected with predicted depth",
        cav_range,
        opt.max_bev_points,
        weights=seed_diag["p_seed"],
        group_ids=seed_diag["cav_id"],
        group_prefix="Drone",
    )

    save_bev_points(
        out_dir / "bev_drone_gtdepth_raw_seeds.png",
        gt_corners,
        seed_diag["gt_ego"][valid],
        "Same Drone FG seeds backprojected with corrected CARLA GT optical-z",
        cav_range,
        opt.max_bev_points,
        weights=seed_diag["p_seed"][valid],
        group_ids=seed_diag["cav_id"][valid],
        group_prefix="Drone",
    )

    stage_plot_specs = [
        ("drone_init", capture.init.get("drone"), drone_cav_ids["init"]),
        ("drone_stage1", capture.stage1.get("drone"), drone_cav_ids["stage1"]),
        ("drone_stage2", capture.stage2.get("drone"), drone_cav_ids["stage2"]),
        ("drone_pooled", pooled.get("drone"), drone_cav_ids["pooled"]),
        ("vehicle_stage2", capture.stage2.get("vehicle"), None),
        ("vehicle_pooled", pooled.get("vehicle"), None),
        ("rsu_stage2", capture.stage2.get("rsu"), None),
        ("rsu_pooled", pooled.get("rsu"), None),
    ]
    for name, gs, group_ids in stage_plot_specs:
        if gs is None:
            continue
        save_bev_points(
            out_dir / f"bev_{name}.png",
            gt_corners,
            gs.mean,
            name,
            cav_range,
            opt.max_bev_points,
            weights=gs.p_fg,
            group_ids=group_ids,
            group_prefix="Drone",
        )

    if capture.stage3_merged is not None:
        save_bev_points(
            out_dir / "bev_stage3_merged.png",
            gt_corners,
            capture.stage3_merged.mean,
            "Stage3 merged pooled Gaussians (geometry unchanged by attention)",
            cav_range,
            opt.max_bev_points,
            weights=None,
        )

    # --------------------------------------------------------------
    # CSV / JSON
    # --------------------------------------------------------------
    write_csv(
        out_dir / "drone_fg_seed_depth.csv",
        seed_csv_rows,
    )
    write_csv(
        out_dir / "per_view_drone_fg_depth.csv",
        view_rows,
    )
    write_csv(
        out_dir / "projected_gt_fg_metrics.csv",
        fg_box_rows,
    )
    write_csv(
        out_dir / "per_gt_gaussian_flow.csv",
        gt_flow_rows,
    )

    report = {
        "timestamp_dir": str(timestamp_dir),
        "dataset_index": int(idx),
        "metadata_path": metadata_path,
        "checkpoint": str(ckpt),
        "config": str(config),
        "fg_threshold": fg_threshold,
        "local_radius_m": float(opt.local_radius),
        "drone_camera": {
            "n_flat_views": n_flat,
            "n_views_per_cav": n_views,
            "image_hw": list(seed_diag["image_hw"]),
            "feature_hw": list(seed_diag["feature_hw"]),
            "gt_depth_source": (
                "batch['ego']['drone']['batch_merged_cam_inputs']"
                "['imgs'][...,3,:,:], transformed CARLA camera ray-range meters; "
                "converted to optical-z before backprojection"
            ),
            "pred_depth_source": (
                "batch['ego']['drone']['depth_z_mean'] "
                "= camera_world_z + delta_pred"
            ),
        },
        "same_predicted_fg_seed_depth_diagnosis": summary_seed,
        "gaussian_stage_flow": stage_flow,
        "view_summary": view_rows,
        "drone_cav_color_legend": {
            f"cav_{i}": DRONE_CAV_COLORS[i % len(DRONE_CAV_COLORS)]
            for i in range(int(geometry.get("n_cav", n_flat // max(n_views, 1))))
        },
        "drone_agent_to_ego_matrices": [
            geometry["agent_to_ego"][view].detach().float().cpu().tolist()
            for view in range(n_flat)
        ],
        "notes": {
            "depth_isolation_logic": (
                "CARLA GT depth channel is treated as ray-range rho and "
                "converted to optical-z using K and inverse image augmentation. "
                "Predicted-depth and corrected-GT-depth points then use identical "
                "predicted FG cells, camera calibration, and Drone->ego transform; "
                "their 3D difference isolates depth."
            ),
            "fg_logic": (
                "RGB+p_fg overlays include projected ego GT boxes. "
                "Low/shifted p_fg inside projected GT boxes indicates an "
                "image-plane FG localization/recall issue."
            ),
            "geometry_logic": (
                "If GT-depth reprojected FG points still exhibit the same "
                "systematic ego-frame shift relative to GT, depth alone "
                "cannot explain it; inspect FG locations and camera/agent "
                "geometry. If GT-depth points align but predicted-depth "
                "points shift, Drone depth is directly implicated."
            ),
        },
    }

    (out_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Console summary
    # --------------------------------------------------------------
    print("\n=== SAME PREDICTED-FG SEEDS: DEPTH ===")
    print(
        f"FG seeds={summary_seed['n_fg_seeds']} | "
        f"valid GT depth={summary_seed['n_valid_gt_depth_at_fg_seed']} | "
        f"valid pairs={summary_seed['n_valid_pred_gt_pairs']}"
    )
    de = summary_seed["pred_minus_gt_optical_z_m"]
    print(
        "pred optical-z - corrected GT optical-z: "
        f"mean={de.get('mean', float('nan')):+.3f} m, "
        f"p50={de.get('p50', float('nan')):+.3f} m, "
        f"p90={de.get('p90', float('nan')):+.3f} m"
    )
    bp = summary_seed[
        "pred_backproject_minus_gtdepth_backproject_ego_m"
    ]
    if bp.get("n", 0):
        print(
            "ego shift caused by replacing GT depth with predicted depth:\n"
            f"  dx mean={bp['dx']['mean']:+.3f} m\n"
            f"  dy mean={bp['dy']['mean']:+.3f} m\n"
            f"  dz mean={bp['dz']['mean']:+.3f} m\n"
            f"  3D norm mean={bp['norm']['mean']:.3f} m "
            f"p50={bp['norm']['p50']:.3f} m "
            f"p90={bp['norm']['p90']:.3f} m"
        )

    print("\n=== DRONE GAUSSIAN FLOW ===")
    print(stage_flow)

    print("\n=== PER VIEW ===")
    for r in per_view_depth_rows:
        print(
            f"view {r['view_index']:02d} "
            f"(cav={r['cav_index']} cam={r['camera_index_within_cav']}): "
            f"fg={r['fg_seed_count']} valid={r['valid_pair_count']} "
            f"height={r['camera_world_z_m']:.3f} m "
            f"delta={r['delta_pred_fg_mean_m']:+.3f} m "
            f"depth_err={r['pred_minus_gt_optical_z_mean_m']:+.3f} m "
            f"ego_dx={r['pred_vs_gtdepth_backproject_mean_dx_ego_m']:+.3f} "
            f"ego_dy={r['pred_vs_gtdepth_backproject_mean_dy_ego_m']:+.3f} "
            f"T=({r['agent_to_ego_tx_m']:+.2f},"
            f"{r['agent_to_ego_ty_m']:+.2f},"
            f"{r['agent_to_ego_tz_m']:+.2f}) "
            f"yaw={r['agent_to_ego_yaw_deg']:+.2f}deg"
        )

    print("\n=== OUTPUTS ===")
    print(out_dir / "summary.json")
    print(out_dir / "drone_fg_seed_depth.csv")
    print(out_dir / "per_view_drone_fg_depth.csv")
    print(out_dir / "projected_gt_fg_metrics.csv")
    print(out_dir / "per_gt_gaussian_flow.csv")
    print(out_dir / "gaussian_flow_all_stages.npz")
    print(out_dir / "drone_seed_depth_backprojection.npz")
    print(out_dir / "drone_gaussian_cav_ids.npz")
    print(out_dir / "bev_same_fg_preddepth_vs_gtdepth.png")
    print(out_dir / "views")


if __name__ == "__main__":
    main()

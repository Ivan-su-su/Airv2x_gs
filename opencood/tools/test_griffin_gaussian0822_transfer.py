#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct-transfer sanity check: AirV2X gaussian_0822 -> Griffin (5 frames).

What it does
------------
1) Loads the AirV2X P1/gaussian_0822-compatible weights from:
   /mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/airv2x_gaussian_final/net_final_branch_v45.pth
2) Reads 5 synchronized Griffin frames directly from griffin-release:
   - vehicle: front/back/left/right RGB
   - drone: bottom RGB only
3) Runs gaussian_0822 image frontend / heatmap / depth heads.
4) Converts selected heatmap cells + predicted camera-z into 3D points.
5) Transforms every predicted point into the VEHICLE EGO frame.
6) Overlays vehicle predictions, drone predictions and Griffin GT boxes in BEV.
7) Separately saves heatmap visualizations.

Important
---------
- This is a ZERO-SHOT transfer test. No Griffin fine-tuning is performed.
- Griffin has a different camera setup/resolution from AirV2X. Images are resized
  to 360x640 and intrinsics are scaled consistently.
- The script intentionally uses ONLY drone bottom view, because that is the
  geometric assumption being tested.
- This script expects the checkpoint to contain gaussian_0822-compatible
  frontend/heatmap/depth parameters. It aborts if key heads were not loaded.

Run from the AirV2X-Perception_gaussian repository root, e.g.:

CUDA_VISIBLE_DEVICES=0 python opencood/tools/test_griffin_gaussian0822_transfer.py \
  --griffin_root /mnt/home/suyi/Griffin/datasets/griffin_50scenes_55m/griffin-release \
  --ckpt /mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/airv2x_gaussian_final/net_final_branch_v45.pth \
  --yaml opencood/hypes_yaml/airv2x/camera/gaussian_p1/airv2x_gaussian_p1_joint.yaml \
  --num_frames 5 \
  --fg_thresh 0.30 \
  --topk_per_view 1200 \
  --gpu 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as SciRot
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.models.airv2x_gaussian_0822 import Airv2xGaussian0822


OUT_H = 360
OUT_W = 640
FEAT_H = 90
FEAT_W = 160
STRIDE = 4

IMAGENET = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

AGENT_COLORS = {
    "vehicle": "#ff7f0e",
    "drone": "#1f77b4",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--griffin_root",
        default="/mnt/home/suyi/Griffin/datasets/griffin_50scenes_55m/griffin-release",
    )
    p.add_argument(
        "--ckpt",
        default="/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
                "airv2x_gaussian_final/net_final_branch_v45.pth",
    )
    p.add_argument(
        "--yaml",
        default="opencood/hypes_yaml/airv2x/camera/gaussian_p1/"
                "airv2x_gaussian_p1_joint.yaml",
    )
    p.add_argument("--num_frames", type=int, default=5)
    p.add_argument("--fg_thresh", type=float, default=0.30)
    p.add_argument("--topk_per_view", type=int, default=1200)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--out_dir",
        default="opencood/logs/airv2x_gaussian_final/griffin_zero_shot_5frames",
    )
    p.add_argument(
        "--bev_range",
        nargs=4,
        type=float,
        default=[-80.0, 80.0, -80.0, 80.0],
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Griffin raw-data helpers
# -----------------------------------------------------------------------------

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def load_pose(base: Path, frame: str) -> Dict[str, float]:
    return load_json(base / "pose" / f"{frame}.json")


def pose_to_T_ego_to_enu(pose: Dict[str, float]) -> np.ndarray:
    """
    Griffin official convention:
      ENU xyz, RPY Euler xyz in degrees.
    """
    R = SciRot.from_euler(
        "xyz",
        [float(pose["roll"]), float(pose["pitch"]), float(pose["yaw"])],
        degrees=True,
    ).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [float(pose["x"]), float(pose["y"]), float(pose["z"])]
    return T


def load_calib(base: Path, sensor: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Griffin calibration:
      extrinsic = T_sensor_to_agent_ego
      intrinsic = K
    """
    data = load_json(base / "calib" / f"{sensor}.json")
    T_cam_to_ego = np.asarray(data["extrinsic"], dtype=np.float64)
    K = np.asarray(data["intrinsic"], dtype=np.float64)
    return K, T_cam_to_ego


def list_common_frames(vehicle_base: Path, drone_base: Path) -> List[str]:
    v = {p.stem for p in (vehicle_base / "pose").glob("*.json")}
    d = {p.stem for p in (drone_base / "pose").glob("*.json")}
    frames = sorted(v & d)
    return frames


def read_rgb_and_scaled_K(
    path: Path,
    K_orig: np.ndarray,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Direct deterministic resize to 360x640.
    Returns:
      normalized CHW tensor,
      scaled K,
      uint8 resized RGB for visualization.
    """
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = img.resize((OUT_W, OUT_H), resample=Image.BILINEAR)

    sx = OUT_W / float(orig_w)
    sy = OUT_H / float(orig_h)
    K = K_orig.copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy

    tensor = IMAGENET(img_resized)
    rgb_np = np.asarray(img_resized)
    return tensor, K, rgb_np


def load_gt_objects(vehicle_base: Path, frame: str) -> List[Dict[str, float]]:
    path = vehicle_base / "label" / f"{frame}.txt"
    out = []
    if not path.exists():
        return out
    for line in path.read_text().strip().splitlines():
        if not line.strip():
            continue
        s = line.strip().split()
        # Griffin: type x y z l w h roll pitch yaw id [visibility]
        if len(s) < 11:
            continue
        out.append({
            "type": s[0],
            "x": float(s[1]),
            "y": float(s[2]),
            "z": float(s[3]),
            "l": float(s[4]),
            "w": float(s[5]),
            "h": float(s[6]),
            "roll": float(s[7]),
            "pitch": float(s[8]),
            "yaw": float(s[9]),
            "id": int(float(s[10])),
            "visibility": float(s[11]) if len(s) > 11 else 1.0,
        })
    return out


def instance_path(base: Path, direction: str, frame: str) -> Path:
    return base / "camera" / f"instance_{direction}" / f"{frame}.png"


def instance_to_binary(path: Path) -> np.ndarray | None:
    """
    Diagnostic binary mask only.
    Griffin's instance images are kept raw. For the first transfer test we use
    all-zero as background when possible. The raw mask is also saved separately
    so this convention can be visually checked.
    """
    if not path.exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 2:
        fg = m != 0
    else:
        fg = np.any(m != 0, axis=2)
    fg = cv2.resize(
        fg.astype(np.uint8),
        (OUT_W, OUT_H),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    return fg


# -----------------------------------------------------------------------------
# Checkpoint/model
# -----------------------------------------------------------------------------

def unwrap_state(raw):
    if isinstance(raw, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    return raw


def canonical_key(k: str) -> str:
    for prefix in ("module.", "model."):
        if k.startswith(prefix):
            k = k[len(prefix):]
    return k


def load_compatible_weights(model: torch.nn.Module, ckpt: str) -> Dict[str, object]:
    raw = torch.load(ckpt, map_location="cpu")
    src0 = unwrap_state(raw)
    if not isinstance(src0, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(src0)}")

    src = {canonical_key(k): v for k, v in src0.items() if torch.is_tensor(v)}
    tgt = model.state_dict()

    matched = {}
    source_for_target = {}
    for tk, tv in tgt.items():
        candidates = []
        if tk in src:
            candidates.append(tk)
        # Allow loading when the P1 module was nested in a larger final model.
        suffix = "." + tk
        candidates.extend([sk for sk in src.keys() if sk.endswith(suffix)])
        for sk in candidates:
            sv = src[sk]
            if tuple(sv.shape) == tuple(tv.shape):
                matched[tk] = sv
                source_for_target[tk] = sk
                break

    result = model.load_state_dict(matched, strict=False)
    ratio = len(matched) / max(len(tgt), 1)
    print(f"[CKPT] matched tensors: {len(matched)}/{len(tgt)} ({ratio:.1%})")
    print(f"[CKPT] missing target tensors: {len(result.missing_keys)}")
    print(f"[CKPT] unexpected: {len(result.unexpected_keys)}")

    critical = [
        "heatmap_heads.vehicle.cls.weight",
        "heatmap_heads.drone.cls.weight",
        "drone_delta_head.pred.weight",
    ]
    # Vehicle depth is also needed for vehicle projection.
    critical.append("depth_heads.vehicle.pred.weight")

    missing_critical = [k for k in critical if k not in matched]
    if missing_critical:
        print("\n[CKPT] Example checkpoint keys:")
        for k in list(src.keys())[:80]:
            print("   ", k)
        raise RuntimeError(
            "Checkpoint is not gaussian_0822-compatible for this test. "
            f"Critical tensors not loaded: {missing_critical}\n"
            "If net_final_branch_v45 is a later integrated architecture, inspect "
            "its state_dict keys and adapt the hook/prefix mapping."
        )

    return {
        "matched": matched,
        "source_for_target": source_for_target,
        "ratio": ratio,
    }


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------

def feature_centers_uv(
    feat_h: int,
    feat_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Map prediction cells to centers in resized 360x640 image.

    Vehicle V45:
        45x80 -> stride 8 -> centers (8j+4, 8i+4)

    Drone:
        90x160 -> stride 4 -> centers (4j+2, 4i+2)
    """
    stride_x = OUT_W / float(feat_w)
    stride_y = OUT_H / float(feat_h)

    yy, xx = torch.meshgrid(
        torch.arange(feat_h, device=device, dtype=dtype),
        torch.arange(feat_w, device=device, dtype=dtype),
        indexing="ij",
    )

    u = (xx + 0.5) * stride_x
    v = (yy + 0.5) * stride_y

    return torch.stack([u, v], dim=-1)


def points_from_prediction(
    p_fg: torch.Tensor,
    z_map: torch.Tensor,
    K: np.ndarray,
    T_cam_to_agent: np.ndarray,
    T_agent_to_vehicle: np.ndarray,
    fg_thresh: float,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    p_fg: [90,160]
    z_map: optical-axis camera z, [90,160]

    Returns:
      xyz_vehicle [N,3]
      scores [N]
    """
    dev = p_fg.device
    valid = torch.isfinite(z_map) & (z_map > 0.1) & (p_fg >= fg_thresh)
    ids = torch.nonzero(valid, as_tuple=False)
    if ids.numel() == 0:
        return np.zeros((0, 3)), np.zeros((0,))

    scores = p_fg[ids[:, 0], ids[:, 1]]
    if topk > 0 and ids.shape[0] > topk:
        val, order = torch.topk(scores, k=topk, largest=True, sorted=False)
        ids = ids[order]
        scores = val

    feat_h, feat_w = p_fg.shape[-2:]
    uv_grid = feature_centers_uv(
        feat_h,
        feat_w,
        dev,
        z_map.dtype,
    )
    uv = uv_grid[ids[:, 0], ids[:, 1]]
    z = z_map[ids[:, 0], ids[:, 1]]

    K_t = torch.tensor(K, device=dev, dtype=z.dtype)
    K_inv = torch.linalg.inv(K_t)
    ones = torch.ones((uv.shape[0], 1), device=dev, dtype=z.dtype)
    pix_h = torch.cat([uv, ones], dim=1)
    ray = (K_inv @ pix_h.T).T
    # DepthHead predicts optical-axis z; K^-1[u,v,1] has ray_z = 1.
    xyz_cam = ray * z[:, None]

    xyz_h = torch.cat(
        [xyz_cam, torch.ones((xyz_cam.shape[0], 1), device=dev, dtype=z.dtype)],
        dim=1,
    )
    T = torch.tensor(
        T_agent_to_vehicle @ T_cam_to_agent,
        device=dev,
        dtype=z.dtype,
    )
    xyz_vehicle = (T @ xyz_h.T).T[:, :3]
    return (
        xyz_vehicle.detach().cpu().numpy(),
        scores.detach().cpu().numpy(),
    )


def gt_box_corners_xy(obj: Dict[str, float]) -> np.ndarray:
    l, w = obj["l"], obj["w"]
    local = np.array([
        [ l/2,  w/2],
        [ l/2, -w/2],
        [-l/2, -w/2],
        [-l/2,  w/2],
        [ l/2,  w/2],
    ])
    a = math.radians(obj["yaw"])
    R = np.array([[math.cos(a), -math.sin(a)],
                  [math.sin(a),  math.cos(a)]])
    return local @ R.T + np.array([obj["x"], obj["y"]])


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def save_bev(
    path: Path,
    frame: str,
    vehicle_pts: np.ndarray,
    vehicle_scores: np.ndarray,
    drone_pts: np.ndarray,
    drone_scores: np.ndarray,
    gt_objects: List[Dict[str, float]],
    bev_range: List[float],
):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    if vehicle_pts.shape[0]:
        ax.scatter(
            vehicle_pts[:, 0],
            vehicle_pts[:, 1],
            s=6,
            alpha=0.55,
            c=AGENT_COLORS["vehicle"],
            label=f"vehicle pred ({len(vehicle_pts)})",
        )
    if drone_pts.shape[0]:
        ax.scatter(
            drone_pts[:, 0],
            drone_pts[:, 1],
            s=7,
            alpha=0.65,
            c=AGENT_COLORS["drone"],
            label=f"drone-bottom pred ({len(drone_pts)})",
        )

    for obj in gt_objects:
        c = gt_box_corners_xy(obj)
        ax.plot(c[:, 0], c[:, 1], c="black", linewidth=0.8, alpha=0.75)
        ax.scatter([obj["x"]], [obj["y"]], marker="x", c="black", s=18)

    ax.scatter([0], [0], marker="*", s=120, c="green", label="vehicle ego")
    ax.set_xlim(bev_range[0], bev_range[1])
    ax.set_ylim(bev_range[2], bev_range[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("vehicle ego x (m)")
    ax.set_ylabel("vehicle ego y (m)")
    ax.set_title(f"Griffin zero-shot projection | frame {frame}\n"
                 "GT=black boxes; predictions colored by agent")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_heatmap_panel(
    path: Path,
    title: str,
    rgb: np.ndarray,
    p_fg: np.ndarray,
    gt_fg: np.ndarray | None,
    raw_instance_path: Path | None,
):
    fig, axes = plt.subplots(1, 4 if gt_fg is not None else 3, figsize=(15, 4))
    axes = np.atleast_1d(axes)

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")

    im = axes[1].imshow(p_fg, vmin=0.0, vmax=1.0, cmap="magma")
    axes[1].set_title("Pred heatmap p(fg)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    pred_up = cv2.resize(p_fg, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    axes[2].imshow(rgb)
    axes[2].imshow(pred_up, alpha=0.45, vmin=0.0, vmax=1.0, cmap="magma")
    axes[2].set_title("Pred overlay")

    if gt_fg is not None:
        axes[3].imshow(gt_fg, cmap="gray")
        axes[3].set_title("Instance-derived GT fg")

    for ax in axes:
        ax.axis("off")

    raw_note = ""
    if raw_instance_path is not None:
        raw_note = f"\ninstance={raw_instance_path.name}"
    fig.suptitle(title + raw_note, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    bev_dir = out_dir / "bev"
    hm_dir = out_dir / "heatmaps"
    rawmask_dir = out_dir / "raw_instance_masks"
    for d in (bev_dir, hm_dir, rawmask_dir):
        d.mkdir(parents=True, exist_ok=True)

    griffin_root = Path(args.griffin_root)
    vehicle_base = griffin_root / "vehicle-side"
    drone_base = griffin_root / "drone-side"

    assert vehicle_base.exists(), vehicle_base
    assert drone_base.exists(), drone_base
    assert Path(args.ckpt).exists(), args.ckpt

    hypes = yaml_utils.load_yaml(args.yaml, None)
    model_args = hypes["model"]["args"]
    model_args = dict(model_args)


    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model = Airv2xGaussian0822(model_args)
    ckpt_info = load_compatible_weights(model, args.ckpt)
    model.to(device)
    model.eval()

    vehicle_dirs = ["front", "back", "left", "right"]
    drone_dirs = ["bottom"]

    vehicle_calib = {d: load_calib(vehicle_base, d) for d in vehicle_dirs}
    drone_calib = {d: load_calib(drone_base, d) for d in drone_dirs}

    common_frames = list_common_frames(vehicle_base, drone_base)
    if len(common_frames) < args.num_frames:
        raise RuntimeError(
            f"Only {len(common_frames)} common frames, requested {args.num_frames}"
        )
    frames = common_frames[: args.num_frames]
    print("[DATA] frames:", frames)

    metrics = []

    with torch.no_grad():
        for frame_i, frame in enumerate(frames):
            v_pose = load_pose(vehicle_base, frame)
            d_pose = load_pose(drone_base, frame)
            T_v_to_enu = pose_to_T_ego_to_enu(v_pose)
            T_d_to_enu = pose_to_T_ego_to_enu(d_pose)
            T_enu_to_v = np.linalg.inv(T_v_to_enu)
            T_d_to_v = T_enu_to_v @ T_d_to_enu

            # -----------------------------
            # Build vehicle 4-view batch
            # -----------------------------
            v_imgs, v_Ks, v_rgbs = [], [], []
            for direction in vehicle_dirs:
                K0, _T = vehicle_calib[direction]
                path = vehicle_base / "camera" / direction / f"{frame}.png"
                t, K, rgb = read_rgb_and_scaled_K(path, K0)
                v_imgs.append(t)
                v_Ks.append(K)
                v_rgbs.append(rgb)

            # -----------------------------
            # Build drone bottom-only batch
            # -----------------------------
            d_imgs, d_Ks, d_rgbs = [], [], []
            for direction in drone_dirs:
                K0, T_cam_to_d = drone_calib[direction]
                path = drone_base / "camera" / direction / f"{frame}.png"
                t, K, rgb = read_rgb_and_scaled_K(path, K0)
                d_imgs.append(t)
                d_Ks.append(K)
                d_rgbs.append(rgb)

            v_imgs_t = torch.stack(v_imgs, dim=0).unsqueeze(0).to(device)
            d_imgs_t = torch.stack(d_imgs, dim=0).unsqueeze(0).to(device)

            # Camera world-z expected by gaussian_0822 drone branch.
            d_heights = []
            for direction in drone_dirs:
                _, T_cam_to_d = drone_calib[direction]
                T_cam_to_enu = T_d_to_enu @ T_cam_to_d
                d_heights.append(float(T_cam_to_enu[2, 3]))
            d_height_t = torch.tensor(
                [d_heights], dtype=torch.float32, device=device
            )

            batch = {
                "vehicle": {
                    "batch_merged_cam_inputs": {
                        "imgs": v_imgs_t,
                    }
                },
                "drone": {
                    "batch_merged_cam_inputs": {
                        "imgs": d_imgs_t,
                        "camera_world_z": d_height_t,
                    }
                },
            }

            model._encode_agent("vehicle", batch)
            model._encode_agent("drone", batch)

            pred = {
                "vehicle": {
                    "heatmap_logits": batch["vehicle"]["heatmap_logits"],
                    "depth_z_mean": batch["vehicle"]["depth_z_mean"],
                },
                "drone": {
                    "heatmap_logits": batch["drone"]["heatmap_logits"],
                    "depth_z_mean": batch["drone"]["depth_z_mean"],
                    "delta_pred": batch["drone"]["delta_pred"],
                },
            }

            # -----------------------------
            # Projection: vehicle
            # -----------------------------
            v_prob = torch.softmax(pred["vehicle"]["heatmap_logits"], dim=1)[:, 1]
            v_z = pred["vehicle"]["depth_z_mean"]

            vehicle_xyz_all, vehicle_s_all = [], []
            for view_i, direction in enumerate(vehicle_dirs):
                _, T_cam_to_v = vehicle_calib[direction]
                xyz, score = points_from_prediction(
                    p_fg=v_prob[view_i],
                    z_map=v_z[view_i],
                    K=v_Ks[view_i],
                    T_cam_to_agent=T_cam_to_v,
                    T_agent_to_vehicle=np.eye(4),
                    fg_thresh=args.fg_thresh,
                    topk=args.topk_per_view,
                )
                vehicle_xyz_all.append(xyz)
                vehicle_s_all.append(score)

                gtmask_path = instance_path(vehicle_base, direction, frame)
                gt_fg = instance_to_binary(gtmask_path)
                if gtmask_path.exists():
                    raw = cv2.imread(str(gtmask_path), cv2.IMREAD_UNCHANGED)
                    if raw is not None:
                        cv2.imwrite(
                            str(rawmask_dir / f"{frame}_vehicle_{direction}.png"),
                            raw,
                        )
                save_heatmap_panel(
                    hm_dir / f"{frame}_vehicle_{direction}.png",
                    f"frame={frame} | vehicle {direction}",
                    v_rgbs[view_i],
                    v_prob[view_i].detach().cpu().numpy(),
                    gt_fg,
                    gtmask_path if gtmask_path.exists() else None,
                )

            vehicle_xyz = np.concatenate(vehicle_xyz_all, axis=0) \
                if vehicle_xyz_all else np.zeros((0, 3))
            vehicle_scores = np.concatenate(vehicle_s_all, axis=0) \
                if vehicle_s_all else np.zeros((0,))

            # -----------------------------
            # Projection: drone bottom
            # -----------------------------
            d_prob = torch.softmax(pred["drone"]["heatmap_logits"], dim=1)[:, 1]
            d_z = pred["drone"]["depth_z_mean"]

            drone_xyz_all, drone_s_all = [], []
            for view_i, direction in enumerate(drone_dirs):
                _, T_cam_to_d = drone_calib[direction]
                xyz, score = points_from_prediction(
                    p_fg=d_prob[view_i],
                    z_map=d_z[view_i],
                    K=d_Ks[view_i],
                    T_cam_to_agent=T_cam_to_d,
                    T_agent_to_vehicle=T_d_to_v,
                    fg_thresh=args.fg_thresh,
                    topk=args.topk_per_view,
                )
                drone_xyz_all.append(xyz)
                drone_s_all.append(score)

                gtmask_path = instance_path(drone_base, direction, frame)
                gt_fg = instance_to_binary(gtmask_path)
                if gtmask_path.exists():
                    raw = cv2.imread(str(gtmask_path), cv2.IMREAD_UNCHANGED)
                    if raw is not None:
                        cv2.imwrite(
                            str(rawmask_dir / f"{frame}_drone_{direction}.png"),
                            raw,
                        )
                save_heatmap_panel(
                    hm_dir / f"{frame}_drone_{direction}.png",
                    f"frame={frame} | drone {direction} | "
                    f"camera_world_z={d_heights[view_i]:.2f}m",
                    d_rgbs[view_i],
                    d_prob[view_i].detach().cpu().numpy(),
                    gt_fg,
                    gtmask_path if gtmask_path.exists() else None,
                )

            drone_xyz = np.concatenate(drone_xyz_all, axis=0) \
                if drone_xyz_all else np.zeros((0, 3))
            drone_scores = np.concatenate(drone_s_all, axis=0) \
                if drone_s_all else np.zeros((0,))

            gt_objects = load_gt_objects(vehicle_base, frame)
            save_bev(
                bev_dir / f"{frame}_bev.png",
                frame,
                vehicle_xyz,
                vehicle_scores,
                drone_xyz,
                drone_scores,
                gt_objects,
                args.bev_range,
            )

            # Simple diagnostic: nearest predicted XY distance to every GT center.
            gt_xy = np.asarray([[o["x"], o["y"]] for o in gt_objects], dtype=np.float64)
            frame_metric = {"frame": frame}
            for name, pts in [("vehicle", vehicle_xyz), ("drone", drone_xyz)]:
                if len(gt_xy) and len(pts):
                    diff = gt_xy[:, None, :] - pts[None, :, :2]
                    dist = np.linalg.norm(diff, axis=-1)
                    nearest = dist.min(axis=1)
                    frame_metric[f"{name}_gt_nearest_mean_m"] = float(nearest.mean())
                    frame_metric[f"{name}_gt_nearest_median_m"] = float(np.median(nearest))
                    frame_metric[f"{name}_n_pred"] = int(len(pts))
                else:
                    frame_metric[f"{name}_gt_nearest_mean_m"] = None
                    frame_metric[f"{name}_gt_nearest_median_m"] = None
                    frame_metric[f"{name}_n_pred"] = int(len(pts))
            metrics.append(frame_metric)

            print(
                f"[{frame_i+1}/{len(frames)}] frame={frame} "
                f"veh_pts={len(vehicle_xyz)} drone_pts={len(drone_xyz)} "
                f"GT={len(gt_objects)}"
            )

    report = {
        "checkpoint": args.ckpt,
        "griffin_root": str(griffin_root),
        "frames": frames,
        "fg_thresh": args.fg_thresh,
        "topk_per_view": args.topk_per_view,
        "checkpoint_match_ratio": ckpt_info["ratio"],
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    print("\nDone.")
    print("BEV:", bev_dir)
    print("Heatmaps:", hm_dir)
    print("Raw instance masks:", rawmask_dir)
    print("Metrics:", out_dir / "metrics.json")


if __name__ == "__main__":
    main()

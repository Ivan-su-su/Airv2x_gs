# -*- coding: utf-8 -*-
"""Preflight checks before Griffin P1 adaptation training.

Checks the four assumptions that matter most:
1) official scene split resolves to raw frames;
2) CARLA instance masks decode target semantic tags correctly;
3) vehicle lidar PLY/BIN can be decoded;
4) lidar->camera calibration projects a sensible number of points into images.

Also saves four vehicle camera overlays for one frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opencood.data_utils.datasets.griffin_p1_dataset import (
    VEHICLE_CAMS,
    GriffinP1Dataset,
    _load_calib,
    _load_lidar_xyz,
    _mask_to_foreground,
    _resolve_lidar_path,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="opencood/hypes_yaml/griffin/camera/gaussian_p1/griffin_gaussian_p1_v45.yaml",
    )
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out", default="opencood/logs/griffin_p1_preflight")
    return p.parse_args()


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def project_raw_lidar(
    xyz: np.ndarray,
    t_lidar_to_ego: np.ndarray,
    t_cam_to_ego: np.ndarray,
    k: np.ndarray,
    h: int,
    w: int,
):
    xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
    t = np.linalg.inv(t_cam_to_ego) @ t_lidar_to_ego
    cam = (t @ xyz_h.T).T[:, :3]
    z = cam[:, 2]
    keep = np.isfinite(z) & (z > 0.05)
    cam, z = cam[keep], z[keep]
    uvw = (k @ cam.T).T
    u = uvw[:, 0] / z
    v = uvw[:, 1] / z
    keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u[keep], v[keep], z[keep]


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    ds_cfg = cfg["dataset"]
    ds = GriffinP1Dataset(ds_cfg, args.split)
    index = max(0, min(args.index, len(ds) - 1))
    frame = ds.frames[index]
    print(f"[split] {args.split}: {len(ds)} usable frames; checking frame {frame}")

    root = Path(ds_cfg["root"]).expanduser().resolve()
    vehicle_root = root / "vehicle-side"
    drone_root = root / "drone-side"

    # Lidar layout diagnostics. Official Griffin release uses PLY.
    lidar_path = _resolve_lidar_path(vehicle_root, frame)
    if not lidar_path.exists():
        raise FileNotFoundError(lidar_path)
    if lidar_path.suffix.lower() == ".bin":
        raw = np.fromfile(str(lidar_path), dtype=np.float32)
        candidates = [n for n in range(3, 9) if raw.size % n == 0]
        print(
            f"[lidar] {lidar_path.name}: {raw.size} float32 values; "
            f"possible feature counts={candidates}; yaml={ds.lidar_num_features}"
        )
    else:
        print(
            f"[lidar] {lidar_path.name}: PLY release point cloud "
            f"({lidar_path.stat().st_size} bytes)"
        )
    xyz = _load_lidar_xyz(lidar_path, ds.lidar_num_features)
    print(f"[lidar] decoded points={len(xyz)}")
    finite = np.isfinite(xyz).all(axis=1)
    xyz_f = xyz[finite]
    print(
        "[lidar] xyz finite=%.4f min=%s max=%s" % (
            finite.mean(),
            np.round(xyz_f.min(axis=0), 2).tolist(),
            np.round(xyz_f.max(axis=0), 2).tolist(),
        )
    )

    # Mask diagnostics. Griffin raw masks are CARLA instance segmentation:
    # OpenCV reads BGRA, so channel 2 is the semantic-tag R channel.
    mask_paths = [
        vehicle_root / "camera" / "instance_front" / f"{frame}.png",
        drone_root / "camera" / "instance_bottom" / f"{frame}.png",
    ]
    for path in mask_paths:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(path)
        if mask.ndim != 3 or mask.shape[2] < 3:
            raise ValueError(
                f"{path}: expected raw CARLA BGR/BGRA instance PNG, got {mask.shape}"
            )

        semantic = mask[..., 2]
        sem_ids, sem_counts = np.unique(semantic, return_counts=True)
        order = np.argsort(sem_counts)[::-1]
        top_semantic = [
            (int(sem_ids[i]), int(sem_counts[i]), float(sem_counts[i] / semantic.size))
            for i in order[:20]
        ]

        fg = _mask_to_foreground(
            mask,
            ds.mask_mode,
            target_semantic_tags=ds.target_semantic_tags,
        )
        target_counts = {
            int(tag): int((semantic == int(tag)).sum())
            for tag in ds.target_semantic_tags
        }
        print(
            f"[mask] {path.parent.name}: shape={mask.shape} dtype={mask.dtype} "
            f"mode={ds.mask_mode} target_fg_ratio={fg.mean():.4f}"
        )
        print(
            f"       semantic_top(id,count,ratio)={top_semantic}"
        )
        print(
            f"       target_tag_counts={target_counts}"
        )

    # Dataset target ratios / depth coverage.
    item = ds[index]
    print(
        "[targets] vehicle heatmap fg ratio per cam:",
        [round(float(x.float().mean()), 4) for x in item["vehicle"]["heatmap_target"]],
    )
    print(
        "[targets] drone-bottom heatmap fg ratio:",
        round(float(item["drone"]["heatmap_target"].float().mean()), 4),
    )
    print(
        "[targets] vehicle depth valid ratio per cam:",
        [round(float(x.float().mean()), 4) for x in item["vehicle"]["depth_valid"]],
    )

    # Raw lidar overlays: this is the calibration sanity check that must pass.
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t_lidar_to_ego, _ = _load_calib(vehicle_root, "lidar_top")
    projection_stats = {}
    for cam in VEHICLE_CAMS:
        t_cam_to_ego, k = _load_calib(vehicle_root, cam)
        assert k is not None
        img_path = vehicle_root / "camera" / cam / f"{frame}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        h, w = img.shape[:2]
        u, v, z = project_raw_lidar(xyz, t_lidar_to_ego, t_cam_to_ego, k, h, w)
        projection_stats[cam] = {
            "n_in_image": int(len(z)),
            "z_median": float(np.median(z)) if len(z) else None,
            "z_min": float(np.min(z)) if len(z) else None,
            "z_max": float(np.max(z)) if len(z) else None,
        }
        if len(z):
            z_clip = np.clip(z, 0.0, 50.0)
            norm = (255.0 * z_clip / 50.0).astype(np.uint8)
            colors = cv2.applyColorMap(norm.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
            # Thin points for readability.
            step = max(1, len(z) // 15000)
            for uu, vv, color in zip(u[::step], v[::step], colors[::step]):
                cv2.circle(img, (int(round(uu)), int(round(vv))), 1, color.tolist(), -1)
        save_path = out_dir / f"{frame}_{cam}_lidar_overlay.png"
        cv2.imwrite(str(save_path), img)
        print(f"[projection] {cam}: {projection_stats[cam]} -> {save_path}")

    (out_dir / f"{frame}_projection_stats.json").write_text(
        json.dumps(projection_stats, indent=2)
    )
    print("\nPRE-FLIGHT PASS CRITERIA:")
    print(
        "  1) CARLA target-semantic mask ratios should be sparse/object-like; "
        "target tag counts should be non-zero when objects are visible;"
    )
    print("  2) lidar overlays must land on visible surfaces/vehicles, not mirrored/rotated;")
    print("  3) each useful vehicle camera should have non-trivial depth_valid coverage;")
    print("  4) PLY/BIN lidar decoding must produce finite XYZ with plausible ranges.")


if __name__ == "__main__":
    main()

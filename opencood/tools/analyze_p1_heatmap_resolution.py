#!/usr/bin/env python3
"""Empirical P1 heatmap-resolution study. Does not change training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from opencood.models.sub_modules.lss_submodule import CamEncode
from opencood.utils.box_utils import boxes_to_corners_3d

CLASS_NAMES: Dict[int, str] = {
    1: "bicycle",
    2: "motorcycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "bus",
}
AGENT_KEYS: Dict[str, str] = {
    "vehicle": "veh",
    "rsu": "rsu",
    "drone": "drone",
}
RESOLUTIONS: Tuple[Tuple[int, int], ...] = ((180, 320), (90, 160), (45, 80))
SOURCE_HW: Tuple[int, int] = (360, 640)


def percentile(values: Sequence[float], q: float) -> float:
    """Return a percentile; NaN if empty."""
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_counts(values: Sequence[float]) -> Dict[str, float]:
    """Distribution of per-object cell counts."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "n": float(arr.size),
        "min": float(arr.min()),
        "p1": percentile(values, 1),
        "p5": percentile(values, 5),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "median": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "mean": float(arr.mean()),
        "frac_0": float(np.mean(arr == 0)),
        "frac_1": float(np.mean(arr == 1)),
        "frac_2": float(np.mean(arr == 2)),
        "frac_le2": float(np.mean(arr < 2)),
        "frac_le3": float(np.mean(arr < 3)),
        "frac_le4": float(np.mean(arr <= 4)),
        "frac_le5": float(np.mean(arr < 5)),
        "frac_le9": float(np.mean(arr <= 9)),
        "frac_gt9": float(np.mean(arr > 9)),
    }


def nearest_downsample(semantic: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Match P1 ``F.interpolate(..., mode='nearest')``."""
    tensor = torch.from_numpy(semantic.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=out_hw, mode="nearest")
    return resized[0, 0].long().numpy()


def fg_preserve_downsample(semantic: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Foreground-preserving majority vote. Analysis only; not a training target.

    Each output cell takes the majority foreground class among source pixels
    in its block. Background only if the block is all background. Multiclass
    conflict is resolved by pixel count, not class id order.
    """
    height, width = semantic.shape
    out_h, out_w = out_hw
    if height % out_h != 0 or width % out_w != 0:
        raise ValueError(f"cannot block-pool {semantic.shape} -> {out_hw}")
    factor_h = height // out_h
    factor_w = width // out_w
    onehot = np.eye(7, dtype=np.int32)[np.clip(semantic, 0, 6)]
    onehot[..., 0] = 0
    occupancy = (
        onehot.reshape(out_h, factor_h, out_w, factor_w, 7)
        .sum(axis=(1, 3))
    )
    fg_count = occupancy[..., 1:].sum(axis=-1)
    fg_class = occupancy[..., 1:].argmax(axis=-1) + 1
    return np.where(fg_count > 0, fg_class, 0).astype(np.int64)


def rasterize_convex_polygon(
    points_xy: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Boolean mask of a convex polygon in image coordinates."""
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    xy = [(float(x), float(y)) for x, y in points_xy]
    draw.polygon(xy, fill=1)
    return np.asarray(canvas, dtype=np.uint8) > 0


def project_box_to_image(
    corners_lidar: np.ndarray,
    intrinsic: np.ndarray,
    extrinsics_cam_to_lidar: np.ndarray,
    post_rot: np.ndarray,
    post_trans: np.ndarray,
    image_hw: Tuple[int, int],
) -> Optional[Tuple[np.ndarray, float]]:
    """Project 8 lidar-frame corners into the augmented image.

    ``extrinsics`` follows the dataset after ``ue4_to_lss``: camera-to-lidar.
    """
    ones = np.ones((corners_lidar.shape[0], 1), dtype=np.float64)
    lidar_h = np.concatenate([corners_lidar.astype(np.float64), ones], axis=1).T
    lidar_to_cam = np.linalg.inv(extrinsics_cam_to_lidar.astype(np.float64))
    cam_h = lidar_to_cam @ lidar_h
    xyz = cam_h[:3]
    front = xyz[2] > 0.1
    if int(front.sum()) < 3:
        return None
    uv = intrinsic.astype(np.float64) @ xyz[:, front]
    uv = uv[:2] / np.clip(uv[2:3], 1e-6, None)
    post_r = post_rot[:2, :2].astype(np.float64)
    post_t = post_trans[:2].astype(np.float64).reshape(2, 1)
    uv_aug = post_r @ uv + post_t
    height, width = image_hw
    in_frame = (
        (uv_aug[0] >= -32)
        & (uv_aug[0] < width + 32)
        & (uv_aug[1] >= -32)
        & (uv_aug[1] < height + 32)
    )
    if int(in_frame.sum()) < 3:
        return None
    pts = np.stack([uv_aug[0, in_frame], uv_aug[1, in_frame]], axis=-1)
    camera_z = float(xyz[2, front].mean())
    return pts, camera_z


def dummy_lidar_preprocess(pcd_np: np.ndarray) -> Dict[str, np.ndarray]:
    """Skip voxelization during the resolution study."""
    del pcd_np
    return {
        "voxel_features": np.zeros((1, 32, 4), dtype=np.float32),
        "voxel_coords": np.zeros((1, 3), dtype=np.int32),
        "voxel_num_points": np.ones((1,), dtype=np.int32),
    }


def measure_object_on_map(
    semantic: np.ndarray,
    box_mask: np.ndarray,
    class_id: int,
) -> Dict[str, float]:
    """Intersect projected box with same-class semantic pixels."""
    class_mask = semantic == int(class_id)
    support = class_mask & box_mask
    ys, xs = np.where(support)
    if ys.size == 0:
        return {
            "cells": 0.0,
            "width": 0.0,
            "height": 0.0,
            "area_bbox": 0.0,
        }
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    return {
        "cells": float(ys.size),
        "width": width,
        "height": height,
        "area_bbox": width * height,
    }


def process_camera(
    semantic: np.ndarray,
    cam_inputs: Dict[str, Any],
    view_idx: int,
    corners_lidar: np.ndarray,
    class_ids: Sequence[int],
    image_hw: Tuple[int, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Project each GT box into one camera and measure support at each resolution."""
    intrinsic = cam_inputs["intrinsics"][view_idx].numpy()
    extrinsics = cam_inputs["extrinsics"][view_idx].numpy()
    post_rot = cam_inputs["post_rots"][view_idx].numpy()
    post_trans = cam_inputs["post_trans"][view_idx].numpy()
    order = "hwl"
    corners = corners_lidar
    records: List[Dict[str, Any]] = []
    maps: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {
        "nearest": {},
        "fg_preserve": {},
    }
    maps["nearest"][SOURCE_HW] = semantic
    maps["fg_preserve"][SOURCE_HW] = semantic
    for out_hw in RESOLUTIONS:
        maps["nearest"][out_hw] = nearest_downsample(semantic, out_hw)
        maps["fg_preserve"][out_hw] = fg_preserve_downsample(semantic, out_hw)

    source_fg = int((semantic > 0).sum())
    birth: Dict[str, Dict[str, int]] = {}
    for method, method_maps in maps.items():
        birth[method] = {}
        for hw, label_map in method_maps.items():
            key = f"{hw[0]}x{hw[1]}"
            birth[method][key] = int((label_map > 0).sum())

    for box_idx, class_id in enumerate(class_ids):
        if int(class_id) not in CLASS_NAMES:
            continue
        projected = project_box_to_image(
            corners[box_idx],
            intrinsic,
            extrinsics,
            post_rot,
            post_trans,
            image_hw,
        )
        if projected is None:
            continue
        pts, camera_z = projected
        src_mask = rasterize_convex_polygon(pts, image_hw[0], image_hw[1])
        src_stats = measure_object_on_map(semantic, src_mask, int(class_id))
        if src_stats["cells"] <= 0:
            continue
        record: Dict[str, Any] = {
            "class_id": int(class_id),
            "camera_z": camera_z,
            "src_cells": src_stats["cells"],
            "src_width": src_stats["width"],
            "src_height": src_stats["height"],
            "birth": birth,
        }
        for method, method_maps in maps.items():
            for out_hw, label_map in method_maps.items():
                if out_hw == SOURCE_HW:
                    continue
                scale_y = out_hw[0] / float(image_hw[0])
                scale_x = out_hw[1] / float(image_hw[1])
                scaled = np.stack(
                    [pts[:, 0] * scale_x, pts[:, 1] * scale_y], axis=-1
                )
                box_mask = rasterize_convex_polygon(scaled, out_hw[0], out_hw[1])
                stats = measure_object_on_map(label_map, box_mask, int(class_id))
                key = f"{method}_{out_hw[0]}x{out_hw[1]}"
                record[key] = stats
        records.append(record)
    extra = {"source_fg": source_fg, "birth": birth, "n_visible_objects": len(records)}
    return records, extra


def collect_from_sample(
    sample: Dict[str, Any],
    img_pairwise: np.ndarray,
    object_boxes: np.ndarray,
    object_classes: Sequence[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """Measure one uncollated dataset sample."""
    ego = sample["ego"]
    records: List[Dict[str, Any]] = []
    birth_by_agent: Dict[str, List[int]] = defaultdict(list)
    cav_offset = 0
    for agent_type, abb in AGENT_KEYS.items():
        n_cav = int(ego.get(f"num_{abb}", 0))
        cam_list = ego.get(f"cam_inputs_{abb}", [])
        for local_idx in range(n_cav):
            T_cav2ego = np.asarray(img_pairwise[cav_offset + local_idx, 0], dtype=np.float64)
            if T_cav2ego.shape != (4, 4):
                raise ValueError(f"unexpected pairwise slice shape {T_cav2ego.shape}")
            det = float(np.linalg.det(T_cav2ego))
            if abs(det) < 1e-8:
                T_cav2ego = np.eye(4, dtype=np.float64)
            T_ego2cav = np.linalg.inv(T_cav2ego)
            corners_ego = boxes_to_corners_3d(object_boxes, "hwl")
            if torch.is_tensor(corners_ego):
                corners_ego = corners_ego.numpy()
            n_box = int(corners_ego.shape[0])
            if n_box == 0:
                continue
            ones = np.ones((n_box * 8, 1), dtype=np.float64)
            xyz_h = np.concatenate(
                [corners_ego.reshape(-1, 3).astype(np.float64), ones], axis=1
            ).T
            xyz_cav = (T_ego2cav @ xyz_h).T[:, :3].reshape(n_box, 8, 3)
            cam_inputs = cam_list[local_idx]
            semantic_all = cam_inputs.get("image_semantic_gts")
            if not torch.is_tensor(semantic_all):
                continue
            imgs = cam_inputs["imgs"]
            num_views = int(imgs.shape[0])
            image_hw = (int(imgs.shape[-2]), int(imgs.shape[-1]))
            for view_idx in range(num_views):
                semantic = semantic_all[view_idx].numpy()
                view_records, extra = process_camera(
                    semantic=semantic,
                    cam_inputs=cam_inputs,
                    view_idx=view_idx,
                    corners_lidar=xyz_cav,
                    class_ids=object_classes,
                    image_hw=image_hw,
                )
                for rec in view_records:
                    rec["agent"] = agent_type
                records.extend(view_records)
                birth_by_agent[f"{agent_type}_nearest_45x80"].append(
                    extra["birth"]["nearest"]["45x80"]
                )
                birth_by_agent[f"{agent_type}_nearest_90x160"].append(
                    extra["birth"]["nearest"]["90x160"]
                )
                birth_by_agent[f"{agent_type}_nearest_180x320"].append(
                    extra["birth"]["nearest"]["180x320"]
                )
        cav_offset += n_cav
    return records, birth_by_agent


def probe_camencode_resolutions() -> Dict[str, Any]:
    """Actual EfficientNet / Up spatial sizes for 360x640 RGB."""
    encoder = CamEncode(
        D=48,
        C=64,
        downsample=8,
        ddiscr=[2, 50, 48],
        mode="LID",
        use_gt_depth=False,
        depth_supervision=True,
    )
    encoder.eval()
    sizes: Dict[str, Tuple[int, ...]] = {}

    def _hook(name: str):
        def fn(_module, _inp, out):
            sizes[name] = tuple(int(v) for v in out.shape)

        return fn

    handles = []
    handles.append(encoder.trunk._conv_stem.register_forward_hook(_hook("stem")))
    dummy = torch.zeros(1, 3, 360, 640)
    with torch.no_grad():
        x = encoder.trunk._swish(encoder.trunk._bn0(encoder.trunk._conv_stem(dummy)))
        sizes["after_stem"] = tuple(int(v) for v in x.shape)
        endpoints: Dict[str, torch.Tensor] = {}
        prev_x = x
        for idx, block in enumerate(encoder.trunk._blocks):
            x = block(x)
            if prev_x.size(2) > x.size(2):
                endpoints[f"reduction_{len(endpoints) + 1}"] = prev_x
            prev_x = x
        endpoints[f"reduction_{len(endpoints) + 1}"] = x
        for key, tensor in endpoints.items():
            sizes[key] = tuple(int(v) for v in tensor.shape)
        up1 = encoder.up1(endpoints["reduction_5"], endpoints["reduction_4"])
        sizes["up1"] = tuple(int(v) for v in up1.shape)
        up2 = encoder.up2(up1, endpoints["reduction_3"])
        sizes["up2_F_img"] = tuple(int(v) for v in up2.shape)
    for handle in handles:
        handle.remove()
    return sizes


def smoke_metrics_nograd() -> None:
    """Metrics must not retain autograd graphs."""
    from opencood.models.gaussian_modules_0822.heatmap.metrics import (
        compute_heatmap_metrics,
    )
    from opencood.models.gaussian_modules_0822.lss.metrics import compute_depth_metrics

    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    target = torch.randint(0, 2, (2, 8, 8))
    metrics = compute_heatmap_metrics(logits, target)
    assert logits.grad is None
    loss = logits.sum()
    loss.backward()
    assert logits.grad is not None
    depth = torch.randn(2, 8, 8, requires_grad=True)
    gt = torch.rand(2, 8, 8) * 40 + 2
    compute_depth_metrics(depth, gt, 2.0, 50.0, foreground_mask=target.ne(0))
    print("[smoke] metrics no_grad ok", sorted(metrics.keys()))


def distance_bin(agent: str, camera_z: float) -> str:
    """Distance buckets chosen after inspecting typical camera-z ranges."""
    if agent == "drone":
        if camera_z < 30:
            return "0-30m"
        if camera_z < 80:
            return "30-80m"
        if camera_z < 150:
            return "80-150m"
        return ">150m"
    if camera_z < 15:
        return "0-15m"
    if camera_z < 30:
        return "15-30m"
    if camera_z < 50:
        return "30-50m"
    return ">50m"


def print_table(rows: List[Dict[str, Any]]) -> None:
    """Print a compact markdown-like table."""
    if not rows:
        print("(empty)")
        return
    keys = list(rows[0].keys())
    print(" | ".join(keys))
    print(" | ".join("---" for _ in keys))
    for row in rows:
        print(" | ".join(str(row[k]) for k in keys))


def main() -> None:
    """Run smoke probes and optional dataset measurement."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y",
        default=str(
            ROOT
            / "opencood/hypes_yaml/airv2x/camera/gaussian_p1/airv2x_gaussian_p1_joint.yaml"
        ),
    )
    parser.add_argument(
        "--data-root",
        default="/home/dell/suyi/AirV2X-Perception/train/train",
    )
    parser.add_argument("--max-samples", type=int, default=120)
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument(
        "--out",
        default=str(ROOT / "opencood/logs/p1_heatmap_resolution_study.json"),
    )
    args = parser.parse_args()

    print("=== CamEncode feature resolutions for 360x640 ===")
    sizes = probe_camencode_resolutions()
    for key, value in sizes.items():
        print(f"  {key}: {value}")
    print("reduction_2 is the official ~1/4 feature; up2/F_img is 1/8.")
    smoke_metrics_nograd()
    if args.skip_data:
        return

    if args.skip_data:
        return

    import types

    cuda_stub = types.ModuleType("opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_cuda")

    def _stub_points_in_boxes_cpu(boxes, points, point_indices):
        point_indices.zero_()

    cuda_stub.points_in_boxes_cpu = _stub_points_in_boxes_cpu
    sys.modules["opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_cuda"] = cuda_stub
    import opencood.pcdet_utils.roiaware_pool3d as _roiaware_pkg
    _roiaware_pkg.roiaware_pool3d_cuda = cuda_stub

    from opencood.hypes_yaml import yaml_utils
    from opencood.data_utils.datasets.airv2x.intermediate_fusion_dataset import (
        IntermediateFusionDatasetAirv2x,
    )

    hypes = yaml_utils.load_yaml(args.y, None)
    hypes["root_dir"] = args.data_root
    hypes["validate_dir"] = args.data_root
    dataset = IntermediateFusionDatasetAirv2x(
        params=hypes, visualize=False, train=False
    )
    dataset.pre_processor.preprocess = dummy_lidar_preprocess
    n_total = len(dataset)
    stride = max(1, n_total // max(args.max_samples, 1))
    indices = list(range(0, n_total, stride))[: args.max_samples]
    print(f"Dataset len={n_total} stride={stride} using {len(indices)} samples")

    all_records: List[Dict[str, Any]] = []
    birth_store: Dict[str, List[int]] = defaultdict(list)
    n_ok = 0
    for idx in tqdm(indices, desc="resolution-study"):
        try:
            sample = dataset[idx]
        except Exception as exc:
            print(f"[skip] idx={idx} {type(exc).__name__}: {exc}")
            continue
        ego = sample["ego"]
        mask = np.asarray(ego["object_bbx_mask"]).astype(bool)
        boxes = np.asarray(ego["object_bbx_center"])[mask]
        classes = list(ego["class_ids"])
        if boxes.shape[0] != len(classes):
            classes = classes[: boxes.shape[0]]
        pairwise = np.asarray(ego["img_pairwise_t_matrix_collab"])
        try:
            recs, birth = collect_from_sample(sample, pairwise, boxes, classes)
        except Exception as exc:
            print(f"[skip-collect] idx={idx} {type(exc).__name__}: {exc}")
            continue
        all_records.extend(recs)
        for key, values in birth.items():
            birth_store[key].extend(values)
        n_ok += 1

    print(f"loaded samples={n_ok} visible object-views={len(all_records)}")
    z_all = [r["camera_z"] for r in all_records]
    print(
        "camera_z percentiles:",
        {q: percentile(z_all, q) for q in (5, 10, 25, 50, 75, 90, 95)},
    )

    payload: Dict[str, Any] = {
        "n_samples": n_ok,
        "n_object_views": len(all_records),
        "camencode_sizes": {k: list(v) for k, v in sizes.items()},
        "camera_z_percentiles": {str(q): percentile(z_all, q) for q in (5, 50, 95)},
    }

    table_rows: List[Dict[str, Any]] = []
    for agent in ("vehicle", "rsu", "drone"):
        agent_recs = [r for r in all_records if r["agent"] == agent]
        for group_name, subset in [("all", agent_recs)] + [
            (name, [r for r in agent_recs if r["class_id"] == cid])
            for cid, name in CLASS_NAMES.items()
        ]:
            if not subset:
                continue
            cells = [r["nearest_45x80"]["cells"] for r in subset]
            disappear_45 = float(np.mean([c == 0 for c in cells]))
            cells_90 = [r["nearest_90x160"]["cells"] for r in subset]
            cells_180 = [r["nearest_180x320"]["cells"] for r in subset]
            fg_cells = [r["fg_preserve_45x80"]["cells"] for r in subset]
            stats = summarize_counts(cells)
            birth_key = f"{agent}_nearest_45x80"
            birth_vals = birth_store.get(birth_key, [])
            row = {
                "Agent": agent,
                "Class/Group": group_name,
                "Resolution": "45x80-nearest",
                "N": int(len(subset)),
                "Median cells/object": round(stats.get("median", float("nan")), 3),
                "P10": round(stats.get("p10", float("nan")), 3),
                "% 0 cells": round(100 * stats.get("frac_0", float("nan")), 2),
                "% 1 cell": round(100 * stats.get("frac_1", float("nan")), 2),
                "% <=4": round(100 * stats.get("frac_le4", float("nan")), 2),
                "Mean Gaussian candidates/image": round(
                    float(np.mean(birth_vals)) if birth_vals else float("nan"), 1
                ),
            }
            table_rows.append(row)
            payload.setdefault("groups", []).append(
                {
                    "agent": agent,
                    "group": group_name,
                    "n": len(subset),
                    "nearest_45x80": stats,
                    "nearest_90x160": summarize_counts(cells_90),
                    "nearest_180x320": summarize_counts(cells_180),
                    "fg_preserve_45x80": summarize_counts(fg_cells),
                    "disappear_nearest_45": disappear_45,
                    "disappear_nearest_90": float(np.mean([c == 0 for c in cells_90])),
                    "disappear_nearest_180": float(np.mean([c == 0 for c in cells_180])),
                    "disappear_fg_preserve_45": float(
                        np.mean([c == 0 for c in fg_cells])
                    ),
                    "median_src_wh": [
                        percentile([r["src_width"] for r in subset], 50),
                        percentile([r["src_height"] for r in subset], 50),
                    ],
                    "frac_lt2_45": stats.get("frac_le2"),
                    "frac_lt3_45": stats.get("frac_le3"),
                    "frac_lt5_45": stats.get("frac_le5"),
                }
            )
        for bin_name in ("0-15m", "15-30m", "30-50m", ">50m", "0-30m", "30-80m", "80-150m", ">150m"):
            subset = [
                r
                for r in agent_recs
                if distance_bin(agent, r["camera_z"]) == bin_name
            ]
            if not subset:
                continue
            cells = [r["nearest_45x80"]["cells"] for r in subset]
            payload.setdefault("distance", []).append(
                {
                    "agent": agent,
                    "bin": bin_name,
                    "n": len(subset),
                    "median_cells_45": percentile(cells, 50),
                    "p10_45": percentile(cells, 10),
                    "disappear_45": float(np.mean([c == 0 for c in cells])),
                    "median_src_cells": percentile([r["src_cells"] for r in subset], 50),
                }
            )

    print("\n=== Verdict table (45x80 nearest, instance-level) ===")
    print_table(table_rows)

    print("\n=== Downsample method at 45x80 (all agents) ===")
    for method in ("nearest_45x80", "fg_preserve_45x80"):
        cells = [r[method]["cells"] for r in all_records]
        stats = summarize_counts(cells)
        print(method, {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()})

    print("\n=== Gaussian birth counts per camera image ===")
    birth_summary = {}
    for key, values in sorted(birth_store.items()):
        birth_summary[key] = {
            "mean": float(np.mean(values)),
            "median": percentile(values, 50),
            "p10": percentile(values, 10),
            "p90": percentile(values, 90),
        }
        print(key, {k: round(v, 1) for k, v in birth_summary[key].items()})
    payload["birth_per_image"] = birth_summary
    for agent in ("vehicle", "rsu", "drone"):
        a45 = birth_store.get(f"{agent}_nearest_45x80", [])
        a90 = birth_store.get(f"{agent}_nearest_90x160", [])
        a180 = birth_store.get(f"{agent}_nearest_180x320", [])
        if a45 and a90:
            print(
                f"{agent} candidate multiply 45->90: {float(np.mean(a90))/max(float(np.mean(a45)), 1e-6):.2f}x"
            )
        if a45 and a180:
            print(
                f"{agent} candidate multiply 45->180: {float(np.mean(a180))/max(float(np.mean(a45)), 1e-6):.2f}x"
            )

    print("\n=== Distance bins ===")
    for item in payload.get("distance", []):
        print(item)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

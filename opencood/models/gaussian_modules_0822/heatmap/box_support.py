"""Projected-GT-box object-support maps for drone TRAIN heatmap targets.

Geometry is copied from the verified diagnostic
``opencood/tools/analyze_p1_heatmap_resolution.py``:
``project_box_to_image`` and ``rasterize_convex_polygon``.

Does not load or modify ``*_seg.bin``. Vehicle/RSU must not call this.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from opencood.models.gaussian_modules_0822.heatmap.target import (
    binary_objectness_target,
    build_semantic_target,
)
from opencood.utils.box_utils import boxes_to_corners_3d

VALID_BOX_CLASS_IDS = {1, 2, 3, 4, 5, 6}
_FRONT_Z = 0.1
_IN_FRAME_PAD = 32
_DEGENERATE_SPAN = 1e-3
_DEFAULT_AGENT_ORDER = ("vehicle", "rsu", "drone")


def rasterize_convex_polygon(
    points_xy: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Boolean mask of a convex polygon in image coordinates.

    Copied from ``analyze_p1_heatmap_resolution.rasterize_convex_polygon``.
    PIL clips to the canvas, so partial-out-of-image polygons are bounded.
    """
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

    Copied from ``analyze_p1_heatmap_resolution.project_box_to_image``.
    ``extrinsics`` follows the dataset after ``ue4_to_lss``: camera-to-lidar.

    Filtering (same as the diagnostic, plus conservative finite/degenerate
    guards requested for TRAIN):

    * non-finite extrinsics / corners / projected uv → invalid
    * fewer than 3 corners with camera ``z > 0.1`` (behind camera) → invalid
    * after ``post_rots`` / ``post_trans``, fewer than 3 corners inside the
      image padded by 32 px → invalid
    * remaining points collapse to a degenerate location → invalid
    """
    if not np.isfinite(corners_lidar).all():
        return None
    if not np.isfinite(extrinsics_cam_to_lidar).all():
        return None
    if not np.isfinite(intrinsic).all():
        return None
    ones = np.ones((corners_lidar.shape[0], 1), dtype=np.float64)
    lidar_h = np.concatenate([corners_lidar.astype(np.float64), ones], axis=1).T
    try:
        lidar_to_cam = np.linalg.inv(extrinsics_cam_to_lidar.astype(np.float64))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(lidar_to_cam).all():
        return None
    cam_h = lidar_to_cam @ lidar_h
    xyz = cam_h[:3]
    front = xyz[2] > _FRONT_Z
    if int(front.sum()) < 3:
        return None
    uv = intrinsic.astype(np.float64) @ xyz[:, front]
    uv = uv[:2] / np.clip(uv[2:3], 1e-6, None)
    if not np.isfinite(uv).all():
        return None
    post_r = post_rot[:2, :2].astype(np.float64)
    post_t = post_trans[:2].astype(np.float64).reshape(2, 1)
    uv_aug = post_r @ uv + post_t
    if not np.isfinite(uv_aug).all():
        return None
    height, width = image_hw
    in_frame = (
        (uv_aug[0] >= -_IN_FRAME_PAD)
        & (uv_aug[0] < width + _IN_FRAME_PAD)
        & (uv_aug[1] >= -_IN_FRAME_PAD)
        & (uv_aug[1] < height + _IN_FRAME_PAD)
    )
    if int(in_frame.sum()) < 3:
        return None
    pts = np.stack([uv_aug[0, in_frame], uv_aug[1, in_frame]], axis=-1)
    if not np.isfinite(pts).all():
        return None
    if pts.shape[0] < 3:
        return None
    if (
        float(np.ptp(pts[:, 0])) < _DEGENERATE_SPAN
        and float(np.ptp(pts[:, 1])) < _DEGENERATE_SPAN
    ):
        return None
    camera_z = float(xyz[2, front].mean())
    return pts, camera_z


def _as_numpy(value: Any) -> np.ndarray:
    """Detach tensors to CPU numpy; pass ndarrays through."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _class_ids_for_batch(class_ids: Any, batch_index: int, n_box: int) -> np.ndarray:
    """Align collated ``class_ids`` with the unpadded GT boxes of one sample."""
    if class_ids is None:
        return np.ones((n_box,), dtype=np.int64)
    if torch.is_tensor(class_ids):
        row = class_ids[batch_index]
        ids = _as_numpy(row).reshape(-1)
        return ids[:n_box].astype(np.int64)
    row = class_ids[batch_index]
    ids = np.asarray(row, dtype=np.int64).reshape(-1)
    if ids.size >= n_box:
        return ids[:n_box]
    padded = np.ones((n_box,), dtype=np.int64)
    padded[: ids.size] = ids
    return padded


def _record_len(ego: Mapping[str, Any], agent_type: str, batch_index: int) -> int:
    """Per-sample CAV count for one agent type."""
    agent = ego.get(agent_type)
    if not isinstance(agent, Mapping):
        return 0
    record_len = agent.get("record_len")
    if record_len is None:
        return 0
    value = record_len[batch_index]
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def paint_drone_box_support_maps(
    ego: Mapping[str, Any],
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Rasterize union of valid projected GT boxes onto each drone view.

    Args:
        ego: Collated P1 ``batch["ego"]`` dict.

    Returns:
        A tuple ``(maps, stats)`` where ``maps`` is ``[N, H, W]`` long
        ``{0, 1}`` at the current 360x640 image size, matching the flattened
        drone camera order used by the heatmap head, and ``stats`` counts GT
        boxes and valid box-view projections.

    Raises:
        KeyError: If drone camera tensors required for projection are missing.
        ValueError: If camera tensors have an unexpected rank.
    """
    drone = ego.get("drone")
    if not isinstance(drone, Mapping):
        raise KeyError("ego['drone'] is required for box-support targets")
    cam_inputs = drone.get("batch_merged_cam_inputs")
    if not isinstance(cam_inputs, Mapping):
        raise KeyError("drone batch_merged_cam_inputs is required")
    imgs = cam_inputs.get("imgs")
    if not torch.is_tensor(imgs):
        raise KeyError("drone imgs is required")
    if imgs.dim() == 5:
        n_drones, n_views, _, height, width = imgs.shape
        n_flat = int(n_drones * n_views)
    elif imgs.dim() == 4:
        n_flat, _, height, width = imgs.shape
        n_views = 1
        n_drones = n_flat
    else:
        raise ValueError(f"drone imgs must be 4D or 5D, got {tuple(imgs.shape)}")

    maps = np.zeros((n_flat, int(height), int(width)), dtype=np.uint8)
    stats = {
        "n_gt_boxes": 0,
        "n_valid_projections": 0,
        "n_source_fg_pixels": 0,
    }
    if n_flat == 0:
        return torch.from_numpy(maps).long(), stats

    boxes_all = ego.get("object_bbx_center")
    mask_all = ego.get("object_bbx_mask")
    if not torch.is_tensor(boxes_all) or not torch.is_tensor(mask_all):
        raise KeyError("object_bbx_center / object_bbx_mask are required")
    if boxes_all.dim() == 2:
        boxes_all = boxes_all.unsqueeze(0)
        mask_all = mask_all.unsqueeze(0)
    batch_size = int(boxes_all.shape[0])
    pairwise_all = ego.get("img_pairwise_t_matrix_collab")
    if not torch.is_tensor(pairwise_all):
        raise KeyError("img_pairwise_t_matrix_collab is required")
    if pairwise_all.dim() == 4:
        pairwise_all = pairwise_all.unsqueeze(0)
    agent_order: Sequence[str] = tuple(ego.get("agent_order") or _DEFAULT_AGENT_ORDER)
    class_ids = ego.get("class_ids")
    image_hw = (int(height), int(width))

    intrinsics = _as_numpy(cam_inputs["intrinsics"])
    extrinsics = _as_numpy(cam_inputs["extrinsics"])
    post_rots = _as_numpy(cam_inputs["post_rots"])
    post_trans = _as_numpy(cam_inputs["post_trans"])

    drone_tensor_idx = 0
    n_gt_boxes_sum = 0
    n_valid = 0
    for batch_index in range(batch_size):
        mask = _as_numpy(mask_all[batch_index]).reshape(-1) == 1
        boxes = _as_numpy(boxes_all[batch_index][mask]).astype(np.float64)
        n_box = int(boxes.shape[0])
        n_gt_boxes_sum += n_box
        class_row = _class_ids_for_batch(class_ids, batch_index, n_box)
        n_drone = _record_len(ego, "drone", batch_index)
        if n_drone == 0:
            continue
        drone_offset = 0
        for name in agent_order:
            if name == "drone":
                break
            drone_offset += _record_len(ego, name, batch_index)
        pairwise = _as_numpy(pairwise_all[batch_index])
        keep = np.ones((n_box,), dtype=bool)
        for box_idx in range(n_box):
            class_id = int(class_row[box_idx]) if box_idx < class_row.size else 1
            if class_id not in VALID_BOX_CLASS_IDS:
                keep[box_idx] = False
            if not np.isfinite(boxes[box_idx]).all():
                keep[box_idx] = False
        boxes = boxes[keep]
        n_box = int(boxes.shape[0])
        if n_box == 0:
            drone_tensor_idx += n_drone
            continue
        corners_ego = boxes_to_corners_3d(boxes, "hwl")
        if torch.is_tensor(corners_ego):
            corners_ego = corners_ego.numpy()
        for local_idx in range(n_drone):
            T_cav2ego = np.asarray(pairwise[drone_offset + local_idx, 0], dtype=np.float64)
            if T_cav2ego.shape != (4, 4):
                raise ValueError(f"unexpected pairwise slice shape {T_cav2ego.shape}")
            det = float(np.linalg.det(T_cav2ego))
            if abs(det) < 1e-8:
                T_cav2ego = np.eye(4, dtype=np.float64)
            T_ego2cav = np.linalg.inv(T_cav2ego)
            ones = np.ones((n_box * 8, 1), dtype=np.float64)
            xyz_h = np.concatenate(
                [corners_ego.reshape(-1, 3).astype(np.float64), ones], axis=1
            ).T
            xyz_cav = (T_ego2cav @ xyz_h).T[:, :3].reshape(n_box, 8, 3)
            if drone_tensor_idx >= n_drones:
                raise AssertionError(
                    f"drone tensor idx {drone_tensor_idx} exceeds stacked n={n_drones}"
                )
            for view_idx in range(int(n_views)):
                flat_idx = drone_tensor_idx * int(n_views) + view_idx
                view_mask = np.zeros((int(height), int(width)), dtype=np.uint8)
                for box_idx in range(n_box):
                    projected = project_box_to_image(
                        xyz_cav[box_idx],
                        intrinsics[drone_tensor_idx, view_idx],
                        extrinsics[drone_tensor_idx, view_idx],
                        post_rots[drone_tensor_idx, view_idx],
                        post_trans[drone_tensor_idx, view_idx],
                        image_hw,
                    )
                    if projected is None:
                        continue
                    pts, _camera_z = projected
                    poly = rasterize_convex_polygon(pts, image_hw[0], image_hw[1])
                    if not bool(poly.any()):
                        continue
                    view_mask[poly] = 1
                    n_valid += 1
                maps[flat_idx] = view_mask
            drone_tensor_idx += 1

    stats["n_gt_boxes"] = int(n_gt_boxes_sum)
    stats["n_valid_projections"] = int(n_valid)
    stats["n_source_fg_pixels"] = int(maps.sum())
    return torch.from_numpy(maps).long(), stats


def build_drone_box_support_target(
    ego: Mapping[str, Any],
    tau: int = 1,
) -> torch.Tensor:
    """Binary R90 object-support target from projected official GT boxes.

    Rasterizes the union of valid polygons on the 360x640 image, then applies
    the existing tau=1 4x4 occupancy rule. Output ids are ``{0, 1}``.

    Args:
        ego: Collated P1 ``batch["ego"]`` dict.
        tau: Occupancy threshold. Production uses 1.

    Returns:
        ``[N, 90, 160]`` long tensor, values in ``{0, 1}``.
    """
    maps, _stats = paint_drone_box_support_maps(ego)
    return binary_objectness_target(maps, tau=int(tau))


def build_drone_union_target(
    ego: Mapping[str, Any],
    tau: int = 1,
) -> torch.Tensor:
    """Binary R90 target: projected GT-box OR SAM3/seg.bin foreground.

    Reuses ``build_drone_box_support_target`` and ``build_semantic_target``.
    For tau=1 this matches occupancy of the 360x640 pixel-wise union.
    Values stay in ``{0, 1}``. Does not write ``*_seg.bin``.
    """
    box = build_drone_box_support_target(ego, tau=int(tau))
    drone = ego.get("drone")
    if not isinstance(drone, Mapping):
        raise KeyError("ego['drone'] is required for union targets")
    cam_inputs = drone.get("batch_merged_cam_inputs")
    if not isinstance(cam_inputs, Mapping):
        raise KeyError("drone batch_merged_cam_inputs is required")
    sam3 = build_semantic_target(cam_inputs, tau=int(tau))
    if tuple(box.shape) != tuple(sam3.shape):
        raise AssertionError(
            f"drone union shape mismatch box {tuple(box.shape)} vs sam3 {tuple(sam3.shape)}"
        )
    return torch.maximum(box.to(device=sam3.device), sam3)

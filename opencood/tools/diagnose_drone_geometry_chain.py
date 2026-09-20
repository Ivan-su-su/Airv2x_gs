#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose AirV2X drone geometry / GT-to-RGB projection chain.

Checks before Stage-1/2/3:
  1) current GT vs delayed RGB/depth timestamp (time_delay)
  2) actual Gaussian attach_camera_geometry() transforms
  3) Gaussian lift vs official LSS get_geometry algebra
  4) A/B/C drone camera->ego chains
  5) raw metadata extrinsic vs pose-derived exact/additive extrinsics
  6) current-GT and delayed-GT overlays on delayed raw RGB

A = current dataset/Gaussian chain, with GT in the CURRENT dataset ego frame
B = exact rigid SE(3) pose chain, with the SAME raw world GT re-expressed in
    the exact ego-lidar frame (fixed in v2; v1 mixed coordinate frames here)
C = old commented drone-special branch AS WRITTEN (diagnostic only)
D = direct WORLD-GT -> CAMERA projection, completely bypassing the ego frame

Overlay colors:
  RED   current-timestamp GT
  GREEN delayed-timestamp GT

V2 key change:
  The previous B panel projected dataset/additive-ego GT using an exact-ego
  camera transform, which mixed two different ego frame definitions. V2 fixes
  that and adds D, the decisive direct world->camera projection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


def _find_repo_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    candidates.extend(Path(__file__).resolve().parents)
    for p in candidates:
        if (p / "opencood").is_dir():
            return p
    raise RuntimeError("Run from the Airv2x_gs/AirV2X-Perception_gaussian repo root.")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_pickle
from opencood.models.gaussian_modules_0822.base.projection import (
    attach_camera_geometry,
    lift_optical_z_to_lidar,
    project_lidar_to_image,
)
from opencood.utils import airv2x_utils, box_utils, camera_utils
from opencood.utils.transformation_utils import get_abs_world_pose, x1_to_x2, x_to_world


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def as_np(x: Any, dtype=np.float64) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def transform_points(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    shape = xyz.shape
    pts = xyz.reshape(-1, 3)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    out = (np.asarray(T, dtype=np.float64) @ pts_h.T).T[:, :3]
    return out.reshape(shape)


def rotation_angle_deg(R: np.ndarray) -> float:
    v = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(v, -1.0, 1.0))))


def transform_delta(T_ref: np.ndarray, T_test: np.ndarray) -> Dict[str, float]:
    delta = np.linalg.inv(T_ref) @ T_test
    return {
        "translation_m": float(np.linalg.norm(delta[:3, 3])),
        "rotation_deg": rotation_angle_deg(delta[:3, :3]),
        "matrix_max_abs": float(np.max(np.abs(T_ref - T_test))),
    }


def pose_exact_world(body_pose: Sequence[float], rel_pose: Sequence[float]) -> np.ndarray:
    return as_np(x_to_world(body_pose)) @ as_np(x_to_world(rel_pose))


def pose_additive_world(body_pose: Sequence[float], rel_pose: Sequence[float]) -> np.ndarray:
    return as_np(x_to_world(get_abs_world_pose(rel_pose, body_pose)))


def optical_to_ue4_matrix() -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    return T


def infer_camera_key(meta: Mapping[str, Any]) -> str:
    if meta.get("agent_type") == "drone" and "bev_camera" in meta:
        return "bev_camera"
    for k, v in meta.items():
        if "camera" in str(k) and isinstance(v, Mapping) and "intrinsic" in v:
            return str(k)
    raise KeyError("No camera entry with intrinsic found")


def find_ego(base_data: Mapping[Any, Mapping[str, Any]]):
    for cav_id, cav in base_data.items():
        if bool(cav.get("ego", False)):
            return cav_id, cav
    raise RuntimeError("No ego CAV found")


def timestamp_dir_from_metadata(metadata_path: str) -> Path:
    return Path(metadata_path).resolve().parent.parent


def load_delayed_objects(metadata_path: str):
    p = timestamp_dir_from_metadata(metadata_path) / "objects.pkl"
    if not p.is_file():
        raise FileNotFoundError(f"Delayed objects.pkl not found: {p}")
    return load_pickle(str(p))


def get_gt_for_objects(dataset, source_cav, ego_lidar_pose, objects=None):
    cav = copy.deepcopy(source_cav)
    if objects is not None:
        cav["params"] = dict(cav["params"])
        cav["params"]["objects"] = airv2x_utils.filter_objects(objects)
    centers, mask, ids, class_ids = dataset.post_processor.generate_object_center_airv2x(
        [cav], ego_lidar_pose
    )
    return np.asarray(centers), np.asarray(mask), list(ids), list(class_ids)


def centers_to_corners(centers: np.ndarray, mask: np.ndarray, order: str) -> np.ndarray:
    valid = np.asarray(mask).astype(bool)
    c = np.asarray(centers)[valid]
    if len(c) == 0:
        return np.zeros((0, 8, 3), dtype=np.float64)
    corners = box_utils.boxes_to_corners_3d(torch.as_tensor(c, dtype=torch.float32), order=order)
    return as_np(corners)


def _lookup_object(objects: Mapping[Any, Any], object_id: Any) -> Mapping[str, Any]:
    if object_id in objects:
        return objects[object_id]
    s = str(object_id)
    for k, v in objects.items():
        if str(k) == s:
            return v
    raise KeyError(f"object id {object_id!r} not found in raw objects.pkl")


def world_corners_from_object_ids(
    objects: Mapping[Any, Any],
    object_ids: Sequence[Any],
) -> np.ndarray:
    """Build raw 8-corner boxes directly in WORLD coordinates.

    This deliberately bypasses generate_object_center_airv2x(), ego lidar pose,
    pairwise matrices, and Gaussian geometry.
    """
    out = []
    for object_id in object_ids:
        obj = _lookup_object(objects, object_id)
        location = obj["location"][:3]
        center = obj["center"]
        extent = obj["extent"]
        rotation = [obj["location"][3], obj["location"][4], obj["location"][5]]
        object_pose = [
            float(location[0]) + float(center[0]),
            float(location[1]) + float(center[1]),
            float(location[2]) + float(center[2]),
            float(rotation[0]), float(rotation[1]), float(rotation[2]),
        ]
        T_world_object = as_np(x_to_world(object_pose))
        bbx = box_utils.create_bbx(extent).T  # [3,8], same primitive as project_world_objects_airv2x
        bbx_h = np.concatenate(
            [bbx, np.ones((1, bbx.shape[1]), dtype=np.float64)], axis=0
        )
        out.append((T_world_object @ bbx_h).T[:, :3])
    if not out:
        return np.zeros((0, 8, 3), dtype=np.float64)
    return np.stack(out, axis=0)


def object_world_centers_from_ids(
    objects: Mapping[Any, Any], object_ids: Sequence[Any]
) -> np.ndarray:
    centers = []
    for object_id in object_ids:
        obj = _lookup_object(objects, object_id)
        location = obj["location"][:3]
        center = obj["center"]
        centers.append([
            float(location[0]) + float(center[0]),
            float(location[1]) + float(center[1]),
            float(location[2]) + float(center[2]),
        ])
    return np.asarray(centers, dtype=np.float64).reshape(-1, 3)


def gt_ego_center_back_to_world_error(
    centers_ego: np.ndarray,
    mask: np.ndarray,
    object_ids: Sequence[Any],
    objects_raw: Mapping[Any, Any],
    T_world_from_ego: np.ndarray,
) -> Dict[str, float]:
    valid_centers = np.asarray(centers_ego, dtype=np.float64)[
        np.asarray(mask).astype(bool)
    ][:, :3]
    n = min(len(valid_centers), len(object_ids))
    if n == 0:
        return {"n": 0, "mean_m": float("nan"), "median_m": float("nan"), "p90_m": float("nan"), "max_m": float("nan")}
    world_from_dataset = transform_points(T_world_from_ego, valid_centers[:n])
    world_raw = object_world_centers_from_ids(objects_raw, object_ids[:n])
    d = np.linalg.norm(world_from_dataset - world_raw, axis=-1)
    return {
        "n": int(d.size),
        "mean_m": float(np.mean(d)),
        "median_m": float(np.median(d)),
        "p90_m": float(np.percentile(d, 90)),
        "max_m": float(np.max(d)),
    }


def project_xyz_world_to_raw_image(xyz_world, T_world_from_camopt, K):
    xyz_cam = transform_points(np.linalg.inv(T_world_from_camopt), xyz_world)
    z = xyz_cam[..., 2]
    flat = xyz_cam.reshape(-1, 3)
    uvh = (np.asarray(K, dtype=np.float64) @ flat.T).T
    denom = uvh[:, 2:3].copy()
    denom[np.abs(denom) < 1e-9] = 1e-9
    uv = uvh[:, :2] / denom
    return uv.reshape(*xyz_cam.shape[:-1], 2), z


def projection_difference_mixed(corners_a, projector_a, corners_b, projector_b):
    if corners_a.shape != corners_b.shape or corners_a.size == 0:
        return {"n": 0, "median_px": float("nan"), "p90_px": float("nan"), "max_px": float("nan")}
    uv_a, z_a = projector_a(corners_a)
    uv_b, z_b = projector_b(corners_b)
    valid = (
        (z_a > 0.1) & (z_b > 0.1)
        & np.isfinite(uv_a).all(axis=-1)
        & np.isfinite(uv_b).all(axis=-1)
    )
    if not np.any(valid):
        return {"n": 0, "median_px": float("nan"), "p90_px": float("nan"), "max_px": float("nan")}
    d = np.linalg.norm(uv_a[valid] - uv_b[valid], axis=-1)
    return {
        "n": int(d.size),
        "median_px": float(np.median(d)),
        "p90_px": float(np.percentile(d, 90)),
        "max_px": float(np.max(d)),
    }


def project_xyz_ego_to_raw_image(xyz_ego, T_ego_from_camopt, K):
    xyz_cam = transform_points(np.linalg.inv(T_ego_from_camopt), xyz_ego)
    z = xyz_cam[..., 2]
    flat = xyz_cam.reshape(-1, 3)
    uvh = (np.asarray(K, dtype=np.float64) @ flat.T).T
    denom = uvh[:, 2:3].copy()
    denom[np.abs(denom) < 1e-9] = 1e-9
    uv = uvh[:, :2] / denom
    return uv.reshape(*xyz_cam.shape[:-1], 2), z


def projection_difference_px(corners, T_a, T_b, K):
    if corners.size == 0:
        return {"n": 0, "median_px": float("nan"), "p90_px": float("nan"), "max_px": float("nan")}
    uv_a, z_a = project_xyz_ego_to_raw_image(corners, T_a, K)
    uv_b, z_b = project_xyz_ego_to_raw_image(corners, T_b, K)
    valid = (z_a > 0.1) & (z_b > 0.1) & np.isfinite(uv_a).all(-1) & np.isfinite(uv_b).all(-1)
    if not np.any(valid):
        return {"n": 0, "median_px": float("nan"), "p90_px": float("nan"), "max_px": float("nan")}
    d = np.linalg.norm(uv_a[valid] - uv_b[valid], axis=-1)
    return {
        "n": int(d.size),
        "median_px": float(np.median(d)),
        "p90_px": float(np.percentile(d, 90)),
        "max_px": float(np.max(d)),
    }


def temporal_gt_displacement(cur_centers, cur_mask, cur_ids, del_centers, del_mask, del_ids):
    cc = np.asarray(cur_centers)[np.asarray(cur_mask).astype(bool)]
    dc = np.asarray(del_centers)[np.asarray(del_mask).astype(bool)]
    cm = {str(obj_id): cc[i] for i, obj_id in enumerate(cur_ids[:len(cc)])}
    dm = {str(obj_id): dc[i] for i, obj_id in enumerate(del_ids[:len(dc)])}
    common = sorted(set(cm).intersection(dm))
    if not common:
        return {
            "common_objects": 0,
            "median_center_shift_m": float("nan"),
            "p90_center_shift_m": float("nan"),
            "max_center_shift_m": float("nan"),
        }
    d = np.asarray([np.linalg.norm(cm[k][:3] - dm[k][:3]) for k in common])
    return {
        "common_objects": int(len(d)),
        "median_center_shift_m": float(np.median(d)),
        "p90_center_shift_m": float(np.percentile(d, 90)),
        "max_center_shift_m": float(np.max(d)),
    }


def draw_projected_boxes_bgr(img_bgr, corners, T, K, color, thickness=2):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    if corners.size == 0:
        return out
    uv, z = project_xyz_ego_to_raw_image(corners, T, K)
    rect = (0, 0, int(w), int(h))
    for b in range(corners.shape[0]):
        for i, j in BOX_EDGES:
            if z[b, i] <= 0.1 or z[b, j] <= 0.1:
                continue
            p1, p2 = uv[b, i], uv[b, j]
            if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
                continue
            q1 = (int(np.clip(round(float(p1[0])), -1000000, 1000000)),
                  int(np.clip(round(float(p1[1])), -1000000, 1000000)))
            q2 = (int(np.clip(round(float(p2[0])), -1000000, 1000000)),
                  int(np.clip(round(float(p2[1])), -1000000, 1000000)))
            ok, r1, r2 = cv2.clipLine(rect, q1, q2)
            if ok:
                cv2.line(out, r1, r2, color, thickness, cv2.LINE_AA)
    return out


def draw_projected_world_boxes_bgr(
    img_bgr, corners_world, T_world_from_camopt, K, color, thickness=2
):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    if corners_world.size == 0:
        return out
    uv, z = project_xyz_world_to_raw_image(corners_world, T_world_from_camopt, K)
    rect = (0, 0, int(w), int(h))
    for b in range(corners_world.shape[0]):
        for i, j in BOX_EDGES:
            if z[b, i] <= 0.1 or z[b, j] <= 0.1:
                continue
            p1, p2 = uv[b, i], uv[b, j]
            if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
                continue
            q1 = (int(np.clip(round(float(p1[0])), -1000000, 1000000)),
                  int(np.clip(round(float(p1[1])), -1000000, 1000000)))
            q2 = (int(np.clip(round(float(p2[0])), -1000000, 1000000)),
                  int(np.clip(round(float(p2[1])), -1000000, 1000000)))
            ok, r1, r2 = cv2.clipLine(rect, q1, q2)
            if ok:
                cv2.line(out, r1, r2, color, thickness, cv2.LINE_AA)
    return out


def make_world_panel(
    rgb, current_world_corners, delayed_world_corners,
    T_world_from_camopt, K, title, subtitle
):
    panel = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    panel = draw_projected_world_boxes_bgr(
        panel, current_world_corners, T_world_from_camopt, K, (0, 0, 255), 2
    )
    panel = draw_projected_world_boxes_bgr(
        panel, delayed_world_corners, T_world_from_camopt, K, (0, 255, 0), 2
    )
    bar_h = 60
    canvas = np.zeros((panel.shape[0] + bar_h, panel.shape[1], 3), dtype=np.uint8)
    canvas[bar_h:] = panel
    cv2.putText(canvas, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210,210,210), 1, cv2.LINE_AA)
    cv2.line(canvas, (canvas.shape[1]-220, 18), (canvas.shape[1]-194, 18), (0,0,255), 3)
    cv2.putText(canvas, "current GT", (canvas.shape[1]-188, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
    cv2.line(canvas, (canvas.shape[1]-220, 44), (canvas.shape[1]-194, 44), (0,255,0), 3)
    cv2.putText(canvas, "delayed GT", (canvas.shape[1]-188, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
    return canvas


def make_panel(rgb, current_corners, delayed_corners, T, K, title, subtitle):
    panel = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    panel = draw_projected_boxes_bgr(panel, current_corners, T, K, (0, 0, 255), 2)
    panel = draw_projected_boxes_bgr(panel, delayed_corners, T, K, (0, 255, 0), 2)
    bar_h = 60
    canvas = np.zeros((panel.shape[0] + bar_h, panel.shape[1], 3), dtype=np.uint8)
    canvas[bar_h:] = panel
    cv2.putText(canvas, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210,210,210), 1, cv2.LINE_AA)
    cv2.line(canvas, (canvas.shape[1]-220, 18), (canvas.shape[1]-194, 18), (0,0,255), 3)
    cv2.putText(canvas, "current GT", (canvas.shape[1]-188, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
    cv2.line(canvas, (canvas.shape[1]-220, 44), (canvas.shape[1]-194, 44), (0,255,0), 3)
    cv2.putText(canvas, "delayed GT", (canvas.shape[1]-188, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
    return canvas


def build_drone_chains(drone_base, ego_base):
    params = drone_base["params"]
    delayed_meta = load_pickle(drone_base["metadata_path"])
    ego_meta = load_pickle(ego_base["metadata_path"])
    cam_key = infer_camera_key(delayed_meta)
    cam = delayed_meta[cam_key]

    K = as_np(cam["intrinsic"])
    E_raw_meta = as_np(cam["extrinsic"])
    E_raw_param = as_np(params["delay_extrinsic"]).reshape(-1, 4, 4)[0]

    # A: exact chain currently used by dataset/Gaussian.
    cam2agent_lss = as_np(camera_utils.ue4_to_lss(E_raw_param))
    T_agent_to_ego_current = as_np(params["transformation_matrix"])
    T_A = T_agent_to_ego_current @ cam2agent_lss

    T_opt_to_ue4 = optical_to_ue4_matrix()

    drone_body_pose = delayed_meta["odometry"]["ego_pos"]
    drone_lidar_rel = delayed_meta["lidar"]["lidar_pose"]
    drone_cam_rel = cam["cords"]
    ego_body_pose = ego_meta["odometry"]["ego_pos"]
    ego_lidar_rel = ego_meta["lidar"]["lidar_pose"]

    # B: proper rigid composition.
    Tw_drone_lidar_exact = pose_exact_world(drone_body_pose, drone_lidar_rel)
    Tw_drone_cam_exact = pose_exact_world(drone_body_pose, drone_cam_rel)
    Tw_ego_lidar_exact = pose_exact_world(ego_body_pose, ego_lidar_rel)

    T_agent_to_ego_exact = np.linalg.inv(Tw_ego_lidar_exact) @ Tw_drone_lidar_exact
    E_lidar_to_cam_exact = np.linalg.inv(Tw_drone_cam_exact) @ Tw_drone_lidar_exact
    T_camopt_to_agent_exact = np.linalg.inv(E_lidar_to_cam_exact) @ T_opt_to_ue4
    Tw_camopt_exact = Tw_drone_cam_exact @ T_opt_to_ue4
    # B uses exact ego frame. V2 will also express GT in this exact frame.
    T_B = np.linalg.inv(Tw_ego_lidar_exact) @ Tw_camopt_exact
    T_B_via_agent = T_agent_to_ego_exact @ T_camopt_to_agent_exact

    # D(raw): camera optical->WORLD reconstructed using raw lidar->camera extrinsic.
    Tw_camopt_from_raw = (
        Tw_drone_lidar_exact @ np.linalg.inv(E_raw_param) @ T_opt_to_ue4
    )

    # Additive get_abs_world_pose reconstruction.
    Tw_drone_lidar_add = pose_additive_world(drone_body_pose, drone_lidar_rel)
    Tw_drone_cam_add = pose_additive_world(drone_body_pose, drone_cam_rel)
    Tw_ego_lidar_add = pose_additive_world(ego_body_pose, ego_lidar_rel)
    T_agent_to_ego_add = np.linalg.inv(Tw_ego_lidar_add) @ Tw_drone_lidar_add
    E_lidar_to_cam_add = np.linalg.inv(Tw_drone_cam_add) @ Tw_drone_lidar_add

    # Re-express A in WORLD so A and D have the same destination frame.
    Tw_camopt_from_A = Tw_ego_lidar_add @ T_A

    # C: old commented drone special branch AS WRITTEN.
    cam_world_add_pose = get_abs_world_pose(drone_cam_rel, drone_body_pose)
    ego_lidar_world_add_pose = get_abs_world_pose(ego_lidar_rel, ego_body_pose)
    special_raw = as_np(x1_to_x2(cam_world_add_pose, drone_body_pose))
    special_cam2body_lss = as_np(camera_utils.ue4_to_lss(special_raw))
    special_body_to_ego = as_np(x1_to_x2(drone_body_pose, ego_lidar_world_add_pose))
    T_C = special_body_to_ego @ special_cam2body_lss
    Tw_camopt_from_C = Tw_ego_lidar_add @ T_C

    return {
        "K": K,
        "camera_key": cam_key,
        "E_raw_meta": E_raw_meta,
        "E_raw_param": E_raw_param,
        "cam2agent_lss_current": cam2agent_lss,
        "T_agent_to_ego_current": T_agent_to_ego_current,
        "T_agent_to_ego_exact": T_agent_to_ego_exact,
        "T_agent_to_ego_add": T_agent_to_ego_add,
        "E_lidar_to_cam_exact": E_lidar_to_cam_exact,
        "E_lidar_to_cam_add": E_lidar_to_cam_add,
        "Tw_drone_lidar_exact": Tw_drone_lidar_exact,
        "Tw_drone_lidar_add": Tw_drone_lidar_add,
        "Tw_drone_cam_exact": Tw_drone_cam_exact,
        "Tw_drone_cam_add": Tw_drone_cam_add,
        "Tw_ego_lidar_exact": Tw_ego_lidar_exact,
        "Tw_ego_lidar_add": Tw_ego_lidar_add,
        "Tw_camopt_exact": Tw_camopt_exact,
        "Tw_camopt_from_raw": Tw_camopt_from_raw,
        "Tw_camopt_from_A": Tw_camopt_from_A,
        "Tw_camopt_from_C": Tw_camopt_from_C,
        "T_A_current": T_A,
        "T_B_exact": T_B,
        "T_B_via_agent": T_B_via_agent,
        "T_C_old_special_as_written": T_C,
        "drone_body_pose": list(map(float, drone_body_pose)),
        "drone_lidar_rel": list(map(float, drone_lidar_rel)),
        "drone_cam_rel": list(map(float, drone_cam_rel)),
        "ego_body_pose": list(map(float, ego_body_pose)),
        "ego_lidar_rel": list(map(float, ego_lidar_rel)),
    }


def actual_gaussian_geometry(batch_ego):
    attach_camera_geometry(batch_ego)
    payload = batch_ego["drone"]
    geom = payload["camera_geometry"]
    return {
        "geometry": geom,
        "cav_ids": list(payload.get("cav_ids", [])),
        "n_flat": int(geom["n_flat"]),
        "n_cav": int(geom.get("n_cav", 0)),
        "n_views": int(geom["n_views"]),
    }


def lss_gaussian_lift_check(geom, flat_view_index):
    vi = int(flat_view_index)
    image_h, image_w = tuple(int(v) for v in geom["image_hw"])
    uv = torch.tensor(
        [
            [0.25*image_w, 0.25*image_h],
            [0.50*image_w, 0.50*image_h],
            [0.75*image_w, 0.25*image_h],
            [0.25*image_w, 0.75*image_h],
            [0.75*image_w, 0.75*image_h],
        ], dtype=torch.float64
    )
    depth = torch.tensor([10., 30., 60., 90., 120.], dtype=torch.float64)
    n = len(uv)
    K = geom["intrinsics"][vi].detach().cpu().to(torch.float64).unsqueeze(0).expand(n,-1,-1)
    R = geom["cam2lidar_rot"][vi].detach().cpu().to(torch.float64).unsqueeze(0).expand(n,-1,-1)
    t = geom["cam2lidar_trans"][vi].detach().cpu().to(torch.float64).unsqueeze(0).expand(n,-1)
    pR = geom["post_rots"][vi].detach().cpu().to(torch.float64).unsqueeze(0).expand(n,-1,-1)
    pt = geom["post_trans"][vi].detach().cpu().to(torch.float64).unsqueeze(0).expand(n,-1)

    mean_g, _, _ = lift_optical_z_to_lidar(uv, depth, K, R, t, pR, pt)

    frustum = torch.stack([uv[:,0], uv[:,1], depth], dim=-1)
    points = frustum - pt
    points = torch.matmul(torch.linalg.inv(pR), points.unsqueeze(-1)).squeeze(-1)
    x_img = torch.cat([points[:,:2]*points[:,2:3], points[:,2:3]], dim=-1)
    mean_lss = torch.matmul(torch.matmul(R, torch.linalg.inv(K)), x_img.unsqueeze(-1)).squeeze(-1) + t

    uv_rt, z_rt, front = project_lidar_to_image(mean_g, K, R, t, pR, pt)
    return {
        "gaussian_vs_lss_max_abs_m": float((mean_g-mean_lss).abs().max().item()),
        "gaussian_vs_lss_mean_l2_m": float(torch.linalg.norm(mean_g-mean_lss, dim=-1).mean().item()),
        "lift_project_roundtrip_max_px": float(torch.linalg.norm(uv_rt-uv, dim=-1).max().item()),
        "lift_project_roundtrip_max_depth_m": float((z_rt-depth).abs().max().item()),
        "front_fraction": float(front.float().mean().item()),
    }


def sanitize_json(x):
    if isinstance(x, dict):
        return {str(k): sanitize_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sanitize_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, np.integer):
        x = int(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def fmt_delta(name, d):
    return (f"{name}: trans={d['translation_m']:.6f} m, "
            f"rot={d['rotation_deg']:.6f} deg, max|dT|={d['matrix_max_abs']:.6g}")


def build_hints(delay_frames, checks, temporal, proj_diff):
    hints = []
    if delay_frames > 0:
        hints.append(
            f"time_delay={delay_frames} frame(s): RGB/depth are delayed while dataset GT is current; "
            "red(current GT) and green(delayed GT) can differ for moving objects."
        )
    else:
        hints.append("time_delay=0: current-GT vs delayed-RGB temporal mismatch is not the cause.")

    ga = checks["gaussian_agent_to_ego_vs_A"]
    gc = checks["gaussian_cam2agent_vs_A"]
    if ga["matrix_max_abs"] < 1e-4 and gc["matrix_max_abs"] < 1e-6:
        hints.append("Gaussian consumes the same agent_to_ego and cam2agent geometry as dataset chain A.")
    else:
        hints.append("Gaussian geometry does not exactly match A; inspect retained-CAV/view ordering.")

    aw = checks["A_world_reconstructed_vs_D_pose_exact"]
    if aw["translation_m"] < 1e-4 and aw["rotation_deg"] < 1e-4:
        hints.append(
            "A reconstructed back into WORLD matches D. The additive-vs-exact ego-frame difference "
            "cancels when GT and camera use the same frame."
        )
    else:
        hints.append("A reconstructed into WORLD differs from D; current ego/agent composition is inconsistent.")

    bd = proj_diff.get("B_corrected_vs_D_world_current_GT", {})
    if bd.get("n", 0) > 0 and bd.get("max_px", float("inf")) < 1e-3:
        hints.append("Corrected B and direct WORLD->camera D agree numerically; the v2 exact-SE3 test is frame-consistent.")
    elif bd.get("n", 0) > 0:
        hints.append("Corrected B and D disagree; inspect v2 world/ego conversion or raw object reconstruction.")

    ad = proj_diff.get("A_vs_D_world_current_GT", {})
    if ad.get("n", 0) > 0:
        if ad.get("median_px", float("inf")) < 0.5:
            hints.append(
                "A and direct WORLD->camera D project GT to nearly identical pixels. If both are visibly shifted "
                "from RGB vehicles, the mismatch is upstream in raw object annotation vs rendered RGB/camera metadata."
            )
        else:
            hints.append(
                f"A vs D differs by median {ad['median_px']:.3f}px: ego GT conversion / frame construction affects projection."
            )

    rex = checks["raw_extrinsic_vs_pose_exact"]
    if rex["translation_m"] < 1e-5 and rex["rotation_deg"] < 1e-4:
        hints.append("Raw lidar->camera extrinsic agrees with pose-derived exact camera geometry.")
    else:
        hints.append("Raw lidar->camera extrinsic disagrees with pose-derived exact geometry; camera metadata is a strong suspect.")

    gtback = checks["current_GT_ego_center_back_to_world"]
    if gtback.get("n", 0) > 0:
        if gtback.get("median_m", float("inf")) < 1e-3:
            hints.append("Dataset current GT centers map back to raw objects.pkl world centers correctly.")
        else:
            hints.append(
                f"Dataset GT ego conversion is inconsistent with raw objects.pkl: median world-center error={gtback['median_m']:.4f}m."
            )

    med = temporal.get("median_center_shift_m", float("nan"))
    if temporal.get("common_objects", 0) > 0 and np.isfinite(med) and med > 0.2:
        hints.append(f"Matched current/delayed GT move by median {med:.3f} m; temporal offset can visibly shift boxes.")
    return hints

def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-dir", type=str, help="Log dir containing config.yaml; checkpoint not needed.")
    src.add_argument("--hypes", type=str, help="Explicit YAML config path.")
    p.add_argument("--index", type=int, default=35)
    p.add_argument("--split", choices=["validate", "test"], default="validate")
    p.add_argument("--out-dir", type=str, default="opencood/logs/diagnose_drone_geometry_chain_v2")
    p.add_argument("--no-images", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model_dir:
        hypes = yaml_utils.load_yaml(None, model_dir=args.model_dir)
    else:
        hypes = yaml_utils.load_yaml(args.hypes)

    if args.split == "test":
        if "test_dir" not in hypes:
            raise KeyError("Config has no test_dir")
        hypes["validate_dir"] = hypes["test_dir"]

    dataset = build_dataset(hypes, visualize=False, train=False)
    if not (0 <= args.index < len(dataset)):
        raise IndexError(f"index {args.index} outside [0, {len(dataset)-1}]")

    # Actual collated tensors consumed by Gaussian projection.py.
    sample = dataset[args.index]
    batch = dataset.collate_batch_test([sample])
    batch_ego = batch["ego"]
    gauss_info = actual_gaussian_geometry(batch_ego)

    # Raw current/delayed dataset records for the same eval index.
    base_data, scenario_index, timestamp_key = dataset.retrieve_base_data(
        args.index, cur_ego_pos_flag=dataset.cur_ego_pose_flag
    )
    ego_id, ego_base = find_ego(base_data)
    ego_lidar_pose = ego_base["params"]["delay_ego_lidar_pose"]

    drone_ids = list(gauss_info["cav_ids"])
    if not drone_ids:
        raise RuntimeError(f"index {args.index} has no retained drone CAVs")

    cur_centers, cur_mask, cur_ids, _ = get_gt_for_objects(dataset, ego_base, ego_lidar_pose)
    order = hypes["postprocess"]["order"]
    cur_corners = centers_to_corners(cur_centers, cur_mask, order)

    # V2: raw WORLD GT directly from current objects.pkl.
    current_objects_raw = load_delayed_objects(ego_base["metadata_path"])
    cur_world_corners = world_corners_from_object_ids(current_objects_raw, cur_ids)

    geom = gauss_info["geometry"]
    n_views = gauss_info["n_views"]

    summary = {
        "repo_root": str(REPO_ROOT),
        "index": int(args.index),
        "scenario_index": int(scenario_index),
        "timestamp_key_current": str(timestamp_key),
        "split": args.split,
        "ego_id": str(ego_id),
        "agent_order": list(batch_ego.get("agent_order", [])),
        "drone_cav_ids": [str(x) for x in drone_ids],
        "drone_n_cav": int(gauss_info["n_cav"]),
        "drone_n_views": int(n_views),
        "drone_n_flat": int(gauss_info["n_flat"]),
        "config_async": bool(hypes.get("wild_setting", {}).get("async", False)),
        "config_async_mode": hypes.get("wild_setting", {}).get("async_mode", "sim"),
        "config_async_overhead_ms": hypes.get("wild_setting", {}).get("async_overhead", 0),
        "config_loc_err": bool(hypes.get("wild_setting", {}).get("loc_err", False)),
        "drones": {},
    }

    lines = [
        "="*88,
        "AirV2X DRONE GEOMETRY DIAGNOSTIC V2",
        "="*88,
        f"index={args.index} scenario={scenario_index} current_timestamp={timestamp_key} ego_id={ego_id}",
        f"retained drones={drone_ids} n_views={n_views} async={summary['config_async']} loc_err={summary['config_loc_err']}",
        "",
    ]

    for cav_idx, cav_id in enumerate(drone_ids):
        base_key = cav_id if cav_id in base_data else next((k for k in base_data if str(k)==str(cav_id)), None)
        if base_key is None:
            raise KeyError(f"retained drone {cav_id} not found in raw base_data")
        drone = base_data[base_key]
        delay_frames = int(drone.get("time_delay", 0))
        chains = build_drone_chains(drone, ego_base)
        K = chains["K"]

        delayed_objects = load_delayed_objects(drone["metadata_path"])
        del_centers, del_mask, del_ids, _ = get_gt_for_objects(
            dataset, drone, ego_lidar_pose, objects=delayed_objects
        )
        del_corners = centers_to_corners(del_centers, del_mask, order)
        del_world_corners = world_corners_from_object_ids(delayed_objects, del_ids)

        # V2 corrected B: exact world GT -> exact ego frame before T_B projection.
        cur_corners_exact_ego = transform_points(
            np.linalg.inv(chains["Tw_ego_lidar_exact"]), cur_world_corners
        )
        del_corners_exact_ego = transform_points(
            np.linalg.inv(chains["Tw_ego_lidar_exact"]), del_world_corners
        )

        flat_vi = cav_idx * n_views
        if flat_vi >= gauss_info["n_flat"]:
            raise RuntimeError(f"flat view index {flat_vi} >= n_flat {gauss_info['n_flat']}")
        T_gauss_agent_to_ego = as_np(geom["agent_to_ego"][flat_vi])
        T_gauss_cam2agent = as_np(geom["cam2lidar"][flat_vi])

        checks = {
            "gaussian_agent_to_ego_vs_A": transform_delta(
                chains["T_agent_to_ego_current"], T_gauss_agent_to_ego
            ),
            "gaussian_cam2agent_vs_A": transform_delta(
                chains["cam2agent_lss_current"], T_gauss_cam2agent
            ),
            "B_direct_vs_B_via_agent": transform_delta(
                chains["T_B_exact"], chains["T_B_via_agent"]
            ),
            # Valid same-destination WORLD comparisons in V2.
            "A_world_reconstructed_vs_D_pose_exact": transform_delta(
                chains["Tw_camopt_exact"], chains["Tw_camopt_from_A"]
            ),
            "D_raw_extrinsic_world_vs_D_pose_exact": transform_delta(
                chains["Tw_camopt_exact"], chains["Tw_camopt_from_raw"]
            ),
            "C_world_reconstructed_vs_D_pose_exact": transform_delta(
                chains["Tw_camopt_exact"], chains["Tw_camopt_from_C"]
            ),
            "raw_param_vs_raw_metadata": transform_delta(
                chains["E_raw_meta"], chains["E_raw_param"]
            ),
            "raw_extrinsic_vs_pose_exact": transform_delta(
                chains["E_lidar_to_cam_exact"], chains["E_raw_param"]
            ),
            "raw_extrinsic_vs_pose_additive": transform_delta(
                chains["E_lidar_to_cam_add"], chains["E_raw_param"]
            ),
            "drone_lidar_world_additive_vs_exact": transform_delta(
                chains["Tw_drone_lidar_exact"], chains["Tw_drone_lidar_add"]
            ),
            "drone_camera_world_additive_vs_exact": transform_delta(
                chains["Tw_drone_cam_exact"], chains["Tw_drone_cam_add"]
            ),
            "ego_lidar_world_additive_vs_exact": transform_delta(
                chains["Tw_ego_lidar_exact"], chains["Tw_ego_lidar_add"]
            ),
            "current_GT_ego_center_back_to_world": gt_ego_center_back_to_world_error(
                cur_centers, cur_mask, cur_ids, current_objects_raw,
                chains["Tw_ego_lidar_add"],
            ),
        }

        lift_check = lss_gaussian_lift_check(geom, flat_vi)
        proj_diff = {
            "A_vs_D_world_current_GT": projection_difference_mixed(
                cur_corners,
                lambda x: project_xyz_ego_to_raw_image(x, chains["T_A_current"], K),
                cur_world_corners,
                lambda x: project_xyz_world_to_raw_image(x, chains["Tw_camopt_exact"], K),
            ),
            "B_corrected_vs_D_world_current_GT": projection_difference_mixed(
                cur_corners_exact_ego,
                lambda x: project_xyz_ego_to_raw_image(x, chains["T_B_exact"], K),
                cur_world_corners,
                lambda x: project_xyz_world_to_raw_image(x, chains["Tw_camopt_exact"], K),
            ),
            "C_vs_D_world_current_GT": projection_difference_mixed(
                cur_corners,
                lambda x: project_xyz_ego_to_raw_image(x, chains["T_C_old_special_as_written"], K),
                cur_world_corners,
                lambda x: project_xyz_world_to_raw_image(x, chains["Tw_camopt_exact"], K),
            ),
            "A_vs_D_world_delayed_GT": projection_difference_mixed(
                del_corners,
                lambda x: project_xyz_ego_to_raw_image(x, chains["T_A_current"], K),
                del_world_corners,
                lambda x: project_xyz_world_to_raw_image(x, chains["Tw_camopt_exact"], K),
            ),
            "D_pose_vs_D_raw_current_GT": projection_difference_mixed(
                cur_world_corners,
                lambda x: project_xyz_world_to_raw_image(x, chains["Tw_camopt_exact"], K),
                cur_world_corners,
                lambda x: project_xyz_world_to_raw_image(x, chains["Tw_camopt_from_raw"], K),
            ),
        }
        temporal = temporal_gt_displacement(
            cur_centers, cur_mask, cur_ids, del_centers, del_mask, del_ids
        )
        hints = build_hints(delay_frames, checks, temporal, proj_diff)

        rgb = np.asarray(drone["cameras"][0].convert("RGB"))
        h, w = rgb.shape[:2]
        if not args.no_images:
            subtitle = f"delay={delay_frames} frame(s), raw RGB={w}x{h}"
            panels = [
                make_panel(
                    rgb, cur_corners, del_corners, chains["T_A_current"], K,
                    "A: current dataset / Gaussian chain", subtitle
                ),
                make_panel(
                    rgb, cur_corners_exact_ego, del_corners_exact_ego, chains["T_B_exact"], K,
                    "B: exact SE(3), GT in exact ego frame (V2 fixed)", subtitle
                ),
                make_panel(
                    rgb, cur_corners, del_corners, chains["T_C_old_special_as_written"], K,
                    "C: old commented drone branch (as written)", subtitle
                ),
                make_world_panel(
                    rgb, cur_world_corners, del_world_corners, chains["Tw_camopt_exact"], K,
                    "D: DIRECT raw WORLD GT -> camera (decisive)", subtitle
                ),
            ]
            cv2.imwrite(
                str(out_dir / f"drone_{cav_id}_projection_compare_v2.png"),
                np.concatenate(panels, axis=1),
            )
            cv2.imwrite(
                str(out_dir / f"drone_{cav_id}_D_world_direct.png"), panels[3]
            )

        ds = {
            "cav_id": str(cav_id),
            "metadata_path_delayed": str(drone["metadata_path"]),
            "delayed_timestamp_dir": str(timestamp_dir_from_metadata(drone["metadata_path"])),
            "time_delay_frames": delay_frames,
            "distance_to_ego_m": float(drone.get("distance_to_ego", float("nan"))),
            "raw_rgb_hw": [int(h), int(w)],
            "camera_key": chains["camera_key"],
            "body_pose_delay": chains["drone_body_pose"],
            "lidar_rel_pose": chains["drone_lidar_rel"],
            "camera_rel_pose": chains["drone_cam_rel"],
            "ego_body_pose_current": chains["ego_body_pose"],
            "ego_lidar_rel_pose": chains["ego_lidar_rel"],
            "checks": checks,
            "lss_gaussian_lift_check": lift_check,
            "projection_pixel_difference": proj_diff,
            "current_vs_delayed_gt": temporal,
            "hints": hints,
            "matrices": {
                "T_A_current_camopt_to_additive_ego": chains["T_A_current"],
                "T_B_exact_camopt_to_exact_ego": chains["T_B_exact"],
                "T_C_old_special_as_written_camopt_to_additive_ego": chains["T_C_old_special_as_written"],
                "T_D_exact_camopt_to_world": chains["Tw_camopt_exact"],
                "T_D_rawextrinsic_camopt_to_world": chains["Tw_camopt_from_raw"],
                "T_world_from_additive_ego": chains["Tw_ego_lidar_add"],
                "T_world_from_exact_ego": chains["Tw_ego_lidar_exact"],
                "T_gaussian_agent_to_ego": T_gauss_agent_to_ego,
                "T_gaussian_cam2agent": T_gauss_cam2agent,
                "T_agent_to_ego_current": chains["T_agent_to_ego_current"],
                "T_agent_to_ego_exact": chains["T_agent_to_ego_exact"],
                "E_raw_param_lidar_to_cam_ue4": chains["E_raw_param"],
                "E_pose_exact_lidar_to_cam_ue4": chains["E_lidar_to_cam_exact"],
            },
        }
        summary["drones"][str(cav_id)] = ds

        lines += ["-"*88, f"DRONE CAV {cav_id} (V2)", "-"*88]
        lines.append(f"time_delay={delay_frames} frame(s), distance={ds['distance_to_ego_m']:.3f} m")
        lines.append(f"metadata(delay RGB): {drone['metadata_path']}")
        lines.append(f"body pose(delay): {chains['drone_body_pose']}")
        lines.append(f"lidar rel pose:   {chains['drone_lidar_rel']}")
        lines.append(f"camera rel pose:  {chains['drone_cam_rel']}")
        lines.append("\n[matrix / GT-frame checks]")
        for k, v in checks.items():
            if "translation_m" in v:
                lines.append("  " + fmt_delta(k, v))
            else:
                lines.append(
                    f"  {k}: n={v.get('n', 0)} "
                    f"mean={v.get('mean_m', float('nan')):.6f}m "
                    f"median={v.get('median_m', float('nan')):.6f}m "
                    f"p90={v.get('p90_m', float('nan')):.6f}m "
                    f"max={v.get('max_m', float('nan')):.6f}m"
                )
        lines.append("\n[Gaussian lift vs normal LSS]")
        for k, v in lift_check.items():
            lines.append(f"  {k}: {v:.9g}")
        lines.append("\n[projection pixel difference: frame-consistent V2]")
        for name, d in proj_diff.items():
            lines.append(f"  {name}: n={d['n']} median={d['median_px']:.3f}px p90={d['p90_px']:.3f}px max={d['max_px']:.3f}px")
        lines.append("\n[current GT vs delayed GT]")
        lines.append(
            "  common_objects={common_objects}, median_center_shift={median_center_shift_m:.3f}m, "
            "p90={p90_center_shift_m:.3f}m, max={max_center_shift_m:.3f}m".format(**temporal)
        )
        lines.append("\n[diagnostic hints]")
        lines.extend("  - " + h for h in hints)
        lines.append("")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(sanitize_json(summary), f, indent=2, ensure_ascii=False)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved: {out_dir / 'summary.txt'}")
    print(f"Saved: {out_dir / 'summary.json'}")
    if not args.no_images:
        print(f"Projection overlays: {out_dir}/drone_*_projection_compare_v2.png")
        print(f"Direct WORLD panels: {out_dir}/drone_*_D_world_direct.png")


if __name__ == "__main__":
    main()

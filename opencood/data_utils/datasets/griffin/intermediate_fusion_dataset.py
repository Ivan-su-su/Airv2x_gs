# -*- coding: utf-8 -*-
"""Minimal raw-Griffin adapter for full Gaussian-0822 detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRot
from torch.utils.data import Dataset

import opencood.data_utils.post_processor as post_processor
from opencood.data_utils.datasets.griffin_p1_dataset import (
    VEHICLE_CAMS,
    _load_calib,
    _read_json,
    _rgb_to_normalized_tensor,
    photometric_distortion_bgr,
    resolve_split_frames,
)

DRONE_CAMS: Tuple[str, ...] = ("bottom",)

# Detector class 0 is background. Griffin official 3 classes are 1..3 here.
_CLASS_ID = {
    "car": 1, "truck": 1, "van": 1, "bus": 1,
    "bicycle": 2, "cyclist": 2, "motorcycle": 2, "motorcyclist": 2,
    "tricyclist": 2, "barrowlist": 2,
    "pedestrian": 3,
}


def _pose_to_matrix(pose: Mapping[str, Any]) -> np.ndarray:
    rot = SciRot.from_euler(
        "xyz",
        [float(pose["roll"]), float(pose["pitch"]), float(pose["yaw"])],
        degrees=True,
    ).as_matrix()
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = [float(pose["x"]), float(pose["y"]), float(pose["z"])]
    return out


def _scaled_intrinsic(
    intrinsic: np.ndarray,
    original_hw: Tuple[int, int],
    final_hw: Tuple[int, int],
) -> np.ndarray:
    orig_h, orig_w = original_hw
    final_h, final_w = final_hw
    out = intrinsic.astype(np.float64, copy=True)
    out[0, 0] *= float(final_w) / float(orig_w)
    out[0, 2] *= float(final_w) / float(orig_w)
    out[1, 1] *= float(final_h) / float(orig_h)
    out[1, 2] *= float(final_h) / float(orig_h)
    return out


def _ideal_ground_depth(
    feat_h: int,
    feat_w: int,
    image_hw: Tuple[int, int],
    intrinsic: np.ndarray,
    t_cam_to_enu: np.ndarray,
    ground_z: float,
    max_depth: float,
) -> np.ndarray:
    """Bottom-camera optical-z from ray / z=ground_z intersection."""
    image_h, image_w = image_hw
    stride_y = float(image_h) / float(feat_h)
    stride_x = float(image_w) / float(feat_w)
    yy, xx = np.meshgrid(
        np.arange(feat_h, dtype=np.float64),
        np.arange(feat_w, dtype=np.float64),
        indexing="ij",
    )
    u = (xx + 0.5) * stride_x
    v = (yy + 0.5) * stride_y
    pix = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
    rays_cam = (np.linalg.inv(intrinsic) @ pix.T).T
    rays_enu = (t_cam_to_enu[:3, :3] @ rays_cam.T).T
    denom = rays_enu[:, 2]
    origin_z = float(t_cam_to_enu[2, 3])
    depth = np.full((pix.shape[0],), np.nan, dtype=np.float32)
    valid = denom < -1.0e-6
    t = np.full_like(denom, np.nan, dtype=np.float64)
    t[valid] = (float(ground_z) - origin_z) / denom[valid]
    valid &= np.isfinite(t) & (t > 0.0) & (t <= float(max_depth))
    depth[valid] = t[valid].astype(np.float32)
    return depth.reshape(feat_h, feat_w)


class IntermediateFusionDatasetGriffin(Dataset):
    """One synchronized vehicle + drone-bottom pair per Griffin frame."""

    def __init__(self, params: Dict[str, Any], visualize: bool, train: bool = True):
        super().__init__()
        self.params = params
        self.visualize = bool(visualize)
        self.train = bool(train)
        self.split = "train" if self.train else "val"

        root_value = (
            params["root_dir"]
            if self.train
            else params.get("validate_dir", params["root_dir"])
        )
        self.root = Path(root_value).expanduser().resolve()
        self.vehicle_root = self.root / "vehicle-side"
        self.drone_root = self.root / "drone-side"

        cfg = params.get("griffin") or {}
        split_json_value = cfg.get("split_json")
        split_json = (
            Path(split_json_value).expanduser().resolve()
            if split_json_value
            else None
        )
        self.final_hw = tuple(int(v) for v in cfg.get("final_dim", [360, 640]))
        self.drone_stride = int(cfg.get("drone_stride", 4))
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.max_ideal_depth = float(cfg.get("max_ideal_depth", 150.0))
        self.photometric = bool(cfg.get("photometric_distortion", True)) and self.train
        self.seed = int(cfg.get("seed", 20260923))

        raw_frames = resolve_split_frames(
            self.vehicle_root, self.split, split_json
        )
        self.frames = [f for f in raw_frames if self._frame_complete(f)]
        if not self.frames:
            raise RuntimeError(f"No usable Griffin frames for split={self.split}")

        self.vehicle_calib = {
            cam: _load_calib(self.vehicle_root, cam) for cam in VEHICLE_CAMS
        }
        self.drone_calib = {
            cam: _load_calib(self.drone_root, cam) for cam in DRONE_CAMS
        }

        self.post_processor = post_processor.build_postprocessor(
            params["postprocess"], dataset="airv2x", train=train
        )
        self.anchor_box = self.post_processor.generate_anchor_box()
        self.max_num = int(params["postprocess"].get("max_num", 300))
        self.det_range = np.asarray(
            params["postprocess"]["anchor_args"]["cav_lidar_range"],
            dtype=np.float32,
        )
        print(
            f"[GriffinDet] split={self.split} frames={len(self.frames)} "
            f"vehicle=4cam drone=bottom final={self.final_hw}"
        )

    def _frame_complete(self, frame: str) -> bool:
        required = [
            self.vehicle_root / "pose" / f"{frame}.json",
            self.drone_root / "pose" / f"{frame}.json",
            self.vehicle_root / "label" / f"{frame}.txt",
            self.drone_root / "camera" / "bottom" / f"{frame}.png",
        ]
        for cam in VEHICLE_CAMS:
            required.append(self.vehicle_root / "camera" / cam / f"{frame}.png")
        return all(path.exists() for path in required)

    def __len__(self) -> int:
        return len(self.frames)

    def _rng(self, index: int, view_index: int) -> np.random.RandomState:
        if self.photometric:
            return np.random.RandomState(np.random.randint(0, 2**31 - 1))
        return np.random.RandomState(self.seed + 37 * index + 1009 * view_index)

    def _camera_input(
        self,
        side_root: Path,
        cam: str,
        frame: str,
        t_agent_to_enu: np.ndarray,
        rng: np.random.RandomState,
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, float]:
        path = side_root / "camera" / cam / f"{frame}.png"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        original_hw = (int(bgr.shape[0]), int(bgr.shape[1]))
        if self.photometric:
            bgr = photometric_distortion_bgr(bgr.astype(np.float32), rng)
        img = _rgb_to_normalized_tensor(bgr, self.final_hw)

        t_cam_to_agent, intrinsic = _load_calib(side_root, cam)
        if intrinsic is None:
            raise ValueError(f"Camera {cam} has no intrinsic")
        k_scaled = _scaled_intrinsic(intrinsic, original_hw, self.final_hw)
        t_cam_to_enu = t_agent_to_enu @ t_cam_to_agent
        return img, k_scaled, t_cam_to_agent, float(t_cam_to_enu[2, 3])

    @staticmethod
    def _pack_cam_inputs(
        imgs: List[torch.Tensor],
        intrinsics: List[np.ndarray],
        extrinsics: List[np.ndarray],
        camera_world_z: List[float],
    ) -> Dict[str, torch.Tensor]:
        ext = torch.from_numpy(np.stack(extrinsics)).float()
        return {
            "imgs": torch.stack(imgs, dim=0),
            "intrinsics": torch.from_numpy(np.stack(intrinsics)).float(),
            "extrinsics": ext,
            "rots": ext[:, :3, :3].clone(),
            "trans": ext[:, :3, 3].clone(),
            "post_rots": torch.eye(3, dtype=torch.float32)
                .unsqueeze(0).repeat(len(imgs), 1, 1),
            "post_trans": torch.zeros((len(imgs), 3), dtype=torch.float32),
            "camera_world_z": torch.tensor(camera_world_z, dtype=torch.float32),
        }

    def _load_boxes(
        self, frame: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        path = self.vehicle_root / "label" / f"{frame}.txt"
        boxes: List[List[float]] = []
        object_ids: List[str] = []
        class_ids: List[int] = []

        for line in path.read_text().splitlines():
            fields = line.strip().split()
            if len(fields) < 10:
                continue
            class_id = _CLASS_ID.get(fields[0].lower())
            if class_id is None:
                continue
            x, y, z = map(float, fields[1:4])
            l, w, h = map(float, fields[4:7])
            yaw_deg = float(fields[9])
            object_id = fields[10] if len(fields) > 10 else f"{frame}_{len(boxes)}"

            if not (
                self.det_range[0] <= x <= self.det_range[3]
                and self.det_range[1] <= y <= self.det_range[4]
                and self.det_range[2] <= z <= self.det_range[5]
            ):
                continue

            # OpenCOOD hwl order: [x,y,z,h,w,l,yaw(rad)].
            boxes.append([x, y, z, h, w, l, np.deg2rad(yaw_deg)])
            object_ids.append(str(object_id))
            class_ids.append(int(class_id))

        center = np.zeros((self.max_num, 7), dtype=np.float32)
        mask = np.zeros((self.max_num,), dtype=np.float32)
        n = min(len(boxes), self.max_num)
        if n:
            center[:n] = np.asarray(boxes[:n], dtype=np.float32)
            mask[:n] = 1.0
        return center, mask, object_ids[:n], class_ids[:n]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        frame = self.frames[index]
        t_vehicle_to_enu = _pose_to_matrix(
            _read_json(self.vehicle_root / "pose" / f"{frame}.json")
        )
        t_drone_to_enu = _pose_to_matrix(
            _read_json(self.drone_root / "pose" / f"{frame}.json")
        )
        t_drone_to_vehicle = np.linalg.inv(t_vehicle_to_enu) @ t_drone_to_enu
        t_vehicle_to_drone = np.linalg.inv(t_drone_to_vehicle)

        v_imgs, v_ks, v_exts, v_z = [], [], [], []
        for view_idx, cam in enumerate(VEHICLE_CAMS):
            img, k, ext, world_z = self._camera_input(
                self.vehicle_root, cam, frame, t_vehicle_to_enu,
                self._rng(index, view_idx),
            )
            v_imgs.append(img); v_ks.append(k); v_exts.append(ext); v_z.append(world_z)

        d_imgs, d_ks, d_exts, d_z, d_depth = [], [], [], [], []
        for view_idx, cam in enumerate(DRONE_CAMS):
            img, k, ext, world_z = self._camera_input(
                self.drone_root, cam, frame, t_drone_to_enu,
                self._rng(index, 100 + view_idx),
            )
            d_imgs.append(img); d_ks.append(k); d_exts.append(ext); d_z.append(world_z)
            t_cam_to_enu = t_drone_to_enu @ ext
            depth = _ideal_ground_depth(
                self.final_hw[0] // self.drone_stride,
                self.final_hw[1] // self.drone_stride,
                self.final_hw,
                k,
                t_cam_to_enu,
                self.ground_z,
                self.max_ideal_depth,
            )
            d_depth.append(torch.from_numpy(depth).float())

        center, mask, object_ids, class_ids = self._load_boxes(frame)
        class_ids_padded = np.zeros((self.max_num,), dtype=np.int64)
        class_ids_padded[:len(class_ids)] = np.asarray(class_ids, dtype=np.int64)
        label_dict = self.post_processor.generate_label_airv2x(
            gt_box_center=center,
            anchors=self.anchor_box,
            mask=mask,
            class_ids_padded=class_ids_padded,
        )
        label_dict["anchor_box"] = self.anchor_box

        pairwise = np.tile(np.eye(4, dtype=np.float32), (2, 2, 1, 1))
        pairwise[0, 1] = t_vehicle_to_drone.astype(np.float32)
        pairwise[1, 0] = t_drone_to_vehicle.astype(np.float32)

        return {
            "ego": {
                "frame_id": frame,
                "object_bbx_center": center,
                "object_bbx_mask": mask,
                "object_ids": object_ids,
                "class_ids": class_ids,
                "label_dict": label_dict,
                "anchor_box": self.anchor_box,
                "img_pairwise_t_matrix_collab": pairwise,
                "agent_order": ["vehicle", "drone"],
                "vehicle": {
                    "cam_inputs": self._pack_cam_inputs(v_imgs, v_ks, v_exts, v_z),
                },
                "drone": {
                    "cam_inputs": self._pack_cam_inputs(d_imgs, d_ks, d_exts, d_z),
                    "ideal_depth_z": torch.stack(d_depth, dim=0),
                },
            }
        }

    def collate_batch_train(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(batch) != 1:
            raise ValueError(
                "Griffin full Gaussian currently requires batch_size=1 per GPU."
            )
        sample = batch[0]["ego"]
        return {
            "ego": {
                "frame_id": [sample["frame_id"]],
                "object_bbx_center": torch.from_numpy(
                    sample["object_bbx_center"]
                ).unsqueeze(0),
                "object_bbx_mask": torch.from_numpy(
                    sample["object_bbx_mask"]
                ).unsqueeze(0),
                "object_ids": [sample["object_ids"]],
                "class_ids": [sample["class_ids"]],
                "label_dict": self.post_processor.collate_batch_airv2x(
                    [sample["label_dict"]]
                ),
                "anchor_box": torch.from_numpy(
                    np.asarray(self.anchor_box)
                ).float(),
                "transformation_matrix": torch.eye(4, dtype=torch.float32),
                "img_pairwise_t_matrix_collab": torch.from_numpy(
                    sample["img_pairwise_t_matrix_collab"]
                ).float().unsqueeze(0),
                "agent_order": ["vehicle", "drone"],
                "record_len": torch.tensor([2], dtype=torch.int64),
                "vehicle": {
                    "batch_merged_cam_inputs": sample["vehicle"]["cam_inputs"],
                    "record_len": torch.tensor([1], dtype=torch.int64),
                    "batch_idxs": torch.tensor([0], dtype=torch.int64),
                },
                "drone": {
                    "batch_merged_cam_inputs": sample["drone"]["cam_inputs"],
                    "ideal_depth_z": sample["drone"]["ideal_depth_z"],
                    "record_len": torch.tensor([1], dtype=torch.int64),
                    "batch_idxs": torch.tensor([0], dtype=torch.int64),
                },
            }
        }

    def collate_batch_test(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.collate_batch_train(batch)

    def post_process(self, data_dict, output_dict):
        pred_box, pred_score, pred_labels, _ = (
            self.post_processor.post_process_airv2x(data_dict, output_dict)
        )
        gt_box, gt_class_ids, gt_track_ids = (
            self.post_processor.generate_gt_bbx_airv2x(data_dict)
        )
        return pred_box, pred_score, pred_labels, gt_box, gt_class_ids, gt_track_ids

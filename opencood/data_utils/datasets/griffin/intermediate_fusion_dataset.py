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
    _bgr_to_normalized_tensor,
    _load_calib,
    _read_json,
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
        self.image_scale = float(cfg.get("image_scale", 0.5))
        self.drone_stride = int(cfg.get("drone_stride", 4))
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.max_ideal_depth = float(cfg.get("max_ideal_depth", 150.0))

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

        self.det_range = np.asarray(
            params["postprocess"]["anchor_args"]["cav_lidar_range"],
            dtype=np.float32,
        )
        self._configure_anchor_priors(split_json)
        self.post_processor = post_processor.build_postprocessor(
            params["postprocess"], dataset="airv2x", train=train
        )
        self.anchor_box = self.post_processor.generate_anchor_box()
        self.max_num = int(params["postprocess"].get("max_num", 300))
        print(
            f"[GriffinDet] split={self.split} frames={len(self.frames)} "
            f"vehicle=4cam drone=bottom final={self.final_hw} "
            f"image_scale={self.image_scale} photometric=False"
        )

    def _configure_anchor_priors(self, split_json: Path) -> None:
        """Use Griffin TRAIN-label medians for class-specific anchor priors."""
        anchor_args = self.params["postprocess"]["anchor_args"]
        if anchor_args.get("sizes") != "auto_griffin_train_median":
            return

        train_frames = resolve_split_frames(
            self.vehicle_root, "train", split_json
        )
        by_class = {1: [], 2: [], 3: []}
        for frame in train_frames:
            path = self.vehicle_root / "label" / f"{frame}.txt"
            for fields in self._read_label_rows(path):
                if len(fields) < 10:
                    continue
                class_id = _CLASS_ID.get(fields[0].lower())
                if class_id is None:
                    continue
                x, y, z = map(float, fields[1:4])
                if not (
                    self.det_range[0] <= x <= self.det_range[3]
                    and self.det_range[1] <= y <= self.det_range[4]
                    and self.det_range[2] <= z <= self.det_range[5]
                ):
                    continue
                l, w, h = map(float, fields[4:7])
                by_class[class_id].append([h, w, l, z])

        sizes = []
        for class_id in (1, 2, 3):
            values = np.asarray(by_class[class_id], dtype=np.float32)
            if values.size == 0:
                raise RuntimeError(
                    f"No Griffin TRAIN boxes found for class_id={class_id}"
                )
            sizes.append(np.median(values, axis=0).tolist())

        anchor_args["sizes"] = sizes
        anchor_args["num"] = len(sizes) * len(anchor_args["r"])
        expected = int(anchor_args["num"])
        model_args = self.params["model"]["args"]
        if int(model_args["anchor_number"]) != expected:
            raise ValueError(
                f"model.anchor_number={model_args['anchor_number']} "
                f"but Griffin anchors require {expected}"
            )
        print(
            "[GriffinDet] anchor priors [h,w,l,z] "
            f"car/bicycle/pedestrian={sizes}"
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

    def _camera_input(
        self,
        side_root: Path,
        cam: str,
        frame: str,
        t_agent_to_enu: np.ndarray,
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, float]:
        path = side_root / "camera" / cam / f"{frame}.png"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        original_hw = (int(bgr.shape[0]), int(bgr.shape[1]))
        img = _bgr_to_normalized_tensor(
            bgr, self.final_hw, self.image_scale
        )

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
    ) -> Dict[str, Any]:
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
            # Griffin P1 targets/depth use cell centers, e.g. stride-4 -> 2,6,10,...
            "feature_sample_mode": "cell_center",
        }

    @staticmethod
    def _read_label_rows(path: Path) -> List[List[str]]:
        if not path.exists():
            return []
        return [
            line.strip().split()
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    def _load_boxes(
        self,
        frame: str,
        t_drone_to_vehicle: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        """Match Griffin cooperative GT: vehicle labels + drone-only labels."""
        vehicle_rows = self._read_label_rows(
            self.vehicle_root / "label" / f"{frame}.txt"
        )
        drone_rows = self._read_label_rows(
            self.drone_root / "label" / f"{frame}.txt"
        )

        vehicle_ids = {
            row[10] for row in vehicle_rows if len(row) > 10
        }
        rows: List[Tuple[List[str], bool]] = [
            (row, False) for row in vehicle_rows
        ]
        rows.extend(
            (row, True)
            for row in drone_rows
            if len(row) > 10 and row[10] not in vehicle_ids
        )

        boxes: List[List[float]] = []
        object_ids: List[str] = []
        class_ids: List[int] = []

        for fields, from_drone in rows:
            if len(fields) < 10:
                continue
            class_id = _CLASS_ID.get(fields[0].lower())
            if class_id is None:
                continue

            x, y, z = map(float, fields[1:4])
            l, w, h = map(float, fields[4:7])
            roll_deg, pitch_deg, yaw_deg = map(float, fields[7:10])
            object_id = (
                fields[10]
                if len(fields) > 10
                else f"{frame}_{len(boxes)}"
            )

            if from_drone:
                t_obj_to_drone = np.eye(4, dtype=np.float64)
                t_obj_to_drone[:3, :3] = SciRot.from_euler(
                    "xyz",
                    [roll_deg, pitch_deg, yaw_deg],
                    degrees=True,
                ).as_matrix()
                t_obj_to_drone[:3, 3] = [x, y, z]
                t_obj_to_vehicle = t_drone_to_vehicle @ t_obj_to_drone
                x, y, z = t_obj_to_vehicle[:3, 3].tolist()
                yaw = float(
                    SciRot.from_matrix(
                        t_obj_to_vehicle[:3, :3]
                    ).as_euler("xyz", degrees=False)[2]
                )
                # Griffin converter removes the ego vehicle if it appears
                # only in the aerial annotation list.
                if np.linalg.norm(t_obj_to_vehicle[:2, 3]) < 0.1:
                    continue
            else:
                yaw = float(np.deg2rad(yaw_deg))

            if not (
                self.det_range[0] <= x <= self.det_range[3]
                and self.det_range[1] <= y <= self.det_range[4]
                and self.det_range[2] <= z <= self.det_range[5]
            ):
                continue

            boxes.append([x, y, z, h, w, l, yaw])
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
                self.vehicle_root, cam, frame, t_vehicle_to_enu
            )
            v_imgs.append(img); v_ks.append(k); v_exts.append(ext); v_z.append(world_z)

        d_imgs, d_ks, d_exts, d_z, d_depth = [], [], [], [], []
        for view_idx, cam in enumerate(DRONE_CAMS):
            img, k, ext, world_z = self._camera_input(
                self.drone_root, cam, frame, t_drone_to_enu
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

        center, mask, object_ids, class_ids = self._load_boxes(
            frame, t_drone_to_vehicle
        )
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

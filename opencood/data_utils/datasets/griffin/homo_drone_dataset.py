# -*- coding: utf-8 -*-
"""Drone-only Griffin homogeneous Stage-0 dataset.

Camera count matches the cooperative Griffin baseline: bottom camera only.
``ideal_depth_z`` is the same analytic ground intersection used by that
baseline. Boxes stay in the drone local frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import opencood.data_utils.post_processor as post_processor
from opencood.data_utils.datasets.griffin.intermediate_fusion_dataset import (
    DRONE_CAMS,
    IntermediateFusionDatasetGriffin,
    _CLASS_ID,
    _ideal_ground_depth,
    _pose_to_matrix,
)
from opencood.data_utils.datasets.griffin_p1_dataset import (
    _read_json,
    resolve_split_frames,
)


class GriffinHomoDroneDataset(Dataset):
    """One Griffin drone-bottom sample in the drone local frame.

    Collate leaves vehicle and RSU empty. The pairwise matrix is identity.
    """

    def __init__(self, params: Dict[str, Any], visualize: bool, train: bool = True) -> None:
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
        if split_json is None or not split_json.exists():
            raise FileNotFoundError(
                "Griffin drone Stage-0 requires the official split_json; "
                f"got {split_json}"
            )
        self.final_hw = tuple(int(v) for v in cfg.get("final_dim", [360, 640]))
        self.image_scale = float(cfg.get("image_scale", 0.5))
        self.drone_stride = int(cfg.get("drone_stride", 4))
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.max_ideal_depth = float(cfg.get("max_ideal_depth", 150.0))

        raw_frames = resolve_split_frames(self.vehicle_root, self.split, split_json)
        self.frames = [frame for frame in raw_frames if self._frame_complete(frame)]
        if not self.frames:
            raise RuntimeError(f"No usable Griffin drone frames for split={self.split}")

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
            f"[GriffinHomoDrone] split={self.split} frames={len(self.frames)} "
            f"cams={list(DRONE_CAMS)} stride={self.drone_stride} "
            f"final={self.final_hw} image_scale={self.image_scale} "
            f"range={self.det_range.tolist()}"
        )

    def _configure_anchor_priors(self, split_json: Path) -> None:
        """Median ``[h, w, l, z]`` from drone-local TRAIN boxes inside range.

        This is the cooperative median procedure, applied to drone-side labels
        in the drone frame. Vehicle labels are not used.
        """
        anchor_args = self.params["postprocess"]["anchor_args"]
        if anchor_args.get("sizes") != "auto_griffin_train_median":
            return

        train_frames = resolve_split_frames(self.vehicle_root, "train", split_json)
        by_class: Dict[int, List[List[float]]] = {1: [], 2: [], 3: []}
        for frame in train_frames:
            path = self.drone_root / "label" / f"{frame}.txt"
            for fields in IntermediateFusionDatasetGriffin._read_label_rows(path):
                parsed = self._parse_local_box(fields, frame, len(by_class[1]))
                if parsed is None:
                    continue
                _box, class_id = parsed
                by_class[class_id].append(
                    [float(_box[3]), float(_box[4]), float(_box[5]), float(_box[2])]
                )

        sizes = []
        for class_id in (1, 2, 3):
            values = np.asarray(by_class[class_id], dtype=np.float32)
            if values.size == 0:
                raise RuntimeError(
                    "No Griffin drone TRAIN boxes inside the drone range for "
                    f"class_id={class_id}"
                )
            sizes.append(np.median(values, axis=0).tolist())

        anchor_args["sizes"] = sizes
        anchor_args["num"] = len(sizes) * len(anchor_args["r"])
        expected = int(anchor_args["num"])
        model_args = self.params["model"]["args"]
        if int(model_args["anchor_number"]) != expected:
            raise ValueError(
                f"model.anchor_number={model_args['anchor_number']} "
                f"but Griffin drone anchors require {expected}"
            )
        print(
            "[GriffinHomoDrone] anchor priors [h,w,l,z] "
            f"car/bicycle/pedestrian={sizes}"
        )

    def _frame_complete(self, frame: str) -> bool:
        required = [
            self.drone_root / "pose" / f"{frame}.json",
            self.drone_root / "label" / f"{frame}.txt",
        ]
        for cam in DRONE_CAMS:
            required.append(self.drone_root / "camera" / cam / f"{frame}.png")
        return all(path.exists() for path in required)

    def __len__(self) -> int:
        return len(self.frames)

    def _parse_local_box(
        self, fields: List[str], frame: str, index: int
    ) -> Optional[Tuple[List[float], int]]:
        """Parse one drone-label row and keep it in the drone frame."""
        if len(fields) < 10:
            return None
        class_id = _CLASS_ID.get(fields[0].lower())
        if class_id is None:
            return None
        x, y, z = map(float, fields[1:4])
        length, width, height = map(float, fields[4:7])
        yaw = float(np.deg2rad(float(fields[9])))
        if not (
            self.det_range[0] <= x <= self.det_range[3]
            and self.det_range[1] <= y <= self.det_range[4]
            and self.det_range[2] <= z <= self.det_range[5]
        ):
            return None
        return [x, y, z, height, width, length, yaw], int(class_id)

    def _load_drone_boxes(
        self, frame: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        rows = IntermediateFusionDatasetGriffin._read_label_rows(
            self.drone_root / "label" / f"{frame}.txt"
        )
        boxes: List[List[float]] = []
        object_ids: List[str] = []
        class_ids: List[int] = []
        for fields in rows:
            parsed = self._parse_local_box(fields, frame, len(boxes))
            if parsed is None:
                continue
            box, class_id = parsed
            object_id = fields[10] if len(fields) > 10 else f"{frame}_{len(boxes)}"
            boxes.append(box)
            object_ids.append(str(object_id))
            class_ids.append(class_id)

        center = np.zeros((self.max_num, 7), dtype=np.float32)
        mask = np.zeros((self.max_num,), dtype=np.float32)
        count = min(len(boxes), self.max_num)
        if count:
            center[:count] = np.asarray(boxes[:count], dtype=np.float32)
            mask[:count] = 1.0
        return center, mask, object_ids[:count], class_ids[:count]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        frame = self.frames[index]
        t_drone_to_enu = _pose_to_matrix(
            _read_json(self.drone_root / "pose" / f"{frame}.json")
        )
        imgs: List[torch.Tensor] = []
        intrinsics: List[np.ndarray] = []
        extrinsics: List[np.ndarray] = []
        world_z: List[float] = []
        depths: List[torch.Tensor] = []
        for cam in DRONE_CAMS:
            img, intrinsic, extrinsic, cam_z = IntermediateFusionDatasetGriffin._camera_input(
                self, self.drone_root, cam, frame, t_drone_to_enu
            )
            imgs.append(img)
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
            world_z.append(cam_z)
            t_cam_to_enu = t_drone_to_enu @ extrinsic
            depth = _ideal_ground_depth(
                self.final_hw[0] // self.drone_stride,
                self.final_hw[1] // self.drone_stride,
                self.final_hw,
                intrinsic,
                t_cam_to_enu,
                self.ground_z,
                self.max_ideal_depth,
            )
            depths.append(torch.from_numpy(depth).float())

        center, mask, object_ids, class_ids = self._load_drone_boxes(frame)
        class_ids_padded = np.zeros((self.max_num,), dtype=np.int64)
        class_ids_padded[: len(class_ids)] = np.asarray(class_ids, dtype=np.int64)
        label_dict = self.post_processor.generate_label_airv2x(
            gt_box_center=center,
            anchors=self.anchor_box,
            mask=mask,
            class_ids_padded=class_ids_padded,
        )
        label_dict["anchor_box"] = self.anchor_box
        pairwise = np.eye(4, dtype=np.float32)[None, None, :, :]
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
                "agent_order": ["drone"],
                "drone": {
                    "cam_inputs": IntermediateFusionDatasetGriffin._pack_cam_inputs(
                        imgs, intrinsics, extrinsics, world_z
                    ),
                    "ideal_depth_z": torch.stack(depths, dim=0),
                },
            }
        }

    def collate_batch_train(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate one drone-only scene.

        Raises:
            ValueError: If ``batch`` is not length 1.
        """
        if len(batch) != 1:
            raise ValueError(
                "Griffin drone Stage-0 currently requires batch_size=1 per GPU."
            )
        sample = batch[0]["ego"]
        pairwise = torch.from_numpy(sample["img_pairwise_t_matrix_collab"]).float()
        pairwise = pairwise.unsqueeze(0)
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
                "anchor_box": torch.from_numpy(np.asarray(self.anchor_box)).float(),
                "transformation_matrix": torch.eye(4, dtype=torch.float32),
                "pairwise_t_matrix_collab": pairwise,
                "img_pairwise_t_matrix_collab": pairwise,
                "agent_order": ["drone"],
                "record_len": torch.tensor([1], dtype=torch.int64),
                "vehicle": {
                    "batch_merged_cam_inputs": {},
                    "record_len": torch.tensor([0], dtype=torch.int64),
                    "batch_idxs": [],
                },
                "rsu": {
                    "batch_merged_cam_inputs": {},
                    "record_len": torch.tensor([0], dtype=torch.int64),
                    "batch_idxs": [],
                },
                "drone": {
                    "batch_merged_cam_inputs": IntermediateFusionDatasetGriffin._batch_cam_inputs(
                        sample["drone"]["cam_inputs"]
                    ),
                    "ideal_depth_z": sample["drone"]["ideal_depth_z"].unsqueeze(0),
                    "record_len": torch.tensor([1], dtype=torch.int64),
                    "batch_idxs": [0],
                },
            }
        }

    def collate_batch_test(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.collate_batch_train(batch)

    def post_process(self, data_dict: Dict[str, Any], output_dict: Dict[str, Any]):
        pred_box, pred_score, pred_labels, _ = (
            self.post_processor.post_process_airv2x(data_dict, output_dict)
        )
        gt_box, gt_class_ids, gt_track_ids = (
            self.post_processor.generate_gt_bbx_airv2x(data_dict)
        )
        return pred_box, pred_score, pred_labels, gt_box, gt_class_ids, gt_track_ids

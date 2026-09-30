# -*- coding: utf-8 -*-
"""Vehicle-only Griffin homogeneous Stage-0 dataset.

Reuses the cooperative Griffin camera packing, ImageNet normalization,
calibration scaling and vehicle-label anchor priors. It does not read drone
images and does not build a vehicle-drone transform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import opencood.data_utils.post_processor as post_processor
from opencood.data_utils.datasets.griffin.intermediate_fusion_dataset import (
    IntermediateFusionDatasetGriffin,
    _CLASS_ID,
    _pose_to_matrix,
)
from opencood.data_utils.datasets.griffin_p1_dataset import (
    VEHICLE_CAMS,
    _read_json,
    resolve_split_frames,
)


class GriffinHomoVehicleDataset(Dataset):
    """One Griffin vehicle, in the vehicle local frame, per sample.

    Collate matches the ``Airv2xBase`` agent contract with a single vehicle
    and empty RSU / drone slots. Pairwise transforms are identity because
    Stage-0 has no collaborator.
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

        cfg = params.get("griffin") or {}
        split_json_value = cfg.get("split_json")
        split_json = (
            Path(split_json_value).expanduser().resolve()
            if split_json_value
            else None
        )
        if split_json is None or not split_json.exists():
            raise FileNotFoundError(
                "Griffin vehicle Stage-0 requires the official split_json; "
                f"got {split_json}"
            )
        self.final_hw = tuple(int(v) for v in cfg.get("final_dim", [360, 640]))
        self.image_scale = float(cfg.get("image_scale", 0.5))

        raw_frames = resolve_split_frames(self.vehicle_root, self.split, split_json)
        self.frames = [frame for frame in raw_frames if self._frame_complete(frame)]
        if not self.frames:
            raise RuntimeError(
                f"No usable Griffin vehicle frames for split={self.split}"
            )

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
            f"[GriffinHomoVehicle] split={self.split} frames={len(self.frames)} "
            f"cams={list(VEHICLE_CAMS)} final={self.final_hw} "
            f"image_scale={self.image_scale} range={self.det_range.tolist()}"
        )

    def _configure_anchor_priors(self, split_json: Path) -> None:
        """Reuse the cooperative vehicle TRAIN-median anchor priors."""
        IntermediateFusionDatasetGriffin._configure_anchor_priors(self, split_json)

    @staticmethod
    def _read_label_rows(path: Path) -> List[List[str]]:
        return IntermediateFusionDatasetGriffin._read_label_rows(path)

    def _frame_complete(self, frame: str) -> bool:
        required = [
            self.vehicle_root / "pose" / f"{frame}.json",
            self.vehicle_root / "label" / f"{frame}.txt",
        ]
        for cam in VEHICLE_CAMS:
            required.append(self.vehicle_root / "camera" / cam / f"{frame}.png")
        return all(path.exists() for path in required)

    def __len__(self) -> int:
        return len(self.frames)

    def _load_vehicle_boxes(
        self, frame: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        """Keep vehicle-label boxes in the vehicle local frame."""
        rows = self._read_label_rows(self.vehicle_root / "label" / f"{frame}.txt")
        boxes: List[List[float]] = []
        object_ids: List[str] = []
        class_ids: List[int] = []
        for fields in rows:
            if len(fields) < 10:
                continue
            class_id = _CLASS_ID.get(fields[0].lower())
            if class_id is None:
                continue
            x, y, z = map(float, fields[1:4])
            length, width, height = map(float, fields[4:7])
            yaw = float(np.deg2rad(float(fields[9])))
            object_id = fields[10] if len(fields) > 10 else f"{frame}_{len(boxes)}"
            if not (
                self.det_range[0] <= x <= self.det_range[3]
                and self.det_range[1] <= y <= self.det_range[4]
                and self.det_range[2] <= z <= self.det_range[5]
            ):
                continue
            boxes.append([x, y, z, height, width, length, yaw])
            object_ids.append(str(object_id))
            class_ids.append(int(class_id))

        center = np.zeros((self.max_num, 7), dtype=np.float32)
        mask = np.zeros((self.max_num,), dtype=np.float32)
        count = min(len(boxes), self.max_num)
        if count:
            center[:count] = np.asarray(boxes[:count], dtype=np.float32)
            mask[:count] = 1.0
        return center, mask, object_ids[:count], class_ids[:count]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        frame = self.frames[index]
        t_vehicle_to_enu = _pose_to_matrix(
            _read_json(self.vehicle_root / "pose" / f"{frame}.json")
        )
        imgs: List[torch.Tensor] = []
        intrinsics: List[np.ndarray] = []
        extrinsics: List[np.ndarray] = []
        world_z: List[float] = []
        for cam in VEHICLE_CAMS:
            img, intrinsic, extrinsic, cam_z = IntermediateFusionDatasetGriffin._camera_input(
                self, self.vehicle_root, cam, frame, t_vehicle_to_enu
            )
            imgs.append(img)
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
            world_z.append(cam_z)

        center, mask, object_ids, class_ids = self._load_vehicle_boxes(frame)
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
                "agent_order": ["vehicle"],
                "vehicle": {
                    "cam_inputs": IntermediateFusionDatasetGriffin._pack_cam_inputs(
                        imgs, intrinsics, extrinsics, world_z
                    ),
                },
            }
        }

    def collate_batch_train(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate one vehicle-only scene.

        Raises:
            ValueError: If ``batch`` is not length 1. Griffin camera training
                currently runs one scene per step, matching the cooperative loader.
        """
        if len(batch) != 1:
            raise ValueError(
                "Griffin vehicle Stage-0 currently requires batch_size=1 per GPU."
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
                "agent_order": ["vehicle"],
                "record_len": torch.tensor([1], dtype=torch.int64),
                "vehicle": {
                    "batch_merged_cam_inputs": IntermediateFusionDatasetGriffin._batch_cam_inputs(
                        sample["vehicle"]["cam_inputs"]
                    ),
                    "record_len": torch.tensor([1], dtype=torch.int64),
                    "batch_idxs": [0],
                },
                "rsu": {
                    "batch_merged_cam_inputs": {},
                    "record_len": torch.tensor([0], dtype=torch.int64),
                    "batch_idxs": [],
                },
                "drone": {
                    "batch_merged_cam_inputs": {},
                    "record_len": torch.tensor([0], dtype=torch.int64),
                    "batch_idxs": [],
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

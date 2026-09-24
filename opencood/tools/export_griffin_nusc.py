# -*- coding: utf-8 -*-
"""Export Griffin Gaussian detections to the official nuScenes result JSON.

Predicted boxes are in the vehicle/ego frame used by the raw Griffin labels:
``[x, y, z, h, w, l, yaw]`` with yaw in radians. Griffin's official evaluator
loads boxes in the global frame, so each box is transformed by the sample's
``ego2global`` pose. Class ids follow the training head: 1 car, 2 bicycle,
3 pedestrian.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

CLASS_NAMES = {1: "car", 2: "bicycle", 3: "pedestrian"}
CLASS_RANGE = {"car": 50.0, "bicycle": 50.0, "pedestrian": 50.0}
ATTRIBUTES = {
    "car": "vehicle.parked",
    "bicycle": "cycle.without_rider",
    "pedestrian": "pedestrian.standing",
}


def ego_box_to_global(
    box7: np.ndarray,
    score: float,
    class_id: int,
    info: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Convert one ego-frame box to a nuScenes detection record.

    Args:
        box7: ``[x, y, z, h, w, l, yaw]`` in the vehicle/ego frame.
        score: Objectness score.
        class_id: 1 car, 2 bicycle, 3 pedestrian.
        info: One ``griffin_infos_*.pkl`` sample, used for ``ego2global``.

    Returns:
        A detection dict in the global frame, or None when the class is
        unknown or the ego-frame center is outside ``CLASS_RANGE``.
    """
    name = CLASS_NAMES.get(int(class_id))
    if name is None:
        return None
    center = np.asarray(box7[:3], dtype=np.float64)
    if float(np.linalg.norm(center[:2])) > CLASS_RANGE[name]:
        return None
    height, width, length, yaw = [float(v) for v in box7[3:7]]
    box = NuScenesBox(
        center,
        [width, length, height],
        Quaternion(axis=[0.0, 0.0, 1.0], radians=yaw),
        score=float(score),
        velocity=(0.0, 0.0, 0.0),
    )
    box.rotate(Quaternion(info["ego2global_rotation"]))
    box.translate(np.asarray(info["ego2global_translation"], dtype=np.float64))
    return {
        "sample_token": info["token"],
        "translation": box.center.tolist(),
        "size": box.wlh.tolist(),
        "rotation": box.orientation.elements.tolist(),
        "velocity": [0.0, 0.0],
        "detection_name": name,
        "detection_score": float(score),
        "attribute_name": ATTRIBUTES[name],
    }


def load_infos_by_frame(info_path: Path) -> Dict[str, Mapping[str, Any]]:
    """Index Griffin infos by the raw frame id in ``CAM_FRONT``."""
    payload = pickle.load(open(info_path, "rb"))
    infos = payload["infos"] if isinstance(payload, dict) else payload
    by_frame: Dict[str, Mapping[str, Any]] = {}
    for info in infos:
        frame = Path(info["cams"]["CAM_FRONT"]["data_path"]).stem
        by_frame[frame] = info
    return by_frame


def export_split(
    model_dir: str,
    info_path: Path,
    out_json: Path,
    epoch: Optional[int],
    gpu: int,
    workers: int,
) -> None:
    """Run the detector and write ``results_nusc.json`` for one split."""

    class Opt:
        config_file = "config.yaml"

    Opt.model_dir = model_dir
    hypes = yaml_utils.load_yaml(None, Opt())
    dataset = build_dataset(hypes, visualize=False, train=False)
    infos = load_infos_by_frame(info_path)
    missing = [frame for frame in dataset.frames if frame not in infos]
    if missing:
        raise RuntimeError(f"{len(missing)} val frames have no Griffin info, e.g. {missing[:3]}")

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = train_utils.create_model(hypes).to(device)
    epoch_id, model = train_utils.load_model(
        model_dir, model, epoch=epoch, start_from_best=False, device=device
    )
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    results: Dict[str, List[Dict[str, Any]]] = {
        infos[frame]["token"]: [] for frame in dataset.frames
    }
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"export epoch={epoch_id}", total=len(dataset)):
            frame = batch["ego"]["frame_id"][0]
            batch = train_utils.to_device(batch, device)
            output = {"ego": model(batch["ego"])}
            _, scores, labels, boxes7 = dataset.post_processor.post_process_airv2x(
                batch, output
            )
            info = infos[frame]
            if boxes7 is None or scores is None:
                continue
            boxes_np = boxes7.detach().cpu().numpy()
            scores_np = scores.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy().astype(np.int64)
            for box, score, class_id in zip(boxes_np, scores_np, labels_np):
                record = ego_box_to_global(box, float(score), int(class_id), info)
                if record is not None:
                    results[info["token"]].append(record)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": True,
        },
        "results": results,
    }
    out_json.write_text(json.dumps(payload))
    n_boxes = sum(len(v) for v in results.values())
    print(f"epoch={epoch_id} samples={len(results)} boxes={n_boxes} saved={out_json}")


def evaluate_nusc(result_json: Path, info_path: Path, output_dir: Path) -> None:
    """Run Griffin's nuScenes detection eval and print mean AP."""
    import importlib.util

    griffin_root = Path("/mnt/home/suyi/Griffin")
    eval_path = (
        griffin_root
        / "projects/mmdet3d_plugin/datasets/eval_utils/nuscenes_eval.py"
    )
    spec = importlib.util.spec_from_file_location("griffin_nuscenes_eval", eval_path)
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)
    NuScenesEval_custom = eval_mod.NuScenesEval_custom
    from nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory

    infos = pickle.load(open(info_path, "rb"))["infos"]
    split_json = json.load(
        open(griffin_root / "data/split_datas/griffin_50scenes_55m.json")
    )
    nusc = NuScenes(
        version="v1.0-trainval",
        dataroot=str(
            griffin_root / "datasets/griffin_50scenes_55m/griffin-nuscenes/cooperative"
        ),
        verbose=True,
    )
    cfg = config_factory("detection_cvpr_2019")
    cfg.class_names = ["car", "bicycle", "pedestrian"]
    cfg.class_range = CLASS_RANGE
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = NuScenesEval_custom(
        nusc,
        config=cfg,
        result_path=str(result_json),
        eval_set="val",
        output_dir=str(output_dir),
        verbose=True,
        data_infos=infos,
        splits=split_json["batch_split"],
        category_to_type_name=lambda name: name if name in cfg.class_names else None,
        class_range=CLASS_RANGE,
    )
    evaluator.main(plot_examples=0, render_curves=False)
    summary = json.load(open(output_dir / "metrics_summary.json"))
    print(f"Griffin mean_ap={summary['mean_ap']:.4f} NDS={summary['nd_score']:.4f}")
    for name, aps in summary["label_aps"].items():
        print(name, {k: round(v, 4) for k, v in aps.items()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--eval_epoch", type=int, default=13)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--info",
        default="/mnt/home/suyi/Griffin/data/infos/griffin_50scenes_55m/cooperative/griffin_infos_val.pkl",
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--skip_export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(
        args.out_dir
        or os.path.join(args.model_dir, f"griffin_nusc_epoch{args.eval_epoch}")
    )
    result_json = out_dir / "results_nusc.json"
    if args.skip_export and result_json.exists():
        print(f"reuse {result_json}")
    else:
        export_split(
        args.model_dir,
        Path(args.info),
        result_json,
        args.eval_epoch,
        args.gpu,
        args.workers,
    )
    evaluate_nusc(result_json, Path(args.info), out_dir / "det")


if __name__ == "__main__":
    main()

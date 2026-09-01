#!/usr/bin/env python3
"""4x4 tau sweep and true per-scene foreground candidate counts. Diagnostics only."""

from __future__ import annotations

import argparse
import json
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from opencood.tools.analyze_p1_heatmap_resolution import (
    AGENT_KEYS,
    CLASS_NAMES,
    dummy_lidar_preprocess,
    measure_object_on_map,
    percentile,
    project_box_to_image,
    rasterize_convex_polygon,
    summarize_counts,
)
from opencood.utils.box_utils import boxes_to_corners_3d

SOURCE_HW: Tuple[int, int] = (360, 640)
HW45: Tuple[int, int] = (45, 80)
HW90: Tuple[int, int] = (90, 160)
TAUS: Tuple[int, ...] = (1, 2, 3, 4)


def _block_class_and_stats(
    semantic: np.ndarray,
    factor: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized per-block fg count, majority class with geometric tie-break.

    Block center is pixel (factor*i + factor/2, factor*j + factor/2).
    """
    height, width = semantic.shape
    if height % factor != 0 or width % factor != 0:
        raise ValueError(f"cannot block {semantic.shape} by {factor}")
    out_h = height // factor
    out_w = width // factor
    blocks = semantic.reshape(out_h, factor, out_w, factor).transpose(0, 2, 1, 3)
    n_fg = (blocks > 0).sum(axis=(2, 3)).astype(np.int32)
    yy, xx = np.mgrid[0:factor, 0:factor]
    center = float(factor) / 2.0
    dist2 = (yy.astype(np.float64) - center) ** 2 + (xx.astype(np.float64) - center) ** 2
    occ = np.zeros((out_h, out_w, 6), dtype=np.int32)
    mean_d = np.full((out_h, out_w, 6), 1.0e9, dtype=np.float64)
    for class_id in range(1, 7):
        mask = blocks == class_id
        count = mask.sum(axis=(2, 3))
        occ[..., class_id - 1] = count
        dist_sum = (mask.astype(np.float64) * dist2).sum(axis=(2, 3))
        valid = count > 0
        mean_d[..., class_id - 1] = np.where(valid, dist_sum / np.maximum(count, 1), 1.0e9)
    score = occ.astype(np.float64) * 1.0e6 - mean_d
    majority = score.argmax(axis=-1).astype(np.int32) + 1
    n_fg_classes = (occ > 0).sum(axis=-1)
    return n_fg, majority, n_fg_classes


def apply_tau(n_fg: np.ndarray, majority: np.ndarray, tau: int) -> np.ndarray:
    """Background if fewer than tau foreground pixels, else majority fg class."""
    return np.where(n_fg >= int(tau), majority, 0).astype(np.int64)


def pixel_fg_confusion(
    target: np.ndarray,
    semantic: np.ndarray,
    factor: int,
) -> Tuple[int, int, int]:
    """Nearest-expand target cells to source pixels; binary fg vs gt semantic."""
    pred = np.repeat(np.repeat(target > 0, factor, axis=0), factor, axis=1)
    gt = semantic > 0
    true_pos = int(np.logical_and(pred, gt).sum())
    false_pos = int(np.logical_and(pred, np.logical_not(gt)).sum())
    false_neg = int(np.logical_and(np.logical_not(pred), gt).sum())
    return true_pos, false_pos, false_neg


def pr_iou(true_pos: int, false_pos: int, false_neg: int) -> Dict[str, float]:
    """Precision / recall / IoU from confusion counts."""
    prec_den = true_pos + false_pos
    rec_den = true_pos + false_neg
    iou_den = true_pos + false_pos + false_neg
    return {
        "precision": float(true_pos / prec_den) if prec_den else float("nan"),
        "recall": float(true_pos / rec_den) if rec_den else float("nan"),
        "iou": float(true_pos / iou_den) if iou_den else float("nan"),
    }


def summarize_scene(values: Sequence[float]) -> Dict[str, float]:
    """Scene-level count summary."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def load_dataset(yaml_path: str, data_root: str) -> Any:
    """Build the same IntermediateFusion dataset stub as the resolution study."""
    cuda_stub = types.ModuleType("opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_cuda")

    def _stub_points_in_boxes_cpu(boxes, points, point_indices):
        point_indices.zero_()

    cuda_stub.points_in_boxes_cpu = _stub_points_in_boxes_cpu
    sys.modules["opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_cuda"] = cuda_stub
    import opencood.pcdet_utils.roiaware_pool3d as roiaware_pkg

    roiaware_pkg.roiaware_pool3d_cuda = cuda_stub

    from opencood.data_utils.datasets.airv2x.intermediate_fusion_dataset import (
        IntermediateFusionDatasetAirv2x,
    )
    from opencood.hypes_yaml import yaml_utils

    hypes = yaml_utils.load_yaml(yaml_path, None)
    hypes["root_dir"] = data_root
    hypes["validate_dir"] = data_root
    dataset = IntermediateFusionDatasetAirv2x(params=hypes, visualize=False, train=False)
    dataset.pre_processor.preprocess = dummy_lidar_preprocess
    return dataset


def process_view(
    semantic: np.ndarray,
    cam_inputs: Dict[str, Any],
    view_idx: int,
    corners_lidar: np.ndarray,
    class_ids: Sequence[int],
    image_hw: Tuple[int, int],
    maps90: Dict[int, np.ndarray],
    map45: np.ndarray,
) -> List[Dict[str, Any]]:
    """Visible-object support cells on each candidate target map."""
    intrinsic = cam_inputs["intrinsics"][view_idx].numpy()
    extrinsics = cam_inputs["extrinsics"][view_idx].numpy()
    post_rot = cam_inputs["post_rots"][view_idx].numpy()
    post_trans = cam_inputs["post_trans"][view_idx].numpy()
    records: List[Dict[str, Any]] = []
    for box_idx, class_id in enumerate(class_ids):
        if int(class_id) not in CLASS_NAMES:
            continue
        projected = project_box_to_image(
            corners_lidar[box_idx],
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
        }
        scaled45 = np.stack([pts[:, 0] * (HW45[1] / image_hw[1]), pts[:, 1] * (HW45[0] / image_hw[0])], axis=-1)
        mask45 = rasterize_convex_polygon(scaled45, HW45[0], HW45[1])
        record["cells_45"] = measure_object_on_map(map45, mask45, int(class_id))["cells"]
        scaled90 = np.stack([pts[:, 0] * (HW90[1] / image_hw[1]), pts[:, 1] * (HW90[0] / image_hw[0])], axis=-1)
        mask90 = rasterize_convex_polygon(scaled90, HW90[0], HW90[1])
        for tau, label_map in maps90.items():
            record[f"cells_90_tau{tau}"] = measure_object_on_map(
                label_map, mask90, int(class_id)
            )["cells"]
        records.append(record)
    return records


def main() -> None:
    """Run the two missing empirical checks on the same 80-frame stride sample."""
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
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument(
        "--out",
        default=str(ROOT / "opencood/logs/p1_tau4x4_and_scene.json"),
    )
    args = parser.parse_args()

    dataset = load_dataset(args.y, args.data_root)
    n_total = len(dataset)
    stride = max(1, n_total // max(args.max_samples, 1))
    indices = list(range(0, n_total, stride))[: args.max_samples]
    print(f"Dataset len={n_total} stride={stride} using {len(indices)} samples")

    object_by_agent_tau: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    object_by_agent_class_tau: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
    object_45: Dict[str, List[float]] = defaultdict(list)
    confusion: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0, 0])
    fg_cells_image: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    conflict_fg_blocks: Dict[str, List[int]] = defaultdict(list)
    conflict_multi: Dict[str, List[int]] = defaultdict(list)

    scene_rows: List[Dict[str, Any]] = []
    all_object_records = 0
    n_ok = 0

    for idx in tqdm(indices, desc="tau-scene"):
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

        scene: Dict[str, Any] = {
            "idx": int(idx),
            "n_agent_vehicle": 0,
            "n_agent_rsu": 0,
            "n_agent_drone": 0,
            "n_views_vehicle": 0,
            "n_views_rsu": 0,
            "n_views_drone": 0,
            "p45_vehicle": 0,
            "p45_rsu": 0,
            "p45_drone": 0,
        }
        for tau in TAUS:
            scene[f"p90t{tau}_vehicle"] = 0
            scene[f"p90t{tau}_rsu"] = 0
            scene[f"p90t{tau}_drone"] = 0

        cav_offset = 0
        try:
            for agent_type, abb in AGENT_KEYS.items():
                n_cav = int(ego.get(f"num_{abb}", 0))
                scene[f"n_agent_{agent_type}"] = n_cav
                cam_list = ego.get(f"cam_inputs_{abb}", [])
                for local_idx in range(n_cav):
                    t_cav2ego = np.asarray(
                        pairwise[cav_offset + local_idx, 0], dtype=np.float64
                    )
                    if t_cav2ego.shape != (4, 4):
                        raise ValueError(f"unexpected pairwise slice {t_cav2ego.shape}")
                    if abs(float(np.linalg.det(t_cav2ego))) < 1e-8:
                        t_cav2ego = np.eye(4, dtype=np.float64)
                    t_ego2cav = np.linalg.inv(t_cav2ego)
                    corners_ego = boxes_to_corners_3d(boxes, "hwl")
                    if hasattr(corners_ego, "numpy"):
                        corners_ego = corners_ego.numpy()
                    n_box = int(corners_ego.shape[0])
                    if n_box == 0:
                        xyz_cav = np.zeros((0, 8, 3), dtype=np.float64)
                    else:
                        ones = np.ones((n_box * 8, 1), dtype=np.float64)
                        xyz_h = np.concatenate(
                            [corners_ego.reshape(-1, 3).astype(np.float64), ones],
                            axis=1,
                        ).T
                        xyz_cav = (t_ego2cav @ xyz_h).T[:, :3].reshape(n_box, 8, 3)
                    cam_inputs = cam_list[local_idx]
                    semantic_all = cam_inputs.get("image_semantic_gts")
                    if not hasattr(semantic_all, "numpy"):
                        continue
                    imgs = cam_inputs["imgs"]
                    num_views = int(imgs.shape[0])
                    scene[f"n_views_{agent_type}"] += num_views
                    image_hw = (int(imgs.shape[-2]), int(imgs.shape[-1]))
                    for view_idx in range(num_views):
                        semantic = semantic_all[view_idx].numpy()
                        if semantic.shape != SOURCE_HW:
                            continue
                        n_fg8, maj8, _ = _block_class_and_stats(semantic, 8)
                        map45 = apply_tau(n_fg8, maj8, 1)
                        n_fg4, maj4, n_cls4 = _block_class_and_stats(semantic, 4)
                        maps90 = {tau: apply_tau(n_fg4, maj4, tau) for tau in TAUS}
                        fg_blocks = n_fg4 > 0
                        conflict_fg_blocks[agent_type].append(int(fg_blocks.sum()))
                        conflict_multi[agent_type].append(
                            int(np.logical_and(fg_blocks, n_cls4 > 1).sum())
                        )
                        p45 = int((map45 > 0).sum())
                        scene[f"p45_{agent_type}"] += p45
                        fg_cells_image[(agent_type, "45")] .append(p45)
                        for tau in TAUS:
                            p90 = int((maps90[tau] > 0).sum())
                            scene[f"p90t{tau}_{agent_type}"] += p90
                            fg_cells_image[(agent_type, f"90t{tau}")].append(p90)
                        tp, fp, fn = pixel_fg_confusion(map45, semantic, 8)
                        confusion[(agent_type, "45")][0] += tp
                        confusion[(agent_type, "45")][1] += fp
                        confusion[(agent_type, "45")][2] += fn
                        for tau in TAUS:
                            tp, fp, fn = pixel_fg_confusion(maps90[tau], semantic, 4)
                            key = (agent_type, f"90t{tau}")
                            confusion[key][0] += tp
                            confusion[key][1] += fp
                            confusion[key][2] += fn
                        recs = process_view(
                            semantic,
                            cam_inputs,
                            view_idx,
                            xyz_cav,
                            classes,
                            image_hw,
                            maps90,
                            map45,
                        )
                        for rec in recs:
                            rec["agent"] = agent_type
                            object_45[agent_type].append(float(rec["cells_45"]))
                            for tau in TAUS:
                                cells = float(rec[f"cells_90_tau{tau}"])
                                object_by_agent_tau[(agent_type, tau)].append(cells)
                                object_by_agent_class_tau[
                                    (agent_type, tau, rec["class_id"])
                                ].append(cells)
                        all_object_records += len(recs)
                cav_offset += n_cav
        except Exception as exc:
            print(f"[skip-collect] idx={idx} {type(exc).__name__}: {exc}")
            continue

        scene["n_views"] = (
            scene["n_views_vehicle"] + scene["n_views_rsu"] + scene["n_views_drone"]
        )
        scene["n_agents"] = (
            scene["n_agent_vehicle"] + scene["n_agent_rsu"] + scene["n_agent_drone"]
        )
        scene["p45"] = scene["p45_vehicle"] + scene["p45_rsu"] + scene["p45_drone"]
        for tau in TAUS:
            scene[f"p90t{tau}"] = (
                scene[f"p90t{tau}_vehicle"]
                + scene[f"p90t{tau}_rsu"]
                + scene[f"p90t{tau}_drone"]
            )
        scene_rows.append(scene)
        n_ok += 1

    payload: Dict[str, Any] = {
        "n_samples": n_ok,
        "n_object_views": all_object_records,
        "stride": stride,
        "taus": list(TAUS),
        "tau_tables": [],
        "conflict": {},
        "scene": {},
        "agents_per_scene": {},
        "views_per_scene": {},
        "candidates_by_agent_type": {},
    }

    print("\n=== 4x4 tau sweep ===")
    header = (
        "agent group tau N disappear% med p10 %<2 %<3 %<5 prec rec IoU "
        "fg_cells/img conflict_among_fg%"
    )
    print(header)
    for agent in ("vehicle", "rsu", "drone"):
        n_fg_b = sum(conflict_fg_blocks[agent])
        n_multi = sum(conflict_multi[agent])
        conflict_rate = float(n_multi / n_fg_b) if n_fg_b else float("nan")
        payload["conflict"][agent] = {
            "fg_blocks": n_fg_b,
            "multi_class_blocks": n_multi,
            "rate_among_fg_blocks": conflict_rate,
        }
        for tau in TAUS:
            cells = object_by_agent_tau[(agent, tau)]
            stats = summarize_counts(cells)
            conf = pr_iou(*confusion[(agent, f"90t{tau}")])
            fg_per_img = fg_cells_image[(agent, f"90t{tau}")]
            row = {
                "agent": agent,
                "group": "all",
                "tau": tau,
                "n": int(stats.get("n", 0)),
                "disappear": float(stats.get("frac_0", float("nan"))),
                "median": float(stats.get("median", float("nan"))),
                "p10": float(stats.get("p10", float("nan"))),
                "frac_lt2": float(stats.get("frac_le2", float("nan"))),
                "frac_lt3": float(stats.get("frac_le3", float("nan"))),
                "frac_lt5": float(stats.get("frac_le5", float("nan"))),
                "precision": conf["precision"],
                "recall": conf["recall"],
                "iou": conf["iou"],
                "fg_cells_per_image_mean": float(np.mean(fg_per_img)) if fg_per_img else float("nan"),
                "conflict_among_fg": conflict_rate,
            }
            payload["tau_tables"].append(row)
            print(
                f"{agent:8} all        {tau} {row['n']:5d} "
                f"{100*row['disappear']:6.2f} {row['median']:6.1f} {row['p10']:5.1f} "
                f"{100*row['frac_lt2']:5.1f} {100*row['frac_lt3']:5.1f} {100*row['frac_lt5']:5.1f} "
                f"{100*row['precision']:5.1f} {100*row['recall']:5.1f} {100*row['iou']:5.1f} "
                f"{row['fg_cells_per_image_mean']:7.1f} {100*conflict_rate:5.2f}"
            )
        cells45 = object_45[agent]
        stats45 = summarize_counts(cells45)
        conf45 = pr_iou(*confusion[(agent, "45")])
        fg45 = fg_cells_image[(agent, "45")]
        row45 = {
            "agent": agent,
            "group": "all",
            "tau": "45x80_corrected_tau1",
            "n": int(stats45.get("n", 0)),
            "disappear": float(stats45.get("frac_0", float("nan"))),
            "median": float(stats45.get("median", float("nan"))),
            "p10": float(stats45.get("p10", float("nan"))),
            "frac_lt2": float(stats45.get("frac_le2", float("nan"))),
            "frac_lt3": float(stats45.get("frac_le3", float("nan"))),
            "frac_lt5": float(stats45.get("frac_le5", float("nan"))),
            "precision": conf45["precision"],
            "recall": conf45["recall"],
            "iou": conf45["iou"],
            "fg_cells_per_image_mean": float(np.mean(fg45)) if fg45 else float("nan"),
        }
        payload["tau_tables"].append(row45)
        print(
            f"{agent:8} all        45 {row45['n']:5d} "
            f"{100*row45['disappear']:6.2f} {row45['median']:6.1f} {row45['p10']:5.1f} "
            f"{100*row45['frac_lt2']:5.1f} {100*row45['frac_lt3']:5.1f} {100*row45['frac_lt5']:5.1f} "
            f"{100*row45['precision']:5.1f} {100*row45['recall']:5.1f} {100*row45['iou']:5.1f} "
            f"{row45['fg_cells_per_image_mean']:7.1f}"
        )
        for class_id, name in CLASS_NAMES.items():
            for tau in TAUS:
                cells = object_by_agent_class_tau[(agent, tau, class_id)]
                if len(cells) < 10:
                    continue
                stats = summarize_counts(cells)
                row = {
                    "agent": agent,
                    "group": name,
                    "tau": tau,
                    "n": int(stats["n"]),
                    "disappear": float(stats["frac_0"]),
                    "median": float(stats["median"]),
                    "p10": float(stats["p10"]),
                    "frac_lt2": float(stats["frac_le2"]),
                    "frac_lt3": float(stats["frac_le3"]),
                    "frac_lt5": float(stats["frac_le5"]),
                }
                payload["tau_tables"].append(row)
                print(
                    f"{agent:8} {name:11} {tau} {row['n']:5d} "
                    f"{100*row['disappear']:6.2f} {row['median']:6.1f} {row['p10']:5.1f} "
                    f"{100*row['frac_lt2']:5.1f} {100*row['frac_lt3']:5.1f} {100*row['frac_lt5']:5.1f}"
                )

    print("\n=== per-scene composition and P_scene ===")
    for key in (
        "n_agent_vehicle",
        "n_agent_rsu",
        "n_agent_drone",
        "n_agents",
        "n_views_vehicle",
        "n_views_rsu",
        "n_views_drone",
        "n_views",
        "p45",
        "p45_vehicle",
        "p45_rsu",
        "p45_drone",
        "p90t1",
        "p90t2",
        "p90t3",
        "p90t4",
        "p90t1_vehicle",
        "p90t1_rsu",
        "p90t1_drone",
        "p90t2_vehicle",
        "p90t2_rsu",
        "p90t2_drone",
    ):
        values = [row[key] for row in scene_rows]
        summary = summarize_scene(values)
        payload["scene"][key] = summary
        print(
            f"{key:22} n={summary.get('n', 0):.0f} mean={summary.get('mean', float('nan')):8.2f} "
            f"med={summary.get('median', float('nan')):8.2f} p90={summary.get('p90', float('nan')):8.2f} "
            f"p95={summary.get('p95', float('nan')):8.2f} max={summary.get('max', float('nan')):8.2f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

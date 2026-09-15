"""FP source audit for AirV2X Gaussian epoch5 on test index 0.

One-to-one GT matching over the production postprocess output, per-FP
attribution, and threshold-policy comparison (obj only / obj+cls /
obj*cls). Analysis only: production postprocess, loss, and model code
are not touched.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import box_utils, common_utils

DEFAULT_MODEL_DIR = (
    ROOT
    / "opencood/logs/airv2x_gaussian_0822_camera"
    / "v45_scale_asym_2026_09_12_01_11_47"
)
DEFAULT_CKPT = "net_epoch5.pth"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return _jsonable(obj.detach().cpu())
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return obj


def load_hypes(model_dir: Path):
    import argparse as _ap
    from opencood.hypes_yaml import yaml_utils

    return yaml_utils.load_yaml(
        str(ROOT / "opencood/hypes_yaml/airv2x/camera/det/airv2x_gaussian_0822.yaml"),
        _ap.Namespace(model_dir=str(model_dir), config_file="config.yaml"),
    )


def load_model(hypes, ckpt: Path, device: torch.device):
    model = train_utils.create_model(hypes).to(device)
    blob = torch.load(str(ckpt), map_location=device)
    src = blob.get("model_state_dict", blob)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in src.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {ckpt.name}: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


def forward_heads(model, batch, device) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        output = model(batch["ego"])
    return output


def decode_scores(
    output: Dict[str, torch.Tensor],
    anchors: torch.Tensor,
    hypes: Dict[str, Any],
    tfm: torch.Tensor,
):
    """Mirror production post_process_airv2x decode, keeping per-anchor arrays."""
    obj = output["obj"].permute(0, 2, 3, 1).contiguous()
    objectness = torch.sigmoid(obj).view(-1)
    psm = output["psm"]
    b, ac, h, w = psm.shape
    c = int(hypes["num_class"])
    a = ac // c
    psm_v = psm.view(b, c, a, h, w).permute(0, 3, 4, 2, 1).contiguous()
    class_scores, class_labels = torch.max(torch.sigmoid(psm_v)[..., 1:], dim=-1)
    class_scores = class_scores.view(-1)
    class_labels = (class_labels + 1).view(-1)
    boxes = VoxelPostprocessor_delta_to_boxes3d(output["rm"], anchors)
    corners_ego = (
        box_utils.boxes_to_corners_3d(boxes, order=hypes["postprocess"]["order"])
        if boxes.shape[0]
        else boxes.new_zeros((0, 8, 3))
    )
    corners = box_utils.project_box3d(corners_ego, tfm)
    return objectness, class_scores, class_labels, boxes, corners


def VoxelPostprocessor_delta_to_boxes3d(
    deltas: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    from opencood.data_utils.post_processor.voxel_postprocessor import (
        VoxelPostprocessor,
    )

    return VoxelPostprocessor.delta_to_boxes3d(deltas, anchors)[0]


def iou_matrix(pred_corners: torch.Tensor, gt_corners: torch.Tensor) -> np.ndarray:
    if pred_corners is None or pred_corners.shape[0] == 0:
        return np.zeros((pred_corners.shape[0] if pred_corners is not None else 0, gt_corners.shape[0]), dtype=np.float32)
    gt_p = common_utils.convert_format(gt_corners.detach().cpu().numpy())
    pr_p = common_utils.convert_format(pred_corners.detach().cpu().numpy())
    n_pr, n_gt = len(pr_p), len(gt_p)
    ious = np.zeros((n_pr, n_gt), dtype=np.float32)
    for i, p in enumerate(pr_p):
        ious[i] = common_utils.compute_iou(p, gt_p)
    return ious


def center_distance(pred_boxes: torch.Tensor, gt_centers_xy: np.ndarray) -> np.ndarray:
    if pred_boxes.shape[0] == 0 or gt_centers_xy.shape[0] == 0:
        return np.zeros((pred_boxes.shape[0], gt_centers_xy.shape[0]), dtype=np.float32)
    pc = pred_boxes[:, :2].detach().cpu().numpy().astype(np.float64)
    gc = np.asarray(gt_centers_xy, dtype=np.float64)
    return np.linalg.norm(pc[:, None, :] - gc[None, :, :], axis=-1).astype(np.float32)


def gt_centers_from_corners(gt_corners: torch.Tensor) -> torch.Tensor:
    return gt_corners.mean(dim=1)  # [N,3]


def gt_boxes_center(ego) -> torch.Tensor:
    mask = ego["object_bbx_mask"][0] == 1
    return ego["object_bbx_center"][0][mask][:, :7].detach()


def gt_center_from_corners_xy(gt_corners: torch.Tensor) -> np.ndarray:
    # polygon centroid (2D, BEV)
    return gt_corners[:, :, :2].mean(dim=1).detach().cpu().numpy()


def one_to_one_match(
    pred_corners: torch.Tensor,
    gt_corners: torch.Tensor,
    scores: torch.Tensor,
    iou_thr: float,
) -> Tuple[set, set, List[Dict[str, Any]]]:
    """Greedy highest-score-first one-to-one matching at IoU >= thr."""
    n_pr = int(pred_corners.shape[0])
    n_gt = int(gt_corners.shape[0])
    matched_pr: set = set()
    matched_gt: set = set()
    pairs: List[Dict[str, Any]] = []
    if n_pr == 0 or n_gt == 0:
        return matched_pr, matched_gt, pairs
    ious = iou_matrix(pred_corners, gt_corners)
    order = torch.argsort(scores, descending=True).tolist()
    for pi in order:
        if ious.shape[1] == 0:
            break
        gi = int(np.argmax(ious[pi]))
        if ious[pi, gi] >= iou_thr and gi not in matched_gt:
            matched_pr.add(pi)
            matched_gt.add(gi)
            pairs.append(
                {
                    "pred_index": int(pi),
                    "gt_index": gi,
                    "iou": float(ious[pi, gi]),
                    "score": float(scores[pi]),
                }
            )
    return matched_pr, matched_gt, pairs


def classify_fp(
    best_iou: float,
    nearest_dist_m: float,
    lidar_range: List[float],
) -> str:
    diag = math.hypot(
        float(lidar_range[3]) - float(lidar_range[0]),
        float(lidar_range[4]) - float(lidar_range[1]),
    )
    far_thr = 0.10 * diag  # ~28 m on 281 m diagonal
    if best_iou >= 0.3:
        return "1_duplicate_near_gt"
    if best_iou >= 0.1:
        return "2_localization_fp"
    if nearest_dist_m > far_thr:
        return "3_far_background"
    return "2_localization_fp_low_iou"


def evaluate_policy(
    policy_scores: torch.Tensor,
    keep_mask: torch.Tensor,
    objectness: torch.Tensor,
    class_scores: torch.Tensor,
    class_labels: torch.Tensor,
    boxes: torch.Tensor,
    corners: torch.Tensor,
    gt_corners: torch.Tensor,
    gt_centers_xy: np.ndarray,
    post,
    hypes: Dict[str, Any],
    nms_thresh: float,
    lidar_range: List[float],
    score_name: str = "obj",
) -> Dict[str, Any]:
    idx = torch.nonzero(keep_mask, as_tuple=False).view(-1)
    kept_corners = corners[idx]
    kept_scores = policy_scores[idx]
    kept_obj = objectness[idx]
    kept_cls = class_scores[idx]
    kept_labels = class_labels[idx]
    kept_boxes = boxes[idx]
    # production NMS + range filter
    n_before_nms = int(kept_corners.shape[0])
    if n_before_nms:
        keep_nms = box_utils.nms_rotated(kept_corners, kept_scores, nms_thresh)
        keep_nms = torch.as_tensor(keep_nms, device=kept_corners.device, dtype=torch.long)
        kept_corners = kept_corners[keep_nms]
        kept_scores = kept_scores[keep_nms]
        kept_obj = kept_obj[keep_nms]
        kept_cls = kept_cls[keep_nms]
        kept_labels = kept_labels[keep_nms]
        kept_boxes = kept_boxes[keep_nms]
    if int(kept_corners.shape[0]):
        keep_r = box_utils.get_mask_for_boxes_within_range_torch(kept_corners, lidar_range)
        kept_corners = kept_corners[keep_r]
        kept_scores = kept_scores[keep_r]
        kept_obj = kept_obj[keep_r]
        kept_cls = kept_cls[keep_r]
        kept_labels = kept_labels[keep_r]
        kept_boxes = kept_boxes[keep_r]
    n_final = int(kept_corners.shape[0])
    n_gt = int(gt_corners.shape[0])

    matched_pr, matched_gt, pairs = one_to_one_match(
        kept_corners, gt_corners, kept_scores, iou_thr=0.3
    )
    tp = len(matched_gt)
    fp = n_final - len(matched_pr)
    precision = tp / max(n_final, 1)
    recall = tp / max(n_gt, 1)

    ious = iou_matrix(kept_corners, gt_corners)
    dists = center_distance(kept_boxes, gt_centers_xy)
    fp_records = []
    for pi in range(n_final):
        if pi in matched_pr:
            continue
        best_iou = float(ious[pi].max()) if n_gt else 0.0
        nearest = float(dists[pi].min()) if n_gt else float("inf")
        cat = classify_fp(best_iou, nearest, lidar_range)
        fp_records.append(
            {
                "pred_index": pi,
                "objectness": float(kept_obj[pi]),
                "class_score": float(kept_cls[pi]),
                "pred_class": int(kept_labels[pi]),
                "max_iou_to_gt": best_iou,
                "nearest_gt_center_dist_m": nearest,
                "category": cat,
                "box_x": float(kept_boxes[pi, 0]),
                "box_y": float(kept_boxes[pi, 1]),
                "box_z": float(kept_boxes[pi, 2]),
                "box_h": float(kept_boxes[pi, 3]),
                "box_w": float(kept_boxes[pi, 4]),
                "box_l": float(kept_boxes[pi, 5]),
                "box_yaw": float(kept_boxes[pi, 6]),
            }
        )
    tp_records = [
        {
            "pred_index": p["pred_index"],
            "gt_index": p["gt_index"],
            "iou": p["iou"],
            "score": p["score"],
            "objectness": float(kept_obj[p["pred_index"]]),
            "class_score": float(kept_cls[p["pred_index"]]),
            "pred_class": int(kept_labels[p["pred_index"]]),
        }
        for p in pairs
    ]
    return {
        "policy": {
            "obj_thresh": 0.2,
            "cls_used": None,
            "score_used": score_name,
        },
        "n_threshold": n_before_nms,
        "n_final": n_final,
        "gt_count": n_gt,
        "tp": tp,
        "fp": fp,
        "precision": precision,
        "recall_at_0.3": recall,
        "fp_list": fp_records,
        "tp_list": tp_records,
        "unmatched_gt": sorted(set(range(n_gt)) - matched_gt),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    hypes = load_hypes(model_dir)
    model = load_model(hypes, model_dir / args.ckpt, device)

    ds_hypes = dict(hypes)
    ds_hypes["validate_dir"] = hypes["test_dir"]
    test_set = build_dataset(ds_hypes, visualize=False, train=False)
    batch = test_set.collate_batch_test([test_set[args.index]])
    batch = train_utils.to_device(batch, device)
    batch["ego"]["epoch"] = 0

    ego = batch["ego"]
    tfm = ego["transformation_matrix"].to(device)
    anchors = ego["anchor_box"].to(device)
    output = forward_heads(model, batch, device)
    objectness, class_scores, class_labels, boxes, corners = decode_scores(
        output, anchors, hypes, tfm
    )
    # GT in ego frame (production generate_gt_bbx_airv2x)
    gt_corners, _, _ = test_set.post_processor.generate_gt_bbx_airv2x(batch)
    gt_centers_xy = gt_center_from_corners_xy(gt_corners)

    nms_thresh = float(hypes["postprocess"]["nms_thresh"])
    lidar_range = hypes["postprocess"]["anchor_args"]["cav_lidar_range"]

    policies: Dict[str, Dict[str, Any]] = {}
    common = dict(
        objectness=objectness,
        class_scores=class_scores,
        class_labels=class_labels,
        boxes=boxes,
        corners=corners,
        gt_corners=gt_corners,
        gt_centers_xy=gt_centers_xy,
        post=test_set.post_processor,
        hypes=hypes,
        nms_thresh=nms_thresh,
        lidar_range=lidar_range,
    )
    policies["A_obj_only"] = evaluate_policy(
        objectness, objectness > 0.2, **common, score_name="obj"
    )
    policies["B_obj_and_cls"] = evaluate_policy(
        (objectness + class_scores),  # only ranking; mask is what matters
        (objectness > 0.2) & (class_scores > 0.2),
        **common,
        score_name="obj>0.2 & cls>0.2; rank=obj+cls",
    )
    policies["C_obj_times_cls"] = evaluate_policy(
        objectness * class_scores,
        (objectness * class_scores) > 0.2,
        **common,
        score_name="obj*cls>0.2",
    )
    for name, res in policies.items():
        print(
            f"{name}: final={res['n_final']} TP={res['tp']} FP={res['fp']} "
            f"prec={res['precision']:.3f} R@0.3={res['recall_at_0.3']:.3f}"
        )

    # FP category summary under production policy A
    cat_counts: Dict[str, int] = {}
    for rec in policies["A_obj_only"]["fp_list"]:
        cat_counts[rec["category"]] = cat_counts.get(rec["category"], 0) + 1

    report = {
        "ckpt": str(model_dir / args.ckpt),
        "test_index": int(args.index),
        "objectness_stats": {
            "mean": float(objectness.mean()),
            "max": float(objectness.max()),
            "n_gt_0.2": int((objectness > 0.2).sum()),
        },
        "class_score_stats": {
            "mean": float(class_scores.mean()),
            "max": float(class_scores.max()),
            "n_gt_0.2": int((class_scores > 0.2).sum()),
        },
        "gt_count": int(gt_corners.shape[0]),
        "fp_category_counts_policyA": cat_counts,
        "policies": policies,
    }
    out = Path(args.out) if args.out else model_dir / "fp_source_audit_epoch5.json"
    out.write_text(json.dumps(_jsonable(report), indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()

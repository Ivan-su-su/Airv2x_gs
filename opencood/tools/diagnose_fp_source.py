#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AirV2X Gaussian FP causal diagnostic.

What it does on the SAME checkpoint / SAME frames:
1) Stage-3 residual alpha black-box: alpha in {0, 0.5, 1} by default.
2) Tracks every FINAL post-NMS prediction back to its original anchor id.
3) Labels that source anchor as positive / explicit_negative / ignore.
4) Reports objectness distributions on all positive/negative anchors.
5) Greedy TP/FP matching exactly follows eval_utils_opv2v logic at IoU=0.3.
6) Saves per-prediction CSV and aggregate JSON.

Required one-line debug hook in current local Stage-3 forward:
    alpha = float(getattr(self, "debug_residual_alpha", 1.0))
    fused = pooled_feature + alpha * delta

Default alpha=1.0 preserves production behavior and adds no checkpoint parameter.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(__file__)
# If saved under opencood/tools/, repo root is 3 levels up; otherwise use cwd.
_candidate = "/".join(ROOT.split("/")[:-3])
if os.path.isdir(os.path.join(_candidate, "opencood")):
    sys.path.insert(0, _candidate)
else:
    sys.path.insert(0, os.getcwd())

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import box_utils, common_utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--alphas", type=str, default="0,0.5,1")
    p.add_argument("--iou", type=float, default=0.30)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def finite_stats(x):
    if x is None or x.numel() == 0:
        return {"n": 0}
    x = x.detach().float().reshape(-1)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {"n": 0}
    q = torch.quantile(x, torch.tensor([0.5, 0.9, 0.99], device=x.device))
    return {
        "n": int(x.numel()),
        "mean": float(x.mean().item()),
        "median": float(q[0].item()),
        "p90": float(q[1].item()),
        "p99": float(q[2].item()),
        "max": float(x.max().item()),
    }


def find_label_dict(obj):
    """Find dict containing pos_equal_one and neg_equal_one without assuming nesting."""
    if isinstance(obj, dict):
        if "pos_equal_one" in obj and "neg_equal_one" in obj:
            return obj
        for v in obj.values():
            found = find_label_dict(v)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found = find_label_dict(v)
            if found is not None:
                return found
    return None


def extract_official(dataset, batch_data, model_output):
    """Run the repo's official post-process only to obtain official pred + GT."""
    output_dict = OrderedDict(ego=model_output)
    result = dataset.post_process(batch_data, output_dict)
    if not isinstance(result, (tuple, list)):
        raise RuntimeError(f"Unexpected dataset.post_process type: {type(result)}")
    # AirV2X: pred, score, labels, pred_boxes3d, gt, gt_cls, gt_track
    if len(result) >= 7:
        return result[0], result[1], result[4]
    if len(result) >= 3:
        return result[0], result[1], result[2]
    raise RuntimeError(f"Unexpected dataset.post_process return length: {len(result)}")


def trace_post_process_airv2x(batch_data, model_output, post_processor):
    """Replicate current AirV2X production post-process while carrying anchor ids."""
    if "ego" not in batch_data:
        raise KeyError("Expected batch_data['ego'] for intermediate fusion")
    cav = batch_data["ego"]
    device = model_output["obj"].device

    transformation_matrix = cav["transformation_matrix"]
    anchor_box = cav["anchor_box"]
    if torch.is_tensor(transformation_matrix):
        transformation_matrix = transformation_matrix.to(device)
    if torch.is_tensor(anchor_box):
        anchor_box = anchor_box.to(device)

    # ---------- raw heads ----------
    obj_logits = model_output["obj"].permute(0, 2, 3, 1).contiguous()  # [1,H,W,A]
    objectness = torch.sigmoid(obj_logits).reshape(1, -1)

    psm = model_output["psm"]
    B, AC, H, W = psm.shape
    C = int(post_processor.num_class)
    A = AC // C
    cls_prob = torch.sigmoid(
        psm.view(B, C, A, H, W).permute(0, 3, 4, 2, 1).contiguous()
    ).reshape(B, -1, C)
    # production ignores background channel 0 for class label, but candidate gate is obj-only
    fg_prob = cls_prob[:, :, 1:]
    class_scores, class_labels = torch.max(fg_prob, dim=-1)
    class_labels = class_labels + 1

    N = objectness.shape[1]
    anchor_ids_all = torch.arange(N, device=device, dtype=torch.long)

    obj_thresh = float(post_processor.params["target_args"]["obj_threshold"])
    candidate_mask = objectness[0] > obj_thresh
    candidate_count = int(candidate_mask.sum().item())

    # regression decode
    reg = model_output["rm"]
    reg_flat = reg.permute(0, 2, 3, 1).contiguous().view(B, -1, 7)
    decoded_all = post_processor.delta_to_boxes3d(reg, anchor_box)[0]  # [N,7]
    anchor_flat = anchor_box.reshape(-1, 7).float()
    if anchor_flat.shape[0] != N:
        raise RuntimeError(f"anchor count {anchor_flat.shape[0]} != obj count {N}")

    anchor_ids = anchor_ids_all[candidate_mask]
    decoded = decoded_all[candidate_mask]
    deltas = reg_flat[0][candidate_mask]
    anchors = anchor_flat[candidate_mask]
    scores = objectness[0][candidate_mask]
    cls_scores = class_scores[0][candidate_mask]
    labels = class_labels[0][candidate_mask]

    if decoded.numel() == 0:
        return {
            "corners": None,
            "scores": None,
            "anchor_ids": None,
            "candidate_count": 0,
            "objectness": objectness[0],
            "class_scores_all": class_scores[0],
        }

    corners = box_utils.boxes_to_corners_3d(decoded, order=post_processor.params["order"])
    projected = box_utils.project_box3d(corners, transformation_matrix)

    # production prefilters
    keep1 = box_utils.remove_large_pred_bbx(projected, post_processor.dataset)
    z_min, z_max = post_processor.lidar_range[2], post_processor.lidar_range[5]
    try:
        keep2 = box_utils.remove_bbx_abnormal_z(projected, z_min=z_min, z_max=z_max)
    except TypeError:
        keep2 = box_utils.remove_bbx_abnormal_z(projected)
    keep = torch.logical_and(keep1, keep2)

    def apply_mask(m):
        nonlocal projected, scores, labels, decoded, deltas, anchors, anchor_ids, cls_scores
        projected = projected[m]
        scores = scores[m]
        labels = labels[m]
        decoded = decoded[m]
        deltas = deltas[m]
        anchors = anchors[m]
        anchor_ids = anchor_ids[m]
        cls_scores = cls_scores[m]

    apply_mask(keep)
    pre_nms_count = int(projected.shape[0])

    if projected.shape[0] > 0:
        keep_nms = box_utils.nms_rotated(
            projected, scores, post_processor.params["nms_thresh"]
        )
        apply_mask(keep_nms)
    post_nms_count = int(projected.shape[0])

    if projected.shape[0] > 0:
        keep_range = box_utils.get_mask_for_boxes_within_range_torch(
            projected, post_processor.lidar_range
        )
        apply_mask(keep_range)

    return {
        "corners": projected,
        "scores": scores,
        "labels": labels,
        "class_scores": cls_scores,
        "decoded": decoded,
        "deltas": deltas,
        "anchors": anchors,
        "anchor_ids": anchor_ids,
        "candidate_count": candidate_count,
        "pre_nms_count": pre_nms_count,
        "post_nms_count": post_nms_count,
        "objectness": objectness[0],
        "class_scores_all": class_scores[0],
        "obj_thresh": obj_thresh,
    }


def greedy_match_like_repo(pred_corners, scores, gt_corners, iou_thresh):
    """Match exactly like eval_utils_opv2v.caluclate_tp_fp; return per-original-pred info."""
    n = 0 if pred_corners is None else int(pred_corners.shape[0])
    out = [
        {"is_tp": False, "is_fp": True, "nearest_iou": 0.0, "nearest_gt": -1}
        for _ in range(n)
    ]
    if n == 0:
        return out

    det_np = common_utils.torch_tensor_to_numpy(pred_corners)
    score_np = common_utils.torch_tensor_to_numpy(scores)
    gt_np = common_utils.torch_tensor_to_numpy(gt_corners)
    det_polys = list(common_utils.convert_format(det_np))
    gt_polys_all = list(common_utils.convert_format(gt_np))
    available = list(range(len(gt_polys_all)))
    order = np.argsort(-score_np)

    for det_idx in order:
        # nearest GT over all GTs (diagnostic, independent of greedy removal)
        if gt_polys_all:
            ious_all = common_utils.compute_iou(det_polys[det_idx], gt_polys_all)
            nearest_local = int(np.argmax(ious_all))
            nearest_iou = float(np.max(ious_all))
        else:
            nearest_local, nearest_iou = -1, 0.0
        out[int(det_idx)]["nearest_gt"] = nearest_local
        out[int(det_idx)]["nearest_iou"] = nearest_iou

        # official greedy assignment uses only unmatched GTs
        if not available:
            continue
        avail_polys = [gt_polys_all[j] for j in available]
        ious = common_utils.compute_iou(det_polys[det_idx], avail_polys)
        if len(ious) == 0 or float(np.max(ious)) < iou_thresh:
            continue
        j = int(np.argmax(ious))
        available.pop(j)
        out[int(det_idx)]["is_tp"] = True
        out[int(det_idx)]["is_fp"] = False

    return out


def anchor_source(anchor_id, pos_flat, neg_flat):
    if pos_flat is None or neg_flat is None:
        return "unknown"
    if anchor_id < 0 or anchor_id >= pos_flat.numel():
        return "out_of_range"
    if float(pos_flat[anchor_id].item()) > 0.5:
        return "positive"
    if float(neg_flat[anchor_id].item()) > 0.5:
        return "explicit_negative"
    return "ignore"


def center_xy_from_corners(corners):
    if corners is None or corners.numel() == 0:
        return torch.empty((0, 2), device=corners.device if corners is not None else "cpu")
    return corners[:, :, :2].mean(dim=1)


def run_one_frame(frame_idx, batch_data, model, dataset, alpha, iou_thresh):
    model.stage3.debug_residual_alpha = float(alpha)
    with torch.no_grad():
        output = model(batch_data["ego"])

    official_pred, official_score, gt = extract_official(dataset, batch_data, output)
    trace = trace_post_process_airv2x(batch_data, output, dataset.post_processor)

    trace_n = 0 if trace["corners"] is None else int(trace["corners"].shape[0])
    official_n = 0 if official_pred is None else int(official_pred.shape[0])
    count_match = (trace_n == official_n)

    label_dict = find_label_dict(batch_data.get("ego", batch_data))
    pos_flat = neg_flat = None
    if label_dict is not None:
        pos_flat = label_dict["pos_equal_one"].reshape(-1).detach()
        neg_flat = label_dict["neg_equal_one"].reshape(-1).detach()

    obj_all = trace["objectness"]
    pos_obj = obj_all[pos_flat > 0.5] if pos_flat is not None and pos_flat.numel() == obj_all.numel() else None
    neg_obj = obj_all[neg_flat > 0.5] if neg_flat is not None and neg_flat.numel() == obj_all.numel() else None
    ign_obj = obj_all[(pos_flat <= 0.5) & (neg_flat <= 0.5)] if pos_flat is not None and pos_flat.numel() == obj_all.numel() else None

    matches = greedy_match_like_repo(trace["corners"], trace["scores"], gt, iou_thresh)
    n_gt = int(gt.shape[0])
    tp = sum(int(m["is_tp"]) for m in matches)
    fp = sum(int(m["is_fp"]) for m in matches)

    gt_centers = center_xy_from_corners(gt)
    pred_centers = center_xy_from_corners(trace["corners"]) if trace["corners"] is not None else None

    rows = []
    fp_sources = Counter()
    if trace_n > 0:
        for j in range(trace_n):
            aid = int(trace["anchor_ids"][j].item())
            src = anchor_source(aid, pos_flat, neg_flat)
            m = matches[j]
            if m["is_fp"]:
                fp_sources[src] += 1

            nearest_gt = int(m["nearest_gt"])
            if nearest_gt >= 0 and gt_centers.numel() > 0:
                center_dist = float(torch.linalg.vector_norm(pred_centers[j] - gt_centers[nearest_gt]).item())
            else:
                center_dist = float("nan")

            anchor = trace["anchors"][j]
            decoded = trace["decoded"][j]
            delta = trace["deltas"][j]
            shift_xy = float(torch.linalg.vector_norm(decoded[:2] - anchor[:2]).item())

            rows.append({
                "frame": int(frame_idx),
                "alpha": float(alpha),
                "pred_index": int(j),
                "is_tp_iou03": int(m["is_tp"]),
                "is_fp_iou03": int(m["is_fp"]),
                "anchor_id": aid,
                "anchor_source": src,
                "obj": float(trace["scores"][j].item()),
                "class_score": float(trace["class_scores"][j].item()),
                "class_label": int(trace["labels"][j].item()),
                "nearest_gt_iou": float(m["nearest_iou"]),
                "nearest_gt_center_dist_m": center_dist,
                "anchor_x": float(anchor[0].item()),
                "anchor_y": float(anchor[1].item()),
                "decoded_x": float(decoded[0].item()),
                "decoded_y": float(decoded[1].item()),
                "reg_center_shift_xy_m": shift_xy,
                "reg_dx": float(delta[0].item()),
                "reg_dy": float(delta[1].item()),
                "reg_dz": float(delta[2].item()),
                "reg_dh": float(delta[3].item()),
                "reg_dw": float(delta[4].item()),
                "reg_dl": float(delta[5].item()),
                "reg_dyaw": float(delta[6].item()),
            })

    summary = {
        "frame": int(frame_idx),
        "alpha": float(alpha),
        "n_gt": n_gt,
        "candidate_obj_pass": int(trace["candidate_count"]),
        "pre_nms": int(trace.get("pre_nms_count", 0)),
        "post_nms": int(trace.get("post_nms_count", 0)),
        "final": trace_n,
        "official_final": official_n,
        "trace_count_matches_official": bool(count_match),
        "tp_iou03": tp,
        "fp_iou03": fp,
        "precision_iou03": float(tp / max(tp + fp, 1)),
        "recall_iou03": float(tp / max(n_gt, 1)),
        "fp_anchor_sources": dict(fp_sources),
        "pos_obj": finite_stats(pos_obj),
        "neg_obj": finite_stats(neg_obj),
        "ignore_obj": finite_stats(ign_obj),
    }
    return summary, rows


def aggregate(frame_summaries):
    if not frame_summaries:
        return {}
    tp = sum(x["tp_iou03"] for x in frame_summaries)
    fp = sum(x["fp_iou03"] for x in frame_summaries)
    gt = sum(x["n_gt"] for x in frame_summaries)
    src = Counter()
    for x in frame_summaries:
        src.update(x["fp_anchor_sources"])
    return {
        "frames": len(frame_summaries),
        "gt": gt,
        "tp_iou03": tp,
        "fp_iou03": fp,
        "precision_iou03": float(tp / max(tp + fp, 1)),
        "recall_iou03": float(tp / max(gt, 1)),
        "candidate_obj_pass_mean": float(np.mean([x["candidate_obj_pass"] for x in frame_summaries])),
        "final_mean": float(np.mean([x["final"] for x in frame_summaries])),
        "fp_anchor_sources": dict(src),
        "trace_count_match_fraction": float(np.mean([x["trace_count_matches_official"] for x in frame_summaries])),
        "neg_obj_mean_mean": float(np.mean([x["neg_obj"].get("mean", np.nan) for x in frame_summaries])),
        "neg_obj_p90_mean": float(np.mean([x["neg_obj"].get("p90", np.nan) for x in frame_summaries])),
        "neg_obj_p99_mean": float(np.mean([x["neg_obj"].get("p99", np.nan) for x in frame_summaries])),
        "pos_obj_mean_mean": float(np.mean([x["pos_obj"].get("mean", np.nan) for x in frame_summaries])),
    }


def main():
    args = parse_args()
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    out_dir = args.out or os.path.join(args.model_dir, f"fp_diag_epoch{args.epoch}")
    os.makedirs(out_dir, exist_ok=True)

    hypes = yaml_utils.load_yaml(None, args)
    if "test_dir" in hypes:
        hypes["validate_dir"] = hypes["test_dir"]

    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _, model = train_utils.load_model(args.model_dir, model, args.epoch, start_from_best=False)
    model.eval()
    model.zero_grad(set_to_none=True)

    if not hasattr(model, "stage3"):
        raise RuntimeError("model.stage3 not found")

    all_json = {
        "model_dir": args.model_dir,
        "epoch": args.epoch,
        "alphas": alphas,
        "iou": args.iou,
        "per_alpha": {},
    }
    all_rows = []
    per_alpha_frames = {a: [] for a in alphas}

    end = args.start + args.num_frames
    for i, batch_data in tqdm(enumerate(loader), total=min(len(loader), end)):
        if i < args.start:
            continue
        if i >= end:
            break
        batch_data = train_utils.to_device(batch_data, device)

        for alpha in alphas:
            summary, rows = run_one_frame(i, batch_data, model, dataset, alpha, args.iou)
            per_alpha_frames[alpha].append(summary)
            all_rows.extend(rows)
            print(
                f"[frame {i:04d} alpha={alpha:g}] "
                f"cand={summary['candidate_obj_pass']} final={summary['final']} "
                f"TP={summary['tp_iou03']} FP={summary['fp_iou03']} "
                f"P={summary['precision_iou03']:.3f} R={summary['recall_iou03']:.3f} "
                f"FPsrc={summary['fp_anchor_sources']}"
            )

    for alpha in alphas:
        frames = per_alpha_frames[alpha]
        all_json["per_alpha"][str(alpha)] = {
            "aggregate": aggregate(frames),
            "frames": frames,
        }

    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_json, f, ensure_ascii=False, indent=2, allow_nan=True)

    csv_path = os.path.join(out_dir, "final_predictions.csv")
    if all_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    print("\n=== AGGREGATE ===")
    for alpha in alphas:
        a = all_json["per_alpha"][str(alpha)]["aggregate"]
        print(
            f"alpha={alpha:g}: TP={a.get('tp_iou03',0)} FP={a.get('fp_iou03',0)} "
            f"P={a.get('precision_iou03',0):.4f} R={a.get('recall_iou03',0):.4f} "
            f"cand_mean={a.get('candidate_obj_pass_mean',0):.1f} "
            f"final_mean={a.get('final_mean',0):.1f} "
            f"FPsrc={a.get('fp_anchor_sources',{})} "
            f"neg_p90={a.get('neg_obj_p90_mean',float('nan')):.4f} "
            f"neg_p99={a.get('neg_obj_p99_mean',float('nan')):.4f}"
        )
    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()

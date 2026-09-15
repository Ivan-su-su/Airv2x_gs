#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference-only diagnostic for AirV2X:
test whether the current objectness-only post-process is producing many FPs
that the trained class head could reject.

IMPORTANT
---------
This script DOES NOT modify production code, checkpoint, loss, NMS, Gaussian
geometry, or model parameters.

For every frame:
  1) run the model ONCE
  2) baseline = exact current dataset.post_process output
  3) correctly reshape the class head exactly as training loss does:
         [B, A*C, H, W]
      -> [B, H, W, A, C]
  4) for each class threshold, create an inference-only gate:
         objectness > production obj_threshold
         AND
         max foreground class score > class_threshold
     The gate is injected by setting obj logits to a very negative value ONLY
     at class-rejected anchors, then calling the exact same production
     dataset.post_process again.
     Therefore decode / abnormal-box filtering / NMS / range filtering stay
     exactly unchanged.
  5) compare FINAL post-NMS TP / FP / precision / recall at IoU=0.3
  6) save BEV visualizations for baseline and one chosen class threshold.

Outputs
-------
<model_dir>/class_gate_diag_epoch<E>/
  class_gate_summary.json
  per_frame_metrics.csv
  class_score_anchor_stats.csv
  metrics_vs_class_threshold.png
  vis/
      frame_XXXX_baseline.png
      frame_XXXX_class_0p20.png
      ...

Typical usage
-------------
cd /data2/home/suyi/AirV2X-Perception_gaussian

python opencood/tools/diagnose_class_gate.py \
  --model_dir /mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/airv2x_gaussian_0822_camera/v45_scale_asym_2026_09_13_16_57_09 \
  --epoch 3 \
  --num_frames 50 \
  --workers 4 \
  --iou 0.3 \
  --class_thresholds 0.05,0.10,0.20,0.30,0.50 \
  --vis_class_thresh 0.20 \
  --vis_frames 10
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# repo import
# ---------------------------------------------------------------------
_THIS = os.path.abspath(__file__)
_REPO = "/".join(_THIS.split("/")[:-3])
if os.path.isdir(os.path.join(_REPO, "opencood")):
    sys.path.insert(0, _REPO)
else:
    sys.path.insert(0, os.getcwd())

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils_opv2v as eval_utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--num_frames", type=int, default=50)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--iou", type=float, default=0.3)

    p.add_argument(
        "--class_thresholds",
        type=str,
        default="0.05,0.10,0.20,0.30,0.50",
        help="Foreground-class thresholds for inference-only class gate.",
    )
    p.add_argument(
        "--vis_class_thresh",
        type=float,
        default=0.20,
        help="Which class-gate threshold to visualize.",
    )
    p.add_argument(
        "--vis_frames",
        type=int,
        default=10,
        help="Number of frames to visualize from the processed range.",
    )
    p.add_argument(
        "--reject_logit",
        type=float,
        default=-30.0,
        help="Obj logit assigned to anchors rejected by the class gate.",
    )
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def qstats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p99": float(np.quantile(x, 0.99)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def threshold_key(x: float) -> str:
    return f"{x:.2f}"


def filename_threshold(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def clone_output_dict_with_obj(model_output: Dict[str, torch.Tensor],
                               new_obj: torch.Tensor):
    out = dict(model_output)
    out["obj"] = new_obj
    return OrderedDict([("ego", out)])


def baseline_output_dict(model_output):
    return OrderedDict([("ego", model_output)])


def correct_class_scores(model_output: Dict[str, torch.Tensor],
                         num_class: int):
    """
    EXACT same channel interpretation as the current training loss:

        cls_preds = psm.permute(0,2,3,1)
        input = input.view(B,H,W,A,C)

    Returns
    -------
    foreground_score : [B,H,W,A]
        max sigmoid score over classes 1..C-1
    foreground_label : [B,H,W,A]
        class id in 1..C-1
    background_score : [B,H,W,A]
        sigmoid score of class 0
    full_prob : [B,H,W,A,C]
    """
    psm = model_output["psm"]
    B, AC, H, W = psm.shape
    C = int(num_class)
    if AC % C != 0:
        raise RuntimeError(
            f"psm channels {AC} not divisible by num_class={C}"
        )
    A = AC // C

    logits = (
        psm.permute(0, 2, 3, 1)
           .contiguous()
           .view(B, H, W, A, C)
    )
    prob = torch.sigmoid(logits)

    if C <= 1:
        raise RuntimeError(f"Expected background + foreground classes, C={C}")

    fg_score, fg_label0 = torch.max(prob[..., 1:], dim=-1)
    fg_label = fg_label0 + 1
    bg_score = prob[..., 0]
    return fg_score, fg_label, bg_score, prob


def gated_obj_logits(model_output: Dict[str, torch.Tensor],
                     fg_score: torch.Tensor,
                     class_thresh: float,
                     reject_logit: float):
    """
    Keep production objectness values unchanged for class-approved anchors.
    For class-rejected anchors only, force obj logit << production threshold.

    This lets us reuse EXACT production postprocess/NMS.
    """
    obj = model_output["obj"]  # [B,A,H,W]
    obj_hw = obj.permute(0, 2, 3, 1).contiguous()  # [B,H,W,A]

    if obj_hw.shape != fg_score.shape:
        raise RuntimeError(
            f"obj/class shape mismatch: obj={tuple(obj_hw.shape)} "
            f"class={tuple(fg_score.shape)}"
        )

    keep_cls = fg_score > float(class_thresh)
    new_obj_hw = torch.where(
        keep_cls,
        obj_hw,
        torch.full_like(obj_hw, float(reject_logit)),
    )
    return new_obj_hw.permute(0, 3, 1, 2).contiguous(), keep_cls


def parse_postprocess_result(result):
    """
    Supports common AirV2X/OpenCOOD post_process return layouts.

    We need only:
      pred_box: first item
      pred_score: second item
      gt_box: first later tensor shaped [N,8,3]

    Examples seen in this repo:
      (pred_box, pred_score, gt_box)
      (pred_box, pred_score, labels, boxes3d, gt_box, gt_cls, gt_track)
      (pred_box, pred_score, labels, gt_box, gt_cls, gt_track)
    """
    if result is None:
        return None, None, None

    if not isinstance(result, (tuple, list)):
        raise RuntimeError(
            f"Unexpected dataset.post_process return type: {type(result)}"
        )

    if len(result) < 2:
        raise RuntimeError(
            f"Unexpected dataset.post_process return length: {len(result)}"
        )

    pred_box = result[0]
    pred_score = result[1]
    gt_box = None

    # GT corners are normally [N,8,3]. Search only after pred_score.
    for item in result[2:]:
        if torch.is_tensor(item) and item.ndim == 3:
            if item.shape[-2:] == (8, 3):
                gt_box = item
                break
        elif isinstance(item, np.ndarray) and item.ndim == 3:
            if item.shape[-2:] == (8, 3):
                gt_box = item
                break

    # Three-return standard path: item 2 is GT.
    if gt_box is None and len(result) >= 3:
        cand = result[2]
        if torch.is_tensor(cand) and cand.ndim == 3 and cand.shape[-2:] == (8, 3):
            gt_box = cand
        elif isinstance(cand, np.ndarray) and cand.ndim == 3 and cand.shape[-2:] == (8, 3):
            gt_box = cand

    if gt_box is None:
        raise RuntimeError(
            "Could not locate gt_box [N,8,3] in dataset.post_process result. "
            f"Returned shapes/types: "
            f"{[(type(x).__name__, tuple(x.shape) if hasattr(x,'shape') else None) for x in result]}"
        )

    return pred_box, pred_score, gt_box


def run_post_process(dataset, batch_data, output_dict):
    result = dataset.post_process(batch_data, output_dict)
    return parse_postprocess_result(result)


def init_stat(iou):
    return {iou: {"tp": [], "fp": [], "gt": 0, "score": []}}


def add_eval(stat, pred_box, pred_score, gt_box, iou):
    """
    Use repo's exact evaluation helper when predictions exist.
    Handle None prediction explicitly.
    """
    if gt_box is None:
        n_gt = 0
    else:
        n_gt = int(len(gt_box))

    if pred_box is None or pred_score is None or len(pred_box) == 0:
        stat[iou]["gt"] += n_gt
        return 0, 0, n_gt, 0

    # For a per-frame count, use a temporary stat.
    tmp = init_stat(iou)
    eval_utils.caluclate_tp_fp(
        pred_box, pred_score, gt_box, tmp, iou
    )
    tp = int(np.sum(tmp[iou]["tp"]))
    fp = int(np.sum(tmp[iou]["fp"]))
    gt = int(tmp[iou]["gt"])
    pred = tp + fp

    # Global stat for AP.
    eval_utils.caluclate_tp_fp(
        pred_box, pred_score, gt_box, stat, iou
    )
    return tp, fp, gt, pred


def safe_ap(stat, iou):
    if stat[iou]["gt"] <= 0:
        return float("nan")
    if len(stat[iou]["tp"]) == 0:
        return 0.0
    st = copy.deepcopy(stat)
    try:
        ap, _, _ = eval_utils.calculate_ap(
            st, iou, global_sort_detections=False
        )
        return float(ap)
    except Exception:
        return float("nan")


def as_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rectangle_xy(corners):
    """
    Convert one [8,3] box-corner tensor to ordered unique XY polygon.
    Does not assume a particular corner ordering.
    """
    pts = np.asarray(corners, dtype=np.float64)[:, :2]
    # z duplicates the same XY footprint; round to remove duplicated top/bottom.
    pts = np.unique(np.round(pts, decimals=6), axis=0)
    if len(pts) < 3:
        return None
    center = pts.mean(axis=0)
    angle = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angle)]
    pts = np.vstack([pts, pts[0]])
    return pts


def plot_bev(pred_box, gt_box, title, save_path, lidar_range=None):
    pred_np = as_numpy(pred_box)
    gt_np = as_numpy(gt_box)

    fig, ax = plt.subplots(figsize=(10, 5.2))

    if gt_np is not None:
        first = True
        for box in gt_np:
            poly = rectangle_xy(box)
            if poly is None:
                continue
            ax.plot(
                poly[:, 0], poly[:, 1],
                linewidth=1.5,
                linestyle="--",
                label="GT" if first else None,
            )
            first = False

    if pred_np is not None:
        first = True
        for box in pred_np:
            poly = rectangle_xy(box)
            if poly is None:
                continue
            ax.plot(
                poly[:, 0], poly[:, 1],
                linewidth=1.0,
                label="Prediction" if first else None,
            )
            first = False

    if lidar_range is not None and len(lidar_range) >= 5:
        ax.set_xlim(float(lidar_range[0]), float(lidar_range[3]))
        ax.set_ylim(float(lidar_range[1]), float(lidar_range[4]))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_metric_curve(rows, save_path):
    """
    Single-axes summary plot. Baseline is represented at x=-0.05.
    """
    # Separate baseline and gate points.
    gate = [r for r in rows if r["variant"] != "baseline"]
    gate = sorted(gate, key=lambda r: r["class_threshold"])

    if not gate:
        return

    x = [r["class_threshold"] for r in gate]
    fp = [r["fp_per_frame"] for r in gate]
    recall = [r["recall"] * 100.0 for r in gate]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, fp, marker="o", label="Final FP / frame")
    ax.plot(x, recall, marker="s", label="Recall (%)")
    ax.set_xlabel("Foreground class threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Inference-only class gate: final post-NMS behavior")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def get_num_class(dataset, model_output):
    # Prefer postprocessor field.
    pp = getattr(dataset, "post_processor", None)
    if pp is not None and hasattr(pp, "num_class"):
        return int(pp.num_class)

    # Fallback: common AirV2X value.
    # We intentionally do not infer C from AC alone because A is also unknown.
    raise RuntimeError(
        "Could not find dataset.post_processor.num_class. "
        "Please pass through the AirV2X postprocessor that defines num_class."
    )


def get_lidar_range(dataset):
    pp = getattr(dataset, "post_processor", None)
    if pp is not None and hasattr(pp, "lidar_range"):
        return pp.lidar_range
    return None


def main():
    args = parse_args()

    class_thresholds = [
        float(x.strip())
        for x in args.class_thresholds.split(",")
        if x.strip()
    ]
    class_thresholds = sorted(set(class_thresholds))

    # Ensure requested visualization threshold is evaluated.
    if not any(abs(x - args.vis_class_thresh) < 1e-9 for x in class_thresholds):
        class_thresholds.append(float(args.vis_class_thresh))
        class_thresholds = sorted(set(class_thresholds))

    out_dir = args.out or os.path.join(
        args.model_dir,
        f"class_gate_diag_epoch{args.epoch}",
    )
    vis_dir = os.path.join(out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    hypes = yaml_utils.load_yaml(None, args)
    if "test_dir" in hypes:
        hypes["validate_dir"] = hypes["test_dir"]

    dataset = build_dataset(hypes, visualize=False, train=False)
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
    _, model = train_utils.load_model(
        args.model_dir,
        model,
        args.epoch,
        start_from_best=False,
    )
    model.eval()
    model.zero_grad(set_to_none=True)

    lidar_range = get_lidar_range(dataset)

    variants = ["baseline"] + [
        f"class_{threshold_key(t)}" for t in class_thresholds
    ]
    global_stats = {v: init_stat(args.iou) for v in variants}
    frame_rows = []
    anchor_stat_rows = []

    # Distribution buffers: class behavior conditional on obj gate.
    dist = defaultdict(list)

    start = int(args.start)
    stop = start + int(args.num_frames)
    processed = 0

    for frame_idx, batch_data in tqdm(enumerate(loader)):
        if frame_idx < start:
            continue
        if frame_idx >= stop:
            break

        batch_data = train_utils.to_device(batch_data, device)
        cav_content = batch_data["ego"]

        # -------------------------------------------------------------
        # ONE model forward only
        # -------------------------------------------------------------
        with torch.no_grad():
            model_output = model(cav_content)

        if "obj" not in model_output or "psm" not in model_output:
            raise KeyError(
                f"model_output needs obj and psm, keys={list(model_output.keys())}"
            )

        num_class = get_num_class(dataset, model_output)
        fg_score, fg_label, bg_score, full_prob = correct_class_scores(
            model_output, num_class
        )

        obj_hw = (
            torch.sigmoid(model_output["obj"])
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        if obj_hw.shape != fg_score.shape:
            raise RuntimeError(
                f"obj/class mismatch obj={tuple(obj_hw.shape)}, "
                f"fg={tuple(fg_score.shape)}"
            )

        # -------------------------------------------------------------
        # Baseline = untouched production postprocess
        # -------------------------------------------------------------
        with torch.no_grad():
            b_pred, b_score, gt_box = run_post_process(
                dataset,
                batch_data,
                baseline_output_dict(model_output),
            )

        b_tp, b_fp, b_gt, b_n = add_eval(
            global_stats["baseline"],
            b_pred, b_score, gt_box, args.iou
        )

        frame_rows.append({
            "frame": frame_idx,
            "variant": "baseline",
            "class_threshold": "",
            "pred": b_n,
            "gt": b_gt,
            "tp": b_tp,
            "fp": b_fp,
            "precision": b_tp / max(b_tp + b_fp, 1),
            "recall": b_tp / max(b_gt, 1),
        })

        # -------------------------------------------------------------
        # Anchor-level diagnostic only: how much class head disagrees
        # with current objectness gate?
        # -------------------------------------------------------------
        # Try to read production obj threshold; fallback only if needed.
        obj_thresh = None
        try:
            obj_thresh = float(
                dataset.post_processor.params["target_args"]["obj_threshold"]
            )
        except Exception:
            pass
        if obj_thresh is None:
            raise RuntimeError(
                "Could not read production obj_threshold from "
                "dataset.post_processor.params['target_args']['obj_threshold']"
            )

        obj_pass = obj_hw > obj_thresh
        n_obj_pass = int(obj_pass.sum().item())
        obj_pass_fg = fg_score[obj_pass]
        obj_pass_bg = bg_score[obj_pass]

        if obj_pass_fg.numel() > 0:
            dist["obj_pass_fg_score"].append(
                obj_pass_fg.detach().cpu().numpy()
            )
            dist["obj_pass_bg_score"].append(
                obj_pass_bg.detach().cpu().numpy()
            )

        astat = {
            "frame": frame_idx,
            "obj_threshold": obj_thresh,
            "obj_pass_anchors": n_obj_pass,
            "obj_pass_fg_score_mean":
                float(obj_pass_fg.mean().item()) if obj_pass_fg.numel() else float("nan"),
            "obj_pass_fg_score_median":
                float(obj_pass_fg.median().item()) if obj_pass_fg.numel() else float("nan"),
        }

        for t in class_thresholds:
            keep = obj_pass & (fg_score > t)
            astat[f"obj_and_cls_{threshold_key(t)}_anchors"] = int(
                keep.sum().item()
            )
        anchor_stat_rows.append(astat)

        # -------------------------------------------------------------
        # Class-gated variants, SAME production postprocess/NMS
        # -------------------------------------------------------------
        vis_variant_pred = None
        vis_variant_score = None
        vis_variant_metrics = None

        for t in class_thresholds:
            new_obj, keep_cls = gated_obj_logits(
                model_output,
                fg_score,
                t,
                args.reject_logit,
            )

            with torch.no_grad():
                pred, score, gt_box_t = run_post_process(
                    dataset,
                    batch_data,
                    clone_output_dict_with_obj(model_output, new_obj),
                )

            tp, fp, gt, n_pred = add_eval(
                global_stats[f"class_{threshold_key(t)}"],
                pred, score, gt_box_t, args.iou
            )

            frame_rows.append({
                "frame": frame_idx,
                "variant": f"class_{threshold_key(t)}",
                "class_threshold": t,
                "pred": n_pred,
                "gt": gt,
                "tp": tp,
                "fp": fp,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(gt, 1),
            })

            if abs(t - args.vis_class_thresh) < 1e-9:
                vis_variant_pred = pred
                vis_variant_score = score
                vis_variant_metrics = (tp, fp, gt, n_pred)

        # -------------------------------------------------------------
        # Visualization
        # -------------------------------------------------------------
        if processed < args.vis_frames:
            plot_bev(
                b_pred,
                gt_box,
                (
                    f"Frame {frame_idx} | baseline obj-only | "
                    f"pred={b_n}, TP={b_tp}, FP={b_fp}, GT={b_gt}"
                ),
                os.path.join(
                    vis_dir,
                    f"frame_{frame_idx:04d}_baseline.png",
                ),
                lidar_range=lidar_range,
            )

            if vis_variant_metrics is not None:
                tp, fp, gt, n_pred = vis_variant_metrics
                plot_bev(
                    vis_variant_pred,
                    gt_box,
                    (
                        f"Frame {frame_idx} | obj + class>{args.vis_class_thresh:.2f} | "
                        f"pred={n_pred}, TP={tp}, FP={fp}, GT={gt}"
                    ),
                    os.path.join(
                        vis_dir,
                        f"frame_{frame_idx:04d}_class_"
                        f"{filename_threshold(args.vis_class_thresh)}.png",
                    ),
                    lidar_range=lidar_range,
                )

        # Compact terminal line.
        chosen = [
            r for r in frame_rows
            if r["frame"] == frame_idx
            and r["variant"] == f"class_{threshold_key(args.vis_class_thresh)}"
        ][-1]

        print(
            f"[frame {frame_idx:04d}] "
            f"baseline pred={b_n:3d} TP={b_tp:3d} FP={b_fp:3d} | "
            f"class>{args.vis_class_thresh:.2f} "
            f"pred={chosen['pred']:3d} TP={chosen['tp']:3d} FP={chosen['fp']:3d} | "
            f"obj-pass anchors={n_obj_pass}"
        )

        processed += 1

    # -----------------------------------------------------------------
    # Aggregate final metrics
    # -----------------------------------------------------------------
    summary_rows = []

    for v in variants:
        rows = [r for r in frame_rows if r["variant"] == v]
        if not rows:
            continue

        total_tp = int(sum(r["tp"] for r in rows))
        total_fp = int(sum(r["fp"] for r in rows))
        total_gt = int(sum(r["gt"] for r in rows))
        total_pred = int(sum(r["pred"] for r in rows))

        if v == "baseline":
            t = None
        else:
            t = float(v.split("_", 1)[1])

        summary_rows.append({
            "variant": v,
            "class_threshold": t,
            "frames": len(rows),
            "pred_per_frame": total_pred / len(rows),
            "gt_per_frame": total_gt / len(rows),
            "tp_per_frame": total_tp / len(rows),
            "fp_per_frame": total_fp / len(rows),
            "precision": total_tp / max(total_tp + total_fp, 1),
            "recall": total_tp / max(total_gt, 1),
            "ap": safe_ap(global_stats[v], args.iou),
        })

    baseline = next(r for r in summary_rows if r["variant"] == "baseline")
    for r in summary_rows:
        r["delta_fp_per_frame"] = (
            r["fp_per_frame"] - baseline["fp_per_frame"]
        )
        r["delta_tp_per_frame"] = (
            r["tp_per_frame"] - baseline["tp_per_frame"]
        )
        r["delta_recall"] = (
            r["recall"] - baseline["recall"]
        )
        r["delta_precision"] = (
            r["precision"] - baseline["precision"]
        )

    # Distribution summary
    anchor_distribution = {}
    for k, chunks in dist.items():
        arr = np.concatenate(chunks) if chunks else np.empty(0)
        anchor_distribution[k] = qstats(arr)

    summary = {
        "model_dir": args.model_dir,
        "epoch": args.epoch,
        "frames_processed": processed,
        "iou": args.iou,
        "diagnostic_contract": {
            "model_forward_per_frame": 1,
            "production_postprocess_reused": True,
            "production_nms_reused": True,
            "nms_score_unchanged": "objectness",
            "class_layout": "[B,A*C,H,W] -> permute(B,H,W,A*C) -> view(B,H,W,A,C)",
            "foreground_class_score": "max(sigmoid(class_logits[...,1:]))",
            "production_code_modified": False,
        },
        "results": summary_rows,
        "obj_pass_class_score_distribution": anchor_distribution,
    }

    # JSON
    json_path = os.path.join(out_dir, "class_gate_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=True)

    # Per-frame CSV
    frame_csv = os.path.join(out_dir, "per_frame_metrics.csv")
    with open(frame_csv, "w", encoding="utf-8", newline="") as f:
        fields = [
            "frame", "variant", "class_threshold",
            "pred", "gt", "tp", "fp", "precision", "recall",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(frame_rows)

    # Anchor CSV
    anchor_csv = os.path.join(out_dir, "class_score_anchor_stats.csv")
    if anchor_stat_rows:
        with open(anchor_csv, "w", encoding="utf-8", newline="") as f:
            fields = list(anchor_stat_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(anchor_stat_rows)

    # Summary plot
    plot_metric_curve(
        summary_rows,
        os.path.join(out_dir, "metrics_vs_class_threshold.png"),
    )

    # Terminal summary
    print("\n================ FINAL POST-NMS CLASS-GATE A/B ================")
    print(
        f"{'variant':>14s} {'pred/fr':>9s} {'TP/fr':>8s} {'FP/fr':>8s} "
        f"{'Prec':>8s} {'Recall':>8s} {'AP':>8s} {'dFP/fr':>9s}"
    )
    for r in summary_rows:
        print(
            f"{r['variant']:>14s} "
            f"{r['pred_per_frame']:9.2f} "
            f"{r['tp_per_frame']:8.2f} "
            f"{r['fp_per_frame']:8.2f} "
            f"{r['precision']:8.4f} "
            f"{r['recall']:8.4f} "
            f"{r['ap']:8.4f} "
            f"{r['delta_fp_per_frame']:9.2f}"
        )

    print("\nInterpretation:")
    print(
        "- If FP/frame drops strongly with little recall loss, the class head "
        "is rejecting many objectness hard-negatives; this supports an obj-head "
        "calibration/loss problem."
    )
    print(
        "- If FP/frame barely drops, or recall collapses together with FP, "
        "adding a class gate is not a useful fix; then go directly back to "
        "the objectness loss/calibration."
    )
    print(
        "- NMS scoring remains objectness in ALL variants, so this experiment "
        "isolates only the added foreground-class gate."
    )

    print(f"\nsaved: {json_path}")
    print(f"saved: {frame_csv}")
    print(f"saved: {anchor_csv}")
    print(f"saved visualizations under: {vis_dir}")


if __name__ == "__main__":
    main()

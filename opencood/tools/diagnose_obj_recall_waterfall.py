#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Inference-only objectness threshold + recall-waterfall diagnostic for AirV2X.

One model forward per frame. The script only sweeps the production objectness
threshold in post-processing, restores it afterwards, and reports where GT
coverage is lost: regression -> obj gate -> pre-NMS -> NMS -> final.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.abspath(__file__)
_REPO = "/".join(_THIS.split("/")[:-3])
if os.path.isdir(os.path.join(_REPO, "opencood")):
    sys.path.insert(0, _REPO)
else:
    sys.path.insert(0, os.getcwd())

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import box_utils, common_utils
from opencood.utils import eval_utils_opv2v as eval_utils

DEFAULT_MODEL_DIR = (
    "/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
    "airv2x_gaussian_0822_objfocal/v45_objfocal_2026_09_13_21_11_11"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--epoch", type=int, default=13)
    p.add_argument("--num_frames", type=int, default=50)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--thresholds", default="0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.30")
    p.add_argument("--coverage_ious", default="0.3,0.5")
    p.add_argument("--eval_ious", default="0.3,0.5,0.7")
    p.add_argument("--out", default=None)
    return p.parse_args()


def find_label_dict(obj: Any) -> Optional[Mapping]:
    if isinstance(obj, Mapping):
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


def stats_np(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p10": float(np.quantile(a, 0.10)),
        "median": float(np.quantile(a, 0.50)),
        "p90": float(np.quantile(a, 0.90)),
        "p99": float(np.quantile(a, 0.99)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def init_result_stat(ious):
    return {float(i): {"tp": [], "fp": [], "gt": 0, "score": []} for i in ious}


def parse_postprocess_result(result):
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise RuntimeError(f"Unexpected post_process result: {type(result)}")
    pred, score = result[0], result[1]
    gt = None
    for item in result[2:]:
        if torch.is_tensor(item) and item.ndim == 3 and tuple(item.shape[-2:]) == (8, 3):
            gt = item
            break
        if isinstance(item, np.ndarray) and item.ndim == 3 and tuple(item.shape[-2:]) == (8, 3):
            gt = item
            break
    if gt is None:
        raise RuntimeError(
            "Could not locate GT corners [N,8,3]. Returned: " +
            str([(type(x).__name__, tuple(x.shape) if hasattr(x, "shape") else None) for x in result])
        )
    return pred, score, gt


def official_postprocess(dataset, batch_data, model_output):
    return parse_postprocess_result(
        dataset.post_process(batch_data, OrderedDict(ego=model_output))
    )


def accumulate_ap(stat, pred, score, gt, ious):
    n_gt = 0 if gt is None else int(len(gt))
    for iou in ious:
        iou = float(iou)
        if pred is None or score is None or len(pred) == 0:
            stat[iou]["gt"] += n_gt
        else:
            eval_utils.caluclate_tp_fp(pred, score, gt, stat, iou)


def safe_ap(stat, iou):
    st = copy.deepcopy(stat)
    iou = float(iou)
    if st[iou]["gt"] <= 0:
        return float("nan")
    if len(st[iou]["tp"]) == 0:
        return 0.0
    try:
        ap, _, _ = eval_utils.calculate_ap(st, iou, global_sort_detections=False)
        return float(ap)
    except Exception:
        return float("nan")


def greedy_tp_fp(pred, score, gt, iou_thresh):
    n_gt = 0 if gt is None else int(gt.shape[0])
    if pred is None or score is None or int(pred.shape[0]) == 0:
        return 0, 0, n_gt
    det_np = common_utils.torch_tensor_to_numpy(pred)
    score_np = common_utils.torch_tensor_to_numpy(score)
    gt_np = common_utils.torch_tensor_to_numpy(gt)
    det_polys = list(common_utils.convert_format(det_np))
    gt_polys = list(common_utils.convert_format(gt_np))
    available = list(range(len(gt_polys)))
    tp = fp = 0
    for det_idx in np.argsort(-score_np):
        if not available:
            fp += 1
            continue
        avail_polys = [gt_polys[j] for j in available]
        ious = common_utils.compute_iou(det_polys[int(det_idx)], avail_polys)
        if len(ious) == 0 or float(np.max(ious)) < float(iou_thresh):
            fp += 1
        else:
            available.pop(int(np.argmax(ious)))
            tp += 1
    return tp, fp, n_gt


def gt_coverage(boxes, gt, iou_thresh):
    if gt is None or int(gt.shape[0]) == 0:
        return 0, 0, 0.0
    n_gt = int(gt.shape[0])
    if boxes is None or int(boxes.shape[0]) == 0:
        return 0, n_gt, 0.0
    box_polys = list(common_utils.convert_format(common_utils.torch_tensor_to_numpy(boxes)))
    gt_polys = list(common_utils.convert_format(common_utils.torch_tensor_to_numpy(gt)))
    hit = 0
    for gp in gt_polys:
        ious = common_utils.compute_iou(gp, box_polys)
        if len(ious) and float(np.max(ious)) >= float(iou_thresh):
            hit += 1
    return hit, n_gt, float(hit / max(n_gt, 1))


def decode_stages(batch_data, model_output, pp, threshold, pos_flat):
    """Build stage-wise box sets without changing the model output."""
    cav = batch_data["ego"]
    device = model_output["obj"].device
    tfm = cav["transformation_matrix"]
    anchor_box = cav["anchor_box"]
    if torch.is_tensor(tfm):
        tfm = tfm.to(device)
    if torch.is_tensor(anchor_box):
        anchor_box = anchor_box.to(device)

    obj = torch.sigmoid(
        model_output["obj"].permute(0, 2, 3, 1).contiguous()
    ).reshape(-1)
    decoded_all = pp.delta_to_boxes3d(model_output["rm"], anchor_box)[0]
    if decoded_all.shape[0] != obj.numel():
        raise RuntimeError(f"decoded={decoded_all.shape[0]} obj={obj.numel()}")
    if pos_flat.numel() != obj.numel():
        raise RuntimeError(f"pos={pos_flat.numel()} obj={obj.numel()}")

    corners_all = box_utils.boxes_to_corners_3d(decoded_all, order=pp.params["order"])
    projected_all = box_utils.project_box3d(corners_all, tfm)

    pos_mask = pos_flat > 0.5
    positive_decoded_raw = projected_all[pos_mask]
    positive_after_obj_raw = projected_all[pos_mask & (obj > threshold)]

    cand_mask = obj > float(threshold)
    projected = projected_all[cand_mask]
    scores = obj[cand_mask]
    candidate_count = int(cand_mask.sum().item())

    if projected.shape[0] == 0:
        empty = projected
        return {
            "objectness": obj,
            "positive_decoded_raw": positive_decoded_raw,
            "positive_after_obj_raw": positive_after_obj_raw,
            "candidate_prefilter": empty,
            "post_nms": empty,
            "candidate_count": candidate_count,
            "prefilter_count": 0,
            "post_nms_count": 0,
        }

    keep1 = box_utils.remove_large_pred_bbx(projected, pp.dataset)
    z_min, z_max = pp.lidar_range[2], pp.lidar_range[5]
    try:
        keep2 = box_utils.remove_bbx_abnormal_z(projected, z_min=z_min, z_max=z_max)
    except TypeError:
        keep2 = box_utils.remove_bbx_abnormal_z(projected)
    keep = torch.logical_and(keep1, keep2)
    projected = projected[keep]
    scores = scores[keep]
    candidate_prefilter = projected

    if projected.shape[0] > 0:
        keep_nms = box_utils.nms_rotated(projected, scores, pp.params["nms_thresh"])
        projected = projected[keep_nms]
        scores = scores[keep_nms]
    post_nms = projected

    return {
        "objectness": obj,
        "positive_decoded_raw": positive_decoded_raw,
        "positive_after_obj_raw": positive_after_obj_raw,
        "candidate_prefilter": candidate_prefilter,
        "post_nms": post_nms,
        "candidate_count": candidate_count,
        "prefilter_count": int(candidate_prefilter.shape[0]),
        "post_nms_count": int(post_nms.shape[0]),
    }


def aggregate_waterfall(rows, thresholds, coverage_ious):
    out = {}
    stages = [
        "positive_decoded_raw",
        "positive_after_obj_raw",
        "candidate_prefilter",
        "post_nms",
        "final_official",
    ]
    for t in thresholds:
        tk = f"{t:.4f}"
        rt = [r for r in rows if abs(r["threshold"] - t) < 1e-12]
        out[tk] = {}
        for iou in coverage_ious:
            ik = f"{iou:.2f}"
            out[tk][ik] = {}
            for s in stages:
                hit = sum(r[f"{s}_hit_iou{ik}"] for r in rt)
                gt = sum(r[f"{s}_gt_iou{ik}"] for r in rt)
                out[tk][ik][s] = {
                    "hit": int(hit),
                    "gt": int(gt),
                    "coverage": float(hit / max(gt, 1)),
                }
    return out


def plot_tradeoff(metrics, path):
    rows = sorted(metrics, key=lambda r: r["threshold"])
    x = [r["threshold"] for r in rows]
    recall = [100*r["recall_iou0.30"] for r in rows]
    precision = [100*r["precision_iou0.30"] for r in rows]
    fp = [r["fp_per_frame_iou0.30"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    ax1.plot(x, recall, marker="o", label="Recall @0.3 (%)")
    ax1.plot(x, precision, marker="s", label="Precision @0.3 (%)")
    ax1.set_xlabel("Objectness threshold")
    ax1.set_ylabel("Precision / Recall (%)")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, fp, marker="^", linestyle="--", label="FP / frame")
    ax2.set_ylabel("FP / frame")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="best")
    ax1.set_title("Final post-NMS threshold trade-off")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def plot_waterfall(wf, thresholds, iou, path):
    ik = f"{iou:.2f}"
    stages = ["positive_decoded_raw", "positive_after_obj_raw", "candidate_prefilter", "post_nms", "final_official"]
    labels = ["Pos decoded\n(raw)", "Pos + obj gate", "All candidates\npre-NMS", "Post NMS", "Final official"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for t in thresholds:
        s = wf[f"{t:.4f}"][ik]
        y = [100*s[k]["coverage"] for k in stages]
        ax.plot(labels, y, marker="o", label=f"T={t:g}")
    ax.set_ylabel(f"GT coverage @ IoU {iou:.2f} (%)")
    ax.set_title("Recall waterfall: where GT coverage is lost")
    ax.grid(True, alpha=0.25); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def main():
    args = parse_args()
    thresholds = sorted({float(x) for x in args.thresholds.split(",") if x.strip()})
    coverage_ious = sorted({float(x) for x in args.coverage_ious.split(",") if x.strip()})
    eval_ious = sorted({float(x) for x in args.eval_ious.split(",") if x.strip()})

    out_dir = args.out or os.path.join(args.model_dir, f"obj_recall_waterfall_epoch{args.epoch}")
    os.makedirs(out_dir, exist_ok=True)

    hypes = yaml_utils.load_yaml(None, args)
    if "test_dir" in hypes:
        hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset, batch_size=1, num_workers=args.workers,
        collate_fn=dataset.collate_batch_test, shuffle=False,
        pin_memory=False, drop_last=False,
    )

    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _, model = train_utils.load_model(args.model_dir, model, args.epoch, start_from_best=False)
    model.eval(); model.zero_grad(set_to_none=True)

    pp = dataset.post_processor
    original_threshold = float(pp.params["target_args"]["obj_threshold"])
    result_stats = {t: init_result_stat(eval_ious) for t in thresholds}
    rows = []
    pos_obj_chunks, neg_obj_chunks, ign_obj_chunks = [], [], []
    end = args.start + args.num_frames
    processed = 0

    try:
        for frame_idx, batch_data in tqdm(enumerate(loader), total=min(len(loader), end)):
            if frame_idx < args.start:
                continue
            if frame_idx >= end:
                break
            batch_data = train_utils.to_device(batch_data, device)
            with torch.no_grad():
                model_output = model(batch_data["ego"])

            label_dict = find_label_dict(batch_data["ego"])
            if label_dict is None:
                raise RuntimeError("Cannot find pos_equal_one / neg_equal_one")
            pos_flat = label_dict["pos_equal_one"].reshape(-1).detach()
            neg_flat = label_dict["neg_equal_one"].reshape(-1).detach()
            obj_all = torch.sigmoid(model_output["obj"].permute(0,2,3,1).contiguous()).reshape(-1).detach()
            if pos_flat.numel() != obj_all.numel():
                raise RuntimeError(f"label/objectness mismatch {pos_flat.numel()} vs {obj_all.numel()}")

            pos_sel = pos_flat > 0.5
            neg_sel = neg_flat > 0.5
            ign_sel = (~pos_sel) & (~neg_sel)
            pos_obj_chunks.append(obj_all[pos_sel].cpu().numpy())
            neg_obj_chunks.append(obj_all[neg_sel].cpu().numpy())
            ign_obj_chunks.append(obj_all[ign_sel].cpu().numpy())

            for threshold in thresholds:
                trace = decode_stages(batch_data, model_output, pp, threshold, pos_flat)
                pp.params["target_args"]["obj_threshold"] = float(threshold)
                with torch.no_grad():
                    pred, score, gt = official_postprocess(dataset, batch_data, model_output)
                accumulate_ap(result_stats[threshold], pred, score, gt, eval_ious)

                pos_pass = int((pos_sel & (obj_all > threshold)).sum().item())
                row = {
                    "frame": int(frame_idx), "threshold": float(threshold),
                    "gt_count": int(gt.shape[0]),
                    "positive_anchor_count": int(pos_sel.sum().item()),
                    "negative_anchor_count": int(neg_sel.sum().item()),
                    "positive_anchor_pass_count": pos_pass,
                    "positive_anchor_pass_rate": float(pos_pass / max(int(pos_sel.sum().item()), 1)),
                    "candidate_obj_pass": int(trace["candidate_count"]),
                    "candidate_prefilter_count": int(trace["prefilter_count"]),
                    "post_nms_count": int(trace["post_nms_count"]),
                    "official_final_count": 0 if pred is None else int(pred.shape[0]),
                }

                for iou in eval_ious:
                    tp, fp, ngt = greedy_tp_fp(pred, score, gt, iou)
                    ik = f"{iou:.2f}"
                    row[f"tp_iou{ik}"] = int(tp); row[f"fp_iou{ik}"] = int(fp); row[f"gt_iou{ik}"] = int(ngt)

                for iou in coverage_ious:
                    ik = f"{iou:.2f}"
                    stage_boxes = {
                        "positive_decoded_raw": trace["positive_decoded_raw"],
                        "positive_after_obj_raw": trace["positive_after_obj_raw"],
                        "candidate_prefilter": trace["candidate_prefilter"],
                        "post_nms": trace["post_nms"],
                        "final_official": pred,
                    }
                    for stage, boxes in stage_boxes.items():
                        hit, ngt, cov = gt_coverage(boxes, gt, iou)
                        row[f"{stage}_hit_iou{ik}"] = int(hit)
                        row[f"{stage}_gt_iou{ik}"] = int(ngt)
                        row[f"{stage}_coverage_iou{ik}"] = float(cov)
                rows.append(row)

            pp.params["target_args"]["obj_threshold"] = original_threshold
            processed += 1
    finally:
        pp.params["target_args"]["obj_threshold"] = original_threshold

    metrics = []
    for t in thresholds:
        rt = [r for r in rows if abs(r["threshold"] - t) < 1e-12]
        nfr = max(len(rt), 1)
        m = {
            "threshold": float(t), "frames": len(rt),
            "gt_per_frame": float(np.mean([r["gt_count"] for r in rt])),
            "positive_anchor_pass_rate": float(np.mean([r["positive_anchor_pass_rate"] for r in rt])),
            "candidate_obj_pass_per_frame": float(np.mean([r["candidate_obj_pass"] for r in rt])),
            "candidate_prefilter_per_frame": float(np.mean([r["candidate_prefilter_count"] for r in rt])),
            "post_nms_per_frame": float(np.mean([r["post_nms_count"] for r in rt])),
            "pred_per_frame": float(np.mean([r["official_final_count"] for r in rt])),
        }
        for iou in eval_ious:
            ik = f"{iou:.2f}"
            tp = sum(r[f"tp_iou{ik}"] for r in rt)
            fp = sum(r[f"fp_iou{ik}"] for r in rt)
            gt = sum(r[f"gt_iou{ik}"] for r in rt)
            m[f"tp_per_frame_iou{ik}"] = float(tp/nfr)
            m[f"fp_per_frame_iou{ik}"] = float(fp/nfr)
            m[f"precision_iou{ik}"] = float(tp/max(tp+fp, 1))
            m[f"recall_iou{ik}"] = float(tp/max(gt, 1))
            m[f"ap_iou{ik}"] = safe_ap(result_stats[t], iou)
        metrics.append(m)

    wf = aggregate_waterfall(rows, thresholds, coverage_ious)
    cat = lambda chunks: np.concatenate([x for x in chunks if x.size]) if any(x.size for x in chunks) else np.empty(0)
    obj_dist = {
        "positive": stats_np(cat(pos_obj_chunks)),
        "explicit_negative": stats_np(cat(neg_obj_chunks)),
        "ignore": stats_np(cat(ign_obj_chunks)),
    }

    summary = {
        "model_dir": args.model_dir,
        "epoch": args.epoch,
        "frames_processed": processed,
        "original_production_obj_threshold": original_threshold,
        "thresholds": thresholds,
        "coverage_ious": coverage_ious,
        "eval_ious": eval_ious,
        "obj_distributions": obj_dist,
        "threshold_metrics": metrics,
        "recall_waterfall": wf,
        "definitions": {
            "positive_decoded_raw": "label-positive anchors decoded/projected before obj gate; regression upper bound",
            "positive_after_obj_raw": "same positive decoded boxes after obj threshold",
            "candidate_prefilter": "all obj-passing candidates after production large-box + abnormal-z filters, before NMS",
            "post_nms": "same candidates after production rotated NMS, before final range filter",
            "final_official": "exact dataset.post_process final boxes",
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=True)
    with open(os.path.join(out_dir, "obj_distributions.json"), "w", encoding="utf-8") as f:
        json.dump(obj_dist, f, indent=2, ensure_ascii=False, allow_nan=True)
    if metrics:
        with open(os.path.join(out_dir, "threshold_metrics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics[0].keys())); w.writeheader(); w.writerows(metrics)
    if rows:
        with open(os.path.join(out_dir, "per_frame.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    plot_tradeoff(metrics, os.path.join(out_dir, "threshold_tradeoff.png"))
    for iou in coverage_ious:
        plot_waterfall(wf, thresholds, iou, os.path.join(out_dir, f"recall_waterfall_iou{str(iou).replace('.', 'p')}.png"))

    print("\n================ OBJECTNESS DISTRIBUTION ================")
    print(json.dumps(obj_dist, indent=2, ensure_ascii=False))
    print("\n================ FINAL THRESHOLD SWEEP ================")
    print(f"{'T':>6} {'PosPass':>8} {'Cand/fr':>9} {'Pred/fr':>9} {'TP/fr@.3':>9} {'FP/fr@.3':>9} {'P@.3':>8} {'R@.3':>8} {'AP30':>8} {'AP50':>8}")
    for m in metrics:
        print(f"{m['threshold']:6.2f} {m['positive_anchor_pass_rate']:8.3f} {m['candidate_obj_pass_per_frame']:9.1f} {m['pred_per_frame']:9.2f} {m.get('tp_per_frame_iou0.30', float('nan')):9.2f} {m.get('fp_per_frame_iou0.30', float('nan')):9.2f} {m.get('precision_iou0.30', float('nan')):8.3f} {m.get('recall_iou0.30', float('nan')):8.3f} {m.get('ap_iou0.30', float('nan')):8.3f} {m.get('ap_iou0.50', float('nan')):8.3f}")

    for iou in coverage_ious:
        ik = f"{iou:.2f}"
        print(f"\n================ RECALL WATERFALL IoU={iou:.2f} ================")
        print(f"{'T':>6} {'PosDecoded':>11} {'Pos+Obj':>10} {'PreNMS':>10} {'PostNMS':>10} {'Final':>10}")
        for t in thresholds:
            s = wf[f"{t:.4f}"][ik]
            print(f"{t:6.2f} {s['positive_decoded_raw']['coverage']:11.3f} {s['positive_after_obj_raw']['coverage']:10.3f} {s['candidate_prefilter']['coverage']:10.3f} {s['post_nms']['coverage']:10.3f} {s['final_official']['coverage']:10.3f}")

    print("\nInterpretation:")
    print("- PosDecoded low: regression/anchor localization is the bottleneck; stop tuning obj.")
    print("- PosDecoded high but Pos+Obj drops: objectness threshold/loss suppresses true positives.")
    print("- Pos+Obj high but PreNMS drops: production geometry filters remove useful boxes.")
    print("- PreNMS high but PostNMS drops: NMS suppresses GT-covering candidates.")
    print("- Lower threshold restores recall with acceptable FP/AP: retune threshold before retraining.")
    print(f"\nsaved under: {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
READ-ONLY diagnostic for AirV2X objectness target / regression alignment.

Question answered
-----------------
For each GT:
  1) find the ALL-DECODED box with highest exact BEV IoU;
  2) map that box back to the SAME anchor index;
  3) inspect that anchor's TRAINING objectness target:
       positive / negative / ignore
  4) reconstruct the repository's anchor assignment independently and
     verify it matches batch["ego"]["label_dict"];
  5) report whether the oracle-best anchor was assigned to THIS SAME GT;
  6) compare regression quality available from:
       - all anchors
       - train-positive anchors
       - train-positive anchors assigned to this GT
       - train-negative anchors
       - ignored anchors

This distinguishes:

A. Objectness optimization problem:
   good decoded boxes mainly come from true positive anchors, but their
   objectness remains too low.

B. Target/regression mismatch:
   good decoded boxes mainly come from negative/ignore anchors, while the
   anchors supervised positive for that GT regress substantially worse.

Verified repository interfaces
------------------------------
- batch["ego"]["label_dict"]:
    pos_equal_one, neg_equal_one, targets, class_ids
- flatten order:
    target: [B,H,W,A] -> flatten
    obj:    [B,A,H,W] -> permute(B,H,W,A) -> flatten
    anchors:[H,W,A,7] -> reshape(-1,7)
- target assignment in VoxelPostprocessor.generate_label_airv2x:
    stand-up 2D anchor/GT IoU
    positive: IoU > pos_threshold
    negative: anchor has IoU < neg_threshold to ALL GT
    plus each GT's highest-IoU anchor is forced positive
- decoding:
    VoxelPostprocessor.delta_to_boxes3d(rm, anchor_box)

No optimizer, no checkpoint writes, no model/threshold mutation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.tools import train_utils
from opencood.utils import box_utils, common_utils
from opencood.utils.box_overlaps import bbox_overlaps


DEFAULT_CKPT = (
    "/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
    "airv2x_gaussian_0822_geom_sup/geom_sup_2026_09_17_00_34_27/"
    "net_epoch25.pth"
)

DEFAULT_IOUS = (0.3, 0.5, 0.7)


# ---------------------------------------------------------------------
# CLI / setup
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--config", default="")
    p.add_argument("--num-frames", type=int, default=100)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--worker", type=int, default=2)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--eval-ious", default="0.3,0.5,0.7")
    p.add_argument(
        "--out-dir",
        default="opencood/logs/diagnose_obj_target_alignment_ep25",
    )
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_config(ckpt: Path, explicit: str) -> Path:
    p = Path(explicit) if explicit else ckpt.parent / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"config not found: {p}; pass --config explicitly"
        )
    return p


def unwrap_state_dict(blob):
    if not isinstance(blob, dict):
        raise TypeError(f"checkpoint must be dict-like, got {type(blob)}")
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        if isinstance(blob.get(key), dict):
            return blob[key]
    return blob


def load_checkpoint_exact(model, ckpt: Path, device):
    blob = torch.load(str(ckpt), map_location=device)
    state = dict(unwrap_state_dict(blob))
    if state and next(iter(state)).startswith("module."):
        state = {k[7:]: v for k, v in state.items()}

    current = model.state_dict()
    shape_bad = [
        k for k, v in state.items()
        if k in current and tuple(v.shape) != tuple(current[k].shape)
    ]
    if shape_bad:
        raise RuntimeError(f"shape mismatch examples: {shape_bad[:20]}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch\n"
            f"missing({len(missing)}): {missing[:30]}\n"
            f"unexpected({len(unexpected)}): {unexpected[:30]}"
        )
    print(f"[ckpt] exact-compatible: {ckpt}")


def make_loader(hypes, worker):
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=worker,
        collate_fn=dataset.collate_batch_test,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


# ---------------------------------------------------------------------
# Generic stats
# ---------------------------------------------------------------------

def finite_np(values):
    x = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return x[np.isfinite(x)]


def stats(values) -> Dict[str, float]:
    x = finite_np(values)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "p50": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def ratio(num, den):
    return float(num / den) if den else float("nan")


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------
# Target assignment: exact repository logic reconstructed independently
# ---------------------------------------------------------------------

def reconstruct_assignment(
    anchors_np: np.ndarray,
    gt_boxes_np: np.ndarray,
    order: str,
    pos_threshold: float,
    neg_threshold: float,
) -> Dict[str, np.ndarray]:
    """
    Reproduce VoxelPostprocessor.generate_label_airv2x assignment.

    Returns flattened arrays length N_anchor:
      status:  1 pos, 0 ignore, -1 neg
      assigned_gt: GT index for positive anchors, -1 otherwise
      anchor_gt_iou: stand-up IoU matrix [N_anchor, N_gt]
    """
    n_anchor = int(anchors_np.shape[0])
    n_gt = int(gt_boxes_np.shape[0])

    status = np.zeros((n_anchor,), dtype=np.int8)
    assigned_gt = np.full((n_anchor,), -1, dtype=np.int64)

    if n_gt == 0:
        status[:] = -1
        return {
            "status": status,
            "assigned_gt": assigned_gt,
            "iou": np.zeros((n_anchor, 0), dtype=np.float32),
        }

    anchors_corner = box_utils.boxes_to_corners_3d(
        anchors_np, order=order
    )
    gt_corner = box_utils.boxes_to_corners_3d(
        gt_boxes_np, order=order
    )

    anchor_standup = box_utils.corner2d_to_standup_box(
        anchors_corner
    )
    gt_standup = box_utils.corner2d_to_standup_box(
        gt_corner
    )

    iou = bbox_overlaps(
        np.ascontiguousarray(anchor_standup).astype(np.float32),
        np.ascontiguousarray(gt_standup).astype(np.float32),
    )

    # Each GT's highest-IoU anchor, if IoU > 0.
    id_highest = np.argmax(iou.T, axis=1)
    id_highest_gt = np.arange(iou.T.shape[0])
    valid_highest = iou.T[id_highest_gt, id_highest] > 0
    id_highest = id_highest[valid_highest]
    id_highest_gt = id_highest_gt[valid_highest]

    # Normal positives.
    id_pos, id_pos_gt = np.where(iou > pos_threshold)

    # Exactly repository order: concatenate then np.unique keeping first.
    id_pos = np.concatenate([id_pos, id_highest])
    id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
    id_pos_unique, first_idx = np.unique(id_pos, return_index=True)
    id_pos_gt_unique = id_pos_gt[first_idx]

    # Negative iff below neg threshold against EVERY GT.
    id_neg = np.where(
        np.sum(iou < neg_threshold, axis=1) == iou.shape[1]
    )[0]

    status[id_neg] = -1
    status[id_pos_unique] = 1
    assigned_gt[id_pos_unique] = id_pos_gt_unique

    # Repository explicitly removes highest anchors from neg.
    status[id_highest] = 1

    return {
        "status": status,
        "assigned_gt": assigned_gt,
        "iou": iou,
    }


def status_name(v: int) -> str:
    if int(v) == 1:
        return "positive"
    if int(v) == -1:
        return "negative"
    return "ignore"


def validate_batch_labels(
    label_dict: Dict[str, torch.Tensor],
    reconstructed_status: np.ndarray,
    n_anchor: int,
):
    required = ("pos_equal_one", "neg_equal_one", "class_ids")
    missing = [k for k in required if k not in label_dict]
    if missing:
        raise RuntimeError(
            f"label_dict missing keys {missing}; got {list(label_dict.keys())}"
        )

    pos = label_dict["pos_equal_one"][0].reshape(-1).detach().cpu().numpy()
    neg = label_dict["neg_equal_one"][0].reshape(-1).detach().cpu().numpy()

    if pos.size != n_anchor or neg.size != n_anchor:
        raise RuntimeError(
            f"label flatten mismatch: pos={pos.size}, neg={neg.size}, anchors={n_anchor}"
        )

    if np.any((pos > 0) & (neg > 0)):
        bad = int(np.sum((pos > 0) & (neg > 0)))
        raise RuntimeError(f"{bad} anchors are simultaneously pos and neg")

    batch_status = np.zeros((n_anchor,), dtype=np.int8)
    batch_status[neg > 0] = -1
    batch_status[pos > 0] = 1

    mismatch = batch_status != reconstructed_status
    mismatch_count = int(mismatch.sum())
    if mismatch_count:
        idx = np.nonzero(mismatch)[0][:20]
        examples = [
            {
                "idx": int(i),
                "batch": status_name(batch_status[i]),
                "reconstructed": status_name(reconstructed_status[i]),
            }
            for i in idx
        ]
        raise RuntimeError(
            "Reconstructed target assignment does not match batch labels: "
            f"{mismatch_count}/{n_anchor} mismatches. Examples={examples}"
        )

    return {
        "status": batch_status,
        "pos": pos,
        "neg": neg,
        "class_ids": label_dict["class_ids"][0].reshape(-1).detach().cpu().numpy(),
    }


# ---------------------------------------------------------------------
# Exact decoded-box IoU with lossless AABB prefilter
# ---------------------------------------------------------------------

def xy_aabb(corners_np: np.ndarray):
    xy = corners_np[..., :2]
    return (
        xy[..., 0].min(axis=1),
        xy[..., 1].min(axis=1),
        xy[..., 0].max(axis=1),
        xy[..., 1].max(axis=1),
    )


def exact_iou_candidates_for_gt(
    pred_corners_np: np.ndarray,
    pred_aabb,
    gt_corner_np: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return global prediction indices and exact BEV IoUs.

    AABB overlap is a lossless prefilter: polygons with positive intersection
    must have overlapping AABBs.
    """
    p_xmin, p_ymin, p_xmax, p_ymax = pred_aabb

    gxmin = float(gt_corner_np[:, 0].min())
    gymin = float(gt_corner_np[:, 1].min())
    gxmax = float(gt_corner_np[:, 0].max())
    gymax = float(gt_corner_np[:, 1].max())

    mask = (
        (p_xmax >= gxmin)
        & (p_xmin <= gxmax)
        & (p_ymax >= gymin)
        & (p_ymin <= gymax)
    )
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return idx, np.zeros((0,), dtype=np.float64)

    gt_poly = common_utils.convert_format(
        gt_corner_np[None, ...]
    )[0]
    pred_polys = list(
        common_utils.convert_format(pred_corners_np[idx])
    )
    ious = np.asarray(
        common_utils.compute_iou(gt_poly, pred_polys),
        dtype=np.float64,
    )
    return idx, ious


def best_for_mask(
    cand_idx: np.ndarray,
    cand_iou: np.ndarray,
    anchor_mask: np.ndarray,
) -> Tuple[int, float]:
    if cand_idx.size == 0:
        return -1, 0.0
    keep = anchor_mask[cand_idx]
    if not np.any(keep):
        return -1, 0.0
    sub_idx = cand_idx[keep]
    sub_iou = cand_iou[keep]
    j = int(np.argmax(sub_iou))
    return int(sub_idx[j]), float(sub_iou[j])


# ---------------------------------------------------------------------
# One-frame audit
# ---------------------------------------------------------------------

def audit_frame(
    batch: Dict[str, Any],
    output: Dict[str, torch.Tensor],
    hypes: Dict[str, Any],
    frame_idx: int,
    iou_thresholds: Sequence[float],
) -> Dict[str, Any]:
    ego = batch["ego"]

    pp = VoxelPostprocessor(
        hypes["postprocess"],
        dataset="airv2x",
        train=False,
    )
    params = hypes["postprocess"]
    order = params["order"]
    pos_thr = float(params["target_args"]["pos_threshold"])
    neg_thr = float(params["target_args"]["neg_threshold"])
    obj_thr = float(params["target_args"]["obj_threshold"])

    anchors = ego["anchor_box"].to(output["rm"].device)
    anchors_np = anchors.detach().cpu().numpy().reshape(-1, 7)
    n_anchor = int(anchors_np.shape[0])

    # Current GT boxes in the exact order used to generate labels.
    valid = ego["object_bbx_mask"][0] == 1
    gt_boxes = ego["object_bbx_center"][0][valid]
    gt_boxes_np = gt_boxes.detach().cpu().numpy()
    gt_corners = box_utils.boxes_to_corners_3d(
        gt_boxes, order=order
    )
    gt_corners_np = gt_corners.detach().cpu().numpy()
    n_gt = int(gt_boxes_np.shape[0])

    # Reconstruct assignment independently.
    recon = reconstruct_assignment(
        anchors_np=anchors_np,
        gt_boxes_np=gt_boxes_np,
        order=order,
        pos_threshold=pos_thr,
        neg_threshold=neg_thr,
    )
    labels = validate_batch_labels(
        ego["label_dict"],
        recon["status"],
        n_anchor,
    )
    status = labels["status"]
    assigned_gt = recon["assigned_gt"]
    anchor_gt_iou = recon["iou"]

    # Outputs in exact anchor-flatten order [H,W,A].
    obj = torch.sigmoid(
        output["obj"].permute(0, 2, 3, 1).contiguous()
    ).view(-1)

    psm = output["psm"]
    B, AC, H, W = psm.shape
    C = int(hypes["num_class"])
    A = AC // C
    class_prob = torch.sigmoid(
        psm.view(B, C, A, H, W)
        .permute(0, 3, 4, 2, 1)
        .contiguous()
    ).view(-1, C)
    # Production ignores background column 0.
    fg_class_scores, fg_class_labels = torch.max(
        class_prob[:, 1:], dim=-1
    )
    fg_class_labels = fg_class_labels + 1

    boxes_all = pp.delta_to_boxes3d(
        output["rm"], anchors
    )[0].reshape(-1, 7)

    if int(boxes_all.shape[0]) != n_anchor:
        raise RuntimeError(
            f"decoded={boxes_all.shape[0]} anchors={n_anchor}"
        )
    if int(obj.numel()) != n_anchor:
        raise RuntimeError(
            f"obj={obj.numel()} anchors={n_anchor}"
        )
    if int(fg_class_scores.numel()) != n_anchor:
        raise RuntimeError(
            f"class_scores={fg_class_scores.numel()} anchors={n_anchor}"
        )

    # ego transformation is identity for intermediate-fusion test batches,
    # but use the actual stored transform to exactly match postprocess.
    tfm = ego["transformation_matrix"].to(boxes_all.device)
    decoded_local_corners = box_utils.boxes_to_corners_3d(
        boxes_all, order=order
    )
    decoded_corners = box_utils.project_box3d(
        decoded_local_corners, tfm
    )

    decoded_np = decoded_corners.detach().cpu().numpy()
    pred_aabb = xy_aabb(decoded_np)
    obj_np = obj.detach().cpu().numpy()
    cls_np = fg_class_scores.detach().cpu().numpy()
    pred_cls_np = fg_class_labels.detach().cpu().numpy()
    target_cls_np = labels["class_ids"]

    pos_mask = status == 1
    neg_mask = status == -1
    ign_mask = status == 0

    rows = []

    for gi in range(n_gt):
        cand_idx, cand_iou = exact_iou_candidates_for_gt(
            decoded_np,
            pred_aabb,
            gt_corners_np[gi],
        )

        # Best across ALL anchors.
        if cand_idx.size:
            j = int(np.argmax(cand_iou))
            best_idx = int(cand_idx[j])
            best_iou = float(cand_iou[j])
        else:
            best_idx = -1
            best_iou = 0.0

        # Best by training target status.
        best_pos_idx, best_pos_iou = best_for_mask(
            cand_idx, cand_iou, pos_mask
        )
        best_neg_idx, best_neg_iou = best_for_mask(
            cand_idx, cand_iou, neg_mask
        )
        best_ign_idx, best_ign_iou = best_for_mask(
            cand_idx, cand_iou, ign_mask
        )

        same_gt_pos_mask = (
            pos_mask & (assigned_gt == gi)
        )
        best_same_idx, best_same_iou = best_for_mask(
            cand_idx, cand_iou, same_gt_pos_mask
        )

        # Among boxes that are geometrically valid at each IoU threshold,
        # quantify which supervision status supplies them.
        threshold_fields = {}
        for thr in iou_thresholds:
            good = cand_iou >= float(thr)
            good_idx = cand_idx[good]

            if good_idx.size:
                g_status = status[good_idx]
                threshold_fields[f"n_decoded_iou_ge_{thr}"] = int(good_idx.size)
                threshold_fields[f"n_positive_iou_ge_{thr}"] = int(
                    np.sum(g_status == 1)
                )
                threshold_fields[f"n_negative_iou_ge_{thr}"] = int(
                    np.sum(g_status == -1)
                )
                threshold_fields[f"n_ignore_iou_ge_{thr}"] = int(
                    np.sum(g_status == 0)
                )
                threshold_fields[f"max_obj_iou_ge_{thr}"] = float(
                    obj_np[good_idx].max()
                )

                pos_good = good_idx[g_status == 1]
                same_good = good_idx[
                    (g_status == 1)
                    & (assigned_gt[good_idx] == gi)
                ]
                threshold_fields[
                    f"max_obj_positive_iou_ge_{thr}"
                ] = (
                    float(obj_np[pos_good].max())
                    if pos_good.size else float("nan")
                )
                threshold_fields[
                    f"max_obj_same_gt_positive_iou_ge_{thr}"
                ] = (
                    float(obj_np[same_good].max())
                    if same_good.size else float("nan")
                )
            else:
                threshold_fields[f"n_decoded_iou_ge_{thr}"] = 0
                threshold_fields[f"n_positive_iou_ge_{thr}"] = 0
                threshold_fields[f"n_negative_iou_ge_{thr}"] = 0
                threshold_fields[f"n_ignore_iou_ge_{thr}"] = 0
                threshold_fields[f"max_obj_iou_ge_{thr}"] = float("nan")
                threshold_fields[
                    f"max_obj_positive_iou_ge_{thr}"
                ] = float("nan")
                threshold_fields[
                    f"max_obj_same_gt_positive_iou_ge_{thr}"
                ] = float("nan")

        row = {
            "frame": frame_idx,
            "gt_index": gi,
            "oracle_best_anchor_index": best_idx,
            "oracle_best_decoded_iou": best_iou,
            "oracle_best_status": (
                status_name(status[best_idx])
                if best_idx >= 0 else "none"
            ),
            "oracle_best_objectness": (
                float(obj_np[best_idx])
                if best_idx >= 0 else float("nan")
            ),
            "oracle_best_class_score": (
                float(cls_np[best_idx])
                if best_idx >= 0 else float("nan")
            ),
            "oracle_best_pred_class": (
                int(pred_cls_np[best_idx])
                if best_idx >= 0 else -1
            ),
            "oracle_best_target_class": (
                int(target_cls_np[best_idx])
                if best_idx >= 0 else -1
            ),
            "oracle_best_anchor_iou_to_this_gt": (
                float(anchor_gt_iou[best_idx, gi])
                if best_idx >= 0 else float("nan")
            ),
            "oracle_best_anchor_max_iou_any_gt": (
                float(anchor_gt_iou[best_idx].max())
                if best_idx >= 0 and n_gt else float("nan")
            ),
            "oracle_best_assigned_gt_index": (
                int(assigned_gt[best_idx])
                if best_idx >= 0 else -1
            ),
            "oracle_best_assigned_to_same_gt": (
                bool(
                    best_idx >= 0
                    and status[best_idx] == 1
                    and assigned_gt[best_idx] == gi
                )
            ),
            "oracle_best_passes_obj_threshold": (
                bool(best_idx >= 0 and obj_np[best_idx] > obj_thr)
            ),
            "best_positive_anchor_index": best_pos_idx,
            "best_positive_decoded_iou": best_pos_iou,
            "best_positive_objectness": (
                float(obj_np[best_pos_idx])
                if best_pos_idx >= 0 else float("nan")
            ),
            "best_same_gt_positive_anchor_index": best_same_idx,
            "best_same_gt_positive_decoded_iou": best_same_iou,
            "best_same_gt_positive_objectness": (
                float(obj_np[best_same_idx])
                if best_same_idx >= 0 else float("nan")
            ),
            "best_negative_decoded_iou": best_neg_iou,
            "best_ignore_decoded_iou": best_ign_iou,
        }
        row.update(threshold_fields)
        rows.append(row)

    return {
        "rows": rows,
        "counts": {
            "n_gt": n_gt,
            "n_anchor": n_anchor,
            "n_pos": int(pos_mask.sum()),
            "n_neg": int(neg_mask.sum()),
            "n_ignore": int(ign_mask.sum()),
        },
        "objectness": {
            "pos": stats(obj_np[pos_mask]),
            "neg": stats(obj_np[neg_mask]),
            "ignore": stats(obj_np[ign_mask]),
            "threshold": obj_thr,
            "fraction_pos_above_threshold": (
                float(np.mean(obj_np[pos_mask] > obj_thr))
                if np.any(pos_mask) else float("nan")
            ),
            "fraction_neg_above_threshold": (
                float(np.mean(obj_np[neg_mask] > obj_thr))
                if np.any(neg_mask) else float("nan")
            ),
            "fraction_ignore_above_threshold": (
                float(np.mean(obj_np[ign_mask] > obj_thr))
                if np.any(ign_mask) else float("nan")
            ),
        },
    }


# ---------------------------------------------------------------------
# Aggregate interpretation
# ---------------------------------------------------------------------

def summarize_rows(rows: List[Dict[str, Any]], iou_thresholds):
    summary = {
        "n_gt": len(rows),
        "oracle_best_decoded_iou": stats(
            r["oracle_best_decoded_iou"] for r in rows
        ),
        "oracle_best_objectness": stats(
            r["oracle_best_objectness"] for r in rows
        ),
        "best_positive_decoded_iou": stats(
            r["best_positive_decoded_iou"] for r in rows
        ),
        "best_same_gt_positive_decoded_iou": stats(
            r["best_same_gt_positive_decoded_iou"] for r in rows
        ),
        "best_negative_decoded_iou": stats(
            r["best_negative_decoded_iou"] for r in rows
        ),
        "best_ignore_decoded_iou": stats(
            r["best_ignore_decoded_iou"] for r in rows
        ),
    }

    status_counts = Counter(r["oracle_best_status"] for r in rows)
    summary["oracle_best_status_counts_all_gt"] = dict(status_counts)
    summary["oracle_best_status_fraction_all_gt"] = {
        k: ratio(v, len(rows))
        for k, v in status_counts.items()
    }

    same = sum(bool(r["oracle_best_assigned_to_same_gt"]) for r in rows)
    summary["oracle_best_assigned_to_same_gt_fraction_all"] = ratio(
        same, len(rows)
    )

    for thr in iou_thresholds:
        oracle_rows = [
            r for r in rows
            if r["oracle_best_decoded_iou"] >= float(thr)
        ]
        key = f"iou_ge_{thr}"
        counts = Counter(
            r["oracle_best_status"] for r in oracle_rows
        )
        same_n = sum(
            bool(r["oracle_best_assigned_to_same_gt"])
            for r in oracle_rows
        )

        summary[key] = {
            "n_oracle_gt": len(oracle_rows),
            "oracle_recall": ratio(len(oracle_rows), len(rows)),
            "oracle_best_status_counts": dict(counts),
            "oracle_best_status_fraction": {
                k: ratio(v, len(oracle_rows))
                for k, v in counts.items()
            },
            "oracle_best_assigned_to_same_gt_fraction": ratio(
                same_n, len(oracle_rows)
            ),
            "oracle_best_objectness": stats(
                r["oracle_best_objectness"]
                for r in oracle_rows
            ),
            "best_positive_decoded_iou": stats(
                r["best_positive_decoded_iou"]
                for r in oracle_rows
            ),
            "best_same_gt_positive_decoded_iou": stats(
                r["best_same_gt_positive_decoded_iou"]
                for r in oracle_rows
            ),
            "best_same_gt_positive_objectness": stats(
                r["best_same_gt_positive_objectness"]
                for r in oracle_rows
            ),
            "fraction_oracle_gt_with_positive_candidate_at_same_iou": (
                ratio(
                    sum(
                        r["best_positive_decoded_iou"] >= float(thr)
                        for r in oracle_rows
                    ),
                    len(oracle_rows),
                )
            ),
            "fraction_oracle_gt_with_same_gt_positive_candidate_at_same_iou": (
                ratio(
                    sum(
                        r["best_same_gt_positive_decoded_iou"] >= float(thr)
                        for r in oracle_rows
                    ),
                    len(oracle_rows),
                )
            ),
            "fraction_oracle_best_passes_obj_threshold": (
                ratio(
                    sum(
                        bool(r["oracle_best_passes_obj_threshold"])
                        for r in oracle_rows
                    ),
                    len(oracle_rows),
                )
            ),
            "max_obj_positive_iou_ge": stats(
                r[f"max_obj_positive_iou_ge_{thr}"]
                for r in oracle_rows
            ),
            "max_obj_same_gt_positive_iou_ge": stats(
                r[f"max_obj_same_gt_positive_iou_ge_{thr}"]
                for r in oracle_rows
            ),
        }

    return summary


def build_verdict(summary, obj_thr):
    s03 = summary.get("iou_ge_0.3", {})
    if not s03:
        return "NO_IOU03_SUMMARY"

    status_frac = s03.get("oracle_best_status_fraction", {})
    pos_frac = float(status_frac.get("positive", 0.0))
    neg_frac = float(status_frac.get("negative", 0.0))
    ign_frac = float(status_frac.get("ignore", 0.0))
    same_frac = float(
        s03.get(
            "oracle_best_assigned_to_same_gt_fraction",
            0.0,
        )
    )
    positive_candidate_frac = float(
        s03.get(
            "fraction_oracle_gt_with_same_gt_positive_candidate_at_same_iou",
            0.0,
        )
    )

    pos_obj = s03.get("max_obj_same_gt_positive_iou_ge", {})
    pos_obj_median = float(pos_obj.get("p50", float("nan")))

    if neg_frac + ign_frac >= 0.50 and positive_candidate_frac < 0.60:
        return (
            "TARGET_REGRESSION_MISMATCH: at least half of oracle-best IoU>=0.3 "
            "boxes come from negative/ignore anchors and same-GT positive anchors "
            "often cannot match that regression quality. Fix target/quality "
            "alignment before tuning objectness focal weights."
        )

    if positive_candidate_frac >= 0.70 and (
        math.isfinite(pos_obj_median) and pos_obj_median < obj_thr
    ):
        return (
            "OBJECTNESS_OPTIMIZATION: same-GT train-positive anchors usually do "
            "produce IoU>=0.3 boxes, but their objectness remains below the "
            "production threshold. Inspect focal alpha/gamma, prior bias, loss "
            "weighting, and positive confidence learning."
        )

    if pos_frac >= 0.60 or same_frac >= 0.50:
        return (
            "MOSTLY_POSITIVE_BUT_MIXED: oracle-good boxes are often from positive "
            "anchors, so objectness optimization/calibration is likely important, "
            "but assignment mismatch is not negligible."
        )

    return (
        "MIXED_ASSIGNMENT_AND_OBJECTNESS: neither pure target mismatch nor pure "
        "positive-confidence failure dominates. Use the per-GT CSV breakdown."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    opt = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    torch.cuda.set_device(opt.gpu_id)
    device = torch.device(f"cuda:{opt.gpu_id}")
    seed_all(opt.seed)

    ckpt = Path(opt.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    config = resolve_config(ckpt, opt.config)
    hypes = yaml_utils.load_yaml(str(config))
    iou_thresholds = sorted({
        float(x.strip())
        for x in opt.eval_ious.split(",")
        if x.strip()
    })

    print("[config]", config)
    print("[checkpoint]", ckpt)

    dataset, loader = make_loader(hypes, opt.worker)
    model = train_utils.create_model(hypes).to(device)
    load_checkpoint_exact(model, ckpt, device)
    model.eval()

    out_dir = Path(opt.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    frame_summaries = []

    processed = 0
    with torch.no_grad():
        for fi, batch in tqdm(
            enumerate(loader),
            total=min(len(loader), opt.start + opt.num_frames),
            desc="obj-target alignment",
        ):
            if fi < opt.start:
                continue
            if processed >= opt.num_frames:
                break
            if batch is None:
                continue

            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            ego["epoch"] = 0

            with amp.autocast(enabled=bool(opt.amp)):
                output = model(ego)

            required = ("psm", "rm", "obj")
            missing = [k for k in required if k not in output]
            if missing:
                raise RuntimeError(
                    f"model output missing {missing}; got {list(output.keys())}"
                )

            fr = audit_frame(
                batch=batch,
                output=output,
                hypes=hypes,
                frame_idx=fi,
                iou_thresholds=iou_thresholds,
            )
            all_rows.extend(fr["rows"])
            frame_summaries.append({
                "frame": fi,
                **fr["counts"],
                "pos_obj_mean": fr["objectness"]["pos"].get(
                    "mean", float("nan")
                ),
                "pos_obj_p50": fr["objectness"]["pos"].get(
                    "p50", float("nan")
                ),
                "neg_obj_mean": fr["objectness"]["neg"].get(
                    "mean", float("nan")
                ),
                "ignore_obj_mean": fr["objectness"]["ignore"].get(
                    "mean", float("nan")
                ),
                "fraction_pos_above_threshold": fr["objectness"][
                    "fraction_pos_above_threshold"
                ],
                "fraction_neg_above_threshold": fr["objectness"][
                    "fraction_neg_above_threshold"
                ],
            })
            processed += 1

    if not all_rows:
        raise RuntimeError("no GT rows collected")

    obj_thr = float(
        hypes["postprocess"]["target_args"]["obj_threshold"]
    )
    pos_thr = float(
        hypes["postprocess"]["target_args"]["pos_threshold"]
    )
    neg_thr = float(
        hypes["postprocess"]["target_args"]["neg_threshold"]
    )

    summary = summarize_rows(
        all_rows,
        iou_thresholds,
    )
    verdict = build_verdict(summary, obj_thr)

    # Aggregate anchor target counts.
    total_pos = sum(x["n_pos"] for x in frame_summaries)
    total_neg = sum(x["n_neg"] for x in frame_summaries)
    total_ign = sum(x["n_ignore"] for x in frame_summaries)
    total_anchor = sum(x["n_anchor"] for x in frame_summaries)

    report = {
        "checkpoint": str(ckpt),
        "config": str(config),
        "n_frames": processed,
        "n_gt": len(all_rows),
        "target_assignment": {
            "pos_threshold": pos_thr,
            "neg_threshold": neg_thr,
            "forced_highest_anchor_per_gt": True,
            "iou_type": "standup_2d_iou",
            "flatten_order_verified": "H,W,A",
            "batch_label_reconstruction_match_required": True,
        },
        "production_obj_threshold": obj_thr,
        "anchor_target_counts": {
            "total_anchor": total_anchor,
            "positive": total_pos,
            "negative": total_neg,
            "ignore": total_ign,
            "positive_fraction": ratio(total_pos, total_anchor),
            "negative_fraction": ratio(total_neg, total_anchor),
            "ignore_fraction": ratio(total_ign, total_anchor),
        },
        "summary": summary,
        "verdict": verdict,
    }

    json_path = out_dir / "obj_target_alignment_summary.json"
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    gt_csv = out_dir / "gt_obj_target_alignment.csv"
    write_csv(gt_csv, all_rows)

    frame_csv = out_dir / "frame_obj_target_stats.csv"
    write_csv(frame_csv, frame_summaries)

    print("\n=== TARGET COUNTS ===")
    print(
        f"anchors={total_anchor} "
        f"pos={total_pos} ({ratio(total_pos,total_anchor):.5f}) "
        f"neg={total_neg} ({ratio(total_neg,total_anchor):.5f}) "
        f"ignore={total_ign} ({ratio(total_ign,total_anchor):.5f})"
    )

    print("\n=== ORACLE-BEST ANCHOR STATUS ===")
    for thr in iou_thresholds:
        s = summary[f"iou_ge_{thr}"]
        print(
            f"IoU>={thr}: oracle GT={s['n_oracle_gt']} "
            f"recall={s['oracle_recall']:.4f}"
        )
        print(
            "  best-anchor status:",
            s["oracle_best_status_counts"],
        )
        print(
            "  fractions:",
            {
                k: round(v, 4)
                for k, v in s[
                    "oracle_best_status_fraction"
                ].items()
            },
        )
        print(
            "  oracle best assigned to SAME GT: "
            f"{s['oracle_best_assigned_to_same_gt_fraction']:.4f}"
        )
        print(
            "  has same-GT POSITIVE box at same IoU: "
            f"{s['fraction_oracle_gt_with_same_gt_positive_candidate_at_same_iou']:.4f}"
        )
        pobj = s["max_obj_same_gt_positive_iou_ge"]
        if pobj.get("n", 0):
            print(
                "  max obj among same-GT positive valid boxes: "
                f"p10={pobj.get('p10',float('nan')):.4f} "
                f"p25={pobj.get('p25',float('nan')):.4f} "
                f"p50={pobj.get('p50',float('nan')):.4f} "
                f"p75={pobj.get('p75',float('nan')):.4f} "
                f"p90={pobj.get('p90',float('nan')):.4f}"
            )

    print("\n=== REGRESSION QUALITY BY TARGET STATUS ===")
    for key in (
        "best_positive_decoded_iou",
        "best_same_gt_positive_decoded_iou",
        "best_negative_decoded_iou",
        "best_ignore_decoded_iou",
    ):
        s = summary[key]
        if s.get("n", 0):
            print(
                f"{key}: mean={s['mean']:.4f} "
                f"p50={s['p50']:.4f} p90={s['p90']:.4f}"
            )

    print("\n=== VERDICT ===")
    print(verdict)

    print("\n[wrote]", json_path)
    print("[wrote]", gt_csv)
    print("[wrote]", frame_csv)


if __name__ == "__main__":
    main()

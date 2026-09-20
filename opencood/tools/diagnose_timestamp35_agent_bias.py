#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-scene diagnosis for AirV2X Gaussian: timestamp_000035 agent bias.

This script is deliberately scoped to ONE timestamp:
    /mnt/home/suyi/AirV2X-Perception/test/
        2025_05_05_21_46_39/timestamp_000035

It compares:
    old_full
    new_full
    new_no_drone
    new_drone_only
    new_drone_no_mean_refine
    new_drone_no_geom_refine

Main questions:
1) Did the new objectness training actually change regression (`rm`), or
   merely expose boxes that already existed?
2) Are the systematically shifted boxes specifically introduced by Drone?
3) Are Drone Stage2 mean refinements causing the shift?
4) Are accurate and shifted candidates both present pre-NMS, with NMS
   selecting the wrong one after objectness changes?

Important implementation detail:
--------------------------------
For agent ablations we DO NOT delete the entire agent payload.  We remove
only `batch_merged_cam_inputs` from the disabled agent.  This keeps its
`record_len` in the batch so `extract_agent_to_ego()` still uses the
original offsets into `img_pairwise_t_matrix_collab`.  Directly deleting
Vehicle before a Drone-only run would otherwise shift Drone's pairwise
matrix slice and create an artificial coordinate error.

Verified against repository interfaces:
- Airv2xGaussian0822.forward
- present_camera_agents
- GaussianInitializer.forward
- CrossAgentInteraction.forward
- GaussianSet (dataclasses.replace compatible)
- VoxelPostprocessor.delta_to_boxes3d / post_process_airv2x
- IntermediateFusionDatasetAirv2x._metadata_path_for_idx

Outputs:
    summary.json
    per_gt.csv
    old_new_anchor_compare.csv              (if --old-checkpoint provided)
    drone_stage12_<variant>.npz
    <variant>_pre_nms.png
    <variant>_final.png

Typical command:
    CUDA_VISIBLE_DEVICES=0 python opencood/tools/diagnose_timestamp35_agent_bias.py \
      --new-checkpoint /path/to/new/net_epochXX.pth \
      --old-checkpoint /mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/airv2x_gaussian_0822_geom_sup/geom_sup_2026_09_17_00_34_27/net_epoch25.pth \
      --timestamp-dir /mnt/home/suyi/AirV2X-Perception/test/2025_05_05_21_46_39/timestamp_000035 \
      --amp \
      --out-dir opencood/logs/diagnose_timestamp35_agent_bias
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda import amp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.tools import train_utils
from opencood.utils import box_utils, common_utils


DEFAULT_TIMESTAMP = (
    "/mnt/home/suyi/AirV2X-Perception/test/"
    "2025_05_05_21_46_39/timestamp_000035"
)
DEFAULT_OLD_CKPT = (
    "/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
    "airv2x_gaussian_0822_geom_sup/"
    "geom_sup_2026_09_17_00_34_27/net_epoch25.pth"
)
AGENTS = ("vehicle", "rsu", "drone")
IOU_LEVELS = (0.3, 0.5, 0.7)


# ---------------------------------------------------------------------
# CLI / setup
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--new-checkpoint",
        required=True,
        help="checkpoint after the objectness alpha/loss change",
    )
    p.add_argument(
        "--old-checkpoint",
        default=DEFAULT_OLD_CKPT,
        help="checkpoint before the objectness change; pass '' to skip",
    )
    p.add_argument(
        "--config",
        default="",
        help="model/dataset config; defaults to new checkpoint's config.yaml",
    )
    p.add_argument(
        "--timestamp-dir",
        default=DEFAULT_TIMESTAMP,
        help="exact AirV2X timestamp directory",
    )
    p.add_argument(
        "--test-root",
        default="",
        help="override validate_dir; defaults to timestamp_dir/../..",
    )
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument(
        "--obj-threshold",
        type=float,
        default=None,
        help="diagnostic threshold override; default uses config value",
    )
    p.add_argument(
        "--split-x",
        type=float,
        default=0.0,
        help="GT/Gaussian x < split_x is summarized as 'left'",
    )
    p.add_argument(
        "--plot-score",
        action="store_true",
        help="draw score text next to pre-NMS/final predictions",
    )
    p.add_argument(
        "--out-dir",
        default="opencood/logs/diagnose_timestamp35_agent_bias",
    )
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_config(new_ckpt: Path, explicit: str) -> Path:
    p = Path(explicit) if explicit else new_ckpt.parent / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"config not found: {p}. Pass --config explicitly."
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
        k
        for k, v in state.items()
        if k in current and tuple(v.shape) != tuple(current[k].shape)
    ]
    if shape_bad:
        raise RuntimeError(
            f"{ckpt}: checkpoint/model shape mismatch: {shape_bad[:20]}"
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{ckpt}: checkpoint/model mismatch\n"
            f"missing({len(missing)}): {missing[:30]}\n"
            f"unexpected({len(unexpected)}): {unexpected[:30]}"
        )
    print(f"[ckpt] exact-compatible: {ckpt}")


def build_model(hypes, checkpoint: Path, device):
    model = train_utils.create_model(hypes).to(device)
    load_checkpoint_exact(model, checkpoint, device)
    model.eval()
    return model


# ---------------------------------------------------------------------
# Exact timestamp lookup
# ---------------------------------------------------------------------

def find_timestamp_index(dataset, timestamp_dir: Path) -> Tuple[int, str]:
    scene_name = timestamp_dir.parent.name
    timestamp_name = timestamp_dir.name

    if not hasattr(dataset, "_metadata_path_for_idx"):
        raise RuntimeError(
            "Dataset has no _metadata_path_for_idx(); this script expects "
            "the current IntermediateFusionDatasetAirv2x interface."
        )

    n = len(dataset)
    hits = []
    for idx in range(n):
        meta = str(dataset._metadata_path_for_idx(idx))
        parts = Path(meta).parts
        if scene_name in parts and timestamp_name in parts:
            hits.append((idx, meta))

    if len(hits) == 0:
        examples = []
        for idx in range(min(n, 5)):
            examples.append(str(dataset._metadata_path_for_idx(idx)))
        raise RuntimeError(
            "Target timestamp was not found in the dataset index.\n"
            f"scene={scene_name}, timestamp={timestamp_name}\n"
            f"dataset_len={n}\n"
            f"first metadata paths={examples}"
        )
    if len(hits) > 1:
        # Multiple metadata paths can theoretically represent the same timestamp
        # only if the dataset index is malformed; dataset index should be per frame.
        raise RuntimeError(
            f"Expected one dataset index for {timestamp_dir}, got {hits[:10]}"
        )

    idx, meta = hits[0]
    print(f"[target] idx={idx}")
    print(f"[target] metadata={meta}")
    return idx, meta


def fresh_cpu_batch(dataset, idx: int):
    sample = dataset[idx]
    if sample is None:
        raise RuntimeError(f"dataset[{idx}] returned None")
    return dataset.collate_batch_test([sample])


def disable_agents_keep_offsets(
    batch: Dict[str, Any],
    disabled_agents: Iterable[str],
):
    """
    Disable camera participation without deleting record_len.

    Keeping the payload is intentional: projection.extract_agent_to_ego()
    computes offsets in img_pairwise_t_matrix_collab using record_len of
    preceding agents.
    """
    ego = batch["ego"]
    for agent in disabled_agents:
        payload = ego.get(agent)
        if not isinstance(payload, dict):
            continue
        payload.pop("batch_merged_cam_inputs", None)


def present_agents_from_batch(batch: Mapping[str, Any]) -> List[str]:
    out = []
    ego = batch["ego"]
    for agent in AGENTS:
        payload = ego.get(agent)
        if not isinstance(payload, dict):
            continue
        cam = payload.get("batch_merged_cam_inputs")
        if isinstance(cam, dict):
            imgs = cam.get("imgs")
            if torch.is_tensor(imgs) and imgs.numel() > 0:
                out.append(agent)
    return out


# ---------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------

def stats(values) -> Dict[str, float]:
    x = np.asarray(list(values), dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
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


def corners_center_xy(corners: torch.Tensor) -> torch.Tensor:
    if corners is None or int(corners.shape[0]) == 0:
        device = corners.device if torch.is_tensor(corners) else "cpu"
        return torch.empty((0, 2), device=device)
    return corners[..., :2].mean(dim=1)


def _xy_aabb(corners_np: np.ndarray):
    xy = corners_np[..., :2]
    return (
        xy[..., 0].min(axis=1),
        xy[..., 1].min(axis=1),
        xy[..., 0].max(axis=1),
        xy[..., 1].max(axis=1),
    )


def exact_iou_one_gt(
    pred_np: np.ndarray,
    pred_aabb,
    gt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lossless AABB prefilter, then exact repository polygon IoU."""
    if pred_np.shape[0] == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
        )

    p_xmin, p_ymin, p_xmax, p_ymax = pred_aabb
    gxmin = float(gt[:, 0].min())
    gymin = float(gt[:, 1].min())
    gxmax = float(gt[:, 0].max())
    gymax = float(gt[:, 1].max())

    mask = (
        (p_xmax >= gxmin)
        & (p_xmin <= gxmax)
        & (p_ymax >= gymin)
        & (p_ymin <= gymax)
    )
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return idx, np.zeros((0,), dtype=np.float64)

    gt_poly = common_utils.convert_format(gt[None, ...])[0]
    pred_polys = list(common_utils.convert_format(pred_np[idx]))
    vals = np.asarray(
        common_utils.compute_iou(gt_poly, pred_polys),
        dtype=np.float64,
    )
    return idx, vals


def best_prediction_for_gt(
    pred_corners: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_corner: torch.Tensor,
) -> Dict[str, Any]:
    n_pred = int(pred_corners.shape[0])
    if n_pred == 0:
        return {
            "best_iou": 0.0,
            "best_pred_local_index": -1,
            "score": float("nan"),
            "pred_x": float("nan"),
            "pred_y": float("nan"),
            "dx": float("nan"),
            "dy": float("nan"),
            "center_error": float("nan"),
            "nearest_pred_local_index": -1,
            "nearest_score": float("nan"),
            "nearest_dx": float("nan"),
            "nearest_dy": float("nan"),
            "nearest_center_error": float("nan"),
        }

    p_np = pred_corners.detach().cpu().numpy()
    g_np = gt_corner.detach().cpu().numpy()
    idx, vals = exact_iou_one_gt(p_np, _xy_aabb(p_np), g_np)

    gt_xy = g_np[:, :2].mean(axis=0)
    pred_xy = p_np[:, :, :2].mean(axis=1)

    # nearest center is useful even when a systematic shift makes IoU very low.
    dist = np.linalg.norm(pred_xy - gt_xy[None, :], axis=1)
    nearest = int(np.argmin(dist))
    ndx = float(pred_xy[nearest, 0] - gt_xy[0])
    ndy = float(pred_xy[nearest, 1] - gt_xy[1])

    if vals.size == 0:
        best = -1
        best_iou = 0.0
        bx = by = dx = dy = cerr = float("nan")
        score = float("nan")
    else:
        j = int(np.argmax(vals))
        best = int(idx[j])
        best_iou = float(vals[j])
        bx = float(pred_xy[best, 0])
        by = float(pred_xy[best, 1])
        dx = float(bx - gt_xy[0])
        dy = float(by - gt_xy[1])
        cerr = float(math.hypot(dx, dy))
        score = float(pred_scores[best].detach().cpu().item())

    return {
        "best_iou": best_iou,
        "best_pred_local_index": best,
        "score": score,
        "pred_x": bx,
        "pred_y": by,
        "dx": dx,
        "dy": dy,
        "center_error": cerr,
        "nearest_pred_local_index": nearest,
        "nearest_score": float(pred_scores[nearest].detach().cpu().item()),
        "nearest_dx": ndx,
        "nearest_dy": ndy,
        "nearest_center_error": float(dist[nearest]),
    }


# ---------------------------------------------------------------------
# Exact post-process decomposition
# ---------------------------------------------------------------------

def resolve_num_class(hypes, pp):
    if hasattr(pp, "num_class"):
        return int(pp.num_class)
    if "num_class" in hypes:
        return int(hypes["num_class"])
    return int(hypes["model"]["args"]["num_class"])


def decompose_postprocess(
    batch: Dict[str, Any],
    output: Dict[str, torch.Tensor],
    hypes: Dict[str, Any],
    obj_threshold_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Reproduce current AirV2X postprocess while retaining every stage."""
    ego = batch["ego"]
    pp = VoxelPostprocessor(
        hypes["postprocess"],
        dataset="airv2x",
        train=False,
    )
    params = hypes["postprocess"]
    order = params["order"]
    lidar_range = params["anchor_args"]["cav_lidar_range"]
    nms_thresh = float(params["nms_thresh"])
    obj_thresh = (
        float(obj_threshold_override)
        if obj_threshold_override is not None
        else float(params["target_args"]["obj_threshold"])
    )

    anchors = ego["anchor_box"].to(output["rm"].device)
    tfm = ego["transformation_matrix"].to(output["rm"].device)

    # Objectness flatten order: H,W,A
    obj_logits = output["obj"].permute(0, 2, 3, 1).contiguous()
    objectness = torch.sigmoid(obj_logits).view(-1)

    # Class flatten order: H,W,A,C
    psm = output["psm"]
    B, AC, H, W = psm.shape
    C = resolve_num_class(hypes, pp)
    if AC % C != 0:
        raise RuntimeError(f"psm AC={AC} not divisible by C={C}")
    A = AC // C
    prob = torch.sigmoid(
        psm.view(B, C, A, H, W)
        .permute(0, 3, 4, 2, 1)
        .contiguous()
    ).view(-1, C)
    if C > 1:
        class_scores, class_labels = torch.max(prob[:, 1:], dim=-1)
        class_labels = class_labels + 1
    else:
        class_scores = prob[:, 0]
        class_labels = torch.zeros_like(class_scores, dtype=torch.long)

    boxes_all = pp.delta_to_boxes3d(output["rm"], anchors)[0].reshape(-1, 7)
    n_all = int(boxes_all.shape[0])
    if n_all != int(objectness.numel()):
        raise RuntimeError(
            f"decoded N={n_all} != objectness N={objectness.numel()}"
        )

    corners_local_all = box_utils.boxes_to_corners_3d(
        boxes_all, order=order
    )
    corners_all = box_utils.project_box3d(corners_local_all, tfm)
    all_idx = torch.arange(
        n_all, device=boxes_all.device, dtype=torch.long
    )

    # Stage 1: production objectness threshold.
    mask_obj = objectness > obj_thresh
    idx_thr = all_idx[mask_obj]
    corners_thr = corners_all[idx_thr]
    scores_thr = objectness[idx_thr]
    labels_thr = class_labels[idx_thr]

    # Optional quality only changes NMS ordering in current repo.
    if (
        params.get("quality_aware_nms", False)
        and "quality" in output
    ):
        quality_all = torch.sigmoid(
            output["quality"].permute(0, 2, 3, 1).contiguous()
        ).view(-1)
        quality_thr = quality_all[idx_thr]
    else:
        quality_thr = torch.ones_like(scores_thr)

    # Stage 2: official large + z filters.
    if int(corners_thr.shape[0]) > 0:
        keep_large = box_utils.remove_large_pred_bbx(
            corners_thr, "airv2x"
        )
        keep_z = box_utils.remove_bbx_abnormal_z(
            corners_thr,
            z_min=float(lidar_range[2]),
            z_max=float(lidar_range[5]),
        )
        keep_pre = torch.logical_and(keep_large, keep_z)
    else:
        keep_pre = torch.zeros(
            (0,), dtype=torch.bool, device=boxes_all.device
        )

    idx_pre = idx_thr[keep_pre]
    corners_pre = corners_all[idx_pre]
    scores_pre = objectness[idx_pre]
    labels_pre = class_labels[idx_pre]
    quality_pre = quality_thr[keep_pre]

    # Stage 3: production rotated NMS.
    if int(corners_pre.shape[0]) > 0:
        nms_scores = scores_pre * quality_pre
        keep_nms = box_utils.nms_rotated(
            corners_pre, nms_scores, nms_thresh
        )
        keep_nms = torch.as_tensor(
            keep_nms,
            dtype=torch.long,
            device=boxes_all.device,
        )
        idx_nms = idx_pre[keep_nms]
    else:
        idx_nms = idx_pre

    corners_nms = corners_all[idx_nms]
    scores_nms = objectness[idx_nms]
    labels_nms = class_labels[idx_nms]

    # Stage 4: production range filter.
    if int(corners_nms.shape[0]) > 0:
        keep_range = box_utils.get_mask_for_boxes_within_range_torch(
            corners_nms, lidar_range
        )
        idx_final = idx_nms[keep_range]
    else:
        idx_final = idx_nms

    corners_final = corners_all[idx_final]
    scores_final = objectness[idx_final]
    labels_final = class_labels[idx_final]

    gt_corners, gt_labels, gt_tracks = pp.generate_gt_bbx_airv2x(batch)

    def stage(corners, scores, labels, anchor_idx):
        return {
            "corners": corners,
            "scores": scores,
            "labels": labels,
            "anchor_idx": anchor_idx,
        }

    return {
        "obj_threshold": obj_thresh,
        "objectness": objectness,
        "class_scores": class_scores,
        "class_labels": class_labels,
        "boxes_all_7d": boxes_all,
        "all_decoded": stage(
            corners_all,
            objectness,
            class_labels,
            all_idx,
        ),
        "obj_threshold_stage": stage(
            corners_thr,
            scores_thr,
            labels_thr,
            idx_thr,
        ),
        "pre_nms": stage(
            corners_pre,
            scores_pre,
            labels_pre,
            idx_pre,
        ),
        "after_nms": stage(
            corners_nms,
            scores_nms,
            labels_nms,
            idx_nms,
        ),
        "final": stage(
            corners_final,
            scores_final,
            labels_final,
            idx_final,
        ),
        "gt_corners": gt_corners,
        "gt_labels": gt_labels,
        "gt_tracks": gt_tracks,
    }


# ---------------------------------------------------------------------
# Stage1 / Stage2 capture and intervention
# ---------------------------------------------------------------------

class CrossAgentCapture:
    def __init__(self, model, mode: str):
        self.model = model
        self.mode = mode
        self.stage1 = None
        self.stage2_raw = None
        self.stage2_used = None
        self._h_pre = model.cross_agent.register_forward_pre_hook(
            self._pre
        )
        self._h_post = model.cross_agent.register_forward_hook(
            self._post
        )

    def _pre(self, module, inputs):
        # cross_agent(stage1_gaussians, sources)
        self.stage1 = inputs[0]

    def _post(self, module, inputs, output):
        self.stage2_raw = output

        if self.mode not in (
            "drone_no_mean_refine",
            "drone_no_geom_refine",
        ):
            self.stage2_used = output
            return output

        stage1 = self.stage1
        if (
            not isinstance(stage1, Mapping)
            or "drone" not in stage1
            or "drone" not in output
        ):
            self.stage2_used = output
            return output

        s1 = stage1["drone"]
        s2 = output["drone"]
        if s1.n_gaussians != s2.n_gaussians:
            raise RuntimeError(
                "Stage1/Stage2 drone Gaussian count changed; cannot "
                "perform row-aligned no-refine intervention."
            )

        if self.mode == "drone_no_mean_refine":
            drone_mod = replace(s2, mean=s1.mean)
        else:
            # Preserve Stage2 ATTENDED FEATURE, but restore all geometry.
            drone_mod = replace(
                s2,
                mean=s1.mean,
                scale=s1.scale,
                quaternion=s1.quaternion,
            )

        modified = dict(output)
        modified["drone"] = drone_mod
        self.stage2_used = modified
        return modified

    def close(self):
        self._h_pre.remove()
        self._h_post.remove()


def tensor_region_stats(delta: torch.Tensor, mask: torch.Tensor):
    if int(mask.sum().item()) == 0:
        return {"n": 0}
    d = delta[mask].detach().float().cpu().numpy()
    norm = np.linalg.norm(d, axis=1)
    return {
        "n": int(d.shape[0]),
        "mean_dx": float(d[:, 0].mean()),
        "std_dx": float(d[:, 0].std()),
        "median_dx": float(np.median(d[:, 0])),
        "mean_dy": float(d[:, 1].mean()),
        "std_dy": float(d[:, 1].std()),
        "median_dy": float(np.median(d[:, 1])),
        "mean_dz": float(d[:, 2].mean()),
        "std_dz": float(d[:, 2].std()),
        "median_dz": float(np.median(d[:, 2])),
        "norm": stats(norm),
    }


def summarize_drone_stage12(
    stage1: Optional[Mapping[str, Any]],
    stage2: Optional[Mapping[str, Any]],
    split_x: float,
):
    if (
        not isinstance(stage1, Mapping)
        or not isinstance(stage2, Mapping)
        or "drone" not in stage1
        or "drone" not in stage2
    ):
        return {"present": False}

    s1 = stage1["drone"]
    s2 = stage2["drone"]
    if s1.n_gaussians != s2.n_gaussians:
        return {
            "present": True,
            "count_mismatch": [s1.n_gaussians, s2.n_gaussians],
        }

    delta = s2.mean - s1.mean
    left = s1.mean[:, 0] < float(split_x)
    right = ~left
    return {
        "present": True,
        "n": s1.n_gaussians,
        "all": tensor_region_stats(
            delta,
            torch.ones_like(left, dtype=torch.bool),
        ),
        "left_x_lt_split": tensor_region_stats(delta, left),
        "right_x_ge_split": tensor_region_stats(delta, right),
        "split_x": float(split_x),
    }


def save_drone_npz(
    path: Path,
    stage1: Optional[Mapping[str, Any]],
    stage2_raw: Optional[Mapping[str, Any]],
    stage2_used: Optional[Mapping[str, Any]],
):
    if (
        not isinstance(stage1, Mapping)
        or "drone" not in stage1
    ):
        return

    arrays = {}
    s1 = stage1["drone"]
    arrays["stage1_mean"] = s1.mean.detach().float().cpu().numpy()
    if s1.p_fg is not None:
        arrays["p_fg"] = s1.p_fg.detach().float().cpu().numpy()

    if (
        isinstance(stage2_raw, Mapping)
        and "drone" in stage2_raw
    ):
        arrays["stage2_raw_mean"] = (
            stage2_raw["drone"].mean.detach().float().cpu().numpy()
        )

    if (
        isinstance(stage2_used, Mapping)
        and "drone" in stage2_used
    ):
        arrays["stage2_used_mean"] = (
            stage2_used["drone"].mean.detach().float().cpu().numpy()
        )

    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------
# Per-GT diagnostics
# ---------------------------------------------------------------------

def make_per_gt_rows(
    variant: str,
    decomp: Dict[str, Any],
    split_x: float,
):
    gt = decomp["gt_corners"]
    gt_xy = corners_center_xy(gt).detach().float().cpu().numpy()

    rows = []
    for gi in range(int(gt.shape[0])):
        region = (
            "left" if float(gt_xy[gi, 0]) < float(split_x)
            else "right"
        )
        row = {
            "variant": variant,
            "gt_index": gi,
            "gt_x": float(gt_xy[gi, 0]),
            "gt_y": float(gt_xy[gi, 1]),
            "region": region,
        }

        for stage_name in (
            "all_decoded",
            "obj_threshold_stage",
            "pre_nms",
            "final",
        ):
            st = decomp[stage_name]
            best = best_prediction_for_gt(
                st["corners"],
                st["scores"],
                gt[gi],
            )
            prefix = {
                "all_decoded": "all",
                "obj_threshold_stage": "threshold",
                "pre_nms": "pre_nms",
                "final": "final",
            }[stage_name]
            for key, value in best.items():
                row[f"{prefix}_{key}"] = value

            local_idx = int(best["best_pred_local_index"])
            if local_idx >= 0:
                row[f"{prefix}_anchor_index"] = int(
                    st["anchor_idx"][local_idx].detach().cpu().item()
                )
            else:
                row[f"{prefix}_anchor_index"] = -1

        rows.append(row)

    return rows


def region_bias_summary(
    rows: List[Dict[str, Any]],
    prefix: str,
    region: str,
    min_iou: float,
):
    selected = [
        r for r in rows
        if r["region"] == region
        and float(r[f"{prefix}_best_iou"]) >= float(min_iou)
        and math.isfinite(float(r[f"{prefix}_dx"]))
        and math.isfinite(float(r[f"{prefix}_dy"]))
    ]
    if not selected:
        return {"n": 0}
    dx = np.asarray([r[f"{prefix}_dx"] for r in selected])
    dy = np.asarray([r[f"{prefix}_dy"] for r in selected])
    err = np.sqrt(dx ** 2 + dy ** 2)
    return {
        "n": len(selected),
        "mean_dx": float(dx.mean()),
        "std_dx": float(dx.std()),
        "median_dx": float(np.median(dx)),
        "mean_dy": float(dy.mean()),
        "std_dy": float(dy.std()),
        "median_dy": float(np.median(dy)),
        "mean_center_error": float(err.mean()),
        "median_center_error": float(np.median(err)),
    }


def nearest_region_summary(
    rows: List[Dict[str, Any]],
    prefix: str,
    region: str,
):
    selected = [
        r for r in rows
        if r["region"] == region
        and math.isfinite(
            float(r[f"{prefix}_nearest_center_error"])
        )
    ]
    if not selected:
        return {"n": 0}
    dx = np.asarray(
        [r[f"{prefix}_nearest_dx"] for r in selected],
        dtype=np.float64,
    )
    dy = np.asarray(
        [r[f"{prefix}_nearest_dy"] for r in selected],
        dtype=np.float64,
    )
    err = np.asarray(
        [r[f"{prefix}_nearest_center_error"] for r in selected],
        dtype=np.float64,
    )
    return {
        "n": len(selected),
        "mean_dx": float(dx.mean()),
        "std_dx": float(dx.std()),
        "median_dx": float(np.median(dx)),
        "mean_dy": float(dy.mean()),
        "std_dy": float(dy.std()),
        "median_dy": float(np.median(dy)),
        "mean_center_error": float(err.mean()),
        "median_center_error": float(np.median(err)),
    }


def recall_summary(rows, prefix: str):
    return {
        f"recall@{thr}": float(
            np.mean([
                float(r[f"{prefix}_best_iou"]) >= thr
                for r in rows
            ])
        )
        for thr in IOU_LEVELS
    }


def summarize_variant(
    variant: str,
    decomp: Dict[str, Any],
    rows: List[Dict[str, Any]],
    drone_raw_summary: Dict[str, Any],
    drone_used_summary: Dict[str, Any],
):
    return {
        "variant": variant,
        "obj_threshold": decomp["obj_threshold"],
        "counts": {
            "gt": int(decomp["gt_corners"].shape[0]),
            "all_decoded": int(
                decomp["all_decoded"]["corners"].shape[0]
            ),
            "obj_threshold": int(
                decomp["obj_threshold_stage"]["corners"].shape[0]
            ),
            "pre_nms": int(
                decomp["pre_nms"]["corners"].shape[0]
            ),
            "final": int(
                decomp["final"]["corners"].shape[0]
            ),
        },
        "recall": {
            "all": recall_summary(rows, "all"),
            "threshold": recall_summary(rows, "threshold"),
            "pre_nms": recall_summary(rows, "pre_nms"),
            "final": recall_summary(rows, "final"),
        },
        "final_bias_iou_ge_0.1": {
            "left": region_bias_summary(
                rows, "final", "left", 0.1
            ),
            "right": region_bias_summary(
                rows, "final", "right", 0.1
            ),
        },
        "final_bias_iou_ge_0.3": {
            "left": region_bias_summary(
                rows, "final", "left", 0.3
            ),
            "right": region_bias_summary(
                rows, "final", "right", 0.3
            ),
        },
        "pre_nms_bias_iou_ge_0.1": {
            "left": region_bias_summary(
                rows, "pre_nms", "left", 0.1
            ),
            "right": region_bias_summary(
                rows, "pre_nms", "right", 0.1
            ),
        },
        "nearest_final_bias": {
            "left": nearest_region_summary(
                rows, "final", "left"
            ),
            "right": nearest_region_summary(
                rows, "final", "right"
            ),
        },
        "drone_stage12_raw": drone_raw_summary,
        "drone_stage12_used": drone_used_summary,
    }


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def draw_box(ax, box: np.ndarray, linewidth=1.0, alpha=1.0):
    """
    Draw bottom/top BEV rectangle using the first four xy corners.
    OpenCOOD boxes_to_corners_3d keeps one 4-corner face contiguous.
    """
    xy = box[:4, :2]
    xy = np.concatenate([xy, xy[:1]], axis=0)
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        linewidth=linewidth,
        alpha=alpha,
    )


def plot_bev(
    path: Path,
    gt_corners: torch.Tensor,
    pred_stage: Dict[str, torch.Tensor],
    title: str,
    lidar_range: Sequence[float],
    plot_score: bool,
):
    gt_np = gt_corners.detach().float().cpu().numpy()
    pred_np = pred_stage["corners"].detach().float().cpu().numpy()
    scores = pred_stage["scores"].detach().float().cpu().numpy()

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)

    for box in gt_np:
        draw_box(ax, box, linewidth=1.4, alpha=0.9)

    for i, box in enumerate(pred_np):
        draw_box(ax, box, linewidth=1.2, alpha=0.9)
        if plot_score:
            center = box[:, :2].mean(axis=0)
            ax.text(
                float(center[0]),
                float(center[1]),
                f"{scores[i]:.2f}",
                fontsize=6,
            )

    ax.set_xlim(float(lidar_range[0]), float(lidar_range[3]))
    ax.set_ylim(float(lidar_range[1]), float(lidar_range[4]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ego x (m)")
    ax.set_ylabel("ego y (m)")
    ax.set_title(
        f"{title} | GT={len(gt_np)} Pred={len(pred_np)}"
    )
    ax.grid(True, linewidth=0.3, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------
# One model / variant forward
# ---------------------------------------------------------------------

def run_variant(
    model,
    dataset,
    idx: int,
    hypes,
    device,
    use_amp: bool,
    variant: str,
    split_x: float,
    obj_threshold_override: Optional[float],
    out_dir: Path,
    plot_score: bool,
):
    batch = fresh_cpu_batch(dataset, idx)

    if variant == "no_drone":
        disable_agents_keep_offsets(batch, ("drone",))
        intervention = "normal"
    elif variant == "drone_only":
        disable_agents_keep_offsets(
            batch, ("vehicle", "rsu")
        )
        intervention = "normal"
    elif variant == "vehicle_only":
        disable_agents_keep_offsets(
            batch, ("rsu", "drone")
        )
        intervention = "normal"
    elif variant == "drone_no_mean_refine":
        intervention = "drone_no_mean_refine"
    elif variant == "drone_no_geom_refine":
        intervention = "drone_no_geom_refine"
    elif variant == "full":
        intervention = "normal"
    else:
        raise KeyError(variant)

    print(
        f"\n[variant={variant}] "
        f"present={present_agents_from_batch(batch)}"
    )

    batch = train_utils.to_device(batch, device)
    ego = batch["ego"]
    ego["epoch"] = 0

    capture = CrossAgentCapture(model, intervention)
    try:
        with torch.no_grad():
            with amp.autocast(enabled=bool(use_amp)):
                output = model(ego)

        decomp = decompose_postprocess(
            batch=batch,
            output=output,
            hypes=hypes,
            obj_threshold_override=obj_threshold_override,
        )

        rows = make_per_gt_rows(
            variant=variant,
            decomp=decomp,
            split_x=split_x,
        )

        raw_summary = summarize_drone_stage12(
            capture.stage1,
            capture.stage2_raw,
            split_x,
        )
        used_summary = summarize_drone_stage12(
            capture.stage1,
            capture.stage2_used,
            split_x,
        )

        save_drone_npz(
            out_dir / f"drone_stage12_{variant}.npz",
            capture.stage1,
            capture.stage2_raw,
            capture.stage2_used,
        )

        lidar_range = hypes["postprocess"]["anchor_args"][
            "cav_lidar_range"
        ]
        plot_bev(
            out_dir / f"{variant}_pre_nms.png",
            decomp["gt_corners"],
            decomp["pre_nms"],
            title=f"{variant} PRE-NMS",
            lidar_range=lidar_range,
            plot_score=plot_score,
        )
        plot_bev(
            out_dir / f"{variant}_final.png",
            decomp["gt_corners"],
            decomp["final"],
            title=f"{variant} FINAL",
            lidar_range=lidar_range,
            plot_score=plot_score,
        )

        # Keep compact CPU tensors required for old/new direct comparison.
        compact = {
            "rm": output["rm"].detach().float().cpu(),
            "obj_prob": torch.sigmoid(
                output["obj"]
                .permute(0, 2, 3, 1)
                .contiguous()
            ).view(-1).detach().float().cpu(),
            "boxes_all_7d": decomp[
                "boxes_all_7d"
            ].detach().float().cpu(),
            "corners_all": decomp[
                "all_decoded"
            ]["corners"].detach().float().cpu(),
        }

        summary = summarize_variant(
            variant=variant,
            decomp=decomp,
            rows=rows,
            drone_raw_summary=raw_summary,
            drone_used_summary=used_summary,
        )

        return {
            "summary": summary,
            "rows": rows,
            "compact": compact,
        }
    finally:
        capture.close()


# ---------------------------------------------------------------------
# Old / new full direct comparison
# ---------------------------------------------------------------------

def compare_old_new(
    old_compact: Dict[str, torch.Tensor],
    new_compact: Dict[str, torch.Tensor],
    obj_threshold: float,
):
    old_rm = old_compact["rm"].numpy()
    new_rm = new_compact["rm"].numpy()
    if old_rm.shape != new_rm.shape:
        raise RuntimeError(
            f"old/new rm shape mismatch {old_rm.shape} vs {new_rm.shape}"
        )

    abs_rm = np.abs(new_rm - old_rm).reshape(-1)

    old_obj = old_compact["obj_prob"].numpy()
    new_obj = new_compact["obj_prob"].numpy()

    old_centers = (
        old_compact["corners_all"][..., :2]
        .mean(dim=1)
        .numpy()
    )
    new_centers = (
        new_compact["corners_all"][..., :2]
        .mean(dim=1)
        .numpy()
    )
    dxy = new_centers - old_centers
    dnorm = np.linalg.norm(dxy, axis=1)

    newly_visible = (
        (new_obj > obj_threshold)
        & (old_obj <= obj_threshold)
    )
    old_visible = old_obj > obj_threshold
    new_visible = new_obj > obj_threshold

    return {
        "rm_abs_new_minus_old": stats(abs_rm),
        "obj_probability_delta": stats(new_obj - old_obj),
        "decoded_center_shift_all_anchor_m": {
            "dx": stats(dxy[:, 0]),
            "dy": stats(dxy[:, 1]),
            "norm": stats(dnorm),
        },
        "threshold_crossing": {
            "obj_threshold": float(obj_threshold),
            "old_pass": int(old_visible.sum()),
            "new_pass": int(new_visible.sum()),
            "newly_visible": int(newly_visible.sum()),
            "newly_hidden": int(
                ((old_obj > obj_threshold)
                 & (new_obj <= obj_threshold)).sum()
            ),
        },
        "decoded_center_shift_newly_visible_m": (
            {
                "dx": stats(dxy[newly_visible, 0]),
                "dy": stats(dxy[newly_visible, 1]),
                "norm": stats(dnorm[newly_visible]),
            }
            if np.any(newly_visible)
            else {"n": 0}
        ),
    }


def old_new_anchor_rows(
    old_compact,
    new_compact,
    obj_threshold: float,
):
    old_obj = old_compact["obj_prob"].numpy()
    new_obj = new_compact["obj_prob"].numpy()
    old_xy = (
        old_compact["corners_all"][..., :2]
        .mean(dim=1)
        .numpy()
    )
    new_xy = (
        new_compact["corners_all"][..., :2]
        .mean(dim=1)
        .numpy()
    )

    keep = (
        (old_obj > obj_threshold)
        | (new_obj > obj_threshold)
    )
    idxs = np.nonzero(keep)[0]

    rows = []
    for i in idxs:
        rows.append({
            "anchor_index": int(i),
            "old_obj": float(old_obj[i]),
            "new_obj": float(new_obj[i]),
            "crossed_up": bool(
                new_obj[i] > obj_threshold
                and old_obj[i] <= obj_threshold
            ),
            "crossed_down": bool(
                old_obj[i] > obj_threshold
                and new_obj[i] <= obj_threshold
            ),
            "old_x": float(old_xy[i, 0]),
            "old_y": float(old_xy[i, 1]),
            "new_x": float(new_xy[i, 0]),
            "new_y": float(new_xy[i, 1]),
            "new_minus_old_dx": float(
                new_xy[i, 0] - old_xy[i, 0]
            ),
            "new_minus_old_dy": float(
                new_xy[i, 1] - old_xy[i, 1]
            ),
            "new_minus_old_center_shift": float(
                np.linalg.norm(new_xy[i] - old_xy[i])
            ),
        })
    return rows


# ---------------------------------------------------------------------
# CSV / console
# ---------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_variant_summary(name: str, s: Dict[str, Any]):
    c = s["counts"]
    r = s["recall"]["final"]
    print(
        f"\n[{name}] "
        f"preNMS={c['pre_nms']} final={c['final']} | "
        f"R30={r['recall@0.3']:.3f} "
        f"R50={r['recall@0.5']:.3f} "
        f"R70={r['recall@0.7']:.3f}"
    )

    for region in ("left", "right"):
        b = s["final_bias_iou_ge_0.1"][region]
        if b.get("n", 0):
            print(
                f"  final {region} IoU>=.1: n={b['n']} "
                f"dx={b['mean_dx']:+.3f}±{b['std_dx']:.3f} m "
                f"dy={b['mean_dy']:+.3f}±{b['std_dy']:.3f} m "
                f"err={b['mean_center_error']:.3f} m"
            )

    d = s["drone_stage12_raw"]
    if d.get("present"):
        for region_key in (
            "left_x_lt_split",
            "right_x_ge_split",
        ):
            q = d.get(region_key, {})
            if q.get("n", 0):
                print(
                    f"  raw Drone S2-S1 {region_key}: "
                    f"n={q['n']} "
                    f"dx={q['mean_dx']:+.3f} "
                    f"dy={q['mean_dy']:+.3f} "
                    f"dz={q['mean_dz']:+.3f} m"
                )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    opt = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    seed_all(opt.seed)
    torch.cuda.set_device(opt.gpu_id)
    device = torch.device(f"cuda:{opt.gpu_id}")

    new_ckpt = Path(opt.new_checkpoint)
    if not new_ckpt.exists():
        raise FileNotFoundError(new_ckpt)

    old_ckpt = (
        Path(opt.old_checkpoint)
        if str(opt.old_checkpoint).strip()
        else None
    )
    if old_ckpt is not None and not old_ckpt.exists():
        raise FileNotFoundError(old_ckpt)

    config = resolve_config(new_ckpt, opt.config)
    hypes = yaml_utils.load_yaml(str(config))

    timestamp_dir = Path(opt.timestamp_dir)
    if not timestamp_dir.exists():
        raise FileNotFoundError(timestamp_dir)

    test_root = (
        Path(opt.test_root)
        if opt.test_root
        else timestamp_dir.parent.parent
    )
    if not test_root.exists():
        raise FileNotFoundError(test_root)

    # Force the validation dataset to the exact test tree containing scene 35.
    hypes["validate_dir"] = str(test_root)

    print("[config]", config)
    print("[validate_dir]", hypes["validate_dir"])
    print("[timestamp]", timestamp_dir)

    dataset = build_dataset(
        hypes,
        visualize=False,
        train=False,
    )
    target_idx, target_meta = find_timestamp_index(
        dataset,
        timestamp_dir,
    )

    out_dir = Path(opt.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine the production threshold actually used.
    cfg_obj_thr = float(
        hypes["postprocess"]["target_args"]["obj_threshold"]
    )
    effective_obj_thr = (
        float(opt.obj_threshold)
        if opt.obj_threshold is not None
        else cfg_obj_thr
    )

    report = {
        "timestamp_dir": str(timestamp_dir),
        "dataset_index": int(target_idx),
        "metadata_path": target_meta,
        "config": str(config),
        "new_checkpoint": str(new_ckpt),
        "old_checkpoint": (
            str(old_ckpt) if old_ckpt is not None else None
        ),
        "configured_obj_threshold": cfg_obj_thr,
        "effective_obj_threshold": effective_obj_thr,
        "split_x": float(opt.split_x),
        "variants": {},
    }
    all_gt_rows = []

    # --------------------------------------------------------------
    # OLD FULL
    # --------------------------------------------------------------
    old_full = None
    if old_ckpt is not None:
        print("\n========== OLD FULL ==========")
        old_model = build_model(
            hypes, old_ckpt, device
        )
        old_full = run_variant(
            old_model,
            dataset,
            target_idx,
            hypes,
            device,
            opt.amp,
            variant="full",
            split_x=opt.split_x,
            obj_threshold_override=opt.obj_threshold,
            out_dir=out_dir,
            plot_score=opt.plot_score,
        )
        # Rename visual outputs to keep old/new distinct.
        for suffix in ("pre_nms.png", "final.png"):
            src = out_dir / f"full_{suffix}"
            dst = out_dir / f"old_full_{suffix}"
            if src.exists():
                src.replace(dst)
        src_npz = out_dir / "drone_stage12_full.npz"
        if src_npz.exists():
            src_npz.replace(
                out_dir / "drone_stage12_old_full.npz"
            )

        for r in old_full["rows"]:
            r["variant"] = "old_full"
        report["variants"]["old_full"] = old_full["summary"]
        all_gt_rows.extend(old_full["rows"])
        print_variant_summary(
            "old_full", old_full["summary"]
        )

        del old_model
        torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # NEW MODEL: full + agent/refinement ablations
    # --------------------------------------------------------------
    print("\n========== NEW MODEL ==========")
    new_model = build_model(
        hypes, new_ckpt, device
    )

    new_results = {}
    variant_specs = (
        ("new_full", "full"),
        ("new_no_drone", "no_drone"),
        ("new_drone_only", "drone_only"),
        ("new_vehicle_only", "vehicle_only"),
        (
            "new_drone_no_mean_refine",
            "drone_no_mean_refine",
        ),
        (
            "new_drone_no_geom_refine",
            "drone_no_geom_refine",
        ),
    )

    for public_name, mode in variant_specs:
        print(f"\n========== {public_name} ==========")
        res = run_variant(
            new_model,
            dataset,
            target_idx,
            hypes,
            device,
            opt.amp,
            variant=mode,
            split_x=opt.split_x,
            obj_threshold_override=opt.obj_threshold,
            out_dir=out_dir,
            plot_score=opt.plot_score,
        )

        # Rename files from internal mode to stable public name.
        for suffix in ("pre_nms.png", "final.png"):
            src = out_dir / f"{mode}_{suffix}"
            dst = out_dir / f"{public_name}_{suffix}"
            if src.exists():
                src.replace(dst)
        src_npz = out_dir / f"drone_stage12_{mode}.npz"
        if src_npz.exists():
            src_npz.replace(
                out_dir / f"drone_stage12_{public_name}.npz"
            )

        for r in res["rows"]:
            r["variant"] = public_name

        new_results[public_name] = res
        report["variants"][public_name] = res["summary"]
        all_gt_rows.extend(res["rows"])
        print_variant_summary(public_name, res["summary"])

    # --------------------------------------------------------------
    # Old vs New FULL: direct rm/obj/decoded-center comparison
    # --------------------------------------------------------------
    if old_full is not None:
        compare = compare_old_new(
            old_full["compact"],
            new_results["new_full"]["compact"],
            obj_threshold=effective_obj_thr,
        )
        report["old_vs_new_full"] = compare

        anchor_rows = old_new_anchor_rows(
            old_full["compact"],
            new_results["new_full"]["compact"],
            obj_threshold=effective_obj_thr,
        )
        write_csv(
            out_dir / "old_new_anchor_compare.csv",
            anchor_rows,
        )

        print("\n========== OLD vs NEW FULL ==========")
        rm = compare["rm_abs_new_minus_old"]
        print(
            "rm |new-old|: "
            f"mean={rm.get('mean', float('nan')):.6f}, "
            f"p50={rm.get('p50', float('nan')):.6f}, "
            f"p90={rm.get('p90', float('nan')):.6f}, "
            f"p99={rm.get('p99', float('nan')):.6f}"
        )
        shift = compare[
            "decoded_center_shift_all_anchor_m"
        ]["norm"]
        print(
            "decoded center new-old shift: "
            f"mean={shift.get('mean', float('nan')):.4f} m, "
            f"p50={shift.get('p50', float('nan')):.4f} m, "
            f"p90={shift.get('p90', float('nan')):.4f} m, "
            f"p99={shift.get('p99', float('nan')):.4f} m"
        )
        print(
            "threshold crossings:",
            compare["threshold_crossing"],
        )

    # --------------------------------------------------------------
    # Write report
    # --------------------------------------------------------------
    write_csv(
        out_dir / "per_gt.csv",
        all_gt_rows,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    print("\n========== KEY FILES ==========")
    print(out_dir / "summary.json")
    print(out_dir / "per_gt.csv")
    if old_full is not None:
        print(out_dir / "old_new_anchor_compare.csv")
    print(
        "\nLook first at:\n"
        "  new_full_pre_nms.png\n"
        "  new_full_final.png\n"
        "  new_no_drone_final.png\n"
        "  new_drone_only_final.png\n"
        "  new_drone_no_mean_refine_final.png\n"
        "  new_drone_no_geom_refine_final.png"
    )


if __name__ == "__main__":
    main()

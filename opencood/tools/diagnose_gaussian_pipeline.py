#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference-only ablations for AirV2X Gaussian 0822.

Tests on the SAME sampled test frames:
  1) Objectness-threshold sweep.
  2) Ranking-score sweep: obj / cls / obj*cls / sqrt(obj*cls).
  3) NMS-threshold sweep.
  4) Stage-3 opacity ablation: learned / p_fg / ones.

No model weights are changed and no training code is edited.

Designed for:
  Ivan-su-su/Airv2x_gs, baseline
  opencood/models/airv2x_gaussian_0822.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import types
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import box_utils
from opencood.utils import eval_utils_opv2v as eval_utils


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference-only objectness/NMS/opacity ablations for AirV2X Gaussian 0822"
    )
    p.add_argument("--model_dir", required=True, type=str)
    p.add_argument("--config_file", default="config.yaml", type=str)
    p.add_argument("--eval_epoch", default=13, type=int)
    p.add_argument("--eval_best_epoch", action="store_true")
    p.add_argument("--gpu_id", default=0, type=int)

    p.add_argument("--samples_per_scenario", default=10, type=int)
    p.add_argument("--expected_scenarios", default=5, type=int)
    p.add_argument("--seed", default=20260923, type=int)
    p.add_argument(
        "--selected_frames_json",
        default=None,
        type=str,
        help=(
            "Optional selected_frames.json from diagnose_gaussian_pipeline.py. "
            "Use this to guarantee exactly the same 50 frames."
        ),
    )
    p.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Default: <model_dir>/gaussian_ablation_epochXXX",
    )

    p.add_argument(
        "--obj_thresholds",
        nargs="+",
        type=float,
        default=[0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
    )
    p.add_argument(
        "--nms_thresholds",
        nargs="+",
        type=float,
        default=[0.10, 0.15, 0.20, 0.30, 0.50],
    )
    p.add_argument(
        "--ranking_obj_threshold",
        default=0.05,
        type=float,
        help="Objectness gate used by ranking and NMS experiments.",
    )
    p.add_argument(
        "--relaxed_nms",
        default=0.30,
        type=float,
        help="NMS used in the relaxed opacity comparison.",
    )
    return p.parse_args()


# =============================================================================
# Generic utilities
# =============================================================================

def empty_result_stat() -> Dict[float, Dict[str, Any]]:
    return {
        t: {"tp": [], "fp": [], "gt": 0, "score": []}
        for t in (0.3, 0.5, 0.7)
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def extract_gt_hwl(ego: Mapping[str, Any]) -> torch.Tensor:
    boxes = ego["object_bbx_center"]
    mask = ego["object_bbx_mask"]

    if boxes.dim() == 3:
        boxes = boxes[0]
    if mask.dim() == 2:
        mask = mask[0]

    return boxes[mask > 0, :7].float()


def scenario_name_from_database(dataset: Any, scenario_idx: int, test_root: str) -> str:
    scenario = dataset.scenario_database[scenario_idx]
    first_cav = next(iter(scenario.values()))

    metadata = None
    for _, value in first_cav.items():
        if isinstance(value, Mapping) and "metadata_path" in value:
            metadata = str(value["metadata_path"])
            break

    if metadata is None:
        return f"scenario_{scenario_idx:02d}"

    try:
        rel = os.path.relpath(metadata, test_root)
        first = rel.split(os.sep)[0]
        if first not in (".", "..") and not first.startswith(".."):
            return first
    except Exception:
        pass

    parts = Path(metadata).parts
    return parts[-5] if len(parts) >= 5 else f"scenario_{scenario_idx:02d}"


def sample_test_indices(
    dataset: Any,
    test_root: str,
    samples_per_scenario: int,
    seed: int,
    expected_scenarios: int,
) -> List[Dict[str, Any]]:
    n_scenarios = len(dataset.len_record)
    if expected_scenarios > 0 and n_scenarios != expected_scenarios:
        raise RuntimeError(
            f"Expected {expected_scenarios} scenarios, found {n_scenarios}. "
            "Set --expected_scenarios 0 to disable the check."
        )

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []

    for s in range(n_scenarios):
        start = 0 if s == 0 else int(dataset.len_record[s - 1])
        end = int(dataset.len_record[s])
        n = end - start
        k = min(int(samples_per_scenario), n)
        ids = sorted(rng.sample(range(start, end), k))
        name = scenario_name_from_database(dataset, s, test_root)

        for idx in ids:
            selected.append(
                {
                    "scenario_index": s,
                    "scenario": name,
                    "dataset_index": int(idx),
                    "scenario_local_index": int(idx - start),
                }
            )

    return selected


def load_or_sample_frames(
    opt: argparse.Namespace,
    dataset: Any,
    test_root: str,
) -> List[Dict[str, Any]]:
    if opt.selected_frames_json:
        path = Path(opt.selected_frames_json)
        selected = json.loads(path.read_text(encoding="utf-8"))
        print(f"[sampling] loaded {len(selected)} frames from {path}")
        return selected

    return sample_test_indices(
        dataset=dataset,
        test_root=test_root,
        samples_per_scenario=opt.samples_per_scenario,
        seed=opt.seed,
        expected_scenarios=opt.expected_scenarios,
    )


# =============================================================================
# Stage-3 opacity override
# =============================================================================

@contextmanager
def opacity_mode(model_core: torch.nn.Module, mode: str):
    """
    learned:
        Exact current Stage-3 behavior.

    pfg:
        alpha_pre = p_fg
        opacity-weighted pool uses p_fg;
        alpha_final = pooled p_fg;
        splat uses pooled p_fg.

    ones:
        alpha_pre = 1
        pool becomes ordinary mean;
        alpha_final = 1;
        splat has no opacity attenuation.
    """
    stage3 = model_core.stage3
    old_enabled = bool(stage3.opacity_enabled)
    old_calibrate = stage3._calibrate_opacity

    if mode == "learned":
        try:
            yield
        finally:
            pass
        return

    if not old_enabled:
        raise RuntimeError(
            "Loaded model has stage3.opacity_enabled=False. "
            "This opacity ablation expects opacity to be enabled."
        )

    def replacement(self, agent: str, gaussians):
        if mode == "pfg":
            if gaussians.p_fg is None:
                raise ValueError("p_fg opacity mode requires GaussianSet.p_fg")
            return gaussians.p_fg.to(
                device=gaussians.feature.device,
                dtype=gaussians.feature.dtype,
            ).clamp(1.0e-4, 1.0)
        if mode == "ones":
            return torch.ones(
                (gaussians.n_gaussians,),
                device=gaussians.feature.device,
                dtype=gaussians.feature.dtype,
            )
        raise ValueError(mode)

    stage3._calibrate_opacity = types.MethodType(replacement, stage3)

    try:
        yield
    finally:
        stage3.opacity_enabled = old_enabled
        stage3._calibrate_opacity = old_calibrate


# =============================================================================
# Decode once; custom post-processing many times
# =============================================================================

def decode_raw(
    output: Mapping[str, torch.Tensor],
    ego: Mapping[str, Any],
    post: Any,
) -> Dict[str, torch.Tensor]:
    """
    Decode ALL anchors, before threshold/NMS.

    Ordering exactly follows VoxelPostprocessor.post_process_airv2x:
      obj -> [B,H,W,A]
      psm -> [B,H,W,A,C]
      rm  -> H*W*A boxes
    """
    obj_logits = output["obj"]
    psm = output["psm"]
    rm = output["rm"]

    obj = torch.sigmoid(
        obj_logits.permute(0, 2, 3, 1).contiguous()
    ).reshape(-1)

    B, AC, H, W = psm.shape
    A = int(obj_logits.shape[1])
    if AC % A != 0:
        raise ValueError(f"psm channels={AC} not divisible by anchor count={A}")
    C = AC // A

    cls_prob_all = torch.sigmoid(
        psm.view(B, C, A, H, W)
        .permute(0, 3, 4, 2, 1)
        .contiguous()
        .view(-1, C)
    )

    # Repository post-process ignores channel 0 when choosing semantic class.
    if C > 1:
        cls_score, cls_local = cls_prob_all[:, 1:].max(dim=-1)
        cls_label = cls_local + 1
    else:
        cls_score = cls_prob_all[:, 0]
        cls_label = torch.zeros_like(cls_score, dtype=torch.long)

    anchor = ego["anchor_box"]
    if anchor.dim() == 5 and int(anchor.shape[0]) == 1:
        anchor = anchor[0]

    boxes_hwl = post.delta_to_boxes3d(rm, anchor)[0]

    if not (
        boxes_hwl.shape[0]
        == obj.shape[0]
        == cls_score.shape[0]
        == cls_label.shape[0]
    ):
        raise RuntimeError(
            "Decoded box/objectness/class ordering mismatch: "
            f"box={boxes_hwl.shape}, obj={obj.shape}, cls={cls_score.shape}"
        )

    finite = (
        torch.isfinite(boxes_hwl).all(dim=-1)
        & torch.isfinite(obj)
        & torch.isfinite(cls_score)
    )

    return {
        "boxes_hwl": boxes_hwl[finite],
        "obj": obj[finite],
        "cls": cls_score[finite],
        "label": cls_label[finite],
    }


def select_and_rank(
    raw: Dict[str, torch.Tensor],
    rank_mode: str,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obj = raw["obj"]
    cls = raw["cls"]

    if rank_mode == "obj":
        keep = obj > float(threshold)
        score = obj
    elif rank_mode == "cls":
        keep = cls > float(threshold)
        score = cls
    elif rank_mode == "obj_cls":
        # Deliberately use objectness only as the permissive gate;
        # semantic confidence participates in ranking.
        keep = obj > float(threshold)
        score = obj * cls
    elif rank_mode == "sqrt_obj_cls":
        keep = obj > float(threshold)
        score = torch.sqrt((obj * cls).clamp_min(0.0))
    else:
        raise ValueError(f"Unknown rank_mode={rank_mode}")

    return (
        raw["boxes_hwl"][keep],
        score[keep],
        raw["label"][keep],
    )


def custom_postprocess(
    raw: Dict[str, torch.Tensor],
    ego: Mapping[str, Any],
    post: Any,
    dataset: Any,
    rank_mode: str,
    threshold: float,
    nms_thresh: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Reproduce the repository's post_process_airv2x after candidate selection,
    but with explicit configurable selection score and NMS threshold.

    Returns:
        projected corner boxes [N,8,3], score [N], labels [N]
    """
    boxes_hwl, score, labels = select_and_rank(raw, rank_mode, threshold)

    if boxes_hwl.numel() == 0:
        return None, None, None

    transform = ego["transformation_matrix"]
    if transform.is_cuda != boxes_hwl.is_cuda:
        transform = transform.to(boxes_hwl.device)

    corners = box_utils.boxes_to_corners_3d(
        boxes_hwl,
        order=post.params["order"],
    )
    projected = box_utils.project_box3d(corners, transform)

    # Same pre-filters as current post_process_airv2x.
    keep_large = box_utils.remove_large_pred_bbx(projected, dataset)

    z_min = float(post.lidar_range[2])
    z_max = float(post.lidar_range[5])
    keep_z = box_utils.remove_bbx_abnormal_z(
        projected,
        z_min=z_min,
        z_max=z_max,
    )
    keep = torch.logical_and(keep_large, keep_z)

    projected = projected[keep]
    score = score[keep]
    labels = labels[keep]

    if projected.shape[0] == 0:
        return None, None, None

    nms_keep = box_utils.nms_rotated(
        projected,
        score,
        float(nms_thresh),
    )
    projected = projected[nms_keep]
    score = score[nms_keep]
    labels = labels[nms_keep]

    range_keep = box_utils.get_mask_for_boxes_within_range_torch(
        projected,
        post.lidar_range,
    )
    projected = projected[range_keep]
    score = score[range_keep]
    labels = labels[range_keep]

    if projected.shape[0] == 0:
        return None, None, None

    return projected, score, labels


# =============================================================================
# Experiment definitions
# =============================================================================

def build_experiments(opt: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Intentionally NOT a full Cartesian product.

    Each group changes one thing at a time, so its result is interpretable.
    """
    exps: List[Dict[str, Any]] = []

    # 1) Objectness threshold only: exact current ranking/NMS.
    for t in opt.obj_thresholds:
        exps.append(
            {
                "experiment": "01_obj_threshold",
                "opacity": "learned",
                "rank_mode": "obj",
                "threshold": float(t),
                "nms": 0.15,
            }
        )

    # 2) Ranking score only, all under a permissive gate.
    for rank in ("obj", "cls", "obj_cls", "sqrt_obj_cls"):
        exps.append(
            {
                "experiment": "02_ranking",
                "opacity": "learned",
                "rank_mode": rank,
                "threshold": float(opt.ranking_obj_threshold),
                "nms": 0.15,
            }
        )

    # 3) NMS: compare current obj ranking vs localization/ranking proxy obj*cls.
    for rank in ("obj", "obj_cls"):
        for nms in opt.nms_thresholds:
            exps.append(
                {
                    "experiment": "03_nms",
                    "opacity": "learned",
                    "rank_mode": rank,
                    "threshold": float(opt.ranking_obj_threshold),
                    "nms": float(nms),
                }
            )

    # 4a) Opacity under CURRENT postprocess.
    for opacity in ("learned", "pfg", "ones"):
        exps.append(
            {
                "experiment": "04_opacity_current_post",
                "opacity": opacity,
                "rank_mode": "obj",
                "threshold": 0.20,
                "nms": 0.15,
            }
        )

    # 4b) Opacity after reducing the known postprocess bottleneck.
    for opacity in ("learned", "pfg", "ones"):
        exps.append(
            {
                "experiment": "05_opacity_relaxed_post",
                "opacity": opacity,
                "rank_mode": "obj_cls",
                "threshold": float(opt.ranking_obj_threshold),
                "nms": float(opt.relaxed_nms),
            }
        )

    # De-duplicate identical configs but preserve experiment labels by
    # retaining every row in the final result. Forward outputs are cached
    # by opacity mode, so duplicate evaluation is cheap.
    return exps


def config_key(exp: Mapping[str, Any]) -> Tuple[str, str, float, float]:
    return (
        str(exp["opacity"]),
        str(exp["rank_mode"]),
        float(exp["threshold"]),
        float(exp["nms"]),
    )


# =============================================================================
# Metrics accumulation
# =============================================================================

def add_eval(
    stat: Dict[float, Dict[str, Any]],
    pred_box: Optional[torch.Tensor],
    pred_score: Optional[torch.Tensor],
    gt_corners: torch.Tensor,
) -> None:
    for iou in (0.3, 0.5, 0.7):
        eval_utils.caluclate_tp_fp(
            pred_box,
            pred_score,
            gt_corners,
            stat,
            iou,
        )


def summarize_eval(stat: Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
    row = {}
    for iou in (0.3, 0.5, 0.7):
        tmp = copy.deepcopy(stat)
        ap, _, _ = eval_utils.calculate_ap(
            tmp,
            iou,
            global_sort_detections=False,
        )
        gt = int(stat[iou]["gt"])
        tp = int(sum(stat[iou]["tp"]))
        row[f"ap_{int(iou * 100)}"] = float(ap)
        row[f"recall_{int(iou * 100)}"] = float(tp / gt) if gt > 0 else None
        row[f"tp_{int(iou * 100)}"] = tp
        row[f"gt_{int(iou * 100)}"] = gt
    return row


class ObjectnessCollector:
    def __init__(self, thresholds: Sequence[float]):
        self.thresholds = [float(x) for x in thresholds]
        self.pos_obj: List[torch.Tensor] = []
        self.neg_obj: List[torch.Tensor] = []
        self.pos_cls: List[torch.Tensor] = []
        self.neg_cls: List[torch.Tensor] = []

    def add(
        self,
        output: Mapping[str, torch.Tensor],
        ego: Mapping[str, Any],
    ) -> None:
        if "label_dict" not in ego:
            return
        label = ego["label_dict"]
        if "pos_equal_one" not in label or "neg_equal_one" not in label:
            return

        obj_logits = output["obj"]
        obj = torch.sigmoid(
            obj_logits.permute(0, 2, 3, 1).contiguous()
        ).reshape(-1)

        psm = output["psm"]
        B, AC, H, W = psm.shape
        A = int(obj_logits.shape[1])
        C = AC // A
        cls_all = torch.sigmoid(
            psm.view(B, C, A, H, W)
            .permute(0, 3, 4, 2, 1)
            .contiguous()
            .view(-1, C)
        )
        cls = cls_all[:, 1:].max(dim=-1).values if C > 1 else cls_all[:, 0]

        pos = label["pos_equal_one"].reshape(-1) > 0
        neg = label["neg_equal_one"].reshape(-1) > 0

        n = min(obj.numel(), pos.numel())
        obj, cls = obj[:n], cls[:n]
        pos, neg = pos[:n], neg[:n]

        self.pos_obj.append(obj[pos].detach().float().cpu())
        self.neg_obj.append(obj[neg].detach().float().cpu())
        self.pos_cls.append(cls[pos].detach().float().cpu())
        self.neg_cls.append(cls[neg].detach().float().cpu())

    def summary(self) -> Dict[str, Any]:
        def cat(xs):
            xs = [x for x in xs if x.numel() > 0]
            return torch.cat(xs) if xs else torch.empty((0,))

        pos_obj = cat(self.pos_obj)
        neg_obj = cat(self.neg_obj)
        pos_cls = cat(self.pos_cls)
        neg_cls = cat(self.neg_cls)

        out: Dict[str, Any] = {
            "n_positive_anchors": int(pos_obj.numel()),
            "n_negative_anchors": int(neg_obj.numel()),
        }

        for name, x in (
            ("pos_obj", pos_obj),
            ("neg_obj", neg_obj),
            ("pos_cls", pos_cls),
            ("neg_cls", neg_cls),
        ):
            if x.numel():
                out[f"{name}_mean"] = float(x.mean().item())
                out[f"{name}_median"] = float(x.median().item())
                out[f"{name}_p10"] = float(torch.quantile(x, 0.10).item())
                out[f"{name}_p90"] = float(torch.quantile(x, 0.90).item())

        for t in self.thresholds:
            if pos_obj.numel():
                out[f"positive_anchor_obj_recall_at_{t:g}"] = float(
                    (pos_obj > t).float().mean().item()
                )
            if neg_obj.numel():
                out[f"negative_anchor_obj_pass_at_{t:g}"] = float(
                    (neg_obj > t).float().mean().item()
                )

        return out


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    opt = parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    hypes = yaml_utils.load_yaml(None, opt)
    if "test_dir" not in hypes:
        raise KeyError("config.yaml has no test_dir")
    hypes["validate_dir"] = hypes["test_dir"]

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_id)
        device = torch.device(f"cuda:{opt.gpu_id}")
    else:
        device = torch.device("cpu")

    print(f"[device] {device}")
    print(f"[test_dir] {hypes['test_dir']}")

    print("[dataset] building...")
    dataset = build_dataset(
        hypes,
        visualize=False,
        train=False,
    )
    print(
        f"[dataset] samples={len(dataset)} "
        f"scenarios={len(dataset.len_record)}"
    )

    selected = load_or_sample_frames(
        opt,
        dataset,
        hypes["test_dir"],
    )

    out_dir = Path(
        opt.output_dir
        or os.path.join(
            opt.model_dir,
            f"gaussian_ablation_epoch{opt.eval_epoch:03d}",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_frames.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[model] creating...")
    model = train_utils.create_model(hypes, dataset)
    model.to(device)

    print("[model] loading checkpoint...")
    loaded_epoch, model = train_utils.load_model(
        opt.model_dir,
        model,
        opt.eval_epoch,
        start_from_best=opt.eval_best_epoch,
    )
    model.eval()
    core = model.module if hasattr(model, "module") else model

    if core.__class__.__name__.lower() != "airv2xgaussian0822":
        print(
            f"[warning] model class={core.__class__.__name__}; "
            "this script is for Airv2xGaussian0822."
        )

    if not bool(core.stage3.opacity_enabled):
        raise RuntimeError(
            "stage3.opacity_enabled=False in the loaded model. "
            "The requested opacity ablation cannot be run faithfully."
        )

    post = dataset.post_processor
    experiments = build_experiments(opt)
    unique_cfgs = {}
    for exp in experiments:
        unique_cfgs[config_key(exp)] = exp

    print(f"[sampling] frames={len(selected)}")
    print(f"[experiments] rows={len(experiments)}, unique configs={len(unique_cfgs)}")
    print(
        "[opacity modes] learned / pfg / ones "
        "(3 model forwards per sampled frame in total)"
    )

    # result_stat[config_key]["all" or scenario]
    stats: Dict[
        Tuple[str, str, float, float],
        Dict[str, Dict[float, Dict[str, Any]]],
    ] = {}
    pred_counts: Dict[Tuple[str, str, float, float], Dict[str, List[int]]] = {}

    for key in unique_cfgs:
        stats[key] = defaultdict(empty_result_stat)
        pred_counts[key] = defaultdict(list)

    obj_diag = {
        mode: ObjectnessCollector(opt.obj_thresholds)
        for mode in ("learned", "pfg", "ones")
    }

    # Group required configs by opacity so each model forward is reused.
    cfgs_by_opacity: Dict[str, List[Tuple[str, str, float, float]]] = defaultdict(list)
    for key in unique_cfgs:
        cfgs_by_opacity[key[0]].append(key)

    for frame_i, item in enumerate(selected, start=1):
        scenario = str(item["scenario"])
        idx = int(item["dataset_index"])
        print(
            f"\n[{frame_i:02d}/{len(selected):02d}] "
            f"{scenario} idx={idx}"
        )

        sample = dataset[idx]
        batch = dataset.collate_batch_test([sample])
        batch = train_utils.to_device(batch, device)
        ego = batch["ego"]

        gt_hwl = extract_gt_hwl(ego)
        gt_corners = box_utils.boxes_to_corners_3d(
            gt_hwl,
            order=post.params["order"],
        )

        for mode in ("learned", "pfg", "ones"):
            with opacity_mode(core, mode):
                with torch.no_grad():
                    output = model(ego)

            raw = decode_raw(output, ego, post)
            obj_diag[mode].add(output, ego)

            print(
                f"  opacity={mode:7s} "
                f"raw={raw['boxes_hwl'].shape[0]} "
                f"obj_mean={raw['obj'].mean().item():.4f} "
                f"cls_mean={raw['cls'].mean().item():.4f}"
            )

            for key in cfgs_by_opacity[mode]:
                _, rank_mode, threshold, nms_thresh = key

                with torch.no_grad():
                    pred_box, pred_score, _ = custom_postprocess(
                        raw=raw,
                        ego=ego,
                        post=post,
                        dataset=dataset,
                        rank_mode=rank_mode,
                        threshold=threshold,
                        nms_thresh=nms_thresh,
                    )

                n_pred = 0 if pred_box is None else int(pred_box.shape[0])
                for scope in ("all", scenario):
                    add_eval(
                        stats[key][scope],
                        pred_box,
                        pred_score,
                        gt_corners,
                    )
                    pred_counts[key][scope].append(n_pred)

    # -------------------------------------------------------------------------
    # Flatten metrics back to experiment rows.
    # -------------------------------------------------------------------------
    result_rows: List[Dict[str, Any]] = []

    scopes = ["all"] + sorted(set(str(x["scenario"]) for x in selected))
    for exp in experiments:
        key = config_key(exp)
        for scope in scopes:
            metric = summarize_eval(stats[key][scope])
            counts = pred_counts[key][scope]
            result_rows.append(
                {
                    **exp,
                    "scope": scope,
                    **metric,
                    "pred_boxes_per_frame": (
                        float(np.mean(counts)) if counts else None
                    ),
                }
            )

    # Objectness calibration summary per opacity mode.
    obj_summary = {
        mode: collector.summary()
        for mode, collector in obj_diag.items()
    }

    # -------------------------------------------------------------------------
    # Write output.
    # -------------------------------------------------------------------------
    write_csv(out_dir / "ablation_results.csv", result_rows)
    (out_dir / "ablation_results.json").write_text(
        json.dumps(result_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "objectness_calibration.json").write_text(
        json.dumps(obj_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    config_dump = {
        "loaded_epoch": int(loaded_epoch),
        "model_class": core.__class__.__name__,
        "n_frames": len(selected),
        "scenario_counts": dict(Counter(str(x["scenario"]) for x in selected)),
        "test_dir": hypes["test_dir"],
        "original_obj_threshold": float(
            post.params["target_args"].get("obj_threshold", 0.2)
        ),
        "original_score_threshold": float(
            post.params["target_args"].get("score_threshold", 0.2)
        ),
        "original_nms_threshold": float(post.params.get("nms_thresh", 0.15)),
        "opacity_enabled": bool(core.stage3.opacity_enabled),
        "obj_thresholds": [float(x) for x in opt.obj_thresholds],
        "nms_thresholds": [float(x) for x in opt.nms_thresholds],
        "ranking_obj_threshold": float(opt.ranking_obj_threshold),
        "relaxed_nms": float(opt.relaxed_nms),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config_dump, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Console summary: ALL scope only.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 112)
    print("ABLATION RESULTS -- ALL 50 SAMPLED FRAMES")
    print("=" * 112)

    all_rows = [r for r in result_rows if r["scope"] == "all"]
    by_exp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        by_exp[r["experiment"]].append(r)

    for exp_name in sorted(by_exp):
        print(f"\n[{exp_name}]")
        rows = by_exp[exp_name]
        rows = sorted(
            rows,
            key=lambda r: (
                -1.0 if r["ap_50"] is None else -float(r["ap_50"]),
                str(r["opacity"]),
                str(r["rank_mode"]),
            ),
        )
        for r in rows:
            print(
                "  "
                f"opacity={r['opacity']:7s} "
                f"rank={r['rank_mode']:12s} "
                f"thr={r['threshold']:.3f} "
                f"nms={r['nms']:.2f} | "
                f"AP30={r['ap_30']:.4f} "
                f"AP50={r['ap_50']:.4f} "
                f"AP70={r['ap_70']:.4f} | "
                f"R30={r['recall_30']:.4f} "
                f"R50={r['recall_50']:.4f} "
                f"R70={r['recall_70']:.4f} | "
                f"boxes/frame={r['pred_boxes_per_frame']:.1f}"
            )

    print("\n[objectness calibration]")
    for mode in ("learned", "pfg", "ones"):
        s = obj_summary[mode]
        print(
            f"  {mode:7s}: "
            f"pos_obj_mean={s.get('pos_obj_mean')} "
            f"neg_obj_mean={s.get('neg_obj_mean')} "
            f"pos_obj_median={s.get('pos_obj_median')}"
        )
        for t in opt.obj_thresholds:
            print(
                f"    t={t:g}: "
                f"positive-pass={s.get(f'positive_anchor_obj_recall_at_{t:g}')} "
                f"negative-pass={s.get(f'negative_anchor_obj_pass_at_{t:g}')}"
            )

    print("\n[written]")
    for name in (
        "selected_frames.json",
        "ablation_results.csv",
        "ablation_results.json",
        "objectness_calibration.json",
        "run_config.json",
    ):
        print(" ", out_dir / name)


if __name__ == "__main__":
    main()

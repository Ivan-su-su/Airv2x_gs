"""Read-only low-recall joint audit for AirV2X Gaussian 0911version.

Does not modify model / loss / clip / threshold / NMS. No optimizer.step.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opencood.data_utils.datasets import build_dataset
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.hypes_yaml import yaml_utils
from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.fusion.gaussian_bev_splat import (
    GaussianBEVSplat,
)
from opencood.models.airv2x_gaussian_0822 import (
    CAMERA_GEOMETRY_KEY,
    present_camera_agents,
)
from opencood.tools import train_utils
from opencood.utils import box_utils, common_utils

DEFAULT_MODEL_DIR = (
    ROOT
    / "opencood/logs/airv2x_gaussian_0822_camera"
    / "v45_scale_asym_2026_09_11_15_54_05"
)
CKPT_NAME = "net_epoch2.pth"
YAML_PATH = ROOT / "opencood/hypes_yaml/airv2x/camera/det/airv2x_gaussian_0822.yaml"
CELL_FAIL_M = None  # filled from voxel * feature_stride
SYNTH_XY = [
    ("center", 0.0, 0.0),
    ("q1", 50.0, 10.0),
    ("q2", -50.0, 10.0),
    ("q3", -50.0, -10.0),
    ("q4", 50.0, -10.0),
]


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


def _pct(x: np.ndarray, qs: Sequence[float]) -> Dict[str, float]:
    if x.size == 0:
        return {f"p{int(q * 100)}": float("nan") for q in qs}
    return {f"p{int(q * 100)}": float(np.percentile(x, q * 100)) for q in qs}


def _dist_stats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), **_pct(x, (0.5, 0.9, 0.99))}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        **_pct(x, (0.5, 0.9, 0.99)),
    }


def _load_hypes(model_dir: Path) -> Dict[str, Any]:
    opt = argparse.Namespace(model_dir=str(model_dir), config_file="config.yaml")
    return yaml_utils.load_yaml(str(YAML_PATH), opt=opt)


def _load_weights(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    blob = torch.load(str(ckpt_path), map_location=device)
    src = blob.get("model_state_dict", blob)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in src.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {ckpt_path.name}: missing={len(missing)} unexpected={len(unexpected)}")


def _argmax2d(amp: torch.Tensor) -> Tuple[int, int]:
    idx = int(amp.reshape(-1).argmax())
    width = int(amp.shape[-1])
    return idx // width, idx % width


def expected_splat_idx(
    x: float, y: float, xmin: float, ymin: float, vx: float, vy: float
) -> Tuple[int, int]:
    ix = int(math.floor((x - xmin) / vx))
    iy = int(math.floor((y - ymin) / vy))
    return ix, iy


def splat_cell_center(
    ix: int, iy: int, xmin: float, ymin: float, vx: float, vy: float
) -> Tuple[float, float]:
    return xmin + (ix + 0.5) * vx, ymin + (iy + 0.5) * vy


def make_one_gaussian(
    x: float,
    y: float,
    z: float,
    feature_dim: int,
    scale_xyz: Tuple[float, float, float],
    device: torch.device,
) -> GaussianSet:
    mean = torch.tensor([[x, y, z]], device=device, dtype=torch.float32)
    scale = torch.tensor([scale_xyz], device=device, dtype=torch.float32)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    feat = torch.ones((1, feature_dim), device=device, dtype=torch.float32)
    return GaussianSet(
        mean=mean,
        scale=scale,
        quaternion=quat,
        feature=feat,
        batch_index=torch.zeros(1, dtype=torch.long, device=device),
        agent="vehicle",
        frame="ego",
    )


def make_aniso_gaussian(
    axis: str, feature_dim: int, device: torch.device, pc_range: Sequence[float]
) -> GaussianSet:
    x = 0.5 * (float(pc_range[0]) + float(pc_range[3]))
    y = 0.5 * (float(pc_range[1]) + float(pc_range[4]))
    if axis == "x":
        scale = (4.0, 0.3, 0.3)
    else:
        scale = (0.3, 4.0, 0.3)
    return make_one_gaussian(x, y, -1.0, feature_dim, scale, device)


def audit_static_mapping(hypes: Dict[str, Any]) -> Dict[str, Any]:
    pc = [float(v) for v in hypes["preprocess"]["cav_lidar_range"]]
    vx, vy, vz = [float(v) for v in hypes["model"]["args"]["gaussian_stage3"]["voxel_size"]]
    splat_nx = int((pc[3] - pc[0]) / vx)
    splat_ny = int((pc[4] - pc[1]) / vy)
    aa = hypes["postprocess"]["anchor_args"]
    ah, aw = int(aa["H"]), int(aa["W"])
    stride = int(aa.get("feature_stride", 2))
    det_h, det_w = ah // stride, aw // stride
    dummy = type("P", (), {"params": hypes["postprocess"], "dataset": "airv2x"})()
    post = VoxelPostprocessor(hypes["postprocess"], dataset="airv2x", train=False)
    anchors = post.generate_anchor_box()
    x = np.linspace(pc[0] + float(aa["vw"]), pc[3] - float(aa["vw"]), det_w)
    y = np.linspace(pc[1] + float(aa["vh"]), pc[4] - float(aa["vh"]), det_h)
    return {
        "point_cloud_range": pc,
        "splat_voxel_xy": [vx, vy],
        "splat_hw_ny_nx": [splat_ny, splat_nx],
        "anchor_yaml_HW": [ah, aw],
        "feature_stride": stride,
        "anchor_grid_hw": [int(anchors.shape[0]), int(anchors.shape[1])],
        "det_expected_hw": [det_h, det_w],
        "splat_vs_anchor_fullres_match": [splat_ny == ah, splat_nx == aw],
        "det_vs_anchor_match": [
            int(anchors.shape[0]) == det_h,
            int(anchors.shape[1]) == det_w,
        ],
        "convention": {
            "splat_tensor": "[B,C,ny,nx] rows=y cols=x",
            "ix": "floor((x-xmin)/vx)",
            "iy": "floor((y-ymin)/vy)",
            "splat_cell_center": "xmin+(ix+0.5)*vx",
            "anchor_x": "linspace(xmin+vw, xmax-vw, W//stride)",
            "anchor_y": "linspace(ymin+vh, ymax-vh, H//stride)",
            "meshgrid": "np.meshgrid(x,y) default xy -> (H,W)",
            "backbone": "layer_strides[2,2,2] + upsample[1,2,4] => H/2,W/2",
            "shrink_header_stride": hypes["model"]["args"]["modality_fusion"][
                "shrink_header"
            ]["stride"],
        },
        "anchor0_xy": [float(anchors[0, 0, 0, 0]), float(anchors[0, 0, 0, 1])],
        "anchor_last_xy": [
            float(anchors[-1, -1, 0, 0]),
            float(anchors[-1, -1, 0, 1]),
        ],
        "linspace_x0_y0": [float(x[0]), float(y[0])],
        "notes": [
            "splat radius[:,0]=rx along x/col, radius[:,1]=ry along y/row",
            "covariance Σ_xy = Σ_3d[:2,:2]; inv uses (dx,dy) matching tensor axes",
            "anchor cz is hardcoded -1.0 in generate_anchor_box",
        ],
        "half_cell_offset_m": {
            "splat_vs_anchor_at_origin_est": 0.4,
            "det_cell_m": [vx * stride, vy * stride],
        },
        "_anchors": anchors,
        "_x": x,
        "_y": y,
        "_pc": pc,
        "_vx": vx,
        "_vy": vy,
        "_stride": stride,
        "_post": post,
    }


def audit_synthetic(
    mapping: Dict[str, Any], device: torch.device, model: Optional[nn.Module]
) -> Dict[str, Any]:
    pc = mapping["_pc"]
    vx, vy = mapping["_vx"], mapping["_vy"]
    stride = mapping["_stride"]
    anchors = mapping["_anchors"]
    xmin, ymin = pc[0], pc[1]
    ny, nx = mapping["splat_hw_ny_nx"]
    splat = GaussianBEVSplat(
        feature_dim=128,
        voxel_size=hypes_voxel(mapping),
        point_cloud_range=pc,
        n_sigma=3.0,
    ).to(device)
    rows = []
    fail = False
    det_cell = vx * stride
    for name, x, y in SYNTH_XY:
        ix, iy = expected_splat_idx(x, y, xmin, ymin, vx, vy)
        gs = make_one_gaussian(x, y, -1.0, 128, (0.25, 0.25, 0.25), device)
        bev = splat(gs)
        amp = bev[0].abs().sum(dim=0)
        iy_hat, ix_hat = _argmax2d(amp)
        det_r, det_c = iy // stride, ix // stride
        det_r_hat, det_c_hat = iy_hat // stride, ix_hat // stride
        det_r_hat = min(max(det_r_hat, 0), anchors.shape[0] - 1)
        det_c_hat = min(max(det_c_hat, 0), anchors.shape[1] - 1)
        ax, ay = float(anchors[det_r, det_c, 0, 0]), float(anchors[det_r, det_c, 0, 1])
        ax_hat = float(anchors[det_r_hat, det_c_hat, 0, 0])
        ay_hat = float(anchors[det_r_hat, det_c_hat, 0, 1])
        err_cell = max(abs(ix - ix_hat), abs(iy - iy_hat))
        err_m = math.hypot(ax_hat - x, ay_hat - y)
        rec = {
            "name": name,
            "world_xy": [x, y],
            "expected_splat_col_row_ix_iy": [ix, iy],
            "actual_splat_argmax_ix_iy": [ix_hat, iy_hat],
            "splat_cell_err": err_cell,
            "det_expected_col_row": [det_c, det_r],
            "det_from_argmax_col_row": [det_c_hat, det_r_hat],
            "anchor_xy_from_expected_cell": [ax, ay],
            "anchor_xy_from_argmax_cell": [ax_hat, ay_hat],
            "anchor_vs_world_m": err_m,
            "anchor_vs_world_cells": err_m / det_cell,
            "in_grid": 0 <= ix < nx and 0 <= iy < ny,
        }
        if err_cell > 1 or (err_m / det_cell) > 1.0:
            rec["FAIL"] = True
            fail = True
        if model is not None:
            rec.update(_backbone_peak(model, bev, anchors, stride))
        rows.append(rec)
    aniso = {}
    for axis in ("x", "y"):
        gs = make_aniso_gaussian(axis, 128, device, pc)
        bev = splat(gs)
        amp = bev[0].abs().sum(dim=0)
        mass = amp.sum().clamp_min(1e-8)
        ys = torch.arange(ny, device=device, dtype=torch.float32)
        xs = torch.arange(nx, device=device, dtype=torch.float32)
        cy = (amp.sum(1) * ys).sum() / mass
        cx = (amp.sum(0) * xs).sum() / mass
        var_y = (amp.sum(1) * (ys - cy) ** 2).sum() / mass
        var_x = (amp.sum(0) * (xs - cx) ** 2).sum() / mass
        aniso[axis] = {
            "var_col_x": float(var_x),
            "var_row_y": float(var_y),
            "elongated_along_expected_axis": bool(
                (var_x > var_y * 2) if axis == "x" else (var_y > var_x * 2)
            ),
        }
    cov_axis_ok = bool(aniso["x"]["elongated_along_expected_axis"] and aniso["y"]["elongated_along_expected_axis"])
    return {
        "status": "FAIL" if fail or not cov_axis_ok else "PASS",
        "fail_if_gt_1_cell": fail,
        "covariance_axis_ok": cov_axis_ok,
        "points": rows,
        "aniso": aniso,
        "det_cell_m": det_cell,
    }


def hypes_voxel(mapping: Dict[str, Any]) -> Tuple[float, float, float]:
    return (mapping["_vx"], mapping["_vy"], 0.125)


def _backbone_peak(
    model: nn.Module, bev: torch.Tensor, anchors: np.ndarray, stride: int
) -> Dict[str, Any]:
    data = {"spatial_features": bev}
    with torch.no_grad():
        data = model.backbone(data)
        fused = model.shrink_conv(data["spatial_features_2d"])
        obj = torch.sigmoid(model.obj_head(fused))[0].max(0).values
        r, c = _argmax2d(obj)
    r = min(max(r, 0), anchors.shape[0] - 1)
    c = min(max(c, 0), anchors.shape[1] - 1)
    return {
        "detection_map_hw": [int(obj.shape[0]), int(obj.shape[1])],
        "obj_argmax_row_col": [r, c],
        "obj_argmax_anchor_xy": [float(anchors[r, c, 0, 0]), float(anchors[r, c, 0, 1])],
        "obj_max": float(obj.max()),
    }


def _cat_gaussians(sets: Sequence[GaussianSet]) -> GaussianSet:
    device = sets[0].mean.device
    return GaussianSet(
        mean=torch.cat([g.mean for g in sets]),
        scale=torch.cat([g.scale for g in sets]),
        quaternion=torch.cat([g.quaternion for g in sets]),
        feature=torch.cat([g.feature for g in sets]),
        batch_index=torch.cat(
            [
                g.batch_index
                if g.batch_index is not None
                else torch.zeros(g.n_gaussians, dtype=torch.long, device=device)
                for g in sets
            ]
        ),
        agent="mixed",
        frame="ego",
    )


def _gt_centers_and_corners(batch: Dict[str, Any], post: VoxelPostprocessor):
    ego = batch["ego"]
    mask = ego["object_bbx_mask"][0] == 1
    centers = ego["object_bbx_center"][0][mask]
    corners, _, _ = post.generate_gt_bbx_airv2x(batch)
    return centers, corners


def point_in_corners_xy(px: float, py: float, corners: np.ndarray) -> bool:
    poly = common_utils.convert_format(corners[None, ...])[0]
    from shapely.geometry import Point

    return bool(poly.contains(Point(px, py)) or poly.touches(Point(px, py)))


def recall_at(
    pred_corners: Optional[torch.Tensor],
    gt_corners: torch.Tensor,
    thr: float,
) -> float:
    if gt_corners is None or int(gt_corners.shape[0]) == 0:
        return float("nan")
    n_gt = int(gt_corners.shape[0])
    if pred_corners is None or int(pred_corners.shape[0]) == 0:
        return 0.0
    gt_p = common_utils.convert_format(gt_corners.detach().cpu().numpy())
    pr_p = common_utils.convert_format(pred_corners.detach().cpu().numpy())
    hit = 0
    for g in gt_p:
        ious = common_utils.compute_iou(g, pr_p)
        if float(np.max(ious)) >= thr:
            hit += 1
    return hit / n_gt


def best_iou_each_gt(
    pred_corners: Optional[torch.Tensor], gt_corners: torch.Tensor
) -> np.ndarray:
    n_gt = int(gt_corners.shape[0])
    out = np.zeros((n_gt,), dtype=np.float32)
    if pred_corners is None or int(pred_corners.shape[0]) == 0:
        return out
    gt_p = common_utils.convert_format(gt_corners.detach().cpu().numpy())
    pr_p = common_utils.convert_format(pred_corners.detach().cpu().numpy())
    for i, g in enumerate(gt_p):
        out[i] = float(np.max(common_utils.compute_iou(g, pr_p)))
    return out


def run_stage3_bev(model: nn.Module, data: Dict[str, Any]):
    for agent in present_camera_agents(data):
        model._encode_agent(agent, data)
    gaussians = model.initializer.forward(data)
    init = {a: gs for a, gs in gaussians.items()}
    for agent, gs in gaussians.items():
        gaussians[agent] = model.intra_view(
            gs, data[agent]["f90"], data[agent][CAMERA_GEOMETRY_KEY], agent=agent
        )
    sources = {
        agent: {
            "features": data[agent]["f90"],
            "geometry": data[agent][CAMERA_GEOMETRY_KEY],
        }
        for agent in gaussians
    }
    g2 = model.cross_agent(gaussians, sources)
    fused_gs, bev = model.stage3(g2)
    return init, g2, fused_gs, bev


def audit_real_index0(
    model: nn.Module,
    batch: Dict[str, Any],
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    data = batch["ego"]
    post: VoxelPostprocessor = mapping["_post"]
    pc = mapping["_pc"]
    vx, vy = mapping["_vx"], mapping["_vy"]
    stride = mapping["_stride"]
    anchors = mapping["_anchors"]
    with torch.no_grad():
        init, g2, fused_gs, bev = run_stage3_bev(model, data)
        data["spatial_features"] = bev
        data = model.backbone(data)
        fused = model.shrink_conv(data["spatial_features_2d"])
        psm = model.cls_head(fused)
        rm = model.reg_head(fused)
        obj = model.obj_head(fused)
        output = {"psm": psm, "rm": rm, "obj": obj}
    ny, nx = int(bev.shape[2]), int(bev.shape[3])
    feat_norm = bev[0].float().norm(dim=0)
    centers, gt_corners = _gt_centers_and_corners(batch, post)
    centers_np = centers.detach().cpu().numpy()
    gt_c_np = gt_corners.detach().cpu().numpy()
    gt_cells = []
    gt_norms = []
    seed_hits = 0
    init_all = _cat_gaussians([gs for gs in init.values() if gs.n_gaussians > 0])
    init_xy = init_all.mean[:, :2].detach().cpu().numpy()
    for i in range(centers_np.shape[0]):
        x, y = float(centers_np[i, 0]), float(centers_np[i, 1])
        ix, iy = expected_splat_idx(x, y, pc[0], pc[1], vx, vy)
        iy = min(max(iy, 0), ny - 1)
        ix = min(max(ix, 0), nx - 1)
        nrm = float(feat_norm[iy, ix])
        det_r, det_c = iy // stride, ix // stride
        det_r = min(max(det_r, 0), fused.shape[2] - 1)
        det_c = min(max(det_c, 0), fused.shape[3] - 1)
        fused_n = float(fused[0, :, det_r, det_c].float().norm())
        inside = 0
        for j in range(init_xy.shape[0]):
            if point_in_corners_xy(float(init_xy[j, 0]), float(init_xy[j, 1]), gt_c_np[i]):
                inside += 1
        if inside > 0:
            seed_hits += 1
        gt_cells.append(
            {
                "gt_xy": [x, y],
                "splat_ix_iy": [ix, iy],
                "splat_feat_norm": nrm,
                "det_row_col": [det_r, det_c],
                "fused_feat_norm": fused_n,
                "n_init_seeds_inside": inside,
            }
        )
        gt_norms.append(nrm)
    occ = (feat_norm > 0).float()
    bg_mask = torch.ones_like(feat_norm, dtype=torch.bool)
    for rec in gt_cells:
        ix, iy = rec["splat_ix_iy"]
        r = 4
        bg_mask[
            max(0, iy - r) : min(ny, iy + r + 1),
            max(0, ix - r) : min(nx, ix + r + 1),
        ] = False
    bg_vals = feat_norm[bg_mask]
    gt_vals = feat_norm.new_tensor(gt_norms) if gt_norms else feat_norm.new_zeros((0,))
    return {
        "bev_hw": [ny, nx],
        "fused_hw": [int(fused.shape[2]), int(fused.shape[3])],
        "anchor_hw": [int(anchors.shape[0]), int(anchors.shape[1])],
        "hw_match_fused_anchor": [
            int(fused.shape[2]) == int(anchors.shape[0]),
            int(fused.shape[3]) == int(anchors.shape[1]),
        ],
        "n_gt": int(centers_np.shape[0]),
        "n_gt_with_init_seed": seed_hits,
        "frac_gt_with_init_seed": seed_hits / max(len(gt_cells), 1),
        "n_init_gaussians": int(init_all.n_gaussians),
        "n_pooled_gaussians": int(fused_gs.n_gaussians),
        "splat_occupied_frac": float(occ.mean()),
        "gt_cell_feat_norm": _dist_stats(np.array(gt_norms, dtype=np.float64)),
        "background_cell_feat_norm": _dist_stats(bg_vals.detach().cpu().numpy()),
        "gt_over_bg_mean_norm": float(
            (float(np.mean(gt_norms)) / max(float(bg_vals.mean()), 1e-8))
            if gt_norms
            else float("nan")
        ),
        "gt_cells_sample": gt_cells[:12],
        "output": output,
        "gt_corners": gt_corners,
        "centers": centers,
    }


def decode_all_boxes(rm: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    return VoxelPostprocessor.delta_to_boxes3d(rm, anchors)[0]


def corners_from_boxes(boxes: torch.Tensor, order: str) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 8, 3))
    return box_utils.boxes_to_corners_3d(boxes, order=order)


def project_corners(corners: torch.Tensor, tfm: torch.Tensor) -> torch.Tensor:
    if corners.numel() == 0:
        return corners
    return box_utils.project_box3d(corners, tfm)


def audit_postprocess(
    batch: Dict[str, Any],
    output: Dict[str, torch.Tensor],
    gt_corners: torch.Tensor,
    hypes: Dict[str, Any],
) -> Dict[str, Any]:
    ego = batch["ego"]
    post = VoxelPostprocessor(hypes["postprocess"], dataset="airv2x", train=False)
    order = hypes["postprocess"]["order"]
    obj_thresh = float(hypes["postprocess"]["target_args"]["obj_threshold"])
    score_thresh = float(hypes["postprocess"]["target_args"].get("score_threshold", 0.2))
    nms_thresh = float(hypes["postprocess"]["nms_thresh"])
    lidar_range = hypes["postprocess"]["anchor_args"]["cav_lidar_range"]
    z_min, z_max = float(lidar_range[2]), float(lidar_range[5])
    tfm = ego["transformation_matrix"].to(output["obj"].device)
    anchors = ego["anchor_box"].to(output["obj"].device)

    obj_preds = output["obj"].permute(0, 2, 3, 1).contiguous()
    objectness = torch.sigmoid(obj_preds).view(1, -1)
    psm = output["psm"]
    b, ac, h, w = psm.shape
    c = int(hypes["num_class"])
    a = ac // c
    psm_v = psm.view(b, c, a, h, w).permute(0, 3, 4, 2, 1).contiguous()
    class_scores, class_labels = torch.max(torch.sigmoid(psm_v)[..., 1:], dim=-1)
    class_labels = class_labels + 1
    mask = objectness > obj_thresh
    n_total = int(objectness.numel())
    n0 = int(mask.sum())
    boxes_all = decode_all_boxes(output["rm"], anchors)
    boxes_n0 = boxes_all[mask.view(-1)]
    scores_n0 = objectness.view(-1)[mask.view(-1)]
    n1 = int(boxes_n0.shape[0])
    corners_n0 = project_corners(corners_from_boxes(boxes_n0, order), tfm)

    keep_large = (
        box_utils.remove_large_pred_bbx(corners_n0, "airv2x")
        if n1
        else torch.zeros((0,), dtype=torch.bool, device=boxes_all.device)
    )
    n_drop_large = int((~keep_large).sum()) if n1 else 0
    boxes_l = boxes_n0[keep_large]
    corners_l = corners_n0[keep_large]
    scores_l = scores_n0[keep_large]
    n2 = int(boxes_l.shape[0])
    keep_z = (
        box_utils.remove_bbx_abnormal_z(corners_l, z_min=z_min, z_max=z_max)
        if n2
        else torch.zeros((0,), dtype=torch.bool, device=boxes_all.device)
    )
    n_drop_z = int((~keep_z).sum()) if n2 else 0
    boxes_z = boxes_l[keep_z]
    corners_z = corners_l[keep_z]
    scores_z = scores_l[keep_z]
    n3 = int(boxes_z.shape[0])
    if n3:
        keep_nms = box_utils.nms_rotated(corners_z, scores_z, nms_thresh)
        keep_nms = torch.as_tensor(keep_nms, device=boxes_z.device, dtype=torch.long)
        boxes_n = boxes_z[keep_nms]
        corners_n = corners_z[keep_nms]
        scores_n = scores_z[keep_nms]
    else:
        boxes_n = boxes_z
        corners_n = corners_z
        scores_n = scores_z
    n4 = int(boxes_n.shape[0])
    if n4:
        keep_r = box_utils.get_mask_for_boxes_within_range_torch(corners_n, lidar_range)
        corners_f = corners_n[keep_r]
        boxes_f = boxes_n[keep_r]
        scores_f = scores_n[keep_r]
    else:
        keep_r = torch.zeros((0,), dtype=torch.bool)
        corners_f = corners_n
        boxes_f = boxes_n
        scores_f = scores_n
    n5 = int(corners_f.shape[0])

    rm = output["rm"].permute(0, 2, 3, 1).contiguous().view(-1, 7)
    rm_np = rm.detach().cpu().numpy()
    boxes_all_np = boxes_all.detach().cpu().numpy()
    boxes_n0_np = boxes_n0.detach().cpu().numpy() if n1 else np.zeros((0, 7))
    names = ["dx", "dy", "dz", "dw", "dl", "dh", "dyaw"]
    # decode order is dx,dy,dz,dh,dw,dl,dyaw in hwl
    box_names = ["x", "y", "z", "h", "w", "l", "yaw"]

    stages = {
        "N0_threshold": corners_n0,
        "N1_decode": corners_n0,
        "N2_large_filter": corners_l,
        "N3_z_filter": corners_z,
        "N4_nms": corners_n,
        "N5_range": corners_f,
    }
    rec_map = {}
    for k, v in stages.items():
        rec_map[k] = {f"recall@{t}": recall_at(v, gt_corners, t) for t in (0.1, 0.3, 0.5)}

    best0 = best_iou_each_gt(corners_n0, gt_corners)
    best2 = best_iou_each_gt(corners_l, gt_corners)
    best3 = best_iou_each_gt(corners_z, gt_corners)
    best4 = best_iou_each_gt(corners_n, gt_corners)
    best5 = best_iou_each_gt(corners_f, gt_corners)
    gt_death = []
    for i in range(int(gt_corners.shape[0])):
        death = "survived"
        if best0[i] < 0.1:
            death = "threshold_or_no_candidate"
        elif best2[i] < 0.1 <= best0[i]:
            death = "large_box_filter"
        elif best3[i] < 0.1 <= best2[i]:
            death = "abnormal_z_filter"
        elif best4[i] < 0.1 <= best3[i]:
            death = "nms"
        elif best5[i] < 0.1 <= best4[i]:
            death = "range_filter"
        elif best5[i] < 0.3:
            death = "survived_but_low_iou"
        gt_death.append(
            {
                "gt_index": i,
                "has_pre_nms_candidate_iou01": bool(best3[i] >= 0.1),
                "best_iou_N0": float(best0[i]),
                "best_iou_N2": float(best2[i]),
                "best_iou_N3": float(best3[i]),
                "best_iou_N4": float(best4[i]),
                "best_iou_N5": float(best5[i]),
                "death": death,
            }
        )
    death_counts = defaultdict(int)
    for g in gt_death:
        death_counts[g["death"]] += 1

    if rm_np.size:
        size_m = np.exp(np.clip(rm_np[:, 3:6], -20.0, 20.0)) * np.array([1.56, 1.6, 3.9])
        huge = int((size_m.max(axis=1) > 20.0).sum())
    else:
        huge = 0
    n_gt = int(gt_corners.shape[0])
    verdict_votes = []
    if n0 <= 2 and death_counts.get("threshold_or_no_candidate", 0) >= 0.8 * max(n_gt, 1):
        verdict_votes.append("1_threshold_no_candidate")
    if n1 and float(np.mean(best0)) < 0.1 and n0 > 10:
        verdict_votes.append("2_regression_decode")
    if n_drop_large + n_drop_z > 0 and n4 < n1 * 0.2:
        verdict_votes.append("3_filter")
    if n3 > 10 and n4 <= 2:
        verdict_votes.append("4_nms")
    if n4 > n5 * 3 and n5 <= 2:
        verdict_votes.append("5_range")
    if len(verdict_votes) > 1:
        primary = "6_multi_factor"
    elif verdict_votes:
        primary = verdict_votes[0]
    else:
        primary = "inspect_counts"

    return {
        "call_chain": "dataset.post_process -> VoxelPostprocessor.post_process_airv2x",
        "score_source": "objectness=sigmoid(obj); class_scores computed but NOT used in mask",
        "mask": "objectness > obj_threshold (0.2); score_threshold is logged only",
        "nms_top": 1000,
        "gt_count": n_gt,
        "n_total_anchors": n_total,
        "N0_threshold": n0,
        "N1_decode": n1,
        "N2_large_box_filter": n2,
        "N3_abnormal_z_filter": n3,
        "N4_nms": n4,
        "N5_range": n5,
        "N_final": n5,
        "dropped_large": n_drop_large,
        "dropped_z": n_drop_z,
        "objectness": {
            "mean": float(objectness.mean()),
            "max": float(objectness.max()),
            "min": float(objectness.min()),
            "frac_gt_obj_thresh": float((objectness > obj_thresh).float().mean()),
            "obj_thresh": obj_thresh,
        },
        "class_scores_unused_in_mask": {
            "mean": float(class_scores.mean()),
            "max": float(class_scores.max()),
            "n_pass_score_thresh": int((class_scores > score_thresh).sum()),
        },
        "rm_delta": {names[i]: _dist_stats(rm_np[:, i]) for i in range(7)},
        "rm_size_exp": {
            "exp_dh": _dist_stats(np.exp(np.clip(rm_np[:, 3], -20, 20))),
            "exp_dw": _dist_stats(np.exp(np.clip(rm_np[:, 4], -20, 20))),
            "exp_dl": _dist_stats(np.exp(np.clip(rm_np[:, 5], -20, 20))),
            "n_decoded_size_gt_20m_if_applied_to_all_anchors": int(huge),
        },
        "decoded_all_boxes": {box_names[i]: _dist_stats(boxes_all_np[:, i]) for i in range(7)},
        "decoded_N0_boxes": {box_names[i]: _dist_stats(boxes_n0_np[:, i]) for i in range(7)},
        "recall_by_stage": rec_map,
        "gt_death": gt_death,
        "death_counts": dict(death_counts),
        "verdict": primary,
        "verdict_votes": verdict_votes,
        "final_scores": _dist_stats(scores_f.detach().cpu().numpy()) if n5 else _dist_stats(np.array([])),
    }


def split_det_losses(criterion, output, target) -> Dict[str, torch.Tensor]:
    """Mirror PointPillarLossMultiClass.forward without extra terms."""
    rm = output["rm"]
    psm = output["psm"]
    obj = output["obj"]
    targets = target["targets"]
    cls_preds = psm.permute(0, 2, 3, 1).contiguous()
    obj_preds = obj.permute(0, 2, 3, 1).contiguous()
    box_cls_labels = target["pos_equal_one"].view(psm.shape[0], -1).contiguous()
    positives = box_cls_labels > 0
    negatives = box_cls_labels == 0
    cls_weights = (negatives * 1.0 + positives * 1.0).float()
    reg_weights = positives.float()
    pos_mask = target["pos_equal_one"]
    pos_normalizer = positives.sum(1, keepdim=True).float()
    reg_weights = reg_weights / torch.clamp(pos_normalizer, min=1.0)
    cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)
    cls_targets = target["class_ids"]
    one_hot = torch.zeros(
        *list(cls_targets.shape),
        criterion.cls_num,
        dtype=cls_preds.dtype,
        device=cls_targets.device,
    )
    one_hot.scatter_(-1, cls_targets.unsqueeze(-1).long(), 1.0)
    cls_labels = one_hot.view(
        cls_targets.shape[0], cls_targets.shape[1], cls_targets.shape[2], -1
    )
    cls_loss_src = criterion.cls_loss_func(cls_preds, cls_labels, weights=cls_weights)
    conf_loss = cls_loss_src.sum() / psm.shape[0] * criterion.cls_weight
    rm_p = rm.permute(0, 2, 3, 1).contiguous().view(rm.size(0), -1, 7)
    tgt = targets.view(targets.size(0), -1, 7)
    box_preds_sin, reg_targets_sin = criterion.add_sin_difference(rm_p, tgt)
    loc_loss_src = criterion.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)
    reg_loss = loc_loss_src.sum() / rm.shape[0] * criterion.reg_coe
    obj_preds_expanded = obj_preds.unsqueeze(-1)
    pos_mask_expanded = pos_mask.unsqueeze(-1).float()
    obj_loss_src = criterion.obj_loss_func(
        obj_preds_expanded, pos_mask_expanded, torch.ones_like(pos_mask_expanded)
    )
    obj_loss = obj_loss_src.mean() * criterion.obj_weight
    total = reg_loss + conf_loss + obj_loss
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    return {
        "cls": conf_loss,
        "reg": reg_loss,
        "obj": obj_loss,
        "total": total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_frac": n_pos / max(n_pos + n_neg, 1),
    }


def global_grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        if not torch.isfinite(g).all():
            return float("nan")
        total += float(g.pow(2).sum())
    return math.sqrt(total)


def module_grad_norm(module: Optional[nn.Module]) -> float:
    if module is None:
        return 0.0
    return global_grad_norm(module.parameters())


def named_grad_norm(model: nn.Module, substrings: Sequence[str]) -> float:
    params = []
    for n, p in model.named_parameters():
        if any(s in n for s in substrings):
            params.append(p)
    return global_grad_norm(params)


def collect_module_grads(model: nn.Module) -> Dict[str, float]:
    s3 = model.stage3
    self0 = s3.self_blocks[0] if len(s3.self_blocks) else None
    return {
        "frontend": named_grad_norm(model, ("frontend", "highres", "heatmap_heads", "depth_heads")),
        "Stage1_intra_view": module_grad_norm(model.intra_view),
        "Stage2_cross_agent": module_grad_norm(model.cross_agent),
        "Stage3_split_mlp": module_grad_norm(s3.split_mlp),
        "Stage3_self_q": global_grad_norm(
            [p for n, p in model.named_parameters() if "stage3.self_blocks" in n and "q_proj" in n]
        ),
        "Stage3_self_k": global_grad_norm(
            [p for n, p in model.named_parameters() if "stage3.self_blocks" in n and "k_proj" in n]
        ),
        "Stage3_self_v": global_grad_norm(
            [p for n, p in model.named_parameters() if "stage3.self_blocks" in n and "v_proj" in n]
        ),
        "Stage3_self_out": global_grad_norm(
            [p for n, p in model.named_parameters() if "stage3.self_blocks" in n and "out_proj" in n]
        ),
        "Stage3_self_ffn": global_grad_norm(
            [p for n, p in model.named_parameters() if "stage3.self_blocks" in n and "ffn" in n]
        ),
        "Stage3_adapter": module_grad_norm(s3.adapter),
        "fusion_mlp": module_grad_norm(s3.fusion_mlp),
        "fusion_gamma": global_grad_norm([s3.fusion_gamma]),
        "scale_heads": global_grad_norm(
            [p for n, p in model.named_parameters() if "scale_head" in n]
        ),
        "BEV_backbone": global_grad_norm(
            list(model.backbone.parameters()) + list(model.shrink_conv.parameters())
        ),
        "psm_head": module_grad_norm(model.cls_head),
        "rm_head": module_grad_norm(model.reg_head),
        "obj_head": module_grad_norm(model.obj_head),
    }


def q_proj_cos(model: nn.Module) -> Dict[str, float]:
    w = model.stage3.self_blocks[0].q_proj.weight
    g = w.grad
    out = {
        "weight_rms": float(w.detach().float().pow(2).mean().sqrt()),
        "weight_fro": float(w.detach().float().norm()),
    }
    if g is None:
        out["cos"] = float("nan")
        out["grad_rms"] = 0.0
        return out
    wf = w.detach().float().reshape(-1)
    gf = g.detach().float().reshape(-1)
    denom = float(wf.norm() * gf.norm())
    cos = float(torch.dot(wf, gf) / denom) if denom > 0 else float("nan")
    out["cos_weight_grad"] = cos
    out["grad_rms"] = float(gf.pow(2).mean().sqrt())
    out["shrinks_weight"] = bool(cos > 0)
    return out


def fusion_gamma_stats(model: nn.Module) -> Dict[str, float]:
    g = model.stage3.fusion_gamma
    rec = {
        "gamma_mean": float(g.detach().mean()),
        "gamma_rms": float(g.detach().float().pow(2).mean().sqrt()),
        "gamma_min": float(g.detach().min()),
        "gamma_max": float(g.detach().max()),
    }
    if g.grad is None:
        rec["dL_dgamma_mean"] = 0.0
        rec["pushes_to_zero"] = False
        return rec
    dg = g.grad.detach().float()
    rec["dL_dgamma_mean"] = float(dg.mean())
    rec["dL_dgamma_rms"] = float(dg.pow(2).mean().sqrt())
    # SGD: gamma <- gamma - lr * dL/dgamma. If gamma>0 and dL/dgamma>0, gamma shrinks.
    rec["pushes_to_zero"] = bool(float(g.detach().mean()) * float(dg.mean()) > 0)
    return rec


def audit_grads(
    model: nn.Module,
    criterion,
    loader,
    device: torch.device,
    n_batches: int,
) -> Dict[str, Any]:
    model.train()
    recs = []
    n_done = 0
    for batch in loader:
        if batch is None:
            continue
        batch = train_utils.to_device(batch, device)
        batch["ego"]["epoch"] = 1
        model.zero_grad(set_to_none=True)
        output = model(batch["ego"])
        target = batch["ego"]["label_dict"]
        comps = split_det_losses(criterion, output, target)
        row: Dict[str, Any] = {
            "n_pos": comps["n_pos"],
            "n_neg": comps["n_neg"],
            "pos_frac": comps["pos_frac"],
            "loss_cls": float(comps["cls"].detach()),
            "loss_reg": float(comps["reg"].detach()),
            "loss_obj": float(comps["obj"].detach()),
            "loss_total": float(comps["total"].detach()),
        }
        by_comp = {}
        for name in ("total", "cls", "reg", "obj"):
            model.zero_grad(set_to_none=True)
            comps[name].backward(retain_graph=True)
            gnorm = global_grad_norm([p for p in model.parameters() if p.requires_grad])
            clip_factor = min(1.0, 1.0 / max(gnorm, 1e-12)) if math.isfinite(gnorm) else 0.0
            entry = {
                "global_grad_norm": gnorm,
                "clip_factor_max_norm1": clip_factor,
                "would_clip": bool(math.isfinite(gnorm) and gnorm > 1.0),
                "modules": collect_module_grads(model),
            }
            if name in ("total", "cls", "obj"):
                entry["fusion_gamma"] = fusion_gamma_stats(model)
                entry["q_proj"] = q_proj_cos(model)
            by_comp[name] = entry
        row["by_component"] = by_comp
        recs.append(row)
        n_done += 1
        print(
            f"[grad] {n_done}/{n_batches} totalG={by_comp['total']['global_grad_norm']:.3f} "
            f"clsL={row['loss_cls']:.3f} regL={row['loss_reg']:.3f} objL={row['loss_obj']:.3f} "
            f"pos={row['n_pos']}"
        )
        del output, comps
        torch.cuda.empty_cache()
        if n_done >= n_batches:
            break

    def mean_key(path: Sequence[str]) -> float:
        vals = []
        for r in recs:
            cur: Any = r
            ok = True
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    ok = False
                    break
                cur = cur[p]
            if ok and isinstance(cur, (int, float)) and math.isfinite(float(cur)):
                vals.append(float(cur))
        return float(np.mean(vals)) if vals else float("nan")

    g_tot = [r["by_component"]["total"]["global_grad_norm"] for r in recs]
    g_cls = [r["by_component"]["cls"]["global_grad_norm"] for r in recs]
    g_reg = [r["by_component"]["reg"]["global_grad_norm"] for r in recs]
    g_obj = [r["by_component"]["obj"]["global_grad_norm"] for r in recs]
    summary = {
        "n_batches": n_done,
        "loss_components_in_train_py": ["cls/conf", "reg", "obj"],
        "absent_from_criterion": ["depth", "heatmap", "iou (weight=0)", "recall (weight=0)"],
        "cls_reduction": "focal BCE, weights=(pos+neg)/n_pos, then sum/B, then /B again in forward",
        "reg_reduction": "smooth-L1 on positives only, weights=1/n_pos, sum/B * 2.0",
        "obj_reduction": "focal BCE mean over ALL anchors * obj_weight=1.0",
        "weights": {"cls_weight": 1.0, "reg": 2.0, "obj_weight": 1.0},
        "mean_loss": {
            "cls": mean_key(("loss_cls",)),
            "reg": mean_key(("loss_reg",)),
            "obj": mean_key(("loss_obj",)),
            "total": mean_key(("loss_total",)),
        },
        "mean_global_grad": {
            "total": float(np.mean(g_tot)),
            "cls": float(np.mean(g_cls)),
            "reg": float(np.mean(g_reg)),
            "obj": float(np.mean(g_obj)),
        },
        "clip": {
            "max_norm": 1.0,
            "mean_unclipped_G": float(np.mean(g_tot)),
            "median_unclipped_G": float(np.median(g_tot)),
            "max_unclipped_G": float(np.max(g_tot)),
            "min_unclipped_G": float(np.min(g_tot)),
            "mean_clip_factor": float(np.mean([min(1.0, 1.0 / max(g, 1e-12)) for g in g_tot])),
            "frac_G_gt_1": float(np.mean([g > 1.0 for g in g_tot])),
            "frac_G_gt_10": float(np.mean([g > 10.0 for g in g_tot])),
            "mild_overshoot_1_to_2": float(np.mean([1.0 < g <= 2.0 for g in g_tot])),
        },
        "mean_module_grad_total": {
            k: mean_key(("by_component", "total", "modules", k))
            for k in recs[0]["by_component"]["total"]["modules"]
        },
        "mean_fusion_gamma_total": {
            k: mean_key(("by_component", "total", "fusion_gamma", k))
            for k in recs[0]["by_component"]["total"]["fusion_gamma"]
            if k != "pushes_to_zero"
        },
        "frac_gamma_pushes_to_zero": float(
            np.mean(
                [
                    int(r["by_component"]["total"]["fusion_gamma"]["pushes_to_zero"])
                    for r in recs
                ]
            )
        ),
        "mean_q_proj_total": {
            k: mean_key(("by_component", "total", "q_proj", k))
            for k in recs[0]["by_component"]["total"]["q_proj"]
            if k != "shrinks_weight"
        },
        "frac_q_proj_cos_positive": float(
            np.mean(
                [
                    int(r["by_component"]["total"]["q_proj"].get("shrinks_weight", False))
                    for r in recs
                ]
            )
        ),
        "batches": recs,
    }
    return summary


def build_verdict(
    mapping: Dict[str, Any],
    synth: Dict[str, Any],
    real: Dict[str, Any],
    post: Dict[str, Any],
    grads: Dict[str, Any],
) -> Dict[str, Any]:
    prio = []
    align_fail = synth["status"] != "PASS" or (not all(real["hw_match_fused_anchor"]))
    prio.append(
        {
            "rank": 1,
            "name": "spatial_alignment_bug",
            "status": "FAIL" if align_fail else "PASS",
            "evidence": {
                "synth": synth["status"],
                "fused_vs_anchor": real["hw_match_fused_anchor"],
                "gt_over_bg": real.get("gt_over_bg_mean_norm"),
            },
        }
    )
    prio.append(
        {
            "rank": 2,
            "name": "postprocess_decode_nms",
            "status": "FAIL" if post["N_final"] <= 2 and post["gt_count"] > 5 else "PASS",
            "verdict": post["verdict"],
            "N": {
                "GT": post["gt_count"],
                "N0": post["N0_threshold"],
                "N4": post["N4_nms"],
                "N_final": post["N_final"],
            },
            "death_counts": post["death_counts"],
        }
    )
    gmean = grads["clip"]["mean_unclipped_G"]
    cls_dom = grads["mean_global_grad"]["cls"] >= max(
        grads["mean_global_grad"]["reg"], grads["mean_global_grad"]["obj"]
    )
    prio.append(
        {
            "rank": 3,
            "name": "loss_gradient_imbalance",
            "status": "FAIL" if gmean > 10 or cls_dom else "WATCH",
            "mean_unclipped_G": gmean,
            "cls_dominates_global_grad": cls_dom,
            "mean_loss": grads["mean_loss"],
        }
    )
    qcos = grads.get("mean_q_proj_total", {})
    prio.append(
        {
            "rank": 4,
            "name": "stage3_optimization_shortcut",
            "status": "FAIL",
            "fusion_gamma_mean": grads["mean_fusion_gamma_total"].get("gamma_mean"),
            "frac_gamma_pushes_to_zero": grads["frac_gamma_pushes_to_zero"],
            "q_proj_weight_rms": qcos.get("weight_rms"),
            "q_proj_cos": qcos.get("cos_weight_grad"),
        }
    )
    return {"priority": prio}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--ckpt", type=str, default=CKPT_NAME)
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--n_grad_batches", type=int, default=20)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    hypes = _load_hypes(model_dir)
    mapping = audit_static_mapping(hypes)
    print("=== A static ===")
    print("splat", mapping["splat_hw_ny_nx"], "anchor", mapping["anchor_grid_hw"], "stride", mapping["feature_stride"])

    print("building model")
    model = train_utils.create_model(hypes).to(device)
    _load_weights(model, model_dir / args.ckpt, device)
    model.eval()

    synth = audit_synthetic(mapping, device, model)
    print("=== A synthetic ===", synth["status"])
    for p in synth["points"]:
        print(p["name"], "splat_err", p["splat_cell_err"], "anchor_cells", f"{p['anchor_vs_world_cells']:.3f}")

    ds_hypes = dict(hypes)
    ds_hypes["validate_dir"] = hypes["test_dir"]
    test_set = build_dataset(ds_hypes, visualize=False, train=False)
    batch = test_set.collate_batch_test([test_set[args.index]])
    batch = train_utils.to_device(batch, device)
    batch["ego"]["epoch"] = 0
    mapping["_post"] = test_set.post_processor
    print("=== A real index", args.index, "===")
    real = audit_real_index0(model, batch, mapping)
    print(
        "GT", real["n_gt"], "seeded", real["n_gt_with_init_seed"],
        "gt/bg", real["gt_over_bg_mean_norm"], "fused_hw", real["fused_hw"]
    )

    print("=== B postprocess ===")
    post = audit_postprocess(batch, real.pop("output"), real.pop("gt_corners"), hypes)
    print("N", {k: post[k] for k in (
        "gt_count", "N0_threshold", "N1_decode", "N2_large_box_filter",
        "N3_abnormal_z_filter", "N4_nms", "N5_range", "N_final"
    )})
    print("death", post["death_counts"], "verdict", post["verdict"])

    print("=== C grads ===")
    train_set = build_dataset(hypes, visualize=False, train=True)
    loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=train_set.collate_batch_train,
        drop_last=True,
    )
    criterion = train_utils.create_loss(hypes)
    grads = audit_grads(model, criterion, loader, device, int(args.n_grad_batches))

    for k in ("_anchors", "_x", "_y", "_pc", "_vx", "_vy", "_stride", "_post"):
        mapping.pop(k, None)
    real.pop("centers", None)
    report = {
        "ckpt": str(model_dir / args.ckpt),
        "branch": "0911version",
        "A_static": mapping,
        "A_synthetic": synth,
        "A_real_index0": real,
        "B_postprocess": post,
        "C_loss_grad": {k: v for k, v in grads.items() if k != "batches"},
        "C_batches": grads["batches"],
        "verdict": build_verdict(mapping, synth, real, post, grads),
    }
    out = Path(args.out) if args.out else model_dir / "low_recall_joint_audit_epoch2.json"
    out.write_text(json.dumps(_jsonable(report), indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()

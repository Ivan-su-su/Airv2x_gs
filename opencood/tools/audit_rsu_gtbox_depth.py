# -*- coding: utf-8 -*-
"""RSU depth audit on a fixed GT-box mask (epochs compared on the same pixels).

Projects official 3D boxes into RSU cameras, downsamples with tau=1 4x4
occupancy, and scores depth only on ``box_mask & in-range z``. Heatmap FG is
logged for contrast but does not define the mask.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.gaussian_modules_0822.heatmap.box_support import (
    VALID_BOX_CLASS_IDS,
    project_box_to_image,
    rasterize_convex_polygon,
)
from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    PRIMARY_OBJECTNESS_THRESHOLD,
)
from opencood.models.gaussian_modules_0822.heatmap.target import binary_objectness_target
from opencood.models.gaussian_modules_0822.lss.target import (
    depth_valid_mask,
    extract_camera_z_gt,
)
from opencood.tools import train_utils
from opencood.tools.eval_gaussian_p1 import (
    denormalize_rgb,
    load_epoch_checkpoint,
)
from opencood.tools.train import setup_dataloader
from opencood.tools.train_gaussian_p1 import _unwrap_model
from opencood.utils.box_utils import boxes_to_corners_3d

_DEFAULT_AGENT_ORDER = ("vehicle", "rsu", "drone")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    """CLI for the RSU GT-box depth audit."""
    parser = argparse.ArgumentParser(description="RSU GT-box mask depth audit")
    parser.add_argument("--hypes_yaml", "-y", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--worker", type=int, default=2)
    parser.add_argument("--sample_n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fg_tau", type=float, default=PRIMARY_OBJECTNESS_THRESHOLD)
    return parser.parse_args()


def _as_numpy(value: Any) -> np.ndarray:
    """Detach tensors to CPU numpy."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _record_len(ego: Mapping[str, Any], agent_type: str, batch_index: int) -> int:
    """Per-sample CAV count."""
    agent = ego.get(agent_type)
    if not isinstance(agent, Mapping):
        return 0
    record_len = agent.get("record_len")
    if record_len is None:
        return 0
    value = record_len[batch_index]
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def paint_rsu_box_maps(ego: Mapping[str, Any]) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Rasterize official GT boxes onto stacked RSU views at image resolution."""
    rsu = ego.get("rsu")
    if not isinstance(rsu, Mapping):
        raise KeyError("ego['rsu'] is required")
    cam_inputs = rsu.get("batch_merged_cam_inputs")
    if not isinstance(cam_inputs, Mapping):
        raise KeyError("rsu batch_merged_cam_inputs is required")
    imgs = cam_inputs.get("imgs")
    if not torch.is_tensor(imgs):
        raise KeyError("rsu imgs is required")
    if imgs.dim() == 5:
        n_rsu, n_views, _, height, width = imgs.shape
        n_flat = int(n_rsu * n_views)
    elif imgs.dim() == 4:
        n_flat, _, height, width = imgs.shape
        n_views = 1
        n_rsu = n_flat
    else:
        raise ValueError(f"rsu imgs must be 4D or 5D, got {tuple(imgs.shape)}")

    maps = np.zeros((n_flat, int(height), int(width)), dtype=np.uint8)
    stats = {"n_gt_boxes": 0, "n_valid_projections": 0, "n_source_fg_pixels": 0}
    if n_flat == 0:
        return torch.from_numpy(maps).long(), stats

    boxes_all = ego.get("object_bbx_center")
    mask_all = ego.get("object_bbx_mask")
    if not torch.is_tensor(boxes_all) or not torch.is_tensor(mask_all):
        raise KeyError("object_bbx_center / object_bbx_mask are required")
    if boxes_all.dim() == 2:
        boxes_all = boxes_all.unsqueeze(0)
        mask_all = mask_all.unsqueeze(0)
    batch_size = int(boxes_all.shape[0])
    pairwise_all = ego.get("img_pairwise_t_matrix_collab")
    if not torch.is_tensor(pairwise_all):
        raise KeyError("img_pairwise_t_matrix_collab is required")
    if pairwise_all.dim() == 4:
        pairwise_all = pairwise_all.unsqueeze(0)
    agent_order: Sequence[str] = tuple(ego.get("agent_order") or _DEFAULT_AGENT_ORDER)
    class_ids = ego.get("class_ids")
    image_hw = (int(height), int(width))
    intrinsics = _as_numpy(cam_inputs["intrinsics"])
    extrinsics = _as_numpy(cam_inputs["extrinsics"])
    post_rots = _as_numpy(cam_inputs["post_rots"])
    post_trans = _as_numpy(cam_inputs["post_trans"])

    rsu_tensor_idx = 0
    n_valid = 0
    n_gt = 0
    for batch_index in range(batch_size):
        mask = _as_numpy(mask_all[batch_index]).reshape(-1) == 1
        boxes = _as_numpy(boxes_all[batch_index][mask]).astype(np.float64)
        n_box = int(boxes.shape[0])
        n_gt += n_box
        if class_ids is None:
            class_row = np.ones((n_box,), dtype=np.int64)
        else:
            raw = class_ids[batch_index] if not torch.is_tensor(class_ids) else class_ids[batch_index]
            ids = _as_numpy(raw).reshape(-1).astype(np.int64)
            class_row = ids[:n_box] if ids.size >= n_box else np.ones((n_box,), dtype=np.int64)
        n_this = _record_len(ego, "rsu", batch_index)
        if n_this == 0:
            continue
        rsu_offset = 0
        for name in agent_order:
            if name == "rsu":
                break
            rsu_offset += _record_len(ego, name, batch_index)
        pairwise = _as_numpy(pairwise_all[batch_index])
        keep = np.ones((n_box,), dtype=bool)
        for box_idx in range(n_box):
            class_id = int(class_row[box_idx]) if box_idx < class_row.size else 1
            if class_id not in VALID_BOX_CLASS_IDS or not np.isfinite(boxes[box_idx]).all():
                keep[box_idx] = False
        boxes = boxes[keep]
        n_box = int(boxes.shape[0])
        if n_box == 0:
            rsu_tensor_idx += n_this
            continue
        corners_ego = boxes_to_corners_3d(boxes, "hwl")
        if torch.is_tensor(corners_ego):
            corners_ego = corners_ego.numpy()
        for local_idx in range(n_this):
            t_cav2ego = np.asarray(pairwise[rsu_offset + local_idx, 0], dtype=np.float64)
            if t_cav2ego.shape != (4, 4) or abs(float(np.linalg.det(t_cav2ego))) < 1e-8:
                t_cav2ego = np.eye(4, dtype=np.float64)
            t_ego2cav = np.linalg.inv(t_cav2ego)
            ones = np.ones((n_box * 8, 1), dtype=np.float64)
            xyz_h = np.concatenate(
                [corners_ego.reshape(-1, 3).astype(np.float64), ones], axis=1
            ).T
            xyz_cav = (t_ego2cav @ xyz_h).T[:, :3].reshape(n_box, 8, 3)
            for view_idx in range(int(n_views)):
                flat_idx = rsu_tensor_idx * int(n_views) + view_idx
                view_mask = np.zeros((int(height), int(width)), dtype=np.uint8)
                k = _slice_cam(intrinsics, rsu_tensor_idx, view_idx, n_views)
                ext = _slice_cam(extrinsics, rsu_tensor_idx, view_idx, n_views)
                prot = _slice_cam(post_rots, rsu_tensor_idx, view_idx, n_views)
                ptra = _slice_cam(post_trans, rsu_tensor_idx, view_idx, n_views)
                ptra = np.asarray(ptra).reshape(-1)
                # Stored extrinsics are camera-to-lidar after ``ue4_to_lss``.
                ext_cam_to_lidar = np.asarray(ext, dtype=np.float64)
                for box_idx in range(n_box):
                    projected = project_box_to_image(
                        xyz_cav[box_idx], k, ext_cam_to_lidar, prot, ptra, image_hw
                    )
                    if projected is None:
                        continue
                    pts, _z = projected
                    poly = rasterize_convex_polygon(pts, image_hw[0], image_hw[1])
                    if not bool(poly.any()):
                        continue
                    view_mask[poly] = 1
                    n_valid += 1
                maps[flat_idx] = view_mask
            rsu_tensor_idx += 1
    stats["n_gt_boxes"] = int(n_gt)
    stats["n_valid_projections"] = int(n_valid)
    stats["n_source_fg_pixels"] = int(maps.sum())
    return torch.from_numpy(maps).long(), stats


def _slice_cam(arr: np.ndarray, cav_idx: int, view_idx: int, n_views: int) -> np.ndarray:
    """Index stacked camera matrices ``[N,V,...]`` or ``[N,...]``."""
    if arr.ndim >= 3 and arr.shape[0] > cav_idx and arr.shape[1] == n_views:
        return np.asarray(arr[cav_idx, view_idx])
    flat = cav_idx * n_views + view_idx
    return np.asarray(arr[flat])


def save_box_depth_panel(
    out_path: Path,
    rgb: np.ndarray,
    box_mask: np.ndarray,
    gt_z: np.ndarray,
    pred_z: np.ndarray,
    d_min: float,
    d_max: float,
    title: str,
) -> None:
    """RGB | GT-box mask | GT z | pred z | abs error, all on the box mask."""
    vmax = max(d_max, 1.0)
    gt_show = np.where(box_mask, gt_z, np.nan)
    pred_show = np.where(box_mask, pred_z, np.nan)
    err_show = np.where(box_mask, np.abs(pred_z - gt_z), np.nan)
    fig, axes = plt.subplots(1, 5, figsize=(18.0, 3.4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[1].imshow(box_mask.astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("GT-box mask")
    im1 = axes[2].imshow(gt_show, cmap="turbo", vmin=d_min, vmax=vmax)
    axes[2].set_title("GT z on box")
    fig.colorbar(im1, ax=axes[2], fraction=0.046)
    im2 = axes[3].imshow(pred_show, cmap="turbo", vmin=d_min, vmax=vmax)
    axes[3].set_title("Pred z on box")
    fig.colorbar(im2, ax=axes[3], fraction=0.046)
    im3 = axes[4].imshow(err_show, cmap="magma", vmin=0.0, vmax=min(20.0, vmax))
    axes[4].set_title("|pred-GT| box (m)")
    fig.colorbar(im3, ax=axes[4], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _add_stats(
    sums: Dict[str, float],
    counts: Dict[str, float],
    prefix: str,
    pred_z: torch.Tensor,
    gt_z: torch.Tensor,
    mask: torch.Tensor,
) -> int:
    """Accumulate MAE/MSE/AbsRel/delta1 on ``mask``."""
    n = int(mask.sum().item())
    if n <= 0:
        return 0
    diff = pred_z[mask] - gt_z[mask]
    abs_err = diff.abs()
    rel = abs_err / gt_z[mask].clamp_min(1e-3)
    ratio = torch.maximum(
        pred_z[mask] / gt_z[mask].clamp_min(1e-3),
        gt_z[mask] / pred_z[mask].clamp_min(1e-3),
    )
    sums[f"{prefix}/mae"] += float(abs_err.sum().item())
    sums[f"{prefix}/mse"] += float((diff * diff).sum().item())
    sums[f"{prefix}/absrel"] += float(rel.sum().item())
    sums[f"{prefix}/delta1"] += float((ratio < 1.25).float().sum().item())
    counts[prefix] += n
    return n


def _pack(sums: Dict[str, float], counts: Dict[str, float], prefix: str) -> Dict[str, Any]:
    """Mean metrics for one mask prefix."""
    n = counts.get(prefix, 0.0)
    if n <= 0:
        return {"n_pixels": 0}
    return {
        "n_pixels": int(n),
        "mae_m": round(sums[f"{prefix}/mae"] / n, 4),
        "rmse_m": round((sums[f"{prefix}/mse"] / n) ** 0.5, 4),
        "absrel": round(sums[f"{prefix}/absrel"] / n, 4),
        "delta1": round(sums[f"{prefix}/delta1"] / n, 4),
    }


def main() -> None:
    """Run the RSU GT-box depth audit for one checkpoint."""
    opt = parse_args()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    dummy = Namespace(
        distributed=False,
        worker=opt.worker,
        gpu_id=opt.gpu_id,
        amp=False,
        tag="audit",
        model_dir="",
        hypes_yaml=opt.hypes_yaml,
    )
    print("Building val dataset...")
    dataset = build_dataset(hypes, visualize=False, train=False)
    n_all = len(dataset)
    n_keep = min(int(opt.sample_n), n_all)
    rng = np.random.RandomState(int(opt.seed))
    sample_ids = rng.choice(n_all, size=n_keep, replace=False).astype(int).tolist()
    sample_ids.sort()
    collate_fn = dataset.collate_batch_train
    dataset = torch.utils.data.Subset(dataset, sample_ids)
    dataset.collate_batch_train = collate_fn  # type: ignore[attr-defined]
    print(f"sampled {n_keep}/{n_all} seed={opt.seed} ids={sample_ids}")
    data_loader = setup_dataloader(dataset, hypes, dummy, is_train=False)

    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = train_utils.create_model(hypes)
    load_epoch_checkpoint(model, opt.model_dir, opt.epoch)
    model.to(device)
    model.eval()
    core = _unwrap_model(model)
    camencode = core.frontend.encoders["rsu"]
    d_min = float(camencode.d_min)
    d_max = float(camencode.d_max)
    fg_tau = float(opt.fg_tau)

    out_dir = (
        Path(opt.model_dir)
        / "visualization"
        / f"audit_rsu_gtbox_e{opt.epoch}_n{n_keep}s{opt.seed}"
    )
    vis_dir = out_dir / "panels"
    vis_dir.mkdir(parents=True, exist_ok=True)

    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, float] = defaultdict(float)
    overlap_inter = 0.0
    overlap_union = 0.0
    n_batch = 0
    n_skip_no_rsu = 0
    n_skip_empty_box = 0
    per_frame: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch_i, batch_data in enumerate(
            tqdm(data_loader, desc=f"rsu-box epoch{opt.epoch}")
        ):
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, device)
            ego = batch_data["ego"]
            if "rsu" not in ego or "batch_merged_cam_inputs" not in ego["rsu"]:
                n_skip_no_rsu += 1
                continue
            maps, proj_stats = paint_rsu_box_maps(ego)
            maps = maps.to(device)
            box_r90 = binary_objectness_target(maps, tau=1)
            predictions = model(ego)
            if "rsu" not in predictions:
                n_skip_no_rsu += 1
                continue
            pred = predictions["rsu"]
            gt_z = extract_camera_z_gt(ego["rsu"]["batch_merged_cam_inputs"]["imgs"])
            pred_z = pred["depth_z_mean"]
            if tuple(box_r90.shape) != tuple(pred_z.shape):
                raise AssertionError(
                    f"box mask {tuple(box_r90.shape)} vs depth {tuple(pred_z.shape)}"
                )
            valid = depth_valid_mask(gt_z, d_min, d_max)
            box_mask = box_r90.gt(0) & valid
            p_fg = torch.softmax(pred["heatmap_logits"], dim=1)[:, 1]
            pred_fg = p_fg.ge(fg_tau) & valid
            n_box = _add_stats(sums, counts, "gtbox", pred_z, gt_z, box_mask)
            _add_stats(sums, counts, "pred_fg", pred_z, gt_z, pred_fg)
            _add_stats(sums, counts, "full", pred_z, gt_z, valid)
            overlap_inter += float((box_mask & pred_fg).sum().item())
            overlap_union += float((box_mask | pred_fg).sum().item())
            if n_box == 0:
                n_skip_empty_box += 1
            n_batch += 1
            view = int(box_mask.view(box_mask.shape[0], -1).sum(dim=1).argmax().item())
            n_view = int(box_mask[view].sum().item())
            mae_view = (
                float((pred_z[view][box_mask[view]] - gt_z[view][box_mask[view]]).abs().mean())
                if n_view > 0
                else float("nan")
            )
            imgs = ego["rsu"]["batch_merged_cam_inputs"]["imgs"]
            if imgs.dim() == 5:
                rgb_chw = imgs.reshape(-1, *imgs.shape[2:])[view]
            else:
                rgb_chw = imgs[view]
            rgb = denormalize_rgb(rgb_chw)
            save_box_depth_panel(
                vis_dir / f"{batch_i:02d}_rsu.png",
                rgb,
                box_mask[view].detach().cpu().numpy().astype(bool),
                gt_z[view].detach().cpu().numpy(),
                pred_z[view].detach().cpu().numpy(),
                d_min,
                d_max,
                f"epoch{opt.epoch} idx={sample_ids[batch_i]} RSU GT-box  "
                f"n={n_view} MAE={mae_view:.2f}m  D=[{d_min:g},{d_max:g}]",
            )
            per_frame.append(
                {
                    "batch": batch_i,
                    "idx": sample_ids[batch_i],
                    "n_box_valid": n_box,
                    "n_pred_fg": int(pred_fg.sum().item()),
                    "mae_box": round(mae_view, 4) if n_view > 0 else None,
                    "n_proj": proj_stats["n_valid_projections"],
                }
            )

    iou = overlap_inter / max(overlap_union, 1.0)
    report = {
        "checkpoint": f"net_epoch{opt.epoch}.pth",
        "agent": "rsu",
        "mask": "official GT-box projection, tau=1 R90, AND in-range depth",
        "geometry": (
            "boxes in ego lidar; T_ego2cav = inv(img_pairwise[rsu,0]); "
            "RSU extrinsics treated as lidar-to-camera"
        ),
        "sample_n": n_keep,
        "seed": int(opt.seed),
        "sample_ids": sample_ids,
        "n_batch_used": n_batch,
        "n_skip_no_rsu": n_skip_no_rsu,
        "n_skip_empty_box": n_skip_empty_box,
        "d_min": d_min,
        "d_max": d_max,
        "fg_tau": fg_tau,
        "gtbox_depth": _pack(sums, counts, "gtbox"),
        "pred_fg_depth": _pack(sums, counts, "pred_fg"),
        "full_image_depth": _pack(sums, counts, "full"),
        "pred_fg_vs_gtbox_iou": round(iou, 4),
        "per_frame": per_frame,
    }
    out_json = out_dir / "metrics.json"
    out_txt = out_dir / "metrics.txt"
    out_json.write_text(json.dumps(report, indent=2))
    box = report["gtbox_depth"]
    pred = report["pred_fg_depth"]
    full = report["full_image_depth"]
    lines = [
        f"RSU GT-box depth audit  epoch={opt.epoch}  frames={n_batch}",
        f"mask = projected official boxes @ R90 (tau=1) AND z in [{d_min:g},{d_max:g}]",
        f"GT-box  n={box.get('n_pixels', 0):8d}  MAE={box.get('mae_m', float('nan')):.3f} m  "
        f"RMSE={box.get('rmse_m', float('nan')):.3f} m  "
        f"AbsRel={box.get('absrel', float('nan')):.3f}  delta1={box.get('delta1', float('nan')):.3f}",
        f"pred-FG n={pred.get('n_pixels', 0):8d}  MAE={pred.get('mae_m', float('nan')):.3f} m  "
        f"(heatmap tau={fg_tau:g}, not the audit mask)",
        f"full    n={full.get('n_pixels', 0):8d}  MAE={full.get('mae_m', float('nan')):.3f} m",
        f"pred-FG vs GT-box IoU = {iou:.3f}",
        f"skipped no-rsu={n_skip_no_rsu} empty-box={n_skip_empty_box}",
        "",
    ]
    text = "\n".join(lines)
    out_txt.write_text(text + "\n")
    print(text)
    print(f"wrote {out_json}")
    print(f"wrote {out_txt}")
    print(f"panels: {vis_dir}")


if __name__ == "__main__":
    main()

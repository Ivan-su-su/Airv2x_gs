# -*- coding: utf-8 -*-
"""Test-set heatmap vis (RGB + p_fg) and pixel recall vs projected GT boxes.

Test has no SAM3 ``*_seg.bin``. GT is official 3D boxes rasterized to R90
(tau=1). Collated ``extrinsics`` after ``ue4_to_lss`` are camera-to-lidar
for vehicle / RSU / drone; ``project_box_to_image`` inverts to lidar-to-cam.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

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
    compute_heatmap_metrics,
)
from opencood.models.gaussian_modules_0822.heatmap.target import binary_objectness_target
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.tools import train_utils
from opencood.tools.eval_gaussian_p1 import denormalize_rgb, load_epoch_checkpoint
from opencood.tools.train_gaussian_p1 import _unwrap_model
from opencood.utils.box_utils import boxes_to_corners_3d

AGENT_ORDER = ("vehicle", "rsu", "drone")
SCENE_RE = re.compile(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}")
# Stored extrinsics are camera-to-lidar for every agent. Do not invert RSU.
LIDAR2CAM_STORED = {"vehicle": False, "rsu": False, "drone": False}
SEMANTIC_RED = np.array([220, 20, 60], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    """CLI for test heatmap vis and recall."""
    parser = argparse.ArgumentParser(description="Test heatmap vis + GT recall")
    parser.add_argument("--hypes_yaml", "-y", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epoch", type=int, default=9)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--frames_per_scene", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fg_tau", type=float, default=PRIMARY_OBJECTNESS_THRESHOLD)
    parser.add_argument(
        "--out_root",
        default="/mnt/home/suyi/visualization/concat128_p1_test_heatmap",
    )
    return parser.parse_args()


def _as_numpy(value: Any) -> np.ndarray:
    """Detach tensors to CPU numpy."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _record_len(ego: Mapping[str, Any], agent_type: str, batch_index: int = 0) -> int:
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


def _slice_cam(arr: np.ndarray, cav_idx: int, view_idx: int, n_views: int) -> np.ndarray:
    """Index stacked camera matrices ``[N,V,...]`` or ``[N,...]``."""
    if arr.ndim >= 3 and arr.shape[0] > cav_idx and arr.shape[1] == n_views:
        return np.asarray(arr[cav_idx, view_idx])
    flat = cav_idx * n_views + view_idx
    return np.asarray(arr[flat])


def scene_from_path(path: Any) -> str:
    """Extract ``YYYY_MM_DD_HH_MM_SS`` from a dataset path."""
    match = SCENE_RE.search(str(path))
    return match.group(0) if match else "unknown"


def sample_plan(dataset: Any, frames_per_scene: int, seed: int) -> List[Tuple[str, int]]:
    """Random ``frames_per_scene`` indices per scenario, sorted within scene."""
    rng = np.random.RandomState(int(seed))
    plan: List[Tuple[str, int]] = []
    for scene_i, end in enumerate(dataset.len_record):
        start = 0 if scene_i == 0 else int(dataset.len_record[scene_i - 1])
        scene_db = dataset.scenario_database[scene_i]
        first_cav = next(iter(scene_db.values()))
        ts0 = dataset.return_timestamp_key(scene_db, 0)
        scene = scene_from_path(first_cav[ts0]["metadata_path"])
        n = int(end) - start
        k = min(int(frames_per_scene), n)
        chosen = rng.choice(n, size=k, replace=False)
        for local in sorted(int(x) for x in chosen):
            plan.append((scene, start + local))
    return plan


def paint_agent_box_maps(ego: Mapping[str, Any], agent_type: str) -> torch.Tensor:
    """Rasterize official GT boxes onto one agent's stacked camera views."""
    agent = ego.get(agent_type)
    if not isinstance(agent, Mapping):
        return torch.zeros(0, 1, 1, dtype=torch.long)
    cam_inputs = agent.get("batch_merged_cam_inputs")
    if not isinstance(cam_inputs, Mapping) or not torch.is_tensor(cam_inputs.get("imgs")):
        return torch.zeros(0, 1, 1, dtype=torch.long)
    imgs = cam_inputs["imgs"]
    if imgs.dim() == 5:
        n_cav, n_views, _, height, width = imgs.shape
    elif imgs.dim() == 4:
        n_cav, n_views = imgs.shape[0], 1
        height, width = int(imgs.shape[-2]), int(imgs.shape[-1])
    else:
        raise ValueError(f"{agent_type} imgs must be 4D or 5D, got {tuple(imgs.shape)}")
    n_flat = int(n_cav * n_views)
    maps = np.zeros((n_flat, int(height), int(width)), dtype=np.uint8)
    boxes_all = ego.get("object_bbx_center")
    mask_all = ego.get("object_bbx_mask")
    pairwise_all = ego.get("img_pairwise_t_matrix_collab")
    if not torch.is_tensor(boxes_all) or not torch.is_tensor(mask_all):
        return torch.from_numpy(maps).long()
    if not torch.is_tensor(pairwise_all):
        return torch.from_numpy(maps).long()
    if boxes_all.dim() == 2:
        boxes_all = boxes_all.unsqueeze(0)
        mask_all = mask_all.unsqueeze(0)
    if pairwise_all.dim() == 4:
        pairwise_all = pairwise_all.unsqueeze(0)
    mask = _as_numpy(mask_all[0]).reshape(-1) == 1
    boxes = _as_numpy(boxes_all[0][mask]).astype(np.float64)
    n_box = int(boxes.shape[0])
    if n_box == 0:
        return torch.from_numpy(maps).long()
    class_ids = ego.get("class_ids")
    if class_ids is None:
        class_row = np.ones((n_box,), dtype=np.int64)
    else:
        raw = class_ids[0] if not torch.is_tensor(class_ids) else class_ids[0]
        ids = _as_numpy(raw).reshape(-1).astype(np.int64)
        class_row = ids[:n_box] if ids.size >= n_box else np.ones((n_box,), dtype=np.int64)
    keep = np.ones((n_box,), dtype=bool)
    for box_idx in range(n_box):
        class_id = int(class_row[box_idx]) if box_idx < class_row.size else 1
        if class_id not in VALID_BOX_CLASS_IDS or not np.isfinite(boxes[box_idx]).all():
            keep[box_idx] = False
    boxes = boxes[keep]
    n_box = int(boxes.shape[0])
    if n_box == 0:
        return torch.from_numpy(maps).long()
    corners_ego = boxes_to_corners_3d(boxes, "hwl")
    if torch.is_tensor(corners_ego):
        corners_ego = corners_ego.numpy()
    n_this = _record_len(ego, agent_type, 0)
    if n_this == 0:
        return torch.from_numpy(maps).long()
    offset = 0
    for name in AGENT_ORDER:
        if name == agent_type:
            break
        offset += _record_len(ego, name, 0)
    pairwise = _as_numpy(pairwise_all[0])
    intrins = _as_numpy(cam_inputs["intrinsics"])
    extrinsics = _as_numpy(cam_inputs["extrinsics"])
    post_rots = _as_numpy(cam_inputs["post_rots"])
    post_trans = _as_numpy(cam_inputs["post_trans"])
    lidar2cam = bool(LIDAR2CAM_STORED[agent_type])
    image_hw = (int(height), int(width))
    ones = np.ones((n_box * 8, 1), dtype=np.float64)
    xyz_h = np.concatenate(
        [corners_ego.reshape(-1, 3).astype(np.float64), ones], axis=1
    ).T
    for local_idx in range(n_this):
        t_cav2ego = np.asarray(pairwise[offset + local_idx, 0], dtype=np.float64)
        if t_cav2ego.shape != (4, 4) or abs(float(np.linalg.det(t_cav2ego))) < 1e-8:
            t_cav2ego = np.eye(4, dtype=np.float64)
        t_ego2cav = np.linalg.inv(t_cav2ego)
        xyz_cav = (t_ego2cav @ xyz_h).T[:, :3].reshape(n_box, 8, 3)
        for view_idx in range(int(n_views)):
            flat_idx = local_idx * int(n_views) + view_idx
            k = _slice_cam(intrins, local_idx, view_idx, int(n_views))
            ext = _slice_cam(extrinsics, local_idx, view_idx, int(n_views))
            prot = _slice_cam(post_rots, local_idx, view_idx, int(n_views))
            ptra = np.asarray(_slice_cam(post_trans, local_idx, view_idx, int(n_views))).reshape(-1)
            try:
                ext_cam_to_lidar = (
                    np.linalg.inv(np.asarray(ext, dtype=np.float64))
                    if lidar2cam
                    else ext
                )
            except np.linalg.LinAlgError:
                continue
            view_mask = np.zeros((int(height), int(width)), dtype=np.uint8)
            for box_idx in range(n_box):
                projected = project_box_to_image(
                    xyz_cav[box_idx], k, ext_cam_to_lidar, prot, ptra, image_hw
                )
                if projected is None:
                    continue
                pts, _z = projected
                poly = rasterize_convex_polygon(pts, image_hw[0], image_hw[1])
                if bool(poly.any()):
                    view_mask[poly] = 1
            maps[flat_idx] = view_mask
    return torch.from_numpy(maps).long()


def overlay_fg(rgb: np.ndarray, fg_hw: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Nearest-upsample a binary map and tint foreground red."""
    ids_up = np.array(
        Image.fromarray(fg_hw.astype(np.uint8)).resize(
            (rgb.shape[1], rgb.shape[0]), Image.NEAREST
        )
    )
    mix = rgb.astype(np.float32)
    fg = ids_up > 0
    mix[fg] = (1.0 - alpha) * mix[fg] + alpha * SEMANTIC_RED.astype(np.float32)
    return np.clip(mix, 0, 255).astype(np.uint8)


def save_panel(
    path: Path,
    rgb: np.ndarray,
    p_fg: np.ndarray,
    pred_fg: np.ndarray,
    gt_fg: np.ndarray,
    title: str,
) -> None:
    """RGB | p_fg | pred overlay | GT-box overlay."""
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    im = axes[1].imshow(p_fg, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("heatmap p_fg")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(overlay_fg(rgb, pred_fg))
    axes[2].set_title("pred FG overlay")
    axes[3].imshow(overlay_fg(rgb, gt_fg))
    axes[3].set_title("GT-box overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def flatten_imgs(imgs: torch.Tensor) -> torch.Tensor:
    """``[A,V,C,H,W]`` or ``[N,C,H,W]`` → ``[N,C,H,W]``."""
    if imgs.dim() == 5:
        return imgs.reshape(-1, *imgs.shape[2:])
    return imgs


def main() -> None:
    """Sample test scenes, dump RGB+heatmap panels, aggregate recall."""
    opt = parse_args()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    hypes["validate_dir"] = hypes["test_dir"]
    hypes["train"] = False
    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print("Building test dataset...")
    dataset = build_dataset(hypes, visualize=False, train=False)
    plan = sample_plan(dataset, opt.frames_per_scene, opt.seed)
    print(
        f"test n={len(dataset)} scenes={len(dataset.len_record)} "
        f"sampled={len(plan)} seed={opt.seed}"
    )

    model = train_utils.create_model(hypes)
    load_epoch_checkpoint(model, opt.model_dir, opt.epoch)
    model.to(device)
    model.eval()
    _unwrap_model(model)
    fg_tau = float(opt.fg_tau)
    out_root = Path(opt.out_root)
    vis_dir = out_root / f"epoch{opt.epoch}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    pixel_tp: Dict[str, float] = defaultdict(float)
    pixel_fn: Dict[str, float] = defaultdict(float)
    pixel_fp: Dict[str, float] = defaultdict(float)
    scene_tp: Dict[Tuple[str, str], float] = defaultdict(float)
    scene_fn: Dict[Tuple[str, str], float] = defaultdict(float)
    scene_fp: Dict[Tuple[str, str], float] = defaultdict(float)
    n_gt_empty: Dict[str, int] = defaultdict(int)
    n_agent_frames: Dict[str, int] = defaultdict(int)
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for sample_i, (scene, idx) in enumerate(plan):
            sample = dataset[idx]
            batch = dataset.collate_batch_test([sample])
            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            meta = ego.get("metadata_path_list", [""])[0]
            scene = scene_from_path(meta) if meta else scene
            pred_all = model(ego)
            present = present_camera_agents(ego)
            print(f"[{sample_i + 1}/{len(plan)}] scene={scene} idx={idx} agents={present}")
            for agent in AGENT_ORDER:
                if agent not in present or agent not in pred_all:
                    continue
                cam = ego[agent]["batch_merged_cam_inputs"]
                if not torch.is_tensor(cam.get("imgs")):
                    continue
                logits = pred_all[agent]["heatmap_logits"]
                maps = paint_agent_box_maps(ego, agent).to(device=logits.device)
                if maps.numel() == 0 or maps.shape[0] != logits.shape[0]:
                    print(
                        f"  skip {agent}: maps {tuple(maps.shape)} vs logits {tuple(logits.shape)}"
                    )
                    continue
                gt_r90 = binary_objectness_target(maps, tau=1)
                metrics = compute_heatmap_metrics(logits, gt_r90)
                p_fg = torch.softmax(logits, dim=1)[:, 1]
                pred_fg = p_fg.ge(fg_tau)
                gt_fg = gt_r90.gt(0)
                tp = int((pred_fg & gt_fg).sum().item())
                fn = int((~pred_fg & gt_fg).sum().item())
                fp = int((pred_fg & ~gt_fg).sum().item())
                pixel_tp[agent] += tp
                pixel_fn[agent] += fn
                pixel_fp[agent] += fp
                scene_tp[(scene, agent)] += tp
                scene_fn[(scene, agent)] += fn
                scene_fp[(scene, agent)] += fp
                n_agent_frames[agent] += 1
                if int(gt_fg.sum().item()) == 0:
                    n_gt_empty[agent] += 1
                n_gt = tp + fn
                n_pred = tp + fp
                recall = tp / max(n_gt, 1)
                precision = tp / max(n_pred, 1)
                p_np = p_fg.detach().cpu().numpy()
                gt_np = gt_fg.detach().cpu().numpy().astype(np.uint8)
                pred_np = pred_fg.detach().cpu().numpy().astype(np.uint8)
                view_scores = gt_np.reshape(gt_np.shape[0], -1).sum(axis=1)
                if float(view_scores.max()) <= 0:
                    view_scores = p_np.reshape(p_np.shape[0], -1).mean(axis=1)
                view = int(np.argmax(view_scores))
                imgs = flatten_imgs(cam["imgs"])
                rgb = denormalize_rgb(imgs[view])
                out_png = vis_dir / scene / agent / f"{sample_i:02d}_{idx:04d}.png"
                title = (
                    f"test ep{opt.epoch} {scene} idx={idx} {agent} view={view}  "
                    f"recall={recall:.3f} prec={precision:.3f}  "
                    f"gt_n={n_gt} pred_n={n_pred} tau={fg_tau:g}"
                )
                save_panel(out_png, rgb, p_np[view], pred_np[view], gt_np[view], title)
                rows.append(
                    {
                        "scene": scene,
                        "idx": int(idx),
                        "agent": agent,
                        "view": view,
                        "recall": round(recall, 4),
                        "precision": round(precision, 4),
                        "tp": tp,
                        "fn": fn,
                        "fp": fp,
                        "metric_recall@0.3": float(metrics.get("recall@0.3", 0.0)),
                        "png": str(out_png),
                    }
                )

    def pack(tp: float, fn: float, fp: float) -> Dict[str, Any]:
        n_gt = tp + fn
        n_pred = tp + fp
        return {
            "n_gt_pixels": int(n_gt),
            "n_pred_pixels": int(n_pred),
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "recall": round(tp / max(n_gt, 1.0), 4) if n_gt > 0 else None,
            "precision": round(tp / max(n_pred, 1.0), 4) if n_pred > 0 else None,
        }

    by_agent = {
        agent: {
            **pack(pixel_tp[agent], pixel_fn[agent], pixel_fp[agent]),
            "n_frames": n_agent_frames[agent],
            "n_gt_empty_frames": n_gt_empty[agent],
        }
        for agent in AGENT_ORDER
    }
    by_scene: Dict[str, Dict[str, Any]] = {}
    scenes = sorted({scene for scene, _ in plan})
    for scene in scenes:
        by_scene[scene] = {
            agent: pack(
                scene_tp[(scene, agent)],
                scene_fn[(scene, agent)],
                scene_fp[(scene, agent)],
            )
            for agent in AGENT_ORDER
        }
    report = {
        "checkpoint": f"net_epoch{opt.epoch}.pth",
        "split": "test",
        "n_scenes": len(scenes),
        "frames_per_scene": int(opt.frames_per_scene),
        "seed": int(opt.seed),
        "fg_tau": fg_tau,
        "gt": "official 3D GT boxes projected to each agent camera, tau=1 R90",
        "note": "test has no SAM3 seg.bin; coverage = pixel recall of pred heatmap vs GT boxes",
        "by_agent": by_agent,
        "by_scene": by_scene,
        "plan": [{"scene": s, "idx": i} for s, i in plan],
        "rows": rows,
    }
    out_json = out_root / f"epoch{opt.epoch}_metrics.json"
    out_txt = out_root / f"epoch{opt.epoch}_metrics.txt"
    out_json.write_text(json.dumps(report, indent=2))
    lines = [
        f"test heatmap recall  epoch={opt.epoch}  scenes={len(scenes)}  "
        f"{opt.frames_per_scene}/scene  seed={opt.seed}  tau={fg_tau:g}",
        "GT = projected official 3D boxes (test has no SAM3). recall = covered GT pixels.",
        "",
        f"{'agent':8s}  {'recall':>8s}  {'prec':>8s}  {'gt_px':>10s}  {'pred_px':>10s}  frames  emptyGT",
    ]
    for agent in AGENT_ORDER:
        block = by_agent[agent]
        rec = block["recall"]
        prec = block["precision"]
        rec_s = f"{rec:.3f}" if rec is not None else "  n/a"
        prec_s = f"{prec:.3f}" if prec is not None else "  n/a"
        lines.append(
            f"{agent:8s}  {rec_s:>8s}  {prec_s:>8s}  {block['n_gt_pixels']:10d}  "
            f"{block['n_pred_pixels']:10d}  {block['n_frames']:6d}  {block['n_gt_empty_frames']:7d}"
        )
    lines.append("")
    for scene in scenes:
        lines.append(f"scene {scene}")
        for agent in AGENT_ORDER:
            block = by_scene[scene][agent]
            rec = block["recall"]
            rec_s = f"{rec:.3f}" if rec is not None else "n/a"
            lines.append(
                f"  {agent:8s} recall={rec_s}  gt_px={block['n_gt_pixels']:6d}  "
                f"pred_px={block['n_pred_pixels']:7d}"
            )
    text = "\n".join(lines) + "\n"
    out_txt.write_text(text)
    print(text)
    print(f"wrote {out_json}")
    print(f"wrote {out_txt}")
    print(f"panels: {vis_dir}")


if __name__ == "__main__":
    main()

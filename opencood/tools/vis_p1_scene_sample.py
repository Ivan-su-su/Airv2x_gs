# -*- coding: utf-8 -*-
"""Per-scene P1 objectness vis: 5 frames/scene, seg.bin or 3D-box GT."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.loss.gaussian_p1_semantic_loss import softmax_focal_loss
from opencood.models.gaussian_modules_0822.heatmap.metrics import compute_heatmap_metrics
from opencood.models.gaussian_modules_0822.heatmap.target import binary_objectness_target
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.tools import train_utils
from opencood.tools.analyze_p1_heatmap_resolution import (
    project_box_to_image,
    rasterize_convex_polygon,
)
from opencood.tools.train_gaussian_p1 import _unwrap_model
from opencood.utils.box_utils import boxes_to_corners_3d

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SEMANTIC_COLORS = np.array([[40, 40, 40], [220, 20, 60]], dtype=np.uint8)
AGENT_ORDER = ("vehicle", "rsu", "drone")
SCENE_RE = re.compile(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}")
AGENT_ABBR = {"vehicle": "veh", "rsu": "rsu", "drone": "drone"}


def parse_args() -> argparse.Namespace:
    """CLI for per-scene train/test heatmap visualization."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--hypes_yaml", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--frames_per_scene", type=int, default=5)
    parser.add_argument(
        "--out_root",
        default="/mnt/home/suyi/visualization",
    )
    return parser.parse_args()


def scene_from_path(path: Any) -> str:
    """Extract ``YYYY_MM_DD_HH_MM_SS`` from a dataset path."""
    match = SCENE_RE.search(str(path))
    return match.group(0) if match else "unknown"


def evenly_sample(start: int, end: int, k: int) -> List[int]:
    """Inclusive ``[start, end)`` evenly spaced unique indices."""
    n = end - start
    if n <= 0:
        return []
    if n <= k:
        return list(range(start, end))
    raw = np.linspace(start, end - 1, num=k)
    return sorted({int(round(float(x))) for x in raw})


def sample_plan(dataset: Any, frames_per_scene: int) -> List[Tuple[str, int, str]]:
    """Return ``(scene, idx, timestamp_key)`` tuples."""
    plan: List[Tuple[str, int, str]] = []
    for scene_i, end in enumerate(dataset.len_record):
        start = 0 if scene_i == 0 else int(dataset.len_record[scene_i - 1])
        scene_db = dataset.scenario_database[scene_i]
        first_cav = next(iter(scene_db.values()))
        ts0 = dataset.return_timestamp_key(scene_db, 0)
        scene = scene_from_path(first_cav[ts0]["metadata_path"])
        for idx in evenly_sample(start, int(end), frames_per_scene):
            local = idx if scene_i == 0 else idx - int(dataset.len_record[scene_i - 1])
            ts = dataset.return_timestamp_key(scene_db, local)
            plan.append((scene, idx, str(ts)))
    return plan


def denorm_rgb(chw: torch.Tensor) -> np.ndarray:
    """ImageNet CHW → HWC uint8."""
    rgb = chw.detach().float().cpu().numpy()[:3]
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def overlay_seg(rgb: np.ndarray, ids_hw: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Nearest-upsample {0,1} map onto RGB."""
    color = SEMANTIC_COLORS[np.clip(ids_hw.astype(np.int64), 0, 1)]
    color_up = np.array(
        Image.fromarray(color).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)
    )
    ids_up = np.array(
        Image.fromarray(ids_hw.astype(np.uint8)).resize(
            (rgb.shape[1], rgb.shape[0]), Image.NEAREST
        )
    )
    mix = rgb.astype(np.float32)
    fg = ids_up > 0
    mix[fg] = (1.0 - alpha) * mix[fg] + alpha * color_up[fg].astype(np.float32)
    return np.clip(mix, 0, 255).astype(np.uint8)


def flatten_cam_matrix(tensor: torch.Tensor, n_views: int) -> np.ndarray:
    """Align camera matrices to flattened view count ``N`` as numpy."""
    arr = tensor.detach().float().cpu().numpy()
    if int(arr.shape[0]) == n_views:
        if arr.ndim >= 3 and int(arr.shape[1]) == 1 and int(arr.shape[2]) in (3, 4):
            return arr.reshape(n_views, *arr.shape[2:])
        return arr
    if arr.ndim >= 3 and int(arr.shape[0] * arr.shape[1]) == n_views:
        return arr.reshape(n_views, *arr.shape[2:])
    if n_views == 1 and arr.ndim >= 2:
        return arr.reshape(1, *arr.shape[-2:])
    raise ValueError(f"cannot flatten camera tensor {arr.shape} to N={n_views}")


def flatten_semantic(semantic: torch.Tensor) -> torch.Tensor:
    """``[A,V,H,W]`` or ``[N,H,W]`` → ``[N,H,W]``."""
    if semantic.dim() == 4:
        return semantic.reshape(-1, semantic.shape[-2], semantic.shape[-1])
    if semantic.dim() != 3:
        raise ValueError(f"semantic rank {semantic.dim()}")
    return semantic


def flatten_imgs(imgs: torch.Tensor) -> torch.Tensor:
    """``[A,V,C,H,W]`` or ``[N,C,H,W]`` → ``[N,C,H,W]``."""
    if imgs.dim() == 5:
        return imgs.reshape(-1, *imgs.shape[2:])
    return imgs


def heatmap_head_stats(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """cls bias / |weight| for the 2-class HeatmapHead."""
    report: Dict[str, Any] = {}
    for agent in AGENT_ORDER:
        bias_key = f"heatmap_heads.{agent}.cls.bias"
        w_key = f"heatmap_heads.{agent}.cls.weight"
        if bias_key not in state:
            continue
        bias = state[bias_key].detach().float().cpu().tolist()
        weight = state[w_key].detach().float().abs().mean(dim=(1, 2, 3)).cpu().tolist()
        report[agent] = {
            "cls_bias": bias,
            "bias_fg_minus_bg": float(bias[1] - bias[0]),
            "cls_weight_abs_mean": weight,
        }
    return report


def valid_boxes(ego: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Ego-frame boxes ``[M,7]`` and class ids ``[M]``."""
    centers = ego["object_bbx_center"]
    mask = ego["object_bbx_mask"]
    if torch.is_tensor(centers):
        centers = centers.detach().cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    if centers.ndim == 3:
        centers = centers[0]
        mask = mask[0]
    keep = mask.reshape(-1) > 0.5
    boxes = centers[keep]
    class_ids = ego.get("class_ids", [[]])
    if isinstance(class_ids, list) and class_ids:
        raw = class_ids[0] if isinstance(class_ids[0], (list, tuple, np.ndarray)) else class_ids
        ids = np.asarray(raw, dtype=np.int64).reshape(-1)
        if ids.size >= int(keep.sum()):
            ids = ids[: int(keep.sum())]
        else:
            ids = np.ones((int(keep.sum()),), dtype=np.int64)
    else:
        ids = np.ones((int(keep.sum()),), dtype=np.int64)
    return boxes.astype(np.float64), ids


def agent_offsets(ego: Dict[str, Any]) -> Dict[str, int]:
    """Pairwise-matrix offset of each agent type in vehicle→rsu→drone order."""
    offset = 0
    out: Dict[str, int] = {}
    for agent in AGENT_ORDER:
        out[agent] = offset
        rec = ego.get(agent, {}).get("record_len", 0)
        if torch.is_tensor(rec):
            n_cav = int(rec.reshape(-1)[0].item())
        else:
            n_cav = int(rec) if rec else 0
        offset += n_cav
    return out


def project_boxes_to_views(
    ego: Dict[str, Any],
    agent_type: str,
    cam_inputs: Dict[str, Any],
    image_hw: Tuple[int, int],
) -> np.ndarray:
    """Rasterize ego 3D boxes into per-view occupancy ``[N, H, W]``."""
    imgs = flatten_imgs(cam_inputs["imgs"])
    n_views = int(imgs.shape[0])
    height, width = image_hw
    occupancy = np.zeros((n_views, height, width), dtype=np.uint8)
    boxes, _class_ids = valid_boxes(ego)
    if boxes.shape[0] == 0:
        return occupancy
    print(
        f"[project] {agent_type} imgs={tuple(cam_inputs['imgs'].shape)} "
        f"K={tuple(cam_inputs['intrinsics'].shape)} "
        f"E={tuple(cam_inputs['extrinsics'].shape)} "
        f"n_views={n_views} n_box={boxes.shape[0]}",
        flush=True,
    )
    corners = boxes_to_corners_3d(boxes, "hwl")
    if torch.is_tensor(corners):
        corners = corners.detach().cpu().numpy()
    pairwise = ego["img_pairwise_t_matrix_collab"]
    if torch.is_tensor(pairwise):
        pairwise = pairwise.detach().cpu().numpy()
    if pairwise.ndim == 5:
        pairwise = pairwise[0]
    raw_imgs = cam_inputs["imgs"]
    n_cav = int(raw_imgs.shape[0]) if raw_imgs.dim() == 5 else 1
    views_per_cav = n_views // max(n_cav, 1)
    cav_offset = agent_offsets(ego)[agent_type]
    intrins = flatten_cam_matrix(cam_inputs["intrinsics"], n_views)
    extrinsics = flatten_cam_matrix(cam_inputs["extrinsics"], n_views)
    post_rots = flatten_cam_matrix(cam_inputs["post_rots"], n_views)
    post_trans = flatten_cam_matrix(cam_inputs["post_trans"], n_views)

    for view_i in range(n_views):
        local_cav = view_i // max(views_per_cav, 1)
        t_cav2ego = np.asarray(pairwise[cav_offset + local_cav, 0], dtype=np.float64)
        if t_cav2ego.shape != (4, 4) or abs(float(np.linalg.det(t_cav2ego))) < 1e-8:
            t_cav2ego = np.eye(4, dtype=np.float64)
        t_ego2cav = np.linalg.inv(t_cav2ego)
        n_box = int(corners.shape[0])
        ones = np.ones((n_box * 8, 1), dtype=np.float64)
        xyz_h = np.concatenate(
            [corners.reshape(-1, 3).astype(np.float64), ones], axis=1
        ).T
        xyz_cav = (t_ego2cav @ xyz_h).T[:, :3].reshape(n_box, 8, 3)
        mask = np.zeros((height, width), dtype=bool)
        for box_i in range(n_box):
            corners_i = np.asarray(xyz_cav[box_i], dtype=np.float64).reshape(8, 3)
            K = np.asarray(intrins[view_i], dtype=np.float64)
            ext = np.asarray(extrinsics[view_i], dtype=np.float64)
            prot = np.asarray(post_rots[view_i], dtype=np.float64)
            ptra = np.asarray(post_trans[view_i], dtype=np.float64).reshape(-1)
            if ext.shape != (4, 4) or K.ndim != 2:
                continue
            try:
                projected = project_box_to_image(
                    corners_i, K, ext, prot, ptra, image_hw
                )
            except Exception as exc:
                print(
                    f"[project skip] {agent_type} view={view_i} "
                    f"corners={corners_i.shape} K={K.shape} ext={ext.shape} "
                    f"prot={prot.shape} ptra={ptra.shape}: {exc}",
                    flush=True,
                )
                continue
            if projected is None:
                continue
            pts, _z = projected
            mask |= rasterize_convex_polygon(pts, height, width)
        occupancy[view_i] = mask.astype(np.uint8)
    return occupancy


def source_semantic(
    ego: Dict[str, Any],
    agent_type: str,
    cam_inputs: Dict[str, Any],
) -> Tuple[torch.Tensor, List[str]]:
    """Per-view GT ids. Use 3D-box occupancy when a view has no seg.bin."""
    imgs = flatten_imgs(cam_inputs["imgs"])
    n_views = int(imgs.shape[0])
    height, width = int(imgs.shape[-2]), int(imgs.shape[-1])
    semantic = cam_inputs.get("image_semantic_gts")
    if torch.is_tensor(semantic):
        semantic = flatten_semantic(semantic.long().cpu())
        if int(semantic.shape[0]) != n_views:
            semantic = semantic.reshape(n_views, height, width)
    else:
        semantic = torch.zeros(n_views, height, width, dtype=torch.long)
    sources: List[str] = []
    box_maps: Optional[np.ndarray] = None
    for view_i in range(n_views):
        if int(semantic[view_i].max().item()) > 0:
            sources.append("seg")
            continue
        if box_maps is None:
            box_maps = project_boxes_to_views(
                ego, agent_type, cam_inputs, (height, width)
            )
        semantic[view_i] = torch.from_numpy(box_maps[view_i].astype(np.int64))
        sources.append("box" if int(box_maps[view_i].max()) > 0 else "empty")
    return semantic, sources


def save_panel(
    path: Path,
    rgb: np.ndarray,
    gt_ids: np.ndarray,
    p_fg: np.ndarray,
    title: str,
) -> None:
    """RGB | GT overlay | p_fg | pred@0.3."""
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.6))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[1].imshow(overlay_seg(rgb, gt_ids))
    axes[1].set_title("GT overlay")
    im2 = axes[2].imshow(p_fg, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("p_fg")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[3].imshow(overlay_seg(rgb, (p_fg >= 0.3).astype(np.int64)))
    axes[3].set_title("pred @0.3")
    for ax in axes:
        ax.axis("off")
    handles = [
        mpatches.Patch(color=SEMANTIC_COLORS[0] / 255.0, label="bg"),
        mpatches.Patch(color=SEMANTIC_COLORS[1] / 255.0, label="fg"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def mean_or_nan(values: Sequence[float]) -> float:
    """Mean of a list, NaN if empty."""
    arr = [float(v) for v in values if v == v]
    if not arr:
        return float("nan")
    return float(sum(arr) / len(arr))


def main() -> None:
    """Load one epoch, sample 5 frames/scene, dump panels + metrics."""
    opt = parse_args()
    model_dir = Path(opt.model_dir)
    ckpt = model_dir / f"net_epoch{opt.epoch}.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    out_dir = (
        Path(opt.out_root)
        / model_dir.name
        / f"epoch{opt.epoch}"
        / opt.split
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    if opt.split == "train":
        hypes["validate_dir"] = hypes["root_dir"]
    else:
        hypes["validate_dir"] = hypes["test_dir"]
    hypes["train"] = False

    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    raw = torch.load(str(ckpt), map_location="cpu")
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    head_stats = heatmap_head_stats(state)
    (out_dir.parent / "heatmap_head_stats.json").write_text(
        json.dumps(head_stats, indent=2)
    )

    print(f"Creating model on {device} from {ckpt}", flush=True)
    model = train_utils.create_model(hypes)
    missing, unexpected = model.load_state_dict(state, strict=True)
    print("missing", missing, "unexpected", unexpected, flush=True)
    model.to(device)
    model.eval()
    _unwrap_model(model)

    print(f"Building {opt.split} dataset from {hypes['validate_dir']}", flush=True)
    dataset = build_dataset(hypes, visualize=False, train=False)
    plan = sample_plan(dataset, opt.frames_per_scene)
    print(f"{opt.split}: {len(dataset)} frames, {len(dataset.len_record)} scenes, "
          f"{len(plan)} sampled", flush=True)

    rows: List[Dict[str, Any]] = []
    metric_acc: Dict[str, List[float]] = {
        "heatmap_loss": [],
        "p_fg_mean": [],
        "pred_fg@0.3": [],
        "gt_fg_ratio": [],
        "mean_p_fg_gt": [],
        "recall@0.3": [],
        "precision@0.3": [],
        "f1@0.3": [],
        "logit_fg_mean": [],
        "logit_bg_mean": [],
    }

    with torch.no_grad():
        for sample_i, (scene, idx, timestamp) in enumerate(plan):
            sample = dataset[idx]
            batch = dataset.collate_batch_test([sample])
            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            pred = model(ego)
            meta = ego.get("metadata_path_list", [""])[0]
            scene = scene_from_path(meta) if meta else scene
            present = present_camera_agents(ego)
            print(
                f"[{opt.split} {sample_i+1}/{len(plan)}] scene={scene} idx={idx} "
                f"ts={timestamp} agents={present}",
                flush=True,
            )
            for agent in present:
                cam = ego[agent]["batch_merged_cam_inputs"]
                logits = pred[agent]["heatmap_logits"]
                imgs = flatten_imgs(cam["imgs"])
                semantic, sources = source_semantic(ego, agent, cam)
                semantic = semantic.to(device=logits.device)
                target = binary_objectness_target(semantic.long(), tau=1)
                if tuple(target.shape[-2:]) != tuple(logits.shape[-2:]):
                    raise AssertionError(
                        f"{agent} target {tuple(target.shape)} vs logits {tuple(logits.shape)}"
                    )
                hm_loss = float(
                    softmax_focal_loss(logits, target, gamma=2.0).item()
                )
                metrics = compute_heatmap_metrics(logits, target)
                probs = torch.softmax(logits, dim=1)
                p_fg = probs[:, 1]
                logit_fg = logits[:, 1]
                logit_bg = logits[:, 0]
                p_np = p_fg.detach().cpu().numpy()
                gt_np = target.detach().cpu().numpy()
                scores = [
                    float(gt_np[view_i].mean()) if sources[view_i] != "empty"
                    else float(p_np[view_i].mean())
                    for view_i in range(p_np.shape[0])
                ]
                view = int(np.argmax(scores)) if scores else 0
                rgb = denorm_rgb(imgs[view])
                png = out_dir / scene / (
                    f"{sample_i:02d}_{timestamp}_{agent}_{sources[view]}.png"
                )
                title = (
                    f"{opt.split} ep{opt.epoch} {scene} ts={timestamp} {agent} "
                    f"view{view}/{p_np.shape[0]} gt={sources[view]}  "
                    f"p_fg={float(p_np[view].mean()):.3f} "
                    f"gt_fg={float(gt_np[view].mean()):.3f} "
                    f"pred@0.3={float((p_np[view] >= 0.3).mean()):.3f} "
                    f"loss={hm_loss:.4f}"
                )
                save_panel(png, rgb, gt_np[view], p_np[view], title)
                row = {
                    "split": opt.split,
                    "epoch": opt.epoch,
                    "scene": scene,
                    "idx": int(idx),
                    "timestamp": timestamp,
                    "agent": agent,
                    "view": view,
                    "n_views": int(p_np.shape[0]),
                    "gt_source_view": sources[view],
                    "gt_sources": sources,
                    "png": str(png),
                    "heatmap_loss": hm_loss,
                    "p_fg_mean": float(p_np.mean()),
                    "p_fg_max": float(p_np.max()),
                    "pred_fg@0.1": float((p_np >= 0.1).mean()),
                    "pred_fg@0.2": float((p_np >= 0.2).mean()),
                    "pred_fg@0.3": float((p_np >= 0.3).mean()),
                    "pred_fg@0.5": float((p_np >= 0.5).mean()),
                    "gt_fg_ratio": float(gt_np.mean()),
                    "logit_fg_mean": float(logit_fg.mean().item()),
                    "logit_bg_mean": float(logit_bg.mean().item()),
                    **{f"metric_{k}": float(v) for k, v in metrics.items()},
                }
                rows.append(row)
                metric_acc["heatmap_loss"].append(hm_loss)
                metric_acc["p_fg_mean"].append(float(p_np.mean()))
                metric_acc["pred_fg@0.3"].append(float((p_np >= 0.3).mean()))
                metric_acc["gt_fg_ratio"].append(float(gt_np.mean()))
                metric_acc["mean_p_fg_gt"].append(float(metrics.get("mean_p_fg_gt", 0.0)))
                metric_acc["recall@0.3"].append(float(metrics.get("recall@0.3", 0.0)))
                metric_acc["precision@0.3"].append(float(metrics.get("precision@0.3", 0.0)))
                metric_acc["f1@0.3"].append(float(metrics.get("f1@0.3", 0.0)))
                metric_acc["logit_fg_mean"].append(float(logit_fg.mean().item()))
                metric_acc["logit_bg_mean"].append(float(logit_bg.mean().item()))

    summary = {key: mean_or_nan(vals) for key, vals in metric_acc.items()}
    summary["n_panels"] = len(rows)
    summary["n_samples"] = len(plan)
    summary["collapse_all_bg"] = bool(summary["pred_fg@0.3"] < 1.0e-4)
    summary["fg_logit_below_bg"] = bool(
        summary["logit_fg_mean"] < summary["logit_bg_mean"]
    )
    report = {
        "ckpt": str(ckpt),
        "epoch": opt.epoch,
        "split": opt.split,
        "heatmap_head": head_stats,
        "summary": summary,
        "rows": rows,
    }
    report_path = out_dir.parent / f"report_{opt.split}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"summary": summary, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()

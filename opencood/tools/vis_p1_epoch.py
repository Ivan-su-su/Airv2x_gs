# -*- coding: utf-8 -*-
"""Quick P1 visualization: 10 val depth frames + train heatmap RGB/GT/pred.

Does not run the full val set. Also dumps per-agent weight stats from the ckpt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.gaussian_modules_0822.heatmap.target import build_semantic_target
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.models.gaussian_modules_0822.lss.target import extract_camera_z_gt
from opencood.tools import train_utils
from opencood.tools.train import setup_dataloader
from opencood.tools.train_gaussian_p1 import _unwrap_model

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SEMANTIC_COLORS = np.array(
    [
        [40, 40, 40],
        [220, 20, 60],
    ],
    dtype=np.uint8,
)
SEMANTIC_NAMES = ["bg", "fg"]


def parse_args() -> argparse.Namespace:
    """CLI for 10-frame P1 visualization."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--hypes_yaml", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epoch", type=int, default=9)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_depth", type=int, default=10)
    parser.add_argument("--n_heatmap_per_agent", type=int, default=2)
    return parser.parse_args()


def ckpt_path(model_dir: str, epoch: int) -> str:
    """Return ``net_epoch{epoch}.pth`` path."""
    path = os.path.join(model_dir, f"net_epoch{epoch}.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def load_state(model: torch.nn.Module, path: str) -> None:
    """Load checkpoint weights strictly."""
    raw = torch.load(path, map_location="cpu")
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"Loaded {path}")
    if missing:
        print("missing", missing)
    if unexpected:
        print("unexpected", unexpected)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> Any:
    """Cosine of two flattened tensors, or a mismatch marker."""
    if a.numel() != b.numel():
        return {"status": "shape_mismatch", "shapes": [list(a.shape), list(b.shape)]}
    return round(
        float(
            torch.nn.functional.cosine_similarity(
                a.flatten().float(), b.flatten().float(), dim=0
            )
        ),
        4,
    )


def inspect_weights(path: str, out_json: Path) -> Dict[str, Any]:
    """Per-agent trainable stats and cross-agent cosine of matched tensors."""
    raw = torch.load(path, map_location="cpu")
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    agents = ["vehicle", "rsu", "drone"]
    report: Dict[str, Any] = {
        "n_tensors": len(state),
        "architecture_keys": {},
        "agents": {},
        "cosine": {},
        "shapes": {},
    }
    for agent in agents:
        report["architecture_keys"][agent] = {
            "encoder": any(k.startswith(f"frontend.encoders.{agent}.") for k in state),
            "highres": any(k.startswith(f"highres.{agent}.") for k in state),
            "heatmap_head": any(k.startswith(f"heatmap_heads.{agent}.") for k in state),
            "depth_head": any(k.startswith(f"depth_heads.{agent}.") for k in state),
        }

    grouped: Dict[str, Dict[str, List[float]]] = {a: defaultdict(list) for a in agents}
    grouped["other"] = defaultdict(list)
    for name, tensor in state.items():
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            continue
        agent = next((a for a in agents if f".{a}." in name), "other")
        family = "other"
        for token in ("up1", "up2", "depth_heads", "depth_head", "heatmap_heads", "highres", "trunk", "image_head"):
            if token in name:
                family = token
                break
        grouped[agent][family].append(float(tensor.detach().float().abs().mean()))
    for agent, fams in grouped.items():
        report["agents"][agent] = {k: round(float(np.mean(v)), 6) for k, v in fams.items() if v}

    named = {
        "trunk_stem": "frontend.encoders.{a}.trunk._conv_stem.weight",
        "up1_first": "frontend.encoders.{a}.up1.conv.0.weight",
        "up2_first": "frontend.encoders.{a}.up2.conv.0.weight",
        "depth_head": "depth_heads.{a}.pred.weight",
        "heatmap_cls": "heatmap_heads.{a}.cls.weight",
        "conv1": "highres.{a}.conv1.weight",
        "conv1_bias": "highres.{a}.conv1.bias",
        "conv2": "highres.{a}.conv2.weight",
    }
    for label, template in named.items():
        tensors = {}
        for agent in agents:
            key = template.format(a=agent)
            if key in state:
                tensors[agent] = state[key]
                report["shapes"][f"{label}.{agent}"] = list(state[key].shape)
        pair: Dict[str, Any] = {}
        names = list(tensors.keys())
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                pair[f"{left}-vs-{right}"] = _cosine(tensors[left], tensors[right])
        pair["abs_mean"] = {
            agent: round(float(t.detach().float().abs().mean()), 6) for agent, t in tensors.items()
        }
        if label == "conv1":
            pair["max_abs"] = {
                agent: round(float(t.detach().float().abs().max()), 6) for agent, t in tensors.items()
            }
        report["cosine"][label] = pair

    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def denorm_rgb(chw: torch.Tensor) -> np.ndarray:
    """ImageNet CHW → HWC uint8."""
    rgb = chw.detach().float().cpu().numpy()[:3]
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def colorize_seg(ids: np.ndarray) -> np.ndarray:
    """Map {0,1} objectness ids to RGB."""
    clipped = np.clip(ids.astype(np.int64), 0, 1)
    return SEMANTIC_COLORS[clipped]


def overlay_seg(rgb: np.ndarray, seg_hw: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Resize nearest seg to RGB and blend."""
    from PIL import Image

    color = colorize_seg(seg_hw)
    color_up = np.array(Image.fromarray(color).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))
    mix = (1.0 - alpha) * rgb.astype(np.float32) + alpha * color_up.astype(np.float32)
    return np.clip(mix, 0, 255).astype(np.uint8)


def flatten_views(imgs: torch.Tensor) -> torch.Tensor:
    """``[B,V,C,H,W]`` or ``[N,C,H,W]`` → ``[N,C,H,W]``."""
    if imgs.dim() == 5:
        batch, views, channels, height, width = imgs.shape
        return imgs.reshape(batch * views, channels, height, width)
    return imgs


def pick_view(maps: np.ndarray, prefer_fg: bool) -> int:
    """Pick a camera index; heatmap prefers foreground, depth prefers z variation."""
    if maps.ndim != 3:
        return 0
    scores = []
    for idx in range(maps.shape[0]):
        if prefer_fg:
            scores.append(float((maps[idx] > 0).mean()))
        else:
            scores.append(float(np.std(maps[idx])))
    return int(np.argmax(scores)) if scores else 0


def save_depth_row(
    path: Path,
    rgb: np.ndarray,
    gt_z: np.ndarray,
    pred_z: np.ndarray,
    d_min: float,
    d_max: float,
    title: str,
) -> None:
    """RGB | GT depth | pred depth | error."""
    err = np.abs(pred_z - gt_z)
    valid = (gt_z >= d_min) & (gt_z <= d_max)
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.3))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    im1 = axes[1].imshow(gt_z, cmap="turbo", vmin=d_min, vmax=d_max)
    axes[1].set_title("GT z (m)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(pred_z, cmap="turbo", vmin=d_min, vmax=d_max)
    axes[2].set_title("Pred z (m)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    im3 = axes[3].imshow(np.where(valid, err, np.nan), cmap="magma", vmin=0, vmax=min(15.0, d_max))
    axes[3].set_title("|err| valid (m)")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_heatmap_row(
    path: Path,
    rgb: np.ndarray,
    gt_ids: np.ndarray,
    p_fg: np.ndarray,
    title: str,
) -> None:
    """RGB | GT overlay | p_fg | overlay at tau=0.3."""
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.6))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB (train)")
    axes[1].imshow(overlay_seg(rgb, gt_ids))
    axes[1].set_title("GT objectness overlay")
    im2 = axes[2].imshow(p_fg, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("p_fg")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    axes[3].imshow(overlay_seg(rgb, (p_fg >= 0.3).astype(np.int64)))
    axes[3].set_title("pred overlay @0.3")
    for ax in axes:
        ax.axis("off")
    handles = [
        mpatches.Patch(color=SEMANTIC_COLORS[i] / 255.0, label=f"{i}:{SEMANTIC_NAMES[i]}")
        for i in range(len(SEMANTIC_NAMES))
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def dump_live_arch(core: torch.nn.Module) -> Dict[str, Any]:
    """Confirm three independent agent modules on the live model."""
    return {
        "encoders": list(core.frontend.encoders.keys()),
        "highres": list(core.highres.keys()),
        "heatmap_heads": list(core.heatmap_heads.keys()),
        "depth_heads": list(core.depth_heads.keys()),
        "depth_head_out": {
            agent: int(core.depth_heads[agent].num_bins)
            for agent in core.depth_heads
        },
        "d_range": {
            agent: [
                float(core.frontend.encoders[agent].d_min),
                float(core.frontend.encoders[agent].d_max),
            ]
            for agent in core.frontend.encoders
        },
        "separate_modules": {
            "encoder_vehicle_vs_rsu": id(core.frontend.encoders["vehicle"])
            != id(core.frontend.encoders["rsu"]),
            "heatmap_vehicle_vs_rsu": id(core.heatmap_heads["vehicle"])
            != id(core.heatmap_heads["rsu"]),
            "highres_vehicle_vs_rsu": id(core.highres["vehicle"]) != id(core.highres["rsu"]),
        },
    }


def main() -> None:
    """Inspect weights, then dump 10 depth + a few train heatmap panels."""
    opt = parse_args()
    out_dir = Path(opt.model_dir) / "visualization" / f"eval_epoch{opt.epoch}"
    depth_dir = out_dir / "depth"
    heat_dir = out_dir / "heatmap"
    depth_dir.mkdir(parents=True, exist_ok=True)
    heat_dir.mkdir(parents=True, exist_ok=True)

    path = ckpt_path(opt.model_dir, opt.epoch)
    print("=== weight stats ===")
    inspect_weights(path, out_dir / "weight_stats.json")

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    dummy = Namespace(
        distributed=False,
        worker=0,
        gpu_id=opt.gpu_id,
        amp=False,
        tag="vis",
        model_dir="",
        hypes_yaml=opt.hypes_yaml,
    )
    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print("Creating model...")
    model = train_utils.create_model(hypes)
    load_state(model, path)
    model.to(device)
    model.eval()
    core = _unwrap_model(model)
    live_arch = dump_live_arch(core)
    print("live architecture:", json.dumps(live_arch, indent=2))

    depth_stats: Dict[str, List[float]] = defaultdict(list)
    n_depth = 0
    print("Building val dataset (depth vis, 10 frames)...")
    val_dataset = build_dataset(hypes, visualize=False, train=False)
    val_loader = setup_dataloader(val_dataset, hypes, dummy, is_train=False)
    with torch.no_grad():
        for batch_i, batch in enumerate(val_loader):
            if batch is None:
                continue
            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            pred = model(ego)
            present = present_camera_agents(ego)
            print(f"[val {batch_i}] present_agents={present} pred_keys={list(pred.keys())}")
            for agent in present:
                if n_depth >= opt.n_depth:
                    break
                encoder = core.frontend.encoders[agent]
                imgs = ego[agent]["batch_merged_cam_inputs"]["imgs"]
                gt_all = extract_camera_z_gt(imgs).detach().cpu().numpy()
                pred_all = pred[agent]["depth_z_mean"].detach().cpu().numpy()
                view = pick_view(gt_all, prefer_fg=False)
                gt_z = gt_all[view]
                pred_z = pred_all[view]
                d_min, d_max = float(encoder.d_min), float(encoder.d_max)
                valid = (gt_z >= d_min) & (gt_z <= d_max)
                mae = float(np.abs(pred_z[valid] - gt_z[valid]).mean()) if valid.any() else float("nan")
                pred_std = float(pred_z.std())
                gt_std = float(gt_z[valid].std()) if valid.any() else float("nan")
                depth_stats[agent].append(mae)
                rgb = denorm_rgb(flatten_views(imgs)[view])
                save_depth_row(
                    depth_dir / f"{n_depth:02d}_{agent}.png",
                    rgb,
                    gt_z,
                    pred_z,
                    d_min,
                    d_max,
                    f"val#{batch_i} {agent} view{view}  MAE={mae:.2f}m  "
                    f"pred_std={pred_std:.2f} gt_std={gt_std:.2f}  "
                    f"D={core.depth_heads[agent].num_bins} [{d_min:g},{d_max:g}]",
                )
                n_depth += 1
            if n_depth >= opt.n_depth:
                break
    del val_loader, val_dataset

    heat_counts: Dict[str, int] = defaultdict(int)
    n_heat = 0
    print("Building train dataset (heatmap vis)...")
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    train_loader = setup_dataloader(train_dataset, hypes, dummy, is_train=True)
    with torch.no_grad():
        for batch_i, batch in enumerate(train_loader):
            if batch is None:
                continue
            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            pred = model(ego)
            present = present_camera_agents(ego)
            print(f"[train {batch_i}] present_agents={present}")
            for agent in present:
                if heat_counts[agent] >= opt.n_heatmap_per_agent:
                    continue
                imgs = ego[agent]["batch_merged_cam_inputs"]["imgs"]
                cam = ego[agent]["batch_merged_cam_inputs"]
                if cam.get("image_semantic_gts") is None:
                    print(f"  skip {agent}: no image_semantic_gts")
                    continue
                gt_all = build_semantic_target(cam, tau=1).detach().cpu().numpy()
                p_fg_all = (
                    pred[agent]["heatmap_logits"]
                    .softmax(1)[:, 1]
                    .detach()
                    .cpu()
                    .numpy()
                )
                view = pick_view(gt_all, prefer_fg=True)
                if float((gt_all[view] > 0).mean()) < 1e-4:
                    print(f"  skip {agent}: empty semantic GT")
                    continue
                gt_ids = gt_all[view]
                p_fg = p_fg_all[view]
                rgb = denorm_rgb(flatten_views(imgs)[view])
                save_heatmap_row(
                    heat_dir / f"{agent}_{heat_counts[agent]:02d}.png",
                    rgb,
                    gt_ids,
                    p_fg,
                    f"train#{batch_i} {agent} view{view}  "
                    f"gt_fg={(gt_ids > 0).mean():.3f} pred_fg@0.3={(p_fg >= 0.3).mean():.3f}",
                )
                heat_counts[agent] += 1
                n_heat += 1
            if all(heat_counts[a] >= opt.n_heatmap_per_agent for a in ("vehicle", "rsu", "drone")):
                break
            if batch_i >= 40:
                break
    del train_loader, train_dataset

    summary = {
        "n_depth_panels": n_depth,
        "n_heatmap_panels": n_heat,
        "heatmap_per_agent": dict(heat_counts),
        "depth_mae_by_agent": {k: round(float(np.mean(v)), 3) for k, v in depth_stats.items()},
        "live_architecture": live_arch,
    }
    (out_dir / "vis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"depth panels: {depth_dir}")
    print(f"heatmap panels: {heat_dir}")


if __name__ == "__main__":
    main()

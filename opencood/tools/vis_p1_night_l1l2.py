# -*- coding: utf-8 -*-
"""Night test L1/L2 pre-lighten comparison. Matches dell camera_utils formulas."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.loss.gaussian_p1_semantic_loss import softmax_focal_loss
from opencood.models.gaussian_modules_0822.heatmap.metrics import compute_heatmap_metrics
from opencood.models.gaussian_modules_0822.heatmap.target import binary_objectness_target
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.tools import train_utils
from opencood.tools.vis_p1_scene_sample import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEMANTIC_COLORS,
    denorm_rgb,
    flatten_imgs,
    overlay_seg,
    scene_from_path,
    source_semantic,
)

SCENE = "2025_05_10_19_54_35"
AGENTS = ("vehicle", "rsu", "drone")
CONDITIONS = ("raw", "L1", "L2")
TAU = 0.3
L1_GAIN, L1_GAMMA = 0.35, 1.8
L2_GAIN, L2_GAMMA = 0.25, 2.0
FRAME_IDS = (3736, 3843, 3950, 4056, 4163)

CKPT = ROOT / (
    "opencood/logs/airv2x_gaussian_p1_joint/"
    "objectness_focal_2026_08_27_11_05_32/net_epoch26.pth"
)
YAML = ROOT / (
    "opencood/logs/airv2x_gaussian_p1_joint/"
    "objectness_focal_2026_08_27_11_05_32/config.yaml"
)
OUT = Path(
    "/mnt/home/suyi/visualization/objectness_focal_2026_08_27_11_05_32/"
    "epoch26/l1l2_night_2025_05_10_19_54_35"
)


def apply_gain_gamma(rgb: torch.Tensor, gain: float, gamma: float) -> torch.Tensor:
    """``clip((I/gain)**(1/gamma), 0, 1)`` on display RGB channels."""
    out = rgb.clone()
    scaled = out[:3].clamp(0.0, 1.0) / float(gain)
    scaled = torch.where(scaled < 0, scaled.new_full(scaled.shape, 1.0e-8), scaled)
    out[:3] = torch.clamp(torch.pow(scaled, 1.0 / float(gamma)), 0.0, 1.0)
    return out


def apply_condition_nchw(imgs: torch.Tensor, condition: str) -> torch.Tensor:
    """Denorm → optional L1/L2 → ImageNet. Depth channel unchanged."""
    leading = tuple(imgs.shape[:-3])
    channels, height, width = (int(imgs.shape[-3]), int(imgs.shape[-2]), int(imgs.shape[-1]))
    flat = imgs.reshape(-1, channels, height, width).clone()
    mean = flat.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = flat.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    display = (flat[:, :3] * std + mean).clamp(0.0, 1.0)
    if condition == "L1":
        for idx in range(int(display.shape[0])):
            display[idx] = apply_gain_gamma(display[idx], L1_GAIN, L1_GAMMA)
    elif condition == "L2":
        for idx in range(int(display.shape[0])):
            display[idx] = apply_gain_gamma(display[idx], L2_GAIN, L2_GAMMA)
    elif condition != "raw":
        raise ValueError(condition)
    flat[:, :3] = (display - mean) / std
    return flat.reshape(*leading, channels, height, width)


def save_comparison(
    path: Path,
    rows: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    title: str,
) -> None:
    """3 conditions × RGB | GT | p_fg | pred@0.3."""
    fig, axes = plt.subplots(len(rows), 4, figsize=(15.2, 3.2 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, 0)
    for row_i, (name, rgb, gt_ids, p_fg) in enumerate(rows):
        axes[row_i, 0].imshow(rgb)
        axes[row_i, 0].set_title(f"{name} RGB")
        axes[row_i, 1].imshow(overlay_seg(rgb, gt_ids))
        axes[row_i, 1].set_title("GT overlay")
        im = axes[row_i, 2].imshow(p_fg, cmap="magma", vmin=0.0, vmax=1.0)
        axes[row_i, 2].set_title("p_fg")
        plt.colorbar(im, ax=axes[row_i, 2], fraction=0.046)
        axes[row_i, 3].imshow(overlay_seg(rgb, (p_fg >= TAU).astype(np.int64)))
        axes[row_i, 3].set_title(f"pred @{TAU:g}")
        for ax in axes[row_i]:
            ax.axis("off")
    handles = [
        mpatches.Patch(color=SEMANTIC_COLORS[0] / 255.0, label="bg"),
        mpatches.Patch(color=SEMANTIC_COLORS[1] / 255.0, label="fg"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pick_view(gt_np: np.ndarray, sources: List[str], p_np: np.ndarray) -> int:
    """Same view for raw/L1/L2: prefer GT fg, else p_fg."""
    scores = []
    for view_i in range(gt_np.shape[0]):
        if sources[view_i] != "empty":
            scores.append(float(gt_np[view_i].mean()))
        else:
            scores.append(float(p_np[view_i].mean()) * 1.0e-3)
    return int(np.argmax(scores)) if scores else 0


def main() -> None:
    """Run raw/L1/L2 on the five night-test frames already visualized."""
    hypes = yaml_utils.load_yaml(str(YAML), None)
    hypes["validate_dir"] = hypes["test_dir"]
    hypes["train"] = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print(f"Creating model from {CKPT}", flush=True)
    model = train_utils.create_model(hypes)
    raw = torch.load(str(CKPT), map_location="cpu")
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    print(f"Building test dataset from {hypes['validate_dir']}", flush=True)
    dataset = build_dataset(hypes, visualize=False, train=False)

    report: Dict[str, Any] = {
        "ckpt": str(CKPT),
        "scene": SCENE,
        "l1": "inverse C2: (I/0.35)**(1/1.8) then ImageNet",
        "l2": "inverse C3: (I/0.25)**(1/2.0) then ImageNet",
        "raw": "no pre-lighten (inference-only; dataset L2 gate not in this repo)",
        "protocol": "denorm display RGB → L1/L2 → ImageNet; depth kept",
        "frames": [],
    }
    OUT.mkdir(parents=True, exist_ok=True)

    for idx in FRAME_IDS:
        sample = dataset[idx]
        batch = dataset.collate_batch_test([sample])
        batch = train_utils.to_device(batch, device)
        ego = batch["ego"]
        present = present_camera_agents(ego)
        meta = ego.get("metadata_path_list", [""])[0]
        scene = scene_from_path(meta) if meta else SCENE
        ts = str(ego.get("timestamp_key_list", [idx])[0])
        print(f"idx={idx} scene={scene} ts={ts} agents={present}", flush=True)
        raw_imgs = {
            agent: ego[agent]["batch_merged_cam_inputs"]["imgs"].clone()
            for agent in AGENTS
            if agent in present
        }
        if not raw_imgs:
            continue
        cond_pred: Dict[str, Dict[str, Any]] = {}
        with torch.no_grad():
            for cond in CONDITIONS:
                for agent, tensor in raw_imgs.items():
                    ego[agent]["batch_merged_cam_inputs"]["imgs"] = apply_condition_nchw(
                        tensor, cond
                    )
                cond_pred[cond] = model(ego)

        frame_rec: Dict[str, Any] = {
            "idx": int(idx),
            "timestamp": ts,
            "scene": scene,
            "agents": {},
        }
        for agent in AGENTS:
            if agent not in raw_imgs:
                continue
            cam = ego[agent]["batch_merged_cam_inputs"]
            semantic, sources = source_semantic(ego, agent, cam)
            target = binary_objectness_target(semantic.long(), tau=1)
            gt_np = target.detach().cpu().numpy()
            rows = []
            agent_rec: Dict[str, Any] = {"gt_sources": sources, "conditions": {}}
            view = None
            for cond in CONDITIONS:
                logits = cond_pred[cond][agent]["heatmap_logits"]
                p_fg = torch.softmax(logits, dim=1)[:, 1]
                p_np = p_fg.detach().cpu().numpy()
                if view is None:
                    view = pick_view(gt_np, sources, p_np)
                rgb = denorm_rgb(flatten_imgs(ego[agent]["batch_merged_cam_inputs"]["imgs"])[view])
                if cond == "raw":
                    rgb = denorm_rgb(flatten_imgs(raw_imgs[agent])[view])
                else:
                    rgb = denorm_rgb(
                        flatten_imgs(apply_condition_nchw(raw_imgs[agent], cond))[view]
                    )
                hm_loss = float(softmax_focal_loss(logits, target.to(logits.device), gamma=2.0).item())
                metrics = compute_heatmap_metrics(logits, target.to(logits.device))
                rows.append((cond, rgb, gt_np[view], p_np[view]))
                agent_rec["conditions"][cond] = {
                    "view": int(view),
                    "heatmap_loss": hm_loss,
                    "p_fg_mean": float(p_np.mean()),
                    "p_fg_view": float(p_np[view].mean()),
                    "pred_fg@0.3": float((p_np >= TAU).mean()),
                    "pred_fg@0.3_view": float((p_np[view] >= TAU).mean()),
                    "gt_fg_ratio": float(gt_np.mean()),
                    "gt_fg_view": float(gt_np[view].mean()),
                    "rgb_mean": float(rgb.mean() / 255.0),
                    "recall@0.3": float(metrics.get("recall@0.3", 0.0)),
                    "precision@0.3": float(metrics.get("precision@0.3", 0.0)),
                    "mean_p_fg_gt": float(metrics.get("mean_p_fg_gt", 0.0)),
                }
            agent_rec["view"] = int(view)
            agent_rec["gt_source_view"] = sources[view]
            png = OUT / f"{idx}_{ts}_{agent}.png"
            title = (
                f"ep26 {scene} ts={ts} {agent} view{view} gt={sources[view]}  "
                + "  |  ".join(
                    f"{c}: p={agent_rec['conditions'][c]['p_fg_view']:.3f} "
                    f"pred@0.3={agent_rec['conditions'][c]['pred_fg@0.3_view']:.3f} "
                    f"rec={agent_rec['conditions'][c]['recall@0.3']:.3f}"
                    for c in CONDITIONS
                )
            )
            save_comparison(png, rows, title)
            agent_rec["png"] = str(png)
            frame_rec["agents"][agent] = agent_rec
            print(f"  saved {png.name}", flush=True)
        report["frames"].append(frame_rec)

    (OUT / "meta.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"out": str(OUT), "n_frames": len(report["frames"])}, indent=2))


if __name__ == "__main__":
    main()

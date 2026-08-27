# -*- coding: utf-8 -*-
"""Evaluate a joint P1 checkpoint on val, focusing on camera-z depth.

Heatmap val metrics are reported but unusable if semantic GT is missing
(dataset fills all-zero maps). Depth MAE/RMSE use the camera depth channel
and do not depend on semantic GT.
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
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.gaussian_modules_0822.heatmap.target import build_semantic_target
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.models.gaussian_modules_0822.lss.target import (
    build_depth_class_target,
    extract_camera_z_gt,
)
from opencood.tools import train_utils
from opencood.tools.train import setup_dataloader
from opencood.tools.train_gaussian_p1 import (
    _unwrap_model,
    build_p1_criteria,
    compute_p1_metrics,
    build_depth_targets,
)
from opencood.loss.gaussian_p1_depth_loss import GaussianP1DepthLoss

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    """CLI for P1 val evaluation."""
    parser = argparse.ArgumentParser(description="Eval joint P1 checkpoint")
    parser.add_argument("--hypes_yaml", "-y", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epoch", type=int, default=9)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--worker", type=int, default=4)
    parser.add_argument("--vis_n", type=int, default=12)
    parser.add_argument("--max_batches", type=int, default=0, help="0 = full split")
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="val has no semantic bins; use train for Focal/heatmap metrics",
    )
    return parser.parse_args()


def load_epoch_checkpoint(model: torch.nn.Module, model_dir: str, epoch: int) -> str:
    """Load ``net_epoch{epoch}.pth`` into ``model``."""
    path = os.path.join(model_dir, f"net_epoch{epoch}.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    raw = torch.load(path, map_location="cpu")
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"Loaded {path}")
    if missing:
        print("missing", missing)
    if unexpected:
        print("unexpected", unexpected)
    return path


def denormalize_rgb(chw: torch.Tensor) -> np.ndarray:
    """ImageNet-normalized CHW tensor → HWC uint8."""
    rgb = chw.detach().float().cpu().numpy()[:3]
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def save_depth_panel(
    out_path: Path,
    rgb: np.ndarray,
    gt_z: np.ndarray,
    pred_z: np.ndarray,
    d_min: float,
    d_max: float,
    title: str,
) -> None:
    """RGB | GT z | pred z | abs error for one camera crop."""
    err = np.abs(pred_z - gt_z)
    valid = (gt_z >= d_min) & (gt_z <= d_max)
    vmax = max(d_max, 1.0)
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    im1 = axes[1].imshow(gt_z, cmap="turbo", vmin=d_min, vmax=vmax)
    axes[1].set_title("GT camera-z (m)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(pred_z, cmap="turbo", vmin=d_min, vmax=vmax)
    axes[2].set_title("Pred z_mean (m)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    im3 = axes[3].imshow(np.where(valid, err, np.nan), cmap="magma", vmin=0.0, vmax=min(20.0, vmax))
    axes[3].set_title("|pred-GT| valid (m)")
    fig.colorbar(im3, ax=axes[3], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    """Run val depth evaluation for one checkpoint."""
    opt = parse_args()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    dummy = Namespace(
        distributed=False,
        worker=opt.worker,
        gpu_id=opt.gpu_id,
        amp=False,
        tag="eval",
        model_dir="",
        hypes_yaml=opt.hypes_yaml,
    )
    use_train = opt.split == "train"
    print(f"Building {opt.split} dataset (train={use_train})...")
    dataset = build_dataset(hypes, visualize=False, train=use_train)
    data_loader = setup_dataloader(dataset, hypes, dummy, is_train=False)
    print(f"{opt.split} samples={len(dataset)} batches={len(data_loader)}")

    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = train_utils.create_model(hypes)
    load_epoch_checkpoint(model, opt.model_dir, opt.epoch)
    model.to(device)
    model.eval()
    core = _unwrap_model(model)
    heatmap_criterion, depth_criterion = build_p1_criteria(hypes)
    heatmap_criterion.to(device)
    depth_criterion.to(device)

    out_dir = Path(opt.model_dir) / "visualization" / f"eval_epoch{opt.epoch}_{opt.split}"
    vis_dir = out_dir / "panels"
    vis_dir.mkdir(parents=True, exist_ok=True)

    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, float] = defaultdict(float)
    metric_sum: Dict[str, float] = defaultdict(float)
    loss_sum = 0.0
    heatmap_loss_sum = 0.0
    n_batch = 0
    vis_saved = 0

    with torch.no_grad():
        for batch_i, batch_data in enumerate(
            tqdm(data_loader, desc=f"eval {opt.split} epoch{opt.epoch}")
        ):
            if batch_data is None:
                continue
            if opt.max_batches and batch_i >= opt.max_batches:
                break
            batch_data = train_utils.to_device(batch_data, device)
            ego = batch_data["ego"]
            predictions = model(ego)
            heatmap_targets: Dict[str, torch.Tensor] = {}
            for agent_type in present_camera_agents(ego):
                cam_inputs = ego[agent_type]["batch_merged_cam_inputs"]
                heatmap_targets[agent_type] = build_semantic_target(cam_inputs, tau=1)
            depth_targets = build_depth_targets(ego, predictions, core)
            heatmap_loss = heatmap_criterion(predictions, heatmap_targets)
            depth_loss = depth_criterion(predictions, depth_targets, heatmap_targets)
            heatmap_loss_sum += float(heatmap_loss.item())
            loss_sum += float(depth_loss.item())
            n_batch += 1
            batch_metrics = compute_p1_metrics(ego, predictions, heatmap_targets, core)
            for key, value in batch_metrics.items():
                metric_sum[key] += float(value)

            for agent_type, pred in predictions.items():
                camencode = core.frontend.encoders[agent_type]
                d_min = float(camencode.d_min)
                d_max = float(camencode.d_max)
                gt_z = extract_camera_z_gt(ego[agent_type]["batch_merged_cam_inputs"]["imgs"])
                pred_z = pred["depth_z_mean"]
                valid = (gt_z >= d_min) & (gt_z <= d_max)
                n_valid = int(valid.sum().item())
                if n_valid == 0:
                    continue
                diff = pred_z[valid] - gt_z[valid]
                abs_err = diff.abs()
                sq_err = diff * diff
                rel = abs_err / gt_z[valid].clamp_min(1e-3)
                ratio = torch.maximum(
                    pred_z[valid] / gt_z[valid].clamp_min(1e-3),
                    gt_z[valid] / pred_z[valid].clamp_min(1e-3),
                )
                key = agent_type
                fg = heatmap_targets[agent_type].ne(0) & valid
                n_fg = int(fg.sum().item())
                if n_fg > 0:
                    fg_abs = (pred_z[fg] - gt_z[fg]).abs()
                    sums[f"{key}/mae_fg"] += float(fg_abs.sum().item())
                    counts[f"{key}_fg"] += n_fg
                    sums["all/mae_fg"] += float(fg_abs.sum().item())
                    counts["all_fg"] += n_fg
                sums[f"{key}/mae"] += float(abs_err.sum().item())
                sums[f"{key}/mse"] += float(sq_err.sum().item())
                sums[f"{key}/absrel"] += float(rel.sum().item())
                sums[f"{key}/delta1"] += float((ratio < 1.25).float().sum().item())
                counts[key] += n_valid
                sums["all/mae"] += float(abs_err.sum().item())
                sums["all/mse"] += float(sq_err.sum().item())
                sums["all/absrel"] += float(rel.sum().item())
                sums["all/delta1"] += float((ratio < 1.25).float().sum().item())
                counts["all"] += n_valid

                if vis_saved < opt.vis_n:
                    rgb = denormalize_rgb(ego[agent_type]["batch_merged_cam_inputs"]["imgs"][0, 0])
                    save_depth_panel(
                        vis_dir / f"{vis_saved:03d}_{agent_type}.png",
                        rgb,
                        gt_z[0].detach().cpu().numpy(),
                        pred_z[0].detach().cpu().numpy(),
                        d_min,
                        d_max,
                        f"epoch{opt.epoch} batch{batch_i} {agent_type}  "
                        f"MAE={float(abs_err.mean()):.2f}m  D=[{d_min:g},{d_max:g}]",
                    )
                    vis_saved += 1

    trainer_metrics = {
        key: metric_sum[key] / max(n_batch, 1) for key in metric_sum
    }
    report: Dict[str, Any] = {
        "checkpoint": f"net_epoch{opt.epoch}.pth",
        "split": opt.split,
        "n_batch": n_batch,
        "heatmap_ce_mean": heatmap_loss_sum / max(n_batch, 1),
        "depth_focal_mean": loss_sum / max(n_batch, 1),
        "trainer_metrics": {key: round(value, 4) for key, value in trainer_metrics.items()},
        "agents": {},
    }
    for key in ["all", "vehicle", "rsu", "drone"]:
        n = counts.get(key, 0.0)
        if n <= 0:
            continue
        mae = sums[f"{key}/mae"] / n
        rmse = (sums[f"{key}/mse"] / n) ** 0.5
        n_fg = counts.get(f"{key}_fg", 0.0)
        entry = {
            "n_valid_pixels": int(n),
            "mae_m": round(mae, 4),
            "rmse_m": round(rmse, 4),
            "absrel": round(sums[f"{key}/absrel"] / n, 4),
            "delta1": round(sums[f"{key}/delta1"] / n, 4),
        }
        if n_fg > 0:
            entry["n_fg_pixels"] = int(n_fg)
            entry["mae_fg_m"] = round(sums[f"{key}/mae_fg"] / n_fg, 4)
        report["agents"][key] = entry

    out_json = out_dir / "metrics.json"
    out_txt = out_dir / "metrics.txt"
    out_json.write_text(json.dumps(report, indent=2))
    lines = [
        f"P1 {opt.split} eval  epoch={opt.epoch}  batches={n_batch}",
        f"heatmap CE (mean over batches) = {report['heatmap_ce_mean']:.4f}",
        f"depth Focal (mean over batches) = {report['depth_focal_mean']:.4f}",
        "Depth MAE/RMSE: all camera-z in [d_min, d_max]. mae_fg uses semantic != 0.",
        "val semantic GT is missing (all-zero maps) so Focal/mae_fg collapse; train has bins.",
        "",
    ]
    if trainer_metrics:
        lines.append("trainer-style metrics (batch-mean):")
        for key, value in trainer_metrics.items():
            lines.append(f"  {key}: {value:.4f}")
        lines.append("")
    for key, stats in report["agents"].items():
        fg_txt = ""
        if "mae_fg_m" in stats:
            fg_txt = f"  n_fg={stats['n_fg_pixels']:9d}  MAE_fg={stats['mae_fg_m']:.3f} m"
        lines.append(
            f"{key:8s}  n={stats['n_valid_pixels']:9d}  "
            f"MAE={stats['mae_m']:.3f} m  RMSE={stats['rmse_m']:.3f} m  "
            f"AbsRel={stats['absrel']:.3f}  delta1={stats['delta1']:.3f}"
            f"{fg_txt}"
        )
    text = "\n".join(lines) + "\n"
    out_txt.write_text(text)
    print(text)
    print(f"wrote {out_json}")
    print(f"wrote {out_txt}")
    print(f"panels: {vis_dir} ({vis_saved} images)")


if __name__ == "__main__":
    main()

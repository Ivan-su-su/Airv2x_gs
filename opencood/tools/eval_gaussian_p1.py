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
from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    PRIMARY_OBJECTNESS_THRESHOLD,
)
from opencood.models.gaussian_modules_0822.heatmap.target import build_semantic_target
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.models.gaussian_modules_0822.lss.target import (
    extract_camera_z_gt,
    depth_valid_mask,
)
from opencood.tools import train_utils
from opencood.tools.train import setup_dataloader
from opencood.tools.train_gaussian_p1 import (
    _unwrap_model,
    build_p1_criteria,
    compute_p1_metrics,
    build_depth_targets,
    build_depth_valid_masks,
)

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
        "--sample_n",
        type=int,
        default=0,
        help="if >0, randomly sample this many frames (fixed --seed)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="val has no semantic bins; use train for Focal/heatmap metrics",
    )
    parser.add_argument(
        "--fg_tau",
        type=float,
        default=PRIMARY_OBJECTNESS_THRESHOLD,
        help="heatmap p_fg threshold for object-only depth (pred FG, not GT)",
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


def save_object_depth_panel(
    out_path: Path,
    rgb: np.ndarray,
    p_fg: np.ndarray,
    gt_z: np.ndarray,
    pred_z: np.ndarray,
    obj_mask: np.ndarray,
    d_min: float,
    d_max: float,
    title: str,
) -> None:
    """RGB | p_fg | GT z on pred-FG | pred z on pred-FG | abs error."""
    vmax = max(d_max, 1.0)
    gt_show = np.where(obj_mask, gt_z, np.nan)
    pred_show = np.where(obj_mask, pred_z, np.nan)
    err_show = np.where(obj_mask, np.abs(pred_z - gt_z), np.nan)
    fig, axes = plt.subplots(1, 5, figsize=(18.0, 3.4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    im0 = axes[1].imshow(p_fg, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("heatmap p_fg")
    fig.colorbar(im0, ax=axes[1], fraction=0.046)
    im1 = axes[2].imshow(gt_show, cmap="turbo", vmin=d_min, vmax=vmax)
    axes[2].set_title("GT z on pred-FG")
    fig.colorbar(im1, ax=axes[2], fraction=0.046)
    im2 = axes[3].imshow(pred_show, cmap="turbo", vmin=d_min, vmax=vmax)
    axes[3].set_title("Pred z on pred-FG")
    fig.colorbar(im2, ax=axes[3], fraction=0.046)
    im3 = axes[4].imshow(err_show, cmap="magma", vmin=0.0, vmax=min(20.0, vmax))
    axes[4].set_title("|pred-GT| pred-FG (m)")
    fig.colorbar(im3, ax=axes[4], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _add_masked_depth(
    sums: Dict[str, float],
    counts: Dict[str, float],
    prefix: str,
    pred_z: torch.Tensor,
    gt_z: torch.Tensor,
    mask: torch.Tensor,
) -> int:
    """Accumulate MAE/MSE/AbsRel/delta1 on ``mask``. Return valid count."""
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
    sample_ids: List[int] = []
    if int(opt.sample_n) > 0:
        n_all = len(dataset)
        n_keep = min(int(opt.sample_n), n_all)
        rng = np.random.RandomState(int(opt.seed))
        sample_ids = rng.choice(n_all, size=n_keep, replace=False).astype(int).tolist()
        sample_ids.sort()
        collate_fn = dataset.collate_batch_train
        dataset = torch.utils.data.Subset(dataset, sample_ids)
        dataset.collate_batch_train = collate_fn  # type: ignore[attr-defined]
        print(f"sampled {n_keep}/{n_all} frames seed={opt.seed} ids={sample_ids}")
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

    sample_tag = f"_n{opt.sample_n}s{opt.seed}" if int(opt.sample_n) > 0 else ""
    out_dir = Path(opt.model_dir) / "visualization" / f"eval_epoch{opt.epoch}_{opt.split}{sample_tag}"
    vis_dir = out_dir / "panels"
    obj_dir = out_dir / "panels_obj"
    vis_dir.mkdir(parents=True, exist_ok=True)
    obj_dir.mkdir(parents=True, exist_ok=True)
    fg_tau = float(opt.fg_tau)
    vis_limit = int(opt.vis_n)
    if int(opt.sample_n) > 0 and vis_limit < int(opt.sample_n) * 3:
        vis_limit = int(opt.sample_n) * 3

    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, float] = defaultdict(float)
    metric_sum: Dict[str, float] = defaultdict(float)
    loss_sum = 0.0
    heatmap_loss_sum = 0.0
    n_batch = 0
    vis_saved = 0
    obj_saved = 0

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
            depth_valid_masks = build_depth_valid_masks(ego, predictions, core)
            heatmap_loss = heatmap_criterion(predictions, heatmap_targets)
            depth_loss = depth_criterion(
                predictions, depth_targets, heatmap_targets, depth_valid_masks
            )
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
                valid = depth_valid_mask(gt_z, d_min, d_max)
                p_fg = torch.softmax(pred["heatmap_logits"], dim=1)[:, 1]
                pred_fg = p_fg.ge(fg_tau) & valid
                gt_fg = heatmap_targets[agent_type].ne(0) & valid
                _add_masked_depth(sums, counts, agent_type, pred_z, gt_z, valid)
                _add_masked_depth(sums, counts, "all", pred_z, gt_z, valid)
                _add_masked_depth(sums, counts, f"{agent_type}_pred_fg", pred_z, gt_z, pred_fg)
                _add_masked_depth(sums, counts, "all_pred_fg", pred_z, gt_z, pred_fg)
                _add_masked_depth(sums, counts, f"{agent_type}_gt_fg", pred_z, gt_z, gt_fg)
                _add_masked_depth(sums, counts, "all_gt_fg", pred_z, gt_z, gt_fg)

                if vis_saved < vis_limit:
                    rgb = denormalize_rgb(ego[agent_type]["batch_merged_cam_inputs"]["imgs"][0, 0])
                    gt_np = gt_z[0].detach().cpu().numpy()
                    pred_np = pred_z[0].detach().cpu().numpy()
                    p_np = p_fg[0].detach().cpu().numpy()
                    obj_np = pred_fg[0].detach().cpu().numpy().astype(bool)
                    n_view = int(pred_fg[0].sum().item())
                    mae_view = (
                        float((pred_z[0][pred_fg[0]] - gt_z[0][pred_fg[0]]).abs().mean())
                        if n_view > 0
                        else float("nan")
                    )
                    save_depth_panel(
                        vis_dir / f"{batch_i:02d}_{agent_type}.png",
                        rgb,
                        gt_np,
                        pred_np,
                        d_min,
                        d_max,
                        f"epoch{opt.epoch} idx={sample_ids[batch_i] if sample_ids else batch_i} "
                        f"{agent_type} MAE={float((pred_z[valid] - gt_z[valid]).abs().mean()):.2f}m "
                        f"D=[{d_min:g},{d_max:g}]",
                    )
                    vis_saved += 1
                    save_object_depth_panel(
                        obj_dir / f"{batch_i:02d}_{agent_type}.png",
                        rgb,
                        p_np,
                        gt_np,
                        pred_np,
                        obj_np,
                        d_min,
                        d_max,
                        f"epoch{opt.epoch} idx={sample_ids[batch_i] if sample_ids else batch_i} "
                        f"{agent_type} object-only tau={fg_tau:g} n={n_view} MAE={mae_view:.2f}m",
                    )
                    obj_saved += 1

    trainer_metrics = {
        key: metric_sum[key] / max(n_batch, 1) for key in metric_sum
    }
    report: Dict[str, Any] = {
        "checkpoint": f"net_epoch{opt.epoch}.pth",
        "split": opt.split,
        "n_batch": n_batch,
        "sample_n": int(opt.sample_n),
        "seed": int(opt.seed),
        "sample_ids": sample_ids,
        "fg_tau": fg_tau,
        "heatmap_ce_mean": heatmap_loss_sum / max(n_batch, 1),
        "depth_focal_mean": loss_sum / max(n_batch, 1),
        "trainer_metrics": {key: round(value, 4) for key, value in trainer_metrics.items()},
        "agents": {},
        "object_only_pred_fg": {},
        "object_only_gt_fg": {},
    }

    def _pack_agent(prefix: str) -> Dict[str, Any]:
        n = counts.get(prefix, 0.0)
        if n <= 0:
            return {}
        return {
            "n_valid_pixels": int(n),
            "mae_m": round(sums[f"{prefix}/mae"] / n, 4),
            "rmse_m": round((sums[f"{prefix}/mse"] / n) ** 0.5, 4),
            "absrel": round(sums[f"{prefix}/absrel"] / n, 4),
            "delta1": round(sums[f"{prefix}/delta1"] / n, 4),
        }

    for key in ["all", "vehicle", "rsu", "drone"]:
        full = _pack_agent(key)
        if full:
            report["agents"][key] = full
        pred_fg_stats = _pack_agent(f"{key}_pred_fg" if key != "all" else "all_pred_fg")
        if pred_fg_stats:
            report["object_only_pred_fg"][key] = pred_fg_stats
        gt_fg_stats = _pack_agent(f"{key}_gt_fg" if key != "all" else "all_gt_fg")
        if gt_fg_stats:
            report["object_only_gt_fg"][key] = gt_fg_stats

    out_json = out_dir / "metrics.json"
    out_txt = out_dir / "metrics.txt"
    out_json.write_text(json.dumps(report, indent=2))

    def _fmt_block(title: str, block: Dict[str, Any]) -> List[str]:
        rows = [title]
        for key, stats in block.items():
            rows.append(
                f"{key:8s}  n={stats['n_valid_pixels']:9d}  "
                f"MAE={stats['mae_m']:.3f} m  RMSE={stats['rmse_m']:.3f} m  "
                f"AbsRel={stats['absrel']:.3f}  delta1={stats['delta1']:.3f}"
            )
        return rows

    lines = [
        f"P1 {opt.split} eval  epoch={opt.epoch}  batches={n_batch}  fg_tau={fg_tau:g}",
        f"heatmap CE (mean over batches) = {report['heatmap_ce_mean']:.4f}",
        f"depth Focal (mean over batches) = {report['depth_focal_mean']:.4f}",
        "Full-image: all camera-z in [d_min, d_max].",
        f"Object-only pred-FG: heatmap p_fg >= {fg_tau:g} AND in-range depth.",
        "Object-only GT-FG: semantic != 0 AND in-range (val semantic is all-zero).",
        "",
    ]
    if trainer_metrics:
        lines.append("trainer-style metrics (batch-mean):")
        for key, value in trainer_metrics.items():
            lines.append(f"  {key}: {value:.4f}")
        lines.append("")
    lines.extend(_fmt_block("full-image depth", report["agents"]))
    lines.append("")
    lines.extend(_fmt_block(f"object-only depth (pred FG, tau={fg_tau:g})", report["object_only_pred_fg"]))
    if report["object_only_gt_fg"]:
        lines.append("")
        lines.extend(_fmt_block("object-only depth (GT semantic FG)", report["object_only_gt_fg"]))
    text = "\n".join(lines) + "\n"
    out_txt.write_text(text)
    print(text)
    print(f"wrote {out_json}")
    print(f"wrote {out_txt}")
    print(f"panels: {vis_dir} ({vis_saved} images)")
    print(f"object panels: {obj_dir} ({obj_saved} images)")


if __name__ == "__main__":
    main()

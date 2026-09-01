# -*- coding: utf-8 -*-
"""Val-only P1 vis: camera-z histograms + RGB/pred heatmap (no semantic GT)."""

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
from PIL import Image
from tqdm import tqdm

root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
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
AGENTS = ("vehicle", "rsu", "drone")


def parse_args() -> argparse.Namespace:
    """CLI for val depth histograms and heatmap overlays."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--hypes_yaml", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epoch", type=int, default=7)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--worker", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=80)
    parser.add_argument("--n_depth", type=int, default=12)
    parser.add_argument("--n_heatmap_per_agent", type=int, default=4)
    return parser.parse_args()


def load_epoch(model: torch.nn.Module, model_dir: str, epoch: int) -> str:
    """Load ``net_epoch{epoch}.pth``."""
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


def denorm_rgb(chw: torch.Tensor) -> np.ndarray:
    """ImageNet CHW → HWC uint8."""
    rgb = chw.detach().float().cpu().numpy()[:3]
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def overlay_pred(rgb: np.ndarray, p_fg: np.ndarray, alpha: float = 0.5, tau: float = 0.3) -> np.ndarray:
    """Nearest-upsample p_fg>=tau onto RGB. Background stays RGB-only."""
    pred_ids = (p_fg >= tau).astype(np.int64)
    color = SEMANTIC_COLORS[pred_ids]
    color_up = np.array(
        Image.fromarray(color).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)
    )
    ids_up = np.array(
        Image.fromarray(pred_ids.astype(np.uint8)).resize(
            (rgb.shape[1], rgb.shape[0]), Image.NEAREST
        )
    )
    mix = rgb.astype(np.float32)
    fg = ids_up > 0
    mix[fg] = (1.0 - alpha) * mix[fg] + alpha * color_up[fg].astype(np.float32)
    return np.clip(mix, 0, 255).astype(np.uint8)


def flatten_imgs(imgs: torch.Tensor) -> torch.Tensor:
    """``[B,V,C,H,W]`` → ``[N,C,H,W]``."""
    if imgs.dim() == 5:
        batch, views = imgs.shape[:2]
        return imgs.reshape(batch * views, *imgs.shape[2:])
    return imgs


def pick_view(maps: np.ndarray, prefer_fg: bool) -> int:
    """Choose a camera crop."""
    if maps.ndim != 3:
        return 0
    scores = []
    for idx in range(maps.shape[0]):
        if prefer_fg:
            scores.append(float(maps[idx].mean()))
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
    """RGB | GT z | pred z | error."""
    err = np.abs(pred_z - gt_z)
    valid = (gt_z >= d_min) & (gt_z <= d_max)
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.3))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    im1 = axes[1].imshow(gt_z, cmap="turbo", vmin=d_min, vmax=d_max)
    axes[1].set_title("GT camera-z (m)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(pred_z, cmap="turbo", vmin=d_min, vmax=d_max)
    axes[2].set_title("Pred z_mean (m)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    im3 = axes[3].imshow(np.where(valid, err, np.nan), cmap="magma", vmin=0.0, vmax=min(20.0, d_max))
    axes[3].set_title("|err| valid (m)")
    fig.colorbar(im3, ax=axes[3], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_heatmap_row(
    path: Path,
    rgb: np.ndarray,
    p_fg: np.ndarray,
    title: str,
) -> None:
    """RGB | p_fg overlay. No GT."""
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    im1 = axes[1].imshow(p_fg, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("p_fg")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[2].imshow(overlay_pred(rgb, p_fg))
    axes[2].set_title("Pred overlay @0.3")
    for ax in axes:
        ax.axis("off")
    handles = [
        mpatches.Patch(color=SEMANTIC_COLORS[i] / 255.0, label=f"{i}:{SEMANTIC_NAMES[i]}")
        for i in range(len(SEMANTIC_NAMES))
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def percentile_dict(values: np.ndarray) -> Dict[str, float]:
    """Compact distribution summary."""
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": round(float(values.mean()), 3),
        "std": round(float(values.std()), 3),
        "p10": round(float(np.percentile(values, 10)), 3),
        "p50": round(float(np.percentile(values, 50)), 3),
        "p90": round(float(np.percentile(values, 90)), 3),
        "min": round(float(values.min()), 3),
        "max": round(float(values.max()), 3),
    }


def main() -> None:
    """Run val histograms, depth panels, and heatmap overlays without GT."""
    opt = parse_args()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    dummy = Namespace(
        distributed=False,
        worker=opt.worker,
        gpu_id=opt.gpu_id,
        amp=False,
        tag="vis_val",
        model_dir="",
        hypes_yaml=opt.hypes_yaml,
    )
    print("Building val dataset...")
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = setup_dataloader(dataset, hypes, dummy, is_train=False)
    print(f"val samples={len(dataset)} batches={len(loader)}")

    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = train_utils.create_model(hypes)
    load_epoch(model, opt.model_dir, opt.epoch)
    model.to(device)
    model.eval()
    core = _unwrap_model(model)

    out_dir = Path(opt.model_dir) / "visualization" / f"eval_epoch{opt.epoch}_val"
    depth_dir = out_dir / "depth"
    heat_dir = out_dir / "heatmap"
    depth_dir.mkdir(parents=True, exist_ok=True)
    heat_dir.mkdir(parents=True, exist_ok=True)

    n_bins = 60
    gt_hists: Dict[str, np.ndarray] = {}
    pred_hists: Dict[str, np.ndarray] = {}
    edges: Dict[str, np.ndarray] = {}
    ranges: Dict[str, List[float]] = {}
    for agent in AGENTS:
        encoder = core.frontend.encoders[agent]
        d_min, d_max = float(encoder.d_min), float(encoder.d_max)
        ranges[agent] = [d_min, d_max]
        edges[agent] = np.linspace(d_min, d_max, n_bins + 1)
        gt_hists[agent] = np.zeros(n_bins, dtype=np.float64)
        pred_hists[agent] = np.zeros(n_bins, dtype=np.float64)

    abs_sum = defaultdict(float)
    n_valid = defaultdict(int)
    n_all = defaultdict(int)
    n_below = defaultdict(int)
    n_above = defaultdict(int)
    pred_fg_sum = defaultdict(float)
    pred_fg_n = defaultdict(int)

    n_depth = 0
    heat_counts: Dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for batch_i, batch in enumerate(tqdm(loader, desc=f"val vis epoch{opt.epoch}")):
            if batch is None:
                continue
            if opt.max_batches and batch_i >= opt.max_batches:
                break
            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            pred = model(ego)
            for agent in present_camera_agents(ego):
                encoder = core.frontend.encoders[agent]
                d_min, d_max = float(encoder.d_min), float(encoder.d_max)
                imgs = ego[agent]["batch_merged_cam_inputs"]["imgs"]
                gt_z = extract_camera_z_gt(imgs)
                pred_z = pred[agent]["depth_z_mean"]
                p_fg = pred[agent]["heatmap_logits"].softmax(1)[:, 1]
                valid = (gt_z >= d_min) & (gt_z <= d_max)
                n_all[agent] += int(gt_z.numel())
                n_below[agent] += int((gt_z < d_min).sum().item())
                n_above[agent] += int((gt_z > d_max).sum().item())
                n_ok = int(valid.sum().item())
                n_valid[agent] += n_ok
                if n_ok > 0:
                    abs_sum[agent] += float((pred_z[valid] - gt_z[valid]).abs().sum().item())
                    gt_np = gt_z[valid].detach().cpu().numpy()
                    pred_np = pred_z[valid].detach().cpu().numpy()
                    gt_hists[agent] += np.histogram(gt_np, bins=edges[agent])[0]
                    pred_hists[agent] += np.histogram(pred_np, bins=edges[agent])[0]

                p_fg_np = p_fg.detach().cpu().numpy()
                pred_fg_sum[agent] += float((p_fg_np >= 0.3).mean())
                pred_fg_n[agent] += 1

                if n_depth < opt.n_depth:
                    gt_all = gt_z.detach().cpu().numpy()
                    pred_all = pred_z.detach().cpu().numpy()
                    view = pick_view(gt_all, prefer_fg=False)
                    rgb = denorm_rgb(flatten_imgs(imgs)[view])
                    valid_v = (gt_all[view] >= d_min) & (gt_all[view] <= d_max)
                    mae = (
                        float(np.abs(pred_all[view][valid_v] - gt_all[view][valid_v]).mean())
                        if valid_v.any()
                        else float("nan")
                    )
                    save_depth_row(
                        depth_dir / f"{n_depth:02d}_{agent}.png",
                        rgb,
                        gt_all[view],
                        pred_all[view],
                        d_min,
                        d_max,
                        f"val#{batch_i} {agent} view{view} MAE={mae:.2f}m "
                        f"D=[{d_min:g},{d_max:g}]",
                    )
                    n_depth += 1

                if heat_counts[agent] < opt.n_heatmap_per_agent:
                    view = pick_view(p_fg_np, prefer_fg=True)
                    rgb = denorm_rgb(flatten_imgs(imgs)[view])
                    save_heatmap_row(
                        heat_dir / f"{agent}_{heat_counts[agent]:02d}.png",
                        rgb,
                        p_fg_np[view],
                        f"val#{batch_i} {agent} view{view}  "
                        f"pred_fg@0.3={(p_fg_np[view] >= 0.3).mean():.3f}",
                    )
                    heat_counts[agent] += 1

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.5), sharex=False)
    dist_report: Dict[str, Any] = {}
    for row, agent in enumerate(AGENTS):
        d_min, d_max = ranges[agent]
        centers = 0.5 * (edges[agent][:-1] + edges[agent][1:])
        gt_c = gt_hists[agent]
        pred_c = pred_hists[agent]
        gt_pdf = gt_c / max(gt_c.sum(), 1.0)
        pred_pdf = pred_c / max(pred_c.sum(), 1.0)
        axes[row].plot(centers, gt_pdf, label="GT z", color="tab:blue", lw=1.8)
        axes[row].plot(centers, pred_pdf, label="pred z_mean", color="tab:orange", lw=1.8)
        axes[row].set_xlim(d_min, d_max)
        axes[row].set_ylabel("mass")
        axes[row].set_title(
            f"{agent}  range=[{d_min:g},{d_max:g}]  "
            f"n_valid={n_valid[agent]}  "
            f"MAE={abs_sum[agent] / max(n_valid[agent], 1):.2f}m"
        )
        axes[row].legend(frameon=False)
        axes[row].grid(alpha=0.25)
        n_tot = max(n_all[agent], 1)
        dist_report[agent] = {
            "d_min": d_min,
            "d_max": d_max,
            "n_pixels": n_all[agent],
            "n_valid": n_valid[agent],
            "frac_valid": round(n_valid[agent] / n_tot, 4),
            "frac_below_dmin": round(n_below[agent] / n_tot, 4),
            "frac_above_dmax": round(n_above[agent] / n_tot, 4),
            "mae_valid_m": round(abs_sum[agent] / max(n_valid[agent], 1), 4),
            "pred_fg_ratio": round(pred_fg_sum[agent] / max(pred_fg_n[agent], 1), 4),
        }
    axes[-1].set_xlabel("camera-z (m)")
    fig.suptitle(f"Val camera-z distribution  epoch={opt.epoch}  batches={opt.max_batches}", fontsize=12)
    fig.tight_layout()
    hist_path = out_dir / "depth_hist_gt_vs_pred.png"
    fig.savefig(hist_path, dpi=140)
    plt.close(fig)

    report = {
        "checkpoint": f"net_epoch{opt.epoch}.pth",
        "split": "val",
        "n_batch": opt.max_batches,
        "n_depth_panels": n_depth,
        "heatmap_panels": dict(heat_counts),
        "agents": dist_report,
    }
    (out_dir / "val_depth_dist.json").write_text(json.dumps(report, indent=2))
    lines = [
        f"P1 val vis  epoch={opt.epoch}  batches={opt.max_batches}",
        "Depth uses camera-z GT (val has no semantic bins).",
        "Heatmap panels are RGB + pred overlay only (no GT).",
        "",
    ]
    for agent, stats in dist_report.items():
        lines.append(
            f"{agent:8s}  valid={stats['n_valid']:9d}/{stats['n_pixels']:9d} "
            f"({100 * stats['frac_valid']:.1f}%)  "
            f"<dmin={100 * stats['frac_below_dmin']:.1f}%  "
            f">dmax={100 * stats['frac_above_dmax']:.1f}%  "
            f"MAE={stats['mae_valid_m']:.3f} m  "
            f"pred_fg={stats['pred_fg_ratio']:.3f}"
        )
        share = " ".join(
            f"{name}:{100 * frac:.1f}%"
            for name, frac in stats["pred_class_share"].items()
            if frac > 0
        )
        lines.append(f"         pred classes  {share}")
    text = "\n".join(lines) + "\n"
    (out_dir / "val_depth_dist.txt").write_text(text)
    print(text)
    print(f"hist: {hist_path}")
    print(f"depth panels: {depth_dir}")
    print(f"heatmap panels: {heat_dir}")


if __name__ == "__main__":
    main()

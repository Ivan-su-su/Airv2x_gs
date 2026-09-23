# -*- coding: utf-8 -*-
"""Sampled evaluation of Griffin P1 checkpoints: heatmap visualizations + MAE.

For each requested checkpoint:
* Sample ``--per-scene`` frames per official val scene (uniformly spaced with
  a fixed seed, no shuffle), always drawn from the *val-only* frames (i.e.
  excluding the 30% that were moved into train when
  ``val_frame_train_fraction`` > 0).
* Report vehicle depth MAE / MedAE on those sampled frames and also
  separately on the moved-to-train val frames ("被抽取数据集") to measure
  memorization vs generalization.
* Save heatmap visualization panels (vehicle 4 views + drone bottom) for the
  sampled frames.

Usage:
  python opencood/tools/eval_griffin_p1_sampled.py \
      --config opencood/hypes_yaml/griffin/camera/gaussian_p1/griffin_gaussian_p1_v45_aug.yaml \
      --ckpt opencood/logs/griffin_gaussian_p1_v45_scratch50_aug/epoch_005.pth \
      --out opencood/logs/griffin_gaussian_p1_v45_scratch50_aug/eval_sampled/epoch_005 \
      --per-scene 5 --gpu 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opencood.data_utils.datasets.griffin_p1_dataset import (
    GriffinP1Dataset,
    _pick_val_frames_per_scene_for_train,
    _scene_frames,
    _scene_name,
    _load_scene_infos,
    _split_scene_names,
)
from opencood.models.griffin_gaussian_p1 import GriffinGaussianP1
from opencood.utils.camera_utils import imagenet_normalize_display_rgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="opencood/hypes_yaml/griffin/camera/gaussian_p1/griffin_gaussian_p1_v45_aug.yaml",
    )
    p.add_argument(
        "--ckpt",
        default="opencood/logs/griffin_gaussian_p1_v45_scratch50_aug/best.pth",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output dir; default: <ckpt_dir>/eval_sampled/<ckpt_stem>",
    )
    p.add_argument("--per-scene", type=int, default=5)
    p.add_argument("--sample-seed", type=int, default=20260923)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--fg-threshold", type=float, default=0.3)
    return p.parse_args()


def denorm_rgb(img_t: torch.Tensor) -> np.ndarray:
    """De-normalize a [3,H,W] ImageNet tensor back to HxWx3 RGB uint8."""
    x = img_t.clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = x * std + mean
    x = x.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (x * 255.0).astype(np.uint8)


def save_heatmap_panel(
    path: Path,
    rgb: np.ndarray,
    gt_fg: np.ndarray,
    p_fg: np.ndarray,
    threshold: float,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[1].imshow(gt_fg, cmap="gray")
    axes[1].set_title("GT foreground")
    im = axes[2].imshow(p_fg, vmin=0.0, vmax=1.0, cmap="jet")
    axes[2].set_title("Pred p_fg")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    axes[3].imshow(p_fg >= threshold, cmap="gray")
    axes[3].set_title(f"Pred FG @{threshold:g}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    epoch = ckpt.get("epoch")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model = GriffinGaussianP1(cfg["model"]).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[eval] ckpt={ckpt_path.name} epoch(0-based)={epoch}")

    # Val dataset (with moved frames excluded) for sampling.
    val_ds = GriffinP1Dataset(cfg["dataset"], "val")
    ds_cfg = cfg["dataset"]

    # Reconstruct which frames were moved into train (the "被抽取" set).
    split_json = Path(ds_cfg["split_json"])
    scenes = _load_scene_infos(val_ds.vehicle_root)
    val_names = _split_scene_names(split_json, "val")
    frac = float(ds_cfg.get("val_frame_train_fraction", 0.0))
    seed = int(ds_cfg.get("val_frame_split_seed", 20260921))
    moved_frames = set()
    if frac > 0.0:
        picked = _pick_val_frames_per_scene_for_train(scenes, val_names, frac, seed)
        for frames in picked.values():
            moved_frames.update(frames)

    # Per official val scene, sample `per_scene` frames from VAL-ONLY frames.
    rng = np.random.RandomState(args.sample_seed)
    val_frame_set = set(val_ds.frames)
    scene_to_val_frames: Dict[str, List[str]] = {name: [] for name in val_names}
    for i, scene in enumerate(scenes):
        raw_name = _scene_name(scene, i)
        for name in val_names:
            if raw_name == name or name.endswith("-" + raw_name):
                scene_to_val_frames[name].extend(
                    [f for f in _scene_frames(scene) if f in val_frame_set]
                )
                break

    sampled: List[str] = []
    for name in sorted(scene_to_val_frames):
        pool = scene_to_val_frames[name]
        if not pool:
            continue
        k = min(args.per_scene, len(pool))
        idx = sorted(int(j) for j in rng.choice(len(pool), size=k, replace=False))
        sampled.extend(pool[j] for j in idx)
        print(f"[sample] {name}: {k} frames")

    frame_to_index = {f: i for i, f in enumerate(val_ds.frames)}
    sampled_idx = [frame_to_index[f] for f in sampled]

    # Also collect moved ("被抽取") frames present in val_ds? They are NOT in
    # val_ds; build a temporary train-side dataset to fetch them if needed.
    moved_in_ds: List[int] = []
    if moved_frames:
        train_ds = GriffinP1Dataset(cfg["dataset"], "train")
        train_frame_to_index = {f: i for i, f in enumerate(train_ds.frames)}
        moved_in_ds = [train_frame_to_index[f] for f in sorted(moved_frames)]

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else ckpt_path.parent / "eval_sampled" / ckpt_path.stem
    )
    heat_dir = out_dir / "heatmap"
    heat_dir.mkdir(parents=True, exist_ok=True)

    ddiscr = cfg["model"]["vehicle"]["cam"]["grid_conf"]["ddiscr"]
    d_min, d_max = float(ddiscr[0]), float(ddiscr[1])

    def eval_frames(ds, indices, tag, visualize=False) -> Dict[str, float]:
        err_all: List[float] = []
        per_view_err = [[], [], [], []]
        n_valid_cells = 0
        with torch.no_grad():
            for k, ds_i in enumerate(indices):
                batch = ds[ds_i]
                v_imgs = batch["vehicle"]["imgs"].to(device)  # [4,3,H,W]
                # Full forward on single-frame dicts, same as training.
                model_in = {
                    "vehicle": {"imgs": v_imgs.unsqueeze(0)},
                    "drone": {
                        "imgs": batch["drone"]["imgs"].unsqueeze(0).to(device)
                    },
                }
                pred = model(model_in)
                # NOTE: model forward flattens views -> logits [B*V, C, H, W],
                # depth_z_mean [B*V, H, W] with B=1, V=4 here.
                z_pred = pred["vehicle"]["depth_z_mean"]  # [4,45,80]
                z_gt = batch["vehicle"]["camera_z_gt"].to(device)  # [4,45,80]
                valid = batch["vehicle"]["depth_valid"].to(device)
                valid = (
                    valid
                    & torch.isfinite(z_gt)
                    & (z_gt >= d_min)
                    & (z_gt <= d_max)
                )
                err = torch.abs(z_pred - z_gt)
                for v in range(4):
                    m = valid[v]
                    if m.any():
                        err_all.extend(err[v][m].cpu().numpy().tolist())
                        per_view_err[v].extend(err[v][m].cpu().numpy().tolist())
                n_valid_cells += int(valid.sum().item())

                if visualize:
                    p_fg = torch.softmax(pred["vehicle"]["heatmap_logits"], dim=1)[:, 1]  # [4,45,80]
                    gt_hm = batch["vehicle"]["heatmap_target"]
                    d_p_fg = torch.softmax(pred["drone"]["heatmap_logits"], dim=1)[:, 1]  # [1,90,160]
                    d_gt = batch["drone"]["heatmap_target"]
                    frame = batch["frame_id"]
                    for v in range(4):
                        save_heatmap_panel(
                            heat_dir / f"{frame}_vehicle_{['front','back','left','right'][v]}.png",
                            denorm_rgb(batch["vehicle"]["imgs"][v]),
                            gt_hm[v].bool().numpy(),
                            p_fg[v].cpu().numpy(),
                            args.fg_threshold,
                            f"{ckpt_path.stem} ep{epoch} {frame} veh/{['front','back','left','right'][v]}",
                        )
                    save_heatmap_panel(
                        heat_dir / f"{frame}_drone_bottom.png",
                        denorm_rgb(batch["drone"]["imgs"][0]),
                        d_gt[0].bool().numpy(),
                        d_p_fg[0].cpu().numpy(),
                        args.fg_threshold,
                        f"{ckpt_path.stem} ep{epoch} {frame} drone/bottom",
                    )

        err_np = np.asarray(err_all, dtype=np.float64) if err_all else np.zeros(0)
        report: Dict[str, float] = {
            "n_frames": len(indices),
            "n_valid_cells": n_valid_cells,
            "mae_m": float(err_np.mean()) if err_np.size else float("nan"),
            "medae_m": float(np.median(err_np)) if err_np.size else float("nan"),
            "p90_m": float(np.percentile(err_np, 90)) if err_np.size else float("nan"),
        }
        for v, cam in enumerate(("front", "back", "left", "right")):
            e = np.asarray(per_view_err[v])
            report[f"mae_{cam}"] = float(e.mean()) if e.size else float("nan")
        print(
            f"[{tag}] frames={report['n_frames']} valid_cells={report['n_valid_cells']} "
            f"MAE={report['mae_m']:.4f}m MedAE={report['medae_m']:.4f}m P90={report['p90_m']:.4f}m"
        )
        return report

    sampled_report = eval_frames(val_ds, sampled_idx, "sampled-val", visualize=True)
    moved_report = (
        eval_frames(train_ds, moved_in_ds, "moved-to-train(被抽取)")
        if moved_in_ds
        else {}
    )

    result = {
        "ckpt": str(ckpt_path),
        "epoch_0based": epoch,
        "per_scene": args.per_scene,
        "sample_seed": args.sample_seed,
        "sampled_val": sampled_report,
        "moved_to_train": moved_report,
        "sampled_frames": sampled,
    }
    out_file = out_dir / "sampled_metrics.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[done] metrics -> {out_file}")
    print(f"[done] heatmap panels -> {heat_dir}")


if __name__ == "__main__":
    main()

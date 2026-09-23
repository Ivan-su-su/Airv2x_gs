# -*- coding: utf-8 -*-
"""
Exact Griffin P1 evaluator for baseline branch.

Matches:
  opencood/data_utils/datasets/griffin_p1_dataset.py
  opencood/models/griffin_gaussian_p1.py
  opencood/tools/train_griffin_p1.py
  opencood/hypes_yaml/griffin/camera/gaussian_p1/griffin_gaussian_p1_v45.yaml

Evaluates:
  1) Vehicle + drone heatmap visualization and metrics.
  2) Vehicle depth error on predicted foreground:
       (p_fg >= threshold) & depth_valid
  3) Vehicle global depth error:
       all valid sparse LiDAR-projected depth cells
  4) Extra diagnostic: vehicle depth error on GT foreground.

Important:
  Griffin P1 DOES NOT learn drone depth. Drone only has a V90 heatmap branch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opencood.data_utils.datasets.griffin_p1_dataset import GriffinP1Dataset
from opencood.models.griffin_gaussian_p1 import GriffinGaussianP1
from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    OBJECTNESS_THRESHOLDS,
    PRIMARY_OBJECTNESS_THRESHOLD,
    compute_heatmap_metrics,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate Griffin Gaussian P1")
    p.add_argument(
        "--config",
        default="opencood/hypes_yaml/griffin/camera/gaussian_p1/"
                "griffin_gaussian_p1_v45.yaml",
    )
    p.add_argument(
        "--model_dir",
        default="/mnt/home/suyi/AirV2X-Perception_gaussian/"
                "opencood/logs/griffin_gaussian_p1_v45",
    )
    p.add_argument(
        "--checkpoint",
        default="best.pth",
        help="best.pth / latest.pth / epoch_015.pth, or an absolute path",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument(
        "--fg_threshold",
        type=float,
        default=PRIMARY_OBJECTNESS_THRESHOLD,
        help="predicted foreground threshold used for requested pred-FG depth MAE",
    )
    p.add_argument("--max_batches", type=int, default=0, help="0 = full val")
    p.add_argument("--vis_samples", type=int, default=12)
    p.add_argument("--out_dir", default="")
    return p.parse_args()


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"bad yaml: {path}")
    return cfg


def move_nested(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_nested(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move_nested(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_nested(v, device) for v in x)
    return x


def flatten_views(x: torch.Tensor) -> torch.Tensor:
    """[B,V,...] -> [B*V,...], same as train_griffin_p1.py."""
    if x.dim() < 3:
        raise ValueError(f"expected [B,V,...], got {tuple(x.shape)}")
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def resolve_checkpoint(model_dir: str, checkpoint: str) -> Path:
    p = Path(checkpoint)
    if not p.is_absolute():
        p = Path(model_dir) / checkpoint
    return p.expanduser().resolve()


def load_checkpoint(
    model: GriffinGaussianP1,
    path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    ckpt = torch.load(str(path), map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise KeyError(
            f"{path} is not a Griffin trainer checkpoint with model_state_dict"
        )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"[checkpoint] loaded {path}")
    print(f"[checkpoint] epoch={ckpt.get('epoch')}")
    if "val_metrics" in ckpt:
        vm = ckpt["val_metrics"] or {}
        print(
            "[checkpoint val] "
            f"loss={vm.get('loss_total', float('nan')):.4f} "
            f"veh_F1@.3={vm.get('vehicle_hm/f1@0.3', float('nan')):.4f} "
            f"drone_F1@.3={vm.get('drone_hm/f1@0.3', float('nan')):.4f} "
            f"veh_depth_MAE={vm.get('vehicle_depth_mae_m', float('nan')):.4f}m"
        )
    return ckpt


def denorm_rgb(chw: torch.Tensor) -> np.ndarray:
    x = chw.detach().float().cpu().numpy()
    x = np.transpose(x[:3], (1, 2, 0))
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = np.clip(x, 0.0, 1.0)
    return x


class ErrorAccumulator:
    def __init__(self, keep_values: bool = True) -> None:
        self.n = 0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.values: List[torch.Tensor] = []
        self.keep_values = keep_values

    def add(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        mask = mask.bool()
        n = int(mask.sum().item())
        if n == 0:
            return
        err = (pred[mask] - gt[mask]).detach().float()
        ae = err.abs()
        self.n += n
        self.abs_sum += float(ae.sum().item())
        self.sq_sum += float((err * err).sum().item())
        if self.keep_values:
            self.values.append(ae.cpu())

    def report(self) -> Dict[str, float]:
        if self.n == 0:
            return {
                "n": 0,
                "mae_m": float("nan"),
                "rmse_m": float("nan"),
                "median_ae_m": float("nan"),
                "p90_ae_m": float("nan"),
                "p95_ae_m": float("nan"),
            }
        out = {
            "n": int(self.n),
            "mae_m": self.abs_sum / self.n,
            "rmse_m": math.sqrt(self.sq_sum / self.n),
        }
        if self.values:
            v = torch.cat(self.values)
            out.update({
                "median_ae_m": float(v.median().item()),
                "p90_ae_m": float(torch.quantile(v, 0.90).item()),
                "p95_ae_m": float(torch.quantile(v, 0.95).item()),
            })
        else:
            out.update({
                "median_ae_m": float("nan"),
                "p90_ae_m": float("nan"),
                "p95_ae_m": float("nan"),
            })
        return out


class HeatmapGlobalAccumulator:
    """Pixel-weighted global objectness metrics at the same fixed thresholds."""
    def __init__(self) -> None:
        self.counts = {
            float(t): {"tp": 0, "fp": 0, "fn": 0}
            for t in OBJECTNESS_THRESHOLDS
        }
        self.gt_fg = 0
        self.n_pixels = 0
        self.gt_score_sum = 0.0

    def add(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        p_fg = torch.softmax(logits.detach(), dim=1)[:, 1]
        gt = target.ne(0)
        self.gt_fg += int(gt.sum().item())
        self.n_pixels += int(gt.numel())
        if int(gt.sum().item()) > 0:
            self.gt_score_sum += float(p_fg[gt].sum().item())
        for tau in OBJECTNESS_THRESHOLDS:
            pred = p_fg.ge(float(tau))
            c = self.counts[float(tau)]
            c["tp"] += int((pred & gt).sum().item())
            c["fp"] += int((pred & ~gt).sum().item())
            c["fn"] += int((~pred & gt).sum().item())

    def report(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "gt_fg_ratio": self.gt_fg / max(self.n_pixels, 1),
            "mean_p_fg_gt": self.gt_score_sum / max(self.gt_fg, 1),
        }
        for tau, c in self.counts.items():
            p = c["tp"] / max(c["tp"] + c["fp"], 1)
            r = c["tp"] / max(c["tp"] + c["fn"], 1)
            f1 = 2 * p * r / max(p + r, 1e-12)
            tag = f"{tau:g}"
            out[f"precision@{tag}"] = p
            out[f"recall@{tag}"] = r
            out[f"f1@{tag}"] = f1
        return out


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

    axes[1].imshow(gt_fg)
    axes[1].set_title("GT foreground")

    im = axes[2].imshow(p_fg, vmin=0.0, vmax=1.0)
    axes[2].set_title("Pred p_fg")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    axes[3].imshow(p_fg >= threshold)
    axes[3].set_title(f"Pred FG @{threshold:g}")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_vehicle_depth_panel(
    path: Path,
    rgb: np.ndarray,
    gt_z: np.ndarray,
    pred_z: np.ndarray,
    valid: np.ndarray,
    pred_fg: np.ndarray,
    gt_fg: np.ndarray,
    title: str,
) -> None:
    abs_err = np.abs(pred_z - gt_z)
    pred_fg_valid = valid & pred_fg
    gt_fg_valid = valid & gt_fg

    fig, axes = plt.subplots(1, 6, figsize=(21, 3.6))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")

    im1 = axes[1].imshow(np.where(valid, gt_z, np.nan), vmin=0.0, vmax=50.0)
    axes[1].set_title("Sparse GT z")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(pred_z, vmin=0.0, vmax=50.0)
    axes[2].set_title("Pred z_mean")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(np.where(valid, abs_err, np.nan))
    axes[3].set_title("|err| global valid")
    fig.colorbar(im3, ax=axes[3], fraction=0.046)

    im4 = axes[4].imshow(np.where(pred_fg_valid, abs_err, np.nan))
    axes[4].set_title("|err| predicted FG")
    fig.colorbar(im4, ax=axes[4], fraction=0.046)

    im5 = axes[5].imshow(np.where(gt_fg_valid, abs_err, np.nan))
    axes[5].set_title("|err| GT FG")
    fig.colorbar(im5, ax=axes[5], fraction=0.046)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # Exact same val dataset class as training.
    val_ds = GriffinP1Dataset(cfg["dataset"], "val")
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(cfg["train"].get("batch_size", 2))
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )
    print(
        f"[val] frames={len(val_ds)} batches={len(val_loader)} "
        f"batch_size={batch_size}"
    )

    # Exact same model class as training.
    model = GriffinGaussianP1(cfg["model"]).to(device)
    ckpt_path = resolve_checkpoint(args.model_dir, args.checkpoint)
    ckpt = load_checkpoint(model, ckpt_path, device)
    model.eval()
    model.assert_train_eval_state(False)

    tag = ckpt_path.stem
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path(args.model_dir).expanduser().resolve()
             / "eval_griffin_p1"
             / tag
    )
    heat_dir = out_dir / "heatmap"
    depth_dir = out_dir / "depth"
    heat_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap:
    # - batch_mean reproduces train_griffin_p1.validate behavior.
    # - global_acc gives pixel-weighted whole-validation metrics.
    hm_batch_sums: Dict[str, Dict[str, float]] = {
        "vehicle": defaultdict(float),
        "drone": defaultdict(float),
    }
    hm_batch_count = {"vehicle": 0, "drone": 0}
    hm_global = {
        "vehicle": HeatmapGlobalAccumulator(),
        "drone": HeatmapGlobalAccumulator(),
    }

    depth_global = ErrorAccumulator()
    depth_pred_fg = ErrorAccumulator()
    depth_gt_fg = ErrorAccumulator()

    # Also retain the trainer-style batch-mean MAE for exact comparison with
    # the printed ~6.8-7.0 m validation logs.
    trainer_depth_mae_sum = 0.0
    trainer_depth_mae_batches = 0

    pred_fg_valid_cells = 0
    gt_fg_valid_cells = 0
    all_valid_cells = 0

    vis_saved = 0

    ddiscr = cfg["model"]["vehicle"]["cam"]["grid_conf"]["ddiscr"]
    d_min, d_max = float(ddiscr[0]), float(ddiscr[1])

    with torch.no_grad():
        for batch_i, batch in enumerate(tqdm(val_loader, desc="griffin val eval")):
            if args.max_batches > 0 and batch_i >= args.max_batches:
                break

            batch = move_nested(batch, device)
            pred = model(batch)

            v_hm = flatten_views(batch["vehicle"]["heatmap_target"]).long()
            d_hm = flatten_views(batch["drone"]["heatmap_target"]).long()

            # Heatmap metrics exactly via the same training helper.
            for agent, target in (("vehicle", v_hm), ("drone", d_hm)):
                logits = pred[agent]["heatmap_logits"]
                metrics = compute_heatmap_metrics(logits, target)
                hm_batch_count[agent] += 1
                for k, v in metrics.items():
                    hm_batch_sums[agent][k] += float(v)
                hm_global[agent].add(logits, target)

            # Vehicle only: Griffin has no learned drone depth.
            z_gt = flatten_views(batch["vehicle"]["camera_z_gt"]).float()
            valid0 = flatten_views(batch["vehicle"]["depth_valid"]).bool()
            z_pred = pred["vehicle"]["depth_z_mean"]

            valid = (
                valid0
                & torch.isfinite(z_gt)
                & torch.isfinite(z_pred)
                & (z_gt >= d_min)
                & (z_gt <= d_max)
            )

            v_p_fg = torch.softmax(
                pred["vehicle"]["heatmap_logits"], dim=1
            )[:, 1]
            pred_fg = v_p_fg.ge(float(args.fg_threshold))
            gt_fg = v_hm.ne(0)

            pred_fg_valid = valid & pred_fg
            gt_fg_valid = valid & gt_fg

            depth_global.add(z_pred, z_gt, valid)
            depth_pred_fg.add(z_pred, z_gt, pred_fg_valid)
            depth_gt_fg.add(z_pred, z_gt, gt_fg_valid)

            n_valid = int(valid.sum().item())
            all_valid_cells += n_valid
            pred_fg_valid_cells += int(pred_fg_valid.sum().item())
            gt_fg_valid_cells += int(gt_fg_valid.sum().item())

            # Same as train_griffin_p1.py:
            # mean per batch, then validation averages those batch scalars.
            if n_valid > 0:
                mae_batch = float(
                    torch.abs(z_pred - z_gt)[valid].mean().item()
                )
                trainer_depth_mae_sum += mae_batch
                trainer_depth_mae_batches += 1

            # Visualization: save complete frames. Vehicle has four views;
            # drone has one bottom view.
            if vis_saved < args.vis_samples:
                B = int(batch["vehicle"]["imgs"].shape[0])
                for b in range(B):
                    if vis_saved >= args.vis_samples:
                        break

                    frame_raw = batch.get("frame_id", [f"b{batch_i}_{b}"] * B)
                    if isinstance(frame_raw, (list, tuple)):
                        frame_id = str(frame_raw[b])
                    else:
                        frame_id = str(frame_raw)

                    # Vehicle indices in flattened output.
                    v_views = int(batch["vehicle"]["imgs"].shape[1])
                    for v in range(v_views):
                        idx = b * v_views + v
                        rgb = denorm_rgb(batch["vehicle"]["imgs"][b, v])
                        gt_np = v_hm[idx].detach().cpu().numpy().astype(bool)
                        p_np = v_p_fg[idx].detach().cpu().numpy()
                        save_heatmap_panel(
                            heat_dir / f"{vis_saved:03d}_{frame_id}_vehicle_v{v}.png",
                            rgb,
                            gt_np,
                            p_np,
                            args.fg_threshold,
                            f"{frame_id} vehicle view={v}",
                        )

                        zgt_np = z_gt[idx].detach().cpu().numpy()
                        zpred_np = z_pred[idx].detach().cpu().numpy()
                        valid_np = valid[idx].detach().cpu().numpy()
                        pfg_np = pred_fg[idx].detach().cpu().numpy()
                        gtfg_np = gt_fg[idx].detach().cpu().numpy()
                        save_vehicle_depth_panel(
                            depth_dir / f"{vis_saved:03d}_{frame_id}_vehicle_v{v}.png",
                            rgb,
                            zgt_np,
                            zpred_np,
                            valid_np,
                            pfg_np,
                            gtfg_np,
                            f"{frame_id} vehicle view={v}",
                        )

                    # Drone bottom heatmap.
                    d_views = int(batch["drone"]["imgs"].shape[1])
                    for v in range(d_views):
                        idx = b * d_views + v
                        d_prob = torch.softmax(
                            pred["drone"]["heatmap_logits"], dim=1
                        )[:, 1]
                        rgb = denorm_rgb(batch["drone"]["imgs"][b, v])
                        gt_np = d_hm[idx].detach().cpu().numpy().astype(bool)
                        p_np = d_prob[idx].detach().cpu().numpy()
                        save_heatmap_panel(
                            heat_dir / f"{vis_saved:03d}_{frame_id}_drone_v{v}.png",
                            rgb,
                            gt_np,
                            p_np,
                            args.fg_threshold,
                            f"{frame_id} drone-bottom",
                        )

                    vis_saved += 1

    batch_mean_hm: Dict[str, Dict[str, float]] = {}
    global_hm: Dict[str, Dict[str, float]] = {}
    for agent in ("vehicle", "drone"):
        n = max(hm_batch_count[agent], 1)
        batch_mean_hm[agent] = {
            k: v / n for k, v in hm_batch_sums[agent].items()
        }
        global_hm[agent] = hm_global[agent].report()

    report = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch_zero_based": ckpt.get("epoch"),
        "checkpoint_val_metrics": ckpt.get("val_metrics"),
        "val_frames": len(val_ds),
        "evaluated_batches": hm_batch_count["vehicle"],
        "fg_threshold": float(args.fg_threshold),
        "heatmap_batch_mean_same_as_trainer": batch_mean_hm,
        "heatmap_global_pixel_weighted": global_hm,
        "vehicle_depth": {
            "note": (
                "GT depth is sparse LiDAR projected into 45x80 camera cells; "
                "'global' means all valid projected LiDAR cells, not dense image depth."
            ),
            "valid_depth_range_m": [d_min, d_max],
            "global_valid": depth_global.report(),
            "predicted_foreground_valid": depth_pred_fg.report(),
            "gt_foreground_valid": depth_gt_fg.report(),
            "trainer_style_batch_mean_mae_m": (
                trainer_depth_mae_sum / max(trainer_depth_mae_batches, 1)
            ),
            "n_global_valid": all_valid_cells,
            "n_pred_fg_valid": pred_fg_valid_cells,
            "n_gt_fg_valid": gt_fg_valid_cells,
            "pred_fg_share_of_valid": (
                pred_fg_valid_cells / max(all_valid_cells, 1)
            ),
            "gt_fg_share_of_valid": (
                gt_fg_valid_cells / max(all_valid_cells, 1)
            ),
        },
        "drone_depth": {
            "evaluated": False,
            "reason": (
                "griffin_gaussian_p1.py has no drone depth head; "
                "downstream uses analytic bottom-camera geometry."
            ),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    v_b = batch_mean_hm["vehicle"]
    d_b = batch_mean_hm["drone"]
    dep_g = report["vehicle_depth"]["global_valid"]
    dep_p = report["vehicle_depth"]["predicted_foreground_valid"]
    dep_t = report["vehicle_depth"]["gt_foreground_valid"]

    lines = [
        "=" * 88,
        "Griffin P1 validation",
        f"checkpoint: {ckpt_path}",
        f"checkpoint epoch (0-based): {ckpt.get('epoch')}",
        "",
        "[Heatmap - trainer-compatible batch mean]",
        f"vehicle F1@0.3 = {v_b.get('f1@0.3', float('nan')):.4f}  "
        f"P={v_b.get('precision@0.3', float('nan')):.4f}  "
        f"R={v_b.get('recall@0.3', float('nan')):.4f}",
        f"drone   F1@0.3 = {d_b.get('f1@0.3', float('nan')):.4f}  "
        f"P={d_b.get('precision@0.3', float('nan')):.4f}  "
        f"R={d_b.get('recall@0.3', float('nan')):.4f}",
        "",
        "[Vehicle depth - sparse LiDAR projected GT]",
        f"trainer-style batch-mean MAE = "
        f"{report['vehicle_depth']['trainer_style_batch_mean_mae_m']:.4f} m",
        f"GLOBAL valid      : n={dep_g['n']}  "
        f"MAE={dep_g['mae_m']:.4f} m  RMSE={dep_g['rmse_m']:.4f} m  "
        f"MedAE={dep_g['median_ae_m']:.4f} m  P90={dep_g['p90_ae_m']:.4f} m",
        f"PRED-FG valid @{args.fg_threshold:g}: n={dep_p['n']}  "
        f"MAE={dep_p['mae_m']:.4f} m  RMSE={dep_p['rmse_m']:.4f} m  "
        f"MedAE={dep_p['median_ae_m']:.4f} m  P90={dep_p['p90_ae_m']:.4f} m",
        f"GT-FG valid       : n={dep_t['n']}  "
        f"MAE={dep_t['mae_m']:.4f} m  RMSE={dep_t['rmse_m']:.4f} m  "
        f"MedAE={dep_t['median_ae_m']:.4f} m  P90={dep_t['p90_ae_m']:.4f} m",
        "",
        f"pred-FG share among valid depth cells = "
        f"{100.0 * report['vehicle_depth']['pred_fg_share_of_valid']:.2f}%",
        f"GT-FG share among valid depth cells   = "
        f"{100.0 * report['vehicle_depth']['gt_fg_share_of_valid']:.2f}%",
        "",
        "[Drone depth]",
        "Not evaluated: Griffin P1 drone branch has heatmap only; no learned depth head.",
        "",
        f"heatmap images: {heat_dir}",
        f"depth images:   {depth_dir}",
        f"metrics:        {out_dir / 'metrics.json'}",
        "=" * 88,
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    (out_dir / "metrics.txt").write_text(text)


if __name__ == "__main__":
    main()

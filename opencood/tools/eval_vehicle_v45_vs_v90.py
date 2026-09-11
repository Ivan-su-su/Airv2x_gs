# -*- coding: utf-8 -*-
"""TEST compare: Vehicle V45 P1 vs V90 P1 2x2-downsampled to 45x80.

Heatmap GT is projected 3D boxes (TEST has no SAM3). Depth GT is optical-axis
camera-z at stride-8 cell centers. Visualizes 5 frames per TEST scene.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    OBJECTNESS_THRESHOLDS,
    PRIMARY_OBJECTNESS_THRESHOLD,
)
from opencood.models.gaussian_modules_0822.heatmap.target import binary_objectness_target
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.models.gaussian_modules_0822.lss.target import (
    depth_valid_mask,
    extract_camera_z_gt,
)
from opencood.models.gaussian_modules_0822.vehicle_v45_p1 import VehicleV45P1
from opencood.tools import train_utils
from opencood.tools.eval_gaussian_p1 import denormalize_rgb
from opencood.tools.train_gaussian_p1 import _unwrap_model
from opencood.tools.vis_p1_scene_sample import evenly_sample, scene_from_path
from opencood.tools.vis_test_heatmap_recall import flatten_imgs, paint_agent_box_maps

AGENT = "vehicle"
STRIDE_V45 = 8
STRIDE_V90 = 4
FAR_Z_M = 20.0
DEFAULT_V90_CKPT = (
    "/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
    "airv2x_gaussian_p1_vehicle_branch/vehicle_branch_2026_09_01_22_09_38/"
    "net_epoch15.pth"
)
SEMANTIC_RED = np.array([220, 20, 60], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    """CLI for V45 vs V90-downsampled TEST compare."""
    parser = argparse.ArgumentParser(description="V45 vs V90-ds TEST compare")
    parser.add_argument(
        "--model_dir",
        default=(
            "/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
            "vehicle_v45_p1/v45_p1_2gpu_2026_09_09_23_25_42"
        ),
    )
    parser.add_argument("--epoch", type=int, default=0, help="0 = latest complete")
    parser.add_argument("--v90_ckpt", default=DEFAULT_V90_CKPT)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--frames_per_scene", type=int, default=5)
    parser.add_argument("--fg_tau", type=float, default=PRIMARY_OBJECTNESS_THRESHOLD)
    parser.add_argument(
        "--out_root",
        default="/mnt/home/suyi/visualization/vehicle_v45_p1",
    )
    return parser.parse_args()


def latest_complete_epoch(model_dir: Path) -> Tuple[int, Path]:
    """Pick the newest ``net_epochN.pth`` (ignore mid-epoch ``_q*`` dumps)."""
    found: List[Tuple[int, Path]] = []
    for path in model_dir.glob("net_epoch*.pth"):
        rest = path.stem[len("net_epoch") :]
        if "_q" in rest:
            continue
        found.append((int(rest), path))
    if not found:
        raise FileNotFoundError(f"no complete epoch under {model_dir}")
    return max(found, key=lambda item: item[0])


def load_mix_keys(path: Path) -> Set[Tuple[str, str]]:
    """Held-out TEST timestamps that were mixed into TRAIN."""
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text())
    return {
        (str(row["scene"]), str(row["timestamp"]))
        for row in payload.get("records", [])
    }


def sample_plan_heldout(
    dataset: Any,
    frames_per_scene: int,
    mix_keys: Set[Tuple[str, str]],
) -> List[Tuple[str, int, str, bool]]:
    """Evenly sample ``frames_per_scene`` non-mix timestamps per TEST scene."""
    plan: List[Tuple[str, int, str, bool]] = []
    for scene_i, end in enumerate(dataset.len_record):
        start = 0 if scene_i == 0 else int(dataset.len_record[scene_i - 1])
        scene_db = dataset.scenario_database[scene_i]
        first_cav = next(iter(scene_db.values()))
        ts0 = dataset.return_timestamp_key(scene_db, 0)
        scene = scene_from_path(first_cav[ts0]["metadata_path"])
        candidates: List[Tuple[int, str, bool]] = []
        for local in range(int(end) - start):
            timestamp = str(dataset.return_timestamp_key(scene_db, local))
            mixed = (scene, timestamp) in mix_keys
            candidates.append((start + local, timestamp, mixed))
        heldout = [row for row in candidates if not row[2]]
        pool = heldout if heldout else candidates
        chosen = evenly_sample(0, len(pool), frames_per_scene)
        for local_i in chosen:
            idx, timestamp, mixed = pool[local_i]
            plan.append((scene, idx, timestamp, mixed))
    return plan


def load_v45_state(model: torch.nn.Module, path: Path) -> None:
    """Load a VehicleV45P1 checkpoint, stripping optional DDP prefixes."""
    raw = torch.load(str(path), map_location="cpu")
    state = (
        raw["model_state_dict"]
        if isinstance(raw, dict) and "model_state_dict" in raw
        else raw
    )
    first = next(iter(state))
    if first.startswith("module."):
        state = {key[7:]: value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"Loaded V45 {path}")
    if missing:
        print("missing", missing)
    if unexpected:
        print("unexpected", unexpected)


def forward_v90_native(
    model: VehicleV45P1, imgs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Production V90 P1 path: HighResFusion on native R2 (90x160)."""
    r2, f45 = model.extract_r2_f45(imgs)
    fused = model.highres(r2, f45)
    heatmap_logits = model.heatmap_head(fused)
    depth_logits = model.depth_head(fused)
    depth_z_mean, _ = model.depth_moments(depth_logits)
    return heatmap_logits, depth_z_mean


def downsample_2x2(map_hw: torch.Tensor) -> torch.Tensor:
    """Average-pool ``[N,H,W]`` by 2 so 90x160 → 45x80."""
    return F.avg_pool2d(map_hw.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)


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


def pick_view(gt_fg: np.ndarray, p_fg: np.ndarray) -> int:
    """Camera with most box-GT pixels, else highest mean p_fg."""
    scores = gt_fg.reshape(gt_fg.shape[0], -1).sum(axis=1)
    if float(scores.max()) <= 0:
        scores = p_fg.reshape(p_fg.shape[0], -1).mean(axis=1)
    return int(np.argmax(scores))


class HeatmapAcc:
    """Pixel TP/FP/FN plus p_fg on GT at several thresholds."""

    def __init__(self) -> None:
        self.tp: Dict[float, float] = defaultdict(float)
        self.fp: Dict[float, float] = defaultdict(float)
        self.fn: Dict[float, float] = defaultdict(float)
        self.p_fg_gt_sum = 0.0
        self.p_fg_gt_n = 0.0
        self.gt_n = 0.0
        self.total_n = 0.0
        self.n_views = 0
        self.n_empty = 0

    def add(self, p_fg: torch.Tensor, gt_fg: torch.Tensor) -> None:
        """Accumulate one ``[N,H,W]`` batch."""
        gt = gt_fg.bool()
        n_gt = int(gt.sum().item())
        self.gt_n += n_gt
        self.total_n += int(gt.numel())
        self.n_views += int(gt.shape[0])
        if n_gt == 0:
            self.n_empty += 1
        else:
            self.p_fg_gt_sum += float(p_fg[gt].sum().item())
            self.p_fg_gt_n += n_gt
        for tau in OBJECTNESS_THRESHOLDS:
            pred = p_fg.ge(float(tau))
            self.tp[tau] += float((pred & gt).sum().item())
            self.fp[tau] += float((pred & ~gt).sum().item())
            self.fn[tau] += float((~pred & gt).sum().item())

    def pack(self) -> Dict[str, Any]:
        """Serialize pixel-pooled metrics."""
        out: Dict[str, Any] = {
            "n_views": int(self.n_views),
            "n_gt_empty_views": int(self.n_empty),
            "gt_fg_ratio": round(self.gt_n / max(self.total_n, 1.0), 6),
            "mean_p_fg_gt": round(
                self.p_fg_gt_sum / max(self.p_fg_gt_n, 1.0), 6
            ),
        }
        for tau in OBJECTNESS_THRESHOLDS:
            tp = self.tp[tau]
            fp = self.fp[tau]
            fn = self.fn[tau]
            rec = tp / max(tp + fn, 1.0)
            prec = tp / max(tp + fp, 1.0)
            f1 = (2.0 * prec * rec) / max(prec + rec, 1.0e-12)
            tag = f"{tau:g}"
            out[f"recall@{tag}"] = round(rec, 4)
            out[f"precision@{tag}"] = round(prec, 4)
            out[f"f1@{tag}"] = round(f1, 4)
            out[f"tp@{tag}"] = int(tp)
            out[f"fp@{tag}"] = int(fp)
            out[f"fn@{tag}"] = int(fn)
        return out


class DepthAcc:
    """Masked MAE/RMSE accumulator."""

    def __init__(self) -> None:
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.n = 0.0

    def add(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> None:
        """Accumulate one mask."""
        n = int(mask.sum().item())
        if n <= 0:
            return
        diff = pred[mask] - gt[mask]
        self.abs_sum += float(diff.abs().sum().item())
        self.sq_sum += float((diff * diff).sum().item())
        self.n += n

    def pack(self) -> Dict[str, Any]:
        """Serialize MAE/RMSE."""
        if self.n <= 0:
            return {"mae": None, "rmse": None, "n": 0}
        mae = self.abs_sum / self.n
        rmse = float(np.sqrt(self.sq_sum / self.n))
        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "n": int(self.n),
        }


def save_heatmap_panel(
    path: Path,
    rgb: np.ndarray,
    gt_fg: np.ndarray,
    p_v45: np.ndarray,
    p_v90: np.ndarray,
    title: str,
    tau: float,
) -> None:
    """RGB | GT overlay | V45 45x80 p_fg | V90-ds 45x80 p_fg."""
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.2))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    axes[0, 1].imshow(overlay_fg(rgb, gt_fg))
    axes[0, 1].set_title("GT box overlay (block=8)")
    im1 = axes[0, 2].imshow(p_v45, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[0, 2].set_title(f"V45 p_fg  {p_v45.shape[0]}x{p_v45.shape[1]}")
    fig.colorbar(im1, ax=axes[0, 2], fraction=0.046)
    axes[1, 0].imshow(overlay_fg(rgb, (p_v45 >= tau).astype(np.uint8)))
    axes[1, 0].set_title(f"V45 pred @{tau:g}")
    axes[1, 1].imshow(overlay_fg(rgb, (p_v90 >= tau).astype(np.uint8)))
    axes[1, 1].set_title(f"V90-ds pred @{tau:g}")
    im2 = axes[1, 2].imshow(p_v90, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[1, 2].set_title(f"V90 2x2-avg p_fg  {p_v90.shape[0]}x{p_v90.shape[1]}")
    fig.colorbar(im2, ax=axes[1, 2], fraction=0.046)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_depth_panel(
    path: Path,
    rgb: np.ndarray,
    gt_z: np.ndarray,
    z_v45: np.ndarray,
    z_v90: np.ndarray,
    valid: np.ndarray,
    d_min: float,
    d_max: float,
    title: str,
) -> None:
    """RGB | GT z | V45 z | V90-ds z | abs errors."""
    vmax = max(d_max, 1.0)
    err45 = np.where(valid, np.abs(z_v45 - gt_z), np.nan)
    err90 = np.where(valid, np.abs(z_v90 - gt_z), np.nan)
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.2))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    im0 = axes[0, 1].imshow(
        np.where(valid, gt_z, np.nan),
        cmap="turbo",
        vmin=d_min,
        vmax=vmax,
        interpolation="nearest",
    )
    axes[0, 1].set_title(f"GT camera-z  {gt_z.shape[0]}x{gt_z.shape[1]}")
    fig.colorbar(im0, ax=axes[0, 1], fraction=0.046)
    im1 = axes[0, 2].imshow(
        z_v45, cmap="turbo", vmin=d_min, vmax=vmax, interpolation="nearest"
    )
    axes[0, 2].set_title("V45 z_mean")
    fig.colorbar(im1, ax=axes[0, 2], fraction=0.046)
    im2 = axes[1, 0].imshow(
        z_v90, cmap="turbo", vmin=d_min, vmax=vmax, interpolation="nearest"
    )
    axes[1, 0].set_title("V90 2x2-avg z_mean")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    im3 = axes[1, 1].imshow(err45, cmap="magma", vmin=0.0, vmax=min(20.0, vmax), interpolation="nearest")
    axes[1, 1].set_title("|V45-GT| valid (m)")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    im4 = axes[1, 2].imshow(err90, cmap="magma", vmin=0.0, vmax=min(20.0, vmax), interpolation="nearest")
    axes[1, 2].set_title("|V90-ds-GT| valid (m)")
    fig.colorbar(im4, ax=axes[1, 2], fraction=0.046)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fmt_block(name: str, heat: Dict[str, Any], depth: Dict[str, Any], tau: float) -> str:
    """One-line summary for a model."""
    rec = heat.get(f"recall@{tau:g}")
    prec = heat.get(f"precision@{tau:g}")
    f1 = heat.get(f"f1@{tau:g}")
    mae = depth["valid"]["mae"]
    mae_fg = depth["fg"]["mae"]
    mae_pred = depth["pred_fg"]["mae"]
    mae_far = depth["far"]["mae"]
    return (
        f"{name:10s}  rec@{tau:g}={rec}  prec={prec}  f1={f1}  "
        f"mean_p_fg_gt={heat.get('mean_p_fg_gt')}  "
        f"mae_valid={mae}  mae_fg={mae_fg}  mae_pred_fg={mae_pred}  mae_far={mae_far}"
    )


def main() -> None:
    """Run held-out TEST compare and dump 45x80 panels."""
    opt = parse_args()
    model_dir = Path(opt.model_dir)
    if int(opt.epoch) > 0:
        epoch = int(opt.epoch)
        v45_ckpt = model_dir / f"net_epoch{epoch}.pth"
        if not v45_ckpt.is_file():
            raise FileNotFoundError(v45_ckpt)
    else:
        epoch, v45_ckpt = latest_complete_epoch(model_dir)
    v90_ckpt = Path(opt.v90_ckpt)
    if not v90_ckpt.is_file():
        raise FileNotFoundError(v90_ckpt)
    hypes_path = model_dir / "config.yaml"
    hypes = yaml_utils.load_yaml(str(hypes_path), None)
    hypes["validate_dir"] = hypes["test_dir"]
    hypes["train"] = False
    hypes.pop("extra_train_scenes", None)

    out_dir = Path(opt.out_root) / model_dir.name / f"test_compare_e{epoch}"
    heat_dir = out_dir / "heatmap_45x80"
    depth_dir = out_dir / "depth_45x80"
    heat_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print(f"Creating V45 + V90-native models on {device}", flush=True)
    model_v45 = train_utils.create_model(hypes)
    load_v45_state(model_v45, v45_ckpt)
    model_v45.to(device)
    model_v45.eval()
    core_v45 = _unwrap_model(model_v45)

    model_v90 = VehicleV45P1(hypes["model"]["args"])
    report_v90 = model_v90.load_v90_vehicle_weights(str(v90_ckpt), torch.device("cpu"))
    print(
        f"Loaded V90 vehicle weights from {v90_ckpt}  "
        f"n_loaded={len(report_v90['loaded'])}  new={report_v90['new']}  "
        f"mismatched={len(report_v90['mismatched'])}",
        flush=True,
    )
    model_v90.to(device)
    model_v90.eval()

    print(f"Building TEST dataset from {hypes['validate_dir']}", flush=True)
    dataset = build_dataset(hypes, visualize=False, train=False)
    mix_keys = load_mix_keys(model_dir / "extra_train_mix.json")
    plan = sample_plan_heldout(dataset, opt.frames_per_scene, mix_keys)
    print(
        f"TEST frames={len(dataset)} scenes={len(dataset.len_record)}  "
        f"mix_ts={len(mix_keys)}  sampled={len(plan)}",
        flush=True,
    )

    tau = float(opt.fg_tau)
    heat_v45 = HeatmapAcc()
    heat_v90 = HeatmapAcc()
    heat_v90_native = HeatmapAcc()
    depth_v45 = {
        "valid": DepthAcc(),
        "fg": DepthAcc(),
        "pred_fg": DepthAcc(),
        "far": DepthAcc(),
    }
    depth_v90 = {
        "valid": DepthAcc(),
        "fg": DepthAcc(),
        "pred_fg": DepthAcc(),
        "far": DepthAcc(),
    }
    scene_heat_v45: Dict[str, HeatmapAcc] = defaultdict(HeatmapAcc)
    scene_heat_v90: Dict[str, HeatmapAcc] = defaultdict(HeatmapAcc)
    scene_depth_v45: Dict[str, Dict[str, DepthAcc]] = defaultdict(
        lambda: {k: DepthAcc() for k in depth_v45}
    )
    scene_depth_v90: Dict[str, Dict[str, DepthAcc]] = defaultdict(
        lambda: {k: DepthAcc() for k in depth_v90}
    )
    rows: List[Dict[str, Any]] = []
    d_min = float(core_v45.encoder.d_min)
    d_max = float(core_v45.encoder.d_max)

    with torch.no_grad():
        for sample_i, (scene, idx, timestamp, mixed) in enumerate(plan):
            sample = dataset[idx]
            batch = dataset.collate_batch_test([sample])
            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            meta = ego.get("metadata_path_list", [""])[0]
            scene = scene_from_path(meta) if meta else scene
            present = present_camera_agents(ego)
            print(
                f"[{sample_i + 1}/{len(plan)}] scene={scene} idx={idx} ts={timestamp} "
                f"mix={int(mixed)} agents={present}",
                flush=True,
            )
            if AGENT not in present or AGENT not in ego:
                continue
            cam = ego[AGENT]["batch_merged_cam_inputs"]
            imgs = cam["imgs"]
            pred_v45 = core_v45.forward_features(imgs)
            hm_v90, z_v90 = forward_v90_native(model_v90, imgs)
            p_v45 = torch.softmax(pred_v45["heatmap_logits"], dim=1)[:, 1]
            z_v45 = pred_v45["depth_z_mean"]
            p_v90 = torch.softmax(hm_v90, dim=1)[:, 1]
            p_v90_ds = downsample_2x2(p_v90)
            z_v90_ds = downsample_2x2(z_v90)

            maps = paint_agent_box_maps(ego, AGENT).to(device=p_v45.device)
            if maps.numel() == 0 or maps.shape[0] != p_v45.shape[0]:
                print(
                    f"  skip maps {tuple(maps.shape)} vs p_v45 {tuple(p_v45.shape)}",
                    flush=True,
                )
                continue
            gt45 = binary_objectness_target(maps, tau=1, block=STRIDE_V45).gt(0)
            gt90 = binary_objectness_target(maps, tau=1, block=STRIDE_V90).gt(0)
            camera_z = extract_camera_z_gt(imgs, spatial_stride=STRIDE_V45)
            valid = depth_valid_mask(camera_z, d_min, d_max)
            pred_fg_v45 = p_v45.ge(tau)
            pred_fg_v90 = p_v90_ds.ge(tau)

            heat_v45.add(p_v45, gt45)
            heat_v90.add(p_v90_ds, gt45)
            heat_v90_native.add(p_v90, gt90)
            scene_heat_v45[scene].add(p_v45, gt45)
            scene_heat_v90[scene].add(p_v90_ds, gt45)

            for acc, z_pred, p_fg in (
                (depth_v45, z_v45, p_v45),
                (depth_v90, z_v90_ds, p_v90_ds),
            ):
                acc["valid"].add(z_pred, camera_z, valid)
                acc["fg"].add(z_pred, camera_z, valid & gt45)
                acc["pred_fg"].add(z_pred, camera_z, valid & p_fg.ge(tau))
                acc["far"].add(z_pred, camera_z, valid & (camera_z >= FAR_Z_M))
            for acc, z_pred, p_fg in (
                (scene_depth_v45[scene], z_v45, p_v45),
                (scene_depth_v90[scene], z_v90_ds, p_v90_ds),
            ):
                acc["valid"].add(z_pred, camera_z, valid)
                acc["fg"].add(z_pred, camera_z, valid & gt45)
                acc["pred_fg"].add(z_pred, camera_z, valid & p_fg.ge(tau))
                acc["far"].add(z_pred, camera_z, valid & (camera_z >= FAR_Z_M))

            p45_np = p_v45.detach().cpu().numpy()
            p90_np = p_v90_ds.detach().cpu().numpy()
            gt_np = gt45.detach().cpu().numpy().astype(np.uint8)
            z45_np = z_v45.detach().cpu().numpy()
            z90_np = z_v90_ds.detach().cpu().numpy()
            gt_z_np = camera_z.detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()
            view = pick_view(gt_np, p45_np)
            rgb = denormalize_rgb(flatten_imgs(imgs)[view])
            n_gt = int(gt_np[view].sum())
            rec45 = float(
                (pred_fg_v45[view] & gt45[view]).sum().item()
                / max(int(gt45[view].sum().item()), 1)
            )
            rec90 = float(
                (pred_fg_v90[view] & gt45[view]).sum().item()
                / max(int(gt45[view].sum().item()), 1)
            )
            title = (
                f"{scene} ts={timestamp} idx={idx} view={view} mix={int(mixed)}  "
                f"gt_px={n_gt}  V45 rec@{tau:g}={rec45:.3f}  V90-ds rec={rec90:.3f}"
            )
            save_heatmap_panel(
                heat_dir / scene / f"{sample_i:02d}_{idx:04d}_hm.png",
                rgb,
                gt_np[view],
                p45_np[view],
                p90_np[view],
                title,
                tau,
            )
            save_depth_panel(
                depth_dir / scene / f"{sample_i:02d}_{idx:04d}_z.png",
                rgb,
                gt_z_np[view],
                z45_np[view],
                z90_np[view],
                valid_np[view],
                d_min,
                d_max,
                title,
            )
            rows.append(
                {
                    "scene": scene,
                    "idx": int(idx),
                    "timestamp": timestamp,
                    "mixed_in_train": bool(mixed),
                    "view": view,
                    "gt_px_view": n_gt,
                    "v45_recall_view": round(rec45, 4),
                    "v90ds_recall_view": round(rec90, 4),
                    "heatmap_png": str(heat_dir / scene / f"{sample_i:02d}_{idx:04d}_hm.png"),
                    "depth_png": str(depth_dir / scene / f"{sample_i:02d}_{idx:04d}_z.png"),
                }
            )

    def pack_depth(accs: Dict[str, DepthAcc]) -> Dict[str, Any]:
        return {key: acc.pack() for key, acc in accs.items()}

    scenes = sorted({scene for scene, *_ in plan})
    report = {
        "v45_ckpt": str(v45_ckpt),
        "v90_ckpt": str(v90_ckpt),
        "split": "test_heldout",
        "note": (
            "TEST has no SAM3. Heatmap GT = projected 3D boxes, block=8 → 45x80. "
            "V90-ds = 2x2 avg-pool of native 90x160 p_fg / z_mean. "
            "Depth GT = camera-z at stride-8 centers. Mix timestamps excluded."
        ),
        "frames_per_scene": int(opt.frames_per_scene),
        "n_sampled": len(plan),
        "n_mix_excluded": len(mix_keys),
        "fg_tau": tau,
        "heatmap_45x80": {
            "v45": heat_v45.pack(),
            "v90_downsampled": heat_v90.pack(),
        },
        "heatmap_v90_native_90x160": heat_v90_native.pack(),
        "depth_45x80": {
            "v45": pack_depth(depth_v45),
            "v90_downsampled": pack_depth(depth_v90),
        },
        "by_scene": {
            scene: {
                "heatmap_v45": scene_heat_v45[scene].pack(),
                "heatmap_v90_ds": scene_heat_v90[scene].pack(),
                "depth_v45": pack_depth(scene_depth_v45[scene]),
                "depth_v90_ds": pack_depth(scene_depth_v90[scene]),
            }
            for scene in scenes
        },
        "plan": [
            {
                "scene": scene,
                "idx": idx,
                "timestamp": ts,
                "mixed_in_train": mixed,
            }
            for scene, idx, ts, mixed in plan
        ],
        "rows": rows,
    }
    out_json = out_dir / "compare_metrics.json"
    out_txt = out_dir / "compare_metrics.txt"
    out_json.write_text(json.dumps(report, indent=2))
    lines = [
        f"V45 vs V90-ds TEST  v45={v45_ckpt.name}  v90={v90_ckpt.name}",
        f"sampled {len(plan)} frames ({opt.frames_per_scene}/scene, mix excluded)  tau={tau:g}",
        "heatmap GT = projected 3D boxes @45x80; V90-ds = 2x2 avg of 90x160",
        "",
        fmt_block("V45", report["heatmap_45x80"]["v45"], report["depth_45x80"]["v45"], tau),
        fmt_block(
            "V90-ds",
            report["heatmap_45x80"]["v90_downsampled"],
            report["depth_45x80"]["v90_downsampled"],
            tau,
        ),
        (
            "V90-native90  rec@0.3="
            f"{report['heatmap_v90_native_90x160'].get('recall@0.3')}  "
            f"prec={report['heatmap_v90_native_90x160'].get('precision@0.3')}  "
            f"mean_p_fg_gt={report['heatmap_v90_native_90x160'].get('mean_p_fg_gt')}"
        ),
        "",
    ]
    for scene in scenes:
        block = report["by_scene"][scene]
        lines.append(f"scene {scene}")
        lines.append(
            "  "
            + fmt_block("V45", block["heatmap_v45"], block["depth_v45"], tau)
        )
        lines.append(
            "  "
            + fmt_block("V90-ds", block["heatmap_v90_ds"], block["depth_v90_ds"], tau)
        )
    text = "\n".join(lines) + "\n"
    out_txt.write_text(text)
    print(text)
    print(f"wrote {out_json}")
    print(f"wrote {out_txt}")
    print(f"heatmap panels: {heat_dir}")
    print(f"depth panels: {depth_dir}")


if __name__ == "__main__":
    main()

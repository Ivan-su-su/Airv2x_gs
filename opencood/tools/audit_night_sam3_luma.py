# -*- coding: utf-8 -*-
"""Train-night vehicle SAM3 vis + luminance vs p_fg correlation.

Checks whether night heatmap false positives come from SAM3 labeling lights
and/or a luminance–foreground correlation after L2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
from opencood.models.gaussian_modules_0822.heatmap.target import binary_objectness_target
from opencood.tools import train_utils
from opencood.tools.eval_gaussian_p1 import denormalize_rgb, load_epoch_checkpoint
from opencood.tools.train_gaussian_p1 import _unwrap_model
from opencood.utils.camera_utils import apply_l2_highlight_compress

NIGHT_TRAIN = "2025_05_06_10_01_50"
DAY_TRAIN = "2025_05_05_20_05_23"
SEMANTIC_MAP = np.array([0, 1, 2, 4, 6, 5, 3], dtype=np.int8)
VEHICLE_VIEWS = ("front", "front_left", "front_right", "rear", "rear_left", "rear_right")
RED = np.array([220, 20, 60], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--hypes_yaml", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epoch", type=int, default=9)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_vis", type=int, default=6)
    parser.add_argument("--n_corr", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_root",
        default="/mnt/home/suyi/visualization/concat128_p1_night_sam3",
    )
    return parser.parse_args()


def load_seg(path: Path) -> np.ndarray:
    """Raw 720x1280 SAM3 ids, mapped with the dataset table."""
    raw = np.fromfile(path, dtype=np.uint8)
    raw = raw.reshape(720, 1280)
    mapped = np.where(raw < len(SEMANTIC_MAP), SEMANTIC_MAP[raw], 0).astype(np.uint8)
    return raw, mapped


def luma_uint8(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luma in [0, 1]."""
    x = rgb.astype(np.float32) / 255.0
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def overlay(rgb: np.ndarray, fg: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Tint FG red."""
    mix = rgb.astype(np.float32)
    mask = fg.astype(bool)
    mix[mask] = (1.0 - alpha) * mix[mask] + alpha * RED.astype(np.float32)
    return np.clip(mix, 0, 255).astype(np.uint8)


def list_vehicle_fronts(scene_dir: Path) -> List[Tuple[Path, Path]]:
    """``(camera.png, seg.bin)`` for vehicle front cameras."""
    pairs: List[Tuple[Path, Path]] = []
    for ts in sorted(scene_dir.glob("timestamp_*")):
        for agent in sorted(ts.glob("agent_*")):
            cam = agent / "front_camera.png"
            seg = agent / "front_seg.bin"
            if cam.is_file() and seg.is_file() and (agent / "front_right_camera.png").is_file():
                pairs.append((cam, seg))
                break
    return pairs


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r; NaN if degenerate."""
    if a.size < 2 or float(a.std()) < 1e-8 or float(b.std()) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def bin_mean(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean y in x-bins. Returns centers, means, counts."""
    idx = np.digitize(x, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(len(centers), np.nan, dtype=np.float64)
    counts = np.zeros(len(centers), dtype=np.int64)
    for i in range(len(centers)):
        sel = idx == i
        counts[i] = int(sel.sum())
        if counts[i] > 0:
            means[i] = float(y[sel].mean())
    return centers, means, counts


def save_sam3_panel(
    path: Path,
    rgb: np.ndarray,
    rgb_l2: np.ndarray,
    raw: np.ndarray,
    mapped: np.ndarray,
    title: str,
) -> None:
    """Original | SAM3 overlay | L2 | SAM3 on L2 | raw id map."""
    fg = mapped > 0
    fig, axes = plt.subplots(1, 5, figsize=(20.0, 3.6))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB (raw night)")
    axes[1].imshow(overlay(rgb, fg))
    axes[1].set_title(f"SAM3 overlay  fg={float(fg.mean()):.3f}")
    axes[2].imshow(rgb_l2)
    axes[2].set_title("L2_hl RGB")
    axes[3].imshow(overlay(rgb_l2, fg))
    axes[3].set_title("SAM3 on L2")
    im = axes[4].imshow(raw, cmap="tab10", vmin=0, vmax=6, interpolation="nearest")
    axes[4].set_title("raw seg id 0-6")
    fig.colorbar(im, ax=axes[4], fraction=0.046, ticks=list(range(7)))
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def l2_preview(rgb: np.ndarray) -> np.ndarray:
    """Apply L2_hl to uint8 HWC for display."""
    t = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float() / 255.0
    t = apply_l2_highlight_compress(t)
    out = t.clamp(0, 1).cpu().numpy()
    return (np.transpose(out, (1, 2, 0)) * 255.0).astype(np.uint8)


def scene_index_range(dataset: Any, scene_id: str) -> Tuple[int, int]:
    """Inclusive-start exclusive-end global indices for a scene folder."""
    for scene_i, end in enumerate(dataset.len_record):
        start = 0 if scene_i == 0 else int(dataset.len_record[scene_i - 1])
        scene_db = dataset.scenario_database[scene_i]
        first = next(iter(scene_db.values()))
        ts0 = dataset.return_timestamp_key(scene_db, 0)
        path = str(first[ts0]["metadata_path"])
        if scene_id in path:
            return start, int(end)
    raise KeyError(f"scene {scene_id} not in dataset")


def flatten_imgs(imgs: torch.Tensor) -> torch.Tensor:
    """5D or 4D to NCHW."""
    if imgs.dim() == 5:
        return imgs.reshape(-1, *imgs.shape[2:])
    return imgs


def accumulate_corr(
    luma: np.ndarray,
    p_fg: np.ndarray,
    gt_fg: np.ndarray,
    bucket: Dict[str, List[np.ndarray]],
) -> None:
    """Store flattened arrays for later correlation."""
    bucket["luma"].append(luma.reshape(-1))
    bucket["p_fg"].append(p_fg.reshape(-1))
    bucket["gt"].append(gt_fg.reshape(-1).astype(np.uint8))


def summarize_corr(bucket: Dict[str, List[np.ndarray]]) -> Dict[str, Any]:
    """Pearson + luma-bin p_fg, including GT-background only."""
    luma = np.concatenate(bucket["luma"])
    p_fg = np.concatenate(bucket["p_fg"])
    gt = np.concatenate(bucket["gt"]).astype(bool)
    edges = np.linspace(0.0, 1.0, 11)
    c_all, m_all, n_all = bin_mean(luma, p_fg, edges)
    c_bg, m_bg, n_bg = bin_mean(luma[~gt], p_fg[~gt], edges)
    pred = p_fg >= 0.3
    return {
        "n_pixels": int(luma.size),
        "pearson_luma_pfg": round(pearson(luma, p_fg), 4),
        "pearson_luma_pfg_gt_bg_only": round(pearson(luma[~gt], p_fg[~gt]), 4),
        "pearson_luma_sam3": round(pearson(luma, gt.astype(np.float32)), 4),
        "mean_luma_sam3_fg": round(float(luma[gt].mean()) if gt.any() else float("nan"), 4),
        "mean_luma_sam3_bg": round(float(luma[~gt].mean()) if (~gt).any() else float("nan"), 4),
        "mean_pfg_sam3_fg": round(float(p_fg[gt].mean()) if gt.any() else float("nan"), 4),
        "mean_pfg_sam3_bg": round(float(p_fg[~gt].mean()) if (~gt).any() else float("nan"), 4),
        "pred_fg_ratio": round(float(pred.mean()), 4),
        "sam3_fg_ratio": round(float(gt.mean()), 4),
        "precision@0.3": round(
            float((pred & gt).sum() / max(int(pred.sum()), 1)), 4
        ),
        "recall@0.3": round(float((pred & gt).sum() / max(int(gt.sum()), 1)), 4),
        "pfg_by_luma_bin": {
            f"{edges[i]:.1f}-{edges[i+1]:.1f}": {
                "mean_pfg": None if np.isnan(m_all[i]) else round(float(m_all[i]), 4),
                "n": int(n_all[i]),
            }
            for i in range(len(m_all))
        },
        "pfg_by_luma_bin_gt_bg_only": {
            f"{edges[i]:.1f}-{edges[i+1]:.1f}": {
                "mean_pfg": None if np.isnan(m_bg[i]) else round(float(m_bg[i]), 4),
                "n": int(n_bg[i]),
            }
            for i in range(len(m_bg))
        },
        "_plot": {
            "centers": c_all.tolist(),
            "mean_all": [None if np.isnan(v) else float(v) for v in m_all],
            "mean_bg": [None if np.isnan(v) else float(v) for v in m_bg],
        },
    }


def save_corr_plot(path: Path, night: Dict[str, Any], day: Dict[str, Any]) -> None:
    """p_fg vs luma bins, night vs day, all pixels and SAM3-background."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, block, title in (
        (axes[0], night, "TRAIN night vehicle"),
        (axes[1], day, "TRAIN day vehicle"),
    ):
        plot = block["_plot"]
        x = plot["centers"]
        ax.plot(x, plot["mean_all"], "o-", label="all pixels")
        ax.plot(x, plot["mean_bg"], "s--", label="SAM3 background only")
        ax.axhline(0.3, color="gray", ls=":", lw=1, label="tau=0.3")
        ax.set_xlabel("luma")
        ax.set_title(
            f"{title}\nr(luma,p_fg)={block['pearson_luma_pfg']:.3f}  "
            f"r|BG={block['pearson_luma_pfg_gt_bg_only']:.3f}"
        )
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("mean p_fg")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    """SAM3 vis on train night + luma/p_fg correlation night vs day."""
    opt = parse_args()
    out = Path(opt.out_root)
    vis_dir = out / "sam3_panels"
    vis_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(opt.seed)

    night_dir = Path("/mnt/home/suyi/AirV2X-Perception/train/train") / NIGHT_TRAIN
    pairs = list_vehicle_fronts(night_dir)
    print(f"night vehicle fronts: {len(pairs)}")
    vis_idx = np.linspace(0, len(pairs) - 1, num=opt.n_vis)
    vis_idx = sorted({int(round(float(x))) for x in vis_idx})
    sam3_stats: List[Dict[str, Any]] = []
    for k, i in enumerate(vis_idx):
        cam_path, seg_path = pairs[i]
        rgb = np.asarray(Image.open(cam_path).convert("RGB"))
        raw, mapped = load_seg(seg_path)
        fg = mapped > 0
        luma = luma_uint8(rgb)
        rgb_l2 = l2_preview(rgb)
        title = (
            f"TRAIN night {NIGHT_TRAIN}  {cam_path.parent.parent.name}/"
            f"{cam_path.parent.name}/front  "
            f"fg={float(fg.mean()):.3f}  "
            f"luma fg={float(luma[fg].mean()) if fg.any() else float('nan'):.3f} "
            f"bg={float(luma[~fg].mean()):.3f}  "
            f"r(luma,sam3)={pearson(luma, fg.astype(np.float32)):.3f}"
        )
        save_sam3_panel(vis_dir / f"{k:02d}_{cam_path.parent.parent.name}.png", rgb, rgb_l2, raw, mapped, title)
        counts = {int(v): int((raw == v).sum()) for v in range(7)}
        sam3_stats.append(
            {
                "path": str(cam_path),
                "fg_ratio": round(float(fg.mean()), 4),
                "raw_id_hist": counts,
                "mean_luma_fg": round(float(luma[fg].mean()) if fg.any() else float("nan"), 4),
                "mean_luma_bg": round(float(luma[~fg].mean()), 4),
                "pearson_luma_sam3": round(pearson(luma, fg.astype(np.float32)), 4),
            }
        )
        print(f"  vis {k}: {cam_path.parent.parent.name} fg={fg.mean():.3f} r={sam3_stats[-1]['pearson_luma_sam3']}")

    # Broader SAM3-luma on raw files (no model).
    n_raw = min(40, len(pairs))
    raw_pick = rng.choice(len(pairs), size=n_raw, replace=False)
    raw_luma_fg: List[float] = []
    raw_luma_bg: List[float] = []
    raw_r: List[float] = []
    raw_fg_ratio: List[float] = []
    for i in raw_pick:
        cam_path, seg_path = pairs[int(i)]
        rgb = np.asarray(Image.open(cam_path).convert("RGB"))
        _raw, mapped = load_seg(seg_path)
        fg = mapped > 0
        luma = luma_uint8(rgb)
        raw_fg_ratio.append(float(fg.mean()))
        raw_r.append(pearson(luma, fg.astype(np.float32)))
        if fg.any():
            raw_luma_fg.append(float(luma[fg].mean()))
        raw_luma_bg.append(float(luma[~fg].mean()))
    sam3_raw = {
        "n": n_raw,
        "mean_fg_ratio": round(float(np.mean(raw_fg_ratio)), 4),
        "mean_pearson_luma_sam3": round(float(np.nanmean(raw_r)), 4),
        "mean_luma_fg": round(float(np.mean(raw_luma_fg)), 4),
        "mean_luma_bg": round(float(np.mean(raw_luma_bg)), 4),
    }
    print("raw SAM3 vs luma:", sam3_raw)

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    hypes["validate_dir"] = hypes["root_dir"]
    hypes["train"] = False
    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print("Building TRAIN dataset (eval crop, L2 on night)...")
    dataset = build_dataset(hypes, visualize=False, train=False)
    night_lo, night_hi = scene_index_range(dataset, NIGHT_TRAIN)
    try:
        day_lo, day_hi = scene_index_range(dataset, DAY_TRAIN)
    except KeyError:
        day_lo, day_hi = 0, min(200, night_lo)
        print(f"day scene missing, fallback indices [{day_lo},{day_hi})")
    print(f"night idx [{night_lo},{night_hi}) day [{day_lo},{day_hi})")

    model = train_utils.create_model(hypes)
    load_epoch_checkpoint(model, opt.model_dir, opt.epoch)
    model.to(device)
    model.eval()
    _unwrap_model(model)

    night_ids = rng.choice(night_hi - night_lo, size=min(opt.n_corr, night_hi - night_lo), replace=False)
    night_ids = sorted(int(night_lo + x) for x in night_ids)
    day_ids = rng.choice(max(day_hi - day_lo, 1), size=min(opt.n_corr, max(day_hi - day_lo, 1)), replace=False)
    day_ids = sorted(int(day_lo + x) for x in day_ids)

    buckets = {
        "night": {"luma": [], "p_fg": [], "gt": []},
        "day": {"luma": [], "p_fg": [], "gt": []},
    }
    pred_dir = out / "heatmap_panels"
    pred_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for split, ids in (("night", night_ids), ("day", day_ids)):
            for j, idx in enumerate(ids):
                sample = dataset[idx]
                batch = dataset.collate_batch_test([sample])
                batch = train_utils.to_device(batch, device)
                ego = batch["ego"]
                if "vehicle" not in ego or "vehicle" not in (pred := model(ego)):
                    print(f"  skip {split} idx={idx}: no vehicle")
                    continue
                cam = ego["vehicle"]["batch_merged_cam_inputs"]
                imgs = flatten_imgs(cam["imgs"])
                logits = pred["vehicle"]["heatmap_logits"]
                p_fg = torch.softmax(logits, dim=1)[:, 1]
                semantic = cam.get("image_semantic_gts")
                if not torch.is_tensor(semantic) or int(semantic.max().item()) <= 0:
                    print(f"  skip {split} idx={idx}: empty SAM3")
                    continue
                if semantic.dim() == 4:
                    semantic = semantic.reshape(-1, semantic.shape[-2], semantic.shape[-1])
                gt = binary_objectness_target(semantic.long(), tau=1)
                p_np = p_fg.detach().cpu().numpy()
                gt_np = gt.detach().cpu().numpy()
                # luma from denormalized (L2 already in the tensor for night)
                n_view = min(int(imgs.shape[0]), int(p_np.shape[0]))
                luma_maps = []
                for v in range(n_view):
                    rgb = denormalize_rgb(imgs[v])
                    luma_maps.append(luma_uint8(rgb))
                    if split == "night" and j < 4 and v == int(np.argmax(gt_np[:n_view].reshape(n_view, -1).sum(1))):
                        gt_up = np.array(
                            Image.fromarray(gt_np[v].astype(np.uint8)).resize(
                                (rgb.shape[1], rgb.shape[0]), Image.NEAREST
                            )
                        )
                        pred_up = np.array(
                            Image.fromarray((p_np[v] >= 0.3).astype(np.uint8)).resize(
                                (rgb.shape[1], rgb.shape[0]), Image.NEAREST
                            )
                        )
                        fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.5))
                        axes[0].imshow(rgb)
                        axes[0].set_title("L2 RGB (model input)")
                        axes[1].imshow(overlay(rgb, gt_up > 0))
                        axes[1].set_title("SAM3 R90")
                        im = axes[2].imshow(p_np[v], cmap="magma", vmin=0, vmax=1)
                        axes[2].set_title("p_fg")
                        fig.colorbar(im, ax=axes[2], fraction=0.046)
                        axes[3].imshow(overlay(rgb, pred_up > 0))
                        axes[3].set_title("pred@0.3")
                        for ax in axes:
                            ax.axis("off")
                        fig.suptitle(f"train {split} idx={idx} vehicle view={v}")
                        fig.tight_layout()
                        fig.savefig(pred_dir / f"{split}_{j:02d}_{idx}.png", dpi=130)
                        plt.close(fig)
                luma_stack = np.stack(luma_maps, 0)
                # p_fg is R90; downsample luma by 4x4 mean
                h, w = p_np.shape[-2], p_np.shape[-1]
                luma_r90 = luma_stack[:n_view, : h * 4, : w * 4]
                luma_r90 = luma_r90.reshape(n_view, h, 4, w, 4).mean(axis=(2, 4))
                accumulate_corr(luma_r90, p_np[:n_view], gt_np[:n_view], buckets[split])
                print(f"  {split} {j+1}/{len(ids)} idx={idx} sam3={gt_np.mean():.4f} pfg={p_np.mean():.4f}")

    night_sum = summarize_corr(buckets["night"])
    day_sum = summarize_corr(buckets["day"])
    save_corr_plot(out / "luma_pfg_bins.png", night_sum, day_sum)
    night_plot = night_sum.pop("_plot")
    day_plot = day_sum.pop("_plot")
    report = {
        "night_scene": NIGHT_TRAIN,
        "day_scene": DAY_TRAIN,
        "epoch": opt.epoch,
        "sam3_vis": sam3_stats,
        "sam3_raw_night_fronts": sam3_raw,
        "corr_night_vehicle": night_sum,
        "corr_day_vehicle": day_sum,
        "note": (
            "SAM3 vis is original 720x1280. Correlation uses model p_fg at R90 "
            "vs luma of denormalized input (L2_hl on night)."
        ),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    lines = [
        f"TRAIN night SAM3 + luma-p_fg  epoch={opt.epoch}",
        f"night={NIGHT_TRAIN}  day={DAY_TRAIN}",
        "",
        "SAM3 raw night vehicle fronts:",
        f"  fg_ratio={sam3_raw['mean_fg_ratio']:.3f}  "
        f"r(luma,SAM3)={sam3_raw['mean_pearson_luma_sam3']:.3f}  "
        f"luma fg={sam3_raw['mean_luma_fg']:.3f} bg={sam3_raw['mean_luma_bg']:.3f}",
        "",
        "Model p_fg vs luma (R90):",
        f"  NIGHT r={night_sum['pearson_luma_pfg']:.3f}  r|SAM3-BG={night_sum['pearson_luma_pfg_gt_bg_only']:.3f}  "
        f"prec@0.3={night_sum['precision@0.3']:.3f}  rec@0.3={night_sum['recall@0.3']:.3f}  "
        f"pred_fg={night_sum['pred_fg_ratio']:.3f} sam3={night_sum['sam3_fg_ratio']:.3f}",
        f"  DAY   r={day_sum['pearson_luma_pfg']:.3f}  r|SAM3-BG={day_sum['pearson_luma_pfg_gt_bg_only']:.3f}  "
        f"prec@0.3={day_sum['precision@0.3']:.3f}  rec@0.3={day_sum['recall@0.3']:.3f}  "
        f"pred_fg={day_sum['pred_fg_ratio']:.3f} sam3={day_sum['sam3_fg_ratio']:.3f}",
        "",
        f"panels: {vis_dir}",
        f"corr plot: {out / 'luma_pfg_bins.png'}",
    ]
    text = "\n".join(lines) + "\n"
    (out / "metrics.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

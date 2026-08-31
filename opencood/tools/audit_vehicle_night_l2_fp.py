# -*- coding: utf-8 -*-
"""READ-ONLY TEST-night vehicle heatmap audit: RAW vs production L2_hl vs sqrt.

Does not modify training, SAM3, weights, or the 0.3 threshold.
Variant B uses production ``encode_rgb_for_p1(apply_l2=True)`` (L2 + highlight
compress) so it matches current TEST night behavior, not the bare L2 formula
alone.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
import opencood.utils.camera_utils as camera_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    PRIMARY_OBJECTNESS_THRESHOLD,
)
from opencood.models.gaussian_modules_0822.heatmap.target import binary_objectness_target
from opencood.tools import train_utils
from opencood.tools.eval_gaussian_p1 import load_epoch_checkpoint
from opencood.tools.train_gaussian_p1 import _unwrap_model
from opencood.tools.vis_test_heatmap_recall import paint_agent_box_maps

SCENE_ID = "2025_05_10_19_54_35"
VARIANTS = ("raw", "l2", "sqrt")
TAU = float(PRIMARY_OBJECTNESS_THRESHOLD)
SAT_EDGES = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0000001])
SAT_LABELS = [
    "[0.0,0.1)",
    "[0.1,0.2)",
    "[0.2,0.4)",
    "[0.4,0.6)",
    "[0.6,0.8)",
    "[0.8,0.95)",
    "[0.95,1.0]",
]
PFG_EDGES = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0000001])
PFG_LABELS = [
    "[0.0,0.1)",
    "[0.1,0.2)",
    "[0.2,0.4)",
    "[0.4,0.6)",
    "[0.6,0.8)",
    "[0.8,0.9)",
    "[0.9,1.0]",
]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)


def luma01(rgb01: np.ndarray) -> np.ndarray:
    """Rec.709 luma on ``[C,H,W]`` or ``[H,W,C]`` display RGB in ``[0,1]``."""
    if rgb01.ndim == 3 and rgb01.shape[0] == 3:
        r, g, b = rgb01[0], rgb01[1], rgb01[2]
    else:
        r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float64)


def display_rgb01(chw: torch.Tensor) -> np.ndarray:
    """Undo ImageNet norm → clip to ``[0,1]``, shape ``[3,H,W]``."""
    x = chw[:3].detach().float().cpu().numpy().astype(np.float64)
    x = x * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    return np.clip(x, 0.0, 1.0)


def to_uint8(rgb01: np.ndarray) -> np.ndarray:
    """``[3,H,W]`` float → HWC uint8."""
    return (np.clip(np.transpose(rgb01, (1, 2, 0)), 0.0, 1.0) * 255.0).astype(np.uint8)


def downsample_4(maps: np.ndarray) -> np.ndarray:
    """Mean-pool ``[N,H,W]`` by aligned 4x4 blocks."""
    n_map, height, width = maps.shape
    if height % 4 or width % 4:
        raise ValueError(f"cannot 4x4 pool {maps.shape}")
    return maps.reshape(n_map, height // 4, 4, width // 4, 4).mean(axis=(2, 4))


def flatten_imgs(imgs: torch.Tensor) -> torch.Tensor:
    """``[A,V,C,H,W]`` or NCHW → NCHW."""
    if imgs.dim() == 5:
        return imgs.reshape(-1, *imgs.shape[2:])
    return imgs


def scene_range(dataset: Any, scene_id: str) -> Tuple[int, int]:
    """Global ``[start, end)`` for a scenario folder name."""
    for scene_i, end in enumerate(dataset.len_record):
        start = 0 if scene_i == 0 else int(dataset.len_record[scene_i - 1])
        scene_db = dataset.scenario_database[scene_i]
        first = next(iter(scene_db.values()))
        ts0 = dataset.return_timestamp_key(scene_db, 0)
        if scene_id in str(first[ts0]["metadata_path"]):
            return start, int(end)
    raise KeyError(scene_id)


def install_encode_patch(variant: str) -> None:
    """Replace ``camera_utils.encode_rgb_for_p1`` for this process only."""

    def encode_rgb_for_p1(pil_rgb: Any, apply_l2: bool = False) -> torch.Tensor:
        rgb = TF.to_tensor(pil_rgb)
        if variant == "raw":
            disp = rgb
        elif variant == "l2":
            disp = camera_utils.apply_l2_highlight_compress(rgb)
        elif variant == "sqrt":
            disp = rgb.clamp(0.0, 1.0).sqrt()
        else:
            raise ValueError(variant)
        return camera_utils.imagenet_normalize_display_rgb(disp[:3])

    camera_utils.encode_rgb_for_p1 = encode_rgb_for_p1  # type: ignore[method-assign]


def bin_frac(values: np.ndarray, edges: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    """Fraction of pixels per bin."""
    idx = np.digitize(values, edges, right=False) - 1
    idx = np.clip(idx, 0, len(labels) - 1)
    n_all = max(int(values.size), 1)
    out: Dict[str, float] = {}
    for i, lab in enumerate(labels):
        out[lab] = round(float((idx == i).sum()) / n_all, 6)
    return out


def empty_pfg_bins() -> Dict[str, Dict[str, float]]:
    """Zeroed p_fg-by-luma accumulators."""
    return {
        lab: {"n": 0.0, "sum_p": 0.0, "n_ge": 0.0, "p_list": []}
        for lab in PFG_LABELS
    }


def add_pfg_bins(
    acc: Dict[str, Dict[str, float]],
    luma: np.ndarray,
    p_fg: np.ndarray,
) -> None:
    """Accumulate p_fg stats into luma bins."""
    idx = np.digitize(luma, PFG_EDGES, right=False) - 1
    idx = np.clip(idx, 0, len(PFG_LABELS) - 1)
    for i, lab in enumerate(PFG_LABELS):
        sel = idx == i
        n_sel = int(sel.sum())
        if n_sel == 0:
            continue
        p = p_fg[sel]
        acc[lab]["n"] += n_sel
        acc[lab]["sum_p"] += float(p.sum())
        acc[lab]["n_ge"] += float((p >= TAU).sum())
        # median later from a subsample to bound memory
        if len(acc[lab]["p_list"]) < 200000:
            acc[lab]["p_list"].extend(p.astype(np.float32).tolist())


def finalize_pfg_bins(acc: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Convert accumulators to reportable means/medians."""
    out: Dict[str, Dict[str, Any]] = {}
    for lab, block in acc.items():
        n_pix = int(block["n"])
        if n_pix <= 0:
            out[lab] = {"n": 0, "mean_pfg": None, "median_pfg": None, "frac_ge_0.3": None}
            continue
        arr = np.asarray(block["p_list"], dtype=np.float64)
        out[lab] = {
            "n": n_pix,
            "mean_pfg": round(float(block["sum_p"] / n_pix), 4),
            "median_pfg": round(float(np.median(arr)), 4) if arr.size else None,
            "frac_ge_0.3": round(float(block["n_ge"] / n_pix), 4),
        }
    return out


def f1_score(recall: float, precision: float) -> float:
    """Harmonic mean; 0 if both zero."""
    if recall + precision <= 0:
        return 0.0
    return float(2.0 * recall * precision / (recall + precision))


def save_compare_panel(
    path: Path,
    raw: np.ndarray,
    l2: np.ndarray,
    sqrt_im: np.ndarray,
    gt: np.ndarray,
    p_raw: np.ndarray,
    p_l2: np.ndarray,
    p_sqrt: np.ndarray,
    title: str,
) -> None:
    """One-frame 10-column comparison at the fixed GT-argmax view."""
    pred_raw = (p_raw >= TAU).astype(np.uint8)
    pred_l2 = (p_l2 >= TAU).astype(np.uint8)
    pred_sqrt = (p_sqrt >= TAU).astype(np.uint8)
    fig, axes = plt.subplots(2, 5, figsize=(22.0, 7.2))
    tiles = [
        (axes[0, 0], raw, "RAW RGB", None),
        (axes[0, 1], l2, "L2_hl RGB (prod)", None),
        (axes[0, 2], sqrt_im, "sqrt RGB", None),
        (axes[0, 3], gt, "GT-box R90", "gray"),
        (axes[0, 4], p_raw, "p_fg RAW", "magma"),
        (axes[1, 0], p_l2, "p_fg L2_hl", "magma"),
        (axes[1, 1], p_sqrt, "p_fg sqrt", "magma"),
        (axes[1, 2], pred_raw, "FG@0.3 RAW", "gray"),
        (axes[1, 3], pred_l2, "FG@0.3 L2_hl", "gray"),
        (axes[1, 4], pred_sqrt, "FG@0.3 sqrt", "gray"),
    ]
    for ax, img, lab, cmap in tiles:
        if cmap == "magma":
            ax.imshow(img, cmap="magma", vmin=0.0, vmax=1.0)
        elif cmap == "gray":
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(img)
        ax.set_title(lab, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    """Run the three-variant TEST-night vehicle heatmap audit."""
    log_dir = Path(
        "/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
        "airv2x_gaussian_p1_joint/concat128_2026_08_29_16_25_05"
    )
    ckpt_epoch = 9
    out_root = Path("/mnt/home/suyi/visualization/concat128_p1_night_l2_fp_audit")
    vis_dir = out_root / "panels"
    vis_dir.mkdir(parents=True, exist_ok=True)
    hypes = yaml_utils.load_yaml(str(log_dir / "config.yaml"), None)
    hypes["validate_dir"] = hypes["test_dir"]
    hypes["train"] = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print("Building TEST dataset...")
    dataset = build_dataset(hypes, visualize=False, train=False)
    lo, hi = scene_range(dataset, SCENE_ID)
    n_scene = hi - lo
    rng = np.random.RandomState(42)
    n_keep = min(40, n_scene)
    sample_ids = sorted(int(lo + x) for x in rng.choice(n_scene, size=n_keep, replace=False))
    print(f"scene {SCENE_ID} idx [{lo},{hi}) sampled {n_keep} seed=42")

    model = train_utils.create_model(hypes)
    ckpt_path = load_epoch_checkpoint(model, str(log_dir), ckpt_epoch)
    model.to(device)
    model.eval()
    _unwrap_model(model)

    gt_store: Dict[int, np.ndarray] = {}
    view_store: Dict[int, int] = {}
    rgb_store: Dict[str, Dict[int, np.ndarray]] = {v: {} for v in VARIANTS}
    pfg_store: Dict[str, Dict[int, np.ndarray]] = {v: {} for v in VARIANTS}
    rows: List[Dict[str, Any]] = []
    sat_acc: Dict[str, List[np.ndarray]] = {v: [] for v in VARIANTS}
    pfg_all = {v: empty_pfg_bins() for v in VARIANTS}
    pfg_gt = {v: empty_pfg_bins() for v in VARIANTS}
    pfg_bg = {v: empty_pfg_bins() for v in VARIANTS}
    hl_stats = {
        v: {
            "hl08_bg": 0,
            "hl08_fp": 0,
            "hl095_bg": 0,
            "hl095_fp": 0,
            "fp_all": 0,
            "fp_hl08": 0,
            "fp_hl095": 0,
            "pred_all": 0,
            "tp": 0,
            "fn": 0,
            "fp": 0,
        }
        for v in VARIANTS
    }

    with torch.no_grad():
        for variant in VARIANTS:
            install_encode_patch(variant)
            print(f"=== variant {variant} ===")
            for i, idx in enumerate(sample_ids):
                sample = dataset[idx]
                batch = dataset.collate_batch_test([sample])
                batch = train_utils.to_device(batch, device)
                ego = batch["ego"]
                if "vehicle" not in ego:
                    print(f"  skip idx={idx}: no vehicle")
                    continue
                pred = model(ego)
                if "vehicle" not in pred:
                    print(f"  skip idx={idx}: no vehicle pred")
                    continue
                imgs = flatten_imgs(ego["vehicle"]["batch_merged_cam_inputs"]["imgs"])
                logits = pred["vehicle"]["heatmap_logits"]
                p_fg = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                if idx not in gt_store:
                    maps = paint_agent_box_maps(ego, "vehicle")
                    gt_r90 = binary_objectness_target(maps, tau=1).cpu().numpy().astype(np.uint8)
                    gt_store[idx] = gt_r90
                    scores = gt_r90.reshape(gt_r90.shape[0], -1).sum(axis=1)
                    view_store[idx] = int(np.argmax(scores)) if scores.size else 0
                gt_r90 = gt_store[idx]
                n_view = min(int(p_fg.shape[0]), int(gt_r90.shape[0]), int(imgs.shape[0]))
                p_fg = p_fg[:n_view]
                gt_r90 = gt_r90[:n_view]
                disp = np.stack([display_rgb01(imgs[v]) for v in range(n_view)], axis=0)
                luma_hi = np.stack([luma01(disp[v]) for v in range(n_view)], axis=0)
                luma = downsample_4(luma_hi)
                pred_fg = p_fg >= TAU
                gt_fg = gt_r90 > 0
                tp = int((pred_fg & gt_fg).sum())
                fn = int((~pred_fg & gt_fg).sum())
                fp = int((pred_fg & ~gt_fg).sum())
                n_gt = tp + fn
                n_pred = tp + fp
                recall = tp / max(n_gt, 1)
                precision = tp / max(n_pred, 1)
                sat_acc[variant].append(luma.reshape(-1))
                add_pfg_bins(pfg_all[variant], luma.reshape(-1), p_fg.reshape(-1))
                add_pfg_bins(pfg_gt[variant], luma[gt_fg], p_fg[gt_fg])
                add_pfg_bins(pfg_bg[variant], luma[~gt_fg], p_fg[~gt_fg])
                bg = ~gt_fg
                hl08 = luma >= 0.8
                hl095 = luma >= 0.95
                hl_stats[variant]["hl08_bg"] += int((hl08 & bg).sum())
                hl_stats[variant]["hl08_fp"] += int((hl08 & bg & pred_fg).sum())
                hl_stats[variant]["hl095_bg"] += int((hl095 & bg).sum())
                hl_stats[variant]["hl095_fp"] += int((hl095 & bg & pred_fg).sum())
                hl_stats[variant]["fp_all"] += fp
                hl_stats[variant]["fp_hl08"] += int((pred_fg & bg & hl08).sum())
                hl_stats[variant]["fp_hl095"] += int((pred_fg & bg & hl095).sum())
                hl_stats[variant]["pred_all"] += n_pred
                hl_stats[variant]["tp"] += tp
                hl_stats[variant]["fn"] += fn
                hl_stats[variant]["fp"] += fp
                view = view_store[idx]
                view = min(view, n_view - 1)
                rgb_store[variant][idx] = to_uint8(disp[view])
                pfg_store[variant][idx] = p_fg[view]
                rows.append(
                    {
                        "frame_id": int(idx),
                        "variant": variant,
                        "gt_box_pixels": n_gt,
                        "pred_fg_pixels": n_pred,
                        "pred_fg_ratio": round(float(pred_fg.mean()), 6),
                        "recall_03": round(recall, 4),
                        "precision_03": round(precision, 4),
                        "mean_pfg_gt": round(float(p_fg[gt_fg].mean()), 4) if n_gt else None,
                        "mean_pfg_non_gt": round(float(p_fg[bg].mean()), 4) if int(bg.sum()) else None,
                        "high_luma08_pixels": int(hl08.sum()),
                        "high_luma08_fp_pixels": int((hl08 & bg & pred_fg).sum()),
                        "high_luma095_pixels": int(hl095.sum()),
                        "high_luma095_fp_pixels": int((hl095 & bg & pred_fg).sum()),
                        "n_sat_near1": int((luma >= 0.99).sum()),
                        "view": view,
                    }
                )
                print(
                    f"  {variant} {i+1}/{n_keep} idx={idx} gt={n_gt} pred={n_pred} "
                    f"R={recall:.3f} P={precision:.3f}"
                )

    gt_pixels_total = int(sum(int((gt_store[i] > 0).sum()) for i in gt_store))
    n_with_gt = int(sum(1 for i in gt_store if int((gt_store[i] > 0).sum()) > 0))

    def pack_variant(variant: str) -> Dict[str, Any]:
        st = hl_stats[variant]
        tp, fn, fp = st["tp"], st["fn"], st["fp"]
        n_gt = tp + fn
        n_pred = tp + fp
        rec = tp / max(n_gt, 1)
        prec = tp / max(n_pred, 1)
        luma_cat = np.concatenate(sat_acc[variant]) if sat_acc[variant] else np.zeros(1)
        hl08_bg = max(st["hl08_bg"], 1)
        hl095_bg = max(st["hl095_bg"], 1)
        fp_all = max(st["fp_all"], 1)
        return {
            "pred_fg_pixels": int(n_pred),
            "pred_fg_ratio": round(float(n_pred / max(luma_cat.size, 1)), 6),
            "gt_box_recall@0.3": round(rec, 4),
            "gt_box_precision@0.3": round(prec, 4),
            "gt_box_f1@0.3": round(f1_score(rec, prec), 4),
            "saturation": {
                **bin_frac(luma_cat, SAT_EDGES, SAT_LABELS),
                "frac_luma_ge_0.8": round(float((luma_cat >= 0.8).mean()), 6),
                "frac_luma_ge_0.95": round(float((luma_cat >= 0.95).mean()), 6),
                "frac_luma_ge_0.99": round(float((luma_cat >= 0.99).mean()), 6),
            },
            "pfg_by_luma_all": finalize_pfg_bins(pfg_all[variant]),
            "pfg_by_luma_gtbox": finalize_pfg_bins(pfg_gt[variant]),
            "pfg_by_luma_non_gtbox": finalize_pfg_bins(pfg_bg[variant]),
            "highlight_bg_fp": {
                "n_luma_ge_0.8_bg": int(st["hl08_bg"]),
                "n_luma_ge_0.8_bg_pred": int(st["hl08_fp"]),
                "fp_rate_luma_ge_0.8_bg": round(st["hl08_fp"] / hl08_bg, 4),
                "frac_of_all_fp_from_luma_ge_0.8": round(st["fp_hl08"] / fp_all, 4),
                "n_luma_ge_0.95_bg": int(st["hl095_bg"]),
                "n_luma_ge_0.95_bg_pred": int(st["hl095_fp"]),
                "fp_rate_luma_ge_0.95_bg": round(st["hl095_fp"] / hl095_bg, 4),
                "frac_of_all_fp_from_luma_ge_0.95": round(st["fp_hl095"] / fp_all, 4),
            },
            "fp_decomposition": {
                "predicted_non_gt": int(st["fp_all"]),
                "high_highlight_luma_ge_0.8": int(st["fp_hl08"]),
                "high_highlight_luma_ge_0.8_frac": round(st["fp_hl08"] / fp_all, 4),
                "remaining_non_gt_background": int(st["fp_all"] - st["fp_hl08"]),
                "remaining_non_gt_background_frac": round(
                    (st["fp_all"] - st["fp_hl08"]) / fp_all, 4
                ),
                "ego_roi": None,
                "note": (
                    "ego ROI skipped: side-mirror location is view-dependent "
                    "(bottom-right / bottom-left / left edge) and not a stable "
                    "pixel box across stacked vehicle cameras."
                ),
            },
        }

    by_variant = {v: pack_variant(v) for v in VARIANTS}

    # Representative frames from L2 rows.
    l2_rows = [r for r in rows if r["variant"] == "l2" and r["frame_id"] in rgb_store["l2"]]
    picks: List[Tuple[str, int]] = []
    if l2_rows:
        picks.append(("severe_fp", max(l2_rows, key=lambda r: r["pred_fg_pixels"])["frame_id"]))
        picks.append(
            ("highlight_fp", max(l2_rows, key=lambda r: r["high_luma08_fp_pixels"])["frame_id"])
        )
        with_gt = [r for r in l2_rows if r["gt_box_pixels"] >= 20]
        if with_gt:
            picks.append(("true_vehicle", max(with_gt, key=lambda r: r["recall_03"])["frame_id"]))
            picks.append(("l2_better", min(with_gt, key=lambda r: r["pred_fg_pixels"])["frame_id"]))
        med = sorted(l2_rows, key=lambda r: r["pred_fg_pixels"])[len(l2_rows) // 2]
        picks.append(("median_fp", med["frame_id"]))
    seen = set()
    unique_picks: List[Tuple[str, int]] = []
    for name, idx in picks:
        if idx in seen:
            continue
        seen.add(idx)
        unique_picks.append((name, idx))
        gt_v = gt_store[idx]
        view = min(view_store[idx], gt_v.shape[0] - 1)
        save_compare_panel(
            vis_dir / f"{name}_{idx}.png",
            rgb_store["raw"][idx],
            rgb_store["l2"][idx],
            rgb_store["sqrt"][idx],
            gt_v[view].astype(np.float32),
            pfg_store["raw"][idx],
            pfg_store["l2"][idx],
            pfg_store["sqrt"][idx],
            f"TEST night {SCENE_ID} idx={idx} view={view}  "
            f"ckpt=net_epoch{ckpt_epoch}  tau={TAU:g}  "
            f"L2=production L2_hl  (not bare sqrt(I/0.25))",
        )

    csv_path = out_root / "per_frame.csv"
    fieldnames = [
        "frame_id",
        "variant",
        "gt_box_pixels",
        "pred_fg_pixels",
        "pred_fg_ratio",
        "recall_03",
        "precision_03",
        "mean_pfg_gt",
        "mean_pfg_non_gt",
        "high_luma08_pixels",
        "high_luma08_fp_pixels",
        "high_luma095_pixels",
        "high_luma095_fp_pixels",
        "n_sat_near1",
        "view",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Verdict
    l2m, rawm, sqm = by_variant["l2"], by_variant["raw"], by_variant["sqrt"]
    extra_hl = (
        l2m["saturation"]["frac_luma_ge_0.8"] - rawm["saturation"]["frac_luma_ge_0.8"]
    )
    rec_keep = sqm["gt_box_recall@0.3"] >= 0.8 * max(l2m["gt_box_recall@0.3"], 1e-6)
    fp_drop = sqm["pred_fg_pixels"] < 0.7 * max(l2m["pred_fg_pixels"], 1)
    hl_pfg_l2 = l2m["pfg_by_luma_non_gtbox"]["[0.8,0.9)"]["mean_pfg"]
    hl_pfg_sq = sqm["pfg_by_luma_non_gtbox"]["[0.8,0.9)"]["mean_pfg"]
    case_a = extra_hl > 0.01 and fp_drop and rec_keep
    case_b = False
    case_c = (not fp_drop) and (not case_b)
    if case_a and not case_b:
        verdict = "CASE A — L2 saturation is important"
    elif case_b and not case_a:
        verdict = "CASE B — Mirror dominates"
    elif case_a:
        verdict = "CASE D — Mixed (L2 saturation + other; ego ROI not quantified)"
    else:
        verdict = "CASE C — General night-domain problem"
        if extra_hl > 0.005 and not fp_drop:
            verdict = "CASE D — Mixed / weak L2 effect, FP remain after sqrt"

    report = {
        "checkpoint": ckpt_path,
        "scene": SCENE_ID,
        "agent": "vehicle",
        "n_sampled": n_keep,
        "seed": 42,
        "sample_ids": sample_ids,
        "tau": TAU,
        "gt": {
            "definition": "official 3D boxes projected to vehicle cameras, tau=1 R90",
            "identical_across_variants": True,
            "n_gt_box_pixels": gt_pixels_total,
            "n_frames_with_projected_objects": n_with_gt,
            "n_frames": len(gt_store),
            "precision_caveat": (
                "pixel precision vs projected 3D-box support is a diagnostic "
                "proxy; box hulls are not SAM3 silhouettes."
            ),
        },
        "preprocessing": {
            "raw": "ToTensor + ImageNet; no night lift",
            "l2": (
                "PRODUCTION TEST path: apply_l2_highlight_compress then ImageNet. "
                "This is clip(sqrt(I/0.25)) PLUS highlight knee/blend, not bare L2."
            ),
            "sqrt": "I_sqrt=sqrt(I) on [0,1] then ImageNet; no /0.25; no highlight compress",
        },
        "ego_roi": {
            "defined": False,
            "reason": (
                "Side mirror appears in different corners depending on camera "
                "(bottom-right, bottom-left, left edge) and the GT-argmax view "
                "index is not constant across frames. No single stable pixel ROI."
            ),
        },
        "by_variant": by_variant,
        "verdict": verdict,
        "verdict_numbers": {
            "extra_frac_luma_ge_0.8_L2_minus_RAW": round(float(extra_hl), 6),
            "sqrt_pred_fg_over_l2": round(
                sqm["pred_fg_pixels"] / max(l2m["pred_fg_pixels"], 1), 4
            ),
            "sqrt_recall_over_l2": round(
                sqm["gt_box_recall@0.3"] / max(l2m["gt_box_recall@0.3"], 1e-6), 4
            ),
            "non_gt_mean_pfg_luma_0.8_0.9_l2": hl_pfg_l2,
            "non_gt_mean_pfg_luma_0.8_0.9_sqrt": hl_pfg_sq,
        },
        "production_code_modified": False,
        "panels": [str(vis_dir / f"{name}_{idx}.png") for name, idx in unique_picks],
        "per_frame_csv": str(csv_path),
    }
    out_json = out_root / "report.json"
    out_txt = out_root / "report.txt"
    out_json.write_text(json.dumps(report, indent=2))
    lines = [
        "TEST vehicle-night heatmap audit (READ-ONLY)",
        f"checkpoint: {ckpt_path}",
        f"scene: {SCENE_ID}  n={n_keep} seed=42  ids={sample_ids}",
        f"GT-box pixels={gt_pixels_total}  frames_with_objects={n_with_gt}/{len(gt_store)}",
        "tau=0.3 fixed. Precision is vs 3D-box support, not SAM3.",
        "",
        f"{'var':6s} {'predFG':>10s} {'ratio':>8s} {'rec':>7s} {'prec':>7s} {'F1':>7s} "
        f"{'luma>=.8':>10s} {'luma>=.95':>10s}",
    ]
    for variant in VARIANTS:
        b = by_variant[variant]
        s = b["saturation"]
        lines.append(
            f"{variant:6s} {b['pred_fg_pixels']:10d} {b['pred_fg_ratio']:8.4f} "
            f"{b['gt_box_recall@0.3']:7.3f} {b['gt_box_precision@0.3']:7.3f} "
            f"{b['gt_box_f1@0.3']:7.3f} {s['frac_luma_ge_0.8']:10.4f} "
            f"{s['frac_luma_ge_0.95']:10.4f}"
        )
    lines.append("")
    lines.append("non-GT p_fg in high-luma bins:")
    for variant in VARIANTS:
        bg = by_variant[variant]["pfg_by_luma_non_gtbox"]
        lines.append(
            f"  {variant}: [0.8,0.9) mean_pfg={bg['[0.8,0.9)']['mean_pfg']} "
            f"frac>=0.3={bg['[0.8,0.9)']['frac_ge_0.3']} n={bg['[0.8,0.9)']['n']} | "
            f"[0.9,1.0] mean_pfg={bg['[0.9,1.0]']['mean_pfg']} "
            f"frac>=0.3={bg['[0.9,1.0]']['frac_ge_0.3']} n={bg['[0.9,1.0]']['n']}"
        )
    lines.append("")
    lines.append(f"ego ROI: not defined (view-dependent mirror location)")
    lines.append(f"VERDICT: {verdict}")
    lines.append("production/training code was NOT modified.")
    lines.append(f"csv: {csv_path}")
    lines.append(f"panels: {vis_dir}")
    text = "\n".join(lines) + "\n"
    out_txt.write_text(text)
    print(text)


if __name__ == "__main__":
    main()

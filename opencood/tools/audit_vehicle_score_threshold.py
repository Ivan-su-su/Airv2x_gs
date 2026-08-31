# -*- coding: utf-8 -*-
"""READ-ONLY vehicle heatmap score/threshold calibration audit.

Compares TEST day vs TEST night p_fg distributions and a tau sweep.
Does not change production threshold, L2, SAM3, or training.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

root_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.gaussian_modules_0822.heatmap.target import (
    binary_objectness_target,
    build_semantic_target,
)
from opencood.tools import train_utils
from opencood.tools.audit_vehicle_night_l2_fp import (
    display_rgb01,
    downsample_4,
    flatten_imgs,
    luma01,
    scene_range,
    to_uint8,
)
from opencood.tools.eval_gaussian_p1 import load_epoch_checkpoint
from opencood.tools.train_gaussian_p1 import _unwrap_model
from opencood.tools.vis_test_heatmap_recall import paint_agent_box_maps

LOG_DIR = Path(
    "/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
    "airv2x_gaussian_p1_joint/concat128_2026_08_29_16_25_05"
)
CKPT_EPOCH = 9
NIGHT_SCENE = "2025_05_10_19_54_35"
DAY_SCENE = "2025_05_05_21_46_39"
TRAIN_NIGHT_SCENE = "2025_05_06_10_01_50"
NIGHT_IDS = [
    3745, 3766, 3775, 3791, 3806, 3808, 3811, 3813, 3829, 3862,
    3876, 3877, 3881, 3888, 3891, 3901, 3911, 3913, 3930, 3932,
    3944, 3956, 3986, 4007, 4013, 4033, 4054, 4067, 4087, 4088,
    4098, 4103, 4110, 4123, 4139, 4146, 4153, 4156, 4159, 4160,
]
TAUS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
Q_LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
OUT = Path("/mnt/home/suyi/visualization/concat128_p1_vehicle_score_audit")


def quantiles(arr: np.ndarray) -> Dict[str, Optional[float]]:
    """Named quantiles; None if empty."""
    if arr.size == 0:
        out_empty = {f"q{int(q * 100):02d}": None for q in Q_LEVELS}
        out_empty.update({"mean": None, "std": None, "median": None, "n": 0})
        return out_empty
    qs = np.quantile(arr, Q_LEVELS)
    out = {f"q{int(q * 100):02d}": round(float(v), 4) for q, v in zip(Q_LEVELS, qs)}
    out["mean"] = round(float(arr.mean()), 4)
    out["std"] = round(float(arr.std()), 4)
    out["median"] = round(float(np.median(arr)), 4)
    out["n"] = int(arr.size)
    return out


def frac_ge(arr: np.ndarray, tau: float) -> Optional[float]:
    """Fraction of values >= tau."""
    if arr.size == 0:
        return None
    return round(float((arr >= tau).mean()), 4)


def auroc_auprc(y: np.ndarray, s: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """GT-box-proxy AUROC / AUPRC; skip if one class missing."""
    if y.size == 0 or int(y.min()) == int(y.max()):
        return None, None
    # subsample if huge to keep sklearn tractable
    if y.size > 2_000_000:
        rng = np.random.RandomState(0)
        pick = rng.choice(y.size, size=2_000_000, replace=False)
        y, s = y[pick], s[pick]
    return (
        round(float(roc_auc_score(y, s)), 4),
        round(float(average_precision_score(y, s)), 4),
    )


def sweep_metrics(p_fg: np.ndarray, gt: np.ndarray, tau: float) -> Dict[str, float]:
    """Pixel metrics at one threshold."""
    pred = p_fg >= tau
    gt_b = gt.astype(bool)
    tp = int((pred & gt_b).sum())
    fn = int((~pred & gt_b).sum())
    fp = int((pred & ~gt_b).sum())
    n_gt = tp + fn
    n_pred = tp + fp
    n_bg = int((~gt_b).sum())
    rec = tp / max(n_gt, 1)
    prec = tp / max(n_pred, 1)
    f1 = 0.0 if rec + prec <= 0 else 2 * rec * prec / (rec + prec)
    return {
        "pred_fg_pixels": n_pred,
        "pred_fg_ratio": round(float(pred.mean()), 6),
        "gt_box_recall": round(rec, 4),
        "gt_box_precision": round(prec, 4),
        "gt_box_f1": round(f1, 4),
        "bg_fp_pixels": fp,
        "bg_fp_rate": round(fp / max(n_bg, 1), 6),
        "n_gt": n_gt,
        "n_bg": n_bg,
    }


def lowest_tau_for_ratio(sweep: Dict[str, Dict[str, float]], cap: float) -> Optional[str]:
    """Smallest tau whose pred FG ratio is <= cap."""
    for tau in TAUS:
        key = f"{tau:.2f}"
        if sweep[key]["pred_fg_ratio"] <= cap:
            return key
    return None


def highest_tau_for_recall(sweep: Dict[str, Dict[str, float]], floor: float) -> Optional[str]:
    """Largest tau that still has recall >= floor."""
    hit = None
    for tau in TAUS:
        key = f"{tau:.2f}"
        if sweep[key]["gt_box_recall"] >= floor:
            hit = key
    return hit


def nearest_tau_for_ratio(sweep: Dict[str, Dict[str, float]], target: float) -> str:
    """Tau whose pred FG ratio is closest to target."""
    best = TAUS[0]
    best_d = 1e9
    for tau in TAUS:
        d = abs(sweep[f"{tau:.2f}"]["pred_fg_ratio"] - target)
        if d < best_d:
            best_d = d
            best = tau
    return f"{best:.2f}"


def plot_hist(path: Path, gt: np.ndarray, bg: np.ndarray, title: str, bg_label: str = "BG") -> None:
    """Overlay GT vs BG p_fg histograms on [0,1]."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bins = np.linspace(0.0, 1.0, 41)
    if gt.size:
        ax.hist(gt, bins=bins, density=True, alpha=0.55, label=f"GT n={gt.size}", color="#d62728")
    if bg.size:
        ax.hist(bg, bins=bins, density=True, alpha=0.45, label=f"{bg_label} n={bg.size}", color="#1f77b4")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("p_fg")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_curves(path: Path, sweep: Dict[str, Dict[str, float]], title: str) -> None:
    """Recall / pred-FG / BG-FP vs tau on twin axes."""
    xs = [float(t) for t in TAUS]
    rec = [sweep[f"{t:.2f}"]["gt_box_recall"] for t in TAUS]
    ratio = [sweep[f"{t:.2f}"]["pred_fg_ratio"] for t in TAUS]
    fpr = [sweep[f"{t:.2f}"]["bg_fp_rate"] for t in TAUS]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(xs, rec, "o-", color="#d62728", label="GT-box recall")
    ax.set_ylabel("GT-box recall", color="#d62728")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("tau")
    ax2 = ax.twinx()
    ax2.plot(xs, ratio, "s--", color="#1f77b4", label="pred FG ratio")
    ax2.plot(xs, fpr, "^:", color="#2ca02c", label="BG FP rate")
    ax2.set_ylabel("pred FG ratio / BG FP rate")
    ax2.set_ylim(0.0, 1.05)
    ax.set_title(title)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_frame_panel(
    path: Path,
    rgb: np.ndarray,
    gt: np.ndarray,
    p_fg: np.ndarray,
    title: str,
) -> None:
    """RGB | GT | p_fg | FG@0.3 | 0.5 | 0.7."""
    fig, axes = plt.subplots(1, 6, figsize=(20.0, 3.3))
    axes[0].imshow(rgb)
    axes[0].set_title("L2_hl RGB")
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT-box")
    im = axes[2].imshow(p_fg, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("p_fg")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax, tau, lab in (
        (axes[3], 0.3, "FG@0.3"),
        (axes[4], 0.5, "FG@0.5"),
        (axes[5], 0.7, "FG@0.7"),
    ):
        ax.imshow((p_fg >= tau).astype(np.float32), cmap="gray", vmin=0, vmax=1)
        ax.set_title(lab)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_split(
    dataset: Any,
    model: torch.nn.Module,
    device: torch.device,
    ids: Sequence[int],
    split: str,
    scene: str,
    gt_mode: str,
) -> Dict[str, Any]:
    """Infer vehicle p_fg on a fixed id list. gt_mode is box or sam3."""
    frames: List[Dict[str, Any]] = []
    gt_all: List[np.ndarray] = []
    bg_all: List[np.ndarray] = []
    dark_bg_all: List[np.ndarray] = []
    mid_bg_all: List[np.ndarray] = []
    cam_acc: Dict[int, Dict[str, List[np.ndarray]]] = {}
    with torch.no_grad():
        for i, idx in enumerate(ids):
            sample = dataset[idx]
            batch = dataset.collate_batch_test([sample])
            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            if "vehicle" not in ego:
                print(f"  skip {split} idx={idx}: no vehicle")
                continue
            pred = model(ego)
            if "vehicle" not in pred:
                print(f"  skip {split} idx={idx}: no pred")
                continue
            imgs = flatten_imgs(ego["vehicle"]["batch_merged_cam_inputs"]["imgs"])
            p_fg = torch.softmax(pred["vehicle"]["heatmap_logits"], dim=1)[:, 1]
            p_np = p_fg.detach().cpu().numpy()
            if gt_mode == "sam3":
                cam = ego["vehicle"]["batch_merged_cam_inputs"]
                try:
                    gt_t = build_semantic_target(cam, tau=1)
                    gt_np = gt_t.detach().cpu().numpy().astype(np.uint8)
                except KeyError:
                    print(f"  skip {split} idx={idx}: no SAM3")
                    continue
            else:
                maps = paint_agent_box_maps(ego, "vehicle")
                gt_np = binary_objectness_target(maps, tau=1).cpu().numpy().astype(np.uint8)
            n_view = min(int(p_np.shape[0]), int(gt_np.shape[0]), int(imgs.shape[0]))
            p_np = p_np[:n_view]
            gt_np = gt_np[:n_view]
            disp = np.stack([display_rgb01(imgs[v]) for v in range(n_view)], 0)
            luma = downsample_4(np.stack([luma01(disp[v]) for v in range(n_view)], 0))
            gt_b = gt_np > 0
            bg_b = ~gt_b
            dark_b = bg_b & (luma < 0.1)
            mid_b = bg_b & (luma >= 0.1) & (luma < 0.4)
            gt_all.append(p_np[gt_b])
            bg_all.append(p_np[bg_b])
            dark_bg_all.append(p_np[dark_b])
            mid_bg_all.append(p_np[mid_b])
            scores = gt_np.reshape(n_view, -1).sum(1)
            view = int(np.argmax(scores)) if scores.max() > 0 else 0
            frames.append(
                {
                    "idx": int(idx),
                    "p_fg": p_np,
                    "gt": gt_np,
                    "luma": luma,
                    "rgb": to_uint8(disp[view]),
                    "view": view,
                    "n_view": n_view,
                }
            )
            for cam_i in range(n_view):
                cam_acc.setdefault(cam_i, {"p": [], "gt": []})
                cam_acc[cam_i]["p"].append(p_np[cam_i].reshape(-1))
                cam_acc[cam_i]["gt"].append(gt_np[cam_i].reshape(-1))
            print(
                f"  {split} {i+1}/{len(ids)} idx={idx} views={n_view} "
                f"gt={int(gt_b.sum())} p_mean={float(p_np.mean()):.3f}"
            )
    p_gt = np.concatenate(gt_all) if gt_all else np.zeros(0, np.float32)
    p_bg = np.concatenate(bg_all) if bg_all else np.zeros(0, np.float32)
    p_dark = np.concatenate(dark_bg_all) if dark_bg_all else np.zeros(0, np.float32)
    p_mid = np.concatenate(mid_bg_all) if mid_bg_all else np.zeros(0, np.float32)
    p_cat = np.concatenate([fr["p_fg"].reshape(-1) for fr in frames]) if frames else np.zeros(0)
    g_cat = np.concatenate([fr["gt"].reshape(-1) for fr in frames]) if frames else np.zeros(0, np.uint8)
    y = (g_cat > 0).astype(np.int32)
    auroc, auprc = auroc_auprc(y, p_cat.astype(np.float64))
    sweep = {f"{tau:.2f}": sweep_metrics(p_cat, g_cat, tau) for tau in TAUS}
    per_frame_ratio = []
    per_frame_rec = []
    for fr in frames:
        m03 = sweep_metrics(fr["p_fg"], fr["gt"], 0.30)
        per_frame_ratio.append(m03["pred_fg_ratio"])
        per_frame_rec.append(m03["gt_box_recall"])
    cam_report: Dict[str, Any] = {}
    for cam_i, block in sorted(cam_acc.items()):
        p = np.concatenate(block["p"])
        g = np.concatenate(block["gt"])
        gb = g > 0
        cam_sweep = {f"{tau:.2f}": sweep_metrics(p, g, tau) for tau in TAUS}
        cam_report[str(cam_i)] = {
            "n_pixels": int(p.size),
            "gt_box_pixels": int(gb.sum()),
            "mean_bg_pfg": round(float(p[~gb].mean()), 4) if (~gb).any() else None,
            "bg_frac_ge_0.3": frac_ge(p[~gb], 0.3),
            "pred_fg_ratio@0.3": cam_sweep["0.30"]["pred_fg_ratio"],
            "gt_recall@0.3": cam_sweep["0.30"]["gt_box_recall"],
            "sweep": cam_sweep,
        }
    return {
        "split": split,
        "scene": scene,
        "gt_mode": gt_mode,
        "n_frames": len(frames),
        "ids": [fr["idx"] for fr in frames],
        "gt_pixels": int(p_gt.size),
        "bg_pixels": int(p_bg.size),
        "frames_with_gt": int(sum(1 for fr in frames if int(fr["gt"].sum()) > 0)),
        "gt_scores": quantiles(p_gt),
        "bg_scores": quantiles(p_bg),
        "dark_bg_scores": quantiles(p_dark),
        "mid_bg_scores": quantiles(p_mid),
        "mean_pfg_gt": round(float(p_gt.mean()), 4) if p_gt.size else None,
        "median_pfg_gt": round(float(np.median(p_gt)), 4) if p_gt.size else None,
        "mean_pfg_bg": round(float(p_bg.mean()), 4) if p_bg.size else None,
        "median_pfg_bg": round(float(np.median(p_bg)), 4) if p_bg.size else None,
        "gt_frac_ge": {f"{t:.1f}": frac_ge(p_gt, t) for t in (0.3, 0.5, 0.7)},
        "bg_frac_ge": {f"{t:.1f}": frac_ge(p_bg, t) for t in (0.3, 0.5, 0.7)},
        "dark_bg_frac_ge": {f"{t:.1f}": frac_ge(p_dark, t) for t in (0.3, 0.5, 0.7)},
        "auroc_gtbox_proxy": auroc,
        "auprc_gtbox_proxy": auprc,
        "sweep": sweep,
        "per_frame_pred_fg_ratio@0.3": {
            "median": round(float(np.median(per_frame_ratio)), 4) if per_frame_ratio else None,
            "q10": round(float(np.quantile(per_frame_ratio, 0.1)), 4) if per_frame_ratio else None,
            "q90": round(float(np.quantile(per_frame_ratio, 0.9)), 4) if per_frame_ratio else None,
        },
        "per_frame_recall@0.3_median": (
            round(float(np.median(per_frame_rec)), 4) if per_frame_rec else None
        ),
        "per_camera": cam_report,
        "_arrays": {
            "gt": p_gt,
            "bg": p_bg,
            "dark": p_dark,
            "mid": p_mid,
        },
        "_frames": frames,
    }


def main() -> None:
    """Run day/night vehicle score audit."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    (OUT / "panels").mkdir(exist_ok=True)
    hypes = yaml_utils.load_yaml(str(LOG_DIR / "config.yaml"), None)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    hypes_test = dict(hypes)
    hypes_test["validate_dir"] = hypes["test_dir"]
    hypes_test["train"] = False
    print("Building TEST dataset...")
    test_ds = build_dataset(hypes_test, visualize=False, train=False)
    day_lo, day_hi = scene_range(test_ds, DAY_SCENE)
    rng = np.random.RandomState(42)
    day_ids = sorted(int(day_lo + x) for x in rng.choice(day_hi - day_lo, size=min(40, day_hi - day_lo), replace=False))
    print(f"DAY scene {DAY_SCENE} [{day_lo},{day_hi}) ids={day_ids}")
    print(f"NIGHT scene {NIGHT_SCENE} ids={NIGHT_IDS}")

    model = train_utils.create_model(hypes)
    ckpt = load_epoch_checkpoint(model, str(LOG_DIR), CKPT_EPOCH)
    model = _unwrap_model(model)
    model.to(device)
    model.eval()

    print("=== TEST NIGHT ===")
    night = run_split(test_ds, model, device, NIGHT_IDS, "test_night", NIGHT_SCENE, "box")
    print("=== TEST DAY ===")
    day = run_split(test_ds, model, device, day_ids, "test_day", DAY_SCENE, "box")

    train_night = None
    try:
        hypes_tr = dict(hypes)
        hypes_tr["validate_dir"] = hypes["root_dir"]
        hypes_tr["train"] = False
        print("Building TRAIN dataset (eval crop, L2 on night, no fog)...")
        train_ds = build_dataset(hypes_tr, visualize=False, train=False)
        tn_lo, tn_hi = scene_range(train_ds, TRAIN_NIGHT_SCENE)
        tn_ids = sorted(
            int(tn_lo + x)
            for x in rng.choice(tn_hi - tn_lo, size=min(40, tn_hi - tn_lo), replace=False)
        )
        print(f"TRAIN NIGHT {TRAIN_NIGHT_SCENE} [{tn_lo},{tn_hi}) ids={tn_ids}")
        print("=== TRAIN NIGHT (SAM3 target, not mixed with TEST box) ===")
        train_night = run_split(
            train_ds, model, device, tn_ids, "train_night", TRAIN_NIGHT_SCENE, "sam3"
        )
    except Exception as exc:
        print(f"TRAIN night skipped: {exc}")

    ns, ds_ = night["sweep"], day["sweep"]
    constrained = {
        "lowest_tau_pred_fg_le_0.20": None,
        "lowest_tau_pred_fg_le_0.10": None,
        "lowest_tau_pred_fg_le_0.05": None,
        "highest_tau_recall_ge_0.70": None,
        "highest_tau_recall_ge_0.60": None,
        "highest_tau_recall_ge_0.50": None,
    }
    for cap, key in ((0.20, "lowest_tau_pred_fg_le_0.20"), (0.10, "lowest_tau_pred_fg_le_0.10"), (0.05, "lowest_tau_pred_fg_le_0.05")):
        hit = lowest_tau_for_ratio(ns, cap)
        constrained[key] = None if hit is None else {"tau": float(hit), **ns[hit]}
    for floor, key in ((0.70, "highest_tau_recall_ge_0.70"), (0.60, "highest_tau_recall_ge_0.60"), (0.50, "highest_tau_recall_ge_0.50")):
        hit = highest_tau_for_recall(ns, floor)
        constrained[key] = None if hit is None else {"tau": float(hit), **ns[hit]}

    tau_day_05 = nearest_tau_for_ratio(ds_, 0.05)
    tau_day_10 = nearest_tau_for_ratio(ds_, 0.10)
    transfer = {
        "day_tau_nearest_pred_fg_5pct": {
            "tau": float(tau_day_05),
            "day": ds_[tau_day_05],
            "night": ns[tau_day_05],
        },
        "day_tau_nearest_pred_fg_10pct": {
            "tau": float(tau_day_10),
            "day": ds_[tau_day_10],
            "night": ns[tau_day_10],
        },
    }

    plots = OUT / "plots"
    plot_hist(plots / "day_gt_vs_bg.png", day["_arrays"]["gt"], day["_arrays"]["bg"], "TEST DAY vehicle p_fg (GT-box proxy)")
    plot_hist(plots / "night_gt_vs_bg.png", night["_arrays"]["gt"], night["_arrays"]["bg"], "TEST NIGHT vehicle p_fg (GT-box proxy)")
    plot_hist(
        plots / "night_gt_vs_dark_bg.png",
        night["_arrays"]["gt"],
        night["_arrays"]["dark"],
        "TEST NIGHT GT vs dark BG (luma<0.1)",
        bg_label="dark BG",
    )
    plot_curves(plots / "night_threshold_curves.png", ns, "TEST NIGHT threshold sweep")
    plot_curves(plots / "day_threshold_curves.png", ds_, "TEST DAY threshold sweep")

    # representative night frames
    frs = night["_frames"]
    stats = [(sweep_metrics(fr["p_fg"], fr["gt"], 0.30), fr) for fr in frs]
    picks: List[Tuple[str, Dict[str, Any]]] = []
    if stats:
        picks.append(("severe_fp", max(stats, key=lambda x: x[0]["pred_fg_ratio"])[1]))
        picks.append(("median", sorted(stats, key=lambda x: x[0]["pred_fg_ratio"])[len(stats) // 2][1]))
        with_gt = [x for x in stats if x[1]["gt"].sum() > 0]
        if with_gt:
            picks.append(("relatively_good", min(with_gt, key=lambda x: x[0]["pred_fg_ratio"])[1]))
            picks.append(("object_rich", max(with_gt, key=lambda x: int(x[1]["gt"].sum()))[1]))
    seen = set()
    panel_paths = []
    for name, fr in picks:
        if fr["idx"] in seen:
            continue
        seen.add(fr["idx"])
        v = fr["view"]
        path = OUT / "panels" / f"{name}_{fr['idx']}.png"
        save_frame_panel(
            path,
            fr["rgb"],
            fr["gt"][v],
            fr["p_fg"][v],
            f"{NIGHT_SCENE} idx={fr['idx']} view={v}  {name}",
        )
        panel_paths.append(str(path))

    csv_path = OUT / "per_frame_threshold_audit.csv"
    fieldnames = [
        "split", "scene", "frame_id", "camera_id", "tau",
        "gt_pixels", "pred_fg_pixels", "pred_fg_ratio",
        "gt_recall", "gt_precision_proxy", "gt_f1_proxy",
        "bg_fp_pixels", "bg_fp_rate",
        "mean_gt_pfg", "mean_bg_pfg", "mean_dark_bg_pfg",
        "gt_mode",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        packs = [night, day] + ([train_night] if train_night else [])
        for pack in packs:
            for fr in pack["_frames"]:
                luma = fr["luma"]
                gt_b = fr["gt"] > 0
                dark = (~gt_b) & (luma < 0.1)
                for cam_i in range(fr["n_view"]):
                    p = fr["p_fg"][cam_i]
                    g = fr["gt"][cam_i]
                    gb = g > 0
                    db = dark[cam_i]
                    for tau in TAUS:
                        m = sweep_metrics(p, g, tau)
                        writer.writerow(
                            {
                                "split": pack["split"],
                                "scene": pack["scene"],
                                "frame_id": fr["idx"],
                                "camera_id": cam_i,
                                "tau": f"{tau:.2f}",
                                "gt_pixels": m["n_gt"],
                                "pred_fg_pixels": m["pred_fg_pixels"],
                                "pred_fg_ratio": m["pred_fg_ratio"],
                                "gt_recall": m["gt_box_recall"],
                                "gt_precision_proxy": m["gt_box_precision"],
                                "gt_f1_proxy": m["gt_box_f1"],
                                "bg_fp_pixels": m["bg_fp_pixels"],
                                "bg_fp_rate": m["bg_fp_rate"],
                                "mean_gt_pfg": round(float(p[gb].mean()), 4) if gb.any() else "",
                                "mean_bg_pfg": round(float(p[~gb].mean()), 4) if (~gb).any() else "",
                                "mean_dark_bg_pfg": round(float(fr["p_fg"][cam_i][db].mean()), 4) if db.any() else "",
                                "gt_mode": pack["gt_mode"],
                            }
                        )

    # verdict
    n03, n05, n07 = ns["0.30"], ns["0.50"], ns["0.70"]
    auroc_n = night["auroc_gtbox_proxy"] or 0.0
    auroc_d = day["auroc_gtbox_proxy"] or 0.0
    rec_drop = n03["gt_box_recall"] - n05["gt_box_recall"]
    ratio_drop = n03["pred_fg_ratio"] - n05["pred_fg_ratio"]
    sparse_ok = constrained["lowest_tau_pred_fg_le_0.10"] is not None
    rec_at_sparse = (
        constrained["lowest_tau_pred_fg_le_0.10"]["gt_box_recall"]
        if sparse_ok else 0.0
    )
    cam_ratios = [v["pred_fg_ratio@0.3"] for v in night["per_camera"].values()]
    cam_spread = max(cam_ratios) - min(cam_ratios) if cam_ratios else 0.0
    if cam_spread > 0.35 and max(cam_ratios) > 0.6:
        verdict = "CASE C — CAMERA-SPECIFIC FAILURE"
        can_fix = "No: concentrated in worst cameras; a global tau is not the right lever."
    elif auroc_n >= 0.80 and ratio_drop >= 0.20 and rec_drop <= 0.15 and rec_at_sparse >= 0.50:
        verdict = "CASE A — MAINLY SCORE / CALIBRATION SHIFT"
        can_fix = "Thresholding appears capable of sparsifying night FG while keeping useful GT-box recall. This is diagnosis only; production tau was not changed."
    elif auroc_n < 0.70 or rec_drop >= 0.20 and ratio_drop < 0.15 or (sparse_ok and rec_at_sparse < 0.35) or (not sparse_ok):
        verdict = "CASE B — TRUE NIGHT SEPARABILITY FAILURE"
        can_fix = "No reasonable global tau both sparsifies FG and keeps useful recall. Threshold tuning cannot solve this."
    else:
        verdict = "CASE D — MIXED"
        can_fix = "There is some score shift, but GT/BG overlap remains large enough that thresholding is not a clean fix."

    def strip(pack: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if pack is None:
            return None
        return {k: v for k, v in pack.items() if not k.startswith("_")}

    report = {
        "checkpoint": ckpt,
        "night_scene": NIGHT_SCENE,
        "night_ids": NIGHT_IDS,
        "day_scene": DAY_SCENE,
        "day_ids": day_ids,
        "day_scene_note": (
            "2025_05_10_21_46_39 is not in TEST; used verified clear-day "
            "2025_05_05_21_46_39 from scenario_utils."
        ),
        "gt_proxy": "official 3D boxes -> vehicle camera -> R90 tau=1; not SAM3",
        "precision_caveat": "pixel precision vs box hull is a diagnostic proxy",
        "test_night": strip(night),
        "test_day": strip(day),
        "train_night_sam3_reference": strip(train_night),
        "train_night_note": (
            None if train_night is None else
            "TRAIN night uses SAM3 targets; do not mix precision with TEST GT-box."
        ),
        "recall_constrained": constrained,
        "day_to_night_transfer": transfer,
        "verdict": verdict,
        "thresholding_can_fix_night": can_fix,
        "panels": panel_paths,
        "plots": [str(p) for p in sorted(plots.glob("*.png"))],
        "per_frame_csv": str(csv_path),
        "production_code_modified": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))

    def qline(tag: str, d: Dict[str, Any]) -> str:
        return (
            f"  {tag:10s} n={d.get('n', 0):8d} mean={d.get('mean')} med={d.get('median')} "
            f"q10={d.get('q10')} q50={d.get('q50')} q90={d.get('q90')} q99={d.get('q99')}"
        )

    lines = [
        "VEHICLE heatmap score/threshold audit (READ-ONLY)",
        f"checkpoint: {ckpt}",
        f"NIGHT {NIGHT_SCENE} n=40 seed=42 (same ids as L2 FP audit)",
        f"DAY   {DAY_SCENE} n=40 seed=42 ids={day_ids}",
        "GT TEST = 3D-box R90 proxy. TRAIN night ref = SAM3 (not mixed).",
        "",
        f"DAY   GT mean/med={day['mean_pfg_gt']}/{day['median_pfg_gt']}  "
        f"BG mean/med={day['mean_pfg_bg']}/{day['median_pfg_bg']}  "
        f"AUROC={day['auroc_gtbox_proxy']} AUPRC={day['auprc_gtbox_proxy']}",
        qline("DAY GT", day["gt_scores"]),
        qline("DAY BG", day["bg_scores"]),
        f"NIGHT GT mean/med={night['mean_pfg_gt']}/{night['median_pfg_gt']}  "
        f"BG mean/med={night['mean_pfg_bg']}/{night['median_pfg_bg']}  "
        f"AUROC={night['auroc_gtbox_proxy']} AUPRC={night['auprc_gtbox_proxy']}",
        qline("NIGHT GT", night["gt_scores"]),
        qline("NIGHT BG", night["bg_scores"]),
        qline("NIGHT darkBG", night["dark_bg_scores"]),
        f"  GT  >=0.3/0.5/0.7 = {night['gt_frac_ge']}",
        f"  BG  >=0.3/0.5/0.7 = {night['bg_frac_ge']}",
        f"  darkBG >=0.3/0.5/0.7 = {night['dark_bg_frac_ge']}",
        "",
        f"{'tau':>6} {'Nrec':>7} {'Nratio':>8} {'Nprec':>7} {'Drec':>7} {'Dratio':>8}",
    ]
    for tau in TAUS:
        k = f"{tau:.2f}"
        lines.append(
            f"{tau:6.2f} {ns[k]['gt_box_recall']:7.3f} {ns[k]['pred_fg_ratio']:8.3f} "
            f"{ns[k]['gt_box_precision']:7.3f} {ds_[k]['gt_box_recall']:7.3f} "
            f"{ds_[k]['pred_fg_ratio']:8.3f}"
        )
    lines.append("")
    lines.append("recall-constrained (NIGHT):")
    for key, val in constrained.items():
        if val is None:
            lines.append(f"  {key}: NONE")
        else:
            lines.append(
                f"  {key}: tau={val['tau']:.2f} ratio={val['pred_fg_ratio']:.3f} "
                f"rec={val['gt_box_recall']:.3f} bg_fp={val['bg_fp_rate']:.3f}"
            )
    lines.append("")
    lines.append(
        f"DAY->NIGHT transfer tau~5%FG: tau={transfer['day_tau_nearest_pred_fg_5pct']['tau']:.2f} "
        f"day R={ds_[tau_day_05]['gt_box_recall']:.3f} ratio={ds_[tau_day_05]['pred_fg_ratio']:.3f} | "
        f"night R={ns[tau_day_05]['gt_box_recall']:.3f} ratio={ns[tau_day_05]['pred_fg_ratio']:.3f}"
    )
    lines.append(
        f"DAY->NIGHT transfer tau~10%FG: tau={transfer['day_tau_nearest_pred_fg_10pct']['tau']:.2f} "
        f"day R={ds_[tau_day_10]['gt_box_recall']:.3f} ratio={ds_[tau_day_10]['pred_fg_ratio']:.3f} | "
        f"night R={ns[tau_day_10]['gt_box_recall']:.3f} ratio={ns[tau_day_10]['pred_fg_ratio']:.3f}"
    )
    lines.append("")
    lines.append("NIGHT per-camera @0.3:")
    for cam, block in night["per_camera"].items():
        lines.append(
            f"  cam{cam}: gt={block['gt_box_pixels']} meanBG={block['mean_bg_pfg']} "
            f"BGge0.3={block['bg_frac_ge_0.3']} ratio={block['pred_fg_ratio@0.3']} "
            f"rec={block['gt_recall@0.3']}"
        )
    lines.append("")
    lines.append(f"VERDICT: {verdict}")
    lines.append(f"thresholding can fix night?: {can_fix}")
    lines.append("production/training code was NOT modified.")
    lines.append(f"csv: {csv_path}")
    text = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

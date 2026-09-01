# -*- coding: utf-8 -*-
"""A/B probe: RSU collated-extrinsic invert vs identity.

Diagnostic only. Does not change training, model, or dataset.

Compares two conventions for ``cam_to_lidar``:

* identity: stored ``extrinsics`` are camera-to-lidar after ``ue4_to_lss``
  (same as vehicle / drone / production LSS).
* invert: stored matrices are treated as lidar-to-camera (current RSU
  diagnostic ``LIDAR2CAM_STORED=True``).

Reports 2D GT-box vs pred-FG IoU and 3D seed-to-box distance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.gaussian_modules_0822.heatmap.box_support import (
    project_box_to_image,
    rasterize_convex_polygon,
)
from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    PRIMARY_OBJECTNESS_THRESHOLD,
)
from opencood.models.gaussian_modules_0822.heatmap.target import build_semantic_target
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.models.gaussian_modules_0822.p1_layout import FEAT_H, FEAT_W
from opencood.tools import train_utils
from opencood.tools.audit_gaussian_scale_coverage import (
    _as_numpy,
    _record_len,
    _slice_cam,
    flatten_imgs,
    gt_boxes_from_ego,
    load_split_dataset,
    n_views_of,
    pairwise_for_agent,
    scene_from_path,
)
from opencood.tools.eval_gaussian_p1 import denormalize_rgb, load_epoch_checkpoint
from opencood.tools.gaussian_scale_audit.coverage import (
    boxes_hwl_to_corners,
    points_in_boxes,
)
from opencood.tools.gaussian_scale_audit.geometry import (
    _as_44,
    lift_cam,
    r90_pixel_centers,
    ray_dir_cam,
    transform_points,
)
from opencood.tools.train_gaussian_p1 import _unwrap_model

AGENT_ORDER = ("vehicle", "rsu", "drone")
CONVENTIONS = ("identity", "invert")


def parse_args() -> argparse.Namespace:
    """CLI for the RSU extrinsic A/B probe."""
    parser = argparse.ArgumentParser(description="Probe RSU extrinsic invert")
    parser.add_argument("-y", "--hypes_yaml", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epoch", type=int, default=16)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--indices",
        default="test:2745,val:661,val:220",
        help="split:idx pairs",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/dell/suyi/visualization/rsu_extrinsic_invert_probe",
    )
    parser.add_argument("--fg_tau", type=float, default=PRIMARY_OBJECTNESS_THRESHOLD)
    parser.add_argument("--max_lift_seeds", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_indices(text: str) -> List[Tuple[str, int]]:
    """Parse ``val:12,test:3`` into split/idx pairs."""
    pairs: List[Tuple[str, int]] = []
    for tok in str(text).split(","):
        split, idx_s = tok.strip().split(":")
        pairs.append((split.strip(), int(idx_s)))
    return pairs


def cam2lidar(ext: np.ndarray, convention: str) -> np.ndarray:
    """Return camera-to-lidar under ``identity`` or ``invert``."""
    mat = _as_44(ext)
    if convention == "invert":
        return np.linalg.inv(mat)
    return mat


def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


def project_view(
    xyz_cav: np.ndarray,
    k: np.ndarray,
    ext: np.ndarray,
    prot: np.ndarray,
    ptra: np.ndarray,
    image_hw: Tuple[int, int],
    convention: str,
) -> Tuple[np.ndarray, int, List[Optional[np.ndarray]]]:
    """Rasterize GT boxes for one view. Returns mask, n_valid, polygons."""
    height, width = image_hw
    mask = np.zeros((height, width), dtype=bool)
    n_valid = 0
    polys: List[Optional[np.ndarray]] = []
    cam2l = cam2lidar(ext, convention)
    for box_i in range(int(xyz_cav.shape[0])):
        proj = project_box_to_image(
            xyz_cav[box_i], k, cam2l, prot, ptra, image_hw
        )
        if proj is None:
            polys.append(None)
            continue
        pts, _z = proj
        poly = rasterize_convex_polygon(pts, height, width)
        polys.append(pts if bool(poly.any()) else None)
        if bool(poly.any()):
            mask[poly] = True
            n_valid += 1
        else:
            polys[-1] = None
    return mask, n_valid, polys


def lift_seeds_ego(
    seed_u: np.ndarray,
    seed_v: np.ndarray,
    z_cell: np.ndarray,
    intrinsic: np.ndarray,
    post_rot: np.ndarray,
    post_trans: np.ndarray,
    extrinsic: np.ndarray,
    t_cav2ego: np.ndarray,
    convention: str,
) -> np.ndarray:
    """Lift optical-axis z along the LSS ray into ego XYZ."""
    n_seed = int(seed_u.size)
    if n_seed == 0:
        return np.zeros((0, 3), dtype=np.float64)
    q = ray_dir_cam(seed_u, seed_v, intrinsic, post_rot, post_trans).reshape(n_seed, 3)
    mu_cam = lift_cam(q, z_cell)
    cam2ego = _as_44(t_cav2ego) @ cam2lidar(extrinsic, convention)
    return transform_points(mu_cam, cam2ego)


def nearest_xy_dist(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """BEV distance from each point to the nearest box center."""
    if points.shape[0] == 0 or boxes.shape[0] == 0:
        return np.zeros((points.shape[0],), dtype=np.float64)
    centers = boxes[:, :2]
    d2 = ((points[:, None, :2] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))


def subsample_seeds(
    ys: np.ndarray, xs: np.ndarray, max_n: int, rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    """Cap seed count for 3D lift."""
    n = int(ys.size)
    if n <= max_n:
        return ys, xs
    pick = rng.choice(n, size=int(max_n), replace=False)
    return ys[pick], xs[pick]


def save_overlay(
    rgb: np.ndarray,
    p_fg: np.ndarray,
    polys_id: Sequence[Optional[np.ndarray]],
    polys_inv: Sequence[Optional[np.ndarray]],
    path: Path,
    title: str,
) -> None:
    """RGB | p_fg | identity boxes | invert boxes."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[1].imshow(p_fg, vmin=0.0, vmax=1.0, cmap="magma")
    axes[1].set_title("p_fg")
    axes[2].imshow(rgb)
    axes[2].set_title("GT identity (no invert)")
    axes[3].imshow(rgb)
    axes[3].set_title("GT invert (current RSU diag)")
    for ax, polys in ((axes[2], polys_id), (axes[3], polys_inv)):
        for poly in polys:
            if poly is None or len(poly) < 3:
                continue
            closed = np.vstack([poly, poly[0]])
            ax.plot(closed[:, 0], closed[:, 1], color="lime", linewidth=1.2)
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def agent_probe(
    ego: Mapping[str, Any],
    pred: Mapping[str, Any],
    agent: str,
    boxes: np.ndarray,
    fg_tau: float,
    max_lift_seeds: int,
    rng: np.random.RandomState,
    u_map: np.ndarray,
    v_map: np.ndarray,
    out_dir: Path,
    split: str,
    idx: int,
    scene: str,
) -> Dict[str, Any]:
    """Run invert A/B for one present agent on one frame."""
    cam = ego[agent]["batch_merged_cam_inputs"]
    imgs = flatten_imgs(cam["imgs"])
    n_cav = _record_len(ego, agent)
    n_view = n_views_of(imgs, n_cav, agent)
    n_flat = int(imgs.shape[0])
    logits = pred[agent]["heatmap_logits"]
    p_fg = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    sam3 = build_semantic_target(cam, tau=1).detach().cpu().numpy()
    sam3_fg = float((sam3 > 0).mean()) if sam3.size else 0.0
    pred_fg = float((p_fg >= fg_tau).mean())
    image_hw = (int(imgs.shape[-2]), int(imgs.shape[-1]))
    pairwise = pairwise_for_agent(ego, agent)
    intrins = _as_numpy(cam["intrinsics"])
    extrinsics = _as_numpy(cam["extrinsics"])
    post_rots = _as_numpy(cam["post_rots"])
    post_trans = _as_numpy(cam["post_trans"])
    z_mean = pred[agent]["depth_z_mean"].detach().cpu().numpy()
    corners_ego = boxes_hwl_to_corners(boxes)
    ones = np.ones((max(corners_ego.shape[0], 1) * 8, 1), dtype=np.float64)

    per_conv: Dict[str, Dict[str, Any]] = {
        name: {
            "n_valid_proj": 0,
            "n_visible_gt": 0,
            "iou_sum": 0.0,
            "iou_n": 0,
            "n_seeds": 0,
            "n_seeds_in_box": 0,
            "n_gt_near_2d": 0,
            "xy_d": [],
        }
        for name in CONVENTIONS
    }
    overlay_saved = False

    for flat in range(n_flat):
        local = flat // n_view
        view = flat % n_view
        k = _slice_cam(intrins, local, view, n_view)
        ext = _slice_cam(extrinsics, local, view, n_view)
        prot = _slice_cam(post_rots, local, view, n_view)
        ptra = np.asarray(_slice_cam(post_trans, local, view, n_view)).reshape(-1)
        t_cav2ego = (
            pairwise[local]
            if pairwise is not None
            else np.eye(4, dtype=np.float64)
        )
        if boxes.shape[0] > 0:
            t_ego2cav = np.linalg.inv(t_cav2ego)
            xyz_h = np.concatenate(
                [corners_ego.reshape(-1, 3), ones[: corners_ego.shape[0] * 8]],
                axis=1,
            ).T
            xyz_cav = (t_ego2cav @ xyz_h).T[:, :3].reshape(boxes.shape[0], 8, 3)
        else:
            xyz_cav = np.zeros((0, 8, 3), dtype=np.float64)

        fg = p_fg[flat] >= float(fg_tau)
        ys_all, xs_all = np.where(fg)
        n_seed = int(ys_all.size)
        ys, xs = subsample_seeds(ys_all, xs_all, max_lift_seeds, rng)
        seed_u = u_map[ys, xs] if ys.size else np.zeros((0,), dtype=np.float64)
        seed_v = v_map[ys, xs] if ys.size else np.zeros((0,), dtype=np.float64)
        z_cell = z_mean[flat, ys, xs] if ys.size else np.zeros((0,), dtype=np.float64)
        seed_uv = (
            np.stack([seed_u, seed_v], axis=1)
            if ys.size
            else np.zeros((0, 2), dtype=np.float64)
        )

        view_polys: Dict[str, List[Optional[np.ndarray]]] = {}
        for convention in CONVENTIONS:
            mask, n_valid, polys = project_view(
                xyz_cav, k, ext, prot, ptra, image_hw, convention
            )
            view_polys[convention] = polys
            stats = per_conv[convention]
            stats["n_valid_proj"] += int(n_valid)
            visible = [p is not None for p in polys]
            stats["n_visible_gt"] += int(sum(visible))
            # Downsample GT mask to R90 for IoU with p_fg.
            gt_r90 = np.zeros((FEAT_H, FEAT_W), dtype=bool)
            if mask.any():
                # 4x4 block occupancy, same as heatmap target.
                h, w = mask.shape
                gt_r90 = mask.reshape(FEAT_H, 4, FEAT_W, 4).any(axis=(1, 3))
            iou = iou_binary(gt_r90, fg)
            if iou == iou:
                stats["iou_sum"] += iou
                stats["iou_n"] += 1
            stats["n_seeds"] += n_seed
            if seed_uv.shape[0]:
                assigned = np.zeros((seed_uv.shape[0],), dtype=bool)
                for poly in polys:
                    if poly is None or len(poly) < 3:
                        continue
                    assigned |= MplPath(np.asarray(poly, dtype=np.float64)).contains_points(
                        seed_uv
                    )
                stats["n_gt_near_2d"] += int(assigned.sum())
            mu_ego = lift_seeds_ego(
                seed_u, seed_v, z_cell, k, prot, ptra, ext, t_cav2ego, convention
            )
            if mu_ego.shape[0] and boxes.shape[0]:
                inside = points_in_boxes(mu_ego, boxes).any(axis=1)
                stats["n_seeds_in_box"] += int(inside.sum())
                stats["xy_d"].extend(nearest_xy_dist(mu_ego, boxes).tolist())

        if agent == "rsu" and n_seed > 20 and not overlay_saved:
            rgb = denormalize_rgb(imgs[flat])
            save_overlay(
                rgb,
                p_fg[flat],
                view_polys["identity"],
                view_polys["invert"],
                out_dir / f"{split}_{scene}_{idx}_{agent}_v{flat}.png",
                f"{split} {scene} idx={idx} {agent} view={flat} seeds={n_seed}",
            )
            overlay_saved = True

    out: Dict[str, Any] = {
        "agent": agent,
        "n_views": n_flat,
        "sam3_fg_frac": round(sam3_fg, 4),
        "pred_fg_frac": round(pred_fg, 4),
        "n_gt": int(boxes.shape[0]),
        "conventions": {},
    }
    for convention in CONVENTIONS:
        stats = per_conv[convention]
        xy = np.asarray(stats["xy_d"], dtype=np.float64)
        iou_mean = (
            stats["iou_sum"] / stats["iou_n"] if stats["iou_n"] else float("nan")
        )
        in_box_n = max(int(min(stats["n_seeds"], max_lift_seeds * n_flat)), 1)
        # n_seeds_in_box is on the subsampled set; report vs subsampled count.
        n_lifted = min(int(stats["n_seeds"]), int(max_lift_seeds) * n_flat)
        out["conventions"][convention] = {
            "n_valid_proj": int(stats["n_valid_proj"]),
            "n_visible_gt_view_sum": int(stats["n_visible_gt"]),
            "mean_iou_r90": None if iou_mean != iou_mean else round(float(iou_mean), 4),
            "n_seeds": int(stats["n_seeds"]),
            "n_gt_near_2d_subsample": int(stats["n_gt_near_2d"]),
            "n_lifted": int(xy.size),
            "frac_in_gt_box": (
                None
                if xy.size == 0
                else round(float(stats["n_seeds_in_box"]) / float(xy.size), 4)
            ),
            "median_xy_to_nearest_gt_m": (
                None if xy.size == 0 else round(float(np.median(xy)), 3)
            ),
            "p90_xy_to_nearest_gt_m": (
                None if xy.size == 0 else round(float(np.percentile(xy, 90)), 3)
            ),
        }
        del in_box_n, n_lifted
    return out


def main() -> None:
    """Load concat128 E16 and probe invert vs identity on listed frames."""
    opt = parse_args()
    out_dir = Path(opt.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = train_utils.create_model(hypes)
    load_epoch_checkpoint(model, opt.model_dir, opt.epoch)
    model.to(device)
    model.eval()
    _unwrap_model(model)
    u_map, v_map = r90_pixel_centers()
    rng = np.random.RandomState(int(opt.seed))
    datasets: Dict[str, Any] = {}
    report: List[Dict[str, Any]] = []

    with torch.no_grad():
        for split, idx in parse_indices(opt.indices):
            datasets[split] = datasets.get(split) or load_split_dataset(hypes, split)
            sample = datasets[split][int(idx)]
            batch = datasets[split].collate_batch_test([sample])
            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            pred = model(ego)
            present = present_camera_agents(ego)
            boxes, _ids = gt_boxes_from_ego(ego)
            meta = ego.get("metadata_path_list", [""])[0]
            scene = scene_from_path(meta)
            print(
                f"{split} idx={idx} scene={scene} agents={present} n_gt={boxes.shape[0]}",
                flush=True,
            )
            frame: Dict[str, Any] = {
                "split": split,
                "idx": idx,
                "scene": scene,
                "n_gt": int(boxes.shape[0]),
                "agents": {},
            }
            for agent in AGENT_ORDER:
                if agent not in present or agent not in pred:
                    continue
                stats = agent_probe(
                    ego,
                    pred,
                    agent,
                    boxes,
                    float(opt.fg_tau),
                    int(opt.max_lift_seeds),
                    rng,
                    u_map,
                    v_map,
                    out_dir,
                    split,
                    idx,
                    scene,
                )
                frame["agents"][agent] = stats
                idn = stats["conventions"]["identity"]
                inv = stats["conventions"]["invert"]
                print(
                    f"  {agent} sam3_fg={stats['sam3_fg_frac']} pred_fg={stats['pred_fg_frac']}"
                    f"  identity IoU={idn['mean_iou_r90']} in_box={idn['frac_in_gt_box']}"
                    f" medXY={idn['median_xy_to_nearest_gt_m']}"
                    f"  invert IoU={inv['mean_iou_r90']} in_box={inv['frac_in_gt_box']}"
                    f" medXY={inv['median_xy_to_nearest_gt_m']}",
                    flush=True,
                )
            report.append(frame)

    out_json = out_dir / "probe.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()

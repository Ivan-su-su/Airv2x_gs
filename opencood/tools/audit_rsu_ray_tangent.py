# -*- coding: utf-8 -*-
"""RSU A-seed depth calibration and ray vs tangent scale sweeps.

Diagnostic only. Uses identity cam2lidar (no RSU invert).

For each RSU FG seed inside a projected GT polygon (A):

* |e_z|, sigma_z, |e_z|/sigma_z  vs camera-depth GT and vs ray-box midpoint
* T_z^GT = z_exit - z_enter along the seed ray through the OBB
* 4 sigma_z  (2σ full width along optical z) and 4 sigma_z / T_z^GT

Sweeps (coverage C = 2σ box-sample fraction, A+B objects):

* fix sigma0_tangent=1, k_ray in {0.5,1,2,3,4}
* fix k_ray=1, k_tangent in {0.5,1,1.5,2}

Sigma_ego = R (k_ray^2 Sigma_depth + k_tan^2 Sigma_tangent(σ0=1) + eps I) R^T
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    PRIMARY_OBJECTNESS_THRESHOLD,
)
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.models.gaussian_modules_0822.lss.target import (
    depth_valid_mask,
    extract_camera_z_gt,
)
from opencood.models.gaussian_modules_0822.p1_layout import BLOCK
from opencood.tools import train_utils
from opencood.tools.audit_gaussian_scale_coverage import (
    _as_numpy,
    _record_len,
    _slice_cam,
    flatten_imgs,
    gt_boxes_from_ego,
    n_views_of,
    pairwise_for_agent,
    projected_polygons,
)
from opencood.tools.eval_gaussian_p1 import load_epoch_checkpoint
from opencood.tools.eval_heatmap_ft_abc import obb_surface_distance, weather_tag
from opencood.tools.gaussian_scale_audit.coverage import (
    box_frame_axes,
    coverage_fraction,
    sample_box_points,
)
from opencood.tools.gaussian_scale_audit.geometry import (
    _as_44,
    cam_to_ego_rt,
    invert_spd,
    pixel_jacobian,
    r90_pixel_centers,
    ray_dir_cam,
    transform_points,
)
from opencood.tools.gaussian_scale_audit.orientation import local_orientation
from opencood.tools.train_gaussian_p1 import _unwrap_model
from opencood.tools.vis_test_heatmap_recall import sample_plan, scene_from_path

K_RAY = (0.5, 1.0, 2.0, 3.0, 4.0)
K_TAN = (0.5, 1.0, 1.5, 2.0)
NEAR_M = 2.0
COVER_K = 2.0
COVER_THRESH = 0.5


def parse_args() -> argparse.Namespace:
    """CLI for the RSU ray/tangent diagnostic."""
    parser = argparse.ArgumentParser(description="RSU A-seed ray vs tangent audit")
    parser.add_argument("-y", "--hypes_yaml", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--frames_per_scene", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fg_tau", type=float, default=PRIMARY_OBJECTNESS_THRESHOLD)
    parser.add_argument("--near_m", type=float, default=NEAR_M)
    parser.add_argument("--box_res", type=int, default=5)
    parser.add_argument("--anisotropy_max", type=float, default=4.0)
    parser.add_argument("--orient_window", type=int, default=7)
    parser.add_argument("--eps", type=float, default=1.0e-4)
    parser.add_argument("--max_seeds_per_object", type=int, default=800)
    parser.add_argument(
        "--out_root",
        default="/home/dell/suyi/visualization/rsu_ray_tangent_audit",
    )
    return parser.parse_args()


def summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Mean / median / p90 of finite values."""
    arr = np.asarray([v for v in values if v == v and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
    }


def ray_obb_optical_z(
    q: np.ndarray,
    cam2ego: np.ndarray,
    box: np.ndarray,
    z_front: float = 0.1,
) -> Optional[Tuple[float, float]]:
    """Optical-z interval where ray ``X_cam = z q`` hits the ego OBB."""
    center, rot_box, half = box_frame_axes(box)
    mat = _as_44(cam2ego)
    origin = mat[:3, 3]
    direction = mat[:3, :3] @ np.asarray(q, dtype=np.float64).reshape(3)
    o_local = rot_box.T @ (origin - center)
    d_local = rot_box.T @ direction
    z_enter = -1.0e9
    z_exit = 1.0e9
    for axis in range(3):
        if abs(d_local[axis]) < 1.0e-12:
            if abs(o_local[axis]) > half[axis] + 1.0e-6:
                return None
            continue
        inv_d = 1.0 / d_local[axis]
        a = (-half[axis] - o_local[axis]) * inv_d
        b = (half[axis] - o_local[axis]) * inv_d
        lo, hi = (a, b) if a <= b else (b, a)
        z_enter = max(z_enter, lo)
        z_exit = min(z_exit, hi)
        if z_enter > z_exit:
            return None
    if z_exit < z_front:
        return None
    z_enter = max(z_enter, z_front)
    if z_enter > z_exit:
        return None
    return float(z_enter), float(z_exit)


def split_rsu_gaussians(
    seed_u: np.ndarray,
    seed_v: np.ndarray,
    z_mean: np.ndarray,
    z_var: np.ndarray,
    theta: np.ndarray,
    anisotropy: np.ndarray,
    intrinsic: np.ndarray,
    post_rot: np.ndarray,
    post_trans: np.ndarray,
    extrinsic: np.ndarray,
    t_cav2ego: np.ndarray,
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``mu_ego, sigma_depth_ego, sigma_tan_ego(σ0=1), cam2ego``."""
    n_seed = int(np.asarray(seed_u).reshape(-1).size)
    q = np.asarray(q, dtype=np.float64).reshape(n_seed, 3)
    z_mean = np.asarray(z_mean, dtype=np.float64).reshape(n_seed)
    z_var = np.asarray(z_var, dtype=np.float64).reshape(n_seed)
    mu_cam = z_mean[:, None] * q
    sigma_depth = z_var[:, None, None] * q[:, :, None] * q[:, None, :]
    cam2ego, rot_c2e = cam_to_ego_rt(extrinsic, t_cav2ego, "rsu")
    mu_ego = transform_points(mu_cam, cam2ego)
    jac0 = pixel_jacobian(1.0, intrinsic, post_rot, post_trans)
    jac = z_mean[:, None, None] * jac0[None, :, :]
    theta = np.asarray(theta, dtype=np.float64).reshape(n_seed)
    aniso = np.asarray(anisotropy, dtype=np.float64).reshape(n_seed)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot2 = np.stack(
        [
            np.stack([cos_t, -sin_t], axis=1),
            np.stack([sin_t, cos_t], axis=1),
        ],
        axis=1,
    )
    s_par = np.sqrt(np.clip(aniso, 1.0e-8, None))
    s_perp = 1.0 / np.sqrt(np.clip(aniso, 1.0e-8, None))
    scale = np.zeros((n_seed, 2, 2), dtype=np.float64)
    scale[:, 0, 0] = s_par ** 2
    scale[:, 1, 1] = s_perp ** 2
    sigma_cells = rot2 @ scale @ np.transpose(rot2, (0, 2, 1))
    sigma_px = (float(BLOCK) ** 2) * sigma_cells
    sigma_t = jac @ sigma_px @ np.transpose(jac, (0, 2, 1))
    sigma_d_ego = rot_c2e[None] @ sigma_depth @ rot_c2e.T[None]
    sigma_t_ego = rot_c2e[None] @ sigma_t @ rot_c2e.T[None]
    return mu_ego, sigma_d_ego, sigma_t_ego, cam2ego


def assemble_sigma(
    sigma_d: np.ndarray,
    sigma_t: np.ndarray,
    k_ray: float,
    k_tan: float,
    eps: float,
) -> np.ndarray:
    """``k_ray^2 Sigma_depth + k_tan^2 Sigma_tangent + eps I``."""
    eye = np.eye(3, dtype=np.float64)[None]
    return (
        float(k_ray) ** 2 * sigma_d
        + float(k_tan) ** 2 * sigma_t
        + float(eps) * eye
    )


def subsample(
    n: int, max_n: int, rng: np.random.RandomState
) -> np.ndarray:
    """Indices ``[0, n)`` capped at ``max_n``."""
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=int(max_n), replace=False)


def save_hist(
    path: Path,
    series: Mapping[str, np.ndarray],
    title: str,
    xlabel: str,
    bins: int = 40,
) -> None:
    """Overlaid histograms for A vs A+B seed groups."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, values in series.items():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        ax.hist(finite, bins=bins, density=True, alpha=0.45, label=f"{name} n={finite.size}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_scatter(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """T_z^GT vs 4 sigma_z scatter."""
    finite = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(x[finite], y[finite], s=8, alpha=0.35, linewidths=0)
    lo = 0.0
    hi = float(np.nanpercentile(np.concatenate([x[finite], y[finite]]), 98)) if finite.any() else 1.0
    ax.plot([lo, hi], [lo, hi], color="0.5", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_sweep_plot(
    path: Path,
    xs: Sequence[float],
    mean_c: Sequence[float],
    frac_c: Sequence[float],
    title: str,
    xlabel: str,
) -> None:
    """Mean 2σ coverage and P(C) vs a scale multiplier."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(xs, mean_c, marker="o", label="mean coverage@2σ")
    ax.plot(xs, frac_c, marker="s", label="P(C) coverage≥0.5")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    """Run RSU A-seed calibration and ray/tangent coverage sweeps."""
    opt = parse_args()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    hypes["validate_dir"] = hypes["test_dir"]
    hypes["train"] = False
    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print("Building test dataset...", flush=True)
    dataset = build_dataset(hypes, visualize=False, train=False)
    plan = sample_plan(dataset, opt.frames_per_scene, opt.seed)
    model = train_utils.create_model(hypes)
    load_epoch_checkpoint(model, opt.model_dir, opt.epoch)
    model.to(device)
    model.eval()
    core = _unwrap_model(model)
    z_bins = core.depth_moments["rsu"].z_bins.detach().cpu().numpy().astype(np.float64)
    rsu_grid = (
        hypes.get("rsu", {})
        .get("cam", {})
        .get("grid_conf")
        or hypes["fusion"]["args"]["rsu_grid_conf"]
    )
    d_min = float(rsu_grid["ddiscr"][0])
    d_max = float(rsu_grid["ddiscr"][1])
    del z_bins
    u_map, v_map = r90_pixel_centers()
    rng = np.random.RandomState(int(opt.seed))
    out_root = Path(opt.out_root)
    (out_root / "plots").mkdir(parents=True, exist_ok=True)

    seed_rows: List[Dict[str, Any]] = []
    object_pool: List[Dict[str, Any]] = []

    with torch.no_grad():
        for sample_i, (scene, idx) in enumerate(plan):
            sample = dataset[int(idx)]
            batch = dataset.collate_batch_test([sample])
            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            meta = ego.get("metadata_path_list", [""])[0]
            scene = scene_from_path(meta) if meta else scene
            if "rsu" not in present_camera_agents(ego):
                print(f"[{sample_i + 1}/{len(plan)}] {scene} idx={idx} no rsu", flush=True)
                continue
            pred = model(ego)
            if "rsu" not in pred:
                continue
            boxes, class_ids = gt_boxes_from_ego(ego)
            cam = ego["rsu"]["batch_merged_cam_inputs"]
            imgs = flatten_imgs(cam["imgs"])
            n_cav = _record_len(ego, "rsu")
            n_view = n_views_of(imgs, n_cav, "rsu")
            n_flat = int(imgs.shape[0])
            p_fg = torch.softmax(pred["rsu"]["heatmap_logits"], dim=1)[:, 1]
            pred_fg = p_fg.ge(float(opt.fg_tau)).detach().cpu().numpy()
            p_fg_np = p_fg.detach().cpu().numpy()
            z_mean_map = pred["rsu"]["depth_z_mean"].detach().cpu().numpy()
            z_var_map = pred["rsu"]["depth_z_var"].detach().cpu().numpy()
            z_gt_map = extract_camera_z_gt(imgs).detach().cpu().numpy()
            valid_gt = depth_valid_mask(
                torch.from_numpy(z_gt_map), d_min, d_max
            ).numpy()
            image_hw = (int(imgs.shape[-2]), int(imgs.shape[-1]))
            polygons = projected_polygons(ego, "rsu", boxes, image_hw)
            pairwise = pairwise_for_agent(ego, "rsu")
            intrins = _as_numpy(cam["intrinsics"])
            extrinsics = _as_numpy(cam["extrinsics"])
            post_rots = _as_numpy(cam["post_rots"])
            post_trans = _as_numpy(cam["post_trans"])
            n_box = int(boxes.shape[0])
            print(
                f"[{sample_i + 1}/{len(plan)}] {scene} idx={idx} "
                f"weather={weather_tag(scene)} n_gt={n_box} views={n_flat}",
                flush=True,
            )
            box_mu: List[List[np.ndarray]] = [[] for _ in range(n_box)]
            box_sig_d: List[List[np.ndarray]] = [[] for _ in range(n_box)]
            box_sig_t: List[List[np.ndarray]] = [[] for _ in range(n_box)]
            visible = np.zeros((n_box,), dtype=bool)

            for flat in range(n_flat):
                local = flat // n_view
                view = flat % n_view
                ys, xs = np.where(pred_fg[flat])
                n_seed = int(ys.size)
                if n_seed == 0:
                    for box_i, poly in enumerate(polygons[flat]):
                        if poly is not None and len(poly) >= 3:
                            visible[box_i] = True
                    continue
                seed_u = u_map[ys, xs]
                seed_v = v_map[ys, xs]
                seed_uv = np.stack([seed_u, seed_v], axis=1)
                theta, aniso, _l1, _l2 = local_orientation(
                    p_fg_np[flat],
                    ys,
                    xs,
                    window=int(opt.orient_window),
                    anisotropy_max=float(opt.anisotropy_max),
                )
                k = _slice_cam(intrins, local, view, n_view)
                ext = _slice_cam(extrinsics, local, view, n_view)
                prot = _slice_cam(post_rots, local, view, n_view)
                ptra = np.asarray(_slice_cam(post_trans, local, view, n_view)).reshape(-1)
                t_cav2ego = (
                    pairwise[local] if pairwise is not None else np.eye(4, dtype=np.float64)
                )
                q_all = ray_dir_cam(seed_u, seed_v, k, prot, ptra).reshape(n_seed, 3)
                z_mean = z_mean_map[flat, ys, xs]
                z_var = np.clip(z_var_map[flat, ys, xs], 0.0, None)
                sigma_z = np.sqrt(z_var)
                z_gt = z_gt_map[flat, ys, xs]
                valid = valid_gt[flat, ys, xs]
                mu_all, sig_d_all, sig_t_all, cam2ego = split_rsu_gaussians(
                    seed_u,
                    seed_v,
                    z_mean,
                    z_var,
                    theta,
                    aniso,
                    k,
                    prot,
                    ptra,
                    ext,
                    t_cav2ego,
                    q_all,
                )
                for box_i in range(n_box):
                    poly = polygons[flat][box_i]
                    if poly is None or len(poly) < 3:
                        continue
                    visible[box_i] = True
                    inside = MplPath(np.asarray(poly, dtype=np.float64)).contains_points(
                        seed_uv
                    )
                    hit = np.flatnonzero(inside)
                    if hit.size == 0:
                        continue
                    box_mu[box_i].append(mu_all[hit])
                    box_sig_d[box_i].append(sig_d_all[hit])
                    box_sig_t[box_i].append(sig_t_all[hit])
                    for j in hit.tolist():
                        span = ray_obb_optical_z(q_all[j], cam2ego, boxes[box_i])
                        if span is None:
                            t_gt = float("nan")
                            z_mid = float("nan")
                        else:
                            t_gt = float(span[1] - span[0])
                            z_mid = 0.5 * (span[0] + span[1])
                        e_depth = (
                            abs(float(z_mean[j] - z_gt[j]))
                            if bool(valid[j])
                            else float("nan")
                        )
                        e_box = (
                            abs(float(z_mean[j] - z_mid))
                            if z_mid == z_mid
                            else float("nan")
                        )
                        sz = float(sigma_z[j])
                        seed_rows.append(
                            {
                                "scene": scene,
                                "weather": weather_tag(scene),
                                "idx": int(idx),
                                "box_id": int(box_i),
                                "class_id": int(class_ids[box_i])
                                if box_i < class_ids.size
                                else 1,
                                "z_pred": round(float(z_mean[j]), 4),
                                "sigma_z": round(sz, 4),
                                "z_gt_depth": None
                                if not bool(valid[j])
                                else round(float(z_gt[j]), 4),
                                "abs_ez_depth": None
                                if e_depth != e_depth
                                else round(e_depth, 4),
                                "abs_ez_over_sigma_depth": None
                                if e_depth != e_depth or sz <= 1.0e-8
                                else round(e_depth / sz, 4),
                                "T_z_GT": None if t_gt != t_gt else round(t_gt, 4),
                                "z_gt_box_mid": None if z_mid != z_mid else round(z_mid, 4),
                                "abs_ez_box": None if e_box != e_box else round(e_box, 4),
                                "abs_ez_over_sigma_box": None
                                if e_box != e_box or sz <= 1.0e-8
                                else round(e_box / sz, 4),
                                "four_sigma_z": round(4.0 * sz, 4),
                                "four_sigma_over_T": None
                                if t_gt != t_gt or t_gt <= 1.0e-6
                                else round(4.0 * sz / t_gt, 4),
                            }
                        )

            for box_i in range(n_box):
                if not visible[box_i] or not box_mu[box_i]:
                    continue
                mu = np.concatenate(box_mu[box_i], axis=0)
                sig_d = np.concatenate(box_sig_d[box_i], axis=0)
                sig_t = np.concatenate(box_sig_t[box_i], axis=0)
                dist = obb_surface_distance(mu, boxes[box_i])
                is_b = bool(np.min(dist) <= float(opt.near_m))
                pick = subsample(int(mu.shape[0]), int(opt.max_seeds_per_object), rng)
                object_pool.append(
                    {
                        "scene": scene,
                        "idx": int(idx),
                        "box_id": int(box_i),
                        "is_ab": is_b,
                        "n_seed_a": int(mu.shape[0]),
                        "min_surface_m": float(np.min(dist)),
                        "mu": mu[pick],
                        "sig_d": sig_d[pick],
                        "sig_t": sig_t[pick],
                        "box": boxes[box_i],
                    }
                )
                n_a_this = int(mu.shape[0])
                # Mark last n_a_this seed rows of this box as AB. Cheaper: fill later.
                del n_a_this

    # Attach is_ab onto seed rows.
    ab_keys = {(r["scene"], r["idx"], r["box_id"]) for r in object_pool if r["is_ab"]}
    a_keys = {(r["scene"], r["idx"], r["box_id"]) for r in object_pool}
    for row in seed_rows:
        key = (row["scene"], row["idx"], row["box_id"])
        row["is_a"] = key in a_keys
        row["is_ab"] = key in ab_keys

    def seed_arr(field: str, ab_only: bool) -> np.ndarray:
        vals = []
        for row in seed_rows:
            if ab_only and not row["is_ab"]:
                continue
            val = row.get(field)
            if val is None:
                continue
            vals.append(float(val))
        return np.asarray(vals, dtype=np.float64)

    cal_a = {
        "abs_ez_depth": summarize(seed_arr("abs_ez_depth", False)),
        "sigma_z": summarize(seed_arr("sigma_z", False)),
        "abs_ez_over_sigma_depth": summarize(seed_arr("abs_ez_over_sigma_depth", False)),
        "T_z_GT": summarize(seed_arr("T_z_GT", False)),
        "four_sigma_z": summarize(seed_arr("four_sigma_z", False)),
        "four_sigma_over_T": summarize(seed_arr("four_sigma_over_T", False)),
        "abs_ez_box": summarize(seed_arr("abs_ez_box", False)),
        "abs_ez_over_sigma_box": summarize(seed_arr("abs_ez_over_sigma_box", False)),
    }
    cal_ab = {
        "abs_ez_depth": summarize(seed_arr("abs_ez_depth", True)),
        "sigma_z": summarize(seed_arr("sigma_z", True)),
        "abs_ez_over_sigma_depth": summarize(seed_arr("abs_ez_over_sigma_depth", True)),
        "T_z_GT": summarize(seed_arr("T_z_GT", True)),
        "four_sigma_z": summarize(seed_arr("four_sigma_z", True)),
        "four_sigma_over_T": summarize(seed_arr("four_sigma_over_T", True)),
        "abs_ez_box": summarize(seed_arr("abs_ez_box", True)),
        "abs_ez_over_sigma_box": summarize(seed_arr("abs_ez_over_sigma_box", True)),
    }

    ab_objects = [obj for obj in object_pool if obj["is_ab"]]
    a_objects = object_pool
    print(
        f"A objects with seeds={len(a_objects)}  A+B objects={len(ab_objects)}  "
        f"A seeds={len(seed_rows)}  A+B seeds={sum(1 for r in seed_rows if r['is_ab'])}",
        flush=True,
    )

    def coverage_sweep(
        objects: Sequence[Mapping[str, Any]],
        k_rays: Sequence[float],
        k_tans: Sequence[float],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for k_ray in k_rays:
            for k_tan in k_tans:
                covs: List[float] = []
                n_pass = 0
                for obj in objects:
                    sigma = assemble_sigma(
                        obj["sig_d"], obj["sig_t"], k_ray, k_tan, float(opt.eps)
                    )
                    pts = sample_box_points(obj["box"], int(opt.box_res))
                    cov = coverage_fraction(pts, obj["mu"], invert_spd(sigma), k=COVER_K)
                    covs.append(float(cov))
                    if cov >= COVER_THRESH:
                        n_pass += 1
                rows.append(
                    {
                        "k_ray": float(k_ray),
                        "k_tangent": float(k_tan),
                        "n_object": len(objects),
                        "mean_cov@2s": None
                        if not covs
                        else round(float(np.mean(covs)), 4),
                        "median_cov@2s": None
                        if not covs
                        else round(float(np.median(covs)), 4),
                        "P(C)": None if not objects else round(n_pass / len(objects), 4),
                    }
                )
                print(
                    f"  sweep k_ray={k_ray:g} k_tan={k_tan:g}  "
                    f"meanC={rows[-1]['mean_cov@2s']} P(C)={rows[-1]['P(C)']}",
                    flush=True,
                )
        return rows

    print("sweep k_ray (k_tangent=1, A+B)", flush=True)
    sweep_ray_ab = coverage_sweep(ab_objects, K_RAY, (1.0,))
    print("sweep k_tangent (k_ray=1, A+B)", flush=True)
    sweep_tan_ab = coverage_sweep(ab_objects, (1.0,), K_TAN)
    print("sweep k_ray (k_tangent=1, all A objects)", flush=True)
    sweep_ray_a = coverage_sweep(a_objects, K_RAY, (1.0,))
    print("sweep k_tangent (k_ray=1, all A objects)", flush=True)
    sweep_tan_a = coverage_sweep(a_objects, (1.0,), K_TAN)

    plots = out_root / "plots"
    save_hist(
        plots / "abs_ez_depth.png",
        {"A seeds": seed_arr("abs_ez_depth", False), "A+B seeds": seed_arr("abs_ez_depth", True)},
        "RSU |e_z| vs camera-depth GT",
        "|e_z| (m)",
    )
    save_hist(
        plots / "sigma_z.png",
        {"A seeds": seed_arr("sigma_z", False), "A+B seeds": seed_arr("sigma_z", True)},
        "RSU categorical σ_z",
        "σ_z (m)",
    )
    save_hist(
        plots / "abs_ez_over_sigma.png",
        {
            "A seeds": seed_arr("abs_ez_over_sigma_depth", False),
            "A+B seeds": seed_arr("abs_ez_over_sigma_depth", True),
        },
        "RSU |e_z| / σ_z (camera-depth GT)",
        "|e_z| / σ_z",
    )
    save_hist(
        plots / "T_z_GT.png",
        {"A seeds": seed_arr("T_z_GT", False), "A+B seeds": seed_arr("T_z_GT", True)},
        "RSU GT ray thickness T_z^GT = z_exit − z_enter",
        "T_z^GT (m)",
    )
    save_hist(
        plots / "four_sigma_over_T.png",
        {
            "A seeds": seed_arr("four_sigma_over_T", False),
            "A+B seeds": seed_arr("four_sigma_over_T", True),
        },
        "RSU 4σ_z / T_z^GT  (k_ray=1)",
        "4σ_z / T_z^GT",
    )
    save_scatter(
        plots / "four_sigma_vs_T.png",
        seed_arr("T_z_GT", True),
        seed_arr("four_sigma_z", True),
        "A+B seeds: 4σ_z vs T_z^GT",
        "T_z^GT (m)",
        "4σ_z (m)",
    )
    save_sweep_plot(
        plots / "sweep_k_ray_ab.png",
        [r["k_ray"] for r in sweep_ray_ab],
        [r["mean_cov@2s"] or 0.0 for r in sweep_ray_ab],
        [r["P(C)"] or 0.0 for r in sweep_ray_ab],
        "A+B objects: fix σ0_tangent=1, sweep k_ray",
        "k_ray",
    )
    save_sweep_plot(
        plots / "sweep_k_tangent_ab.png",
        [r["k_tangent"] for r in sweep_tan_ab],
        [r["mean_cov@2s"] or 0.0 for r in sweep_tan_ab],
        [r["P(C)"] or 0.0 for r in sweep_tan_ab],
        "A+B objects: fix k_ray=1, sweep k_tangent",
        "k_tangent",
    )

    report = {
        "checkpoint": f"net_epoch{opt.epoch}.pth",
        "agent": "rsu",
        "rsu_extrinsic": "identity cam2lidar",
        "fg_tau": float(opt.fg_tau),
        "near_m": float(opt.near_m),
        "sigma0_tangent": 1.0,
        "k_ray": list(K_RAY),
        "k_tangent": list(K_TAN),
        "n_a_seeds": len(seed_rows),
        "n_ab_seeds": int(sum(1 for r in seed_rows if r["is_ab"])),
        "n_a_objects": len(a_objects),
        "n_ab_objects": len(ab_objects),
        "calibration_A": cal_a,
        "calibration_AB": cal_ab,
        "sweep_k_ray_AB": sweep_ray_ab,
        "sweep_k_tangent_AB": sweep_tan_ab,
        "sweep_k_ray_A": sweep_ray_a,
        "sweep_k_tangent_A": sweep_tan_a,
        "note": (
            "e_z vs camera-depth GT is depth-head calibration. "
            "T_z^GT is ray-OBB optical-z span. 4 sigma_z is 2σ full width along z."
        ),
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if seed_rows:
        with (out_root / "seeds.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(seed_rows)

    def fmt(block: Dict[str, Any], key: str) -> str:
        item = block[key]
        if item["n"] == 0:
            return "n/a"
        return (
            f"n={item['n']}  mean={item['mean']}  "
            f"med={item['median']}  p90={item['p90']}"
        )

    lines = [
        f"RSU ray/tangent audit  epoch={opt.epoch}  A seeds={len(seed_rows)}  "
        f"A+B seeds={report['n_ab_seeds']}  A+B objects={len(ab_objects)}",
        "",
        "calibration  A seeds",
        f"  |e_z| depth     {fmt(cal_a, 'abs_ez_depth')}",
        f"  sigma_z         {fmt(cal_a, 'sigma_z')}",
        f"  |e_z|/sigma     {fmt(cal_a, 'abs_ez_over_sigma_depth')}",
        f"  T_z^GT          {fmt(cal_a, 'T_z_GT')}",
        f"  4 sigma_z       {fmt(cal_a, 'four_sigma_z')}",
        f"  4s / T          {fmt(cal_a, 'four_sigma_over_T')}",
        "",
        "calibration  A+B seeds",
        f"  |e_z| depth     {fmt(cal_ab, 'abs_ez_depth')}",
        f"  sigma_z         {fmt(cal_ab, 'sigma_z')}",
        f"  |e_z|/sigma     {fmt(cal_ab, 'abs_ez_over_sigma_depth')}",
        f"  T_z^GT          {fmt(cal_ab, 'T_z_GT')}",
        f"  4 sigma_z       {fmt(cal_ab, 'four_sigma_z')}",
        f"  4s / T          {fmt(cal_ab, 'four_sigma_over_T')}",
        "",
        "A+B  k_ray sweep (k_tangent=1)",
    ]
    for row in sweep_ray_ab:
        lines.append(
            f"  k_ray={row['k_ray']:g}  meanC={row['mean_cov@2s']}  P(C)={row['P(C)']}"
        )
    lines.append("A+B  k_tangent sweep (k_ray=1)")
    for row in sweep_tan_ab:
        lines.append(
            f"  k_tan={row['k_tangent']:g}  meanC={row['mean_cov@2s']}  P(C)={row['P(C)']}"
        )
    text = "\n".join(lines) + "\n"
    (out_root / "report.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"wrote {out_root}", flush=True)


if __name__ == "__main__":
    main()

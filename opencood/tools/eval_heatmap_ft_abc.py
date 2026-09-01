# -*- coding: utf-8 -*-
"""Heatmap-ft test vis + per-GT A/B/C after fixing RSU cam2lidar.

A: projected GT image area contains an FG seed (p_fg >= tau).
B: those 2D-correct seeds lift near the official 3D box (surface dist <= near_m).
C: 2σ support of those Gaussians covers >= cover_thresh of the sampled box.

Diagnostic only. Uses identity extrinsics for all agents (no RSU invert).
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
from opencood.models.gaussian_modules_0822.heatmap.box_support import (
    rasterize_convex_polygon,
)
from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    PRIMARY_OBJECTNESS_THRESHOLD,
)
from opencood.models.gaussian_modules_0822.heatmap.target import binary_objectness_target
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.tools import train_utils
from opencood.tools.audit_gaussian_scale_coverage import (
    _as_numpy,
    _record_len,
    _slice_cam,
    flatten_imgs,
    gt_boxes_from_ego,
    pairwise_for_agent,
    projected_polygons,
    n_views_of,
)
from opencood.tools.eval_gaussian_p1 import denormalize_rgb, load_epoch_checkpoint
from opencood.tools.gaussian_scale_audit.coverage import (
    box_frame_axes,
    coverage_fraction,
    sample_box_points,
)
from opencood.tools.gaussian_scale_audit.geometry import (
    invert_spd,
    r90_pixel_centers,
    view_gaussians,
)
from opencood.tools.gaussian_scale_audit.orientation import local_orientation
from opencood.tools.train_gaussian_p1 import _unwrap_model
from opencood.tools.vis_test_heatmap_recall import (
    overlay_fg,
    sample_plan,
    scene_from_path,
)
from opencood.utils.camera_utils import (
    FOG_BETA_RANGE,
    apply_atmospheric_fog_rgb,
    camera_optical_ray_range,
    imagenet_normalize_display_rgb,
)
from opencood.utils.scenario_utils import scenarios_params

AGENT_ORDER = ("vehicle", "rsu", "drone")
ABC_COLORS = {
    "miss": (220 / 255, 20 / 255, 60 / 255),
    "A": (1.0, 0.84, 0.0),
    "AB": (0.0, 0.75, 0.85),
    "ABC": (0.2, 0.85, 0.2),
}


def parse_args() -> argparse.Namespace:
    """CLI for heatmap-ft ABC evaluation."""
    parser = argparse.ArgumentParser(description="Heatmap-ft vis + per-GT A/B/C")
    parser.add_argument("-y", "--hypes_yaml", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--frames_per_scene", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fg_tau", type=float, default=PRIMARY_OBJECTNESS_THRESHOLD)
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--near_m", type=float, default=2.0)
    parser.add_argument("--cover_thresh", type=float, default=0.5)
    parser.add_argument("--box_res", type=int, default=5)
    parser.add_argument("--anisotropy_max", type=float, default=4.0)
    parser.add_argument("--orient_window", type=int, default=7)
    parser.add_argument("--drone_ray_sigma", type=float, default=2.0)
    parser.add_argument("--eps", type=float, default=1.0e-4)
    parser.add_argument(
        "--out_root",
        default="/home/dell/suyi/visualization/airv2x_gaussian_p1_heatmap_ft_abc",
    )
    parser.add_argument(
        "--extra_fog_on_foggy",
        action="store_true",
        help="also forward the real foggy-scene frames with high-end Koschmieder fog",
    )
    return parser.parse_args()


def weather_tag(scene: str) -> str:
    """Map a scenario timestamp to a single weather/time label."""
    params = scenarios_params.get(scene, {})
    if params.get("foggy"):
        return "foggy"
    if params.get("rainy"):
        return "rainy"
    if params.get("nighttime"):
        return "night"
    if params.get("dusk"):
        return "dusk"
    if params.get("clear"):
        return "clear"
    return "other"


def weather_flags(scene: str) -> Dict[str, bool]:
    """Boolean weather flags from scenario_utils."""
    params = scenarios_params.get(scene, {})
    return {
        "foggy": bool(params.get("foggy", False)),
        "rainy": bool(params.get("rainy", False)),
        "nighttime": bool(params.get("nighttime", False)),
        "dusk": bool(params.get("dusk", False)),
        "clear": bool(params.get("clear", False)),
        "cloudy": bool(params.get("cloudy", False)),
    }


def obb_surface_distance(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Euclidean distance from points to the OBB surface (0 if inside)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((0,), dtype=np.float64)
    center, rot, half = box_frame_axes(box)
    local = (pts - center[None, :]) @ rot
    delta = np.maximum(np.abs(local) - half[None, :], 0.0)
    return np.linalg.norm(delta, axis=1)


def apply_heavy_fog_agent(cam: Dict[str, Any], agent: str) -> float:
    """Koschmieder fog on one agent's collated RGB. Returns beta."""
    beta = float(FOG_BETA_RANGE[agent][1])
    imgs = cam["imgs"]
    if imgs.dim() == 5:
        n_cav, n_view = imgs.shape[:2]
        imgs = imgs.reshape(n_cav * n_view, *imgs.shape[2:])
        cam["imgs"] = imgs
    n_flat = int(imgs.shape[0])

    def _flat(name: str) -> torch.Tensor:
        tensor = cam[name]
        if int(tensor.shape[0]) == n_flat:
            return tensor
        if tensor.dim() >= 2 and int(tensor.shape[0] * tensor.shape[1]) == n_flat:
            return tensor.reshape(n_flat, *tensor.shape[2:])
        return tensor

    intrins = _flat("intrinsics")
    post_rots = _flat("post_rots")
    post_trans = _flat("post_trans")
    for idx in range(n_flat):
        rgb_u8 = denormalize_rgb(imgs[idx])
        rgb = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1)
        depth = imgs[idx, 3].detach().float().cpu()
        k = intrins[idx].detach().float().cpu().reshape(-1, 3, 3)[0]
        prot = post_rots[idx].detach().float().cpu().reshape(-1, 3, 3)[0]
        ptra = post_trans[idx].detach().float().cpu().reshape(-1)[:3]
        rho = camera_optical_ray_range(depth, k, prot, ptra)
        fogged = apply_atmospheric_fog_rgb(rgb, rho, beta)
        imgs[idx, :3] = imagenet_normalize_display_rgb(fogged).to(
            device=imgs.device, dtype=imgs.dtype
        )
    cam["imgs"] = imgs
    return beta


def abc_style(a: bool, b: bool, c: bool) -> str:
    """Box color key from A/B/C."""
    if a and b and c:
        return "ABC"
    if a and b:
        return "AB"
    if a:
        return "A"
    return "miss"


def save_panel(
    path: Path,
    rgb: np.ndarray,
    p_fg: np.ndarray,
    pred_fg: np.ndarray,
    gt_fg: np.ndarray,
    polygons: Sequence[Optional[np.ndarray]],
    styles: Sequence[str],
    title: str,
) -> None:
    """RGB | p_fg | pred overlay | GT boxes colored by A/B/C."""
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.6))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    im = axes[1].imshow(p_fg, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("heatmap p_fg")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].imshow(overlay_fg(rgb, pred_fg))
    axes[2].set_title("pred FG overlay")
    axes[3].imshow(rgb)
    axes[3].set_title("GT A/B/C")
    for poly, style in zip(polygons, styles):
        if poly is None or len(poly) < 3:
            continue
        closed = np.vstack([np.asarray(poly), np.asarray(poly)[0]])
        axes[3].plot(
            closed[:, 0],
            closed[:, 1],
            color=ABC_COLORS[style],
            linewidth=1.4,
        )
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def rate(num: int, den: int) -> Optional[float]:
    """num/den rounded, or None if den==0."""
    if den <= 0:
        return None
    return round(float(num) / float(den), 4)


def pack_pixel(tp: float, fn: float, fp: float) -> Dict[str, Any]:
    """Pixel TP/FN/FP → recall/precision."""
    n_gt = tp + fn
    n_pred = tp + fp
    return {
        "n_gt_pixels": int(n_gt),
        "n_pred_pixels": int(n_pred),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "recall": rate(int(tp), int(n_gt)),
        "precision": rate(int(tp), int(n_pred)),
    }


def main() -> None:
    """Run test vis, pixel heatmap metrics, and per-object A/B/C."""
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
    print(
        f"test n={len(dataset)} scenes={len(dataset.len_record)} "
        f"sampled={len(plan)} seed={opt.seed}",
        flush=True,
    )

    model = train_utils.create_model(hypes)
    load_epoch_checkpoint(model, opt.model_dir, opt.epoch)
    model.to(device)
    model.eval()
    core = _unwrap_model(model)
    z_bins = {
        agent: core.depth_moments[agent].z_bins.detach().cpu().numpy().astype(np.float64)
        for agent in core.depth_moments
    }
    u_map, v_map = r90_pixel_centers()
    sigma0_list = [float(opt.sigma0)]
    out_root = Path(opt.out_root)
    vis_dir = out_root / f"epoch{opt.epoch}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    pixel_tp: Dict[Tuple[str, str], float] = defaultdict(float)
    pixel_fn: Dict[Tuple[str, str], float] = defaultdict(float)
    pixel_fp: Dict[Tuple[str, str], float] = defaultdict(float)
    n_gt_empty: Dict[str, int] = defaultdict(int)
    n_agent_frames: Dict[str, int] = defaultdict(int)
    object_rows: List[Dict[str, Any]] = []
    frame_rows: List[Dict[str, Any]] = []
    jobs: List[Dict[str, Any]] = [
        {"scene": scene, "idx": idx, "synth_fog": False} for scene, idx in plan
    ]
    if opt.extra_fog_on_foggy:
        for scene, idx in plan:
            if weather_tag(scene) == "foggy":
                jobs.append({"scene": scene, "idx": idx, "synth_fog": True})

    with torch.no_grad():
        for sample_i, job in enumerate(jobs):
            scene, idx = job["scene"], int(job["idx"])
            sample = dataset[idx]
            batch = dataset.collate_batch_test([sample])
            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            if job["synth_fog"]:
                for agent in present_camera_agents(ego):
                    cam = ego[agent]["batch_merged_cam_inputs"]
                    apply_heavy_fog_agent(cam, agent)
            meta = ego.get("metadata_path_list", [""])[0]
            scene = scene_from_path(meta) if meta else scene
            weather = weather_tag(scene)
            flags = weather_flags(scene)
            tag = f"{weather}_synthfog" if job["synth_fog"] else weather
            pred_all = model(ego)
            present = present_camera_agents(ego)
            boxes, class_ids = gt_boxes_from_ego(ego)
            print(
                f"[{sample_i + 1}/{len(jobs)}] scene={scene} idx={idx} "
                f"tag={tag} agents={present} n_gt={boxes.shape[0]}",
                flush=True,
            )
            for agent in AGENT_ORDER:
                if agent not in present or agent not in pred_all:
                    continue
                cam = ego[agent]["batch_merged_cam_inputs"]
                if not torch.is_tensor(cam.get("imgs")):
                    continue
                imgs = flatten_imgs(cam["imgs"])
                n_cav = _record_len(ego, agent)
                n_view = n_views_of(imgs, n_cav, agent)
                n_flat = int(imgs.shape[0])
                logits = pred_all[agent]["heatmap_logits"]
                p_fg = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                pred_fg = p_fg >= float(opt.fg_tau)
                image_hw = (int(imgs.shape[-2]), int(imgs.shape[-1]))
                polygons = projected_polygons(ego, agent, boxes, image_hw)
                gt_maps = np.zeros((n_flat, image_hw[0], image_hw[1]), dtype=np.uint8)
                for flat in range(n_flat):
                    for poly in polygons[flat]:
                        if poly is None or len(poly) < 3:
                            continue
                        mask = rasterize_convex_polygon(poly, image_hw[0], image_hw[1])
                        gt_maps[flat][mask] = 1
                gt_r90 = binary_objectness_target(
                    torch.from_numpy(gt_maps).long(), tau=1
                ).numpy()
                gt_fg = gt_r90 > 0
                if tuple(gt_fg.shape) != tuple(pred_fg.shape):
                    print(
                        f"  skip {agent}: gt {gt_fg.shape} vs pred {pred_fg.shape}",
                        flush=True,
                    )
                    continue
                tp = int((pred_fg & gt_fg).sum())
                fn = int((~pred_fg & gt_fg).sum())
                fp = int((pred_fg & ~gt_fg).sum())
                pixel_tp[(agent, tag)] += tp
                pixel_fn[(agent, tag)] += fn
                pixel_fp[(agent, tag)] += fp
                n_agent_frames[agent] += 1
                if int(gt_fg.sum()) == 0:
                    n_gt_empty[agent] += 1

                pairwise = pairwise_for_agent(ego, agent)
                intrins = _as_numpy(cam["intrinsics"])
                extrinsics = _as_numpy(cam["extrinsics"])
                post_rots = _as_numpy(cam["post_rots"])
                post_trans = _as_numpy(cam["post_trans"])
                z_mean = pred_all[agent]["depth_z_mean"].detach().cpu().numpy()
                depth_prob = None
                if agent in ("vehicle", "rsu") and "depth_logits" in pred_all[agent]:
                    depth_prob = (
                        torch.softmax(pred_all[agent]["depth_logits"], dim=1)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                n_box = int(boxes.shape[0])
                visible = np.zeros((n_box,), dtype=bool)
                a_flags = np.zeros((n_box,), dtype=bool)
                b_flags = np.zeros((n_box,), dtype=bool)
                c_flags = np.zeros((n_box,), dtype=bool)
                n_seed_2d = np.zeros((n_box,), dtype=np.int64)
                min_surf = np.full((n_box,), np.nan, dtype=np.float64)
                cov2s = np.full((n_box,), np.nan, dtype=np.float64)
                box_mu: List[List[np.ndarray]] = [[] for _ in range(n_box)]
                box_sig: List[List[np.ndarray]] = [[] for _ in range(n_box)]

                for flat in range(n_flat):
                    local = flat // n_view
                    view = flat % n_view
                    ys, xs = np.where(pred_fg[flat])
                    n_seed = int(ys.size)
                    seed_u = u_map[ys, xs] if n_seed else np.zeros((0,), dtype=np.float64)
                    seed_v = v_map[ys, xs] if n_seed else np.zeros((0,), dtype=np.float64)
                    seed_uv = (
                        np.stack([seed_u, seed_v], axis=1)
                        if n_seed
                        else np.zeros((0, 2), dtype=np.float64)
                    )
                    theta = aniso = None
                    k = _slice_cam(intrins, local, view, n_view)
                    ext = _slice_cam(extrinsics, local, view, n_view)
                    prot = _slice_cam(post_rots, local, view, n_view)
                    ptra = np.asarray(
                        _slice_cam(post_trans, local, view, n_view)
                    ).reshape(-1)
                    t_cav2ego = (
                        pairwise[local]
                        if pairwise is not None
                        else np.eye(4, dtype=np.float64)
                    )
                    for box_i in range(n_box):
                        poly = polygons[flat][box_i]
                        if poly is None or len(poly) < 3:
                            continue
                        visible[box_i] = True
                        if n_seed == 0:
                            continue
                        inside = MplPath(np.asarray(poly, dtype=np.float64)).contains_points(
                            seed_uv
                        )
                        hit = np.flatnonzero(inside)
                        if hit.size == 0:
                            continue
                        a_flags[box_i] = True
                        n_seed_2d[box_i] += int(hit.size)
                        if theta is None:
                            theta, aniso, _l1, _l2 = local_orientation(
                                p_fg[flat],
                                ys,
                                xs,
                                window=int(opt.orient_window),
                                anisotropy_max=float(opt.anisotropy_max),
                            )
                        z_cells = z_mean[flat, ys[hit], xs[hit]]
                        prob_sel = (
                            depth_prob[flat][:, ys[hit], xs[hit]]
                            if depth_prob is not None
                            else None
                        )
                        mu_view, sig_view, _diff = view_gaussians(
                            seed_u[hit],
                            seed_v[hit],
                            z_cells,
                            theta[hit],
                            aniso[hit],
                            k,
                            prot,
                            ptra,
                            ext,
                            t_cav2ego,
                            agent,
                            sigma0_list,
                            z_bins.get(agent),
                            prob_sel,
                            float(opt.drone_ray_sigma),
                            float(opt.eps),
                        )
                        if mu_view.shape[0]:
                            box_mu[box_i].append(mu_view)
                            box_sig[box_i].append(sig_view[float(opt.sigma0)])

                for box_i in range(n_box):
                    if not visible[box_i]:
                        continue
                    if box_mu[box_i]:
                        mu = np.concatenate(box_mu[box_i], axis=0)
                        sig = np.concatenate(box_sig[box_i], axis=0)
                        dist = obb_surface_distance(mu, boxes[box_i])
                        min_surf[box_i] = float(np.min(dist))
                        b_flags[box_i] = bool(np.min(dist) <= float(opt.near_m))
                        pts = sample_box_points(boxes[box_i], int(opt.box_res))
                        prec = invert_spd(sig)
                        cov = coverage_fraction(pts, mu, prec, k=2.0)
                        cov2s[box_i] = float(cov)
                        c_flags[box_i] = bool(cov >= float(opt.cover_thresh))
                    object_rows.append(
                        {
                            "scene": scene,
                            "weather": weather,
                            "synth_fog": bool(job["synth_fog"]),
                            "tag": tag,
                            "idx": idx,
                            "agent": agent,
                            "box_id": int(box_i),
                            "class_id": int(class_ids[box_i]) if box_i < class_ids.size else 1,
                            "visible": True,
                            "A": bool(a_flags[box_i]),
                            "B": bool(b_flags[box_i]),
                            "C": bool(c_flags[box_i]),
                            "n_seed_2d": int(n_seed_2d[box_i]),
                            "min_surface_m": None
                            if min_surf[box_i] != min_surf[box_i]
                            else round(float(min_surf[box_i]), 3),
                            "cov@2s": None
                            if cov2s[box_i] != cov2s[box_i]
                            else round(float(cov2s[box_i]), 4),
                            **flags,
                        }
                    )

                n_vis = int(visible.sum())
                n_a = int((visible & a_flags).sum())
                n_b = int((visible & b_flags).sum())
                n_c = int((visible & c_flags).sum())
                frame_rows.append(
                    {
                        "scene": scene,
                        "weather": weather,
                        "synth_fog": bool(job["synth_fog"]),
                        "tag": tag,
                        "idx": idx,
                        "agent": agent,
                        "n_visible": n_vis,
                        "n_A": n_a,
                        "n_B": n_b,
                        "n_C": n_c,
                        "pixel_recall": rate(tp, tp + fn),
                        "pixel_precision": rate(tp, tp + fp),
                    }
                )

                view_scores = gt_fg.reshape(n_flat, -1).sum(axis=1)
                if float(view_scores.max()) <= 0:
                    view_scores = p_fg.reshape(n_flat, -1).mean(axis=1)
                view = int(np.argmax(view_scores))
                styles = [
                    abc_style(bool(a_flags[i]), bool(b_flags[i]), bool(c_flags[i]))
                    if visible[i]
                    else "miss"
                    for i in range(n_box)
                ]
                rgb = denormalize_rgb(imgs[view])
                out_png = vis_dir / scene / tag / agent / f"{sample_i:02d}_{idx:04d}.png"
                rec_s = rate(tp, tp + fn)
                rec_txt = f"{rec_s:.3f}" if rec_s is not None else "n/a"
                title = (
                    f"ep{opt.epoch} {scene} idx={idx} {agent} {tag} view={view}  "
                    f"pixR={rec_txt}  A={n_a}/{n_vis} B={n_b}/{n_vis} C={n_c}/{n_vis}  "
                    f"tau={opt.fg_tau:g} sigma0={opt.sigma0:g} near={opt.near_m:g}m"
                )
                save_panel(
                    out_png,
                    rgb,
                    p_fg[view],
                    pred_fg[view].astype(np.uint8),
                    gt_fg[view].astype(np.uint8),
                    polygons[view],
                    styles,
                    title,
                )

    def abc_block(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        n = len(rows)
        n_a = sum(1 for r in rows if r["A"])
        n_b = sum(1 for r in rows if r["B"])
        n_c = sum(1 for r in rows if r["C"])
        n_ab = sum(1 for r in rows if r["A"] and r["B"])
        n_abc = sum(1 for r in rows if r["A"] and r["B"] and r["C"])
        return {
            "n_visible_gt": n,
            "P(A)": rate(n_a, n),
            "P(B)": rate(n_b, n),
            "P(C)": rate(n_c, n),
            "P(B|A)": rate(n_ab, n_a),
            "P(C|A)": rate(sum(1 for r in rows if r["A"] and r["C"]), n_a),
            "P(ABC)": rate(n_abc, n),
            "n_A": n_a,
            "n_B": n_b,
            "n_C": n_c,
        }

    vis_rows = [r for r in object_rows]
    by_agent = {
        agent: abc_block([r for r in vis_rows if r["agent"] == agent])
        for agent in AGENT_ORDER
    }
    tags = sorted({r["tag"] for r in vis_rows})
    by_tag = {
        tag: {
            agent: abc_block(
                [r for r in vis_rows if r["agent"] == agent and r["tag"] == tag]
            )
            for agent in AGENT_ORDER
        }
        for tag in tags
    }
    pixel_by_agent_tag: Dict[str, Dict[str, Any]] = {}
    for agent, tag in sorted(pixel_tp.keys()):
        pixel_by_agent_tag.setdefault(tag, {})[agent] = pack_pixel(
            pixel_tp[(agent, tag)],
            pixel_fn[(agent, tag)],
            pixel_fp[(agent, tag)],
        )
    pixel_by_agent = {}
    for agent in AGENT_ORDER:
        tp = sum(pixel_tp[(agent, t)] for t in tags if (agent, t) in pixel_tp)
        fn = sum(pixel_fn[(agent, t)] for t in tags if (agent, t) in pixel_fn)
        fp = sum(pixel_fp[(agent, t)] for t in tags if (agent, t) in pixel_fp)
        pixel_by_agent[agent] = {
            **pack_pixel(tp, fn, fp),
            "n_frames": n_agent_frames[agent],
            "n_gt_empty_frames": n_gt_empty[agent],
        }

    report = {
        "checkpoint": f"net_epoch{opt.epoch}.pth",
        "split": "test",
        "n_scenes": len({s for s, _ in plan}),
        "frames_per_scene": int(opt.frames_per_scene),
        "seed": int(opt.seed),
        "fg_tau": float(opt.fg_tau),
        "sigma0_cells": float(opt.sigma0),
        "near_m": float(opt.near_m),
        "cover_thresh": float(opt.cover_thresh),
        "k_sigma": 2.0,
        "rsu_extrinsic": "identity cam2lidar (no invert)",
        "A": "FG seed (p_fg>=tau) inside projected GT polygon",
        "B": "2D-correct seed lifts to OBB surface distance <= near_m",
        "C": "2σ coverage of sampled GT box by those Gaussians >= cover_thresh",
        "pixel_by_agent": pixel_by_agent,
        "pixel_by_weather": pixel_by_agent_tag,
        "abc_by_agent": by_agent,
        "abc_by_weather": by_tag,
        "plan": [{"scene": s, "idx": i, "weather": weather_tag(s)} for s, i in plan],
    }
    out_json = out_root / f"epoch{opt.epoch}_abc.json"
    out_csv = out_root / f"epoch{opt.epoch}_abc_objects.csv"
    out_txt = out_root / f"epoch{opt.epoch}_abc.txt"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if vis_rows:
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(vis_rows[0].keys()))
            writer.writeheader()
            writer.writerows(vis_rows)

    def fmt(val: Optional[float]) -> str:
        return f"{val:.3f}" if val is not None else " n/a"

    lines = [
        f"heatmap-ft ABC  epoch={opt.epoch}  scenes={report['n_scenes']}  "
        f"{opt.frames_per_scene}/scene  seed={opt.seed}  tau={opt.fg_tau:g}",
        "RSU projection = identity cam2lidar (no invert).",
        f"A=2D FG in GT  B=lift within {opt.near_m:g}m  "
        f"C=2σ coverage>={opt.cover_thresh:g} at sigma0={opt.sigma0:g} cells",
        "",
        "pixel heatmap vs projected GT",
        f"{'agent':8s}  {'recall':>8s}  {'prec':>8s}  {'gt_px':>10s}  {'pred_px':>10s}",
    ]
    for agent in AGENT_ORDER:
        block = pixel_by_agent[agent]
        lines.append(
            f"{agent:8s}  {fmt(block['recall']):>8s}  {fmt(block['precision']):>8s}  "
            f"{block['n_gt_pixels']:10d}  {block['n_pred_pixels']:10d}"
        )
    lines.append("")
    lines.append("pixel recall by weather/tag")
    for tag in tags:
        lines.append(f"  [{tag}]")
        for agent in AGENT_ORDER:
            block = pixel_by_agent_tag.get(tag, {}).get(agent)
            if not block:
                continue
            lines.append(
                f"    {agent:8s} recall={fmt(block['recall'])}  "
                f"prec={fmt(block['precision'])}  gt_px={block['n_gt_pixels']}"
            )
    lines.append("")
    lines.append("object ABC (visible GT only)")
    for agent in AGENT_ORDER:
        block = by_agent[agent]
        lines.append(
            f"  {agent:8s} n={block['n_visible_gt']:4d}  "
            f"P(A)={fmt(block['P(A)'])}  P(B|A)={fmt(block['P(B|A)'])}  "
            f"P(C|A)={fmt(block['P(C|A)'])}  P(ABC)={fmt(block['P(ABC)'])}"
        )
    lines.append("")
    for tag in tags:
        lines.append(f"  ABC [{tag}]")
        for agent in AGENT_ORDER:
            block = by_tag[tag][agent]
            if block["n_visible_gt"] == 0:
                continue
            lines.append(
                f"    {agent:8s} n={block['n_visible_gt']:4d}  "
                f"P(A)={fmt(block['P(A)'])}  P(B|A)={fmt(block['P(B|A)'])}  "
                f"P(C|A)={fmt(block['P(C|A)'])}  P(ABC)={fmt(block['P(ABC)'])}"
            )
    text = "\n".join(lines) + "\n"
    out_txt.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"wrote {out_json}", flush=True)
    print(f"wrote {out_csv}", flush=True)
    print(f"wrote {out_txt}", flush=True)
    print(f"panels: {vis_dir}", flush=True)


if __name__ == "__main__":
    main()

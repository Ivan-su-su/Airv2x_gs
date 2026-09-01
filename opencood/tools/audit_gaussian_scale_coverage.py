# -*- coding: utf-8 -*-
"""Gaussian tangent-scale coverage audit. Isolated from production training.

Uses the CURRENT concat128 heatmap / depth predictions as seeds. Does not
change the model, losses, dataset, or checkpoints. Does not pick a final
sigma0.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.gaussian_modules_0822.heatmap.box_support import (
    VALID_BOX_CLASS_IDS,
    project_box_to_image,
)
from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    PRIMARY_OBJECTNESS_THRESHOLD,
)
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.models.gaussian_modules_0822.p1_layout import BLOCK, FEAT_H, FEAT_W
from opencood.tools import train_utils
from opencood.tools.eval_gaussian_p1 import denormalize_rgb, load_epoch_checkpoint
from opencood.tools.gaussian_scale_audit.coverage import (
    SEED_BUCKETS,
    boxes_hwl_to_corners,
    bucket_histogram,
    coverage_fraction,
    sample_box_points,
    sample_expanded_region,
    seed_bucket,
    support_precision,
)
from opencood.tools.gaussian_scale_audit.geometry import (
    audit_layout,
    invert_spd,
    r90_pixel_centers,
    view_gaussians,
)
from opencood.tools.gaussian_scale_audit.orientation import local_orientation
from opencood.tools.gaussian_scale_audit.visualize import (
    save_object_closeup,
    save_scale_sweep_view,
    save_seed_orientation_panel,
)
from opencood.tools.train_gaussian_p1 import _unwrap_model
from opencood.utils.airv2x_utils import CAMERA_KEYS_BY_AGENT
from opencood.utils.camera_utils import (
    FOG_BETA_RANGE,
    apply_atmospheric_fog_rgb,
    camera_optical_ray_range,
    imagenet_normalize_display_rgb,
)

AGENT_ORDER = ("vehicle", "rsu", "drone")
SCENE_RE = re.compile(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}")
DEFAULT_SIGMA0 = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)


def parse_args() -> argparse.Namespace:
    """CLI for the Gaussian scale coverage diagnostic."""
    parser = argparse.ArgumentParser(description="Gaussian scale coverage audit")
    parser.add_argument("-y", "--hypes_yaml", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epoch", type=int, default=16)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--out_root",
        default="/home/dell/suyi/visualization/gaussian_scale_coverage_audit",
    )
    parser.add_argument("--fg_tau", type=float, default=PRIMARY_OBJECTNESS_THRESHOLD)
    parser.add_argument(
        "--sigma0",
        default="0.25,0.50,0.75,1.00,1.50,2.00",
        help="comma-separated R90-cell tangent base scales",
    )
    parser.add_argument("--anisotropy_max", type=float, default=4.0)
    parser.add_argument("--orient_window", type=int, default=7)
    parser.add_argument("--drone_ray_sigma", type=float, default=2.0)
    parser.add_argument("--eps", type=float, default=1.0e-4)
    parser.add_argument("--box_res", type=int, default=5)
    parser.add_argument("--spill_res", type=int, default=6)
    parser.add_argument("--spill_expand", type=float, default=2.0)
    parser.add_argument("--val_frames_per_scene", type=int, default=2)
    parser.add_argument("--test_frames_per_scene", type=int, default=1)
    parser.add_argument("--fog_frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--indices",
        default="",
        help="optional explicit split:idx pairs, e.g. val:10,test:3",
    )
    return parser.parse_args()


def scene_from_path(path: Any) -> str:
    """Extract ``YYYY_MM_DD_HH_MM_SS`` from a dataset path."""
    match = SCENE_RE.search(str(path))
    return match.group(0) if match else "unknown"


def _as_numpy(value: Any) -> np.ndarray:
    """Detach tensors to CPU numpy."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _record_len(ego: Mapping[str, Any], agent_type: str) -> int:
    """Per-sample CAV count."""
    agent = ego.get(agent_type)
    if not isinstance(agent, Mapping):
        return 0
    record_len = agent.get("record_len")
    if record_len is None:
        return 0
    value = record_len[0] if torch.is_tensor(record_len) or np.ndim(record_len) else record_len
    if torch.is_tensor(value):
        return int(value.item())
    return int(np.asarray(value).reshape(-1)[0])


def _slice_cam(arr: np.ndarray, cav_idx: int, view_idx: int, n_views: int) -> np.ndarray:
    """Index stacked camera matrices."""
    if arr.ndim >= 3 and arr.shape[0] > cav_idx and arr.shape[1] == n_views:
        return np.asarray(arr[cav_idx, view_idx])
    return np.asarray(arr[cav_idx * n_views + view_idx])


def sample_plan(dataset: Any, frames_per_scene: int, seed: int) -> List[Tuple[str, int]]:
    """Random ``frames_per_scene`` indices per scenario."""
    rng = np.random.RandomState(int(seed))
    plan: List[Tuple[str, int]] = []
    for scene_i, end in enumerate(dataset.len_record):
        start = 0 if scene_i == 0 else int(dataset.len_record[scene_i - 1])
        scene_db = dataset.scenario_database[scene_i]
        first_cav = next(iter(scene_db.values()))
        ts0 = dataset.return_timestamp_key(scene_db, 0)
        scene = scene_from_path(first_cav[ts0]["metadata_path"])
        n = int(end) - start
        k = min(int(frames_per_scene), n)
        chosen = rng.choice(n, size=k, replace=False)
        for local in sorted(int(x) for x in chosen):
            plan.append((scene, start + local))
    return plan


def parse_sigma0(text: str) -> List[float]:
    """Parse comma-separated sigma0 list."""
    values = [float(tok.strip()) for tok in str(text).split(",") if tok.strip()]
    if not values:
        raise ValueError("sigma0 list is empty")
    return values


def parse_indices(text: str) -> List[Tuple[str, int]]:
    """Parse ``val:12,test:3`` into split/idx pairs."""
    if not str(text).strip():
        return []
    pairs: List[Tuple[str, int]] = []
    for tok in str(text).split(","):
        split, idx_s = tok.strip().split(":")
        pairs.append((split.strip(), int(idx_s)))
    return pairs


def load_split_dataset(hypes: Dict[str, Any], split: str) -> Any:
    """Build a dataset for ``train`` / ``val`` / ``test``."""
    local = dict(hypes)
    local["train"] = False
    if split == "train":
        local["validate_dir"] = local["root_dir"]
    elif split == "test":
        local["validate_dir"] = local["test_dir"]
    else:
        local["validate_dir"] = local["validate_dir"]
    print(f"Building {split} dataset from {local['validate_dir']}", flush=True)
    return build_dataset(local, visualize=False, train=False)


def gt_boxes_from_ego(ego: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Valid official 3D boxes ``[N,7]`` hwl and class ids."""
    boxes_all = ego.get("object_bbx_center")
    mask_all = ego.get("object_bbx_mask")
    if not torch.is_tensor(boxes_all) or not torch.is_tensor(mask_all):
        return np.zeros((0, 7), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    boxes = _as_numpy(boxes_all[0])
    mask = _as_numpy(mask_all[0]).reshape(-1) == 1
    boxes = boxes[mask].astype(np.float64)
    class_ids = ego.get("class_ids")
    if class_ids is None:
        ids = np.ones((boxes.shape[0],), dtype=np.int64)
    else:
        raw = class_ids[0]
        ids = _as_numpy(raw).reshape(-1).astype(np.int64)[: boxes.shape[0]]
        if ids.size < boxes.shape[0]:
            pad = np.ones((boxes.shape[0],), dtype=np.int64)
            pad[: ids.size] = ids
            ids = pad
    keep = np.ones((boxes.shape[0],), dtype=bool)
    for i in range(boxes.shape[0]):
        if int(ids[i]) not in VALID_BOX_CLASS_IDS or not np.isfinite(boxes[i]).all():
            keep[i] = False
    return boxes[keep], ids[keep]


def pairwise_for_agent(
    ego: Mapping[str, Any], agent_type: str
) -> Optional[np.ndarray]:
    """``[n_cav, 4, 4]`` cav-to-ego for one agent type."""
    pairwise_all = ego.get("img_pairwise_t_matrix_collab")
    if not torch.is_tensor(pairwise_all):
        return None
    pairwise = _as_numpy(pairwise_all[0])
    n_this = _record_len(ego, agent_type)
    if n_this == 0:
        return None
    offset = 0
    for name in AGENT_ORDER:
        if name == agent_type:
            break
        offset += _record_len(ego, name)
    out = np.zeros((n_this, 4, 4), dtype=np.float64)
    for local in range(n_this):
        t_cav2ego = np.asarray(pairwise[offset + local, 0], dtype=np.float64)
        if t_cav2ego.shape != (4, 4) or abs(float(np.linalg.det(t_cav2ego))) < 1e-8:
            t_cav2ego = np.eye(4, dtype=np.float64)
        out[local] = t_cav2ego
    return out


def n_views_of(imgs: torch.Tensor, n_cav: int, agent_type: str) -> int:
    """Camera count per CAV."""
    n_flat = int(imgs.shape[0]) if imgs.dim() == 4 else int(imgs.shape[0] * imgs.shape[1])
    if n_cav > 0 and n_flat % n_cav == 0:
        return n_flat // n_cav
    return len(CAMERA_KEYS_BY_AGENT.get(agent_type, ["cam"]))


def flatten_imgs(imgs: torch.Tensor) -> torch.Tensor:
    """``[A,V,C,H,W]`` or ``[N,C,H,W]`` → ``[N,C,H,W]``."""
    if imgs.dim() == 5:
        return imgs.reshape(-1, *imgs.shape[2:])
    return imgs


def apply_heavy_fog(ego: Dict[str, Any]) -> Dict[str, float]:
    """Koschmieder fog on already-collated eval RGB, high-end TRAIN betas."""
    betas = {agent: float(bounds[1]) for agent, bounds in FOG_BETA_RANGE.items()}
    for agent, beta in betas.items():
        if agent not in ego:
            continue
        cam = ego[agent].get("batch_merged_cam_inputs")
        if not isinstance(cam, dict) or not torch.is_tensor(cam.get("imgs")):
            continue
        imgs = cam["imgs"].clone()
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
            # Collated cam matrices may keep a singleton view dim, e.g. [1,3,3].
            k = intrins[idx].detach().float().cpu().reshape(-1, 3, 3)[0]
            prot = post_rots[idx].detach().float().cpu().reshape(-1, 3, 3)[0]
            ptra = post_trans[idx].detach().float().cpu().reshape(-1)[:3]
            rho = camera_optical_ray_range(depth, k, prot, ptra)
            fogged = apply_atmospheric_fog_rgb(rgb, rho, beta)
            imgs[idx, :3] = imagenet_normalize_display_rgb(fogged).to(
                device=imgs.device, dtype=imgs.dtype
            )
        cam["imgs"] = imgs
    return betas


def projected_polygons(
    ego: Mapping[str, Any],
    agent_type: str,
    boxes: np.ndarray,
    image_hw: Tuple[int, int],
) -> List[List[Optional[np.ndarray]]]:
    """Per-view list of projected GT polygons (or None)."""
    agent = ego.get(agent_type)
    cam = agent["batch_merged_cam_inputs"]
    imgs = flatten_imgs(cam["imgs"])
    n_cav = _record_len(ego, agent_type)
    n_view = n_views_of(imgs, n_cav, agent_type)
    n_flat = int(imgs.shape[0])
    pairwise = pairwise_for_agent(ego, agent_type)
    corners_ego = boxes_hwl_to_corners(boxes)
    intrins = _as_numpy(cam["intrinsics"])
    extrinsics = _as_numpy(cam["extrinsics"])
    post_rots = _as_numpy(cam["post_rots"])
    post_trans = _as_numpy(cam["post_trans"])
    from opencood.tools.gaussian_scale_audit.geometry import cam_to_lidar_matrix

    polys: List[List[Optional[np.ndarray]]] = [
        [None] * int(boxes.shape[0]) for _ in range(n_flat)
    ]
    if pairwise is None or boxes.shape[0] == 0:
        return polys
    ones = np.ones((corners_ego.shape[0] * 8, 1), dtype=np.float64)
    xyz_h = np.concatenate([corners_ego.reshape(-1, 3), ones], axis=1).T
    for local in range(n_cav):
        t_ego2cav = np.linalg.inv(pairwise[local])
        xyz_cav = (t_ego2cav @ xyz_h).T[:, :3].reshape(boxes.shape[0], 8, 3)
        for view in range(n_view):
            flat = local * n_view + view
            k = _slice_cam(intrins, local, view, n_view)
            ext = _slice_cam(extrinsics, local, view, n_view)
            prot = _slice_cam(post_rots, local, view, n_view)
            ptra = np.asarray(_slice_cam(post_trans, local, view, n_view)).reshape(-1)
            cam2lidar = cam_to_lidar_matrix(ext, agent_type)
            for box_i in range(int(boxes.shape[0])):
                proj = project_box_to_image(
                    xyz_cav[box_i], k, cam2lidar, prot, ptra, image_hw
                )
                if proj is None:
                    continue
                polys[flat][box_i] = proj[0]
    return polys


def associate_seeds(
    seed_uv: np.ndarray, polygons: Sequence[Optional[np.ndarray]]
) -> np.ndarray:
    """Assign each seed to a projected GT polygon, else -1."""
    n_seed = int(seed_uv.shape[0])
    assigned = np.full(n_seed, -1, dtype=np.int64)
    if n_seed == 0:
        return assigned
    for box_i, poly in enumerate(polygons):
        if poly is None or len(poly) < 3:
            continue
        inside = MplPath(np.asarray(poly, dtype=np.float64)).contains_points(seed_uv)
        take = inside & (assigned < 0)
        assigned[take] = int(box_i)
    return assigned


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    """Mean of finite values, else None."""
    arr = [float(v) for v in values if v == v]
    if not arr:
        return None
    return round(float(np.mean(arr)), 4)


def main() -> None:
    """Run the scale-coverage diagnostic and write report + panels."""
    opt = parse_args()
    sigma0_list = parse_sigma0(opt.sigma0)
    out_root = Path(opt.out_root)
    for sub in (
        "image_seed_orientation",
        "bev_scale_sweep",
        "side_scale_sweep",
        "object_closeups",
    ):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print("Creating model...", flush=True)
    model = train_utils.create_model(hypes)
    ckpt = load_epoch_checkpoint(model, opt.model_dir, opt.epoch)
    model.to(device)
    model.eval()
    core = _unwrap_model(model)
    z_bins = {
        agent: core.depth_moments[agent].z_bins.detach().cpu().numpy().astype(np.float64)
        for agent in core.depth_moments
    }
    layout = audit_layout()
    print("layout", layout, flush=True)

    explicit = parse_indices(opt.indices)
    jobs: List[Dict[str, Any]] = []
    datasets: Dict[str, Any] = {}
    if explicit:
        for split, idx in explicit:
            datasets[split] = datasets.get(split) or load_split_dataset(hypes, split)
            jobs.append({"split": split, "idx": idx, "fog": False, "tag": split})
    else:
        datasets["val"] = load_split_dataset(hypes, "val")
        datasets["test"] = load_split_dataset(hypes, "test")
        for scene, idx in sample_plan(
            datasets["val"], opt.val_frames_per_scene, opt.seed
        ):
            jobs.append(
                {"split": "val", "idx": idx, "fog": False, "tag": "val_normal", "scene": scene}
            )
        for scene, idx in sample_plan(
            datasets["test"], opt.test_frames_per_scene, opt.seed + 1
        ):
            jobs.append(
                {
                    "split": "test",
                    "idx": idx,
                    "fog": False,
                    "tag": "test_normal",
                    "scene": scene,
                }
            )
        fog_candidates = [job for job in jobs if job["split"] in ("val", "test")]
        rng = np.random.RandomState(opt.seed + 7)
        pick = list(fog_candidates)
        rng.shuffle(pick)
        for job in pick[: int(opt.fog_frames)]:
            fog_job = dict(job)
            fog_job["fog"] = True
            fog_job["tag"] = "drone_heavy_fog"
            jobs.append(fog_job)

    u_map, v_map = r90_pixel_centers()
    per_object_rows: List[Dict[str, Any]] = []
    per_frame_rows: List[Dict[str, Any]] = []
    closeup_pool: List[Dict[str, Any]] = []
    sigma_depth_diffs: List[float] = []
    seed_counts: Dict[str, List[int]] = defaultdict(list)
    zero_visible: Dict[str, List[int]] = defaultdict(list)
    n_fwd = 0

    with torch.no_grad():
        for job_i, job in enumerate(jobs):
            dataset = datasets[job["split"]]
            sample = dataset[int(job["idx"])]
            batch = dataset.collate_batch_test([sample])
            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            fog_betas: Dict[str, float] = {}
            if job["fog"]:
                fog_betas = apply_heavy_fog(ego)
            meta = ego.get("metadata_path_list", [""])[0]
            scene = scene_from_path(meta) if meta else job.get("scene", "unknown")
            pred = model(ego)
            present = present_camera_agents(ego)
            boxes, _class_ids = gt_boxes_from_ego(ego)
            corners_ego = boxes_hwl_to_corners(boxes)
            print(
                f"[{job_i + 1}/{len(jobs)}] {job['split']} idx={job['idx']} "
                f"scene={scene} fog={job['fog']} agents={present} n_gt={boxes.shape[0]}",
                flush=True,
            )
            frame_gaussians: Dict[str, Any] = {
                "mu": [],
                "agent": [],
                "box_id": [],
                "near_gt": [],
                "sigma": {float(s): [] for s in sigma0_list},
            }
            vis_done: Dict[str, int] = defaultdict(int)
            n_fwd += 1

            for agent in AGENT_ORDER:
                if agent not in present or agent not in pred:
                    continue
                cam = ego[agent]["batch_merged_cam_inputs"]
                imgs = flatten_imgs(cam["imgs"])
                n_cav = _record_len(ego, agent)
                n_view = n_views_of(imgs, n_cav, agent)
                n_flat = int(imgs.shape[0])
                logits = pred[agent]["heatmap_logits"]
                p_fg = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                feat_h, feat_w = int(p_fg.shape[-2]), int(p_fg.shape[-1])
                if (feat_h, feat_w) != (FEAT_H, FEAT_W):
                    print(
                        f"  WARN {agent} heatmap {feat_h}x{feat_w} != {FEAT_H}x{FEAT_W}"
                    )
                image_hw = (int(imgs.shape[-2]), int(imgs.shape[-1]))
                polygons = projected_polygons(ego, agent, boxes, image_hw)
                pairwise = pairwise_for_agent(ego, agent)
                intrins = _as_numpy(cam["intrinsics"])
                extrinsics = _as_numpy(cam["extrinsics"])
                post_rots = _as_numpy(cam["post_rots"])
                post_trans = _as_numpy(cam["post_trans"])
                z_mean = pred[agent]["depth_z_mean"].detach().cpu().numpy()
                depth_prob = None
                if agent in ("vehicle", "rsu") and "depth_logits" in pred[agent]:
                    depth_prob = (
                        torch.softmax(pred[agent]["depth_logits"], dim=1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                visible = np.zeros((boxes.shape[0],), dtype=bool)
                seeds_per_box = np.zeros((boxes.shape[0],), dtype=np.int64)
                agent_mu: List[np.ndarray] = []
                agent_box: List[int] = []
                agent_sigma: Dict[float, List[np.ndarray]] = {
                    float(s): [] for s in sigma0_list
                }

                for flat in range(n_flat):
                    local = flat // n_view
                    view = flat % n_view
                    fg = p_fg[flat] >= float(opt.fg_tau)
                    ys, xs = np.where(fg)
                    n_seed = int(ys.size)
                    seed_counts[agent].append(n_seed)
                    seed_u = u_map[ys, xs] if n_seed else np.zeros((0,), dtype=np.float64)
                    seed_v = v_map[ys, xs] if n_seed else np.zeros((0,), dtype=np.float64)
                    seed_uv = (
                        np.stack([seed_u, seed_v], axis=1)
                        if n_seed
                        else np.zeros((0, 2), dtype=np.float64)
                    )
                    theta, aniso, _l1, _l2 = local_orientation(
                        p_fg[flat],
                        ys,
                        xs,
                        window=int(opt.orient_window),
                        anisotropy_max=float(opt.anisotropy_max),
                    )
                    assigned = associate_seeds(seed_uv, polygons[flat])
                    for box_i, poly in enumerate(polygons[flat]):
                        if poly is not None:
                            visible[box_i] = True
                    for box_i in assigned[assigned >= 0]:
                        seeds_per_box[int(box_i)] += 1

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
                    if n_seed:
                        z_cells = z_mean[flat, ys, xs]
                        prob_sel = (
                            depth_prob[flat][:, ys, xs]
                            if depth_prob is not None
                            else None
                        )
                        mu_view, sig_view, diff = view_gaussians(
                            seed_u,
                            seed_v,
                            z_cells,
                            theta,
                            aniso,
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
                        if diff:
                            sigma_depth_diffs.append(diff)
                        agent_mu.extend(list(mu_view))
                        agent_box.extend(int(x) for x in assigned)
                        for sigma0 in sigma0_list:
                            agent_sigma[float(sigma0)].extend(
                                list(sig_view[float(sigma0)])
                            )

                    if vis_done[agent] < 2 and (n_seed > 0 or any(p is not None for p in polygons[flat])):
                        rgb = denormalize_rgb(imgs[flat])
                        polys_v = [p for p in polygons[flat] if p is not None]
                        tag = "fog" if job["fog"] else "clear"
                        save_seed_orientation_panel(
                            out_root
                            / "image_seed_orientation"
                            / f"{job['split']}_{scene}_{job['idx']:04d}_{agent}_v{flat}_{tag}.png",
                            rgb,
                            polys_v,
                            p_fg[flat],
                            fg.astype(np.uint8),
                            seed_uv,
                            theta,
                            f"{job['split']} {scene} idx={job['idx']} {agent} view={flat} "
                            f"fog={job['fog']} seeds={n_seed} tau={opt.fg_tau:g}",
                        )
                        vis_done[agent] += 1

                mu_arr = (
                    np.stack(agent_mu, axis=0)
                    if agent_mu
                    else np.zeros((0, 3), dtype=np.float64)
                )
                box_arr = np.asarray(agent_box, dtype=np.int64)
                near = box_arr >= 0
                if mu_arr.size:
                    frame_gaussians["mu"].append(mu_arr)
                    frame_gaussians["agent"].append(
                        np.full((mu_arr.shape[0],), agent, dtype=object)
                    )
                    frame_gaussians["box_id"].append(box_arr)
                    frame_gaussians["near_gt"].append(near)
                    for sigma0 in sigma0_list:
                        frame_gaussians["sigma"][float(sigma0)].append(
                            np.stack(agent_sigma[float(sigma0)], axis=0)
                        )

                vis_n = int(visible.sum())
                zero_n = int(((seeds_per_box == 0) & visible).sum())
                zero_visible[agent].append(zero_n)
                hist = bucket_histogram(seeds_per_box[visible].tolist() if vis_n else [])
                per_frame_rows.append(
                    {
                        "split": job["split"],
                        "scene": scene,
                        "idx": int(job["idx"]),
                        "fog": bool(job["fog"]),
                        "tag": job["tag"],
                        "agent": agent,
                        "n_gt_visible": vis_n,
                        "n_gt_zero_seed": zero_n,
                        "zero_seed_rate": round(zero_n / max(vis_n, 1), 4) if vis_n else None,
                        "n_seeds": int(mu_arr.shape[0]),
                        "n_gt_near_seeds": int(near.sum()) if mu_arr.size else 0,
                        **{f"bucket_{k}": hist[k] for k in SEED_BUCKETS},
                        "fog_beta": fog_betas.get(agent),
                    }
                )

                for box_i in range(int(boxes.shape[0])):
                    if not visible[box_i]:
                        continue
                    n_seed_box = int(seeds_per_box[box_i])
                    pts = sample_box_points(boxes[box_i], int(opt.box_res))
                    spill_pts, spill_in = sample_expanded_region(
                        boxes[box_i], int(opt.spill_res), float(opt.spill_expand)
                    )
                    if mu_arr.size:
                        use = box_arr == box_i
                        mu_obj = mu_arr[use]
                    else:
                        use = np.zeros((0,), dtype=bool)
                        mu_obj = np.zeros((0, 3), dtype=np.float64)
                    row: Dict[str, Any] = {
                        "split": job["split"],
                        "scene": scene,
                        "idx": int(job["idx"]),
                        "fog": bool(job["fog"]),
                        "tag": job["tag"],
                        "agent": agent,
                        "box_i": int(box_i),
                        "n_seed": n_seed_box,
                        "seed_bucket": seed_bucket(n_seed_box),
                        "visible": True,
                    }
                    for sigma0 in sigma0_list:
                        if mu_obj.shape[0] == 0:
                            row[f"cov1_s{sigma0:g}"] = 0.0
                            row[f"cov2_s{sigma0:g}"] = 0.0
                            row[f"prec1_s{sigma0:g}"] = None
                            row[f"prec2_s{sigma0:g}"] = None
                            continue
                        covs = np.stack(agent_sigma[float(sigma0)], axis=0)[use]
                        prec = invert_spd(covs)
                        row[f"cov1_s{sigma0:g}"] = round(
                            coverage_fraction(pts, mu_obj, prec, 1.0), 4
                        )
                        row[f"cov2_s{sigma0:g}"] = round(
                            coverage_fraction(pts, mu_obj, prec, 2.0), 4
                        )
                        p1, _, _ = support_precision(
                            spill_pts, spill_in, mu_obj, prec, 1.0
                        )
                        p2, _, _ = support_precision(
                            spill_pts, spill_in, mu_obj, prec, 2.0
                        )
                        row[f"prec1_s{sigma0:g}"] = None if p1 != p1 else round(p1, 4)
                        row[f"prec2_s{sigma0:g}"] = None if p2 != p2 else round(p2, 4)
                    per_object_rows.append(row)
                    if n_seed_box in (0, 1) or seed_bucket(n_seed_box) in (
                        "2-3",
                        "4-8",
                        ">8",
                    ):
                        closeup_pool.append(
                            {
                                "row": row,
                                "box": boxes[box_i],
                                "corners": corners_ego[box_i],
                                "mu": mu_obj.copy(),
                                "sigma": {
                                    float(s): np.stack(agent_sigma[float(s)], axis=0)[use]
                                    if mu_obj.shape[0]
                                    else np.zeros((0, 3, 3))
                                    for s in sigma0_list
                                },
                            }
                        )

            if frame_gaussians["mu"]:
                mu_all = np.concatenate(frame_gaussians["mu"], axis=0)
                near_all = np.concatenate(frame_gaussians["near_gt"], axis=0)
                sig_all = {
                    float(s): np.concatenate(frame_gaussians["sigma"][float(s)], axis=0)
                    for s in sigma0_list
                }
                tag = "fog" if job["fog"] else "clear"
                stem = f"{job['split']}_{scene}_{job['idx']:04d}_{tag}"
                save_scale_sweep_view(
                    out_root / "bev_scale_sweep" / f"{stem}_all.png",
                    "bev",
                    sigma0_list,
                    mu_all,
                    sig_all,
                    corners_ego,
                    f"{stem} ALL Gaussians  n={mu_all.shape[0]}",
                )
                save_scale_sweep_view(
                    out_root / "bev_scale_sweep" / f"{stem}_gtnear.png",
                    "bev",
                    sigma0_list,
                    mu_all[near_all],
                    {s: sig_all[s][near_all] for s in sig_all},
                    corners_ego,
                    f"{stem} GT-near Gaussians  n={int(near_all.sum())}",
                )
                save_scale_sweep_view(
                    out_root / "side_scale_sweep" / f"{stem}_all.png",
                    "side",
                    sigma0_list,
                    mu_all,
                    sig_all,
                    corners_ego,
                    f"{stem} ALL side  n={mu_all.shape[0]}",
                )
                save_scale_sweep_view(
                    out_root / "side_scale_sweep" / f"{stem}_gtnear.png",
                    "side",
                    sigma0_list,
                    mu_all[near_all],
                    {s: sig_all[s][near_all] for s in sig_all},
                    corners_ego,
                    f"{stem} GT-near side  n={int(near_all.sum())}",
                )

    _write_closeups(out_root, closeup_pool, sigma0_list)
    _write_reports(
        out_root,
        opt,
        ckpt,
        jobs,
        layout,
        sigma0_list,
        per_object_rows,
        per_frame_rows,
        sigma_depth_diffs,
    )
    print(f"wrote {out_root}", flush=True)


def _write_closeups(
    out_root: Path,
    pool: List[Dict[str, Any]],
    sigma0_list: Sequence[float],
) -> None:
    """Save a few per-bucket object close-ups."""
    chosen: List[Dict[str, Any]] = []
    seen = set()
    for want in ("0", "1", "2-3", "4-8", ">8"):
        for item in pool:
            row = item["row"]
            key = (row["agent"], row["seed_bucket"], row["scene"], row["idx"], row["box_i"])
            if row["seed_bucket"] != want or key in seen:
                continue
            if want != "0" and item["mu"].shape[0] == 0:
                continue
            seen.add(key)
            chosen.append(item)
            break
    for item in chosen:
        row = item["row"]
        name = (
            f"{row['agent']}_bkt{row['seed_bucket']}_{row['scene']}_"
            f"{row['idx']:04d}_box{row['box_i']}.png"
        )
        save_object_closeup(
            out_root / "object_closeups" / name,
            item["corners"],
            item["mu"],
            item["sigma"],
            sigma0_list,
            f"{row['agent']} {row['scene']} idx={row['idx']} box={row['box_i']} "
            f"seeds={row['n_seed']} fog={row['fog']}",
        )


def _write_reports(
    out_root: Path,
    opt: argparse.Namespace,
    ckpt: str,
    jobs: Sequence[Mapping[str, Any]],
    layout: Dict[str, Any],
    sigma0_list: Sequence[float],
    per_object: List[Dict[str, Any]],
    per_frame: List[Dict[str, Any]],
    sigma_depth_diffs: Sequence[float],
) -> None:
    """Write csv / json / txt summaries. Does not pick a winner sigma0."""
    if per_object:
        with (out_root / "per_object.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_object[0].keys()))
            writer.writeheader()
            writer.writerows(per_object)
    if per_frame:
        with (out_root / "per_frame.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_frame[0].keys()))
            writer.writeheader()
            writer.writerows(per_frame)

    def _agg_cov(agent: str, sigma0: float, k: str, fog: Optional[bool] = None) -> Optional[float]:
        vals = []
        for row in per_object:
            if row["agent"] != agent:
                continue
            if fog is not None and bool(row["fog"]) != fog:
                continue
            val = row.get(f"{k}_s{sigma0:g}")
            if val is not None:
                vals.append(float(val))
        return mean_or_none(vals)

    by_agent: Dict[str, Any] = {}
    for agent in AGENT_ORDER:
        frames = [r for r in per_frame if r["agent"] == agent and not r["fog"]]
        objs = [r for r in per_object if r["agent"] == agent and not r["fog"]]
        vis = sum(int(r["n_gt_visible"]) for r in frames)
        zero = sum(int(r["n_gt_zero_seed"]) for r in frames)
        seeds = [int(r["n_seeds"]) for r in frames]
        hist = bucket_histogram([int(r["n_seed"]) for r in objs])
        cov = {}
        for sigma0 in sigma0_list:
            cov[str(sigma0)] = {
                "coverage_1sigma": _agg_cov(agent, float(sigma0), "cov1", False),
                "coverage_2sigma": _agg_cov(agent, float(sigma0), "cov2", False),
                "support_precision_1sigma": _agg_cov(agent, float(sigma0), "prec1", False),
                "support_precision_2sigma": _agg_cov(agent, float(sigma0), "prec2", False),
            }
        by_bucket: Dict[str, Any] = {}
        for bucket in SEED_BUCKETS:
            sub = [r for r in objs if r["seed_bucket"] == bucket]
            by_bucket[bucket] = {
                "n_objects": len(sub),
                "coverage_2sigma": {
                    str(s): mean_or_none(
                        [
                            float(r[f"cov2_s{s:g}"])
                            for r in sub
                            if r.get(f"cov2_s{s:g}") is not None
                        ]
                    )
                    for s in sigma0_list
                },
            }
        fog_objs = [r for r in per_object if r["agent"] == agent and r["fog"]]
        fog_frames = [r for r in per_frame if r["agent"] == agent and r["fog"]]
        by_agent[agent] = {
            "n_frames": len(frames),
            "n_seeds_mean": mean_or_none(seeds),
            "n_seeds_sum": int(sum(seeds)),
            "n_gt_visible": vis,
            "n_gt_zero_seed": zero,
            "zero_seed_GT_rate": round(zero / max(vis, 1), 4) if vis else None,
            "seed_count_histogram": hist,
            "by_sigma0": cov,
            "by_seed_bucket": by_bucket,
            "fog": {
                "n_frames": len(fog_frames),
                "zero_seed_GT_rate": (
                    round(
                        sum(int(r["n_gt_zero_seed"]) for r in fog_frames)
                        / max(sum(int(r["n_gt_visible"]) for r in fog_frames), 1),
                        4,
                    )
                    if fog_frames
                    else None
                ),
                "coverage_2sigma": {
                    str(s): _agg_cov(agent, float(s), "cov2", True) for s in sigma0_list
                },
            },
        }

    report = {
        "checkpoint": ckpt,
        "epoch": int(opt.epoch),
        "fg_selection": {
            "rule": "p_fg = softmax(heatmap_logits)[:, 1]; seed iff p_fg >= tau",
            "tau": float(opt.fg_tau),
            "note": "production has no Gaussian seed module; this is the vis/eval FG rule",
        },
        "layout": layout,
        "sigma0_r90_cells": list(sigma0_list),
        "anisotropy_max": float(opt.anisotropy_max),
        "orient_window": int(opt.orient_window),
        "drone_ray_sigma_m": float(opt.drone_ray_sigma),
        "fog_betas_high_end": {k: float(v[1]) for k, v in FOG_BETA_RANGE.items()},
        "jobs": [
            {"split": j["split"], "idx": int(j["idx"]), "fog": bool(j["fog"]), "tag": j["tag"]}
            for j in jobs
        ],
        "n_frames_forward": len(jobs),
        "sigma_depth_direct_vs_varz_qqT_max_abs": (
            round(float(np.max(sigma_depth_diffs)), 8) if sigma_depth_diffs else None
        ),
        "by_agent": by_agent,
        "do_not_autoselect_sigma0": True,
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))

    lines = [
        "Gaussian tangent-scale coverage audit (diagnostic, no scale selected)",
        f"checkpoint: {ckpt}",
        f"FG rule: softmax class-1 >= {opt.fg_tau:g}  (vis/eval threshold; no prod seed module)",
        f"R90: {FEAT_H}x{FEAT_W}  center u=4j+2 v=4i+2  block={BLOCK}",
        f"lift: X_cam = z * K^-1 undo_post([u,v,1])  (not z*normalize(q))",
        f"sigma0 (R90 cells): {list(sigma0_list)}",
        f"anisotropy_max={opt.anisotropy_max:g}  drone_ray_sigma={opt.drone_ray_sigma:g} m",
        f"Var_z q q^T vs direct Sigma_depth max|diff|="
        f"{report['sigma_depth_direct_vs_varz_qqT_max_abs']}",
        "",
    ]
    for agent, block in by_agent.items():
        lines.append(
            f"=== {agent}  frames={block['n_frames']}  seeds={block['n_seeds_sum']}  "
            f"zero-seed GT rate={block['zero_seed_GT_rate']}"
        )
        lines.append(f"  seed-count histogram (visible GT): {block['seed_count_histogram']}")
        lines.append(
            f"  {'sigma0':>8s}  {'cov@1s':>8s}  {'cov@2s':>8s}  {'prec@1s':>8s}  {'prec@2s':>8s}"
        )
        for sigma0 in sigma0_list:
            item = block["by_sigma0"][str(sigma0)]
            def _fmt(val: Optional[float]) -> str:
                return "   n/a" if val is None else f"{val:8.3f}"

            lines.append(
                f"  {sigma0:8.2f}  {_fmt(item['coverage_1sigma'])}  "
                f"{_fmt(item['coverage_2sigma'])}  "
                f"{_fmt(item['support_precision_1sigma'])}  "
                f"{_fmt(item['support_precision_2sigma'])}"
            )
        lines.append("  coverage@2s by seed-count bucket:")
        for bucket in SEED_BUCKETS:
            covs = block["by_seed_bucket"][bucket]["coverage_2sigma"]
            n_obj = block["by_seed_bucket"][bucket]["n_objects"]
            cov_s = " ".join(
                f"{s:g}:{covs[str(s)] if covs[str(s)] is not None else 'n/a'}"
                for s in sigma0_list
            )
            lines.append(f"    [{bucket:4s}] n={n_obj:4d}  {cov_s}")
        fog = block["fog"]
        lines.append(
            f"  heavy-fog frames={fog['n_frames']}  zero-seed GT rate={fog['zero_seed_GT_rate']}"
        )
        lines.append("")
    lines.extend(
        [
            "Interpretation notes (not a selected scale):",
            "- Bucket 0: scale cannot help; heatmap recall failure.",
            "- Rising coverage with sigma0 plus falling support precision = spill tradeoff.",
            "- Compare Vehicle / RSU / Drone before assuming one global sigma0.",
            f"panels: {out_root}",
        ]
    )
    (out_root / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

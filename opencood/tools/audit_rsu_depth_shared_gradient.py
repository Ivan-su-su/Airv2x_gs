# -*- coding: utf-8 -*-
"""READ-ONLY RSU depth residual + shared-F90 gradient audit.

Does not modify production model, losses, targets, optimizer, or training.
Does not implement Gaussian generation or calibration in production.
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
from opencood.loss.gaussian_p1_semantic_loss import softmax_focal_loss
from opencood.loss.point_pillar_depth_loss import FocalLoss
from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    PRIMARY_OBJECTNESS_THRESHOLD,
)
from opencood.models.gaussian_modules_0822.heatmap.target import build_semantic_target
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.models.gaussian_modules_0822.lss.target import (
    depth_valid_mask,
    extract_camera_z_gt,
)
from opencood.models.gaussian_modules_0822.p1_layout import BLOCK, FEAT_H, FEAT_W
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
from opencood.tools.gaussian_scale_audit.geometry import (
    cam_to_ego_rt,
    r90_pixel_centers,
    ray_dir_cam,
    transform_points,
)
from opencood.tools.train_gaussian_p1 import (
    _unwrap_model,
    build_depth_class_target,
    build_depth_targets,
    build_depth_valid_masks,
)
from opencood.tools.vis_test_heatmap_recall import scene_from_path
from opencood.utils.airv2x_utils import CAMERA_KEYS_BY_AGENT
from opencood.utils.camera_utils import depth_discretization

JOINT_EPOCHS = (1, 4, 9, 16)
EPS = 1.0e-8
FG_TAU = PRIMARY_OBJECTNESS_THRESHOLD
NEAR_M = 2.0
RSU_CAM_KEYS = CAMERA_KEYS_BY_AGENT["rsu"]


def parse_args() -> argparse.Namespace:
    """CLI for the read-only RSU audit."""
    parser = argparse.ArgumentParser(description="RSU depth/gradient diagnostic")
    parser.add_argument("-y", "--hypes_yaml", required=True)
    parser.add_argument("--joint_dir", required=True)
    parser.add_argument("--ft_dir", default="")
    parser.add_argument("--ft_epoch", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_train", type=int, default=12)
    parser.add_argument("--n_val", type=int, default=8)
    parser.add_argument("--n_test", type=int, default=8)
    parser.add_argument("--n_grad", type=int, default=8)
    parser.add_argument("--n_drift", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv_cap", type=int, default=6000)
    parser.add_argument(
        "--out_root",
        default="/home/dell/suyi/visualization/rsu_depth_shared_gradient_audit",
    )
    return parser.parse_args()


def verified_architecture(core: torch.nn.Module, hypes: Mapping[str, Any]) -> Dict[str, Any]:
    """Record verified RSU module paths from the live model."""
    enc = core.frontend.encoders["rsu"]
    bins = depth_discretization(
        float(enc.d_min),
        float(enc.d_max),
        int(enc.num_bins),
        str(enc.mode),
    )
    report = {
        "rsu_encoder": "frontend.encoders['rsu']  CamEncode  "
        "opencood/models/sub_modules/lss_submodule.py",
        "efficientnet": "frontend.encoders['rsu'].trunk  EfficientNet-B0 "
        "from_pretrained('efficientnet-b0'); non-BN fine-tuned; BN frozen",
        "up1": "frontend.encoders['rsu'].up1  Up(320+112, 256)  lss_submodule.py",
        "up2": "frontend.encoders['rsu'].up2  Up(256+40, 256)  downsample=8",
        "concat128": "highres['rsu']  HighResFusion cat(R2, up F45) 280→128→128  "
        "opencood/models/gaussian_modules_0822/highres_adapter.py",
        "heatmap_head": "heatmap_heads['rsu']  HeatmapHead 3x3 128→128 ReLU 1x1→2  "
        "heatmap/head.py",
        "depth_head": "depth_heads['rsu']  DepthHead 3x3 128→128 ReLU 1x1→D=48  "
        "lss/head.py; official CamEncode.depth_head unused",
        "heatmap_target": "build_semantic_target SAM3 image_semantic_gts tau=1  "
        "vehicle/RSU never use box union",
        "depth_target": "build_depth_class_target LID bins at R90 centers; "
        "RSU loss = Focal on SAM3 FG ∩ in-range  gaussian_p1_depth_loss.py",
        "heatmap_loss": "softmax_focal_loss gamma=2 no alpha  "
        "opencood/loss/gaussian_p1_semantic_loss.py",
        "depth_loss": "FocalLoss alpha=0.25 gamma=2 reduction=none  "
        "opencood/loss/point_pillar_depth_loss.py",
        "rsu_ddiscr": [float(enc.d_min), float(enc.d_max), int(enc.num_bins)],
        "rsu_mode": str(enc.mode),
        "bin_centers": [round(float(z), 4) for z in bins],
        "bin_centers_first_last": [float(bins[0]), float(bins[-1])],
        "n_bins": int(len(bins)),
        "cam2lidar": "identity after ue4_to_lss; LIDAR2CAM_STORED rsu=False",
        "agents_independent": True,
        "shared_within_rsu": [
            "frontend.encoders.rsu.trunk",
            "frontend.encoders.rsu.up1",
            "frontend.encoders.rsu.up2",
            "highres.rsu",
        ],
        "up1_type": type(enc.up1).__name__,
        "up2_type": type(enc.up2).__name__,
        "trunk_type": type(enc.trunk).__name__,
        "fusion_type": type(core.highres["rsu"]).__name__,
    }
    return report


def load_split_dataset(hypes: Dict[str, Any], split: str) -> Any:
    """Build train/val/test with train=False for deterministic frames."""
    local = dict(hypes)
    local["train"] = False
    if split == "train":
        local["validate_dir"] = local["root_dir"]
    elif split == "test":
        local["validate_dir"] = local["test_dir"]
    print(f"Building {split} from {local['validate_dir']}", flush=True)
    return build_dataset(local, visualize=False, train=False)


def sample_has_rsu(sample: Mapping[str, Any]) -> bool:
    """True if the uncollated sample contains RSU cameras."""
    ego = sample.get("ego", sample)
    agent = ego.get("rsu")
    if not isinstance(agent, Mapping):
        return False
    cam = agent.get("batch_merged_cam_inputs") or agent.get("processed_cam_inputs")
    if isinstance(cam, Mapping) and cam.get("imgs") is not None:
        return True
    if agent.get("record_len"):
        return True
    return "rsu" in present_camera_agents(ego) if isinstance(ego, dict) else False


def collect_rsu_indices(dataset: Any, n_keep: int, seed: int) -> List[int]:
    """Random indices whose collated batch contains RSU imgs."""
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(len(dataset))[: max(int(n_keep) * 12, 40)]
    found: List[int] = []
    for idx in order:
        if len(found) >= int(n_keep):
            break
        sample = dataset[int(idx)]
        batch = dataset.collate_batch_test([sample])
        ego = batch["ego"]
        if "rsu" in present_camera_agents(ego):
            found.append(int(idx))
            print(f"  rsu idx={idx}  ({len(found)}/{n_keep})", flush=True)
    return found


def percentile_dict(values: np.ndarray, keys: Sequence[float]) -> Dict[str, Optional[float]]:
    """Named percentiles of finite values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"p{int(k)}": None for k in keys}
    return {f"p{int(k)}": round(float(np.percentile(finite, k)), 4) for k in keys}


def residual_summary(e: np.ndarray, abs_e: np.ndarray, sigma: np.ndarray) -> Dict[str, Any]:
    """Signed / absolute residual statistics."""
    e = e[np.isfinite(e)]
    abs_e = abs_e[np.isfinite(abs_e)]
    sigma = sigma[np.isfinite(sigma)]
    if e.size == 0:
        return {"n": 0}
    ratio = abs_e / np.clip(sigma[: abs_e.size], EPS, None) if sigma.size else np.array([])
    return {
        "n": int(e.size),
        "mean_e": round(float(e.mean()), 4),
        "median_e": round(float(np.median(e)), 4),
        "std_e": round(float(e.std()), 4),
        "mean_abs": round(float(abs_e.mean()), 4),
        "median_abs": round(float(np.median(abs_e)), 4),
        **percentile_dict(e, (10, 25, 50, 75, 90, 95)),
        "abs_p50": round(float(np.percentile(abs_e, 50)), 4),
        "abs_p90": round(float(np.percentile(abs_e, 90)), 4),
        "abs_p95": round(float(np.percentile(abs_e, 95)), 4),
        "mean_sigma": round(float(sigma.mean()), 4) if sigma.size else None,
        "median_sigma": round(float(np.median(sigma)), 4) if sigma.size else None,
        "median_abs_over_sigma": round(float(np.median(ratio)), 4) if ratio.size else None,
        "p90_abs_over_sigma": round(float(np.percentile(ratio, 90)), 4)
        if ratio.size
        else None,
        "frac_e_gt0": round(float((e > 0).mean()), 4),
        "frac_e_lt0": round(float((e < 0).mean()), 4),
    }


def fit_affine(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Least-squares y ≈ a x + b and R^2."""
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite].astype(np.float64)
    y = y[finite].astype(np.float64)
    if x.size < 8:
        return {"a": float("nan"), "b": float("nan"), "r2": float("nan"), "n": 0}
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + EPS
    return {
        "a": round(float(a), 5),
        "b": round(float(b), 5),
        "r2": round(1.0 - ss_res / ss_tot, 4),
        "n": int(x.size),
    }


def choose_depth_bins(z_gt: np.ndarray) -> List[float]:
    """Bins from observed RSU GT depth percentiles, clipped to LID range."""
    finite = z_gt[np.isfinite(z_gt)]
    if finite.size == 0:
        return [2.0, 8.0, 15.0, 25.0, 40.0, 50.0]
    qs = np.percentile(finite, [0, 20, 40, 60, 80, 100])
    edges = [2.0]
    for q in qs[1:-1]:
        q = float(np.clip(q, 2.5, 49.0))
        if q - edges[-1] >= 2.0:
            edges.append(round(q, 1))
    edges.append(50.0)
    return edges


def box_seed_masks(
    ego: Mapping[str, Any],
    p_fg: np.ndarray,
    z_mean: np.ndarray,
    boxes: np.ndarray,
    image_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean maps: inside projected GT polygon, and A-seed (p_fg≥tau in poly)."""
    n_flat, feat_h, feat_w = p_fg.shape
    in_box = np.zeros((n_flat, feat_h, feat_w), dtype=bool)
    a_seed = np.zeros_like(in_box)
    if boxes.shape[0] == 0:
        return in_box, a_seed
    polygons = projected_polygons(ego, "rsu", boxes, image_hw)
    u_map, v_map = r90_pixel_centers()
    pred_fg = p_fg >= FG_TAU
    for flat in range(n_flat):
        ys, xs = np.where(pred_fg[flat])
        seed_uv = (
            np.stack([u_map[ys, xs], v_map[ys, xs]], axis=1)
            if ys.size
            else np.zeros((0, 2))
        )
        cell_u = u_map.reshape(-1)
        cell_v = v_map.reshape(-1)
        cell_uv = np.stack([cell_u, cell_v], axis=1)
        for poly in polygons[flat]:
            if poly is None or len(poly) < 3:
                continue
            path = MplPath(np.asarray(poly, dtype=np.float64))
            inside_cells = path.contains_points(cell_uv).reshape(feat_h, feat_w)
            in_box[flat] |= inside_cells
            if seed_uv.shape[0]:
                hit = path.contains_points(seed_uv)
                a_seed[flat, ys[hit], xs[hit]] = True
    return in_box, a_seed


def a_plus_b_mask(
    ego: Mapping[str, Any],
    a_seed: np.ndarray,
    z_pred: np.ndarray,
    boxes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """A-seed pixels whose lifted mean is within 2 m of a GT OBB surface.

    Returns:
        ``(ab_mask, min_surface_distance)``. Distance is NaN off A-seeds.
    """
    ab = np.zeros_like(a_seed)
    surf = np.full(a_seed.shape, np.nan, dtype=np.float64)
    if boxes.shape[0] == 0 or not a_seed.any():
        return ab, surf
    cam = ego["rsu"]["batch_merged_cam_inputs"]
    imgs = flatten_imgs(cam["imgs"])
    n_cav = _record_len(ego, "rsu")
    n_view = n_views_of(imgs, n_cav, "rsu")
    pairwise = pairwise_for_agent(ego, "rsu")
    if pairwise is None:
        return ab, surf
    intrins = _as_numpy(cam["intrinsics"])
    extrinsics = _as_numpy(cam["extrinsics"])
    post_rots = _as_numpy(cam["post_rots"])
    post_trans = _as_numpy(cam["post_trans"])
    u_map, v_map = r90_pixel_centers()
    n_flat = int(a_seed.shape[0])
    for flat in range(n_flat):
        ys, xs = np.where(a_seed[flat])
        if ys.size == 0:
            continue
        local = flat // n_view
        view = flat % n_view
        k = np.asarray(_slice_cam(intrins, local, view, n_view), dtype=np.float64).reshape(3, 3)
        ext = _slice_cam(extrinsics, local, view, n_view)
        prot = np.asarray(_slice_cam(post_rots, local, view, n_view), dtype=np.float64).reshape(3, 3)
        ptra = np.asarray(_slice_cam(post_trans, local, view, n_view)).reshape(-1)
        t_cav2ego = pairwise[local] if local < len(pairwise) else np.eye(4)
        cam2ego, _rot = cam_to_ego_rt(ext, t_cav2ego, "rsu")
        q = ray_dir_cam(u_map[ys, xs], v_map[ys, xs], k, prot, ptra)
        mu_ego = transform_points(lift_cam_points(q, z_pred[flat, ys, xs]), cam2ego)
        dist = np.min(
            np.stack([obb_surface_distance(mu_ego, boxes[b]) for b in range(boxes.shape[0])], axis=0),
            axis=0,
        )
        surf[flat, ys, xs] = dist
        ab[flat, ys, xs] = dist <= NEAR_M
    return ab, surf


def lift_cam_points(q: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Optical-axis lift ``X_cam = z * q``."""
    return np.asarray(z, dtype=np.float64).reshape(-1, 1) * np.asarray(q, dtype=np.float64).reshape(-1, 3)


def rsu_shared_param_groups(core: torch.nn.Module) -> Dict[str, List[torch.nn.Parameter]]:
    """Trainable RSU shared parameters (not HeatmapHead / DepthHead)."""
    enc = core.frontend.encoders["rsu"]
    groups = {
        "trunk": [p for p in enc.trunk.parameters() if p.requires_grad],
        "up1": [p for p in enc.up1.parameters() if p.requires_grad],
        "up2": [p for p in enc.up2.parameters() if p.requires_grad],
        "concat128": [p for p in core.highres["rsu"].parameters() if p.requires_grad],
    }
    return groups


def enable_rsu_shared_grad(core: torch.nn.Module) -> None:
    """Turn requires_grad on RSU shared modules for diagnostic autograd."""
    enc = core.frontend.encoders["rsu"]
    for param in enc.trunk.parameters():
        param.requires_grad = True
    core.frontend._freeze_efficientnet_bn(enc.trunk)
    for name in ("up1", "up2"):
        for param in getattr(enc, name).parameters():
            param.requires_grad = True
    for param in core.highres["rsu"].parameters():
        param.requires_grad = True
    for param in core.heatmap_heads["rsu"].parameters():
        param.requires_grad = True
    for param in core.depth_heads["rsu"].parameters():
        param.requires_grad = True


def flatten_task_grad(
    loss: torch.Tensor, params: Sequence[torch.nn.Parameter], retain: bool
) -> torch.Tensor:
    """Concatenate autograd grads; unused params become zeros."""
    if loss is None or (not params):
        return torch.zeros(1, device=loss.device if loss is not None else "cpu")
    grads = torch.autograd.grad(
        loss, list(params), retain_graph=retain, allow_unused=True
    )
    chunks = []
    for param, grad in zip(params, grads):
        if grad is None:
            chunks.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            chunks.append(grad.reshape(-1))
    return torch.cat(chunks)


def cosine_and_norms(g_a: torch.Tensor, g_b: torch.Tensor) -> Tuple[float, float, float]:
    """Return cosine, ||a||, ||b||."""
    na = float(torch.linalg.vector_norm(g_a).item())
    nb = float(torch.linalg.vector_norm(g_b).item())
    dot = float(torch.dot(g_a.float(), g_b.float()).item())
    cos = dot / (na * nb + EPS)
    return cos, na, nb


def save_heatmap(path: Path, array: np.ndarray, title: str, cmap: str = "coolwarm") -> None:
    """Save a 90x160 diagnostic map."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    vmax = np.nanpercentile(np.abs(array[np.isfinite(array)]), 95) if np.isfinite(array).any() else 1.0
    vmax = max(float(vmax), 1.0e-3)
    im = ax.imshow(array, cmap=cmap, vmin=-vmax if cmap == "coolwarm" else 0.0, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_hist(path: Path, values: np.ndarray, title: str, xlabel: str) -> None:
    """Finite-value histogram."""
    finite = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    if finite.size:
        ax.hist(finite, bins=40, density=True, color="0.35")
        ax.axvline(0.0, color="C1", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_scatter_identity(path: Path, x: np.ndarray, y: np.ndarray, title: str) -> None:
    """z_pred vs z_gt with identity line."""
    finite = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.scatter(x[finite][:: max(1, finite.sum() // 4000)], y[finite][:: max(1, finite.sum() // 4000)], s=4, alpha=0.25)
    lo, hi = 2.0, 50.0
    ax.plot([lo, hi], [lo, hi], color="C1", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("z_gt (m)")
    ax.set_ylabel("z_pred (m)")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    """Run the read-only RSU residual and gradient audit."""
    opt = parse_args()
    out = Path(opt.out_root)
    for sub in (
        "residual_histograms",
        "residual_vs_depth",
        "residual_image_maps",
        "gradient_cosine",
        "gradient_spatial_maps",
        "feature_drift_plots",
    ):
        (out / sub).mkdir(parents=True, exist_ok=True)

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print("Creating model for architecture verification...", flush=True)
    model = train_utils.create_model(hypes)
    core = _unwrap_model(model)
    arch = verified_architecture(core, hypes)
    print(json.dumps(arch, indent=2), flush=True)
    (out / "architecture.json").write_text(json.dumps(arch, indent=2), encoding="utf-8")
    del model, core
    torch.cuda.empty_cache()

    datasets = {
        "train": load_split_dataset(hypes, "train"),
        "val": load_split_dataset(hypes, "val"),
        "test": load_split_dataset(hypes, "test"),
    }
    frames: Dict[str, List[int]] = {
        "train": collect_rsu_indices(datasets["train"], opt.n_train, opt.seed),
        "val": collect_rsu_indices(datasets["val"], opt.n_val, opt.seed + 1),
        "test": collect_rsu_indices(datasets["test"], opt.n_test, opt.seed + 2),
    }
    print("frame lists", {k: v for k, v in frames.items()}, flush=True)

    depth_focal = FocalLoss(alpha=0.25, gamma=2.0, reduction="none", smooth_target=False)
    heatmap_gamma = float(hypes.get("loss", {}).get("heatmap", {}).get("args", {}).get("gamma", 2.0))
    u_map, v_map = r90_pixel_centers()

    residual_csv_rows: List[Dict[str, Any]] = []
    residual_groups: Dict[Tuple[str, str, str], List[np.ndarray]] = defaultdict(list)
    # key: (ckpt, split, mask_name) -> list of e, abs, sigma, z_gt, z_pred, i, j
    image_acc: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    scene_stats: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    cam_stats: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    pred_fg_rates: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    box_mae: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    z_gt_all_train: List[np.ndarray] = []
    feature_bank: Dict[Tuple[str, int, str], np.ndarray] = {}
    grad_rows: List[Dict[str, Any]] = []
    identity: Dict[str, Any] = {}

    ckpt_specs: List[Tuple[str, str, int]] = [
        (f"joint_e{ep}", opt.joint_dir, int(ep)) for ep in JOINT_EPOCHS
    ]
    if opt.ft_dir:
        ckpt_specs.append((f"heatmap_ft_e{opt.ft_epoch}", opt.ft_dir, int(opt.ft_epoch)))

    def load_ckpt(label: str, model_dir: str, epoch: int) -> torch.nn.Module:
        local = train_utils.create_model(hypes)
        load_epoch_checkpoint(local, model_dir, epoch)
        local.to(device)
        local.eval()
        print(f"loaded {label}", flush=True)
        return local

    # ---- residuals + drift features ----
    for label, model_dir, epoch in ckpt_specs:
        net = load_ckpt(label, model_dir, epoch)
        core = _unwrap_model(net)
        with torch.no_grad():
            for split, indices in frames.items():
                for idx in indices:
                    sample = datasets[split][int(idx)]
                    batch = datasets[split].collate_batch_test([sample])
                    batch = train_utils.to_device(batch, device)
                    ego = batch["ego"]
                    if "rsu" not in present_camera_agents(ego):
                        continue
                    pred_all = net(ego)
                    pred = pred_all["rsu"]
                    cam = ego["rsu"]["batch_merged_cam_inputs"]
                    imgs = flatten_imgs(cam["imgs"])
                    z_pred = pred["depth_z_mean"].detach().cpu().numpy()
                    z_var = np.clip(pred["depth_z_var"].detach().cpu().numpy(), 0.0, None)
                    sigma = np.sqrt(z_var)
                    z_gt = extract_camera_z_gt(imgs).detach().cpu().numpy()
                    valid = depth_valid_mask(
                        torch.from_numpy(z_gt),
                        float(core.frontend.encoders["rsu"].d_min),
                        float(core.frontend.encoders["rsu"].d_max),
                    ).numpy()
                    p_fg = torch.softmax(pred["heatmap_logits"], dim=1)[:, 1].detach().cpu().numpy()
                    sam3 = None
                    try:
                        sam3 = build_semantic_target(cam, tau=1).detach().cpu().numpy() > 0
                    except (KeyError, ValueError):
                        sam3 = np.zeros_like(valid)
                    boxes, _ids = gt_boxes_from_ego(ego)
                    image_hw = (int(imgs.shape[-2]), int(imgs.shape[-1]))
                    in_box, a_seed = box_seed_masks(ego, p_fg, z_pred, boxes, image_hw)
                    ab_seed, surf_dist = a_plus_b_mask(ego, a_seed, z_pred, boxes)
                    e = z_pred - z_gt
                    abs_e = np.abs(e)
                    masks = {
                        "all_valid": valid,
                        "sam3_fg": valid & sam3,
                        "box": valid & in_box,
                        "a_seed": valid & a_seed,
                        "ab_seed": valid & ab_seed,
                    }
                    meta = ego.get("metadata_path_list", [""])[0]
                    scene = scene_from_path(meta)
                    weather = weather_tag(scene)
                    n_cav = _record_len(ego, "rsu")
                    n_view = n_views_of(imgs, n_cav, "rsu")
                    n_flat = int(z_pred.shape[0])
                    pred_fg_rates[(label, split)].append(float((p_fg >= FG_TAU).mean()))
                    if in_box.any() and valid.any():
                        box_mae[(label, split)].append(
                            float(np.median(abs_e[valid & in_box]))
                            if (valid & in_box).any()
                            else float("nan")
                        )
                    if split == "train":
                        z_gt_all_train.append(z_gt[valid])

                    acc_key = (label, split)
                    if acc_key not in image_acc:
                        image_acc[acc_key] = {
                            "sum_e": np.zeros((FEAT_H, FEAT_W), dtype=np.float64),
                            "sum_abs": np.zeros((FEAT_H, FEAT_W), dtype=np.float64),
                            "sum_sig": np.zeros((FEAT_H, FEAT_W), dtype=np.float64),
                            "count": np.zeros((FEAT_H, FEAT_W), dtype=np.float64),
                        }
                    acc = image_acc[acc_key]
                    acc["sum_e"] += np.nansum(np.where(valid, e, 0.0), axis=0)
                    acc["sum_abs"] += np.nansum(np.where(valid, abs_e, 0.0), axis=0)
                    acc["sum_sig"] += np.nansum(np.where(valid, sigma, 0.0), axis=0)
                    acc["count"] += valid.sum(axis=0).astype(np.float64)

                    for mask_name, mask in masks.items():
                        if not mask.any():
                            continue
                        residual_groups[(label, split, mask_name)].append(
                            np.stack(
                                [e[mask], abs_e[mask], sigma[mask], z_gt[mask], z_pred[mask]],
                                axis=1,
                            )
                        )
                    cap = max(1, int(opt.csv_cap // max(len(indices) * max(n_flat, 1), 1)))
                    rng = np.random.RandomState(opt.seed + idx)
                    for flat in range(n_flat):
                        ys, xs = np.where(valid[flat])
                        if ys.size == 0:
                            continue
                        take = min(cap, int(ys.size))
                        sel = rng.choice(ys.size, size=take, replace=False)
                        local = flat // n_view
                        view = flat % n_view
                        cam_name = RSU_CAM_KEYS[view % len(RSU_CAM_KEYS)]
                        for k in sel:
                            i, j = int(ys[k]), int(xs[k])
                            residual_csv_rows.append(
                                {
                                    "ckpt": label,
                                    "split": split,
                                    "scene": scene,
                                    "weather": weather,
                                    "idx": int(idx),
                                    "cam": cam_name,
                                    "i": i,
                                    "j": j,
                                    "u": float(u_map[i, j]),
                                    "v": float(v_map[i, j]),
                                    "z_pred": float(z_pred[flat, i, j]),
                                    "z_gt": float(z_gt[flat, i, j]),
                                    "e_z": float(e[flat, i, j]),
                                    "abs_e_z": float(abs_e[flat, i, j]),
                                    "sigma_z": float(sigma[flat, i, j]),
                                    "sam3": bool(sam3[flat, i, j]),
                                    "in_box": bool(in_box[flat, i, j]),
                                    "a_seed": bool(a_seed[flat, i, j]),
                                    "ab_seed": bool(ab_seed[flat, i, j]),
                                    "surf_dist": (
                                        float(surf_dist[flat, i, j])
                                        if np.isfinite(surf_dist[flat, i, j])
                                        else ""
                                    ),
                                }
                            )
                            scene_stats[(label, split, scene)].append(float(e[flat, i, j]))
                            cam_stats[(label, split, cam_name)].append(float(e[flat, i, j]))

                    if (
                        split in ("val", "test")
                        and idx in frames[split][: opt.n_drift]
                        and "joint_" in label
                    ):
                        f90 = core.highres["rsu"](
                            *core.frontend.extract_backbone_features("rsu", cam["imgs"])
                        )
                        feat = f90.detach().float().cpu().numpy()
                        fg = sam3 | in_box
                        bg = ~fg
                        for tag, m in (("object", fg), ("background", bg)):
                            if not m.any():
                                continue
                            vecs = feat.transpose(0, 2, 3, 1)[m]
                            if vecs.shape[0] > 256:
                                vecs = vecs[rng.choice(vecs.shape[0], 256, replace=False)]
                            feature_bank[(label, split, int(idx), tag)] = vecs.mean(axis=0)

        del net
        torch.cuda.empty_cache()

    train_z = (
        np.concatenate(z_gt_all_train) if z_gt_all_train else np.array([10.0])
    )
    depth_edges = choose_depth_bins(train_z)
    print("depth bins", depth_edges, "train z percentiles",
          np.percentile(train_z, [10, 25, 50, 75, 90]).round(2), flush=True)

    # summaries
    summaries: Dict[str, Any] = {}
    affine_pred_from_gt: Dict[str, Any] = {}
    affine_e_from_gt: Dict[str, Any] = {}
    depth_bin_tables: Dict[str, Any] = {}
    for key, chunks in residual_groups.items():
        label, split, mask_name = key
        arr = np.concatenate(chunks, axis=0)
        e, abs_e, sigma, z_gt, z_pred = arr.T
        summaries[f"{label}/{split}/{mask_name}"] = residual_summary(e, abs_e, sigma)
        if mask_name == "all_valid":
            affine_pred_from_gt[f"{label}/{split}"] = fit_affine(z_gt, z_pred)
            affine_e_from_gt[f"{label}/{split}"] = fit_affine(z_gt, e)
            rows = []
            for lo, hi in zip(depth_edges[:-1], depth_edges[1:]):
                sel = (z_gt >= lo) & (z_gt < hi)
                if not sel.any():
                    continue
                rows.append(
                    {
                        "lo": lo,
                        "hi": hi,
                        "n": int(sel.sum()),
                        "mean_e": round(float(e[sel].mean()), 4),
                        "median_e": round(float(np.median(e[sel])), 4),
                        "median_abs": round(float(np.median(abs_e[sel])), 4),
                        "median_sigma": round(float(np.median(sigma[sel])), 4),
                        "median_abs_over_sigma": round(
                            float(np.median(abs_e[sel] / np.clip(sigma[sel], EPS, None))),
                            4,
                        ),
                    }
                )
            depth_bin_tables[f"{label}/{split}"] = rows

    # plots for joint_e16
    for split in ("train", "val", "test"):
        key = ("joint_e16", split, "all_valid")
        if key not in residual_groups:
            continue
        arr = np.concatenate(residual_groups[key], axis=0)
        e, abs_e, sigma, z_gt, z_pred = arr.T
        save_hist(out / "residual_histograms" / f"e16_{split}_ez.png", e, f"e16 {split} e_z", "e_z (m)")
        save_hist(out / "residual_histograms" / f"e16_{split}_abs.png", abs_e, f"e16 {split} |e_z|", "|e_z| (m)")
        save_scatter_identity(out / "residual_vs_depth" / f"e16_{split}_pred_vs_gt.png", z_gt, z_pred, f"e16 {split}")
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        edges = depth_edges
        meds = []
        cents = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (z_gt >= lo) & (z_gt < hi)
            cents.append(0.5 * (lo + hi))
            meds.append(float(np.median(e[sel])) if sel.any() else np.nan)
        ax.plot(cents, meds, marker="o")
        ax.axhline(0.0, color="0.5", linewidth=1.0)
        ax.set_xlabel("z_gt bin center (m)")
        ax.set_ylabel("median e_z (m)")
        ax.set_title(f"e16 {split} median e_z vs depth")
        fig.tight_layout()
        fig.savefig(out / "residual_vs_depth" / f"e16_{split}_median_e_by_z.png", dpi=120)
        plt.close(fig)
        acc = image_acc.get(("joint_e16", split))
        if acc is not None:
            cnt = np.clip(acc["count"], 1.0, None)
            save_heatmap(out / "residual_image_maps" / f"e16_{split}_mean_e.png", acc["sum_e"] / cnt, f"e16 {split} mean e_z")
            save_heatmap(out / "residual_image_maps" / f"e16_{split}_mean_abs.png", acc["sum_abs"] / cnt, f"e16 {split} mean |e_z|", cmap="magma")
            save_heatmap(out / "residual_image_maps" / f"e16_{split}_sigma.png", acc["sum_sig"] / cnt, f"e16 {split} mean sigma_z", cmap="magma")
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
        rows = residual_csv_rows
        e16_rows = [r for r in rows if r["ckpt"] == "joint_e16" and r["split"] == split]
        if e16_rows:
            ii = np.array([r["i"] for r in e16_rows], dtype=np.float64)
            jj = np.array([r["j"] for r in e16_rows], dtype=np.float64)
            ee = np.array([r["e_z"] for r in e16_rows], dtype=np.float64)
            axes[0].scatter(ii, ee, s=3, alpha=0.15)
            axes[0].axhline(0.0, color="C1", linewidth=1.0)
            axes[0].set_xlabel("R90 row i")
            axes[0].set_ylabel("e_z (m)")
            axes[1].scatter(jj, ee, s=3, alpha=0.15)
            axes[1].axhline(0.0, color="C1", linewidth=1.0)
            axes[1].set_xlabel("R90 col j")
            fig.suptitle(f"e16 {split} e_z vs image position")
            fig.tight_layout()
            fig.savefig(out / "residual_image_maps" / f"e16_{split}_e_vs_ij.png", dpi=120)
        plt.close(fig)

    # TRAIN-only affine on e16, apply val/test
    calib: Dict[str, Any] = {}
    train_key = ("joint_e16", "train", "all_valid")
    if train_key in residual_groups:
        arr = np.concatenate(residual_groups[train_key], axis=0)
        z_gt_t, z_pred_t = arr[:, 3], arr[:, 4]
        aff = fit_affine(z_pred_t, z_gt_t)  # z_gt ≈ a * z_pred + b  → z_corr = a z_pred + b
        offset = float(np.median(z_gt_t - z_pred_t))
        calib["train_affine_zgt_from_zpred"] = aff
        calib["train_constant_offset"] = round(offset, 4)
        for split in ("val", "test"):
            k = ("joint_e16", split, "all_valid")
            if k not in residual_groups:
                continue
            a2 = np.concatenate(residual_groups[k], axis=0)
            z_gt_s, z_pred_s = a2[:, 3], a2[:, 4]
            z_aff = aff["a"] * z_pred_s + aff["b"]
            z_off = z_pred_s + offset
            def pack_pair(z_c: np.ndarray, z_g: np.ndarray) -> Dict[str, float]:
                err = z_c - z_g
                return {
                    "mae": round(float(np.mean(np.abs(err))), 4),
                    "median_abs": round(float(np.median(np.abs(err))), 4),
                    "median_signed": round(float(np.median(err)), 4),
                }
            calib[split] = {
                "raw": pack_pair(z_pred_s, z_gt_s),
                "affine": pack_pair(z_aff, z_gt_s),
                "offset": pack_pair(z_off, z_gt_s),
            }
            for mask_name in ("box", "a_seed", "ab_seed"):
                mk = ("joint_e16", split, mask_name)
                if mk not in residual_groups:
                    continue
                am = np.concatenate(residual_groups[mk], axis=0)
                z_gt_m, z_pred_m = am[:, 3], am[:, 4]
                calib[split][mask_name] = {
                    "raw": pack_pair(z_pred_m, z_gt_m),
                    "offset": pack_pair(z_pred_m + offset, z_gt_m),
                    "affine": pack_pair(aff["a"] * z_pred_m + aff["b"], z_gt_m),
                }

    if residual_csv_rows:
        with (out / "depth_residuals.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(residual_csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(residual_csv_rows)
        print(f"wrote residual csv n={len(residual_csv_rows)}", flush=True)
    (out / "residual_partial.json").write_text(
        json.dumps(
            {
                "architecture": arch,
                "n_frames": {k: len(v) for k, v in frames.items()},
                "frame_indices": frames,
                "depth_bins": depth_edges,
                "residual_summaries": summaries,
                "z_pred_vs_z_gt_affine": affine_pred_from_gt,
                "e_z_vs_z_gt_affine": affine_e_from_gt,
                "depth_bin_tables": depth_bin_tables,
                "train_only_calibration": calib,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---- gradients on TRAIN (joint epochs) ----
    for label, model_dir, epoch in ckpt_specs:
        if "heatmap_ft" in label:
            continue
        net = load_ckpt(label, model_dir, epoch)
        core = _unwrap_model(net)
        enable_rsu_shared_grad(core)
        net.train()
        core.frontend.apply_train_eval_state(True)
        groups = rsu_shared_param_groups(core)
        grad_indices = frames["train"][: opt.n_grad]
        spatial_saved = 0
        for idx in grad_indices:
            sample = datasets["train"][int(idx)]
            batch = datasets["train"].collate_batch_test([sample])
            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            if "rsu" not in present_camera_agents(ego):
                continue
            cam = ego["rsu"]["batch_merged_cam_inputs"]
            r2, f45 = core.frontend.extract_backbone_features("rsu", cam["imgs"])
            f90 = core.highres["rsu"](r2, f45)
            f90.retain_grad()
            hm_logits = core.heatmap_heads["rsu"](f90)
            depth_logits = core.depth_heads["rsu"](f90)
            pred = {
                "rsu": {
                    "heatmap_logits": hm_logits,
                    "depth_logits": depth_logits,
                    "depth_z_mean": core.depth_moments["rsu"](depth_logits)[0],
                }
            }
            try:
                sem = {"rsu": build_semantic_target(cam, tau=1).to(device=device, dtype=torch.long)}
            except (KeyError, ValueError):
                print(f"  skip grad idx={idx}: no SAM3", flush=True)
                continue
            if int(sem["rsu"].gt(0).sum().item()) == 0:
                print(f"  skip grad idx={idx}: SAM3 empty", flush=True)
                continue
            depth_tgt = build_depth_targets(ego, pred, core)
            valid = build_depth_valid_masks(ego, pred, core)
            l_hm = softmax_focal_loss(hm_logits, sem["rsu"], gamma=heatmap_gamma)
            l_d = depth_focal(depth_logits, depth_tgt["rsu"])
            dmask = valid["rsu"] & sem["rsu"].ne(0)
            if int(dmask.sum().item()) == 0:
                print(f"  skip grad idx={idx}: no RSU depth FG", flush=True)
                continue
            l_d = l_d[dmask].mean()
            g_hm_f90 = torch.autograd.grad(l_hm, f90, retain_graph=True)[0]
            g_d_f90 = torch.autograd.grad(l_d, f90, retain_graph=True)[0]
            ordered: List[torch.nn.Parameter] = []
            spans: List[Tuple[str, int, int]] = []
            for gname, params in groups.items():
                if not params:
                    continue
                start = sum(p.numel() for p in ordered)
                ordered.extend(params)
                end = sum(p.numel() for p in ordered)
                spans.append((gname, start, end))
            g_hm_all = flatten_task_grad(l_hm, ordered, retain=True)
            g_d_all = flatten_task_grad(l_d, ordered, retain=False)
            cos_f90, n_hm_f90, n_d_f90 = cosine_and_norms(g_hm_f90.reshape(-1), g_d_f90.reshape(-1))
            fg = sem["rsu"].gt(0)
            bg = ~fg
            def spatial_norm(g: torch.Tensor) -> torch.Tensor:
                return torch.linalg.vector_norm(g, dim=1)

            nh = spatial_norm(g_hm_f90)
            nd = spatial_norm(g_d_f90)
            hm_fg = float(nh[fg].mean().item()) if bool(fg.any()) else float("nan")
            hm_bg = float(nh[bg].mean().item()) if bool(bg.any()) else float("nan")
            d_fg = float(nd[fg].mean().item()) if bool(fg.any()) else float("nan")
            d_bg = float(nd[bg].mean().item()) if bool(bg.any()) else float("nan")
            def subset_cos(mask: torch.Tensor) -> float:
                if not bool(mask.any()):
                    return float("nan")
                ga = g_hm_f90.permute(0, 2, 3, 1)[mask].reshape(-1)
                gb = g_d_f90.permute(0, 2, 3, 1)[mask].reshape(-1)
                return cosine_and_norms(ga, gb)[0]

            cos_fg = subset_cos(fg)
            cos_bg = subset_cos(bg)
            row_base = {
                "ckpt": label,
                "idx": int(idx),
                "cosine_F90": round(cos_f90, 4),
                "norm_hm_F90": round(n_hm_f90, 4),
                "norm_depth_F90": round(n_d_f90, 4),
                "ratio_F90": round(n_d_f90 / (n_hm_f90 + EPS), 4),
                "hm_grad_fg": round(hm_fg, 5),
                "hm_grad_bg": round(hm_bg, 5),
                "depth_grad_fg": round(d_fg, 5),
                "depth_grad_bg": round(d_bg, 5),
                "cosine_F90_fg": round(cos_fg, 4) if cos_fg == cos_fg else None,
                "cosine_F90_bg": round(cos_bg, 4) if cos_bg == cos_bg else None,
            }
            for gname, start, end in spans:
                g_hm = g_hm_all[start:end]
                g_d = g_d_all[start:end]
                cos, nhm, ndp = cosine_and_norms(g_hm, g_d)
                grad_rows.append(
                    {
                        **row_base,
                        "module": gname,
                        "cosine": round(cos, 4),
                        "norm_hm": round(nhm, 6),
                        "norm_depth": round(ndp, 6),
                        "ratio": round(ndp / (nhm + EPS), 4),
                    }
                )
            grad_rows.append(
                {
                    **row_base,
                    "module": "F90",
                    "cosine": round(cos_f90, 4),
                    "norm_hm": round(n_hm_f90, 6),
                    "norm_depth": round(n_d_f90, 6),
                    "ratio": round(n_d_f90 / (n_hm_f90 + EPS), 4),
                }
            )
            if spatial_saved < 2 and label == "joint_e16":
                save_heatmap(
                    out / "gradient_spatial_maps" / f"e16_idx{idx}_hm.png",
                    nh.mean(0).detach().cpu().numpy(),
                    f"||dL_hm / dF90|| idx={idx}",
                    cmap="magma",
                )
                save_heatmap(
                    out / "gradient_spatial_maps" / f"e16_idx{idx}_depth.png",
                    nd.mean(0).detach().cpu().numpy(),
                    f"||dL_depth / dF90|| idx={idx}",
                    cmap="magma",
                )
                spatial_saved += 1
            net.zero_grad(set_to_none=True)
            print(f"  grad {label} idx={idx} cosF90={cos_f90:.3f} ratio={n_d_f90/(n_hm_f90+EPS):.2f}", flush=True)
        del net
        torch.cuda.empty_cache()

    # gradient aggregates
    grad_agg: Dict[str, Any] = {}
    by = defaultdict(list)
    for row in grad_rows:
        by[(row["ckpt"], row["module"])].append(row)
    for (ckpt, module), rows in sorted(by.items()):
        coss = np.array([r["cosine"] for r in rows], dtype=np.float64)
        grad_agg[f"{ckpt}/{module}"] = {
            "n": len(rows),
            "mean_cos": round(float(coss.mean()), 4),
            "median_cos": round(float(np.median(coss)), 4),
            **percentile_dict(coss, (10, 25, 75, 90)),
            "frac_cos_lt0": round(float((coss < 0).mean()), 4),
            "median_norm_hm": round(float(np.median([r["norm_hm"] for r in rows])), 6),
            "median_norm_depth": round(float(np.median([r["norm_depth"] for r in rows])), 6),
            "median_ratio": round(float(np.median([r["ratio"] for r in rows])), 4),
            "median_cos_F90": round(float(np.median([r["cosine_F90"] for r in rows])), 4),
            "median_hm_fg": round(float(np.nanmedian([r["hm_grad_fg"] for r in rows])), 5),
            "median_hm_bg": round(float(np.nanmedian([r["hm_grad_bg"] for r in rows])), 5),
            "median_d_fg": round(float(np.nanmedian([r["depth_grad_fg"] for r in rows])), 5),
            "median_d_bg": round(float(np.nanmedian([r["depth_grad_bg"] for r in rows])), 5),
        }

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    modules = ["trunk", "up1", "up2", "concat128"]
    for module in modules:
        xs, ys = [], []
        for ep in JOINT_EPOCHS:
            key = f"joint_e{ep}/{module}"
            if key in grad_agg:
                xs.append(ep)
                ys.append(grad_agg[key]["median_cos"])
        if xs:
            ax.plot(xs, ys, marker="o", label=module)
    ax.axhline(0.0, color="0.5", linewidth=1.0)
    ax.set_xlabel("joint epoch")
    ax.set_ylabel("median cosine")
    ax.set_title("RSU shared-module gradient cosine (heatmap vs depth)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "gradient_cosine" / "median_cosine_by_epoch.png", dpi=120)
    plt.close(fig)

    # feature drift
    drift_rows: List[Dict[str, Any]] = []
    pairs = (("joint_e1", "joint_e4"), ("joint_e4", "joint_e9"), ("joint_e9", "joint_e16"), ("joint_e1", "joint_e16"))
    for a, b in pairs:
        for tag in ("object", "background"):
            coss, l2s = [], []
            for split in ("val", "test"):
                for idx in frames[split][: opt.n_drift]:
                    ka, kb = (a, split, int(idx), tag), (b, split, int(idx), tag)
                    if ka not in feature_bank or kb not in feature_bank:
                        continue
                    va, vb = feature_bank[ka], feature_bank[kb]
                    coss.append(float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + EPS)))
                    l2s.append(float(np.linalg.norm(va - vb)))
            if not coss:
                continue
            drift_rows.append(
                {
                    "pair": f"{a}->{b}",
                    "region": tag,
                    "mean_cosine": round(float(np.mean(coss)), 4),
                    "mean_l2": round(float(np.mean(l2s)), 4),
                    "n": len(coss),
                }
            )

    if drift_rows:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        labels = [f"{r['pair']}\n{r['region']}" for r in drift_rows]
        ax.bar(range(len(drift_rows)), [r["mean_l2"] for r in drift_rows], color="0.4")
        ax.set_xticks(range(len(drift_rows)))
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("mean L2 F90 drift")
        ax.set_title("Object vs background F90 drift")
        fig.tight_layout()
        fig.savefig(out / "feature_drift_plots" / "f90_l2_drift.png", dpi=120)
        plt.close(fig)

    # heatmap-ft vs joint e16 identity
    if opt.ft_dir:
        joint = load_ckpt("joint_e16", opt.joint_dir, 16)
        ft = load_ckpt(f"heatmap_ft_e{opt.ft_epoch}", opt.ft_dir, opt.ft_epoch)
        c_j, c_f = _unwrap_model(joint), _unwrap_model(ft)
        diffs = []
        hm_changed = []
        with torch.no_grad():
            for idx in frames["test"][:2]:
                sample = datasets["test"][int(idx)]
                batch = datasets["test"].collate_batch_test([sample])
                batch = train_utils.to_device(batch, device)
                ego = batch["ego"]
                if "rsu" not in present_camera_agents(ego):
                    continue
                cam = ego["rsu"]["batch_merged_cam_inputs"]
                f90_j = c_j.highres["rsu"](*c_j.frontend.extract_backbone_features("rsu", cam["imgs"]))
                f90_f = c_f.highres["rsu"](*c_f.frontend.extract_backbone_features("rsu", cam["imgs"]))
                d_j = c_j.depth_heads["rsu"](f90_j)
                d_f = c_f.depth_heads["rsu"](f90_f)
                z_j, _ = c_j.depth_moments["rsu"](d_j)
                z_f, _ = c_f.depth_moments["rsu"](d_f)
                h_j = c_j.heatmap_heads["rsu"](f90_j)
                h_f = c_f.heatmap_heads["rsu"](f90_f)
                diffs.append(
                    {
                        "idx": int(idx),
                        "max_abs_F90": float((f90_j - f90_f).abs().max().item()),
                        "max_abs_depth_logits": float((d_j - d_f).abs().max().item()),
                        "max_abs_z_mean": float((z_j - z_f).abs().max().item()),
                        "max_abs_heatmap_logits": float((h_j - h_f).abs().max().item()),
                    }
                )
        identity = {
            "source": "concat128 net_epoch16.pth",
            "ft": f"heatmap_ft net_epoch{opt.ft_epoch}.pth",
            "frames": diffs,
            "max_abs_F90": max((d["max_abs_F90"] for d in diffs), default=None),
            "max_abs_depth_logits": max((d["max_abs_depth_logits"] for d in diffs), default=None),
            "max_abs_z_mean": max((d["max_abs_z_mean"] for d in diffs), default=None),
            "heatmap_logits_changed": max((d["max_abs_heatmap_logits"] for d in diffs), default=None),
        }
        del joint, ft
        torch.cuda.empty_cache()

    # write csvs
    if residual_csv_rows:
        with (out / "depth_residuals.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(residual_csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(residual_csv_rows)
    if grad_rows:
        with (out / "gradient_stats.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(grad_rows[0].keys()))
            writer.writeheader()
            writer.writerows(grad_rows)
    if drift_rows:
        with (out / "feature_drift.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(drift_rows[0].keys()))
            writer.writeheader()
            writer.writerows(drift_rows)

    cam_summary = {
        f"{a}/{b}/{c}": residual_summary(np.array(v), np.abs(np.array(v)), np.ones_like(v, dtype=np.float64))
        for (a, b, c), v in cam_stats.items()
        if v
    }
    report = {
        "architecture": arch,
        "checkpoints": [s[0] for s in ckpt_specs],
        "n_frames": {k: len(v) for k, v in frames.items()},
        "frame_indices": frames,
        "depth_bins": depth_edges,
        "train_z_gt_percentiles": np.percentile(train_z, [10, 25, 50, 75, 90, 95]).round(3).tolist(),
        "residual_summaries": summaries,
        "z_pred_vs_z_gt_affine": affine_pred_from_gt,
        "e_z_vs_z_gt_affine": affine_e_from_gt,
        "depth_bin_tables": depth_bin_tables,
        "train_only_calibration": calib,
        "gradient_aggregates": grad_agg,
        "feature_drift": drift_rows,
        "heatmap_ft_identity": identity,
        "pred_fg_rate": {f"{a}/{b}": round(float(np.mean(v)), 4) for (a, b), v in pred_fg_rates.items()},
        "box_median_abs": {f"{a}/{b}": round(float(np.nanmedian(v)), 4) for (a, b), v in box_mae.items() if v},
        "camera_signed": {
            k: {"median_e": v.get("median_e"), "n": v.get("n")} for k, v in cam_summary.items()
        },
        "scene_signed": {
            f"{a}/{b}/{c}": {
                "n": len(v),
                "median_e": round(float(np.median(v)), 4),
                "mean_e": round(float(np.mean(v)), 4),
            }
            for (a, b, c), v in scene_stats.items()
            if v
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def grab(ckpt: str, split: str, mask: str, field: str) -> str:
        block = summaries.get(f"{ckpt}/{split}/{mask}") or {}
        val = block.get(field)
        return "n/a" if val is None else str(val)

    lines = [
        "RSU depth residual + shared-F90 gradient audit (read-only)",
        f"joint={opt.joint_dir}  ft={opt.ft_dir or 'none'}",
        f"frames train={frames['train']} val={frames['val']} test={frames['test']}",
        f"RSU LID ddiscr={arch['rsu_ddiscr']} mode={arch['rsu_mode']}",
        "cam2lidar identity (no RSU invert)",
        "",
        "=== architecture ===",
        json.dumps({k: arch[k] for k in ("rsu_encoder", "efficientnet", "up1", "up2", "concat128", "heatmap_head", "depth_head", "heatmap_loss", "depth_loss", "cam2lidar")}, indent=2),
        "",
        "=== signed residual joint_e16 all_valid ===",
    ]
    for split in ("train", "val", "test"):
        lines.append(
            f"  {split}: n={grab('joint_e16', split, 'all_valid', 'n')}  "
            f"median_e={grab('joint_e16', split, 'all_valid', 'median_e')}  "
            f"mean_e={grab('joint_e16', split, 'all_valid', 'mean_e')}  "
            f"median_abs={grab('joint_e16', split, 'all_valid', 'median_abs')}  "
            f"frac>0={grab('joint_e16', split, 'all_valid', 'frac_e_gt0')}  "
            f"|e|/σ med={grab('joint_e16', split, 'all_valid', 'median_abs_over_sigma')}"
        )
    lines.append("=== object-support e16 test (all_valid / box / A / A+B) ===")
    lines.append(
        f"  all_valid median_e={grab('joint_e16', 'test', 'all_valid', 'median_e')}  "
        f"box={grab('joint_e16', 'test', 'box', 'median_e')}  "
        f"a_seed={grab('joint_e16', 'test', 'a_seed', 'median_e')}  "
        f"ab_seed={grab('joint_e16', 'test', 'ab_seed', 'median_e')}  "
        f"sam3={grab('joint_e16', 'test', 'sam3_fg', 'median_e')}"
    )
    lines.append(
        f"  all_valid median_abs={grab('joint_e16', 'test', 'all_valid', 'median_abs')}  "
        f"box={grab('joint_e16', 'test', 'box', 'median_abs')}  "
        f"a_seed={grab('joint_e16', 'test', 'a_seed', 'median_abs')}  "
        f"ab_seed={grab('joint_e16', 'test', 'ab_seed', 'median_abs')}"
    )
    lines.append("=== TRAIN-only calib applied to val/test ===")
    lines.append(json.dumps(calib, indent=2))
    lines.append("=== gradient aggregates ===")
    for key, block in sorted(grad_agg.items()):
        lines.append(
            f"  {key}: cos_med={block['median_cos']} P(cos<0)={block['frac_cos_lt0']}  "
            f"||g_hm||={block['median_norm_hm']} ||g_d||={block['median_norm_depth']} "
            f"ratio={block['median_ratio']}"
        )
    lines.append("=== heatmap-ft identity vs joint e16 ===")
    lines.append(json.dumps(identity, indent=2))
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

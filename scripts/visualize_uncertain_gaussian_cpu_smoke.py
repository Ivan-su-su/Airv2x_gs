#!/usr/bin/env python3
"""CPU smoke + visualization: synthetic voxel features / lidar mask, no backbone3d, no CUDA LiDAR projector.

This script **does not** call ``ImageConditionGaussianGenerator`` (which uses ``LidarToImageProjector``).
It injects ``gaussian_candidates`` built from synthetic ``voxel_features`` and optional ``lidar_mask``,
runs ``FirstRoundGaussianGenerator`` + ``IntraAgentGaussianRefiner`` on CPU, and saves a PNG.

Usage (from repo root)::

    conda activate airv2x
    python scripts/visualize_uncertain_gaussian_cpu_smoke.py --out /tmp/gaussian_vis.png
    python scripts/visualize_uncertain_gaussian_cpu_smoke.py --no-figure --stats-out /tmp/gaussian_stats.txt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Repo root on sys.path when executed as ``python scripts/...``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from opencood.models.uncertain_gaussian_modules.gaussian_refine.first_round_gaussian_generator import (
    FirstRoundGaussianGenerator,
)
from opencood.models.uncertain_gaussian_modules.gaussian_refine.intra_agent_gaussian_refiner import (
    IntraAgentGaussianRefiner,
)


def _build_lidar2image_and_wh(
    imgs: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    post_rots: torch.Tensor,
    post_trans: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirror ``GaussianToImageProjector._build_projection_and_image_wh`` for plotting."""
    batch_size, num_views = imgs.shape[:2]
    device = imgs.device
    dtype = intrinsics.dtype
    intrinsics_4 = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(
        batch_size, num_views, 1, 1
    )
    intrinsics_4[..., :3, :3] = intrinsics
    post_rots_4 = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(
        batch_size, num_views, 1, 1
    )
    post_rots_4[..., :3, :3] = post_rots
    img_aug_matrix = post_rots_4.clone()
    img_aug_matrix[..., :3, 3] = post_trans[..., :3]
    lidar2image = torch.matmul(
        img_aug_matrix, torch.matmul(intrinsics_4, torch.inverse(extrinsics))
    )
    image_wh = torch.tensor(
        [imgs.shape[-1], imgs.shape[-2]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 2).repeat(batch_size, num_views, 1)
    return lidar2image, image_wh


def project_mean_to_normalized(
    lidar2image: torch.Tensor,
    image_wh: torch.Tensor,
    mean_xyz: torch.Tensor,
    batch_idx: int,
    view_idx: int,
) -> torch.Tensor:
    """Project one 3D point ``[3]`` to normalized ``[2]`` for ``(batch_idx, view_idx)``."""
    li = lidar2image[batch_idx, view_idx]
    wh = image_wh[batch_idx, view_idx]
    ph = torch.cat([mean_xyz, mean_xyz.new_ones(1)], dim=-1)
    proj = li @ ph
    z = proj[2].clamp(min=1e-5)
    uv = proj[:2] / z
    normalized = uv / wh
    if not (
        float(proj[2]) > 1e-5
        and 0 < float(normalized[0]) < 1
        and 0 < float(normalized[1]) < 1
    ):
        return normalized.new_full((2,), float("nan"))
    return normalized


def build_camera_batch(
    b: int,
    v: int,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Synthetic ``batch_merged_cam_inputs`` (identity extrinsics: lidar == camera frame)."""
    imgs = torch.rand(b, v, 3, h, w, device=device, dtype=dtype)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    fx = fy = max(h, w) * 0.8
    intrinsics = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).repeat(b, v, 1, 1)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = cx
    intrinsics[..., 1, 2] = cy
    extrinsics = (
        torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(b, v, 1, 1)
    )
    post_rots = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).repeat(b, v, 1, 1)
    post_trans = torch.zeros(b, v, 3, device=device, dtype=dtype)
    return {
        "imgs": imgs,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "post_rots": post_rots,
        "post_trans": post_trans,
    }


def build_synthetic_batch_dict(
    device: torch.device,
    dtype: torch.dtype,
    num_voxels: int,
    voxel_channels: int,
    image_channels: int,
    b: int,
    v: int,
    h: int,
    w: int,
    fh: int,
    fw: int,
) -> Dict:
    """Full ``batch_dict`` with synthetic voxel/image tensors and injected candidates."""
    torch.manual_seed(0)
    voxel_features = torch.randn(num_voxels, voxel_channels, device=device, dtype=dtype)
    # ``lidar_mask``: 3D occupancy-style grid (not consumed by this CPU path, but stored for realism).
    grid_z, grid_y, grid_x = 8, 16, 16
    lidar_mask = torch.zeros(b, grid_z, grid_y, grid_x, dtype=torch.bool, device=device)
    lidar_mask[:, :, grid_y // 4 : 3 * grid_y // 4, grid_x // 4 : 3 * grid_x // 4] = True

    cam = build_camera_batch(b, v, h, w, device, dtype)
    image_feature = torch.randn(b, v, image_channels, fh, fw, device=device, dtype=dtype)

    # 5 fused groups: 0–2 on batch agent 0 (2 views each); 3–4 on batch agent 1.
    group_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], device=device, dtype=torch.long)
    local_agent_ids = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], device=device, dtype=torch.long)
    view_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], device=device, dtype=torch.long)
    base = torch.tensor(
        [
            [0.15, -0.08, 14.0],
            [0.15, -0.08, 14.0],
            [-0.35, 0.05, 16.0],
            [-0.35, 0.05, 16.0],
            [0.45, 0.12, 13.0],
            [0.45, 0.12, 13.0],
            [-0.2, -0.1, 15.0],
            [-0.2, -0.1, 15.0],
            [0.25, 0.2, 12.5],
            [0.25, 0.2, 12.5],
        ],
        device=device,
        dtype=dtype,
    )
    mean = base + torch.randn_like(base) * 0.02
    normalized_coords = torch.stack(
        [
            torch.tensor([0.42, 0.58, 0.5, 0.52, 0.48, 0.55, 0.4, 0.6, 0.5, 0.5], device=device),
            torch.tensor([0.48, 0.52, 0.5, 0.5, 0.45, 0.55, 0.5, 0.5, 0.42, 0.58], device=device),
        ],
        dim=-1,
    )
    eye3 = torch.eye(3, device=device, dtype=dtype)
    sigma_3d = eye3.view(1, 3, 3).repeat(10, 1, 1) * 0.25
    support = torch.eye(2, device=device, dtype=dtype).view(1, 2, 2).repeat(10, 1, 1) * 0.001
    feat_idx = torch.arange(10, device=device) % num_voxels
    feature = voxel_features[feat_idx]
    source_is_image_only = torch.zeros(10, dtype=torch.bool, device=device)

    candidate = {
        "feature": feature,
        "group_ids": group_ids,
        "mean": mean,
        "normalized_coords": normalized_coords,
        "local_agent_ids": local_agent_ids,
        "view_ids": view_ids,
        "sigma_3d": sigma_3d,
        "support_covariance_2d": support,
        "source_is_image_only": source_is_image_only,
    }

    vehicle = {
        "voxel_features": voxel_features,
        "image_feature": image_feature,
        "batch_merged_cam_inputs": cam,
    }
    batch_dict: Dict = {
        "vehicle": vehicle,
        "lidar_mask": {"vehicle": lidar_mask},
        "gaussian_pipeline": {
            "available_agents": ["vehicle"],
            "gaussian_candidates": {"vehicle": [candidate]},
        },
    }
    return batch_dict


def run_pipeline(
    batch_dict: Dict,
    cross_cfg: Dict,
    multiview_cfg: Dict,
    keypoint_cfg: Dict,
    fgu_cfg: Dict,
) -> Tuple[Dict, FirstRoundGaussianGenerator, IntraAgentGaussianRefiner]:
    first_round = FirstRoundGaussianGenerator({"cross_attention": cross_cfg, "multiview_fuser": multiview_cfg})
    refiner = IntraAgentGaussianRefiner(
        {
            "cross_attention": cross_cfg,
            "multiview_fuser": multiview_cfg,
            "keypoint_generator": keypoint_cfg,
            "gaussian_to_image_projector": {},
            "feature_geometry_update": fgu_cfg,
        }
    )
    device = batch_dict["vehicle"]["image_feature"].device
    first_round.to(device)
    refiner.to(device)
    batch_dict = first_round(batch_dict, agent="vehicle")
    batch_dict = refiner.run_second_round_refinement(batch_dict, available_agents=["vehicle"])
    return batch_dict, first_round, refiner


def _fmt_float(x: float) -> str:
    return f"{x:.6g}"


def _quat_geodesic_deg(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """Rotation angle in degrees between unit quaternions (per row)."""
    a = torch.nn.functional.normalize(q0.float(), dim=-1)
    b = torch.nn.functional.normalize(q1.float(), dim=-1)
    d = torch.abs((a * b).sum(dim=-1)).clamp(max=1.0)
    return (2.0 * torch.acos(d) * (180.0 / math.pi)).cpu()


def _tensor_stats(name: str, t: torch.Tensor) -> List[str]:
    """Scalar summary for a tensor (mean/std/min/max/norm for vectors)."""
    if not torch.is_tensor(t) or t.numel() == 0:
        return [f"{name}: <empty>"]
    tc = t.detach().float().cpu().reshape(-1)
    lines = [
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"mean={_fmt_float(float(tc.mean()))} std={_fmt_float(float(tc.std()))} "
        f"min={_fmt_float(float(tc.min()))} max={_fmt_float(float(tc.max()))}"
    ]
    if tc.numel() <= 32:
        lines.append(f"  values: {tc.numpy().tolist()}")
    return lines


def collect_pipeline_stats(batch_dict: Dict, refiner: IntraAgentGaussianRefiner) -> List[str]:
    """Structured text stats: candidates, fusion, projection, masks, geometry/feature deltas."""
    lines: List[str] = []
    gp = batch_dict["gaussian_pipeline"]
    agent = "vehicle"
    agent_batch = batch_dict[agent]
    cand_list = gp.get("gaussian_candidates", {}).get(agent, [])
    fr = gp.get("first_round_gaussians", {}).get(agent)
    sr = gp.get("second_round_gaussians", {}).get(agent)
    imgs = agent_batch["batch_merged_cam_inputs"]["imgs"]
    b, num_views = int(imgs.shape[0]), int(imgs.shape[1])

    lines.append("=== synthetic inputs ===")
    vf = agent_batch.get("voxel_features")
    if torch.is_tensor(vf):
        lines.extend(_tensor_stats("voxel_features", vf))
    lm = batch_dict.get("lidar_mask", {}).get(agent)
    if torch.is_tensor(lm):
        lines.append(f"lidar_mask shape={tuple(lm.shape)} (B,Z,Y,X)")
        for bi in range(lm.shape[0]):
            m = lm[bi]
            lines.append(
                f"  B={bi}: true_voxels={int(m.sum())} / {int(m.numel())} "
                f"({100.0 * float(m.float().mean()):.2f}%)"
            )

    if cand_list:
        merged = {}
        keys = set().union(*(c.keys() for c in cand_list))
        for k in keys:
            parts = [c[k] for c in cand_list if k in c and torch.is_tensor(c[k])]
            if parts:
                merged[k] = torch.cat(parts, dim=0)
        lines.append("=== injected gaussian_candidates (per-hit, pre-fusion) ===")
        lines.append(f"num_hits={merged['group_ids'].shape[0]} unique_group_ids={torch.unique(merged['group_ids']).tolist()}")
        gid = merged["group_ids"].long().cpu()
        lid = merged["local_agent_ids"].long().cpu()
        vid = merged["view_ids"].long().cpu()
        lines.append("hit_idx | group_id | local_B | view_id")
        for i in range(gid.shape[0]):
            lines.append(f"  {i:2d}    | {int(gid[i]):8d} | {int(lid[i]):7d} | {int(vid[i]):7d}")
        lines.append("hits per (local_B, view_id):")
        for bi in range(b):
            for vi in range(num_views):
                n = int(((lid == bi) & (vid == vi)).sum())
                lines.append(f"  (B={bi}, V={vi}): {n}")

    if fr is None:
        lines.append("=== first_round_gaussians: None ===")
        return lines

    lines.append("=== first_round_gaussians (after cross-attn + multiview fuse) ===")
    lines.append(f"num_groups_G={fr['mean'].shape[0]} source_group_ids={fr['source_group_ids'].cpu().tolist()}")
    lines.extend(_tensor_stats("fused mean [G,3]", fr["mean"]))
    lines.extend(_tensor_stats("fused feature [G,C]", fr["feature"]))
    nvv = fr["num_valid_views"].cpu()
    mvm = fr["multi_view_group_mask"].cpu()
    lines.append(f"num_valid_views (hits merged per group): {nvv.tolist()}")
    lines.append(f"multi_view_group_mask (count>1): {mvm.tolist()}")
    if torch.is_tensor(fr.get("local_agent_mask")):
        lam = fr["local_agent_mask"].bool().cpu()
        lines.append(
            f"local_agent_mask [G,B] shape={tuple(lam.shape)} — "
            "True => this Gaussian is evaluated for that batch column when projecting / refining."
        )
        for gi in range(lam.shape[0]):
            parts = [f"B{j}={'1' if lam[gi, j] else '0'}" for j in range(lam.shape[1])]
            lines.append(
                f"  G={gi} (src_gid={int(fr['source_group_ids'][gi])}): " + " ".join(parts)
            )
    vw = fr["view_weights"].detach().cpu()
    gix = fr["group_indices"].long().cpu()
    lines.append(f"first_round view_weights len={vw.shape[0]} (per original hit row in fuse_groups)")
    for g in torch.unique(gix).tolist():
        sel = gix == int(g)
        wsub = vw[sel]
        lines.append(
            f"  group_row={g} (src hits): weights sum={float(wsub.sum()):.6f} "
            f"min={float(wsub.min()):.6g} max={float(wsub.max()):.6g} values={wsub.numpy().tolist()}"
        )

    if sr is None:
        lines.append("=== second_round_gaussians: None ===")
        return lines

    lines.append("=== second_round_gaussians ===")
    rgm = sr["refined_gaussian_mask"].bool().cpu()
    lines.append(
        f"refined_gaussian_mask [G]: {rgm.int().tolist()} "
        f"(True => had valid projection hit + cross-attn + fuse this round)"
    )
    lines.extend(_tensor_stats("second mean [G,3]", sr["mean"]))
    lines.extend(_tensor_stats("second feature [G,C]", sr["feature"]))
    if torch.is_tensor(sr.get("num_valid_views")):
        lines.append(f"second num_valid_views (subset): {sr['num_valid_views'].cpu().tolist()}")
    if torch.is_tensor(sr.get("multi_view_group_mask")):
        lines.append(f"second multi_view_group_mask: {sr['multi_view_group_mask'].int().cpu().tolist()}")

    lines.append("=== geometry / feature delta (second vs first, same G index) ===")
    dm = (sr["mean"] - fr["mean"]).detach().float().cpu()
    lines.extend(_tensor_stats("|Δmean| per-G", dm.norm(dim=-1)))
    ds = (sr["scale"] - fr["scale"]).detach().float().cpu()
    lines.extend(_tensor_stats("|Δscale| per-G", ds.norm(dim=-1)))
    dq = (sr["rotation"] - fr["rotation"]).detach().float().cpu()
    lines.extend(_tensor_stats("Δquat vec-L2 per-G (not rotation angle)", dq.norm(dim=-1)))
    qdeg = _quat_geodesic_deg(fr["rotation"], sr["rotation"])
    lines.extend(_tensor_stats("quat_geodesic_deg (first↔second)", qdeg))
    dfeat = (sr["feature"] - fr["feature"]).detach().float().cpu()
    lines.extend(_tensor_stats("|Δfeature| per-G", dfeat.norm(dim=-1)))
    lines.append("per-G breakdown (only refined typically has non-zero Δmean from FGU):")
    for gi in range(fr["mean"].shape[0]):
        lines.append(
            f"  G={gi} refined={bool(rgm[gi])} |Δm|={float(dm[gi].norm()):.6g} |Δs|={float(ds[gi].norm()):.6g} "
            f"Δq_vecL2={float(dq[gi].norm()):.6g} rot_deg={float(qdeg[gi]):.4g} |Δc|={float(dfeat[gi].norm()):.6g}"
        )

    lines.append("=== GaussianToImageProjector (second round) — mask / pairing logic ===")
    lines.append(
        "Step A: local_agent_mask[G,B] -> valid (G,B) pairs (Cartesian membership). "
        "Only these pairs are considered for projecting that Gaussian's keypoints."
    )
    lines.append(
        "Step B: For each pair, project ALL keypoints to ALL views; depth>0 and norm coords in (0,1). "
        "hit_mask = any keypoint valid per (pair, view). Emit one row per (pair,view) with a valid hit."
    )
    lines.append(
        "Step C: normalized_coords = sampling_coords[:,0] (first support sample = center path in DA). "
        "Downstream uses gaussian_ids, local_agent_ids, view_ids, sampling_coords, sampling_valid_mask."
    )
    key_dict = refiner.keypoint_generator(
        mean=sr["mean"],
        axis_scales=sr["scale"],
        rotation=sr["rotation"],
        feature=sr["feature"],
    )
    kp = key_dict["key_points"]
    lines.append(f"key_points shape [G, P, 3]: G={kp.shape[0]} P={kp.shape[1]}")
    lam_sr = sr["local_agent_mask"].bool()
    num_pairs = int(lam_sr.sum())
    lines.append(f"local_agent_mask: num_valid_(G,B)_pairs={num_pairs}")

    proj = refiner.gaussian_to_image_projector(
        agent_batch=agent_batch,
        gaussian_key_points=key_dict["key_points"],
        local_agent_mask=lam_sr,
    )
    nh = proj["gaussian_ids"].shape[0]
    lines.append(f"projection output rows (valid (G,B,V) hits with any in-bounds keypoint): {nh}")
    if nh > 0:
        gid = proj["gaussian_ids"].long().cpu()
        lid = proj["local_agent_ids"].long().cpu()
        vid = proj["view_ids"].long().cpu()
        vmask = proj["sampling_valid_mask"].bool().cpu()
        lines.append("projection hits per (B, V):")
        for bi in range(b):
            for vi in range(num_views):
                sel = (lid == bi) & (vid == vi)
                lines.append(f"  (B={bi}, V={vi}): {int(sel.sum())} rows")
        lines.append(
            f"sampling_valid_mask shape [N,P]: P={vmask.shape[1]} — "
            f"fraction True per sample index: "
            f"{[float(vmask[:, j].float().mean()) for j in range(vmask.shape[1])]}"
        )
        lines.append("first 16 projection rows: gaussian_id local_B view_id valid[0:4]...")
        for j in range(min(16, nh)):
            v0 = vmask[j, : min(4, vmask.shape[1])].int().tolist()
            lines.append(
                f"  j={j:2d} g={int(gid[j]):2d} B={int(lid[j])} V={int(vid[j])} valid_head={v0}"
            )

    lines.append("=== lidar_mask note ===")
    lines.append(
        "Synthetic lidar_mask is NOT wired into candidate construction in this script; "
        "it only documents B-major occupancy stats. To couple mask->hits, index voxels where mask True."
    )
    return lines


def draw_figure(
    batch_dict: Dict,
    refiner: IntraAgentGaussianRefiner,
    out_path: Path,
) -> None:
    agent_batch = batch_dict["vehicle"]
    cam = agent_batch["batch_merged_cam_inputs"]
    imgs = cam["imgs"].detach().cpu()
    lidar2image, image_wh = _build_lidar2image_and_wh(
        imgs.to(cam["intrinsics"].device),
        cam["intrinsics"],
        cam["extrinsics"],
        cam["post_rots"],
        cam["post_trans"],
    )
    gp = batch_dict["gaussian_pipeline"]
    fr = gp["first_round_gaussians"]["vehicle"]
    sr = gp["second_round_gaussians"]["vehicle"]
    assert fr is not None and sr is not None

    b, v, _, h, w = imgs.shape
    fig, axes = plt.subplots(2, b * v, figsize=(4 * b * v, 8), squeeze=False)
    fr_mean = fr["mean"].detach().cpu()
    fr_lam = fr["local_agent_mask"].bool().detach().cpu()

    colors = plt.cm.tab10(np.linspace(0, 1, fr_mean.shape[0], endpoint=False))

    for bi in range(b):
        for vi in range(v):
            col = bi * v + vi
            ax = axes[0, col]
            img = imgs[bi, vi].permute(1, 2, 0).numpy().clip(0, 1)
            ax.imshow(img)
            ax.set_title(f"First round (proj mean) B={bi} V={vi}")
            ax.axis("off")
            lidar2i = lidar2image.cpu()
            wh = image_wh.cpu()
            for gi in range(fr_mean.shape[0]):
                if not fr_lam[gi, bi]:
                    continue
                norm = project_mean_to_normalized(
                    lidar2i,
                    wh,
                    fr_mean[gi],
                    bi,
                    vi,
                )
                if bool(torch.isnan(norm).any()):
                    continue
                px = float(norm[0]) * (w - 1)
                py = float(norm[1]) * (h - 1)
                ax.scatter([px], [py], c=[colors[gi]], s=80, marker="o", edgecolors="white", linewidths=0.5)

    lam = sr["local_agent_mask"].bool().cpu()
    key_dict = refiner.keypoint_generator(
        mean=sr["mean"],
        axis_scales=sr["scale"],
        rotation=sr["rotation"],
        feature=sr["feature"],
    )
    proj = refiner.gaussian_to_image_projector(
        agent_batch=agent_batch,
        gaussian_key_points=key_dict["key_points"],
        local_agent_mask=sr["local_agent_mask"],
    )
    samp = proj["sampling_coords"].detach().cpu()
    g_ids = proj["gaussian_ids"].detach().cpu()
    lid_ids = proj["local_agent_ids"].detach().cpu()
    v_ids = proj["view_ids"].detach().cpu()
    valid = proj["sampling_valid_mask"].detach().cpu()

    for bi in range(b):
        for vi in range(v):
            col = bi * v + vi
            ax = axes[1, col]
            img = imgs[bi, vi].permute(1, 2, 0).numpy().clip(0, 1)
            ax.imshow(img)
            ax.set_title(f"Second round (keypt[0]) B={bi} V={vi}")
            ax.axis("off")
            mask = (lid_ids == bi) & (v_ids == vi)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            for j in idx.tolist():
                gi = int(g_ids[j])
                if not lam[gi, bi]:
                    continue
                if not bool(valid[j, 0]):
                    continue
                xy = samp[j, 0]
                px = float(xy[0]) * (w - 1)
                py = float(xy[1]) * (h - 1)
                ax.scatter([px], [py], c=[colors[gi]], s=60, marker="x", linewidths=1.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("gaussian_pipeline_cpu_vis.png"))
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=None,
        help="Write collect_pipeline_stats() lines to this UTF-8 text file.",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip PNG; only print / write stats.",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    dtype = torch.float32
    cross_cfg = {"attention_dim": 64, "align_corners": True, "support_min_offset": 0.02}
    multiview_cfg = {"view_token_dim": 32}
    keypoint_cfg: Dict = {}
    fgu_cfg = {"hidden_dim": 64, "scale_delta_scale": 0.15, "mean_delta_scale": 0.5}

    batch_dict = build_synthetic_batch_dict(
        device=device,
        dtype=dtype,
        num_voxels=32,
        voxel_channels=16,
        image_channels=8,
        b=2,
        v=2,
        h=96,
        w=128,
        fh=12,
        fw=16,
    )
    batch_dict, _first, refiner = run_pipeline(
        batch_dict, cross_cfg=cross_cfg, multiview_cfg=multiview_cfg, keypoint_cfg=keypoint_cfg, fgu_cfg=fgu_cfg
    )
    stat_lines = collect_pipeline_stats(batch_dict, refiner)
    text = "\n".join(stat_lines) + "\n"
    print(text, end="")
    if args.stats_out is not None:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(text, encoding="utf-8")
        print(f"Wrote stats to {args.stats_out.resolve()}")
    if not args.no_figure:
        draw_figure(batch_dict, refiner, args.out)
        print(f"Saved visualization to {args.out.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose remaining dense-region false positives after fixing Gaussian->BEV splat.

Goal
----
Separate two geometry failure modes:

A) mean-near / mean-placement problem
   High-objectness negative locations are covered by Gaussians whose MEANS
   themselves lie very close to that negative location.

B) large-covariance / ellipse-overlap problem
   High-objectness negative locations receive substantial support from Gaussians
   whose means are far away in Euclidean XY, but whose covariance is large enough
   that the location is still within a small Mahalanobis distance.

The script is READ-ONLY:
- does not modify checkpoint
- does not modify training
- does not modify loss
- does not modify production code
- baseline forward is untouched
- optional sigma ablation uses a temporary forward_pre_hook on the splat only

Baseline geometry statistics are measured at the objectness spatial location.
A Gaussian is counted once per objectness spatial location even if it covers
multiple underlying high-resolution BEV cells inside that location.

Outputs
-------
<model_dir>/mean_vs_sigma_diag_epoch<E>/
    mean_vs_sigma_summary.json
    high_obj_negative_geometry.csv

Example
-------
cd /data2/home/suyi/AirV2X-Perception_gaussian

python opencood/tools/diagnose_mean_vs_sigma.py \
  --model_dir /path/to/run \
  --epoch 1 \
  --num_frames 50 \
  --workers 4 \
  --obj_threshold 0.2 \
  --near_m 0.8 \
  --far_m 2.0 \
  --large_sigma_m 1.0 \
  --sigma_scales 1.0,0.75,0.5

Interpretation
--------------
Evidence for mean-placement:
    high-obj negatives have very small nearest_mean_dist and a large fraction
    of contributor weight from means within --near_m.

Evidence for large ellipses:
    high-obj negatives have substantial weight from contributors with
    Euclidean distance > --far_m, especially when sigma_major is large and
    Mahalanobis distance remains <= 2 or <= 3.
    If sigma shrink 0.75/0.5 sharply reduces negative obj>threshold while
    positive pass rate remains similar, this is strong causal evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import replace
from typing import Dict, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------
# repo import
# ---------------------------------------------------------------------
_THIS = os.path.abspath(__file__)
_REPO = "/".join(_THIS.split("/")[:-3])
if os.path.isdir(os.path.join(_REPO, "opencood")):
    sys.path.insert(0, _REPO)
else:
    sys.path.insert(0, os.getcwd())

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


AGENTS = ("vehicle", "rsu", "drone")
AGENT_TO_ID = {"vehicle": 0, "rsu": 1, "drone": 2}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--num_frames", type=int, default=50)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--obj_threshold", type=float, default=0.2)

    # Geometry signatures.
    p.add_argument("--near_m", type=float, default=0.8)
    p.add_argument("--far_m", type=float, default=2.0)
    p.add_argument("--large_sigma_m", type=float, default=1.0)

    # Inference-only causal test. 1.0 is baseline.
    p.add_argument(
        "--sigma_scales",
        type=str,
        default="1.0,0.75,0.5",
        help="Temporary scale multipliers used only inside GaussianBEVSplat.",
    )

    p.add_argument(
        "--save_topk_neg_per_frame",
        type=int,
        default=500,
        help="Save top-K explicit-negative anchors by objectness per frame.",
    )
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def find_target_dict(ego: Mapping) -> Mapping:
    for key in ("label_dict", "label_dict_single", "target_dict"):
        obj = ego.get(key, None)
        if isinstance(obj, Mapping) and "pos_equal_one" in obj:
            return obj
    if "pos_equal_one" in ego:
        return ego
    raise KeyError(
        "Cannot find target dict containing pos_equal_one/neg_equal_one. "
        f"ego keys={list(ego.keys())[:50]}"
    )


def summary_np(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p99": float(np.quantile(x, 0.99)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def safe_scatter_min(dst, index, src):
    if hasattr(dst, "scatter_reduce_"):
        dst.scatter_reduce_(0, index, src, reduce="amin", include_self=True)
    else:
        for i, v in zip(index.tolist(), src.tolist()):
            if v < float(dst[i]):
                dst[i] = v


def safe_scatter_max(dst, index, src):
    if hasattr(dst, "scatter_reduce_"):
        dst.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
    else:
        for i, v in zip(index.tolist(), src.tolist()):
            if v > float(dst[i]):
                dst[i] = v


def scatter_sum(index, src, size):
    out = torch.zeros(size, device=src.device, dtype=torch.float32)
    out.index_add_(0, index, src.float())
    return out


class GeometryInspector:
    """
    Forward hook on Stage-3.

    Reconstructs the exact Gaussian support pairs from the GaussianSet sent to
    the production splat, then deduplicates (objectness spatial cell, Gaussian)
    so each Gaussian counts once per detector spatial location.
    """

    def __init__(self, near_m: float, far_m: float, large_sigma_m: float):
        self.near_m = float(near_m)
        self.far_m = float(far_m)
        self.large_sigma_m = float(large_sigma_m)
        self.enabled = True
        self.last = None

    @torch.no_grad()
    def hook(self, module, inputs, output):
        if not self.enabled:
            return
        gaussians_by_agent = inputs[0]
        out_gs, bev = output
        splat = module.splat

        # Reconstruct source-agent ids in the same merge order as Stage-3.
        pooled = {
            agent: module.pool(gs)
            for agent, gs in gaussians_by_agent.items()
            if gs.n_gaussians > 0
        }
        if not pooled or out_gs.n_gaussians == 0:
            self.last = None
            return

        merged_geom, _ = module._merge(pooled)
        if merged_geom.n_gaussians != out_gs.n_gaussians:
            raise RuntimeError("Stage3 pooled/output Gaussian count mismatch")
        if not torch.allclose(merged_geom.mean, out_gs.mean, atol=1e-5, rtol=1e-5):
            err = float((merged_geom.mean - out_gs.mean).abs().max().item())
            raise RuntimeError(f"Stage3 merge ordering mismatch: max mean err={err}")

        source_id = torch.cat(
            [
                torch.full(
                    (gs.n_gaussians,),
                    AGENT_TO_ID.get(str(agent).lower(), -1),
                    device=out_gs.mean.device,
                    dtype=torch.long,
                )
                for agent, gs in pooled.items()
            ],
            dim=0,
        )

        B, C, ny, nx = bev.shape
        if B != 1:
            raise RuntimeError(f"Diagnostic requires batch_size=1, got {B}")
        n = out_gs.n_gaussians
        device = out_gs.mean.device

        cov_xy = out_gs.covariance()[:, :2, :2]
        cov_xy = cov_xy + torch.eye(
            2, device=device, dtype=cov_xy.dtype
        ) * 1e-4

        cell = splat.cell.to(device=device, dtype=out_gs.mean.dtype)
        xy_min = splat.xy_min.to(device=device, dtype=out_gs.mean.dtype)

        var_xy = cov_xy.diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
        radius = torch.clamp(
            torch.ceil(splat.n_sigma * var_xy.sqrt() / cell).long(),
            min=1,
        )
        mean_xy = out_gs.mean[:, :2]
        center_idx = torch.floor((mean_xy - xy_min) / cell).long()

        g_idx, ix, iy, delta = splat._candidate_pairs(
            center_idx, radius, mean_xy, xy_min, cell, nx, ny
        )
        if g_idx.numel() == 0:
            self.last = None
            return

        cov = cov_xy[g_idx]
        a = cov[:, 0, 0]
        b = cov[:, 0, 1]
        c = cov[:, 1, 1]
        det = (a * c - b * b).clamp_min(1e-6)
        inv00 = c / det
        inv01 = -b / det
        inv11 = a / det
        dx, dy = delta[:, 0], delta[:, 1]
        d2 = dx * (inv00 * dx + inv01 * dy) + dy * (inv01 * dx + inv11 * dy)

        support = d2 <= float(splat.n_sigma) ** 2
        g_idx = g_idx[support]
        ix = ix[support]
        iy = iy[support]

        # Detector objectness spatial resolution is not available inside Stage3.
        # Store the Gaussian geometry and supported splat pairs; finalize after
        # seeing output["obj"] in the caller.
        self.last = {
            "out_gs": out_gs,
            "source_id": source_id,
            "cov_xy": cov_xy,
            "g_idx": g_idx,
            "ix": ix,
            "iy": iy,
            "cell": cell.detach(),
            "xy_min": xy_min.detach(),
            "ny": ny,
            "nx": nx,
            "bev": bev.detach(),
        }

    @torch.no_grad()
    def finalize_for_obj_grid(self, H: int, W: int) -> Dict[str, np.ndarray]:
        c = self.last
        if c is None:
            raise RuntimeError("No Stage3 geometry captured")

        gs = c["out_gs"]
        g_idx = c["g_idx"]
        ix = c["ix"]
        iy = c["iy"]
        ny, nx = c["ny"], c["nx"]
        cell = c["cell"]
        xy_min = c["xy_min"]
        cov_xy = c["cov_xy"]
        source_id = c["source_id"]

        if ny % H != 0 or nx % W != 0:
            raise RuntimeError(
                f"Cannot map splat grid {ny}x{nx} to obj grid {H}x{W}"
            )
        sy, sx = ny // H, nx // W

        # Map each supported high-res BEV pair to detector spatial location.
        oy = torch.div(iy, sy, rounding_mode="floor")
        ox = torch.div(ix, sx, rounding_mode="floor")
        obj_id = oy * W + ox
        n_obj = H * W
        n_g = gs.n_gaussians

        # Deduplicate (obj spatial location, Gaussian).
        key = obj_id.long() * n_g + g_idx.long()
        uniq = torch.unique(key)
        u_obj = torch.div(uniq, n_g, rounding_mode="floor")
        u_g = uniq % n_g

        # Detector spatial-cell physical center.
        u_oy = torch.div(u_obj, W, rounding_mode="floor")
        u_ox = u_obj % W
        cx = xy_min[0] + (u_ox.float() * sx + 0.5 * sx) * cell[0]
        cy = xy_min[1] + (u_oy.float() * sy + 0.5 * sy) * cell[1]
        obj_xy = torch.stack([cx, cy], dim=-1)

        dxy = obj_xy - gs.mean[u_g, :2]
        euclid = torch.linalg.norm(dxy.float(), dim=-1)

        # Mahalanobis distance at detector cell center.
        cov = cov_xy[u_g].float()
        a = cov[:, 0, 0]
        b = cov[:, 0, 1]
        cc = cov[:, 1, 1]
        det = (a * cc - b * b).clamp_min(1e-6)
        inv00 = cc / det
        inv01 = -b / det
        inv11 = a / det
        dx, dy = dxy[:, 0].float(), dxy[:, 1].float()
        d2 = dx * (inv00 * dx + inv01 * dy) + dy * (inv01 * dx + inv11 * dy)
        d2 = d2.clamp_min(0.0)
        maha = torch.sqrt(d2)

        # Gaussian response at detector spatial center. Keep value even if center
        # lies barely outside 3σ while one sub-cell is inside; useful diagnostic.
        weight = torch.exp(-0.5 * d2)

        # XY covariance principal-axis sigmas.
        eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
        sigma_minor = torch.sqrt(eig[:, 0])
        sigma_major = torch.sqrt(eig[:, 1])

        feat_rms = torch.sqrt(
            torch.mean(gs.feature[u_g].float() ** 2, dim=-1).clamp_min(0.0)
        )
        contribution_amp = weight * feat_rms

        # Aggregate per objectness spatial location.
        one = torch.ones_like(weight)
        count = scatter_sum(u_obj, one, n_obj)
        weight_sum = scatter_sum(u_obj, weight, n_obj)
        contribution_sum = scatter_sum(u_obj, contribution_amp, n_obj)

        nearest = torch.full(
            (n_obj,), float("inf"), device=weight.device, dtype=torch.float32
        )
        safe_scatter_min(nearest, u_obj, euclid)
        nearest = torch.where(
            torch.isfinite(nearest), nearest, torch.zeros_like(nearest)
        )

        sigma_major_max = torch.zeros(n_obj, device=weight.device)
        safe_scatter_max(sigma_major_max, u_obj, sigma_major)

        sigma_major_weighted = scatter_sum(
            u_obj, weight * sigma_major, n_obj
        ) / weight_sum.clamp_min(1e-12)

        # Euclidean mean-distance buckets.
        near = euclid < self.near_m
        mid = (euclid >= self.near_m) & (euclid <= self.far_m)
        far = euclid > self.far_m

        near_count_frac = scatter_sum(
            u_obj, near.float(), n_obj
        ) / count.clamp_min(1)
        mid_count_frac = scatter_sum(
            u_obj, mid.float(), n_obj
        ) / count.clamp_min(1)
        far_count_frac = scatter_sum(
            u_obj, far.float(), n_obj
        ) / count.clamp_min(1)

        near_weight_frac = scatter_sum(
            u_obj, weight * near.float(), n_obj
        ) / weight_sum.clamp_min(1e-12)
        mid_weight_frac = scatter_sum(
            u_obj, weight * mid.float(), n_obj
        ) / weight_sum.clamp_min(1e-12)
        far_weight_frac = scatter_sum(
            u_obj, weight * far.float(), n_obj
        ) / weight_sum.clamp_min(1e-12)

        # Mahalanobis buckets.
        core1 = maha <= 1.0
        ring12 = (maha > 1.0) & (maha <= 2.0)
        ring23 = (maha > 2.0) & (maha <= 3.0)
        beyond3 = maha > 3.0

        maha_le1_weight_frac = scatter_sum(
            u_obj, weight * core1.float(), n_obj
        ) / weight_sum.clamp_min(1e-12)
        maha_12_weight_frac = scatter_sum(
            u_obj, weight * ring12.float(), n_obj
        ) / weight_sum.clamp_min(1e-12)
        maha_23_weight_frac = scatter_sum(
            u_obj, weight * ring23.float(), n_obj
        ) / weight_sum.clamp_min(1e-12)
        maha_gt3_weight_frac = scatter_sum(
            u_obj, weight * beyond3.float(), n_obj
        ) / weight_sum.clamp_min(1e-12)

        # "Large ellipse reaches from far away" signatures.
        large_sigma = sigma_major >= self.large_sigma_m
        far_large = far & large_sigma
        far_maha_le2 = far & (maha <= 2.0)
        far_maha_le3 = far & (maha <= 3.0)

        far_large_count_frac = scatter_sum(
            u_obj, far_large.float(), n_obj
        ) / count.clamp_min(1)
        far_large_weight_frac = scatter_sum(
            u_obj, weight * far_large.float(), n_obj
        ) / weight_sum.clamp_min(1e-12)
        far_maha_le2_weight_frac = scatter_sum(
            u_obj, weight * far_maha_le2.float(), n_obj
        ) / weight_sum.clamp_min(1e-12)
        far_maha_le3_weight_frac = scatter_sum(
            u_obj, weight * far_maha_le3.float(), n_obj
        ) / weight_sum.clamp_min(1e-12)

        # Source-agent fractions.
        src = source_id[u_g]
        agent_weight_frac = {}
        agent_count_frac = {}
        for agent, aid in AGENT_TO_ID.items():
            m = src == aid
            agent_weight_frac[agent] = scatter_sum(
                u_obj, weight * m.float(), n_obj
            ) / weight_sum.clamp_min(1e-12)
            agent_count_frac[agent] = scatter_sum(
                u_obj, m.float(), n_obj
            ) / count.clamp_min(1)

        # Raw BEV magnitude pooled to detector spatial grid.
        bev = c["bev"][0].float()  # [C,ny,nx]
        bev_rms = torch.sqrt(torch.mean(bev ** 2, dim=0).clamp_min(0.0))
        bev_rms_mean = (
            bev_rms.reshape(H, sy, W, sx).mean(dim=(1, 3))
        ).reshape(-1)

        def np_hw(x):
            return x.reshape(H, W).detach().cpu().numpy()

        result = {
            "contributor_count": np_hw(count),
            "weight_sum": np_hw(weight_sum),
            "contribution_amp_sum": np_hw(contribution_sum),
            "nearest_mean_dist": np_hw(nearest),
            "sigma_major_max": np_hw(sigma_major_max),
            "sigma_major_weighted": np_hw(sigma_major_weighted),
            "near_count_frac": np_hw(near_count_frac),
            "mid_count_frac": np_hw(mid_count_frac),
            "far_count_frac": np_hw(far_count_frac),
            "near_weight_frac": np_hw(near_weight_frac),
            "mid_weight_frac": np_hw(mid_weight_frac),
            "far_weight_frac": np_hw(far_weight_frac),
            "maha_le1_weight_frac": np_hw(maha_le1_weight_frac),
            "maha_12_weight_frac": np_hw(maha_12_weight_frac),
            "maha_23_weight_frac": np_hw(maha_23_weight_frac),
            "maha_gt3_weight_frac": np_hw(maha_gt3_weight_frac),
            "far_large_count_frac": np_hw(far_large_count_frac),
            "far_large_weight_frac": np_hw(far_large_weight_frac),
            "far_maha_le2_weight_frac": np_hw(far_maha_le2_weight_frac),
            "far_maha_le3_weight_frac": np_hw(far_maha_le3_weight_frac),
            "bev_rms_mean": np_hw(bev_rms_mean),
        }
        for agent in AGENTS:
            result[f"{agent}_weight_frac"] = np_hw(agent_weight_frac[agent])
            result[f"{agent}_count_frac"] = np_hw(agent_count_frac[agent])

        return result


# ---------------------------------------------------------------------
# Temporary sigma shrink at splat only
# ---------------------------------------------------------------------
class SigmaScalePreHook:
    def __init__(self, scale: float):
        self.scale = float(scale)

    def __call__(self, module, inputs):
        if abs(self.scale - 1.0) < 1e-12:
            return None
        gs = inputs[0]
        scaled = replace(gs, scale=gs.scale * self.scale)
        return (scaled,)


METRICS = (
    "contributor_count",
    "weight_sum",
    "contribution_amp_sum",
    "nearest_mean_dist",
    "sigma_major_max",
    "sigma_major_weighted",
    "near_count_frac",
    "mid_count_frac",
    "far_count_frac",
    "near_weight_frac",
    "mid_weight_frac",
    "far_weight_frac",
    "maha_le1_weight_frac",
    "maha_12_weight_frac",
    "maha_23_weight_frac",
    "maha_gt3_weight_frac",
    "far_large_count_frac",
    "far_large_weight_frac",
    "far_maha_le2_weight_frac",
    "far_maha_le3_weight_frac",
    "vehicle_weight_frac",
    "vehicle_count_frac",
    "drone_weight_frac",
    "drone_count_frac",
    "bev_rms_mean",
)


def expand_anchor(arr_hw: np.ndarray, A: int):
    return np.repeat(arr_hw[..., None], A, axis=-1).reshape(-1)


def group_summary(flat, mask):
    out = {"n": int(mask.sum())}
    if out["n"] == 0:
        return out
    out["obj"] = summary_np(flat["obj"][mask])
    for k in METRICS:
        out[k] = summary_np(flat[k][mask])
    return out


def main():
    args = parse_args()
    sigma_scales = [float(x.strip()) for x in args.sigma_scales.split(",") if x.strip()]
    if 1.0 not in sigma_scales:
        sigma_scales = [1.0] + sigma_scales

    start = args.start
    end = start + args.num_frames
    out_dir = args.out or os.path.join(
        args.model_dir, f"mean_vs_sigma_diag_epoch{args.epoch}"
    )
    os.makedirs(out_dir, exist_ok=True)

    hypes = yaml_utils.load_yaml(None, args)
    if "test_dir" in hypes:
        hypes["validate_dir"] = hypes["test_dir"]

    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _, model = train_utils.load_model(
        args.model_dir, model, args.epoch, start_from_best=False
    )
    model.eval()
    model.zero_grad(set_to_none=True)

    inspector = GeometryInspector(
        near_m=args.near_m,
        far_m=args.far_m,
        large_sigma_m=args.large_sigma_m,
    )
    stage3_handle = model.stage3.register_forward_hook(inspector.hook)

    accum = defaultdict(list)
    top_rows = []

    sigma_ablation = {
        str(s): {
            "pos_total": 0,
            "pos_pass": 0,
            "neg_total": 0,
            "neg_pass": 0,
            "obj_pos": [],
            "obj_neg": [],
        }
        for s in sigma_scales
    }

    try:
        for frame_idx, batch_data in tqdm(enumerate(loader), total=min(len(loader), end)):
            if frame_idx < start:
                continue
            if frame_idx >= end:
                break

            batch_data = train_utils.to_device(batch_data, device)

            # ---------------- baseline ----------------
            inspector.enabled = True
            inspector.last = None
            with torch.no_grad():
                output = model(batch_data["ego"])

            obj = torch.sigmoid(output["obj"]).permute(0, 2, 3, 1)[0]
            H, W, A = obj.shape
            obj_np = obj.detach().cpu().numpy()

            target = find_target_dict(batch_data["ego"])
            pos = target["pos_equal_one"][0].detach().cpu().numpy() > 0
            neg = target["neg_equal_one"][0].detach().cpu().numpy() > 0
            if pos.shape != obj_np.shape or neg.shape != obj_np.shape:
                raise RuntimeError(
                    f"shape mismatch obj={obj_np.shape}, pos={pos.shape}, neg={neg.shape}"
                )

            geo = inspector.finalize_for_obj_grid(H, W)

            flat = {
                "obj": obj_np.reshape(-1),
                "pos": pos.reshape(-1),
                "neg": neg.reshape(-1),
            }
            for k in METRICS:
                flat[k] = expand_anchor(geo[k], A).astype(np.float32)

            for k, v in flat.items():
                accum[k].append(v)

            # Save top high-obj explicit negatives with geometry signatures.
            neg_ids = np.flatnonzero(flat["neg"])
            if neg_ids.size:
                order = neg_ids[np.argsort(flat["obj"][neg_ids])[::-1]]
                order = order[: args.save_topk_neg_per_frame]
                for idx in order:
                    if flat["obj"][idx] < args.obj_threshold:
                        break
                    hw = idx // A
                    slot = idx % A
                    yy = hw // W
                    xx = hw % W
                    row = {
                        "frame": frame_idx,
                        "obj_y": int(yy),
                        "obj_x": int(xx),
                        "anchor_slot": int(slot),
                        "obj": float(flat["obj"][idx]),
                    }
                    for k in METRICS:
                        row[k] = float(flat[k][idx])
                    top_rows.append(row)

            # Baseline ablation stats.
            s1 = sigma_ablation[str(1.0)]
            s1["pos_total"] += int(pos.sum())
            s1["pos_pass"] += int((pos & (obj_np > args.obj_threshold)).sum())
            s1["neg_total"] += int(neg.sum())
            s1["neg_pass"] += int((neg & (obj_np > args.obj_threshold)).sum())
            s1["obj_pos"].append(obj_np[pos])
            s1["obj_neg"].append(obj_np[neg])

            # ---------------- sigma-only black-box ablation ----------------
            inspector.enabled = False
            for s in sigma_scales:
                if abs(s - 1.0) < 1e-12:
                    continue
                hook = model.stage3.splat.register_forward_pre_hook(
                    SigmaScalePreHook(s)
                )
                try:
                    with torch.no_grad():
                        out_s = model(batch_data["ego"])
                finally:
                    hook.remove()

                p = (
                    torch.sigmoid(out_s["obj"])
                    .permute(0, 2, 3, 1)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                st = sigma_ablation[str(s)]
                st["pos_total"] += int(pos.sum())
                st["pos_pass"] += int((pos & (p > args.obj_threshold)).sum())
                st["neg_total"] += int(neg.sum())
                st["neg_pass"] += int((neg & (p > args.obj_threshold)).sum())
                st["obj_pos"].append(p[pos])
                st["obj_neg"].append(p[neg])

            print(
                f"[frame {frame_idx:04d}] "
                f"pos={int(pos.sum())} "
                f"neg>{args.obj_threshold:g}="
                f"{int((neg & (obj_np > args.obj_threshold)).sum())} "
                f"high-neg saved={sum(1 for r in top_rows if r['frame']==frame_idx)}"
            )

    finally:
        stage3_handle.remove()

    data = {k: np.concatenate(v, axis=0) for k, v in accum.items()}
    pos = data["pos"].astype(bool)
    neg = data["neg"].astype(bool)
    high_neg = neg & (data["obj"] > args.obj_threshold)
    low_neg = neg & (data["obj"] <= 0.1)

    # Vehicle-dominated subset is especially relevant to the user's visual FP.
    veh_dom = data["vehicle_weight_frac"] >= 0.5
    high_neg_vehicle = high_neg & veh_dom

    groups = {
        "positive": group_summary(data, pos),
        "explicit_negative_all": group_summary(data, neg),
        "explicit_negative_low_obj_le_0p1": group_summary(data, low_neg),
        "explicit_negative_high_obj": group_summary(data, high_neg),
        "high_obj_negative_vehicle_dominated": group_summary(
            data, high_neg_vehicle
        ),
    }

    # Finish sigma-ablation summaries.
    sigma_report = {}
    for s in sigma_scales:
        st = sigma_ablation[str(s)]
        pos_arr = np.concatenate(st["obj_pos"]) if st["obj_pos"] else np.asarray([])
        neg_arr = np.concatenate(st["obj_neg"]) if st["obj_neg"] else np.asarray([])
        sigma_report[str(s)] = {
            "scale_multiplier": s,
            "positive_pass_rate": (
                st["pos_pass"] / st["pos_total"] if st["pos_total"] else float("nan")
            ),
            "negative_pass_rate": (
                st["neg_pass"] / st["neg_total"] if st["neg_total"] else float("nan")
            ),
            "positive_obj": summary_np(pos_arr),
            "negative_obj": summary_np(neg_arr),
            "pos_total": st["pos_total"],
            "neg_total": st["neg_total"],
            "pos_pass": st["pos_pass"],
            "neg_pass": st["neg_pass"],
        }

    # Compact interpretation indicators (not proof by themselves).
    hn = groups["explicit_negative_high_obj"]
    hv = groups["high_obj_negative_vehicle_dominated"]
    indicators = {
        "mean_near_signature": {
            "high_neg_nearest_mean_median":
                hn.get("nearest_mean_dist", {}).get("median", float("nan")),
            "high_neg_near_weight_frac_median":
                hn.get("near_weight_frac", {}).get("median", float("nan")),
            "vehicle_high_neg_nearest_mean_median":
                hv.get("nearest_mean_dist", {}).get("median", float("nan")),
            "vehicle_high_neg_near_weight_frac_median":
                hv.get("near_weight_frac", {}).get("median", float("nan")),
        },
        "large_ellipse_signature": {
            "high_neg_sigma_major_weighted_median":
                hn.get("sigma_major_weighted", {}).get("median", float("nan")),
            "high_neg_far_weight_frac_median":
                hn.get("far_weight_frac", {}).get("median", float("nan")),
            "high_neg_far_large_weight_frac_median":
                hn.get("far_large_weight_frac", {}).get("median", float("nan")),
            "high_neg_far_maha_le3_weight_frac_median":
                hn.get("far_maha_le3_weight_frac", {}).get("median", float("nan")),
            "vehicle_high_neg_sigma_major_weighted_median":
                hv.get("sigma_major_weighted", {}).get("median", float("nan")),
            "vehicle_high_neg_far_large_weight_frac_median":
                hv.get("far_large_weight_frac", {}).get("median", float("nan")),
        },
    }

    result = {
        "model_dir": args.model_dir,
        "epoch": args.epoch,
        "num_frames_processed": min(args.num_frames, len(loader) - args.start),
        "obj_threshold": args.obj_threshold,
        "definitions": {
            "near_m": args.near_m,
            "far_m": args.far_m,
            "large_sigma_major_m": args.large_sigma_m,
            "mean_near": f"Euclidean XY distance from Gaussian mean to detector spatial center < {args.near_m} m",
            "far": f"Euclidean XY distance > {args.far_m} m",
            "far_large": f"distance > {args.far_m} m AND sigma_major >= {args.large_sigma_m} m",
            "far_maha_le3": f"distance > {args.far_m} m but Mahalanobis distance <= 3",
        },
        "groups": groups,
        "sigma_only_inference_ablation": sigma_report,
        "indicators": indicators,
    }

    json_path = os.path.join(out_dir, "mean_vs_sigma_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=True)

    csv_path = os.path.join(out_dir, "high_obj_negative_geometry.csv")
    if top_rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(top_rows[0].keys()))
            w.writeheader()
            w.writerows(top_rows)

    # Terminal report.
    print("\n================ GEOMETRY DIAGNOSIS ================")
    for name in (
        "positive",
        "explicit_negative_low_obj_le_0p1",
        "explicit_negative_high_obj",
        "high_obj_negative_vehicle_dominated",
    ):
        g = groups[name]
        print(
            f"{name:40s} n={g.get('n',0):8d} | "
            f"nearestMean_med={g.get('nearest_mean_dist',{}).get('median',float('nan')):.3f}m | "
            f"nearW_med={g.get('near_weight_frac',{}).get('median',float('nan')):.3f} | "
            f"farW_med={g.get('far_weight_frac',{}).get('median',float('nan')):.3f} | "
            f"farLargeW_med={g.get('far_large_weight_frac',{}).get('median',float('nan')):.3f} | "
            f"sigmaMajorW_med={g.get('sigma_major_weighted',{}).get('median',float('nan')):.3f}m | "
            f"farMaha<=3W_med={g.get('far_maha_le3_weight_frac',{}).get('median',float('nan')):.3f}"
        )

    print("\n================ SIGMA-ONLY BLACKBOX ================")
    base_neg = sigma_report[str(1.0)]["negative_pass_rate"]
    base_pos = sigma_report[str(1.0)]["positive_pass_rate"]
    for s in sigma_scales:
        r = sigma_report[str(s)]
        neg_rate = r["negative_pass_rate"]
        pos_rate = r["positive_pass_rate"]
        print(
            f"sigma x{s:<4g} | "
            f"pos_pass={pos_rate:.4f} "
            f"({pos_rate-base_pos:+.4f}) | "
            f"neg_pass={neg_rate:.4f} "
            f"({neg_rate-base_neg:+.4f})"
        )

    print("\nQuick reading:")
    print(
        f"- Mean issue indicator: look at nearestMean and nearW for high-obj negatives "
        f"(near < {args.near_m}m)."
    )
    print(
        f"- Large ellipse indicator: look at farLargeW / farMaha<=3W "
        f"(far > {args.far_m}m, large sigma >= {args.large_sigma_m}m)."
    )
    print(
        "- Strongest covariance evidence: shrinking sigma greatly lowers negative "
        "pass rate while positive pass rate changes little."
    )

    print(f"\nsaved: {json_path}")
    if top_rows:
        print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()

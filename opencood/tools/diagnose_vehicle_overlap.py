#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vehicle dense-region Gaussian-overlap diagnostic.

Purpose
-------
After changing Gaussian->BEV splatting to the correct direct weighted sum

    BEV(p) = sum_i exp(-0.5 * d_i(p)^2) * feature_i

diagnose whether remaining vehicle-region false-positive candidates are associated
with:
  1) too many Gaussian contributors,
  2) too-large total Gaussian support (sum of weights),
  3) vehicle-source dominance,
  4) 3D->2D z-collapse,
  5) large BEV feature magnitude.

This script is READ-ONLY with respect to model behavior:
- no model/loss/postprocess modification
- no threshold/NMS modification
- no checkpoint writes

It hooks Stage-3, reconstructs the exact Gaussian-cell pairs used by the current
GaussianBEVSplat, verifies production BEV == direct weighted sum, and compares
positive anchors vs explicit-negative anchors.

Optional:
If --fp_csv is provided (from diagnose_fp_source.py), actual final TP/FP rows
are enriched with local overlap statistics as well.

Example
-------
cd /data2/home/suyi/AirV2X-Perception_gaussian

python opencood/tools/diagnose_vehicle_overlap.py \
  --model_dir /path/to/current/run \
  --epoch 2 \
  --num_frames 32 \
  --workers 4 \
  --obj_threshold 0.2

Optional final-box alignment:
  --fp_csv /path/to/fp_diag_epoch2/final_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ----------------------------------------------------------------------
# repo import
# ----------------------------------------------------------------------
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

METRICS = (
    "count_mean",
    "count_max",
    "weight_mean",
    "weight_max",
    "bev_rms_mean",
    "bev_rms_max",
    "z_range_max",
    "weighted_z_std_mean",
    "vehicle_weight_frac",
    "vehicle_count_frac",
    "vehicle_feat_rms_mean",
    "rsu_weight_frac",
    "drone_weight_frac",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--epoch", required=True, type=int)
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--obj_threshold", type=float, default=0.2)
    p.add_argument("--save_neg_topk_per_frame", type=int, default=1000)
    p.add_argument("--fp_csv", type=str, default=None)
    p.add_argument("--fp_csv_alpha", type=float, default=1.0)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def fnum(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def inum(x, default=-1):
    try:
        return int(float(x))
    except Exception:
        return default


def np_summary(x):
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


def torch_summary(x):
    if x.numel() == 0:
        return {"n": 0}
    x = x.detach().float()
    q = torch.quantile(
        x,
        torch.tensor([0.50, 0.90, 0.99], device=x.device, dtype=x.dtype),
    )
    return {
        "n": int(x.numel()),
        "mean": float(x.mean().item()),
        "median": float(q[0].item()),
        "p90": float(q[1].item()),
        "p99": float(q[2].item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def channel_rms(x):
    return torch.sqrt(torch.mean(x.float() ** 2, dim=-1).clamp_min(0.0))


def rankdata(x):
    """Average ranks, no scipy dependency."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty_like(xs, dtype=np.float64)
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and xs[j] == xs[i]:
            j += 1
        r = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = r
        i = j
    return ranks


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def find_target_dict(ego):
    for key in ("label_dict", "label_dict_single", "target_dict"):
        d = ego.get(key, None)
        if isinstance(d, Mapping) and "pos_equal_one" in d:
            return d
    if "pos_equal_one" in ego:
        return ego
    raise KeyError(
        "Cannot find target dict with pos_equal_one. "
        f"ego keys={list(ego.keys())[:50]}"
    )


def load_final_rows(path, alpha, start, end):
    if not path:
        return {}
    out = defaultdict(list)
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            frame = inum(row.get("frame"))
            if frame < start or frame >= end:
                continue
            if "alpha" in row:
                a = fnum(row.get("alpha"))
                if not np.isfinite(a) or abs(a - alpha) > 1e-6:
                    continue
            out[frame].append(row)
    return out


# ----------------------------------------------------------------------
# Stage3 hook
# ----------------------------------------------------------------------
class Stage3OverlapInspector:
    def __init__(self):
        self.last = None

    @torch.no_grad()
    def hook(self, module, inputs, output):
        gaussians_by_agent = inputs[0]
        out_gs, bev = output
        splat = module.splat

        # Reproduce Stage3 pooling order so every output Gaussian gets
        # an exact source-agent label.
        pooled = {
            agent: module.pool(gs)
            for agent, gs in gaussians_by_agent.items()
            if gs.n_gaussians > 0
        }
        if not pooled:
            self.last = None
            return

        merged_geom, _ = module._merge(pooled)
        if merged_geom.n_gaussians != out_gs.n_gaussians:
            raise RuntimeError("Stage3 pooled/output Gaussian count mismatch")
        if not torch.allclose(
            merged_geom.mean, out_gs.mean, atol=1e-5, rtol=1e-5
        ):
            err = float((merged_geom.mean - out_gs.mean).abs().max().item())
            raise RuntimeError(f"Stage3 merge order mismatch, max err={err}")

        source_ids = []
        for agent, gs in pooled.items():
            source_ids.append(
                torch.full(
                    (gs.n_gaussians,),
                    AGENT_TO_ID.get(str(agent).lower(), -1),
                    device=out_gs.mean.device,
                    dtype=torch.long,
                )
            )
        source_id = torch.cat(source_ids)

        B, C, ny, nx = bev.shape
        if B != 1:
            raise RuntimeError(f"Diagnostic requires batch_size=1, got {B}")

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

        # Same Mahalanobis^2 as production splat.
        cov = cov_xy[g_idx]
        a = cov[:, 0, 0]
        b = cov[:, 0, 1]
        c = cov[:, 1, 1]
        det = (a * c - b * b).clamp_min(1e-6)
        inv00 = c / det
        inv01 = -b / det
        inv11 = a / det
        dx, dy = delta[:, 0], delta[:, 1]
        d2 = dx * (inv00 * dx + inv01 * dy) + dy * (
            inv01 * dx + inv11 * dy
        )

        support = d2 <= float(splat.n_sigma) ** 2
        g_idx = g_idx[support]
        ix = ix[support]
        iy = iy[support]
        d2 = d2[support]
        weight = torch.exp(-0.5 * d2).float()

        if out_gs.batch_index is None:
            batch_index = torch.zeros(
                out_gs.n_gaussians,
                dtype=torch.long,
                device=device,
            )
        else:
            batch_index = out_gs.batch_index.long()

        flat = batch_index[g_idx] * (ny * nx) + iy * nx + ix
        ncell = B * ny * nx

        # total density
        count = torch.zeros(ncell, device=device)
        count.index_add_(0, flat, torch.ones_like(weight))

        weight_sum = torch.zeros(ncell, device=device)
        weight_sum.index_add_(0, flat, weight)

        # z-collapse statistics
        z = out_gs.mean[g_idx, 2].float()
        z_sum = torch.zeros(ncell, device=device)
        z2_sum = torch.zeros(ncell, device=device)
        z_sum.index_add_(0, flat, z)
        z2_sum.index_add_(0, flat, z * z)

        z_mean = torch.where(
            count > 0,
            z_sum / count.clamp_min(1.0),
            torch.zeros_like(z_sum),
        )
        z_var = torch.where(
            count > 0,
            z2_sum / count.clamp_min(1.0) - z_mean**2,
            torch.zeros_like(z_sum),
        ).clamp_min(0.0)
        z_std = torch.sqrt(z_var)

        z_min = torch.full((ncell,), float("inf"), device=device)
        z_max = torch.full((ncell,), float("-inf"), device=device)
        if not hasattr(z_min, "scatter_reduce_"):
            raise RuntimeError(
                "This diagnostic requires torch.Tensor.scatter_reduce_."
            )
        z_min.scatter_reduce_(0, flat, z, reduce="amin", include_self=True)
        z_max.scatter_reduce_(0, flat, z, reduce="amax", include_self=True)
        z_range = torch.where(
            count > 0, z_max - z_min, torch.zeros_like(z_max)
        )

        wz = torch.zeros(ncell, device=device)
        wz2 = torch.zeros(ncell, device=device)
        wz.index_add_(0, flat, weight * z)
        wz2.index_add_(0, flat, weight * z * z)
        weighted_z_mean = torch.where(
            weight_sum > 0,
            wz / weight_sum.clamp_min(1e-12),
            torch.zeros_like(wz),
        )
        weighted_z_var = torch.where(
            weight_sum > 0,
            wz2 / weight_sum.clamp_min(1e-12) - weighted_z_mean**2,
            torch.zeros_like(wz),
        ).clamp_min(0.0)
        weighted_z_std = torch.sqrt(weighted_z_var)

        # exact direct weighted feature sum
        weighted_feat = torch.zeros(
            (ncell, C), device=device, dtype=torch.float32
        )
        weighted_feat.index_add_(
            0,
            flat,
            out_gs.feature[g_idx].float() * weight.unsqueeze(-1),
        )

        bev_flat = (
            bev.detach()
            .float()
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(ncell, C)
        )
        bev_rms = channel_rms(bev_flat)

        denom = torch.sqrt(torch.mean(bev_flat**2)).clamp_min(1e-12)
        recon_rel_rms = float(
            (
                torch.sqrt(torch.mean((weighted_feat - bev_flat) ** 2))
                / denom
            ).item()
        )

        # source-specific density and contribution magnitude
        pair_source = source_id[g_idx]
        agent_maps = {}
        for agent, aid in AGENT_TO_ID.items():
            m = pair_source == aid

            a_count = torch.zeros(ncell, device=device)
            a_weight = torch.zeros(ncell, device=device)
            if bool(m.any()):
                a_count.index_add_(
                    0, flat[m], torch.ones_like(weight[m])
                )
                a_weight.index_add_(0, flat[m], weight[m])

            a_feat = torch.zeros(
                (ncell, C), device=device, dtype=torch.float32
            )
            if bool(m.any()):
                a_feat.index_add_(
                    0,
                    flat[m],
                    out_gs.feature[g_idx[m]].float()
                    * weight[m].unsqueeze(-1),
                )

            agent_maps[agent] = {
                "count": a_count,
                "weight": a_weight,
                "feat_rms": channel_rms(a_feat),
            }

        def hw(x):
            return x.view(B, ny, nx)[0].detach().cpu().numpy()

        maps = {
            "count": hw(count),
            "weight_sum": hw(weight_sum),
            "bev_rms": hw(bev_rms),
            "z_std": hw(z_std),
            "z_range": hw(z_range),
            "weighted_z_std": hw(weighted_z_std),
        }
        for agent in AGENTS:
            maps[f"{agent}_count"] = hw(agent_maps[agent]["count"])
            maps[f"{agent}_weight"] = hw(agent_maps[agent]["weight"])
            maps[f"{agent}_feat_rms"] = hw(
                agent_maps[agent]["feat_rms"]
            )

        nz = count > 0
        self.last = {
            "maps": maps,
            "ny": ny,
            "nx": nx,
            "xy_min": [
                float(xy_min[0].item()),
                float(xy_min[1].item()),
            ],
            "cell": [
                float(cell[0].item()),
                float(cell[1].item()),
            ],
            "n_gaussians": int(out_gs.n_gaussians),
            "n_pairs": int(weight.numel()),
            "reconstruction_rel_rms": recon_rel_rms,
            "global_count_nonzero": torch_summary(count[nz]),
            "global_weight_nonzero": torch_summary(weight_sum[nz]),
            "global_bev_rms_nonzero": torch_summary(bev_rms[nz]),
            "global_z_range_nonzero": torch_summary(z_range[nz]),
        }


# ----------------------------------------------------------------------
# map splat grid -> raw objectness grid
# ----------------------------------------------------------------------
def block_reduce(arr, ho, wo, mode):
    hs, ws = arr.shape
    if hs % ho or ws % wo:
        raise RuntimeError(
            f"Splat grid {hs}x{ws} is not integer-compatible "
            f"with obj grid {ho}x{wo}"
        )
    sy, sx = hs // ho, ws // wo
    x = arr.reshape(ho, sy, wo, sx)
    if mode == "mean":
        return x.mean(axis=(1, 3))
    if mode == "sum":
        return x.sum(axis=(1, 3))
    if mode == "max":
        return x.max(axis=(1, 3))
    raise KeyError(mode)


def build_obj_grid_metrics(capture, ho, wo):
    m = capture["maps"]
    out = {
        "count_mean": block_reduce(m["count"], ho, wo, "mean"),
        "count_max": block_reduce(m["count"], ho, wo, "max"),
        "weight_mean": block_reduce(
            m["weight_sum"], ho, wo, "mean"
        ),
        "weight_max": block_reduce(
            m["weight_sum"], ho, wo, "max"
        ),
        "bev_rms_mean": block_reduce(
            m["bev_rms"], ho, wo, "mean"
        ),
        "bev_rms_max": block_reduce(
            m["bev_rms"], ho, wo, "max"
        ),
        "z_range_max": block_reduce(
            m["z_range"], ho, wo, "max"
        ),
        "weighted_z_std_mean": block_reduce(
            m["weighted_z_std"], ho, wo, "mean"
        ),
    }

    total_w = block_reduce(m["weight_sum"], ho, wo, "sum")
    total_c = block_reduce(m["count"], ho, wo, "sum")

    for agent in AGENTS:
        aw = block_reduce(m[f"{agent}_weight"], ho, wo, "sum")
        ac = block_reduce(m[f"{agent}_count"], ho, wo, "sum")
        afr = block_reduce(
            m[f"{agent}_feat_rms"], ho, wo, "mean"
        )
        out[f"{agent}_weight_frac"] = aw / np.maximum(total_w, 1e-12)
        out[f"{agent}_count_frac"] = ac / np.maximum(total_c, 1e-12)
        out[f"{agent}_feat_rms_mean"] = afr

    return out


def group_report(data, mask, obj_threshold):
    rep = {"n": int(mask.sum())}
    if rep["n"] == 0:
        return rep
    rep["obj"] = np_summary(data["obj"][mask])
    rep["obj_pass_rate"] = float(
        np.mean(data["obj"][mask] > obj_threshold)
    )
    for name in METRICS:
        rep[name] = np_summary(data[name][mask])
    return rep


def quantile_bins(x, obj, obj_threshold, n_bins=5):
    x = np.asarray(x)
    obj = np.asarray(obj)
    m = np.isfinite(x) & np.isfinite(obj)
    x, obj = x[m], obj[m]
    if x.size == 0:
        return []
    q = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        lo, hi = float(q[i]), float(q[i + 1])
        mm = (x >= lo) & (x <= hi if i == n_bins - 1 else x < hi)
        if not np.any(mm):
            continue
        out.append(
            {
                "bin": i,
                "lo": lo,
                "hi": hi,
                "n": int(mm.sum()),
                "x_mean": float(x[mm].mean()),
                "obj_mean": float(obj[mm].mean()),
                "obj_median": float(np.quantile(obj[mm], 0.5)),
                "obj_p90": float(np.quantile(obj[mm], 0.9)),
                "obj_p99": float(np.quantile(obj[mm], 0.99)),
                "obj_pass_rate": float(
                    np.mean(obj[mm] > obj_threshold)
                ),
            }
        )
    return out


# ----------------------------------------------------------------------
# optional final TP/FP alignment
# ----------------------------------------------------------------------
def xy_to_cell(x, y, capture):
    xmin, ymin = capture["xy_min"]
    cx, cy = capture["cell"]
    ix = int(math.floor((x - xmin) / cx))
    iy = int(math.floor((y - ymin) / cy))
    ix = min(max(ix, 0), capture["nx"] - 1)
    iy = min(max(iy, 0), capture["ny"] - 1)
    return iy, ix


def patch(arr, iy, ix, r, mode):
    p = arr[
        max(0, iy-r):min(arr.shape[0], iy+r+1),
        max(0, ix-r):min(arr.shape[1], ix+r+1),
    ]
    if mode == "mean":
        return float(p.mean())
    if mode == "max":
        return float(p.max())
    if mode == "sum":
        return float(p.sum())
    raise KeyError(mode)


def enrich_final_row(row, capture):
    x, y = fnum(row.get("anchor_x")), fnum(row.get("anchor_y"))
    if not np.isfinite(x) or not np.isfinite(y):
        raise KeyError("fp_csv needs anchor_x and anchor_y")

    iy, ix = xy_to_cell(x, y, capture)
    m = capture["maps"]
    out = dict(row)

    for key in (
        "count", "weight_sum", "bev_rms", "z_range",
        "weighted_z_std",
        "vehicle_count", "vehicle_weight", "vehicle_feat_rms",
        "rsu_count", "rsu_weight",
        "drone_count", "drone_weight",
    ):
        out[f"{key}_center"] = float(m[key][iy, ix])

    out["vehicle_weight_frac_center"] = (
        out["vehicle_weight_center"]
        / max(out["weight_sum_center"], 1e-12)
    )

    for r in (1, 2, 4):
        for key in ("count", "weight_sum", "bev_rms", "z_range"):
            out[f"{key}_r{r}_mean"] = patch(
                m[key], iy, ix, r, "mean"
            )
            out[f"{key}_r{r}_max"] = patch(
                m[key], iy, ix, r, "max"
            )
        vw = patch(m["vehicle_weight"], iy, ix, r, "sum")
        tw = patch(m["weight_sum"], iy, ix, r, "sum")
        out[f"vehicle_weight_frac_r{r}"] = vw / max(tw, 1e-12)

    return out


def main():
    args = parse_args()
    start = args.start
    end = start + args.num_frames
    out_dir = args.out or os.path.join(
        args.model_dir,
        f"vehicle_overlap_diag_epoch{args.epoch}",
    )
    os.makedirs(out_dir, exist_ok=True)

    final_by_frame = load_final_rows(
        args.fp_csv, args.fp_csv_alpha, start, end
    )

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
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    _, model = train_utils.load_model(
        args.model_dir,
        model,
        args.epoch,
        start_from_best=False,
    )
    model.eval()

    inspector = Stage3OverlapInspector()
    handle = model.stage3.register_forward_hook(inspector.hook)

    store = defaultdict(list)
    top_neg_rows = []
    final_rows = []
    frame_reports = []

    try:
        for frame, batch_data in tqdm(
            enumerate(loader), total=min(len(loader), end)
        ):
            if frame < start:
                continue
            if frame >= end:
                break

            batch_data = train_utils.to_device(batch_data, device)
            inspector.last = None

            with torch.no_grad():
                output = model(batch_data["ego"])

            cap = inspector.last
            if cap is None:
                raise RuntimeError(f"frame {frame}: no Stage3 capture")

            # Important sanity check after the splat fix.
            if cap["reconstruction_rel_rms"] > 1e-5:
                raise RuntimeError(
                    f"frame {frame}: production splat differs from "
                    f"direct weighted sum, rel RMS="
                    f"{cap['reconstruction_rel_rms']:.3e}"
                )

            obj = torch.sigmoid(output["obj"]).permute(
                0, 2, 3, 1
            )[0].detach().cpu().numpy()
            ho, wo, A = obj.shape

            target = find_target_dict(batch_data["ego"])
            pos = (
                target["pos_equal_one"][0]
                .detach().cpu().numpy() > 0
            )
            neg = (
                target["neg_equal_one"][0]
                .detach().cpu().numpy() > 0
            )
            if pos.shape != obj.shape or neg.shape != obj.shape:
                raise RuntimeError(
                    f"target/obj mismatch: pos={pos.shape}, "
                    f"neg={neg.shape}, obj={obj.shape}"
                )
            ign = ~(pos | neg)

            spatial = build_obj_grid_metrics(cap, ho, wo)

            frame_data = {
                "obj": obj.reshape(-1).astype(np.float32),
                "pos": pos.reshape(-1),
                "neg": neg.reshape(-1),
                "ignore": ign.reshape(-1),
            }
            for name in METRICS:
                arr = spatial[name]
                frame_data[name] = np.repeat(
                    arr[..., None], A, axis=-1
                ).reshape(-1).astype(np.float32)

            for key in ("obj", "pos", "neg", "ignore") + METRICS:
                store[key].append(frame_data[key])

            # save only top-K negative anchors per frame
            neg_idx = np.flatnonzero(frame_data["neg"])
            if neg_idx.size:
                k = min(args.save_neg_topk_per_frame, neg_idx.size)
                vals = frame_data["obj"][neg_idx]
                kk = np.argpartition(vals, -k)[-k:]
                idxs = neg_idx[kk]
                idxs = idxs[
                    np.argsort(frame_data["obj"][idxs])[::-1]
                ]
                for idx in idxs:
                    hw = idx // A
                    row = {
                        "frame": frame,
                        "obj_y": int(hw // wo),
                        "obj_x": int(hw % wo),
                        "anchor_slot": int(idx % A),
                        "obj": float(frame_data["obj"][idx]),
                        "obj_pass": int(
                            frame_data["obj"][idx]
                            > args.obj_threshold
                        ),
                    }
                    for name in METRICS:
                        row[name] = float(frame_data[name][idx])
                    top_neg_rows.append(row)

            if frame in final_by_frame:
                for row in final_by_frame[frame]:
                    final_rows.append(enrich_final_row(row, cap))

            frame_reports.append(
                {
                    "frame": frame,
                    "n_gaussians": cap["n_gaussians"],
                    "n_pairs": cap["n_pairs"],
                    "reconstruction_rel_rms":
                        cap["reconstruction_rel_rms"],
                    "n_pos": int(pos.sum()),
                    "n_neg": int(neg.sum()),
                    "n_ignore": int(ign.sum()),
                    "neg_obj_pass": int(
                        (neg & (obj > args.obj_threshold)).sum()
                    ),
                    "global_count_nonzero":
                        cap["global_count_nonzero"],
                    "global_weight_nonzero":
                        cap["global_weight_nonzero"],
                    "global_bev_rms_nonzero":
                        cap["global_bev_rms_nonzero"],
                    "global_z_range_nonzero":
                        cap["global_z_range_nonzero"],
                }
            )

            print(
                f"[frame {frame:04d}] "
                f"G={cap['n_gaussians']} "
                f"pairs={cap['n_pairs']} "
                f"neg>{args.obj_threshold:g}="
                f"{int((neg & (obj > args.obj_threshold)).sum())} "
                f"recon={cap['reconstruction_rel_rms']:.2e}"
            )

    finally:
        handle.remove()

    data = {
        key: np.concatenate(value, axis=0)
        for key, value in store.items()
    }

    pos = data["pos"].astype(bool)
    neg = data["neg"].astype(bool)
    ign = data["ignore"].astype(bool)
    high_neg = neg & (data["obj"] > args.obj_threshold)

    groups = {
        "positive": group_report(
            data, pos, args.obj_threshold
        ),
        "explicit_negative_all": group_report(
            data, neg, args.obj_threshold
        ),
        "explicit_negative_high_obj": group_report(
            data, high_neg, args.obj_threshold
        ),
        "ignore": group_report(
            data, ign, args.obj_threshold
        ),
    }

    # Define "dense" only relative to explicit negatives.
    dense_cut = float(
        np.quantile(data["count_mean"][neg], 0.90)
    )
    dense = data["count_mean"] >= dense_cut
    vehicle_dom = data["vehicle_weight_frac"] >= 0.5

    key_masks = {
        "neg_vehicle_dense":
            neg & vehicle_dom & dense,
        "neg_vehicle_nondense":
            neg & vehicle_dom & (~dense),
        "neg_nonvehicle_dense":
            neg & (~vehicle_dom) & dense,
        "neg_nonvehicle_nondense":
            neg & (~vehicle_dom) & (~dense),
    }
    key_comparison = {
        name: group_report(data, mask, args.obj_threshold)
        for name, mask in key_masks.items()
    }

    # Objectness correlations inside explicit negatives only.
    correlations = {}
    for name in METRICS:
        correlations[name] = {
            "pearson": pearson(
                data["obj"][neg], data[name][neg]
            ),
            "spearman": spearman(
                data["obj"][neg], data[name][neg]
            ),
        }

    bins = {}
    for name in (
        "count_mean",
        "weight_mean",
        "vehicle_weight_frac",
        "bev_rms_mean",
        "z_range_max",
    ):
        bins[name] = quantile_bins(
            data[name][neg],
            data["obj"][neg],
            args.obj_threshold,
            n_bins=5,
        )

    baseline_pass = groups[
        "explicit_negative_all"
    ].get("obj_pass_rate", float("nan"))
    vehicle_dense_pass = key_comparison[
        "neg_vehicle_dense"
    ].get("obj_pass_rate", float("nan"))

    flags = {
        "production_splat_matches_direct_weighted_sum":
            max(
                x["reconstruction_rel_rms"]
                for x in frame_reports
            ) < 1e-5,
        "vehicle_dense_negatives_have_higher_pass_rate":
            np.isfinite(vehicle_dense_pass)
            and vehicle_dense_pass > baseline_pass,
        "vehicle_dense_pass_rate_over_2x_baseline":
            np.isfinite(vehicle_dense_pass)
            and baseline_pass > 0
            and vehicle_dense_pass > 2 * baseline_pass,
        "obj_correlates_with_count_spearman_gt_0p10":
            correlations["count_mean"]["spearman"] > 0.10,
        "obj_correlates_with_weight_spearman_gt_0p10":
            correlations["weight_mean"]["spearman"] > 0.10,
        "obj_correlates_with_bev_rms_spearman_gt_0p10":
            correlations["bev_rms_mean"]["spearman"] > 0.10,
    }

    summary = {
        "model_dir": args.model_dir,
        "epoch": args.epoch,
        "obj_threshold": args.obj_threshold,
        "num_frames_processed": len(frame_reports),
        "dense_definition": {
            "metric": "count_mean",
            "reference": "explicit-negative anchors",
            "quantile": 0.90,
            "threshold": dense_cut,
        },
        "groups": groups,
        "key_vehicle_density_comparison": key_comparison,
        "explicit_negative_obj_correlations": correlations,
        "explicit_negative_quintile_bins": bins,
        "diagnostic_flags": flags,
        "per_frame": frame_reports,
    }

    # Optional actual final predictions.
    if final_rows:
        def final_mask(row, kind):
            if kind == "tp":
                return inum(row.get("is_tp_iou03")) == 1
            if kind == "fp":
                return inum(row.get("is_fp_iou03")) == 1
            if kind == "fp_explicit_negative":
                return (
                    inum(row.get("is_fp_iou03")) == 1
                    and row.get("anchor_source")
                    == "explicit_negative"
                )
            return False

        summary["final_prediction_groups"] = {}
        for kind in ("tp", "fp", "fp_explicit_negative"):
            rows = [r for r in final_rows if final_mask(r, kind)]
            rep = {"n": len(rows)}
            for key in (
                "count_center",
                "weight_sum_center",
                "bev_rms_center",
                "z_range_center",
                "weighted_z_std_center",
                "vehicle_weight_frac_center",
                "count_r2_mean",
                "count_r2_max",
                "weight_sum_r2_mean",
                "weight_sum_r2_max",
                "bev_rms_r2_mean",
                "bev_rms_r2_max",
                "vehicle_weight_frac_r2",
            ):
                rep[key] = np_summary(
                    [fnum(r.get(key)) for r in rows]
                )
            summary["final_prediction_groups"][kind] = rep

    summary_path = os.path.join(
        out_dir, "overlap_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            summary, f, indent=2,
            ensure_ascii=False, allow_nan=True
        )

    neg_csv = os.path.join(
        out_dir, "high_obj_negative_anchors.csv"
    )
    if top_neg_rows:
        with open(neg_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(top_neg_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(top_neg_rows)

    final_csv = None
    if final_rows:
        final_csv = os.path.join(
            out_dir, "final_prediction_overlap_stats.csv"
        )
        with open(final_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(final_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(final_rows)

    # concise terminal report
    print("\n================ OVERLAP DIAGNOSTIC ================")
    print(
        "max direct-sum reconstruction rel RMS:",
        f"{max(x['reconstruction_rel_rms'] for x in frame_reports):.3e}"
    )
    print(f"negative dense cutoff (p90 count_mean): {dense_cut:.3f}")

    def show(name, rep):
        print(
            f"{name:24s} "
            f"n={rep.get('n', 0):8d} "
            f"pass={rep.get('obj_pass_rate', float('nan')):.4f} "
            f"obj50={rep.get('obj',{}).get('median',float('nan')):.4f} "
            f"cnt50={rep.get('count_mean',{}).get('median',float('nan')):.2f} "
            f"w50={rep.get('weight_mean',{}).get('median',float('nan')):.3f} "
            f"vehW50={rep.get('vehicle_weight_frac',{}).get('median',float('nan')):.3f} "
            f"bev50={rep.get('bev_rms_mean',{}).get('median',float('nan')):.3f} "
            f"zR50={rep.get('z_range_max',{}).get('median',float('nan')):.3f}"
        )

    show("positive", groups["positive"])
    show("negative all", groups["explicit_negative_all"])
    show("negative high-obj", groups["explicit_negative_high_obj"])
    print()
    for name, rep in key_comparison.items():
        show(name, rep)

    print("\nSpearman(obj, metric) inside explicit negatives:")
    for name in (
        "count_mean",
        "weight_mean",
        "vehicle_weight_frac",
        "bev_rms_mean",
        "z_range_max",
        "weighted_z_std_mean",
    ):
        print(
            f"  {name:24s}: "
            f"{correlations[name]['spearman']:+.4f}"
        )

    print("\nflags:")
    for key, value in flags.items():
        print(f"  {key}: {value}")

    print(f"\nsaved: {summary_path}")
    if top_neg_rows:
        print(f"saved: {neg_csv}")
    if final_csv:
        print(f"saved: {final_csv}")


if __name__ == "__main__":
    main()

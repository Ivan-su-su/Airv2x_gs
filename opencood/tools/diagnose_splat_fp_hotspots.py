#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose whether Gaussian->BEV splat normalization turns low-weight Gaussian tails
into full-strength BEV features at final TP/FP anchor locations.

IMPORTANT:
- Pure diagnostic: does NOT modify model outputs, loss, checkpoint, or post-processing.
- Uses a forward hook on GaussianBEVSplat and recomputes the SAME support/weights only
  to inspect weight_sum / contributor_count / pre-normalization feature RMS.
- Best used with the final_predictions.csv produced by diagnose_fp_source.py.

What it measures
----------------
For each BEV cell:
    weighted_sum = sum_i w_i * f_i
    weight_sum   = sum_i w_i
    production   = weighted_sum / weight_sum

If a cell has only one contributor:
    production = f_i
regardless of whether w_i is 1.0 or 0.01.

The script correlates these quantities with final TP / FP anchor locations and
reports recurring FP hotspots.

Outputs
-------
<splat_diag_dir>/
    splat_fp_cell_stats.csv
    splat_diag_summary.json

Example
-------
cd /data2/home/suyi/AirV2X-Perception_gaussian

python opencood/tools/diagnose_splat_fp_hotspots.py \
  --model_dir /mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/airv2x_gaussian_0822_camera/v45_scale_asym_2026_09_13_02_48_48 \
  --epoch 16 \
  --fp_csv /mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/airv2x_gaussian_0822_camera/v45_scale_asym_2026_09_13_02_48_48/fp_diag_epoch16/final_predictions.csv \
  --alpha 1.0 \
  --num_frames 32 \
  --workers 4
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# -----------------------------------------------------------------------------
# repo import
# -----------------------------------------------------------------------------
ROOT = os.path.abspath(__file__)
_candidate = "/".join(ROOT.split("/")[:-3])
if os.path.isdir(os.path.join(_candidate, "opencood")):
    sys.path.insert(0, _candidate)
else:
    sys.path.insert(0, os.getcwd())

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


# -----------------------------------------------------------------------------
# args
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--fp_csv", required=True,
                   help="final_predictions.csv from diagnose_fp_source.py")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Which alpha rows to use from fp_csv; usually 1.0")
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patch_radii", type=str, default="0,1,2,4",
                   help="Splat-cell radii around anchor center for local statistics")
    p.add_argument("--top_hotspots", type=int, default=30)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def _int(v):
    try:
        return int(float(v))
    except Exception:
        return -1


def finite_summary(values):
    a = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.quantile(a, 0.50)),
        "p90": float(np.quantile(a, 0.90)),
        "p99": float(np.quantile(a, 0.99)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def load_fp_rows(path, alpha, start, end):
    by_frame = defaultdict(list)
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fr = _int(row.get("frame"))
            a = _float(row.get("alpha"))
            if fr < start or fr >= end:
                continue
            if not np.isfinite(a) or abs(a - alpha) > 1e-6:
                continue
            by_frame[fr].append(row)
    return by_frame


def find_splat_module(model):
    # Preferred current path.
    if hasattr(model, "stage3") and hasattr(model.stage3, "splat"):
        return model.stage3.splat, "model.stage3.splat"

    hits = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "GaussianBEVSplat":
            hits.append((name, module))
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one GaussianBEVSplat, found {[(n, type(m).__name__) for n,m in hits]}"
        )
    return hits[0][1], hits[0][0]


def rms_channels(x):
    # x [..., C] -> [...]
    return torch.sqrt(torch.mean(x.float() * x.float(), dim=-1).clamp_min(0.0))


# -----------------------------------------------------------------------------
# diagnostic hook
# -----------------------------------------------------------------------------
class SplatInspector:
    """
    Forward hook: leaves output untouched, recomputes exact splat bookkeeping
    from GaussianSet input, and stores only scalar BEV maps on CPU.
    """

    def __init__(self):
        self.last = None
        self.frame_stats = []

    @torch.no_grad()
    def hook(self, module, inputs, output):
        if len(inputs) < 1:
            raise RuntimeError("GaussianBEVSplat hook did not receive GaussianSet input")

        gs = inputs[0]
        n = int(gs.n_gaussians)
        if n == 0:
            self.last = None
            return

        device = gs.mean.device
        dtype = gs.feature.dtype
        ny, nx = int(output.shape[-2]), int(output.shape[-1])

        # EXACT same geometry/weighting contract as GaussianBEVSplat.forward.
        cov_xy = gs.covariance()[:, :2, :2]
        cov_xy = cov_xy + torch.eye(2, device=device, dtype=dtype) * 1.0e-4

        cell = module.cell.to(device=device, dtype=dtype)
        xy_min = module.xy_min.to(device=device, dtype=dtype)

        var_xy = cov_xy.diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
        radius = torch.clamp(
            torch.ceil(module.n_sigma * var_xy.sqrt() / cell).long(), min=1
        )

        mean_xy = gs.mean[:, :2]
        center_idx = torch.floor((mean_xy - xy_min) / cell).long()

        g_idx, ix, iy, delta = module._candidate_pairs(
            center_idx, radius, mean_xy, xy_min, cell, nx, ny
        )

        if g_idx.numel() == 0:
            self.last = None
            return

        # Local copy of exact 2x2 Mahalanobis formula used in repository.
        cov = cov_xy[g_idx]
        a = cov[:, 0, 0]
        b = cov[:, 0, 1]
        c = cov[:, 1, 1]
        det = (a * c - b * b).clamp_min(1.0e-6)
        inv00 = c / det
        inv01 = -b / det
        inv11 = a / det
        dx = delta[:, 0]
        dy = delta[:, 1]
        d2 = dx * (inv00 * dx + inv01 * dy) + dy * (inv01 * dx + inv11 * dy)

        support = d2 <= (float(module.n_sigma) ** 2)
        g_idx = g_idx[support]
        ix = ix[support]
        iy = iy[support]
        d2 = d2[support]
        weight = torch.exp(-0.5 * d2)

        if gs.batch_index is None:
            batch_index = torch.zeros(n, dtype=torch.long, device=device)
            n_batch = 1
        else:
            batch_index = gs.batch_index.long()
            n_batch = int(batch_index.max().item()) + 1

        flat = batch_index[g_idx] * (ny * nx) + iy * nx + ix
        ncell = n_batch * ny * nx

        # Scalar support maps.
        weight_sum = torch.zeros(ncell, device=device, dtype=torch.float32)
        weight_sum.index_add_(0, flat, weight.float())

        sum_w2 = torch.zeros(ncell, device=device, dtype=torch.float32)
        sum_w2.index_add_(0, flat, weight.float() ** 2)

        count = torch.zeros(ncell, device=device, dtype=torch.float32)
        count.index_add_(0, flat, torch.ones_like(weight, dtype=torch.float32))

        # Weighted feature sum BEFORE normalization.
        feat = gs.feature[g_idx].float()
        weighted_sum = torch.zeros(
            (ncell, feat.shape[-1]), device=device, dtype=torch.float32
        )
        weighted_sum.index_add_(0, flat, feat * weight.float().unsqueeze(-1))
        pre_rms = rms_channels(weighted_sum)

        # Production splat output, exactly as consumed by the BEV backbone.
        out_flat = (
            output.detach()
            .float()
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(ncell, -1)
        )
        out_rms = rms_channels(out_flat)

        # Reconstruct production output to verify bookkeeping is correct.
        recon = torch.where(
            (weight_sum > 0).unsqueeze(-1),
            weighted_sum / weight_sum.clamp_min(1.0e-8).unsqueeze(-1),
            weighted_sum,
        )
        denom = torch.sqrt(torch.mean(out_flat * out_flat)).clamp_min(1e-12)
        recon_rel_rms = float(
            (
                torch.sqrt(torch.mean((recon - out_flat) ** 2)) / denom
            ).item()
        )

        # Max individual Gaussian weight per cell.
        max_weight = torch.zeros(ncell, device=device, dtype=torch.float32)
        if hasattr(max_weight, "scatter_reduce_"):
            max_weight.scatter_reduce_(
                0, flat, weight.float(), reduce="amax", include_self=True
            )
        else:
            # Slow fallback only for old torch.
            for idx_val, w_val in zip(flat.tolist(), weight.float().tolist()):
                if w_val > float(max_weight[idx_val]):
                    max_weight[idx_val] = w_val

        effective_count = torch.where(
            sum_w2 > 0,
            weight_sum * weight_sum / sum_w2.clamp_min(1e-12),
            torch.zeros_like(weight_sum),
        )

        # The exact scalar amplification due to normalization whenever wsum>0:
        # ||weighted_sum / wsum|| / ||weighted_sum|| = 1 / wsum.
        norm_gain = torch.where(
            weight_sum > 0,
            1.0 / weight_sum.clamp_min(1e-12),
            torch.zeros_like(weight_sum),
        )

        # reshape [B,H,W]; batch=1 for val/test in this project.
        def hw(x):
            return x.view(n_batch, ny, nx)[0].detach().cpu().numpy()

        maps = {
            "weight_sum": hw(weight_sum),
            "count": hw(count),
            "sum_w2": hw(sum_w2),
            "max_weight": hw(max_weight),
            "effective_count": hw(effective_count),
            "pre_rms": hw(pre_rms),
            "out_rms": hw(out_rms),
            "norm_gain": hw(norm_gain),
        }

        nz = weight_sum > 0
        singleton = nz & (count == 1)
        singleton_low_01 = singleton & (weight_sum < 0.1)
        singleton_low_005 = singleton & (weight_sum < 0.05)
        singleton_low_001 = singleton & (weight_sum < 0.01)

        def frac(mask, base):
            den = int(base.sum().item())
            if den == 0:
                return 0.0
            return float(mask.sum().item() / den)

        frame_summary = {
            "n_gaussians": n,
            "n_nonzero_cells": int(nz.sum().item()),
            "nonzero_cell_fraction": float(nz.float().mean().item()),
            "reconstruction_rel_rms": recon_rel_rms,
            "weight_sum_nonzero": tensor_summary(weight_sum[nz]),
            "count_nonzero": tensor_summary(count[nz]),
            "norm_gain_nonzero": tensor_summary(norm_gain[nz]),
            "singleton_fraction_of_nonzero": frac(singleton, nz),
            "singleton_wsum_lt_0p1_fraction_of_nonzero": frac(singleton_low_01, nz),
            "singleton_wsum_lt_0p05_fraction_of_nonzero": frac(singleton_low_005, nz),
            "singleton_wsum_lt_0p01_fraction_of_nonzero": frac(singleton_low_001, nz),
            "radius_x": tensor_summary(radius[:, 0].float()),
            "radius_y": tensor_summary(radius[:, 1].float()),
        }

        # Global support buckets: directly exposes tail normalization behavior.
        bucket_edges = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, float("inf")]
        bucket_stats = []
        for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
            m = nz & (weight_sum >= lo) & (weight_sum < hi)
            if not bool(m.any()):
                continue
            bucket_stats.append({
                "lo": lo,
                "hi": None if math.isinf(hi) else hi,
                "n": int(m.sum().item()),
                "weight_sum_mean": float(weight_sum[m].mean().item()),
                "contrib_count_mean": float(count[m].mean().item()),
                "pre_rms_mean": float(pre_rms[m].mean().item()),
                "out_rms_mean": float(out_rms[m].mean().item()),
                "norm_gain_mean": float(norm_gain[m].mean().item()),
                "max_weight_mean": float(max_weight[m].mean().item()),
            })
        frame_summary["support_buckets"] = bucket_stats

        self.last = {
            "maps": maps,
            "cell": [float(cell[0].item()), float(cell[1].item())],
            "xy_min": [float(xy_min[0].item()), float(xy_min[1].item())],
            "shape": [ny, nx],
            "summary": frame_summary,
        }


def tensor_summary(x):
    if x.numel() == 0:
        return {"n": 0}
    x = x.detach().float()
    q = torch.quantile(x, torch.tensor([0.5, 0.9, 0.99], device=x.device))
    return {
        "n": int(x.numel()),
        "mean": float(x.mean().item()),
        "median": float(q[0].item()),
        "p90": float(q[1].item()),
        "p99": float(q[2].item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


# -----------------------------------------------------------------------------
# prediction -> splat-cell mapping
# -----------------------------------------------------------------------------
def anchor_xy_to_cell(anchor_x, anchor_y, xy_min, cell, ny, nx):
    ix = int(math.floor((anchor_x - xy_min[0]) / cell[0]))
    iy = int(math.floor((anchor_y - xy_min[1]) / cell[1]))
    ix = min(max(ix, 0), nx - 1)
    iy = min(max(iy, 0), ny - 1)
    return iy, ix


def patch_stats(arr, iy, ix, r):
    y0 = max(0, iy - r)
    y1 = min(arr.shape[0], iy + r + 1)
    x0 = max(0, ix - r)
    x1 = min(arr.shape[1], ix + r + 1)
    p = np.asarray(arr[y0:y1, x0:x1], dtype=np.float64)
    if p.size == 0:
        return {"mean": float("nan"), "max": float("nan"), "min": float("nan"),
                "median": float("nan"), "nonzero_frac": 0.0}
    return {
        "mean": float(np.mean(p)),
        "max": float(np.max(p)),
        "min": float(np.min(p)),
        "median": float(np.median(p)),
        "nonzero_frac": float(np.mean(p > 0)),
    }


def enrich_prediction_row(row, capture, radii):
    maps = capture["maps"]
    ny, nx = capture["shape"]
    xy_min = capture["xy_min"]
    cell = capture["cell"]

    ax = _float(row.get("anchor_x"))
    ay = _float(row.get("anchor_y"))
    iy, ix = anchor_xy_to_cell(ax, ay, xy_min, cell, ny, nx)

    out = dict(row)
    out["splat_iy"] = iy
    out["splat_ix"] = ix

    center_metrics = [
        "weight_sum", "count", "max_weight", "effective_count",
        "pre_rms", "out_rms", "norm_gain",
    ]
    for key in center_metrics:
        out[f"splat_{key}_center"] = float(maps[key][iy, ix])

    # Direct flags for the suspected pathology.
    w = out["splat_weight_sum_center"]
    c = out["splat_count_center"]
    out["splat_singleton_center"] = int(abs(c - 1.0) < 0.5)
    out["splat_low_support_lt_0p1_center"] = int(w > 0 and w < 0.1)
    out["splat_low_support_lt_0p05_center"] = int(w > 0 and w < 0.05)
    out["splat_low_support_singleton_lt_0p1_center"] = int(
        out["splat_singleton_center"] and out["splat_low_support_lt_0p1_center"]
    )

    for r in radii:
        for key in center_metrics:
            ps = patch_stats(maps[key], iy, ix, r)
            out[f"splat_{key}_r{r}_mean"] = ps["mean"]
            out[f"splat_{key}_r{r}_max"] = ps["max"]
            if key == "weight_sum":
                out[f"splat_nonzero_frac_r{r}"] = ps["nonzero_frac"]

    return out


# -----------------------------------------------------------------------------
# aggregation
# -----------------------------------------------------------------------------
def group_summary(rows, predicate):
    selected = [r for r in rows if predicate(r)]
    keys = [
        "splat_weight_sum_center",
        "splat_count_center",
        "splat_max_weight_center",
        "splat_effective_count_center",
        "splat_pre_rms_center",
        "splat_out_rms_center",
        "splat_norm_gain_center",
        "obj",
    ]
    out = {"n": len(selected)}
    for k in keys:
        out[k] = finite_summary([_float(r.get(k)) for r in selected])

    if selected:
        out["singleton_fraction"] = float(np.mean([
            _int(r.get("splat_singleton_center")) == 1 for r in selected
        ]))
        out["low_support_lt_0p1_fraction"] = float(np.mean([
            _int(r.get("splat_low_support_lt_0p1_center")) == 1 for r in selected
        ]))
        out["low_support_singleton_lt_0p1_fraction"] = float(np.mean([
            _int(r.get("splat_low_support_singleton_lt_0p1_center")) == 1
            for r in selected
        ]))
    return out


def hotspot_summary(rows, topk):
    fp_rows = [
        r for r in rows
        if _int(r.get("is_fp_iou03")) == 1
        and r.get("anchor_source") == "explicit_negative"
    ]
    by_anchor = defaultdict(list)
    for r in fp_rows:
        by_anchor[_int(r.get("anchor_id"))].append(r)

    ranked = sorted(by_anchor.items(), key=lambda kv: len(kv[1]), reverse=True)[:topk]
    out = []
    for aid, rr in ranked:
        out.append({
            "anchor_id": aid,
            "fp_occurrences": len(rr),
            "frames": sorted({_int(x.get("frame")) for x in rr}),
            "anchor_x_mean": float(np.mean([_float(x.get("anchor_x")) for x in rr])),
            "anchor_y_mean": float(np.mean([_float(x.get("anchor_y")) for x in rr])),
            "obj": finite_summary([_float(x.get("obj")) for x in rr]),
            "weight_sum_center": finite_summary(
                [_float(x.get("splat_weight_sum_center")) for x in rr]
            ),
            "count_center": finite_summary(
                [_float(x.get("splat_count_center")) for x in rr]
            ),
            "max_weight_center": finite_summary(
                [_float(x.get("splat_max_weight_center")) for x in rr]
            ),
            "pre_rms_center": finite_summary(
                [_float(x.get("splat_pre_rms_center")) for x in rr]
            ),
            "out_rms_center": finite_summary(
                [_float(x.get("splat_out_rms_center")) for x in rr]
            ),
            "norm_gain_center": finite_summary(
                [_float(x.get("splat_norm_gain_center")) for x in rr]
            ),
            "singleton_fraction": float(np.mean([
                _int(x.get("splat_singleton_center")) == 1 for x in rr
            ])),
            "low_support_singleton_lt_0p1_fraction": float(np.mean([
                _int(x.get("splat_low_support_singleton_lt_0p1_center")) == 1
                for x in rr
            ])),
        })
    return out


def aggregate_support_buckets(frame_summaries):
    # Merge buckets by textual interval.
    acc = defaultdict(lambda: {
        "n": 0,
        "weight_sum_sum": 0.0,
        "contrib_count_sum": 0.0,
        "pre_rms_sum": 0.0,
        "out_rms_sum": 0.0,
        "norm_gain_sum": 0.0,
        "max_weight_sum": 0.0,
    })
    interval = {}
    for fs in frame_summaries:
        for b in fs.get("support_buckets", []):
            key = (b["lo"], b["hi"])
            n = b["n"]
            interval[key] = {"lo": b["lo"], "hi": b["hi"]}
            a = acc[key]
            a["n"] += n
            a["weight_sum_sum"] += b["weight_sum_mean"] * n
            a["contrib_count_sum"] += b["contrib_count_mean"] * n
            a["pre_rms_sum"] += b["pre_rms_mean"] * n
            a["out_rms_sum"] += b["out_rms_mean"] * n
            a["norm_gain_sum"] += b["norm_gain_mean"] * n
            a["max_weight_sum"] += b["max_weight_mean"] * n

    out = []
    for key in sorted(acc.keys(), key=lambda z: z[0]):
        a = acc[key]
        n = max(a["n"], 1)
        out.append({
            **interval[key],
            "n": a["n"],
            "weight_sum_mean": a["weight_sum_sum"] / n,
            "contrib_count_mean": a["contrib_count_sum"] / n,
            "pre_rms_mean": a["pre_rms_sum"] / n,
            "out_rms_mean": a["out_rms_sum"] / n,
            "norm_gain_mean": a["norm_gain_sum"] / n,
            "max_weight_mean": a["max_weight_sum"] / n,
        })
    return out


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    radii = sorted({
        int(x.strip()) for x in args.patch_radii.split(",") if x.strip()
    })
    end = args.start + args.num_frames

    fp_by_frame = load_fp_rows(args.fp_csv, args.alpha, args.start, end)
    if not fp_by_frame:
        raise RuntimeError(
            f"No rows found in {args.fp_csv} for alpha={args.alpha}, "
            f"frames [{args.start},{end})"
        )

    out_dir = args.out or os.path.join(
        args.model_dir, f"splat_fp_diag_epoch{args.epoch}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Load exact checkpoint config.
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

    splat, splat_name = find_splat_module(model)
    inspector = SplatInspector()
    handle = splat.register_forward_hook(inspector.hook)

    print(f"[diag] splat module: {splat_name}")
    print(f"[diag] fp csv alpha filter: {args.alpha}")
    print(f"[diag] frames: [{args.start}, {end})")
    print(f"[diag] patch radii: {radii}")

    enriched_rows = []
    frame_summaries = []

    try:
        for i, batch_data in tqdm(enumerate(loader), total=min(len(loader), end)):
            if i < args.start:
                continue
            if i >= end:
                break
            if i not in fp_by_frame:
                continue

            batch_data = train_utils.to_device(batch_data, device)
            inspector.last = None

            with torch.no_grad():
                _ = model(batch_data["ego"])

            if inspector.last is None:
                raise RuntimeError(f"Frame {i}: splat hook did not capture anything")

            capture = inspector.last
            fs = dict(capture["summary"])
            fs["frame"] = i
            frame_summaries.append(fs)

            # Strong sanity check: our recomputation should reconstruct production splat.
            if fs["reconstruction_rel_rms"] > 1e-5:
                print(
                    f"[WARN frame {i}] splat reconstruction rel RMS "
                    f"{fs['reconstruction_rel_rms']:.3e} > 1e-5"
                )

            for row in fp_by_frame[i]:
                enriched_rows.append(
                    enrich_prediction_row(row, capture, radii)
                )

            print(
                f"[frame {i:04d}] "
                f"nonzero={fs['n_nonzero_cells']} "
                f"single={fs['singleton_fraction_of_nonzero']:.3f} "
                f"single_w<.1={fs['singleton_wsum_lt_0p1_fraction_of_nonzero']:.3f} "
                f"recon_err={fs['reconstruction_rel_rms']:.2e}"
            )

    finally:
        handle.remove()

    if not enriched_rows:
        raise RuntimeError("No enriched prediction rows produced")

    # Save detailed rows.
    csv_path = os.path.join(out_dir, "splat_fp_cell_stats.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(enriched_rows[0].keys()))
        writer.writeheader()
        writer.writerows(enriched_rows)

    tp_pred = lambda r: _int(r.get("is_tp_iou03")) == 1
    fp_exp = lambda r: (
        _int(r.get("is_fp_iou03")) == 1
        and r.get("anchor_source") == "explicit_negative"
    )
    fp_any = lambda r: _int(r.get("is_fp_iou03")) == 1

    summary = {
        "model_dir": args.model_dir,
        "epoch": args.epoch,
        "fp_csv": args.fp_csv,
        "alpha": args.alpha,
        "frames_requested": args.num_frames,
        "frames_captured": len(frame_summaries),
        "splat_module": splat_name,
        "patch_radii": radii,
        "groups": {
            "tp": group_summary(enriched_rows, tp_pred),
            "fp_all": group_summary(enriched_rows, fp_any),
            "fp_explicit_negative": group_summary(enriched_rows, fp_exp),
        },
        "top_recurring_explicit_negative_fp_hotspots": hotspot_summary(
            enriched_rows, args.top_hotspots
        ),
        "splat_global": {
            "reconstruction_rel_rms": finite_summary(
                [x["reconstruction_rel_rms"] for x in frame_summaries]
            ),
            "nonzero_cell_fraction": finite_summary(
                [x["nonzero_cell_fraction"] for x in frame_summaries]
            ),
            "singleton_fraction_of_nonzero": finite_summary(
                [x["singleton_fraction_of_nonzero"] for x in frame_summaries]
            ),
            "singleton_wsum_lt_0p1_fraction_of_nonzero": finite_summary(
                [x["singleton_wsum_lt_0p1_fraction_of_nonzero"] for x in frame_summaries]
            ),
            "singleton_wsum_lt_0p05_fraction_of_nonzero": finite_summary(
                [x["singleton_wsum_lt_0p05_fraction_of_nonzero"] for x in frame_summaries]
            ),
            "singleton_wsum_lt_0p01_fraction_of_nonzero": finite_summary(
                [x["singleton_wsum_lt_0p01_fraction_of_nonzero"] for x in frame_summaries]
            ),
            "support_buckets": aggregate_support_buckets(frame_summaries),
        },
        "interpretation_checks": {},
    }

    # Automatic high-signal checks.
    tp = summary["groups"]["tp"]
    fp = summary["groups"]["fp_explicit_negative"]
    checks = summary["interpretation_checks"]

    tp_w = tp.get("splat_weight_sum_center", {}).get("median", float("nan"))
    fp_w = fp.get("splat_weight_sum_center", {}).get("median", float("nan"))
    tp_gain = tp.get("splat_norm_gain_center", {}).get("median", float("nan"))
    fp_gain = fp.get("splat_norm_gain_center", {}).get("median", float("nan"))
    tp_out = tp.get("splat_out_rms_center", {}).get("median", float("nan"))
    fp_out = fp.get("splat_out_rms_center", {}).get("median", float("nan"))
    tp_pre = tp.get("splat_pre_rms_center", {}).get("median", float("nan"))
    fp_pre = fp.get("splat_pre_rms_center", {}).get("median", float("nan"))

    checks["fp_has_lower_weight_sum_than_tp"] = (
        bool(np.isfinite(tp_w) and np.isfinite(fp_w) and fp_w < tp_w)
    )
    checks["fp_has_higher_normalization_gain_than_tp"] = (
        bool(np.isfinite(tp_gain) and np.isfinite(fp_gain) and fp_gain > tp_gain)
    )
    checks["fp_pre_norm_weaker_but_post_norm_similar"] = (
        bool(
            np.isfinite(tp_pre) and np.isfinite(fp_pre)
            and np.isfinite(tp_out) and np.isfinite(fp_out)
            and fp_pre < tp_pre
            and fp_out > 0.5 * tp_out
        )
    )

    json_path = os.path.join(out_dir, "splat_diag_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=True)

    print("\n=== TP vs explicit-negative FP @ anchor-center splat cell ===")
    for name in ["tp", "fp_explicit_negative"]:
        g = summary["groups"][name]
        print(
            f"{name:>22s}: n={g['n']} | "
            f"wsum_med={g['splat_weight_sum_center'].get('median', float('nan')):.4g} | "
            f"count_med={g['splat_count_center'].get('median', float('nan')):.3g} | "
            f"gain_med={g['splat_norm_gain_center'].get('median', float('nan')):.3g} | "
            f"preRMS_med={g['splat_pre_rms_center'].get('median', float('nan')):.4g} | "
            f"outRMS_med={g['splat_out_rms_center'].get('median', float('nan')):.4g} | "
            f"single_low<.1={g.get('low_support_singleton_lt_0p1_fraction', float('nan')):.3f}"
        )

    print("\n=== Top recurring explicit-negative FP hotspots ===")
    for h in summary["top_recurring_explicit_negative_fp_hotspots"][:10]:
        print(
            f"anchor={h['anchor_id']:6d} n={h['fp_occurrences']:3d} "
            f"xy=({h['anchor_x_mean']:.1f},{h['anchor_y_mean']:.1f}) "
            f"obj_med={h['obj'].get('median', float('nan')):.3f} "
            f"wsum_med={h['weight_sum_center'].get('median', float('nan')):.4g} "
            f"count_med={h['count_center'].get('median', float('nan')):.3g} "
            f"gain_med={h['norm_gain_center'].get('median', float('nan')):.2f} "
            f"preRMS_med={h['pre_rms_center'].get('median', float('nan')):.3g} "
            f"outRMS_med={h['out_rms_center'].get('median', float('nan')):.3g}"
        )

    print("\n=== Automatic checks ===")
    for k, v in checks.items():
        print(f"{k}: {v}")

    print(f"\nsaved: {csv_path}")
    print(f"saved: {json_path}")


if __name__ == "__main__":
    main()

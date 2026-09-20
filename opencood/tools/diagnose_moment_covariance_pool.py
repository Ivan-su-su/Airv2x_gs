#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference-only diagnostic:
Current GaussianVoxelPool vs moment-preserving covariance pooling.

Verified against Airv2x_gs baseline branch APIs:
  - GaussianContextInteraction.forward():
        pooled = {agent: self.pool(gs)}
        merged, agent_slices = self._merge(pooled)
        ...
        return out, self.splat(out)
  - GaussianVoxelPool is a plain callable object, not nn.Module.
  - GaussianSet.covariance() reconstructs [N,3,3].
  - covariance_to_scale_quaternion() factors [M,3,3].
  - official AirV2X evaluation uses:
        dataset.post_process(batch, OrderedDict(ego=output))
        eval_utils.caluclate_tp_fp(...)
        eval_utils.calculate_ap(..., global_sort_detections=False)

A/B:
  A = current pool:
      mean(mean_i), mean(scale_i), antipodal mean(quaternion_i)

  B = moment pool:
      mu = mean(mu_i)
      Sigma = mean[ Sigma_i + (mu_i-mu)(mu_i-mu)^T ]
      feature = mean(feature_i)
      p_fg = mean(p_fg_i)
      Sigma -> scale/quaternion by covariance_to_scale_quaternion()

No retraining, no optimizer step, no checkpoint modification.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import random
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.base.gaussian_init import (
    covariance_to_scale_quaternion,
)
from opencood.tools import train_utils
from opencood.utils import eval_utils_opv2v as eval_utils

DEFAULT_CKPT = (
    "/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
    "airv2x_gaussian_0822_geom_sup/geom_sup_2026_09_17_00_34_27/"
    "net_epoch25.pth"
)

AGENTS = ("vehicle", "rsu", "drone")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--config", default="")
    p.add_argument("--num-frames", type=int, default=100)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--worker", type=int, default=2)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--eval-ious", default="0.3,0.5,0.7")
    p.add_argument(
        "--out-dir",
        default="opencood/logs/diagnose_moment_pool_ep25",
    )
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_config(ckpt, explicit):
    p = Path(explicit) if explicit else ckpt.parent / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"config not found: {p}; pass --config explicitly"
        )
    return p


def unwrap_state_dict(blob):
    if not isinstance(blob, dict):
        raise TypeError(type(blob))
    for k in ("model_state_dict", "model_state", "state_dict", "model"):
        if isinstance(blob.get(k), dict):
            return blob[k]
    return blob


def load_checkpoint_exact(model, ckpt, device):
    blob = torch.load(str(ckpt), map_location=device)
    state = dict(unwrap_state_dict(blob))
    if state and next(iter(state)).startswith("module."):
        state = {k[7:]: v for k, v in state.items()}

    current = model.state_dict()
    shape_bad = [
        k for k, v in state.items()
        if k in current and tuple(v.shape) != tuple(current[k].shape)
    ]
    if shape_bad:
        raise RuntimeError(f"shape mismatch examples: {shape_bad[:10]}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch\n"
            f"missing({len(missing)}): {missing[:20]}\n"
            f"unexpected({len(unexpected)}): {unexpected[:20]}"
        )
    print(f"[ckpt] exact-compatible: {ckpt}")


def make_loader(hypes, worker):
    ds = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=worker,
        collate_fn=ds.collate_batch_test,
        pin_memory=True,
        drop_last=False,
    )
    return ds, loader


def scatter_mean(values, inverse, n_groups):
    sums = values.new_zeros((n_groups, values.shape[1]))
    sums.index_add_(0, inverse, values)
    counts = torch.bincount(inverse, minlength=n_groups).to(values.dtype)
    return sums / counts.unsqueeze(-1)


class MomentGaussianVoxelPool:
    """
    Drop-in replacement for the real GaussianVoxelPool.

    Group assignment is IDENTICAL to current code:
      floor((mean - pc_min) / voxel_size)

    Only the pooled geometry changes:
      current: average scale + average quaternion
      moment:  second-moment-preserving covariance
    """

    def __init__(self, original_pool):
        # Copy the exact grid definition from the live model.
        self.voxel_size = original_pool.voxel_size
        self.pc_min = original_pool.pc_min
        self.grid = original_pool.grid

    def __call__(self, gaussians: GaussianSet) -> GaussianSet:
        n = gaussians.n_gaussians
        if n == 0:
            return gaussians

        device = gaussians.mean.device
        dtype = gaussians.mean.dtype
        voxel_size = self.voxel_size.to(device=device, dtype=dtype)
        pc_min = self.pc_min.to(device=device, dtype=dtype)
        grid = self.grid.to(device)

        # Exact same voxel assignment as repository GaussianVoxelPool.
        idx = torch.floor((gaussians.mean - pc_min) / voxel_size).long()
        valid = ((idx >= 0) & (idx < grid)).all(dim=-1)
        kept = gaussians.index_select(valid)
        if kept.n_gaussians == 0:
            return kept

        idx = idx[valid]
        batch = (
            kept.batch_index
            if kept.batch_index is not None
            else torch.zeros(
                kept.n_gaussians,
                dtype=torch.long,
                device=device,
            )
        )
        gx, gy, gz = (int(v) for v in grid.tolist())
        code = ((batch * gx + idx[:, 0]) * gy + idx[:, 1]) * gz + idx[:, 2]
        unq_code, inverse = torch.unique(code, return_inverse=True)
        inverse = inverse.long()
        m = int(unq_code.numel())
        pooled_batch = unq_code // (gx * gy * gz)

        # Same pooled mean/feature/p_fg as current pool.
        pooled_mean = scatter_mean(kept.mean, inverse, m)
        pooled_feature = scatter_mean(kept.feature, inverse, m)

        # Moment-preserving Gaussian mixture approximation:
        # E[xx^T] = mean(Sigma_i + mu_i mu_i^T)
        # Cov = E[xx^T] - mu_bar mu_bar^T
        cov_i = kept.covariance()  # [N,3,3]
        mu = kept.mean
        second_i = cov_i + mu.unsqueeze(-1) * mu.unsqueeze(-2)
        second_bar = scatter_mean(
            second_i.reshape(second_i.shape[0], 9),
            inverse,
            m,
        ).reshape(m, 3, 3)
        pooled_cov = (
            second_bar
            - pooled_mean.unsqueeze(-1) * pooled_mean.unsqueeze(-2)
        )
        pooled_cov = 0.5 * (
            pooled_cov + pooled_cov.transpose(-1, -2)
        )

        pooled_scale, pooled_quat = covariance_to_scale_quaternion(
            pooled_cov
        )

        pooled_pfg = None
        if kept.p_fg is not None:
            pooled_pfg = scatter_mean(
                kept.p_fg.unsqueeze(-1),
                inverse,
                m,
            ).squeeze(-1)

        return GaussianSet(
            mean=pooled_mean,
            scale=pooled_scale,
            quaternion=pooled_quat,
            feature=pooled_feature,
            p_fg=pooled_pfg,
            batch_index=pooled_batch,
            agent=kept.agent,
            frame=kept.frame,
        )


@contextlib.contextmanager
def use_moment_pool(model):
    old_pool = model.stage3.pool
    model.stage3.pool = MomentGaussianVoxelPool(old_pool)
    try:
        yield
    finally:
        model.stage3.pool = old_pool


# ---- Official evaluation helpers copied/aligned with repository diagnostic ----

def init_stat(ious):
    return {
        float(i): {"tp": [], "fp": [], "gt": 0, "score": []}
        for i in ious
    }


def add_ap(stat, pred, score, gt, ious):
    ngt = 0 if gt is None else int(len(gt))
    for iou in ious:
        iou = float(iou)
        if pred is None or score is None or len(pred) == 0:
            stat[iou]["gt"] += ngt
        else:
            eval_utils.caluclate_tp_fp(
                pred, score, gt, stat, iou
            )


def ap_safe(stat, iou):
    # calculate_ap mutates fp/tp lists; use deepcopy exactly as repo helper.
    s = copy.deepcopy(stat)
    iou = float(iou)
    if s[iou]["gt"] <= 0:
        return float("nan")
    if len(s[iou]["tp"]) == 0:
        return 0.0
    ap, _, _ = eval_utils.calculate_ap(
        s,
        iou,
        global_sort_detections=False,
    )
    return float(ap)


def parse_pp(result):
    if not isinstance(result, (tuple, list)):
        raise RuntimeError(f"unexpected post_process type: {type(result)}")
    pred, score = result[0], result[1]
    gt = None
    for item in result[2:]:
        if (
            torch.is_tensor(item)
            and item.ndim == 3
            and tuple(item.shape[-2:]) == (8, 3)
        ):
            gt = item
            break
        if (
            isinstance(item, np.ndarray)
            and item.ndim == 3
            and tuple(item.shape[-2:]) == (8, 3)
        ):
            gt = item
            break
    if gt is None:
        raise RuntimeError(
            "Cannot locate GT [N,8,3] in post_process result"
        )
    return pred, score, gt


def official_pp(dataset, batch_data, output):
    return parse_pp(
        dataset.post_process(
            batch_data,
            OrderedDict(ego=output),
        )
    )


def eval_variant(
    dataset,
    loader,
    model,
    device,
    use_amp,
    variant,
    start,
    nframes,
    ious,
):
    stat = init_stat(ious)
    done = 0
    model.eval()

    pool_ctx = (
        use_moment_pool(model)
        if variant == "moment"
        else contextlib.nullcontext()
    )

    with pool_ctx:
        with torch.no_grad():
            for fi, batch in enumerate(loader):
                if fi < start:
                    continue
                if done >= nframes:
                    break
                if batch is None:
                    continue

                batch = train_utils.to_device(batch, device)
                ego = batch["ego"]
                ego["epoch"] = 0

                with amp.autocast(enabled=use_amp):
                    out = model(ego)

                pred, score, gt = official_pp(
                    dataset, batch, out
                )
                add_ap(stat, pred, score, gt, ious)
                done += 1

    ans = {"variant": variant, "n_frames": done}
    for iou in ious:
        iou = float(iou)
        ans[f"AP@{iou}"] = ap_safe(stat, iou)
        ans[f"GT@{iou}"] = int(stat[iou]["gt"])
        ans[f"TP@{iou}"] = int(sum(stat[iou]["tp"]))
        ans[f"FP@{iou}"] = int(sum(stat[iou]["fp"]))
        ans[f"recall@{iou}"] = (
            ans[f"TP@{iou}"] / max(ans[f"GT@{iou}"], 1)
        )
        ans[f"precision@{iou}"] = (
            ans[f"TP@{iou}"]
            / max(ans[f"TP@{iou}"] + ans[f"FP@{iou}"], 1)
        )
    return ans


# ---- Geometry-only comparison on Stage2 inputs ----

class Stage2Capture:
    def __init__(self, model):
        self.value = None
        self.handle = model.stage3.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self.value = inputs[0]

    def clear(self):
        self.value = None

    def close(self):
        self.handle.remove()


def tensor_stats(values):
    if not values:
        return {}
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(x.max()),
    }


def collect_geometry_delta(
    loader,
    model,
    device,
    use_amp,
    start,
    nframes,
):
    """
    Run normal forward only to capture Stage2 input.
    Then apply current_pool and moment_pool directly to the SAME Stage2 sets.

    Compare resulting pooled covariance:
      relative Frobenius difference
      trace ratio moment/current
      XY sqrt(det) area ratio moment/current
    """
    cap = Stage2Capture(model)
    per_agent = {
        a: defaultdict(list)
        for a in AGENTS
    }
    done = 0

    try:
        with torch.no_grad():
            for fi, batch in enumerate(loader):
                if fi < start:
                    continue
                if done >= nframes:
                    break
                if batch is None:
                    continue

                batch = train_utils.to_device(batch, device)
                ego = batch["ego"]
                ego["epoch"] = 0
                cap.clear()

                with amp.autocast(enabled=use_amp):
                    _ = model(ego)

                stage2 = cap.value
                if stage2 is None:
                    raise RuntimeError("failed to capture Stage2 input")

                current_pool = model.stage3.pool
                moment_pool = MomentGaussianVoxelPool(current_pool)

                for agent in AGENTS:
                    gs = stage2.get(agent)
                    if gs is None or gs.n_gaussians == 0:
                        continue

                    cur = current_pool(gs)
                    mom = moment_pool(gs)

                    if cur.n_gaussians != mom.n_gaussians:
                        raise RuntimeError(
                            f"{agent}: current N={cur.n_gaussians} "
                            f"moment N={mom.n_gaussians}"
                        )
                    if cur.n_gaussians == 0:
                        continue

                    # Same grouping => same mean and feature should be numerically equal.
                    mean_err = float(
                        (cur.mean - mom.mean).abs().max().item()
                    )
                    feat_err = float(
                        (cur.feature - mom.feature).abs().max().item()
                    )
                    if mean_err > 1e-4 or feat_err > 1e-4:
                        raise RuntimeError(
                            f"{agent}: grouping mismatch "
                            f"mean_err={mean_err} feat_err={feat_err}"
                        )

                    cc = cur.covariance().float()
                    cm = mom.covariance().float()

                    fro_cur = torch.linalg.matrix_norm(
                        cc, ord="fro", dim=(-2, -1)
                    ).clamp_min(1e-8)
                    rel = torch.linalg.matrix_norm(
                        cm - cc, ord="fro", dim=(-2, -1)
                    ) / fro_cur

                    trace_ratio = (
                        cm.diagonal(dim1=-2, dim2=-1).sum(-1)
                        / cc.diagonal(dim1=-2, dim2=-1)
                        .sum(-1)
                        .clamp_min(1e-8)
                    )

                    det_cur_xy = torch.det(cc[:, :2, :2]).clamp_min(1e-12)
                    det_mom_xy = torch.det(cm[:, :2, :2]).clamp_min(1e-12)
                    # sqrt(det Sigma_xy) is proportional to ellipse area.
                    area_ratio = torch.sqrt(
                        det_mom_xy / det_cur_xy
                    )

                    per_agent[agent]["cov_rel_fro"] += (
                        rel.detach().cpu().tolist()
                    )
                    per_agent[agent]["trace_ratio_moment_over_current"] += (
                        trace_ratio.detach().cpu().tolist()
                    )
                    per_agent[agent]["xy_area_ratio_moment_over_current"] += (
                        area_ratio.detach().cpu().tolist()
                    )

                done += 1
    finally:
        cap.close()

    return {
        agent: {
            metric: tensor_stats(vals)
            for metric, vals in metrics.items()
        }
        for agent, metrics in per_agent.items()
    }


def main():
    opt = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    torch.cuda.set_device(opt.gpu_id)
    device = torch.device(f"cuda:{opt.gpu_id}")
    seed_all(opt.seed)

    ckpt = Path(opt.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    config = resolve_config(ckpt, opt.config)
    hypes = yaml_utils.load_yaml(str(config))
    ious = [
        float(x.strip())
        for x in opt.eval_ious.split(",")
        if x.strip()
    ]

    print("[config]", config)
    print("[checkpoint]", ckpt)

    dataset, loader = make_loader(hypes, opt.worker)
    model = train_utils.create_model(hypes).to(device)
    load_checkpoint_exact(model, ckpt, device)
    model.eval()

    if not hasattr(model, "stage3"):
        raise RuntimeError("model has no stage3")
    if not hasattr(model.stage3, "pool"):
        raise RuntimeError("model.stage3 has no pool")

    out_dir = Path(opt.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== geometry delta: current pool vs moment pool ===")
    geometry_delta = collect_geometry_delta(
        loader=loader,
        model=model,
        device=device,
        use_amp=bool(opt.amp),
        start=opt.start,
        nframes=opt.num_frames,
    )

    # Fresh loaders for fair A/B.
    dataset, loader = make_loader(hypes, opt.worker)
    print("\n=== A: current pooling ===")
    current = eval_variant(
        dataset, loader, model, device,
        bool(opt.amp), "current",
        opt.start, opt.num_frames, ious,
    )
    print(json.dumps(current, indent=2))

    dataset, loader = make_loader(hypes, opt.worker)
    print("\n=== B: moment covariance pooling ===")
    moment = eval_variant(
        dataset, loader, model, device,
        bool(opt.amp), "moment",
        opt.start, opt.num_frames, ious,
    )
    print(json.dumps(moment, indent=2))

    ab = {}
    for iou in ious:
        k = f"AP@{float(iou)}"
        ab[k] = {
            "current": current[k],
            "moment": moment[k],
            "delta": moment[k] - current[k],
        }

    report = {
        "checkpoint": str(ckpt),
        "config": str(config),
        "num_frames": opt.num_frames,
        "voxelization_logic": {
            "assignment": "center_only",
            "formula": "floor((mean - pc_min) / voxel_size)",
            "one_input_gaussian_belongs_to_one_pool_voxel": True,
            "multi_cell_coverage_happens_after_pooling_in_bev_splat": True,
        },
        "geometry_delta": geometry_delta,
        "current": current,
        "moment": moment,
        "ab": ab,
    }

    out_json = out_dir / "moment_pool_ab.json"
    out_json.write_text(
        json.dumps(report, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print("\n=== GEOMETRY DELTA ===")
    for agent in AGENTS:
        print(f"[{agent}]")
        d = geometry_delta.get(agent, {})
        for metric, s in d.items():
            if not s:
                continue
            print(
                f"  {metric}: "
                f"mean={s['mean']:.3f} "
                f"p50={s['p50']:.3f} "
                f"p90={s['p90']:.3f} "
                f"p99={s['p99']:.3f}"
            )

    print("\n=== AP A/B ===")
    for k, v in ab.items():
        print(
            f"{k}: current={v['current']:.4f} "
            f"moment={v['moment']:.4f} "
            f"delta={v['delta']:+.4f}"
        )

    print("\n[wrote]", out_json)


if __name__ == "__main__":
    main()

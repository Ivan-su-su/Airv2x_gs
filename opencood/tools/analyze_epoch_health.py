#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified epoch health audit for AirV2X Gaussian 0822.

Required inputs
---------------
    python opencood/tools/analyze_epoch_health.py /path/to/run/config.yaml EPOCH

Example
-------
    python opencood/tools/analyze_epoch_health.py \
        opencood/logs/airv2x_gaussian_0822_camera/v45_xxx/config.yaml 5

Outputs
-------
    <config_dir>/epoch5_health_report.txt
    <config_dir>/epoch5_health_metrics.json

The audit is read-only: it loads a checkpoint, runs a deterministic fixed
test/val frame, performs one no-step backward probe, and writes diagnostics.
It does not alter model weights, optimizer state, thresholds, or config.

This script is aligned to the 0911version AirV2X Gaussian code path:
  * Airv2xGaussian0822 encode -> initializer -> Stage1 -> Stage2 -> Stage3
  * production Stage3 uses standard PreNorm residuals:
      adapter: x + MLP(LN(x))
      final fusion: pooled + fusion_mlp(cat(LN(self), LN(cross)))
    with no output-side fusion_delta_norm / fusion_gamma
  * VoxelPostprocessor.post_process_airv2x gates production candidates by
    objectness > obj_threshold; class score threshold is computed but is not
    included in the production mask
  * train.py clips global gradient norm with clip_grad_norm_
  * current PointPillarLossMultiClass objectness implementation can be
    identified from source at runtime (all-anchor mean vs balanced variants)

The thresholds below are diagnostic heuristics, not training constraints.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
import re
import sys
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.airv2x_gaussian_0822 import (
    CAMERA_GEOMETRY_KEY,
    present_camera_agents,
)
from opencood.tools import train_utils
from opencood.utils import box_utils
from opencood.utils import eval_utils_opv2v as eval_utils


EPS = 1.0e-12

# Diagnostic-only heuristics.
FEATURE_EXPLOSION_RATIO = 3.0
STAGE3_BYPASS_DELTA_RATIO = 0.05
PARAM_COLLAPSE_RATIO = 1.0e-2
LOW_RECALL_IOU = 0.30
LOW_RECALL_FLOOR = 0.20
CLIP_ALWAYS_RATE = 0.80
SEVERE_CLIP_FACTOR = 0.50
SCALE_BOUND_FRACTION = 0.50
SCALE_BOUND_MARGIN_FRAC = 0.05
SEED_COVERAGE_RADII_M = (2.0, 4.0, 8.0)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def finite_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().reshape(-1)
    return x[torch.isfinite(x)]


def stats(x: Optional[torch.Tensor]) -> Dict[str, Any]:
    if x is None:
        return {"n": 0, "finite_ratio": None}
    flat = x.detach().float().reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return {
            "n": 0,
            "finite_ratio": 1.0,
            "mean": None,
            "std": None,
            "median": None,
            "mean_abs": None,
            "max_abs": None,
            "rms": None,
            "p01": None,
            "p10": None,
            "p90": None,
            "p99": None,
        }
    f = flat[torch.isfinite(flat)]
    nf = int(f.numel())
    if nf == 0:
        return {
            "n": n,
            "finite_ratio": 0.0,
            "mean": None,
            "std": None,
            "median": None,
            "mean_abs": None,
            "max_abs": None,
            "rms": None,
            "p01": None,
            "p10": None,
            "p90": None,
            "p99": None,
        }
    q = torch.quantile(f, torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99], device=f.device))
    return {
        "n": n,
        "finite_ratio": nf / max(n, 1),
        "mean": float(f.mean()),
        "std": float(f.std()) if nf > 1 else 0.0,
        "median": float(q[2]),
        "mean_abs": float(f.abs().mean()),
        "max_abs": float(f.abs().max()),
        "rms": float(f.pow(2).mean().sqrt()),
        "p01": float(q[0]),
        "p10": float(q[1]),
        "p90": float(q[3]),
        "p99": float(q[4]),
    }


def rms(x: torch.Tensor) -> float:
    f = finite_tensor(x)
    if f.numel() == 0:
        return 0.0
    return float(f.pow(2).mean().sqrt())


def l2(x: Optional[torch.Tensor]) -> Optional[float]:
    if x is None:
        return None
    return float(torch.linalg.vector_norm(x.detach().float()))


def ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a) / (float(b) + EPS)


def safe_float(x: Any) -> Optional[float]:
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, bool):
        return "YES" if x else "NO"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            return str(x)
        if abs(x) >= 10000 or (abs(x) > 0 and abs(x) < 1e-4):
            return f"{x:.3e}"
        return f"{x:.{digits}f}"
    return str(x)


def section(title: str) -> str:
    return "\n" + "=" * 68 + f"\n{title}\n" + "=" * 68 + "\n"


def param_l2_dict(module: nn.Module, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, p in module.named_parameters():
        key = f"{prefix}.{name}" if prefix else name
        out[key] = float(torch.linalg.vector_norm(p.detach().float()))
    return out


def find_key_suffix(mapping: Mapping[str, float], suffix: str) -> Optional[float]:
    for key, value in mapping.items():
        if key.endswith(suffix):
            return value
    return None


# ---------------------------------------------------------------------------
# CLI / loading
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified AirV2X Gaussian epoch health diagnostic."
    )
    p.add_argument("config", type=str, help="Run config.yaml (normally beside net_epochN.pth)")
    p.add_argument("epoch", type=int, help="Checkpoint number N in net_epochN.pth")
    p.add_argument("--index", type=int, default=0, help="Fixed dataset index; default 0")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--seed-radius",
        type=float,
        default=4.0,
        help="Primary GT-to-initial-seed coverage radius in meters; default 4",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory; default is the config/checkpoint directory",
    )
    return p.parse_args()


def resolve_paths(config_arg: str, epoch: int) -> Tuple[Path, Path, Path]:
    config = Path(config_arg).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"config not found: {config}")
    model_dir = config.parent
    ckpt = model_dir / f"net_epoch{int(epoch)}.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"checkpoint not found beside config: {ckpt}\n"
            "This script intentionally needs only CONFIG + EPOCH, so use the "
            "copied run config.yaml in the checkpoint directory."
        )
    return config, model_dir, ckpt


def load_hypes(config: Path, split: str) -> Dict[str, Any]:
    hypes = yaml_utils.load_yaml(str(config), None)
    hypes = copy.deepcopy(hypes)
    if split == "test":
        if "test_dir" not in hypes:
            raise KeyError("config has no test_dir")
        hypes["validate_dir"] = hypes["test_dir"]
    return hypes


def load_fixed_batch(
    dataset: Any,
    index: int,
    device: torch.device,
) -> Tuple[Dict[str, Any], str]:
    if index < 0 or index >= len(dataset):
        raise IndexError(f"index {index} out of range 0..{len(dataset)-1}")
    raw = dataset[index]
    batch = dataset.collate_batch_test([raw])
    if batch is None:
        raise RuntimeError(f"collate_batch_test returned None for index={index}")
    batch = train_utils.to_device(batch, device)
    ego = batch["ego"]
    meta = ego.get("metadata_path_list", ["?"])
    if isinstance(meta, (list, tuple)):
        frame = str(meta[0]) if meta else "?"
    else:
        frame = str(meta)
    return batch, frame


def normalize_checkpoint_state(blob: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(blob, dict):
        raise TypeError(f"checkpoint must be dict, got {type(blob)}")
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        if key in blob and isinstance(blob[key], dict):
            src = blob[key]
            break
    else:
        src = blob
    out: Dict[str, torch.Tensor] = {}
    for key, value in src.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("module.") and not key.startswith("module_list"):
            key = key[7:]
        out[key] = value
    return out


def load_checkpoint(
    model: nn.Module,
    ckpt: Path,
    device: torch.device,
) -> Dict[str, Any]:
    blob = torch.load(str(ckpt), map_location=device)
    src = normalize_checkpoint_state(blob)
    current = model.state_dict()
    compatible = {
        k: v for k, v in src.items()
        if k in current and tuple(v.shape) == tuple(current[k].shape)
    }
    shape_mismatch = [
        {
            "key": k,
            "checkpoint": list(v.shape),
            "model": list(current[k].shape),
        }
        for k, v in src.items()
        if k in current and tuple(v.shape) != tuple(current[k].shape)
    ]
    unexpected = [k for k in src if k not in current]
    missing = [k for k in current if k not in compatible]
    model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint_keys": len(src),
        "loaded_keys": len(compatible),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": shape_mismatch,
        "checkpoint_epoch_field": blob.get("epoch") if isinstance(blob, dict) else None,
        "has_optimizer_state": isinstance(blob, dict) and "optimizer_state_dict" in blob,
        "has_scheduler_state": isinstance(blob, dict) and "scheduler_state_dict" in blob,
    }


# ---------------------------------------------------------------------------
# Production source inspection
# ---------------------------------------------------------------------------

def inspect_objectness_loss(criterion: nn.Module) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "class": criterion.__class__.__name__,
        "module": criterion.__class__.__module__,
        "mode": "UNKNOWN",
        "all_anchor_weights": None,
        "mean_over_all": None,
    }
    try:
        src = inspect.getsource(criterion.__class__.forward)
    except Exception as exc:
        rec["source_error"] = str(exc)
        return rec

    compact = re.sub(r"\s+", " ", src)
    all_anchor = (
        "torch.ones_like(pos_mask_expanded)" in src
        or "torch.ones_like(pos_mask" in src
    )
    mean_all = (
        "obj_loss_src.mean()" in compact
        or re.search(r"obj_loss\s*=\s*obj_loss_src\.mean\s*\(", src) is not None
    )
    balanced_tokens = (
        "pos_obj_loss" in src
        and "neg_obj_loss" in src
        and ("num_pos" in src or "pos_normal" in src)
    )
    if all_anchor and mean_all:
        mode = "ALL_ANCHORS_MEAN_POSMASK_TARGET"
    elif balanced_tokens:
        mode = "BALANCED_POS_NEG_NORMALIZATION"
    elif mean_all:
        mode = "MEAN_REDUCTION_UNKNOWN_WEIGHTING"
    else:
        mode = "CUSTOM_OR_UNKNOWN"
    rec.update(
        {
            "mode": mode,
            "all_anchor_weights": all_anchor,
            "mean_over_all": mean_all,
        }
    )
    return rec


def detect_grad_clip_config() -> Dict[str, Any]:
    train_py = ROOT / "opencood/tools/train.py"
    rec = {"source": str(train_py), "max_norm": None}
    try:
        txt = train_py.read_text(encoding="utf-8")
        matches = re.findall(
            r"clip_grad_norm_\s*\([^)]*?max_norm\s*=\s*([0-9eE+\-.]+)",
            txt,
            flags=re.S,
        )
        vals = [float(v) for v in matches]
        if vals:
            rec["max_norm"] = vals[0] if len(set(vals)) == 1 else vals
    except Exception as exc:
        rec["error"] = str(exc)
    return rec


def read_historical_logs(model_dir: Path, checkpoint_epoch: int) -> Dict[str, Any]:
    # train.py writes checkpoint net_epoch{epoch+1}.pth after zero-based epoch.
    train_epoch = int(checkpoint_epoch) - 1
    out: Dict[str, Any] = {"train_epoch_index": train_epoch}

    grad_file = model_dir / "grad_clip.txt"
    if grad_file.is_file():
        pat = re.compile(
            rf"Epoch\[{train_epoch}\].*?steps\[(\d+)\].*?clipped\[(\d+)\].*?"
            rf"clip_rate\[([0-9eE+\-.]+)\].*?nonfinite\[(\d+)\]"
        )
        for line in grad_file.read_text(errors="ignore").splitlines():
            m = pat.search(line)
            if m:
                out["grad_clip"] = {
                    "steps": int(m.group(1)),
                    "clipped": int(m.group(2)),
                    "clip_rate": float(m.group(3)),
                    "nonfinite": int(m.group(4)),
                }

    train_loss_file = model_dir / "train_loss.txt"
    if train_loss_file.is_file():
        vals: List[float] = []
        pat = re.compile(rf"Epoch\[{train_epoch}\].*?loss\[([0-9eE+\-.]+)\]")
        for line in train_loss_file.read_text(errors="ignore").splitlines():
            m = pat.search(line)
            if m:
                try:
                    vals.append(float(m.group(1)))
                except Exception:
                    pass
        if vals:
            out["train_loss"] = {
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "last": float(vals[-1]),
            }

    val_file = model_dir / "validation_loss.txt"
    if val_file.is_file():
        pat = re.compile(rf"Epoch\[{train_epoch}\].*?loss\[([0-9eE+\-.]+)\]")
        for line in val_file.read_text(errors="ignore").splitlines():
            m = pat.search(line)
            if m:
                out["validation_loss"] = float(m.group(1))
    return out


# ---------------------------------------------------------------------------
# Stage-3 exact-production instrumentation
# ---------------------------------------------------------------------------

def _blocks_by_batch_agent_local(
    batch_index: torch.Tensor,
    n_batch: int,
    agent_slices: Mapping[str, Tuple[int, int]],
) -> List[Tuple[int, str, torch.Tensor]]:
    groups = []
    for b in range(n_batch):
        for agent, (start, end) in agent_slices.items():
            rows = torch.arange(start, end, device=batch_index.device)[
                batch_index[start:end] == b
            ]
            if int(rows.numel()) > 0:
                groups.append((b, agent, rows))
    return groups


def _module_param_summary(module: nn.Module) -> Dict[str, Any]:
    vals: Dict[str, float] = {}
    for name, p in module.named_parameters():
        vals[name] = float(torch.linalg.vector_norm(p.detach().float()))
    return vals


def run_stage3_instrumented(
    stage3: nn.Module,
    gaussians_by_agent: Mapping[str, Any],
    init_param_reference: Mapping[str, float],
) -> Tuple[Any, torch.Tensor, Dict[str, Any]]:
    pooled = {
        agent: stage3.pool(gs)
        for agent, gs in gaussians_by_agent.items()
        if gs.n_gaussians > 0
    }
    rec: Dict[str, Any] = {
        "pooled_counts": {a: int(gs.n_gaussians) for a, gs in pooled.items()},
        "adapter": {},
        "param_ratios_to_fresh_init": {},
    }

    if not pooled:
        sample = next(iter(gaussians_by_agent.values()))
        from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
        empty = GaussianSet.empty(
            sample.mean.device, sample.feature.dtype, int(stage3.feature_dim)
        )
        bev = stage3.splat(empty)
        rec["empty"] = True
        return empty, bev, rec

    merged, agent_slices = stage3._merge(pooled)
    pooled_feature = merged.feature
    split = stage3.split_mlp(pooled_feature)
    self_feature = split[:, : stage3.feature_dim]
    cross_feature = split[:, stage3.feature_dim :]

    batch = merged.batch_index
    n_batch = int(batch.max().item()) + 1
    groups = _blocks_by_batch_agent_local(batch, n_batch, agent_slices)

    self_out = self_feature.clone()
    for _b, _agent, rows in groups:
        x = self_feature[rows].unsqueeze(0)
        for block in stage3.self_blocks:
            x = block(x)
        self_out[rows] = x.squeeze(0)

    adapted = cross_feature.clone()
    for agent, (start, end) in agent_slices.items():
        if end <= start:
            continue
        raw = cross_feature[start:end]
        ad = stage3.adapter(raw, agent)
        adapted[start:end] = ad
        rec["adapter"][agent] = {
            "raw": stats(raw),
            "out": stats(ad),
            "delta": stats(ad - raw),
            "delta_to_raw_rms": rms(ad - raw) / (rms(raw) + EPS),
        }

    cross_out = cross_feature.clone()
    for _b, agent, rows in groups:
        others = [
            (a2, r2) for b2, a2, r2 in groups
            if b2 == _b and a2 != agent
        ]
        if not others:
            continue
        query = adapted[rows].unsqueeze(0)
        kv = torch.cat([adapted[r2] for _a2, r2 in others], dim=0).unsqueeze(0)
        for block in stage3.cross_blocks:
            query = block(query, kv)
        cross_out[rows] = query.squeeze(0)

    self_normed = stage3.self_fusion_norm(self_out)
    cross_normed = stage3.cross_fusion_norm(cross_out)
    fusion_input = torch.cat([self_normed, cross_normed], dim=-1)
    # Stage3-standard parameterization: the raw fusion projection is the
    # residual itself. There is intentionally no output-side LN0 or gamma.
    fusion_linear = stage3.fusion_mlp(fusion_input)
    applied_delta = fusion_linear
    fused = pooled_feature + applied_delta

    out = merged.replace_feature(fused)
    bev = stage3.splat(out)

    rec["features"] = {
        "pooled": stats(pooled_feature),
        "split": stats(split),
        "self_feature": stats(self_feature),
        "cross_feature": stats(cross_feature),
        "self_out": stats(self_out),
        "cross_out": stats(cross_out),
        "self_change": stats(self_out - self_feature),
        "cross_change": stats(cross_out - cross_feature),
        "fusion_input": stats(fusion_input),
        "fusion_linear": stats(fusion_linear),
        "applied_delta": stats(applied_delta),
        "final": stats(fused),
    }
    rec["ratios"] = {
        "self_change_to_input_rms": rms(self_out - self_feature)
        / (rms(self_feature) + EPS),
        "cross_change_to_input_rms": rms(cross_out - cross_feature)
        / (rms(cross_feature) + EPS),
        "fusion_linear_to_pooled_rms": rms(fusion_linear)
        / (rms(pooled_feature) + EPS),
        "applied_delta_to_pooled_rms": rms(applied_delta)
        / (rms(pooled_feature) + EPS),
        "final_to_pooled_rms": rms(fused) / (rms(pooled_feature) + EPS),
    }
    # Explicitly record the current Stage3 contract so reports cannot silently
    # fall back to the removed gated/output-normalized parameterization.
    rec["residual_parameterization"] = {
        "adapter": "x + MLP(LN(x))",
        "fusion": "pooled + fusion_mlp(cat(LN(self), LN(cross)))",
        "fusion_output_norm": False,
        "fusion_gamma": False,
    }

    rec["self_block_params"] = [
        _module_param_summary(b) for b in stage3.self_blocks
    ]
    rec["cross_block_params"] = [
        _module_param_summary(b) for b in stage3.cross_blocks
    ]
    rec["fusion_mlp_params"] = _module_param_summary(stage3.fusion_mlp)

    current = param_l2_dict(stage3, prefix="stage3")
    important_suffixes = (
        "self_blocks.0.q_proj.weight",
        "self_blocks.0.out_proj.weight",
        "self_blocks.0.ffn.0.weight",
        "self_blocks.0.ffn.2.weight",
        "cross_blocks.0.q_proj.weight",
        "cross_blocks.0.out_proj.weight",
        "fusion_mlp.weight",
        "adapter.adapters.vehicle.0.weight",
        "adapter.adapters.vehicle.2.weight",
        "adapter.adapters.rsu.0.weight",
        "adapter.adapters.rsu.2.weight",
        "adapter.adapters.drone.0.weight",
        "adapter.adapters.drone.2.weight",
    )
    for suffix in important_suffixes:
        cur = find_key_suffix(current, suffix)
        init = find_key_suffix(init_param_reference, suffix)
        rec["param_ratios_to_fresh_init"][suffix] = {
            "current_l2": cur,
            "fresh_init_l2": init,
            "ratio": ratio(cur, init),
        }

    # Representative LayerNorm affine weights.
    ln = {}
    for name, mod in stage3.named_modules():
        if isinstance(mod, nn.LayerNorm) and mod.elementwise_affine:
            ln[name] = {
                "weight": stats(mod.weight),
                "bias": stats(mod.bias),
            }
    rec["layernorm_affine"] = ln
    rec["splat_stats"] = dict(getattr(stage3.splat, "last_stats", {}) or {})
    return out, bev, rec


# ---------------------------------------------------------------------------
# Gaussian scale audit
# ---------------------------------------------------------------------------

def scale_stage_stats(gs: Any) -> Dict[str, Any]:
    if gs is None or gs.n_gaussians == 0:
        return {"n": 0}
    s = gs.scale.detach().float()
    return {
        "n": int(gs.n_gaussians),
        "xyz_mean": s.mean(0).cpu().tolist(),
        "xyz_median": s.median(0).values.cpu().tolist(),
        "xyz_p10": torch.quantile(s, 0.10, dim=0).cpu().tolist(),
        "xyz_p90": torch.quantile(s, 0.90, dim=0).cpu().tolist(),
        "finite_ratio": float(torch.isfinite(s).float().mean()),
    }


def aligned_log_scale_delta(
    before: Any,
    after: Any,
    lo: float,
    hi: float,
) -> Dict[str, Any]:
    if before is None or after is None:
        return {"available": False}
    n = min(int(before.n_gaussians), int(after.n_gaussians))
    if n <= 0:
        return {"available": False, "n": 0}
    b = before.scale[:n].detach().float().clamp_min(1e-8)
    a = after.scale[:n].detach().float().clamp_min(1e-8)
    d = torch.log(a / b)
    width = max(hi - lo, 1e-8)
    margin = SCALE_BOUND_MARGIN_FRAC * width
    near_lo = d <= (lo + margin)
    near_hi = d >= (hi - margin)
    return {
        "available": True,
        "n": n,
        "count_mismatch": int(before.n_gaussians) != int(after.n_gaussians),
        "delta_log_scale": stats(d),
        "xyz_mean": d.mean(0).cpu().tolist(),
        "xyz_median": d.median(0).values.cpu().tolist(),
        "near_lower_frac_xyz": near_lo.float().mean(0).cpu().tolist(),
        "near_upper_frac_xyz": near_hi.float().mean(0).cpu().tolist(),
        "configured_bounds": [lo, hi],
        "bound_margin": margin,
    }


def analyze_scales(
    init_g: Mapping[str, Any],
    s1_g: Mapping[str, Any],
    s2_g: Mapping[str, Any],
    hypes: Mapping[str, Any],
) -> Dict[str, Any]:
    cfg = (
        hypes.get("model", {})
        .get("args", {})
        .get("gaussian_refinement", {})
        .get("scale", {})
    )
    bounds = cfg.get("delta_range", (-0.20, 0.05))
    lo, hi = float(bounds[0]), float(bounds[1])
    agents = sorted(set(init_g) | set(s1_g) | set(s2_g))
    out: Dict[str, Any] = {
        "delta_bounds": [lo, hi],
        "agents": {},
    }
    for agent in agents:
        i = init_g.get(agent)
        s1 = s1_g.get(agent)
        s2 = s2_g.get(agent)
        out["agents"][agent] = {
            "init": scale_stage_stats(i),
            "stage1": scale_stage_stats(s1),
            "stage2": scale_stage_stats(s2),
            "stage1_over_init": aligned_log_scale_delta(i, s1, lo, hi),
            "stage2_over_stage1": aligned_log_scale_delta(s1, s2, lo, hi),
        }
    return out


# ---------------------------------------------------------------------------
# Manual production pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    model: nn.Module,
    ego: Dict[str, Any],
    init_param_reference: Mapping[str, float],
) -> Dict[str, Any]:
    # Mirrors Airv2xGaussian0822.forward, while retaining all three Gaussian stages.
    for agent in present_camera_agents(ego):
        model._encode_agent(agent, ego)

    init_g = model.initializer.forward(ego)
    s1_g: Dict[str, Any] = {}
    for agent, gs in init_g.items():
        payload = ego[agent]
        s1_g[agent] = model.intra_view(
            gs,
            payload["f90"],
            payload[CAMERA_GEOMETRY_KEY],
            agent=agent,
        )

    sources = {
        agent: {
            "features": ego[agent]["f90"],
            "geometry": ego[agent][CAMERA_GEOMETRY_KEY],
        }
        for agent in s1_g
    }
    s2_g = model.cross_agent(s1_g, sources)
    stage3_g, bev, stage3_rec = run_stage3_instrumented(
        model.stage3, s2_g, init_param_reference
    )

    ego["gaussians"] = s2_g
    ego["spatial_features"] = bev
    backbone_data = model.backbone(ego)
    fused = model.shrink_conv(backbone_data["spatial_features_2d"])
    output = {
        "psm": model.cls_head(fused),
        "rm": model.reg_head(fused),
        "obj": model.obj_head(fused),
    }
    return {
        "output": output,
        "init_g": init_g,
        "stage1_g": s1_g,
        "stage2_g": s2_g,
        "stage3_g": stage3_g,
        "bev": bev,
        "stage3": stage3_rec,
    }


# ---------------------------------------------------------------------------
# GT / spatial alignment
# ---------------------------------------------------------------------------

def get_gt_boxes(dataset: Any, batch: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    pp = dataset.post_processor
    rec: Dict[str, Any] = {}
    try:
        gt = pp.generate_gt_bbx_airv2x(batch)
        if isinstance(gt, tuple):
            boxes = gt[0]
            rec["method"] = "generate_gt_bbx_airv2x"
            if len(gt) > 1:
                rec["class_labels"] = gt[1]
            if len(gt) > 2:
                rec["track_ids"] = gt[2]
            return boxes, rec
        rec["method"] = "generate_gt_bbx_airv2x"
        return gt, rec
    except Exception as exc_air:
        rec["airv2x_error"] = str(exc_air)
    try:
        boxes = pp.generate_gt_bbx(batch)
        rec["method"] = "generate_gt_bbx"
        return boxes, rec
    except Exception as exc:
        rec["fallback_error"] = str(exc)
        return None, rec


def seed_coverage(
    init_g: Mapping[str, Any],
    gt_boxes: Optional[torch.Tensor],
    primary_radius: float,
) -> Dict[str, Any]:
    if gt_boxes is None:
        return {"available": False, "reason": "GT unavailable"}
    if int(gt_boxes.shape[0]) == 0:
        return {"available": False, "reason": "no GT"}
    means = [
        gs.mean.detach().float()
        for gs in init_g.values()
        if gs.n_gaussians > 0
    ]
    if not means:
        return {
            "available": True,
            "n_gt": int(gt_boxes.shape[0]),
            "n_seed": 0,
            "coverage": {str(r): 0.0 for r in SEED_COVERAGE_RADII_M},
        }
    seed_xy = torch.cat(means, dim=0)[:, :2]
    gt_xy = gt_boxes.detach().float().mean(dim=1)[:, :2]
    # Typical counts are manageable; cdist is only GT x seeds.
    d = torch.cdist(gt_xy, seed_xy)
    min_d = d.min(dim=1).values
    radii = sorted(set(tuple(SEED_COVERAGE_RADII_M) + (float(primary_radius),)))
    return {
        "available": True,
        "n_gt": int(gt_xy.shape[0]),
        "n_seed": int(seed_xy.shape[0]),
        "min_distance_m": stats(min_d),
        "coverage": {
            f"{r:g}m": float((min_d <= r).float().mean()) for r in radii
        },
        "primary_radius_m": float(primary_radius),
        "primary_coverage": float((min_d <= float(primary_radius)).float().mean()),
    }


def bev_gt_background(
    bev: torch.Tensor,
    stage3: nn.Module,
    gt_boxes: Optional[torch.Tensor],
) -> Dict[str, Any]:
    if gt_boxes is None or int(gt_boxes.shape[0]) == 0:
        return {"available": False}
    if bev.dim() != 4 or bev.shape[0] < 1:
        return {"available": False, "reason": f"unexpected BEV shape {list(bev.shape)}"}
    feature_norm = torch.linalg.vector_norm(bev[0].detach().float(), dim=0)
    ny, nx = int(feature_norm.shape[0]), int(feature_norm.shape[1])
    xy_min = stage3.splat.xy_min.detach().float().to(feature_norm.device)
    cell = stage3.splat.cell.detach().float().to(feature_norm.device)
    gt_xy = gt_boxes.detach().float().mean(dim=1)[:, :2].to(feature_norm.device)
    idx = torch.floor((gt_xy - xy_min) / cell).long()
    inside = (
        (idx[:, 0] >= 0) & (idx[:, 0] < nx)
        & (idx[:, 1] >= 0) & (idx[:, 1] < ny)
    )
    idx_in = idx[inside]
    if idx_in.numel() == 0:
        return {
            "available": True,
            "n_gt": int(gt_xy.shape[0]),
            "inside_grid": 0,
            "inside_fraction": 0.0,
        }

    gt_vals = feature_norm[idx_in[:, 1], idx_in[:, 0]]
    gt_mask = torch.zeros_like(feature_norm, dtype=torch.bool)
    gt_mask[idx_in[:, 1], idx_in[:, 0]] = True
    bg_vals = feature_norm[~gt_mask]
    bg_nonzero = bg_vals[bg_vals > 0]
    bg_ref = bg_nonzero if bg_nonzero.numel() else bg_vals
    gt_mean = float(gt_vals.mean()) if gt_vals.numel() else 0.0
    bg_mean = float(bg_ref.mean()) if bg_ref.numel() else 0.0
    return {
        "available": True,
        "bev_shape": list(bev.shape),
        "grid_hw": [ny, nx],
        "xy_min": xy_min.cpu().tolist(),
        "cell": cell.cpu().tolist(),
        "n_gt": int(gt_xy.shape[0]),
        "inside_grid": int(inside.sum()),
        "inside_fraction": float(inside.float().mean()),
        "gt_cell_norm": stats(gt_vals),
        "background_nonzero_cell_norm": stats(bg_ref),
        "gt_to_bg_mean_ratio": gt_mean / (bg_mean + EPS),
        "bev_nonzero_fraction": float((feature_norm > 0).float().mean()),
    }


# ---------------------------------------------------------------------------
# Detection / post-processing chain
# ---------------------------------------------------------------------------

def recall_at_iou(
    boxes: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
    gt_boxes: Optional[torch.Tensor],
    iou: float = LOW_RECALL_IOU,
) -> Optional[float]:
    if gt_boxes is None:
        return None
    n_gt = int(gt_boxes.shape[0])
    if n_gt == 0:
        return None
    if boxes is None or scores is None or int(boxes.shape[0]) == 0:
        return 0.0
    result = {float(iou): {"tp": [], "fp": [], "gt": 0, "score": []}}
    eval_utils.caluclate_tp_fp(boxes, scores, gt_boxes, result, float(iou))
    return float(sum(result[float(iou)]["tp"])) / max(n_gt, 1)


def detection_chain(
    dataset: Any,
    ego: Dict[str, Any],
    output: Mapping[str, torch.Tensor],
    gt_boxes: Optional[torch.Tensor],
) -> Dict[str, Any]:
    pp = dataset.post_processor
    rec: Dict[str, Any] = {}
    obj_logits = output["obj"]
    obj = torch.sigmoid(obj_logits.permute(0, 2, 3, 1).contiguous()).view(1, -1)

    psm = output["psm"]
    B, AC, H, W = psm.shape
    C = int(getattr(pp, "num_class", AC))
    A = AC // C
    p = psm.view(B, C, A, H, W).permute(0, 3, 4, 2, 1).contiguous()
    p = torch.sigmoid(p).view(B, -1, C)
    fg_class = p[:, :, 1:] if C > 1 else p
    class_scores, class_labels = torch.max(fg_class, dim=-1)
    class_labels = class_labels + (1 if C > 1 else 0)

    targ = pp.params["target_args"]
    obj_thresh = float(targ["obj_threshold"])
    score_thresh = float(targ.get("score_threshold", 0.2))
    mask = obj > obj_thresh  # exact production mask in 0911version

    rec["thresholds"] = {
        "obj_threshold": obj_thresh,
        "score_threshold": score_thresh,
        "production_mask": "objectness > obj_threshold",
        "class_score_applied_to_mask": False,
    }
    rec["raw"] = {
        "total_anchors": int(obj.numel()),
        "objectness": stats(obj),
        "class_scores": stats(class_scores),
        "obj_pass": int(mask.sum()),
        "class_score_pass_diagnostic_only": int((class_scores > score_thresh).sum()),
        "both_pass_diagnostic_only": int(
            ((obj > obj_thresh) & (class_scores > score_thresh)).sum()
        ),
    }

    anchor_box = ego["anchor_box"]
    if output["rm"].is_cuda:
        anchor_box = anchor_box.to(output["rm"].device)
    decoded_all = pp.delta_to_boxes3d(output["rm"], anchor_box)
    selected_center = decoded_all[0][mask[0]]
    selected_scores = obj[0][mask[0]]
    selected_labels = class_labels[0][mask[0]]

    rec["decode"] = {
        "selected_before_decode": int(mask.sum()),
        "decoded_count": int(selected_center.shape[0]),
        "finite_box_fraction": (
            float(torch.isfinite(selected_center).all(dim=1).float().mean())
            if selected_center.numel() else 1.0
        ),
        "box_center_stats": stats(selected_center[:, :3]) if selected_center.numel() else stats(None),
        "box_size_stats": stats(selected_center[:, 3:6]) if selected_center.numel() else stats(None),
    }

    if selected_center.shape[0] == 0:
        for name in ("projected", "prefilter", "nms", "range", "final"):
            rec[name] = {"count": 0, "recall_iou03": 0.0 if gt_boxes is not None else None}
        rec["final_boxes"] = None
        rec["final_scores"] = None
        return rec

    transformation_matrix = ego["transformation_matrix"]
    if selected_center.is_cuda:
        transformation_matrix = transformation_matrix.to(selected_center.device)

    corners = box_utils.boxes_to_corners_3d(
        selected_center, order=pp.params["order"]
    )
    projected = box_utils.project_box3d(corners, transformation_matrix)
    rec["projected"] = {
        "count": int(projected.shape[0]),
        "recall_iou03": recall_at_iou(projected, selected_scores, gt_boxes),
    }

    # Match production prefilters.
    try:
        keep1 = box_utils.remove_large_pred_bbx(projected, pp.dataset)
    except Exception:
        # Some branches use dataset name differently; do not kill the audit.
        keep1 = torch.ones(projected.shape[0], dtype=torch.bool, device=projected.device)
        rec["large_box_filter_warning"] = traceback.format_exc(limit=1)

    try:
        z_min = float(pp.lidar_range[2])
        z_max = float(pp.lidar_range[5])
        keep2 = box_utils.remove_bbx_abnormal_z(projected, z_min=z_min, z_max=z_max)
    except TypeError:
        keep2 = box_utils.remove_bbx_abnormal_z(projected)
    keep = keep1 & keep2
    pre = projected[keep]
    pre_s = selected_scores[keep]
    pre_l = selected_labels[keep]
    rec["prefilter"] = {
        "count": int(pre.shape[0]),
        "removed": int(projected.shape[0] - pre.shape[0]),
        "recall_iou03": recall_at_iou(pre, pre_s, gt_boxes),
    }

    if pre.shape[0] == 0:
        rec["nms"] = {"count": 0, "recall_iou03": 0.0 if gt_boxes is not None else None}
        rec["range"] = {"count": 0, "recall_iou03": 0.0 if gt_boxes is not None else None}
        rec["final"] = {"count": 0, "recall_iou03": 0.0 if gt_boxes is not None else None}
        rec["final_boxes"] = None
        rec["final_scores"] = None
        return rec

    try:
        keep_nms = box_utils.nms_rotated(pre, pre_s, pp.params["nms_thresh"])
        post_nms = pre[keep_nms]
        post_nms_s = pre_s[keep_nms]
        post_nms_l = pre_l[keep_nms]
        rec["nms"] = {
            "count": int(post_nms.shape[0]),
            "removed": int(pre.shape[0] - post_nms.shape[0]),
            "threshold": float(pp.params["nms_thresh"]),
            "recall_iou03": recall_at_iou(post_nms, post_nms_s, gt_boxes),
        }
    except Exception as exc:
        rec["nms"] = {
            "error": str(exc),
            "count": int(pre.shape[0]),
            "recall_iou03": recall_at_iou(pre, pre_s, gt_boxes),
        }
        post_nms, post_nms_s, post_nms_l = pre, pre_s, pre_l

    range_mask = box_utils.get_mask_for_boxes_within_range_torch(
        post_nms, pp.lidar_range
    )
    final = post_nms[range_mask]
    final_s = post_nms_s[range_mask]
    final_l = post_nms_l[range_mask]
    final_recall = recall_at_iou(final, final_s, gt_boxes)
    rec["range"] = {
        "count": int(final.shape[0]),
        "removed": int(post_nms.shape[0] - final.shape[0]),
        "recall_iou03": final_recall,
    }
    rec["final"] = {
        "count": int(final.shape[0]),
        "recall_iou03": final_recall,
        "score": stats(final_s),
        "labels": stats(final_l.float()),
    }
    # Do not put full tensors into JSON; they are only useful internally.
    rec["final_boxes"] = final
    rec["final_scores"] = final_s
    return rec


# ---------------------------------------------------------------------------
# Objectness and loss
# ---------------------------------------------------------------------------

def objectness_health(
    output: Mapping[str, torch.Tensor],
    label_dict: Mapping[str, torch.Tensor],
    obj_threshold: float,
) -> Dict[str, Any]:
    logits = output["obj"].permute(0, 2, 3, 1).contiguous()
    prob = torch.sigmoid(logits)
    pos = label_dict.get("pos_equal_one")
    neg = label_dict.get("neg_equal_one")
    rec: Dict[str, Any] = {
        "logits": stats(logits),
        "probability": stats(prob),
        "threshold": float(obj_threshold),
        "global_pass_count": int((prob > obj_threshold).sum()),
        "global_pass_fraction": float((prob > obj_threshold).float().mean()),
    }
    if pos is None:
        rec["positive_available"] = False
        return rec

    pos = pos.to(device=prob.device)
    if tuple(pos.shape) != tuple(prob.shape):
        try:
            pos = pos.view_as(prob)
        except Exception:
            rec["positive_available"] = False
            rec["shape_error"] = {
                "prob": list(prob.shape),
                "pos": list(pos.shape),
            }
            return rec

    pos_b = pos > 0
    rec["positive_available"] = True
    rec["n_positive"] = int(pos_b.sum())
    rec["positive_fraction_all_anchors"] = float(pos_b.float().mean())
    pos_prob = prob[pos_b]
    rec["positive_probability"] = stats(pos_prob)
    rec["positive_pass_fraction"] = (
        float((pos_prob > obj_threshold).float().mean())
        if pos_prob.numel() else None
    )

    if neg is not None:
        neg = neg.to(device=prob.device)
        try:
            neg = neg.view_as(prob)
            neg_b = neg > 0
            rec["n_explicit_negative"] = int(neg_b.sum())
            rec["explicit_negative_probability"] = stats(prob[neg_b])
            ignored = ~(pos_b | neg_b)
            rec["n_ignored_anchor_by_labeler"] = int(ignored.sum())
            rec["ignored_anchor_probability"] = stats(prob[ignored])
        except Exception:
            pass
    return rec


# ---------------------------------------------------------------------------
# Backward / gradient audit
# ---------------------------------------------------------------------------

def grad_group_norm(module: nn.Module) -> Dict[str, Any]:
    sq = 0.0
    n = 0
    none = 0
    nonfinite = 0
    max_abs = 0.0
    for p in module.parameters():
        if not p.requires_grad:
            continue
        n += p.numel()
        if p.grad is None:
            none += p.numel()
            continue
        g = p.grad.detach().float()
        nonfinite += int((~torch.isfinite(g)).sum())
        gf = torch.where(torch.isfinite(g), g, torch.zeros_like(g))
        sq += float((gf * gf).sum())
        if gf.numel():
            max_abs = max(max_abs, float(gf.abs().max()))
    return {
        "l2": math.sqrt(max(sq, 0.0)),
        "trainable_params": n,
        "grad_none_params": none,
        "nonfinite_grad_values": nonfinite,
        "max_abs": max_abs,
    }


def whole_model_grad_norm(model: nn.Module) -> Dict[str, Any]:
    sq = 0.0
    nonfinite = 0
    n_with_grad = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        n_with_grad += p.numel()
        g = p.grad.detach().float()
        nonfinite += int((~torch.isfinite(g)).sum())
        gf = torch.where(torch.isfinite(g), g, torch.zeros_like(g))
        sq += float((gf * gf).sum())
    return {
        "l2": math.sqrt(max(sq, 0.0)),
        "params_with_grad": n_with_grad,
        "nonfinite_grad_values": nonfinite,
    }


def backward_probe(
    model: nn.Module,
    criterion: nn.Module,
    batch: Dict[str, Any],
    checkpoint_epoch: int,
    clip_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    ego = batch["ego"]
    ego["epoch"] = int(checkpoint_epoch)
    model.zero_grad(set_to_none=True)
    criterion.zero_grad(set_to_none=True) if hasattr(criterion, "zero_grad") else None

    # Keep eval mode for deterministic one-frame diagnostic. Gradients are still computed.
    model.eval()
    output = model(ego)
    loss = criterion(output, ego["label_dict"])
    rec: Dict[str, Any] = {
        "loss": float(loss.detach()) if torch.isfinite(loss) else None,
        "loss_finite": bool(torch.isfinite(loss)),
        "output_finite": {
            k: bool(torch.isfinite(v).all())
            for k, v in output.items()
            if torch.is_tensor(v)
        },
        "criterion_loss_dict": dict(getattr(criterion, "loss_dict", {}) or {}),
    }
    loss.backward()

    rec["whole_model"] = whole_model_grad_norm(model)
    groups = {}
    for name in ("backbone", "shrink_conv", "cls_head", "reg_head", "obj_head", "stage3"):
        mod = getattr(model, name, None)
        if mod is not None:
            groups[name] = grad_group_norm(mod)
    if getattr(model, "cross_agent", None) is not None:
        groups["stage2_cross_agent"] = grad_group_norm(model.cross_agent)
    if getattr(model, "intra_view", None) is not None:
        groups["stage1_intra_view"] = grad_group_norm(model.intra_view)
    rec["groups"] = groups

    max_norm_raw = clip_cfg.get("max_norm")
    max_norm: Optional[float] = None
    if isinstance(max_norm_raw, (int, float)):
        max_norm = float(max_norm_raw)
    elif isinstance(max_norm_raw, list) and max_norm_raw:
        max_norm = float(max_norm_raw[0])
    total = rec["whole_model"]["l2"]
    rec["configured_clip_max_norm"] = max_norm
    if max_norm is not None:
        rec["would_clip"] = bool(total > max_norm)
        rec["clip_factor"] = min(1.0, max_norm / (total + EPS))
    else:
        rec["would_clip"] = None
        rec["clip_factor"] = None

    model.zero_grad(set_to_none=True)
    return rec


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def diagnose(
    spatial: Mapping[str, Any],
    detect: Mapping[str, Any],
    obj: Mapping[str, Any],
    stage3: Mapping[str, Any],
    scales: Mapping[str, Any],
    grad: Mapping[str, Any],
    historical: Mapping[str, Any],
    loss_impl: Mapping[str, Any],
) -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    excluded: List[str] = []

    def add(code: str, severity: int, evidence: str, action: str) -> None:
        findings.append(
            {
                "code": code,
                "severity": severity,
                "evidence": evidence,
                "action": action,
            }
        )

    # Spatial.
    seed = spatial.get("seed_coverage", {})
    bev = spatial.get("bev_gt_background", {})
    primary_cov = safe_float(seed.get("primary_coverage"))
    inside = safe_float(bev.get("inside_fraction"))
    gt_bg = safe_float(bev.get("gt_to_bg_mean_ratio"))
    spatial_fail = False
    if inside is not None and inside < 0.90:
        spatial_fail = True
    if primary_cov is not None and primary_cov < 0.30 and (gt_bg is None or gt_bg < 0.5):
        spatial_fail = True
    if spatial_fail:
        add(
            "SPATIAL_ALIGNMENT_RISK",
            90,
            f"GT inside BEV={fmt(inside)}, primary seed coverage={fmt(primary_cov)}, "
            f"GT/BG BEV norm ratio={fmt(gt_bg)}.",
            "Check coordinate frame, x/y ordering, range, voxel size, stride and anchor alignment before tuning losses.",
        )
    elif primary_cov is not None or inside is not None:
        excluded.append("major spatial alignment failure")

    # Detection death chain.
    proj_r = safe_float(detect.get("projected", {}).get("recall_iou03"))
    pre_r = safe_float(detect.get("prefilter", {}).get("recall_iou03"))
    nms_r = safe_float(detect.get("nms", {}).get("recall_iou03"))
    final_r = safe_float(detect.get("final", {}).get("recall_iou03"))
    n_obj_pass = int(detect.get("raw", {}).get("obj_pass", 0) or 0)
    n_gt = int(seed.get("n_gt", 0) or 0)

    pos_pass = safe_float(obj.get("positive_pass_fraction"))
    pos_median = safe_float(obj.get("positive_probability", {}).get("median"))
    threshold = safe_float(obj.get("threshold"))
    obj_max = safe_float(obj.get("probability", {}).get("max_abs"))

    obj_collapsed = False
    if obj.get("positive_available"):
        if pos_pass is not None and pos_pass < 0.20:
            obj_collapsed = True
        if pos_median is not None and threshold is not None and pos_median < threshold * 0.5:
            obj_collapsed = True
    if threshold is not None and obj_max is not None and obj_max < threshold:
        obj_collapsed = True

    if obj_collapsed or (final_r is not None and final_r < LOW_RECALL_FLOOR and n_obj_pass < max(5, int(0.2 * max(n_gt, 1)))):
        add(
            "OBJECTNESS_COLLAPSE",
            100,
            f"obj-pass={n_obj_pass}, positive pass fraction={fmt(pos_pass)}, "
            f"positive median={fmt(pos_median)}, threshold={fmt(threshold)}, "
            f"final Recall@0.3={fmt(final_r)}.",
            "Fix objectness supervision/normalization first; do not use a lower inference threshold as a substitute for training the obj head.",
        )
    elif pos_pass is not None and pos_pass >= 0.5:
        excluded.append("objectness collapse")

    if loss_impl.get("mode") == "ALL_ANCHORS_MEAN_POSMASK_TARGET":
        pos_frac = safe_float(obj.get("positive_fraction_all_anchors"))
        if pos_frac is not None and pos_frac < 0.01:
            add(
                "OBJECTNESS_ALL_ANCHOR_IMBALANCE_RISK",
                92,
                f"Current loss is all-anchor focal loss followed by mean; positives are only "
                f"{fmt(100.0 * pos_frac, 3)}% of anchors.",
                "Compare against balanced positive/negative normalization or explicit ignore handling; verify on the same epoch audit before changing thresholds.",
            )

    decode_finite = safe_float(detect.get("decode", {}).get("finite_box_fraction"))
    if decode_finite is not None and decode_finite < 0.99:
        add(
            "DECODE_FAILURE",
            85,
            f"Only {fmt(decode_finite)} of selected decoded boxes are finite.",
            "Inspect rm magnitude, anchor geometry and delta_to_boxes3d numerical range.",
        )
    elif decode_finite is not None:
        excluded.append("non-finite decode failure")

    if pre_r is not None and nms_r is not None and pre_r >= 0.50 and nms_r < LOW_RECALL_FLOOR:
        add(
            "NMS_FAILURE",
            80,
            f"Recall@0.3 drops from {fmt(pre_r)} before NMS to {fmt(nms_r)} after NMS.",
            "Audit duplicate geometry/scores and nms_thresh; do not change NMS unless the recall drop reproduces across frames.",
        )
    elif pre_r is not None and nms_r is not None and nms_r >= pre_r - 0.10:
        excluded.append("major NMS recall collapse")

    if nms_r is not None and final_r is not None and nms_r >= 0.50 and final_r < LOW_RECALL_FLOOR:
        add(
            "RANGE_FILTER_FAILURE",
            78,
            f"Recall@0.3 drops from {fmt(nms_r)} after NMS to {fmt(final_r)} after range filtering.",
            "Check lidar range and coordinate projection.",
        )
    elif nms_r is not None and final_r is not None and final_r >= nms_r - 0.10:
        excluded.append("major range-filter recall collapse")

    # Stage3.
    ratios = stage3.get("ratios", {})
    final_pool = safe_float(ratios.get("final_to_pooled_rms"))
    applied_pool = safe_float(ratios.get("applied_delta_to_pooled_rms"))
    if (
        (final_pool is not None and final_pool > FEATURE_EXPLOSION_RATIO)
        or (applied_pool is not None and applied_pool > FEATURE_EXPLOSION_RATIO)
    ):
        add(
            "STAGE3_FEATURE_EXPLOSION",
            88,
            f"final/pooled RMS={fmt(final_pool)}, applied fusion delta/pooled RMS={fmt(applied_pool)}.",
            "Inspect Stage3 attention/fusion parameter growth and gradients before touching detection thresholds.",
        )
    else:
        excluded.append("Stage3 feature explosion")

    if applied_pool is not None and applied_pool < STAGE3_BYPASS_DELTA_RATIO:
        add(
            "STAGE3_WEAK_RESIDUAL",
            62,
            f"raw fusion residual/pooled RMS={fmt(applied_pool)}.",
            "Stage3 now has no residual gate; if this stays very small across epochs, inspect fusion/attention gradients and representation usefulness rather than a gamma bypass.",
        )

    q_ratio = safe_float(
        stage3.get("param_ratios_to_fresh_init", {})
        .get("self_blocks.0.q_proj.weight", {})
        .get("ratio")
    )
    if q_ratio is not None and q_ratio < PARAM_COLLAPSE_RATIO:
        add(
            "STAGE3_SELF_QPROJ_COLLAPSE",
            72,
            f"self block q_proj L2 is {fmt(q_ratio)}x a fresh-init reference.",
            "Inspect optimizer updates and Stage3 gradients; this is a scale-collapse signal, not an exact comparison to the original run initialization.",
        )

    # Scales.
    for agent, rec in scales.get("agents", {}).items():
        for stage_name in ("stage1_over_init", "stage2_over_stage1"):
            srec = rec.get(stage_name, {})
            lower = srec.get("near_lower_frac_xyz")
            upper = srec.get("near_upper_frac_xyz")
            if isinstance(lower, list) and len(lower) >= 2:
                xy_low = float(np.mean(lower[:2]))
                if xy_low > SCALE_BOUND_FRACTION:
                    add(
                        f"{agent.upper()}_SCALE_SHRINK_SATURATION",
                        60,
                        f"{stage_name}: XY near-lower-bound fraction={fmt(xy_low)}.",
                        "Inspect the agent-specific scale head logits and whether the asymmetric delta range is driving persistent contraction.",
                    )
            if isinstance(upper, list) and len(upper) >= 2:
                xy_hi = float(np.mean(upper[:2]))
                if xy_hi > SCALE_BOUND_FRACTION:
                    add(
                        f"{agent.upper()}_SCALE_EXPAND_SATURATION",
                        60,
                        f"{stage_name}: XY near-upper-bound fraction={fmt(xy_hi)}.",
                        "Inspect the agent-specific scale head logits and whether scale expansion is saturating against the configured upper bound.",
                    )

    # Gradients / clipping.
    clip_hist = historical.get("grad_clip", {})
    clip_rate = safe_float(clip_hist.get("clip_rate"))
    if clip_rate is not None and clip_rate > CLIP_ALWAYS_RATE:
        add(
            "GRADIENT_CLIP_ALWAYS_ON",
            82,
            f"Historical epoch clip rate={fmt(clip_rate)} ({clip_hist.get('clipped')}/{clip_hist.get('steps')}).",
            "Treat the configured max_norm as an active optimizer constraint; inspect which modules dominate the pre-clip norm.",
        )

    clip_factor = safe_float(grad.get("clip_factor"))
    if clip_factor is not None and clip_factor < SEVERE_CLIP_FACTOR:
        add(
            "SEVERE_CLIPPING_ON_PROBE",
            68,
            f"Probe pre-clip norm={fmt(grad.get('whole_model', {}).get('l2'))}, "
            f"clip factor={fmt(clip_factor)}, max_norm={fmt(grad.get('configured_clip_max_norm'))}.",
            "Compare the probe with grad_clip.txt across the full epoch before changing max_norm.",
        )

    groups = grad.get("groups", {})
    reg_g = safe_float(groups.get("reg_head", {}).get("l2"))
    obj_g = safe_float(groups.get("obj_head", {}).get("l2"))
    if reg_g is not None and obj_g is not None and obj_g > 0 and reg_g / obj_g > 10.0:
        add(
            "REG_GRADIENT_DOMINATES_OBJ_HEAD",
            62,
            f"reg_head grad L2 / obj_head grad L2 = {fmt(reg_g / (obj_g + EPS))}.",
            "Inspect loss normalization/weights and shared-backbone gradient competition.",
        )

    stage3_g = safe_float(groups.get("stage3", {}).get("l2"))
    backbone_g = safe_float(groups.get("backbone", {}).get("l2"))
    if (
        stage3_g is not None and backbone_g is not None and backbone_g > 0
        and stage3_g / backbone_g < 1e-3
    ):
        add(
            "STAGE3_GRADIENT_STARVED",
            65,
            f"stage3/backbone gradient L2 ratio={fmt(stage3_g / (backbone_g + EPS))}.",
            "Stage3 has no residual gamma in the standard parameterization; inspect attention/fusion gradients, optimizer dynamics, and downstream loss signal.",
        )

    findings.sort(key=lambda x: (-int(x["severity"]), x["code"]))
    if findings:
        overall = "FAIL" if findings[0]["severity"] >= 80 else "WARN"
    else:
        overall = "PASS"
    return {
        "overall": overall,
        "findings": findings,
        "excluded": excluded,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_report(payload: Mapping[str, Any]) -> str:
    lines: List[str] = []
    meta = payload["meta"]
    lines.append("AirV2X Gaussian Epoch Health Report")
    lines.append("=" * 68)
    lines.append(f"Config:      {meta['config']}")
    lines.append(f"Checkpoint:  {meta['checkpoint']}")
    lines.append(f"Epoch:       {meta['epoch']}")
    lines.append(f"Split/index: {meta['split']} / {meta['index']}")
    lines.append(f"Frame:       {meta['frame']}")
    lines.append(f"Device:      {meta['device']}")
    lines.append(f"Torch:       {meta['torch']}")
    lines.append(f"Model:       {meta['model_core_method']}")

    ck = payload["checkpoint_load"]
    lines.append(section("Checkpoint Load"))
    lines.append(
        f"Loaded compatible keys: {ck['loaded_keys']}/{ck['checkpoint_keys']}\n"
        f"Missing current-model keys: {len(ck['missing_keys'])}\n"
        f"Unexpected checkpoint keys: {len(ck['unexpected_keys'])}\n"
        f"Shape mismatches: {len(ck['shape_mismatch'])}"
    )
    if ck["shape_mismatch"]:
        lines.append("First shape mismatches:")
        for row in ck["shape_mismatch"][:8]:
            lines.append(f"  - {row}")

    src = payload["source_audit"]
    lines.append(section("Production Source Audit"))
    lines.append(
        f"Objectness loss implementation: {src['objectness_loss']['mode']}\n"
        f"Criterion: {src['objectness_loss']['module']}.{src['objectness_loss']['class']}\n"
        f"Configured train.py max_norm: {fmt(src['grad_clip'].get('max_norm'))}\n"
        f"Production post-process gate: objectness > obj_threshold\n"
        f"Production class-score threshold participates in mask: NO"
    )

    hist = payload["historical"]
    lines.append(section("Historical Run Logs"))
    if "train_loss" in hist:
        t = hist["train_loss"]
        lines.append(
            f"Train epoch index: {hist['train_epoch_index']}\n"
            f"Train loss mean/median/last: {fmt(t['mean'])} / {fmt(t['median'])} / {fmt(t['last'])}"
        )
    else:
        lines.append("Train loss: unavailable for this checkpoint epoch.")
    if "validation_loss" in hist:
        lines.append(f"Validation loss: {fmt(hist['validation_loss'])}")
    if "grad_clip" in hist:
        g = hist["grad_clip"]
        lines.append(
            f"Grad clipping: {g['clipped']}/{g['steps']} "
            f"(rate={fmt(g['clip_rate'])}), nonfinite={g['nonfinite']}"
        )
    else:
        lines.append("grad_clip.txt entry: unavailable.")

    sp = payload["spatial"]
    lines.append(section("Spatial Alignment"))
    seed = sp["seed_coverage"]
    bev = sp["bev_gt_background"]
    if seed.get("available"):
        lines.append(
            f"GT: {seed.get('n_gt')} | initial Gaussians: {seed.get('n_seed')}\n"
            f"GT-to-seed min distance median: {fmt(seed.get('min_distance_m', {}).get('median'))} m\n"
            f"Seed coverage: "
            + ", ".join(f"{k}={fmt(v)}" for k, v in seed.get("coverage", {}).items())
        )
    else:
        lines.append(f"Seed coverage unavailable: {seed.get('reason', 'unknown')}")
    if bev.get("available"):
        lines.append(
            f"GT inside Stage3 BEV: {bev.get('inside_grid')}/{bev.get('n_gt')} "
            f"({fmt(bev.get('inside_fraction'))})\n"
            f"GT cell feature norm mean: {fmt(bev.get('gt_cell_norm', {}).get('mean'))}\n"
            f"Nonzero BG feature norm mean: {fmt(bev.get('background_nonzero_cell_norm', {}).get('mean'))}\n"
            f"GT/BG mean norm ratio: {fmt(bev.get('gt_to_bg_mean_ratio'))}\n"
            f"BEV nonzero fraction: {fmt(bev.get('bev_nonzero_fraction'))}"
        )

    det = payload["detection"]
    lines.append(section("Detection Death Chain"))
    raw = det["raw"]
    lines.append(
        f"Total anchors: {raw['total_anchors']}\n"
        f"obj > production threshold: {raw['obj_pass']}\n"
        f"class score > score_threshold (diagnostic only): {raw['class_score_pass_diagnostic_only']}\n"
        f"both pass (diagnostic only): {raw['both_pass_diagnostic_only']}\n"
        f"Objectness mean/max: {fmt(raw['objectness'].get('mean'))} / "
        f"{fmt(raw['objectness'].get('max_abs'))}\n"
        f"Class-score mean/max: {fmt(raw['class_scores'].get('mean'))} / "
        f"{fmt(raw['class_scores'].get('max_abs'))}"
    )
    for name in ("projected", "prefilter", "nms", "range", "final"):
        r = det.get(name, {})
        lines.append(
            f"{name:10s}: count={r.get('count', 'N/A')}, "
            f"Recall@0.3={fmt(r.get('recall_iou03'))}"
        )
    lines.append(
        f"Decode finite fraction: {fmt(det.get('decode', {}).get('finite_box_fraction'))}"
    )

    oh = payload["objectness"]
    lines.append(section("Objectness Health"))
    lines.append(
        f"Threshold: {fmt(oh.get('threshold'))}\n"
        f"Global pass: {oh.get('global_pass_count')} "
        f"({fmt(oh.get('global_pass_fraction'))})\n"
        f"Positive anchors: {oh.get('n_positive', 'N/A')} "
        f"({fmt(oh.get('positive_fraction_all_anchors'))} of all anchors)\n"
        f"Positive obj mean/median/max: "
        f"{fmt(oh.get('positive_probability', {}).get('mean'))} / "
        f"{fmt(oh.get('positive_probability', {}).get('median'))} / "
        f"{fmt(oh.get('positive_probability', {}).get('max_abs'))}\n"
        f"Positive pass fraction: {fmt(oh.get('positive_pass_fraction'))}"
    )
    if "n_explicit_negative" in oh:
        lines.append(
            f"Explicit negatives: {oh.get('n_explicit_negative')} | "
            f"ignored-by-labeler anchors: {oh.get('n_ignored_anchor_by_labeler')}"
        )

    s3 = payload["stage3"]
    lines.append(section("Stage3 Numerical / Collapse Audit"))
    rr = s3.get("ratios", {})
    lines.append(
        f"pooled RMS: {fmt(s3.get('features', {}).get('pooled', {}).get('rms'))}\n"
        f"final RMS: {fmt(s3.get('features', {}).get('final', {}).get('rms'))}\n"
        f"final/pooled RMS: {fmt(rr.get('final_to_pooled_rms'))}\n"
        f"self change/input RMS: {fmt(rr.get('self_change_to_input_rms'))}\n"
        f"cross change/input RMS: {fmt(rr.get('cross_change_to_input_rms'))}\n"
        f"raw/applied fusion delta/pooled RMS: {fmt(rr.get('applied_delta_to_pooled_rms'))}\n"
        f"Stage3 residual parameterization: {s3.get('residual_parameterization', {})}"
    )
    for key, val in s3.get("param_ratios_to_fresh_init", {}).items():
        lines.append(
            f"{key}: current L2={fmt(val.get('current_l2'))}, "
            f"fresh-init L2={fmt(val.get('fresh_init_l2'))}, ratio={fmt(val.get('ratio'))}"
        )
    lines.append(f"Splat stats: {s3.get('splat_stats', {})}")

    sc = payload["scales"]
    lines.append(section("Gaussian Scale Drift"))
    lines.append(f"Configured per-stage log-scale bounds: {sc.get('delta_bounds')}")
    for agent, ar in sc.get("agents", {}).items():
        lines.append(f"\n[{agent}]")
        for stage_name in ("init", "stage1", "stage2"):
            sr = ar.get(stage_name, {})
            lines.append(
                f"  {stage_name:7s} n={sr.get('n')} median xyz={sr.get('xyz_median')}"
            )
        for delta_name in ("stage1_over_init", "stage2_over_stage1"):
            dr = ar.get(delta_name, {})
            if dr.get("available"):
                lines.append(
                    f"  {delta_name}: median log-delta xyz={dr.get('xyz_median')}\n"
                    f"    near lower xyz={dr.get('near_lower_frac_xyz')}\n"
                    f"    near upper xyz={dr.get('near_upper_frac_xyz')}"
                )

    gr = payload["gradient"]
    lines.append(section("Loss / Gradient / Clip"))
    lines.append(
        f"Probe loss: {fmt(gr.get('loss'))} | finite={gr.get('loss_finite')}\n"
        f"Loss dict: {gr.get('criterion_loss_dict', {})}\n"
        f"Whole-model pre-clip grad L2: {fmt(gr.get('whole_model', {}).get('l2'))}\n"
        f"Configured max_norm: {fmt(gr.get('configured_clip_max_norm'))}\n"
        f"Would clip on probe: {gr.get('would_clip')}\n"
        f"Probe clip factor: {fmt(gr.get('clip_factor'))}"
    )
    for name, row in gr.get("groups", {}).items():
        lines.append(
            f"{name:20s} grad L2={fmt(row.get('l2'))}, "
            f"max|g|={fmt(row.get('max_abs'))}, nonfinite={row.get('nonfinite_grad_values')}"
        )

    dg = payload["diagnosis"]
    lines.append(section("FINAL DIAGNOSIS"))
    lines.append(f"Overall: {dg['overall']}")
    if dg["findings"]:
        for i, f in enumerate(dg["findings"], 1):
            lines.append(
                f"\n{i}. {f['code']}  [severity={f['severity']}]\n"
                f"   Evidence: {f['evidence']}\n"
                f"   Next action: {f['action']}"
            )
    else:
        lines.append(
            "\nNo strong failure matched the diagnostic thresholds on this fixed frame."
        )
    if dg.get("excluded"):
        lines.append("\nExcluded / not supported by this probe:")
        for x in dg["excluded"]:
            lines.append(f"  - {x}")

    lines.append(
        "\nNotes:\n"
        "  * This is a deterministic fixed-frame diagnostic, not a replacement for full AP evaluation.\n"
        "  * 'fresh-init' parameter ratios compare against a newly constructed same-architecture "
        "reference, not the exact historical initialization of this run.\n"
        "  * Diagnostic thresholds are heuristics. Reproduce a warning across multiple epochs/frames "
        "before changing training behavior."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    opt = parse_args()
    config, model_dir, ckpt = resolve_paths(opt.config, opt.epoch)
    out_dir = Path(opt.out_dir).expanduser().resolve() if opt.out_dir else model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / f"epoch{opt.epoch}_health_report.txt"
    out_json = out_dir / f"epoch{opt.epoch}_health_metrics.json"

    device = torch.device(
        f"cuda:{int(opt.gpu)}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    hypes = load_hypes(config, opt.split)
    core_method = str(hypes.get("model", {}).get("core_method", ""))
    if "gaussian_0822" not in core_method.lower():
        raise RuntimeError(
            f"This audit targets airv2x_gaussian_0822, but config core_method={core_method!r}"
        )

    print(f"[health] config: {config}")
    print(f"[health] checkpoint: {ckpt}")
    print(f"[health] device: {device}")
    print(f"[health] building {opt.split} dataset ...")
    dataset = build_dataset(hypes, visualize=False, train=False)
    print(f"[health] dataset size: {len(dataset)}")

    # Construct model first and retain a same-architecture fresh-init Stage3 reference.
    torch.manual_seed(0)
    model = train_utils.create_model(hypes).to(device)
    fresh_init_stage3 = param_l2_dict(model.stage3, prefix="stage3")
    load_rec = load_checkpoint(model, ckpt, device)
    model.eval()

    batch, frame = load_fixed_batch(dataset, opt.index, device)
    batch["ego"]["epoch"] = int(opt.epoch)

    print(f"[health] fixed frame: {frame}")
    print("[health] forward + Stage1/2/3 instrumentation ...")
    with torch.no_grad():
        pipe = run_pipeline(model, batch["ego"], fresh_init_stage3)

    gt_boxes, gt_meta = get_gt_boxes(dataset, {"ego": batch["ego"]})
    if gt_boxes is not None:
        print(f"[health] GT boxes: {int(gt_boxes.shape[0])}")
    else:
        print("[health] GT unavailable; recall/spatial checks will be partial")

    spatial = {
        "gt_meta": gt_meta,
        "seed_coverage": seed_coverage(
            pipe["init_g"], gt_boxes, primary_radius=float(opt.seed_radius)
        ),
        "bev_gt_background": bev_gt_background(
            pipe["bev"], model.stage3, gt_boxes
        ),
    }

    print("[health] detection death-chain audit ...")
    detection = detection_chain(
        dataset, batch["ego"], pipe["output"], gt_boxes
    )

    obj_threshold = float(
        hypes["postprocess"]["target_args"]["obj_threshold"]
    )
    objectness = objectness_health(
        pipe["output"], batch["ego"]["label_dict"], obj_threshold
    )
    scales = analyze_scales(
        pipe["init_g"], pipe["stage1_g"], pipe["stage2_g"], hypes
    )

    criterion = train_utils.create_loss(hypes)
    loss_impl = inspect_objectness_loss(criterion)
    clip_cfg = detect_grad_clip_config()
    historical = read_historical_logs(model_dir, opt.epoch)

    print("[health] one-batch no-step backward probe ...")
    # Fresh batch: the first one was mutated by the manual pipeline.
    grad_batch, _ = load_fixed_batch(dataset, opt.index, device)
    gradient = backward_probe(
        model, criterion, grad_batch, opt.epoch, clip_cfg
    )

    # Drop internal full prediction tensors before JSON serialization.
    detection_json = dict(detection)
    detection_json.pop("final_boxes", None)
    detection_json.pop("final_scores", None)

    diagnosis = diagnose(
        spatial=spatial,
        detect=detection,
        obj=objectness,
        stage3=pipe["stage3"],
        scales=scales,
        grad=gradient,
        historical=historical,
        loss_impl=loss_impl,
    )

    payload: Dict[str, Any] = {
        "meta": {
            "config": str(config),
            "model_dir": str(model_dir),
            "checkpoint": str(ckpt),
            "epoch": int(opt.epoch),
            "split": opt.split,
            "index": int(opt.index),
            "frame": frame,
            "device": str(device),
            "torch": torch.__version__,
            "cuda_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda" else None
            ),
            "model_core_method": core_method,
            "dataset_class": dataset.__class__.__name__,
        },
        "checkpoint_load": load_rec,
        "source_audit": {
            "objectness_loss": loss_impl,
            "grad_clip": clip_cfg,
        },
        "historical": historical,
        "spatial": spatial,
        "detection": detection_json,
        "objectness": objectness,
        "stage3": pipe["stage3"],
        "scales": scales,
        "gradient": gradient,
        "diagnosis": diagnosis,
    }

    out_json.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = render_report(jsonable(payload))
    out_txt.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"[health] wrote: {out_txt}")
    print(f"[health] wrote: {out_json}")


if __name__ == "__main__":
    main()

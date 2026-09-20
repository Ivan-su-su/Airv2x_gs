#!/usr/bin/env python3
"""
Offline Initializer/Stage-1/Stage-2 Gaussian geometry-correction audit for AirV2X.

This script measures whether Stage-2 cross-agent interaction moves each
Gaussian center closer to the GT 3D point associated with the SAME source
camera pixel.

Core contract
-------------
1. Run the production pipeline only through Stage-2:
       P1 -> initializer -> Stage-1 -> Stage-2
2. Preserve exact Gaussian identity (NO nearest-neighbor rematching).
3. Read GT optical-z from the already augmentation-aligned depth channel in
   batch_merged_cam_inputs["imgs"][:, :, 3, ...] / flattened equivalent.
4. Backproject that exact pixel with the production camera geometry helpers.
5. Transform GT points to ego frame and compare against Stage-1 / Stage-2
   Gaussian means in ego frame.
6. Recompute Stage-2's valid source-token mask using the current production
   _sample_tokens path, without changing model behavior.

Typical usage
-------------
python opencood/tools/analyze_stage2_geometry_correction.py \
    --model_dir opencood/logs/airv2x_gaussian_0822_camera/<run> \
    --checkpoint opencood/logs/airv2x_gaussian_0822_camera/<run>/net_epoch20.pth \
    --split test \
    --sample_n 200 \
    --seed 42 \
    --gpu_id 0 \
    --out_dir opencood/logs/airv2x_gaussian_0822_camera/<run>/stage2_geometry_audit

Notes
-----
- This is an OFFLINE analysis script. It does not train or modify weights.
- Primary GT mask defaults to each agent's configured camera depth range.
- Use --gt_mask raw to score all finite CARLA depth values below --raw_depth_max.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.models.airv2x_gaussian_0822 import (
    CAMERA_GEOMETRY_KEY,
    present_camera_agents,
)
from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.base.projection import (
    lift_optical_z_to_lidar,
    project_lidar_to_image,
)
from opencood.models.gaussian_modules_0822.cross_agent.interaction import (
    INTERACTION_TYPES,
    interaction_sources,
)
from opencood.tools import train_utils


DEFAULT_YAML = (
    ROOT / "opencood/hypes_yaml/airv2x/camera/det/airv2x_gaussian_0822.yaml"
)
AGENT_ORDER = ("vehicle", "rsu", "drone")


# -----------------------------------------------------------------------------
# CLI / loading
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit Stage-2 Gaussian center correction against GT depth geometry."
    )
    p.add_argument("--hypes_yaml", type=str, default="")
    p.add_argument("--model_dir", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument(
        "--sample_n",
        type=int,
        default=100,
        help="Number of dataset samples. 0 means full split.",
    )
    p.add_argument(
        "--indices",
        type=str,
        default="",
        help="Optional comma-separated dataset indices; overrides --sample_n.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument(
        "--gt_mask",
        choices=["model_range", "raw"],
        default="model_range",
        help="Primary GT validity mask used for reported geometry metrics.",
    )
    p.add_argument("--raw_depth_min", type=float, default=1.0e-3)
    p.add_argument(
        "--raw_depth_max",
        type=float,
        default=999.0,
        help="Exclude CARLA far-plane / invalid values near 1000 m.",
    )
    p.add_argument("--improvement_eps", type=float, default=0.01)
    p.add_argument("--movement_eps", type=float, default=1.0e-4)
    p.add_argument("--error_eps", type=float, default=1.0e-6)
    p.add_argument(
        "--depth_bins",
        type=str,
        default="0,20,40,60,100,150,1000",
    )
    p.add_argument(
        "--initial_error_bins",
        type=str,
        default="0,0.5,1,2,4,8,inf",
    )
    p.add_argument(
        "--plot_max_points",
        type=int,
        default=100000,
        help="Max points in scatter/hist plotting only; CSV/statistics remain complete.",
    )
    p.add_argument(
        "--roundtrip_n",
        type=int,
        default=64,
        help="Max valid points per agent/sample used for camera geometry round-trip check.",
    )
    p.add_argument("--roundtrip_tol_px", type=float, default=0.02)
    p.add_argument("--roundtrip_tol_depth", type=float, default=0.02)
    p.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip matplotlib plots; all numeric outputs are still written.",
    )
    p.add_argument(
        "--strict_checkpoint",
        action="store_true",
        help="Fail on any missing/unexpected checkpoint key instead of only critical Stage1/2 keys.",
    )
    return p.parse_args()


def _resolve_yaml(args: argparse.Namespace) -> Path:
    if args.hypes_yaml:
        path = Path(args.hypes_yaml)
        if not path.is_file():
            raise FileNotFoundError(f"--hypes_yaml not found: {path}")
        return path

    if args.model_dir:
        model_dir = Path(args.model_dir)
        for name in ("config.yaml", "config.yml", "hypes.yaml", "hypes.yml"):
            candidate = model_dir / name
            if candidate.is_file():
                return candidate

    if DEFAULT_YAML.is_file():
        return DEFAULT_YAML
    raise FileNotFoundError(
        "Could not resolve YAML. Pass --hypes_yaml explicitly."
    )


def _checkpoint_epoch_key(path: Path) -> Tuple[int, str]:
    nums = re.findall(r"(?:epoch|net_epoch|ep)[_-]?(\d+)", path.stem, flags=re.I)
    epoch = int(nums[-1]) if nums else -1
    return epoch, path.name


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"--checkpoint not found: {path}")
        return path

    if not args.model_dir:
        raise ValueError("Pass --checkpoint or --model_dir; random Stage1/2 is not meaningful.")

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"--model_dir not found: {model_dir}")

    preferred = [
        model_dir / "net_latest.pth",
        model_dir / "latest.pth",
        model_dir / "net_final.pth",
    ]
    for p in preferred:
        if p.is_file():
            return p

    candidates = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    candidates.sort(key=_checkpoint_epoch_key)
    return candidates[-1]


def _load_hypes(yaml_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    ns = argparse.Namespace(
        model_dir=args.model_dir or "",
        hypes_yaml=str(yaml_path),
    )
    hypes = yaml_utils.load_yaml(str(yaml_path), ns)
    if not isinstance(hypes, dict):
        raise TypeError(f"yaml_utils.load_yaml returned {type(hypes)}")
    return hypes


def _load_weights(
    model: torch.nn.Module,
    checkpoint: Path,
    device: torch.device,
    strict_all: bool,
) -> Dict[str, Any]:
    blob = torch.load(str(checkpoint), map_location=device)
    if isinstance(blob, Mapping):
        src = blob.get("model_state_dict", blob.get("state_dict", blob.get("model", blob)))
    else:
        raise TypeError(f"checkpoint must be dict-like, got {type(blob)}")

    if not isinstance(src, Mapping):
        raise TypeError(f"checkpoint state_dict must be mapping, got {type(src)}")

    state = {
        (k[7:] if str(k).startswith("module.") else str(k)): v
        for k, v in src.items()
        if torch.is_tensor(v)
    }
    missing, unexpected = model.load_state_dict(state, strict=False)

    critical_missing = [
        k for k in missing if k.startswith("intra_view.") or k.startswith("cross_agent.")
    ]
    if critical_missing:
        raise RuntimeError(
            "Checkpoint is missing Stage1/2 parameters; refusing geometry audit. "
            f"Examples: {critical_missing[:12]}"
        )
    if strict_all and (missing or unexpected):
        raise RuntimeError(
            f"strict checkpoint failure: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )

    report = {
        "checkpoint": str(checkpoint),
        "n_tensor_keys": len(state),
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "missing_examples": list(missing[:20]),
        "unexpected_examples": list(unexpected[:20]),
    }
    print(
        f"[checkpoint] {checkpoint} tensors={len(state)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    return report


def _make_dataset(
    hypes: Dict[str, Any], split: str
) -> Tuple[Any, Dict[str, Any], bool]:
    ds_hypes = copy.deepcopy(hypes)
    if split == "test":
        if "test_dir" not in hypes:
            raise KeyError("YAML has no test_dir")
        ds_hypes["validate_dir"] = hypes["test_dir"]
        train_flag = False
    elif split == "val":
        train_flag = False
    else:
        train_flag = True
    dataset = build_dataset(ds_hypes, visualize=False, train=train_flag)
    return dataset, ds_hypes, train_flag


def _select_indices(
    n_total: int, sample_n: int, seed: int, indices_arg: str
) -> List[int]:
    if indices_arg.strip():
        out = [int(x.strip()) for x in indices_arg.split(",") if x.strip()]
        for idx in out:
            if idx < 0 or idx >= n_total:
                raise IndexError(f"dataset index {idx} outside [0, {n_total})")
        return out

    if sample_n <= 0 or sample_n >= n_total:
        return list(range(n_total))

    rng = np.random.RandomState(seed)
    out = rng.choice(n_total, size=sample_n, replace=False).astype(np.int64).tolist()
    out.sort()
    return out


# -----------------------------------------------------------------------------
# Tensor / metadata helpers
# -----------------------------------------------------------------------------


def _flatten_imgs(imgs: torch.Tensor) -> torch.Tensor:
    """Return [N_flat, C, H, W] from [Ncav,V,C,H,W] or [N,C,H,W]."""
    if imgs.dim() == 5:
        return imgs.reshape(-1, *imgs.shape[-3:])
    if imgs.dim() == 4:
        return imgs
    raise ValueError(f"camera imgs must be 4D/5D, got {tuple(imgs.shape)}")


def _safe_scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _safe_scalar(value[0])
        return str(value)
    return value


def _metadata(ego: Mapping[str, Any], dataset_index: int) -> Dict[str, Any]:
    return {
        "dataset_index": int(dataset_index),
        "scenario": _safe_scalar(ego.get("scenario_index", "")),
        "timestamp": _safe_scalar(ego.get("timestamp_key", "")),
        "metadata_path": _safe_scalar(ego.get("metadata_path", "")),
    }


def _agent_depth_range(
    hypes: Mapping[str, Any], agent: str
) -> Tuple[float, float]:
    try:
        ddiscr = hypes["model"]["args"][agent]["cam"]["grid_conf"]["ddiscr"]
        return float(ddiscr[0]), float(ddiscr[1])
    except Exception:
        try:
            ddiscr = hypes["fusion"]["args"][f"{agent if agent != 'vehicle' else 'veh'}_grid_conf"]["ddiscr"]
            return float(ddiscr[0]), float(ddiscr[1])
        except Exception as exc:
            raise KeyError(f"Cannot resolve configured camera depth range for {agent}") from exc


def _record_len(payload: Mapping[str, Any]) -> int:
    value = payload.get("record_len", 0)
    if torch.is_tensor(value):
        return int(value.reshape(-1).sum().item())
    if isinstance(value, (list, tuple)):
        return int(sum(int(v) for v in value))
    return int(value or 0)


def _cav_id_for_view(payload: Mapping[str, Any], local_cav_idx: int) -> Any:
    for key in ("cav_ids", "cav_id", "agent_ids"):
        ids = payload.get(key)
        if ids is None:
            continue
        try:
            if torch.is_tensor(ids):
                ids = ids.detach().cpu().reshape(-1).tolist()
            if isinstance(ids, (list, tuple)) and local_cav_idx < len(ids):
                return ids[local_cav_idx]
        except Exception:
            pass
    return local_cav_idx


# -----------------------------------------------------------------------------
# Pipeline replay + exact Stage-2 valid mask
# -----------------------------------------------------------------------------


def _replay_to_stage2(
    model: torch.nn.Module,
    batch: MutableMapping[str, Any],
) -> Tuple[
    Dict[str, GaussianSet],
    Dict[str, GaussianSet],
    Dict[str, GaussianSet],
    Dict[str, Mapping[str, Any]],
    Dict[str, torch.Tensor],
    Dict[str, Any],
]:
    """Replay production math through Stage-2 and return exact diagnostics."""
    data = batch["ego"]

    for agent in present_camera_agents(data):
        model._encode_agent(agent, data)

    init_sets = model.initializer.forward(data)

    stage1_sets: Dict[str, GaussianSet] = {}
    for agent, gs0 in init_sets.items():
        stage1_sets[agent] = model.intra_view(
            gs0,
            data[agent]["f90"],
            data[agent][CAMERA_GEOMETRY_KEY],
            agent=agent,
        )

    sources: Dict[str, Mapping[str, Any]] = {
        agent: {
            "features": data[agent]["f90"],
            "geometry": data[agent][CAMERA_GEOMETRY_KEY],
        }
        for agent in stage1_sets
    }

    stage2_valid_masks = _exact_stage2_valid_masks(model, stage1_sets, sources)
    stage2_sets = model.cross_agent(stage1_sets, sources)

    stage2_stats = copy.deepcopy(getattr(model.cross_agent, "last_stats", {}))

    for agent in stage1_sets:
        _assert_identity(stage1_sets[agent], stage2_sets[agent], agent)

        valid = stage2_valid_masks.get(agent)
        if valid is not None:
            invalid = ~valid
            if bool(invalid.any()):
                # Production contract: no valid source tokens -> exact identity.
                if not torch.equal(
                    stage1_sets[agent].mean[invalid],
                    stage2_sets[agent].mean[invalid],
                ):
                    max_diff = float(
                        (stage1_sets[agent].mean[invalid] - stage2_sets[agent].mean[invalid])
                        .abs()
                        .max()
                        .item()
                    )
                    raise AssertionError(
                        f"{agent}: Stage2 invalid-token Gaussians changed; max |delta mean|={max_diff}"
                    )

    return (
        init_sets,
        stage1_sets,
        stage2_sets,
        sources,
        stage2_valid_masks,
        stage2_stats,
    )


def _assert_identity(s1: GaussianSet, s2: GaussianSet, agent: str) -> None:
    if s1.n_gaussians != s2.n_gaussians:
        raise AssertionError(
            f"{agent}: Stage1 N={s1.n_gaussians} != Stage2 N={s2.n_gaussians}"
        )
    if s1.agent != s2.agent:
        raise AssertionError(f"{agent}: agent tag changed {s1.agent!r}->{s2.agent!r}")
    if s1.frame != "ego" or s2.frame != "ego":
        raise AssertionError(
            f"{agent}: expected ego frames, got Stage1={s1.frame}, Stage2={s2.frame}"
        )

    for name in ("view_index", "batch_index"):
        a = getattr(s1, name)
        b = getattr(s2, name)
        if a is None or b is None:
            raise AssertionError(f"{agent}: {name} missing")
        if not torch.equal(a, b):
            raise AssertionError(f"{agent}: {name} changed from Stage1 to Stage2")

    if s1.uv is None or s2.uv is None:
        raise AssertionError(f"{agent}: uv provenance missing")
    if not torch.equal(s1.uv, s2.uv):
        max_diff = float((s1.uv - s2.uv).abs().max().item())
        raise AssertionError(f"{agent}: uv changed; max diff={max_diff}")


def _exact_stage2_valid_masks(
    model: torch.nn.Module,
    stage1_sets: Mapping[str, GaussianSet],
    sources: Mapping[str, Mapping[str, Any]],
) -> Dict[str, torch.Tensor]:
    """Recompute the exact `mask.any(dim=1)` used by CrossAgentInteraction.

    This calls the production sampler + `_sample_tokens` implementation but does
    not run attention/refinement and does not mutate model outputs.
    """
    cross = model.cross_agent
    out: Dict[str, torch.Tensor] = {
        agent: torch.zeros(
            gs.n_gaussians, dtype=torch.bool, device=gs.mean.device
        )
        for agent, gs in stage1_sets.items()
    }

    for interaction_type in INTERACTION_TYPES:
        query_agent, source_agents = interaction_sources(interaction_type)
        if query_agent not in stage1_sets:
            continue
        gs = stage1_sets[query_agent]
        if gs.n_gaussians == 0:
            continue

        present_sources = tuple(a for a in source_agents if a in sources)
        if not present_sources:
            continue

        points = cross.sampler(
            gs.mean,
            gs.scale,
            gs.quaternion,
            gs.feature,
            query_agent,
        )["samples"]

        masks: List[torch.Tensor] = []
        for source in present_sources:
            payload = sources[source]
            _tokens, mask = cross._sample_tokens(
                points,
                payload["features"],
                payload["geometry"],
            )
            masks.append(mask)

        if masks:
            valid = torch.cat(masks, dim=1).any(dim=1)
            out[query_agent] |= valid

    return out


# -----------------------------------------------------------------------------
# GT geometry
# -----------------------------------------------------------------------------


def _gt_geometry_for_gaussians(
    s1: GaussianSet,
    payload: Mapping[str, Any],
    hypes: Mapping[str, Any],
    agent: str,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    """Sample source-pixel GT optical-z and backproject it to ego frame."""
    if s1.uv is None or s1.view_index is None:
        raise AssertionError(f"{agent}: Gaussian provenance uv/view_index is required")

    cam_inputs = payload["batch_merged_cam_inputs"]
    imgs = _flatten_imgs(cam_inputs["imgs"])
    if int(imgs.shape[1]) < 4:
        raise ValueError(
            f"{agent}: camera input has C={imgs.shape[1]}; GT depth channel 3 is missing"
        )

    gt_depth_maps = imgs[:, 3].to(device=s1.mean.device, dtype=s1.mean.dtype)
    n_flat, height, width = (
        int(gt_depth_maps.shape[0]),
        int(gt_depth_maps.shape[1]),
        int(gt_depth_maps.shape[2]),
    )

    view = s1.view_index.long()
    if view.numel() and (
        int(view.min().item()) < 0 or int(view.max().item()) >= n_flat
    ):
        raise IndexError(
            f"{agent}: view_index range [{int(view.min())},{int(view.max())}] "
            f"outside flattened camera count {n_flat}"
        )

    uv = s1.uv.to(dtype=s1.mean.dtype)
    u_int = torch.round(uv[:, 0]).long().clamp(0, width - 1)
    v_int = torch.round(uv[:, 1]).long().clamp(0, height - 1)

    gt_depth = gt_depth_maps[view, v_int, u_int]

    raw_valid = (
        torch.isfinite(gt_depth)
        & (gt_depth > float(args.raw_depth_min))
        & (gt_depth < float(args.raw_depth_max))
    )
    d_min, d_max = _agent_depth_range(hypes, agent)
    model_valid = raw_valid & (gt_depth >= d_min) & (gt_depth <= d_max)

    primary_valid = model_valid if args.gt_mask == "model_range" else raw_valid

    geometry = payload[CAMERA_GEOMETRY_KEY]
    gt_agent = torch.full_like(s1.mean, float("nan"))
    gt_ego = torch.full_like(s1.mean, float("nan"))

    valid_idx = primary_valid.nonzero(as_tuple=False).squeeze(1)
    if valid_idx.numel() > 0:
        vi = view[valid_idx]
        uv_i = torch.stack(
            [
                u_int[valid_idx].to(dtype=s1.mean.dtype),
                v_int[valid_idx].to(dtype=s1.mean.dtype),
            ],
            dim=-1,
        )
        depth_i = gt_depth[valid_idx]

        intr = geometry["intrinsics"][vi].to(dtype=s1.mean.dtype)
        post_rot = geometry["post_rots"][vi].to(dtype=s1.mean.dtype)
        post_trans = geometry["post_trans"][vi].to(dtype=s1.mean.dtype)
        cam_rot = geometry["cam2lidar_rot"][vi].to(dtype=s1.mean.dtype)
        cam_trans = geometry["cam2lidar_trans"][vi].to(dtype=s1.mean.dtype)

        xyz_agent, _ray, _uv_orig = lift_optical_z_to_lidar(
            uv_aug=uv_i,
            depth_z=depth_i,
            intrinsic=intr,
            cam2lidar_rot=cam_rot,
            cam2lidar_trans=cam_trans,
            post_rot=post_rot,
            post_trans=post_trans,
        )

        if not bool(torch.isfinite(xyz_agent).all()):
            raise FloatingPointError(f"{agent}: non-finite GT agent-frame xyz")

        t_agent_to_ego = geometry["agent_to_ego"][vi].to(dtype=s1.mean.dtype)
        ones = torch.ones(
            (xyz_agent.shape[0], 1),
            device=xyz_agent.device,
            dtype=xyz_agent.dtype,
        )
        xyz_h = torch.cat([xyz_agent, ones], dim=-1)
        xyz_ego = torch.matmul(t_agent_to_ego, xyz_h.unsqueeze(-1)).squeeze(-1)[
            :, :3
        ]

        if not bool(torch.isfinite(xyz_ego).all()):
            raise FloatingPointError(f"{agent}: non-finite GT ego-frame xyz")

        gt_agent[valid_idx] = xyz_agent
        gt_ego[valid_idx] = xyz_ego

        _roundtrip_check(
            xyz_agent=xyz_agent,
            uv_aug=uv_i,
            depth=depth_i,
            intr=intr,
            cam_rot=cam_rot,
            cam_trans=cam_trans,
            post_rot=post_rot,
            post_trans=post_trans,
            agent=agent,
            args=args,
        )

    return {
        "uv": uv,
        "u_int": u_int,
        "v_int": v_int,
        "gt_depth": gt_depth,
        "raw_valid": raw_valid,
        "model_valid": model_valid,
        "primary_valid": primary_valid,
        "gt_xyz_agent": gt_agent,
        "gt_xyz_ego": gt_ego,
    }


def _roundtrip_check(
    xyz_agent: torch.Tensor,
    uv_aug: torch.Tensor,
    depth: torch.Tensor,
    intr: torch.Tensor,
    cam_rot: torch.Tensor,
    cam_trans: torch.Tensor,
    post_rot: torch.Tensor,
    post_trans: torch.Tensor,
    agent: str,
    args: argparse.Namespace,
) -> None:
    n = min(int(args.roundtrip_n), int(xyz_agent.shape[0]))
    if n <= 0:
        return

    xyz = xyz_agent[:n]
    uv = uv_aug[:n]
    z = depth[:n]
    uv2, z2, front = project_lidar_to_image(
        xyz_lidar=xyz,
        intrinsic=intr[:n],
        cam2lidar_rot=cam_rot[:n],
        cam2lidar_trans=cam_trans[:n],
        post_rot=post_rot[:n],
        post_trans=post_trans[:n],
    )

    if not bool(front.all()):
        raise AssertionError(f"{agent}: GT depth backprojection produced behind-camera points")

    px_err = torch.linalg.vector_norm(uv2 - uv, dim=-1)
    z_err = (z2 - z).abs()
    max_px = float(px_err.max().item())
    max_z = float(z_err.max().item())
    if max_px > float(args.roundtrip_tol_px) or max_z > float(args.roundtrip_tol_depth):
        raise AssertionError(
            f"{agent}: camera round-trip failed: max_uv={max_px:.6f}px "
            f"max_depth={max_z:.6f}m"
        )


# -----------------------------------------------------------------------------
# Per-Gaussian rows
# -----------------------------------------------------------------------------



def _rows_for_agent(
    init: GaussianSet,
    s1: GaussianSet,
    s2: GaussianSet,
    gt: Mapping[str, torch.Tensor],
    cross_valid: torch.Tensor,
    payload: Mapping[str, Any],
    agent: str,
    meta: Mapping[str, Any],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Build exact per-Gaussian Init -> Stage1 -> Stage2 geometry records."""
    _assert_identity(init, s1, f"{agent}: Init->Stage1")
    _assert_identity(s1, s2, f"{agent}: Stage1->Stage2")

    valid = gt["primary_valid"]
    idx = valid.nonzero(as_tuple=False).squeeze(1)

    counts = {
        "n_gaussians": int(s1.n_gaussians),
        "n_raw_valid": int(gt["raw_valid"].sum().item()),
        "n_model_valid": int(gt["model_valid"].sum().item()),
        "n_primary_valid": int(valid.sum().item()),
        "n_cross_valid_all": int(cross_valid.sum().item()),
        "n_cross_valid_primary": int((cross_valid & valid).sum().item()),
    }
    if idx.numel() == 0:
        return [], counts

    gt_xyz = gt["gt_xyz_ego"][idx]
    init_xyz = init.mean[idx]
    s1_xyz = s1.mean[idx]
    s2_xyz = s2.mean[idx]

    init_err_vec = init_xyz - gt_xyz
    err1_vec = s1_xyz - gt_xyz
    err2_vec = s2_xyz - gt_xyz

    init_abs = init_err_vec.abs()
    abs1 = err1_vec.abs()
    abs2 = err2_vec.abs()

    init_err = torch.linalg.vector_norm(init_err_vec, dim=-1)
    err1 = torch.linalg.vector_norm(err1_vec, dim=-1)
    err2 = torch.linalg.vector_norm(err2_vec, dim=-1)

    # Init -> Stage1
    stage1_improvement = init_err - err1
    stage1_relative = stage1_improvement / init_err.clamp_min(float(args.error_eps))
    stage1_move = s1_xyz - init_xyz
    stage1_move_norm = torch.linalg.vector_norm(stage1_move, dim=-1)
    stage1_ideal = gt_xyz - init_xyz
    stage1_ideal_norm = torch.linalg.vector_norm(stage1_ideal, dim=-1)

    stage1_cos = torch.full_like(init_err, float("nan"))
    stage1_directional = (
        (stage1_move_norm > float(args.movement_eps))
        & (stage1_ideal_norm > float(args.error_eps))
    )
    if bool(stage1_directional.any()):
        stage1_cos[stage1_directional] = (
            (stage1_move[stage1_directional] * stage1_ideal[stage1_directional]).sum(dim=-1)
            / (
                stage1_move_norm[stage1_directional]
                * stage1_ideal_norm[stage1_directional]
            ).clamp_min(float(args.error_eps))
        )

    stage1_projection = torch.full_like(init_err, float("nan"))
    stage1_ideal_ok = stage1_ideal_norm > float(args.error_eps)
    if bool(stage1_ideal_ok.any()):
        stage1_projection[stage1_ideal_ok] = (
            (stage1_move[stage1_ideal_ok] * stage1_ideal[stage1_ideal_ok]).sum(dim=-1)
            / stage1_ideal_norm[stage1_ideal_ok].clamp_min(float(args.error_eps))
        )

    # Stage1 -> Stage2
    improvement = err1 - err2
    relative = improvement / err1.clamp_min(float(args.error_eps))
    move = s2_xyz - s1_xyz
    move_norm = torch.linalg.vector_norm(move, dim=-1)
    ideal = gt_xyz - s1_xyz
    ideal_norm = torch.linalg.vector_norm(ideal, dim=-1)

    denom = move_norm * ideal_norm
    cos = torch.full_like(err1, float("nan"))
    directional = (move_norm > float(args.movement_eps)) & (
        ideal_norm > float(args.error_eps)
    )
    if bool(directional.any()):
        cos[directional] = (
            (move[directional] * ideal[directional]).sum(dim=-1)
            / denom[directional].clamp_min(float(args.error_eps))
        )

    projection = torch.full_like(err1, float("nan"))
    ideal_ok = ideal_norm > float(args.error_eps)
    if bool(ideal_ok.any()):
        projection[ideal_ok] = (
            (move[ideal_ok] * ideal[ideal_ok]).sum(dim=-1)
            / ideal_norm[ideal_ok].clamp_min(float(args.error_eps))
        )

    init_to_s1_gain_xyz = init_abs - abs1
    gain_xyz = abs1 - abs2

    n_views = int(payload[CAMERA_GEOMETRY_KEY].get("n_views", 1))
    p_fg = s1.p_fg[idx] if s1.p_fg is not None else torch.full_like(err1, float("nan"))
    pred_depth = (
        init.depth_mean[idx]
        if init.depth_mean is not None
        else (
            s1.depth_mean[idx]
            if s1.depth_mean is not None
            else torch.full_like(err1, float("nan"))
        )
    )

    def _status(value: float) -> str:
        if value > float(args.improvement_eps):
            return "corrected"
        if value < -float(args.improvement_eps):
            return "worsened"
        return "unchanged"

    rows: List[Dict[str, Any]] = []
    for j, gi in enumerate(idx.detach().cpu().tolist()):
        view_index = int(s1.view_index[gi].item())
        local_cav_idx = view_index // max(n_views, 1)
        cam_idx = view_index % max(n_views, 1)

        imp01 = float(stage1_improvement[j].item())
        imp12 = float(improvement[j].item())
        mv01 = float(stage1_move_norm[j].item())
        mv12 = float(move_norm[j].item())

        stage1_cos_value = float(stage1_cos[j].item())
        stage2_cos_value = float(cos[j].item())

        row: Dict[str, Any] = {
            **meta,
            "agent_type": agent,
            "cav_local_index": int(local_cav_idx),
            "cav_id": _cav_id_for_view(payload, local_cav_idx),
            "view_index": view_index,
            "camera_index": int(cam_idx),
            "u": float(gt["uv"][gi, 0].item()),
            "v": float(gt["uv"][gi, 1].item()),
            "u_int": int(gt["u_int"][gi].item()),
            "v_int": int(gt["v_int"][gi].item()),
            "gt_depth_m": float(gt["gt_depth"][gi].item()),
            "gt_x": float(gt_xyz[j, 0].item()),
            "gt_y": float(gt_xyz[j, 1].item()),
            "gt_z": float(gt_xyz[j, 2].item()),

            "init_x": float(init_xyz[j, 0].item()),
            "init_y": float(init_xyz[j, 1].item()),
            "init_z": float(init_xyz[j, 2].item()),
            "init_dx_signed": float(init_err_vec[j, 0].item()),
            "init_dy_signed": float(init_err_vec[j, 1].item()),
            "init_dz_signed": float(init_err_vec[j, 2].item()),
            "init_abs_x": float(init_abs[j, 0].item()),
            "init_abs_y": float(init_abs[j, 1].item()),
            "init_abs_z": float(init_abs[j, 2].item()),
            "init_error_3d": float(init_err[j].item()),

            "stage1_x": float(s1_xyz[j, 0].item()),
            "stage1_y": float(s1_xyz[j, 1].item()),
            "stage1_z": float(s1_xyz[j, 2].item()),
            "stage1_dx_signed": float(err1_vec[j, 0].item()),
            "stage1_dy_signed": float(err1_vec[j, 1].item()),
            "stage1_dz_signed": float(err1_vec[j, 2].item()),
            "stage1_abs_x": float(abs1[j, 0].item()),
            "stage1_abs_y": float(abs1[j, 1].item()),
            "stage1_abs_z": float(abs1[j, 2].item()),
            "stage1_error_3d": float(err1[j].item()),

            "init_to_stage1_gain_x": float(init_to_s1_gain_xyz[j, 0].item()),
            "init_to_stage1_gain_y": float(init_to_s1_gain_xyz[j, 1].item()),
            "init_to_stage1_gain_z": float(init_to_s1_gain_xyz[j, 2].item()),
            "init_to_stage1_improvement_m": imp01,
            "init_to_stage1_relative_improvement": float(stage1_relative[j].item()),
            "stage1_move_x": float(stage1_move[j, 0].item()),
            "stage1_move_y": float(stage1_move[j, 1].item()),
            "stage1_move_z": float(stage1_move[j, 2].item()),
            "stage1_move_norm": mv01,
            "stage1_cos_to_gt": stage1_cos_value,
            "stage1_projection_toward_gt": float(stage1_projection[j].item()),
            "stage1_correction_status": _status(imp01),
            "stage1_move_direction": (
                "unchanged"
                if mv01 <= float(args.movement_eps)
                else (
                    "toward_gt"
                    if math.isfinite(stage1_cos_value) and stage1_cos_value > 0.0
                    else "away_or_orthogonal"
                )
            ),

            "stage2_x": float(s2_xyz[j, 0].item()),
            "stage2_y": float(s2_xyz[j, 1].item()),
            "stage2_z": float(s2_xyz[j, 2].item()),
            "stage2_dx_signed": float(err2_vec[j, 0].item()),
            "stage2_dy_signed": float(err2_vec[j, 1].item()),
            "stage2_dz_signed": float(err2_vec[j, 2].item()),
            "stage2_abs_x": float(abs2[j, 0].item()),
            "stage2_abs_y": float(abs2[j, 1].item()),
            "stage2_abs_z": float(abs2[j, 2].item()),
            "stage2_error_3d": float(err2[j].item()),

            # Backward-compatible Stage1->Stage2 fields
            "gain_x": float(gain_xyz[j, 0].item()),
            "gain_y": float(gain_xyz[j, 1].item()),
            "gain_z": float(gain_xyz[j, 2].item()),
            "improvement_m": imp12,
            "relative_improvement": float(relative[j].item()),
            "stage2_move_x": float(move[j, 0].item()),
            "stage2_move_y": float(move[j, 1].item()),
            "stage2_move_z": float(move[j, 2].item()),
            "stage2_move_norm": mv12,
            "cos_to_gt": stage2_cos_value,
            "projection_toward_gt": float(projection[j].item()),
            "stage2_cross_valid": bool(cross_valid[gi].item()),
            "correction_status": _status(imp12),
            "move_direction": (
                "unchanged"
                if mv12 <= float(args.movement_eps)
                else (
                    "toward_gt"
                    if math.isfinite(stage2_cos_value) and stage2_cos_value > 0.0
                    else "away_or_orthogonal"
                )
            ),
            "p_fg": float(p_fg[j].item()),
            "predicted_initial_depth_m": float(pred_depth[j].item()),
        }
        rows.append(row)

    return rows, counts


def _finite_array(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    return vals[np.isfinite(vals)]



def _summary_for_rows(
    rows: Sequence[Mapping[str, Any]],
    improvement_eps: float,
    movement_eps: float,
) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}

    e0 = _finite_array(rows, "init_error_3d")
    e1 = _finite_array(rows, "stage1_error_3d")
    e2 = _finite_array(rows, "stage2_error_3d")

    def err_pack(x: np.ndarray) -> Dict[str, float]:
        return {
            "mean_m": float(np.mean(x)),
            "median_m": float(np.median(x)),
            "rmse_m": float(np.sqrt(np.mean(x * x))),
            "p75_m": float(np.quantile(x, 0.75)),
            "p90_m": float(np.quantile(x, 0.90)),
            "p95_m": float(np.quantile(x, 0.95)),
        }

    def stage_pack(prefix: str) -> Dict[str, Any]:
        if prefix == "stage1":
            imp_key = "init_to_stage1_improvement_m"
            rel_key = "init_to_stage1_relative_improvement"
            status_key = "stage1_correction_status"
            move_key = "stage1_move_norm"
            cos_key = "stage1_cos_to_gt"
            before, after = e0, e1
        else:
            imp_key = "improvement_m"
            rel_key = "relative_improvement"
            status_key = "correction_status"
            move_key = "stage2_move_norm"
            cos_key = "cos_to_gt"
            before, after = e1, e2

        imp = _finite_array(rows, imp_key)
        rel = _finite_array(rows, rel_key)
        move = _finite_array(rows, move_key)
        cos = _finite_array(rows, cos_key)

        statuses = [str(r[status_key]) for r in rows]
        nonzero_rows = [
            r
            for r in rows
            if float(r[move_key]) > movement_eps
            and math.isfinite(float(r[cos_key]))
        ]
        toward = [r for r in nonzero_rows if float(r[cos_key]) > 0.0]

        return {
            "improvement": {
                "mean_m": float(np.mean(imp)),
                "median_m": float(np.median(imp)),
                "ratio_of_means": float(
                    1.0 - np.mean(after) / max(np.mean(before), 1.0e-12)
                ),
                "mean_per_gaussian_relative": (
                    float(np.mean(rel)) if rel.size else float("nan")
                ),
            },
            "status": {
                "corrected_rate": float(
                    sum(s == "corrected" for s in statuses) / len(rows)
                ),
                "worsened_rate": float(
                    sum(s == "worsened" for s in statuses) / len(rows)
                ),
                "unchanged_rate": float(
                    sum(s == "unchanged" for s in statuses) / len(rows)
                ),
                "improvement_eps_m": float(improvement_eps),
            },
            "movement": {
                "mean_m": float(np.mean(move)) if move.size else float("nan"),
                "median_m": float(np.median(move)) if move.size else float("nan"),
                "nonzero_count": len(nonzero_rows),
                "toward_gt_rate_among_nonzero": (
                    float(len(toward) / len(nonzero_rows))
                    if nonzero_rows
                    else float("nan")
                ),
                "mean_cos_to_gt": float(np.mean(cos)) if cos.size else float("nan"),
                "median_cos_to_gt": (
                    float(np.median(cos)) if cos.size else float("nan")
                ),
                "movement_eps_m": float(movement_eps),
            },
        }

    stage01 = stage_pack("stage1")
    stage12 = stage_pack("stage2")

    return {
        "n": len(rows),
        "init_error_3d": err_pack(e0),
        "stage1_error_3d": err_pack(e1),
        "stage2_error_3d": err_pack(e2),
        "axis_mae_m": {
            "x": {
                "init": float(np.mean(_finite_array(rows, "init_abs_x"))),
                "stage1": float(np.mean(_finite_array(rows, "stage1_abs_x"))),
                "stage2": float(np.mean(_finite_array(rows, "stage2_abs_x"))),
            },
            "y": {
                "init": float(np.mean(_finite_array(rows, "init_abs_y"))),
                "stage1": float(np.mean(_finite_array(rows, "stage1_abs_y"))),
                "stage2": float(np.mean(_finite_array(rows, "stage2_abs_y"))),
            },
            "z": {
                "init": float(np.mean(_finite_array(rows, "init_abs_z"))),
                "stage1": float(np.mean(_finite_array(rows, "stage1_abs_z"))),
                "stage2": float(np.mean(_finite_array(rows, "stage2_abs_z"))),
            },
        },
        "init_to_stage1": stage01,
        "stage1_to_stage2": stage12,
        # old keys retained for compatibility
        "improvement": stage12["improvement"],
        "status": stage12["status"],
        "movement": stage12["movement"],
    }


def _make_group_summaries(
    rows: Sequence[Mapping[str, Any]], args: argparse.Namespace
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for agent in ("global",) + AGENT_ORDER:
        base = list(rows) if agent == "global" else [r for r in rows if r["agent_type"] == agent]
        out[agent] = {
            "all_valid": _summary_for_rows(
                base, args.improvement_eps, args.movement_eps
            ),
            "cross_valid": _summary_for_rows(
                [r for r in base if bool(r["stage2_cross_valid"])],
                args.improvement_eps,
                args.movement_eps,
            ),
            "cross_invalid": _summary_for_rows(
                [r for r in base if not bool(r["stage2_cross_valid"])],
                args.improvement_eps,
                args.movement_eps,
            ),
        }
        n = len(base)
        out[agent]["stage2_cross_coverage_on_valid_gt"] = (
            float(sum(bool(r["stage2_cross_valid"]) for r in base) / n)
            if n
            else float("nan")
        )
    return out


def _parse_bins(text: str) -> List[float]:
    vals: List[float] = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        vals.append(float("inf") if item in ("inf", "+inf", "infinity") else float(item))
    if len(vals) < 2:
        raise ValueError(f"Need >=2 bin edges, got {vals}")
    if any(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
        raise ValueError(f"Bin edges must be strictly increasing: {vals}")
    return vals


def _bin_rows(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    edges: Sequence[float],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    groups = ("global",) + AGENT_ORDER
    for group in groups:
        group_rows = list(rows) if group == "global" else [r for r in rows if r["agent_type"] == group]
        for lo, hi in zip(edges[:-1], edges[1:]):
            picked = [
                r
                for r in group_rows
                if float(r[key]) >= lo and float(r[key]) < hi
            ]
            s = _summary_for_rows(picked, args.improvement_eps, args.movement_eps)
            out.append(
                {
                    "group": group,
                    "metric": key,
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "n": int(s.get("n", 0)),
                    "stage1_mean_error_m": (
                        s.get("stage1_error_3d", {}).get("mean_m", float("nan"))
                    ),
                    "stage2_mean_error_m": (
                        s.get("stage2_error_3d", {}).get("mean_m", float("nan"))
                    ),
                    "mean_improvement_m": (
                        s.get("improvement", {}).get("mean_m", float("nan"))
                    ),
                    "corrected_rate": (
                        s.get("status", {}).get("corrected_rate", float("nan"))
                    ),
                    "cross_valid_rate": (
                        float(sum(bool(r["stage2_cross_valid"]) for r in picked) / len(picked))
                        if picked
                        else float("nan")
                    ),
                }
            )
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fieldnames.append(str(k))
                seen.add(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))



def _by_agent_csv(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group, entry in summary.items():
        for subset in ("all_valid", "cross_valid", "cross_invalid"):
            s = entry[subset]
            row: Dict[str, Any] = {
                "group": group,
                "subset": subset,
                "n": s.get("n", 0),
                "cross_coverage_on_valid_gt": entry.get(
                    "stage2_cross_coverage_on_valid_gt", float("nan")
                ),
            }
            if s.get("n", 0):
                s01 = s["init_to_stage1"]
                s12 = s["stage1_to_stage2"]
                row.update(
                    {
                        "init_mean_error_m": s["init_error_3d"]["mean_m"],
                        "stage1_mean_error_m": s["stage1_error_3d"]["mean_m"],
                        "stage2_mean_error_m": s["stage2_error_3d"]["mean_m"],
                        "init_median_error_m": s["init_error_3d"]["median_m"],
                        "stage1_median_error_m": s["stage1_error_3d"]["median_m"],
                        "stage2_median_error_m": s["stage2_error_3d"]["median_m"],

                        "init_to_stage1_mean_improvement_m": s01["improvement"]["mean_m"],
                        "init_to_stage1_median_improvement_m": s01["improvement"]["median_m"],
                        "init_to_stage1_improvement_ratio_of_means": s01["improvement"]["ratio_of_means"],
                        "stage1_corrected_rate": s01["status"]["corrected_rate"],
                        "stage1_worsened_rate": s01["status"]["worsened_rate"],
                        "stage1_unchanged_rate": s01["status"]["unchanged_rate"],
                        "stage1_mean_move_m": s01["movement"]["mean_m"],
                        "stage1_toward_gt_rate_among_nonzero": s01["movement"][
                            "toward_gt_rate_among_nonzero"
                        ],
                        "stage1_mean_cos_to_gt": s01["movement"]["mean_cos_to_gt"],

                        "mean_improvement_m": s12["improvement"]["mean_m"],
                        "median_improvement_m": s12["improvement"]["median_m"],
                        "improvement_ratio_of_means": s12["improvement"]["ratio_of_means"],
                        "mean_per_gaussian_relative_improvement": s12["improvement"][
                            "mean_per_gaussian_relative"
                        ],
                        "corrected_rate": s12["status"]["corrected_rate"],
                        "worsened_rate": s12["status"]["worsened_rate"],
                        "unchanged_rate": s12["status"]["unchanged_rate"],
                        "mean_move_m": s12["movement"]["mean_m"],
                        "toward_gt_rate_among_nonzero": s12["movement"][
                            "toward_gt_rate_among_nonzero"
                        ],
                        "mean_cos_to_gt": s12["movement"]["mean_cos_to_gt"],

                        "init_mae_x_m": s["axis_mae_m"]["x"]["init"],
                        "stage1_mae_x_m": s["axis_mae_m"]["x"]["stage1"],
                        "stage2_mae_x_m": s["axis_mae_m"]["x"]["stage2"],
                        "init_mae_y_m": s["axis_mae_m"]["y"]["init"],
                        "stage1_mae_y_m": s["axis_mae_m"]["y"]["stage1"],
                        "stage2_mae_y_m": s["axis_mae_m"]["y"]["stage2"],
                        "init_mae_z_m": s["axis_mae_m"]["z"]["init"],
                        "stage1_mae_z_m": s["axis_mae_m"]["z"]["stage1"],
                        "stage2_mae_z_m": s["axis_mae_m"]["z"]["stage2"],
                    }
                )
            rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------



def _plot_results(
    rows: Sequence[Mapping[str, Any]],
    depth_bin_rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    seed: int,
    max_points: int,
) -> None:
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plots] skipped: matplotlib unavailable: {exc}")
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    if len(rows) > max_points > 0:
        ids = rng.choice(len(rows), size=max_points, replace=False)
        p_rows = [rows[int(i)] for i in ids]
    else:
        p_rows = list(rows)

    e0 = np.asarray([r["init_error_3d"] for r in p_rows], dtype=float)
    e1 = np.asarray([r["stage1_error_3d"] for r in p_rows], dtype=float)
    e2 = np.asarray([r["stage2_error_3d"] for r in p_rows], dtype=float)
    imp01 = np.asarray(
        [r["init_to_stage1_improvement_m"] for r in p_rows], dtype=float
    )
    imp12 = np.asarray([r["improvement_m"] for r in p_rows], dtype=float)

    for xerr, yerr, xlabel, ylabel, title, filename in (
        (
            e0, e1,
            "Initializer 3D error (m)",
            "Stage-1 3D error (m)",
            "Initializer vs Stage-1 Gaussian center error",
            "init_vs_stage1_scatter.png",
        ),
        (
            e1, e2,
            "Stage-1 3D error (m)",
            "Stage-2 3D error (m)",
            "Stage-1 vs Stage-2 Gaussian center error",
            "stage1_vs_stage2_scatter.png",
        ),
    ):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(xerr, yerr, s=4, alpha=0.25)
        vmax = float(np.nanpercentile(np.concatenate([xerr, yerr]), 99.5))
        vmax = max(vmax, 1.0)
        ax.plot([0, vmax], [0, vmax], linestyle="--")
        ax.set_xlim(0, vmax)
        ax.set_ylim(0, vmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for xvals, label in (
        (e0, "Initializer"),
        (e1, "Stage-1"),
        (e2, "Stage-2"),
    ):
        xvals = np.sort(xvals[np.isfinite(xvals)])
        y = np.arange(1, len(xvals) + 1) / max(len(xvals), 1)
        ax.plot(xvals, y, label=label)
    ax.set_xlabel("3D center error (m)")
    ax.set_ylabel("CDF")
    ax.legend()
    ax.set_title("Gaussian 3D localization error CDF")
    fig.tight_layout()
    fig.savefig(plot_dir / "error_cdf.png", dpi=180)
    plt.close(fig)

    for values, title, xlabel, filename in (
        (
            imp01,
            "Stage-1 geometry correction",
            "Initializer error - Stage1 error (m)",
            "init_to_stage1_improvement_hist.png",
        ),
        (
            imp12,
            "Stage-2 geometry correction",
            "Stage1 error - Stage2 error (m)",
            "improvement_hist.png",
        ),
    ):
        fig, ax = plt.subplots(figsize=(7, 5))
        finite = values[np.isfinite(values)]
        if finite.size:
            lo, hi = np.quantile(finite, [0.01, 0.99])
            clipped = finite[(finite >= lo) & (finite <= hi)]
            ax.hist(clipped, bins=80)
        ax.axvline(0.0, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)

    agents = [a for a in AGENT_ORDER if any(r["agent_type"] == a for r in rows)]
    if agents:
        x = np.arange(len(agents))
        w = 0.25
        m0 = [
            np.mean([r["init_error_3d"] for r in rows if r["agent_type"] == a])
            for a in agents
        ]
        m1 = [
            np.mean([r["stage1_error_3d"] for r in rows if r["agent_type"] == a])
            for a in agents
        ]
        m2 = [
            np.mean([r["stage2_error_3d"] for r in rows if r["agent_type"] == a])
            for a in agents
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - w, m0, width=w, label="Initializer")
        ax.bar(x, m1, width=w, label="Stage-1")
        ax.bar(x + w, m2, width=w, label="Stage-2")
        ax.set_xticks(x)
        ax.set_xticklabels(agents)
        ax.set_ylabel("Mean 3D error (m)")
        ax.set_title("Geometry error progression by agent")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "by_agent_mean_error.png", dpi=180)
        plt.close(fig)

    axes = ("x", "y", "z")
    x = np.arange(3)
    w = 0.25
    m0 = [np.mean([r[f"init_abs_{a}"] for r in rows]) for a in axes]
    m1 = [np.mean([r[f"stage1_abs_{a}"] for r in rows]) for a in axes]
    m2 = [np.mean([r[f"stage2_abs_{a}"] for r in rows]) for a in axes]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w, m0, width=w, label="Initializer")
    ax.bar(x, m1, width=w, label="Stage-1")
    ax.bar(x + w, m2, width=w, label="Stage-2")
    ax.set_xticks(x)
    ax.set_xticklabels(["X", "Y", "Z"])
    ax.set_ylabel("MAE (m)")
    ax.set_title("Axis-wise Gaussian center error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "axis_mae.png", dpi=180)
    plt.close(fig)

    global_bins = [
        r for r in depth_bin_rows
        if r["group"] == "global" and int(r["n"]) > 0
    ]
    if global_bins:
        labels = [
            (
                f"{r['bin_lo']:g}-{r['bin_hi']:g}"
                if math.isfinite(float(r["bin_hi"]))
                else f">={r['bin_lo']:g}"
            )
            for r in global_bins
        ]
        vals = [float(r["mean_improvement_m"]) for r in global_bins]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(np.arange(len(vals)), vals, marker="o")
        ax.axhline(0.0, linestyle="--")
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Mean Stage-2 improvement (m)")
        ax.set_xlabel("GT optical depth bin (m)")
        ax.set_title("Stage-2 correction vs depth")
        fig.tight_layout()
        fig.savefig(plot_dir / "improvement_vs_depth.png", dpi=180)
        plt.close(fig)

    status_names = ("corrected", "worsened", "unchanged")
    stage1_rates = [
        sum(r["stage1_correction_status"] == s for r in rows) / len(rows)
        for s in status_names
    ]
    stage2_rates = [
        sum(r["correction_status"] == s for r in rows) / len(rows)
        for s in status_names
    ]
    x = np.arange(len(status_names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, stage1_rates, width=w, label="Init→Stage1")
    ax.bar(x + w / 2, stage2_rates, width=w, label="Stage1→Stage2")
    ax.set_xticks(x)
    ax.set_xticklabels(status_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction")
    ax.set_title("Geometry correction outcome by stage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "correction_status.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    yaml_path = _resolve_yaml(args)
    hypes = _load_hypes(yaml_path, args)
    checkpoint = _resolve_checkpoint(args)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (
            Path(args.model_dir) / "stage2_geometry_audit"
            if args.model_dir
            else checkpoint.parent / "stage2_geometry_audit"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, _ds_hypes, train_flag = _make_dataset(hypes, args.split)
    indices = _select_indices(len(dataset), args.sample_n, args.seed, args.indices)
    print(
        f"[dataset] split={args.split} n_total={len(dataset)} "
        f"n_selected={len(indices)} train_flag={train_flag}"
    )
    print(f"[dataset] first indices={indices[:20]}")

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model = train_utils.create_model(hypes).to(device)
    checkpoint_report = _load_weights(
        model, checkpoint, device, strict_all=bool(args.strict_checkpoint)
    )
    model.eval()

    all_rows: List[Dict[str, Any]] = []
    sample_reports: List[Dict[str, Any]] = []
    aggregate_counts: Dict[str, Dict[str, int]] = {
        agent: defaultdict(int) for agent in AGENT_ORDER
    }
    interaction_counts: Dict[str, Dict[str, int]] = {
        itype: defaultdict(int) for itype in INTERACTION_TYPES
    }

    with torch.no_grad():
        for pos, dataset_index in enumerate(indices, start=1):
            item = dataset[dataset_index]
            if item is None:
                print(f"[sample {dataset_index}] dataset returned None; skipped")
                continue

            if train_flag:
                batch = dataset.collate_batch_train([item])
            else:
                batch = dataset.collate_batch_test([item])
            if batch is None:
                print(f"[sample {dataset_index}] collate returned None; skipped")
                continue

            batch = train_utils.to_device(batch, device)
            batch["ego"]["epoch"] = 0
            meta = _metadata(batch["ego"], dataset_index)

            (
                init_sets,
                stage1_sets,
                stage2_sets,
                _sources,
                cross_valid_masks,
                stage2_stats,
            ) = _replay_to_stage2(model, batch)

            sample_report: Dict[str, Any] = {
                **meta,
                "agents": {},
                "stage2_stats": stage2_stats,
            }

            for itype in INTERACTION_TYPES:
                st = stage2_stats.get(itype, {})
                if isinstance(st, Mapping):
                    for key in ("n_query", "n_valid", "n_invalid"):
                        interaction_counts[itype][key] += int(st.get(key, 0))

            for agent in AGENT_ORDER:
                if agent not in stage1_sets:
                    continue
                init = init_sets[agent]
                s1 = stage1_sets[agent]
                s2 = stage2_sets[agent]
                payload = batch["ego"][agent]
                cross_valid = cross_valid_masks[agent]

                gt = _gt_geometry_for_gaussians(
                    s1=s1,
                    payload=payload,
                    hypes=hypes,
                    agent=agent,
                    args=args,
                )

                rows, counts = _rows_for_agent(
                    init=init,
                    s1=s1,
                    s2=s2,
                    gt=gt,
                    cross_valid=cross_valid,
                    payload=payload,
                    agent=agent,
                    meta=meta,
                    args=args,
                )
                all_rows.extend(rows)
                for key, value in counts.items():
                    aggregate_counts[agent][key] += int(value)

                agent_summary = _summary_for_rows(
                    rows, args.improvement_eps, args.movement_eps
                )
                sample_report["agents"][agent] = {
                    **counts,
                    "metrics": agent_summary,
                }

                if pos <= 3:
                    init_mean = (
                        float(np.mean([r["init_error_3d"] for r in rows]))
                        if rows
                        else float("nan")
                    )
                    s1_mean = (
                        float(np.mean([r["stage1_error_3d"] for r in rows]))
                        if rows
                        else float("nan")
                    )
                    s2_mean = (
                        float(np.mean([r["stage2_error_3d"] for r in rows]))
                        if rows
                        else float("nan")
                    )
                    print(
                        f"[sample {dataset_index}] {agent}: "
                        f"N={s1.n_gaussians} valid={counts['n_primary_valid']} "
                        f"cross_valid={counts['n_cross_valid_primary']} "
                        f"Init={init_mean:.4f}m Stage1={s1_mean:.4f}m "
                        f"Stage2={s2_mean:.4f}m identity=OK frame={s1.frame}"
                    )

            sample_reports.append(sample_report)

            if pos == 1 or pos % 10 == 0 or pos == len(indices):
                print(
                    f"[progress] {pos}/{len(indices)} samples, "
                    f"valid Gaussian rows={len(all_rows)}"
                )

    if not all_rows:
        raise RuntimeError(
            "No valid GT-linked Gaussians were produced. "
            "Check depth channel, GT mask, checkpoint, and dataset split."
        )

    group_summary = _make_group_summaries(all_rows, args)

    depth_edges = _parse_bins(args.depth_bins)
    error_edges = _parse_bins(args.initial_error_bins)
    depth_bin_rows = _bin_rows(
        all_rows, "gt_depth_m", depth_edges, args
    )
    initializer_error_bin_rows = _bin_rows(
        all_rows, "init_error_3d", error_edges, args
    )
    initial_error_bin_rows = _bin_rows(
        all_rows, "stage1_error_3d", error_edges, args
    )

    by_agent_rows = _by_agent_csv(group_summary)

    best = sorted(all_rows, key=lambda r: float(r["improvement_m"]), reverse=True)[:50]
    worst = sorted(all_rows, key=lambda r: float(r["improvement_m"]))[:50]

    interaction_summary: Dict[str, Any] = {}
    for itype, cnt in interaction_counts.items():
        q = int(cnt.get("n_query", 0))
        v = int(cnt.get("n_valid", 0))
        interaction_summary[itype] = {
            "n_query": q,
            "n_valid": v,
            "n_invalid": int(cnt.get("n_invalid", q - v)),
            "coverage": float(v / q) if q else float("nan"),
        }

    summary: Dict[str, Any] = {
        "script": "analyze_stage2_geometry_correction.py",
        "yaml": str(yaml_path),
        "checkpoint": checkpoint_report,
        "split": args.split,
        "seed": int(args.seed),
        "selected_indices": indices,
        "n_selected_samples": len(indices),
        "n_processed_samples": len(sample_reports),
        "gt_contract": {
            "source": 'batch["ego"][agent]["batch_merged_cam_inputs"]["imgs"] channel 3',
            "meaning": "augmentation-aligned GT camera optical-axis z in meters",
            "pixel_sampling": "nearest integer pixel from preserved Gaussian uv",
            "backprojection": "production lift_optical_z_to_lidar",
            "final_frame": "ego",
            "primary_mask": args.gt_mask,
            "raw_depth_min_m": args.raw_depth_min,
            "raw_depth_max_m": args.raw_depth_max,
        },
        "identity_contract": {
            "matching": "exact same Gaussian index; no nearest-neighbor matching",
            "assertions": [
                "same n_gaussians across Initializer/Stage1/Stage2",
                "same view_index across Initializer/Stage1/Stage2",
                "same batch_index across Initializer/Stage1/Stage2",
                "same uv across Initializer/Stage1/Stage2",
                "Initializer.frame == Stage1.frame == Stage2.frame == ego",
                "Stage2 cross-invalid points are exact mean identity",
            ],
        },
        "thresholds": {
            "improvement_eps_m": args.improvement_eps,
            "movement_eps_m": args.movement_eps,
        },
        "aggregate_counts": {
            agent: dict(values) for agent, values in aggregate_counts.items()
        },
        "stage2_interaction_counts": interaction_summary,
        "metrics": group_summary,
        "sample_reports": sample_reports,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True, default=str)
    )
    _write_csv(out_dir / "per_gaussian.csv", all_rows)
    _write_csv(out_dir / "by_agent.csv", by_agent_rows)
    _write_csv(out_dir / "by_depth_bin.csv", depth_bin_rows)
    _write_csv(out_dir / "by_initializer_error_bin.csv", initializer_error_bin_rows)
    _write_csv(out_dir / "by_initial_error_bin.csv", initial_error_bin_rows)
    _write_csv(out_dir / "examples_best_corrected.csv", best)
    _write_csv(out_dir / "examples_most_worsened.csv", worst)

    if not args.no_plots:
        _plot_results(
            all_rows,
            depth_bin_rows,
            out_dir,
            seed=args.seed,
            max_points=args.plot_max_points,
        )

    print("\n=== PRIMARY SUMMARY ===")
    for group in ("global",) + AGENT_ORDER:
        s = group_summary[group]["all_valid"]
        cv = group_summary[group]["cross_valid"]
        if int(s.get("n", 0)) == 0:
            continue
        print(
            f"{group:8s} ALL n={s['n']:7d} "
            f"Init={s['init_error_3d']['mean_m']:.4f}m "
            f"Stage1={s['stage1_error_3d']['mean_m']:.4f}m "
            f"Stage2={s['stage2_error_3d']['mean_m']:.4f}m "
            f"gain01={s['init_to_stage1']['improvement']['mean_m']:.4f}m "
            f"gain12={s['stage1_to_stage2']['improvement']['mean_m']:.4f}m "
            f"S1corr={100*s['init_to_stage1']['status']['corrected_rate']:.2f}% "
            f"S2corr={100*s['stage1_to_stage2']['status']['corrected_rate']:.2f}% "
            f"coverage={100*group_summary[group]['stage2_cross_coverage_on_valid_gt']:.2f}%"
        )
        if int(cv.get("n", 0)) > 0:
            print(
                f"{'':8s} XVALID n={cv['n']:7d} "
                f"Init={cv['init_error_3d']['mean_m']:.4f}m "
                f"Stage1={cv['stage1_error_3d']['mean_m']:.4f}m "
                f"Stage2={cv['stage2_error_3d']['mean_m']:.4f}m "
                f"gain01={cv['init_to_stage1']['improvement']['mean_m']:.4f}m "
                f"gain12={cv['stage1_to_stage2']['improvement']['mean_m']:.4f}m "
                f"S1toward={100*cv['init_to_stage1']['movement']['toward_gt_rate_among_nonzero']:.2f}% "
                f"S2toward={100*cv['stage1_to_stage2']['movement']['toward_gt_rate_among_nonzero']:.2f}%"
            )

    print(f"\nWrote: {out_dir / 'summary.json'}")
    print(f"Wrote: {out_dir / 'per_gaussian.csv'}")
    print("Metric contract verified: SAME Gaussian vs SAME source-pixel GT 3D point at Initializer, Stage1, and Stage2.")


if __name__ == "__main__":
    main()

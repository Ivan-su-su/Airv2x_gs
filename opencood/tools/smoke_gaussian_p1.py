#!/usr/bin/env python3
"""Joint R90 P1 smoke: leakage, one F90, zero-init, grads, sanity, optional real batch."""

from __future__ import annotations

import argparse
import py_compile
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import types

_cuda_stub = types.ModuleType("opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_cuda")


def _stub_points_in_boxes_cpu(boxes, points, point_indices):
    point_indices.zero_()


_cuda_stub.points_in_boxes_cpu = _stub_points_in_boxes_cpu
sys.modules["opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_cuda"] = _cuda_stub
import opencood.pcdet_utils.roiaware_pool3d as _roiaware_pkg

_roiaware_pkg.roiaware_pool3d_cuda = _cuda_stub

from opencood.hypes_yaml import yaml_utils
from opencood.models.gaussian_modules_0822.p1_layout import FEAT_H, FEAT_W, F90_CHANNELS, NUM_CLASSES
from opencood.tools.train_gaussian_p1 import (
    _forward_loss_metrics,
    _unwrap_model,
    build_depth_targets,
    build_heatmap_targets,
    build_p1_criteria,
    print_trainable_families,
    setup_p1_optimizer,
)
from opencood.tools import train_utils
from opencood.utils.camera_utils import bin_depths, depth_discretization

YAML_NAME = "airv2x_gaussian_p1_joint.yaml"
PY_FILES = [
    ROOT / "opencood/models/airv2x_gaussian_0822.py",
    ROOT / "opencood/models/gaussian_modules_0822/highres_adapter.py",
    ROOT / "opencood/models/gaussian_modules_0822/p1_layout.py",
    ROOT / "opencood/models/gaussian_modules_0822/image_frontend.py",
    ROOT / "opencood/models/gaussian_modules_0822/heatmap/head.py",
    ROOT / "opencood/models/gaussian_modules_0822/heatmap/target.py",
    ROOT / "opencood/models/gaussian_modules_0822/heatmap/metrics.py",
    ROOT / "opencood/models/gaussian_modules_0822/lss/head.py",
    ROOT / "opencood/models/gaussian_modules_0822/lss/target.py",
    ROOT / "opencood/models/gaussian_modules_0822/lss/categorical_depth.py",
    ROOT / "opencood/loss/gaussian_p1_semantic_loss.py",
    ROOT / "opencood/loss/gaussian_p1_depth_loss.py",
    ROOT / "opencood/tools/train_gaussian_p1.py",
    ROOT / "opencood/utils/airv2x_utils.py",
    ROOT / "opencood/data_utils/datasets/airv2x/intermediate_fusion_dataset.py",
]


def _yaml() -> Path:
    return ROOT / "opencood/hypes_yaml/airv2x/camera/gaussian_p1" / YAML_NAME


def compile_modified() -> None:
    """Fail fast on syntax errors."""
    for path in PY_FILES:
        py_compile.compile(str(path), doraise=True)
        print(f"[compile] {path.relative_to(ROOT)}")


def fake_ego(device: torch.device, n_views: int = 1) -> Dict[str, Any]:
    """Minimal ego dict with vehicle, RSU, and drone cameras."""
    ego: Dict[str, Any] = {}
    specs = {
        "vehicle": (50.0, 5.0),
        "rsu": (40.0, 8.0),
        "drone": (80.0, 40.0),
    }
    for agent_type, (scale, offset) in specs.items():
        imgs = torch.rand(1, n_views, 4, 360, 640, device=device)
        imgs[:, :, 3] = imgs[:, :, 3] * scale + offset
        semantic = torch.zeros(1, n_views, 360, 640, dtype=torch.long, device=device)
        semantic[:, :, 40:80, 80:160] = 3
        cam: Dict[str, Any] = {
            "imgs": imgs,
            "image_semantic_gts": semantic,
        }
        if agent_type == "drone":
            cam["camera_world_z"] = torch.full(
                (1, n_views), 90.0, device=device, dtype=torch.float32
            )
        ego[agent_type] = {"batch_merged_cam_inputs": cam}
    return ego


def load_model(
    device: torch.device,
) -> Tuple[torch.nn.Module, Any, Any]:
    """Build joint model and both criteria from the final yaml."""
    hypes = yaml_utils.load_yaml(str(_yaml()), None)
    model = train_utils.create_model(hypes)
    model.to(device)
    semantic_criterion, depth_criterion = build_p1_criteria(hypes)
    return model, semantic_criterion, depth_criterion


def leakage_test(model: torch.nn.Module, device: torch.device) -> None:
    """Shuffle GT depth/semantic; predictions and features must not change."""
    model.eval()
    core = _unwrap_model(model)
    imgs = torch.rand(1, 1, 4, 360, 640, device=device)
    semantic = torch.zeros(1, 1, 360, 640, dtype=torch.long, device=device)
    semantic[:, :, 10:40, 20:80] = 2
    ego_a = {
        "vehicle": {
            "batch_merged_cam_inputs": {
                "imgs": imgs,
                "image_semantic_gts": semantic,
            }
        }
    }
    imgs_b = imgs.clone()
    imgs_b[:, :, 3] = torch.rand_like(imgs_b[:, :, 3]) * 80.0
    semantic_b = semantic.clone()
    semantic_b[:, :, 100:200, 200:400] = 5
    ego_b = {
        "vehicle": {
            "batch_merged_cam_inputs": {
                "imgs": imgs_b,
                "image_semantic_gts": semantic_b,
            }
        }
    }
    with torch.no_grad():
        r2_a, f45_a = core.frontend.extract_backbone_features("vehicle", imgs)
        r2_b, f45_b = core.frontend.extract_backbone_features("vehicle", imgs_b)
        if not torch.equal(r2_a, r2_b) or not torch.equal(f45_a, f45_b):
            raise AssertionError("depth channel leaked into R2/F45")
        f90_a = core.highres["vehicle"](r2_a, f45_a)
        f90_b = core.highres["vehicle"](r2_b, f45_b)
        if not torch.equal(f90_a, f90_b):
            raise AssertionError("depth channel leaked into F90")
        pred_a = model(ego_a)
        pred_b = model(ego_b)
        for key in ("heatmap_logits",):
            if not torch.equal(pred_a["vehicle"][key], pred_b["vehicle"][key]):
                raise AssertionError(f"GT leaked into {key}")
        if not torch.equal(
            pred_a["vehicle"]["depth_logits"], pred_b["vehicle"]["depth_logits"]
        ):
            raise AssertionError("GT leaked into depth_logits")
    print("[leakage] RGB-only features and predictions OK")


def one_f90_test(model: torch.nn.Module, device: torch.device) -> None:
    """HighResFusion runs once per present agent; both heads share that F90."""
    core = _unwrap_model(model)
    calls = {"n": 0}

    def _count(_module: torch.nn.Module, _inputs: Any, _output: torch.Tensor) -> None:
        calls["n"] += 1

    handle = core.highres["vehicle"].register_forward_hook(_count)
    model.eval()
    with torch.no_grad():
        model(fake_ego(device))
    handle.remove()
    if calls["n"] != 1:
        raise AssertionError(f"HighResFusion ran {calls['n']} times, expected 1")
    print("[f90] HighResFusion executed once per agent")


def numerical_check(pred: Dict[str, Dict[str, torch.Tensor]]) -> None:
    """Finite logits/moments and official D for vehicle."""
    for agent_type, agent_pred in pred.items():
        hm = agent_pred["heatmap_logits"]
        if tuple(hm.shape[-3:]) != (NUM_CLASSES, FEAT_H, FEAT_W):
            raise AssertionError(f"{agent_type} heatmap_logits {tuple(hm.shape)}")
        if not torch.isfinite(hm).all():
            raise AssertionError(f"{agent_type} heatmap logits not finite")
        if not torch.isfinite(agent_pred["depth_z_mean"]).all():
            raise AssertionError(f"{agent_type} depth_z_mean not finite")
    veh = pred["vehicle"]
    dep = veh["depth_logits"]
    if tuple(dep.shape[-2:]) != (FEAT_H, FEAT_W) or int(dep.shape[1]) != 50:
        raise AssertionError(f"vehicle depth_logits {tuple(dep.shape)} expected D=50")
    if not torch.isfinite(dep).all():
        raise AssertionError("vehicle depth logits not finite")
    zvar = veh["depth_z_var"]
    if not torch.isfinite(zvar).all() or float(zvar.min()) < 0:
        raise AssertionError("depth_z_var invalid")
    if "rsu" in pred:
        rsu_dep = pred["rsu"]["depth_logits"]
        if tuple(rsu_dep.shape[-2:]) != (FEAT_H, FEAT_W) or int(rsu_dep.shape[1]) != 48:
            raise AssertionError(f"rsu depth_logits {tuple(rsu_dep.shape)} expected D=48")
    if "drone" in pred:
        drone = pred["drone"]
        if "depth_logits" in drone or "depth_z_var" in drone:
            raise AssertionError("drone still emits categorical logits/variance")
        if "delta_pred" not in drone:
            raise AssertionError("drone missing delta_pred")
        recon = drone["camera_world_z"][:, None, None] + drone["delta_pred"]
        if not torch.allclose(drone["depth_z_mean"], recon, atol=1e-6, rtol=0.0):
            raise AssertionError("drone depth_z_mean != height + delta_pred")


def _is_efficientnet_bn_param(name: str) -> bool:
    """True for frozen EfficientNet BN affine tensors."""
    return ".trunk." in name and ("_bn" in name or ".bn" in name)


def _is_official_unused_head(name: str) -> bool:
    """True for frozen official CamEncode image/depth heads."""
    return "image_head" in name or (
        "frontend.encoders." in name and ".depth_head." in name
    )


def joint_step(
    model: torch.nn.Module,
    semantic_criterion: Any,
    depth_criterion: Any,
    device: torch.device,
) -> Dict[str, float]:
    """Shared 128-ch F90; one joint backward; concat fusion and heads move."""
    core = _unwrap_model(model)
    stats: Dict[str, float] = {}
    conv1 = core.highres["vehicle"].conv1
    conv2 = core.highres["vehicle"].conv2
    if float(conv1.weight.detach().abs().max()) == 0.0:
        raise AssertionError("concat fusion conv1 is zero-initialized")
    if float(conv2.weight.detach().abs().max()) == 0.0:
        raise AssertionError("concat fusion conv2 is zero-initialized")
    imgs = torch.rand(1, 1, 4, 360, 640, device=device)
    with torch.no_grad():
        r2, f45 = core.frontend.extract_backbone_features("vehicle", imgs)
        f45_up = F.interpolate(
            f45, size=r2.shape[-2:], mode="bilinear", align_corners=True
        )
        fused = torch.cat([r2, f45_up], dim=1)
        f90 = core.highres["vehicle"](r2, f45)
        heatmap_logits = core.heatmap_heads["vehicle"](f90)
        depth_logits = core.depth_heads["vehicle"](f90)
    if tuple(r2.shape[1:]) != (24, FEAT_H, FEAT_W):
        raise AssertionError(f"R2 {tuple(r2.shape)} expected [N,24,90,160]")
    if tuple(f45.shape[1:]) != (256, FEAT_H // 2, FEAT_W // 2):
        raise AssertionError(f"F45 {tuple(f45.shape)} expected [N,256,45,80]")
    if tuple(f45_up.shape[1:]) != (256, FEAT_H, FEAT_W):
        raise AssertionError(f"F45_up {tuple(f45_up.shape)} expected [N,256,90,160]")
    if tuple(fused.shape[1:]) != (280, FEAT_H, FEAT_W):
        raise AssertionError(f"concat {tuple(fused.shape)} expected [N,280,90,160]")
    if tuple(f90.shape[1:]) != (F90_CHANNELS, FEAT_H, FEAT_W):
        raise AssertionError(
            f"F90 {tuple(f90.shape)} expected [N,{F90_CHANNELS},90,160]"
        )
    if tuple(heatmap_logits.shape[1:]) != (NUM_CLASSES, FEAT_H, FEAT_W):
        raise AssertionError(
            f"heatmap logits {tuple(heatmap_logits.shape)} expected [N,2,90,160]"
        )
    if tuple(depth_logits.shape[1:]) != (50, FEAT_H, FEAT_W):
        raise AssertionError(
            f"vehicle depth logits {tuple(depth_logits.shape)} expected [N,50,90,160]"
        )
    print(
        f"[shapes] R2={tuple(r2.shape)} F45={tuple(f45.shape)} "
        f"F45_up={tuple(f45_up.shape)} concat={tuple(fused.shape)} "
        f"F90={tuple(f90.shape)} heatmap={tuple(heatmap_logits.shape)} "
        f"veh_depth={tuple(depth_logits.shape)} "
        "interp=bilinear align_corners=True"
    )
    stats["conv1_max_abs"] = float(conv1.weight.detach().abs().max())

    model.train()
    core.frontend.assert_train_eval_state(True)
    for module in core.frontend.encoders["vehicle"].trunk.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            if module.training:
                raise AssertionError("EfficientNet BN is in train mode")
    if not core.frontend.encoders["vehicle"].up1.training:
        raise AssertionError("up1 is not in train mode")
    if not core.frontend.encoders["vehicle"].up2.training:
        raise AssertionError("up2 is not in train mode")
    ego = fake_ego(device)
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=0.002
    )
    total_loss, heatmap_loss, depth_loss, _metrics = _forward_loss_metrics(
        model, ego, semantic_criterion, depth_criterion, scaler=None
    )
    if not torch.isfinite(total_loss):
        raise AssertionError(f"total_loss not finite: {total_loss}")
    stats["total_before"] = float(total_loss.item())
    stats["heatmap_before"] = float(heatmap_loss.item())
    stats["depth_before"] = float(depth_loss.item())
    print(
        f"[loss] total={stats['total_before']:.4f} "
        f"heatmap={stats['heatmap_before']:.4f} "
        f"depth={stats['depth_before']:.4f}"
    )
    model.zero_grad(set_to_none=True)
    total_loss.backward()
    seen = {
        "trunk": False,
        "up1": False,
        "up2": False,
        "highres": False,
        "heatmap": False,
        "depth_head": False,
        "rsu_depth_head": False,
        "drone_height_embed": False,
        "drone_delta_head": False,
        "drone_heatmap": False,
        "drone_fpn": False,
    }
    for name, param in model.named_parameters():
        if _is_official_unused_head(name) or _is_efficientnet_bn_param(name):
            if param.grad is not None:
                raise AssertionError(f"frozen {name} has grad")
            continue
        if ".trunk." in name and "vehicle" in name and (
            "_conv_stem" in name or "._blocks.0." in name
        ):
            if param.grad is None:
                raise AssertionError(f"{name} grad is None")
            seen["trunk"] = True
        if ".up1." in name and "vehicle" in name:
            if param.grad is None:
                raise AssertionError(f"{name} grad is None")
            seen["up1"] = True
        if ".up2." in name and "vehicle" in name:
            if param.grad is None:
                raise AssertionError(f"{name} grad is None")
            seen["up2"] = True
        if "highres.vehicle.conv1" in name or "highres.vehicle.conv2" in name:
            if param.grad is None or float(param.grad.abs().sum()) == 0.0:
                raise AssertionError(f"{name} got no gradient")
            seen["highres"] = True
        if "heatmap_heads.vehicle" in name:
            if param.grad is None:
                raise AssertionError(f"{name} grad is None")
            seen["heatmap"] = True
        if "depth_heads.vehicle" in name:
            if param.grad is None:
                raise AssertionError(f"{name} grad is None")
            seen["depth_head"] = True
        if "depth_heads.rsu" in name:
            if param.grad is None:
                raise AssertionError(f"{name} grad is None")
            seen["rsu_depth_head"] = True
        if "depth_heads.drone" in name:
            raise AssertionError(f"obsolete drone DepthHead is active: {name}")
        if "drone_height_embed" in name:
            if param.grad is None:
                raise AssertionError(f"{name} grad is None")
            seen["drone_height_embed"] = True
        if "drone_delta_head" in name:
            if param.grad is None:
                raise AssertionError(f"{name} grad is None")
            seen["drone_delta_head"] = True
        if "heatmap_heads.drone" in name:
            if param.grad is None:
                raise AssertionError(f"{name} grad is None")
            seen["drone_heatmap"] = True
        if "highres.drone" in name:
            if param.grad is None:
                raise AssertionError(f"{name} grad is None")
            seen["drone_fpn"] = True
        if "frontend.encoders.vehicle.depth_head" in name and param.grad is not None:
            raise AssertionError(f"official CamEncode depth_head has grad: {name}")
        if ".trunk." in name and "_bn" in name and param.grad is not None:
            raise AssertionError(f"EfficientNet BN has grad: {name}")
    missing = [key for key, ok in seen.items() if not ok]
    if missing:
        raise AssertionError(f"missing joint grads: {missing}")
    print("[grad] joint freeze/trainable contract OK")
    before = float(core.highres["vehicle"].conv1.weight.detach().abs().sum())
    optimizer.step()
    after = float(core.highres["vehicle"].conv1.weight.detach().abs().sum())
    if after == before:
        raise AssertionError("concat fusion conv1 did not move after one step")
    stats["conv1_moved"] = abs(after - before)
    print(f"[step] concat fusion conv1 moved, abs-sum delta={stats['conv1_moved']:.4e}")
    return stats


def sanity_loop(
    model: torch.nn.Module,
    semantic_criterion: Any,
    depth_criterion: Any,
    device: torch.device,
    steps: int = 5,
) -> Dict[str, List[float]]:
    """A few joint steps on the same fake batch."""
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=0.002
    )
    frozen_before = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if _is_official_unused_head(name) or _is_efficientnet_bn_param(name)
    }
    trainable_before = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and (
            ".trunk." in name or ".up1." in name or ".up2." in name
            or "highres" in name or "heatmap_heads" in name or "depth_heads" in name
            or "drone_height_embed" in name or "drone_delta_head" in name
        )
    }
    totals: List[float] = []
    heatmaps: List[float] = []
    depths: List[float] = []
    ego = fake_ego(device)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss, heatmap_loss, depth_loss, _metrics = _forward_loss_metrics(
            model, ego, semantic_criterion, depth_criterion, scaler=None
        )
        total_loss.backward()
        optimizer.step()
        totals.append(float(total_loss.item()))
        heatmaps.append(float(heatmap_loss.item()))
        depths.append(float(depth_loss.item()))
    for name, snapshot in frozen_before.items():
        now = dict(model.named_parameters())[name].detach().cpu()
        if not torch.equal(snapshot, now):
            raise AssertionError(f"frozen {name} changed during sanity")
    moved_families = set()
    named = dict(model.named_parameters())
    for name, snapshot in trainable_before.items():
        if not torch.equal(snapshot, named[name].detach().cpu()):
            if ".up1." in name:
                moved_families.add("up1")
            elif ".up2." in name:
                moved_families.add("up2")
            elif ".trunk." in name:
                moved_families.add("trunk")
            elif "highres" in name:
                moved_families.add("highres")
            elif "heatmap_heads" in name:
                moved_families.add("heatmap")
            elif "depth_heads" in name:
                moved_families.add("depth_head")
            elif "drone_height_embed" in name:
                moved_families.add("drone_height_embed")
            elif "drone_delta_head" in name:
                moved_families.add("drone_delta_head")
    expected = {
        "trunk",
        "up1",
        "up2",
        "highres",
        "heatmap",
        "depth_head",
        "drone_height_embed",
        "drone_delta_head",
    }
    if moved_families != expected:
        raise AssertionError(f"trainable families that moved={moved_families} expected={expected}")
    print(
        f"[sanity] total={['%.4f' % v for v in totals]} "
        f"heatmap={['%.4f' % v for v in heatmaps]} "
        f"depth={['%.4f' % v for v in depths]}"
    )
    return {"total": totals, "heatmap": heatmaps, "depth": depths}


def checkpoint_smoke(device: torch.device) -> None:
    """Save/load restores trainable modules."""
    model, semantic_criterion, depth_criterion = load_model(device)
    model.train()
    ego = fake_ego(device)
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=0.002
    )
    total_loss, _, _, _ = _forward_loss_metrics(
        model, ego, semantic_criterion, depth_criterion, scaler=None
    )
    total_loss.backward()
    optimizer.step()
    snapshot = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p1_joint.pth"
        torch.save({"model_state_dict": model.state_dict()}, path)
        restored, _, _ = load_model(device)
        raw = torch.load(path, map_location="cpu")
        restored.load_state_dict(raw["model_state_dict"])
    for name, tensor in snapshot.items():
        now = dict(restored.named_parameters())[name].detach().cpu()
        if not torch.equal(tensor, now):
            raise AssertionError(f"checkpoint mismatch: {name}")
    print("[ckpt] save/load restored trainable weights")
    del model, restored
    if device.type == "cuda":
        torch.cuda.empty_cache()


def real_batch_smoke(device: torch.device) -> Dict[str, float]:
    """One real dataset batch if data/GPU exist."""
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools.analyze_p1_heatmap_resolution import dummy_lidar_preprocess

    hypes = yaml_utils.load_yaml(str(_yaml()), None)
    train_candidates = [
        Path("/mnt/home/suyi/AirV2X-Perception/train/train"),
        Path("/home/dell/suyi/AirV2X-Perception/train/train"),
    ]
    root = next((p for p in train_candidates if p.is_dir()), None)
    if root is None:
        raise FileNotFoundError("no local/dell train split for real-batch smoke")
    hypes["root_dir"] = str(root)
    hypes["validate_dir"] = hypes["root_dir"]
    dataset = build_dataset(hypes, visualize=False, train=True)
    dataset.pre_processor.preprocess = dummy_lidar_preprocess
    sample = dataset[0]
    batch = dataset.collate_batch_train([sample])
    batch = train_utils.to_device(batch, device)
    ego = batch["ego"]
    model = train_utils.create_model(hypes)
    model.to(device)
    semantic_criterion, depth_criterion = build_p1_criteria(hypes)
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=0.002
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.train()
    total_loss, heatmap_loss, depth_loss, metrics = _forward_loss_metrics(
        model, ego, semantic_criterion, depth_criterion, scaler=None
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0
    t1 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_bwd = time.perf_counter() - t1
    result = {
        "total_loss": float(total_loss.item()),
        "heatmap_loss": float(heatmap_loss.item()),
        "depth_loss": float(depth_loss.item()),
        "forward_s": t_fwd,
        "backward_s": t_bwd,
    }
    if device.type == "cuda":
        result["peak_alloc_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
        result["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024**2)
    result.update(metrics)
    core = _unwrap_model(model)
    with torch.no_grad():
        pred = model(ego)
        heatmap_targets = build_heatmap_targets(ego, pred)
        depth_targets = build_depth_targets(ego, pred, core)
        for agent, logits in pred.items():
            depth_shape = tuple(logits["depth_logits"].shape) if "depth_logits" in logits else None
            z_var_shape = tuple(logits["depth_z_var"].shape) if "depth_z_var" in logits else None
            extra = ""
            if agent == "drone":
                extra = (
                    f" delta={tuple(logits['delta_pred'].shape)} "
                    f"h={tuple(logits['camera_world_z'].shape)}"
                )
            print(
                f"[real-shape] {agent} heatmap={tuple(logits['heatmap_logits'].shape)} "
                f"depth={depth_shape} "
                f"z_mean={tuple(logits['depth_z_mean'].shape)} "
                f"z_var={z_var_shape} "
                f"sem_t={tuple(heatmap_targets[agent].shape)} "
                f"dep_t={tuple(depth_targets[agent].shape)}"
                f"{extra}"
            )
        if "drone" in pred:
            _report_drone_batch(ego, pred, heatmap_targets, depth_targets, core)
    print(f"[real-batch] {YAML_NAME} {result}")
    return result


def _report_drone_batch(
    ego: Dict[str, Any],
    pred: Dict[str, Dict[str, torch.Tensor]],
    heatmap_targets: Dict[str, torch.Tensor],
    depth_targets: Dict[str, torch.Tensor],
    core: torch.nn.Module,
) -> None:
    """Print real-batch drone residual stats and height-to-view alignment."""
    from opencood.hypes_yaml.yaml_utils import load_pickle
    from opencood.models.gaussian_modules_0822.image_frontend import flatten_camera_world_z
    from opencood.models.gaussian_modules_0822.lss.head import HEIGHT_SCALE_M
    from opencood.models.gaussian_modules_0822.lss.target import extract_camera_z_gt
    from opencood.utils.airv2x_utils import compute_camera_world_z

    drone = pred["drone"]
    cam = ego["drone"]["batch_merged_cam_inputs"]
    imgs = cam["imgs"]
    f90_n = int(drone["depth_z_mean"].shape[0])
    height = drone["camera_world_z"]
    delta_pred = drone["delta_pred"]
    delta_gt = depth_targets["drone"]
    z_gt = extract_camera_z_gt(imgs)
    recon = height[:, None, None] + delta_pred
    if not torch.allclose(drone["depth_z_mean"], recon, atol=1e-5, rtol=0.0):
        raise AssertionError("depth_z_mean != camera_world_z + delta_pred")
    fg = heatmap_targets["drone"].ne(0)
    valid_z = torch.isfinite(z_gt) & (z_gt >= 6.0) & (z_gt <= 150.0)
    valid = fg & valid_z
    n_fg = int(fg.sum().item())
    n_valid = int(valid.sum().item())
    print(
        f"[drone] F90-aligned N={f90_n} "
        f"height={tuple(height.shape)} embed_in={HEIGHT_SCALE_M}m scale "
        f"delta_pred={tuple(delta_pred.shape)} delta_gt={tuple(delta_gt.shape)} "
        f"z_mean={tuple(drone['depth_z_mean'].shape)}"
    )
    h = height.detach().float().cpu()
    print(
        f"[drone] height min/median/max="
        f"{float(h.min()):.3f}/{float(h.median()):.3f}/{float(h.max()):.3f}"
    )

    def _stats(name: str, tensor: torch.Tensor, mask: torch.Tensor) -> None:
        values = tensor.detach().float()[mask].cpu()
        if values.numel() == 0:
            print(f"[drone] {name}: empty")
            return
        q = torch.quantile(values, torch.tensor([0.05, 0.50, 0.95]))
        print(
            f"[drone] {name}: mean={float(values.mean()):.4f} std={float(values.std()):.4f} "
            f"median={float(q[1]):.4f} P5={float(q[0]):.4f} P95={float(q[2]):.4f} "
            f"min={float(values.min()):.4f} max={float(values.max()):.4f}"
        )

    print(f"[drone] fg_cells={n_fg} valid_reg_cells={n_valid} excluded={n_fg - n_valid}")
    _stats("delta_gt[valid]", delta_gt, valid)
    _stats("delta_pred[valid]", delta_pred, valid)
    flat_h = flatten_camera_world_z(cam["camera_world_z"], imgs)
    ids = list(ego["drone"].get("cav_ids", []))
    paths = list(ego["drone"].get("metadata_paths", []))
    n_show = min(8, int(flat_h.shape[0]))
    for i in range(n_show):
        z_tensor = float(flat_h[i].item())
        cav_id = ids[i] if i < len(ids) else "?"
        z_meta = None
        if i < len(paths) and paths[i]:
            params = load_pickle(paths[i])
            z_meta = float(compute_camera_world_z(params, "drone")[0])
            if abs(z_meta - z_tensor) > 1e-3:
                raise AssertionError(
                    f"height/view mismatch i={i} cav={cav_id} "
                    f"meta={z_meta} tensor={z_tensor}"
                )
        print(
            f"[align] view={i} cav_id={cav_id} meta_z={z_meta} "
            f"tensor_z={z_tensor:.4f} f90_row={i}"
        )


def vehicle_lid_check() -> None:
    """Verify official LID binning for vehicle [0, 50], D=50, target=True."""
    z_vals = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 25.0, 50.0]
    z = torch.tensor(z_vals, dtype=torch.float32)
    indices, _mask = bin_depths(z, "LID", 0.0, 50.0, 50, target=True)
    centers = depth_discretization(0.0, 50.0, 50, "LID")
    print("[LID] vehicle d_min=0 d_max=50 D=50 mode=LID")
    print("[LID] first 8 bin centers:", [round(float(x), 4) for x in centers[:8]])
    print("[LID] last 4 bin centers:", [round(float(x), 4) for x in centers[-4:]])
    classes = [int(v) for v in indices.tolist()]
    for z_i, cls in zip(z_vals, classes):
        print(f"  z={z_i:g} -> class {cls}")
    if min(classes) < 0 or max(classes) > 49:
        raise AssertionError(f"vehicle LID classes out of [0,49]: {classes}")
    near = [cls for z_i, cls in zip(z_vals, classes) if 0.0 < z_i < 2.0]
    if any(cls == 49 for cls in near):
        raise AssertionError(f"0<z<2 mapped to last class: {list(zip(z_vals, classes))}")
    if classes[-1] != 49:
        raise AssertionError(f"z=50 should map to last class 49, got {classes[-1]}")
    print("[LID] near-field occupies low classes; z=50 -> 49; range [0,49] OK")


def compute_report() -> None:
    """Parameter counts and conv MACs for concat fusion / HeatmapHead / DepthHeads."""

    def conv_stats(height: int, width: int, k: int, cin: int, cout: int) -> Tuple[int, int]:
        params = cin * cout * k * k + cout
        macs = height * width * k * k * cin * cout
        return params, macs

    old_fpn_p, _ = conv_stats(90, 160, 1, 24, 64)
    old_p2, _ = conv_stats(45, 80, 1, 256, 64)
    old_fpn_p += old_p2
    fpn_p, fpn_m = conv_stats(90, 160, 3, 280, 128)
    p2, m2 = conv_stats(90, 160, 3, 128, 128)
    fpn_p += p2
    fpn_m += m2
    print(f"[compute] OLD add-fusion params/agent={old_fpn_p:,}")
    print(f"[compute] NEW concat-fusion params/agent={fpn_p:,} macs={fpn_m:,}")
    print(
        f"[compute] fusion increase/agent={fpn_p - old_fpn_p:,} "
        f"across_3_agents={3 * (fpn_p - old_fpn_p):,}"
    )
    hm_p, hm_m = conv_stats(90, 160, 3, 128, 128)
    p2, m2 = conv_stats(90, 160, 1, 128, 2)
    hm_p += p2
    hm_m += m2
    print(f"[compute] HeatmapHead params={hm_p:,} macs={hm_m:,}")
    for name, depth_d in (("vehicle", 50), ("rsu", 48)):
        dp, dm = conv_stats(90, 160, 3, 128, 128)
        p2, m2 = conv_stats(90, 160, 1, 128, depth_d)
        print(f"[compute] {name} DepthHead D={depth_d} params={dp + p2:,} macs={dm + m2:,}")
    dh_p, dh_m = conv_stats(90, 160, 3, 144, 128)
    p2, m2 = conv_stats(90, 160, 1, 128, 1)
    print(f"[compute] drone DeltaHead params={dh_p + p2:,} macs={dh_m + m2:,}")
    print("[compute] interpolate bilinear align_corners=True has no params")


def independent_backbone_check(model: torch.nn.Module) -> None:
    """Three CamEncode copies must not share parameter identity."""
    core = _unwrap_model(model)
    encoders = core.frontend.encoders
    names = ("vehicle", "rsu", "drone")
    for left, right in ((0, 1), (0, 2), (1, 2)):
        if encoders[names[left]] is encoders[names[right]]:
            raise AssertionError(f"shared encoder object {names[left]}/{names[right]}")
    param_ids = {
        name: {id(param) for param in encoders[name].parameters()} for name in names
    }
    if param_ids["vehicle"] & param_ids["rsu"]:
        raise AssertionError("vehicle/RSU share backbone parameters")
    if param_ids["vehicle"] & param_ids["drone"]:
        raise AssertionError("vehicle/drone share backbone parameters")
    if param_ids["rsu"] & param_ids["drone"]:
        raise AssertionError("RSU/drone share backbone parameters")
    print("[backbone] three independent EfficientNet-B0 CamEncode copies")


def objectness_target_unit_test() -> None:
    """tau=1 occupancy: any fg pixel -> 1; subclass identity ignored."""
    from opencood.models.gaussian_modules_0822.heatmap.target import (
        binary_objectness_target,
        build_semantic_target,
    )

    all_bg = torch.zeros(1, 4, 4, dtype=torch.long)
    if int(binary_objectness_target(all_bg, tau=1).item()) != 0:
        raise AssertionError("all-background 4x4 must be target 0")

    one_fg = torch.zeros(1, 4, 4, dtype=torch.long)
    one_fg[0, 0, 0] = 3
    if int(binary_objectness_target(one_fg, tau=1).item()) != 1:
        raise AssertionError("one foreground pixel must be target 1")

    multi = torch.zeros(1, 4, 4, dtype=torch.long)
    multi[0, 0, 0] = 1
    multi[0, 1, 2] = 6
    if int(binary_objectness_target(multi, tau=1).item()) != 1:
        raise AssertionError("multiple foreground classes must be target 1")

    cls_a = torch.zeros(1, 4, 4, dtype=torch.long)
    cls_b = torch.zeros(1, 4, 4, dtype=torch.long)
    cls_a[0, 2, 2] = 2
    cls_b[0, 2, 2] = 5
    if not torch.equal(
        binary_objectness_target(cls_a, tau=1),
        binary_objectness_target(cls_b, tau=1),
    ):
        raise AssertionError("foreground subclass must not change objectness target")

    semantic = torch.zeros(1, 1, 360, 640, dtype=torch.long)
    semantic[:, :, 0, 0] = 4
    cam = {"image_semantic_gts": semantic}
    target = build_semantic_target(cam, tau=1)
    if tuple(target.shape) != (1, FEAT_H, FEAT_W) or target.dtype != torch.long:
        raise AssertionError(f"full-res target {tuple(target.shape)} {target.dtype}")
    unique = set(int(v) for v in target.unique().tolist())
    if not unique.issubset({0, 1}):
        raise AssertionError(f"target unique {unique} not subset of {{0,1}}")
    if int(target[0, 0, 0].item()) != 1:
        raise AssertionError("pixel (0,0) in first 4x4 block should set cell (0,0)=1")
    print("[target] binary occupancy A/B/C/D ok")


def focal_numerical_unit_test() -> None:
    """gamma=0 == CE; easy cells down-weighted; hard cells keep mass; mean matches."""
    from opencood.loss.gaussian_p1_semantic_loss import softmax_focal_loss

    torch.manual_seed(0)
    logits = torch.randn(3, 2, 5, 7)
    target = torch.randint(0, 2, (3, 5, 7))
    ce_mean = F.cross_entropy(logits, target, reduction="mean")
    fl_gamma0 = softmax_focal_loss(logits, target, gamma=0.0)
    if not torch.allclose(fl_gamma0, ce_mean, atol=1e-6, rtol=1e-6):
        raise AssertionError(
            f"gamma=0 focal {float(fl_gamma0)} != CE {float(ce_mean)}"
        )

    easy = torch.zeros(1, 2, 1, 1)
    easy[0, 1, 0, 0] = 4.59512
    easy_t = torch.ones(1, 1, 1, dtype=torch.long)
    ce_easy = F.cross_entropy(easy, easy_t, reduction="none")
    p_easy = torch.softmax(easy, dim=1)[0, 1, 0, 0]
    fl_easy = softmax_focal_loss(easy, easy_t, gamma=2.0)
    manual_easy = ((1.0 - p_easy) ** 2) * ce_easy.mean()
    if not torch.allclose(fl_easy, manual_easy, atol=1e-6, rtol=1e-6):
        raise AssertionError("easy-cell focal does not match manual (1-p_t)^2 * CE")
    if float(fl_easy) >= 0.05 * float(ce_easy.mean()):
        raise AssertionError(
            f"easy p_t={float(p_easy):.4f} focal {float(fl_easy)} not << CE {float(ce_easy.mean())}"
        )

    hard = torch.zeros(1, 2, 1, 1)
    hard[0, 0, 0, 0] = 5.0
    hard[0, 1, 0, 0] = -5.0
    hard_t = torch.ones(1, 1, 1, dtype=torch.long)
    ce_hard = F.cross_entropy(hard, hard_t, reduction="none")
    p_hard = torch.softmax(hard, dim=1)[0, 1, 0, 0]
    fl_hard = softmax_focal_loss(hard, hard_t, gamma=2.0)
    if float(fl_hard) < 0.9 * float(ce_hard.mean()):
        raise AssertionError(
            f"hard p_t={float(p_hard):.6f} focal {float(fl_hard)} should stay close to CE {float(ce_hard.mean())}"
        )

    ce_none = F.cross_entropy(logits, target, reduction="none")
    p_t = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
    manual_mean = (((1.0 - p_t) ** 2) * ce_none).mean()
    fl_mean = softmax_focal_loss(logits, target, gamma=2.0)
    if not torch.allclose(fl_mean, manual_mean, atol=1e-6, rtol=1e-6):
        raise AssertionError("mean reduction does not match manual focal")
    print(
        f"[focal] gamma0==CE easy_fl={float(fl_easy):.6e} "
        f"hard_fl={float(fl_hard):.4f} p_easy={float(p_easy):.4f} p_hard={float(p_hard):.2e}"
    )


def objectness_contract_check(
    pred: Dict[str, Dict[str, torch.Tensor]],
    ego: Dict[str, Any],
    semantic_criterion: Any,
) -> None:
    """Logits [N,2,90,160], long {0,1} targets, finite focal, matching gamma."""
    if abs(float(semantic_criterion.gamma) - 2.0) > 1e-8:
        raise AssertionError(f"expected gamma=2.0, got {semantic_criterion.gamma}")
    targets = build_heatmap_targets(ego, pred)
    for agent_type, agent_pred in pred.items():
        logits = agent_pred["heatmap_logits"]
        target = targets[agent_type]
        if tuple(logits.shape[-3:]) != (2, FEAT_H, FEAT_W):
            raise AssertionError(f"{agent_type} logits {tuple(logits.shape)}")
        if target.dtype != torch.long:
            raise AssertionError(f"{agent_type} target dtype {target.dtype}")
        unique = set(int(v) for v in target.unique().tolist())
        if not unique.issubset({0, 1}):
            raise AssertionError(f"{agent_type} target unique {unique}")
        if tuple(target.shape) != tuple(logits.shape[:1] + logits.shape[-2:]):
            raise AssertionError(
                f"{agent_type} target {tuple(target.shape)} vs logits {tuple(logits.shape)}"
            )
        print(
            f"[contract] {agent_type} logits={tuple(logits.shape)} "
            f"target={tuple(target.shape)} unique={sorted(unique)}"
        )
    loss = semantic_criterion(pred, targets)
    if not torch.isfinite(loss):
        raise AssertionError(f"focal loss not finite: {loss}")
    print(f"[contract] focal={float(loss.item()):.6f} gamma={semantic_criterion.gamma}")


def _train_split_exists() -> bool:
    """True if a real AirV2X train split is present on this machine."""
    return any(
        Path(p).is_dir()
        for p in (
            "/mnt/home/suyi/AirV2X-Perception/train/train",
            "/home/dell/suyi/AirV2X-Perception/train/train",
        )
    )


def main() -> None:
    """Compile, import, joint smokes, optional real batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-real-batch", action="store_true")
    args = parser.parse_args()
    compile_modified()
    vehicle_lid_check()
    compute_report()
    objectness_target_unit_test()
    focal_numerical_unit_test()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    print(f"\n=== {YAML_NAME} ===")
    model, semantic_criterion, depth_criterion = load_model(device)
    print_trainable_families(model)
    independent_backbone_check(model)
    setup_p1_optimizer(yaml_utils.load_yaml(str(_yaml()), None), model)
    model.eval()
    ego = fake_ego(device)
    with torch.no_grad():
        pred = model(ego)
    numerical_check(pred)
    objectness_contract_check(pred, ego, semantic_criterion)
    print(
        f"heatmap {tuple(pred['vehicle']['heatmap_logits'].shape)} "
        f"depth {tuple(pred['vehicle']['depth_logits'].shape)}"
    )
    one_f90_test(model, device)
    leakage_test(model, device)
    joint_stats = joint_step(model, semantic_criterion, depth_criterion, device)
    sanity = sanity_loop(model, semantic_criterion, depth_criterion, device, steps=5)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    checkpoint_smoke(device)

    real_stats: Dict[str, float] = {}
    if not args.skip_real_batch and _train_split_exists():
        real_stats = real_batch_smoke(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("\nall smokes passed")
    print(f"init losses: {joint_stats}")
    print(f"sanity: {sanity}")
    if real_stats:
        print(f"real-batch: {real_stats}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Standalone Vehicle V45 P1 trainer. Reuses P1 loss / optimizer / logging.

Targets are built at ``output_stride=8`` (45x80). The V90 trainer and
90x160 path are not used.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch.cuda import amp
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.loss.gaussian_p1_depth_loss import GaussianP1DepthLoss
from opencood.loss.gaussian_p1_semantic_loss import GaussianP1SemanticLoss
from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    BACKGROUND_CLASS_ID,
    PRIMARY_OBJECTNESS_THRESHOLD,
    compute_heatmap_metrics,
)
from opencood.models.gaussian_modules_0822.heatmap.target import build_semantic_target
from opencood.models.gaussian_modules_0822.image_frontend import p1_agent_predictions
from opencood.models.gaussian_modules_0822.lss.metrics import compute_depth_metrics
from opencood.models.gaussian_modules_0822.lss.target import (
    build_depth_class_target,
    depth_valid_mask,
    extract_camera_z_gt,
)
from opencood.models.gaussian_modules_0822.p1_layout import BLOCK, SPATIAL_STRIDE
from opencood.models.gaussian_modules_0822.vehicle_v45_p1 import (
    F45_CHANNELS,
    OUTPUT_STRIDE,
    R2_CHANNELS,
    VehicleV45P1,
)
from opencood.tools import multi_gpu_utils, train_utils
from opencood.tools.p1_test_mix import (
    apply_test_mix_to_hypes,
    background_heatmap_targets,
    batch_is_test_mix,
    dump_mix_manifest,
)
from opencood.tools.train import (
    is_main_process,
    resume_training_from_checkpoint,
    setup_dataloader,
)
from opencood.tools.train_gaussian_p1 import (
    _log_metrics,
    _save_p1_checkpoint,
    _unwrap_model,
    build_p1_criteria,
    setup_p1_optimizer,
)

DEFAULT_YAML = (
    "opencood/hypes_yaml/airv2x/camera/gaussian_p1/vehicle_v45_p1.yaml"
)
FAR_Z_M = 20.0
AGENT = "vehicle"


def train_parser() -> argparse.Namespace:
    """CLI for Vehicle V45 P1 training and ``--verify``."""
    parser = argparse.ArgumentParser(description="Vehicle V45 P1 training")
    parser.add_argument("--hypes_yaml", "-y", type=str, default=str(DEFAULT_YAML))
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--fusion_method", "-f", default="intermediate")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--worker", default=0, type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gpu_id", default=0)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Shape / backward / latency / V90-import checks, then exit.",
    )
    return parser.parse_args()


def _output_stride(hypes: Dict[str, Any]) -> int:
    """Read ``model.args.output_stride``, default 8."""
    return int(hypes.get("model", {}).get("args", {}).get("output_stride", OUTPUT_STRIDE))


def print_v45_trainable(model: torch.nn.Module) -> None:
    """Print trainable families, including the new R2 stride-2 conv."""
    families = {
        "trunk": [],
        "up1": [],
        "up2": [],
        "r2_downsample": [],
        "highres": [],
        "heatmap_head": [],
        "depth_head": [],
        "other": [],
    }
    frozen: List[str] = []
    named = dict(model.named_parameters())
    for name, param in named.items():
        if not param.requires_grad:
            frozen.append(name)
            continue
        if ".trunk." in name:
            families["trunk"].append(name)
        elif ".up1." in name:
            families["up1"].append(name)
        elif ".up2." in name:
            families["up2"].append(name)
        elif "r2_downsample" in name:
            families["r2_downsample"].append(name)
        elif "highres" in name:
            families["highres"].append(name)
        elif "heatmap_head" in name:
            families["heatmap_head"].append(name)
        elif "depth_head" in name:
            families["depth_head"].append(name)
        else:
            families["other"].append(name)
    print("Trainable parameter families:")
    for family, names in families.items():
        print(f"  {family}: {len(names)} tensors")
        for name in names:
            print(f"    {name} {tuple(named[name].shape)}")
    if families["other"]:
        raise AssertionError(f"unexpected trainable params: {families['other']}")
    for family in ("r2_downsample", "highres", "heatmap_head", "depth_head"):
        if not families[family]:
            raise AssertionError(f"missing trainable family: {family}")
    n_frozen_bn = sum(
        1 for name in frozen if ".trunk." in name and ("_bn" in name or ".bn" in name)
    )
    print(f"Frozen EfficientNet BN tensors: {n_frozen_bn}")


def build_v45_heatmap_target(
    ego: Dict[str, Any], logits: torch.Tensor, stride: int
) -> torch.Tensor:
    """SAM3 occupancy on the stride-8 Vehicle grid."""
    cam_inputs = ego[AGENT]["batch_merged_cam_inputs"]
    target = build_semantic_target(cam_inputs, tau=1, block=int(stride))
    if tuple(target.shape[-2:]) != tuple(logits.shape[-2:]):
        raise AssertionError(
            f"semantic_target {tuple(target.shape)} vs heatmap_logits {tuple(logits.shape)}"
        )
    return target.to(device=logits.device)


def build_v45_depth_target(
    ego: Dict[str, Any],
    core: VehicleV45P1,
    logits: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """Categorical Vehicle depth indices on the stride-8 grid."""
    imgs = ego[AGENT]["batch_merged_cam_inputs"]["imgs"]
    target = build_depth_class_target(core.encoder, imgs, spatial_stride=int(stride))
    if tuple(target.shape[-2:]) != tuple(logits.shape[-2:]):
        raise AssertionError(
            f"depth target {tuple(target.shape)} vs depth_logits {tuple(logits.shape)}"
        )
    return target


def compute_v45_metrics(
    ego: Dict[str, Any],
    pred: Dict[str, torch.Tensor],
    heatmap_target: torch.Tensor,
    core: VehicleV45P1,
    stride: int,
) -> Dict[str, float]:
    """Reuse P1 heatmap/depth metrics at stride 8. Adds predFG and far MAE."""
    with torch.no_grad():
        metrics: Dict[str, float] = {}
        hm = compute_heatmap_metrics(pred["heatmap_logits"].detach(), heatmap_target)
        for key, value in hm.items():
            metrics[f"heatmap/{AGENT}/{key}"] = value
            metrics[f"heatmap/{key}"] = value
        imgs = ego[AGENT]["batch_merged_cam_inputs"]["imgs"]
        camera_z_gt = extract_camera_z_gt(imgs, spatial_stride=int(stride))
        d_min = float(core.encoder.d_min)
        d_max = float(core.encoder.d_max)
        fg_mask = heatmap_target.ne(BACKGROUND_CLASS_ID)
        depth = compute_depth_metrics(
            depth_z_mean=pred["depth_z_mean"].detach(),
            camera_z_gt=camera_z_gt,
            d_min=d_min,
            d_max=d_max,
            foreground_mask=fg_mask,
        )
        for key, value in depth.items():
            metrics[f"depth/{key}"] = value
        valid = depth_valid_mask(camera_z_gt, d_min, d_max)
        p_fg = torch.softmax(pred["heatmap_logits"].detach(), dim=1)[:, 1]
        pred_fg = p_fg.ge(PRIMARY_OBJECTNESS_THRESHOLD) & valid
        if int(pred_fg.sum().item()) > 0:
            metrics["depth/mae_pred_fg"] = float(
                (pred["depth_z_mean"].detach()[pred_fg] - camera_z_gt[pred_fg])
                .abs()
                .mean()
                .item()
            )
        else:
            metrics["depth/mae_pred_fg"] = 0.0
        far = valid & (camera_z_gt >= FAR_Z_M)
        if int(far.sum().item()) > 0:
            metrics["depth/mae_far"] = float(
                (pred["depth_z_mean"].detach()[far] - camera_z_gt[far]).abs().mean().item()
            )
        else:
            metrics["depth/mae_far"] = 0.0
        return metrics


def forward_v45_loss(
    model: torch.nn.Module,
    ego: Dict[str, Any],
    semantic_criterion: GaussianP1SemanticLoss,
    depth_criterion: GaussianP1DepthLoss,
    scaler: Optional[amp.GradScaler],
    stride: int,
    skip_heatmap: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Vehicle-only heatmap Focal + categorical depth. No Gaussian/det loss.

    When ``skip_heatmap`` (TEST-mix batch), heatmap targets are all-background
    so Focal is identically zero. Depth still trains. TEST has no SAM3.
    """
    core = _unwrap_model(model)
    with amp.autocast(enabled=scaler is not None):
        predictions = p1_agent_predictions(model(ego))
        pred = predictions[AGENT]
        if skip_heatmap:
            heatmap_targets = background_heatmap_targets(predictions)
            heatmap_target = heatmap_targets[AGENT]
            heatmap_loss = pred["heatmap_logits"].sum() * 0.0
        else:
            heatmap_target = build_v45_heatmap_target(
                ego, pred["heatmap_logits"], stride
            )
            heatmap_targets = {AGENT: heatmap_target}
            heatmap_loss = semantic_criterion(predictions, heatmap_targets)
        depth_target = build_v45_depth_target(
            ego, core, pred["depth_logits"], stride
        )
        imgs = ego[AGENT]["batch_merged_cam_inputs"]["imgs"]
        camera_z_gt = extract_camera_z_gt(imgs, spatial_stride=int(stride))
        valid = depth_valid_mask(camera_z_gt, core.encoder.d_min, core.encoder.d_max)
        depth_loss = depth_criterion(
            predictions,
            {AGENT: depth_target},
            heatmap_targets,
            {AGENT: valid},
            camera_z_gts={AGENT: camera_z_gt},
        )
        total_loss = heatmap_loss + depth_loss
    metrics = compute_v45_metrics(ego, pred, heatmap_target, core, stride)
    metrics.update(depth_criterion.loss_dict)
    return total_loss, heatmap_loss, depth_loss, metrics


def validate_v45(
    model: torch.nn.Module,
    val_loader: DataLoader,
    semantic_criterion: GaussianP1SemanticLoss,
    depth_criterion: GaussianP1DepthLoss,
    epoch: int,
    device: torch.device,
    scaler: Optional[amp.GradScaler],
    stride: int,
) -> Tuple[float, float, float, int, Dict[str, float]]:
    """Validation losses and mean diagnostic metrics."""
    model.eval()
    total_sum = 0.0
    heatmap_sum = 0.0
    depth_sum = 0.0
    n_sample = 0
    metric_sum: Dict[str, float] = {}
    show_progress = multi_gpu_utils.get_dist_info()[0] == 0
    with torch.no_grad():
        for _, batch_data in tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc="Validation",
            leave=False,
            disable=not show_progress,
        ):
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, device)
            ego = batch_data["ego"]
            ego["epoch"] = epoch
            total_loss, heatmap_loss, depth_loss, metrics = forward_v45_loss(
                model, ego, semantic_criterion, depth_criterion, scaler, stride
            )
            total_sum += float(total_loss.item())
            heatmap_sum += float(heatmap_loss.item())
            depth_sum += float(depth_loss.item())
            n_sample += 1
            for key, value in metrics.items():
                metric_sum[key] = metric_sum.get(key, 0.0) + value
    metric_mean = {key: value / max(n_sample, 1) for key, value in metric_sum.items()}
    return total_sum, heatmap_sum, depth_sum, n_sample, metric_mean


def _sync() -> None:
    """GPU barrier used around timed regions."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _summarize_ms(samples: List[float]) -> Dict[str, float]:
    """Median / P90 / max in milliseconds."""
    ordered = sorted(samples)
    n = len(ordered)
    p90_i = min(n - 1, max(0, int(round(0.9 * (n - 1)))))
    return {
        "median_ms": ordered[n // 2],
        "p90_ms": ordered[p90_i],
        "max_ms": ordered[-1],
    }


def _time_ms(fn: Any, warmup: int, repeats: int) -> Dict[str, float]:
    """Wall time of ``fn`` with CUDA sync, after warmup."""
    for _ in range(warmup):
        fn()
        _sync()
    samples: List[float] = []
    for _ in range(repeats):
        _sync()
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    return _summarize_ms(samples)


def _make_synthetic_ego(
    device: torch.device, n_cam: int, image_hw: Tuple[int, int], d_max: float
) -> Dict[str, Any]:
    """Synthetic Vehicle batch matching 360x640 + SAM3/depth channels."""
    height, width = image_hw
    rgb = torch.rand(n_cam, 3, height, width, device=device)
    depth = torch.rand(n_cam, 1, height, width, device=device) * float(d_max)
    imgs = torch.cat([rgb, depth], dim=1).view(1, n_cam, 4, height, width)
    semantic = torch.zeros(1, n_cam, height, width, dtype=torch.long, device=device)
    semantic[:, :, height // 4 : height // 2, width // 4 : width // 2] = 1
    return {
        AGENT: {
            "batch_merged_cam_inputs": {
                "imgs": imgs,
                "image_semantic_gts": semantic,
            }
        }
    }


def verify_v45(hypes: Dict[str, Any], device: torch.device) -> None:
    """Shape check, one training step, latency, V90 import smoke."""
    import py_compile

    print("=" * 60)
    print("Vehicle V45 P1 verify")
    print("=" * 60)
    stride = _output_stride(hypes)
    if stride != OUTPUT_STRIDE:
        raise AssertionError(f"output_stride={stride} expected {OUTPUT_STRIDE}")
    if BLOCK != 4 or SPATIAL_STRIDE != 4:
        raise AssertionError("V90 p1_layout BLOCK/SPATIAL_STRIDE changed")

    print("\n[1] V90 import / layout smoke")
    from opencood.models.airv2x_gaussian_0822 import Airv2xGaussian0822
    from opencood.models.gaussian_modules_0822.highres_adapter import HighResFusion
    from opencood.models.gaussian_modules_0822.p1_layout import FEAT_H, FEAT_W
    from opencood.tools import train_gaussian_p1 as v90_trainer

    _ = Airv2xGaussian0822
    _ = v90_trainer
    if (FEAT_H, FEAT_W) != (90, 160):
        raise AssertionError(f"V90 grid became {(FEAT_H, FEAT_W)}")
    py_compile.compile(
        str(root_path / "opencood/models/airv2x_gaussian_0822.py"), doraise=True
    )
    py_compile.compile(
        str(root_path / "opencood/tools/train_gaussian_p1.py"), doraise=True
    )
    print("  PASS: V90 imports, FEAT_HW=(90,160), HighResFusion unchanged")

    print("\n[2] Build VehicleV45P1")
    model = train_utils.create_model(hypes)
    if type(model).__name__ != "VehicleV45P1":
        raise AssertionError(f"unexpected model class {type(model).__name__}")
    model.to(device)
    pretrained = (hypes.get("finetune") or {}).get("pretrained")
    load_report: Dict[str, List[str]] = {}
    if pretrained and os.path.isfile(str(pretrained)):
        load_report = model.load_v90_vehicle_weights(str(pretrained), device)
    else:
        print(f"  skip V90 weight load (missing {pretrained})")
    model.train()
    model.assert_train_eval_state(True)
    print_v45_trainable(model)

    n_cam = 4
    image_hw = (360, 640)
    ego = _make_synthetic_ego(device, n_cam, image_hw, float(model.encoder.d_max))
    d_bins = int(model.num_depth_bins)
    feat_h, feat_w = int(model.feat_h), int(model.feat_w)

    print("\n[3] Runtime shapes")
    imgs = ego[AGENT]["batch_merged_cam_inputs"]["imgs"]
    feats = model.forward_features(imgs)
    expected = {
        "r2": (n_cam, R2_CHANNELS, 90, 160),
        "f45": (n_cam, F45_CHANNELS, feat_h, feat_w),
        "r2_down": (n_cam, R2_CHANNELS, feat_h, feat_w),
        "fused": (n_cam, 128, feat_h, feat_w),
        "heatmap_logits": (n_cam, 2, feat_h, feat_w),
        "depth_logits": (n_cam, d_bins, feat_h, feat_w),
        "depth_z_mean": (n_cam, feat_h, feat_w),
    }
    shape_ok = True
    for name, want in expected.items():
        got = tuple(feats[name].shape)
        status = "PASS" if got == want else "FAIL"
        if status == "FAIL":
            shape_ok = False
        print(f"  {status} {name}: {got} expected {want}")
    concat_c = int(feats["r2_down"].shape[1] + feats["f45"].shape[1])
    print(f"  concat channels: {concat_c} expected 280")
    if concat_c != 280:
        shape_ok = False
    if not shape_ok:
        raise AssertionError("runtime shape check failed")

    print("\n[4] One-step backward")
    semantic_criterion, depth_criterion = build_p1_criteria(hypes)
    optimizer = setup_p1_optimizer(hypes, model)
    optimizer.zero_grad()
    total_loss, heatmap_loss, depth_loss, _metrics = forward_v45_loss(
        model, ego, semantic_criterion, depth_criterion, scaler=None, stride=stride
    )
    total_loss.backward()
    grad_checks = {
        "r2_downsample": model.r2_downsample.weight.grad,
        "highres": model.highres.conv1.weight.grad,
        "heatmap_head": model.heatmap_head.spatial.weight.grad,
        "depth_head": model.depth_head.spatial.weight.grad,
        "encoder.up1": next(model.encoder.up1.parameters()).grad,
        "encoder.trunk_nonBN": next(
            p for n, p in model.encoder.trunk.named_parameters() if p.requires_grad
        ).grad,
    }
    backward_ok = True
    for name, grad in grad_checks.items():
        ok = grad is not None and float(grad.abs().sum().item()) > 0.0
        if not ok:
            backward_ok = False
        print(f"  {'PASS' if ok else 'FAIL'} grad {name}")
    bn_trainable = any(
        p.requires_grad
        for n, p in model.encoder.named_parameters()
        if ".trunk." in n and ("_bn" in n or ".bn" in n)
    )
    print(f"  {'PASS' if not bn_trainable else 'FAIL'} EfficientNet BN frozen")
    if bn_trainable or not backward_ok:
        raise AssertionError("backward check failed")
    print(
        f"  loss total={float(total_loss.item()):.4f} "
        f"heatmap={float(heatmap_loss.item()):.4f} "
        f"depth={float(depth_loss.item()):.4f}"
    )

    print("\n[5] Latency (N_cam=4, 360x640)")
    model.eval()
    warmup, repeats = 10, 50
    with torch.no_grad():
        timings: Dict[str, Dict[str, float]] = {}

        def backbone() -> Tuple[torch.Tensor, torch.Tensor]:
            return model.extract_r2_f45(imgs)

        r2, f45 = backbone()
        timings["backbone"] = _time_ms(backbone, warmup, repeats)
        timings["r2_downsample"] = _time_ms(
            lambda: model.r2_downsample(r2), warmup, repeats
        )
        r2_down = model.r2_downsample(r2)
        timings["highres"] = _time_ms(
            lambda: model.highres(r2_down, f45), warmup, repeats
        )
        fused = model.highres(r2_down, f45)
        timings["heatmap_head"] = _time_ms(
            lambda: model.heatmap_head(fused), warmup, repeats
        )
        timings["depth_head"] = _time_ms(
            lambda: model.depth_head(fused), warmup, repeats
        )
        timings["total_v45"] = _time_ms(
            lambda: model.forward_features(imgs), warmup, repeats
        )

        v90_highres = HighResFusion().to(device)
        v90_highres.load_state_dict(model.highres.state_dict())
        v90_highres.eval()

        def v90_highres_heads() -> None:
            fused90 = v90_highres(r2, f45)
            model.heatmap_head(fused90)
            model.depth_head(fused90)

        def v90_p1() -> None:
            r2_b, f45_b = model.extract_r2_f45(imgs)
            fused90 = v90_highres(r2_b, f45_b)
            model.heatmap_head(fused90)
            model.depth_head(fused90)

        timings["v90_highres_heads"] = _time_ms(v90_highres_heads, warmup, repeats)
        timings["total_v90_p1_stack"] = _time_ms(v90_p1, warmup, repeats)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model.forward_features(imgs)
        _sync()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            v90_p1()
        _sync()
        peak_v90_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    else:
        peak_mb = 0.0
        peak_v90_mb = 0.0
    for name, stats in timings.items():
        print(
            f"  {name}: median={stats['median_ms']:.2f} "
            f"p90={stats['p90_ms']:.2f} max={stats['max_ms']:.2f} ms"
        )
    print(f"  peak VRAM V45: {peak_mb:.1f} MiB")
    print(f"  peak VRAM V90 P1-stack: {peak_v90_mb:.1f} MiB")
    print(
        "  V90 vs V45 P1-stack comparison is latency-only; "
        "do not claim accuracy until both are measured on the same split."
    )
    if load_report:
        print(
            f"\n[6] V90 load report: loaded={len(load_report['loaded'])} "
            f"new={load_report['new']} mismatched={len(load_report['mismatched'])}"
        )
    print("\nVERIFY PASS")


def main() -> None:
    """Train standalone Vehicle V45 P1, or run ``--verify``."""
    opt = train_parser()
    yaml_path = opt.hypes_yaml
    if not os.path.isabs(yaml_path):
        yaml_path = str(root_path / yaml_path)
    hypes = yaml_utils.load_yaml(yaml_path, opt)
    hypes["tag"] = opt.tag
    stride = _output_stride(hypes)
    finetune = hypes.get("finetune") or {}
    test_mix_on = finetune.get("test_mix_keep_frac") is not None
    skip_heatmap_on_test_mix = bool(finetune.get("skip_heatmap_on_test_mix"))
    if test_mix_on:
        apply_test_mix_to_hypes(hypes)
        if not skip_heatmap_on_test_mix:
            print(
                "[test-mix] WARNING: skip_heatmap_on_test_mix=false — TEST "
                "frames have no SAM3 labels, heatmap WILL train on empty GT."
            )
    print("load from yaml file:", yaml_path)
    print(
        "P1 mode: standalone Vehicle V45 "
        f"(output_stride={stride}, expected 45x80). "
        "Loss = vehicle heatmap + vehicle depth. No Gaussian / RSU / Drone."
    )
    if test_mix_on:
        print(
            "TEST mix ON: first 20% of every TEST scene, stride 1; "
            "those batches train depth only."
        )

    if torch.cuda.is_available():
        gpu_id = int(opt.gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device("cpu")

    if opt.verify:
        verify_v45(hypes, device)
        return

    print("Building datasets...")
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    val_dataset = build_dataset(hypes, visualize=False, train=False)

    multi_gpu_utils.init_distributed_mode(opt)
    main_process = is_main_process(opt)
    train_loader = setup_dataloader(train_dataset, hypes, opt, is_train=True)
    val_loader = setup_dataloader(val_dataset, hypes, opt, is_train=False)

    print("Creating model...")
    model = train_utils.create_model(hypes)
    if opt.distributed:
        device = torch.device(f"cuda:{opt.gpu}")
        torch.cuda.set_device(opt.gpu)
    model.to(device)
    pretrained = (hypes.get("finetune") or {}).get("pretrained")
    if pretrained and not opt.model_dir:
        model.load_v90_vehicle_weights(str(pretrained), device)

    total_params = sum(p.nelement() for p in model.parameters())
    trainable_params = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print_v45_trainable(model)

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.gpu],
            output_device=opt.gpu,
            find_unused_parameters=False,
        )

    semantic_criterion, depth_criterion = build_p1_criteria(hypes)
    optimizer = setup_p1_optimizer(hypes, model)
    scaler = amp.GradScaler() if opt.amp and torch.cuda.is_available() else None
    scheduler = train_utils.setup_lr_schedular(
        hypes, optimizer, n_iter_per_epoch=len(train_loader)
    )
    save_quarters = bool((hypes.get("finetune") or {}).get("save_quarters"))

    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, _ = resume_training_from_checkpoint(
            saved_path=saved_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
    else:
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes) if main_process else None
        if opt.distributed:
            saved_path_holder = [saved_path]
            dist.broadcast_object_list(saved_path_holder, src=0)
            saved_path = saved_path_holder[0]
        if main_process:
            print(f"Results will be saved to: {saved_path}")
    assert saved_path is not None

    writer = SummaryWriter(saved_path) if main_process else None
    if test_mix_on and main_process:
        dump_mix_manifest(train_dataset, saved_path)
    print("Starting Vehicle V45 P1 training...")
    epochs = hypes["train_params"]["epoches"]

    for epoch in range(init_epoch, epochs):
        if opt.distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_dataset, "set_fog_epoch"):
            train_dataset.set_fog_epoch(epoch)
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        model.train()
        _unwrap_model(model).assert_train_eval_state(True)

        n_iter = len(train_loader)
        quarter_hits = {}
        if save_quarters:
            from opencood.tools.train_gaussian_p1 import _quarter_save_iters

            quarter_hits = _quarter_save_iters(n_iter)
        pbar = tqdm(enumerate(train_loader), total=n_iter, disable=not main_process)
        for i, batch_data in pbar:
            if batch_data is None:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            ego = batch_data["ego"]
            ego["epoch"] = epoch
            is_test_mix = test_mix_on and batch_is_test_mix(ego)
            total_loss, heatmap_loss, depth_loss, metrics = forward_v45_loss(
                model,
                ego,
                semantic_criterion,
                depth_criterion,
                scaler,
                stride,
                skip_heatmap=is_test_mix and skip_heatmap_on_test_mix,
            )
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            if main_process:
                assert writer is not None
                step = epoch * n_iter + i
                heatmap_v = float(heatmap_loss.item())
                depth_v = float(depth_loss.item())
                total_v = float(total_loss.item())
                writer.add_scalar("Train/total_loss", total_v, step)
                writer.add_scalar("Train/heatmap_loss", heatmap_v, step)
                writer.add_scalar("Train/depth_loss", depth_v, step)
                pbar.set_description(
                    "[epoch %d][%d/%d] || total: %.4f | focal: %.4f | depth: %.4f"
                    % (epoch, i + 1, n_iter, total_v, heatmap_v, depth_v)
                )
                _log_metrics(writer, metrics, step, "Train")
                with open(os.path.join(saved_path, "train_loss.txt"), "a+") as handle:
                    handle.write(
                        f"Epoch[{epoch}], iter[{i}/{n_iter}], "
                        f"total[{total_v:.4f}], heatmap[{heatmap_v:.4f}], "
                        f"depth[{depth_v:.4f}]\n"
                    )

            if save_quarters and (i + 1) in quarter_hits:
                if opt.distributed:
                    torch.cuda.synchronize()
                    torch.distributed.barrier()
                if opt.rank == 0:
                    quarter = quarter_hits[i + 1]
                    _save_p1_checkpoint(
                        saved_path,
                        f"net_epoch{epoch + 1}_q{quarter}.pth",
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                    )
                    if quarter == 4:
                        _save_p1_checkpoint(
                            saved_path,
                            f"net_epoch{epoch + 1}.pth",
                            epoch,
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                        )

        if opt.distributed:
            torch.cuda.synchronize()
            torch.distributed.barrier()

        if (
            not save_quarters
            and opt.rank == 0
            and epoch % hypes["train_params"]["save_freq"] == 0
        ):
            _save_p1_checkpoint(
                saved_path,
                f"net_epoch{epoch + 1}.pth",
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
            )

        if epoch % hypes["train_params"]["eval_freq"] == 0:
            if opt.distributed and isinstance(val_loader.sampler, DistributedSampler):
                val_loader.sampler.set_epoch(epoch)
            local_total, local_hm, local_dep, local_cnt, extra_metrics = validate_v45(
                model,
                val_loader,
                semantic_criterion,
                depth_criterion,
                epoch,
                device,
                scaler,
                stride,
            )
            if opt.distributed:
                stats = torch.tensor(
                    [local_total, local_hm, local_dep, local_cnt],
                    dtype=torch.float32,
                    device=device,
                )
                torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                global_total, global_hm, global_dep, global_cnt = stats.tolist()
            else:
                global_total, global_hm, global_dep, global_cnt = (
                    local_total,
                    local_hm,
                    local_dep,
                    local_cnt,
                )
            if main_process:
                assert writer is not None
                denom = max(global_cnt, 1)
                val_total = global_total / denom
                val_hm = global_hm / denom
                val_dep = global_dep / denom
                print(
                    f"Epoch {epoch}: val total={val_total:.4f} "
                    f"heatmap={val_hm:.4f} depth={val_dep:.4f}"
                )
                writer.add_scalar("Validate/total_loss", val_total, epoch)
                writer.add_scalar("Validate/heatmap_loss", val_hm, epoch)
                writer.add_scalar("Validate/depth_loss", val_dep, epoch)
                _log_metrics(writer, extra_metrics, epoch, "Validate")

        scheduler.step(epoch)
        if opt.distributed:
            torch.distributed.barrier(device_ids=[opt.gpu])

    torch.cuda.empty_cache()
    if writer is not None:
        writer.close()
    if opt.distributed:
        dist.barrier()
        dist.destroy_process_group()
    if main_process:
        print(f"Vehicle V45 P1 training finished. Checkpoints saved to {saved_path}")


if __name__ == "__main__":
    main()

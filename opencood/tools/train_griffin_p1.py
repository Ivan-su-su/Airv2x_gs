# -*- coding: utf-8 -*-
"""Fine-tune AirV2X Gaussian-0822 P1 on Griffin 55m raw data.

Trainable paths:
* Vehicle: independent EfficientNet-B0/CamEncode + V45 HighResFusion + heatmap + depth.
* Drone: independent EfficientNet-B0/CamEncode + V90 HighResFusion + bottom heatmap.

Drone depth is deliberately NOT trained. Downstream Griffin inference should
use analytic bottom-camera ideal ground projection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
import yaml
from tensorboardX import SummaryWriter
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opencood.data_utils.datasets.griffin_p1_dataset import GriffinP1Dataset
from opencood.loss.gaussian_p1_depth_loss import GaussianP1DepthLoss
from opencood.loss.gaussian_p1_semantic_loss import softmax_focal_loss
from opencood.models.griffin_gaussian_p1 import GriffinGaussianP1
from opencood.models.gaussian_modules_0822.heatmap.metrics import compute_heatmap_metrics
from opencood.utils.camera_utils import bin_depths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="opencood/hypes_yaml/griffin/camera/gaussian_p1/griffin_gaussian_p1_v45.yaml",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"bad yaml: {path}")
    return cfg



def _dist_context(args: argparse.Namespace) -> Tuple[bool, int, int, int, torch.device]:
    """Initialize torchrun DDP when WORLD_SIZE>1; otherwise use --gpu."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP Griffin P1 training requires CUDA")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = int(args.gpu)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    return distributed, rank, world_size, local_rank, device


def _core_model(model: torch.nn.Module) -> GriffinGaussianP1:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _mean_metrics_across_ranks(
    sums: Dict[str, float],
    count: int,
    device: torch.device,
    distributed: bool,
) -> Dict[str, float]:
    """Mean scalar metrics over local batches and, under DDP, over all ranks."""
    if not sums:
        return {}
    keys = sorted(sums.keys())
    values = [float(sums[k]) for k in keys] + [float(count)]
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    denom = max(float(tensor[-1].item()), 1.0)
    return {
        key: float(tensor[i].item()) / denom
        for i, key in enumerate(keys)
    }


def move_nested(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_nested(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_nested(v, device) for v in batch)
    return batch


def flatten_views(x: torch.Tensor) -> torch.Tensor:
    if x.dim() < 3:
        raise ValueError(f"expected [B,V,...], got {tuple(x.shape)}")
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def _parameter_agent(name: str) -> str:
    """Return the unique Griffin P1 owner of one trainable parameter."""
    vehicle_prefixes = (
        "frontend.encoders.vehicle.",
        "highres.vehicle.",
        "heatmap_heads.vehicle.",
        "depth_heads.vehicle.",
        "r2_downsample.",
    )
    drone_prefixes = (
        "frontend.encoders.drone.",
        "highres.drone.",
        "heatmap_heads.drone.",
    )
    if name.startswith(vehicle_prefixes):
        return "vehicle"
    if name.startswith(drone_prefixes):
        return "drone"
    raise AssertionError(f"trainable parameter has no unique agent owner: {name}")


def _parameter_family(name: str) -> str:
    if ".trunk." in name:
        return "trunk"
    if (
        ".up1." in name
        or ".up2." in name
        or name.startswith("highres.")
        or name.startswith("r2_downsample.")
    ):
        return "feature"
    return "head"


def assert_branch_parameter_isolation(model: torch.nn.Module) -> None:
    """Assert Vehicle/Drone own disjoint trainable parameter objects."""
    owners = {"vehicle": {}, "drone": {}}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        agent = _parameter_agent(name)
        owners[agent][name] = id(param)

    if not owners["vehicle"] or not owners["drone"]:
        raise AssertionError(
            f"empty trainable branch: vehicle={len(owners['vehicle'])}, "
            f"drone={len(owners['drone'])}"
        )
    shared_ids = set(owners["vehicle"].values()) & set(owners["drone"].values())
    if shared_ids:
        raise AssertionError(
            f"Vehicle/Drone unexpectedly share {len(shared_ids)} trainable tensors"
        )

    print(
        "[branch isolation] independent trainable parameter objects: "
        f"vehicle={len(owners['vehicle'])}, drone={len(owners['drone'])}, shared=0"
    )


def setup_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW with explicit Vehicle/Drone x trunk/feature/head groups.

    Splitting groups does not change the loss graph; it makes agent ownership
    and future branch-specific LR changes explicit.
    """
    opt = cfg["optimizer"]
    lrs = {
        "trunk": float(opt.get("trunk_lr", 2e-5)),
        "feature": float(opt.get("feature_lr", 1e-4)),
        "head": float(opt.get("head_lr", 2e-4)),
    }
    wd = float(opt.get("weight_decay", 1e-4))

    grouped: Dict[Tuple[str, str], list] = {
        (agent, family): []
        for agent in ("vehicle", "drone")
        for family in ("trunk", "feature", "head")
    }
    grouped_names: Dict[Tuple[str, str], list] = {
        key: [] for key in grouped
    }

    seen_ids = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen_ids:
            raise AssertionError(f"duplicate trainable parameter object: {name}")
        seen_ids.add(id(param))
        key = (_parameter_agent(name), _parameter_family(name))
        grouped[key].append(param)
        grouped_names[key].append(name)

    # All six groups should be populated in this two-agent design.
    empty = [key for key, params in grouped.items() if not params]
    if empty:
        raise AssertionError(f"empty optimizer groups: {empty}")

    param_groups = []
    for agent in ("vehicle", "drone"):
        for family in ("trunk", "feature", "head"):
            params = grouped[(agent, family)]
            param_groups.append(
                {
                    "params": params,
                    "lr": lrs[family],
                    "agent": agent,
                    "family": family,
                }
            )
            print(
                f"Optimizer group {agent:7s}/{family:7s}: "
                f"{len(params)} tensors lr={lrs[family]:g}"
            )

    return torch.optim.AdamW(param_groups, weight_decay=wd)


def assert_branch_gradient_isolation(
    model: torch.nn.Module,
    loss_parts: Dict[str, torch.Tensor],
) -> None:
    """One-time autograd check: each agent loss must be disconnected from the other branch."""
    vehicle_params = []
    drone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _parameter_agent(name) == "vehicle":
            vehicle_params.append(param)
        else:
            drone_params.append(param)

    veh_to_drone = torch.autograd.grad(
        loss_parts["vehicle"],
        drone_params,
        retain_graph=True,
        allow_unused=True,
    )
    drone_to_vehicle = torch.autograd.grad(
        loss_parts["drone"],
        vehicle_params,
        retain_graph=True,
        allow_unused=True,
    )
    if any(grad is not None for grad in veh_to_drone):
        raise AssertionError("Vehicle loss has an autograd path into Drone parameters")
    if any(grad is not None for grad in drone_to_vehicle):
        raise AssertionError("Drone loss has an autograd path into Vehicle parameters")

    veh_to_vehicle = torch.autograd.grad(
        loss_parts["vehicle"],
        vehicle_params,
        retain_graph=True,
        allow_unused=True,
    )
    drone_to_drone = torch.autograd.grad(
        loss_parts["drone"],
        drone_params,
        retain_graph=True,
        allow_unused=True,
    )
    if not any(grad is not None for grad in veh_to_vehicle):
        raise AssertionError("Vehicle loss is disconnected from its own branch")
    if not any(grad is not None for grad in drone_to_drone):
        raise AssertionError("Drone loss is disconnected from its own branch")

    print(
        "[gradient isolation] PASS: "
        "Vehicle loss -> Vehicle only; Drone loss -> Drone only"
    )


def build_depth_target(
    z_gt: torch.Tensor,
    valid_from_dataset: torch.Tensor,
    ddiscr: Tuple[float, float, int],
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dmin, dmax, nbins = float(ddiscr[0]), float(ddiscr[1]), int(ddiscr[2])
    valid = (
        valid_from_dataset.bool()
        & torch.isfinite(z_gt)
        & (z_gt >= dmin)
        & (z_gt <= dmax)
    )
    safe = torch.where(valid, z_gt, torch.full_like(z_gt, dmax))
    target, _ = bin_depths(safe, mode, dmin, dmax, nbins, target=True)
    return target.long(), valid


def batch_loss_metrics(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    depth_criterion: GaussianP1DepthLoss,
    cfg: Dict[str, Any],
    use_amp: bool,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    loss_cfg = cfg.get("loss", {})
    gamma = float(loss_cfg.get("heatmap_gamma", 2.0))
    w_v_hm = float(loss_cfg.get("vehicle_heatmap_weight", 1.0))
    w_d_hm = float(loss_cfg.get("drone_heatmap_weight", 1.0))
    w_v_depth = float(loss_cfg.get("vehicle_depth_weight", 1.0))

    with amp.autocast(enabled=use_amp):
        pred = model(batch)

        v_hm = flatten_views(batch["vehicle"]["heatmap_target"]).long()
        d_hm = flatten_views(batch["drone"]["heatmap_target"]).long()
        v_hm_loss = softmax_focal_loss(pred["vehicle"]["heatmap_logits"], v_hm, gamma)
        d_hm_loss = softmax_focal_loss(pred["drone"]["heatmap_logits"], d_hm, gamma)

        z_gt = flatten_views(batch["vehicle"]["camera_z_gt"]).float()
        valid0 = flatten_views(batch["vehicle"]["depth_valid"]).bool()
        veh_cam_cfg = cfg["model"]["vehicle"]["cam"]
        ddiscr = tuple(veh_cam_cfg["grid_conf"]["ddiscr"])
        mode = str(veh_cam_cfg["grid_conf"]["mode"])
        depth_target, depth_valid = build_depth_target(z_gt, valid0, ddiscr, mode)

        v_depth_loss = depth_criterion(
            {"vehicle": pred["vehicle"]},
            {"vehicle": depth_target},
            {"vehicle": v_hm},
            {"vehicle": depth_valid},
            camera_z_gts={"vehicle": z_gt},
        )
        vehicle_total = w_v_hm * v_hm_loss + w_v_depth * v_depth_loss
        drone_total = w_d_hm * d_hm_loss
        total = vehicle_total + drone_total

    metrics: Dict[str, float] = {
        "loss_total": float(total.detach().item()),
        "loss_vehicle_hm": float(v_hm_loss.detach().item()),
        "loss_drone_hm": float(d_hm_loss.detach().item()),
        "loss_vehicle_depth": float(v_depth_loss.detach().item()),
        "vehicle_depth_valid_ratio": float(depth_valid.float().mean().item()),
        "vehicle_mask_raw_fg_ratio": float(batch["vehicle"]["mask_fg_ratio"].mean().item()),
        "drone_mask_raw_fg_ratio": float(batch["drone"]["mask_fg_ratio"].mean().item()),
    }
    for prefix, logits, target in (
        ("vehicle_hm", pred["vehicle"]["heatmap_logits"], v_hm),
        ("drone_hm", pred["drone"]["heatmap_logits"], d_hm),
    ):
        for key, value in compute_heatmap_metrics(logits, target).items():
            metrics[f"{prefix}/{key}"] = value

    if int(depth_valid.sum().item()) > 0:
        err = torch.abs(pred["vehicle"]["depth_z_mean"] - z_gt)[depth_valid]
        metrics["vehicle_depth_mae_m"] = float(err.mean().item())
        metrics["vehicle_depth_median_ae_m"] = float(err.median().item())
    else:
        metrics["vehicle_depth_mae_m"] = 0.0
        metrics["vehicle_depth_median_ae_m"] = 0.0
    return total, metrics, {
        "vehicle": vehicle_total,
        "drone": drone_total,
    }


def reduce_metric_sums(sums: Dict[str, float], count: int) -> Dict[str, float]:
    return {k: v / max(count, 1) for k, v in sums.items()}


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: GaussianP1DepthLoss,
    cfg: Dict[str, Any],
    device: torch.device,
    use_amp: bool,
    distributed: bool,
    rank: int,
) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="val",
            leave=False,
            disable=rank != 0,
        ):
            batch = move_nested(batch, device)
            _loss, metrics, _parts = batch_loss_metrics(
                model, batch, criterion, cfg, use_amp
            )
            count += 1
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v)
    return _mean_metrics_across_ranks(
        sums, count, device=device, distributed=distributed
    )


def save_ckpt(
    path: Path,
    epoch: int,
    model: GriffinGaussianP1,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: amp.GradScaler,
    cfg: Dict[str, Any],
    init_report: Dict[str, Any],
    val_metrics: Dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": _core_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": cfg,
            "airv2x_init_report": init_report,
            "val_metrics": val_metrics,
        },
        str(path),
    )


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    if args.workers is not None:
        cfg["train"]["workers"] = args.workers
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    distributed, rank, world_size, local_rank, device = _dist_context(args)
    main_process = rank == 0
    if main_process:
        print(
            f"[runtime] distributed={distributed} world_size={world_size} "
            f"device={device}"
        )

    train_ds = GriffinP1Dataset(cfg["dataset"], "train")
    val_ds = GriffinP1Dataset(cfg["dataset"], "val")
    batch_size = int(cfg["train"].get("batch_size", 2))
    workers = int(cfg["train"].get("workers", 8))

    train_sampler = (
        DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if distributed
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=workers > 0,
    )

    # Build and initialize each rank from the same AirV2X V45 P1 checkpoint.
    core = GriffinGaussianP1(cfg["model"]).to(device)
    init_report = core.load_airv2x_v45(cfg["train"]["pretrained"], device)
    core.assert_train_eval_state(True)
    assert_branch_parameter_isolation(core)

    optimizer = setup_optimizer(core, cfg)
    epochs = int(cfg["train"].get("epochs", 15))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    use_amp = bool(args.amp or cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = amp.GradScaler(enabled=use_amp)
    depth_criterion = GaussianP1DepthLoss(cfg.get("depth_loss", {})).to(device)

    output_dir = Path(cfg["train"]["output_dir"]).expanduser().resolve()
    if main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(output_dir))
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False)
        )
        (output_dir / "airv2x_init_report.json").write_text(
            json.dumps(init_report, indent=2)
        )
    else:
        writer = None

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        core.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(
            (ckpt.get("val_metrics") or {}).get("loss_total", best_val)
        )
        if main_process:
            print(f"Resumed {args.resume} at epoch {start_epoch}")

    # Verify the actual autograd graph once before attaching DDP hooks.
    if main_process:
        core.train()
        core.assert_train_eval_state(True)
        probe_batch = next(iter(train_loader))
        probe_batch = move_nested(probe_batch, device)
        _probe_total, _probe_metrics, probe_parts = batch_loss_metrics(
            core, probe_batch, depth_criterion, cfg, use_amp
        )
        assert_branch_gradient_isolation(core, probe_parts)
        del probe_batch, _probe_total, _probe_metrics, probe_parts
        torch.cuda.empty_cache()
    if distributed:
        dist.barrier()

    model: torch.nn.Module
    if distributed:
        model = DistributedDataParallel(
            core,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        model = core

    if main_process:
        global_batch = batch_size * world_size
        print(
            f"[loader] batch_size_per_gpu={batch_size} global_batch={global_batch} "
            f"workers_per_rank={workers}; train_iters_per_rank={len(train_loader)}"
        )

    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        _core_model(model).assert_train_eval_state(True)

        running: Dict[str, float] = {}
        n = 0
        pbar = tqdm(
            train_loader,
            desc=f"train {epoch+1}/{epochs}",
            disable=not main_process,
        )
        for batch in pbar:
            batch = move_nested(batch, device)
            optimizer.zero_grad(set_to_none=True)
            total, metrics, _loss_parts = batch_loss_metrics(
                model, batch, depth_criterion, cfg, use_amp
            )
            scaler.scale(total).backward()
            if float(cfg["train"].get("grad_clip", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg["train"]["grad_clip"])
                )
            scaler.step(optimizer)
            scaler.update()

            n += 1
            global_step += 1
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + float(v)
                if writer is not None:
                    writer.add_scalar(f"Train/{k}", float(v), global_step)
            if main_process:
                pbar.set_postfix(
                    total=f"{metrics['loss_total']:.3f}",
                    vh=f"{metrics['loss_vehicle_hm']:.3f}",
                    dh=f"{metrics['loss_drone_hm']:.3f}",
                    vd=f"{metrics['loss_vehicle_depth']:.3f}",
                )

        train_mean = _mean_metrics_across_ranks(
            running, n, device=device, distributed=distributed
        )
        val_metrics = validate(
            model,
            val_loader,
            depth_criterion,
            cfg,
            device,
            use_amp,
            distributed=distributed,
            rank=rank,
        )
        scheduler.step()

        if main_process:
            assert writer is not None
            for k, v in train_mean.items():
                writer.add_scalar(f"EpochTrain/{k}", v, epoch + 1)
            for k, v in val_metrics.items():
                writer.add_scalar(f"Validate/{k}", v, epoch + 1)

            print(
                f"Epoch {epoch+1}: train={train_mean.get('loss_total', 0):.4f} "
                f"val={val_metrics.get('loss_total', 0):.4f} | "
                f"veh hm F1@.3={val_metrics.get('vehicle_hm/f1@0.3', 0):.3f} | "
                f"drone hm F1@.3={val_metrics.get('drone_hm/f1@0.3', 0):.3f} | "
                f"veh depth MAE={val_metrics.get('vehicle_depth_mae_m', 0):.3f}m"
            )

            latest = output_dir / "latest.pth"
            save_ckpt(
                latest,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                cfg,
                init_report,
                val_metrics,
            )
            if val_metrics.get("loss_total", float("inf")) < best_val:
                best_val = float(val_metrics["loss_total"])
                save_ckpt(
                    output_dir / "best.pth",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    cfg,
                    init_report,
                    val_metrics,
                )
            save_every = int(cfg["train"].get("save_every", 5))
            if save_every > 0 and (epoch + 1) % save_every == 0:
                save_ckpt(
                    output_dir / f"epoch_{epoch+1:03d}.pth",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    cfg,
                    init_report,
                    val_metrics,
                )

        if distributed:
            dist.barrier()

    if writer is not None:
        writer.close()
    if main_process:
        print(f"Finished. Checkpoints: {output_dir}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

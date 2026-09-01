# -*- coding: utf-8 -*-
"""Joint P1 trainer: shared F90, CE + Focal, one backward.

Reuses official ``train_utils`` (create_model, optimizer, scheduler,
checkpoint, DDP, AMP, to_device). Instantiates the two P1 criteria
directly. Does not use detection ``label_dict``. Does not call heatmap
geometry.
"""

from __future__ import annotations

import os
import sys
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
from opencood.models.gaussian_modules_0822.image_frontend import flatten_camera_world_z
from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    BACKGROUND_CLASS_ID,
    compute_heatmap_metrics,
)
from opencood.models.gaussian_modules_0822.heatmap.box_support import (
    build_drone_union_target,
)
from opencood.models.gaussian_modules_0822.heatmap.target import build_semantic_target
from opencood.models.gaussian_modules_0822.lss.metrics import compute_depth_metrics
from opencood.models.gaussian_modules_0822.lss.target import (
    build_depth_class_target,
    depth_valid_mask,
    extract_camera_z_gt,
)
from opencood.tools import multi_gpu_utils, train_utils
from opencood.tools.train import (
    _adapt_state_dict_for_model,
    _get_checkpoint_value,
    _get_model_state_dict_for_save,
    is_main_process,
    resume_training_from_checkpoint,
    setup_dataloader,
    train_parser,
)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module when wrapped by DDP."""
    return model.module if hasattr(model, "module") else model


def _heatmap_only(hypes: Dict[str, Any]) -> bool:
    """True when yaml ``p1_finetune.heatmap_only`` is set."""
    return bool(hypes.get("p1_finetune", {}).get("heatmap_only", False))


def apply_heatmap_finetune_freeze(model: torch.nn.Module) -> None:
    """Freeze trunk / up / fusion / depth; keep the three HeatmapHeads trainable."""
    core = _unwrap_model(model)
    core.heatmap_only = True
    core.frontend.freeze_backbone = True
    for param in core.parameters():
        param.requires_grad = False
    for head in core.heatmap_heads.values():
        for param in head.parameters():
            param.requires_grad = True
    core.frontend.apply_train_eval_state(True)
    core.highres.eval()
    core.depth_heads.eval()
    core.drone_height_embed.eval()
    core.drone_delta_head.eval()
    print(
        "P1 heatmap finetune: frozen EfficientNet trunk, up1/up2, concat128, "
        "Vehicle/RSU DepthHead, drone delta-depth; trainable HeatmapHead x3"
    )


def load_pretrained_weights(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
) -> None:
    """Load model weights only. Does not restore optimizer or scheduler."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"pretrained ckpt not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Checkpoint '{ckpt_path}' must be a dict, got {type(checkpoint)}"
        )
    model_state_dict = _get_checkpoint_value(
        checkpoint,
        ["model_state_dict", "state_dict", "model"],
        f"Checkpoint '{ckpt_path}' is missing required field 'model_state_dict'.",
    )
    ckpt_state = _adapt_state_dict_for_model(model, model_state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(ckpt_state, strict=True)
    source_epoch = checkpoint.get("epoch", None)
    print(f"Loaded pretrained weights from {ckpt_path}")
    if source_epoch is not None:
        print(
            f"  source checkpoint epoch field={source_epoch}; "
            "heatmap finetune starts at epoch 0 in a new log dir"
        )
    if missing_keys or unexpected_keys:
        raise AssertionError(
            f"pretrained load mismatch missing={missing_keys} unexpected={unexpected_keys}"
        )


def _quarter_save_marks(n_iter: int) -> Dict[int, int]:
    """Map 1-based iter → quarter index 1..4. Last mark is always the epoch end."""
    if n_iter <= 0:
        raise ValueError(f"n_iter must be positive, got {n_iter}")
    marks: Dict[int, int] = {}
    for quarter in (1, 2, 3, 4):
        step = n_iter if quarter == 4 else max(1, (quarter * n_iter) // 4)
        marks[step] = quarter
    marks[n_iter] = 4
    return marks


def _save_p1_checkpoint(
    saved_path: str,
    filename: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[amp.GradScaler],
) -> str:
    """Write one P1 checkpoint dict to ``saved_path/filename``."""
    save_dict = {
        "epoch": epoch,
        "model_state_dict": _get_model_state_dict_for_save(model),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        save_dict["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        save_dict["scaler_state_dict"] = scaler.state_dict()
    out_path = os.path.join(saved_path, filename)
    torch.save(save_dict, out_path)
    return out_path


def _mean_dicts(items: List[Dict[str, float]]) -> Dict[str, float]:
    """Average scalar metrics across agents."""
    if not items:
        return {}
    keys = items[0].keys()
    return {key: sum(item[key] for item in items) / len(items) for key in keys}


def build_p1_criteria(
    hypes: Dict[str, Any],
) -> Tuple[GaussianP1SemanticLoss, GaussianP1DepthLoss]:
    """Instantiate the two P1 criteria from yaml ``loss.heatmap`` / ``loss.depth``."""
    loss_cfg = hypes.get("loss", {})
    heatmap_args = dict(loss_cfg.get("heatmap", {}).get("args", {}) or {})
    depth_args = dict(loss_cfg.get("depth", {}).get("args", {}) or {})
    return GaussianP1SemanticLoss(heatmap_args), GaussianP1DepthLoss(depth_args)


def print_trainable_families(model: torch.nn.Module) -> None:
    """Print trainable parameter families once at startup."""
    families = {
        "trunk": [],
        "up1": [],
        "up2": [],
        "highres": [],
        "heatmap_head": [],
        "depth_head": [],
        "drone_height_embed": [],
        "drone_delta_head": [],
        "other": [],
    }
    frozen = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen.append(name)
            continue
        if ".trunk." in name:
            families["trunk"].append(name)
        elif ".up1." in name:
            families["up1"].append(name)
        elif ".up2." in name:
            families["up2"].append(name)
        elif "highres" in name:
            families["highres"].append(name)
        elif "heatmap_heads" in name:
            families["heatmap_head"].append(name)
        elif "depth_heads" in name:
            families["depth_head"].append(name)
        elif "drone_height_embed" in name:
            families["drone_height_embed"].append(name)
        elif "drone_delta_head" in name:
            families["drone_delta_head"].append(name)
        else:
            families["other"].append(name)
    named = dict(model.named_parameters())
    print("Trainable parameter families:")
    for family, names in families.items():
        print(f"  {family}: {len(names)} tensors")
        for name in names:
            print(f"    {name} {tuple(named[name].shape)}")
    n_frozen_bn = sum(
        1 for name in frozen if ".trunk." in name and ("_bn" in name or ".bn" in name)
    )
    n_frozen_image = sum(1 for name in frozen if "image_head" in name)
    n_frozen_official_depth = sum(
        1
        for name in frozen
        if "frontend.encoders." in name and ".depth_head." in name
    )
    print(
        "Frozen: EfficientNet BN="
        f"{n_frozen_bn} tensors, image_head={n_frozen_image} tensors, "
        f"official depth_head={n_frozen_official_depth} tensors"
    )
    unexpected_trainable = families["other"]
    if unexpected_trainable:
        raise AssertionError(f"unexpected trainable params: {unexpected_trainable}")
    heatmap_only = bool(getattr(_unwrap_model(model), "heatmap_only", False))
    if heatmap_only:
        required = ("heatmap_head",)
        frozen_required = (
            "trunk",
            "up1",
            "up2",
            "highres",
            "depth_head",
            "drone_height_embed",
            "drone_delta_head",
        )
        for family in frozen_required:
            if families[family]:
                raise AssertionError(
                    f"heatmap finetune: {family} still trainable: {families[family]}"
                )
    else:
        required = (
            "trunk",
            "up1",
            "up2",
            "highres",
            "heatmap_head",
            "depth_head",
            "drone_height_embed",
            "drone_delta_head",
        )
    for family in required:
        if not families[family]:
            raise AssertionError(f"missing trainable family: {family}")
    drone_categorical = [
        name
        for name in families["depth_head"]
        if "depth_heads.drone" in name or "depth_heads.drone" in name
    ]
    if drone_categorical:
        raise AssertionError(
            f"obsolete drone categorical DepthHead still trainable: {drone_categorical}"
        )
    official_trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and (
            "image_head" in name
            or ("frontend.encoders." in name and ".depth_head." in name)
        )
    ]
    if official_trainable:
        raise AssertionError(f"official heads still trainable: {official_trainable}")


def setup_p1_optimizer(hypes: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    """P1-local Adam groups: non-BN trunk at 0.1 * base_lr, new modules at base_lr."""
    opt_cfg = hypes["optimizer"]
    optimizer_cls = getattr(torch.optim, opt_cfg["core_method"], None)
    if optimizer_cls is None:
        raise ValueError(f"{opt_cfg['core_method']} is not supported")
    base_lr = float(opt_cfg["lr"])
    trunk_lr = 0.1 * base_lr
    extra = dict(opt_cfg.get("args") or {})
    trunk_params = []
    new_params = []
    trunk_ids = set()
    new_ids = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "image_head" in name or (
            "frontend.encoders." in name and ".depth_head." in name
        ):
            raise AssertionError(f"official head entered optimizer: {name}")
        if ".trunk." in name:
            trunk_params.append(param)
            trunk_ids.add(id(param))
        else:
            new_params.append(param)
            new_ids.add(id(param))
    if trunk_ids & new_ids:
        raise AssertionError("optimizer groups share parameter ids")
    trainable_ids = {id(param) for param in model.parameters() if param.requires_grad}
    grouped_ids = trunk_ids | new_ids
    if trainable_ids != grouped_ids:
        raise AssertionError("optimizer groups do not cover trainable params exactly once")
    frozen_ids = {id(param) for param in model.parameters() if not param.requires_grad}
    if frozen_ids & grouped_ids:
        raise AssertionError("frozen params appeared in optimizer")
    heatmap_only = _heatmap_only(hypes)
    if heatmap_only:
        if trunk_params:
            raise AssertionError(
                f"heatmap finetune optimizer still has trunk params n={len(trunk_params)}"
            )
        if not new_params:
            raise AssertionError("heatmap finetune optimizer has no HeatmapHead params")
        optimizer = optimizer_cls([{"params": new_params, "lr": base_lr}], **extra)
        print(
            f"P1 heatmap-finetune optimizer: heatmap_heads n={len(new_params)} lr={base_lr}"
        )
        return optimizer
    if not trunk_params or not new_params:
        raise AssertionError("empty P1 optimizer group")
    optimizer = optimizer_cls(
        [
            {"params": trunk_params, "lr": trunk_lr},
            {"params": new_params, "lr": base_lr},
        ],
        **extra,
    )
    print(
        f"P1 optimizer groups: trunk_nonBN n={len(trunk_params)} lr={trunk_lr} ; "
        f"new_modules n={len(new_params)} lr={base_lr}"
    )
    return optimizer


def build_heatmap_targets(
    ego: Dict[str, Any],
    predictions: Dict[str, Dict[str, torch.Tensor]],
    use_drone_box_support: bool = False,
) -> Dict[str, torch.Tensor]:
    """``semantic_target`` per present agent on the 90x160 grid.

    When ``use_drone_box_support`` is True (TRAIN loop only), the drone
    heatmap target is the binary union of projected official GT boxes and
    SAM3 ``image_semantic_gts``. Vehicle/RSU always keep SAM3 only.
    Validation must pass False so val behavior stays production SAM3.
    """
    targets: Dict[str, torch.Tensor] = {}
    for agent_type, pred in predictions.items():
        logits = pred["heatmap_logits"]
        if use_drone_box_support and agent_type == "drone":
            target = build_drone_union_target(ego, tau=1)
            target = target.to(device=logits.device)
        else:
            cam_inputs = ego[agent_type]["batch_merged_cam_inputs"]
            target = build_semantic_target(cam_inputs, tau=1)
        if tuple(target.shape[-2:]) != tuple(logits.shape[-2:]):
            raise AssertionError(
                f"{agent_type} semantic_target {tuple(target.shape)} vs "
                f"heatmap_logits {tuple(logits.shape)}"
            )
        if tuple(target.shape[:1]) != tuple(logits.shape[:1]):
            raise AssertionError(
                f"{agent_type} target N={tuple(target.shape)} vs "
                f"heatmap_logits {tuple(logits.shape)}"
            )
        targets[agent_type] = target
    return targets


def build_depth_targets(
    ego: Dict[str, Any],
    predictions: Dict[str, Dict[str, torch.Tensor]],
    core_model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    """Class indices for vehicle/RSU; residual ``delta_gt`` for drone."""
    targets: Dict[str, torch.Tensor] = {}
    for agent_type, pred in predictions.items():
        cam_inputs = ego[agent_type]["batch_merged_cam_inputs"]
        imgs = cam_inputs["imgs"]
        if agent_type == "drone":
            camera_z_gt = extract_camera_z_gt(imgs)
            height = flatten_camera_world_z(cam_inputs["camera_world_z"], imgs)
            height = height.to(device=camera_z_gt.device, dtype=camera_z_gt.dtype)
            if int(height.shape[0]) != int(camera_z_gt.shape[0]):
                raise AssertionError(
                    f"drone height {tuple(height.shape)} vs z_gt N={int(camera_z_gt.shape[0])}"
                )
            delta_gt = camera_z_gt - height[:, None, None]
            if tuple(delta_gt.shape[-2:]) != tuple(pred["delta_pred"].shape[-2:]):
                raise AssertionError(
                    f"drone delta_gt {tuple(delta_gt.shape)} vs "
                    f"delta_pred {tuple(pred['delta_pred'].shape)}"
                )
            targets[agent_type] = delta_gt
            continue
        camencode = core_model.frontend.encoders[agent_type]
        target = build_depth_class_target(camencode, imgs)
        logits = pred["depth_logits"]
        if tuple(target.shape[-2:]) != tuple(logits.shape[-2:]):
            raise AssertionError(
                f"{agent_type} depth target {tuple(target.shape)} vs "
                f"depth_logits {tuple(logits.shape)}"
            )
        if int(target.max().item()) >= int(logits.shape[1]) or int(target.min().item()) < 0:
            raise AssertionError(
                f"{agent_type} depth target ids out of range "
                f"[{int(target.min())},{int(target.max())}] D={logits.shape[1]}"
            )
        targets[agent_type] = target
    return targets


def build_depth_valid_masks(
    ego: Dict[str, Any],
    predictions: Dict[str, Dict[str, torch.Tensor]],
    core_model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    """In-range GT depth masks for vehicle/RSU. Drone is skipped."""
    masks: Dict[str, torch.Tensor] = {}
    for agent_type in predictions:
        if agent_type == "drone":
            continue
        enc = core_model.frontend.encoders[agent_type]
        z = extract_camera_z_gt(ego[agent_type]["batch_merged_cam_inputs"]["imgs"])
        masks[agent_type] = depth_valid_mask(z, enc.d_min, enc.d_max)
    return masks


def compute_p1_metrics(
    ego: Dict[str, Any],
    predictions: Dict[str, Dict[str, torch.Tensor]],
    heatmap_targets: Dict[str, torch.Tensor],
    core_model: torch.nn.Module,
) -> Dict[str, float]:
    """Heatmap and depth diagnostics. No autograd."""
    with torch.no_grad():
        heatmap_items: List[Dict[str, float]] = []
        depth_items: List[Dict[str, float]] = []
        metrics: Dict[str, float] = {}
        for agent_type, pred in predictions.items():
            agent_heatmap = compute_heatmap_metrics(
                pred["heatmap_logits"].detach(), heatmap_targets[agent_type]
            )
            heatmap_items.append(agent_heatmap)
            for key, value in agent_heatmap.items():
                metrics[f"heatmap/{agent_type}/{key}"] = value
            cam_inputs = ego[agent_type]["batch_merged_cam_inputs"]
            camera_z_gt = extract_camera_z_gt(cam_inputs["imgs"])
            if tuple(camera_z_gt.shape[-2:]) != tuple(pred["depth_z_mean"].shape[-2:]):
                raise AssertionError(
                    f"{agent_type} camera_z_gt {tuple(camera_z_gt.shape)} vs "
                    f"depth_z_mean {tuple(pred['depth_z_mean'].shape)}"
                )
            foreground_mask = heatmap_targets[agent_type].ne(BACKGROUND_CLASS_ID)
            if agent_type == "drone":
                d_min = 6.0
                d_max = 150.0
            else:
                camencode = core_model.frontend.encoders[agent_type]
                d_min = float(camencode.d_min)
                d_max = float(camencode.d_max)
            agent_depth = compute_depth_metrics(
                depth_z_mean=pred["depth_z_mean"].detach(),
                camera_z_gt=camera_z_gt,
                d_min=d_min,
                d_max=d_max,
                foreground_mask=foreground_mask,
            )
            if agent_type == "drone" and "delta_pred" in pred:
                height = pred["camera_world_z"].detach().reshape(-1, 1, 1)
                delta_gt = camera_z_gt - height
                valid = depth_valid_mask(camera_z_gt, d_min, d_max) & foreground_mask
                if int(valid.sum().item()) > 0:
                    agent_depth["delta_mae_fg"] = float(
                        (pred["delta_pred"].detach()[valid] - delta_gt[valid])
                        .abs()
                        .mean()
                    )
                else:
                    agent_depth["delta_mae_fg"] = 0.0
            depth_items.append(agent_depth)
        for key, value in _mean_dicts(heatmap_items).items():
            metrics[f"heatmap/{key}"] = value
        for key, value in _mean_dicts(depth_items).items():
            metrics[f"depth/{key}"] = value
        return metrics


def _forward_loss_metrics(
    model: torch.nn.Module,
    ego: Dict[str, Any],
    semantic_criterion: GaussianP1SemanticLoss,
    depth_criterion: GaussianP1DepthLoss,
    scaler: Optional[amp.GradScaler],
    use_drone_box_support: bool = False,
    heatmap_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """One batch: joint predict, two targets, heatmap Focal + depth, metrics."""
    core_model = _unwrap_model(model)
    with amp.autocast(enabled=scaler is not None):
        predictions = model(ego)
        heatmap_targets = build_heatmap_targets(
            ego, predictions, use_drone_box_support=use_drone_box_support
        )
        heatmap_loss = semantic_criterion(predictions, heatmap_targets)
        if heatmap_only:
            depth_loss = heatmap_loss.new_zeros(())
            total_loss = heatmap_loss
        else:
            depth_targets = build_depth_targets(ego, predictions, core_model)
            depth_valid_masks = build_depth_valid_masks(ego, predictions, core_model)
            depth_loss = depth_criterion(
                predictions, depth_targets, heatmap_targets, depth_valid_masks
            )
            total_loss = heatmap_loss + depth_loss
    metrics = compute_p1_metrics(ego, predictions, heatmap_targets, core_model)
    if not heatmap_only:
        metrics.update(depth_criterion.loss_dict)
    return total_loss, heatmap_loss, depth_loss, metrics


def _log_metrics(
    writer: SummaryWriter,
    metrics: Dict[str, float],
    step: int,
    prefix: str,
) -> None:
    """Write diagnostic scalars. Separate from loss logging."""
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)


def validate_p1(
    model: torch.nn.Module,
    val_loader: DataLoader,
    semantic_criterion: GaussianP1SemanticLoss,
    depth_criterion: GaussianP1DepthLoss,
    epoch: int,
    device: torch.device,
    scaler: Optional[amp.GradScaler],
    heatmap_only: bool = False,
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
            total_loss, heatmap_loss, depth_loss, metrics = _forward_loss_metrics(
                model,
                ego,
                semantic_criterion,
                depth_criterion,
                scaler,
                heatmap_only=heatmap_only,
            )
            total_sum += float(total_loss.item())
            heatmap_sum += float(heatmap_loss.item())
            depth_sum += float(depth_loss.item())
            n_sample += 1
            for key, value in metrics.items():
                metric_sum[key] = metric_sum.get(key, 0.0) + value
    metric_mean = {key: value / max(n_sample, 1) for key, value in metric_sum.items()}
    return total_sum, heatmap_sum, depth_sum, n_sample, metric_mean


def main() -> None:
    """Train the joint R90 P1 frontend from yaml."""
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    hypes["tag"] = opt.tag
    heatmap_only = _heatmap_only(hypes)
    pretrained_ckpt = str(hypes.get("p1_finetune", {}).get("pretrained_ckpt") or "")
    save_quarters = bool(hypes.get("p1_finetune", {}).get("save_quarters", False))
    print("load from yaml file: ", opt.hypes_yaml)
    if heatmap_only:
        if opt.model_dir:
            raise ValueError(
                "heatmap finetune loads pretrained_ckpt from yaml into a NEW log dir. "
                "Do not pass --model_dir (that would load the original config.yaml "
                "and resume optimizer into the concat128 folder)."
            )
        if not pretrained_ckpt:
            raise ValueError("p1_finetune.heatmap_only requires p1_finetune.pretrained_ckpt")
        print("P1 mode: heatmap-only finetune (depth forward unused for loss)")
        print(f"P1 pretrained ckpt: {pretrained_ckpt}")
    else:
        print("P1 mode: joint heatmap + depth (single architecture)")
    print(
        "P1 experiment: TRAIN drone heatmap = GT-box OR SAM3; "
        "TRAIN night RGB = random L1–L2 lift + HL; "
        "TEST night RGB = L2_hl; "
        "TRAIN fog on 40% of non-night timestamps. "
        "Do not resume a pre-experiment checkpoint as the same run."
    )

    # Build the AirV2X file index before NCCL. parse_seq reads tens of
    # thousands of metadata.pkl files; doing that after the first ALLREDUCE
    # makes the NCCL watchdog time out while GPUs sit idle.
    print("Building datasets...")
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    val_dataset = build_dataset(hypes, visualize=False, train=False)

    multi_gpu_utils.init_distributed_mode(opt)
    main_process = is_main_process(opt)
    train_loader = setup_dataloader(train_dataset, hypes, opt, is_train=True)
    val_loader = setup_dataloader(val_dataset, hypes, opt, is_train=False)

    print("Creating model...")
    model = train_utils.create_model(hypes)
    if heatmap_only:
        apply_heatmap_finetune_freeze(model)
    total_params = sum(p.nelement() for p in model.parameters())
    trainable_params = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print_trainable_families(model)

    if opt.distributed:
        assert torch.cuda.is_available(), "Distributed training requires CUDA"
        device = torch.device(f"cuda:{opt.gpu}")
        torch.cuda.set_device(opt.gpu)
    else:
        if torch.cuda.is_available():
            gpu_id = int(opt.gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
        else:
            device = torch.device("cpu")

    model.to(device)
    if pretrained_ckpt:
        load_pretrained_weights(model, pretrained_ckpt, device)
        if heatmap_only:
            apply_heatmap_finetune_freeze(model)
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.gpu],
            output_device=opt.gpu,
            find_unused_parameters=bool(heatmap_only),
        )

    semantic_criterion, depth_criterion = build_p1_criteria(hypes)
    optimizer = setup_p1_optimizer(hypes, model)
    scaler = amp.GradScaler() if opt.amp and torch.cuda.is_available() else None
    if scaler:
        print("Using mixed precision training")
    scheduler = train_utils.setup_lr_schedular(
        hypes, optimizer, n_iter_per_epoch=len(train_loader)
    )

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
    print("Starting heatmap-only P1 finetune..." if heatmap_only else "Starting joint P1 training...")
    epochs = hypes["train_params"]["epoches"]
    n_iter = len(train_loader)
    quarter_marks = _quarter_save_marks(n_iter) if save_quarters else {}

    for epoch in range(init_epoch, epochs):
        if opt.distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_dataset, "set_fog_epoch"):
            train_dataset.set_fog_epoch(epoch)
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        model.train()
        _unwrap_model(model).frontend.assert_train_eval_state(True)

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            disable=not main_process,
        )
        for i, batch_data in pbar:
            if batch_data is None:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            ego = batch_data["ego"]
            ego["epoch"] = epoch
            total_loss, heatmap_loss, depth_loss, metrics = _forward_loss_metrics(
                model,
                ego,
                semantic_criterion,
                depth_criterion,
                scaler,
                use_drone_box_support=True,
                heatmap_only=heatmap_only,
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
                step = epoch * len(train_loader) + i
                heatmap_v = float(heatmap_loss.item())
                depth_v = float(depth_loss.item())
                total_v = float(total_loss.item())
                writer.add_scalar("Train/total_loss", total_v, step)
                writer.add_scalar("Train/heatmap_loss", heatmap_v, step)
                writer.add_scalar("Train/depth_loss", depth_v, step)
                for agent_key in (
                    "depth_loss_vehicle",
                    "depth_loss_rsu",
                    "depth_loss_drone",
                ):
                    if agent_key in metrics:
                        writer.add_scalar(f"Train/{agent_key}", metrics[agent_key], step)
                print_msg = (
                    "[epoch %d][%d/%d] || total: %.4f | focal: %.4f | depth: %.4f"
                    % (epoch, i + 1, len(train_loader), total_v, heatmap_v, depth_v)
                )
                pbar.set_description(print_msg)
                _log_metrics(writer, metrics, step, "Train")
                with open(os.path.join(saved_path, "train_loss.txt"), "a+") as handle:
                    handle.write(
                        f"Epoch[{epoch}], iter[{i}/{len(train_loader)}], "
                        f"total[{total_v:.4f}], heatmap[{heatmap_v:.4f}], "
                        f"depth[{depth_v:.4f}]\n"
                    )

            if save_quarters and opt.rank == 0:
                quarter = quarter_marks.get(i + 1)
                if quarter is not None:
                    q_name = f"net_epoch{epoch + 1}_q{quarter}.pth"
                    q_path = _save_p1_checkpoint(
                        saved_path,
                        q_name,
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                    )
                    print(f"Saved quarter checkpoint q{quarter}: {q_path}")
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
            local_total, local_hm, local_dep, local_cnt, extra_metrics = validate_p1(
                model,
                val_loader,
                semantic_criterion,
                depth_criterion,
                epoch,
                device,
                scaler,
                heatmap_only=heatmap_only,
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
                for key, value in extra_metrics.items():
                    print(f"  {key}: {value:.4f}")
                with open(os.path.join(saved_path, "validation_loss.txt"), "a+") as handle:
                    handle.write(
                        f"Epoch[{epoch}], total[{val_total:.4f}], "
                        f"heatmap[{val_hm:.4f}], depth[{val_dep:.4f}]\n"
                    )

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
        print(f"Joint P1 training finished. Checkpoints saved to {saved_path}")


if __name__ == "__main__":
    main()

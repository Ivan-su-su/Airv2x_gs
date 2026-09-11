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
from opencood.models.gaussian_modules_0822.image_frontend import (
    flatten_camera_world_z,
    p1_agent_predictions,
)
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
# TEMPORARY TEST→TRAIN mix helpers (delete together with p1_test_mix.py).
from opencood.tools.p1_test_mix import (
    apply_test_mix_to_hypes,
    background_heatmap_targets,
    batch_is_test_mix,
    dump_mix_manifest,
)
from opencood.tools.train import (
    _adapt_state_dict_for_model,
    _get_model_state_dict_for_save,
    is_main_process,
    resume_training_from_checkpoint,
    setup_dataloader,
    train_parser,
)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module when wrapped by DDP."""
    return model.module if hasattr(model, "module") else model


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


def _finetune_cfg(hypes: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``hypes['finetune']`` or an empty dict."""
    cfg = hypes.get("finetune") or {}
    return cfg if isinstance(cfg, dict) else {}


def _keep_drone_branch_param(name: str) -> bool:
    """True iff ``name`` belongs to the trainable drone P1 branch.

    Keeps drone CamEncode non-BN trunk / up1 / up2, HighResFusion, heatmap,
    height embedding, and residual depth. Drops EfficientNet BN, official
    unused heads, and every vehicle/RSU tensor.
    """
    if not (
        name.startswith("frontend.encoders.drone.")
        or name.startswith("highres.drone.")
        or name.startswith("heatmap_heads.drone.")
        or name.startswith("drone_height_embed.")
        or name.startswith("drone_delta_head.")
    ):
        return False
    if "image_head" in name:
        return False
    if "frontend.encoders.drone." in name and ".depth_head." in name:
        return False
    if "._conv_head." in name or "._fc." in name:
        return False
    if ".trunk." in name and ("_bn" in name or ".bn" in name):
        return False
    return True


def _keep_vehicle_branch_param(name: str) -> bool:
    """True iff ``name`` belongs to the trainable vehicle P1 branch.

    Keeps vehicle CamEncode non-BN trunk / up1 / up2, HighResFusion, heatmap,
    and categorical depth head. Drops EfficientNet BN, official unused heads,
    and every drone/RSU tensor.
    """
    if not (
        name.startswith("frontend.encoders.vehicle.")
        or name.startswith("highres.vehicle.")
        or name.startswith("heatmap_heads.vehicle.")
        or name.startswith("depth_heads.vehicle.")
    ):
        return False
    if "image_head" in name:
        return False
    if "frontend.encoders.vehicle." in name and ".depth_head." in name:
        return False
    if "._conv_head." in name or "._fc." in name:
        return False
    if ".trunk." in name and ("_bn" in name or ".bn" in name):
        return False
    return True


def freeze_except_drone_heatmap(model: torch.nn.Module) -> None:
    """Train only ``heatmap_heads.drone``; freeze every other parameter."""
    core = _unwrap_model(model)
    kept: List[str] = []
    for name, param in core.named_parameters():
        keep = name.startswith("heatmap_heads.drone")
        param.requires_grad = keep
        if keep:
            kept.append(name)
    if not kept:
        raise AssertionError("heatmap_heads.drone not found")
    print("Finetune trainable:", kept)


def freeze_except_drone_branch(model: torch.nn.Module) -> None:
    """Train the drone encoder + heatmap + residual depth; freeze vehicle/RSU."""
    core = _unwrap_model(model)
    kept: List[str] = []
    for name, param in core.named_parameters():
        keep = _keep_drone_branch_param(name)
        param.requires_grad = keep
        if keep:
            kept.append(name)
    if not kept:
        raise AssertionError("drone branch has no trainable params")
    print(f"Finetune drone branch trainable: {len(kept)} tensors")


def freeze_except_vehicle_branch(model: torch.nn.Module) -> None:
    """Train vehicle encoder + heatmap + depth; freeze drone/RSU."""
    core = _unwrap_model(model)
    kept: List[str] = []
    for name, param in core.named_parameters():
        keep = _keep_vehicle_branch_param(name)
        param.requires_grad = keep
        if keep:
            kept.append(name)
    if not kept:
        raise AssertionError("vehicle branch has no trainable params")
    print(f"Finetune vehicle branch trainable: {len(kept)} tensors")


def _keep_rsu_branch_param(name: str) -> bool:
    """True iff ``name`` belongs to the trainable RSU P1 branch.

    Keeps RSU CamEncode non-BN trunk / up1 / up2, HighResFusion, heatmap,
    and categorical depth head. Drops EfficientNet BN, official unused heads,
    and every vehicle/drone tensor. Heatmap IS trainable (full branch).
    """
    if not (
        name.startswith("frontend.encoders.rsu.")
        or name.startswith("highres.rsu.")
        or name.startswith("heatmap_heads.rsu.")
        or name.startswith("depth_heads.rsu.")
    ):
        return False
    if "image_head" in name:
        return False
    if "frontend.encoders.rsu." in name and ".depth_head." in name:
        return False
    if "._conv_head." in name or "._fc." in name:
        return False
    if ".trunk." in name and ("_bn" in name or ".bn" in name):
        return False
    return True


def freeze_except_rsu_branch(model: torch.nn.Module) -> None:
    """Train RSU encoder + heatmap + categorical depth; freeze vehicle/drone."""
    core = _unwrap_model(model)
    kept: List[str] = []
    for name, param in core.named_parameters():
        keep = _keep_rsu_branch_param(name)
        param.requires_grad = keep
        if keep:
            kept.append(name)
    if not kept:
        raise AssertionError("rsu branch has no trainable params")
    print(f"Finetune rsu branch trainable: {len(kept)} tensors")


def load_p1_weights(model: torch.nn.Module, path: str, device: torch.device) -> None:
    """Load ``model_state_dict`` only. Does not restore optimizer or epoch."""
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise KeyError(f"{path} has no model_state_dict")
    state = _adapt_state_dict_for_model(model, ckpt["model_state_dict"])
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"Loaded pretrained weights from {path} (epoch={ckpt.get('epoch')})")
    if missing:
        print("missing:", missing)
    if unexpected:
        print("unexpected:", unexpected)


def _quarter_save_iters(n_iter: int) -> Dict[int, int]:
    """Map 1-based iteration index to quarter 1..4."""
    hits: Dict[int, int] = {}
    for quarter in (1, 2, 3, 4):
        hits[max(1, (int(n_iter) * quarter) // 4)] = quarter
    return hits


def _save_p1_checkpoint(
    saved_path: str,
    filename: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[amp.GradScaler],
) -> None:
    """Write one checkpoint dict under ``saved_path/filename``."""
    save_dict = {
        "epoch": epoch,
        "model_state_dict": _get_model_state_dict_for_save(model),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        save_dict["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        save_dict["scaler_state_dict"] = scaler.state_dict()
    torch.save(save_dict, os.path.join(saved_path, filename))
    print(f"saved {filename}")


def print_trainable_families(
    model: torch.nn.Module,
    required: bool = True,
    drone_branch: bool = False,
    vehicle_branch: bool = False,
    rsu_branch: bool = False,
) -> None:
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
    if required:
        required_families = (
            "trunk",
            "up1",
            "up2",
            "highres",
            "heatmap_head",
            "depth_head",
            "drone_height_embed",
            "drone_delta_head",
        )
        for family in required_families:
            if not families[family]:
                raise AssertionError(f"missing trainable family: {family}")
    elif drone_branch:
        leaked = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and not _keep_drone_branch_param(name)
        ]
        if leaked:
            raise AssertionError(f"drone-branch finetune leaked trainable: {leaked}")
        for family in (
            "trunk",
            "up1",
            "up2",
            "highres",
            "heatmap_head",
            "drone_height_embed",
            "drone_delta_head",
        ):
            if not families[family]:
                raise AssertionError(f"missing drone-branch family: {family}")
        veh_rsu = [
            name
            for name in families["heatmap_head"] + families["highres"]
            if ".vehicle." in name or ".rsu." in name
        ]
        if veh_rsu:
            raise AssertionError(f"vehicle/RSU still trainable: {veh_rsu}")
    elif vehicle_branch:
        leaked = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and not _keep_vehicle_branch_param(name)
        ]
        if leaked:
            raise AssertionError(f"vehicle-branch finetune leaked trainable: {leaked}")
        for family in ("trunk", "up1", "up2", "highres", "heatmap_head", "depth_head"):
            if not families[family]:
                raise AssertionError(f"missing vehicle-branch family: {family}")
        if families["drone_height_embed"] or families["drone_delta_head"]:
            raise AssertionError("drone residual modules still trainable")
        non_veh = [
            name
            for name in (
                families["heatmap_head"]
                + families["highres"]
                + families["depth_head"]
                + families["trunk"]
                + families["up1"]
                + families["up2"]
            )
            if ".drone." in name or ".rsu." in name
        ]
        if non_veh:
            raise AssertionError(f"drone/RSU still trainable: {non_veh}")
    elif rsu_branch:
        leaked = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and not _keep_rsu_branch_param(name)
        ]
        if leaked:
            raise AssertionError(f"rsu-branch finetune leaked trainable: {leaked}")
        for family in ("trunk", "up1", "up2", "highres", "heatmap_head", "depth_head"):
            if not families[family]:
                raise AssertionError(f"missing rsu-branch family: {family}")
        if families["drone_height_embed"] or families["drone_delta_head"]:
            raise AssertionError("drone residual modules still trainable")
        non_rsu = [
            name
            for name in (
                families["heatmap_head"]
                + families["highres"]
                + families["depth_head"]
                + families["trunk"]
                + families["up1"]
                + families["up2"]
            )
            if ".drone." in name or ".vehicle." in name
        ]
        if non_rsu:
            raise AssertionError(f"drone/vehicle still trainable: {non_rsu}")
    else:
        extra = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and not name.startswith("heatmap_heads.drone")
        ]
        if extra:
            raise AssertionError(f"drone-heatmap finetune leaked trainable: {extra}")
        if not families["heatmap_head"]:
            raise AssertionError("drone heatmap head is not trainable")
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
    extra = dict(opt_cfg.get("args") or {})
    if _finetune_cfg(hypes).get("drone_heatmap_only"):
        params = [param for param in model.parameters() if param.requires_grad]
        if not params:
            raise AssertionError("no trainable params for drone heatmap finetune")
        optimizer = optimizer_cls(params, lr=base_lr, **extra)
        print(f"P1 optimizer drone-heatmap-only n={len(params)} lr={base_lr}")
        return optimizer
    trunk_lr = 0.1 * base_lr
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


def build_camera_z_gts(
    ego: Dict[str, Any],
    predictions: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Continuous GT camera-z for vehicle/RSU. Drone is skipped.

    Used by the depth loss for auxiliary mean supervision and RSU far-range
    detection. Mirrors :func:`build_depth_valid_masks` without the mask step.
    """
    z_gts: Dict[str, torch.Tensor] = {}
    for agent_type in predictions:
        if agent_type == "drone":
            continue
        imgs = ego[agent_type]["batch_merged_cam_inputs"]["imgs"]
        z_gts[agent_type] = extract_camera_z_gt(imgs)
    return z_gts


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
    agent_only: Optional[str] = None,
    skip_depth: bool = False,
    skip_heatmap: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """One batch: joint predict, two targets, heatmap Focal + depth, metrics.

    When ``skip_heatmap`` (TEST-mix batch), heatmap targets are forced to
    all-background so the Focal loss is identically zero: depth-only grad.
    """
    core_model = _unwrap_model(model)
    with amp.autocast(enabled=scaler is not None):
        predictions = p1_agent_predictions(model(ego))
        if agent_only is not None:
            if agent_only not in predictions:
                zero = sum(
                    param.sum() * 0.0
                    for param in model.parameters()
                    if param.requires_grad
                )
                empty: Dict[str, float] = {}
                return zero, zero, zero, empty
            predictions = {agent_only: predictions[agent_only]}
        if skip_heatmap:
            # TEST frames carry no SAM3 ``*_seg.bin``; all-background targets
            # make heatmap Focal exactly 0, so no heatmap gradient leaks in.
            heatmap_targets = background_heatmap_targets(predictions)
            first_logits = next(iter(predictions.values()))["heatmap_logits"]
            heatmap_loss = first_logits.sum() * 0.0
            depth_targets = build_depth_targets(ego, predictions, core_model)
            depth_valid_masks = build_depth_valid_masks(ego, predictions, core_model)
            camera_z_gts = build_camera_z_gts(ego, predictions)
            depth_loss = depth_criterion(
                predictions,
                depth_targets,
                heatmap_targets,
                depth_valid_masks,
                camera_z_gts=camera_z_gts,
            )
            total_loss = heatmap_loss + depth_loss
        else:
            heatmap_targets = build_heatmap_targets(
                ego, predictions, use_drone_box_support=use_drone_box_support
            )
            if skip_depth:
                heatmap_loss = semantic_criterion(predictions, heatmap_targets)
                depth_loss = heatmap_loss.detach() * 0.0
                total_loss = heatmap_loss
            else:
                depth_targets = build_depth_targets(ego, predictions, core_model)
                depth_valid_masks = build_depth_valid_masks(ego, predictions, core_model)
                camera_z_gts = build_camera_z_gts(ego, predictions)
                heatmap_loss = semantic_criterion(predictions, heatmap_targets)
                depth_loss = depth_criterion(
                    predictions,
                    depth_targets,
                    heatmap_targets,
                    depth_valid_masks,
                    camera_z_gts=camera_z_gts,
                )
                total_loss = heatmap_loss + depth_loss
    metrics = compute_p1_metrics(ego, predictions, heatmap_targets, core_model)
    if not skip_depth:
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
                model, ego, semantic_criterion, depth_criterion, scaler
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
    finetune = _finetune_cfg(hypes)
    drone_hm = bool(finetune.get("drone_heatmap_only"))
    drone_branch = bool(finetune.get("drone_branch"))
    vehicle_branch = bool(finetune.get("vehicle_branch"))
    rsu_branch = bool(finetune.get("rsu_branch"))
    if (
        sum(bool(x) for x in (drone_hm, drone_branch, vehicle_branch, rsu_branch))
        > 1
    ):
        raise ValueError(
            "finetune flags are mutually exclusive: "
            "drone_heatmap_only / drone_branch / vehicle_branch / rsu_branch"
        )
    branch_only = drone_hm or drone_branch or vehicle_branch or rsu_branch
    if drone_hm or drone_branch:
        agent_only: Optional[str] = "drone"
    elif vehicle_branch:
        agent_only = "vehicle"
    elif rsu_branch:
        agent_only = "rsu"
    else:
        agent_only = None
    save_quarters = bool(finetune.get("save_quarters"))
    pretrained = finetune.get("pretrained")
    # TEMPORARY TEST→TRAIN mix (delete with p1_test_mix.py): when
    # ``test_mix_keep_frac`` is set, extra_train_scenes is replaced by the
    # first frac of every TEST scene; those batches train depth only.
    test_mix_on = finetune.get("test_mix_keep_frac") is not None
    skip_heatmap_on_test_mix = bool(finetune.get("skip_heatmap_on_test_mix"))
    if test_mix_on:
        apply_test_mix_to_hypes(hypes)
        if not skip_heatmap_on_test_mix:
            print(
                "[test-mix] WARNING: skip_heatmap_on_test_mix=false — TEST "
                "frames have no SAM3 labels, heatmap WILL train on empty GT."
            )
    print("load from yaml file: ", opt.hypes_yaml)
    if drone_hm:
        print("P1 mode: drone heatmap finetune (all other modules frozen)")
        print(
            "Mix TEST fog 2025_05_05_23_26_24 (first 70%, stride 4, 219 frames). "
            "No synthetic fog on those frames. Loss = drone heatmap only."
        )
    elif drone_branch:
        print("P1 mode: drone-branch finetune (vehicle/RSU frozen)")
        print(
            "Train drone encoder + heatmap + residual depth. "
            "Mix TEST fog 2025_05_05_23_26_24 (first 70%, stride 4, 219 frames). "
            "Loss = drone heatmap + drone depth."
        )
    elif vehicle_branch:
        print("P1 mode: vehicle-branch finetune (drone/RSU frozen)")
        print(
            "Train vehicle encoder + heatmap + categorical depth from "
            "concat128 pretrained. Loss = vehicle heatmap + vehicle depth."
        )
    elif rsu_branch:
        print("P1 mode: rsu-branch finetune (vehicle/drone frozen)")
        print(
            "Train RSU encoder + heatmap + categorical depth from "
            "concat128 pretrained. Heatmap IS trained (full branch). "
            "Loss = rsu heatmap + rsu depth (+ optional aux mean / far-weight)."
        )
    else:
        print("P1 mode: joint heatmap + depth (single architecture)")
        print(
            "P1 experiment: TRAIN drone heatmap = GT-box OR SAM3; "
            "L2_hl RGB on TRAIN 2025_05_06_10_01_50 and TEST 2025_05_10_19_54_35; "
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
    if pretrained and not opt.model_dir:
        load_p1_weights(model, str(pretrained), device)
    if drone_hm:
        freeze_except_drone_heatmap(model)
    elif drone_branch:
        freeze_except_drone_branch(model)
    elif vehicle_branch:
        freeze_except_vehicle_branch(model)
    elif rsu_branch:
        freeze_except_rsu_branch(model)
    total_params = sum(p.nelement() for p in model.parameters())
    trainable_params = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print_trainable_families(
        model,
        required=not branch_only,
        drone_branch=drone_branch,
        vehicle_branch=vehicle_branch,
        rsu_branch=rsu_branch,
    )

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.gpu],
            output_device=opt.gpu,
            # branch finetune freezes vehicle/drone via requires_grad=False, so
            # DDP never sees them -> there are no unused parameters. The flag
            # was causing "marked ready twice" with the aux-mean depth path.
            find_unused_parameters=False,
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
    # TEMPORARY (delete with p1_test_mix.py): record which TEST timestamps
    # entered TRAIN so TEST eval can hold them out.
    if test_mix_on and main_process:
        dump_mix_manifest(train_dataset, saved_path)
    print("Starting joint P1 training...")
    epochs = hypes["train_params"]["epoches"]

    for epoch in range(init_epoch, epochs):
        if opt.distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_dataset, "set_fog_epoch"):
            train_dataset.set_fog_epoch(epoch)
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        model.train()
        if not branch_only:
            _unwrap_model(model).frontend.assert_train_eval_state(True)

        n_iter = len(train_loader)
        quarter_hits = _quarter_save_iters(n_iter) if save_quarters else {}
        pbar = tqdm(
            enumerate(train_loader),
            total=n_iter,
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
            # TEMPORARY (delete with p1_test_mix.py): depth-only on TEST mix.
            is_test_mix = test_mix_on and batch_is_test_mix(ego)
            total_loss, heatmap_loss, depth_loss, metrics = _forward_loss_metrics(
                model,
                ego,
                semantic_criterion,
                depth_criterion,
                scaler,
                use_drone_box_support=True,
                agent_only=agent_only,
                skip_depth=drone_hm,
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
                for agent_key in (
                    "depth_loss_vehicle",
                    "depth_loss_rsu",
                    "depth_loss_drone",
                ):
                    if agent_key in metrics:
                        writer.add_scalar(f"Train/{agent_key}", metrics[agent_key], step)
                print_msg = (
                    "[epoch %d][%d/%d] || total: %.4f | focal: %.4f | depth: %.4f"
                    % (epoch, i + 1, n_iter, total_v, heatmap_v, depth_v)
                )
                pbar.set_description(print_msg)
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

        if (not branch_only) and epoch % hypes["train_params"]["eval_freq"] == 0:
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

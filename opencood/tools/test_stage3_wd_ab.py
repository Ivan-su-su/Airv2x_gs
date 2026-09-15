#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage3 weight-decay A/B causal test for AirV2X Gaussian 0911version.

Purpose
-------
Test whether the observed Stage3 collapse
    q/k/v/out/FFN weights -> ~0
    fusion_gamma -> ~0
is primarily caused by weight decay feedback or by the identity shortcut /
lack of task utility.

This script intentionally runs TWO INDEPENDENT single-GPU experiments in
parallel, instead of one 2-GPU DDP experiment:

    GPU 4: baseline
        - same optimizer grouping as production
        - Stage3 core 2-D weights keep the configured weight_decay

    GPU 5: no_stage3_core_wd
        - everything identical
        - ONLY these Stage3 2-D weights get weight_decay=0:
            stage3.split_mlp.*
            stage3.self_blocks.*
            stage3.cross_blocks.*
            stage3.fusion_mlp.*
        - Stage3 adapter MLP is deliberately left unchanged

Using one GPU per variant is the clean causal comparison: both runs see the
same seed, same shuffled sample order, same config, same LR/scheduler and same
gradient clipping policy.

No production source file is modified.

Usage
-----
python opencood/tools/test_stage3_wd_ab.py \
    --config /path/to/config.yaml \
    --gpu-a 4 --gpu-b 5 \
    --epochs 2

Optional quick test:
python opencood/tools/test_stage3_wd_ab.py \
    --config /path/to/config.yaml \
    --gpu-a 4 --gpu-b 5 \
    --epochs 1 --steps-per-epoch 300

Outputs
-------
<out-dir>/
    baseline_gpu4.json
    no_stage3_core_wd_gpu5.json
    stage3_wd_ab_summary.txt

Notes
-----
* Starts from fresh model initialization (while the model may internally load
  its configured frozen/pretrained P1 frontend, exactly as normal creation).
* Does NOT resume net_epoch*.pth because we need to observe how collapse starts.
* Default --clip-norm auto reads the current local opencood/tools/train.py and
  uses its numeric max_norm if found. Both A/B runs use the same value.
* "wd_term_norm = weight_decay * ||W||" is a raw L2 regularization diagnostic.
  PyTorch Adam applies weight decay through the gradient before Adam moments,
  so this is evidence about relative pressure, not the exact final update norm.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage3 WD A/B causal test")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--gpu-a", default=4, type=int)
    p.add_argument("--gpu-b", default=5, type=int)
    p.add_argument("--epochs", default=2, type=int)
    p.add_argument(
        "--steps-per-epoch",
        default=0,
        type=int,
        help="0 means the full training epoch",
    )
    p.add_argument("--workers", default=0, type=int)
    p.add_argument("--seed", default=20260911, type=int)
    p.add_argument("--log-every", default=100, type=int)
    p.add_argument(
        "--clip-norm",
        default="auto",
        type=str,
        help="'auto', 'none', or a numeric max_norm; both variants use the same setting",
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument(
        "--out-dir",
        default="",
        type=str,
        help="default: <config-dir>/stage3_wd_ab_<timestamp>",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Determinism / JSON helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism is more important than throughput for this causal A/B.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_jsonable(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def fmt(x: Any, n: int = 4) -> str:
    if x is None:
        return "N/A"
    try:
        f = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(f):
        return str(f)
    if abs(f) >= 10000 or (abs(f) > 0 and abs(f) < 1e-4):
        return f"{f:.3e}"
    return f"{f:.{n}f}"


# ---------------------------------------------------------------------------
# Match current production training settings
# ---------------------------------------------------------------------------

def detect_clip_norm(train_py: Path) -> Optional[float]:
    """
    Read the current LOCAL train.py, not GitHub, because the user may have
    locally changed max_norm while keeping the same branch.
    """
    try:
        txt = train_py.read_text(encoding="utf-8")
    except Exception:
        return None

    # Handles:
    # clip_grad_norm_(model.parameters(), max_norm=1.0)
    matches = re.findall(
        r"clip_grad_norm_\s*\([^;]*?max_norm\s*=\s*([0-9eE+\-.]+)",
        txt,
        flags=re.S,
    )
    values: List[float] = []
    for raw in matches:
        try:
            values.append(float(raw))
        except Exception:
            pass
    if not values:
        return None
    # train.py normally contains the AMP and non-AMP copy with the same value.
    return float(values[0])


def resolve_clip_norm(raw: str) -> Optional[float]:
    s = raw.strip().lower()
    if s == "none":
        return None
    if s == "auto":
        return detect_clip_norm(ROOT / "opencood/tools/train.py")
    value = float(s)
    return value if value > 0 else None


# ---------------------------------------------------------------------------
# Parameter grouping
# ---------------------------------------------------------------------------

STAGE3_CORE_PREFIXES = (
    "stage3.split_mlp.",
    "stage3.self_blocks.",
    "stage3.cross_blocks.",
    "stage3.fusion_mlp.",
)


def canonical_name(name: str) -> str:
    return name[7:] if name.startswith("module.") else name


def is_stage3_core_matrix(name: str, p: torch.nn.Parameter) -> bool:
    name = canonical_name(name)
    return p.ndim > 1 and any(name.startswith(pref) for pref in STAGE3_CORE_PREFIXES)


def build_ab_optimizer(
    hypes: Mapping[str, Any],
    model: torch.nn.Module,
    disable_stage3_core_wd: bool,
) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
    """
    Reproduce train_utils.setup_optimizer grouping, but split ordinary matrix
    weights into:
      (a) Stage3 core matrices
      (b) all other matrices
    so A/B group structure stays the same and only Stage3-core weight_decay
    changes.
    """
    cfg = hypes["optimizer"]
    opt_cls = getattr(torch.optim, cfg["core_method"], None)
    if opt_cls is None:
        raise ValueError(f"Unsupported optimizer: {cfg['core_method']}")

    extra = dict(cfg.get("args") or {})
    base_wd = float(extra.pop("weight_decay", 0.0))

    other_decay: List[torch.nn.Parameter] = []
    stage3_core: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    names = {"other_decay": [], "stage3_core": [], "no_decay": []}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        clean = canonical_name(name)
        if p.ndim <= 1 or clean.endswith(".bias"):
            no_decay.append(p)
            names["no_decay"].append(clean)
        elif is_stage3_core_matrix(clean, p):
            stage3_core.append(p)
            names["stage3_core"].append(clean)
        else:
            other_decay.append(p)
            names["other_decay"].append(clean)

    groups = []
    if other_decay:
        groups.append(
            {
                "params": other_decay,
                "weight_decay": base_wd,
                "ab_group": "other_decay",
            }
        )
    if stage3_core:
        groups.append(
            {
                "params": stage3_core,
                "weight_decay": 0.0 if disable_stage3_core_wd else base_wd,
                "ab_group": "stage3_core",
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "weight_decay": 0.0,
                "ab_group": "no_decay",
            }
        )

    optimizer = opt_cls(groups, lr=float(cfg["lr"]), **extra)
    meta = {
        "optimizer": cfg["core_method"],
        "lr": float(cfg["lr"]),
        "base_weight_decay": base_wd,
        "stage3_core_weight_decay": 0.0 if disable_stage3_core_wd else base_wd,
        "n_other_decay_tensors": len(other_decay),
        "n_stage3_core_tensors": len(stage3_core),
        "n_no_decay_tensors": len(no_decay),
        "stage3_core_names": names["stage3_core"],
    }
    return optimizer, meta


# ---------------------------------------------------------------------------
# Stage3 metrics
# ---------------------------------------------------------------------------

WATCH_SUFFIXES = (
    "stage3.split_mlp.weight",
    "stage3.self_blocks.0.q_proj.weight",
    "stage3.self_blocks.0.k_proj.weight",
    "stage3.self_blocks.0.v_proj.weight",
    "stage3.self_blocks.0.out_proj.weight",
    "stage3.self_blocks.0.ffn.0.weight",
    "stage3.self_blocks.0.ffn.2.weight",
    "stage3.cross_blocks.0.q_proj.weight",
    "stage3.cross_blocks.0.k_proj.weight",
    "stage3.cross_blocks.0.v_proj.weight",
    "stage3.cross_blocks.0.out_proj.weight",
    "stage3.cross_blocks.0.ffn.0.weight",
    "stage3.cross_blocks.0.ffn.2.weight",
    "stage3.fusion_mlp.weight",
    "stage3.fusion_gamma",
    "stage3.adapter.gamma",
)


def named_params(model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    return {canonical_name(n): p for n, p in model.named_parameters()}


def stage3_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(named_params(model).items()):
        if not name.startswith("stage3."):
            continue
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def vector_l2(t: Optional[torch.Tensor]) -> Optional[float]:
    if t is None:
        return None
    return float(torch.linalg.vector_norm(t.detach().float()))


def cos_w_grad(p: torch.nn.Parameter) -> Optional[float]:
    if p.grad is None:
        return None
    w = p.detach().float().reshape(-1)
    g = p.grad.detach().float().reshape(-1)
    if w.numel() == 0 or g.numel() == 0:
        return None
    denom = torch.linalg.vector_norm(w) * torch.linalg.vector_norm(g)
    if float(denom) <= 1e-30:
        return None
    return float(torch.dot(w, g) / denom)


def total_grad_l2(model: torch.nn.Module) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        gf = torch.where(torch.isfinite(g), g, torch.zeros_like(g))
        sq += float((gf * gf).sum())
    return math.sqrt(max(sq, 0.0))


def stage3_grad_l2(model: torch.nn.Module) -> float:
    sq = 0.0
    for name, p in model.named_parameters():
        if not canonical_name(name).startswith("stage3.") or p.grad is None:
            continue
        g = p.grad.detach().float()
        gf = torch.where(torch.isfinite(g), g, torch.zeros_like(g))
        sq += float((gf * gf).sum())
    return math.sqrt(max(sq, 0.0))


def tensor_stats(x: Optional[torch.Tensor]) -> Dict[str, Optional[float]]:
    if x is None:
        return {"mean": None, "median": None, "mean_abs": None, "max_abs": None}
    f = x.detach().float().reshape(-1)
    f = f[torch.isfinite(f)]
    if f.numel() == 0:
        return {"mean": None, "median": None, "mean_abs": None, "max_abs": None}
    return {
        "mean": float(f.mean()),
        "median": float(f.median()),
        "mean_abs": float(f.abs().mean()),
        "max_abs": float(f.abs().max()),
    }


def snapshot_stage3(
    model: torch.nn.Module,
    initial_norms: Mapping[str, float],
    base_wd: float,
    variant_stage3_wd: float,
    epoch: int,
    step: int,
    global_step: int,
    loss_value: Optional[float] = None,
    criterion_loss_dict: Optional[Mapping[str, Any]] = None,
    pre_clip_grad_norm: Optional[float] = None,
    clip_factor: Optional[float] = None,
) -> Dict[str, Any]:
    params = named_params(model)
    rec: Dict[str, Any] = {
        "epoch": int(epoch),
        "step": int(step),
        "global_step": int(global_step),
        "loss": loss_value,
        "criterion_loss_dict": dict(criterion_loss_dict or {}),
        "pre_clip_grad_norm": pre_clip_grad_norm,
        "clip_factor": clip_factor,
        "stage3_grad_l2": stage3_grad_l2(model),
        "weights": {},
    }

    for name in WATCH_SUFFIXES:
        p = params.get(name)
        if p is None:
            continue
        pn = vector_l2(p)
        gn = vector_l2(p.grad) if p.grad is not None else None
        init_n = initial_norms.get(name)
        actual_wd = (
            variant_stage3_wd
            if is_stage3_core_matrix(name, p)
            else (0.0 if (p.ndim <= 1 or name.endswith(".bias")) else base_wd)
        )
        counterfactual_decay = base_wd * (pn or 0.0)
        rec["weights"][name] = {
            "param_l2": pn,
            "init_l2": init_n,
            "retention": (pn / init_n) if (pn is not None and init_n and init_n > 0) else None,
            "task_grad_l2": gn,
            "actual_weight_decay": actual_wd,
            "actual_raw_decay_term_l2": actual_wd * (pn or 0.0),
            "counterfactual_base_wd_term_l2": counterfactual_decay,
            "task_grad_to_base_wd_term": (
                gn / counterfactual_decay
                if gn is not None and counterfactual_decay > 0
                else None
            ),
            "cos_weight_task_grad": cos_w_grad(p),
        }

    fg = params.get("stage3.fusion_gamma")
    if fg is not None:
        rec["fusion_gamma"] = tensor_stats(fg)
        if fg.grad is not None:
            rec["fusion_gamma_grad"] = tensor_stats(fg.grad)
            g = fg.grad.detach().float()
            rec["fusion_gamma_grad_positive_frac"] = float((g > 0).float().mean())
            rec["fusion_gamma_grad_negative_frac"] = float((g < 0).float().mean())

    ag = params.get("stage3.adapter.gamma")
    if ag is not None:
        rec["adapter_gamma"] = tensor_stats(ag)

    return rec


def initial_norm_snapshot(model: torch.nn.Module) -> Dict[str, float]:
    params = named_params(model)
    out = {}
    for name in WATCH_SUFFIXES:
        if name in params:
            out[name] = float(torch.linalg.vector_norm(params[name].detach().float()))
    return out


# ---------------------------------------------------------------------------
# One A/B worker
# ---------------------------------------------------------------------------

def make_loader(
    dataset: Any,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init_fn(worker_id: int) -> None:
        s = seed + worker_id
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    kwargs = dict(
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=dataset.collate_batch_train,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )
    if workers > 0:
        kwargs["worker_init_fn"] = worker_init_fn
        kwargs["prefetch_factor"] = 1
    return DataLoader(dataset, **kwargs)


def run_variant(
    variant: str,
    gpu: int,
    config_path: str,
    epochs: int,
    steps_per_epoch: int,
    workers: int,
    seed: int,
    log_every: int,
    clip_norm: Optional[float],
    use_amp: bool,
    out_json: str,
) -> None:
    result: Dict[str, Any] = {
        "variant": variant,
        "gpu": int(gpu),
        "status": "STARTED",
    }
    out_path = Path(out_json)
    try:
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
        seed_everything(seed)

        hypes = yaml_utils.load_yaml(config_path, None)
        hypes = copy.deepcopy(hypes)

        # Dataset construction may consume RNG. Reset model RNG immediately
        # before model creation so A and B start from identical model weights.
        dataset = build_dataset(hypes, visualize=False, train=True)
        loader = make_loader(
            dataset,
            batch_size=int(hypes["train_params"]["batch_size"]),
            workers=int(workers),
            seed=seed + 17,
        )

        seed_everything(seed + 101)
        model = train_utils.create_model(hypes).to(device)

        init_hash = stage3_hash(model)
        init_norms = initial_norm_snapshot(model)

        criterion = train_utils.create_loss(hypes)
        disable = variant == "no_stage3_core_wd"
        optimizer, opt_meta = build_ab_optimizer(
            hypes, model, disable_stage3_core_wd=disable
        )
        scheduler = train_utils.setup_lr_schedular(
            hypes, optimizer, n_iter_per_epoch=len(loader)
        )
        scaler = amp.GradScaler() if use_amp else None

        result.update(
            {
                "status": "RUNNING",
                "config": str(config_path),
                "device": str(device),
                "seed": int(seed),
                "stage3_initial_hash": init_hash,
                "initial_norms": init_norms,
                "optimizer": opt_meta,
                "clip_norm": clip_norm,
                "epochs": int(epochs),
                "steps_per_epoch": int(steps_per_epoch),
                "history": [],
            }
        )

        base_wd = float(opt_meta["base_weight_decay"])
        stage3_wd = float(opt_meta["stage3_core_weight_decay"])

        # Initial state before any training update.
        result["history"].append(
            snapshot_stage3(
                model,
                init_norms,
                base_wd,
                stage3_wd,
                epoch=-1,
                step=-1,
                global_step=0,
            )
        )

        global_step = 0
        epoch_loss: List[float] = []
        t0 = time.time()

        for epoch in range(int(epochs)):
            model.train()
            losses_this_epoch: List[float] = []
            for i, batch_data in enumerate(loader):
                if batch_data is None:
                    continue
                if steps_per_epoch > 0 and i >= steps_per_epoch:
                    break

                batch_data = train_utils.to_device(batch_data, device)
                batch_data["ego"]["epoch"] = epoch

                optimizer.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)

                with amp.autocast(enabled=scaler is not None):
                    output = model(batch_data["ego"])
                    loss = criterion(output, batch_data["ego"]["label_dict"])

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                pre_clip = total_grad_l2(model)
                if clip_norm is not None:
                    clip_factor = min(1.0, clip_norm / (pre_clip + 1e-12))
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=float(clip_norm)
                    )
                else:
                    clip_factor = 1.0

                # Snapshot BEFORE optimizer.step so task gradients are visible.
                should_log = (
                    global_step == 0
                    or ((global_step + 1) % max(int(log_every), 1) == 0)
                )
                if should_log:
                    result["history"].append(
                        snapshot_stage3(
                            model,
                            init_norms,
                            base_wd,
                            stage3_wd,
                            epoch=epoch,
                            step=i,
                            global_step=global_step + 1,
                            loss_value=float(loss.detach()),
                            criterion_loss_dict=getattr(criterion, "loss_dict", {}),
                            pre_clip_grad_norm=pre_clip,
                            clip_factor=clip_factor,
                        )
                    )

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                losses_this_epoch.append(float(loss.detach()))
                global_step += 1

            scheduler.step(epoch)

            # End-of-epoch post-update snapshot. No gradient is expected here.
            result["history"].append(
                snapshot_stage3(
                    model,
                    init_norms,
                    base_wd,
                    stage3_wd,
                    epoch=epoch,
                    step=-2,
                    global_step=global_step,
                    loss_value=(
                        float(np.mean(losses_this_epoch))
                        if losses_this_epoch else None
                    ),
                    criterion_loss_dict=getattr(criterion, "loss_dict", {}),
                )
            )
            epoch_loss.append(
                float(np.mean(losses_this_epoch))
                if losses_this_epoch else float("nan")
            )

        result["epoch_loss_mean"] = epoch_loss
        result["final_norms"] = {
            name: float(torch.linalg.vector_norm(p.detach().float()))
            for name, p in named_params(model).items()
            if name in WATCH_SUFFIXES
        }
        result["final_stage3_hash"] = stage3_hash(model)
        result["runtime_sec"] = time.time() - t0
        result["status"] = "OK"
        dump_json(out_path, result)

    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        dump_json(out_path, result)
        raise


# ---------------------------------------------------------------------------
# Parent-side automated comparison
# ---------------------------------------------------------------------------

def final_history_row(run: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    hist = run.get("history", [])
    return hist[-1] if hist else None


def retention(run: Mapping[str, Any], name: str) -> Optional[float]:
    row = final_history_row(run)
    if not row:
        return None
    w = row.get("weights", {}).get(name, {})
    return w.get("retention")


def gamma_abs(run: Mapping[str, Any]) -> Optional[float]:
    row = final_history_row(run)
    if not row:
        return None
    return row.get("fusion_gamma", {}).get("mean_abs")


def mean_loss(run: Mapping[str, Any]) -> Optional[float]:
    vals = [
        float(v)
        for v in run.get("epoch_loss_mean", [])
        if v is not None and math.isfinite(float(v))
    ]
    return float(np.mean(vals)) if vals else None


def conclusion_from_ab(a: Mapping[str, Any], b: Mapping[str, Any]) -> Tuple[str, List[str]]:
    qn = "stage3.self_blocks.0.q_proj.weight"
    cqn = "stage3.cross_blocks.0.q_proj.weight"
    a_q, b_q = retention(a, qn), retention(b, qn)
    a_cq, b_cq = retention(a, cqn), retention(b, cqn)
    a_g, b_g = gamma_abs(a), gamma_abs(b)

    notes: List[str] = []
    notes.append(
        f"self q retention: baseline={fmt(a_q)}, no_stage3_wd={fmt(b_q)}"
    )
    notes.append(
        f"cross q retention: baseline={fmt(a_cq)}, no_stage3_wd={fmt(b_cq)}"
    )
    notes.append(
        f"|fusion_gamma| mean: baseline={fmt(a_g)}, no_stage3_wd={fmt(b_g)}"
    )
    notes.append(
        f"mean train loss: baseline={fmt(mean_loss(a))}, no_stage3_wd={fmt(mean_loss(b))}"
    )

    collapse_a = (
        a_q is not None and a_cq is not None
        and a_q < 1e-2 and a_cq < 1e-2
    )
    preserve_b = (
        b_q is not None and b_cq is not None
        and b_q > 0.1 and b_cq > 0.1
    )
    gamma_dead_a = a_g is not None and a_g < 0.01
    gamma_dead_b = b_g is not None and b_g < 0.01

    if collapse_a and preserve_b and not gamma_dead_b:
        label = "WEIGHT_DECAY_GATE_FEEDBACK_STRONGLY_IMPLICATED"
        notes.append(
            "Removing WD only from Stage3 core matrices preserves both attention "
            "weights and the fusion gate. This strongly supports the WD + shortcut "
            "feedback mechanism."
        )
    elif collapse_a and preserve_b and gamma_dead_b:
        label = "WD_CAUSES_MATRIX_COLLAPSE_BUT_GATE_SHORTCUT_PERSISTS"
        notes.append(
            "No-WD keeps Stage3 matrices alive, but fusion_gamma still collapses. "
            "WD explains matrix death; identity-shortcut/task-utility remains an "
            "independent gate-collapse mechanism."
        )
    elif collapse_a and not preserve_b:
        label = "WEIGHT_DECAY_NOT_SUFFICIENT_TO_EXPLAIN_COLLAPSE"
        notes.append(
            "Stage3 still collapses when its core matrices have WD=0. The primary "
            "cause is likely downstream gating/identity shortcut or lack of useful "
            "task gradients rather than matrix weight decay alone."
        )
    elif not collapse_a and gamma_dead_a:
        label = "GATE_BYPASS_WITHOUT_MATRIX_COLLAPSE"
        notes.append(
            "Baseline matrices remain non-collapsed while fusion_gamma dies. Focus "
            "on the residual gate / identity shortcut, not weight decay."
        )
    else:
        label = "NO_CLEAR_STAGE3_COLLAPSE_IN_TEST_WINDOW"
        notes.append(
            "The chosen training window did not reproduce the original collapse "
            "strongly enough. Increase --epochs or run the full epoch count."
        )
    return label, notes


def render_summary(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    clip_norm: Optional[float],
) -> str:
    label, notes = conclusion_from_ab(a, b)
    same_init = a.get("stage3_initial_hash") == b.get("stage3_initial_hash")

    lines = [
        "Stage3 Weight-Decay A/B Test",
        "=" * 72,
        f"A: baseline             GPU {a.get('gpu')}  status={a.get('status')}",
        f"B: no_stage3_core_wd    GPU {b.get('gpu')}  status={b.get('status')}",
        f"Identical Stage3 initial hash: {same_init}",
        f"Shared clip norm: {clip_norm}",
        "",
        "A/B design:",
        "  A keeps configured WD on Stage3 core 2-D matrices.",
        "  B sets WD=0 ONLY on split/self/cross/fusion core matrices.",
        "  Adapter MLP, all non-Stage3 params, LR, data order and seed are unchanged.",
        "",
        "Key comparison:",
    ]
    for note in notes:
        lines.append(f"  - {note}")

    lines += [
        "",
        "AUTOMATED CONCLUSION",
        "-" * 72,
        label,
        "",
        "Interpretation rules:",
        "  * B preserves q/k/v/out/FFN + gamma -> WD/shortcut feedback is strongly implicated.",
        "  * B preserves matrices but gamma still dies -> WD causes matrix collapse, gate shortcut is separate.",
        "  * B still collapses -> WD is not the primary cause; task utility/gating dominates.",
        "  * Matrices stay alive but gamma dies -> focus on identity residual gate.",
        "",
        "Caution:",
        "  Raw ||weight_decay * W|| is compared with task-gradient norm before Adam.",
        "  Adam preconditioning/moments mean it is not the exact parameter-update norm.",
    ]

    if not same_init:
        lines += [
            "",
            "WARNING: initial Stage3 hashes differ. The A/B is not strictly causal; "
            "check RNG/data/model construction before interpreting the result.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    opt = parse_args()
    config = Path(opt.config).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)

    clip_norm = resolve_clip_norm(opt.clip_norm)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = (
        Path(opt.out_dir).expanduser().resolve()
        if opt.out_dir
        else config.parent / f"stage3_wd_ab_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    a_json = out_dir / f"baseline_gpu{opt.gpu_a}.json"
    b_json = out_dir / f"no_stage3_core_wd_gpu{opt.gpu_b}.json"

    print("=" * 72)
    print("Stage3 WD A/B causal test")
    print(f"config: {config}")
    print(f"A baseline GPU: {opt.gpu_a}")
    print(f"B no-Stage3-core-WD GPU: {opt.gpu_b}")
    print(f"epochs: {opt.epochs}")
    print(f"steps/epoch: {'FULL' if opt.steps_per_epoch <= 0 else opt.steps_per_epoch}")
    print(f"clip_norm: {clip_norm}")
    print(f"output: {out_dir}")
    print("=" * 72)

    ctx = mp.get_context("spawn")
    common = dict(
        config_path=str(config),
        epochs=int(opt.epochs),
        steps_per_epoch=int(opt.steps_per_epoch),
        workers=int(opt.workers),
        seed=int(opt.seed),
        log_every=int(opt.log_every),
        clip_norm=clip_norm,
        use_amp=bool(opt.amp),
    )

    pa = ctx.Process(
        target=run_variant,
        kwargs=dict(
            variant="baseline",
            gpu=int(opt.gpu_a),
            out_json=str(a_json),
            **common,
        ),
    )
    pb = ctx.Process(
        target=run_variant,
        kwargs=dict(
            variant="no_stage3_core_wd",
            gpu=int(opt.gpu_b),
            out_json=str(b_json),
            **common,
        ),
    )

    pa.start()
    pb.start()
    pa.join()
    pb.join()

    if not a_json.exists() or not b_json.exists():
        raise RuntimeError(
            f"A/B result missing. exitcodes: A={pa.exitcode}, B={pb.exitcode}"
        )

    a = json.loads(a_json.read_text(encoding="utf-8"))
    b = json.loads(b_json.read_text(encoding="utf-8"))

    summary = render_summary(a, b, clip_norm)
    summary_path = out_dir / "stage3_wd_ab_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print("\n" + summary)
    print(f"Saved: {summary_path}")
    print(f"Saved: {a_json}")
    print(f"Saved: {b_json}")

    if pa.exitcode != 0 or pb.exitcode != 0:
        raise SystemExit(
            f"One worker failed: A exit={pa.exitcode}, B exit={pb.exitcode}. "
            "Inspect the two JSON files for tracebacks."
        )


if __name__ == "__main__":
    main()

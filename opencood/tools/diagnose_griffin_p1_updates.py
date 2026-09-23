#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose whether Griffin P1 parameters actually update.

Run from AirV2X-Perception_gaussian repo root, e.g.

python diagnose_griffin_p1_updates.py \
  --config opencood/hypes_yaml/griffin/camera/gaussian_p1/griffin_gaussian_p1_v45.yaml \
  --latest opencood/logs/griffin_gaussian_p1_v45/latest.pth

First run WITHOUT --amp. If that updates correctly, run again WITH --amp.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.cuda import amp
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd().resolve()
if not (ROOT / "opencood").is_dir():
    ROOT = HERE
sys.path.insert(0, str(ROOT))

from opencood.data_utils.datasets.griffin_p1_dataset import GriffinP1Dataset
from opencood.loss.gaussian_p1_depth_loss import GaussianP1DepthLoss
from opencood.models.griffin_gaussian_p1 import GriffinGaussianP1
from opencood.tools.train_griffin_p1 import (
    batch_loss_metrics,
    move_nested,
    setup_optimizer,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="opencood/hypes_yaml/griffin/camera/gaussian_p1/"
                "griffin_gaussian_p1_v45.yaml",
    )
    p.add_argument("--latest", default=None)
    p.add_argument(
        "--init",
        default=None,
        help=(
            "AirV2X V45 initialization checkpoint. If omitted, infer from "
            "config train.pretrained or sibling airv2x_init_report.json."
        ),
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def strip_state(obj):
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    out = {}
    for k, v in obj.items():
        while k.startswith("module.") or k.startswith("model."):
            k = k.split(".", 1)[1]
        out[k] = v
    return out


def family(name):
    if name.startswith("frontend.encoders.vehicle."):
        return "vehicle/backbone"
    if name.startswith("frontend.encoders.drone."):
        return "drone/backbone"
    if name.startswith("highres.vehicle.") or name.startswith("r2_downsample."):
        return "vehicle/feature"
    if name.startswith("highres.drone."):
        return "drone/feature"
    if name.startswith("heatmap_heads.vehicle."):
        return "vehicle/heatmap"
    if name.startswith("heatmap_heads.drone."):
        return "drone/heatmap"
    if name.startswith("depth_heads.vehicle."):
        return "vehicle/depth"
    return "other"


def compare_checkpoint(init_path, latest_path):
    init = strip_state(torch.load(init_path, map_location="cpu"))
    latest_obj = torch.load(latest_path, map_location="cpu")
    latest = strip_state(latest_obj)

    stats = {}
    for k, v0 in init.items():
        if k not in latest or not torch.is_tensor(v0) or not torch.is_tensor(latest[k]):
            continue
        if tuple(v0.shape) != tuple(latest[k].shape):
            continue
        d = (latest[k].float() - v0.float()).abs()
        fam = family(k)
        rec = stats.setdefault(
            fam, {"n": 0, "changed": 0, "max": 0.0, "sum_mean": 0.0}
        )
        rec["n"] += 1
        mx = float(d.max().item()) if d.numel() else 0.0
        mean = float(d.mean().item()) if d.numel() else 0.0
        if mx > 0:
            rec["changed"] += 1
        rec["max"] = max(rec["max"], mx)
        rec["sum_mean"] += mean

    print("\n=== checkpoint delta: latest vs AirV2X init ===")
    for fam in sorted(stats):
        r = stats[fam]
        print(
            "%-20s tensors=%4d changed=%4d max_abs=%10.4e mean_abs(avg)=%10.4e"
            % (
                fam,
                r["n"],
                r["changed"],
                r["max"],
                r["sum_mean"] / max(r["n"], 1),
            )
        )

    if isinstance(latest_obj, dict):
        opt_state = latest_obj.get("optimizer_state_dict")
        if isinstance(opt_state, dict):
            lrs = [g.get("lr") for g in opt_state.get("param_groups", [])]
            print("saved optimizer lrs:", lrs)
        sc = latest_obj.get("scaler_state_dict")
        if isinstance(sc, dict):
            print("saved GradScaler scale:", sc.get("scale"))
        print("saved epoch:", latest_obj.get("epoch"))


def grad_stats(model):
    out = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        fam = family(name)
        rec = out.setdefault(
            fam, {"params": 0, "with_grad": 0, "finite": 0, "norm2": 0.0, "max": 0.0}
        )
        rec["params"] += 1
        if p.grad is None:
            continue
        rec["with_grad"] += 1
        g = p.grad.detach().float()
        finite = torch.isfinite(g)
        if bool(finite.all()):
            rec["finite"] += 1
        gf = torch.where(finite, g, torch.zeros_like(g))
        rec["norm2"] += float((gf * gf).sum().item())
        if gf.numel():
            rec["max"] = max(rec["max"], float(gf.abs().max().item()))
    return out


def print_grad_stats(stats):
    for fam in sorted(stats):
        r = stats[fam]
        print(
            "  %-20s grad=%3d/%3d finite=%3d norm=%9.3e max=%9.3e"
            % (
                fam,
                r["with_grad"],
                r["params"],
                r["finite"],
                math.sqrt(r["norm2"]),
                r["max"],
            )
        )


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    device = torch.device(
        "cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    init_path = args.init
    if init_path is None:
        init_path = (cfg.get("train") or {}).get("pretrained")

    if init_path is None and args.latest:
        report_path = Path(args.latest).expanduser().resolve().parent / "airv2x_init_report.json"
        if report_path.exists():
            import json
            report = json.loads(report_path.read_text())
            init_path = report.get("path")
            if init_path:
                print(f"[diagnose] inferred AirV2X init from {report_path}: {init_path}")

    if init_path is None:
        # Last compatibility fallback for configs copied from the full 0822 model.
        model_cfg = cfg.get("model") or {}
        model_args = model_cfg.get("args") if isinstance(model_cfg, dict) else None
        if isinstance(model_args, dict):
            init_path = model_args.get("pretrained_v45") or model_args.get("pretrained")

    if init_path is None:
        raise KeyError(
            "Cannot determine AirV2X initialization checkpoint. "
            "Pass --init /path/to/net_final_branch_v45.pth. "
            "This resolved_config.yaml has no train.pretrained."
        )

    init_path = str(Path(init_path).expanduser().resolve())
    print(f"[diagnose] AirV2X init checkpoint: {init_path}")

    if args.latest:
        compare_checkpoint(init_path, args.latest)

    ds = GriffinP1Dataset(cfg["dataset"], "train")
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    batch = next(iter(loader))
    batch = move_nested(batch, device)

    model = GriffinGaussianP1(cfg["model"]).to(device)
    model.load_airv2x_v45(init_path, device)
    model.train()
    model.assert_train_eval_state(True)

    optimizer = setup_optimizer(model, cfg)
    criterion = GaussianP1DepthLoss(cfg.get("depth_loss", {})).to(device)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = amp.GradScaler(enabled=use_amp)

    # Track three decisive heads.
    tracked = [
        "heatmap_heads.vehicle.cls.weight",
        "heatmap_heads.drone.cls.weight",
        "depth_heads.vehicle.pred.weight",
    ]
    named = dict(model.named_parameters())
    for name in tracked:
        if name not in named:
            raise KeyError("missing tracked parameter: %s" % name)

    initial = {name: named[name].detach().float().clone() for name in tracked}

    print("\n=== fixed-batch optimization test ===")
    print("AMP:", use_amp)
    print("optimizer lrs:", [g["lr"] for g in optimizer.param_groups])

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        result = batch_loss_metrics(model, batch, criterion, cfg, use_amp)
        total = result[0]
        metrics = result[1]

        if use_amp:
            scale_before = float(scaler.get_scale())
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
        else:
            scale_before = 1.0
            total.backward()

        gs = grad_stats(model)
        print(
            "\nstep %d before optimizer: loss=%.6f vh=%.6f dh=%.6f vd=%.6f scale=%g"
            % (
                step,
                float(total.detach().item()),
                metrics["loss_vehicle_hm"],
                metrics["loss_drone_hm"],
                metrics["loss_vehicle_depth"],
                scale_before,
            )
        )
        print_grad_stats(gs)

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
        else:
            optimizer.step()
            scale_after = 1.0

        deltas = {}
        for name in tracked:
            deltas[name] = float(
                (named[name].detach().float() - initial[name]).abs().max().item()
            )
        print(
            "  parameter max|delta from init|:",
            {k: "%.4e" % v for k, v in deltas.items()},
        )
        if use_amp:
            print("  GradScaler scale after:", scale_after)

    print("\nINTERPRETATION")
    print("* all tracked deltas == 0 => optimizer is not changing model weights.")
    print("* non-finite grads or AMP scale repeatedly falling => rerun without AMP.")
    print("* deltas > 0 and fixed-batch loss falls => core training path is healthy.")
    print("* checkpoint delta == 0 after many epochs => those epochs did not update weights.")


if __name__ == "__main__":
    main()

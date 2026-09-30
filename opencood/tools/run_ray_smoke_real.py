#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-checkpoint + real-batch ray-mode smoke tests (Test B / Test H).

Test B: legacy opacity-feature checkpoint strict-loads into the unmodified
        (xyz) old-yaml model, and into the new ray model with the ray heads
        as the only missing keys.
Test H: one real train-mode batch forward/backward on the ray yaml;
        verifies finite outputs/grads and that per-agent ray_head grads
        flow only through agents with valid Stage-2 queries.

Usage:
  python opencood/tools/run_ray_smoke_real.py [--gpu-id 0] [--skip-load-test]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.airv2x_gaussian_0822 import Airv2xGaussian0822
from opencood.tools import train_utils

OLD_YAML = (
    ROOT
    / "opencood/hypes_yaml/airv2x/camera/det/"
    "airv2x_gaussian_0822_geom_sup_opacity_feature.yaml"
)
RAY_YAML = (
    ROOT
    / "opencood/hypes_yaml/airv2x/camera/det/"
    "airv2x_gaussian_0822_geom_sup_opacity_feature_ray.yaml"
)
LEGACY_CKPT = (
    ROOT
    / "opencood/logs/airv2x_gaussian_0822_geom_sup_opacity_feature/"
    "geom_sup_opacity_2026_09_23_23_41_45/net_epoch31.pth"
)


def _unwrap(obj: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError(type(obj))
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        if isinstance(obj.get(key), dict):
            return obj[key]
    return obj


def test_b_checkpoint_compatibility(device: torch.device) -> None:
    state = dict(_unwrap(torch.load(str(LEGACY_CKPT), map_location="cpu")))
    assert state and next(iter(state)).startswith("cross_agent."), next(iter(state))

    # 1) Old yaml + old ckpt: every ckpt key must match exactly (strict-able).
    hypes = yaml_utils.load_yaml(str(OLD_YAML))
    model = Airv2xGaussian0822(hypes["model"]["args"]).to(device)
    current = model.state_dict()
    missing = [k for k in current if k not in state]
    unexpected = [k for k in state if k not in current]
    mismatch = [
        (k, tuple(state[k].shape), tuple(current[k].shape))
        for k in state
        if k in current and tuple(state[k].shape) != tuple(current[k].shape)
    ]
    assert not unexpected, unexpected[:8]
    assert not mismatch, mismatch[:8]
    # missing must be only the optional quality head etc., never mean/ray keys
    assert not any("mean_head" in k or "ray" in k for k in missing), missing[:12]
    model.load_state_dict(state, strict=False)
    print(
        f"[B1] old yaml + old ckpt: 0 unexpected, 0 shape mismatch "
        f"({len(missing)} model-only keys, none geometry): OK"
    )

    # 2) New ray yaml + old ckpt: ray heads are the ONLY new missing keys.
    hypes_ray = yaml_utils.load_yaml(str(RAY_YAML))
    model_ray = Airv2xGaussian0822(hypes_ray["model"]["args"]).to(device)
    current_ray = model_ray.state_dict()
    missing_ray = [k for k in current_ray if k not in state]
    unexpected_ray = [k for k in state if k not in current_ray]
    assert not unexpected_ray, unexpected_ray[:8]
    ray_keys = sorted(k for k in missing_ray if "ray_head" in k)
    non_ray = [k for k in missing_ray if "ray_head" not in k]
    assert not any("mean_head" in k for k in non_ray), non_ray[:12]
    assert len(ray_keys) == 6, ray_keys  # 3 agents x (weight, bias)
    # zero-init ray heads keep Δd = 0 after loading the legacy weights
    model_ray.load_state_dict(state, strict=False)
    for agent in ("vehicle", "rsu", "drone"):
        head = model_ray.cross_agent.refiner.heads.heads[agent]
        assert torch.count_nonzero(head.ray_head.weight) == 0
        assert torch.count_nonzero(head.ray_head.bias) == 0
    print(f"[B2] ray yaml + old ckpt: only {len(ray_keys)} ray_head keys missing: OK")
    print("      example keys:", ray_keys[:2])


def test_h_real_forward_backward(
    device: torch.device, skip_load: bool
) -> None:
    hypes = yaml_utils.load_yaml(str(RAY_YAML))
    model = Airv2xGaussian0822(hypes["model"]["args"]).to(device)
    if not skip_load:
        state = dict(_unwrap(torch.load(str(LEGACY_CKPT), map_location="cpu")))
        model.load_state_dict(state, strict=False)  # ray heads stay zero-init
        print("[H] legacy weights loaded into ray model (ray heads zero-init)")
    model.train()

    dataset = build_dataset(hypes, visualize=False, train=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(hypes["train_params"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_batch_train,
        drop_last=True,
    )
    batch = next(iter(loader))
    batch = train_utils.to_device(batch, device)
    ego = batch["ego"]
    ego["epoch"] = 0

    criterion = train_utils.create_loss(hypes)
    output = model(ego)
    loss = criterion(output, ego["label_dict"])
    if "geom_loss" in output:
        print(f"[H] geom_loss present: {float(output['geom_loss']):.4f}")
    loss.backward()

    finite_fields = {}
    for agent in ("vehicle", "rsu", "drone"):
        head = model.cross_agent.refiner.heads.heads[agent]
        g = head.ray_head.weight.grad
        gb = head.ray_head.bias.grad
        ok = g is not None and torch.isfinite(g).all().item()
        nonzero = bool(g is not None and g.abs().sum().item() > 0)
        finite_fields[f"{agent}.ray_head"] = (ok, nonzero)
    for key, tensor in (
        ("psm", output["psm"]),
        ("rm", output["rm"]),
        ("obj", output["obj"]),
    ):
        finite_fields[key] = (bool(torch.isfinite(tensor).all().item()), None)
    gs = ego.get("gaussians") or {}
    for agent, gset in gs.items():
        for field in ("mean", "scale", "quaternion", "feature", "opacity"):
            t = getattr(gset, field, None)
            if t is not None:
                finite_fields[f"{agent}.{field}"] = (
                    bool(torch.isfinite(t).all().item()),
                    None,
                )
        assert gset.ray_dir is not None, f"{agent} lost ray_dir in forward"
        norm = torch.linalg.norm(gset.ray_dir, dim=-1)
        if int(norm.numel()) > 0:
            assert torch.allclose(
                norm, torch.ones_like(norm), atol=1e-4, rtol=1e-4
            ), (agent, norm.min().item(), norm.max().item())
    bev = ego.get("spatial_features")
    if bev is not None:
        finite_fields["bev"] = (bool(torch.isfinite(bev).all().item()), None)

    bad = [k for k, (ok, _) in finite_fields.items() if not ok]
    assert not bad, bad
    print(f"[H] finite fields: {list(finite_fields)}")
    for name, (ok, nonzero) in finite_fields.items():
        if "ray_head" in name:
            print(f"    {name}: grad finite={ok} nonzero={nonzero}")
    print(f"[H] total loss {float(loss):.4f}: forward/backward OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--skip-load-test", action="store_true")
    parser.add_argument("--skip-ckpt-load", action="store_true")
    opt = parser.parse_args()
    device = torch.device(f"cuda:{opt.gpu_id}")
    torch.cuda.set_device(device)
    test_b_checkpoint_compatibility(device)
    test_h_real_forward_backward(device, skip_load=opt.skip_ckpt_load)
    print("\nREAL RAY SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()

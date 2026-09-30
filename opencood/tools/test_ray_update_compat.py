#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ray-aligned mean update compatibility tests (Test A-H, CPU-only parts).

Covers the spec sections 2/7/10/15:
  A: old yaml adds no ray_head, stays "xyz"
  B: old xyz checkpoint keeps loading (state_dict key/shape contract)
  C: new ray yaml -> mode=ray, per-agent ray_delta_max, Linear(H,1) heads
  D: zero-init ray head -> delta_depth==0, new_mean==mean
  E: movement strictly along the ray (lateral residual < 1e-5)
  F: |delta_depth| <= ray_delta_max
  G: geometry supervision xyz=3 coords / ray=1 scalar, finite fwd+bwd
  H: (real-batch part runs separately on GPU; see run_ray_smoke_real.py)

Run:
  python opencood/tools/test_ray_update_compat.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.refinement.gaussian_refiner import (
    GaussianRefiner,
)
from opencood.models.gaussian_modules_0822.refinement.geometry_supervision import (
    compute_stage2_delta_mean_loss,
)
from opencood.models.gaussian_modules_0822.refinement.refine_heads import (
    AgentRefineHeads,
    GaussianRefineHeads,
)

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


def _refiner(hypes: Dict[str, Any]) -> GaussianRefiner:
    """Build a fresh GaussianRefiner from a loaded yaml (tiny hidden dim)."""
    cfg = dict(hypes["model"]["args"]["gaussian_refinement"])
    cfg["fusion"] = {"hidden_dim": 16}
    from opencood.models.gaussian_modules_0822.base.gaussian_geometry_encoder import (
        GaussianGeometryEncoder,
    )

    return GaussianRefiner(
        GaussianGeometryEncoder(geo_dim=8),
        feature_dim=4,
        cfg=cfg,
    )


def test_a_old_yaml_xyz() -> None:
    hypes = load_yaml(str(OLD_YAML))
    refiner = _refiner(hypes)
    assert refiner.mean_update_mode == "xyz"
    for agent, head in refiner.heads.heads.items():
        assert not hasattr(head, "ray_head"), f"{agent} unexpectedly has ray_head"
        assert head.mean_head.weight.shape == (3, 16)
    print("[A] old yaml -> xyz, no ray_head anywhere: OK")


def test_b_old_ckpt_contract() -> None:
    # mean_head keys/shape unchanged; no new mandatory key in state_dict.
    hypes = load_yaml(str(OLD_YAML))
    refiner = _refiner(hypes)
    sd = refiner.state_dict()
    for agent in ("vehicle", "rsu", "drone"):
        w = f"heads.heads.{agent}.mean_head.weight"
        b = f"heads.heads.{agent}.mean_head.bias"
        assert sd[w].shape == (3, 16), (w, sd[w].shape)
        assert sd[b].shape == (3,)
        assert not any("ray" in k for k in sd), sorted(sd)
    # Simulate an old checkpoint: strict load of a same-shape old state.
    old_state = {k: v.clone() for k, v in sd.items()}
    missing, unexpected = refiner.load_state_dict(old_state, strict=True), None
    assert missing is not None
    print("[B] xyz state_dict keeps legacy mean_head keys/shape, strict-loadable: OK")


def test_c_new_yaml_ray() -> None:
    hypes = load_yaml(str(RAY_YAML))
    args = hypes["model"]["args"]
    assert args["gaussian_refinement"]["mean_update_mode"] == "ray"
    # Deep-merge kept the parent tree.
    assert args["gaussian_refinement"]["agents"]["vehicle"]["unit_xyz"] == [2.0, 2.0, 1.0]
    assert args["stage1_feature_only"] is True
    assert args["gaussian_stage3"]["opacity"]["feature_conditioned"] is True
    assert args["gaussian_stage3"]["opacity"]["final_opacity_source"] == "stage3"
    refiner = _refiner(hypes)
    assert refiner.mean_update_mode == "ray"
    want = {"vehicle": 2.0, "rsu": 2.0, "drone": 1.0}
    for agent, head in refiner.heads.heads.items():
        assert hasattr(head, "ray_head"), agent
        assert tuple(head.ray_head.weight.shape) == (1, 16)
        assert float(head.ray_delta_max.item()) == want[agent]
        assert head.mean_update_mode == "ray"
        # legacy mean_head still exists with the legacy shape
        assert tuple(head.mean_head.weight.shape) == (3, 16)
    print("[C] ray yaml -> mode=ray, delta_max v2/r2/d1, Linear(H,1) heads: OK")


def test_d_zero_init() -> None:
    head = AgentRefineHeads(
        hidden_dim=16,
        unit_xyz=(1.0, 1.0, 1.0),
        scale_delta_range=(-0.2, 0.05),
        rotation_max_angle=0.2,
        mean_update_mode="ray",
        ray_delta_max=2.0,
    )
    assert torch.count_nonzero(head.ray_head.weight) == 0
    assert torch.count_nonzero(head.ray_head.bias) == 0
    n = 32
    feat = torch.randn(n, 16)
    mean = torch.randn(n, 3)
    ray = torch.nn.functional.normalize(torch.randn(n, 3), dim=-1)
    out = head(feat, mean, torch.ones(n, 3), torch.tensor([[1.0, 0, 0, 0]]).expand(n, -1).contiguous(), ray_dir=ray)
    assert torch.count_nonzero(out["delta_depth"]) == 0
    assert torch.allclose(out["mean"], mean, atol=0.0, rtol=0.0)
    print("[D] zero-init ray head -> delta_depth=0, mean unchanged: OK")


def test_e_ray_only_movement() -> None:
    torch.manual_seed(0)
    head = AgentRefineHeads(
        hidden_dim=16,
        unit_xyz=(1.0, 1.0, 1.0),
        scale_delta_range=(-0.2, 0.05),
        rotation_max_angle=0.2,
        mean_update_mode="ray",
        ray_delta_max=2.0,
    )
    torch.nn.init.normal_(head.ray_head.weight, std=0.5)
    torch.nn.init.normal_(head.ray_head.bias, std=0.5)
    n = 512
    feat = torch.randn(n, 16)
    mean = torch.randn(n, 3)
    ray_dir = torch.nn.functional.normalize(torch.randn(n, 3), dim=-1)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, -1).contiguous()
    out = head(feat, mean, torch.ones(n, 3), quat, ray_dir=ray_dir)
    delta = out["mean"] - mean
    parallel = (delta * ray_dir).sum(dim=-1, keepdim=True) * ray_dir
    lateral = delta - parallel
    assert lateral.abs().max().item() < 1.0e-5, lateral.abs().max().item()
    print(f"[E] lateral residual max {lateral.abs().max().item():.3e} < 1e-5: OK")


def test_f_bound() -> None:
    for dmax in (2.0, 1.0):
        head = AgentRefineHeads(
            hidden_dim=16,
            unit_xyz=(1.0, 1.0, 1.0),
            scale_delta_range=(-0.2, 0.05),
            rotation_max_angle=0.2,
            mean_update_mode="ray",
            ray_delta_max=dmax,
        )
        torch.nn.init.normal_(head.ray_head.weight, std=50.0)
        torch.nn.init.normal_(head.ray_head.bias, std=50.0)
        n = 256
        ray = torch.nn.functional.normalize(torch.randn(n, 3), dim=-1)
        out = head(
            torch.randn(n, 16) * 10.0,
            torch.randn(n, 3),
            torch.ones(n, 3),
            torch.tensor([[1.0, 0, 0, 0]]).expand(n, -1).contiguous(),
            ray_dir=ray,
        )
        assert out["delta_depth"].abs().max().item() <= dmax + 1.0e-5
    print("[F] |delta_depth| <= ray_delta_max (saturated inputs): OK")


def _fake_sets(n: int, with_ray: bool) -> Dict[str, GaussianSet]:
    ray = torch.nn.functional.normalize(torch.randn(n, 3), dim=-1) if with_ray else None
    mean = torch.randn(n, 3)
    return {
        "vehicle": GaussianSet(
            mean=mean,
            scale=torch.ones(n, 3) * 0.5,
            quaternion=torch.tensor([[1.0, 0, 0, 0]]).expand(n, -1).contiguous(),
            feature=torch.randn(n, 4),
            uv=torch.rand(n, 2) * 100.0,
            view_index=torch.zeros(n, dtype=torch.long),
            batch_index=torch.zeros(n, dtype=torch.long),
            ray_dir=ray,
            agent="vehicle",
            frame="ego",
        )
    }


def _fake_data_dict(n_views: int = 1) -> Dict[str, Any]:
    # Minimal payload with a GT-depth channel (channel 3) and geometry keys.
    from opencood.models.gaussian_modules_0822.base.projection import (
        CAMERA_GEOMETRY_KEY,
    )

    h, w = 4, 4
    imgs = torch.zeros(n_views, 4, 4, h, w)
    imgs[:, 3] = 20.0  # GT depth everywhere
    k = torch.tensor([[[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]]])
    rot = torch.tensor([[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]])
    geometry = {
        "intrinsics": k,
        "cam2lidar_rot": rot,
        "cam2lidar_trans": torch.zeros(1, 3),
        "post_rots": torch.eye(3).unsqueeze(0),
        "post_trans": torch.zeros(1, 3),
        "agent_to_ego": torch.eye(4).unsqueeze(0),
    }
    return {
        "vehicle": {
            "batch_merged_cam_inputs": {"imgs": imgs},
            CAMERA_GEOMETRY_KEY: geometry,
        }
    }


def test_g_geometry_supervision() -> None:
    n = 64
    s1 = _fake_sets(n, with_ray=True)
    # s2 differs from s1 by a random (differentiable) shift.
    s2_mean = s1["vehicle"].mean + torch.randn(n, 3)
    s2 = GaussianSet(
        mean=s2_mean,
        scale=s1["vehicle"].scale,
        quaternion=s1["vehicle"].quaternion,
        feature=s1["vehicle"].feature,
        uv=s1["vehicle"].uv,
        view_index=s1["vehicle"].view_index,
        batch_index=s1["vehicle"].batch_index,
        ray_dir=s1["vehicle"].ray_dir,
        agent="vehicle",
        frame="ego",
    )
    s2_mean.requires_grad_(True)
    valid = {"vehicle": torch.ones(n, dtype=torch.bool)}
    ranges = {"vehicle": (0.0, 50.0)}

    # Recompute s2 with a graph: rebuild mean from a leaf.
    base = s1["vehicle"].mean
    delta = torch.randn(n, 3, requires_grad=True)
    s2 = GaussianSet(
        mean=base + delta,
        scale=s1["vehicle"].scale,
        quaternion=s1["vehicle"].quaternion,
        feature=s1["vehicle"].feature,
        uv=s1["vehicle"].uv,
        view_index=s1["vehicle"].view_index,
        batch_index=s1["vehicle"].batch_index,
        ray_dir=s1["vehicle"].ray_dir,
        agent="vehicle",
        frame="ego",
    )
    data = _fake_data_dict()
    for mode, expect in (("xyz", 3), ("ray", 1)):
        if mode == "ray":
            loss = compute_stage2_delta_mean_loss(
                s1, {"vehicle": s2}, data, valid, ranges, beta=1.0,
                mean_update_mode="ray",
            )
        else:
            loss = compute_stage2_delta_mean_loss(
                s1, {"vehicle": s2}, data, valid, ranges, beta=1.0,
                mean_update_mode="xyz",
            )
        assert torch.isfinite(loss), (mode, loss)
        loss.backward(retain_graph=True)
        assert torch.isfinite(delta.grad).all(), mode
        g = delta.grad.clone()
        if mode == "xyz":
            # All 3 coordinates receive gradient.
            assert (g.abs().sum(dim=-1) > 0).all()
        else:
            # 1-DoF: gradient lives only along the ray direction.
            g_perp = g - (g * s1["vehicle"].ray_dir).sum(-1, keepdim=True) * s1["vehicle"].ray_dir
            assert g_perp.abs().max().item() < 1.0e-6, g_perp.abs().max().item()
        delta.grad.zero_()
    print("[G] supervision xyz=3-coord / ray=1-DoF, finite fwd+bwd: OK")


def test_ray_provenance_rotation() -> None:
    # to_ego must rotate the ray (R @ r) and keep it unit, no translation.
    n = 8
    torch.manual_seed(1)
    ray = torch.nn.functional.normalize(torch.randn(n, 3), dim=-1)
    gs = GaussianSet(
        mean=torch.randn(n, 3) * 10.0,
        scale=torch.ones(n, 3),
        quaternion=torch.tensor([[1.0, 0, 0, 0]]).expand(n, -1).contiguous(),
        feature=torch.randn(n, 4),
        ray_dir=ray,
        agent="vehicle",
        frame="agent",
    )
    yaw = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    t = torch.eye(4)
    t[:3, :3] = yaw
    t[:3, 3] = torch.tensor([10.0, -2.0, 0.5])
    out = gs.to_ego(t)
    assert out.frame == "ego"
    expected = torch.matmul(yaw, ray.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(out.ray_dir, expected, atol=1.0e-5)
    norm = torch.linalg.norm(out.ray_dir, dim=-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1.0e-4, rtol=1.0e-4)
    # translation must NOT enter the ray
    shifted = t.clone()
    shifted[:3, 3] = torch.tensor([99.0, 99.0, 99.0])
    out2 = gs.to_ego(shifted)
    assert torch.allclose(out2.ray_dir, out.ray_dir, atol=1.0e-6)
    # per-camera [n_flat,4,4] + view_index gather path
    n_flat = 3
    stack = t.unsqueeze(0).repeat(n_flat, 1, 1)
    stack[1, :3, :3] = torch.eye(3)
    vi = torch.tensor([0, 1, 1, 2, 2, 0, 1, 2])
    gs2 = GaussianSet(
        mean=torch.randn(n, 3),
        scale=torch.ones(n, 3),
        quaternion=torch.tensor([[1.0, 0, 0, 0]]).expand(n, -1).contiguous(),
        feature=torch.randn(n, 4),
        ray_dir=ray,
        view_index=vi,
        agent="vehicle",
        frame="agent",
    )
    out3 = gs2.to_ego(stack)  # uses self.view_index
    exp_rot = stack[vi.long(), :3, :3]
    exp = torch.matmul(exp_rot, ray.unsqueeze(-1)).squeeze(-1)
    exp = exp / exp.norm(dim=-1, keepdim=True)
    assert torch.allclose(out3.ray_dir, exp, atol=1.0e-5)
    # index_select keeps ray provenance
    sub = out3.index_select(torch.tensor([0, 3, 5]))
    assert torch.allclose(sub.ray_dir, out3.ray_dir[[0, 3, 5]])
    # ray_dir=None legacy sets are untouched
    legacy = GaussianSet(
        mean=torch.randn(2, 3),
        scale=torch.ones(2, 3),
        quaternion=torch.tensor([[1.0, 0, 0, 0]]).expand(2, -1).contiguous(),
        feature=torch.randn(2, 4),
        agent="vehicle",
    )
    legacy_ego = legacy.to_ego(t)
    assert legacy_ego.ray_dir is None
    print("[extra] to_ego rotates ray only, unit norm, gather/index_select ok: OK")


def main() -> None:
    test_a_old_yaml_xyz()
    test_b_old_ckpt_contract()
    test_c_new_yaml_ray()
    test_d_zero_init()
    test_e_ray_only_movement()
    test_f_bound()
    test_g_geometry_supervision()
    test_ray_provenance_rotation()
    print("\nALL RAY-UPDATE COMPATIBILITY TESTS PASSED")


if __name__ == "__main__":
    main()

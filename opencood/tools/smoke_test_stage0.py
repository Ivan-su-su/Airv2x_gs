# -*- coding: utf-8 -*-
"""Stage-0 homo smoke tests: P1 init, freeze, RGB-only, fwd/bwd, identity."""

import argparse
import sys
from pathlib import Path

import torch

root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.utils.airv2x_freeze import (
    assert_lss_frozen,
    assert_lss_not_in_optimizer,
    assert_no_lss_grad,
    lss_parameter_checksum,
)

CKPT = "/home/dell/suyi/AirV2X-Perception_gs/opencood/logs/airv2x_gaussian_final/net_final_branch_v45.pth"

CONFIGS = {
    "vehicle": "opencood/hypes_yaml/airv2x/camera/det/airv2x_homo/airv2x_homo_vehicle_camera.yaml",
    "rsu": "opencood/hypes_yaml/airv2x/camera/det/airv2x_homo/airv2x_homo_rsu_camera.yaml",
    "drone": "opencood/hypes_yaml/airv2x/camera/det/airv2x_homo/airv2x_homo_drone_camera.yaml",
}


def lss_bn_checksum(model) -> float:
    """Checksum over frozen P1 BN running_mean / running_var."""
    total = 0.0
    for name, buf in model.named_buffers():
        if not ("running_mean" in name or "running_var" in name):
            continue
        if name.startswith("p1_core."):
            total += float(buf.detach().float().abs().sum().cpu())
    return total


def run_agent(agent: str, device: torch.device) -> bool:
    print(f"\n{'='*70}\n[SMOKE {agent.upper()}]\n{'='*70}")
    hypes = yaml_utils.load_yaml(CONFIGS[agent])
    hypes["model"]["args"]["p1_checkpoint"] = CKPT
    assert hypes["model"]["args"]["homo_agent_type"] == agent
    assert hypes.get("rgb_only_no_depth_gt") is True

    model = train_utils.create_model(hypes).to(device)
    assert_lss_frozen(model)
    assert getattr(model, "p1_core", None) is not None

    model.train()
    assert not model.p1_core.training, "p1_core must stay eval after model.train()"
    splat_attr = {"vehicle": "veh_models", "rsu": "rsu_models", "drone": "drone_models"}[agent]
    splat = getattr(model, splat_attr)[0]
    assert splat.bevencode.training, "bevencode must stay trainable"

    optimizer = train_utils.setup_optimizer(hypes, model)
    assert_lss_not_in_optimizer(model, optimizer)

    B, L, Ncam = 1, {"vehicle": 3, "rsu": 2, "drone": 2}[agent], 4
    cam_cfg = hypes["fusion"]["args"][f"{'veh' if agent == 'vehicle' else agent}_data_aug_conf"]
    H, W = cam_cfg["final_dim"]
    imgs = torch.rand(B * L, Ncam, 3, H, W, device=device)
    dd = hypes["fusion"]["args"][f"{'veh' if agent == 'vehicle' else agent}_grid_conf"]
    D = dd["ddiscr"][2]
    cam_inputs = {
        "imgs": imgs,
        "rots": torch.eye(3, device=device).view(1, 1, 3, 3).expand(B * L, Ncam, 3, 3).contiguous(),
        "trans": torch.zeros(B * L, Ncam, 3, device=device),
        "intrinsics": torch.eye(3, device=device).view(1, 1, 3, 3).expand(B * L, Ncam, 3, 3).contiguous(),
        "post_rots": torch.eye(3, device=device).view(1, 1, 3, 3).expand(B * L, Ncam, 3, 3).contiguous(),
        "post_trans": torch.zeros(B * L, Ncam, 3, device=device),
    }
    if agent == "drone":
        cam_inputs["camera_world_z"] = torch.full((B * L, Ncam), 80.0, device=device)
    data = {
        "vehicle": {"batch_idxs": [], "record_len": torch.zeros(B, dtype=torch.long)},
        "rsu": {"batch_idxs": [], "record_len": torch.zeros(B, dtype=torch.long)},
        "drone": {"batch_idxs": [], "record_len": torch.zeros(B, dtype=torch.long)},
    }
    data[agent] = {
        "batch_merged_cam_inputs": cam_inputs,
        "batch_idxs": list(range(B)),
        "record_len": torch.tensor([L] * B, dtype=torch.long, device=device),
    }
    data["img_pairwise_t_matrix_collab"] = (
        torch.eye(4, device=device).view(1, 1, 1, 4, 4).expand(B, L, L, 4, 4).contiguous()
    )

    out = model(data)
    loss = out["psm"].sum() + out["rm"].sum() + (out["obj"].sum() if "obj" in out else 0)
    assert torch.isfinite(loss), "non-finite loss"
    loss.backward()
    assert_no_lss_grad(model)

    before_w, before_bn = lss_parameter_checksum(model), lss_bn_checksum(model)
    optimizer.step()
    after_w, after_bn = lss_parameter_checksum(model), lss_bn_checksum(model)
    assert before_w == after_w, "frozen P1 weights changed after optimizer.step()!"
    assert before_bn == after_bn, "frozen P1 BN running stats changed!"

    print(f"[SMOKE {agent.upper()}] PASS "
          f"(max_cav={L}, RGB C=3, D={D}, loss={loss.item():.3f})")
    return True


def main():
    global CKPT
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default="vehicle,rsu,drone")
    parser.add_argument("--ckpt", default=CKPT)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CKPT = args.ckpt
    ok = True
    for agent in args.agents.split(","):
        ok &= run_agent(agent.strip(), device)
    print("\nALL SMOKE TESTS PASSED" if ok else "\nSMOKE TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

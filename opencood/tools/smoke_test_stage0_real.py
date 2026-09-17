# -*- coding: utf-8 -*-
"""Stage-0 real-data smoke: real batch → C==3 check → full fwd/bwd/step."""

import argparse
import sys
from pathlib import Path

import torch

root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
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
    total = 0.0
    for name, buf in model.named_buffers():
        if name.startswith("p1_core.") and (
            "running_mean" in name or "running_var" in name
        ):
            total += float(buf.detach().float().abs().sum().cpu())
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="vehicle", choices=list(CONFIGS))
    parser.add_argument("--ckpt", default=CKPT)
    args = parser.parse_args()
    agent = args.agent
    device = torch.device("cuda:0")
    hypes = yaml_utils.load_yaml(CONFIGS[agent])
    hypes["model"]["args"]["p1_checkpoint"] = args.ckpt

    print("[real-smoke] building train dataset ...")
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    print(f"[real-smoke] dataset len = {len(train_dataset)}")

    sample = train_dataset[0]
    batch = train_dataset.collate_batch_train([sample])
    imgs = batch["ego"][agent]["batch_merged_cam_inputs"]["imgs"]
    print(f"[real-smoke] imgs shape = {tuple(imgs.shape)}")
    assert imgs.shape[-3] == 3, f"Test E failed: C={imgs.shape[-3]} != 3"
    print("[real-smoke] Test E PASS: RGB-only C==3")

    model = train_utils.create_model(hypes).to(device)
    assert_lss_frozen(model)
    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)
    assert_lss_not_in_optimizer(model, optimizer)

    model.train()
    assert not model.p1_core.training
    batch = train_utils.to_device(batch, device)
    batch["ego"]["epoch"] = 0
    out = model(batch["ego"])
    loss = criterion(out, batch["ego"]["label_dict"])
    assert torch.isfinite(loss), "non-finite loss"
    loss.backward()
    assert_no_lss_grad(model)

    before_w = lss_parameter_checksum(model)
    before_bn = lss_bn_checksum(model)
    optimizer.step()
    after_w = lss_parameter_checksum(model)
    after_bn = lss_bn_checksum(model)
    assert before_w == after_w, "Test D failed: P1 weights moved"
    assert before_bn == after_bn, "Test D failed: P1 BN stats moved"

    print(f"[real-smoke] loss = {loss.item():.4f} (finite)")
    print("[real-smoke] Test D PASS: frozen P1 weights + BN identical after step")
    print("[real-smoke] ALL PASS")


if __name__ == "__main__":
    main()

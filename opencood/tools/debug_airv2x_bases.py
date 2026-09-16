# -*- coding: utf-8 -*-
"""Smoke tests for Stage-0 checkpoint remapping and agent packing."""

from __future__ import annotations

import torch
from torch import nn

from opencood.utils.airv2x_agent_repack import repack_agent_features
from opencood.utils.airv2x_base_checkpoint import (
    copy_matching_tensors,
    extract_state_dict,
    load_agent_encoder,
    load_local_branch,
    remap_prefix,
)
from opencood.utils.airv2x_freeze import (
    assert_lss_frozen,
    assert_lss_not_in_optimizer,
    assert_no_lss_grad,
    freeze_lss_encoders,
    grouped_trainable_parameters,
    lss_parameter_checksum,
    set_module_trainable,
)


class DummyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)


class DummyDet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.veh_models = nn.ModuleList([DummyEncoder()])
        self.rsu_models = nn.ModuleList([DummyEncoder()])
        self.drone_models = nn.ModuleList([DummyEncoder()])
        self.backbone = nn.Conv2d(4, 4, 1)
        self.pyramid_backbone = nn.Conv2d(4, 4, 1)
        self.cls_head = nn.Conv2d(4, 2, 1)
        self.reg_head = nn.Conv2d(4, 7, 1)
        self.obj_head = nn.Conv2d(4, 1, 1)


class DummyBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = DummyEncoder()
        self.backbone = nn.Conv2d(4, 4, 1)
        self.pyramid_backbone = nn.Conv2d(4, 4, 1)
        self.cls_head = nn.Conv2d(4, 2, 1)
        self.reg_head = nn.Conv2d(4, 7, 1)
        self.obj_head = nn.Conv2d(4, 1, 1)


def test_extract_state_dict_formats() -> None:
    raw = {"a": torch.ones(1)}
    assert "a" in extract_state_dict(raw)
    wrapped = {"model_state_dict": {"b": torch.ones(2)}}
    assert "b" in extract_state_dict(wrapped)
    wrapped = {"state_dict": {"module.c": torch.ones(3)}}
    assert "c" in extract_state_dict(wrapped)


def test_heal_style_explicit_load() -> None:
    src = DummyDet()
    dst = DummyDet()
    with torch.no_grad():
        src.veh_models[0].conv.weight.fill_(1.5)
        src.rsu_models[0].conv.weight.fill_(2.5)
        src.drone_models[0].conv.weight.fill_(3.5)
        src.backbone.weight.fill_(4.5)
    veh_state = {f"veh_models.{k}": v for k, v in src.veh_models.state_dict().items()}
    rsu_state = {f"rsu_models.{k}": v for k, v in src.rsu_models.state_dict().items()}
    drone_state = {f"drone_models.{k}": v for k, v in src.drone_models.state_dict().items()}
    veh_state.update(src.state_dict())
    load_agent_encoder(dst, veh_state, "vehicle", source="veh")
    load_agent_encoder(dst, rsu_state, "rsu", source="rsu")
    load_agent_encoder(dst, drone_state, "drone", source="drone")
    mapped = remap_prefix({"backbone.weight": src.backbone.weight, "backbone.bias": src.backbone.bias}, "backbone.", "")
    copy_matching_tensors(dst.backbone, mapped, source="veh", destination="backbone")
    assert torch.allclose(dst.veh_models[0].conv.weight, src.veh_models[0].conv.weight)
    assert torch.allclose(dst.rsu_models[0].conv.weight, src.rsu_models[0].conv.weight)
    assert torch.allclose(dst.drone_models[0].conv.weight, src.drone_models[0].conv.weight)
    assert torch.allclose(dst.backbone.weight, src.backbone.weight)


def test_local_branch_remap() -> None:
    src = DummyDet()
    branch = DummyBranch()
    with torch.no_grad():
        src.veh_models[0].conv.weight.fill_(9.0)
        src.cls_head.weight.fill_(8.0)
    load_local_branch(branch, src.state_dict(), "vehicle", source="veh_base")
    assert torch.allclose(branch.encoder.conv.weight, src.veh_models[0].conv.weight)
    assert torch.allclose(branch.cls_head.weight, src.cls_head.weight)


def test_freeze_lss() -> None:
    model = DummyDet()
    freeze_lss_encoders(model)
    assert_lss_frozen(model)
    groups = grouped_trainable_parameters(model)
    assert "veh_models" not in groups
    assert "backbone" in groups


def test_repack_order() -> None:
    data_dict = {
        "agent_order": ["vehicle", "rsu", "drone"],
        "vehicle": {
            "batch_idxs": [0],
            "record_len": torch.tensor([2]),
        },
        "rsu": {
            "batch_idxs": [0],
            "record_len": torch.tensor([1]),
        },
        "drone": {
            "batch_idxs": [0],
            "record_len": torch.tensor([1]),
        },
    }
    veh = torch.stack([torch.full((1, 2, 2), 1.0), torch.full((1, 2, 2), 2.0)])
    rsu = torch.full((1, 1, 2, 2), 3.0)
    drone = torch.full((1, 1, 2, 2), 4.0)
    packed, record_len = repack_agent_features(
        {"vehicle": veh, "rsu": rsu, "drone": drone}, data_dict, batch_size=1
    )
    assert record_len.tolist() == [4]
    assert packed[0, 0, 0, 0].item() == 1.0
    assert packed[1, 0, 0, 0].item() == 2.0
    assert packed[2, 0, 0, 0].item() == 3.0
    assert packed[3, 0, 0, 0].item() == 4.0


class ScaleAdapter(nn.Module):
    """Tiny adapter used to prove adapted features actually reach fusion packing."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return feat * self.scale


def test_stamp_adapter_reaches_pack() -> None:
    data_dict = {
        "agent_order": ["vehicle", "rsu"],
        "vehicle": {"batch_idxs": [0], "record_len": torch.tensor([1])},
        "rsu": {"batch_idxs": [0], "record_len": torch.tensor([1])},
    }
    veh = torch.ones(1, 1, 2, 2)
    rsu = torch.ones(1, 1, 2, 2)
    adapters = nn.ModuleDict({"vehicle": ScaleAdapter(1.0), "rsu": ScaleAdapter(3.0)})
    adapted = {
        "vehicle": adapters["vehicle"](veh),
        "rsu": adapters["rsu"](rsu),
    }
    packed, record_len = repack_agent_features(adapted, data_dict, batch_size=1)
    bypass, _ = repack_agent_features({"vehicle": veh, "rsu": rsu}, data_dict, batch_size=1)
    assert record_len.tolist() == [2]
    assert torch.allclose(packed[0], bypass[0])
    assert not torch.allclose(packed[1], bypass[1])
    loss = packed.sum()
    loss.backward()
    assert adapters["rsu"].scale.grad is not None
    assert adapters["rsu"].scale.grad.abs().item() > 0


def test_lss_checksum_after_optimizer_step() -> None:
    model = DummyDet()
    freeze_lss_encoders(model)
    assert_lss_frozen(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.1)
    assert_lss_not_in_optimizer(model, optimizer)
    before = lss_parameter_checksum(model)
    x = torch.randn(1, 4, 8, 8)
    loss = model.backbone(x).sum() + model.cls_head(x).sum()
    loss.backward()
    assert_no_lss_grad(model)
    optimizer.step()
    after = lss_parameter_checksum(model)
    assert abs(after - before) < 1e-6, (before, after)
    assert "veh_models" not in grouped_trainable_parameters(model)


if __name__ == "__main__":
    test_extract_state_dict_formats()
    test_heal_style_explicit_load()
    test_local_branch_remap()
    test_freeze_lss()
    test_repack_order()
    test_stamp_adapter_reaches_pack()
    test_lss_checksum_after_optimizer_step()
    print("checkpoint / repack tests passed")

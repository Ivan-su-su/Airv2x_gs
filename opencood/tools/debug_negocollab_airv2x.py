# -*- coding: utf-8 -*-
"""Import / freeze / gradient smoke checks for NegoCollab AirV2X modules."""

from __future__ import annotations

import torch
from torch import nn

from opencood.models.negocollab_modules.comm_in_pub import ComminPub
from opencood.models.negocollab_modules.converter import Converter, CrossdomianConverter
from opencood.models.negocollab_modules.resize_net import ResizeNet
from opencood.utils.airv2x_freeze import grouped_trainable_parameters, set_module_trainable


def _comm_args() -> dict:
    return {
        "local_range": [-140.8, -40, -3, 140.8, 40, 1],
        "modality": "vehicle",
        "unify_parameters": {"C_uni": 16, "granularity_H": 0.8, "granularity_W": 0.8},
        "resizer": {"method": "conv", "local_dim": 16},
        "recombiner": {"core_method": "identity"},
        "converter": {
            "num_of_blocks": 1,
            "dim": 16,
            "window_size": 8,
            "heads": 4,
            "drop_out": 0.0,
        },
        "prompt_generator": {"core_method": "identity"},
    }


def test_send_receive_shapes() -> None:
    comm = ComminPub(_comm_args())
    feat = torch.randn(3, 16, 16, 32)
    sent = comm.send(feat)
    assert sent.shape[0] == 3
    affine = torch.eye(3)[:2].view(1, 1, 1, 2, 3).repeat(1, 3, 3, 1, 1)
    record_len = torch.tensor([3])
    comm.local_feature = comm.local_feature
    restored = comm.receive(sent, record_len, affine)
    assert restored.shape[0] == 3


def test_stage_freeze_name_patterns() -> None:
    comm = ComminPub(_comm_args())
    set_module_trainable(comm, False)
    for name, param in comm.named_parameters():
        if any(token in name for token in ("receiver", "local_prompt_generator", "unify2local")):
            param.requires_grad = True
    groups = grouped_trainable_parameters(comm)
    assert groups, groups
    trainable = [n for n, p in comm.named_parameters() if p.requires_grad]
    assert any("receiver" in n or "unify2local" in n or "local_prompt" in n for n in trainable)
    frozen_sender = [
        n
        for n, p in comm.named_parameters()
        if (not p.requires_grad) and "sender" in n
    ]
    assert frozen_sender


def test_resize_and_converter() -> None:
    resizer = ResizeNet(8, 16)
    x = torch.randn(2, 8, 10, 12)
    y = resizer(x, (16, 16, 24))
    assert y.shape == (2, 16, 16, 24)
    conv = Converter(
        {"num_of_blocks": 1, "dim": 16, "window_size": 8, "heads": 4, "drop_out": 0.0}
    )
    z = conv(torch.randn(2, 16, 16, 24))
    assert z.shape == (2, 16, 16, 24)
    cross = CrossdomianConverter(
        {"num_of_blocks": 1, "dim": 16, "window_size": 8, "heads": 4, "drop_out": 0.0}
    )
    out = cross(z, z)
    assert out.shape == z.shape


def test_receiver_gets_grad() -> None:
    comm = ComminPub(_comm_args())
    set_module_trainable(comm, False)
    for name, param in comm.named_parameters():
        if "receiver" in name or "unify2local" in name:
            param.requires_grad = True
    feat = torch.randn(2, 16, 16, 32, requires_grad=False)
    sent = comm.send(feat)
    affine = torch.eye(3)[:2].view(1, 1, 1, 2, 3).repeat(1, 2, 2, 1, 1)
    restored = comm.receive(sent, torch.tensor([2]), affine)
    loss = restored.pow(2).mean()
    loss.backward()
    grads = [
        (n, p.grad.abs().sum().item())
        for n, p in comm.named_parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert any(g > 0 for _, g in grads), grads
    assert any("receiver" in n or "unify2local" in n for n, _ in grads)
    sender_grads = [
        n
        for n, p in comm.named_parameters()
        if (not p.requires_grad) and p.grad is not None and p.grad.abs().sum().item() != 0
    ]
    assert not sender_grads, sender_grads


def test_nego_stage_freeze_patterns() -> None:
    """Stage-1 trains send/receive; Stage-2 trains receiver-side names only."""
    comm = ComminPub(_comm_args())
    set_module_trainable(comm, True)
    stage1 = [n for n, p in comm.named_parameters() if p.requires_grad]
    assert any("sender" in n for n in stage1)
    assert any("receiver" in n or "unify2local" in n for n in stage1)

    set_module_trainable(comm, False)
    for name, param in comm.named_parameters():
        if any(token in name for token in ("receiver", "local_prompt_generator", "unify2local")):
            param.requires_grad = True
    stage2 = [n for n, p in comm.named_parameters() if p.requires_grad]
    assert stage2
    assert all("sender_aligner" not in n and "sender_recombiner" not in n for n in stage2)
    assert any("receiver" in n or "unify2local" in n for n in stage2)


if __name__ == "__main__":
    test_resize_and_converter()
    test_send_receive_shapes()
    test_stage_freeze_name_patterns()
    test_receiver_gets_grad()
    test_nego_stage_freeze_patterns()
    print("negocollab module tests passed")

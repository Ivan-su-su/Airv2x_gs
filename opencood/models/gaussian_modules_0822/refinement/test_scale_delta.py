"""Scale residual: logit 0 → multiplier 1, range [lo, hi]."""

from __future__ import annotations

import torch

from opencood.models.gaussian_modules_0822.refinement.refine_heads import (
    AgentRefineHeads,
)


def test_scale_delta_identity_and_range() -> None:
    head = AgentRefineHeads(
        hidden_dim=8,
        unit_xyz=(1.0, 1.0, 1.0),
        scale_delta_range=(-0.3, 0.1),
        rotation_max_angle=0.2,
    )
    n = 4
    feat = torch.zeros(n, 8)
    mean = torch.zeros(n, 3)
    scale = torch.ones(n, 3)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, -1).contiguous()
    with torch.no_grad():
        head.scale_head.weight.zero_()
        head.scale_head.bias.zero_()
        out = head(feat, mean, scale, quat)
    assert torch.allclose(out["scale"], scale)

    with torch.no_grad():
        head.scale_head.bias.fill_(-80.0)
        shrink = head(feat, mean, scale, quat)["scale"]
        head.scale_head.bias.fill_(80.0)
        grow = head(feat, mean, scale, quat)["scale"]
    assert torch.allclose(shrink, scale * torch.exp(torch.tensor(-0.3)), atol=1e-5)
    assert torch.allclose(grow, scale * torch.exp(torch.tensor(0.1)), atol=1e-5)


if __name__ == "__main__":
    test_scale_delta_identity_and_range()
    print("scale delta mapping ok")

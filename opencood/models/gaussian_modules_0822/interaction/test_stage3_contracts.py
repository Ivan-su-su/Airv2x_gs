"""Stage-3 residual contracts, routing, ordering, and optimizer grouping."""

from __future__ import annotations

import torch
from torch import nn

from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.cross_agent.adapter import AgentResidualAdapter
from opencood.models.gaussian_modules_0822.interaction.gaussian_context import (
    build_gaussian_context_interaction,
)
from opencood.tools.train_utils import setup_optimizer


def _set(n: int, agent: str, dim: int = 8, mean=None) -> GaussianSet:
    if mean is None:
        mean = torch.arange(n, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
        mean[:, 1] = 0.0
        mean[:, 2] = 0.0
    feat = torch.zeros(n, dim)
    feat[:, 0] = torch.arange(n, dtype=torch.float32)
    return GaussianSet(
        mean=mean,
        scale=torch.ones(n, 3),
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, -1).contiguous(),
        feature=feat,
        batch_index=torch.zeros(n, dtype=torch.long),
        agent=agent,
        frame="ego",
    )


def test_stage2_adapter_is_single_residual() -> None:
    adapter = AgentResidualAdapter(feature_dim=8, bottleneck=8)
    x = torch.randn(5, 8)
    with torch.no_grad():
        adapter.adapters["vehicle"][0].weight.zero_()
        adapter.adapters["vehicle"][0].bias.zero_()
        adapter.adapters["vehicle"][2].weight.zero_()
        adapter.adapters["vehicle"][2].bias.fill_(0.25)
    y = adapter(x, "vehicle")
    assert torch.allclose(y, x + 0.25)
    assert adapter.scaled is False


def test_stage3_adapter_gamma_init() -> None:
    adapter = AgentResidualAdapter(feature_dim=8, bottleneck=8, gamma_init=0.1)
    assert torch.allclose(adapter.gamma.detach(), torch.full((8,), 0.1))
    x = torch.randn(4, 8)
    y = adapter(x, "drone")
    assert y.shape == x.shape
    # normalized delta has unit variance per channel before gamma
    delta = y - x
    assert 0.05 < delta.std() < 0.2
    assert not torch.equal(y, x + adapter.adapters["drone"](x))


def test_identity_and_index_order() -> None:
    cfg = {
        "voxel_size": [1.0, 1.0, 1.0],
        "point_cloud_range": [-0.5, -0.5, -0.5, 7.5, 1.5, 1.5],
        "adapter_bottleneck": 8,
        "bev_n_sigma": 3.0,
    }
    model = build_gaussian_context_interaction(
        feature_dim=8,
        stage3_cfg=cfg,
        attn_cfg={"heads": 4, "ffn_dim": 16, "stage3_self_layers": 1, "stage3_cross_layers": 1},
    )
    with torch.no_grad():
        model.fusion_gamma.zero_()
        model.adapter.gamma.zero_()
        for block in list(model.self_blocks) + list(model.cross_blocks):
            block.forward = lambda query, context=None, mask=None: query
    veh = _set(4, "vehicle")
    drone = _set(3, "drone", mean=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]))
    pooled = {a: model.pool(gs) for a, gs in (("vehicle", veh), ("drone", drone))}
    merged, _ = model._merge(pooled)
    out, _bev = model({"vehicle": veh, "drone": drone})
    assert torch.allclose(out.feature, merged.feature, atol=1e-5)
    # synthetic index: feature[i,0] must return to the same row
    assert torch.allclose(out.feature[:, 0], merged.feature[:, 0], atol=1e-5)
    assert out.n_gaussians == merged.n_gaussians


def test_source_adapter_once() -> None:
    cfg = {
        "voxel_size": [10.0, 10.0, 10.0],
        "point_cloud_range": [-20, -20, -20, 20, 20, 20],
        "adapter_bottleneck": 8,
    }
    model = build_gaussian_context_interaction(
        feature_dim=8,
        stage3_cfg=cfg,
        attn_cfg={"heads": 4, "ffn_dim": 16},
    )
    calls = []
    orig = model.adapter.forward

    def spy(feature, agent):
        calls.append((agent, int(feature.shape[0])))
        return orig(feature, agent)

    model.adapter.forward = spy
    veh = _set(2, "vehicle")
    drone = _set(2, "drone", mean=torch.tensor([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]))
    model({"vehicle": veh, "drone": drone})
    agents = [c[0] for c in calls]
    assert agents.count("vehicle") == 1
    assert agents.count("drone") == 1


def test_optimizer_no_decay_on_norm_and_bias() -> None:
    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stage3 = build_gaussian_context_interaction(
                feature_dim=8,
                stage3_cfg={
                    "voxel_size": [4.0, 4.0, 4.0],
                    "point_cloud_range": [-8, -8, -8, 8, 8, 8],
                    "adapter_bottleneck": 8,
                },
                attn_cfg={"heads": 4, "ffn_dim": 16},
            )

    hypes = {
        "optimizer": {
            "core_method": "Adam",
            "lr": 1e-3,
            "args": {"weight_decay": 1e-4, "eps": 1e-10},
        }
    }
    model = Toy()
    opt = setup_optimizer(hypes, model)
    assert len(opt.param_groups) == 2
    decay_ids = {id(p) for p in opt.param_groups[0]["params"]}
    nod_ids = {id(p) for p in opt.param_groups[1]["params"]}
    assert opt.param_groups[0]["weight_decay"] == 1e-4
    assert opt.param_groups[1]["weight_decay"] == 0.0
    named = dict(model.named_parameters())
    assert id(named["stage3.self_blocks.0.query_norm.weight"]) in nod_ids
    assert id(named["stage3.self_blocks.0.context_norm.weight"]) in nod_ids
    assert id(named["stage3.self_blocks.0.ffn_norm.weight"]) in nod_ids
    assert id(named["stage3.adapter.gamma"]) in nod_ids
    assert id(named["stage3.fusion_gamma"]) in nod_ids
    assert id(named["stage3.fusion_mlp.weight"]) in decay_ids
    assert id(named["stage3.adapter.adapters.vehicle.0.weight"]) in decay_ids
    assert id(named["stage3.fusion_mlp.bias"]) in nod_ids
    assert not (decay_ids & nod_ids)


def test_scale_delta_range_shared() -> None:
    from opencood.models.gaussian_modules_0822.refinement.refine_heads import (
        DEFAULT_UNIT_XYZ,
        GaussianRefineHeads,
    )

    heads = GaussianRefineHeads(hidden_dim=8, unit_xyz=DEFAULT_UNIT_XYZ)
    rng = heads.heads["vehicle"].scale_delta_range
    assert torch.allclose(rng, torch.tensor([-0.20, 0.05]))


if __name__ == "__main__":
    test_stage2_adapter_is_single_residual()
    test_stage3_adapter_gamma_init()
    test_identity_and_index_order()
    test_source_adapter_once()
    test_optimizer_no_decay_on_norm_and_bias()
    test_scale_delta_range_shared()
    print("STAGE3 CONTRACTS OK")

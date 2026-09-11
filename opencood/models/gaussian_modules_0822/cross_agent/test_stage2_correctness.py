"""Stage-2 adapter semantics, invalid identity, and parameter ownership."""

from __future__ import annotations

import sys

import torch

from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.base.projection import _camera_batch_index
from opencood.models.gaussian_modules_0822.cross_agent.adapter import AgentResidualAdapter
from opencood.models.gaussian_modules_0822.cross_agent.interaction import (
    CrossAgentInteraction,
    build_cross_agent_interactions,
)
from opencood.models.gaussian_modules_0822.interaction.gaussian_context import (
    build_gaussian_context_interaction,
)
from opencood.models.gaussian_modules_0822.refinement.gaussian_refiner import (
    GaussianRefiner,
)
from opencood.models.gaussian_modules_0822.base.gaussian_geometry_encoder import (
    GaussianGeometryEncoder,
)


def _set(n: int, agent: str, mean=None) -> GaussianSet:
    if mean is None:
        mean = torch.tensor([[0.0, 0.0, 5.0]]).expand(n, -1).contiguous()
    return GaussianSet(
        mean=mean,
        scale=torch.ones(n, 3),
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, -1).contiguous(),
        feature=torch.randn(n, 8),
        view_index=torch.zeros(n, dtype=torch.long),
        batch_index=torch.zeros(n, dtype=torch.long),
        agent=agent,
        frame="ego",
    )


def _geom(n_flat: int = 2) -> dict:
    eye = torch.eye(4).unsqueeze(0).expand(n_flat, -1, -1).contiguous()
    lidar = eye.unsqueeze(0).clone()
    lidar[0, :, 2, 3] = 0.0
    return {
        "n_flat": n_flat,
        "n_views": n_flat,
        "n_cav": 1,
        "image_hw": (32, 48),
        "lidar2image_cam_bv": lidar.clone(),
        "lidar2image_ego_bv": lidar,
        "img_aug_matrix_bv": eye.unsqueeze(0).clone(),
    }


def test_invalid_query_is_exact_identity() -> None:
    torch.manual_seed(0)
    drone = _set(4, "drone")
    feat0 = drone.feature.clone()
    mean0 = drone.mean.clone()
    scale0 = drone.scale.clone()
    quat0 = drone.quaternion.clone()
    module = CrossAgentInteraction(feature_dim=8, refine=True)
    calls = []
    orig = module.adapter.forward

    def spy(feature, agent):
        calls.append(agent)
        return orig(feature, agent)

    module.adapter.forward = spy

    def dead_project(points, geometry, points_frame="ego", feature_maps=None, feature_hw=None):
        n, s = int(points.shape[0]), int(points.shape[1])
        v = int(feature_maps.shape[0])
        c = int(feature_maps.shape[1])
        return {
            "sampled_features": feature_maps.new_zeros(1, v, n, s, c),
            "valid_mask": torch.zeros(1, v, n, s, dtype=torch.bool, device=points.device),
        }

    module.projector.forward = dead_project
    sources = {"vehicle": {"features": torch.randn(2, 8, 8, 8), "geometry": _geom(2)}}
    out = module({"drone": drone}, sources)
    got = out["drone"]
    assert torch.equal(got.feature, feat0)
    assert torch.equal(got.mean, mean0)
    assert torch.equal(got.scale, scale0)
    assert torch.equal(got.quaternion, quat0)
    assert calls == []
    assert module.last_stats["drone_from_ground"]["n_valid"] == 0
    print("invalid identity PASS")


def test_adapter_runs_on_source_tokens_not_query() -> None:
    torch.manual_seed(1)
    drone = _set(3, "drone")
    query0 = drone.feature.clone()
    module = CrossAgentInteraction(feature_dim=8, refine=False)
    agents = []
    orig = module.adapter.forward

    def spy(feature, agent):
        agents.append((agent, tuple(feature.shape)))
        return orig(feature, agent)

    module.adapter.forward = spy

    def some_valid(points, geometry, points_frame="ego", feature_maps=None, feature_hw=None):
        n, s = int(points.shape[0]), int(points.shape[1])
        v = int(feature_maps.shape[0])
        c = int(feature_maps.shape[1])
        mask = torch.zeros(1, v, n, s, dtype=torch.bool, device=points.device)
        mask[0, 0, :, 0] = True
        sampled = torch.randn(1, v, n, s, c, device=points.device, dtype=feature_maps.dtype)
        return {"sampled_features": sampled, "valid_mask": mask}

    module.projector.forward = some_valid
    sources = {
        "vehicle": {"features": torch.randn(2, 8, 8, 8), "geometry": _geom(2)},
        "rsu": {"features": torch.randn(1, 8, 8, 8), "geometry": _geom(1)},
    }
    out = module({"drone": drone}, sources, interaction_type="drone_from_ground")
    names = [a for a, _ in agents]
    assert names.count("vehicle") == 1
    assert names.count("rsu") == 1
    assert "drone" not in names
    assert not torch.allclose(out["drone"].feature, query0)
    print("source adapter PASS", names)


def test_mixed_valid_subset_skips_invalid() -> None:
    torch.manual_seed(2)
    drone = _set(4, "drone")
    feat0 = drone.feature.clone()
    module = CrossAgentInteraction(feature_dim=8, refine=False)
    seen_n = []

    def mixed(points, geometry, points_frame="ego", feature_maps=None, feature_hw=None):
        n, s = int(points.shape[0]), int(points.shape[1])
        v = int(feature_maps.shape[0])
        c = int(feature_maps.shape[1])
        mask = torch.zeros(1, v, n, s, dtype=torch.bool, device=points.device)
        mask[0, 0, :2, 0] = True
        sampled = torch.randn(1, v, n, s, c, device=points.device, dtype=feature_maps.dtype)
        return {"sampled_features": sampled, "valid_mask": mask}

    orig_block = module.attention["drone_from_ground"][0].forward

    def spy_block(query, context=None, mask=None):
        seen_n.append(int(query.shape[0]))
        return orig_block(query, context, mask)

    module.projector.forward = mixed
    module.attention["drone_from_ground"][0].forward = spy_block
    sources = {"vehicle": {"features": torch.randn(1, 8, 8, 8), "geometry": _geom(1)}}
    out = module({"drone": drone}, sources, interaction_type="drone_from_ground")
    assert seen_n == [2]
    assert torch.equal(out["drone"].feature[2:], feat0[2:])
    assert not torch.allclose(out["drone"].feature[:2], feat0[:2])
    print("mixed valid subset PASS")


def test_parameter_ownership() -> None:
    encoder = GaussianGeometryEncoder(geo_dim=16, hidden_dim=16)
    r1 = GaussianRefiner(encoder, feature_dim=8, cfg={"enabled": True, "geometry": {"geo_dim": 16}})
    r2 = GaussianRefiner(encoder, feature_dim=8, cfg={"enabled": True, "geometry": {"geo_dim": 16}})
    a2 = AgentResidualAdapter(feature_dim=8, bottleneck=8)
    a3 = AgentResidualAdapter(feature_dim=8, bottleneck=8)
    s2 = build_cross_agent_interactions(
        feature_dim=8, adapter=a2, refiner=r2, refine=True, attn_cfg={"stage2_layers": 1}
    )
    s3 = build_gaussian_context_interaction(
        feature_dim=8,
        adapter=a3,
        attn_cfg={"stage3_self_layers": 1, "stage3_cross_layers": 1, "heads": 4, "ffn_dim": 16},
        stage3_cfg={"adapter_bottleneck": 8, "voxel_size": [4.0, 4.0, 4.0],
                    "point_cloud_range": [-20, -20, -20, 20, 20, 20]},
    )
    assert id(r1) != id(r2)
    assert id(s2.refiner) != id(s3)
    assert id(s2.adapter) != id(s3.adapter)
    for agent in ("vehicle", "rsu", "drone"):
        assert id(s2.adapter.adapters[agent]) != id(s3.adapter.adapters[agent])
    s2_names = [n for n, _ in s2.named_parameters()]
    assert any(n.startswith("adapter.adapters.vehicle") for n in s2_names)
    assert any(n.startswith("refiner.") for n in s2_names)
    assert not any(n.startswith("cross_agent") for n in s2_names)
    print("ownership PASS")


def test_camera_batch_index() -> None:
    like = torch.zeros(1)
    idx = _camera_batch_index(
        {"batch_idxs": torch.tensor([0, 0, 1])}, n_flat=6, n_views=2, like=like
    )
    assert idx.tolist() == [0, 0, 0, 0, 1, 1]
    view_index = torch.tensor([0, 3, 5])
    assert idx[view_index].tolist() == [0, 0, 1]
    print("batch_index PASS")


if __name__ == "__main__":
    test_invalid_query_is_exact_identity()
    test_adapter_runs_on_source_tokens_not_query()
    test_mixed_valid_subset_skips_invalid()
    test_parameter_ownership()
    test_camera_batch_index()
    print("STAGE2 CORRECTNESS OK")
    sys.exit(0)

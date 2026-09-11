"""Stage-1 same-CAV routing: tokens are V_cav * S, never all agent cameras."""

from __future__ import annotations

from typing import Any, Dict

import torch

from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.interaction.intra_view import (
    IntraViewInteraction,
)
from opencood.models.gaussian_modules_0822.sampling.gaussian_offset_sampler import (
    DEFAULT_NUM_OFFSETS,
)


def test_intra_view_tokens_are_same_cav_not_all_vehicles() -> None:
    n_cav, n_views, channels = 5, 6, 8
    n_samples = 2 * DEFAULT_NUM_OFFSETS + 1
    n_flat = n_cav * n_views
    eye = torch.eye(4).unsqueeze(0).expand(n_flat, -1, -1).contiguous()
    lidar2image = eye.unsqueeze(0).clone()
    for cav in range(n_cav):
        lidar2image[0, cav * n_views : (cav + 1) * n_views, 0, 3] = float(cav)
    geometry: Dict[str, Any] = {
        "n_flat": n_flat,
        "n_views": n_views,
        "n_cav": n_cav,
        "image_hw": (32, 48),
        "lidar2image_cam_bv": lidar2image.clone(),
        "lidar2image_ego_bv": lidar2image,
        "img_aug_matrix_bv": eye.unsqueeze(0).clone(),
    }
    n = 5
    gaussians = GaussianSet(
        mean=torch.tensor([[0.0, 0.0, 5.0]]).expand(n, -1).contiguous(),
        scale=torch.ones(n, 3),
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, -1).contiguous(),
        feature=torch.randn(n, channels),
        p_fg=torch.ones(n),
        view_index=torch.tensor([0, 0, 15, 15, 15]),
        batch_index=torch.zeros(n, dtype=torch.long),
        agent="vehicle",
        frame="ego",
    )
    maps = torch.randn(n_flat, channels, 8, 8)
    module = IntraViewInteraction(feature_dim=channels, refine=False)
    orig_project = module.projector.project
    orig_block = module.blocks[0].forward
    project_calls = []
    n_tokens_seen = []
    n_query_seen = []

    def spy_project(points, lidar2image, image_aug_matrix, image_hw, feature_hw=None):
        project_calls.append(
            (tuple(points.shape), tuple(lidar2image.shape), lidar2image[:, 0, 0, 3].tolist())
        )
        return orig_project(
            points, lidar2image, image_aug_matrix, image_hw, feature_hw=feature_hw
        )

    def spy_block(query, context=None, mask=None):
        n_tokens_seen.append(int(context.shape[1]))
        n_query_seen.append(int(query.shape[0]))
        return orig_block(query, context, mask)

    module.projector.project = spy_project
    module.blocks[0].forward = spy_block
    out = module(gaussians, maps, geometry, agent="vehicle")
    assert out.n_gaussians == 5
    (out.feature.sum()).backward()
    named = {n: p for n, p in module.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert any(n.startswith("blocks.0") for n in named)
    assert any(n.startswith("blocks.1") for n in named)
    assert len(project_calls) == 1
    assert project_calls[0][0] == (n_cav, 3, n_samples, 3)
    assert project_calls[0][1] == (n_cav, n_views, 4, 4)
    assert n_tokens_seen[0] == n_views * n_samples
    assert n_views * n_samples == 42
    assert n_query_seen == [5]
    assert module.last_stats["n_real"] == 5
    assert module.last_stats["n_padded"] == n_cav * 3
    assert module.last_stats["n_cav_nonempty"] == 2
    assert len(module.blocks) == 2


def test_interaction_block_mask_and_self() -> None:
    from opencood.models.gaussian_modules_0822.attention.interaction_block import (
        GaussianInteractionBlock,
    )

    torch.manual_seed(0)
    block = GaussianInteractionBlock(dim=8, heads=4, ffn_dim=16)
    query = torch.randn(3, 8)
    context = torch.randn(3, 5, 8)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    out_empty = block(query, context, mask)
    assert out_empty.shape == query.shape
    assert torch.allclose(out_empty, query)
    mask[0, :3] = True
    out = block(query, context, mask)
    assert not torch.allclose(out[0], query[0])
    assert torch.allclose(out[1], query[1])
    self_out = block(query)
    assert self_out.shape == query.shape
    query2 = torch.randn(3, 8, requires_grad=True)
    mask2 = torch.ones(3, 5, dtype=torch.bool)
    block(query2, context, mask2).sum().backward()
    assert block.q_proj.weight.grad.abs().sum() > 0


if __name__ == "__main__":
    test_intra_view_tokens_are_same_cav_not_all_vehicles()
    test_interaction_block_mask_and_self()
    print("intra_view same-CAV routing ok")

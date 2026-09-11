"""Stage-3 smoke test: pooling -> context interaction -> BEV + backward.

Checks:
1. cfg-driven construction (voxel size / range read at init, not forward)
2. raw vs pooled Gaussian counts
3. BEV shape / layout [B, C, ny, nx] matching the detection grid
4. batch isolation: zeroing one sample's features never changes the other
5. backward: split / self / cross / adapter / fusion all receive grads

Usage:
    cd /mnt/home/suyi/AirV2X-Perception_gaussian && \
    PYTHONPATH=$PWD python \
    opencood/models/gaussian_modules_0822/interaction/test_stage3_smoke.py
"""

from __future__ import annotations

import sys

import torch

from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.fusion.gaussian_voxel_pool import (
    GaussianVoxelPool,
)
from opencood.models.gaussian_modules_0822.interaction.gaussian_context import (
    build_gaussian_context_interaction,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STAGE3_CFG = {
    "voxel_size": [0.4, 0.4, 0.125],
    "point_cloud_range": [-140.8, -40.0, -150.0, 140.8, 40.0, 30.0],
    "num_heads": 4,
    "adapter_bottleneck": 32,
    "bev_n_sigma": 3.0,
}


def make_set(agent: str, n: int, n_batch: int, feature_dim: int) -> GaussianSet:
    """Random Stage-2-like Gaussian set with ego-frame means."""
    mean = torch.rand(n, 3) * torch.tensor([80.0, 40.0, 10.0]) - torch.tensor(
        [40.0, 20.0, 2.0]
    )
    scale = torch.rand(n, 3) * 0.8 + 0.2
    quaternion = torch.randn(n, 4)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    feature = torch.randn(n, feature_dim)
    batch = torch.randint(0, n_batch, (n,))
    return GaussianSet(
        mean=mean,
        scale=scale,
        quaternion=quaternion,
        feature=feature,
        batch_index=batch,
        agent=agent,
        frame="ego",
    )


def build_inputs(n_batch: int, feature_dim: int):
    """Deterministic per-agent Gaussian sets."""
    torch.manual_seed(0)
    return {
        "vehicle": make_set("vehicle", 4000, n_batch, feature_dim),
        "rsu": make_set("rsu", 1200, n_batch, feature_dim),
        "drone": make_set("drone", 600, n_batch, feature_dim),
    }


def main() -> None:
    feature_dim = 64
    n_batch = 2
    device = torch.device(DEVICE)

    # --- voxel pool alone -------------------------------------------------
    pool = GaussianVoxelPool(
        voxel_size=STAGE3_CFG["voxel_size"],
        point_cloud_range=STAGE3_CFG["point_cloud_range"],
    )
    for agent, gs in build_inputs(n_batch, feature_dim).items():
        pooled = pool(gs)
        print(f"[pool] {agent}: raw={gs.n_gaussians} pooled={pooled.n_gaussians}")
        assert pooled.n_gaussians <= gs.n_gaussians
        assert torch.allclose(
            pooled.quaternion.norm(dim=-1), torch.ones(pooled.n_gaussians), atol=1e-5
        )
        assert torch.isfinite(pooled.feature).all()

    # --- full Stage-3, cfg-driven -----------------------------------------
    model = build_gaussian_context_interaction(
        feature_dim=feature_dim, stage3_cfg=STAGE3_CFG
    ).to(device)
    sets = {a: gs.to(device) for a, gs in build_inputs(n_batch, feature_dim).items()}
    out, bev = model(sets)

    print(f"[stage3] merged pooled={out.n_gaussians} bev={tuple(bev.shape)}")
    assert bev.shape == (n_batch, feature_dim, 200, 704)  # [B, C, ny, nx]
    assert torch.isfinite(bev).all()
    assert bev.grad_fn is not None  # splat is inside the autograd graph
    assert out.batch_index.unique().numel() == n_batch

    # --- batch isolation: batch-1 BEV independent of batch-0 features ------
    perturbed = {
        a: gs.replace_feature(
            torch.where(
                (gs.batch_index == 0).unsqueeze(-1),
                torch.zeros_like(gs.feature),
                gs.feature,
            )
        )
        for a, gs in sets.items()
    }
    _, bev_ref = model(perturbed)
    delta = (bev[1] - bev_ref[1]).abs().max().item()
    print(f"[isolation] batch-1 max delta after zeroing batch-0: {delta:.3e}")
    # CUDA index_add_ float nondeterminism is ~1e-7; real leakage was ~1e-2.
    assert delta < 1e-5, "cross-batch leakage in attention or splatting"

    # --- backward ----------------------------------------------------------
    bev.sum().backward()
    grads = [
        name for name, p in model.named_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    expected = ["split_mlp", "self_blocks", "cross_blocks", "adapter.adapters", "fusion_mlp"]
    missing = [e for e in expected if not any(g.startswith(e) for g in grads)]
    print(f"[grad] {len(grads)} params with nonzero grad; missing={missing}")
    assert not missing, f"missing gradients: {missing}"

    pooled_means = torch.cat(
        [model.pool(sets[a]).mean for a in ("vehicle", "rsu", "drone")]
    )
    assert torch.equal(out.mean, pooled_means)
    assert torch.equal(
        out.scale,
        torch.cat([model.pool(sets[a]).scale for a in ("vehicle", "rsu", "drone")]),
    )

    print("STAGE3 SMOKE TEST OK")


if __name__ == "__main__":
    sys.exit(main())

"""Synthetic correctness tests for Gaussian BEV splat."""

from __future__ import annotations

import math

import torch

from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.base.gaussian_transform import (
    reconstruct_covariance,
    rotation_matrix_to_quaternion,
)
from opencood.models.gaussian_modules_0822.fusion.gaussian_bev_splat import (
    GaussianBEVSplat,
)


def _splat() -> GaussianBEVSplat:
    return GaussianBEVSplat(
        feature_dim=1,
        voxel_size=(0.4, 0.4, 0.125),
        point_cloud_range=(-8.0, -8.0, -3.0, 8.0, 8.0, 1.0),
        n_sigma=3.0,
    )


def _one(
    scale: torch.Tensor,
    quaternion: torch.Tensor,
    feature: float = 1.0,
) -> GaussianSet:
    return GaussianSet(
        mean=torch.zeros(1, 3),
        scale=scale.reshape(1, 3),
        quaternion=quaternion.reshape(1, 4),
        feature=torch.full((1, 1), float(feature)),
        batch_index=torch.zeros(1, dtype=torch.long),
        agent="vehicle",
        frame="ego",
    )


def _energy_axes(bev: torch.Tensor) -> tuple:
    """Second-moment of |BEV| about the grid center, along x and y."""
    amp = bev[0, 0].abs()
    ny, nx = amp.shape
    ys = torch.arange(ny, dtype=torch.float32) - (ny - 1) * 0.5
    xs = torch.arange(nx, dtype=torch.float32) - (nx - 1) * 0.5
    mass = amp.sum().clamp_min(1.0e-8)
    var_y = (amp.sum(dim=1) * (ys ** 2)).sum() / mass
    var_x = (amp.sum(dim=0) * (xs ** 2)).sum() / mass
    return float(var_x.item()), float(var_y.item())


def test_isotropic_quaternion_invariant() -> None:
    splat = _splat()
    scale = torch.tensor([1.0, 1.0, 1.0])
    q_i = torch.tensor([1.0, 0.0, 0.0, 0.0])
    yaw = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    q_yaw = rotation_matrix_to_quaternion(yaw.unsqueeze(0))[0]
    a = splat(_one(scale, q_i))
    b = splat(_one(scale, q_yaw))
    assert torch.allclose(a, b, atol=1.0e-5)


def test_horizontal_anisotropic_follows_quaternion() -> None:
    splat = _splat()
    scale = torch.tensor([0.5, 0.5, 5.0])
    # Long principal axis (column 2) -> ego X.
    r_x = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    # Long principal axis -> ego Y.
    r_y = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    q_x = rotation_matrix_to_quaternion(r_x.unsqueeze(0))[0]
    q_y = rotation_matrix_to_quaternion(r_y.unsqueeze(0))[0]
    vx, vy = _energy_axes(splat(_one(scale, q_x)))
    wx, wy = _energy_axes(splat(_one(scale, q_y)))
    assert vx > vy * 2.0, f"long-X expected vx>>vy, got {vx:.3f} vs {vy:.3f}"
    assert wy > wx * 2.0, f"long-Y expected wy>>wx, got {wx:.3f} vs {wy:.3f}"


def test_vertical_anisotropic_stays_compact() -> None:
    splat = _splat()
    scale = torch.tensor([0.5, 0.5, 5.0])
    q_i = torch.tensor([1.0, 0.0, 0.0, 0.0])
    r_x = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    q_x = rotation_matrix_to_quaternion(r_x.unsqueeze(0))[0]
    compact_x, compact_y = _energy_axes(splat(_one(scale, q_i)))
    long_x, _ = _energy_axes(splat(_one(scale, q_x)))
    assert compact_x < long_x * 0.5, (
        f"vertical should be compact in XY, {compact_x:.3f} vs long-X {long_x:.3f}"
    )
    assert compact_y < long_x * 0.5


def test_sigma_xy_matches_gaussian_covariance() -> None:
    scale = torch.tensor([[0.5, 0.8, 3.0]])
    quat = torch.tensor([[0.8, 0.2, 0.4, 0.4]])
    quat = quat / quat.norm(dim=-1, keepdim=True)
    gs = _one(scale[0], quat[0])
    expected = gs.covariance()[:, :2, :2]
    rebuilt = reconstruct_covariance(gs.scale, gs.quaternion)[:, :2, :2]
    assert torch.allclose(expected, rebuilt, atol=1.0e-6)
    splat = _splat()
    splat(gs)
    # The splat radius is n_sigma * sqrt(diag(Σ_xy)); a z-aligned long
    # axis must not inflate XY the way scale[:, :2] would.
    rx = splat.last_stats["radius_x_max"]
    # σ_xx from the true XY block, not from scale[0].
    sigma_x = float(expected[0, 0, 0].clamp_min(0).sqrt().item())
    cell = 0.4
    expect_r = max(1, int(torch.ceil(torch.tensor(3.0 * sigma_x / cell)).item()))
    assert int(rx) == expect_r


def test_outlier_does_not_expand_all_neighborhoods() -> None:
    """One huge XY Gaussian must not force every Gaussian onto its grid."""
    splat = _splat()
    n_small = 32
    scale = torch.ones(n_small + 1, 3) * 0.4
    scale[-1] = torch.tensor([0.4, 0.4, 8.0])
    r_x = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    q_x = rotation_matrix_to_quaternion(r_x.unsqueeze(0))[0]
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n_small + 1, -1).clone()
    quat[-1] = q_x
    mean = torch.zeros(n_small + 1, 3)
    gs = GaussianSet(
        mean=mean,
        scale=scale,
        quaternion=quat,
        feature=torch.ones(n_small + 1, 1),
        batch_index=torch.zeros(n_small + 1, dtype=torch.long),
        frame="ego",
    )
    splat(gs)
    rx = torch.clamp(
        torch.ceil(
            splat.n_sigma
            * gs.covariance()[:, 0, 0].clamp_min(0).sqrt()
            / splat.cell[0]
        ).long(),
        min=1,
    )
    # Global r_max scheme would evaluate (2*r_max+1)^2 per Gaussian.
    r_max = int(rx.max().item())
    old_pairs = (n_small + 1) * (2 * r_max + 1) ** 2
    new_pairs = int(splat.last_stats["n_candidate"])
    assert new_pairs < old_pairs, f"new {new_pairs} vs global-rmax {old_pairs}"
    assert splat.last_stats["radius_x_median"] < splat.last_stats["radius_x_max"]


def test_feature_gradient() -> None:
    splat = _splat()
    gs = _one(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    gs.feature = gs.feature.clone().requires_grad_(True)
    bev = splat(gs)
    bev.sum().backward()
    assert gs.feature.grad is not None
    assert float(gs.feature.grad.abs().sum().item()) > 0.0


def test_single_gaussian_preserves_radial_decay() -> None:
    """A single Gaussian must keep its absolute Gaussian response.

    With feature = 1 the BEV value at Mahalanobis distance d must equal
    exp(-0.5 d^2). In particular a cell 2 sigma away must read
    exp(-2) ~ 0.1353, NOT ~1 (which the old weight-normalized average
    produced by computing (f * w) / w).
    """
    splat = _splat()  # cell 0.4 m, range [-8, 8]
    # Cell center: (-8) + (ix + 0.5) * 0.4. ix=20 -> x = 0.2.
    gs = GaussianSet(
        mean=torch.tensor([[0.2, 0.2, 0.0]]),
        scale=torch.ones(1, 3),
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        feature=torch.ones(1, 1),
        batch_index=torch.zeros(1, dtype=torch.long),
        agent="vehicle",
        frame="ego",
    )
    # Isotropic unit covariance: sigma_xy = 1 m, so 5 cells = 2 m = 2 sigma.
    assert torch.allclose(
        gs.covariance()[0, :2, :2], torch.eye(2), atol=1.0e-6
    )
    bev = splat(gs)[0, 0]
    assert torch.isfinite(bev).all()
    ix_center, iy_center = 20, 20
    center = float(bev[iy_center, ix_center])
    far = float(bev[iy_center, ix_center + 5])
    expected = math.exp(-2.0)
    assert center > far > 0.0, f"center {center:.4f} far {far:.4f}"
    assert torch.allclose(
        torch.tensor(center), torch.tensor(1.0), atol=1.0e-4, rtol=1.0e-3
    ), f"center should be ~1, got {center}"
    assert torch.allclose(
        torch.tensor(far), torch.tensor(expected), atol=1.0e-4, rtol=1.0e-3
    ), f"2-sigma should be exp(-2)={expected:.4f}, got {far}"


def test_overlapping_gaussians_sum_contributions() -> None:
    """Co-located Gaussians must add directly, not average.

    Two identical Gaussians with features 1 and 2 give center output 3
    under direct summation; the old normalized average gave
    (1 + 2) / 2 = 1.5.
    """
    splat = _splat()
    mean = torch.tensor([[0.2, 0.2, 0.0]])
    scale = torch.ones(1, 3)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    a = GaussianSet(
        mean=mean,
        scale=scale,
        quaternion=quat,
        feature=torch.ones(1, 1),
        batch_index=torch.zeros(1, dtype=torch.long),
        agent="vehicle",
        frame="ego",
    )
    b = GaussianSet(
        mean=mean.clone(),
        scale=scale.clone(),
        quaternion=quat.clone(),
        feature=torch.full((1, 1), 2.0),
        batch_index=torch.zeros(1, dtype=torch.long),
        agent="vehicle",
        frame="ego",
    )
    both = GaussianSet(
        mean=torch.cat([mean, mean]),
        scale=torch.cat([scale, scale]),
        quaternion=torch.cat([quat, quat]),
        feature=torch.tensor([[1.0], [2.0]]),
        batch_index=torch.zeros(2, dtype=torch.long),
        agent="vehicle",
        frame="ego",
    )
    center_i, center_j = 20, 20
    off_i, off_j = 20, 23  # 3 cells = 1.2 m = 1.2 sigma along x
    va = float(splat(a)[0, 0, center_i, center_j])
    vb = float(splat(b)[0, 0, center_i, center_j])
    v_both = float(splat(both)[0, 0, center_i, center_j])
    assert torch.allclose(
        torch.tensor(v_both), torch.tensor(va + vb), atol=1.0e-4, rtol=1.0e-3
    ), f"overlap must sum: {va} + {vb} != {v_both}"
    assert torch.allclose(
        torch.tensor(v_both), torch.tensor(3.0), atol=1.0e-4, rtol=1.0e-3
    ), f"co-located 1 + 2 should give 3, got {v_both}"
    # Off-center cell: identical distance to both, still sums with decay.
    w_off = math.exp(-0.5 * 1.2 ** 2)
    off = float(splat(both)[0, 0, off_i, off_j])
    assert torch.allclose(
        torch.tensor(off), torch.tensor(3.0 * w_off), atol=1.0e-4, rtol=1.0e-3
    ), f"off-center should be 3 * exp(-0.5*1.44)={3.0 * w_off:.4f}, got {off}"
    assert off < v_both


if __name__ == "__main__":
    test_isotropic_quaternion_invariant()
    test_horizontal_anisotropic_follows_quaternion()
    test_vertical_anisotropic_stays_compact()
    test_sigma_xy_matches_gaussian_covariance()
    test_outlier_does_not_expand_all_neighborhoods()
    test_feature_gradient()
    test_single_gaussian_preserves_radial_decay()
    test_overlapping_gaussians_sum_contributions()
    print("gaussian_bev_splat tests ok")

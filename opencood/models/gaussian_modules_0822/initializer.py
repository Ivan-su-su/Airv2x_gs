"""Gaussian initialization from heatmap seeds and agent-specific depth.

Builds uncertainty-aware feature Gaussians. One Gaussian per foreground
cell. No NMS, Top-K, or confidence pruning. After agent→ego, Gaussians
outside the ego ``cav_lidar_range`` are dropped.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional

import torch

from opencood.models.gaussian_modules_0822.base.covariance import assemble_covariance
from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.base.gaussian_filter import (
    GaussianRangeFilter,
)
from opencood.models.gaussian_modules_0822.base.gaussian_init import (
    covariance_to_scale_quaternion,
)
from opencood.models.gaussian_modules_0822.base.projection import (
    CAMERA_GEOMETRY_KEY,
    attach_camera_geometry,
    backproject_optical_z,
    build_camera_geometry,
    image_direction_to_tangent,
)
from opencood.models.gaussian_modules_0822.depth.depth_initializer import (
    categorical_depth_moments,
    drone_depth_moments,
)
from opencood.models.gaussian_modules_0822.depth.uncertainty import (
    LOCAL_WINDOW_M,
    SIGMA_D_DRONE,
)
from opencood.models.gaussian_modules_0822.heatmap.orientation import (
    PCA_RADIUS_PX,
    heatmap_pca_direction,
    pca_cell_radius,
)
from opencood.models.gaussian_modules_0822.heatmap.seed_generator import (
    FG_THRESHOLD,
    foreground_probability,
    generate_seeds,
)
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents

SIGMA_T1_M = 1.0
SIGMA_T2_M = 0.5
DEBUG_SAMPLE_N = 8


class GaussianInitializer:
    """Vectorized Gaussian initialization for vehicle / RSU / drone.

    Args:
        z_bins: Optional metric bin centers per categorical agent
            ``{"vehicle": [D], "rsu": [D]}``.
        fg_threshold: Heatmap occupancy cut.
        local_window_m: Peak window for local depth moments.
        sigma_t1: Fixed major tangent scale (meters).
        sigma_t2: Fixed minor tangent scale (meters).
        sigma_d_drone: Fixed drone ray scale (meters).
        pca_radius_px: Heatmap PCA patch half-width in image pixels.
        range_filter: Ego-range filter applied after agent→ego.
        debug: If True, store a small sanity dict on the last forward.
    """

    def __init__(
        self,
        z_bins: Optional[Mapping[str, torch.Tensor]] = None,
        fg_threshold: float = FG_THRESHOLD,
        local_window_m: float = LOCAL_WINDOW_M,
        sigma_t1: float = SIGMA_T1_M,
        sigma_t2: float = SIGMA_T2_M,
        sigma_d_drone: float = SIGMA_D_DRONE,
        pca_radius_px: float = PCA_RADIUS_PX,
        range_filter: Optional[GaussianRangeFilter] = None,
        debug: bool = False,
    ) -> None:
        self.z_bins = {k: v.detach().float().reshape(-1) for k, v in (z_bins or {}).items()}
        self.fg_threshold = float(fg_threshold)
        self.local_window_m = float(local_window_m)
        self.sigma_t1 = float(sigma_t1)
        self.sigma_t2 = float(sigma_t2)
        self.sigma_d_drone = float(sigma_d_drone)
        self.pca_radius_px = float(pca_radius_px)
        self.range_filter = range_filter
        self.debug = bool(debug)
        self.last_debug: Optional[Dict[str, Any]] = None
        self.last_filter_stats: Optional[Dict[str, Any]] = None
        self.last_pca_by_agent: Dict[str, Dict[str, Any]] = {}

    def set_z_bins(self, agent: str, z_bins: torch.Tensor) -> None:
        """Register categorical depth bins for ``agent``."""
        self.z_bins[str(agent)] = z_bins.detach().float().reshape(-1)

    def forward_agent(
        self,
        agent: str,
        heatmap_logits: torch.Tensor,
        features: torch.Tensor,
        geometry: Mapping[str, Any],
        depth_logits: Optional[torch.Tensor] = None,
        depth_z: Optional[torch.Tensor] = None,
        z_bins: Optional[torch.Tensor] = None,
        agent_to_ego: Optional[torch.Tensor] = None,
    ) -> GaussianSet:
        """Initialize Gaussians for one agent batch (flattened cameras).

        Args:
            agent: ``vehicle``, ``rsu``, or ``drone``.
            heatmap_logits: ``[N, 2, H, W]``.
            features: ``[N, C, H, W]`` F90 (or any aligned feature).
            geometry: Packed ``camera_geometry`` (built once).
            depth_logits: Categorical logits ``[N, D, H, W]`` (vehicle/RSU).
            depth_z: Continuous depth ``[N, H, W]`` (drone).
            z_bins: Optional override of categorical bin centers ``[D]``.
            agent_to_ego: Optional ``[N, 4, 4]`` (or ``[4, 4]``) agent lidar
                → ego. When given, returned Gaussians are in ego.

        Returns:
            ``GaussianSet`` with one Gaussian per FG cell.
        """
        agent = str(agent).lower()
        p_fg = foreground_probability(heatmap_logits)
        seeds = generate_seeds(p_fg, features, fg_threshold=self.fg_threshold)
        n_cam = int(heatmap_logits.shape[0])
        feat_hw = (int(heatmap_logits.shape[2]), int(heatmap_logits.shape[3]))
        image_hw = tuple(geometry["image_hw"])
        if int(geometry["n_flat"]) != n_cam:
            raise ValueError(
                f"{agent}: geometry n_flat={geometry['n_flat']} vs heatmap N={n_cam}"
            )
        n_ch = int(features.shape[1])
        device = features.device
        dtype = features.dtype

        if int(seeds["view_index"].shape[0]) == 0:
            empty = GaussianSet.empty(device, dtype, n_ch, agent=agent)
            if agent_to_ego is not None:
                empty = empty.to_ego(agent_to_ego)
            if self.debug:
                self.last_debug = {"agent": agent, "n": 0, "empty": True}
            return empty

        view_index = seeds["view_index"]
        y_idx = seeds["y_indices"]
        x_idx = seeds["x_indices"]

        if agent == "drone":
            if depth_z is None:
                raise ValueError("drone initialization requires depth_z")
            depth = drone_depth_moments(
                depth_z, view_index, y_idx, x_idx, sigma_d=self.sigma_d_drone
            )
        else:
            bins = z_bins if z_bins is not None else self.z_bins.get(agent)
            if bins is None:
                raise ValueError(f"{agent} initialization requires z_bins")
            if depth_logits is None:
                raise ValueError(f"{agent} initialization requires depth_logits")
            bins = bins.to(device=device, dtype=dtype)
            depth = categorical_depth_moments(
                depth_logits,
                bins,
                view_index,
                y_idx,
                x_idx,
                agent=agent,
                window_m=self.local_window_m,
            )

        depth_mean = depth["depth_mean"]
        depth_var = depth["depth_var"]
        radius_cells = pca_cell_radius(image_hw, feat_hw, self.pca_radius_px)
        self.last_pca_by_agent[agent] = {
            "image_hw": image_hw,
            "feature_hw": feat_hw,
            "stride": float(image_hw[0]) / float(feat_hw[0]),
            "radius_px": self.pca_radius_px,
            "radius_cells": radius_cells,
        }
        v_major, _ = heatmap_pca_direction(
            p_fg, view_index, y_idx, x_idx, radius=radius_cells
        )
        mean, unit_ray, uv = backproject_optical_z(
            x_indices=x_idx,
            y_indices=y_idx,
            depth_z=depth_mean,
            view_index=view_index,
            geometry=geometry,
            feature_hw=feat_hw,
            image_hw=image_hw,
        )
        t1, t2 = image_direction_to_tangent(
            mean=mean,
            unit_ray=unit_ray,
            v_major_xy=v_major,
            depth_z=depth_mean,
            x_indices=x_idx.to(dtype=depth_mean.dtype),
            y_indices=y_idx.to(dtype=depth_mean.dtype),
            view_index=view_index,
            geometry=geometry,
            feature_hw=feat_hw,
            image_hw=image_hw,
        )
        covariance = assemble_covariance(
            unit_ray=unit_ray,
            tangent_major=t1,
            tangent_minor=t2,
            sigma_z_sq=depth_var,
            sigma_t1=self.sigma_t1,
            sigma_t2=self.sigma_t2,
        )
        scale, quaternion = covariance_to_scale_quaternion(covariance)
        batch_index = geometry["camera_batch_index"].to(
            device=device, dtype=torch.long
        )[view_index.long()]
        gaussians = GaussianSet(
            mean=mean,
            scale=scale,
            quaternion=quaternion,
            feature=seeds["feature"],
            p_fg=seeds["p_fg"],
            uv=uv,
            view_index=view_index,
            batch_index=batch_index,
            depth_mean=depth_mean,
            sigma_z=torch.sqrt(depth_var.clamp_min(0.0)),
            agent=agent,
            frame="agent",
        )
        if agent_to_ego is not None:
            gaussians = gaussians.to_ego(agent_to_ego, view_index=view_index)
        if self.range_filter is not None:
            gaussians = self.range_filter(gaussians, agent_type=agent)
            self.last_filter_stats = dict(self.range_filter.last_stats)
        if self.debug:
            self.last_debug = debug_gaussian_set(
                gaussians,
                agent=agent,
                peak_z=depth["peak_z"],
                v_major=v_major,
            )
        return gaussians

    def forward(self, ego_batch: MutableMapping[str, Any]) -> Dict[str, GaussianSet]:
        """Initialize every present agent from tensors already on ``ego_batch``.

        Expects each agent payload to hold ``heatmap_logits``, ``f90``, and
        either ``depth_logits`` (vehicle/RSU) or ``depth_z_mean`` (drone).
        Camera matrices are packed onto the batch once.

        Args:
            ego_batch: Collated ``batch['ego']``.

        Returns:
            ``{agent: GaussianSet}`` in the ego frame.
        """
        attach_camera_geometry(ego_batch)
        out: Dict[str, GaussianSet] = {}
        for agent in present_camera_agents(ego_batch):
            payload = ego_batch[agent]
            geometry = payload[CAMERA_GEOMETRY_KEY]
            if agent == "drone":
                depth_logits, depth_z = None, payload["depth_z_mean"]
            else:
                depth_logits, depth_z = payload["depth_logits"], None
            out[agent] = self.forward_agent(
                agent=agent,
                heatmap_logits=payload["heatmap_logits"],
                features=payload["f90"],
                geometry=geometry,
                depth_logits=depth_logits,
                depth_z=depth_z,
                agent_to_ego=geometry["agent_to_ego"],
            )
        return out


def debug_gaussian_set(
    gaussians: GaussianSet,
    agent: str,
    peak_z: torch.Tensor,
    v_major: torch.Tensor,
    n_samples: int = DEBUG_SAMPLE_N,
) -> Dict[str, Any]:
    """Sanity checks plus a few random samples (no Python loop over all N).

    Args:
        gaussians: Built set.
        agent: Agent name.
        peak_z: ``[N]`` peak depth.
        v_major: ``[N, 2]`` image PCA direction.
        n_samples: Random sample count.

    Returns:
        JSON-serializable debug dict.
    """
    cov = gaussians.covariance()
    n_g = gaussians.n_gaussians
    if n_g == 0:
        return {"agent": agent, "n": 0}
    quat_norm = gaussians.quaternion.norm(dim=-1)
    sym_err = (cov - cov.transpose(-1, -2)).abs().max().item()
    evals = torch.linalg.eigvalsh(cov)
    min_eig = float(evals.min().item())
    axis_ratio = evals[:, 2] / evals[:, 0].clamp_min(1.0e-12)
    n_take = min(int(n_samples), n_g)
    perm = torch.randperm(n_g, device=cov.device)[:n_take]
    samples = []
    for i in perm.tolist():
        samples.append(
            {
                "agent": agent,
                "view_index": int(gaussians.view_index[i].item())
                if gaussians.view_index is not None
                else -1,
                "uv": gaussians.uv[i].detach().float().cpu().tolist()
                if gaussians.uv is not None
                else None,
                "depth_peak": float(peak_z[i].item()),
                "depth_mean": float(gaussians.depth_mean[i].item())
                if gaussians.depth_mean is not None
                else None,
                "sigma_z": float(gaussians.sigma_z[i].item())
                if gaussians.sigma_z is not None
                else None,
                "covariance_eigenvalues": evals[i].detach().float().cpu().tolist(),
                "covariance_axis_ratio": float(axis_ratio[i].item()),
                "pca_xy": v_major[i].detach().float().cpu().tolist(),
            }
        )
    return {
        "agent": agent,
        "n": n_g,
        "mean_shape": list(gaussians.mean.shape),
        "scale_shape": list(gaussians.scale.shape),
        "quaternion_shape": list(gaussians.quaternion.shape),
        "feature_shape": list(gaussians.feature.shape),
        "max_quaternion_norm_error": float((quat_norm - 1.0).abs().max().item()),
        "max_symmetry_error": sym_err,
        "min_eigenvalue": min_eig,
        "psd_ok": min_eig >= -1.0e-5,
        "symmetric_ok": sym_err < 1.0e-5,
        "samples": samples,
    }


def _smoke_test() -> None:
    """CPU smoke test with random tensors (no dataset / checkpoint)."""
    torch.manual_seed(0)
    n_cam, channels, feat_h, feat_w = 2, 8, 12, 16
    image_hw = (48, 64)
    d_bins = 10
    z_bins = torch.linspace(1.0, 40.0, d_bins)
    heatmap = torch.zeros(n_cam, 2, feat_h, feat_w)
    heatmap[:, 1, 4:8, 6:10] = 4.0
    features = torch.randn(n_cam, channels, feat_h, feat_w)
    depth_logits = torch.randn(n_cam, d_bins, feat_h, feat_w)
    depth_z = torch.rand(n_cam, feat_h, feat_w) * 20.0 + 5.0
    cam = {
        "imgs": torch.zeros(n_cam, 3, image_hw[0], image_hw[1]),
        "intrinsics": torch.tensor(
            [[[32.0, 0.0, 32.0], [0.0, 32.0, 24.0], [0.0, 0.0, 1.0]]]
        ).expand(n_cam, -1, -1).contiguous(),
        "extrinsics": torch.eye(4).unsqueeze(0).expand(n_cam, -1, -1).contiguous(),
        "rots": torch.eye(3).unsqueeze(0).expand(n_cam, -1, -1).contiguous(),
        "trans": torch.zeros(n_cam, 3),
        "post_rots": torch.eye(3).unsqueeze(0).expand(n_cam, -1, -1).contiguous(),
        "post_trans": torch.zeros(n_cam, 3),
    }
    geometry = build_camera_geometry(cam)
    init = GaussianInitializer(
        z_bins={"vehicle": z_bins, "rsu": z_bins},
        debug=True,
    )
    for agent, kwargs in (
        ("vehicle", {"depth_logits": depth_logits}),
        ("rsu", {"depth_logits": depth_logits}),
        ("drone", {"depth_z": depth_z}),
    ):
        gs = init.forward_agent(
            agent=agent,
            heatmap_logits=heatmap,
            features=features,
            geometry=geometry,
            **kwargs,
        )
        dbg = init.last_debug or {}
        print(
            f"{agent}: N={gs.n_gaussians} mean={tuple(gs.mean.shape)} "
            f"scale={tuple(gs.scale.shape)} quat={tuple(gs.quaternion.shape)} "
            f"feat={tuple(gs.feature.shape)} "
            f"sym={dbg.get('max_symmetry_error')} min_eig={dbg.get('min_eigenvalue')}"
        )
        if gs.n_gaussians == 0:
            raise RuntimeError("expected foreground seeds")
        if not dbg.get("psd_ok") or not dbg.get("symmetric_ok"):
            raise RuntimeError(f"covariance check failed: {dbg}")

    # Rigid mean + orientation push: 90° yaw, scales must be invariant.
    from opencood.models.gaussian_modules_0822.base.covariance import (
        transform_covariance,
    )
    from opencood.models.gaussian_modules_0822.base.gaussian_init import (
        covariance_to_scale_quaternion as _cov2sq,
    )
    from opencood.models.gaussian_modules_0822.base.gaussian_transform import (
        reconstruct_covariance,
        transform_gaussians as transform_srq,
    )
    from opencood.models.gaussian_modules_0822.base.projection import (
        lift_optical_z_to_lidar,
        project_lidar_to_image,
    )

    sigma = torch.diag(torch.tensor([4.0, 1.0, 0.25])).unsqueeze(0)
    yaw = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated = transform_covariance(sigma, yaw)
    evals_src = torch.linalg.eigvalsh(sigma)
    evals_dst = torch.linalg.eigvalsh(rotated)
    if not torch.allclose(evals_src, evals_dst, atol=1.0e-5):
        raise RuntimeError(f"covariance eigenvalues changed: {evals_src} vs {evals_dst}")

    k_mat = torch.tensor([[[32.0, 0.0, 32.0], [0.0, 32.0, 24.0], [0.0, 0.0, 1.0]]])
    cam_rot = torch.tensor(
        [[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]]
    )
    cam_trans = torch.tensor([[1.5, -0.4, 0.8]])
    post_r = torch.eye(3).unsqueeze(0)
    post_t = torch.zeros(1, 3)
    xyz = torch.tensor([[12.0, -3.0, 1.2]])
    uv, z_cam, front = project_lidar_to_image(
        xyz, k_mat, cam_rot, cam_trans, post_r, post_t
    )
    if not bool(front[0]):
        raise RuntimeError("synthetic point should be in front of camera")
    rec, _, _ = lift_optical_z_to_lidar(
        uv, z_cam, k_mat, cam_rot, cam_trans, post_r, post_t
    )
    err = float((rec - xyz).norm().item())
    if err > 1.0e-4:
        raise RuntimeError(f"LSS lift roundtrip error {err:.3e} m")

    t_ae = torch.eye(4)
    t_ae[:3, :3] = yaw
    t_ae[:3, 3] = torch.tensor([10.0, -2.0, 0.5])
    scale0, quat0 = _cov2sq(sigma)
    mean_ego, scale_ego, _quat_ego = transform_srq(xyz, scale0, quat0, t_ae)
    expected = yaw @ xyz[0] + t_ae[:3, 3]
    if not torch.allclose(mean_ego[0], expected, atol=1.0e-5):
        raise RuntimeError("agent→ego mean transform mismatch")
    if not torch.allclose(scale_ego, scale0, atol=1.0e-5):
        raise RuntimeError("rigid transform must leave scale unchanged")
    cov_roundtrip = reconstruct_covariance(scale0, quat0)
    if not torch.allclose(cov_roundtrip, sigma, atol=1.0e-4):
        raise RuntimeError("scale/quaternion does not reconstruct Sigma")
    print(f"lift roundtrip={err:.2e} m  cov eig invariant  agent→ego ok")


if __name__ == "__main__":
    _smoke_test()

"""LSS-matched camera lift and one-shot camera geometry packing.

Lift matches ``_compute_world_coords`` / ``airv2x_encoder.get_geometry``:

1. Undo image aug on the frustum ``[u, v, d]``.
2. ``X = [u' · d, v' · d, d]`` (optical-axis ``d``).
3. ``X_lidar = (R_cam2lidar @ K^{-1}) @ X + t_cam2lidar``.
4. Optional ``X_ego = R_agent2ego @ X_lidar + t_agent2ego``.

Dataset ``batch_merged_cam_inputs`` already stores the full field set
after ``ue4_to_lss``:

    imgs [B,V,C,H,W] or [N,C,H,W]
    intrinsics [..., 3, 3]
    extrinsics [..., 4, 4]   # cam2lidar
    rots [..., 3, 3]         # cam2lidar rotation
    trans [..., 3]           # cam2lidar translation
    post_rots [..., 3, 3]
    post_trans [..., 3]

``build_camera_geometry`` / ``attach_camera_geometry`` construct
``lidar2image_*`` and ``img_aug_matrix`` **once** and stash them on the
agent ``batch_dict``, same pattern as MambaFusion.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

from opencood.models.gaussian_modules_0822.base.covariance import transform_gaussians
from opencood.models.gaussian_modules_0822.base.utils import EPS, safe_normalize
from opencood.models.gaussian_modules_0822.heatmap.geometry import (
    feature_indices_to_lss_normalized_coords,
)

AGENT_ORDER: Tuple[str, ...] = ("vehicle", "rsu", "drone")
CAMERA_GEOMETRY_KEY = "camera_geometry"
_REQUIRED_CAM_FIELDS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("intrinsics", (3, 3)),
    ("extrinsics", (4, 4)),
    ("rots", (3, 3)),
    ("trans", (3,)),
    ("post_rots", (3, 3)),
    ("post_trans", (3,)),
)


def _imgs_nv(imgs: torch.Tensor) -> Tuple[int, int, Tuple[int, int]]:
    """Return ``(n_flat, n_views, image_hw)`` from ``imgs``."""
    if imgs.dim() == 5:
        n_cav, n_views = int(imgs.shape[0]), int(imgs.shape[1])
        return n_cav * n_views, n_views, (int(imgs.shape[-2]), int(imgs.shape[-1]))
    if imgs.dim() == 4:
        n_flat = int(imgs.shape[0])
        return n_flat, n_flat, (int(imgs.shape[-2]), int(imgs.shape[-1]))
    raise ValueError(f"imgs must be 4D or 5D, got {tuple(imgs.shape)}")


def _flat_field(
    cam_inputs: Mapping[str, torch.Tensor],
    name: str,
    last: Tuple[int, ...],
    n_flat: int,
) -> torch.Tensor:
    """Reshape a calibration tensor to ``[n_flat, *last]``. No dtype/shape padding."""
    value = cam_inputs[name]
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor, got {type(value)}")
    if tuple(value.shape[-len(last) :]) != last:
        raise ValueError(
            f"{name} trailing {tuple(value.shape[-len(last):])} != {last}"
        )
    if value.dim() == 1 + len(last) and int(value.shape[0]) == n_flat:
        return value
    if (
        value.dim() == 2 + len(last)
        and int(value.shape[0]) * int(value.shape[1]) == n_flat
    ):
        return value.reshape(n_flat, *last)
    raise ValueError(f"{name} {tuple(value.shape)} incompatible with n_flat={n_flat}")


def _embed_4x4_from_rt(
    rotation: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    """Pack ``R [N,3,3]`` and ``t [N,3]`` into ``[N,4,4]``."""
    n_flat = int(rotation.shape[0])
    matrix = torch.eye(4, device=rotation.device, dtype=rotation.dtype)
    matrix = matrix.unsqueeze(0).expand(n_flat, -1, -1).clone()
    matrix[:, :3, :3] = rotation
    matrix[:, :3, 3] = translation
    return matrix


def _embed_k(intrinsics: torch.Tensor) -> torch.Tensor:
    """Embed ``K [N,3,3]`` as homogeneous ``[N,4,4]``."""
    n_flat = int(intrinsics.shape[0])
    matrix = torch.eye(4, device=intrinsics.device, dtype=intrinsics.dtype)
    matrix = matrix.unsqueeze(0).expand(n_flat, -1, -1).clone()
    matrix[:, :3, :3] = intrinsics
    return matrix


def build_camera_geometry(
    cam_inputs: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    """Build camera matrices once from a complete ``batch_merged_cam_inputs``.

    Args:
        cam_inputs: Dataset camera dict. All ``_REQUIRED_CAM_FIELDS`` must
            already have the exact trailing shapes listed above.

    Returns:
        Geometry dict. Tensor keys are flattened to ``[n_flat, ...]``.
        ``lidar2image_cam`` is agent-lidar → unaugmented image.
        ``img_aug_matrix`` is the 4x4 image post-homography.
        ``lidar2image_cam_bv`` / ``img_aug_matrix_bv`` are ``[1, n_flat, 4, 4]``
        for MambaFusion ``map_points``.
    """
    n_flat, n_views, image_hw = _imgs_nv(cam_inputs["imgs"])
    packed: Dict[str, Any] = {
        "n_flat": n_flat,
        "n_views": n_views,
        "image_hw": image_hw,
    }
    for name, last in _REQUIRED_CAM_FIELDS:
        packed[name] = _flat_field(cam_inputs, name, last, n_flat)

    packed["cam2lidar_rot"] = packed["rots"]
    packed["cam2lidar_trans"] = packed["trans"]
    cam2lidar = _embed_4x4_from_rt(packed["rots"], packed["trans"])
    packed["cam2lidar"] = cam2lidar
    lidar2cam = torch.linalg.inv(cam2lidar)
    lidar2image_cam = torch.matmul(_embed_k(packed["intrinsics"]), lidar2cam)
    img_aug = _embed_4x4_from_rt(packed["post_rots"], packed["post_trans"])
    packed["lidar2image_cam"] = lidar2image_cam
    packed["img_aug_matrix"] = img_aug
    packed["lidar2image_cam_bv"] = lidar2image_cam.unsqueeze(0)
    packed["img_aug_matrix_bv"] = img_aug.unsqueeze(0)
    packed["camera_batch_index"] = torch.zeros(
        n_flat, dtype=torch.long, device=lidar2image_cam.device
    )
    return packed


def flatten_camera_geometry(
    cam_inputs: Mapping[str, torch.Tensor],
    n_flat: Optional[int] = None,
    agent: str = "",
) -> Dict[str, Any]:
    """Alias of ``build_camera_geometry``. ``n_flat`` / ``agent`` are unused."""
    del agent
    geometry = build_camera_geometry(cam_inputs)
    if n_flat is not None and int(geometry["n_flat"]) != int(n_flat):
        raise ValueError(
            f"n_flat={n_flat} vs imgs-derived {geometry['n_flat']}"
        )
    return geometry


def attach_ego_lidar2image(
    geometry: MutableMapping[str, Any],
    agent_to_ego: torch.Tensor,
) -> MutableMapping[str, Any]:
    """Compose ``lidar2image_ego = lidar2image_cam @ inv(T_agent→ego)``.

    Args:
        geometry: Output of ``build_camera_geometry``.
        agent_to_ego: ``[n_flat, 4, 4]``.

    Returns:
        The same dict, with ``agent_to_ego`` / ``lidar2image_ego`` filled.
    """
    n_flat = int(geometry["n_flat"])
    transform = agent_to_ego
    if transform.dim() == 2:
        transform = transform.unsqueeze(0).expand(n_flat, -1, -1)
    if int(transform.shape[0]) != n_flat:
        raise ValueError(
            f"agent_to_ego {tuple(transform.shape)} vs n_flat={n_flat}"
        )
    geometry["agent_to_ego"] = transform
    ego2agent = torch.linalg.inv(transform)
    lidar2image_ego = torch.matmul(geometry["lidar2image_cam"], ego2agent)
    geometry["lidar2image_ego"] = lidar2image_ego
    geometry["lidar2image_ego_bv"] = lidar2image_ego.unsqueeze(0)
    return geometry


def attach_camera_geometry(
    ego_batch: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Write ``camera_geometry`` onto each present agent payload. Idempotent.

    Call once per batch (MambaFusion ``pre_process`` style). Later modules
    only read ``payload['camera_geometry']``.

    Args:
        ego_batch: Collated ``batch['ego']``.

    Returns:
        The same ``ego_batch``.
    """
    for agent in AGENT_ORDER:
        payload = ego_batch.get(agent)
        if not isinstance(payload, MutableMapping):
            continue
        if CAMERA_GEOMETRY_KEY in payload:
            continue
        cam_inputs = payload.get("batch_merged_cam_inputs")
        if not isinstance(cam_inputs, Mapping) or "imgs" not in cam_inputs:
            continue
        geometry = build_camera_geometry(cam_inputs)
        n_flat = int(geometry["n_flat"])
        n_cav = _record_len(ego_batch, agent)
        geometry["n_cav"] = n_cav
        # 5-D imgs already store cameras-per-CAV in n_views. 4-D flattened
        # imgs report n_views == n_flat; recover the true stride from
        # stacked record_len so Stage-1 can group by originating CAV.
        if n_cav > 0:
            if n_flat % n_cav != 0:
                raise ValueError(
                    f"{agent}: n_flat={n_flat} is not divisible by stacked CAVs={n_cav}"
                )
            geometry["n_views"] = n_flat // n_cav
            geometry["camera_batch_index"] = _camera_batch_index(
                payload, n_flat, int(geometry["n_views"]), geometry["lidar2image_cam"]
            )
        attach_ego_lidar2image(
            geometry, extract_agent_to_ego(ego_batch, agent, n_flat)
        )
        payload[CAMERA_GEOMETRY_KEY] = geometry
    return ego_batch


def projection_mats(
    geometry: Mapping[str, Any], points_frame: str
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Return ``(lidar2image_bv, img_aug_bv, image_hw)`` for ``map_points``.

    Args:
        geometry: Packed camera geometry.
        points_frame: ``ego`` or ``agent``.

    Returns:
        ``lidar2image [1, V, 4, 4]``, ``img_aug [1, V, 4, 4]``, ``(H, W)``.
    """
    frame = str(points_frame).lower()
    if frame == "ego":
        lidar2image = geometry["lidar2image_ego_bv"]
    elif frame == "agent":
        lidar2image = geometry["lidar2image_cam_bv"]
    else:
        raise ValueError(f"points_frame must be 'ego' or 'agent', got {points_frame}")
    return lidar2image, geometry["img_aug_matrix_bv"], tuple(geometry["image_hw"])


def _record_len(ego_batch: Mapping[str, Any], agent: str) -> int:
    """Stacked CAV count for ``agent`` (sum over the collated batch)."""
    payload = ego_batch.get(agent)
    if not isinstance(payload, Mapping):
        return 0
    record_len = payload.get("record_len")
    if record_len is None:
        return 0
    if torch.is_tensor(record_len):
        return int(record_len.reshape(-1).sum().item())
    if isinstance(record_len, (list, tuple)):
        return int(sum(int(v) for v in record_len))
    return int(record_len)


def _camera_batch_index(
    payload: Mapping[str, Any],
    n_flat: int,
    n_views: int,
    like: torch.Tensor,
) -> torch.Tensor:
    """Per-camera batch id aligned with flattened ``view_index``."""
    device = like.device
    batch_idxs = payload.get("batch_idxs")
    if batch_idxs is None:
        return torch.zeros(n_flat, dtype=torch.long, device=device)
    if not torch.is_tensor(batch_idxs):
        batch_idxs = torch.as_tensor(batch_idxs, dtype=torch.long, device=device)
    cav_batch = batch_idxs.long().reshape(-1).to(device=device)
    if int(cav_batch.numel()) * n_views != n_flat:
        return torch.zeros(n_flat, dtype=torch.long, device=device)
    return cav_batch.repeat_interleave(int(n_views))


def extract_agent_to_ego(
    ego_batch: Mapping[str, Any],
    agent: str,
    n_flat: int,
    n_views: Optional[int] = None,
) -> torch.Tensor:
    """Per-camera ``T_agent→ego`` aligned with flattened cameras.

    Matches ``airv2x_mambafusion.pre_process``:
    ``img_pairwise_t_matrix_collab[i, 0]`` is agent ``i`` → ego, then
    ``repeat_interleave(n_views)``.

    Args:
        ego_batch: Collated ``batch['ego']``.
        agent: ``vehicle`` / ``rsu`` / ``drone``.
        n_flat: Flattened camera count.
        n_views: Cameras per CAV. Inferred from ``n_flat / record_len``
            when omitted.

    Returns:
        ``[n_flat, 4, 4]`` rigid transforms.
    """
    pairwise = ego_batch["img_pairwise_t_matrix_collab"]
    if not torch.is_tensor(pairwise):
        raise TypeError("img_pairwise_t_matrix_collab must be a tensor")
    if pairwise.dim() == 5:
        pairwise = pairwise[0]
    order: Sequence[str] = tuple(ego_batch.get("agent_order") or AGENT_ORDER)
    offset = 0
    for name in order:
        if name == agent:
            break
        offset += _record_len(ego_batch, name)
    n_cav = _record_len(ego_batch, agent)
    if n_cav <= 0:
        eye = torch.eye(4, device=pairwise.device, dtype=pairwise.dtype)
        return eye.unsqueeze(0).expand(n_flat, -1, -1).clone()
    if n_views is None:
        if n_flat % n_cav != 0:
            raise ValueError(
                f"{agent}: n_flat={n_flat} is not divisible by record_len={n_cav}"
            )
        n_views = n_flat // n_cav
    transforms = pairwise[offset : offset + n_cav, 0]
    if tuple(transforms.shape[-2:]) != (4, 4):
        raise ValueError(
            f"pairwise slice shape {tuple(transforms.shape)} is not [n_cav, 4, 4]"
        )
    out = transforms.repeat_interleave(int(n_views), dim=0)
    if int(out.shape[0]) != n_flat:
        raise ValueError(
            f"{agent}: agent_to_ego {tuple(out.shape)} vs n_flat={n_flat}"
        )
    return out


def _gather_view(tensor: torch.Tensor, view_index: torch.Tensor) -> torch.Tensor:
    return tensor[view_index.long()]


def feature_indices_to_aug_pixels(
    x_indices: torch.Tensor,
    y_indices: torch.Tensor,
    feature_hw: Tuple[int, int],
    image_hw: Tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Feature cell → augmented-image pixel ``(u, v)``, shape ``[N, 2]``."""
    normalized = feature_indices_to_lss_normalized_coords(
        x_indices=x_indices.to(dtype),
        y_indices=y_indices.to(dtype),
        feature_hw=feature_hw,
        image_hw=image_hw,
        dtype=dtype,
    )
    height, width = float(image_hw[0]), float(image_hw[1])
    return torch.stack(
        [normalized[:, 0] * width, normalized[:, 1] * height], dim=-1
    )


def lift_optical_z_to_lidar(
    uv_aug: torch.Tensor,
    depth_z: torch.Tensor,
    intrinsic: torch.Tensor,
    cam2lidar_rot: torch.Tensor,
    cam2lidar_trans: torch.Tensor,
    post_rot: torch.Tensor,
    post_trans: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LSS frustum lift: augmented pixel + optical ``z`` → lidar.

    Args:
        uv_aug: Augmented-image pixels ``[N, 2]``.
        depth_z: Optical-axis depth in meters ``[N]``.
        intrinsic: ``K`` ``[N, 3, 3]``.
        cam2lidar_rot: ``R_cam2lidar`` ``[N, 3, 3]``.
        cam2lidar_trans: ``t_cam2lidar`` ``[N, 3]``.
        post_rot: Image-aug rotation ``[N, 3, 3]``.
        post_trans: Image-aug translation ``[N, 3]``.

    Returns:
        ``(mean_lidar [N,3], unit_ray_lidar [N,3], uv_orig [N,2])``.
    """
    n_points = int(uv_aug.shape[0])
    if n_points == 0:
        empty = depth_z.new_empty((0, 3))
        return empty, empty, depth_z.new_empty((0, 2))
    frustum = torch.stack([uv_aug[:, 0], uv_aug[:, 1], depth_z], dim=-1)
    points = frustum - post_trans
    inv_post = torch.linalg.inv(post_rot)
    points = torch.matmul(inv_post, points.unsqueeze(-1)).squeeze(-1)
    x_img = torch.cat(
        (points[:, :2] * points[:, 2:3], points[:, 2:3]), dim=-1
    )
    k_inv = torch.linalg.inv(intrinsic[:, :3, :3])
    combine = torch.matmul(cam2lidar_rot, k_inv)
    mean = torch.matmul(combine, x_img.unsqueeze(-1)).squeeze(-1) + cam2lidar_trans
    ones = torch.ones(
        n_points, 1, device=uv_aug.device, dtype=uv_aug.dtype
    )
    cam_dir = torch.matmul(
        k_inv, torch.cat([points[:, :2], ones], dim=-1).unsqueeze(-1)
    ).squeeze(-1)
    ray = safe_normalize(
        torch.matmul(cam2lidar_rot, cam_dir.unsqueeze(-1)).squeeze(-1)
    )
    return mean, ray, points[:, :2]


def project_lidar_to_image(
    xyz_lidar: torch.Tensor,
    intrinsic: torch.Tensor,
    cam2lidar_rot: torch.Tensor,
    cam2lidar_trans: torch.Tensor,
    post_rot: torch.Tensor,
    post_trans: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of ``lift_optical_z_to_lidar`` (no visibility filter).

    Args:
        xyz_lidar: ``[N, 3]`` in the camera's lidar / agent frame.
        intrinsic: ``[N, 3, 3]``.
        cam2lidar_rot: ``[N, 3, 3]``.
        cam2lidar_trans: ``[N, 3]``.
        post_rot: ``[N, 3, 3]``.
        post_trans: ``[N, 3]``.

    Returns:
        ``(uv_aug [N,2], optical_z [N], front [N] bool)``.
    """
    n_points = int(xyz_lidar.shape[0])
    if n_points == 0:
        empty = xyz_lidar.new_empty((0, 2))
        return empty, xyz_lidar.new_empty((0,)), xyz_lidar.new_empty((0,), dtype=torch.bool)
    eye = torch.eye(4, device=xyz_lidar.device, dtype=xyz_lidar.dtype)
    cam2lidar = eye.unsqueeze(0).expand(n_points, -1, -1).clone()
    cam2lidar[:, :3, :3] = cam2lidar_rot
    cam2lidar[:, :3, 3] = cam2lidar_trans
    lidar2cam = torch.linalg.inv(cam2lidar)
    x_cam = torch.matmul(
        lidar2cam[:, :3, :3], xyz_lidar.unsqueeze(-1)
    ).squeeze(-1) + lidar2cam[:, :3, 3]
    optical_z = x_cam[:, 2]
    uv_h = torch.matmul(intrinsic[:, :3, :3], x_cam.unsqueeze(-1)).squeeze(-1)
    denom = uv_h[:, 2:3].clone()
    denom = torch.where(denom.abs() < EPS, torch.full_like(denom, EPS), denom)
    uv = uv_h[:, :2] / denom
    ones = torch.ones(n_points, 1, device=xyz_lidar.device, dtype=xyz_lidar.dtype)
    uv_h3 = torch.cat([uv, ones], dim=-1)
    uv_aug_h = torch.matmul(post_rot, uv_h3.unsqueeze(-1)).squeeze(-1) + post_trans
    front = optical_z > 0.1
    return uv_aug_h[:, :2], optical_z, front


def backproject_optical_z(
    x_indices: torch.Tensor,
    y_indices: torch.Tensor,
    depth_z: torch.Tensor,
    view_index: torch.Tensor,
    geometry: Mapping[str, torch.Tensor],
    feature_hw: Tuple[int, int],
    image_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift feature cells at optical-axis ``z`` into the agent lidar frame.

    Args:
        x_indices: Feature columns ``[N]``.
        y_indices: Feature rows ``[N]``.
        depth_z: Optical-axis depth in meters ``[N]``.
        view_index: Flattened camera index ``[N]``.
        geometry: Flattened calibration dict from ``flatten_camera_geometry``.
        feature_hw: ``(H, W)`` of the heatmap / F90 grid.
        image_hw: ``(image_h, image_w)``.

    Returns:
        ``(mean_lidar [N,3], unit_ray_lidar [N,3], uv_aug [N,2])``.
    """
    n_points = int(x_indices.shape[0])
    dtype = depth_z.dtype
    if n_points == 0:
        empty = depth_z.new_empty((0, 3))
        return empty, empty, depth_z.new_empty((0, 2))
    uv_aug = feature_indices_to_aug_pixels(
        x_indices, y_indices, feature_hw, image_hw, dtype
    )
    intrinsic = _gather_view(geometry["intrinsics"], view_index).to(dtype=dtype)
    post_rot = _gather_view(geometry["post_rots"], view_index).to(dtype=dtype)
    post_trans = _gather_view(geometry["post_trans"], view_index).to(dtype=dtype)
    cam_rot = _gather_view(geometry["cam2lidar_rot"], view_index).to(dtype=dtype)
    cam_trans = _gather_view(geometry["cam2lidar_trans"], view_index).to(dtype=dtype)
    mean, ray, _uv_orig = lift_optical_z_to_lidar(
        uv_aug=uv_aug,
        depth_z=depth_z,
        intrinsic=intrinsic,
        cam2lidar_rot=cam_rot,
        cam2lidar_trans=cam_trans,
        post_rot=post_rot,
        post_trans=post_trans,
    )
    return mean, ray, uv_aug


def image_direction_to_tangent(
    mean: torch.Tensor,
    unit_ray: torch.Tensor,
    v_major_xy: torch.Tensor,
    depth_z: torch.Tensor,
    x_indices: torch.Tensor,
    y_indices: torch.Tensor,
    view_index: torch.Tensor,
    geometry: Mapping[str, torch.Tensor],
    feature_hw: Tuple[int, int],
    image_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lift 2D PCA direction to an orthonormal tangent pair perpendicular to ``r``.

    A one-cell offset along ``v_major_xy`` is backprojected at the same
    optical ``z``. The 3D chord is Gram-Schmidt orthogonalized against the
    viewing ray.

    Args:
        mean: ``[N, 3]`` current lidar means.
        unit_ray: ``[N, 3]``.
        v_major_xy: ``[N, 2]`` unit image direction ``(dx, dy)`` in feature cells.
        depth_z: ``[N]``.
        x_indices: ``[N]``.
        y_indices: ``[N]``.
        view_index: ``[N]``.
        geometry: Flattened calibration.
        feature_hw: Heatmap ``(H, W)``.
        image_hw: Image ``(h, w)``.

    Returns:
        ``(t_major, t_minor)`` each ``[N, 3]``.
    """
    if mean.numel() == 0:
        empty = mean.new_empty((0, 3))
        return empty, empty

    x_off = x_indices.to(dtype=depth_z.dtype) + v_major_xy[:, 0]
    y_off = y_indices.to(dtype=depth_z.dtype) + v_major_xy[:, 1]
    feat_h, feat_w = feature_hw
    x_off = x_off.clamp(0.0, float(feat_w - 1))
    y_off = y_off.clamp(0.0, float(feat_h - 1))
    mean_off, _, _ = backproject_optical_z(
        x_indices=x_off,
        y_indices=y_off,
        depth_z=depth_z,
        view_index=view_index,
        geometry=geometry,
        feature_hw=feature_hw,
        image_hw=image_hw,
    )
    chord = mean_off - mean
    chord = chord - unit_ray * (chord * unit_ray).sum(dim=-1, keepdim=True)
    t1 = safe_normalize(chord)
    degenerate = torch.linalg.norm(chord, dim=-1) < 1.0e-5
    if bool(degenerate.any()):
        rotation = _gather_view(geometry["cam2lidar_rot"], view_index).to(
            dtype=mean.dtype
        )
        camera_right = rotation[:, :, 0]
        fallback = camera_right - unit_ray * (camera_right * unit_ray).sum(
            dim=-1, keepdim=True
        )
        fallback = safe_normalize(fallback)
        t1 = torch.where(degenerate.unsqueeze(-1), fallback, t1)
    t2 = safe_normalize(torch.cross(unit_ray, t1, dim=-1))
    return t1, t2


def transform_to_ego(
    mean_agent: torch.Tensor,
    covariance_agent: torch.Tensor,
    agent_to_ego: torch.Tensor,
    view_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rigid-push Gaussians from agent lidar into ego.

    Covariance follows ``Σ_ego = R Σ_agent Rᵀ``.

    Args:
        mean_agent: ``[N, 3]``.
        covariance_agent: ``[N, 3, 3]``.
        agent_to_ego: ``[n_flat, 4, 4]`` or ``[N, 4, 4]`` or ``[4, 4]``.
        view_index: If ``agent_to_ego`` is per-camera ``[n_flat, 4, 4]``,
            gather with these indices.

    Returns:
        ``(mean_ego, covariance_ego)``.
    """
    if mean_agent.numel() == 0:
        return mean_agent, covariance_agent
    transform = agent_to_ego
    if transform.dim() == 2:
        transform = transform.to(device=mean_agent.device, dtype=mean_agent.dtype)
        return transform_gaussians(mean_agent, covariance_agent, transform)
    transform = transform.to(device=mean_agent.device, dtype=mean_agent.dtype)
    if view_index is not None and int(transform.shape[0]) != int(mean_agent.shape[0]):
        transform = _gather_view(transform, view_index)
    return transform_gaussians(mean_agent, covariance_agent, transform)

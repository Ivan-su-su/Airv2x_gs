"""3D feature Gaussian set in scale-quaternion form (not a 3DGS renderer)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch

from opencood.models.gaussian_modules_0822.base.gaussian_transform import (
    quaternion_to_rotation_matrix,
    reconstruct_covariance,
    transform_gaussians,
)


@dataclass
class GaussianSet:
    """Batched 3D feature Gaussians.

    Primary representation matches standard 3D Gaussians:

        mean [N, 3]
        scale [N, 3]
        quaternion [N, 4]  (``wxyz``)
        feature [N, C]

    ``mean`` / orientation live in the lidar / agent frame after camera
    lift, or in ego after ``to_ego``. Full covariance is not stored.

    Attributes:
        mean: ``[N, 3]`` 3D centers.
        scale: ``[N, 3]`` axis standard deviations (meters).
        quaternion: ``[N, 4]`` unit rotation ``[w, x, y, z]``.
        feature: ``[N, C]`` image features.
        p_fg: Optional ``[N]`` heatmap foreground probability.
        opacity: Optional ``[N]`` learned evidence strength used by Stage-3.
        uv: Optional ``[N, 2]`` image-plane pixel coordinates ``(u, v)``.
        view_index: Optional ``[N]`` flattened camera index.
        batch_index: Optional ``[N]`` sample / agent-batch index.
        depth_mean: Optional ``[N]`` optical-axis depth used as the center.
        sigma_z: Optional ``[N]`` depth / ray scale (meters).
        agent: Agent name for this set, or empty if mixed.
        frame: ``\"ego\"`` after agent→ego, else ``\"agent\"``.
    """

    mean: torch.Tensor
    scale: torch.Tensor
    quaternion: torch.Tensor
    feature: torch.Tensor
    p_fg: Optional[torch.Tensor] = None
    uv: Optional[torch.Tensor] = None
    view_index: Optional[torch.Tensor] = None
    batch_index: Optional[torch.Tensor] = None
    depth_mean: Optional[torch.Tensor] = None
    sigma_z: Optional[torch.Tensor] = None
    agent: str = ""
    frame: str = "agent"
    opacity: Optional[torch.Tensor] = None

    @property
    def n_gaussians(self) -> int:
        """Number of Gaussians ``N``."""
        return int(self.mean.shape[0])

    def rotation_matrix(self) -> torch.Tensor:
        """Return ``[N, 3, 3]`` rotations from the stored quaternion."""
        return quaternion_to_rotation_matrix(self.quaternion)

    def covariance(self) -> torch.Tensor:
        """Rebuild ``Σ`` from scale / quaternion (debug only)."""
        return reconstruct_covariance(self.scale, self.quaternion)

    def replace_feature(self, feature: torch.Tensor) -> "GaussianSet":
        """Return a copy with an updated feature tensor.

        Args:
            feature: ``[N, C]``.

        Returns:
            New ``GaussianSet`` sharing the geometry tensors.
        """
        return replace(self, feature=feature)

    def replace_opacity(self, opacity: Optional[torch.Tensor]) -> "GaussianSet":
        return replace(self, opacity=opacity)

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "GaussianSet":
        """Move tensors to ``device`` / ``dtype`` (non-index tensors only)."""

        def _cast(t: Optional[torch.Tensor], is_index: bool) -> Optional[torch.Tensor]:
            if t is None:
                return None
            if is_index:
                return t.to(device=device)
            return t.to(device=device, dtype=dtype) if dtype is not None else t.to(device=device)

        return GaussianSet(
            mean=_cast(self.mean, False),
            scale=_cast(self.scale, False),
            quaternion=_cast(self.quaternion, False),
            feature=_cast(self.feature, False),
            p_fg=_cast(self.p_fg, False),
            uv=_cast(self.uv, False),
            view_index=_cast(self.view_index, True),
            batch_index=_cast(self.batch_index, True),
            depth_mean=_cast(self.depth_mean, False),
            sigma_z=_cast(self.sigma_z, False),
            agent=self.agent,
            frame=self.frame,
            opacity=_cast(self.opacity, False),
        )

    def index_select(self, index: torch.Tensor) -> "GaussianSet":
        """Return a subset of Gaussians by mask or index.

        Geometry parameters are gathered, not modified. This is used for
        range filtering, not NMS or confidence pruning.

        Args:
            index: Boolean mask ``[N]`` or integer index ``[M]``.

        Returns:
            A new ``GaussianSet`` with the same ``agent`` / ``frame``.
        """
        if index.dim() != 1:
            raise ValueError(f"index must be 1-D, got {tuple(index.shape)}")
        if index.dtype == torch.bool:
            if int(index.shape[0]) != self.n_gaussians:
                raise ValueError(
                    f"bool mask length {int(index.shape[0])} != N={self.n_gaussians}"
                )
            gather_index = index
        else:
            gather_index = index.long()

        def _gather(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            return tensor[gather_index]

        return GaussianSet(
            mean=self.mean[gather_index],
            scale=self.scale[gather_index],
            quaternion=self.quaternion[gather_index],
            feature=self.feature[gather_index],
            p_fg=_gather(self.p_fg),
            uv=_gather(self.uv),
            view_index=_gather(self.view_index),
            batch_index=_gather(self.batch_index),
            depth_mean=_gather(self.depth_mean),
            sigma_z=_gather(self.sigma_z),
            agent=self.agent,
            frame=self.frame,
            opacity=_gather(self.opacity),
        )

    def scatter_replace(self, index: torch.Tensor, src: "GaussianSet") -> "GaussianSet":
        """Write ``src`` geometry and features into ``self`` at ``index``.

        Args:
            index: Bool mask ``[N]`` or integer index ``[M]``.
            src: Subset with ``M`` Gaussians aligned to ``index``.

        Returns:
            New set; unmatched rows keep this set's tensors.
        """
        if index.dtype == torch.bool:
            gather = index
        else:
            gather = index.long()
        mean = self.mean.clone()
        scale = self.scale.clone()
        quaternion = self.quaternion.clone()
        feature = self.feature.clone()
        mean[gather] = src.mean
        scale[gather] = src.scale
        quaternion[gather] = src.quaternion
        feature[gather] = src.feature
        return replace(
            self,
            mean=mean,
            scale=scale,
            quaternion=quaternion,
            feature=feature,
        )

    def to_ego(
        self,
        agent_to_ego: torch.Tensor,
        view_index: Optional[torch.Tensor] = None,
    ) -> "GaussianSet":
        """Apply a rigid agent→ego transform to mean and orientation.

        Scale is invariant. Quaternion is updated via ``R_ego @ R_gaussian``.

        Args:
            agent_to_ego: ``[n_flat, 4, 4]``, ``[N, 4, 4]``, or ``[4, 4]``.
            view_index: Camera indices used to gather a per-camera stack.
                Defaults to ``self.view_index``.

        Returns:
            A new set with ``frame='ego'``.
        """
        if self.n_gaussians == 0:
            return replace(self, frame="ego")
        gather_index = view_index if view_index is not None else self.view_index
        mean_ego, scale_ego, quat_ego = transform_gaussians(
            self.mean,
            self.scale,
            self.quaternion,
            agent_to_ego,
            view_index=gather_index,
        )
        return replace(
            self, mean=mean_ego, scale=scale_ego, quaternion=quat_ego, frame="ego"
        )

    @classmethod
    def empty(
        cls,
        device: torch.device,
        dtype: torch.dtype,
        n_channels: int,
        agent: str = "",
        frame: str = "agent",
    ) -> "GaussianSet":
        """Build an empty set with the correct tensor ranks.

        Args:
            device: Tensor device.
            dtype: Floating dtype.
            n_channels: Feature width ``C``.
            agent: Optional agent tag.
            frame: Coordinate frame tag.

        Returns:
            A set with ``N=0``.
        """
        return cls(
            mean=torch.empty((0, 3), device=device, dtype=dtype),
            scale=torch.empty((0, 3), device=device, dtype=dtype),
            quaternion=torch.empty((0, 4), device=device, dtype=dtype),
            feature=torch.empty((0, n_channels), device=device, dtype=dtype),
            p_fg=torch.empty((0,), device=device, dtype=dtype),
            uv=torch.empty((0, 2), device=device, dtype=dtype),
            view_index=torch.empty((0,), device=device, dtype=torch.long),
            batch_index=torch.empty((0,), device=device, dtype=torch.long),
            depth_mean=torch.empty((0,), device=device, dtype=dtype),
            sigma_z=torch.empty((0,), device=device, dtype=dtype),
            agent=agent,
            frame=frame,
        )

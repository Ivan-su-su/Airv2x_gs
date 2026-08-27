from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from opencood.models.uncertain_gaussian_modules.geometry.multi_view_grid_sampler import (
    MultiViewGridSampler,
)


class DeformableGaussianCrossAttention(nn.Module):
    """Update per-view image features with voxel-query-guided local sampling."""

    def __init__(self, model_cfg=None):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.support_scale = float(self.model_cfg.get("support_scale", 1.0))
        self.support_min_offset = float(self.model_cfg.get("support_min_offset", 0.01))
        self.align_corners = bool(self.model_cfg.get("align_corners", False))
        self.query_proj = nn.LazyLinear(int(self.model_cfg.get("attention_dim", 128)))
        self.key_proj = nn.LazyLinear(int(self.model_cfg.get("attention_dim", 128)))
        self.value_proj = nn.LazyLinear(int(self.model_cfg.get("attention_dim", 128)))
        self.grid_sampler = MultiViewGridSampler(
            {"align_corners": self.align_corners}
        )

    def _reshape_feature_map(
        self, feature_map: torch.Tensor, num_views: int
    ) -> torch.Tensor:
        """Convert a feature map into `[B, V, C, H, W]`."""
        return self.grid_sampler.reshape_feature_map(feature_map, num_views)

    def _extract_support_offsets(
        self, support_covariance_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build principal support offsets from the 2D covariance."""
        eigen_values, eigen_vectors = torch.linalg.eigh(support_covariance_2d)
        major_values = eigen_values[:, -1].clamp_min(0.0).sqrt() * self.support_scale
        minor_values = eigen_values[:, 0].clamp_min(0.0).sqrt() * self.support_scale
        major_values = major_values.clamp_min(self.support_min_offset)
        minor_values = minor_values.clamp_min(self.support_min_offset)
        major_vectors = eigen_vectors[:, :, -1] * major_values.unsqueeze(-1)
        minor_vectors = eigen_vectors[:, :, 0] * minor_values.unsqueeze(-1)
        return major_vectors, minor_vectors

    def _build_sampling_coords(
        self,
        normalized_coords: torch.Tensor,
        support_covariance_2d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build center plus principal-direction support sampling coordinates."""
        if support_covariance_2d is None:
            zero_offsets = torch.zeros_like(normalized_coords)
            offsets = [
                zero_offsets,
                torch.full_like(normalized_coords, self.support_min_offset),
                torch.tensor(
                    [self.support_min_offset, -self.support_min_offset],
                    device=normalized_coords.device,
                    dtype=normalized_coords.dtype,
                ).view(1, 2).repeat(normalized_coords.shape[0], 1),
                torch.tensor(
                    [-self.support_min_offset, self.support_min_offset],
                    device=normalized_coords.device,
                    dtype=normalized_coords.dtype,
                ).view(1, 2).repeat(normalized_coords.shape[0], 1),
                -torch.full_like(normalized_coords, self.support_min_offset),
            ]
        else:
            major_vectors, minor_vectors = self._extract_support_offsets(
                support_covariance_2d
            )
            zero_offsets = torch.zeros_like(normalized_coords)
            offsets = [
                zero_offsets,
                major_vectors,
                -major_vectors,
                minor_vectors,
                -minor_vectors,
            ]
        sampling_coords = torch.stack(
            [normalized_coords + offset for offset in offsets], dim=1
        )
        return sampling_coords.clamp(0.0, 1.0)

    def _sample_support_features(
        self,
        feature_map: torch.Tensor,
        sampling_coords: torch.Tensor,
        local_agent_ids: torch.Tensor,
        view_ids: torch.Tensor,
        num_views: int,
    ) -> torch.Tensor:
        """Sample support-region features for each projected hit."""
        num_hits, num_samples, _ = sampling_coords.shape
        flattened_coords = sampling_coords.reshape(-1, 2)
        flattened_local_agent_ids = (
            local_agent_ids.long().unsqueeze(1).repeat(1, num_samples).reshape(-1)
        )
        flattened_view_ids = view_ids.long().unsqueeze(1).repeat(1, num_samples).reshape(-1)
        sampled_support = self.grid_sampler(
            feature_map,
            flattened_coords,
            flattened_local_agent_ids,
            flattened_view_ids,
            num_views,
        )
        if sampled_support is None:
            raise ValueError("Support feature sampling requires a valid image feature map.")
        return sampled_support.view(num_hits, num_samples, -1)

    def forward(
        self,
        voxel_queries: torch.Tensor,
        image_feature_map: torch.Tensor,
        normalized_coords: torch.Tensor,
        local_agent_ids: torch.Tensor,
        view_ids: torch.Tensor,
        num_views: int,
        support_covariance_2d: Optional[torch.Tensor] = None,
        external_sampling_coords: Optional[torch.Tensor] = None,
        sampling_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return query-conditioned per-view features with local support samples."""
        if normalized_coords.numel() == 0:
            feature_map = self._reshape_feature_map(image_feature_map, num_views)
            channels = int(feature_map.shape[2])
            attention_dim = int(self.query_proj.out_features)
            empty_support = normalized_coords.new_empty((0, 5, channels))
            empty_feature = normalized_coords.new_empty((0, channels))
            empty_weights = normalized_coords.new_empty((0, 5))
            empty_token = normalized_coords.new_empty((0, attention_dim))
            return {
                "sampling_coords": normalized_coords.new_empty((0, 5, 2)),
                "sampled_support_features": empty_support,
                "attention_weights": empty_weights,
                "updated_view_features": empty_feature,
                "attended_view_tokens": empty_token,
            }

        if external_sampling_coords is not None:
            sampling_coords = external_sampling_coords.clamp(0.0, 1.0)
        else:
            sampling_coords = self._build_sampling_coords(
                normalized_coords, support_covariance_2d
            )
        sampled_support_features = self._sample_support_features(
            image_feature_map,
            sampling_coords,
            local_agent_ids,
            view_ids,
            num_views,
        )

        query_token = self.query_proj(voxel_queries)
        key_token = self.key_proj(sampled_support_features)
        value_token = self.value_proj(sampled_support_features)
        attention_logits = torch.einsum("nd,nkd->nk", query_token, key_token)
        attention_logits = attention_logits / (query_token.shape[-1] ** 0.5)
        if sampling_valid_mask is not None:
            valid_mask = sampling_valid_mask.bool()
            masked_logits = attention_logits.masked_fill(
                ~valid_mask,
                torch.finfo(attention_logits.dtype).min,
            )
            all_invalid = ~valid_mask.any(dim=1, keepdim=True)
            attention_weights = torch.softmax(masked_logits, dim=1)
            attention_weights = attention_weights.masked_fill(all_invalid, 0.0)
        else:
            attention_weights = torch.softmax(attention_logits, dim=1)
        updated_view_features = torch.einsum(
            "nk,nkd->nd", attention_weights, value_token
        )
        attended_view_token = torch.einsum(
            "nk,nkc->nc", attention_weights, sampled_support_features
        )

        return {
            "sampling_coords": sampling_coords,
            "sampled_support_features": sampled_support_features,
            "attention_weights": attention_weights,
            "updated_view_features": updated_view_features,
            "attended_view_tokens": attended_view_token,
        }

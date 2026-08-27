from typing import Dict, Optional

import torch
import torch.nn as nn


class MultiViewGaussianFuser(nn.Module):
    """Fuse repeated multi-view gaussian observations by source voxel index."""

    def __init__(self, model_cfg=None):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.view_token_dim = int(self.model_cfg.get("view_token_dim", 64))
        self.score_token_mlp = nn.Sequential(
            nn.LazyLinear(self.view_token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.view_token_dim, self.view_token_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(self.view_token_dim, self.view_token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.view_token_dim, 1),
        )

    def _build_group_weights(
        self, inverse_indices: torch.Tensor, score_logits: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        """Build per-hit fusion weights with segment softmax on repeated-view groups."""
        weights = torch.ones_like(score_logits)
        multi_view_hit_mask = counts[inverse_indices] > 1
        if not torch.any(multi_view_hit_mask):
            return weights

        multi_group_indices = inverse_indices[multi_view_hit_mask]
        multi_scores = score_logits[multi_view_hit_mask]
        num_groups = int(counts.shape[0])
        max_scores = torch.full(
            (num_groups,),
            torch.finfo(score_logits.dtype).min,
            device=score_logits.device,
            dtype=score_logits.dtype,
        )
        max_scores.scatter_reduce_(
            0,
            multi_group_indices,
            multi_scores,
            reduce="amax",
            include_self=True,
        )
        stabilized_scores = torch.exp(
            multi_scores - max_scores[multi_group_indices]
        )
        score_denominator = torch.zeros(
            num_groups,
            device=score_logits.device,
            dtype=score_logits.dtype,
        )
        score_denominator.index_add_(0, multi_group_indices, stabilized_scores)
        weights[multi_view_hit_mask] = stabilized_scores / score_denominator[
            multi_group_indices
        ].clamp_min(1e-8)
        return weights

    def fuse_groups(
        self,
        source_group_ids: torch.Tensor,
        hit_points_3d: torch.Tensor,
        updated_view_features: torch.Tensor,
        attended_view_tokens: torch.Tensor,
        sigma_3d: torch.Tensor,
        view_ids: torch.Tensor,
        local_agent_ids: Optional[torch.Tensor] = None,
        num_local_agents: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Fuse observations that originate from the same source group."""
        unique_group_ids, inverse_indices, counts = torch.unique(
            source_group_ids.long(),
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        score_input = torch.cat([updated_view_features, attended_view_tokens], dim=-1)
        score_tokens = self.score_token_mlp(score_input)
        score_logits = self.score_head(score_tokens).squeeze(-1)
        group_weights = self._build_group_weights(
            inverse_indices=inverse_indices,
            score_logits=score_logits,
            counts=counts,
        )

        num_groups = int(unique_group_ids.shape[0])
        feature_dim = int(updated_view_features.shape[-1])
        fused_means = torch.zeros(
            num_groups, 3, device=hit_points_3d.device, dtype=hit_points_3d.dtype
        )
        fused_features = torch.zeros(
            num_groups,
            feature_dim,
            device=updated_view_features.device,
            dtype=updated_view_features.dtype,
        )
        fused_covariances = torch.zeros(
            num_groups,
            sigma_3d.shape[-2],
            sigma_3d.shape[-1],
            device=sigma_3d.device,
            dtype=sigma_3d.dtype,
        )
        fused_means.index_add_(0, inverse_indices, hit_points_3d * group_weights.unsqueeze(-1))
        fused_features.index_add_(
            0, inverse_indices, updated_view_features * group_weights.unsqueeze(-1)
        )
        fused_covariances.index_add_(
            0, inverse_indices, sigma_3d * group_weights.view(-1, 1, 1)
        )
        local_agent_mask = None
        if (
            local_agent_ids is not None
            and torch.is_tensor(local_agent_ids)
            and num_local_agents is not None
            and num_local_agents > 0
        ):
            local_agent_mask = torch.zeros(
                num_groups,
                int(num_local_agents),
                dtype=torch.bool,
                device=local_agent_ids.device,
            )
            local_agent_mask[inverse_indices, local_agent_ids.long()] = True

        fused_dict = {
            "mean": fused_means,
            "feature": fused_features,
            "covariance": fused_covariances,
            "source_group_ids": unique_group_ids,
            "num_valid_views": counts,
            "multi_view_group_mask": counts > 1,
            "view_weights": group_weights,
            "view_ids": view_ids.long(),
            "group_indices": inverse_indices,
            "local_agent_ids": local_agent_ids.long() if local_agent_ids is not None else None,
        }
        if local_agent_mask is not None:
            fused_dict["local_agent_mask"] = local_agent_mask
        return fused_dict

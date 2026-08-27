from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewGridSampler(nn.Module):
    """Sample projected image features with continuous grid coordinates."""

    def __init__(self, model_cfg=None):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.align_corners = bool(self.model_cfg.get("align_corners", False))

    def reshape_feature_map(
        self, feature_map: torch.Tensor, num_views: int
    ) -> torch.Tensor:
        """Convert image feature maps to [B, V, C, H, W]."""
        assert feature_map.ndim == 5, "image feature maps must have shape [B, num_views, C, H, W]."
        assert int(feature_map.shape[1]) == num_views, "image feature num_views mismatch."
        return feature_map

    def sample(
        self,
        feature_map,
        normalized_coords: torch.Tensor,
        local_agent_ids: torch.Tensor,
        view_ids: torch.Tensor,
        num_views: int,
    ) -> Optional[torch.Tensor]:
        """Sample one feature vector for each projected observation."""

        feature_map = self.reshape_feature_map(feature_map, num_views)
        grid = normalized_coords * 2.0 - 1.0
        local_agent_ids = local_agent_ids.long()
        view_ids = view_ids.long()

        channels = int(feature_map.shape[2])
        sampled_features = torch.empty(
            (grid.shape[0], channels), device=feature_map.device, dtype=feature_map.dtype
        )

        # Group by (agent, view) and sample many points from one map each call.
        flat_pair_ids = local_agent_ids * int(num_views) + view_ids
        unique_pair_ids, inverse_indices = torch.unique(
            flat_pair_ids, sorted=False, return_inverse=True
        )

        for pair_idx in range(unique_pair_ids.shape[0]):
            hit_mask = inverse_indices == pair_idx
            if not torch.any(hit_mask):
                continue
            pair_id = unique_pair_ids[pair_idx]
            agent_id = torch.div(pair_id, int(num_views), rounding_mode="floor")
            view_id = torch.remainder(pair_id, int(num_views))

            current_feature_map = feature_map[
                agent_id : agent_id + 1, view_id : view_id + 1
            ].squeeze(1)
            current_grid = grid[hit_mask].view(1, -1, 1, 2)
            current_sampled = F.grid_sample(
                current_feature_map,
                current_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=self.align_corners,
            )
            sampled_features[hit_mask] = current_sampled[0, :, :, 0].transpose(0, 1).contiguous()

        return sampled_features

    def sample_feature_dict(
        self,
        feature_dict: Dict[str, Optional[torch.Tensor]],
        normalized_coords: torch.Tensor,
        local_agent_ids: torch.Tensor,
        view_ids: torch.Tensor,
        num_views: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Sample multiple projected feature maps with one shared set of coordinates."""
        outputs: Dict[str, Optional[torch.Tensor]] = {}
        import pdb; pdb.set_trace()
        for name, feature_map in feature_dict.items():
            outputs[name] = self.sample(
                feature_map,
                normalized_coords,
                local_agent_ids,
                view_ids,
                num_views,
            )
        return outputs

    def forward(
        self,
        feature_map,
        normalized_coords: torch.Tensor,
        local_agent_ids: torch.Tensor,
        view_ids: torch.Tensor,
        num_views: int,
    ) -> Optional[torch.Tensor]:
        return self.sample(
            feature_map,
            normalized_coords,
            local_agent_ids,
            view_ids,
            num_views,
        )

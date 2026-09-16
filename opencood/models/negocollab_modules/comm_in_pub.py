# -*- coding: utf-8 -*-
"""Official NegoCollab ComminPub (send / receive) without debug visualization."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.negocollab_modules.converter import Converter, CrossdomianConverter
from opencood.models.negocollab_modules.resize_net import ResizeNet
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


class ComminPub(nn.Module):
    """Local ↔ common-space communication module.

    Sender: local BEV → unify size → recombiner → converter.
    Receiver: ego-conditioned cross-domain converter → recombiner → local size.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.local_range = args["local_range"]
        self.inference = bool(args.get("inference", False))
        unify = args["unify_parameters"]
        self.modality = args.get("modality", "unknown")
        self.unify_granularity_H = unify["granularity_H"]
        self.unify_granularity_W = unify["granularity_W"]
        self.unify_channel = unify["C_uni"]

        resizer = args["resizer"]
        self.local2unify = ResizeNet(
            resizer["local_dim"], self.unify_channel, unify_method=resizer.get("method", "conv")
        )
        self.unify2local = ResizeNet(
            self.unify_channel, resizer["local_dim"], unify_method=resizer.get("method", "conv")
        )
        self.sender_recombiner = AlignNet(args["recombiner"])
        self.sender_aligner = Converter(args["converter"])
        self.local_prompt_generator = AlignNet(args["prompt_generator"])
        self.receiver_converter = CrossdomianConverter(args["converter"])
        self.receiver_recombiner = AlignNet(args["recombiner"])
        self.local_feature: Optional[Tensor] = None
        self.local_size = None
        self.unify_size = None

    def send(self, feature: Tensor) -> Tensor:
        """Map local-backbone features into the common representation."""
        self.local_size = feature.size()[1:]
        local_granularity = (
            (self.local_range[4] - self.local_range[1]) / self.local_size[1],
            (self.local_range[3] - self.local_range[0]) / self.local_size[2],
        )
        unify_ratio = (
            local_granularity[0] / self.unify_granularity_H,
            local_granularity[1] / self.unify_granularity_W,
        )
        self.unify_size = (
            self.unify_channel,
            int(unify_ratio[0] * self.local_size[1]),
            int(unify_ratio[1] * self.local_size[2]),
        )
        feature = self.local2unify(feature, self.unify_size)
        feature = self.sender_recombiner(feature)
        self.local_feature = feature
        return self.sender_aligner(feature)

    def receive(
        self,
        feature: Tensor,
        record_len=None,
        affine_matrix=None,
        record_len_modality=None,
    ) -> Tensor:
        """Map public features back into the ego local feature space."""
        feature = self.convert_by_ego(
            feature, record_len, affine_matrix, record_len_modality
        )
        feature = self.receiver_recombiner(feature)
        feature = self.unify2local(feature, self.local_size)
        return feature

    def convert_by_ego(
        self,
        feature: Tensor,
        record_len=None,
        affine_matrix=None,
        record_len_modality=None,
    ) -> Tensor:
        """Ego-conditioned conversion of neighbor public features.

        `local_feature` must have been set by a preceding `send` call and
        must contain one entry per packed agent (`sum(record_len)`). Only
        the first agent of each scene is used as the ego prompt.
        """
        if self.local_feature is None:
            raise RuntimeError("ComminPub.receive requires a prior send() to set local_feature")
        if affine_matrix is None or record_len is None:
            raise ValueError("receive() requires record_len and affine_matrix")

        batch_size = int(affine_matrix.shape[0])
        _, _, height, width = feature.shape
        local_feature = self.local_prompt_generator(self.local_feature)
        if self.inference and record_len_modality is not None:
            local_split = regroup(local_feature, record_len_modality)
        else:
            local_split = regroup(local_feature, record_len)
        received_split = regroup(feature, record_len)

        out = []
        for batch_idx in range(batch_size):
            n_agents = int(record_len[batch_idx])
            t_matrix = affine_matrix[batch_idx][:n_agents, :n_agents, :, :]
            ego = local_split[batch_idx][0].unsqueeze(0).expand(n_agents, -1, -1, -1)
            ego_in_neb = warp_affine_simple(
                ego, t_matrix[:, 0, :, :], (height, width), align_corners=True
            )
            out.append(self.receiver_converter(ego_in_neb, received_split[batch_idx]))
        return torch.cat(out, dim=0)

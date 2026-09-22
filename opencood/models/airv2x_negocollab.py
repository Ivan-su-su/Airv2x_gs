# -*- coding: utf-8 -*-
"""AirV2X NegoCollab: three independent local bases + common-space negotiation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from opencood.models.airv2x_detector_parts import Airv2xSharedDetector
from opencood.models.common_modules.airv2x_base_model import Airv2xBase
from opencood.models.lss_pretrain_modules.frozen_p1 import FrozenP1
from opencood.models.lss_pretrain_modules.p1_mixin import make_p1_splat_encoder
from opencood.models.negocollab_modules.comm_in_pub import ComminPub
from opencood.models.negocollab_modules.negotiator import Negotiator
from opencood.models.negocollab_modules.resize_net import ResizeNet
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.utils.airv2x_agent_repack import (
    flatten_scene_agent_types,
    get_agent_order,
    infer_batch_size,
    repack_agent_features,
    scene_agent_types,
)
from opencood.utils.airv2x_freeze import (
    assert_lss_frozen,
    freeze_lss_encoders,
    print_trainable_parameter_groups,
    set_module_trainable,
)
from opencood.utils.negocollab_transform import to_normalized_affine


class LocalPerceptionBranch(nn.Module):
    """One homogeneous Stage-0 stack: LSS → backbone → aligner → fusion → heads."""

    def __init__(self, args: Dict[str, Any], agent_type: str, p1_core: FrozenP1) -> None:
        super().__init__()
        self.agent_type = agent_type
        self.encoder = make_p1_splat_encoder(args, agent_type, p1_core.encode, p1_core)
        self.detector = Airv2xSharedDetector(args)
        self.backbone = self.detector.backbone
        self.pyramid_backbone = self.detector.pyramid_backbone
        self.shrink_conv = self.detector.shrink_conv
        self.cls_head = self.detector.cls_head
        self.reg_head = self.detector.reg_head
        self.obj_head = self.detector.obj_head
        aligner_cfg = args[agent_type].get("aligner") or {"core_method": "identity"}
        self.aligner = AlignNet(aligner_cfg)

    def encode_local(self, data_dict: Dict[str, Any]) -> Tensor:
        """Frozen LSS → BEV backbone → optional identity aligner."""
        encoded = self.encoder(data_dict)
        bev = self.backbone(encoded)["spatial_features_2d"]
        return self.aligner(bev)


class Airv2xNegoCollab(Airv2xBase):
    """Faithful NegoCollab on AirV2X camera agents.

    Communication happens AFTER the local BEV backbone and BEFORE the
    local PyramidFusion / detection head.

    Stage 1 (`stage='nego'`) freezes every local perception stack and
    trains senders, the negotiator, and receivers.

    Stage 2 (`stage='ft'`) freezes local stacks + senders and trains
    receiver-side modules only.

    Inference uses the ego (default: vehicle) local fusion/head, with
    the official ego bypass (ego keeps its original local BEV feature).
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__(args)
        self.args = args
        self.stage = str(args.get("stage", "inf"))
        self.ego_type = str(args.get("ego_type", "vehicle"))
        self.cav_range = args.get("cav_range") or args.get("lidar_range")
        if self.cav_range is None:
            raise KeyError("NegoCollab requires args['cav_range']")
        self.collaborators = list(args["collaborators"])
        self.active_sensors = args["active_sensors"]
        if not args.get("p1_checkpoint"):
            raise ValueError("NegoCollab requires --p1_checkpoint for the P1 homo bases")
        self.p1_core = FrozenP1(args)

        self.local_branches = nn.ModuleDict()
        self.comms = nn.ModuleDict()
        self.nego_resizers = nn.ModuleDict()
        for agent_type in self.collaborators:
            self.local_branches[agent_type] = LocalPerceptionBranch(
                args, agent_type, self.p1_core
            )
            comm_args = dict(args[agent_type]["comm_args"])
            comm_args.setdefault("local_range", list(self.cav_range))
            comm_args["modality"] = agent_type
            comm_args["inference"] = self.stage in ("inf", "test")
            self.comms[agent_type] = ComminPub(comm_args)
            local_dim = int(args[agent_type].get("local_dim", 64))
            self.nego_resizers[agent_type] = ResizeNet(
                local_dim, int(args["pub"]["C_uni"]), unify_method="conv"
            )

        self.negotiator = Negotiator(args["pub"]["negotiator"])
        self.occ_head_nego = nn.Conv2d(int(args["pub"]["in_head"]), 1, kernel_size=1)
        self._apply_stage_freeze()

    def _apply_stage_freeze(self) -> None:
        freeze_lss_encoders(self)
        for branch in self.local_branches.values():
            set_module_trainable(branch, False)
        for comm in self.comms.values():
            set_module_trainable(comm, False)
        set_module_trainable(self.negotiator, False)
        set_module_trainable(self.occ_head_nego, False)
        for resizer in self.nego_resizers.values():
            set_module_trainable(resizer, False)

        if self.stage == "nego":
            for comm in self.comms.values():
                set_module_trainable(comm, True)
            set_module_trainable(self.negotiator, True)
            set_module_trainable(self.occ_head_nego, True)
            for resizer in self.nego_resizers.values():
                set_module_trainable(resizer, True)
        elif self.stage == "ft":
            for name, param in self.named_parameters():
                if any(
                    token in name
                    for token in ("receiver", "local_prompt_generator", "unify2local")
                ):
                    param.requires_grad = True
            for name, module in self.named_modules():
                if any(
                    token in name
                    for token in ("receiver", "local_prompt_generator", "unify2local")
                ):
                    module.train()
        elif self.stage in ("inf", "test"):
            pass
        else:
            raise ValueError(f"Unknown NegoCollab stage: {self.stage}")

        freeze_lss_encoders(self)
        assert_lss_frozen(self)
        groups = print_trainable_parameter_groups(self, title=f"NegoCollab stage={self.stage}")
        if self.stage in ("nego", "ft") and sum(groups.values()) == 0:
            raise RuntimeError(
                f"NegoCollab stage={self.stage} has zero trainable parameters"
            )

    def train(self, mode: bool = True) -> "Airv2xNegoCollab":
        super().train(mode)
        # The local detector and P1 frontend remain frozen in both stages.
        self.p1_core.eval()
        self.local_branches.eval()
        if self.stage == "ft":
            self.negotiator.eval()
            self.occ_head_nego.eval()
            self.nego_resizers.eval()
            for comm in self.comms.values():
                for name in ("local2unify", "sender_recombiner", "sender_aligner"):
                    module = getattr(comm, name, None)
                    if module is not None:
                        module.eval()
        return self

    def _encode_local_by_type(self, data_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        local_by_type: Dict[str, Tensor] = {}
        for agent_type in self.collaborators:
            agent = data_dict.get(agent_type)
            if not isinstance(agent, dict) or len(agent.get("batch_idxs") or []) == 0:
                continue
            local_by_type[agent_type] = self.local_branches[agent_type].encode_local(
                data_dict
            )
        if not local_by_type:
            raise RuntimeError("NegoCollab batch has no collaborating agents")
        return local_by_type

    def _send_by_type(self, local_by_type: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sent: Dict[str, Tensor] = {}
        for agent_type, feat in local_by_type.items():
            sent[agent_type] = self.comms[agent_type].send(feat)
        return sent

    def _pack(
        self, feat_by_type: Dict[str, Tensor], data_dict: Dict[str, Any]
    ) -> Tuple[Tensor, Tensor]:
        return repack_agent_features(
            feat_by_type, data_dict, batch_size=infer_batch_size(data_dict)
        )

    def _normalized_affine(self, data_dict: Dict[str, Any]) -> Tensor:
        return to_normalized_affine(
            data_dict["img_pairwise_t_matrix_collab"],
            cav_lidar_range=self.cav_range,
        )

    def _expand_scene_feature(self, scene_feat: Tensor, record_len: Tensor) -> Tensor:
        """Broadcast `[B, C, H, W]` to `[sum N, C, H, W]` by record_len."""
        parts: List[Tensor] = []
        for batch_idx, n_agents in enumerate(record_len.tolist()):
            parts.append(scene_feat[batch_idx : batch_idx + 1].expand(int(n_agents), -1, -1, -1))
        return torch.cat(parts, dim=0)

    def _type_mean_per_scene(
        self,
        local_by_type: Dict[str, Tensor],
        data_dict: Dict[str, Any],
        batch_size: int,
    ) -> Dict[str, Tensor]:
        """Mean-pool each agent type inside every scene → `[B, C, H, W]`.

        AirV2X scenes have unequal agent counts per type, so official
        NegoCollab's equal-N stack cannot be used directly. Type-mean
        prototypes keep the negotiator's "one tensor per modality" API.
        """
        order = get_agent_order(data_dict)
        ref = next(iter(local_by_type.values()))
        means: Dict[str, Tensor] = {}
        for agent_type in order:
            feat = local_by_type.get(agent_type)
            per_scene: List[Tensor] = []
            for batch_idx in range(batch_size):
                if feat is None:
                    per_scene.append(torch.zeros_like(ref[:1]))
                    continue
                agent = data_dict[agent_type]
                n = int(agent["record_len"][batch_idx].item())
                if n <= 0 or batch_idx not in list(agent.get("batch_idxs") or []):
                    per_scene.append(torch.zeros_like(ref[:1]))
                    continue
                tensor_idx = list(agent["batch_idxs"]).index(batch_idx)
                valid = agent["record_len"][agent["record_len"] > 0]
                split = torch.split(feat, [int(x) for x in valid.tolist()], dim=0)
                per_scene.append(split[tensor_idx].mean(dim=0, keepdim=True))
            means[agent_type] = torch.cat(per_scene, dim=0)
        return means

    def _ego_indices(self, record_len: Tensor) -> List[int]:
        idxs: List[int] = []
        cursor = 0
        for n_agents in record_len.tolist():
            idxs.append(cursor)
            cursor += int(n_agents)
        return idxs

    def _receive_all(
        self,
        packed_sent: Tensor,
        packed_local_common: Tensor,
        record_len: Tensor,
        affine: Tensor,
        local_size: Sequence[int],
    ) -> Dict[str, Tensor]:
        restored: Dict[str, Tensor] = {}
        for agent_type, comm in self.comms.items():
            comm.local_feature = packed_local_common
            comm.local_size = tuple(local_size)
            restored[agent_type] = comm.receive(packed_sent, record_len, affine)
        return restored

    def _detect_from_local(
        self,
        restored_local: Tensor,
        record_len: Tensor,
        data_dict: Dict[str, Any],
        packed_local: Tensor,
    ) -> Dict[str, Any]:
        ego_idxs = self._ego_indices(record_len)
        restored_local = restored_local.clone()
        restored_local[ego_idxs] = packed_local[ego_idxs]
        ego_branch = self.local_branches[self.ego_type]
        fused, _occ = ego_branch.detector.fuse_and_decode(
            restored_local,
            record_len,
            data_dict["img_pairwise_t_matrix_collab"],
        )
        output = ego_branch.detector.decode_heads(fused)
        output["pyramid"] = "collab"
        output["comm_rate"] = packed_local.count_nonzero().item()
        return output

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run NegoCollab stage-1 / stage-2 / inference."""
        batch_size = infer_batch_size(data_dict)
        local_by_type = self._encode_local_by_type(data_dict)
        packed_local, record_len = self._pack(local_by_type, data_dict)
        types_flat = flatten_scene_agent_types(data_dict, batch_size=batch_size)
        if len(types_flat) != packed_local.shape[0]:
            raise AssertionError("Packed local features are not aligned with agent_order")
        if types_flat[0] != self.ego_type:
            print(
                f"[NegoCollab] warning: first packed agent is {types_flat[0]}, "
                f"configured ego_type={self.ego_type}"
            )

        sent_by_type = self._send_by_type(local_by_type)
        packed_sent, packed_len = self._pack(sent_by_type, data_dict)
        if packed_len.tolist() != record_len.tolist():
            raise AssertionError("Sent features and local features have different record_len")

        local_common_by_type = {
            agent_type: self.comms[agent_type].local_feature
            for agent_type in sent_by_type
        }
        packed_local_common, _ = self._pack(local_common_by_type, data_dict)
        affine = self._normalized_affine(data_dict)
        local_size = tuple(packed_local.shape[1:])

        output: Dict[str, Any] = {
            "nego_stage": self.stage,
            "modality_name_list": list(self.collaborators),
            "agent_order": get_agent_order(data_dict),
            "scene_agent_types": scene_agent_types(data_dict, batch_size=batch_size),
        }

        if self.stage == "nego":
            type_means = self._type_mean_per_scene(local_by_type, data_dict, batch_size)
            standard_size = packed_sent[0].size()
            standard_list = []
            for agent_type in get_agent_order(data_dict):
                if agent_type not in type_means:
                    continue
                resized = self.nego_resizers[agent_type](type_means[agent_type], standard_size)
                standard_list.append(resized)
            common_rep = self.negotiator(standard_list)
            common_expanded = self._expand_scene_feature(common_rep, record_len)
            restored = self._receive_all(
                packed_sent, packed_local_common, record_len, affine, local_size
            )
            output.update(
                {
                    "common_rep": common_rep,
                    "packed_sent": packed_sent,
                    "packed_local": packed_local,
                    "fc_before_send": packed_local,
                    "fc_after_receive": restored[self.ego_type],
                    "fc_after_receive_by_type": restored,
                    "occ_preds_nego": self.occ_head_nego(common_rep),
                    "common_expanded": common_expanded,
                }
            )
            return output

        restored = self._receive_all(
            packed_sent, packed_local_common, record_len, affine, local_size
        )[self.ego_type]
        det = self._detect_from_local(restored, record_len, data_dict, packed_local)
        output.update(det)
        return output

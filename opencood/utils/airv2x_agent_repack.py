# -*- coding: utf-8 -*-
"""Agent-order helpers shared by HEAL / STAMP / NegoCollab.

AirV2X collate builds `record_len` and `img_pairwise_t_matrix_collab` in
`data_dict['agent_order']` order. Default ego=vehicle therefore yields:

    vehicles → rsus → drones

Do not hard-code this order. Always read it from the batch.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


AGENT_TYPES: Tuple[str, ...] = ("vehicle", "rsu", "drone")
DEFAULT_AGENT_ORDER: Tuple[str, ...] = ("vehicle", "rsu", "drone")


def get_agent_order(data_dict: Mapping) -> List[str]:
    """Return the scene agent-type order used by this batch."""
    order = data_dict.get("agent_order")
    if order:
        return list(order)
    return list(DEFAULT_AGENT_ORDER)


def infer_batch_size(data_dict: Mapping) -> int:
    """Infer B from per-type `batch_idxs` / `record_len`."""
    sizes: List[int] = []
    for agent_type in AGENT_TYPES:
        agent = data_dict.get(agent_type)
        if not isinstance(agent, dict):
            continue
        record_len = agent.get("record_len")
        if torch.is_tensor(record_len):
            sizes.append(int(record_len.shape[0]))
        batch_idxs = agent.get("batch_idxs") or []
        sizes.append(len(batch_idxs))
    if not sizes:
        raise KeyError("Cannot infer batch size: no agent records in data_dict")
    return max(sizes)


def type_tensor_index(data_dict: Mapping, agent_type: str, batch_idx: int) -> Optional[int]:
    """Index of `batch_idx` inside the concatenated per-type feature tensor."""
    agent = data_dict.get(agent_type)
    if not isinstance(agent, dict):
        return None
    batch_idxs = list(agent.get("batch_idxs") or [])
    if batch_idx not in batch_idxs:
        return None
    return batch_idxs.index(batch_idx)


def scene_agent_types(
    data_dict: Mapping,
    batch_size: Optional[int] = None,
) -> List[List[str]]:
    """Per-sample list of agent types in the same order as `record_len`.

    Example for ego=vehicle, one sample with 2 veh / 1 rsu / 1 drone::

        [["vehicle", "vehicle", "rsu", "drone"]]
    """
    order = get_agent_order(data_dict)
    B = infer_batch_size(data_dict) if batch_size is None else int(batch_size)
    scenes: List[List[str]] = []
    for batch_idx in range(B):
        types: List[str] = []
        for agent_type in order:
            agent = data_dict.get(agent_type)
            if not isinstance(agent, dict):
                continue
            record_len = agent.get("record_len")
            if record_len is None:
                continue
            n = int(record_len[batch_idx].item()) if torch.is_tensor(record_len) else int(record_len[batch_idx])
            if n <= 0:
                continue
            if type_tensor_index(data_dict, agent_type, batch_idx) is None:
                continue
            types.extend([agent_type] * n)
        scenes.append(types)
    return scenes


def flatten_scene_agent_types(
    data_dict: Mapping,
    batch_size: Optional[int] = None,
) -> List[str]:
    """Flattened agent-type list aligned with packed features `[sum(record_len)]`."""
    types: List[str] = []
    for scene in scene_agent_types(data_dict, batch_size=batch_size):
        types.extend(scene)
    return types


def _split_type_tensor(
    feat: Tensor,
    record_len: Tensor,
    tensor_idx: int,
) -> Tensor:
    """Slice one sample out of a concatenated per-type tensor."""
    valid = record_len[record_len > 0]
    if valid.numel() == 0:
        raise RuntimeError("record_len has no positive entries")
    splits = torch.split(feat, [int(x) for x in valid.tolist()], dim=0)
    return splits[tensor_idx]


def repack_agent_features(
    feature_dict_by_type: Mapping[str, Tensor],
    data_dict: Mapping,
    batch_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Pack per-type feature tensors into scene-agent order.

    Args:
        feature_dict_by_type: `{agent_type: Tensor[N_type, C, H, W]}` produced
            after LSS / backbone / adapter. Missing types may be omitted.
        data_dict: Collated ego batch (must include per-type `record_len`
            and `batch_idxs`, plus optional `agent_order`).
        batch_size: Optional explicit B.

    Returns:
        packed: `[sum(n_cav), C, H, W]`
        record_len: `[B]` int tensor on the feature device.

    Notes:
        Order matches `Airv2xBase.repack_batch` / pairwise transforms:
        within each sample, agents follow `agent_order`, skipping empty types.
    """
    order = get_agent_order(data_dict)
    B = infer_batch_size(data_dict) if batch_size is None else int(batch_size)
    sample_feats: List[Tensor] = []
    record_len_list: List[int] = []
    ref_feat: Optional[Tensor] = None

    for batch_idx in range(B):
        parts: List[Tensor] = []
        n_agents = 0
        for agent_type in order:
            feat = feature_dict_by_type.get(agent_type)
            if feat is None:
                continue
            ref_feat = feat
            tensor_idx = type_tensor_index(data_dict, agent_type, batch_idx)
            if tensor_idx is None:
                continue
            record_len = data_dict[agent_type]["record_len"]
            n = int(record_len[batch_idx].item())
            if n <= 0:
                continue
            sliced = _split_type_tensor(feat, record_len, tensor_idx)
            if sliced.shape[0] != n:
                raise AssertionError(
                    f"{agent_type} sample {batch_idx}: sliced {sliced.shape[0]} "
                    f"agents, record_len={n}"
                )
            parts.append(sliced)
            n_agents += n
        if not parts:
            raise RuntimeError(f"Sample {batch_idx} has no agents to pack")
        sample_feats.append(torch.cat(parts, dim=0))
        record_len_list.append(n_agents)

    packed = torch.cat(sample_feats, dim=0)
    device = packed.device if ref_feat is None else ref_feat.device
    record_len = torch.tensor(record_len_list, device=device, dtype=torch.int32)

    expected = int(record_len.sum().item())
    if packed.shape[0] != expected:
        raise AssertionError(
            f"Packed feature N={packed.shape[0]} != sum(record_len)={expected}"
        )
    scene_types = flatten_scene_agent_types(data_dict, batch_size=B)
    if len(scene_types) != packed.shape[0]:
        raise AssertionError(
            f"Agent-type list length {len(scene_types)} != packed N={packed.shape[0]}"
        )
    return packed, record_len

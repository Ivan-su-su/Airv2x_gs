# -*- coding: utf-8 -*-
"""Explicit Stage-0 checkpoint loading for AirV2X HEAL / STAMP / NegoCollab.

Never sequentially load full checkpoints into a shared namespace. Every copy
operation has a semantic source → destination mapping and fails loudly on
shape / coverage problems.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from opencood.utils.airv2x_freeze import AGENT_ENCODER_ATTR


SHARED_DETECTOR_PREFIXES: Tuple[str, ...] = (
    "backbone.",
    "pyramid_backbone.",
    "shrink_conv.",
    "cls_head.",
    "reg_head.",
    "obj_head.",
    "seg_head.",
)


@dataclass
class CheckpointLoadReport:
    """Audited result of a semantic weight copy."""

    source: str
    destination: str
    matched: int = 0
    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    shape_mismatch: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a one-block human-readable summary."""
        lines = [
            f"[ckpt] {self.source} → {self.destination}",
            f"  matched={self.matched}",
            f"  missing={len(self.missing)} unexpected={len(self.unexpected)} "
            f"shape_mismatch={len(self.shape_mismatch)}",
        ]
        if self.missing[:8]:
            lines.append("  missing examples: " + ", ".join(self.missing[:8]))
        if self.unexpected[:8]:
            lines.append("  unexpected examples: " + ", ".join(self.unexpected[:8]))
        if self.shape_mismatch[:8]:
            lines.append("  shape mismatch: " + ", ".join(self.shape_mismatch[:8]))
        return "\n".join(lines)


def _strip_module_prefix(key: str) -> str:
    if key.startswith("module.") and not key.startswith("module_list"):
        return key[7:]
    return key


def extract_state_dict(blob: object) -> Dict[str, torch.Tensor]:
    """Unwrap common checkpoint containers into a flat state_dict."""
    if not isinstance(blob, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(blob)}")

    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        inner = blob.get(key)
        if isinstance(inner, dict) and inner:
            blob = inner
            break

    state: Dict[str, torch.Tensor] = {}
    for raw_key, value in blob.items():
        if not torch.is_tensor(value):
            continue
        state[_strip_module_prefix(str(raw_key))] = value
    if not state:
        raise RuntimeError("Checkpoint contains no tensor parameters.")
    return state


def load_raw_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load a `.pth` file and unwrap it to a flat state_dict."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    blob = torch.load(path, map_location=map_location)
    return extract_state_dict(blob)


def _find_last_epoch(save_dir: str) -> int:
    file_list = glob.glob(os.path.join(save_dir, "*epoch*.pth"))
    epochs: List[int] = []
    for file_ in file_list:
        match = re.findall(r".*epoch(.*).pth.*", os.path.basename(file_))
        if not match:
            continue
        try:
            epochs.append(int(match[0]))
        except ValueError:
            continue
    return max(epochs) if epochs else 0


def resolve_checkpoint_path(
    model_dir_or_file: str,
    epoch: Optional[int] = None,
    start_from_best: bool = True,
) -> str:
    """Resolve a directory or file path to a concrete checkpoint file."""
    if os.path.isfile(model_dir_or_file):
        return model_dir_or_file
    if not os.path.isdir(model_dir_or_file):
        raise FileNotFoundError(model_dir_or_file)

    if start_from_best:
        best = glob.glob(os.path.join(model_dir_or_file, "net_epoch_bestval_at*.pth"))
        if len(best) == 1:
            return best[0]

    epoch_id = epoch if epoch is not None else _find_last_epoch(model_dir_or_file)
    if epoch_id <= 0:
        raise FileNotFoundError(
            f"No epoch checkpoint found under {model_dir_or_file}"
        )
    path = os.path.join(model_dir_or_file, f"net_epoch{epoch_id}.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def filter_by_prefix(
    state: Mapping[str, torch.Tensor],
    prefixes: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Keep keys that start with any of `prefixes`."""
    kept: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            kept[key] = value
    return kept


def remap_prefix(
    state: Mapping[str, torch.Tensor],
    src_prefix: str,
    dst_prefix: str,
) -> Dict[str, torch.Tensor]:
    """Rewrite keys from `src_prefix` to `dst_prefix`."""
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith(src_prefix):
            remapped[dst_prefix + key[len(src_prefix) :]] = value
    return remapped


def copy_matching_tensors(
    module: nn.Module,
    mapped_state: Mapping[str, torch.Tensor],
    *,
    source: str,
    destination: str,
    min_match: int = 1,
    min_match_ratio: float = 0.5,
) -> CheckpointLoadReport:
    """Copy tensors into `module` with explicit shape / coverage checks.

    Args:
        module: Destination module.
        mapped_state: Keys must already match `module.state_dict()` names.
        source: Human-readable source label.
        destination: Human-readable destination label.
        min_match: Fail if fewer tensors match.
        min_match_ratio: Fail if matched / destination_keys is below this.

    Returns:
        CheckpointLoadReport describing the copy.
    """
    dst_state = module.state_dict()
    report = CheckpointLoadReport(source=source, destination=destination)
    compatible: Dict[str, torch.Tensor] = {}

    for key, tensor in mapped_state.items():
        if key not in dst_state:
            report.unexpected.append(key)
            continue
        if tuple(dst_state[key].shape) != tuple(tensor.shape):
            report.shape_mismatch.append(
                f"{key}: ckpt {tuple(tensor.shape)} vs model {tuple(dst_state[key].shape)}"
            )
            continue
        compatible[key] = tensor

    for key in dst_state:
        if key not in compatible:
            report.missing.append(key)

    report.matched = len(compatible)
    n_dst = max(len(dst_state), 1)
    print(report.summary())

    if report.shape_mismatch:
        raise RuntimeError(
            f"Shape mismatch while loading {source} → {destination}: "
            + "; ".join(report.shape_mismatch[:8])
        )
    if report.matched < min_match:
        raise RuntimeError(
            f"Suspiciously few parameters matched while loading {source} → "
            f"{destination}: matched={report.matched}, dest_keys={len(dst_state)}"
        )
    if report.matched / n_dst < min_match_ratio:
        raise RuntimeError(
            f"Low coverage while loading {source} → {destination}: "
            f"{report.matched}/{len(dst_state)} "
            f"({report.matched / n_dst:.1%}) < min_match_ratio={min_match_ratio:.1%}"
        )

    incompatible = module.load_state_dict(compatible, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys after filtered load into {destination}: "
            f"{incompatible.unexpected_keys[:8]}"
        )
    return report


def load_agent_encoder(
    model: nn.Module,
    ckpt_state: Mapping[str, torch.Tensor],
    agent_type: str,
    *,
    source: str,
) -> CheckpointLoadReport:
    """Load one agent-type LSS encoder from a Stage-0 checkpoint."""
    attr = AGENT_ENCODER_ATTR[agent_type]
    module = getattr(model, attr, None)
    if module is None:
        raise AttributeError(f"{type(model).__name__} has no encoder attr '{attr}'")
    prefix = f"{attr}."
    mapped = remap_prefix(filter_by_prefix(ckpt_state, (prefix,)), prefix, "")
    return copy_matching_tensors(
        module,
        mapped,
        source=f"{source}:{prefix}*",
        destination=f"{type(model).__name__}.{attr}",
    )


def load_shared_detector(
    model: nn.Module,
    ckpt_state: Mapping[str, torch.Tensor],
    *,
    source: str,
    prefixes: Sequence[str] = SHARED_DETECTOR_PREFIXES,
) -> List[CheckpointLoadReport]:
    """Load shared backbone / fusion / shrink / heads from a Stage-0 checkpoint."""
    reports: List[CheckpointLoadReport] = []
    for prefix in prefixes:
        attr = prefix.rstrip(".")
        module = getattr(model, attr, None)
        if module is None:
            continue
        mapped = remap_prefix(filter_by_prefix(ckpt_state, (prefix,)), prefix, "")
        if not mapped and sum(1 for _ in module.parameters()) == 0:
            continue
        reports.append(
            copy_matching_tensors(
                module,
                mapped,
                source=f"{source}:{prefix}*",
                destination=f"{type(model).__name__}.{attr}",
            )
        )
    if not reports:
        raise RuntimeError(
            f"No shared detector modules were loaded from {source}. "
            f"Looked for prefixes {prefixes}."
        )
    return reports


def load_local_branch(
    branch: nn.Module,
    ckpt_state: Mapping[str, torch.Tensor],
    agent_type: str,
    *,
    source: str,
) -> List[CheckpointLoadReport]:
    """Load a full homogeneous base into one NegoCollab local branch.

    Stage-0 keys such as `backbone.*` are remapped onto
    `branch.backbone.*`, and the agent encoder is remapped onto
    `branch.encoder.*`.
    """
    reports: List[CheckpointLoadReport] = []
    encoder_attr = AGENT_ENCODER_ATTR[agent_type]
    encoder_prefix = f"{encoder_attr}.0."
    encoder = getattr(branch, "encoder", None)
    if encoder is None:
        raise AttributeError("local branch has no `.encoder`")
    mapped_encoder = remap_prefix(
        filter_by_prefix(ckpt_state, (encoder_prefix,)), encoder_prefix, ""
    )
    reports.append(
        copy_matching_tensors(
            encoder,
            mapped_encoder,
            source=f"{source}:{encoder_prefix}*",
            destination=f"local_branches.{agent_type}.encoder",
        )
    )

    branch_map = {
        "backbone.": "backbone",
        "pyramid_backbone.": "pyramid_backbone",
        "shrink_conv.": "shrink_conv",
        "cls_head.": "cls_head",
        "reg_head.": "reg_head",
        "obj_head.": "obj_head",
    }
    for src_prefix, attr in branch_map.items():
        module = getattr(branch, attr, None)
        if module is None:
            continue
        mapped = remap_prefix(filter_by_prefix(ckpt_state, (src_prefix,)), src_prefix, "")
        reports.append(
            copy_matching_tensors(
                module,
                mapped,
                source=f"{source}:{src_prefix}*",
                destination=f"local_branches.{agent_type}.{attr}",
            )
        )
    return reports


def load_stage0_bases_for_heal_or_stamp(
    model: nn.Module,
    *,
    vehicle_ckpt: str,
    rsu_ckpt: str,
    drone_ckpt: str,
    vehicle_epoch: Optional[int] = None,
    rsu_epoch: Optional[int] = None,
    drone_epoch: Optional[int] = None,
    map_location: str = "cpu",
) -> Dict[str, List[CheckpointLoadReport]]:
    """Load three Stage-0 bases into an AirV2X HEAL / STAMP model.

    Encoders come from the matching agent-type checkpoint. Shared
    backbone / fusion / heads always come from the **vehicle** checkpoint.
    """
    vehicle_path = resolve_checkpoint_path(vehicle_ckpt, vehicle_epoch)
    rsu_path = resolve_checkpoint_path(rsu_ckpt, rsu_epoch)
    drone_path = resolve_checkpoint_path(drone_ckpt, drone_epoch)
    vehicle_state = load_raw_checkpoint(vehicle_path, map_location)
    rsu_state = load_raw_checkpoint(rsu_path, map_location)
    drone_state = load_raw_checkpoint(drone_path, map_location)

    reports: Dict[str, List[CheckpointLoadReport]] = {
        "vehicle_encoder": [
            load_agent_encoder(model, vehicle_state, "vehicle", source=vehicle_path)
        ],
        "rsu_encoder": [
            load_agent_encoder(model, rsu_state, "rsu", source=rsu_path)
        ],
        "drone_encoder": [
            load_agent_encoder(model, drone_state, "drone", source=drone_path)
        ],
        "shared_detector_from_vehicle": load_shared_detector(
            model, vehicle_state, source=vehicle_path
        ),
    }
    print("[ckpt] HEAL/STAMP mapping: encoders from matching bases; "
          "shared backbone/fusion/heads from vehicle base.")
    return reports

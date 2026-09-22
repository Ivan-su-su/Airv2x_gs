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

    if start_from_best and epoch is None:
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
    if min_match_ratio == 1.0 and report.unexpected:
        raise RuntimeError(
            f"Unexpected pretrained tensors while loading {source} → {destination}: "
            + "; ".join(report.unexpected[:8])
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
    if len(module) != 1 or not hasattr(module[0], "bevencode"):
        raise RuntimeError(f"{attr} must contain exactly one P1SplatEncoder")
    prefix = f"{attr}.0.bevencode."
    mapped = remap_prefix(filter_by_prefix(ckpt_state, (prefix,)), prefix, "")
    return copy_matching_tensors(
        module[0].bevencode,
        mapped,
        source=f"{source}:{prefix}*",
        destination=f"{type(model).__name__}.{attr}[0].bevencode",
        min_match_ratio=1.0,
    )


def verify_frozen_p1(model: nn.Module, ckpt_state: Mapping[str, torch.Tensor],
                     agent_type: str, *, source: str) -> None:
    """Fail if the Stage-0 frozen P1 differs from the Stage-2 P1 frontend."""
    prefix = "p1_core."
    source_keys = filter_by_prefix(ckpt_state, (prefix,))
    if not source_keys:
        raise RuntimeError(f"{source} has no p1_core tensors; expected P1 homo checkpoint")
    dst = model.p1_core.state_dict()
    wrong = []
    for key, value in source_keys.items():
        suffix = key[len(prefix):]
        if suffix not in dst or dst[suffix].shape != value.shape or not torch.equal(
            dst[suffix].cpu(), value.cpu()
        ):
            wrong.append(key)
    if wrong:
        raise RuntimeError(
            f"P1 frontend differs for {agent_type} in {source}: {wrong[:8]}. "
            "Use the same net_final_branch checkpoint that initialized Stage-0."
        )
    print(f"[ckpt] verified {len(source_keys)} frozen P1 tensors for {agent_type}")


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
                min_match_ratio=1.0,
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
    encoder_prefix = f"{encoder_attr}.0.bevencode."
    encoder = getattr(branch, "encoder", None)
    if encoder is None:
        raise AttributeError("local branch has no `.encoder`")
    if not hasattr(encoder, "bevencode"):
        raise RuntimeError("NegoCollab local branch must use P1SplatEncoder")
    mapped_encoder = remap_prefix(
        filter_by_prefix(ckpt_state, (encoder_prefix,)), encoder_prefix, ""
    )
    reports.append(
        copy_matching_tensors(
            encoder.bevencode,
            mapped_encoder,
            source=f"{source}:{encoder_prefix}*",
            destination=f"local_branches.{agent_type}.encoder.bevencode",
            min_match_ratio=1.0,
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
                min_match_ratio=1.0,
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
    for agent_type, state, path in (
        ("vehicle", vehicle_state, vehicle_path),
        ("rsu", rsu_state, rsu_path),
        ("drone", drone_state, drone_path),
    ):
        verify_frozen_p1(model, state, agent_type, source=path)

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


# ---------------------------------------------------------------------------
# Homo Stage-0 initialization from the Gaussian P1 branch checkpoint
# ---------------------------------------------------------------------------

# Source prefix of the official CamEncode image branch inside
# `net_final_branch.pth` (ImageFrontend.encoders is a ModuleDict keyed by
# agent type; each entry is a full CamEncode: trunk/up1/up2/depth_head/image_head).
STAGE0_LSS_SOURCE_PREFIX = "frontend.encoders.{agent}."


def _structural_prefix_match(
    module: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> Optional[str]:
    """Find a source prefix that exactly covers ``module`` (names + shapes).

    Scans every 1-3 level prefix of ``state``. A prefix qualifies only when
    remapping it onto ``module`` matches 100% of ``module.state_dict()``
    keys with zero shape mismatches. Never fuzzy-matches partial coverage.

    Args:
        module: Destination module (e.g. BEV backbone, bevencode).
        state: Flat checkpoint state_dict.

    Returns:
        The qualifying source prefix, or None when no prefix qualifies.
    """
    dst = module.state_dict()
    if not dst:
        return None
    candidates: set = set()
    for key in state:
        parts = key.split(".")
        for depth in (1, 2, 3):
            if len(parts) > depth:
                candidates.add(".".join(parts[:depth]) + ".")
    for prefix in sorted(candidates):
        mapped = remap_prefix(filter_by_prefix(state, (prefix,)), prefix, "")
        matched = sum(
            1
            for key, tensor in mapped.items()
            if key in dst and tuple(dst[key].shape) == tuple(tensor.shape)
        )
        if matched == len(dst):
            return prefix
    return None


def load_homo_stage0_initialization(
    model: nn.Module,
    checkpoint_path: str,
    agent_type: str,
    map_location: str = "cpu",
) -> CheckpointLoadReport:
    """Initialize one homo Stage-0 LSS from the pretrained branch checkpoint.

    Loads ``frontend.encoders.<agent_type>.*`` (official CamEncode:
    trunk / up1 / up2 / depth_head / image_head) onto the matching
    ``<agent>_models[0].camencode`` with exact 1:1 coverage — any missing /
    unexpected / shape-mismatched tensor fails loudly, partial loads are
    refused. BEV backbone and LSS bevencode availability is probed
    structurally (prefix-agnostic, 100%-coverage required) and reported.

    Args:
        model: ``Airv2xHomoBase`` (before DDP wrap, already on device).
        checkpoint_path: e.g. ``.../airv2x_gaussian_final/net_final_branch.pth``.
        agent_type: ``vehicle`` / ``rsu`` / ``drone`` — selects the source
            branch; no cross-agent loading ever happens.
        map_location: torch.load map_location.

    Returns:
        CheckpointLoadReport of the CamEncode copy.

    Raises:
        RuntimeError / ValueError / FileNotFoundError: on any non-exact or
            ambiguous mapping. Never silently skips.
    """
    if agent_type not in AGENT_ENCODER_ATTR:
        raise ValueError(f"Unknown agent_type: {agent_type}")

    state = load_raw_checkpoint(checkpoint_path, map_location)
    src_prefix = STAGE0_LSS_SOURCE_PREFIX.format(agent=agent_type)
    agent_state = filter_by_prefix(state, (src_prefix,))

    print("[stage0-init]")
    print(f"checkpoint:\n  {os.path.abspath(checkpoint_path)}")
    print(f"agent:\n  {agent_type}")

    if not agent_state:
        top_groups = sorted({key.split(".")[0] for key in state})
        raise RuntimeError(
            f"[stage0-init] source checkpoint has no keys under '{src_prefix}'. "
            f"Top-level groups present: {top_groups}. "
            "Refusing to guess a different source prefix."
        )

    depth_head_keys = sorted(k for k in agent_state if ".depth_head." in k)
    attr = AGENT_ENCODER_ATTR[agent_type]
    print(f"LSS source:\n  {src_prefix}")
    print(f"LSS destination:\n  {attr}[0].camencode")
    if not depth_head_keys:
        raise RuntimeError(
            f"[stage0-init] '{src_prefix}' contains no predicted-depth head "
            "(camencode.depth_head.*). This checkpoint cannot initialize an "
            "RGB predicted-depth LSS for Stage-0 (use_depth_gt=false)."
        )
    print(f"depth_head:\n  FOUND ({len(depth_head_keys)} tensors)")

    module_list = getattr(model, attr, None)
    if not isinstance(module_list, nn.ModuleList) or len(module_list) != 1:
        raise RuntimeError(
            f"[stage0-init] destination model must expose exactly one "
            f"{attr} entry (homo Stage-0 trains a single agent type)."
        )
    lss = module_list[0]
    camencode = getattr(lss, "camencode", None)
    if not isinstance(camencode, nn.Module):
        raise RuntimeError(
            f"[stage0-init] {attr}[0] has no .camencode — unexpected model "
            "structure for LiftSplatShootEncoder."
        )

    mapped = remap_prefix(agent_state, src_prefix, "")
    report = copy_matching_tensors(
        camencode,
        mapped,
        source=f"{checkpoint_path}:{src_prefix}*",
        destination=f"{attr}[0].camencode",
        min_match_ratio=1.0,
    )
    if report.missing or report.unexpected or report.shape_mismatch:
        raise RuntimeError(
            "[stage0-init] LSS load was NOT exact: "
            f"missing={len(report.missing)} unexpected={len(report.unexpected)} "
            f"shape_mismatch={len(report.shape_mismatch)}. "
            "Partial / fuzzy loads are not allowed."
        )
    print(f"matched:\n  {report.matched} / {len(camencode.state_dict())}")
    print("missing:\n  0")
    print("unexpected:\n  0")
    print("shape mismatch:\n  0")

    # -- BEV backbone probe: structural, never by name guessing. ----------
    backbone = getattr(model, "backbone", None)
    if isinstance(backbone, nn.Module):
        prefix = _structural_prefix_match(backbone, state)
        if prefix is None:
            print(
                "BEV backbone:\n  NOT AVAILABLE — training Stage-0 backbone "
                "from random initialization"
            )
        else:
            mapped_bb = remap_prefix(filter_by_prefix(state, (prefix,)), prefix, "")
            bb_report = copy_matching_tensors(
                backbone,
                mapped_bb,
                source=f"{checkpoint_path}:{prefix}*",
                destination="backbone",
                min_match_ratio=1.0,
            )
            print(f"BEV backbone source:\n  {prefix}")
            print(f"matched:\n  {bb_report.matched} / {len(backbone.state_dict())}")
            print("shape mismatch:\n  0")

    # -- LSS bevencode probe (ResNet18 BEV encoder inside LSS). ------------
    bevencode = getattr(lss, "bevencode", None)
    if isinstance(bevencode, nn.Module):
        prefix = _structural_prefix_match(bevencode, state)
        if prefix is None:
            print(
                "LSS bevencode:\n  NOT AVAILABLE in source checkpoint — stays "
                "at random init (frozen together with LSS per freeze_lss "
                "protocol)"
            )
        else:
            mapped_bv = remap_prefix(filter_by_prefix(state, (prefix,)), prefix, "")
            copy_matching_tensors(
                bevencode,
                mapped_bv,
                source=f"{checkpoint_path}:{prefix}*",
                destination=f"{attr}[0].bevencode",
                min_match_ratio=1.0,
            )
            print(f"LSS bevencode source:\n  {prefix}")

    return report

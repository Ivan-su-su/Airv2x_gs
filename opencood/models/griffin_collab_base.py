# -*- coding: utf-8 -*-
"""Griffin HEAL / STAMP Stage-1 collab base.

Two frozen Griffin-P1 frontends (vehicle + drone) feed a shared
``Airv2xSharedDetector`` in the vehicle domain, exactly like the AirV2X
HEAL / STAMP protocol but with the Griffin Stage-0 checkpoints:

* ``veh_models.0`` (P1SplatEncoder.bevencode) ← vehicle Stage-0 best.pth
* ``drone_models.0`` (P1SplatEncoder.bevencode) ← drone Stage-0 best.pth
* shared backbone / pyramid / shrink / heads ← vehicle Stage-0 best.pth
* ``p1_core`` ← one frozen FrozenP1 shared by both agents (the P1 weights
  are identical in both Stage-0 checkpoints, so only one copy is kept)

``IntermediateFusionDatasetGriffin`` supplies record_len=[2] cooperative
batches in the unified vehicle detection range.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn

from opencood.models.airv2x_detector_parts import Airv2xSharedDetector
from opencood.models.common_modules.airv2x_base_model import Airv2xBase
from opencood.models.lss_pretrain_modules.p1_mixin import P1CamMixin
from opencood.utils.airv2x_base_checkpoint import SHARED_DETECTOR_PREFIXES
from opencood.utils.airv2x_freeze import print_trainable_parameter_groups
from opencood.utils.transformation_utils import normalize_pairwise_tfm


_SHARED_HEADS = {prefix.rstrip(".") for prefix in SHARED_DETECTOR_PREFIXES}
_SINGLE_HEAD_PREFIX = "detector.pyramid_backbone.single_head_"


class GriffinCollabBase(P1CamMixin, Airv2xBase):
    """Common Stage-1 wiring for GriffinHEALCollab / GriffinSTAMPCollab."""

    def __init__(self, args: Dict[str, Any]) -> None:
        stage_args = dict(args)
        if not stage_args.get("p1_checkpoint"):
            raise ValueError("GriffinCollabBase requires model.args.p1_checkpoint")
        if stage_args.get("freeze_lss") is False:
            # Stage-1 protocol: LSS (P1 + BevEncode) stays frozen.
            print("[griffin-collab] freeze_lss=false is ignored; Stage-1 freezes LSS")
        stage_args["freeze_lss"] = True
        if sorted(stage_args.get("collaborators", [])) != ["drone", "vehicle"]:
            raise ValueError(
                "Griffin collab requires collaborators ['vehicle', 'drone'], got "
                f"{stage_args.get('collaborators')}"
            )
        stage_args["collaborators"] = ["vehicle", "drone"]

        Airv2xBase.__init__(self, stage_args)
        self.args = stage_args
        P1CamMixin.init_encoders(self, stage_args)
        self.detector = Airv2xSharedDetector(stage_args)
        self.shrink_flag = self.detector.shrink_flag
        _alias_module(self, "backbone", self.detector.backbone)
        _alias_module(self, "pyramid_backbone", self.detector.pyramid_backbone)
        _alias_module(self, "shrink_conv", self.detector.shrink_conv)
        _alias_module(self, "cls_head", self.detector.cls_head)
        _alias_module(self, "reg_head", self.detector.reg_head)
        _alias_module(self, "obj_head", self.detector.obj_head)
        _alias_module(self, "seg_head", self.detector.seg_head)

        self._load_stage0_weights(stage_args)
        self._freeze_stage1(stage_args)
        self.p1_core.eval()
        leaked = [
            name
            for name, param in self.named_parameters()
            if ("veh_models" in name or "drone_models" in name)
            and param.requires_grad
        ]
        if leaked:
            raise RuntimeError(
                "Griffin Stage-1 LSS (P1 + BevEncode) must stay frozen, got "
                f"trainable: {leaked[:8]}"
            )
        print_trainable_parameter_groups(self, type(self).__name__)

    # ------------------------------------------------------------------
    # Stage-0 initialization
    # ------------------------------------------------------------------
    def _load_stage0_weights(self, args: Dict[str, Any]) -> None:
        vehicle_path = args.get("stage0_vehicle_ckpt")
        drone_path = args.get("stage0_drone_ckpt")
        if not vehicle_path or not drone_path:
            raise ValueError(
                "GriffinCollabBase requires model.args.stage0_vehicle_ckpt and "
                "stage0_drone_ckpt pointing at the two Stage-0 best.pth files"
            )
        vehicle_state = _load_flat_state(vehicle_path)
        drone_state = _load_flat_state(drone_path)

        # Shared detector body always comes from the vehicle base (HEAL
        # semantics; user decision).
        _copy_prefixed(self.detector, vehicle_state, SHARED_DETECTOR_PREFIXES,
                       source=vehicle_path, strip_prefix="")

        # BevEncode of each agent comes from its own Stage-0 base. The
        # splat encoder key layout is ``veh_models.0.*`` / ``drone_models.0.*``.
        _copy_prefixed(self.veh_models, vehicle_state, ("veh_models.",),
                       source=vehicle_path, strip_prefix="veh_models.")
        _copy_prefixed(self.drone_models, drone_state, ("drone_models.",),
                       source=drone_path, strip_prefix="drone_models.")

        # p1_core was already populated by FrozenP1 from p1_checkpoint, which
        # is the same frontend both Stage-0 runs used. Cross-check that the
        # vehicle Stage-0 checkpoint agrees so silent divergence is caught.
        self._assert_p1_matches(vehicle_state, "vehicle")
        self._assert_p1_matches(drone_state, "drone")

    def _assert_p1_matches(self, state: Mapping[str, torch.Tensor], tag: str) -> None:
        own = self.p1_core.state_dict()
        checked = 0
        for key, tensor in state.items():
            if not key.startswith("p1_core."):
                continue
            sub = key[len("p1_core."):]
            if sub not in own:
                raise KeyError(f"[griffin-collab] p1_core key missing: {key}")
            if tuple(own[sub].shape) != tuple(tensor.shape):
                raise RuntimeError(
                    f"[griffin-collab] p1_core shape mismatch on {key}: "
                    f"model {tuple(own[sub].shape)} vs {tag} ckpt {tuple(tensor.shape)}"
                )
            if not torch.equal(own[sub].cpu(), tensor.cpu()):
                max_diff = (own[sub].cpu().float() - tensor.cpu().float()).abs().max()
                raise RuntimeError(
                    f"[griffin-collab] p1_core diverged from the {tag} Stage-0 "
                    f"checkpoint on {key} (max_abs={max_diff}). Re-train with a "
                    "matching p1_checkpoint or update the Stage-0 bases."
                )
            checked += 1
        if checked < 300:
            raise RuntimeError(
                f"[griffin-collab] only {checked} p1_core tensors found in the "
                f"{tag} Stage-0 checkpoint; expected the full frozen frontend"
            )
        print(f"[griffin-collab] p1_core verified against {tag} Stage-0 ({checked} tensors)")

    # ------------------------------------------------------------------
    # Stage-1 freeze policy
    # ------------------------------------------------------------------
    def _freeze_stage1(self, args: Dict[str, Any]) -> None:
        """Freeze LSS (P1 + both BevEncode) and apply the per-variant policy."""
        for module in (self.veh_models, self.drone_models):
            for param in module.parameters():
                param.requires_grad = False
        for param in self.p1_core.parameters():
            param.requires_grad = False

        freeze_bev = bool(args.get("freeze_bev_backbone", False))
        freeze_fusion = bool(args.get("freeze_fusion", False))
        freeze_heads = bool(args.get("freeze_heads", False))
        freeze_map = [
            (self.detector.backbone, freeze_bev),
            (self.detector.pyramid_backbone, freeze_fusion),
            (self.detector.cls_head, freeze_heads),
            (self.detector.reg_head, freeze_heads),
            (self.detector.shrink_conv, freeze_fusion),
            (self.detector.obj_head, freeze_heads),
            (self.detector.seg_head, freeze_heads),
        ]
        for module, should_freeze in freeze_map:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = not should_freeze

        # STAMP decision: when the whole fusion is frozen, the six
        # single_head_* tensors would stay at their Stage-0 random values
        # (zero-gradient under N=1). Release them so the N=2 softmax can
        # actually calibrate once training starts.
        if freeze_fusion and args.get("train_single_head", True):
            released = 0
            for name, param in self.named_parameters():
                if name.startswith(_SINGLE_HEAD_PREFIX):
                    param.requires_grad = True
                    released += 1
            if released:
                print(
                    "[griffin-collab] released pyramid single_head_* for training "
                    f"({released} tensors) despite freeze_fusion"
                )

    # ------------------------------------------------------------------
    # nn.Module plumbing
    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> "GriffinCollabBase":
        super().train(mode)
        # Frozen P1 and frozen Stage-0 LSS must stay in eval mode so BN
        # statistics never update.
        self.p1_core.eval()
        for module in (self.veh_models, self.drone_models):
            for encoder in module:
                encoder.bevencode.eval()
        return self

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Expose shared detector keys without the ``detector.`` prefix."""
        raw = super().state_dict(*args, **kwargs)
        return {_public_state_key(key): value for key, value in raw.items()}

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
    ) -> nn.modules.module._IncompatibleKeys:
        internal = {_internal_state_key(key): value for key, value in state_dict.items()}
        return super().load_state_dict(internal, strict=strict)

    def collab_affine(self, spatial_features_2d: torch.Tensor, pairwise: torch.Tensor) -> torch.Tensor:
        """Metric 4x4 pairwise → normalized 2x3 affine for PyramidFusion.

        Griffin stores rigid transforms in meters. ``affine_grid`` expects
        translation in the feature's normalized frame, so a raw xy slice
        sends the other agent off the canvas and the agent-softmax collapses
        to a one-hot. Normalize with the physical size of this BEV map.
        """
        _, _, height, width = spatial_features_2d.shape
        cav_range = self.args["cav_range"]
        x_extent = float(cav_range[3] - cav_range[0])
        y_extent = float(cav_range[4] - cav_range[1])
        if abs(x_extent / width - y_extent / height) > 1e-3:
            raise ValueError(
                "Griffin collab BEV must be square in meters-per-pixel, got "
                f"x {x_extent}/{width} vs y {y_extent}/{height}"
            )
        discrete_ratio = x_extent / float(width)
        return normalize_pairwise_tfm(
            pairwise, height, width, discrete_ratio, downsample_rate=1
        )

    def iter_frozen_bevencode(self) -> List[nn.Module]:
        """Return both frozen ``BevEncode`` modules (BN sanity checks)."""
        out: List[nn.Module] = []
        for bucket in (self.veh_models, self.drone_models):
            for encoder in bucket:
                bev = getattr(encoder, "bevencode", None)
                if bev is not None:
                    out.append(bev)
        return out


def _alias_module(model: nn.Module, name: str, module: Optional[nn.Module]) -> None:
    """Expose ``module`` as ``model.name`` without double registration."""
    if module is None:
        object.__setattr__(model, name, None)
        return
    object.__setattr__(model, name, module)


def _public_state_key(key: str) -> str:
    if key.startswith("detector."):
        rest = key[len("detector."):]
        if rest.split(".", 1)[0] in _SHARED_HEADS:
            return rest
    return key


def _internal_state_key(key: str) -> str:
    head = key.split(".", 1)[0]
    if head in _SHARED_HEADS and not key.startswith("detector."):
        return "detector." + key
    return key


def _load_flat_state(path: str) -> Dict[str, torch.Tensor]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    blob = torch.load(path, map_location="cpu")
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        if isinstance(blob, dict) and key in blob:
            return blob[key]
    return blob


def _copy_prefixed(
    module: nn.Module,
    state: Mapping[str, torch.Tensor],
    prefixes: Tuple[str, ...],
    *,
    source: str,
    strip_prefix: str = "detector.",
) -> None:
    """Copy ``prefix*`` tensors (optionally stripping a prefix) into ``module``.

    Follows the strict audit semantics of ``copy_matching_tensors``: every
    destination key must be covered, shapes must match exactly.
    """
    dst = module.state_dict()
    mapped: Dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                mapped[key[len(strip_prefix):] if strip_prefix else key] = tensor
                break
    missing = [k for k in dst if k not in mapped]
    unexpected = [k for k in mapped if k not in dst]
    if missing or unexpected:
        raise RuntimeError(
            f"[griffin-collab] Stage-0 load failed from {source}: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    for key, tensor in mapped.items():
        if tuple(dst[key].shape) != tuple(tensor.shape):
            raise RuntimeError(
                f"[griffin-collab] shape mismatch {key}: ckpt "
                f"{tuple(tensor.shape)} vs model {tuple(dst[key].shape)}"
            )
    module.load_state_dict(mapped, strict=True)
    print(f"[griffin-collab] loaded {len(mapped)} tensors from {source} "
          f"(prefixes={prefixes})")

# -*- coding: utf-8 -*-
"""Griffin homogeneous Stage-0 detector.

BEV backbone, PyramidFusion and detection heads come from
``Airv2xSharedDetector``. The image frontend is the frozen Griffin P1 used by
Griffin V2X-ViT / Where2comm. ``Airv2xHomoBase.__init__`` is not used: it
always asserts that every ``veh_models`` / ``drone_models`` parameter is
frozen, which would also freeze the trainable post-LSS ``BevEncode``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn

from opencood.models.airv2x_detector_parts import Airv2xSharedDetector
from opencood.models.airv2x_homo_base import Airv2xHomoBase
from opencood.models.common_modules.airv2x_base_model import Airv2xBase
from opencood.models.lss_pretrain_modules.p1_mixin import P1CamMixin
from opencood.utils.airv2x_base_checkpoint import SHARED_DETECTOR_PREFIXES
from opencood.utils.airv2x_freeze import (
    apply_named_freeze_policy,
    print_trainable_parameter_groups,
)


_SHARED_HEADS = {prefix.rstrip(".") for prefix in SHARED_DETECTOR_PREFIXES}


class GriffinHomoBase(P1CamMixin, Airv2xHomoBase):
    """Single-agent Griffin detector with a frozen Griffin P1 frontend.

    Saved checkpoints expose ``backbone.*`` / ``pyramid_backbone.*`` /
    ``cls_head.*`` rather than ``detector.*``, matching
    ``load_shared_detector``. ``BevEncode`` stays under ``veh_models`` or
    ``drone_models`` and is trainable. ``p1_core.*`` stays frozen.
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        stage_args = dict(args)
        if not stage_args.get("p1_checkpoint"):
            raise ValueError("GriffinHomoBase requires model.args.p1_checkpoint")
        # AirV2X freeze_lss would freeze BevEncode because it lives on the
        # splat encoder. Griffin P1 freezes itself inside FrozenP1.
        stage_args["freeze_lss"] = False
        Airv2xBase.__init__(self, stage_args)
        self.args = stage_args
        self.homo_agent_type = stage_args.get("homo_agent_type")
        if self.homo_agent_type is None:
            if len(self.collaborators) != 1:
                raise ValueError(
                    "GriffinHomoBase requires exactly one collaborator or an "
                    "explicit homo_agent_type"
                )
            self.homo_agent_type = self.collaborators[0]
        if self.homo_agent_type not in ("vehicle", "drone"):
            raise ValueError(
                "Griffin Stage-0 homo_agent_type must be vehicle or drone, "
                f"got {self.homo_agent_type}"
            )
        if self.collaborators != [self.homo_agent_type]:
            raise ValueError(
                "Griffin Stage-0 collaborators must be "
                f"['{self.homo_agent_type}'], got {self.collaborators}"
            )

        P1CamMixin.init_encoders(self, stage_args)
        self.detector = Airv2xSharedDetector(stage_args)
        # Keep attribute access for the freeze policy, but do not register
        # these modules twice. A second registration makes strict
        # load_state_dict demand both ``detector.backbone.*`` and
        # ``backbone.*`` for the same tensors.
        self.shrink_flag = self.detector.shrink_flag
        _alias_module(self, "backbone", self.detector.backbone)
        _alias_module(self, "pyramid_backbone", self.detector.pyramid_backbone)
        _alias_module(self, "shrink_conv", self.detector.shrink_conv)
        _alias_module(self, "cls_head", self.detector.cls_head)
        _alias_module(self, "reg_head", self.detector.reg_head)
        _alias_module(self, "obj_head", self.detector.obj_head)
        _alias_module(self, "seg_head", self.detector.seg_head)

        freeze_modules = {
            "backbone": bool(stage_args.get("freeze_bev_backbone", False)),
            "pyramid_backbone": bool(stage_args.get("freeze_fusion", False)),
            "cls_head": bool(stage_args.get("freeze_heads", False)),
            "reg_head": bool(stage_args.get("freeze_heads", False)),
        }
        if self.shrink_conv is not None:
            freeze_modules["shrink_conv"] = bool(stage_args.get("freeze_fusion", False))
        if self.obj_head is not None:
            freeze_modules["obj_head"] = bool(stage_args.get("freeze_heads", False))
        if self.seg_head is not None:
            freeze_modules["seg_head"] = bool(stage_args.get("freeze_heads", False))
        apply_named_freeze_policy(
            self,
            freeze_lss=False,
            freeze_modules=freeze_modules,
            assembled_inference=bool(stage_args.get("assembled_inference", False)),
        )
        self.p1_core.eval()
        self._assert_stage0_ownership()
        print_trainable_parameter_groups(self, "GriffinHomoBase")

    def train(self, mode: bool = True) -> "GriffinHomoBase":
        super().train(mode)
        self.p1_core.eval()
        return self

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Expose shared detector weights without the ``detector.`` prefix."""
        raw = super().state_dict(*args, **kwargs)
        renamed: Dict[str, torch.Tensor] = {}
        for key, value in raw.items():
            renamed[_public_state_key(key)] = value
        return renamed

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
    ) -> nn.modules.module._IncompatibleKeys:
        """Accept either public ``backbone.*`` keys or raw ``detector.*`` keys."""
        internal: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            internal[_internal_state_key(key)] = value
        return super().load_state_dict(internal, strict=strict)

    def _assert_stage0_ownership(self) -> None:
        """Fail if P1 can train or if BevEncode / detector body cannot."""
        leaked = [
            name
            for name, param in self.p1_core.named_parameters()
            if param.requires_grad
        ]
        if leaked:
            raise RuntimeError(
                "Griffin P1 parameters must be frozen, got trainable: "
                f"{leaked[:8]}"
            )
        if self.p1_core.training:
            raise RuntimeError("Griffin P1 must stay in eval mode")

        bev_trainable = [
            name
            for name, param in self.named_parameters()
            if "bevencode" in name and param.requires_grad
        ]
        if not bev_trainable:
            raise RuntimeError(
                "Griffin Stage-0 BevEncode has no trainable parameters"
            )

        public_names = {_public_state_key(name) for name, _param in self.named_parameters()}
        required = ("backbone.", "pyramid_backbone.", "cls_head.", "reg_head.")
        missing = [
            prefix
            for prefix in required
            if not any(
                name.startswith(prefix) and _param_requires_grad(self, name)
                for name in public_names
            )
        ]
        if missing:
            raise RuntimeError(
                "Griffin Stage-0 detector body is missing trainable modules: "
                f"{missing}"
            )

    def iter_bevencode(self) -> List[nn.Module]:
        """Return splat ``BevEncode`` modules for the active agent."""
        modules: List[nn.Module] = []
        for bucket_name in ("veh_models", "drone_models", "rsu_models"):
            bucket = getattr(self, bucket_name, None)
            if bucket is None:
                continue
            for encoder in bucket:
                bevencode = getattr(encoder, "bevencode", None)
                if bevencode is not None:
                    modules.append(bevencode)
        return modules


def _alias_module(model: nn.Module, name: str, module: nn.Module) -> None:
    """Expose ``module`` as ``model.name`` without a second ``_modules`` entry.

    ``nn.Module.__setattr__`` would register the same child under both
    ``detector.<name>`` and ``<name>``.
    """
    if module is None:
        object.__setattr__(model, name, None)
        return
    object.__setattr__(model, name, module)


def _public_state_key(key: str) -> str:
    """Map ``detector.backbone.*`` to the HEAL loader prefix ``backbone.*``."""
    if key.startswith("detector."):
        rest = key[len("detector.") :]
        head = rest.split(".", 1)[0]
        if head in _SHARED_HEADS:
            return rest
    return key


def _internal_state_key(key: str) -> str:
    """Map a public shared-detector key back onto ``detector.*``."""
    head = key.split(".", 1)[0]
    if head in _SHARED_HEADS and not key.startswith("detector."):
        return "detector." + key
    return key


def _param_requires_grad(model: nn.Module, public_name: str) -> bool:
    internal = _internal_state_key(public_name)
    for name, param in model.named_parameters():
        if name == internal or _public_state_key(name) == public_name:
            return bool(param.requires_grad)
    return False

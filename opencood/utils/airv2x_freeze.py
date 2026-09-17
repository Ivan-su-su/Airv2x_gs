# -*- coding: utf-8 -*-
"""Explicit freeze / trainability helpers for AirV2X heterogeneous baselines."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn


AGENT_ENCODER_ATTR: Dict[str, str] = {
    "vehicle": "veh_models",
    "rsu": "rsu_models",
    "drone": "drone_models",
}

LSS_NAME_TOKENS: Tuple[str, ...] = (
    "veh_models",
    "rsu_models",
    "drone_models",
    "encoder",
    "p1_core",
)


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def iter_named_modules(
    model: nn.Module, names: Sequence[str]
) -> Iterable[Tuple[str, nn.Module]]:
    """Yield `(name, module)` for attributes that exist on `model`."""
    for name in names:
        module = getattr(model, name, None)
        if module is None:
            continue
        yield name, module


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    """Set `requires_grad` for all parameters of `module`."""
    for param in module.parameters():
        param.requires_grad = bool(trainable)
    if trainable:
        module.train()
    else:
        module.eval()


def iter_lss_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    """Yield LSS / agent-encoder modules attached to an AirV2X model.

    When a frozen P1 stack is present, only ``p1_core`` is treated as LSS.
    ``P1SplatEncoder.bevencode`` stays trainable.
    """
    core = _unwrap(model)
    p1_core = getattr(core, "p1_core", None)
    if p1_core is not None:
        yield "p1_core", p1_core
        return
    for agent_type, attr in AGENT_ENCODER_ATTR.items():
        module = getattr(core, attr, None)
        if module is not None:
            yield f"{attr}", module
    local_branches = getattr(core, "local_branches", None)
    if isinstance(local_branches, nn.ModuleDict):
        for agent_type, branch in local_branches.items():
            encoder = getattr(branch, "encoder", None)
            if encoder is not None:
                yield f"local_branches.{agent_type}.encoder", encoder


def iter_lss_named_parameters(model: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    """Yield ``(name, param)`` for every frozen-LSS parameter on ``model``."""
    lss_ids = {
        id(param)
        for _, module in iter_lss_modules(model)
        for param in module.parameters()
    }
    for name, param in model.named_parameters():
        if id(param) in lss_ids:
            yield name, param


def freeze_lss_encoders(model: nn.Module) -> List[str]:
    """Freeze every LSS encoder. Returns frozen module names."""
    frozen: List[str] = []
    for name, module in iter_lss_modules(model):
        set_module_trainable(module, False)
        frozen.append(name)
    return frozen


def is_lss_param_name(name: str) -> bool:
    """Return True if a parameter name belongs to an LSS encoder."""
    clean = name[7:] if name.startswith("module.") else name
    if clean.startswith("p1_core") or ".p1_core." in f".{clean}":
        return True
    for token in LSS_NAME_TOKENS:
        if token == "p1_core":
            continue
        if clean.startswith(token) or f".{token}." in f".{clean}":
            if token == "encoder" and "local_prompt" in clean:
                return False
            if token == "encoder":
                return ".encoder." in f".{clean}" or clean.startswith("encoder")
            return True
    return False


def assert_lss_frozen(model: nn.Module) -> None:
    """Fail if any LSS parameter is trainable."""
    trainable: List[str] = []
    for name, param in iter_lss_named_parameters(model):
        if param.requires_grad:
            trainable.append(name)
    if trainable:
        preview = "\n  ".join(trainable[:20])
        raise AssertionError(
            "LSS encoders must stay frozen, but trainable LSS params found:\n"
            f"  {preview}"
        )


def assert_lss_not_in_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Fail if the optimizer contains any LSS parameter."""
    lss_ids = {id(param) for _, param in iter_lss_named_parameters(model)}
    leaked: List[str] = []
    opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    for name, param in model.named_parameters():
        if id(param) in lss_ids and id(param) in opt_ids:
            leaked.append(name)
    if leaked:
        preview = "\n  ".join(leaked[:20])
        raise AssertionError(
            "Frozen LSS parameters must not be in the optimizer:\n"
            f"  {preview}"
        )


def assert_no_lss_grad(model: nn.Module) -> None:
    """Fail if any LSS parameter has a non-zero gradient."""
    bad: List[str] = []
    for name, param in iter_lss_named_parameters(model):
        if param.grad is None:
            continue
        if torch.isfinite(param.grad).any() and param.grad.abs().sum().item() != 0:
            bad.append(name)
    if bad:
        preview = "\n  ".join(bad[:20])
        raise AssertionError(
            "LSS gradients must stay empty / zero after backward:\n"
            f"  {preview}"
        )


def lss_parameter_checksum(model: nn.Module) -> float:
    """Scalar checksum of all LSS weights (for post-step identity checks)."""
    total = 0.0
    for _, param in iter_lss_named_parameters(model):
        total += float(param.detach().float().abs().sum().cpu())
    return total


def grouped_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable parameters grouped by top-level module name."""
    groups: Dict[str, int] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean = name[7:] if name.startswith("module.") else name
        group = clean.split(".", 1)[0]
        groups[group] = groups.get(group, 0) + param.numel()
    return groups


def print_trainable_parameter_groups(model: nn.Module, title: str = "") -> Dict[str, int]:
    """Print trainable parameter counts grouped by top-level module."""
    groups = grouped_trainable_parameters(model)
    header = title or type(model).__name__
    print(f"[trainability] {header}")
    if not groups:
        print("  (no trainable parameters)")
        return groups
    total = 0
    for name in sorted(groups):
        count = groups[name]
        total += count
        print(f"  {name}: {count:,}")
    print(f"  TOTAL trainable: {total:,}")
    return groups


def apply_named_freeze_policy(
    model: nn.Module,
    *,
    freeze_lss: bool = True,
    freeze_modules: Optional[Mapping[str, bool]] = None,
    allow_zero_trainable: bool = False,
    assembled_inference: bool = False,
) -> Dict[str, int]:
    """Apply an explicit freeze policy and print the trainable groups.

    Args:
        model: Target model.
        freeze_lss: Always freeze LSS when True.
        freeze_modules: Mapping of attribute name → freeze flag.
        allow_zero_trainable: If True, zero trainable params is not an error.
        assembled_inference: Explicit inference-only mode (zero trainable OK).

    Returns:
        Trainable parameter counts grouped by top-level module.

    Raises:
        RuntimeError: If LSS is frozen, nothing is trainable, and inference
            mode was not declared.
    """
    if freeze_lss:
        freeze_lss_encoders(model)
        assert_lss_frozen(model)

    if freeze_modules:
        for name, should_freeze in freeze_modules.items():
            module = getattr(model, name, None)
            if module is None:
                continue
            set_module_trainable(module, trainable=not should_freeze)

    groups = print_trainable_parameter_groups(model)
    n_trainable = sum(groups.values())
    if n_trainable == 0 and not (allow_zero_trainable or assembled_inference):
        raise RuntimeError(
            "This training configuration has zero trainable parameters. "
            "LSS is frozen by protocol. Set freeze_bev_backbone / freeze_fusion "
            "/ freeze_heads explicitly, or set assembled_inference: true if you "
            "intentionally want a frozen assembled model."
        )
    return groups

"""Frozen Griffin P1 image stack for cooperative camera baselines.

Vehicle:
    frozen CamEncode -> R2/F45 -> r2_downsample -> HighResFusion -> F90
    -> trained categorical DepthHead.

Drone:
    frozen CamEncode -> R2/F45 -> HighResFusion -> F90.
    No learned drone depth is loaded for Griffin. Downstream LSS receives
    dataset-provided analytic bottom-camera ideal_depth_z.

Heatmap heads are intentionally not loaded here: they are Gaussian-only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from opencood.models.gaussian_modules_0822.highres_adapter import HighResFusion
from opencood.models.gaussian_modules_0822.image_frontend import (
    _camencode_r2_and_f45,
    present_camera_agents,
)
from opencood.models.gaussian_modules_0822.lss.head import DepthHead
from opencood.models.sub_modules.lss_submodule import CamEncode

AGENT_TYPES = ("vehicle", "rsu", "drone")
CAMENCODE_C = 64
CAMENCODE_DOWNSAMPLE = 8


def _load_exact(module: nn.Module, state: Dict[str, torch.Tensor], prefix: str) -> int:
    src = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    dst = module.state_dict()
    missing = sorted(set(dst) - set(src))
    unexpected = sorted(set(src) - set(dst))
    if missing or unexpected:
        raise RuntimeError(
            f"[frozen-p1] non-exact mapping for {prefix!r}: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    for key, tensor in src.items():
        if tuple(dst[key].shape) != tuple(tensor.shape):
            raise RuntimeError(
                f"[frozen-p1] shape mismatch {prefix}{key}: "
                f"ckpt={tuple(tensor.shape)} model={tuple(dst[key].shape)}"
            )
    module.load_state_dict(src, strict=True)
    return len(src)


class FrozenP1(nn.Module):
    """Frozen Griffin P1 producing F90 and vehicle categorical depth."""

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__()
        ckpt_path = str(args["p1_checkpoint"])
        raw = torch.load(ckpt_path, map_location="cpu")
        state = (
            raw["model_state_dict"]
            if isinstance(raw, dict) and "model_state_dict" in raw
            else raw
        )
        if not isinstance(state, dict) or not state:
            raise RuntimeError(f"[frozen-p1] invalid checkpoint: {ckpt_path}")
        state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
            if torch.is_tensor(v)
        }

        self.encoders = nn.ModuleDict()
        self.highres = nn.ModuleDict()
        self.depth_heads = nn.ModuleDict()
        self.r2_downsample: Optional[nn.Conv2d] = None

        needed = [
            a for a in AGENT_TYPES
            if a in set(args.get("collaborators") or AGENT_TYPES)
        ]
        if not needed:
            raise ValueError("[frozen-p1] no supported collaborators")

        n_total = 0
        for agent in needed:
            if agent == "rsu":
                raise ValueError(
                    "Griffin has no RSU branch; collaborators must be vehicle/drone."
                )
            cam_cfg = args[agent]["cam"]
            ddiscr = list(cam_cfg["grid_conf"]["ddiscr"])
            mode = str(cam_cfg["grid_conf"]["mode"])
            enc = CamEncode(
                D=int(ddiscr[2]),
                C=CAMENCODE_C,
                downsample=CAMENCODE_DOWNSAMPLE,
                ddiscr=ddiscr,
                mode=mode,
                use_gt_depth=False,
                depth_supervision=False,
            )
            n_total += _load_exact(
                enc, state, f"frontend.encoders.{agent}."
            )
            self.encoders[agent] = enc

            hr = HighResFusion()
            n_total += _load_exact(hr, state, f"highres.{agent}.")
            self.highres[agent] = hr

            if agent == "vehicle":
                head = DepthHead(num_bins=int(ddiscr[2]))
                n_total += _load_exact(
                    head, state, "depth_heads.vehicle."
                )
                self.depth_heads[agent] = head

        if "vehicle" in needed:
            if "r2_downsample.weight" not in state:
                raise RuntimeError(
                    "[frozen-p1] Griffin vehicle P1 requires r2_downsample.*"
                )
            conv = nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1)
            n_total += _load_exact(conv, state, "r2_downsample.")
            self.r2_downsample = conv

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        self._cache_obj: Optional[Dict[str, Any]] = None
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._logged_shapes: set[str] = set()

        print(
            f"[frozen-p1] Griffin loaded {n_total} tensors from {ckpt_path}; "
            "heatmap heads skipped; drone depth=analytic geometry"
        )

    def train(self, mode: bool = True) -> "FrozenP1":
        super().train(False)
        return self

    @torch.no_grad()
    def extract(
        self, agent: str, cam_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        imgs = cam_inputs["imgs"]
        flat = imgs.reshape(-1, *imgs.shape[-3:]) if imgs.dim() == 5 else imgs
        rgb = flat[:, :3]
        r2, f45 = _camencode_r2_and_f45(self.encoders[agent], rgb)
        if agent == "vehicle":
            if self.r2_downsample is None:
                raise RuntimeError("[frozen-p1] missing vehicle r2_downsample")
            r2 = self.r2_downsample(r2)
        f90 = self.highres[agent](r2, f45)
        out: Dict[str, torch.Tensor] = {"f90": f90}
        if agent == "vehicle":
            out["depth"] = F.softmax(self.depth_heads[agent](f90), dim=1)
        return out

    @torch.no_grad()
    def encode(self, data_dict: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        if self._cache_obj is data_dict:
            return self._cache
        self._cache_obj = data_dict
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for agent in present_camera_agents(data_dict):
            if agent not in self.encoders:
                continue
            cam = data_dict[agent]["batch_merged_cam_inputs"]
            pack = self.extract(agent, cam)
            if agent not in self._logged_shapes:
                extra = (
                    f" depth={tuple(pack['depth'].shape)}"
                    if "depth" in pack else " depth=analytic"
                )
                print(
                    f"[frozen-p1] {agent} f90={tuple(pack['f90'].shape)}{extra}"
                )
                self._logged_shapes.add(agent)
            out[agent] = pack
        self._cache = out
        return out

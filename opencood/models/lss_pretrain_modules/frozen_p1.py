"""Frozen P1 image stack for homo Stage-0: CamEncode→F90→trained depth.

Loads ONLY the P1 image branch tensors from a merged branch checkpoint
(``net_final_branch.pth`` / ``net_final_branch_v45.pth``) into lightweight
modules — never builds the full ``Airv2xGaussian0822`` (no Gaussian sampler /
interaction / detection stack). All parameters stay frozen in eval.

Source mapping (explicit, per agent)::

    frontend.encoders.<agent>.*  -> CamEncode        (trunk/up1/up2 trained)
    r2_downsample.*              -> Conv2d 24->24 s2 (v45 only, vehicle 90x160->45x80)
    highres.<agent>.*            -> HighResFusion    -> F90 [N,128,fH,fW]
    depth_heads.<agent>.*        -> DepthHead (veh/rsu, TRAINED categorical depth)
    drone_height_embed.*         -> HeightEmbedding
    drone_delta_head.*           -> DeltaHead        (drone per-pixel z = height + delta)

``r2_downsample`` is optional: when absent (old checkpoint) vehicle keeps the
native 90x160 grid like rsu/drone.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from opencood.models.gaussian_modules_0822.highres_adapter import HighResFusion
from opencood.models.gaussian_modules_0822.image_frontend import (
    _camencode_r2_and_f45,
    flatten_camera_world_z,
    present_camera_agents,
)
from opencood.models.gaussian_modules_0822.lss.head import (
    CATEGORICAL_DEPTH_AGENTS,
    DeltaHead,
    DepthHead,
    HeightEmbedding,
)
from opencood.models.sub_modules.lss_submodule import CamEncode

AGENT_TYPES = ("vehicle", "rsu", "drone")
# Official CamEncode in the P1 checkpoint: image_head is 64-ch, up2 exists
# (downsample=8). Splat frustum / BevEncode use a different img_downsample
# and img_features; do not reuse those yaml fields here.
CAMENCODE_C = 64
CAMENCODE_DOWNSAMPLE = 8


def _load_exact(module: nn.Module, state: Dict[str, torch.Tensor], prefix: str) -> int:
    """Copy ``prefix*`` tensors into ``module`` with exact 1:1 coverage."""
    src = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    dst = module.state_dict()
    missing = sorted(set(dst) - set(src))
    unexpected = sorted(set(src) - set(dst))
    if missing or unexpected:
        raise RuntimeError(
            f"[frozen-p1] non-exact mapping for '{prefix}': "
            f"missing={missing[:6]} unexpected={unexpected[:6]}"
        )
    for key, tensor in src.items():
        if tuple(dst[key].shape) != tuple(tensor.shape):
            raise RuntimeError(
                f"[frozen-p1] shape mismatch '{prefix}{key}': "
                f"ckpt {tuple(tensor.shape)} vs model {tuple(dst[key].shape)}"
            )
    module.load_state_dict(src, strict=True)
    return len(src)


class FrozenP1(nn.Module):
    """Frozen P1 frontend producing F90 + trained depth per agent.

    ``encode(data_dict)`` returns ``{agent: {"f90": ..., "depth"|"z": ...}}``
    and caches per batch dict identity so vehicle/rsu/drone encoders spawned
    from one homo model share a single forward pass.
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__()
        ckpt_path = str(args["p1_checkpoint"])
        raw = torch.load(ckpt_path, map_location="cpu")
        state = (
            raw["model_state_dict"]
            if isinstance(raw, dict) and "model_state_dict" in raw
            else raw
        )
        state = {
            (k[7:] if k.startswith("module.") else k): v for k, v in state.items()
        }

        self.encoders = nn.ModuleDict()
        self.highres = nn.ModuleDict()
        self.depth_heads = nn.ModuleDict()
        self.r2_downsample: Optional[nn.Conv2d] = None
        self.drone_height_embed: Optional[HeightEmbedding] = None
        self.drone_delta_head: Optional[DeltaHead] = None

        needed = [a for a in AGENT_TYPES if a in set(args.get("collaborators") or AGENT_TYPES)]
        if not needed:
            raise ValueError("[frozen-p1] collaborators did not match vehicle/rsu/drone")

        n_total = 0
        for agent in needed:
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
            n_total += _load_exact(enc, state, f"frontend.encoders.{agent}.")
            self.encoders[agent] = enc

            hr = HighResFusion()
            n_total += _load_exact(hr, state, f"highres.{agent}.")
            self.highres[agent] = hr

            if agent in CATEGORICAL_DEPTH_AGENTS:
                head = DepthHead(num_bins=int(ddiscr[2]))
                n_total += _load_exact(head, state, f"depth_heads.{agent}.")
                self.depth_heads[agent] = head

        if "vehicle" in needed and "r2_downsample.weight" in state:
            conv = nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1)
            n_total += _load_exact(conv, state, "r2_downsample.")
            self.r2_downsample = conv

        if "drone" in needed:
            if "drone_height_embed.net.0.weight" not in state:
                raise RuntimeError(
                    "[frozen-p1] drone collaborator requested but checkpoint "
                    "has no drone_height_embed / drone_delta_head"
                )
            he = HeightEmbedding()
            n_total += _load_exact(he, state, "drone_height_embed.")
            de = DeltaHead()
            n_total += _load_exact(de, state, "drone_delta_head.")
            self.drone_height_embed = he
            self.drone_delta_head = de

        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        print(
            f"[frozen-p1] loaded {n_total} tensors (exact) from {ckpt_path}; "
            f"r2_downsample={'yes' if self.r2_downsample is not None else 'no'}"
        )

        self._cache_obj: Optional[Dict[str, Any]] = None
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._logged_shapes: set = set()

    def train(self, mode: bool = True) -> "FrozenP1":
        """Frozen forever: eval regardless of the requested mode."""
        super().train(False)
        return self

    @torch.no_grad()
    def extract(
        self, agent: str, cam_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """One agent's F90 + depth/z from its camera inputs."""
        imgs = cam_inputs["imgs"]
        flat = imgs.reshape(-1, *imgs.shape[-3:]) if imgs.dim() == 5 else imgs
        rgb = flat[:, :3, :, :]
        r2, f45 = _camencode_r2_and_f45(self.encoders[agent], rgb)
        if agent == "vehicle" and self.r2_downsample is not None:
            r2 = self.r2_downsample(r2)
        f90 = self.highres[agent](r2, f45)
        out: Dict[str, torch.Tensor] = {"f90": f90}
        if agent == "drone":
            height = flatten_camera_world_z(cam_inputs["camera_world_z"], imgs)
            height = height.to(device=f90.device, dtype=f90.dtype)
            embed = self.drone_height_embed(
                height, (int(f90.shape[2]), int(f90.shape[3]))
            )
            delta = self.drone_delta_head(f90, embed.to(dtype=f90.dtype))
            out["z"] = height[:, None, None] + delta
        else:
            out["depth"] = F.softmax(self.depth_heads[agent](f90), dim=1)
        return out

    @torch.no_grad()
    def encode(self, data_dict: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Cache one forward per batch dict; shared by all agent encoders."""
        if self._cache_obj is data_dict:
            return self._cache
        self._cache_obj = data_dict
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for agent in present_camera_agents(data_dict):
            if agent not in self.encoders:
                continue
            cam = data_dict[agent]["batch_merged_cam_inputs"]
            packed = self.extract(agent, cam)
            if agent not in self._logged_shapes:
                extra = (
                    f" depth {tuple(packed['depth'].shape)}"
                    if "depth" in packed
                    else f" z {tuple(packed['z'].shape)}"
                )
                print(f"[frozen-p1] {agent} f90 {tuple(packed['f90'].shape)}{extra}")
                self._logged_shapes.add(agent)
            out[agent] = packed
        self._cache = out
        return out
